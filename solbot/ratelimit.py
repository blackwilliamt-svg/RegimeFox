"""Token-bucket rate limiting with a reserved lane for high-priority calls.

Rate limits are the binding constraint on this bot, not CPU. Jupiter's free key
allows 1 request/second; a keyless caller gets 0.5. The scan loop, the quote
path and the safety scanner all draw on the same budget, so a naive limiter
lets background scanning starve an entry that is trying to fire.

Two properties matter here:

* **Burst is separate from sustained rate.** The bucket starts *empty* of burst
  credit and refills at `rate` per second up to `burst`. Starting it full at a
  per-minute allowance would release a whole minute of budget instantly and trip
  the upstream limiter on the first cycle.
* **Low-priority callers cannot drain the bucket.** They may only consume down
  to `reserve`, leaving headroom so a high-priority call (a quote for an entry
  that is about to fire, or an exit) never queues behind a scan.

On HTTP 429 the caller reports back via :meth:`penalise`, which halves the
effective rate for a cooling-off window instead of retrying immediately.
"""
from __future__ import annotations

import threading
import time


class TokenBucket:
    def __init__(
        self,
        rate: float,
        burst: int = 3,
        *,
        reserve: float = 1.0,
        name: str = "bucket",
    ) -> None:
        self.name = name
        self._lock = threading.Condition()
        self._rate = max(0.01, float(rate))
        self._burst = max(1.0, float(burst))
        self._reserve = min(float(reserve), self._burst)
        self._tokens = 0.0          # start empty: no instant burst on boot
        self._updated = time.monotonic()
        self._penalty_until = 0.0
        self._penalty_factor = 1.0
        self.total_acquired = 0
        self.total_throttled = 0
        self.total_429 = 0

    # -- configuration -----------------------------------------------------
    def configure(self, rate: float, burst: int, reserve: float | None = None) -> None:
        with self._lock:
            self._rate = max(0.01, float(rate))
            self._burst = max(1.0, float(burst))
            if reserve is not None:
                self._reserve = min(float(reserve), self._burst)
            self._tokens = min(self._tokens, self._burst)
            self._lock.notify_all()

    @property
    def effective_rate(self) -> float:
        if time.monotonic() < self._penalty_until:
            return self._rate * self._penalty_factor
        return self._rate

    # -- internals ---------------------------------------------------------
    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._updated
        if elapsed <= 0:
            return
        self._updated = now
        self._tokens = min(self._burst, self._tokens + elapsed * self.effective_rate)

    def _floor(self, priority: str) -> float:
        """Lowest token count a caller of this priority is allowed to leave."""
        return 0.0 if priority == "high" else self._reserve

    # -- acquisition -------------------------------------------------------
    def try_acquire(self, cost: float = 1.0, priority: str = "normal") -> bool:
        with self._lock:
            self._refill()
            if self._tokens - cost >= self._floor(priority):
                self._tokens -= cost
                self.total_acquired += 1
                return True
            return False

    def acquire(
        self, cost: float = 1.0, priority: str = "normal", timeout: float | None = None
    ) -> bool:
        """Block until `cost` tokens are available. False if `timeout` elapsed."""
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._lock:
            while True:
                self._refill()
                floor = self._floor(priority)
                if self._tokens - cost >= floor:
                    self._tokens -= cost
                    self.total_acquired += 1
                    return True
                needed = (cost + floor) - self._tokens
                wait = max(0.005, needed / max(self.effective_rate, 0.01))
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return False
                    wait = min(wait, remaining)
                self.total_throttled += 1
                self._lock.wait(wait)

    # -- backoff -----------------------------------------------------------
    def penalise(self, seconds: float = 30.0, factor: float = 0.5) -> None:
        """Called on HTTP 429: back the sustained rate off for a window."""
        with self._lock:
            now = time.monotonic()
            self.total_429 += 1
            self._penalty_until = now + seconds
            self._penalty_factor = max(0.05, factor)
            self._tokens = 0.0
            # Restart the refill clock as well. Zeroing the tokens alone is not
            # enough: the next refill would credit back everything that accrued
            # before the 429, undoing the backoff on the very next call.
            self._updated = now
            self._lock.notify_all()

    def stats(self) -> dict[str, float | int | str]:
        with self._lock:
            self._refill()
            return {
                "name": self.name,
                "rate": round(self._rate, 3),
                "effective_rate": round(self.effective_rate, 3),
                "burst": self._burst,
                "reserve": self._reserve,
                "tokens": round(self._tokens, 2),
                "acquired": self.total_acquired,
                "throttled": self.total_throttled,
                "rate_limited": self.total_429,
                "penalised": time.monotonic() < self._penalty_until,
            }


