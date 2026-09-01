"""Two-tier price scanning, served exclusively by the Jupiter Price API.

Tier 1 sweeps the whole eligible universe about every 15 seconds. Tier 2 polls
"tokens of interest" - those already showing a volume spike or building
momentum - about every second, so the entry window is not missed while the
signal is still being confirmed.

Neither tier touches Birdeye. Birdeye's free tier allows one request per second
in total and would be exhausted immediately; it is reserved for historical
backfill.

**Cadence is budgeted, not assumed.** One sweep of N tokens costs ceil(N/50)
requests. At Jupiter's free-tier 1 rps a 300-token universe needs 6 requests per
sweep, so a 15s broad sweep plus a 1s hot poll wants roughly 1.4 rps - more than
the free key allows. Rather than silently falling behind, :meth:`Scanner.budget`
reports the shortfall, the engine surfaces it on the dashboard, and the limiter
degrades the *broad* tier first so the hot tier keeps its cadence.
"""
from __future__ import annotations

import logging
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any, Iterable

from . import db
from .clients import ApiError, JupiterClient, PriceInfo
from .clients.jupiter import PRICE_BATCH_LIMIT

log = logging.getLogger(__name__)


@dataclass(slots=True)
class ScanBudget:
    universe_size: int
    hot_size: int
    broad_calls_per_sweep: int
    hot_calls_per_poll: int
    required_rps: float
    available_rps: float

    @property
    def sustainable(self) -> bool:
        return self.required_rps <= self.available_rps + 1e-9

    def message(self) -> str:
        if self.sustainable:
            return (
                f"Scan cadence sustainable: needs {self.required_rps:.2f} rps of "
                f"{self.available_rps:.2f} available."
            )
        return (
            f"Scan cadence exceeds the API budget: needs {self.required_rps:.2f} rps "
            f"but the key allows {self.available_rps:.2f}. The broad sweep will run "
            f"slower than configured. Raise broad_scan_seconds, shrink the universe, "
            f"or move to a higher Jupiter tier."
        )


@dataclass(slots=True)
class ScanResult:
    prices: dict[str, PriceInfo] = field(default_factory=dict)
    tier: str = "broad"
    calls: int = 0
    duration: float = 0.0
    errors: int = 0


class Scanner:
    """Owns the price polling cadence and the hot list."""

    def __init__(self, jupiter: JupiterClient, cfg: dict[str, Any]) -> None:
        self.jupiter = jupiter
        self.cfg = cfg
        self._hot: dict[str, float] = {}      # mint -> monotonic time added
        self._last_broad = 0.0
        self._last_hot = 0.0
        self.last_prices: dict[str, PriceInfo] = {}
        self.broad_sweeps = 0
        self.hot_polls = 0

    def update_config(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg

    # ------------------------------------------------------------------
    # hot list
    # ------------------------------------------------------------------
    @property
    def hot(self) -> list[str]:
        return list(self._hot)

    def mark_interesting(self, mint: str) -> None:
        """Escalate a token to the 1s tier."""
        if mint in self._hot:
            self._hot[mint] = time.monotonic()
            return
        limit = int(self.cfg["hot_list_max"])
        if len(self._hot) >= limit:
            # Evict the stalest entry rather than refusing the newest signal.
            oldest = min(self._hot, key=lambda m: self._hot[m])
            self._hot.pop(oldest, None)
        self._hot[mint] = time.monotonic()

    def cool(self, mint: str) -> None:
        self._hot.pop(mint, None)

    def expire_hot(self, max_age_seconds: float = 300.0) -> list[str]:
        """Drop tokens whose interest has gone stale, freeing budget."""
        now = time.monotonic()
        stale = [m for m, t in self._hot.items() if now - t > max_age_seconds]
        for m in stale:
            self._hot.pop(m, None)
        return stale

    def pin(self, mints: Iterable[str]) -> None:
        """Keep open positions permanently hot - they need per-second exit checks."""
        for m in mints:
            self._hot[m] = time.monotonic()

    # ------------------------------------------------------------------
    # cadence
    # ------------------------------------------------------------------
    def broad_due(self) -> bool:
        return (time.monotonic() - self._last_broad) >= float(self.cfg["broad_scan_seconds"])

    def hot_due(self) -> bool:
        return bool(self._hot) and (time.monotonic() - self._last_hot) >= float(
            self.cfg["hot_scan_seconds"]
        )

    def budget(self, universe_size: int, available_rps: float | None = None) -> ScanBudget:
        """What the configured cadence actually costs against the key's rate limit."""
        hot_size = len(self._hot)
        broad_calls = max(0, -(-universe_size // PRICE_BATCH_LIMIT))
        hot_calls = max(0, -(-hot_size // PRICE_BATCH_LIMIT))
        broad_rps = broad_calls / max(1.0, float(self.cfg["broad_scan_seconds"]))
        hot_rps = hot_calls / max(1.0, float(self.cfg["hot_scan_seconds"]))
        return ScanBudget(
            universe_size=universe_size,
            hot_size=hot_size,
            broad_calls_per_sweep=broad_calls,
            hot_calls_per_poll=hot_calls,
            required_rps=broad_rps + hot_rps,
            available_rps=float(
                available_rps if available_rps is not None else self.cfg["jupiter_rps"]
            ),
        )

    # ------------------------------------------------------------------
    # scanning
    # ------------------------------------------------------------------
    def scan_broad(self, mints: list[str]) -> ScanResult:
        """Sweep the full eligible universe at low priority."""
        started = time.monotonic()
        self._last_broad = started
        self.broad_sweeps += 1
        result = self._scan(mints, tier="broad", priority="low")
        result.duration = time.monotonic() - started
        return result

    def scan_hot(self) -> ScanResult:
        """Poll the tokens of interest at high priority so entries are not missed."""
        started = time.monotonic()
        self._last_hot = started
        self.hot_polls += 1
        result = self._scan(list(self._hot), tier="hot", priority="high")
        result.duration = time.monotonic() - started
        return result

    def _scan(self, mints: list[str], *, tier: str, priority: str) -> ScanResult:
        result = ScanResult(tier=tier)
        if not mints:
            return result
        try:
            prices = self.jupiter.prices(mints, priority=priority)
        except ApiError as exc:
            log.warning("%s scan failed: %s", tier, exc)
            result.errors = 1
            return result
        result.prices = prices
        result.calls = self.jupiter.price_calls_for(len(mints))
        self.last_prices.update(prices)
        return result

    # ------------------------------------------------------------------
    def record_ticks(
        self, prices: dict[str, PriceInfo], conn: sqlite3.Connection | None = None
    ) -> None:
        """Persist ticks so live candles and correlation have something to read."""
        if not prices:
            return
        conn = conn or db.connect()
        ts = db.now()
        conn.executemany(
            "INSERT INTO price_ticks(mint, ts, price) VALUES (?, ?, ?) "
            "ON CONFLICT(mint, ts) DO UPDATE SET price = excluded.price",
            [(p.mint, ts, p.price) for p in prices.values()],
        )

    def stats(self) -> dict[str, Any]:
        return {
            "hot_count": len(self._hot),
            "hot": list(self._hot)[:20],
            "broad_sweeps": self.broad_sweeps,
            "hot_polls": self.hot_polls,
            "tracked_prices": len(self.last_prices),
        }
