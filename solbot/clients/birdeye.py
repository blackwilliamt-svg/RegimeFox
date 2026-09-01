"""Birdeye client - historical OHLCV only.

Deliberately *not* the live feed. Birdeye's free Standard tier allows 1 request
per second and 30,000 compute units per month; an OHLCV call costs 35 CU, so the
entire month is about 857 calls. Routing the live scan through it would exhaust
the allowance in minutes, which is why the scan tiers are Jupiter-only and this
client exists purely to backfill candles for the backtest and indicator warm-up.

Endpoint verified 2026-08-31:

    GET https://public-api.birdeye.so/defi/ohlcv
        ?address=<mint>&type=<15m|1H|...>&time_from=<unix>&time_to=<unix>
        headers: X-API-KEY, x-chain: solana

Returns at most 1000 candles per call, so a long backfill is paged.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable

from ..ratelimit import MonthlyBudget, TokenBucket
from .base import ApiError, BudgetExhausted, HttpClient

log = logging.getLogger(__name__)

OHLCV_CU_COST = 35
TOKENLIST_CU_COST = 30
MAX_CANDLES_PER_CALL = 1000

# Birdeye's interval vocabulary. The bot trades 5-15m, so anything outside this
# set is rejected early rather than silently returning the wrong timeframe.
SUPPORTED_INTERVALS = {
    "1m": 60,
    "3m": 180,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1H": 3600,
    "2H": 7200,
    "4H": 14400,
    "1D": 86400,
}


@dataclass(slots=True)
class Candle:
    ts: int
    open: float
    high: float
    low: float
    close: float
    volume: float

    def as_row(self) -> tuple[int, float, float, float, float, float]:
        return (self.ts, self.open, self.high, self.low, self.close, self.volume)


def nearest_interval(minutes: int) -> str:
    """Map the configured candle length onto an interval Birdeye actually serves.

    The bot's default is 10 minutes, which Birdeye does not offer; 5m is
    returned so candles can be aggregated up locally without gaps.
    """
    target = minutes * 60
    exact = {v: k for k, v in SUPPORTED_INTERVALS.items()}
    if target in exact:
        return exact[target]
    divisors = [k for k, v in SUPPORTED_INTERVALS.items() if target % v == 0 and v < target]
    if divisors:
        return max(divisors, key=lambda k: SUPPORTED_INTERVALS[k])
    return "5m"


class BirdeyeClient(HttpClient):
    provider = "birdeye"
    base_url = "https://public-api.birdeye.so"

    def __init__(
        self,
        bucket: TokenBucket,
        api_key: str = "",
        budget: MonthlyBudget | None = None,
        *,
        on_spend: Callable[[int], None] | None = None,
        **kw: Any,
    ) -> None:
        headers = {"x-chain": "solana"}
        if api_key:
            headers["X-API-KEY"] = api_key
        super().__init__(bucket, headers=headers, **kw)
        self.api_key = api_key
        self.budget = budget or MonthlyBudget(0)
        self._on_spend = on_spend

    def set_api_key(self, api_key: str) -> None:
        self.api_key = api_key
        self.set_header("X-API-KEY", api_key or None)

    def _spend(self, units: int) -> None:
        self.budget.spend(units)
        if self._on_spend:
            try:
                self._on_spend(units)
            except Exception:
                log.debug("budget callback failed", exc_info=True)

    def _guard(self, units: int) -> None:
        if not self.budget.can_spend(units):
            raise BudgetExhausted(
                f"birdeye: monthly compute-unit budget exhausted "
                f"({self.budget.stats()['spent']}/{self.budget.limit} CU). "
                "Raise birdeye_monthly_cu_budget or upgrade the plan.",
                provider=self.provider,
            )

    # ------------------------------------------------------------------
    def ohlcv(
        self,
        mint: str,
        interval: str,
        time_from: int,
        time_to: int,
        *,
        priority: str = "low",
    ) -> list[Candle]:
        """One page of candles. Caller pages via :meth:`ohlcv_range`."""
        if interval not in SUPPORTED_INTERVALS:
            raise ValueError(f"birdeye does not serve interval {interval!r}")
        self._guard(OHLCV_CU_COST)
        data = self.get(
            "/defi/ohlcv",
            params={
                "address": mint,
                "type": interval,
                "time_from": int(time_from),
                "time_to": int(time_to),
            },
            priority=priority,
        )
        self._spend(OHLCV_CU_COST)

        if not isinstance(data, dict) or not data.get("success"):
            raise ApiError(
                f"birdeye ohlcv failed for {mint}: {str(data)[:200]}",
                provider=self.provider,
            )
        items = ((data.get("data") or {}).get("items")) or []
        out: list[Candle] = []
        for it in items:
            try:
                out.append(
                    Candle(
                        ts=int(it["unixTime"]),
                        open=float(it["o"]),
                        high=float(it["h"]),
                        low=float(it["l"]),
                        close=float(it["c"]),
                        # vUsd is the USD-denominated volume; the volume-spike
                        # rule compares tokens against each other, so USD is the
                        # only comparable unit.
                        volume=float(it.get("vUsd") or it.get("v") or 0.0),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
        out.sort(key=lambda c: c.ts)
        return out

    def ohlcv_range(
        self,
        mint: str,
        interval: str,
        time_from: int,
        time_to: int,
        *,
        max_pages: int = 40,
        priority: str = "low",
    ) -> list[Candle]:
        """Page through a long window, respecting the 1000-candle cap per call."""
        step = SUPPORTED_INTERVALS[interval] * MAX_CANDLES_PER_CALL
        out: list[Candle] = []
        cursor = int(time_from)
        end = int(time_to)
        pages = 0
        while cursor < end and pages < max_pages:
            chunk_end = min(cursor + step, end)
            try:
                page = self.ohlcv(mint, interval, cursor, chunk_end, priority=priority)
            except BudgetExhausted:
                raise
            except ApiError as exc:
                log.warning("birdeye page failed (%s %s): %s", mint, interval, exc)
                break
            out.extend(page)
            pages += 1
            if not page:
                # A gap with no trades is normal for a thin token; step past it
                # rather than spinning on the same window.
                cursor = chunk_end
                continue
            cursor = max(page[-1].ts + SUPPORTED_INTERVALS[interval], chunk_end)
        # De-duplicate on timestamp; overlapping pages are possible at the seams.
        seen: dict[int, Candle] = {c.ts: c for c in out}
        return [seen[k] for k in sorted(seen)]

    def calls_needed(self, interval: str, days: int) -> int:
        """Requests one token's backfill will cost - used to budget before pulling."""
        seconds = days * 86400
        per_call = SUPPORTED_INTERVALS[interval] * MAX_CANDLES_PER_CALL
        return max(1, -(-seconds // per_call))

    def ping(self) -> bool:
        self._guard(OHLCV_CU_COST)
        now = int(time.time())
        self.get(
            "/defi/ohlcv",
            params={
                "address": "So11111111111111111111111111111111111111112",
                "type": "1H",
                "time_from": now - 7200,
                "time_to": now,
            },
            priority="high",
        )
        self._spend(OHLCV_CU_COST)
        return True
