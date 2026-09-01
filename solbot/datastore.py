"""Candle storage: Birdeye backfill, live tick roll-up, and retention.

Two sources feed the same ``candles`` table:

* **Birdeye**, pulled once per token and then extended incrementally. This is
  the backtest's data and the indicator warm-up on a cold start.
* **Live ticks**, rolled up locally into candles at the configured timeframe.
  This is what keeps the current bar moving between Birdeye pulls, and it costs
  nothing extra - the ticks are already being collected by the scanner.

Birdeye does not serve a 10-minute interval, so the base interval stored is the
largest one that divides the configured timeframe (5m for the 10m default) and
:func:`candles_for` aggregates it up. Trading a timeframe that merely resembles
the configured one would quietly invalidate every backtest.

Retention is enforced on a schedule because the target box is a 1 vCPU / 2GB
droplet; an unbounded candle table is the thing most likely to fill it.
"""
from __future__ import annotations

import logging
import sqlite3
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

import pandas as pd

from . import db
from .clients import ApiError, BirdeyeClient, BudgetExhausted
from .clients.birdeye import OHLCV_CU_COST, SUPPORTED_INTERVALS, nearest_interval
from .indicators import resample, to_frame

log = logging.getLogger(__name__)

JOB_INITIAL_PULL = "historical_pull"


@dataclass(slots=True)
class PullReport:
    tokens_requested: int = 0
    tokens_done: int = 0
    tokens_failed: int = 0
    candles_written: int = 0
    calls: int = 0
    cu_spent: int = 0
    stopped_early: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "tokens_requested": self.tokens_requested,
            "tokens_done": self.tokens_done,
            "tokens_failed": self.tokens_failed,
            "candles_written": self.candles_written,
            "calls": self.calls,
            "cu_spent": self.cu_spent,
            "stopped_early": self.stopped_early,
        }


class DataStore:
    def __init__(self, birdeye: BirdeyeClient, cfg: dict[str, Any]) -> None:
        self.birdeye = birdeye
        self.cfg = cfg

    def update_config(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg

    # ------------------------------------------------------------------
    @property
    def base_interval(self) -> str:
        """The interval actually fetched and stored."""
        return nearest_interval(int(self.cfg["candle_minutes"]))

    @property
    def target_seconds(self) -> int:
        return int(self.cfg["candle_minutes"]) * 60

    def needs_aggregation(self) -> bool:
        return SUPPORTED_INTERVALS[self.base_interval] != self.target_seconds

    # ------------------------------------------------------------------
    # reading
    # ------------------------------------------------------------------
    def candles_for(
        self,
        mint: str,
        *,
        limit: int | None = None,
        since: int | None = None,
        until: int | None = None,
        conn: sqlite3.Connection | None = None,
    ) -> pd.DataFrame:
        """Candles at the configured timeframe, aggregating the base interval up."""
        interval = self.base_interval
        # Over-fetch so aggregation still yields `limit` finished bars.
        raw_limit = None
        if limit:
            factor = max(1, self.target_seconds // SUPPORTED_INTERVALS[interval])
            raw_limit = limit * factor + factor
        rows = db.load_candles(
            mint, interval, limit=raw_limit, since=since, until=until, conn=conn
        )
        df = to_frame(rows)
        if df.empty:
            return df
        if self.needs_aggregation():
            df = resample(df, self.target_seconds)
        if limit:
            df = df.tail(limit).reset_index(drop=True)
        return df

    def drop_forming_bar(self, df: pd.DataFrame, now_ts: int | None = None) -> pd.DataFrame:
        """Remove the still-forming candle.

        Signals are evaluated on closed candles only. Acting on a partial bar
        means the volume and momentum readings change under the strategy's feet
        and the backtest stops describing the same rules.
        """
        if df.empty:
            return df
        now_ts = now_ts or db.now()
        current_open = (now_ts // self.target_seconds) * self.target_seconds
        return df[df["ts"] < current_open].reset_index(drop=True)

    # ------------------------------------------------------------------
    # live tick roll-up
    # ------------------------------------------------------------------
    def rollup_ticks(
        self, *, lookback_seconds: int | None = None, conn: sqlite3.Connection | None = None
    ) -> int:
        """Fold recent price ticks into candles at the base interval.

        Volume is not available from the price feed, so a tick-built candle
        carries zero volume and is marked ``source='ticks'``. The volume-spike
        rule tolerates this: the rolling average skips NaN, and a zero-volume
        bar simply fails the spike test rather than producing a false one.
        """
        conn = conn or db.connect()
        interval = self.base_interval
        seconds = SUPPORTED_INTERVALS[interval]
        since = db.now() - (lookback_seconds or seconds * 4)

        rows = conn.execute(
            "SELECT mint, ts, price FROM price_ticks WHERE ts >= ? ORDER BY mint, ts",
            (since,),
        ).fetchall()
        if not rows:
            return 0

        buckets: dict[tuple[str, int], list[float]] = {}
        for r in rows:
            key = (r["mint"], (int(r["ts"]) // seconds) * seconds)
            buckets.setdefault(key, []).append(float(r["price"]))

        # Never overwrite a Birdeye candle with a thinner tick-built one.
        written = 0
        payload: list[tuple[Any, ...]] = []
        for (mint, bucket_ts), prices in buckets.items():
            if not prices:
                continue
            payload.append(
                (mint, interval, bucket_ts, prices[0], max(prices), min(prices), prices[-1], 0.0)
            )
        if payload:
            conn.executemany(
                "INSERT INTO candles(mint, interval, ts, open, high, low, close, volume, source) "
                "VALUES (?,?,?,?,?,?,?,?, 'ticks') "
                "ON CONFLICT(mint, interval, ts) DO UPDATE SET "
                "high = MAX(candles.high, excluded.high), "
                "low  = MIN(candles.low,  excluded.low), "
                "close = excluded.close "
                "WHERE candles.source = 'ticks'",
                payload,
            )
            written = len(payload)
        return written

    # ------------------------------------------------------------------
    # Birdeye backfill
    # ------------------------------------------------------------------
    def estimate_pull(self, token_count: int, days: int) -> dict[str, Any]:
        """Cost of a pull before running it, so the budget is visible up front."""
        interval = self.base_interval
        per_token = self.birdeye.calls_needed(interval, days)
        calls = per_token * max(0, token_count)
        cu = calls * OHLCV_CU_COST
        stats = self.birdeye.budget.stats()
        remaining = stats["remaining"]
        return {
            "interval": interval,
            "tokens": token_count,
            "days": days,
            "calls_per_token": per_token,
            "total_calls": calls,
            "cu_required": cu,
            "cu_remaining": remaining,
            "affordable": remaining < 0 or cu <= remaining,
            "seconds_estimate": int(calls / max(0.1, float(self.cfg["birdeye_rps"]))),
        }

    def backfill(
        self,
        mints: Sequence[str],
        *,
        days: int,
        conn: sqlite3.Connection | None = None,
        progress: Callable[[int, int, str], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
        incremental: bool = True,
    ) -> PullReport:
        """Pull history for each mint, resuming from what is already stored.

        `incremental=True` starts each token from its newest stored candle
        rather than re-pulling the whole window, which is what makes the daily
        run cheap enough to schedule.
        """
        conn = conn or db.connect()
        report = PullReport(tokens_requested=len(mints))
        interval = self.base_interval
        now = db.now()
        window_start = now - days * 86400

        for idx, mint in enumerate(mints, start=1):
            if should_stop and should_stop():
                report.stopped_early = "cancelled"
                break

            start = window_start
            if incremental:
                latest = db.latest_candle_ts(mint, interval, conn=conn)
                if latest:
                    start = max(window_start, latest - SUPPORTED_INTERVALS[interval])
            if start >= now:
                report.tokens_done += 1
                if progress:
                    progress(idx, len(mints), mint)
                continue

            try:
                candles = self.birdeye.ohlcv_range(mint, interval, start, now)
            except BudgetExhausted as exc:
                report.stopped_early = str(exc)
                log.warning("backfill halted: %s", exc)
                break
            except ApiError as exc:
                log.warning("backfill failed for %s: %s", mint, exc)
                report.tokens_failed += 1
                if progress:
                    progress(idx, len(mints), mint)
                continue

            if candles:
                written = db.upsert_candles(
                    mint, interval, [c.as_row() for c in candles], conn=conn
                )
                report.candles_written += written
            report.tokens_done += 1
            if progress:
                progress(idx, len(mints), mint)

        report.cu_spent = self.birdeye.budget.stats()["spent"]
        return report

    def run_initial_pull(
        self,
        mints: Sequence[str],
        *,
        days: int | None = None,
        conn: sqlite3.Connection | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> PullReport:
        """Backfill with dashboard progress reporting attached.

        Wired to the dashboard's manual button - the spec keeps this heavy,
        rate-limited, one-time load under the user's control rather than firing
        it automatically at startup.
        """
        conn = conn or db.connect()
        days = days or int(self.cfg["backtest_lookback_days"])
        total = len(mints)
        db.set_progress(
            JOB_INITIAL_PULL,
            status="running",
            done=0,
            total=total,
            message=f"Starting {days}-day pull for {total} tokens",
            conn=conn,
        )

        def report_progress(done: int, tot: int, mint: str) -> None:
            db.set_progress(
                JOB_INITIAL_PULL,
                status="running",
                done=done,
                total=tot,
                message=f"{done}/{tot} tokens - {mint[:8]}…",
                conn=conn,
            )

        try:
            result = self.backfill(
                mints,
                days=days,
                conn=conn,
                progress=report_progress,
                should_stop=should_stop,
                incremental=True,
            )
        except Exception as exc:  # surface the failure instead of a silent stall
            db.set_progress(
                JOB_INITIAL_PULL, status="failed", message=str(exc)[:300], conn=conn
            )
            db.log_event(
                f"Historical pull failed: {exc}", level="alert", category="system", conn=conn
            )
            raise

        status = "done" if not result.stopped_early else "failed"
        db.set_progress(
            JOB_INITIAL_PULL,
            status=status,
            done=result.tokens_done,
            total=total,
            message=result.stopped_early
            or f"{result.candles_written:,} candles across {result.tokens_done} tokens",
            conn=conn,
        )
        db.log_event(
            f"Historical pull finished: {result.candles_written:,} candles, "
            f"{result.tokens_done}/{total} tokens"
            + (f" - stopped early: {result.stopped_early}" if result.stopped_early else ""),
            level="warn" if result.stopped_early else "info",
            category="system",
            detail=result.as_dict(),
            conn=conn,
        )
        return result

    # ------------------------------------------------------------------
    def warm(self, mint: str, conn: sqlite3.Connection | None = None) -> bool:
        """Does this token have enough history for the strategy to evaluate it?"""
        df = self.candles_for(mint, limit=int(self.cfg["min_candles_required"]) + 2, conn=conn)
        return len(df) >= int(self.cfg["min_candles_required"])

    def prune(self, conn: sqlite3.Connection | None = None) -> dict[str, int]:
        return db.prune(
            candle_days=int(self.cfg["candle_retention_days"]),
            tick_hours=int(self.cfg["price_tick_retention_hours"]),
            event_days=int(self.cfg["event_retention_days"]),
            conn=conn,
        )

    def coverage(self, conn: sqlite3.Connection | None = None) -> dict[str, Any]:
        conn = conn or db.connect()
        row = conn.execute(
            "SELECT COUNT(*) AS candles, COUNT(DISTINCT mint) AS tokens, "
            "MIN(ts) AS first_ts, MAX(ts) AS last_ts FROM candles WHERE interval = ?",
            (self.base_interval,),
        ).fetchone()
        return {
            "interval": self.base_interval,
            "candles": int(row["candles"] or 0),
            "tokens": int(row["tokens"] or 0),
            "first_ts": row["first_ts"],
            "last_ts": row["last_ts"],
            "days": round(((row["last_ts"] or 0) - (row["first_ts"] or 0)) / 86400.0, 1),
        }
