"""Candle storage: Binance.US backfill, live tick roll-up, and the Parquet store.

Candle history lives entirely in :class:`~solbot.candlestore.ParquetCandleStore`
(spec 3) - never as SQLite rows. Two Binance.US-sourced feeds populate it:

* **The one-time bulk backfill** (spec 3b), pulling a trailing year of
  1-minute candles per coin by paging the ordinary klines REST endpoint
  month by month. Global Binance publishes pre-built monthly zip archives
  (``data.binance.vision``) for this; Binance.US - which this bot uses
  instead, since global Binance geo-blocks US-origin traffic - has no
  documented equivalent, so this pages the rate-limited REST API instead.
  More requests for a one-time job, but still comfortably inside
  ``binance_rps``.
* **The daily incremental pull**, against the same klines endpoint, cheap
  enough at daily-request volume to run on a schedule with no special budget
  tracking.

Live ticks are rolled up locally into 1-minute candles between daily pulls, so
the current trading day is never stale by a full day - but only *closed*
minute buckets are ever written; the still-forming bar is never persisted; the
live engine drops it before computing indicators, same as before.
"""
from __future__ import annotations

import calendar
import logging
import sqlite3
import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import pandas as pd

from . import db
from .candlestore import ParquetCandleStore
from .clients import ApiError, BinanceClient
from .clients.binance import KLINE_INTERVAL, MAX_KLINES_PER_CALL
from .indicators import resample, to_frame

log = logging.getLogger(__name__)

JOB_INITIAL_PULL = "historical_pull"
JOB_DAILY_INCREMENTAL = "daily_incremental_pull"

BASE_INTERVAL = KLINE_INTERVAL   # "1m" - fixed, spec 3b
BASE_SECONDS = 60


def _month_bounds(year: int, month: int) -> tuple[int, int]:
    """``[start, end)`` unix seconds spanning a calendar month, in UTC."""
    start = calendar.timegm((year, month, 1, 0, 0, 0))
    next_year, next_month = (year + 1, 1) if month == 12 else (year, month + 1)
    end = calendar.timegm((next_year, next_month, 1, 0, 0, 0))
    return start, end


@dataclass(slots=True)
class PullReport:
    tokens_requested: int = 0
    tokens_done: int = 0
    tokens_failed: int = 0
    candles_written: int = 0
    calls: int = 0
    stopped_early: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "tokens_requested": self.tokens_requested,
            "tokens_done": self.tokens_done,
            "tokens_failed": self.tokens_failed,
            "candles_written": self.candles_written,
            "calls": self.calls,
            "stopped_early": self.stopped_early,
        }


class DataStore:
    def __init__(
        self,
        binance: BinanceClient,
        cfg: dict[str, Any],
        *,
        candles: ParquetCandleStore | None = None,
    ) -> None:
        self.binance = binance
        self.cfg = cfg
        self.candles = candles or ParquetCandleStore()

    def update_config(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg

    # ------------------------------------------------------------------
    @property
    def base_interval(self) -> str:
        return BASE_INTERVAL

    @property
    def target_seconds(self) -> int:
        return int(self.cfg["candle_minutes"]) * 60

    def needs_aggregation(self) -> bool:
        return self.target_seconds != BASE_SECONDS

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
        target_seconds: int | None = None,
        conn: sqlite3.Connection | None = None,   # kept for call-site symmetry; unused
    ) -> pd.DataFrame:
        """Candles at the configured timeframe, aggregating the 1m base up.

        ``target_seconds``, when given, overrides the configured trading
        timeframe for this one call - the market chart's timeframe control
        (dashboard fix-up section 2) asks for whatever interval the operator
        picked, independent of ``candle_minutes``, without that picking
        also changing what the live strategy itself trades on.
        """
        target = int(target_seconds) if target_seconds else self.target_seconds
        raw_limit = None
        if limit:
            factor = max(1, target // BASE_SECONDS)
            raw_limit = limit * factor + factor
        raw = self.candles.read_range(
            mint, self.base_interval, since=since, until=until, limit=raw_limit
        )
        df = to_frame(raw)
        if df.empty:
            return df
        if target != BASE_SECONDS:
            df = resample(df, target)
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
    # live tick roll-up - only ever writes CLOSED minute buckets
    # ------------------------------------------------------------------
    def rollup_ticks(
        self, *, lookback_seconds: int | None = None, conn: sqlite3.Connection | None = None
    ) -> int:
        """Fold recently-closed minute buckets of price ticks into the store.

        Volume is not available from the price feed, so a tick-built candle
        carries zero volume; the volume-spike rule tolerates this (a
        zero-volume bar simply fails the spike test rather than producing a
        false one). Only buckets that have fully closed are written - the
        forming bucket is never persisted, matching what the live engine
        already discards via :meth:`drop_forming_bar`, so a call made
        mid-minute costs nothing more than a metadata read per mint.
        """
        conn = conn or db.connect()
        now = db.now()
        current_bucket = (now // BASE_SECONDS) * BASE_SECONDS
        since = now - (lookback_seconds or BASE_SECONDS * 10)

        rows = conn.execute(
            "SELECT mint, ts, price FROM price_ticks WHERE ts >= ? ORDER BY mint, ts",
            (since,),
        ).fetchall()
        if not rows:
            return 0

        buckets: dict[tuple[str, int], list[float]] = {}
        for r in rows:
            bucket_ts = (int(r["ts"]) // BASE_SECONDS) * BASE_SECONDS
            if bucket_ts >= current_bucket:
                continue  # still forming; never persisted
            buckets.setdefault((r["mint"], bucket_ts), []).append(float(r["price"]))

        per_mint: dict[str, list[tuple[Any, ...]]] = {}
        for (mint, bucket_ts), prices in buckets.items():
            if not prices:
                continue
            per_mint.setdefault(mint, []).append(
                (bucket_ts, prices[0], max(prices), min(prices), prices[-1], 0.0)
            )

        written = 0
        for mint, mint_rows in per_mint.items():
            written += self.candles.append(mint, self.base_interval, mint_rows, incremental=True)
        return written

    # ------------------------------------------------------------------
    # Binance backfill
    # ------------------------------------------------------------------
    # A deep-history backfill (up to bulk_backfill_months' 96-month/8-year
    # ceiling) across up to 100 coins is meant to fit inside a working day -
    # the number the estimate below is judged against.
    BACKFILL_TARGET_HOURS = 12.0
    AVG_MONTH_SECONDS = 30.44 * 86400  # calendar months vary; the estimate only needs to be close

    def estimate_pull(self, token_count: int, months: int) -> dict[str, Any]:
        """Real cost of a bulk pull before running it - REST calls at the
        one-call-per-1000-candles page size, throttled to ``binance_rps`` -
        not a guess, so the dashboard can tell an operator *before* they
        commit whether an 8-year/100-coin pull will land inside the 12-hour
        target or needs a narrower request (fewer months, fewer tokens, or a
        raised binance_rps) instead.
        """
        calls_per_month = max(1, -(-int(self.AVG_MONTH_SECONDS) // (MAX_KLINES_PER_CALL * BASE_SECONDS)))
        calls = token_count * months * calls_per_month
        rps = max(0.1, float(self.cfg.get("binance_rps", 5.0)))
        seconds = calls / rps
        hours = seconds / 3600.0
        return {
            "tokens": token_count,
            "months": months,
            "calls": calls,
            "seconds_estimate": int(seconds),
            "hours_estimate": round(hours, 2),
            "within_target": hours <= self.BACKFILL_TARGET_HOURS,
            "target_hours": self.BACKFILL_TARGET_HOURS,
        }

    def bulk_backfill(
        self,
        pairs: dict[str, str],
        *,
        months: int,
        conn: sqlite3.Connection | None = None,
        progress: Callable[[int, int, str], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> PullReport:
        """One-time setup: pull a trailing ``months`` of history per coin.

        ``pairs`` maps mint -> Binance pair (e.g. "BTCUSDT"), from the
        universe table. Pages the klines REST endpoint month by month rather
        than downloading a pre-built archive - Binance.US has no documented
        equivalent of global Binance's monthly zip archives (spec 3b).
        """
        conn = conn or db.connect()
        report = PullReport(tokens_requested=len(pairs))
        now = db.now()
        now_struct = time.gmtime(now)

        month_list: list[tuple[int, int]] = []
        year, month = now_struct.tm_year, now_struct.tm_mon
        for _ in range(months):
            month_list.append((year, month))
            month -= 1
            if month == 0:
                month, year = 12, year - 1

        for idx, (mint, pair) in enumerate(pairs.items(), start=1):
            if should_stop and should_stop():
                report.stopped_early = "cancelled"
                break
            written_for_mint = 0
            for y, m in month_list:
                start, end = _month_bounds(y, m)
                end = min(end, now)
                if start >= end:
                    continue
                def _log_page_skipped(chunk_start: int, exc: Exception, pair: str = pair, y: int = y, m: int = m) -> None:
                    # klines_range itself already retried this page a few
                    # times (BinanceClient.PAGE_RETRY_ATTEMPTS) before
                    # giving up on it - a gap here is better than this
                    # month, and everything behind it, hanging on one page.
                    log.warning(
                        "bulk backfill: a page of %s %04d-%02d never came back after "
                        "retrying - skipping the rest of that month (%s)",
                        pair, y, m, exc,
                    )
                    db.log_event(
                        f"Bulk backfill: gave up on a page of {pair} {y:04d}-{m:02d} after "
                        f"retrying - the rest of that month is skipped ({exc})",
                        level="warn", category="system", conn=conn,
                    )

                try:
                    candles = self.binance.klines_range(
                        pair, since=start, until=end, on_page_skipped=_log_page_skipped,
                    )
                except ApiError as exc:
                    log.warning(
                        "bulk backfill failed for %s %04d-%02d: %s", pair, y, m, exc
                    )
                    continue
                report.calls += max(1, -(-(end - start) // (MAX_KLINES_PER_CALL * BASE_SECONDS)))
                if candles:
                    # Not incremental: months are walked newest-first, so an
                    # older month's rows would otherwise all be older than the
                    # latest timestamp already written and get silently
                    # dropped. _merge_month() already dedupes by ts on its
                    # own, so this stays safe to re-run.
                    written_for_mint += self.candles.append(
                        mint, self.base_interval, [c.as_row() for c in candles],
                        incremental=False,
                    )
            report.candles_written += written_for_mint
            report.tokens_done += 1
            if progress:
                progress(idx, len(pairs), pair)

        return report

    def daily_incremental_pull(
        self,
        pairs: dict[str, str],
        *,
        conn: sqlite3.Connection | None = None,
        progress: Callable[[int, int, str], None] | None = None,
    ) -> PullReport:
        """Catch each coin up from its newest stored candle to now, via REST.

        Safe from rate limits at daily-request volume (spec 3b) - one klines
        call per coin covers a whole day of 1-minute bars in a single page.
        """
        conn = conn or db.connect()
        report = PullReport(tokens_requested=len(pairs))
        now = db.now()

        for idx, (mint, pair) in enumerate(pairs.items(), start=1):
            latest = self.candles.latest_ts(mint, self.base_interval)
            since = (latest + BASE_SECONDS) if latest else (now - 7 * 86400)
            if since >= now:
                report.tokens_done += 1
                if progress:
                    progress(idx, len(pairs), pair)
                continue
            try:
                candles = self.binance.klines_range(pair, since=since, until=now)
            except ApiError as exc:
                log.warning("daily incremental pull failed for %s: %s", pair, exc)
                report.tokens_failed += 1
                if progress:
                    progress(idx, len(pairs), pair)
                continue

            if candles:
                report.candles_written += self.candles.append(
                    mint, self.base_interval, [c.as_row() for c in candles]
                )
            report.tokens_done += 1
            if progress:
                progress(idx, len(pairs), pair)
        return report

    def run_initial_pull(
        self,
        pairs: dict[str, str],
        *,
        months: int | None = None,
        conn: sqlite3.Connection | None = None,
        should_stop: Callable[[], bool] | None = None,
        top_n: int | None = None,
    ) -> PullReport:
        """The bulk backfill with dashboard progress reporting attached.

        Wired to the dashboard's manual button - the spec keeps this heavy,
        one-time load under the user's control rather than firing it
        automatically at startup. `pairs` is expected to already be scoped
        (e.g. via `pair_map(top_n=...)`) by the caller; `top_n` here is only
        used to phrase the starting message ("top N by market cap" instead
        of a bare token count) - it does not do any filtering itself.
        """
        conn = conn or db.connect()
        months = months or int(self.cfg["bulk_backfill_months"])
        total = len(pairs)
        scope = f"the top {top_n} tokens by market cap" if top_n else f"{total} tokens"
        db.set_progress(
            JOB_INITIAL_PULL,
            status="running",
            done=0,
            total=total,
            message=f"Starting a {months}-month pull for {scope}",
            conn=conn,
        )

        def report_progress(done: int, tot: int, pair: str) -> None:
            db.set_progress(
                JOB_INITIAL_PULL,
                status="running",
                done=done,
                total=tot,
                message=f"Pulling historical data — {done} of {tot} tokens complete ({pair} now)",
                conn=conn,
            )

        try:
            result = self.bulk_backfill(
                pairs, months=months, conn=conn,
                progress=report_progress, should_stop=should_stop,
            )
        except Exception as exc:
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
    # Full Binance.US history pull (dashboard fix-up section 6) - "replace
    # the historical pull entirely": every Binance.US pair, its full
    # available history, never scoped by the routed `universe` table or its
    # liquidity/volume floors. Deliberately separate methods from
    # bulk_backfill/estimate_pull/run_initial_pull above rather than a mode
    # flag bolted onto them - manage.py's own `pull` CLI command still uses
    # those, scoped to the routed universe with a fixed months window, and
    # section 6 never asked to change that command.
    # ------------------------------------------------------------------
    # Binance.US launched in 2019; no pair can have more history than that,
    # though almost every pair's own listing is much shorter. There is no
    # exchange-info "listed since" field to compute an exact figure from
    # before paging actually discovers it, so this is always reported as an
    # upper bound, never as a real estimate.
    ASSUMED_MAX_HISTORY_YEARS = 7

    def full_binance_pairs(self, conn: sqlite3.Connection | None = None) -> dict[str, str]:
        """``{storage_key: pair}`` for every base asset Binance.US currently
        lists - entirely independent of the routed `universe` table's own
        liquidity/volume floors (nothing is excluded because of them), but
        for a pair that *is* currently routed, the storage key is that
        coin's own mint - the same key the daily incremental pull, the
        backtester, and every dashboard summary already read candles under
        - rather than a second, disconnected entry keyed by the bare
        Binance symbol. Without this, a full-history pull would silently
        leave every already-routed coin's own coverage unchanged: it would
        still download real candles, just under a key nothing else ever
        looks at.
        """
        conn = conn or db.connect()
        mint_by_pair = {
            row["binance_pair"]: row["mint"]
            for row in conn.execute(
                "SELECT mint, binance_pair FROM universe "
                "WHERE binance_pair IS NOT NULL AND binance_pair != ''"
            ).fetchall()
        }
        return {
            mint_by_pair.get(a.pair, a.symbol): a.pair for a in self.binance.all_bases()
        }

    def estimate_full_pull(self, token_count: int) -> dict[str, Any]:
        """Upper-bound cost of a full-history pull across `token_count`
        pairs, assuming every one of them goes all the way back to
        ASSUMED_MAX_HISTORY_YEARS - always the ceiling, since most pairs'
        real history is much shorter and there is no way to know how much
        shorter before actually paging backward through it."""
        seconds_of_history = self.ASSUMED_MAX_HISTORY_YEARS * 365.25 * 86400
        calls_per_token = max(
            1, -(-int(seconds_of_history) // (MAX_KLINES_PER_CALL * BASE_SECONDS))
        )
        calls = token_count * calls_per_token
        rps = max(0.1, float(self.cfg.get("binance_rps", 5.0)))
        seconds = calls / rps
        return {
            "tokens": token_count,
            "calls": calls,
            "seconds_estimate": int(seconds),
            "hours_estimate": round(seconds / 3600.0, 2),
            "upper_bound": True,
            "assumed_years": self.ASSUMED_MAX_HISTORY_YEARS,
        }

    def full_history_backfill(
        self,
        pairs: dict[str, str],
        *,
        conn: sqlite3.Connection | None = None,
        progress: Callable[[int, int, str], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
        page_progress: Callable[[int, int, str, int], None] | None = None,
    ) -> PullReport:
        """Every pair's full available history - paging backward per pair
        until Binance.US returns an empty page, never a fixed months
        window. `pairs` is expected to be `full_binance_pairs()`'s own
        mapping (base symbol -> Binance pair), not the routed universe.

        ``should_stop`` is threaded all the way into
        ``BinanceClient.klines_backward``, not just checked between pairs
        here - a pair with a lot of real history (or one stuck slowly
        retrying through rate-limit backoff) can take a very long time on
        its own, and without this a Stop request would have no effect until
        that single pair finally finished. ``page_progress``, forwarded as
        klines_backward's own on_page hook, is what keeps the dashboard's
        progress readout live while that one pair is still paging, instead
        of looking frozen between the "N of M pairs" ticks either side of
        this call.
        """
        conn = conn or db.connect()
        report = PullReport(tokens_requested=len(pairs))

        for idx, (key, pair) in enumerate(pairs.items(), start=1):
            if should_stop and should_stop():
                report.stopped_early = "cancelled"
                break
            def _log_page_skipped(page_num: int, exc: Exception, pair: str = pair) -> None:
                # A page that still fails after klines_backward's own
                # page-level retries - a gap in this pair's history is
                # better than this pair (and the whole 269-pair job behind
                # it) hanging or aborting on one unlucky page.
                log.warning(
                    "full-history backfill: page %d of %s never came back "
                    "after retrying - skipping the rest of this pair (%s)",
                    page_num, pair, exc,
                )
                db.log_event(
                    f"Historical pull: gave up on page {page_num} of {pair} after "
                    f"retrying - the rest of that pair's history is skipped, not "
                    f"the whole job ({exc})",
                    level="warn", category="system", conn=conn,
                )

            try:
                candles = self.binance.klines_backward(
                    pair,
                    should_stop=should_stop,
                    on_page=(
                        (lambda page_num, idx=idx, pair=pair:
                            page_progress(idx - 1, len(pairs), pair, page_num))
                        if page_progress else None
                    ),
                    on_page_skipped=_log_page_skipped,
                )
            except ApiError as exc:
                log.warning("full-history backfill failed for %s: %s", pair, exc)
                report.tokens_failed += 1
                if progress:
                    progress(idx, len(pairs), pair)
                continue

            report.calls += max(1, -(-len(candles) // MAX_KLINES_PER_CALL))
            if candles:
                # Written even when should_stop cut this pair's own paging
                # short below - whatever was fetched before the stop is
                # real, deduped, safe-to-resume-from history, not discarded.
                report.candles_written += self.candles.append(
                    key, self.base_interval, [c.as_row() for c in candles],
                    incremental=False,
                )
            report.tokens_done += 1
            if progress:
                progress(idx, len(pairs), pair)

            if should_stop and should_stop():
                report.stopped_early = "cancelled"
                break

        return report

    def run_full_history_pull(
        self,
        pairs: dict[str, str],
        *,
        conn: sqlite3.Connection | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> PullReport:
        """The full-history backfill with dashboard progress reporting
        attached - reuses the historical-pull job key/progress slot
        (JOB_INITIAL_PULL); the dashboard's pull screen always runs this
        now instead of the old months/top-N-scoped pull."""
        conn = conn or db.connect()
        total = len(pairs)
        db.set_progress(
            JOB_INITIAL_PULL,
            status="running",
            done=0,
            total=total,
            message=f"Starting a full-history pull for all {total} Binance.US pairs",
            conn=conn,
        )

        def report_progress(done: int, tot: int, pair: str) -> None:
            db.set_progress(
                JOB_INITIAL_PULL,
                status="running",
                done=done,
                total=tot,
                message=f"Pulling full Binance.US history — {done} of {tot} pairs complete ({pair} now)",
                conn=conn,
            )

        def report_page_progress(done: int, tot: int, pair: str, page_num: int) -> None:
            # Every page rather than throttled: klines_backward is already
            # rate-limited to a handful of requests/second, so this writes
            # no faster than the pull itself is already going - and it is
            # exactly what keeps a pair with a lot of real history (or one
            # stuck slowly retrying) from making the dashboard look frozen
            # between "N of M pairs" ticks either side of it.
            db.set_progress(
                JOB_INITIAL_PULL,
                status="running",
                done=done,
                total=tot,
                message=(
                    f"Pulling full Binance.US history — {done} of {tot} pairs complete "
                    f"({pair} now, page {page_num})"
                ),
                conn=conn,
            )

        try:
            result = self.full_history_backfill(
                pairs, conn=conn, progress=report_progress, should_stop=should_stop,
                page_progress=report_page_progress,
            )
        except Exception as exc:
            db.set_progress(
                JOB_INITIAL_PULL, status="failed", message=str(exc)[:300], conn=conn
            )
            db.log_event(
                f"Full-history pull failed: {exc}", level="alert", category="system", conn=conn
            )
            raise

        # "cancelled" (an operator hit Stop) is not the same situation as
        # "failed" (an exception blew the whole pull up) even though both
        # leave the pull incomplete - the dashboard shows them differently,
        # and only "failed" should read as an actual error.
        if result.stopped_early == "cancelled":
            status = "cancelled"
            message = (
                f"Stopped by operator after {result.tokens_done} of {total} pairs "
                f"({result.candles_written:,} candles written)"
            )
        elif result.stopped_early:
            status = "failed"
            message = result.stopped_early
        else:
            status = "done"
            message = f"{result.candles_written:,} candles across {result.tokens_done} pairs"
        db.set_progress(
            JOB_INITIAL_PULL,
            status=status,
            done=result.tokens_done,
            total=total,
            message=message,
            conn=conn,
        )
        db.log_event(
            f"Full Binance.US history pull finished: {result.candles_written:,} candles, "
            f"{result.tokens_done}/{total} pairs"
            + (f" - stopped early: {result.stopped_early}" if result.stopped_early else ""),
            level="warn" if result.stopped_early else "info",
            category="system",
            detail=result.as_dict(),
            conn=conn,
        )
        return result

    def run_daily_incremental(
        self, pairs: dict[str, str], *, conn: sqlite3.Connection | None = None
    ) -> PullReport:
        conn = conn or db.connect()
        db.set_progress(JOB_DAILY_INCREMENTAL, status="running", total=len(pairs), conn=conn)
        try:
            result = self.daily_incremental_pull(pairs, conn=conn)
        except Exception as exc:
            db.set_progress(
                JOB_DAILY_INCREMENTAL, status="failed", message=str(exc)[:300], conn=conn
            )
            raise
        db.set_progress(
            JOB_DAILY_INCREMENTAL,
            status="done",
            done=result.tokens_done,
            total=len(pairs),
            message=f"{result.candles_written:,} candles across {result.tokens_done} tokens",
            conn=conn,
        )
        db.log_event(
            f"Daily incremental candle pull: {result.candles_written:,} candles across "
            f"{result.tokens_done}/{len(pairs)} tokens.",
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

    def coverage(self, conn: sqlite3.Connection | None = None) -> dict[str, Any]:
        conn = conn or db.connect()
        mints = [r["mint"] for r in conn.execute("SELECT mint FROM universe").fetchall()]
        cov = self.candles.universe_coverage(mints, self.base_interval)
        cov["disk"] = self.candles.disk_usage()
        return cov

    def prune(self, conn: sqlite3.Connection | None = None) -> dict[str, int]:
        """Retention for everything EXCEPT candles, which are kept indefinitely."""
        deleted = db.prune(
            tick_hours=int(self.cfg["price_tick_retention_hours"]),
            event_days=int(self.cfg["event_retention_days"]),
            wfmc_days=int(self.cfg.get("wfmc_result_retention_days", 180)),
            conn=conn,
        )
        try:
            from .wfmc import DAILY_STORE_PATH
            from solopt.store import RunStore

            store = RunStore(DAILY_STORE_PATH)
            local = store.purge_older_than(int(self.cfg.get("wfmc_result_retention_days", 180)))
            deleted["wfmc_local_runs"] = local["runs"]
            # Persistent library (gap-closure item 4): capped by entry count,
            # not age - see library_max_entries's own comment in config.py.
            deleted["library_pruned"] = store.prune_library(
                keep=int(self.cfg.get("library_max_entries", 500))
            )
        except Exception:
            pass  # the local WFMC store may not exist yet on a fresh install
        return deleted

    def pair_map(
        self, conn: sqlite3.Connection | None = None, *, top_n: int | None = None
    ) -> dict[str, str]:
        """``{mint: binance_pair}`` for every routed universe token.

        `top_n`, when given, narrows this to the top `top_n` mints ranked by
        market cap (``universe.mcap``, sourced from Jupiter's token API at
        every universe refresh - real data already in this table, not a
        proxy), coins with no known market cap sorted last and broken by
        24h volume. This is a one-off scoping for the manual historical-pull
        button (dashboard fix-up section 6) - every other caller (daily
        incremental pull, backtest, walk-forward, regime pass) calls this
        with the default `top_n=None` and gets the exact same full routed
        universe as before; it does not touch the universe table's own
        liquidity/volume floor filters or the live trading universe.
        """
        conn = conn or db.connect()
        if top_n:
            # Known-market-cap rows always sort ahead of null/zero ones
            # (first ORDER BY key); the CASE pins every null/zero row's
            # second key to the same 0, so within THAT bucket the third key
            # (volume) is what actually breaks ties, rather than mcap's own
            # NULL-sorts-last-in-DESC behaviour doing it by accident.
            rows = conn.execute(
                "SELECT mint, binance_pair FROM universe "
                "WHERE binance_pair IS NOT NULL AND binance_pair != '' "
                "ORDER BY (mcap IS NULL OR mcap <= 0) ASC, "
                "CASE WHEN mcap IS NULL OR mcap <= 0 THEN 0 ELSE mcap END DESC, "
                "volume_24h_usd DESC "
                "LIMIT ?",
                (int(top_n),),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT mint, binance_pair FROM universe WHERE binance_pair IS NOT NULL "
                "AND binance_pair != ''"
            ).fetchall()
        return {r["mint"]: r["binance_pair"] for r in rows}
