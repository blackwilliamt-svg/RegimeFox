"""Binance.US-sourced candle backfill (spec 3b): month-boundary arithmetic and
the bulk backfill's write path, paging the klines REST endpoint rather than
downloading a pre-built archive - Binance.US has no documented equivalent of
global Binance's monthly zip archives."""
from __future__ import annotations

import time

from solbot import db
from solbot.candlestore import ParquetCandleStore
from solbot.clients.base import ApiError
from solbot.clients.binance import Candle
from solbot.datastore import JOB_INITIAL_PULL, DataStore, _month_bounds

from fakes import FakeBinance

MINT_A = "MintAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
MINT_B = "MintBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB"


def test_month_bounds_spans_exactly_one_calendar_month():
    start, end = _month_bounds(2024, 2)   # a leap year February
    assert time.strftime("%Y-%m-%d", time.gmtime(start)) == "2024-02-01"
    assert time.strftime("%Y-%m-%d", time.gmtime(end)) == "2024-03-01"
    assert (end - start) / 86400 == 29


def test_month_bounds_rolls_over_the_year():
    start, end = _month_bounds(2025, 12)
    assert time.strftime("%Y-%m-%d", time.gmtime(start)) == "2025-12-01"
    assert time.strftime("%Y-%m-%d", time.gmtime(end)) == "2026-01-01"


def test_bulk_backfill_writes_candles_from_paged_klines(tmp_path):
    def klines_fn(pair, since, until):
        return [
            Candle(ts=since, open=1.0, high=1.0, low=1.0, close=1.0, volume=10.0)
        ]

    binance = FakeBinance(klines_fn=klines_fn)
    store = DataStore(binance, {"candle_minutes": 1}, candles=ParquetCandleStore(tmp_path / "candles"))

    report = store.bulk_backfill({MINT_A: "BTCUSD"}, months=3)

    assert report.tokens_done == 1
    assert report.candles_written == 3   # one candle per month requested
    assert binance.calls == 3
    # coverage(), not candles_for(): candles_for()'s limit-based file
    # selection heuristic assumes a realistically dense month (tens of
    # thousands of bars) and isn't meant to be exercised by a synthetic
    # one-candle-per-month fixture like this one.
    assert store.candles.coverage(MINT_A, "1m")["bars"] == 3


def test_bulk_backfill_clips_the_current_month_to_now(tmp_path):
    """The current, still-forming month must never be requested past "now"."""
    seen: list[tuple[int, int]] = []

    def klines_fn(pair, since, until):
        seen.append((since, until))
        return []

    binance = FakeBinance(klines_fn=klines_fn)
    store = DataStore(binance, {"candle_minutes": 1}, candles=ParquetCandleStore(tmp_path / "candles"))
    store.bulk_backfill({MINT_A: "BTCUSD"}, months=1)

    now = int(time.time())
    assert seen, "expected at least one klines_range call"
    for _, until in seen:
        assert until <= now


def test_bulk_backfill_skips_a_failed_month_but_keeps_going(tmp_path):
    calls: list[int] = []

    def klines_fn(pair, since, until):
        calls.append(since)
        if len(calls) == 1:
            raise ApiError("binance: rate limited", provider="binance")
        return [Candle(ts=since, open=1.0, high=1.0, low=1.0, close=1.0, volume=10.0)]

    binance = FakeBinance(klines_fn=klines_fn)
    store = DataStore(binance, {"candle_minutes": 1}, candles=ParquetCandleStore(tmp_path / "candles"))
    report = store.bulk_backfill({MINT_A: "BTCUSD"}, months=2)

    assert report.tokens_done == 1          # the token still completes...
    assert report.candles_written == 1      # ...just with one month's data missing
    assert len(calls) == 2                  # both months were attempted


def test_run_initial_pull_reports_overall_status_not_a_bare_fraction(tmp_path, workspace, monkeypatch):
    """Progress bar sizing/clarity fix-up (spec item 3): the per-token
    progress message reads as overall status ("X of Y tokens complete"),
    not a raw "X/Y tokens - <pair>" counter fragment."""
    binance = FakeBinance(klines_fn=lambda pair, since, until: [])
    store = DataStore(binance, {"candle_minutes": 1}, candles=ParquetCandleStore(tmp_path / "candles"))

    seen_messages: list[str] = []
    original_set_progress = db.set_progress

    def spy(job, **kwargs):
        if job == JOB_INITIAL_PULL and kwargs.get("status") == "running":
            seen_messages.append(kwargs.get("message", ""))
        return original_set_progress(job, **kwargs)

    monkeypatch.setattr("solbot.datastore.db.set_progress", spy)
    store.run_initial_pull({MINT_A: "BTCUSD", MINT_B: "ETHUSD"}, months=1)

    per_token = [m for m in seen_messages if "of" in m]
    assert per_token, seen_messages
    assert any(
        m == "Pulling historical data — 1 of 2 tokens complete (BTCUSD now)"
        or m == "Pulling historical data — 2 of 2 tokens complete (ETHUSD now)"
        for m in per_token
    )
    # the old raw-counter phrasing is gone entirely, not just supplemented
    assert not any(m.startswith(("1/2", "2/2")) for m in seen_messages)


def test_bulk_backfill_respects_should_stop_between_tokens(tmp_path):
    binance = FakeBinance(klines_fn=lambda pair, since, until: [])
    store = DataStore(binance, {"candle_minutes": 1}, candles=ParquetCandleStore(tmp_path / "candles"))

    report = store.bulk_backfill(
        {MINT_A: "BTCUSD", MINT_B: "ETHUSD"}, months=1, should_stop=lambda: True
    )

    assert report.stopped_early == "cancelled"
    assert report.tokens_done == 0


# --------------------------------------------------------------------------
# Timing estimate: a deep-history backfill (up to the 96-month/8-year bound)
# across up to ~100 coins is meant to fit inside a 12-hour target.
# --------------------------------------------------------------------------
def test_estimate_pull_scales_linearly_with_tokens_and_months():
    store = DataStore(None, {"binance_rps": 5.0})

    one = store.estimate_pull(1, 1)
    ten_tokens = store.estimate_pull(10, 1)
    two_months = store.estimate_pull(1, 2)

    assert ten_tokens["calls"] == one["calls"] * 10
    assert two_months["calls"] == one["calls"] * 2


def test_estimate_pull_respects_the_configured_binance_rps():
    slow = DataStore(None, {"binance_rps": 1.0}).estimate_pull(10, 12)
    fast = DataStore(None, {"binance_rps": 10.0}).estimate_pull(10, 12)

    assert slow["calls"] == fast["calls"]                       # same request volume...
    assert slow["seconds_estimate"] > fast["seconds_estimate"]  # ...just throttled harder


def test_estimate_pull_flags_a_deep_backfill_that_misses_the_12_hour_target():
    store = DataStore(None, {"binance_rps": 5.0})

    # 8 years across 100 coins at the modest default throttle: a real,
    # honestly-reported miss - not something this estimate should hide.
    deep = store.estimate_pull(100, 96)
    assert deep["hours_estimate"] > deep["target_hours"]
    assert deep["within_target"] is False

    # The same scope comfortably fits once binance_rps is raised - the lever
    # an operator actually has, within its existing (0.5, 50.0) bound.
    fast_store = DataStore(None, {"binance_rps": 12.0})
    assert fast_store.estimate_pull(100, 96)["within_target"] is True


def test_estimate_pull_falls_back_to_a_sane_default_rps_when_unconfigured():
    store = DataStore(None, {})
    estimate = store.estimate_pull(5, 3)
    assert estimate["seconds_estimate"] > 0
