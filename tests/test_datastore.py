"""Binance.US-sourced candle backfill (spec 3b): month-boundary arithmetic and
the bulk backfill's write path, paging the klines REST endpoint rather than
downloading a pre-built archive - Binance.US has no documented equivalent of
global Binance's monthly zip archives."""
from __future__ import annotations

import time

from solbot.candlestore import ParquetCandleStore
from solbot.clients.base import ApiError
from solbot.clients.binance import Candle
from solbot.datastore import DataStore, _month_bounds

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


def test_bulk_backfill_respects_should_stop_between_tokens(tmp_path):
    binance = FakeBinance(klines_fn=lambda pair, since, until: [])
    store = DataStore(binance, {"candle_minutes": 1}, candles=ParquetCandleStore(tmp_path / "candles"))

    report = store.bulk_backfill(
        {MINT_A: "BTCUSD", MINT_B: "ETHUSD"}, months=1, should_stop=lambda: True
    )

    assert report.stopped_early == "cancelled"
    assert report.tokens_done == 0
