"""Combined fix-up section 6: full Binance.US historical pull (every pair,
full available history, no scoping by months or the routed universe) and
the per-coin coverage list.
"""
from __future__ import annotations

from solbot import db
from solbot.candlestore import ParquetCandleStore
from solbot.clients import BinanceAsset
from solbot.clients.binance import BinanceClient, Candle
from solbot.datastore import DataStore
from solbot.ratelimit import TokenBucket

from fakes import FakeBinance

MINT_A = "MintAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"


# --------------------------------------------------------------------------
# BinanceClient.klines_backward / all_bases - pure paging/dedup logic,
# monkeypatching the already-rate-limited .klines() rather than the network.
# --------------------------------------------------------------------------
def _client() -> BinanceClient:
    return BinanceClient(TokenBucket(rate=1000.0, burst=1000))


def test_klines_backward_stops_at_the_first_empty_page(monkeypatch):
    client = _client()
    pages = {
        # newest chunk first, since backward paging walks from `until`.
        900_000: [Candle(ts=900_000 - i * 60, open=1, high=1, low=1, close=1, volume=1) for i in range(3)],
    }

    def fake_klines(pair, *, start_ms, end_ms, limit=1000):
        key = end_ms // 1000
        return pages.get(key, [])

    monkeypatch.setattr(client, "klines", fake_klines)
    out = client.klines_backward("BTCUSDT", until=900_000)

    assert len(out) == 3
    assert out == sorted(out, key=lambda c: c.ts)


def test_klines_backward_keeps_paging_while_pages_are_non_empty(monkeypatch):
    client = _client()
    step = 1000 * 60  # MAX_KLINES_PER_CALL * 60 seconds
    until = 10_000_000
    calls = []

    def fake_klines(pair, *, start_ms, end_ms, limit=1000):
        calls.append((start_ms, end_ms))
        start_s = start_ms // 1000
        # Three real pages (one candle each, at the start of its window, so
        # the next call's window steps back by exactly one page), then
        # nothing further back - the pair's own listing.
        if start_s >= until - 3 * step:
            return [Candle(ts=start_s, open=1, high=1, low=1, close=1, volume=1)]
        return []

    monkeypatch.setattr(client, "klines", fake_klines)
    out = client.klines_backward("ETHUSDT", until=until)

    assert len(out) == 3
    assert len(calls) == 4   # three real pages, then the empty one that stops it


def test_klines_backward_respects_max_pages_as_a_safety_cap(monkeypatch):
    client = _client()

    def always_one_candle(pair, *, start_ms, end_ms, limit=1000):
        return [Candle(ts=end_ms // 1000 - 60, open=1, high=1, low=1, close=1, volume=1)]

    monkeypatch.setattr(client, "klines", always_one_candle)
    out = client.klines_backward("NEVERENDSUSDT", until=10_000_000, max_pages=5)

    assert len(out) == 5   # never stops on its own; the cap is what stops it


def test_all_bases_includes_stablecoins_and_leveraged_tokens_top_bases_excludes():
    client = _client()

    def fake_get(path, **kw):
        return [
            {"symbol": "BTCUSDT", "quoteVolume": "500", "lastPrice": "1", "priceChangePercent": "0", "count": 1},
            {"symbol": "USDCUSDT", "quoteVolume": "9000000", "lastPrice": "1", "priceChangePercent": "0", "count": 1},
            {"symbol": "BTCUPUSDT", "quoteVolume": "8000000", "lastPrice": "1", "priceChangePercent": "0", "count": 1},
        ]

    client.get = fake_get
    all_bases = {a.symbol for a in client.all_bases()}
    top = {a.symbol for a in client.top_bases(limit=10)}

    assert all_bases == {"BTC", "USDC", "BTCUP"}
    assert top == {"BTC"}   # USDC (stable) and BTCUP (leveraged) excluded


# --------------------------------------------------------------------------
# DataStore.full_binance_pairs / estimate_full_pull / full_history_backfill
# --------------------------------------------------------------------------
def test_full_binance_pairs_is_independent_of_the_universe_table(workspace):
    conn = workspace["conn"]
    # A routed universe of exactly one coin...
    conn.execute(
        "INSERT INTO universe(mint, symbol, binance_pair, updated_at) VALUES (?,?,?,?)",
        (MINT_A, "ROUTED", "ROUTEDUSDT", db.now()),
    )
    conn.commit()

    # ...but Binance.US itself lists three completely different bases.
    binance = FakeBinance(assets=[
        BinanceAsset(symbol="AAA", quote_volume_24h=3.0, pair="AAAUSDT"),
        BinanceAsset(symbol="BBB", quote_volume_24h=2.0, pair="BBBUSDT"),
        BinanceAsset(symbol="CCC", quote_volume_24h=1.0, pair="CCCUSDT"),
    ])
    store = DataStore(binance, {"candle_minutes": 1})

    pairs = store.full_binance_pairs()

    assert pairs == {"AAA": "AAAUSDT", "BBB": "BBBUSDT", "CCC": "CCCUSDT"}
    assert "ROUTED" not in pairs   # the routed universe never enters this at all


def test_estimate_full_pull_is_always_labelled_an_upper_bound():
    store = DataStore(None, {"binance_rps": 5.0})
    estimate = store.estimate_full_pull(50)

    assert estimate["upper_bound"] is True
    assert estimate["tokens"] == 50
    assert estimate["calls"] > 0
    assert estimate["assumed_years"] == DataStore.ASSUMED_MAX_HISTORY_YEARS


def test_full_history_backfill_pages_backward_per_pair_not_by_months(tmp_path):
    def backward(pair):
        return [Candle(ts=1000 + i * 60, open=1, high=1, low=1, close=1, volume=1) for i in range(5)]

    binance = FakeBinance(backward_fn=backward)
    store = DataStore(binance, {"candle_minutes": 1}, candles=ParquetCandleStore(tmp_path / "candles"))

    report = store.full_history_backfill({"AAA": "AAAUSDT", "BBB": "BBBUSDT"})

    assert report.tokens_done == 2
    assert report.candles_written == 10
    assert sorted(binance.backward_calls) == ["AAAUSDT", "BBBUSDT"]
    assert binance.klines_calls == []   # the old month-windowed path is never used


def test_full_history_backfill_writes_under_the_symbol_key_not_a_mint(tmp_path):
    def backward(pair):
        return [Candle(ts=1000, open=1, high=1, low=1, close=1, volume=1)]

    binance = FakeBinance(backward_fn=backward)
    store = DataStore(binance, {"candle_minutes": 1}, candles=ParquetCandleStore(tmp_path / "candles"))

    store.full_history_backfill({"AAA": "AAAUSDT"})

    assert store.candles.coverage("AAA", "1m")["bars"] == 1


def test_full_history_backfill_skips_a_failed_pair_but_keeps_going(tmp_path):
    calls = []

    def backward(pair):
        calls.append(pair)
        if pair == "AAAUSDT":
            raise Exception("boom")  # ApiError-shaped in the real client
        return [Candle(ts=1000, open=1, high=1, low=1, close=1, volume=1)]

    from solbot.clients.base import ApiError

    def backward_or_raise(pair):
        if pair == "AAAUSDT":
            raise ApiError("binance: rate limited", provider="binance")
        return backward(pair)

    binance = FakeBinance(backward_fn=backward_or_raise)
    store = DataStore(binance, {"candle_minutes": 1}, candles=ParquetCandleStore(tmp_path / "candles"))

    report = store.full_history_backfill({"AAA": "AAAUSDT", "BBB": "BBBUSDT"})

    assert report.tokens_failed == 1
    assert report.tokens_done == 1
    assert report.candles_written == 1


def test_a_full_history_pull_does_not_touch_the_universe_table_or_other_jobs(workspace):
    """The daily incremental / backtest / walk-forward pair_map() calls must
    see exactly the same routed universe before and after a full pull."""
    conn = workspace["conn"]
    conn.execute(
        "INSERT INTO universe(mint, symbol, binance_pair, updated_at) VALUES (?,?,?,?)",
        (MINT_A, "ROUTED", "ROUTEDUSDT", db.now()),
    )
    conn.commit()

    binance = FakeBinance(assets=[BinanceAsset(symbol="XYZ", quote_volume_24h=1.0, pair="XYZUSDT")])
    store = DataStore(binance, {"candle_minutes": 1})

    before = dict(store.pair_map(conn))
    store.full_history_backfill(store.full_binance_pairs())
    after = dict(store.pair_map(conn))

    assert before == after == {MINT_A: "ROUTEDUSDT"}
    assert conn.execute("SELECT COUNT(*) AS n FROM universe").fetchone()["n"] == 1


# --------------------------------------------------------------------------
# ParquetCandleStore.all_mints / per_mint_coverage
# --------------------------------------------------------------------------
def test_per_mint_coverage_lists_every_symbol_with_history_on_disk(tmp_path):
    store = ParquetCandleStore(tmp_path / "candles")
    store.append("AAA", "1m", [(1000, 1, 1, 1, 1, 10), (1060, 1, 1, 1, 1, 10)])
    store.append("BBB", "1m", [(2000, 1, 1, 1, 1, 10)])

    rows = {r["mint"]: r for r in store.per_mint_coverage("1m")}

    assert set(rows) == {"AAA", "BBB"}
    assert rows["AAA"]["bars"] == 2
    assert rows["BBB"]["bars"] == 1


def test_per_mint_coverage_excludes_mints_with_no_stored_candles(tmp_path):
    store = ParquetCandleStore(tmp_path / "candles")
    store.append("AAA", "1m", [(1000, 1, 1, 1, 1, 10)])
    (tmp_path / "candles" / "1m" / "EMPTY").mkdir(parents=True)

    rows = store.per_mint_coverage("1m")

    assert {r["mint"] for r in rows} == {"AAA"}


# --------------------------------------------------------------------------
# GET /api/candles/coverage
# --------------------------------------------------------------------------
import pyotp
import pytest


@pytest.fixture
def app(workspace, monkeypatch):
    monkeypatch.setenv("FLASK_SECRET_KEY", "x" * 64)
    monkeypatch.setenv("SOLBOT_INSECURE_COOKIES", "1")
    from solbot.config import reset_config_for_tests
    from solbot.web import create_app

    reset_config_for_tests(workspace["config"])
    application = create_app()
    application.config["TESTING"] = True
    return application


@pytest.fixture
def client(app):
    from solbot.web.auth import confirm_totp, create_user

    secret, _codes = create_user("trader", "correct-horse-battery")
    confirm_totp("trader")
    c = app.test_client()
    resp = c.post("/login", data={
        "username": "trader", "password": "correct-horse-battery",
        "token": pyotp.TOTP(secret).now(),
    })
    assert resp.status_code == 302
    return c


def test_coverage_endpoint_includes_coins_not_in_the_routed_universe(client, workspace):
    candles = ParquetCandleStore(workspace["candles_dir"])
    candles.append("NOTROUTED", "1m", [(1000, 1, 1, 1, 1, 10), (1060, 1, 1, 1, 1, 10)])
    candles.append(MINT_A, "1m", [(2000, 1, 1, 1, 1, 10)])
    conn = workspace["conn"]
    conn.execute(
        "INSERT INTO universe(mint, symbol, updated_at) VALUES (?,?,?)",
        (MINT_A, "ROUTED", db.now()),
    )
    conn.commit()

    body = client.get("/api/candles/coverage").get_json()
    by_mint = {r["mint"]: r for r in body}

    assert set(by_mint) == {"NOTROUTED", MINT_A}
    assert by_mint["NOTROUTED"]["bars"] == 2
    assert by_mint["NOTROUTED"]["symbol"] == "NOTROUTED"   # no universe row -> falls back to the key itself
    assert by_mint[MINT_A]["symbol"] == "ROUTED"           # a universe row -> its real symbol


def test_coverage_endpoint_excludes_mints_with_nothing_stored(client, workspace):
    candles = ParquetCandleStore(workspace["candles_dir"])
    candles.append("HASDATA", "1m", [(1000, 1, 1, 1, 1, 10)])
    (workspace["candles_dir"] / "1m" / "NODATA").mkdir(parents=True)

    body = client.get("/api/candles/coverage").get_json()

    assert {r["mint"] for r in body} == {"HASDATA"}
