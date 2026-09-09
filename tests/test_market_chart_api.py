"""Gap-closure item 6: always-visible market chart with symbol search.

The frontend (dashboard.html / app.js) reuses the existing /api/universe and
/api/candles/<mint> endpoints rather than adding parallel ones - this pins
that both already support an arbitrary universe symbol, not just one with an
open position.

Also covers the market-chart upgrade (dashboard step 2): /api/universe must
not silently truncate a universe larger than the old hardcoded LIMIT 500 (the
actual cause of "not all universe coins showing up" - the table itself is
already bounded by universe_max_tokens, up to 2000, at write time), and
/api/candles' opt-in indicator overlays (SMA/EMA/Bollinger/RSI).
"""
from __future__ import annotations

import numpy as np
import pyotp
import pytest

from solbot import db
from solbot.candlestore import ParquetCandleStore


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
        "username": "trader",
        "password": "correct-horse-battery",
        "token": pyotp.TOTP(secret).now(),
    })
    assert resp.status_code == 302
    return c


MINT = "MarketChartMintAAAAAAAAAAAAAAAAAAAAAAAAAAAA"


def _seed_universe_row(mint: str, symbol: str) -> None:
    conn = db.connect()
    conn.execute(
        "INSERT INTO universe(mint, symbol, name, decimals, liquidity_usd, "
        "volume_24h_usd, mcap, holder_count, organic_score, is_verified, "
        "first_pool_at, binance_pair, updated_at) "
        "VALUES (?,?,?,9,100000,500000,0,0,0,1,0,?,?)",
        (mint, symbol, symbol, symbol + "USDT", db.now()),
    )
    conn.commit()


def test_universe_endpoint_lists_a_seeded_symbol(client, workspace):
    _seed_universe_row(MINT, "MKTC")
    rows = client.get("/api/universe").get_json()
    symbols = {r["mint"]: r["symbol"] for r in rows}
    assert symbols.get(MINT) == "MKTC"


def test_candles_endpoint_accepts_any_universe_mint_not_just_open_positions(client, workspace):
    """No position ever exists for MINT - the market chart panel still has to
    be able to ask for its candles (empty history is fine, a 500 is not)."""
    _seed_universe_row(MINT, "MKTC")
    resp = client.get(f"/api/candles/{MINT}?limit=50")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["mint"] == MINT
    assert body["candles"] == []  # no candle history seeded, and that's fine
    assert body["positions"] == []


def test_universe_endpoint_is_not_capped_at_500(client, workspace):
    """Regression: a universe larger than the old hardcoded LIMIT 500 (legal
    whenever universe_max_tokens is raised above 500, up to its 2000 ceiling)
    must come back in full, not silently truncated."""
    for i in range(600):
        _seed_universe_row(f"Mint{i:0>39}", f"T{i}")
    rows = client.get("/api/universe").get_json()
    assert len(rows) == 600


def _seed_candles(mint: str, bars: int = 60) -> None:
    rng = np.random.default_rng(7)
    price = 1.0
    rows = []
    ts = 1_700_000_000
    for i in range(bars):
        price = max(1e-6, price * (1 + rng.normal(0, 0.004)))
        volume = max(50.0, rng.lognormal(np.log(1000.0), 0.5))
        rows.append((ts + i * 600, price * 0.999, price * 1.004, price * 0.996, price, volume))
    ParquetCandleStore().append(mint, "1m", rows)


def test_candles_endpoint_has_no_indicators_by_default(client, workspace):
    _seed_candles(MINT)
    body = client.get(f"/api/candles/{MINT}?limit=60").get_json()
    assert "indicators" not in body


def test_candles_endpoint_sma_ema_bbands_rsi_overlays(client, workspace):
    _seed_candles(MINT)
    resp = client.get(f"/api/candles/{MINT}?limit=60&sma=5&ema=5&bbands=10&rsi=5")
    assert resp.status_code == 200
    body = resp.get_json()
    n = len(body["candles"])
    assert n == 60
    ind = body["indicators"]

    assert ind["sma"] == [{"period": 5, "values": ind["sma"][0]["values"]}]
    sma_values = ind["sma"][0]["values"]
    assert len(sma_values) == n
    assert sma_values[0] is None  # not enough warm-up yet
    assert isinstance(sma_values[-1], float)

    ema_values = ind["ema"][0]["values"]
    assert ind["ema"][0]["period"] == 5
    assert len(ema_values) == n
    assert isinstance(ema_values[0], float)  # ema has no warm-up gap, unlike sma/bbands

    bb = ind["bbands"]
    assert bb["period"] == 10 and bb["num_std"] == 2.0
    assert len(bb["mid"]) == n and len(bb["upper"]) == n and len(bb["lower"]) == n
    assert bb["mid"][0] is None
    assert bb["upper"][-1] >= bb["mid"][-1] >= bb["lower"][-1]

    rsi = ind["rsi"]
    assert rsi["period"] == 5
    assert len(rsi["values"]) == n
    assert all(v is None or 0.0 <= v <= 100.0 for v in rsi["values"])


def test_candles_endpoint_indicator_periods_are_clamped_and_ignore_garbage(client, workspace):
    _seed_candles(MINT)
    resp = client.get(f"/api/candles/{MINT}?limit=60&sma=99999&ema=not-a-number,7")
    assert resp.status_code == 200
    ind = resp.get_json()["indicators"]
    assert ind["sma"][0]["period"] == 400  # clamped to the hi bound, not a 500 or garbage value
    assert ind["ema"] == [{"period": 7, "values": ind["ema"][0]["values"]}]  # bad token dropped, good one kept
