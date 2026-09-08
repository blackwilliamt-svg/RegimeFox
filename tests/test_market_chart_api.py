"""Gap-closure item 6: always-visible market chart with symbol search.

The frontend (dashboard.html / app.js) reuses the existing /api/universe and
/api/candles/<mint> endpoints rather than adding parallel ones - this pins
that both already support an arbitrary universe symbol, not just one with an
open position.
"""
from __future__ import annotations

import pyotp
import pytest

from solbot import db


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
