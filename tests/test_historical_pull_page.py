"""Settings fix-up section 7: the historical-pull form's previously-bare
"months" number field gets a visible label, and market-cap ranking (added
in section 6) is confirmed to use real data already flowing through
/api/universe rather than a proxy.

Real market cap already existed in this codebase before this fix-up:
universe.mcap (solbot/db.py's schema) is populated straight from Jupiter's
token API on every universe refresh (solbot/universe.py's _persist()) -
the same source liquidity_usd/volume_24h_usd already come from. Section
6's top_n ranking (tests/test_datastore.py) uses that column directly, so
there is no CoinGecko integration or new column here - see that section's
commit message for the full explanation.
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


def test_months_field_has_a_visible_label(client, workspace):
    resp = client.get("/backtest")
    assert resp.status_code == 200
    body = resp.data.decode("utf-8")
    # A <label for="pullMonths"> wrapping/preceding the input, not just a
    # bare <input id="pullMonths">.
    assert 'for="pullMonths"' in body
    assert "Months of history" in body


def test_top_n_field_also_has_a_visible_label(client, workspace):
    resp = client.get("/backtest")
    body = resp.data.decode("utf-8")
    assert 'for="pullTopN"' in body
    assert "Top N coins" in body


def test_universe_api_already_surfaces_market_cap(client, workspace):
    """No CoinGecko/new-column work was needed (see module docstring) -
    universe.mcap already flows straight through /api/universe's SELECT *."""
    conn = db.connect()
    conn.execute(
        "INSERT INTO universe(mint, symbol, binance_pair, mcap, volume_24h_usd, updated_at) "
        "VALUES (?,?,?,?,?,?)",
        ("MintMcapCheckAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA", "MCAP", "MCAPUSDT", 12_345_678.0, 500_000.0, db.now()),
    )
    conn.commit()

    rows = client.get("/api/universe").get_json()
    row = next(r for r in rows if r["mint"] == "MintMcapCheckAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
    assert row["mcap"] == 12_345_678.0
