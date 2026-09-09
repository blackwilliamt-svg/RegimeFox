"""The historical-pull page went through two fix-ups in sequence:

1. A settings fix-up gave the previously-bare "months" and "top N coins"
   number fields visible labels.
2. Combined fix-up section 6 ("replace the historical pull entirely") then
   removed both of those fields outright - the pull always covers every
   Binance.US pair's full available history now, with nothing left to
   scope. This file's first two tests were rewritten from "the fields have
   labels" to "the fields are gone" for that reason; see
   tests/test_full_history_pull.py for the new behaviour's own coverage.

The market-cap test below predates and is independent of both fix-ups: it
confirms /api/universe already surfaces real market cap data
(universe.mcap, populated straight from Jupiter's token API on every
universe refresh - solbot/universe.py's _persist()) rather than a proxy,
which pair_map()'s still-existing top_n ranking (tests/test_datastore.py)
relies on.
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


def test_the_months_field_is_gone(client, workspace):
    """Section 6 removed it entirely - there is nothing left to scope."""
    resp = client.get("/backtest")
    assert resp.status_code == 200
    body = resp.data.decode("utf-8")
    assert 'id="pullMonths"' not in body
    assert 'name="months"' not in body


def test_the_top_n_field_is_gone(client, workspace):
    resp = client.get("/backtest")
    body = resp.data.decode("utf-8")
    assert 'id="pullTopN"' not in body
    assert 'name="top_n"' not in body


def test_the_pull_form_posts_with_no_scoping_fields_at_all(client, workspace):
    resp = client.get("/backtest")
    body = resp.data.decode("utf-8")
    assert 'id="pullBtn"' in body
    assert "Start historical pull" in body


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
