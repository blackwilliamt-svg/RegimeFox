"""GET /api/backfill/estimate: an upper-bound cost of the full Binance.US
history pull (dashboard fix-up section 6 - "replace the historical pull
entirely") - every currently-listed pair, no months/scoping parameters at
all, since there is nothing left to scope. See
DataStore.estimate_full_pull().
"""
from __future__ import annotations

import pyotp
import pytest

from fakes import fake_clients


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


def _wire_fake_binance(app, token_count: int) -> None:
    from solbot.clients import BinanceAsset

    assets = [
        BinanceAsset(symbol=f"SYM{i}", quote_volume_24h=float(i), pair=f"SYM{i}USDT")
        for i in range(token_count)
    ]
    app.config["SOLBOT_CLIENTS"] = fake_clients(binance_assets=assets)


def test_estimate_reflects_the_full_binance_pair_count_not_the_universe(app, client, workspace):
    """The routed universe is irrelevant here - the estimate is sized off
    every Binance.US pair, independent of what's currently routed."""
    from solbot import db

    db.connect().execute(
        "INSERT INTO universe(mint, symbol, updated_at) VALUES (?,?,?)",
        ("MintRouted", "ROUTED", db.now()),
    )
    db.connect().commit()
    _wire_fake_binance(app, 7)

    body = client.get("/api/backfill/estimate").get_json()

    assert body["tokens"] == 7
    assert body["calls"] > 0
    assert body["seconds_estimate"] >= 0


def test_estimate_is_always_labelled_an_upper_bound(app, client, workspace):
    _wire_fake_binance(app, 1)

    body = client.get("/api/backfill/estimate").get_json()

    assert body["upper_bound"] is True
    assert "assumed_years" in body


def test_estimate_flags_a_large_pair_count_that_misses_the_12_hour_target(app, client, workspace):
    _wire_fake_binance(app, 150)
    body = client.get("/api/backfill/estimate").get_json()
    assert body["hours_estimate"] > 12.0


def test_estimate_reports_gracefully_when_binance_is_unreachable(app, client, workspace):
    class _BoomBinance:
        def all_bases(self):
            raise RuntimeError("binance.us is unreachable")

    fc = fake_clients()
    fc.binance = _BoomBinance()
    app.config["SOLBOT_CLIENTS"] = fc

    resp = client.get("/api/backfill/estimate")
    assert resp.status_code == 200
    assert "error" in resp.get_json()
