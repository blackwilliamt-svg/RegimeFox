"""GET /api/backfill/estimate: the real (not guessed) cost of a historical
pull, so the dashboard can warn an operator before they commit to one that
would miss the 12-hour deep-history target - see DataStore.estimate_pull()."""
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
        "username": "trader", "password": "correct-horse-battery",
        "token": pyotp.TOTP(secret).now(),
    })
    assert resp.status_code == 302
    return c


def _seed_universe(n: int) -> None:
    conn = db.connect()
    for i in range(n):
        conn.execute(
            "INSERT INTO universe(mint, symbol, updated_at) VALUES (?,?,?)",
            (f"Mint{i:03d}", f"SYM{i}", db.now()),
        )
    conn.commit()


def test_estimate_reflects_the_current_universe_size(client, workspace):
    _seed_universe(7)

    body = client.get("/api/backfill/estimate?months=3").get_json()

    assert body["tokens"] == 7
    assert body["months"] == 3
    assert body["calls"] > 0
    assert body["seconds_estimate"] >= 0


def test_estimate_defaults_to_the_configured_bulk_backfill_months(client, workspace):
    from solbot.config import DEFAULTS

    _seed_universe(1)

    body = client.get("/api/backfill/estimate").get_json()

    assert body["months"] == DEFAULTS["bulk_backfill_months"]


def test_estimate_clamps_an_out_of_range_months_value(client, workspace):
    _seed_universe(1)

    body = client.get("/api/backfill/estimate?months=500").get_json()
    assert body["months"] == 96


def test_estimate_flags_when_the_12_hour_target_is_missed(client, workspace):
    _seed_universe(100)

    body = client.get("/api/backfill/estimate?months=96").get_json()
    assert body["target_hours"] == 12.0
    assert body["within_target"] is False
    assert body["hours_estimate"] > 12.0
