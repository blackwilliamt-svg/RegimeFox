"""GET /api/sensitivity: the parameter-sensitivity dashboard view, backed by
the local daily WFMC store's persistent library."""
from __future__ import annotations

import pyotp
import pytest

from solopt.store import LibraryEntry, RunStore


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


def _seed_library(tmp_path, monkeypatch) -> None:
    import solbot.wfmc as wfmc

    monkeypatch.setattr(wfmc, "DAILY_STORE_PATH", str(tmp_path / "wfmc.db"))
    store = RunStore(tmp_path / "wfmc.db")
    for i, (rsi, wfe) in enumerate([(14, 0.8), (14, 0.7), (21, 0.2), (21, 0.1), (28, 0.5)]):
        store.upsert_library(
            LibraryEntry(
                fingerprint=f"fp{i}",
                params={"rsi_period": rsi, "ema_fast": 9, "ema_slow": 21},
                performance={"walk_forward_efficiency": wfe, "mean_window_return": 0.01},
            )
        )


def test_sensitivity_is_empty_without_a_library(client, workspace, tmp_path, monkeypatch):
    import solbot.wfmc as wfmc

    monkeypatch.setattr(wfmc, "DAILY_STORE_PATH", str(tmp_path / "wfmc.db"))
    body = client.get("/api/sensitivity").get_json()
    assert body["axes"] == {}
    assert body["ranked"] == []


def test_sensitivity_ranks_axes_by_score_range(client, workspace, tmp_path, monkeypatch):
    _seed_library(tmp_path, monkeypatch)

    body = client.get("/api/sensitivity").get_json()
    assert body["entries_used"] == 5
    assert "rsi_period" in body["axes"]
    assert body["ranked"][0] == "rsi_period"
    assert body["axes"]["rsi_period"]["best_value"] == 14


def test_sensitivity_never_500s_when_the_local_store_is_unreachable(client, workspace, monkeypatch):
    import solbot.wfmc as wfmc

    monkeypatch.setattr(wfmc, "DAILY_STORE_PATH", "Z:\\nonexistent\\wfmc.db")
    resp = client.get("/api/sensitivity")
    assert resp.status_code == 200
