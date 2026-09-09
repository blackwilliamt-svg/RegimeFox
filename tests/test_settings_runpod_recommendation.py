"""Settings fix-up section 5: one-click "Apply this recommendation" for the
RunPod GPU-tier benchmark's recommended tier.

run_benchmark itself must never apply anything - see test_wfmc.py's
"...without_touching_runpod_gpu_type" test for that guarantee. This module
covers the separate, deliberate action: the settings page button that goes
through the exact same config.update()/audit-log path a normal settings
save does.
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


def _result_row(gpu_type: str, *, ok: bool = True) -> dict:
    """Shape of BenchmarkResult.as_dict() - what settings.html's results
    table actually iterates, so a test fixture needs every field it reads."""
    return {
        "gpu_type": gpu_type, "ok": ok, "elapsed_seconds": 12.3,
        "price_per_hour": 0.5, "cost_usd": 0.002 if ok else None,
        "combinations_evaluated": 500 if ok else None,
        "cost_per_1000_combinations": 0.004 if ok else None,
        "teardown_clean": True, "error": "" if ok else "simulated failure",
    }


def _store_benchmark_result(recommendation: str, recommended_gpu_type: str) -> None:
    from solbot.wfmc import RUNPOD_BENCHMARK_KEY

    db.kv_set(RUNPOD_BENCHMARK_KEY, {
        "ts": db.now(),
        "results": [_result_row(recommended_gpu_type)] if recommended_gpu_type else [],
        "recommendation": recommendation,
        "recommended_gpu_type": recommended_gpu_type,
    })


def test_apply_recommendation_updates_runpod_gpu_type_and_logs_an_audit_entry(client, workspace):
    # Deliberately not the DEFAULTS value ("NVIDIA RTX 4090") - this test is
    # about applying a genuine change, not the no-op path (that's the next test).
    _store_benchmark_result("NVIDIA RTX 3090 ($0.0035 per 1,000 combinations)", "NVIDIA RTX 3090")

    resp = client.post("/settings/apply-runpod-recommendation", follow_redirects=True)
    assert resp.status_code == 200
    assert b"Applied recommendation" in resp.data

    from solbot.config import get_config
    assert get_config()["runpod_gpu_type"] == "NVIDIA RTX 3090"

    row = db.connect().execute(
        "SELECT * FROM settings_audit WHERE key = 'runpod_gpu_type' ORDER BY id DESC LIMIT 1"
    ).fetchone()
    assert row is not None
    assert row["new_value"] == "NVIDIA RTX 3090"
    assert row["username"] == "trader"


def test_apply_recommendation_is_a_no_op_flash_when_already_applied(client, workspace):
    _store_benchmark_result("NVIDIA RTX 3090 ($0.0035 per 1,000 combinations)", "NVIDIA RTX 3090")
    first = client.post("/settings/apply-runpod-recommendation", follow_redirects=True)
    assert b"Applied recommendation" in first.data  # the real change happens here...

    audit_count_before = db.connect().execute(
        "SELECT COUNT(*) AS n FROM settings_audit WHERE key = 'runpod_gpu_type'"
    ).fetchone()["n"]

    resp = client.post("/settings/apply-runpod-recommendation", follow_redirects=True)
    assert b"already" in resp.data  # ...this one is the genuine no-op repeat

    audit_count_after = db.connect().execute(
        "SELECT COUNT(*) AS n FROM settings_audit WHERE key = 'runpod_gpu_type'"
    ).fetchone()["n"]
    assert audit_count_after == audit_count_before  # no redundant audit row for a no-op


def test_apply_recommendation_fails_cleanly_with_no_stored_benchmark(client, workspace):
    from solbot.config import get_config

    original = get_config()["runpod_gpu_type"]
    resp = client.post("/settings/apply-runpod-recommendation", follow_redirects=True)
    assert resp.status_code == 200
    assert b"No RunPod benchmark recommendation" in resp.data
    assert get_config()["runpod_gpu_type"] == original

    assert db.connect().execute(
        "SELECT COUNT(*) AS n FROM settings_audit WHERE key = 'runpod_gpu_type'"
    ).fetchone()["n"] == 0


def test_apply_recommendation_fails_cleanly_when_benchmark_has_no_recommendation(client, workspace):
    from solbot.config import get_config
    from solbot.wfmc import RUNPOD_BENCHMARK_KEY

    # A benchmark ran but every tier failed - no recommendation to give.
    db.kv_set(RUNPOD_BENCHMARK_KEY, {
        "ts": db.now(), "results": [_result_row("TIER-A", ok=False)],
        "recommendation": "", "recommended_gpu_type": "",
    })
    original = get_config()["runpod_gpu_type"]

    resp = client.post("/settings/apply-runpod-recommendation", follow_redirects=True)
    assert b"No RunPod benchmark recommendation" in resp.data
    assert get_config()["runpod_gpu_type"] == original
