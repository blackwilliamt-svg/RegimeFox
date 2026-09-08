"""The optimizer ingest endpoints (feed, run, bundle) and the walk-forward tab.

There is nothing left to download here (spec 5) - the daily run reads the
droplet's own Parquet files directly, and the monthly RunPod run has its data
shipped to it before it starts. What a RunPod worker still talks to the
droplet for is reporting progress and handing over a finished bundle, so most
of these tests are about what that reporting path refuses: no token, the
wrong token, a malformed bundle.
"""
from __future__ import annotations

import time

import pytest

from solbot import db, paramsync, secrets_store

TOKEN = "a-long-enough-bulk-data-token-for-testing-purposes"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture
def app(workspace, monkeypatch):
    monkeypatch.setenv("FLASK_SECRET_KEY", "x" * 64)
    monkeypatch.setenv("SOLBOT_INSECURE_COOKIES", "1")
    monkeypatch.setenv("BULK_DATA_TOKEN", TOKEN)
    # Needed for the rotation test: without it the at-rest encryption refuses,
    # which is correct behaviour but not what that test is exercising.
    monkeypatch.setenv("SECRET_ENCRYPTION_KEY", "b" * 64)
    from solbot.config import reset_config_for_tests
    from solbot.web import create_app

    reset_config_for_tests(workspace["config"])
    application = create_app()
    application.config["TESTING"] = True
    return application


def good_bundle(**overrides) -> dict:
    payload = {
        "schema": 1,
        "generated_at": int(time.time()),
        "global": {
            "volume_spike_multiple": 2.6, "momentum_min_pct": 0.012,
            "ema_fast": 9, "ema_slow": 21, "stop_atr_mult": 1.4,
            "rr_min": 2.0, "rr_max": 4.0,
        },
        "per_symbol": {},
        "risk": {"p5_max_drawdown": 0.18},
        "evidence": {"walk_forward": {"counted_windows": 8, "profitable_windows": 7}},
        "gates": {
            "accepted": True, "walk_forward_accepted": True, "overfit": False,
            "fragile": False, "fragile_windows": [], "reasons": [],
            "shadow_min_days": 15,
        },
    }
    payload.update(overrides)
    return payload


# --------------------------------------------------------------------------
# Auth
# --------------------------------------------------------------------------
@pytest.mark.parametrize("path", ["/api/optimizer/execution"])
def test_ingest_endpoints_refuse_an_unauthenticated_get(app, path):
    resp = app.test_client().get(path)
    assert resp.status_code == 401
    assert "token" in resp.get_json()["error"]


def test_ingest_endpoints_refuse_an_unauthenticated_post(app):
    resp = app.test_client().post("/api/optimizer/bundle", json=good_bundle())
    assert resp.status_code == 401


def test_ingest_refuses_the_wrong_token(app):
    resp = app.test_client().get(
        "/api/optimizer/execution", headers={"Authorization": "Bearer not-the-token"}
    )
    assert resp.status_code == 401


def test_a_dashboard_session_does_not_grant_ingest_access(app, workspace):
    """The two credentials are separate on purpose.

    A logged-in browser can halt trading; a worker token cannot. Letting a
    session stand in for the token would quietly merge those two privilege
    levels.
    """
    import pyotp
    from solbot.web.auth import confirm_totp, create_user

    secret, _ = create_user("trader", "correct-horse-battery")
    confirm_totp("trader")
    client = app.test_client()
    client.post(
        "/login",
        data={
            "username": "trader", "password": "correct-horse-battery",
            "token": pyotp.TOTP(secret).now(),
        },
    )

    assert client.get("/api/status").status_code == 200
    assert client.get("/api/optimizer/execution").status_code == 401


def test_ingest_can_be_switched_off(app, workspace):
    from solbot.config import get_config

    get_config().update({"runpod_ingest_enabled": False})
    resp = app.test_client().get("/api/optimizer/execution", headers=AUTH)

    assert resp.status_code == 403
    assert "disabled" in resp.get_json()["error"]


def test_a_valid_token_gets_through(app, workspace):
    resp = app.test_client().get("/api/optimizer/execution", headers=AUTH)
    assert resp.status_code == 200


def test_a_rotated_token_takes_precedence_over_the_environment(app, workspace):
    """Rotating from the dashboard must invalidate the environment's token."""
    from solbot.config import get_config

    rotated = "a-freshly-rotated-token-that-is-long-enough"
    secrets_store.store_key(
        "bulk_data", rotated, get_config().secrets.secret_encryption_key
    )
    client = app.test_client()

    assert client.get("/api/optimizer/execution", headers=AUTH).status_code == 401
    assert client.get(
        "/api/optimizer/execution", headers={"Authorization": f"Bearer {rotated}"}
    ).status_code == 200


# --------------------------------------------------------------------------
# Execution profile
# --------------------------------------------------------------------------
def test_execution_profile_says_when_it_is_assuming(app, workspace):
    resp = app.test_client().get("/api/optimizer/execution", headers=AUTH)
    body = resp.get_json()

    assert body["source"] == "assumed"
    assert "30 are needed" in body["note"]


def test_execution_profile_uses_observed_fills_once_there_are_enough(app, workspace):
    conn = workspace["conn"]
    for i in range(60):
        db.record_fill(
            instance="paper", mint="AAA", side="buy", notional_usd=400.0,
            slippage_pct=0.20 + (i % 5) * 0.01, fee_usd=1.0, simulated=True, conn=conn,
        )

    body = app.test_client().get("/api/optimizer/execution", headers=AUTH).get_json()
    assert body["source"] == "observed"
    assert body["samples"] == 60
    assert body["slippage_stdev_pct"] > 0
    assert body["fee_mean_pct"] == pytest.approx(0.25)


# --------------------------------------------------------------------------
# Feed / run ingest
# --------------------------------------------------------------------------
def test_feed_ingest_stores_lines_and_ignores_replays(app, workspace):
    client = app.test_client()
    lines = [
        {"id": 1, "ts": db.now(), "level": "info", "message": "Loaded 12 coins."},
        {"id": 2, "ts": db.now(), "level": "warn", "message": "Window 3 did not hold up."},
    ]
    for _ in range(2):
        resp = client.post(
            "/api/optimizer/feed", json={"run_id": 7, "lines": lines}, headers=AUTH
        )
        assert resp.status_code == 200

    rows = workspace["conn"].execute("SELECT * FROM optimizer_feed").fetchall()
    assert len(rows) == 2, "a resent batch must not duplicate"
    assert rows[1]["level"] == "warn"


def test_feed_ingest_rejects_an_oversized_batch(app):
    lines = [{"id": i, "message": "x"} for i in range(500)]
    resp = app.test_client().post(
        "/api/optimizer/feed", json={"run_id": 1, "lines": lines}, headers=AUTH
    )
    assert resp.status_code == 413


def test_feed_ingest_requires_a_run_id(app):
    resp = app.test_client().post(
        "/api/optimizer/feed", json={"lines": []}, headers=AUTH
    )
    assert resp.status_code == 400


def test_a_finished_run_announces_itself_exactly_once(app, workspace):
    client = app.test_client()
    summary = {
        "accepted": True, "counted_windows": 8, "profitable_windows": 7,
        "walk_forward_efficiency": 0.61,
    }
    for _ in range(3):
        client.post(
            "/api/optimizer/run",
            json={"run_id": 11, "status": "done", "summary": summary},
            headers=AUTH,
        )

    events = workspace["conn"].execute(
        "SELECT * FROM events WHERE message LIKE '%Walk-forward run%'"
    ).fetchall()
    assert len(events) == 1, "the completion notice fires once, not on every post"
    assert "promotable parameter set" in events[0]["message"]


def test_a_run_that_produced_nothing_says_so(app, workspace):
    app.test_client().post(
        "/api/optimizer/run",
        json={
            "run_id": 12, "status": "done",
            "summary": {"accepted": False, "reasons": ["only 40% of windows profitable"]},
        },
        headers=AUTH,
    )
    events = workspace["conn"].execute(
        "SELECT * FROM events WHERE message LIKE '%Walk-forward run%'"
    ).fetchall()

    assert "without a promotable set" in events[0]["message"]
    assert "40%" in events[0]["message"]


# --------------------------------------------------------------------------
# Bundle ingest - a RunPod worker handing over a finished, promotable set
# --------------------------------------------------------------------------
def test_bundle_ingest_accepts_and_installs_a_valid_bundle(app, workspace):
    resp = app.test_client().post(
        "/api/optimizer/bundle", json=good_bundle(), headers=AUTH
    )
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["ok"] and body["accepted"] and body["installed"]

    row = workspace["conn"].execute("SELECT * FROM param_bundles").fetchone()
    assert row["status"] == paramsync.STATUS_SHADOW
    assert row["source_commit"].startswith("runpod:")


def test_bundle_ingest_re_validates_rather_than_trusting_the_worker(app, workspace):
    """The worker's own claim that its set is promotable is not trusted."""
    resp = app.test_client().post(
        "/api/optimizer/bundle",
        json=good_bundle(gates={"accepted": False, "reasons": ["fabricated"]}),
        headers=AUTH,
    )
    assert resp.status_code == 200
    body = resp.get_json()
    assert not body["accepted"]
    assert not body["installed"]


def test_bundle_ingest_refuses_a_malformed_payload(app):
    resp = app.test_client().post(
        "/api/optimizer/bundle", json={"schema": 999}, headers=AUTH
    )
    assert resp.status_code == 400


# --------------------------------------------------------------------------
# The dashboard views
# --------------------------------------------------------------------------
@pytest.fixture
def signed_in(app, workspace):
    import pyotp
    from solbot.web.auth import confirm_totp, create_user

    secret, _ = create_user("trader", "correct-horse-battery")
    confirm_totp("trader")
    client = app.test_client()
    client.post(
        "/login",
        data={
            "username": "trader", "password": "correct-horse-battery",
            "token": pyotp.TOTP(secret).now(),
        },
    )
    return client


def test_every_dashboard_page_renders(signed_in):
    """Jinja errors only surface at render time, so each page is fetched."""
    for path in ("/", "/trades", "/backtest", "/walkforward", "/tax", "/settings"):
        resp = signed_in.get(path)
        assert resp.status_code == 200, f"{path} returned {resp.status_code}"


def test_the_settings_page_renders_the_new_setting_shapes(signed_in):
    body = signed_in.get("/settings").get_data(as_text=True)

    assert 'name="sizing_mode"' in body, "the enum renders as a select"
    assert 'name="confluence_timeframes"' in body, "the list renders as text"
    assert 'name="runpod_gpu_type"' in body, "free-text settings render"
    assert 'name="regime_allowed"' in body
    assert "genBulkToken" in body, "the token generator is offered"


def test_walk_forward_api_is_empty_before_any_run(signed_in):
    body = signed_in.get("/api/walkforward").get_json()
    assert body["run"] is None
    assert body["runs"] == []


def test_walk_forward_api_mirrors_a_reported_run(signed_in, workspace):
    summary = {
        "accepted": True, "counted_windows": 6, "profitable_windows": 5,
        "walk_forward_efficiency": 0.58, "overfit": False,
    }
    signed_in.post(
        "/api/optimizer/run",
        json={
            "run_id": 3, "status": "done", "summary": summary,
            "coverage": {"symbols": 20, "days": 180},
            "monte_carlo": {
                "p5_max_drawdown": 0.21, "median_return": 0.4,
                "drawdown_histogram": {"edges": [0, 0.1, 0.2], "counts": [5, 3]},
            },
            "stress": [{"name": "crash 2024-03-12", "passed": True, "note": "survived"}],
        },
        headers=AUTH,
    )

    body = signed_in.get("/api/walkforward").get_json()
    run = body["run"]
    assert run["run_id"] == 3
    assert run["summary"]["walk_forward_efficiency"] == 0.58
    assert run["monte_carlo"]["p5_max_drawdown"] == 0.21
    assert run["monte_carlo"]["drawdown_histogram"]["counts"] == [5, 3]
    assert run["stress"][0]["passed"] is True


def test_walk_forward_feed_api_pages_from_a_cursor(signed_in, workspace):
    signed_in.post(
        "/api/optimizer/feed",
        json={
            "run_id": 5,
            "lines": [
                {"id": i, "ts": db.now(), "level": "info", "message": f"line {i}"}
                for i in range(1, 6)
            ],
        },
        headers=AUTH,
    )

    first = signed_in.get("/api/walkforward/feed").get_json()
    assert len(first["lines"]) == 5

    cursor = first["lines"][2]["id"]
    rest = signed_in.get(f"/api/walkforward/feed?since_id={cursor}").get_json()
    assert len(rest["lines"]) == 2


def test_drift_and_paramsync_apis_answer_before_any_data(signed_in):
    drift_body = signed_in.get("/api/drift").get_json()
    assert drift_body == {"latest": {}, "samples": []}

    sync_body = signed_in.get("/api/paramsync").get_json()
    assert sync_body["bundles"] == []
    assert sync_body["promotions"] == []
    assert sync_body["last_bundle"] == {}


def test_paramsync_api_reports_the_last_bundle_and_its_source(signed_in):
    signed_in.post("/api/optimizer/bundle", json=good_bundle(), headers=AUTH)
    body = signed_in.get("/api/paramsync").get_json()

    assert len(body["bundles"]) == 1
    assert body["last_bundle"]["source"].startswith("runpod:")


def test_review_api_reports_recent_gate_activity(signed_in, workspace):
    body = signed_in.get("/api/review").get_json()
    assert "market" in body
    assert body["recent"] == []
