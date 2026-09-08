"""Retention for WF/MC output in solbot's own database (spec 6c/7).

Unlike raw candles (kept indefinitely, spec 3a), walk-forward/Monte Carlo
output is the fastest-growing storage component and gets its own purge
policy - this covers db.prune()'s half of it (solbot's `optimizer_runs`/
`param_bundles` tables); solopt.store.RunStore's half is covered in
test_optimizer.py.
"""
from __future__ import annotations

import json

from solbot import db


def _seed_run(conn, run_id: int, *, age_days: float, status: str = "done") -> None:
    updated_at = db.now() - int(age_days * 86400)
    conn.execute(
        "INSERT INTO optimizer_runs(run_id, started_at, updated_at, status) VALUES (?,?,?,?)",
        (run_id, updated_at, updated_at, status),
    )
    conn.execute(
        "INSERT INTO optimizer_feed(run_id, remote_id, ts, level, message) "
        "VALUES (?,?,?,?,?)",
        (run_id, 1, updated_at, "info", "a line"),
    )


def test_prune_removes_old_finished_optimizer_runs_and_their_feed(workspace):
    conn = workspace["conn"]
    _seed_run(conn, 1, age_days=200)
    _seed_run(conn, 2, age_days=5)

    deleted = db.prune(tick_hours=48, event_days=45, wfmc_days=180, conn=conn)

    assert deleted["optimizer_runs"] == 1
    assert deleted["optimizer_feed"] == 1
    remaining = {r["run_id"] for r in conn.execute("SELECT run_id FROM optimizer_runs").fetchall()}
    assert remaining == {2}


def test_prune_never_touches_a_still_running_optimizer_run(workspace):
    conn = workspace["conn"]
    _seed_run(conn, 1, age_days=400, status="running")

    deleted = db.prune(tick_hours=48, event_days=45, wfmc_days=180, conn=conn)

    assert deleted["optimizer_runs"] == 0
    assert conn.execute("SELECT 1 FROM optimizer_runs WHERE run_id = 1").fetchone()


def test_prune_falls_back_to_event_days_when_wfmc_days_is_not_given(workspace):
    conn = workspace["conn"]
    _seed_run(conn, 1, age_days=100)

    deleted = db.prune(tick_hours=48, event_days=45, conn=conn)  # no wfmc_days
    assert deleted["optimizer_runs"] == 1


def test_prune_drops_old_rejected_bundles_but_keeps_the_active_shadow_one(workspace):
    conn = workspace["conn"]
    old_ts = db.now() - 200 * 86400
    conn.execute(
        "INSERT INTO param_bundles(fingerprint, received_at, payload, status) "
        "VALUES (?,?,?,?)",
        ("rejected-old", old_ts, json.dumps({}), "rejected"),
    )
    conn.execute(
        "INSERT INTO param_bundles(fingerprint, received_at, payload, status) "
        "VALUES (?,?,?,?)",
        ("shadow-old", old_ts, json.dumps({}), "shadow"),
    )

    deleted = db.prune(tick_hours=48, event_days=45, wfmc_days=180, conn=conn)

    assert deleted["param_bundles"] == 1
    remaining = {r["fingerprint"] for r in conn.execute("SELECT fingerprint FROM param_bundles")}
    assert remaining == {"shadow-old"}


# --------------------------------------------------------------------------
# settings_audit's reason column (spec 6b) - the audit trail for both a
# manual dashboard edit and a bot-made (auto-promoted) change.
# --------------------------------------------------------------------------
def test_settings_audit_carries_a_reason_for_a_bot_made_change(workspace, cfg):
    from solbot import paramsync

    conn = workspace["conn"]
    verdict = paramsync.PromotionVerdict(
        promoted=True, reason="12 clean days, +40% margin, 96% confidence",
    )
    db.kv_set(paramsync.SHADOW_KEY, {"volume_spike_multiple": 3.3}, conn)
    paramsync.promote(cfg, verdict, conn)

    row = conn.execute(
        "SELECT * FROM settings_audit WHERE key = 'volume_spike_multiple'"
    ).fetchone()
    assert row["username"] == "auto-promotion"
    assert "clean days" in row["reason"]


def test_settings_audit_reason_is_blank_for_an_ordinary_manual_edit(workspace, cfg):
    conn = workspace["conn"]
    conn.execute(
        "INSERT INTO settings_audit(ts, username, key, old_value, new_value) "
        "VALUES (?,?,?,?,?)",
        (db.now(), "trader", "max_slippage_pct", "0.5", "0.75"),
    )
    row = conn.execute(
        "SELECT * FROM settings_audit WHERE key = 'max_slippage_pct'"
    ).fetchone()
    assert row["reason"] is None
