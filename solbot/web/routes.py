"""Dashboard pages and control actions.

Control actions (start/stop, kill switch, manual close, backtest, historical
pull) enqueue a command for the worker rather than acting directly - the web
process never touches a position. Settings changes are the one exception: they
are written to config.json here, validated first, and the worker picks them up
on its next cycle via the mtime check.

Every manual action is logged to the trade journal and event feed, tagged so it
is distinguishable from an automated one in the history and exports.
"""
from __future__ import annotations

import logging
from typing import Any

from flask import (
    Blueprint, Response, current_app, flash, redirect, render_template, request,
    url_for,
)

from .. import db, exports, risk
from ..config import ConfigError
from ..recovery import KEY_RECONCILE_HALT, reconcile_halted
from .. import secrets_store
from .auth import current_user

log = logging.getLogger(__name__)

bp = Blueprint("dashboard", __name__)


def cfg() -> Any:
    return current_app.config["SOLBOT_CONFIG"]


def _enqueue(command: str, payload: dict[str, Any] | None = None) -> int:
    return db.enqueue_command(command, payload, requested_by=current_user() or "dashboard")


def _worker_alive() -> tuple[bool, int]:
    hb = db.kv_get("worker_heartbeat", {}) or {}
    ts = int(hb.get("ts", 0))
    age = db.now() - ts if ts else 10**9
    return age < 60, age


# --------------------------------------------------------------------------
# pages
# --------------------------------------------------------------------------
@bp.route("/")
def index():
    alive, age = _worker_alive()
    return render_template(
        "dashboard.html",
        worker_alive=alive,
        worker_age=age,
        mode=cfg()["trading_mode"],
        kill_switch=risk.kill_switch_state(),
        circuit=risk.circuit_state(),
        reconcile_halt=reconcile_halted(),
        running=bool(db.kv_get("engine_run", True)),
    )


@bp.route("/trades")
def trades():
    instance = request.args.get("instance") or None
    rows = exports.fetch_trades(instance=instance)
    return render_template(
        "trades.html",
        trades=list(reversed(rows)),
        instance=instance or "all",
        instances=db.INSTANCES,
    )


@bp.route("/backtest")
def backtest():
    from ..datastore import JOB_DAILY_INCREMENTAL

    conn = db.connect()
    runs = conn.execute(
        "SELECT * FROM backtest_runs ORDER BY ts DESC LIMIT 30"
    ).fetchall()
    return render_template(
        "backtest.html",
        runs=runs,
        pull=db.get_progress("historical_pull"),
        daily_pull=db.get_progress(JOB_DAILY_INCREMENTAL),
        backtest_progress=db.get_progress("backtest"),
    )


@bp.route("/walkforward")
def walkforward():
    """The walk-forward tab. Every panel on it is populated by the JSON API.

    The run happens on the operator's PC and reports in over the optimizer
    endpoints, so there is nothing to render server-side that would not be stale
    by the time the page loaded.
    """
    return render_template(
        "walkforward.html", mode=cfg()["trading_mode"],
        regime_pass_progress=db.get_progress("regime_pass"),
    )


@bp.route("/settings")
def settings():
    from ..wfmc import RUNPOD_BENCHMARK_JOB, RUNPOD_BENCHMARK_KEY

    config = cfg()
    return render_template(
        "settings.html",
        values=config.as_dict(),
        spec=_spec_rows(),
        keys=secrets_store.key_status(config.secrets, config.secrets.secret_encryption_key),
        audit=db.connect()
        .execute("SELECT * FROM settings_audit ORDER BY id DESC LIMIT 40")
        .fetchall(),
        runpod_benchmark=db.kv_get(RUNPOD_BENCHMARK_KEY, {}),
        runpod_benchmark_progress=db.get_progress(RUNPOD_BENCHMARK_JOB),
    )


def _spec_rows() -> list[dict[str, Any]]:
    """Every editable setting, with the shape the settings page should render.

    Kept derived from the config module rather than duplicated in the template,
    so a new tunable appears on the page by virtue of existing.
    """
    from ..config import BOOLS, ENUMS, LISTS, SPEC, STRINGS

    rows = []
    for key, (typ, low, high) in sorted(SPEC.items()):
        rows.append(
            {"key": key, "type": typ.__name__, "min": low, "max": high, "kind": "number"}
        )
    for key in sorted(BOOLS):
        rows.append({"key": key, "type": "bool", "min": None, "max": None, "kind": "bool"})
    for key, max_len in sorted(STRINGS.items()):
        rows.append(
            {"key": key, "type": "str", "min": None, "max": max_len, "kind": "text"}
        )
    for key, (typ, low, high, max_len) in sorted(LISTS.items()):
        rows.append(
            {
                "key": key, "type": f"list[{typ.__name__}]", "min": low, "max": high,
                "kind": "list", "max_items": max_len,
            }
        )
    # trading_mode is excluded from EDITABLE on purpose - going live is a
    # separate, guarded action, not a dropdown on the settings page.
    for key, options in sorted(ENUMS.items()):
        if key == "trading_mode":
            continue
        rows.append(
            {
                "key": key, "type": "enum", "min": None, "max": None,
                "kind": "enum", "options": sorted(options),
            }
        )
    return rows


# --------------------------------------------------------------------------
# control actions
# --------------------------------------------------------------------------
@bp.route("/control/<action>", methods=["POST"])
def control(action: str):
    allowed = {
        "start": "Bot start requested.",
        "stop": "Bot stop requested.",
        "kill_switch": "Kill switch engaged.",
        "release_kill_switch": "Kill switch released.",
        "reset_circuit": "Circuit breaker reset requested.",
        "clear_reconcile_halt": "Reconciliation halt clear requested.",
        "refresh_universe": "Universe refresh requested.",
    }
    if action not in allowed:
        flash(f"Unknown action: {action}", "error")
        return redirect(url_for("dashboard.index"))

    # The kill switch is the one action that must not depend on a live worker -
    # write the state directly as well as queueing, so it takes effect even if
    # the worker is wedged.
    if action == "kill_switch":
        if request.form.get("confirm") != "yes":
            flash("Kill switch not engaged - confirmation was missing.", "error")
            return redirect(url_for("dashboard.index"))
        risk.engage_kill_switch(
            request.form.get("reason") or "manual dashboard trigger",
            current_user() or "dashboard",
        )
    elif action == "release_kill_switch":
        risk.release_kill_switch(current_user() or "dashboard")

    _enqueue(action, {"reason": request.form.get("reason", "")})
    flash(allowed[action], "success")
    return redirect(request.referrer or url_for("dashboard.index"))


@bp.route("/positions/<int:position_id>/close", methods=["POST"])
def close_position(position_id: int):
    row = db.connect().execute(
        "SELECT * FROM positions WHERE id = ? AND status = 'open'", (position_id,)
    ).fetchone()
    if row is None:
        flash("That position is not open.", "error")
        return redirect(url_for("dashboard.index"))

    _enqueue("close_position", {"position_id": position_id})
    db.log_event(
        f"Manual close requested for {row['symbol'] or row['mint'][:8]} by "
        f"{current_user() or 'dashboard'}.",
        level="warn",
        category="trade",
        instance=row["instance"],
        mint=row["mint"],
    )
    flash("Close requested - the worker will execute it on its next cycle.", "success")
    return redirect(url_for("dashboard.index"))


@bp.route("/backtest/run", methods=["POST"])
def run_backtest():
    days = request.form.get("days", type=int)
    _enqueue("run_backtest", {"days": days} if days else {})
    flash("Backtest queued.", "success")
    return redirect(url_for("dashboard.backtest"))


@bp.route("/walkforward/run/<kind>", methods=["POST"])
def run_wfmc(kind: str):
    if kind not in ("daily", "monthly"):
        flash(f"Unknown WFMC run kind: {kind}", "error")
        return redirect(url_for("dashboard.walkforward"))
    _enqueue(f"run_wfmc_{kind}")
    flash(
        f"{'Daily' if kind == 'daily' else 'Monthly RunPod'} walk-forward run queued.",
        "success",
    )
    return redirect(url_for("dashboard.walkforward"))


@bp.route("/settings/benchmark-runpod", methods=["POST"])
def benchmark_runpod():
    """gap-closure item 7: launches real, billed RunPod pods on 2-3 GPU
    tiers to compare cost-per-run. Never changes runpod_gpu_type itself -
    only records a recommendation for the operator to confirm."""
    _enqueue("run_runpod_benchmark")
    flash(
        "RunPod GPU-tier benchmark queued - this launches real, billed pods "
        "and can take a few minutes. Results will appear below once done.",
        "success",
    )
    return redirect(url_for("dashboard.settings"))


@bp.route("/walkforward/regime-pass", methods=["POST"])
def regime_pass():
    """Fuzzy-regime section, step 5: manual trigger for the full per-coin
    regime discovery + per-regime walk-forward pass, independent of the
    monthly automatic retest."""
    _enqueue("run_regime_pass")
    flash(
        "Regime discovery + per-regime walk-forward pass queued - this can "
        "take a while for a large universe. Progress appears on this page.",
        "success",
    )
    return redirect(url_for("dashboard.walkforward"))


@bp.route("/backtest/pull", methods=["POST"])
def historical_pull():
    months = request.form.get("months", type=int)
    _enqueue("historical_pull", {"months": months} if months else {})
    flash(
        "Historical pull queued. Progress will appear here; this is the heavy "
        "one-time load, so it is rate limited.",
        "success",
    )
    return redirect(url_for("dashboard.backtest"))


# --------------------------------------------------------------------------
# settings
# --------------------------------------------------------------------------
@bp.route("/settings/save", methods=["POST"])
def save_settings():
    config = cfg()
    current = config.as_dict()
    changes: dict[str, Any] = {}

    from ..config import BOOLS, EDITABLE

    for key in EDITABLE:
        if key in BOOLS:
            # An unchecked checkbox submits nothing, so absence means False -
            # but only for booleans that were actually rendered on this form.
            if "_form_rendered" in request.form:
                changes[key] = key in request.form
            elif key in request.form:
                changes[key] = request.form[key]
            continue
        raw = request.form.get(key)
        if raw is None or raw == "":
            continue
        changes[key] = raw

    # Only submit what actually differs, so the audit log stays meaningful.
    pending = {}
    for key, value in changes.items():
        if str(current.get(key)) != str(value):
            pending[key] = value

    if not pending:
        flash("No changes to save.", "success")
        return redirect(url_for("dashboard.settings"))

    try:
        applied = config.update(pending)
    except ConfigError as exc:
        flash(f"Rejected: {exc}", "error")
        return redirect(url_for("dashboard.settings"))

    conn = db.connect()
    who = current_user() or "dashboard"
    for key, (old, new) in applied.items():
        conn.execute(
            "INSERT INTO settings_audit(ts, username, key, old_value, new_value) "
            "VALUES (?,?,?,?,?)",
            (db.now(), who, key, str(old), str(new)),
        )
    db.log_event(
        f"{who} changed {len(applied)} setting(s): "
        + ", ".join(f"{k} {o} -> {n}" for k, (o, n) in list(applied.items())[:8]),
        level="warn",
        category="system",
        detail={k: {"from": o, "to": n} for k, (o, n) in applied.items()},
        conn=conn,
    )
    flash(f"Saved {len(applied)} setting(s). The worker picks them up next cycle.", "success")
    return redirect(url_for("dashboard.settings"))


@bp.route("/settings/shadow", methods=["POST"])
def save_shadow():
    """Promote a candidate rule set into shadow mode (paper, alongside live)."""
    from ..config import SPEC, _coerce

    overrides: dict[str, Any] = {}
    for key in SPEC:
        raw = request.form.get(f"shadow_{key}")
        if raw:
            try:
                overrides[key] = _coerce(key, raw)
            except ConfigError as exc:
                flash(f"Rejected: {exc}", "error")
                return redirect(url_for("dashboard.settings"))

    _enqueue("set_shadow", {"overrides": overrides})
    flash(
        f"Shadow instance updated with {len(overrides)} override(s). It runs in paper "
        "alongside the live bot until you promote it.",
        "success",
    )
    return redirect(url_for("dashboard.settings"))


# --------------------------------------------------------------------------
# exports
# --------------------------------------------------------------------------
@bp.route("/export/trades.csv")
def export_trades():
    instance = request.args.get("instance") or None
    rows = exports.fetch_trades(instance=instance)
    body = exports.trades_csv(rows)
    name = f"trades-{instance or 'all'}.csv"
    return Response(
        body,
        mimetype="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{name}"'},
    )


@bp.route("/export/tax.csv")
def export_tax():
    year = request.args.get("year", type=int)
    summary = exports.tax_summary(year=year)
    body = exports.tax_csv(summary)
    name = f"tax-summary-{year or 'all'}.csv"
    return Response(
        body,
        mimetype="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{name}"'},
    )


@bp.route("/tax")
def tax():
    year = request.args.get("year", type=int)
    return render_template("tax.html", summary=exports.tax_summary(year=year), year=year)
