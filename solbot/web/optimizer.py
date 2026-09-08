"""Optimizer ingest endpoints: a RunPod worker's feed, run summary, and bundle.

Under spec 5 there is nothing left to *download* here - the daily run reads
the droplet's own Parquet files directly, in-process, and the monthly RunPod
run has its data shipped to it before it starts (see :mod:`solbot.runpod`).
What a remote RunPod worker still needs from the droplet is somewhere to
report its progress and, when it finishes with a promotable set, somewhere to
hand it over.

**Authentication is a bearer token, not the dashboard session.** A RunPod
worker is a headless client with no browser and no TOTP device, so it presents
a token the operator generates from the settings page. That token is checked
in constant time, is rate limited, and grants write access to the run feed and
bundle submission only - it cannot halt trading, close a position, change a
setting, or read an API key. A leaked token costs the operator nothing more
than a forged progress report.
"""
from __future__ import annotations

import hmac
import json
import logging
import sqlite3
import time
from collections import deque
from functools import wraps
from typing import Any, Callable

from flask import Blueprint, current_app, jsonify, request

from .. import db, paramsync, secrets_store

log = logging.getLogger(__name__)

bp = Blueprint("optimizer", __name__)

MAX_FEED_LINES = 200

_recent_requests: deque[float] = deque(maxlen=10_000)


def cfg() -> Any:
    return current_app.config["SOLBOT_CONFIG"]


# --------------------------------------------------------------------------
# Auth
# --------------------------------------------------------------------------
def _configured_token() -> str:
    config = cfg()
    try:
        keys = secrets_store.effective_keys(
            config.secrets, config.secrets.secret_encryption_key
        )
    except Exception:
        keys = {}
    return (keys.get("bulk_data") or config.secrets.bulk_data_token or "").strip()


def _presented_token() -> str:
    header = request.headers.get("Authorization", "")
    if header.lower().startswith("bearer "):
        return header[7:].strip()
    return ""


def _rate_limited() -> bool:
    """A simple fixed-window cap, shared across every bulk endpoint.

    The droplet runs the trading loop and live fuzzy-regime classification
    alongside this dashboard process. An optimizer that fans out two hundred
    parallel downloads would starve them, so the ceiling is enforced here
    rather than trusted to the client.
    """
    limit = int(cfg().get("runpod_rate_limit_per_minute", 60))
    now = time.monotonic()
    while _recent_requests and now - _recent_requests[0] > 60.0:
        _recent_requests.popleft()
    if len(_recent_requests) >= limit:
        return True
    _recent_requests.append(now)
    return False


def token_required(fn: Callable) -> Callable:
    """Bearer-token auth. Never falls back to the dashboard session."""

    @wraps(fn)
    def wrapper(*args: Any, **kwargs: Any):
        if not cfg().get("runpod_ingest_enabled", True):
            return jsonify({"error": "the RunPod ingest endpoint is disabled"}), 403

        expected = _configured_token()
        presented = _presented_token()
        if not expected:
            return jsonify(
                {
                    "error": "no bulk data token is configured on this droplet; "
                    "generate one from the dashboard settings page"
                }
            ), 503
        # Constant-time, and length is compared inside compare_digest rather than
        # short-circuiting on it.
        if not presented or not hmac.compare_digest(presented, expected):
            log.warning("bulk data request rejected from %s", request.remote_addr)
            return jsonify({"error": "invalid or missing bearer token"}), 401
        if _rate_limited():
            return jsonify({"error": "rate limit exceeded"}), 429
        return fn(*args, **kwargs)

    return wrapper


# --------------------------------------------------------------------------
# Optimizer ingest
# --------------------------------------------------------------------------
@bp.get("/optimizer/execution")
@token_required
def execution_profile():
    """Observed slippage and fees, for the Monte Carlo's execution model.

    Returned as a distribution rather than a mean: the optimizer samples from it
    per trade, and the spread is the part that matters.
    """
    conn = db.connect()
    samples = db.execution_samples(days=180, conn=conn)
    if len(samples) < 30:
        config = cfg()
        return jsonify(
            {
                "source": "assumed",
                "samples": len(samples),
                "slippage_mean_pct": float(config["max_slippage_pct"]) / 2.0,
                "slippage_stdev_pct": float(config["max_slippage_pct"]) / 8.0,
                "fee_mean_pct": float(config["taker_fee_pct"]),
                "fee_stdev_pct": float(config["taker_fee_pct"]) * 0.05,
                "note": (
                    f"only {len(samples)} recorded fills; 30 are needed before the "
                    "observed distribution replaces the configured constants"
                ),
            }
        )

    import statistics

    slippage = [s for s, _ in samples]
    fees = [f for _, f in samples]
    return jsonify(
        {
            "source": "observed",
            "samples": len(samples),
            "slippage_mean_pct": statistics.fmean(slippage),
            "slippage_stdev_pct": statistics.stdev(slippage) if len(slippage) > 1 else 0.0,
            "fee_mean_pct": statistics.fmean(fees),
            "fee_stdev_pct": statistics.stdev(fees) if len(fees) > 1 else 0.0,
        }
    )


@bp.post("/optimizer/feed")
@token_required
def ingest_feed():
    """Accept a batch of run-feed lines from the optimizer."""
    payload = request.get_json(silent=True) or {}
    try:
        run_id = int(payload.get("run_id", 0))
    except (TypeError, ValueError):
        return jsonify({"error": "run_id must be an integer"}), 400
    lines = payload.get("lines")
    if run_id <= 0 or not isinstance(lines, list):
        return jsonify({"error": "run_id and lines are required"}), 400
    if len(lines) > MAX_FEED_LINES:
        return jsonify({"error": f"at most {MAX_FEED_LINES} lines per request"}), 413

    conn = db.connect()
    conn.execute(
        "INSERT INTO optimizer_runs(run_id, started_at, updated_at, status) "
        "VALUES (?,?,?, 'running') ON CONFLICT(run_id) DO UPDATE SET "
        "updated_at = excluded.updated_at",
        (run_id, db.now(), db.now()),
    )

    stored = 0
    for line in lines:
        if not isinstance(line, dict):
            continue
        message = str(line.get("message", ""))[:2000]
        if not message:
            continue
        level = str(line.get("level", "info"))
        if level not in ("info", "warn", "alert"):
            level = "info"
        detail = line.get("detail")
        try:
            conn.execute(
                "INSERT INTO optimizer_feed(run_id, remote_id, ts, level, message, detail) "
                "VALUES (?,?,?,?,?,?) ON CONFLICT(run_id, remote_id) DO NOTHING",
                (
                    run_id, int(line.get("id", 0)), int(line.get("ts", db.now())),
                    level, message,
                    json.dumps(detail, default=str) if detail is not None else None,
                ),
            )
            stored += 1
        except (sqlite3.Error, TypeError, ValueError):
            continue

    return jsonify({"ok": True, "stored": stored})


@bp.post("/optimizer/run")
@token_required
def ingest_run():
    """Accept a run's status and final summary."""
    payload = request.get_json(silent=True) or {}
    try:
        run_id = int(payload.get("run_id", 0))
    except (TypeError, ValueError):
        return jsonify({"error": "run_id must be an integer"}), 400
    if run_id <= 0:
        return jsonify({"error": "run_id is required"}), 400

    status = str(payload.get("status", "running"))
    if status not in ("running", "done", "failed", "cancelled"):
        status = "running"
    summary = payload.get("summary")
    conn = db.connect()
    conn.execute(
        "INSERT INTO optimizer_runs(run_id, started_at, updated_at, status, label, "
        "coverage, summary, monte_carlo, stress) VALUES (?,?,?,?,?,?,?,?,?) "
        "ON CONFLICT(run_id) DO UPDATE SET updated_at=excluded.updated_at, "
        "status=excluded.status, label=COALESCE(excluded.label, optimizer_runs.label), "
        "coverage=COALESCE(excluded.coverage, optimizer_runs.coverage), "
        "summary=COALESCE(excluded.summary, optimizer_runs.summary), "
        "monte_carlo=COALESCE(excluded.monte_carlo, optimizer_runs.monte_carlo), "
        "stress=COALESCE(excluded.stress, optimizer_runs.stress)",
        (
            run_id, db.now(), db.now(), status,
            str(payload.get("label") or "") or None,
            _dump(payload.get("coverage")),
            _dump(summary),
            # The Monte Carlo block is sent separately from the summary because
            # it carries the histograms the dashboard chart needs, which are far
            # too bulky to duplicate inside every summary.
            _dump(
                payload.get("monte_carlo")
                or ((summary or {}).get("monte_carlo") if isinstance(summary, dict) else None)
            ),
            _dump(payload.get("stress")),
        ),
    )

    # The dashboard's completion notification is driven off this transition, so
    # the event is written once, here, rather than polled for.
    if status == "done":
        row = conn.execute(
            "SELECT notified FROM optimizer_runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        if row is not None and not int(row["notified"] or 0):
            conn.execute(
                "UPDATE optimizer_runs SET notified = 1 WHERE run_id = ?", (run_id,)
            )
            db.log_event(
                _completion_message(run_id, summary if isinstance(summary, dict) else {}),
                level="warn",
                category="system",
                detail=summary if isinstance(summary, dict) else None,
                conn=conn,
            )
    return jsonify({"ok": True})


def _completion_message(run_id: int, summary: dict[str, Any]) -> str:
    if not summary:
        return f"Walk-forward run {run_id} finished."
    if summary.get("accepted"):
        return (
            f"Walk-forward run {run_id} finished and produced a promotable parameter "
            f"set: {summary.get('profitable_windows', 0)}/"
            f"{summary.get('counted_windows', 0)} out-of-sample windows profitable, "
            f"efficiency {summary.get('walk_forward_efficiency', 0):.2f}."
        )
    reasons = summary.get("reasons") or [summary.get("verdict", "did not meet the thresholds")]
    return (
        f"Walk-forward run {run_id} finished without a promotable set: "
        + "; ".join(str(r) for r in reasons)[:400]
    )


def _dump(value: Any) -> str | None:
    if value is None:
        return None
    try:
        return json.dumps(value, default=float)
    except (TypeError, ValueError):
        return None


@bp.post("/optimizer/bundle")
@token_required
def ingest_bundle():
    """Accept a finished, promotable bundle from a RunPod monthly-retest worker.

    Re-validated here exactly like the daily on-droplet run's bundle is - see
    :func:`solbot.paramsync.accept_bundle`. A worker's own claim that its set
    is promotable is not trusted; the gates and bounds are re-checked.
    """
    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, dict):
        return jsonify({"error": "bundle payload must be an object"}), 400

    try:
        result = paramsync.accept_bundle(
            payload, cfg().as_dict(), source=f"runpod:{request.remote_addr}"
        )
    except paramsync.BundleError as exc:
        return jsonify({"error": str(exc)}), 400

    return jsonify({"ok": True, **result})


def _regime_store() -> Any:
    """The same on-droplet library store the live trading loop reads fuzzy
    regime models and per-regime parameter sets from (see
    ``solbot.engine.Engine._get_library_store``) - a RunPod regime-pass
    worker's own local store does not survive its container/volume being torn
    down, so its findings have to land here directly, over this endpoint."""
    from solopt.store import RunStore

    from ..wfmc import DAILY_STORE_PATH

    return RunStore(DAILY_STORE_PATH)


@bp.post("/optimizer/regime-model")
@token_required
def ingest_regime_model():
    """Accept one coin's freshly-discovered fuzzy regime model from a RunPod
    regime-pass worker (fuzzy-regime section, step 5)."""
    payload = request.get_json(silent=True) or {}
    symbol = str(payload.get("symbol") or "").strip()
    model = payload.get("model")
    if not symbol or not isinstance(model, dict):
        return jsonify({"error": "symbol and model are required"}), 400
    try:
        run_id = int(payload.get("run_id") or 0) or None
    except (TypeError, ValueError):
        run_id = None

    _regime_store().save_regime_model(symbol, model, run_id=run_id)
    return jsonify({"ok": True})


@bp.post("/optimizer/regime-library")
@token_required
def ingest_regime_library():
    """Accept one accepted per-coin, per-regime walk-forward result from a
    RunPod regime-pass worker - the regime-scoped counterpart to
    ``/optimizer/bundle``, stored in the library rather than the shadow/
    promotion system since it is keyed by coin and regime cluster, not a
    portfolio-wide parameter set."""
    from solopt.store import LibraryEntry

    payload = request.get_json(silent=True) or {}
    fingerprint = str(payload.get("fingerprint") or "").strip()
    symbol = str(payload.get("symbol") or "").strip()
    params = payload.get("params")
    performance = payload.get("performance")
    if not fingerprint or not symbol or not isinstance(params, dict) or not isinstance(performance, dict):
        return jsonify(
            {"error": "fingerprint, symbol, params, and performance are required"}
        ), 400
    try:
        regime_cluster_id = int(payload.get("regime_cluster_id"))
    except (TypeError, ValueError):
        return jsonify({"error": "regime_cluster_id is required"}), 400
    try:
        run_id = int(payload.get("run_id") or 0) or None
    except (TypeError, ValueError):
        run_id = None

    changed = _regime_store().upsert_library(
        LibraryEntry(
            fingerprint=fingerprint, params=params, performance=performance,
            symbol=symbol, regime_cluster_id=regime_cluster_id, run_id=run_id,
        )
    )
    return jsonify({"ok": True, "changed": changed})
