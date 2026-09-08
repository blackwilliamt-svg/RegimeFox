"""JSON API backing the dashboard's live views.

All read-only. The charting frontend polls these; nothing here mutates trading
state (control actions go through the form routes, which enqueue commands).
"""
from __future__ import annotations

import json
import time
from typing import Any

from flask import Blueprint, current_app, jsonify, request

from .. import db, exports, risk, secrets_store
from ..clients import JupiterClient, RugCheckClient
from ..ratelimit import TokenBucket
from ..recovery import reconcile_halted

bp = Blueprint("api", __name__)


def cfg() -> Any:
    return current_app.config["SOLBOT_CONFIG"]


def _rows(rows: Any) -> list[dict[str, Any]]:
    return [dict(r) for r in rows]


@bp.get("/status")
def status():
    config = cfg()
    hb = db.kv_get("worker_heartbeat", {}) or {}
    age = db.now() - int(hb.get("ts", 0)) if hb.get("ts") else None
    state = db.kv_get("engine_state", {}) or {}

    instances = []
    for name in db.INSTANCES:
        from ..portfolio import Portfolio

        p = Portfolio(name, config.as_dict())
        if name != "live" and not _has_activity(name) and name != "paper":
            continue
        prices = {}
        unreal, detail = p.unrealised(prices)
        instances.append(
            {
                "instance": name,
                "balance": round(p.balance(), 2),
                "deployed": round(p.deployed_usd(), 2),
                "open_positions": len(detail),
                "performance": _round(p.performance()),
            }
        )

    return jsonify(
        {
            "mode": config["trading_mode"],
            "running": bool(db.kv_get("engine_run", True)),
            "worker": {"alive": age is not None and age < 60, "age_seconds": age, **hb},
            "kill_switch": risk.kill_switch_state(),
            "circuit_breaker": risk.circuit_state(),
            "reconcile_halted": reconcile_halted(),
            "engine": state,
            "scan_budget": db.kv_get("scan_budget", {}),
            "instances": instances,
            "server_time": db.now(),
        }
    )


def _has_activity(instance: str) -> bool:
    row = db.connect().execute(
        "SELECT 1 FROM positions WHERE instance = ? LIMIT 1", (instance,)
    ).fetchone()
    return row is not None


def _round(d: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for k, v in d.items():
        if isinstance(v, float):
            out[k] = None if v == float("inf") else round(v, 4)
        else:
            out[k] = v
    return out


@bp.get("/positions")
def positions():
    conn = db.connect()
    rows = conn.execute(
        "SELECT * FROM positions WHERE status = 'open' ORDER BY instance, id"
    ).fetchall()

    prices = db.kv_get("last_prices", {}) or {}
    out = []
    for r in rows:
        d = dict(r)
        price = float(prices.get(r["mint"], 0.0)) or _last_tick(r["mint"], conn)
        d["current_price"] = price
        if price:
            value = float(r["qty"]) * price
            d["unrealised_usd"] = round(value - float(r["size_usd"]), 4)
            d["unrealised_pct"] = round(
                (value / float(r["size_usd"]) - 1.0) * 100.0 if r["size_usd"] else 0.0, 3
            )
        else:
            d["unrealised_usd"] = None
            d["unrealised_pct"] = None
        try:
            d["entry_snapshot"] = json.loads(d.get("entry_snapshot") or "{}")
        except json.JSONDecodeError:
            d["entry_snapshot"] = {}
        out.append(d)
    return jsonify(out)


def _last_tick(mint: str, conn: Any) -> float:
    row = conn.execute(
        "SELECT price FROM price_ticks WHERE mint = ? ORDER BY ts DESC LIMIT 1", (mint,)
    ).fetchone()
    return float(row["price"]) if row else 0.0


@bp.get("/candles/<mint>")
def candles(mint: str):
    """OHLCV for one token plus its markers, for the per-position chart."""
    from ..datastore import DataStore
    from ..clients import build_clients

    config = cfg()
    conn = db.connect()
    limit = min(int(request.args.get("limit", 200)), 1000)

    clients = current_app.config.get("SOLBOT_CLIENTS")
    if clients is None:
        clients = build_clients(config)
        current_app.config["SOLBOT_CLIENTS"] = clients
    store = DataStore(clients.binance, config.as_dict())

    df = store.candles_for(mint, limit=limit, conn=conn)
    series = [
        {
            "time": int(r.ts),
            "open": float(r.open),
            "high": float(r.high),
            "low": float(r.low),
            "close": float(r.close),
            "volume": float(r.volume),
        }
        for r in df.itertuples(index=False)
    ]

    markers = conn.execute(
        "SELECT id, instance, status, entry_ts, entry_price, exit_ts, exit_price, "
        "hard_stop, trailing_stop, take_profit, symbol, entry_reason, exit_reason "
        "FROM positions WHERE mint = ? ORDER BY entry_ts DESC LIMIT 20",
        (mint,),
    ).fetchall()

    return jsonify({"mint": mint, "candles": series, "positions": _rows(markers)})


@bp.get("/events")
def events():
    since_id = request.args.get("since_id", type=int)
    limit = min(int(request.args.get("limit", 100)), 500)
    rows = db.recent_events(limit=limit, since_id=since_id)
    return jsonify(_rows(rows))


@bp.get("/equity")
def equity():
    conn = db.connect()
    since = db.now() - int(request.args.get("days", 30)) * 86400
    rows = conn.execute(
        "SELECT instance, ts, balance, equity FROM equity WHERE ts >= ? ORDER BY ts",
        (since,),
    ).fetchall()
    series: dict[str, list[dict[str, float]]] = {}
    for r in rows:
        series.setdefault(r["instance"], []).append(
            {"time": int(r["ts"]), "value": round(float(r["equity"]), 4)}
        )
    return jsonify(series)


@bp.get("/performance")
def performance():
    from ..portfolio import Portfolio

    config = cfg().as_dict()
    out = {}
    for name in db.INSTANCES:
        out[name] = _round(Portfolio(name, config).performance())

    conn = db.connect()
    latest = conn.execute(
        "SELECT * FROM backtest_runs ORDER BY ts DESC LIMIT 1"
    ).fetchone()
    out["backtest"] = dict(latest) if latest else None

    # The drift the spec wants surfaced passively: expected versus actual.
    if latest and out.get("paper", {}).get("trades"):
        out["drift"] = {
            "backtest_win_rate": latest["win_rate"],
            "paper_win_rate": out["paper"]["win_rate"],
            "live_win_rate": out.get("live", {}).get("win_rate"),
            "backtest_profit_factor": latest["profit_factor"],
            "paper_profit_factor": out["paper"]["profit_factor"],
        }
    return jsonify(out)


@bp.get("/universe")
def universe():
    conn = db.connect()
    rows = conn.execute(
        "SELECT * FROM universe ORDER BY volume_24h_usd DESC LIMIT 500"
    ).fetchall()
    safety = {
        r["mint"]: {"passed": bool(r["passed"]), "reasons": json.loads(r["reasons"] or "[]")}
        for r in conn.execute("SELECT mint, passed, reasons FROM safety_reports").fetchall()
    }
    out = []
    for r in rows:
        d = dict(r)
        d["safety"] = safety.get(r["mint"])
        out.append(d)
    return jsonify(out)


@bp.get("/progress")
def progress():
    from ..candlestore import ParquetCandleStore
    from ..datastore import BASE_INTERVAL, JOB_DAILY_INCREMENTAL

    conn = db.connect()
    mints = [r["mint"] for r in conn.execute("SELECT mint FROM universe").fetchall()]
    candles = ParquetCandleStore()
    return jsonify(
        {
            "historical_pull": db.get_progress("historical_pull"),
            "daily_incremental_pull": db.get_progress(JOB_DAILY_INCREMENTAL),
            "backtest": db.get_progress("backtest"),
            "candle_coverage": {
                **candles.universe_coverage(mints, BASE_INTERVAL),
                "disk": candles.disk_usage(),
            },
        }
    )


@bp.get("/tax")
def tax():
    year = request.args.get("year", type=int)
    return jsonify(exports.tax_summary(year=year).as_dict())


# --------------------------------------------------------------------------
# Walk-forward, Monte Carlo, drift and the parameter pipeline
# --------------------------------------------------------------------------
@bp.get("/walkforward")
def walkforward():
    """The latest optimizer run, as the walk-forward tab renders it.

    The run itself happens on the operator's PC; everything here was posted to
    the droplet over the optimizer endpoints, so this view is a mirror rather
    than a source.
    """
    conn = db.connect()
    run_id = request.args.get("run", type=int)
    if run_id:
        row = conn.execute(
            "SELECT * FROM optimizer_runs WHERE run_id = ?", (run_id,)
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT * FROM optimizer_runs ORDER BY updated_at DESC LIMIT 1"
        ).fetchone()

    if row is None:
        return jsonify({"run": None, "runs": []})

    runs = conn.execute(
        "SELECT run_id, status, started_at, updated_at, label FROM optimizer_runs "
        "ORDER BY updated_at DESC LIMIT 20"
    ).fetchall()

    summary = _loads(row["summary"])
    return jsonify(
        {
            "run": {
                "run_id": int(row["run_id"]),
                "status": row["status"],
                "label": row["label"],
                "started_at": row["started_at"],
                "updated_at": row["updated_at"],
                "age_seconds": db.now() - int(row["updated_at"] or 0),
                "coverage": _loads(row["coverage"]),
                "summary": summary,
                "monte_carlo": _loads(row["monte_carlo"]) or summary.get("monte_carlo"),
                "stress": _loads(row["stress"]) or summary.get("stress"),
            },
            "runs": _rows(runs),
        }
    )


@bp.get("/walkforward/feed")
def walkforward_feed():
    """Plain-language feed lines for one run, oldest first."""
    conn = db.connect()
    run_id = request.args.get("run", type=int)
    if not run_id:
        row = conn.execute(
            "SELECT run_id FROM optimizer_runs ORDER BY updated_at DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return jsonify({"run_id": None, "lines": []})
        run_id = int(row["run_id"])

    since_id = request.args.get("since_id", type=int) or 0
    limit = min(int(request.args.get("limit", 200)), 500)
    rows = conn.execute(
        "SELECT id, ts, level, message FROM optimizer_feed "
        "WHERE run_id = ? AND id > ? ORDER BY id LIMIT ?",
        (run_id, since_id, limit),
    ).fetchall()
    return jsonify({"run_id": run_id, "lines": _rows(rows)})


@bp.get("/drift")
def drift_series():
    """Live-versus-backtest drift over time, per instance (spec 4.5)."""
    from .. import drift as drift_module

    instance = request.args.get("instance") or None
    limit = min(int(request.args.get("limit", 200)), 500)
    samples = drift_module.recent(instance, limit=limit)

    latest: dict[str, Any] = {}
    for sample in samples:
        latest[sample["instance"]] = sample
    return jsonify({"latest": latest, "samples": samples})


@bp.get("/paramsync")
def paramsync_status():
    """Every bundle received (daily or from a RunPod worker) and every promotion decision."""
    from .. import paramsync as paramsync_module

    return jsonify(paramsync_module.status())


@bp.get("/wfmc/storage")
def wfmc_storage():
    """The WF/MC output storage browser (spec 6c): what is stored, by run.

    Backed by the daily on-droplet run's local store - the monthly RunPod
    run's own local store is ephemeral (destroyed with its worker); its
    result is what landed in ``optimizer_runs`` via the bundle/feed endpoints,
    already covered by ``/api/walkforward``.
    """
    from ..wfmc import DAILY_STORE_PATH
    from solopt.store import RunStore

    cursor = request.args.get("cursor", type=int)
    limit = min(int(request.args.get("limit", 50)), 200)
    try:
        store = RunStore(DAILY_STORE_PATH)
        runs = store.list_runs(limit=limit, cursor=cursor)
    except Exception:
        runs = []

    return jsonify(
        {
            "runs": runs,
            "retention_days": cfg()["wfmc_result_retention_days"],
            "next_cursor": runs[-1]["id"] if len(runs) == limit else None,
        }
    )


@bp.get("/review")
def review_status():
    """Recent entry-gate activity and the current market read."""
    from .. import review as review_module

    conn = db.connect()
    recent_rows = conn.execute(
        "SELECT ts, kind, instance, mint, decision, conviction, exit_style "
        "FROM entry_reviews ORDER BY id DESC LIMIT 50"
    ).fetchall()
    return jsonify(
        {
            "market": review_module.build_market_context(conn=conn).as_dict(),
            "recent": _rows(recent_rows),
        }
    )


def _loads(raw: Any) -> Any:
    if raw in (None, ""):
        return None
    try:
        return json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return None


# --------------------------------------------------------------------------
# API key rotation
# --------------------------------------------------------------------------
@bp.post("/keys/<provider>")
def rotate_key(provider: str):
    """Store a new key, test-ping it, and report which one is now in effect.

    A failed validation must not silently keep the old key active nor silently
    discard the new one - the response says explicitly which is live.
    """
    if provider not in secrets_store.PROVIDERS:
        return jsonify({"ok": False, "error": f"unknown provider {provider}"}), 400

    payload = request.get_json(silent=True) or {}
    new_key = (payload.get("key") or "").strip()
    if not new_key:
        return jsonify({"ok": False, "error": "key is required"}), 400

    config = cfg()
    enc = config.secrets.secret_encryption_key
    previous = secrets_store.effective_keys(config.secrets, enc).get(provider, "")

    ok, error = _ping_provider(provider, new_key, config)
    if not ok:
        return jsonify(
            {
                "ok": False,
                "error": error,
                "in_effect": "previous key",
                "in_effect_masked": secrets_store.mask(previous),
                "message": (
                    "The new key failed validation and was NOT saved. "
                    "The previous key remains in effect."
                ),
            }
        ), 400

    try:
        secrets_store.store_key(
            provider, new_key, enc, updated_by=request.headers.get("X-User", "dashboard")
        )
    except secrets_store.SecretsUnavailable as exc:
        return jsonify({"ok": False, "error": str(exc), "in_effect": "previous key"}), 500

    db.log_event(
        f"{provider} API key rotated from the dashboard and validated successfully.",
        level="warn",
        category="auth",
    )
    # Push the new key into this process's live clients too. Binance is keyless
    # and the bulk-data token is read on demand elsewhere, so only these two
    # have a live client to update.
    clients = current_app.config.get("SOLBOT_CLIENTS")
    if clients is not None and provider in ("jupiter", "rugcheck"):
        clients.apply_keys(**{provider: new_key})

    return jsonify(
        {
            "ok": True,
            "in_effect": "new key",
            "in_effect_masked": secrets_store.mask(new_key),
            "message": f"{provider} key validated and saved. It is now in effect.",
        }
    )


@bp.post("/keys/bulk_data/generate")
def generate_bulk_token():
    """Mint a new bearer token for a RunPod worker to report back with.

    Generated here rather than typed in: this is the droplet's own secret, and a
    token the operator invents is a token the operator can make guessable. The
    plaintext is returned exactly once - after this response it exists only
    encrypted at rest.
    """
    import secrets as _secrets

    config = cfg()
    token = _secrets.token_urlsafe(32)
    try:
        secrets_store.store_key(
            "bulk_data",
            token,
            config.secrets.secret_encryption_key,
            updated_by=request.headers.get("X-User", "dashboard"),
        )
    except secrets_store.SecretsUnavailable as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500

    db.log_event(
        "A new RunPod worker token was generated from the dashboard. Any worker "
        "still using the previous token will stop being able to report back.",
        level="warn",
        category="auth",
    )
    return jsonify(
        {
            "ok": True,
            "token": token,
            "message": (
                "Copy this now - it is shown once. Set it as SOLOPT_TOKEN in the "
                "RunPod worker's environment."
            ),
        }
    )


def _ping_provider(provider: str, key: str, config: Any) -> tuple[bool, str]:
    """Test-ping a candidate key without disturbing the running clients."""
    if provider == "bulk_data":
        # This is the droplet's own token; there is nothing external to ask.
        if len(key) < 24:
            return False, "a bulk-data token must be at least 24 characters"
        return True, ""

    bucket = TokenBucket(2.0, 2, reserve=0.0, name=f"{provider}-ping")
    try:
        if provider == "jupiter":
            client = JupiterClient(bucket, key, timeout=15.0, max_retries=0)
        else:
            client = RugCheckClient(bucket, key, timeout=15.0, max_retries=0)
        try:
            client.ping()
            return True, ""
        finally:
            client.close()
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"[:300]


@bp.get("/keys")
def keys():
    config = cfg()
    return jsonify(
        [
            {
                "provider": k.provider,
                "configured": k.configured,
                "masked": k.masked,
                "source": k.source,
                "updated_at": k.updated_at,
                "updated_by": k.updated_by,
            }
            for k in secrets_store.key_status(
                config.secrets, config.secrets.secret_encryption_key
            )
        ]
    )
