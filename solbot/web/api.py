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
from ..clients import JupiterClient, BirdeyeClient, RugCheckClient
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
        clients = build_clients(config, track_budget=False)
        current_app.config["SOLBOT_CLIENTS"] = clients
    store = DataStore(clients.birdeye, config.as_dict())

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
    return jsonify(
        {
            "historical_pull": db.get_progress("historical_pull"),
            "backtest": db.get_progress("backtest"),
            "birdeye_budget": {
                **db.api_usage("birdeye"),
                "limit": cfg()["birdeye_monthly_cu_budget"],
            },
        }
    )


@bp.get("/tax")
def tax():
    year = request.args.get("year", type=int)
    return jsonify(exports.tax_summary(year=year).as_dict())


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
    # Push the new key into this process's live clients too.
    clients = current_app.config.get("SOLBOT_CLIENTS")
    if clients is not None:
        clients.apply_keys(**{provider: new_key})

    return jsonify(
        {
            "ok": True,
            "in_effect": "new key",
            "in_effect_masked": secrets_store.mask(new_key),
            "message": f"{provider} key validated and saved. It is now in effect.",
        }
    )


def _ping_provider(provider: str, key: str, config: Any) -> tuple[bool, str]:
    """Test-ping a candidate key without disturbing the running clients."""
    bucket = TokenBucket(2.0, 2, reserve=0.0, name=f"{provider}-ping")
    try:
        if provider == "jupiter":
            client = JupiterClient(bucket, key, timeout=15.0, max_retries=0)
        elif provider == "birdeye":
            from ..ratelimit import MonthlyBudget

            client = BirdeyeClient(
                bucket, key, MonthlyBudget(0), timeout=15.0, max_retries=0
            )
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
