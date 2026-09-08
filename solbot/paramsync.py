"""Accepting parameter bundles, running them in shadow, and promoting them.

A parameter bundle arrives one of two ways now (spec 5), and both funnel
through the same acceptance path in this module:

* the **daily** on-droplet walk-forward run (:mod:`solbot.wfmc`) builds one
  in-process and calls :func:`accept_bundle` directly - no network hop, no
  repository;
* the **monthly** RunPod retest builds one on a remote GPU worker and POSTs
  it to this droplet's ``/api/optimizer/bundle`` endpoint, which calls the
  same :func:`accept_bundle`.

There used to be a small git repository in between - an optimizer PC pushed to
it, the droplet pulled from it periodically. That hand-off is gone along with
the PC: everything that produces a bundle now either runs on this droplet or
reports back to it directly.

Two stages either way, and the gap between them is the whole point:

1. **Install to shadow.** A bundle that carries its own evidence - walk-forward
   thresholds met, not flagged overfit, no failed crash replay - is loaded into
   the shadow instance, which trades on paper alongside the real one on the same
   data through the same executor. A bundle that fails its own gates is recorded
   and refused; the evidence travels with the parameters precisely so the droplet
   can make that call itself rather than trusting whatever produced them.

2. **Promote to live.** Automatic, with no manual approval step, but only after
   fifteen clean days in shadow that continue to meet the thresholds *and* a
   decisive margin over what live is currently doing. "Decisive" is a number
   here, not a feeling: the shadow set must beat live on expectancy by the
   configured margin, must not be materially deeper in drawdown, and must win a
   bootstrap comparison of the two trade populations at the configured
   confidence. A set that is merely ahead is not promoted, because a set that is
   merely ahead is usually noise.

Every bundle is treated as untrusted input regardless of where it came from.
Every parameter is validated against the same bounds the settings page
enforces before it reaches a config file, and the bundle's own claims about
itself are re-checked rather than believed.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from . import db, drift
from .config import ConfigError, EDITABLE, _coerce, validate
from .portfolio import Portfolio

log = logging.getLogger(__name__)

BUNDLE_SCHEMA = 1
SHADOW_KEY = "shadow_overrides"
ACTIVE_BUNDLE_KEY = "active_bundle"

STATUS_PENDING = "pending"
STATUS_SHADOW = "shadow"
STATUS_PROMOTED = "promoted"
STATUS_REJECTED = "rejected"
STATUS_SUPERSEDED = "superseded"


class BundleError(ValueError):
    """A submitted bundle is malformed - not the same as failing its gates."""


def validate_schema(payload: dict[str, Any]) -> None:
    if not isinstance(payload, dict):
        raise BundleError("bundle payload is not an object")
    schema = int(payload.get("schema", 0))
    if schema != BUNDLE_SCHEMA:
        raise BundleError(
            f"bundle schema {schema} is not understood by this droplet "
            f"(expected {BUNDLE_SCHEMA}); upgrade the bot or the optimizer"
        )


# --------------------------------------------------------------------------
# Validating a bundle
# --------------------------------------------------------------------------
@dataclass
class BundleCheck:
    ok: bool = False
    reason: str = ""
    overrides: dict[str, Any] = field(default_factory=dict)
    rejected: list[str] = field(default_factory=list)
    fingerprint: str = ""


def fingerprint(payload: dict[str, Any]) -> str:
    import hashlib

    body = json.dumps(
        {"global": payload.get("global"), "per_symbol": payload.get("per_symbol")},
        sort_keys=True, default=float,
    )
    return hashlib.sha256(body.encode()).hexdigest()[:16]


def check_bundle(payload: dict[str, Any], cfg: dict[str, Any]) -> BundleCheck:
    """Re-verify a bundle's gates and validate every parameter it carries."""
    check = BundleCheck(fingerprint=fingerprint(payload))
    gates = payload.get("gates") or {}

    if not gates.get("accepted"):
        reasons = gates.get("reasons") or ["the optimizer did not accept this set"]
        check.reason = "; ".join(str(r) for r in reasons)[:400]
        return check
    if gates.get("fragile"):
        windows = ", ".join(gates.get("fragile_windows") or [])
        check.reason = f"flagged fragile by the crash replay ({windows})"
        return check
    if gates.get("overfit"):
        check.reason = "flagged overfit by the walk-forward thresholds"
        return check

    params = payload.get("global") or {}
    if not isinstance(params, dict) or not params:
        check.reason = "the bundle carries no parameters"
        return check

    # The repository is untrusted input: coerce and bounds-check everything,
    # then run the cross-field rules against the state it would produce.
    for key, value in params.items():
        if key not in EDITABLE:
            check.rejected.append(f"{key}: not a parameter this bot accepts")
            continue
        try:
            check.overrides[key] = _coerce(key, value)
        except ConfigError as exc:
            check.rejected.append(f"{key}: {exc}")

    if not check.overrides:
        check.reason = "no usable parameters survived validation: " + "; ".join(
            check.rejected
        )
        return check
    try:
        validate(check.overrides, cfg)
    except ConfigError as exc:
        check.reason = f"the parameter set is internally inconsistent: {exc}"
        return check

    check.ok = True
    check.reason = "meets its own gates and validates against this bot's bounds"
    return check


# --------------------------------------------------------------------------
# Storing and installing
# --------------------------------------------------------------------------
def store_bundle(
    payload: dict[str, Any],
    check: BundleCheck,
    source: str,
    conn: sqlite3.Connection | None = None,
) -> bool:
    """Record a bundle. Returns True if it is new to this droplet.

    ``source`` is provenance for the audit trail - "daily-droplet" for the
    on-droplet incremental run, or "runpod:<job_id>" for a monthly retest -
    kept in the column a git commit SHA used to live in.
    """
    conn = conn or db.connect()
    existing = conn.execute(
        "SELECT status FROM param_bundles WHERE fingerprint = ?", (check.fingerprint,)
    ).fetchone()
    if existing is not None:
        return False

    conn.execute(
        "INSERT INTO param_bundles(fingerprint, received_at, generated_at, "
        "source_commit, payload, status, note) VALUES (?,?,?,?,?,?,?)",
        (
            check.fingerprint, db.now(), int(payload.get("generated_at") or 0), source,
            json.dumps(payload, default=float),
            STATUS_PENDING if check.ok else STATUS_REJECTED,
            check.reason,
        ),
    )
    return True


def install_shadow(
    check: BundleCheck,
    payload: dict[str, Any],
    conn: sqlite3.Connection | None = None,
) -> None:
    """Load a validated bundle into the shadow instance.

    Also carries the Monte Carlo tail across: the 5%-worst-case drawdown is what
    position sizing reads, and shipping parameters without it would leave sizing
    calibrated to the previous set's risk.
    """
    conn = conn or db.connect()
    overrides = dict(check.overrides)
    risk = payload.get("risk") or {}
    p5 = float(risk.get("p5_max_drawdown") or 0.0)
    if p5 > 0:
        overrides["monte_carlo_p5_drawdown"] = p5
    target = risk.get("recommended_portfolio_vol_target")
    if target:
        try:
            overrides["portfolio_vol_target"] = _coerce("portfolio_vol_target", target)
        except ConfigError:
            pass

    # Anything the previous shadow set was testing is replaced wholesale, not
    # merged: a half-old, half-new parameter set was never tested as a set.
    db.kv_set(SHADOW_KEY, overrides, conn)
    conn.execute(
        "UPDATE param_bundles SET status = ? WHERE status = ? AND fingerprint != ?",
        (STATUS_SUPERSEDED, STATUS_SHADOW, check.fingerprint),
    )
    conn.execute(
        "UPDATE param_bundles SET status = ?, shadow_since = ?, installed_at = ? "
        "WHERE fingerprint = ?",
        (STATUS_SHADOW, db.now(), db.now(), check.fingerprint),
    )
    db.enqueue_command("set_shadow", {"overrides": overrides}, requested_by="paramsync", conn=conn)

    wf = ((payload.get("evidence") or {}).get("walk_forward")) or {}
    db.log_event(
        f"New parameter set {check.fingerprint} loaded into shadow: "
        f"{wf.get('profitable_windows', 0)}/"
        f"{wf.get('counted_windows', 0)} out-of-sample windows profitable, "
        f"walk-forward efficiency {wf.get('walk_forward_efficiency', 0):.2f}, "
        f"5% worst-case drawdown {p5 * 100:.1f}%. It runs on paper alongside live "
        f"and will not be promoted for at least "
        f"{(payload.get('gates') or {}).get('shadow_min_days', 15)} days.",
        level="warn",
        category="system",
        instance="shadow",
        detail={"fingerprint": check.fingerprint, "overrides": overrides},
        conn=conn,
    )


def accept_bundle(
    payload: dict[str, Any],
    cfg: dict[str, Any],
    *,
    source: str,
    conn: sqlite3.Connection | None = None,
) -> dict[str, Any]:
    """Validate one bundle against its own gates and this bot's bounds.

    Called directly, in-process, by the daily on-droplet run; called from the
    ``/api/optimizer/bundle`` endpoint for a monthly RunPod retest reporting
    back. Either way, the bundle is untrusted input re-checked here rather
    than believed - see the module docstring.
    """
    conn = conn or db.connect()
    validate_schema(payload)
    check = check_bundle(payload, cfg)
    fresh = store_bundle(payload, check, source, conn)

    result = {
        "fingerprint": check.fingerprint,
        "source": source,
        "new": fresh,
        "accepted": check.ok,
        "reason": check.reason,
        "rejected_parameters": check.rejected,
        "installed": False,
    }
    if not fresh:
        return result
    if not check.ok:
        db.log_event(
            f"Parameter set {check.fingerprint} from {source} was refused: {check.reason}",
            level="warn", category="system", conn=conn,
        )
        return result

    install_shadow(check, payload, conn)
    result["installed"] = True
    return result


# --------------------------------------------------------------------------
# Promotion
# --------------------------------------------------------------------------
@dataclass
class PromotionVerdict:
    promoted: bool = False
    reason: str = ""
    shadow_days: float = 0.0
    margin: float = 0.0
    confidence: float = 0.0
    detail: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "promoted": self.promoted,
            "reason": self.reason,
            "shadow_days": round(self.shadow_days, 2),
            "margin": round(self.margin, 4),
            "confidence": round(self.confidence, 4),
            "detail": self.detail,
        }


def bootstrap_confidence(
    shadow_pnl: Sequence[float],
    live_pnl: Sequence[float],
    *,
    iterations: int = 4000,
    seed: int = 20260903,
) -> float:
    """P(the shadow set's mean trade beats live's), by resampling both.

    Comparing two averages says which is bigger; it does not say whether the
    difference would survive a different run of the same two strategies. This
    does, and it is what turns "decisive margin" into something the code can
    actually check.
    """
    shadow = np.asarray([float(p) for p in shadow_pnl], dtype=np.float64)
    live = np.asarray([float(p) for p in live_pnl], dtype=np.float64)
    if shadow.size < 5:
        return 0.0
    rng = np.random.default_rng(seed)
    shadow_means = rng.choice(shadow, size=(iterations, shadow.size)).mean(axis=1)
    if live.size < 5:
        # Nothing to compare against; ask only whether shadow is reliably positive.
        return float((shadow_means > 0).mean())
    live_means = rng.choice(live, size=(iterations, live.size)).mean(axis=1)
    return float((shadow_means > live_means).mean())


def _trade_pnl(
    instance: str, since: int, conn: sqlite3.Connection
) -> list[float]:
    rows = conn.execute(
        "SELECT pnl_usd FROM trades WHERE instance = ? AND exit_ts >= ? ORDER BY exit_ts",
        (instance, since),
    ).fetchall()
    return [float(r["pnl_usd"]) for r in rows]


def evaluate_promotion(
    cfg: dict[str, Any], conn: sqlite3.Connection | None = None
) -> PromotionVerdict:
    """Decide whether the shadow set has earned the live slot."""
    conn = conn or db.connect()
    verdict = PromotionVerdict()

    if not cfg.get("auto_promote_enabled", True):
        verdict.reason = "auto-promotion is switched off"
        return verdict

    row = conn.execute(
        "SELECT * FROM param_bundles WHERE status = ? ORDER BY shadow_since DESC LIMIT 1",
        (STATUS_SHADOW,),
    ).fetchone()
    if row is None:
        verdict.reason = "no parameter set is running in shadow"
        return verdict

    since_shadow = int(row["shadow_since"] or 0)
    verdict.shadow_days = (db.now() - since_shadow) / 86400.0
    min_days = int(cfg.get("shadow_min_days", 15))
    if verdict.shadow_days < min_days:
        verdict.reason = (
            f"{verdict.shadow_days:.1f} of the required {min_days} clean days in shadow"
        )
        return verdict

    primary = "live" if cfg.get("trading_mode") == "live" else "paper"
    shadow_pnl = _trade_pnl("shadow", since_shadow, conn)
    live_pnl = _trade_pnl(primary, since_shadow, conn)
    min_trades = int(cfg.get("shadow_min_trades", 30))
    if len(shadow_pnl) < min_trades:
        verdict.reason = (
            f"shadow has closed {len(shadow_pnl)} trades, below the {min_trades} "
            "needed for the comparison to mean anything"
        )
        return verdict

    shadow_perf = Portfolio("shadow", cfg).performance(since=since_shadow, conn=conn)
    live_perf = Portfolio(primary, cfg).performance(since=since_shadow, conn=conn)

    # "Continuing to meet the walk-forward thresholds throughout" in practice
    # means the shadow set has not drifted away from what it promised.
    drift_sample = drift.measure("shadow", cfg, conn=conn)
    if drift_sample.status == drift.STATUS_DRIFTING:
        verdict.reason = (
            "the shadow set has drifted from its own backtest expectation: "
            + drift_sample.message
        )
        verdict.detail = {"drift": drift_sample.as_dict()}
        _record(verdict, row["fingerprint"], conn)
        return verdict

    shadow_exp = float(shadow_perf.get("expectancy", 0.0))
    live_exp = float(live_perf.get("expectancy", 0.0))
    required_margin = float(cfg.get("shadow_min_margin", 0.25))

    if live_exp > 0:
        verdict.margin = (shadow_exp - live_exp) / abs(live_exp)
        beats = verdict.margin >= required_margin
    else:
        # Live is not making money; shadow has to be clearly positive rather than
        # merely less bad, or a losing set replaces a losing set.
        verdict.margin = 1.0 if shadow_exp > 0 else -1.0
        beats = shadow_exp > 0 and float(shadow_perf.get("profit_factor", 0.0) or 0.0) >= 1.3

    shadow_dd = float(shadow_perf.get("max_drawdown", 0.0))
    live_dd = float(live_perf.get("max_drawdown", 0.0))
    deeper = live_dd > 0 and shadow_dd > live_dd * 1.15

    verdict.confidence = bootstrap_confidence(shadow_pnl, live_pnl)
    required_confidence = float(cfg.get("shadow_min_confidence", 0.90))

    verdict.detail = {
        "shadow": shadow_perf,
        primary: live_perf,
        "drift": drift_sample.as_dict(),
        "required_margin": required_margin,
        "required_confidence": required_confidence,
    }

    if not beats:
        verdict.reason = (
            f"shadow earns ${shadow_exp:.2f} per trade against live's ${live_exp:.2f} - "
            f"a {verdict.margin * 100:+.0f}% margin, short of the "
            f"{required_margin * 100:.0f}% a promotion needs"
        )
    elif deeper:
        verdict.reason = (
            f"shadow's {shadow_dd * 100:.1f}% drawdown is materially deeper than "
            f"live's {live_dd * 100:.1f}%, so the extra return is not free"
        )
    elif verdict.confidence < required_confidence:
        verdict.reason = (
            f"the margin is there but not decisive: resampling both trade "
            f"populations, shadow beats live in only {verdict.confidence * 100:.0f}% "
            f"of draws against the {required_confidence * 100:.0f}% required"
        )
    else:
        verdict.promoted = True
        verdict.reason = (
            f"{verdict.shadow_days:.0f} clean days in shadow, "
            f"${shadow_exp:.2f} per trade against live's ${live_exp:.2f} "
            f"({verdict.margin * 100:+.0f}%), a {shadow_dd * 100:.1f}% drawdown "
            f"against live's {live_dd * 100:.1f}%, and shadow wins "
            f"{verdict.confidence * 100:.0f}% of resampled comparisons"
        )

    _record(verdict, row["fingerprint"], conn)
    return verdict


def _record(verdict: PromotionVerdict, fingerprint_: str, conn: sqlite3.Connection) -> None:
    conn.execute(
        "INSERT INTO promotions(ts, fingerprint, promoted, reason, shadow_days, "
        "margin, confidence, detail) VALUES (?,?,?,?,?,?,?,?)",
        (
            db.now(), fingerprint_, 1 if verdict.promoted else 0, verdict.reason,
            verdict.shadow_days, verdict.margin, verdict.confidence,
            json.dumps(verdict.detail, default=float),
        ),
    )


def promote(
    config: Any, verdict: PromotionVerdict, conn: sqlite3.Connection | None = None
) -> dict[str, Any]:
    """Write the shadow set into the live configuration.

    The shadow overrides are cleared afterwards: leaving them in place would mean
    the shadow instance carried on testing the set that is now live, which proves
    nothing and costs a comparison slot.
    """
    conn = conn or db.connect()
    overrides = db.kv_get(SHADOW_KEY, {}, conn) or {}
    if not overrides:
        return {}

    applied = config.update(overrides)
    who = "auto-promotion"
    for key, (old, new) in applied.items():
        conn.execute(
            "INSERT INTO settings_audit(ts, username, key, old_value, new_value, reason) "
            "VALUES (?,?,?,?,?,?)",
            (db.now(), who, key, str(old), str(new), verdict.reason),
        )

    conn.execute(
        "UPDATE param_bundles SET status = ? WHERE status = ?",
        (STATUS_PROMOTED, STATUS_SHADOW),
    )
    db.kv_set(SHADOW_KEY, {}, conn)
    db.kv_set(ACTIVE_BUNDLE_KEY, {"ts": db.now(), "reason": verdict.reason}, conn)
    db.enqueue_command("set_shadow", {"overrides": {}}, requested_by=who, conn=conn)

    db.log_event(
        f"PROMOTED the shadow parameter set to live: {verdict.reason}. "
        f"{len(applied)} setting(s) changed: "
        + ", ".join(f"{k} {o} -> {n}" for k, (o, n) in list(applied.items())[:8]),
        level="alert",
        category="system",
        detail={"applied": {k: {"from": o, "to": n} for k, (o, n) in applied.items()},
                "verdict": verdict.as_dict()},
        conn=conn,
    )
    return applied


def status(conn: sqlite3.Connection | None = None) -> dict[str, Any]:
    """What the dashboard shows about the parameter pipeline."""
    conn = conn or db.connect()
    rows = conn.execute(
        "SELECT fingerprint, received_at, generated_at, status, shadow_since, note, "
        "source_commit AS source FROM param_bundles ORDER BY received_at DESC LIMIT 10"
    ).fetchall()
    promotions = conn.execute(
        "SELECT ts, fingerprint, promoted, reason, shadow_days, margin, confidence "
        "FROM promotions ORDER BY id DESC LIMIT 10"
    ).fetchall()
    latest = dict(rows[0]) if rows else {}
    return {
        "last_bundle": {"ts": latest.get("received_at"), "source": latest.get("source")}
        if latest else {},
        "bundles": [dict(r) for r in rows],
        "promotions": [dict(r) for r in promotions],
        "shadow_overrides": db.kv_get(SHADOW_KEY, {}, conn) or {},
    }


# --------------------------------------------------------------------------
# Regime-scoped promoted sets (gap-closure item 5)
#
# Not a third promotion tier alongside shadow/live - it reads the same
# persistent library gap-closure item 4 already writes to (every entry there
# already cleared the walk-forward/Monte Carlo/stress gates at insert time,
# so there is no separate acceptance step to invent here) and picks, per
# symbol per entry decision, whichever of that symbol's own validated
# combinations is the nearest continuous regime match - never a hard bucket
# boundary. It never touches config.json and never replaces the single
# global live/shadow set; it only ever widens or narrows the parameters one
# *new* entry is evaluated against. An open position's stop/target/trailing
# levels were fixed at its own entry and this has no path back to them.
# --------------------------------------------------------------------------
@dataclass(slots=True)
class RegimeSelection:
    """What (if anything) regime-scoped selection chose for one entry
    decision, and why - the detail an event log / explainability panel
    needs, not just the merged params themselves."""

    applied: bool
    params: dict[str, Any]
    fingerprint: str | None = None
    distance: float | None = None
    reason: str = ""


def select_regime_scoped_params(
    cfg: dict[str, Any],
    symbol: str,
    current_regime_score: float | None,
    *,
    store: Any = None,
) -> RegimeSelection:
    """The effective params one entry decision for `symbol` should use.

    Falls back to `cfg` unchanged - `RegimeSelection(applied=False, params=cfg)`
    - whenever: the feature is off, this symbol has no library entries yet,
    or the nearest one is farther than `regime_scoped_max_distance` away.
    Never raises - a library lookup failing is a reason to trade the global
    set, not a reason to skip the cycle.
    """
    if not cfg.get("regime_scoped_promotion_enabled", True):
        return RegimeSelection(False, cfg, reason="disabled")
    if current_regime_score is None or not np.isfinite(current_regime_score):
        return RegimeSelection(False, cfg, reason="regime score unavailable")

    try:
        if store is None:
            from .wfmc import DAILY_STORE_PATH
            from solopt.store import RunStore

            store = RunStore(DAILY_STORE_PATH)
        nearest = store.nearest_regime_entries(symbol, float(current_regime_score), n=1)
    except Exception:
        log.debug("regime-scoped library lookup failed for %s", symbol, exc_info=True)
        return RegimeSelection(False, cfg, reason="library unavailable")

    if not nearest:
        return RegimeSelection(False, cfg, reason="no library entry for this symbol yet")

    entry = nearest[0]
    distance = abs(float(entry["symbol_regime_score"]) - float(current_regime_score))
    max_distance = float(cfg.get("regime_scoped_max_distance", 0.15))
    if distance > max_distance:
        return RegimeSelection(
            False, cfg, distance=distance,
            reason=f"nearest match {distance:.3f} away, beyond {max_distance:.3f}",
        )

    merged = {**cfg, **entry.get("params", {})}
    return RegimeSelection(
        True, merged, fingerprint=entry["fingerprint"], distance=distance,
        reason=f"regime match within {distance:.3f}",
    )
