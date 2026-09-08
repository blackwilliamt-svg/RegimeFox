"""Risk controls: sizing, correlation, circuit breaker, kill switch.

The circuit breaker and kill switch both live in the ``kv`` table rather than in
memory. That is deliberate: the systemd unit restarts the worker on failure, and
if these lived in process memory a crash-restart would silently clear a halt the
bot had just triggered. Persisting them means a restart can never be used - even
accidentally - to resume trading after a breaker trip.
"""
from __future__ import annotations

import logging
import math
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from . import db

log = logging.getLogger(__name__)

KEY_KILL_SWITCH = "kill_switch"
KEY_CIRCUIT = "circuit_breaker"
KEY_RUN = "engine_run"
KEY_HEARTBEAT = "worker_heartbeat"
KEY_DAY_ANCHOR = "day_anchor"


@dataclass(slots=True)
class SizingResult:
    size_usd: float
    qty: float
    reasons: list[str] = field(default_factory=list)
    capped_by: str = ""
    volatility_scalar: float = 1.0

    @property
    def ok(self) -> bool:
        return self.size_usd > 0


@dataclass(slots=True)
class GateResult:
    allowed: bool
    reason: str = ""
    detail: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------
# Kill switch (manual) - distinct from the automatic circuit breaker
# --------------------------------------------------------------------------
def kill_switch_engaged(conn: sqlite3.Connection | None = None) -> bool:
    return bool((db.kv_get(KEY_KILL_SWITCH, {}, conn) or {}).get("engaged"))


def engage_kill_switch(reason: str, by: str = "dashboard", conn: sqlite3.Connection | None = None) -> None:
    db.kv_set(
        KEY_KILL_SWITCH,
        {"engaged": True, "reason": reason, "by": by, "at": db.now()},
        conn,
    )
    db.log_event(
        f"KILL SWITCH ENGAGED by {by}: {reason}. No new entries will be taken; "
        "open positions remain for manual closing.",
        level="alert",
        category="risk",
        conn=conn,
    )


def release_kill_switch(by: str = "dashboard", conn: sqlite3.Connection | None = None) -> None:
    db.kv_set(KEY_KILL_SWITCH, {"engaged": False, "released_by": by, "at": db.now()}, conn)
    db.log_event(
        f"Kill switch released by {by}. Trading may resume.",
        level="warn",
        category="risk",
        conn=conn,
    )


def kill_switch_state(conn: sqlite3.Connection | None = None) -> dict[str, Any]:
    return db.kv_get(KEY_KILL_SWITCH, {"engaged": False}, conn) or {"engaged": False}


# --------------------------------------------------------------------------
# Circuit breaker (automatic) - requires a manual reset, never auto-resumes
# --------------------------------------------------------------------------
def circuit_state(conn: sqlite3.Connection | None = None) -> dict[str, Any]:
    return db.kv_get(KEY_CIRCUIT, {"tripped": False, "consecutive_losses": 0}, conn) or {
        "tripped": False,
        "consecutive_losses": 0,
    }


def circuit_tripped(conn: sqlite3.Connection | None = None) -> bool:
    return bool(circuit_state(conn).get("tripped"))


def reset_circuit(by: str = "dashboard", conn: sqlite3.Connection | None = None) -> None:
    state = circuit_state(conn)
    state.update(
        {"tripped": False, "consecutive_losses": 0, "reset_by": by, "reset_at": db.now()}
    )
    db.kv_set(KEY_CIRCUIT, state, conn)
    db.log_event(
        f"Circuit breaker manually reset by {by}.", level="warn", category="risk", conn=conn
    )


def record_trade_result(
    pnl_usd: float, cfg: dict[str, Any], conn: sqlite3.Connection | None = None
) -> dict[str, Any]:
    """Update the consecutive-loss counter and trip the breaker if warranted."""
    state = circuit_state(conn)
    if pnl_usd < 0:
        state["consecutive_losses"] = int(state.get("consecutive_losses", 0)) + 1
    else:
        state["consecutive_losses"] = 0

    limit = int(cfg["circuit_consecutive_losses"])
    if state["consecutive_losses"] >= limit and not state.get("tripped"):
        state["tripped"] = True
        state["reason"] = f"{state['consecutive_losses']} consecutive losing trades"
        state["tripped_at"] = db.now()
        db.log_event(
            f"CIRCUIT BREAKER TRIPPED: {state['reason']}. New trades are paused until "
            "manually reset from the dashboard.",
            level="alert",
            category="risk",
            conn=conn,
        )
    db.kv_set(KEY_CIRCUIT, state, conn)
    return state


def check_daily_drawdown(
    instance: str, equity_now: float, cfg: dict[str, Any], conn: sqlite3.Connection | None = None
) -> bool:
    """Trip the breaker if today's drawdown exceeds the threshold.

    The day's high-water mark is anchored in the database so an intraday restart
    does not reset the drawdown measurement to the post-loss balance.
    """
    conn = conn or db.connect()
    today = time.strftime("%Y-%m-%d", time.gmtime())
    anchor = db.kv_get(KEY_DAY_ANCHOR, {}, conn) or {}
    key = f"{instance}:{today}"

    if anchor.get("key") != key:
        anchor = {"key": key, "peak": equity_now, "start": equity_now}
    anchor["peak"] = max(float(anchor.get("peak", equity_now)), equity_now)
    db.kv_set(KEY_DAY_ANCHOR, anchor, conn)

    peak = float(anchor["peak"])
    if peak <= 0:
        return False
    drawdown = (peak - equity_now) / peak
    threshold = float(cfg["circuit_daily_drawdown_pct"])
    if drawdown < threshold:
        return False

    state = circuit_state(conn)
    if state.get("tripped"):
        return True
    state.update(
        {
            "tripped": True,
            "reason": f"daily drawdown {drawdown * 100:.1f}% exceeded "
                      f"{threshold * 100:.1f}% (peak ${peak:,.2f} -> ${equity_now:,.2f})",
            "tripped_at": db.now(),
        }
    )
    db.kv_set(KEY_CIRCUIT, state, conn)
    db.log_event(
        f"CIRCUIT BREAKER TRIPPED: {state['reason']}. Manual reset required.",
        level="alert",
        category="risk",
        instance=instance,
        conn=conn,
    )
    return True


# --------------------------------------------------------------------------
# Position sizing
# --------------------------------------------------------------------------
def volatility_scalar(atr_pct: float, cfg: dict[str, Any]) -> float:
    """Size down for choppier tokens.

    A token whose ATR is twice the target volatility gets half the size. The
    floor stops a very quiet token from being scaled up past the wallet cap and
    a very wild one from being scaled to dust.
    """
    target = float(cfg["volatility_target_atr_pct"])
    floor = float(cfg["volatility_size_floor"])
    if atr_pct <= 0 or target <= 0:
        return 1.0
    return float(max(floor, min(1.0, target / atr_pct)))


@dataclass(slots=True)
class OpenExposure:
    """One already-open position, as the sizing maths sees it."""

    size_usd: float
    atr_pct: float
    correlation: float = 0.0


def portfolio_weight(
    *,
    candidate_vol: float,
    open_book: Sequence[OpenExposure],
    wallet_usd: float,
    target_vol: float,
    max_weight: float,
) -> float:
    """The largest weight that keeps portfolio volatility at the target.

    For a book already holding weights ``w_i`` at volatilities ``s_i``, adding
    weight ``w`` at volatility ``s`` gives

        sigma_p^2 = existing + 2*w*s*sum(rho_i * w_i * s_i) + w^2 * s^2

    which is a quadratic in ``w``. Solving it is what turns "scale by
    correlation and volatility" into an actual number: a candidate that moves
    with what is already open gets a smaller slice because its marginal
    contribution to portfolio risk is larger, not because a heuristic said so.

    This mirrors ``solopt.engine.portfolio_weight`` exactly; if the two ever
    diverge the optimizer stops sizing the way the live bot does, and every
    drawdown figure it produces becomes fiction.
    """
    if candidate_vol <= 0 or target_vol <= 0 or wallet_usd <= 0:
        return max_weight

    weights = [max(0.0, e.size_usd / wallet_usd) for e in open_book]
    vols = [max(0.0, e.atr_pct) for e in open_book]

    existing_var = 0.0
    for i, w_i in enumerate(weights):
        existing_var += (w_i * vols[i]) ** 2
        for j in range(i + 1, len(weights)):
            existing_var += 2.0 * w_i * weights[j] * vols[i] * vols[j]
    cross = sum(
        open_book[i].correlation * weights[i] * vols[i] for i in range(len(weights))
    )

    a = candidate_vol**2
    b = 2.0 * candidate_vol * cross
    c = existing_var - target_vol**2
    if c >= 0:
        return 0.0   # the book is already at or over budget
    disc = b * b - 4.0 * a * c
    if disc <= 0:
        return 0.0
    root = (-b + math.sqrt(disc)) / (2.0 * a)
    return float(max(0.0, min(max_weight, root)))


def drawdown_scalar(p5_drawdown: float, tolerance: float, floor: float = 0.25) -> float:
    """Shrink the risk budget when the Monte Carlo tail is worse than tolerated.

    The historical backtest drawdown is one draw from a distribution; the 5th
    percentile of the resampled distribution is the one worth sizing against.
    At or below tolerance this returns 1.0 and changes nothing, so a bot that has
    never had a Monte Carlo run behaves exactly as before.
    """
    if p5_drawdown <= 0 or tolerance <= 0 or p5_drawdown <= tolerance:
        return 1.0
    return float(max(floor, tolerance / p5_drawdown))


def size_position(
    *,
    wallet_usd: float,
    liquidity_usd: float,
    price: float,
    atr_pct: float,
    cfg: dict[str, Any],
    sol_balance: float | None = None,
    sol_price: float | None = None,
    deployed_usd: float = 0.0,
    open_book: Sequence[OpenExposure] = (),
) -> SizingResult:
    """Position size, capped by every constraint the spec lists.

    Order of caps: wallet percentage, pool depth, remaining un-deployed capital,
    then volatility. The gas reserve is removed from the tradeable balance
    *before* any of it, so a new position can never dip into the SOL set aside
    for fees.

    The final step depends on ``sizing_mode``. ``flat`` scales by the token's own
    volatility alone. ``portfolio`` (spec 4.4) instead solves for the weight that
    holds *portfolio* volatility at target given what is already open and how
    correlated the candidate is with it, then shrinks that target further if the
    Monte Carlo 5%-worst-case drawdown exceeds the configured tolerance. Both
    stay inside the hard caps; this only ever sizes down.
    """
    reasons: list[str] = []

    tradeable = wallet_usd
    if sol_balance is not None and sol_price:
        reserve_usd = float(cfg["gas_reserve_sol"]) * float(sol_price)
        tradeable = wallet_usd - reserve_usd
        if tradeable <= 0:
            return SizingResult(
                0.0,
                0.0,
                [
                    f"entire ${wallet_usd:,.2f} balance is inside the "
                    f"{cfg['gas_reserve_sol']} SOL gas reserve"
                ],
                capped_by="gas_reserve",
            )

    by_wallet = tradeable * float(cfg["max_position_pct_of_wallet"])
    by_liquidity = liquidity_usd * float(cfg["max_position_pct_of_liquidity"])

    # Never deploy past the total-deployed cap, so the untouched reserve the
    # spec asks for actually stays untouched. There is deliberately no cap on
    # how many positions make up that total - the bot decides that itself, as
    # an output of this same sizing logic, not from a fixed count.
    max_total = tradeable * float(cfg["max_total_deployed_pct"])
    remaining = max(0.0, max_total - deployed_usd)

    caps = {"wallet": by_wallet, "liquidity": by_liquidity, "remaining_capital": remaining}
    size = min(caps.values())
    capped_by = min(caps, key=lambda k: caps[k])

    # The token's own volatility scales the size in both modes. Portfolio mode
    # then applies a second cap on top - it never lifts this one, which is what
    # keeps switching modes a refinement rather than a silent risk increase.
    scalar = volatility_scalar(atr_pct, cfg)
    if scalar < 1.0:
        reasons.append(f"volatility scaled to {scalar * 100:.0f}% (ATR {atr_pct * 100:.1f}%)")
    size *= scalar

    if str(cfg.get("sizing_mode", "flat")) == "portfolio":
        tolerance = float(cfg.get("drawdown_tolerance", 0.25))
        p5 = float(cfg.get("monte_carlo_p5_drawdown", 0.0))
        shrink = drawdown_scalar(p5, tolerance)
        target = float(cfg.get("portfolio_vol_target", 0.019)) * shrink
        weight = portfolio_weight(
            candidate_vol=atr_pct,
            open_book=open_book,
            wallet_usd=tradeable,
            target_vol=target,
            max_weight=float(cfg["max_position_pct_of_wallet"]),
        )
        by_portfolio = tradeable * weight
        if by_portfolio < size:
            size = by_portfolio
            capped_by = "portfolio_volatility"
            if open_book:
                worst = max(abs(e.correlation) for e in open_book)
                reasons.append(
                    f"portfolio volatility budget: {weight * 100:.1f}% of the wallet "
                    f"(ATR {atr_pct * 100:.1f}%, correlation {worst:.2f} with what is open)"
                )
            else:
                reasons.append(
                    f"portfolio volatility budget: {weight * 100:.1f}% of the wallet "
                    f"(ATR {atr_pct * 100:.1f}%)"
                )
        if shrink < 1.0:
            reasons.append(
                f"risk budget cut to {shrink * 100:.0f}% - the Monte Carlo 5% "
                f"worst-case drawdown of {p5 * 100:.1f}% exceeds the "
                f"{tolerance * 100:.0f}% tolerance"
            )

    if size < float(cfg["min_position_usd"]):
        return SizingResult(
            0.0,
            0.0,
            reasons + [
                f"sized ${size:,.2f}, below the ${cfg['min_position_usd']:,.2f} minimum "
                f"(capped by {capped_by})"
            ],
            capped_by=capped_by,
            volatility_scalar=scalar,
        )

    qty = size / price if price > 0 else 0.0
    reasons.append(f"capped by {capped_by} at ${size:,.2f}")
    return SizingResult(size, qty, reasons, capped_by=capped_by, volatility_scalar=scalar)


# --------------------------------------------------------------------------
# Correlation
# --------------------------------------------------------------------------
def correlation(a: Sequence[float], b: Sequence[float]) -> float | None:
    """Pearson correlation of the two return series, or None if uncomputable."""
    n = min(len(a), len(b))
    if n < 10:
        return None
    x = np.asarray(a[-n:], dtype=float)
    y = np.asarray(b[-n:], dtype=float)
    rx = np.diff(x) / np.where(x[:-1] == 0, np.nan, x[:-1])
    ry = np.diff(y) / np.where(y[:-1] == 0, np.nan, y[:-1])
    mask = ~(np.isnan(rx) | np.isnan(ry))
    rx, ry = rx[mask], ry[mask]
    if len(rx) < 8 or rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def correlation_gate(
    candidate_closes: Sequence[float],
    open_closes: dict[str, Sequence[float]],
    cfg: dict[str, Any],
) -> GateResult:
    """Refuse a second position that is just riding the first one's move."""
    limit = float(cfg["correlation_max"])
    worst_mint, worst = "", 0.0
    for mint, closes in open_closes.items():
        c = correlation(candidate_closes, closes)
        if c is None:
            continue
        if abs(c) > abs(worst):
            worst_mint, worst = mint, c
    if worst_mint and abs(worst) >= limit:
        return GateResult(
            False,
            f"correlation {worst:.2f} with the open {worst_mint[:6]}… position "
            f"exceeds {limit:.2f}",
            {"mint": worst_mint, "correlation": worst},
        )
    return GateResult(True, detail={"max_correlation": worst, "against": worst_mint})


# --------------------------------------------------------------------------
# Combined pre-trade gate
# --------------------------------------------------------------------------
def can_open_position(
    *,
    cfg: dict[str, Any],
    conn: sqlite3.Connection | None = None,
) -> GateResult:
    """Cheap checks that run before any network call is spent on a candidate.

    There is no position-count ceiling here: how many positions may be open at
    once is an output of :func:`size_position`'s total-deployed cap, not a
    fixed number checked up front.
    """
    if kill_switch_engaged(conn):
        state = kill_switch_state(conn)
        return GateResult(False, f"kill switch engaged ({state.get('reason', 'manual')})")
    if circuit_tripped(conn):
        state = circuit_state(conn)
        return GateResult(False, f"circuit breaker tripped: {state.get('reason', 'unknown')}")
    return GateResult(True)
