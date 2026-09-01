"""Startup recovery and on-chain reconciliation.

Runs before the engine takes a single new action. Three steps, in order:

1. **Reload open positions from SQLite.** Never start with a clean slate while
   real funds are exposed.
2. **Re-fetch current prices and recompute stop levels**, so a position that
   moved while the process was down is managed against reality rather than
   against whatever the stops were when it died.
3. **Reconcile against the actual on-chain wallet** when live. If the database
   and the wallet disagree, the engine does *not* guess - it halts, raises an
   alert, and waits for manual review.

Every restart and every recovery action is written to the event feed.
"""
from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from typing import Any

from . import db, risk
from .clients import Clients
from .datastore import DataStore
from .indicators import compute, snapshot_at
from .portfolio import Portfolio

log = logging.getLogger(__name__)

KEY_RECONCILE_HALT = "reconcile_halt"

# A position's on-chain quantity will drift slightly from the recorded fill
# (rounding, transfer dust). Only a material gap is treated as a disagreement.
QTY_TOLERANCE_PCT = 2.0


@dataclass
class RecoveryReport:
    positions_recovered: int = 0
    stops_updated: int = 0
    reconciled: bool = False
    halted: bool = False
    discrepancies: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "positions_recovered": self.positions_recovered,
            "stops_updated": self.stops_updated,
            "reconciled": self.reconciled,
            "halted": self.halted,
            "discrepancies": self.discrepancies,
            "notes": self.notes,
        }


def reconcile_halted(conn: sqlite3.Connection | None = None) -> bool:
    return bool((db.kv_get(KEY_RECONCILE_HALT, {}, conn) or {}).get("halted"))


def clear_reconcile_halt(by: str = "dashboard", conn: sqlite3.Connection | None = None) -> None:
    db.kv_set(KEY_RECONCILE_HALT, {"halted": False, "cleared_by": by, "at": db.now()}, conn)
    db.log_event(
        f"Reconciliation halt cleared by {by} after manual review.",
        level="warn",
        category="system",
        conn=conn,
    )


def _halt(reason: str, detail: dict[str, Any], conn: sqlite3.Connection) -> None:
    db.kv_set(
        KEY_RECONCILE_HALT,
        {"halted": True, "reason": reason, "at": db.now(), "detail": detail},
        conn,
    )
    db.log_event(
        f"TRADING HALTED - state reconciliation failed: {reason}. "
        "Manual review required before trading resumes.",
        level="alert",
        category="system",
        detail=detail,
        conn=conn,
    )


def recover(
    *,
    portfolio: Portfolio,
    clients: Clients,
    store: DataStore,
    cfg: dict[str, Any],
    executor: Any = None,
    conn: sqlite3.Connection | None = None,
) -> RecoveryReport:
    """Restore managed state after a restart. Safe to call on a clean start."""
    conn = conn or db.connect()
    report = RecoveryReport()

    db.log_event(
        f"Worker starting ({portfolio.instance} instance, "
        f"{cfg['trading_mode']} mode). Reloading persisted state.",
        category="system",
        instance=portfolio.instance,
        conn=conn,
    )

    positions = portfolio.open_positions(conn)
    report.positions_recovered = len(positions)

    if not positions:
        report.notes.append("no open positions to recover")
    else:
        mints = [p["mint"] for p in positions]
        prices = clients.jupiter.prices(mints, priority="high")
        report.notes.append(f"re-fetched prices for {len(prices)}/{len(mints)} held tokens")

        for pos in positions:
            price_info = prices.get(pos["mint"])
            if price_info is None:
                # No price for a token being held is itself a warning sign.
                db.log_event(
                    f"Recovered position in {pos['symbol'] or pos['mint'][:8]} but no live "
                    "price is available - the token may have lost liquidity.",
                    level="alert",
                    category="system",
                    instance=portfolio.instance,
                    mint=pos["mint"],
                    conn=conn,
                )
                continue

            changes = _recompute_stops(pos, price_info.price, store, cfg, conn)
            if changes:
                portfolio.update_position(int(pos["id"]), changes, conn)
                report.stops_updated += 1

            pnl_pct = (price_info.price / float(pos["entry_price"]) - 1.0) * 100.0
            db.log_event(
                f"Recovered position {pos['symbol'] or pos['mint'][:8]}: entered "
                f"${float(pos['entry_price']):.6g}, now ${price_info.price:.6g} "
                f"({pnl_pct:+.2f}%). Stop ${float(pos['hard_stop']):.6g}"
                + (
                    f", trail ${float(changes['trailing_stop']):.6g}"
                    if changes.get("trailing_stop")
                    else ""
                ),
                category="system",
                instance=portfolio.instance,
                mint=pos["mint"],
                detail={"position_id": pos["id"], "changes": _jsonable(changes)},
                conn=conn,
            )

    # --- on-chain reconciliation (live only) ------------------------------
    if cfg["trading_mode"] == "live" and executor is not None and hasattr(executor, "wallet_snapshot"):
        report.reconciled = True
        try:
            snapshot = executor.wallet_snapshot()
        except Exception as exc:
            _halt(f"could not read on-chain wallet state ({exc})", {"error": str(exc)}, conn)
            report.halted = True
            report.discrepancies.append(str(exc))
            return report

        discrepancies = _compare_wallet(positions, snapshot, cfg)
        report.discrepancies = discrepancies
        if discrepancies:
            _halt(
                "database and wallet disagree",
                {"discrepancies": discrepancies, "wallet": snapshot.get("pubkey")},
                conn,
            )
            report.halted = True
            return report

        report.notes.append(
            f"wallet reconciled: {snapshot['sol']:.4f} SOL, "
            f"{len(snapshot['tokens'])} token balances match the database"
        )
        db.log_event(
            f"On-chain reconciliation passed: {snapshot['sol']:.4f} SOL held, "
            f"{report.positions_recovered} position(s) match.",
            category="system",
            instance=portfolio.instance,
            conn=conn,
        )

        gas_reserve = float(cfg["gas_reserve_sol"])
        if snapshot["sol"] < gas_reserve:
            db.log_event(
                f"SOL balance {snapshot['sol']:.4f} is below the {gas_reserve} gas reserve. "
                "New positions will be refused until the wallet is topped up.",
                level="alert",
                category="risk",
                instance=portfolio.instance,
                conn=conn,
            )

    _log_persisted_halts(conn)
    return report


def _recompute_stops(
    pos: sqlite3.Row,
    price: float,
    store: DataStore,
    cfg: dict[str, Any],
    conn: sqlite3.Connection,
) -> dict[str, Any]:
    """Re-derive the trailing stop from the price action that happened while down."""
    from .strategy import update_trailing_stop  # local import avoids a cycle

    df = store.candles_for(pos["mint"], limit=int(cfg["min_candles_required"]) + 5, conn=conn)
    atr_value = 0.0
    if not df.empty:
        snap = snapshot_at(compute(df, cfg), -1)
        if snap:
            atr_value = snap.atr
    if atr_value <= 0:
        atr_value = price * 0.01  # 1% fallback when history is thin

    position = dict(pos)
    # The high-water mark may have moved while the process was down.
    position["high_water_price"] = max(
        float(position.get("high_water_price") or position["entry_price"]), price
    )
    changes = update_trailing_stop(position, price, atr_value, cfg)
    changes.pop("_armed_now", None)
    if position["high_water_price"] != (pos["high_water_price"] or 0):
        changes.setdefault("high_water_price", position["high_water_price"])
    return changes


def _compare_wallet(
    positions: list[sqlite3.Row], snapshot: dict[str, Any], cfg: dict[str, Any]
) -> list[str]:
    """Every way the database and the chain can disagree about what is held."""
    problems: list[str] = []
    on_chain = {m: float(v) for m, v in (snapshot.get("tokens") or {}).items()}

    for pos in positions:
        mint = pos["mint"]
        expected = float(pos["qty"])
        actual = on_chain.get(mint, 0.0)
        if actual <= 0:
            problems.append(
                f"database holds {expected:.6g} of {pos['symbol'] or mint[:8]} "
                f"but the wallet holds none"
            )
            continue
        drift = abs(actual - expected) / expected * 100.0 if expected > 0 else 100.0
        if drift > QTY_TOLERANCE_PCT:
            problems.append(
                f"{pos['symbol'] or mint[:8]}: database says {expected:.6g}, "
                f"wallet holds {actual:.6g} ({drift:.1f}% apart)"
            )

    tracked = {p["mint"] for p in positions}
    for mint, amount in on_chain.items():
        if mint in tracked or amount <= 0:
            continue
        # An untracked balance may be legitimate dust, but it may also be a
        # position the database lost. Surfacing it is the point.
        problems.append(
            f"wallet holds {amount:.6g} of untracked token {mint[:8]}… "
            "with no matching open position"
        )

    return problems


def _log_persisted_halts(conn: sqlite3.Connection) -> None:
    """Re-announce any halt that survived the restart, so it is not a surprise."""
    if risk.kill_switch_engaged(conn):
        state = risk.kill_switch_state(conn)
        db.log_event(
            f"Kill switch is still engaged from before the restart "
            f"({state.get('reason', 'manual')}). No new entries will be taken.",
            level="alert",
            category="risk",
            conn=conn,
        )
    if risk.circuit_tripped(conn):
        state = risk.circuit_state(conn)
        db.log_event(
            f"Circuit breaker is still tripped from before the restart "
            f"({state.get('reason', 'unknown')}). Manual reset required.",
            level="alert",
            category="risk",
            conn=conn,
        )
    if reconcile_halted(conn):
        state = db.kv_get(KEY_RECONCILE_HALT, {}, conn) or {}
        db.log_event(
            f"Reconciliation halt is still in force ({state.get('reason', 'unknown')}). "
            "Clear it from the dashboard after reviewing.",
            level="alert",
            category="system",
            conn=conn,
        )


def _jsonable(d: dict[str, Any]) -> dict[str, Any]:
    return {k: (round(v, 10) if isinstance(v, float) else v) for k, v in d.items()}
