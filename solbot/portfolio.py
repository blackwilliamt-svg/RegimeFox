"""Position and balance management.

Every mutation writes through to SQLite immediately - opening, every trailing
stop advance, every invalidation tick, closing. The spec is explicit that the
bot must never lose track of an open position because the process died or the
droplet rebooted, and the only way to guarantee that is to never hold position
state solely in memory.

Each instance (``live``, ``paper``, ``shadow``) keeps its own balance and its
own positions, so the permanent parallel paper run and any shadow rule set can
be compared against live on equal terms.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from typing import Any

from . import db
from .execution import Fill

log = logging.getLogger(__name__)

BALANCE_KEY = "balance:{instance}"


@dataclass(slots=True)
class ClosedTrade:
    position_id: int
    mint: str
    symbol: str
    pnl_usd: float
    pnl_pct: float
    fees_usd: float
    exit_reason: str
    hold_seconds: int


class Portfolio:
    def __init__(self, instance: str, cfg: dict[str, Any]) -> None:
        self.instance = instance
        self.cfg = cfg

    def update_config(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg

    # ------------------------------------------------------------------
    # balance
    # ------------------------------------------------------------------
    def balance(self, conn: sqlite3.Connection | None = None) -> float:
        conn = conn or db.connect()
        value = db.kv_get(BALANCE_KEY.format(instance=self.instance), None, conn)
        if value is None:
            value = float(self.cfg["paper_starting_balance"])
            self.set_balance(value, conn)
        return float(value)

    def set_balance(self, value: float, conn: sqlite3.Connection | None = None) -> None:
        db.kv_set(BALANCE_KEY.format(instance=self.instance), float(value), conn)

    def adjust_balance(self, delta: float, conn: sqlite3.Connection | None = None) -> float:
        conn = conn or db.connect()
        new = self.balance(conn) + delta
        self.set_balance(new, conn)
        return new

    # ------------------------------------------------------------------
    # positions
    # ------------------------------------------------------------------
    def open_positions(self, conn: sqlite3.Connection | None = None) -> list[sqlite3.Row]:
        conn = conn or db.connect()
        return conn.execute(
            "SELECT * FROM positions WHERE instance = ? AND status = 'open' ORDER BY id",
            (self.instance,),
        ).fetchall()

    def open_count(self, conn: sqlite3.Connection | None = None) -> int:
        conn = conn or db.connect()
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM positions WHERE instance = ? AND status = 'open'",
            (self.instance,),
        ).fetchone()
        return int(row["n"])

    def position_for(self, mint: str, conn: sqlite3.Connection | None = None) -> sqlite3.Row | None:
        conn = conn or db.connect()
        return conn.execute(
            "SELECT * FROM positions WHERE instance = ? AND mint = ? AND status = 'open'",
            (self.instance, mint),
        ).fetchone()

    def deployed_usd(self, conn: sqlite3.Connection | None = None) -> float:
        conn = conn or db.connect()
        row = conn.execute(
            "SELECT COALESCE(SUM(size_usd), 0) AS s FROM positions "
            "WHERE instance = ? AND status = 'open'",
            (self.instance,),
        ).fetchone()
        return float(row["s"] or 0.0)

    # ------------------------------------------------------------------
    def open_position(
        self,
        *,
        mint: str,
        symbol: str,
        fill: Fill,
        stop: float,
        target: float,
        rr: float,
        entry_reason: str,
        snapshot: dict[str, Any] | None,
        manual: bool = False,
        conn: sqlite3.Connection | None = None,
    ) -> int:
        conn = conn or db.connect()
        now = db.now()
        risk = max(1e-12, fill.price - stop)

        with db.transaction(conn):
            cur = conn.execute(
                "INSERT INTO positions(instance, mint, symbol, status, entry_price, entry_ts, "
                "qty, size_usd, hard_stop, take_profit, trailing_stop, trailing_armed, "
                "high_water_price, initial_risk, rr_target, entry_reason, entry_snapshot, "
                "entry_fee_usd, entry_tx, manual, updated_at) "
                "VALUES (?,?,?,'open',?,?,?,?,?,?,NULL,0,?,?,?,?,?,?,?,?,?)",
                (
                    self.instance, mint, symbol, fill.price, now, fill.qty, fill.gross_usd,
                    stop, target, fill.price, risk, rr, entry_reason,
                    json.dumps(snapshot or {}), fill.fee_usd, fill.tx_signature,
                    1 if manual else 0, now,
                ),
            )
            position_id = int(cur.lastrowid)
            self.adjust_balance(-fill.gross_usd, conn)

        db.log_event(
            f"OPENED {symbol or mint[:8]} - ${fill.gross_usd:,.2f} at ${fill.price:.6g}, "
            f"stop ${stop:.6g}, target ${target:.6g} ({rr:.1f}:1). {entry_reason}",
            level="info",
            category="trade",
            instance=self.instance,
            mint=mint,
            detail={
                "position_id": position_id,
                "slippage_pct": round(fill.slippage_pct, 3),
                "fee_usd": round(fill.fee_usd, 4),
                "tx": fill.tx_signature,
                "manual": manual,
            },
            conn=conn,
        )
        return position_id

    def update_position(
        self, position_id: int, changes: dict[str, Any], conn: sqlite3.Connection | None = None
    ) -> None:
        """Persist a partial update. Called on every stop advance."""
        conn = conn or db.connect()
        fields = {k: v for k, v in changes.items() if not k.startswith("_")}
        if not fields:
            return
        fields["updated_at"] = db.now()
        assignments = ", ".join(f"{k} = ?" for k in fields)
        conn.execute(
            f"UPDATE positions SET {assignments} WHERE id = ?",
            [*fields.values(), position_id],
        )

    def close_position(
        self,
        position: sqlite3.Row,
        fill: Fill,
        reason: str,
        *,
        manual: bool = False,
        conn: sqlite3.Connection | None = None,
    ) -> ClosedTrade:
        conn = conn or db.connect()
        now = db.now()

        entry_price = float(position["entry_price"])
        size_usd = float(position["size_usd"])
        entry_fee = float(position["entry_fee_usd"] or 0.0)
        proceeds = fill.gross_usd - fill.fee_usd
        total_fees = entry_fee + fill.fee_usd
        pnl = proceeds - size_usd
        pnl_pct = (pnl / size_usd) if size_usd > 0 else 0.0
        hold = now - int(position["entry_ts"])

        with db.transaction(conn):
            conn.execute(
                "UPDATE positions SET status='closed', exit_price=?, exit_ts=?, exit_reason=?, "
                "exit_fee_usd=?, exit_tx=?, pnl_usd=?, pnl_pct=?, manual=?, updated_at=? "
                "WHERE id = ?",
                (
                    fill.price, now, reason, fill.fee_usd, fill.tx_signature, pnl, pnl_pct,
                    1 if (manual or position["manual"]) else 0, now, position["id"],
                ),
            )
            conn.execute(
                "INSERT INTO trades(position_id, instance, mint, symbol, entry_ts, entry_price, "
                "exit_ts, exit_price, qty, size_usd, proceeds_usd, fees_usd, pnl_usd, pnl_pct, "
                "entry_reason, exit_reason, manual, hold_seconds) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    position["id"], self.instance, position["mint"], position["symbol"],
                    position["entry_ts"], entry_price, now, fill.price, fill.qty, size_usd,
                    proceeds, total_fees, pnl, pnl_pct, position["entry_reason"], reason,
                    1 if (manual or position["manual"]) else 0, hold,
                ),
            )
            self.adjust_balance(proceeds, conn)

        tag = "MANUAL " if manual else ""
        db.log_event(
            f"{tag}CLOSED {position['symbol'] or position['mint'][:8]} at ${fill.price:.6g} - "
            f"{'+' if pnl >= 0 else ''}${pnl:,.2f} ({pnl_pct * 100:+.2f}%) after "
            f"${total_fees:,.2f} fees. {reason}",
            level="info" if pnl >= 0 else "warn",
            category="trade",
            instance=self.instance,
            mint=position["mint"],
            detail={
                "position_id": position["id"],
                "pnl_usd": round(pnl, 4),
                "hold_seconds": hold,
                "tx": fill.tx_signature,
                "manual": manual,
            },
            conn=conn,
        )
        return ClosedTrade(
            position_id=int(position["id"]),
            mint=position["mint"],
            symbol=position["symbol"] or "",
            pnl_usd=pnl,
            pnl_pct=pnl_pct,
            fees_usd=total_fees,
            exit_reason=reason,
            hold_seconds=hold,
        )

    # ------------------------------------------------------------------
    # valuation
    # ------------------------------------------------------------------
    def unrealised(
        self, prices: dict[str, Any], conn: sqlite3.Connection | None = None
    ) -> tuple[float, list[dict[str, Any]]]:
        conn = conn or db.connect()
        rows = self.open_positions(conn)
        total = 0.0
        detail: list[dict[str, Any]] = []
        for r in rows:
            price_info = prices.get(r["mint"])
            price = getattr(price_info, "price", None) or float(r["entry_price"])
            value = float(r["qty"]) * price
            pnl = value - float(r["size_usd"])
            total += pnl
            detail.append(
                {
                    "id": r["id"],
                    "mint": r["mint"],
                    "symbol": r["symbol"],
                    "price": price,
                    "value_usd": value,
                    "pnl_usd": pnl,
                    "pnl_pct": pnl / float(r["size_usd"]) if r["size_usd"] else 0.0,
                }
            )
        return total, detail

    def equity(self, prices: dict[str, Any], conn: sqlite3.Connection | None = None) -> float:
        conn = conn or db.connect()
        unreal, _ = self.unrealised(prices, conn)
        return self.balance(conn) + self.deployed_usd(conn) + unreal

    def record_equity(
        self, prices: dict[str, Any], conn: sqlite3.Connection | None = None
    ) -> float:
        conn = conn or db.connect()
        balance = self.balance(conn)
        eq = self.equity(prices, conn)
        conn.execute(
            "INSERT INTO equity(instance, ts, balance, equity) VALUES (?,?,?,?) "
            "ON CONFLICT(instance, ts) DO UPDATE SET balance=excluded.balance, "
            "equity=excluded.equity",
            (self.instance, db.now(), balance, eq),
        )
        return eq

    # ------------------------------------------------------------------
    # performance
    # ------------------------------------------------------------------
    def performance(
        self, *, since: int | None = None, conn: sqlite3.Connection | None = None
    ) -> dict[str, Any]:
        """Win rate, average win/loss, profit factor, total fees, drawdown."""
        conn = conn or db.connect()
        sql = "SELECT pnl_usd, pnl_pct, fees_usd FROM trades WHERE instance = ?"
        args: list[Any] = [self.instance]
        if since:
            sql += " AND exit_ts >= ?"
            args.append(since)
        rows = conn.execute(sql + " ORDER BY exit_ts", args).fetchall()

        if not rows:
            return {
                "instance": self.instance, "trades": 0, "wins": 0, "losses": 0,
                "win_rate": 0.0, "avg_win": 0.0, "avg_loss": 0.0, "profit_factor": 0.0,
                "total_pnl": 0.0, "total_fees": 0.0, "max_drawdown": 0.0,
                "expectancy": 0.0,
            }

        pnls = [float(r["pnl_usd"]) for r in rows]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        gross_win = sum(wins)
        gross_loss = abs(sum(losses))

        equity_curve, peak, max_dd = 0.0, 0.0, 0.0
        for p in pnls:
            equity_curve += p
            peak = max(peak, equity_curve)
            if peak > 0:
                max_dd = max(max_dd, (peak - equity_curve) / peak)

        return {
            "instance": self.instance,
            "trades": len(pnls),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / len(pnls),
            "avg_win": (gross_win / len(wins)) if wins else 0.0,
            "avg_loss": (-gross_loss / len(losses)) if losses else 0.0,
            "profit_factor": (gross_win / gross_loss) if gross_loss > 0 else (
                float("inf") if gross_win > 0 else 0.0
            ),
            "total_pnl": sum(pnls),
            "total_fees": sum(float(r["fees_usd"]) for r in rows),
            "max_drawdown": max_dd,
            "expectancy": sum(pnls) / len(pnls),
        }
