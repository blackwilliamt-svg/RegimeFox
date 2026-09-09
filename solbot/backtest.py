"""Offline backtest.

Runs the identical entry/exit functions the live engine uses, against candles
already stored locally. Nothing here touches the network - re-fetching live data
per run would make results irreproducible for no benefit.

**Survivorship bias is handled at the basket level, not the simulation level.**
The basket comes from ``universe_history`` - what actually passed the filters on
each historical day - and deliberately keeps tokens that later went to zero. A
basket assembled from what is tradeable *today* would silently exclude every
token that rugged or got delisted during the test window and would make the
results look considerably better than reality.

The purpose is to sanity-check thresholds (volume-spike sensitivity, the
reward:risk window, where the trailing stop arms), not to hunt for a curve-fit
"perfect" historical return.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from itertools import groupby
from typing import Any, Callable, Sequence

import pandas as pd

from . import db
from .datastore import DataStore
from .indicators import compute
from .risk import size_position, volatility_scalar
from .strategy import evaluate_entry, evaluate_exit, update_trailing_stop
from .universe import historical_basket

log = logging.getLogger(__name__)

JOB_BACKTEST = "backtest"


@dataclass
class BacktestTrade:
    mint: str
    symbol: str
    entry_ts: int
    entry_price: float
    exit_ts: int
    exit_price: float
    size_usd: float
    qty: float
    pnl_usd: float
    pnl_pct: float
    fees_usd: float
    entry_reason: str
    exit_reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "mint": self.mint, "symbol": self.symbol,
            "entry_ts": self.entry_ts, "entry_price": self.entry_price,
            "exit_ts": self.exit_ts, "exit_price": self.exit_price,
            "size_usd": round(self.size_usd, 2), "qty": self.qty,
            "pnl_usd": round(self.pnl_usd, 4), "pnl_pct": round(self.pnl_pct, 5),
            "fees_usd": round(self.fees_usd, 4),
            "entry_reason": self.entry_reason, "exit_reason": self.exit_reason,
        }


@dataclass
class BacktestResult:
    trades: list[BacktestTrade] = field(default_factory=list)
    starting_balance: float = 0.0
    ending_balance: float = 0.0
    tokens_tested: int = 0
    tokens_with_data: int = 0
    bars_evaluated: int = 0
    from_ts: int = 0
    to_ts: int = 0
    equity_curve: list[tuple[int, float]] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)

    # -- metrics -------------------------------------------------------
    @property
    def total_pnl(self) -> float:
        return sum(t.pnl_usd for t in self.trades)

    @property
    def wins(self) -> list[BacktestTrade]:
        return [t for t in self.trades if t.pnl_usd > 0]

    @property
    def losses(self) -> list[BacktestTrade]:
        return [t for t in self.trades if t.pnl_usd <= 0]

    @property
    def win_rate(self) -> float:
        return len(self.wins) / len(self.trades) if self.trades else 0.0

    @property
    def profit_factor(self) -> float:
        gross_win = sum(t.pnl_usd for t in self.wins)
        gross_loss = abs(sum(t.pnl_usd for t in self.losses))
        if gross_loss > 0:
            return gross_win / gross_loss
        return float("inf") if gross_win > 0 else 0.0

    @property
    def avg_win(self) -> float:
        return sum(t.pnl_usd for t in self.wins) / len(self.wins) if self.wins else 0.0

    @property
    def avg_loss(self) -> float:
        return sum(t.pnl_usd for t in self.losses) / len(self.losses) if self.losses else 0.0

    @property
    def max_drawdown(self) -> float:
        peak, max_dd = self.starting_balance, 0.0
        for _, equity in self.equity_curve:
            peak = max(peak, equity)
            if peak > 0:
                max_dd = max(max_dd, (peak - equity) / peak)
        return max_dd

    @property
    def total_return(self) -> float:
        if self.starting_balance <= 0:
            return 0.0
        return (self.ending_balance - self.starting_balance) / self.starting_balance

    @property
    def total_fees(self) -> float:
        return sum(t.fees_usd for t in self.trades)

    def summary(self) -> dict[str, Any]:
        return {
            "trades": len(self.trades),
            "wins": len(self.wins),
            "losses": len(self.losses),
            "win_rate": round(self.win_rate, 4),
            "profit_factor": (
                round(self.profit_factor, 3) if self.profit_factor != float("inf") else None
            ),
            "avg_win": round(self.avg_win, 2),
            "avg_loss": round(self.avg_loss, 2),
            "total_pnl": round(self.total_pnl, 2),
            "total_fees": round(self.total_fees, 2),
            "max_drawdown": round(self.max_drawdown, 4),
            "total_return": round(self.total_return, 4),
            "starting_balance": round(self.starting_balance, 2),
            "ending_balance": round(self.ending_balance, 2),
            "tokens_tested": self.tokens_tested,
            "tokens_with_data": self.tokens_with_data,
            "bars_evaluated": self.bars_evaluated,
            "from_ts": self.from_ts,
            "to_ts": self.to_ts,
        }


class Backtester:
    def __init__(self, store: DataStore, cfg: dict[str, Any]) -> None:
        self.store = store
        self.cfg = cfg

    def update_config(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg

    # ------------------------------------------------------------------
    def basket(self, days: int, conn: sqlite3.Connection | None = None) -> list[str]:
        """Point-in-time basket, including tokens that later died."""
        conn = conn or db.connect()
        now = db.now()
        day_from = time.strftime("%Y-%m-%d", time.gmtime(now - days * 86400))
        day_to = time.strftime("%Y-%m-%d", time.gmtime(now))
        mints = historical_basket(day_from, day_to, conn=conn)
        if mints:
            return mints
        # No history recorded yet (first run). Fall back to the current universe
        # and say so - these results ARE survivorship-biased until the history
        # table has accumulated real snapshots.
        rows = conn.execute("SELECT mint FROM universe").fetchall()
        log.warning(
            "universe_history is empty; falling back to today's universe. "
            "These results carry survivorship bias until snapshots accumulate."
        )
        return [r["mint"] for r in rows]

    # ------------------------------------------------------------------
    def run(
        self,
        *,
        days: int | None = None,
        mints: Sequence[str] | None = None,
        conn: sqlite3.Connection | None = None,
        progress: Callable[[int, int, str], None] | None = None,
    ) -> BacktestResult:
        conn = conn or db.connect()
        cfg = self.cfg
        days = days or int(cfg["backtest_lookback_days"])
        now = db.now()
        from_ts = now - days * 86400

        basket = list(mints) if mints is not None else self.basket(days, conn)
        result = BacktestResult(
            starting_balance=float(cfg["paper_starting_balance"]),
            ending_balance=float(cfg["paper_starting_balance"]),
            tokens_tested=len(basket),
            from_ts=from_ts,
            to_ts=now,
            params={
                k: cfg[k]
                for k in (
                    "candle_minutes", "volume_spike_multiple", "volume_spike_lookback",
                    "momentum_candles", "momentum_min_pct", "rr_min", "rr_max",
                    "trailing_activate_r", "trailing_distance_atr", "max_slippage_pct",
                    "taker_fee_pct", "max_position_pct_of_wallet",
                    "max_total_deployed_pct", "signal_invalidation_bars",
                )
            },
        )

        # Load every token's frame once, then walk the whole window in
        # chronological order so the concurrent-position cap is respected across
        # tokens rather than per token - simulating each token in isolation
        # would let the backtest hold far more positions than the live bot can.
        frames: dict[str, pd.DataFrame] = {}
        for i, mint in enumerate(basket, start=1):
            if progress:
                progress(i, len(basket), mint)
            df = self.store.candles_for(mint, since=from_ts, until=now, conn=conn)
            if len(df) < int(cfg["min_candles_required"]) + 5:
                continue
            frames[mint] = compute(df, cfg)
        result.tokens_with_data = len(frames)
        if not frames:
            return result

        return self._simulate(frames, result, cfg, conn)

    # ------------------------------------------------------------------
    def _simulate(
        self,
        frames: dict[str, pd.DataFrame],
        result: BacktestResult,
        cfg: dict[str, Any],
        conn: sqlite3.Connection,
    ) -> BacktestResult:
        # A single merged timeline of every bar across every token.
        timeline: list[tuple[int, str, int]] = []
        for mint, df in frames.items():
            for idx, ts in enumerate(df["ts"].tolist()):
                timeline.append((int(ts), mint, idx))
        timeline.sort()

        liquidity = self._liquidity_map(frames.keys(), conn)
        balance = result.starting_balance
        open_positions: dict[str, dict[str, Any]] = {}
        cost_pct = float(cfg["taker_fee_pct"]) / 100.0
        slip_pct = float(cfg["max_slippage_pct"]) / 2.0 / 100.0  # expected, not worst case
        min_bars = int(cfg["min_candles_required"])

        # Each timestamp is processed in two passes: every open position is
        # managed first, then entries are considered. That is exactly what the
        # live engine does on every cycle (manage_positions, then
        # look_for_entries), and it matters at ties: walking a single merged
        # timeline sorted by (ts, mint) would let a token whose name sorts early
        # take an entry while a position exiting on the same bar was still
        # counted as deployed, sizing that entry against capital the live bot
        # would already have released.
        for ts, group in groupby(timeline, key=lambda row: row[0]):
            events = list(group)

            # --- pass one: manage open positions -------------------------
            for _ts, mint, idx in events:
                position = open_positions.get(mint)
                if position is None or idx < min_bars:
                    continue
                df = frames[mint]
                row = df.iloc[idx]
                price = float(row["close"])
                if price <= 0:
                    continue
                result.bars_evaluated += 1
                window = df.iloc[: idx + 1]

                # Intrabar stop: if the bar's low pierced the stop, that is
                # where the exit happened, not at the close.
                stop_level = max(
                    float(position["hard_stop"]), float(position.get("trailing_stop") or 0.0)
                )
                if float(row["low"]) <= stop_level:
                    balance = self._close(
                        result, position, stop_level, ts, "stop hit intrabar",
                        balance, cost_pct, slip_pct,
                    )
                    open_positions.pop(mint, None)
                    result.equity_curve.append((ts, balance))
                    continue

                exit_signal = evaluate_exit(
                    position, price, window, cfg, now_ts=ts, precomputed=True
                )
                if exit_signal.should_exit:
                    balance = self._close(
                        result, position, price, ts, exit_signal.reason,
                        balance, cost_pct, slip_pct,
                    )
                    open_positions.pop(mint, None)
                    result.equity_curve.append((ts, balance))
                    continue

                if "invalidation_count" in exit_signal.detail:
                    position["invalidation_count"] = exit_signal.detail["invalidation_count"]
                atr_value = float(row.get("atr") or 0.0) or price * 0.01
                changes = update_trailing_stop(position, price, atr_value, cfg)
                changes.pop("_armed_now", None)
                position.update(changes)

            # --- pass two: look for entries ------------------------------
            for _ts, mint, idx in events:
                if mint in open_positions or idx < min_bars:
                    continue
                deployed = sum(p["size_usd"] for p in open_positions.values())
                wallet = balance + deployed
                if deployed >= wallet * float(cfg["max_total_deployed_pct"]):
                    break
                df = frames[mint]
                row = df.iloc[idx]
                price = float(row["close"])
                if price <= 0:
                    continue
                result.bars_evaluated += 1

                token_liquidity = liquidity.get(mint, 0.0)

                sizing = size_position(
                    wallet_usd=wallet,
                    liquidity_usd=token_liquidity,
                    price=price,
                    atr_pct=float(row.get("atr_pct") or 0.0),
                    cfg=cfg,
                    deployed_usd=deployed,
                )
                if not sizing.ok or sizing.size_usd > balance:
                    continue

                signal = evaluate_entry(
                    df.iloc[: idx + 1], cfg, mint=mint,
                    liquidity_usd=token_liquidity,
                    intended_size_usd=sizing.size_usd,
                    precomputed=True,
                )
                if not signal.ok:
                    continue

                entry_price = price * (1.0 + slip_pct)
                fee = sizing.size_usd * cost_pct
                qty = (sizing.size_usd - fee) / entry_price
                risk_per_unit = signal.price - signal.stop
                balance -= sizing.size_usd

                open_positions[mint] = {
                    "mint": mint,
                    "symbol": mint[:6],
                    "entry_price": entry_price,
                    "entry_ts": ts,
                    "qty": qty,
                    "size_usd": sizing.size_usd,
                    "hard_stop": entry_price - risk_per_unit,
                    "take_profit": entry_price + risk_per_unit * signal.rr,
                    "trailing_stop": None,
                    "trailing_armed": 0,
                    "high_water_price": entry_price,
                    "initial_risk": risk_per_unit,
                    "rr_target": signal.rr,
                    "entry_reason": signal.summary(),
                    "entry_fee_usd": fee,
                    "invalidation_count": 0,
                }
                result.equity_curve.append((ts, balance + sizing.size_usd))

        # Close anything still open at the final price it traded at.
        for mint, position in list(open_positions.items()):
            df = frames[mint]
            last_price = float(df["close"].iloc[-1])
            last_ts = int(df["ts"].iloc[-1])
            balance = self._close(
                result, position, last_price, last_ts, "backtest window ended",
                balance, cost_pct, slip_pct,
            )

        result.ending_balance = balance
        result.equity_curve.append((result.to_ts, balance))
        return result

    @staticmethod
    def _close(
        result: BacktestResult,
        position: dict[str, Any],
        price: float,
        ts: int,
        reason: str,
        balance: float,
        cost_pct: float,
        slip_pct: float,
    ) -> float:
        exit_price = price * (1.0 - slip_pct)
        gross = position["qty"] * exit_price
        fee = gross * cost_pct
        proceeds = gross - fee
        size = position["size_usd"]
        pnl = proceeds - size

        result.trades.append(
            BacktestTrade(
                mint=position["mint"],
                symbol=position.get("symbol", ""),
                entry_ts=int(position["entry_ts"]),
                entry_price=float(position["entry_price"]),
                exit_ts=int(ts),
                exit_price=exit_price,
                size_usd=size,
                qty=position["qty"],
                pnl_usd=pnl,
                pnl_pct=pnl / size if size else 0.0,
                fees_usd=float(position.get("entry_fee_usd", 0.0)) + fee,
                entry_reason=position.get("entry_reason", ""),
                exit_reason=reason,
            )
        )
        return balance + proceeds

    @staticmethod
    def _liquidity_map(
        mints: Any, conn: sqlite3.Connection
    ) -> dict[str, float]:
        """Best-known liquidity per token, preferring the historical record."""
        out: dict[str, float] = {}
        rows = conn.execute(
            "SELECT mint, AVG(liquidity_usd) AS liq FROM universe_history GROUP BY mint"
        ).fetchall()
        for r in rows:
            out[r["mint"]] = float(r["liq"] or 0.0)
        for r in conn.execute("SELECT mint, liquidity_usd FROM universe").fetchall():
            out.setdefault(r["mint"], float(r["liquidity_usd"] or 0.0))
        return out


# --------------------------------------------------------------------------
def run_and_store(
    store: DataStore,
    cfg: dict[str, Any],
    *,
    days: int | None = None,
    conn: sqlite3.Connection | None = None,
) -> BacktestResult:
    """Run a backtest, persist the result, and report progress to the dashboard."""
    conn = conn or db.connect()
    days = days or int(cfg["backtest_lookback_days"])
    db.set_progress(JOB_BACKTEST, status="running", message="Loading candles", conn=conn)

    def report(done: int, total: int, mint: str) -> None:
        db.set_progress(
            JOB_BACKTEST,
            status="running",
            done=done,
            total=total,
            message=f"Running backtest — {done} of {total} tokens loaded ({mint} now)",
            conn=conn,
        )

    tester = Backtester(store, cfg)
    try:
        result = tester.run(days=days, conn=conn, progress=report)
    except Exception as exc:
        db.set_progress(JOB_BACKTEST, status="failed", message=str(exc)[:300], conn=conn)
        db.log_event(f"Backtest failed: {exc}", level="alert", category="system", conn=conn)
        raise

    summary = result.summary()
    conn.execute(
        "INSERT INTO backtest_runs(ts, from_ts, to_ts, trades, win_rate, profit_factor, "
        "max_drawdown, total_return, avg_win, avg_loss, params, detail) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            db.now(), result.from_ts, result.to_ts, len(result.trades), result.win_rate,
            None if result.profit_factor == float("inf") else result.profit_factor,
            result.max_drawdown, result.total_return, result.avg_win, result.avg_loss,
            json.dumps(result.params), json.dumps(summary),
        ),
    )
    db.set_progress(
        JOB_BACKTEST,
        status="done",
        done=result.tokens_with_data,
        total=result.tokens_tested,
        message=f"{len(result.trades)} trades, {result.win_rate * 100:.1f}% win rate",
        conn=conn,
    )
    db.log_event(
        f"Backtest complete: {len(result.trades)} trades over {days}d, "
        f"{result.win_rate * 100:.1f}% win rate, profit factor "
        f"{result.profit_factor:.2f}, max drawdown {result.max_drawdown * 100:.1f}%.",
        category="system",
        detail=summary,
        conn=conn,
    )
    return result
