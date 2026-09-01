"""The trading loop.

A single polling loop, not a scheduler framework - this has to be light enough
for one vCPU, and the work per cycle is batched HTTP calls plus indicator math,
not model inference.

Each cycle, in order:

1. reload settings if config.json changed
2. drain dashboard commands (start/stop, kill switch, manual close, …)
3. refresh the universe if due
4. poll the hot tier (~1s) - open positions and tokens of interest
5. manage open positions against the three exit conditions
6. poll the broad tier (~15s) and look for entries
7. record equity, roll ticks up into candles, prune, run housekeeping

The engine runs one *primary* instance plus a permanent parallel paper instance
when live. Both evaluate the identical rules on the identical data; only the
executor differs, which is what makes the comparison meaningful.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from . import db, risk
from .clients import Clients, PriceInfo, build_clients
from .config import Config
from .datastore import DataStore
from .execution import Executor, Fill, PaperExecutor, build_executor
from .indicators import compute, snapshot_at
from .portfolio import Portfolio
from .recovery import recover, reconcile_halted
from .safety import SafetyGate
from .scanner import Scanner
from .strategy import evaluate_entry, evaluate_exit, update_trailing_stop
from .universe import UniverseBuilder

log = logging.getLogger(__name__)

HEARTBEAT_KEY = "worker_heartbeat"
STATE_KEY = "engine_state"
RUN_KEY = "engine_run"
SHADOW_KEY = "shadow_overrides"
LAST_BACKTEST_KEY = "last_backtest_day"


@dataclass
class InstanceRunner:
    """One trading instance: its portfolio, its executor, its settings overrides."""

    name: str
    portfolio: Portfolio
    executor: Executor
    overrides: dict[str, Any] = field(default_factory=dict)

    def cfg(self, base: dict[str, Any]) -> dict[str, Any]:
        return {**base, **self.overrides} if self.overrides else base


class Engine:
    def __init__(self, config: Config, clients: Clients | None = None) -> None:
        self.config = config
        self.cfg = config.as_dict()
        self.clients = clients or build_clients(config)
        self.clients.apply_config(config)

        self.store = DataStore(self.clients.birdeye, self.cfg)
        self.universe = UniverseBuilder(self.clients.jupiter, self.cfg)
        self.scanner = Scanner(self.clients.jupiter, self.cfg)
        self.safety = SafetyGate(self.clients.rugcheck, self.cfg)

        self.instances: list[InstanceRunner] = []
        self.primary: InstanceRunner | None = None

        self._stop = threading.Event()
        self._running = False
        self._cycle = 0
        self._last_equity_record = 0.0
        self._last_prune = 0.0
        self._last_budget_warning = 0.0
        self._candle_cache: dict[str, tuple[int, pd.DataFrame]] = {}
        self._backtest_hook = None  # set by run_worker so a daily run can fire

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------
    def build_instances(self) -> None:
        mode = self.cfg["trading_mode"]
        primary_name = "live" if mode == "live" else "paper"

        primary_exec = build_executor(self.clients, self.cfg, mode, self.config.secrets)
        primary = InstanceRunner(
            primary_name, Portfolio(primary_name, self.cfg), primary_exec
        )
        self.instances = [primary]
        self.primary = primary

        # Once live, a permanent parallel paper instance runs the same rules so
        # live performance can be compared against expected performance.
        if mode == "live":
            self.instances.append(
                InstanceRunner(
                    "paper",
                    Portfolio("paper", self.cfg),
                    PaperExecutor(self.clients, self.cfg),
                )
            )

        # Shadow mode: a candidate rule set running in paper alongside the rest,
        # never promoted automatically.
        overrides = db.kv_get(SHADOW_KEY, {}) or {}
        if overrides:
            self.instances.append(
                InstanceRunner(
                    "shadow",
                    Portfolio("shadow", self.cfg),
                    PaperExecutor(self.clients, self.cfg),
                    overrides=dict(overrides),
                )
            )
            db.log_event(
                f"Shadow instance active with {len(overrides)} parameter override(s): "
                + ", ".join(f"{k}={v}" for k, v in list(overrides.items())[:6]),
                category="system",
                instance="shadow",
            )

    def startup(self) -> None:
        db.init_db()
        self.build_instances()
        assert self.primary is not None

        report = recover(
            portfolio=self.primary.portfolio,
            clients=self.clients,
            store=self.store,
            cfg=self.cfg,
            executor=self.primary.executor,
        )
        if report.halted:
            db.log_event(
                "Engine will idle until the reconciliation halt is cleared from the dashboard.",
                level="alert",
                category="system",
            )

        self.universe.load_persisted()
        if not self.universe.tokens:
            try:
                self.universe.refresh()
            except Exception as exc:
                log.warning("initial universe refresh failed: %s", exc)

        self._report_scan_budget(force=True)

    # ------------------------------------------------------------------
    # main loop
    # ------------------------------------------------------------------
    def run(self) -> None:
        self._running = True
        self.startup()
        db.log_event("Trading loop started.", category="system")

        while not self._stop.is_set():
            cycle_started = time.monotonic()
            try:
                self.tick()
            except Exception as exc:  # never let one bad cycle kill the worker
                log.exception("cycle failed")
                db.log_event(
                    f"Unhandled error in the trading cycle: {exc}",
                    level="alert",
                    category="system",
                    detail={"error": repr(exc)},
                )
                time.sleep(2.0)

            # Pace to the hot cadence; that is the fastest tier we owe.
            elapsed = time.monotonic() - cycle_started
            sleep_for = max(0.05, float(self.cfg["hot_scan_seconds"]) - elapsed)
            self._stop.wait(sleep_for)

        self._running = False
        db.log_event("Trading loop stopped.", category="system")

    def stop(self) -> None:
        self._stop.set()

    # ------------------------------------------------------------------
    def tick(self) -> None:
        self._cycle += 1

        if self.config.maybe_reload():
            self._apply_config()

        self.drain_commands()
        self.heartbeat()

        if not self.trading_enabled():
            # Still manage open positions while halted - a kill switch stops new
            # entries, it does not abandon what is already open.
            self.manage_positions()
            return

        if self.universe.due():
            try:
                self.universe.refresh()
                self._report_scan_budget()
            except Exception as exc:
                log.warning("universe refresh failed: %s", exc)

        # --- hot tier -----------------------------------------------------
        self.scanner.pin(self._held_mints())
        if self.scanner.hot_due():
            result = self.scanner.scan_hot()
            if result.prices:
                self.scanner.record_ticks(result.prices)
            self.manage_positions(result.prices)
            self.look_for_entries(result.prices, tier="hot")

        # --- broad tier ---------------------------------------------------
        if self.scanner.broad_due():
            mints = self.universe.mints
            result = self.scanner.scan_broad(mints)
            if result.prices:
                self.scanner.record_ticks(result.prices)
                self.store.rollup_ticks()
                self.triage(result.prices)
                self.look_for_entries(result.prices, tier="broad")
            self.scanner.expire_hot()
            self.housekeeping()

    # ------------------------------------------------------------------
    # gating
    # ------------------------------------------------------------------
    def trading_enabled(self) -> bool:
        if not bool(db.kv_get(RUN_KEY, True)):
            return False
        if risk.kill_switch_engaged():
            return False
        if reconcile_halted():
            return False
        return True

    def _held_mints(self) -> list[str]:
        held: set[str] = set()
        for inst in self.instances:
            held.update(p["mint"] for p in inst.portfolio.open_positions())
        return list(held)

    # ------------------------------------------------------------------
    # triage - which tokens deserve the fast poll
    # ------------------------------------------------------------------
    def triage(self, prices: dict[str, PriceInfo]) -> None:
        """Escalate tokens showing early signal characteristics to the 1s tier."""
        cfg = self.cfg
        for mint in list(prices):
            token = self.universe.tokens.get(mint)
            if token is None:
                continue
            df = self._candles(mint)
            if df.empty or len(df) < cfg["min_candles_required"]:
                continue
            signal = evaluate_entry(
                df, cfg, mint=mint, liquidity_usd=token.liquidity, precomputed=True
            )
            if signal.interest:
                self.scanner.mark_interesting(mint)
            else:
                self.scanner.cool(mint)

    # ------------------------------------------------------------------
    # entries
    # ------------------------------------------------------------------
    def look_for_entries(self, prices: dict[str, PriceInfo], *, tier: str) -> None:
        if not prices:
            return
        for inst in self.instances:
            self._entries_for_instance(inst, prices, tier)

    def _entries_for_instance(
        self, inst: InstanceRunner, prices: dict[str, PriceInfo], tier: str
    ) -> None:
        cfg = inst.cfg(self.cfg)
        conn = db.connect()

        gate = risk.can_open_position(
            open_count=inst.portfolio.open_count(conn), cfg=cfg, conn=conn
        )
        if not gate.allowed:
            return

        balance = inst.portfolio.balance(conn)
        deployed = inst.portfolio.deployed_usd(conn)
        wallet_usd = balance + deployed

        for mint, price_info in prices.items():
            if inst.portfolio.open_count(conn) >= int(cfg["max_concurrent_positions"]):
                break
            if inst.portfolio.position_for(mint, conn) is not None:
                continue
            token = self.universe.tokens.get(mint)
            if token is None:
                continue

            # 4. Safety gate runs BEFORE the signal is evaluated.
            verdict = self.safety.check(mint, token=token, conn=conn)
            if not verdict.passed:
                self.scanner.cool(mint)
                continue

            df = self._candles(mint)
            if df.empty:
                continue

            sizing = risk.size_position(
                wallet_usd=wallet_usd,
                liquidity_usd=token.liquidity,
                price=price_info.price,
                atr_pct=self._atr_pct(df),
                cfg=cfg,
                deployed_usd=deployed,
            )
            if not sizing.ok:
                continue

            signal = evaluate_entry(
                df,
                cfg,
                mint=mint,
                liquidity_usd=token.liquidity,
                intended_size_usd=sizing.size_usd,
                precomputed=True,
            )
            if signal.interest:
                self.scanner.mark_interesting(mint)
            if not signal.ok:
                continue

            # Correlation: do not open a second position riding the same move.
            open_positions = inst.portfolio.open_positions(conn)
            if open_positions:
                held_closes = {
                    p["mint"]: self._candles(p["mint"])["close"].tolist()[
                        -int(cfg["correlation_lookback"]) :
                    ]
                    for p in open_positions
                }
                held_closes = {k: v for k, v in held_closes.items() if len(v) >= 10}
                if held_closes:
                    corr = risk.correlation_gate(
                        df["close"].tolist()[-int(cfg["correlation_lookback"]) :],
                        held_closes,
                        cfg,
                    )
                    if not corr.allowed:
                        db.log_event(
                            f"Skipped {token.symbol or mint[:8]}: {corr.reason}",
                            category="risk",
                            instance=inst.name,
                            mint=mint,
                            conn=conn,
                        )
                        continue

            self._open(inst, cfg, token, price_info, signal, sizing, tier, conn)

    def _open(
        self,
        inst: InstanceRunner,
        cfg: dict[str, Any],
        token: Any,
        price_info: PriceInfo,
        signal: Any,
        sizing: Any,
        tier: str,
        conn: sqlite3.Connection,
    ) -> None:
        fill: Fill = inst.executor.buy(
            token.mint,
            sizing.size_usd,
            price_info.price,
            decimals=token.decimals or price_info.decimals,
        )
        if not fill.ok:
            db.log_event(
                f"Entry skipped for {token.symbol or token.mint[:8]}: {fill.reason}",
                level="warn",
                category="trade",
                instance=inst.name,
                mint=token.mint,
                conn=conn,
            )
            return

        # Re-anchor the stop and target on the price actually filled, not the
        # quoted one - otherwise slippage silently shrinks the real R multiple.
        risk_per_unit = signal.price - signal.stop
        stop = fill.price - risk_per_unit
        target = fill.price + risk_per_unit * signal.rr

        reason = f"[{tier}] " + signal.summary() + f"; sized {', '.join(sizing.reasons)}"
        inst.portfolio.open_position(
            mint=token.mint,
            symbol=token.symbol,
            fill=fill,
            stop=stop,
            target=target,
            rr=signal.rr,
            entry_reason=reason,
            snapshot=signal.snapshot.to_dict() if signal.snapshot else {},
            conn=conn,
        )
        self.scanner.mark_interesting(token.mint)

    # ------------------------------------------------------------------
    # exits
    # ------------------------------------------------------------------
    def manage_positions(self, prices: dict[str, PriceInfo] | None = None) -> None:
        for inst in self.instances:
            self._manage_instance(inst, prices or self.scanner.last_prices)

    def _manage_instance(
        self, inst: InstanceRunner, prices: dict[str, PriceInfo]
    ) -> None:
        cfg = inst.cfg(self.cfg)
        conn = db.connect()
        now = db.now()

        for pos in inst.portfolio.open_positions(conn):
            price_info = prices.get(pos["mint"]) or self.scanner.last_prices.get(pos["mint"])
            if price_info is None:
                continue
            price = price_info.price
            df = self._candles(pos["mint"])

            position = dict(pos)
            exit_signal = evaluate_exit(
                position, price, df if not df.empty else None, cfg,
                now_ts=now, precomputed=True,
            )

            if exit_signal.should_exit:
                self._close(inst, pos, price, exit_signal.reason, conn)
                continue

            changes: dict[str, Any] = {}
            # Persist the invalidation counter so a restart does not reset the
            # 'signal has been failing for N bars' state back to zero.
            if "invalidation_count" in exit_signal.detail:
                count = int(exit_signal.detail["invalidation_count"])
                if count != int(pos["invalidation_count"] or 0):
                    changes["invalidation_count"] = count

            atr_value = self._atr(df) or price * 0.01
            trail = update_trailing_stop(position, price, atr_value, cfg)
            armed_now = trail.pop("_armed_now", False)
            changes.update(trail)

            if changes:
                inst.portfolio.update_position(int(pos["id"]), changes, conn)
            if armed_now:
                db.log_event(
                    f"Trailing stop armed on {pos['symbol'] or pos['mint'][:8]} at "
                    f"${changes.get('trailing_stop', 0):.6g} - round-trip costs are now covered.",
                    category="trade",
                    instance=inst.name,
                    mint=pos["mint"],
                    conn=conn,
                )

    def _close(
        self,
        inst: InstanceRunner,
        pos: sqlite3.Row,
        price: float,
        reason: str,
        conn: sqlite3.Connection,
        *,
        manual: bool = False,
    ) -> None:
        fill = inst.executor.sell(pos["mint"], float(pos["qty"]), price)
        if not fill.ok:
            db.log_event(
                f"Exit attempt failed for {pos['symbol'] or pos['mint'][:8]}: {fill.reason}. "
                "The position remains open and will be retried.",
                level="alert",
                category="trade",
                instance=inst.name,
                mint=pos["mint"],
                conn=conn,
            )
            return

        closed = inst.portfolio.close_position(pos, fill, reason, manual=manual, conn=conn)
        self.scanner.cool(pos["mint"])

        # Only the primary instance drives the circuit breaker; a paper
        # comparison run must not be able to halt live trading.
        if inst is self.primary:
            risk.record_trade_result(closed.pnl_usd, inst.cfg(self.cfg), conn)

    # ------------------------------------------------------------------
    # dashboard commands
    # ------------------------------------------------------------------
    def drain_commands(self) -> None:
        conn = db.connect()
        for row in db.pending_commands(conn):
            command = row["command"]
            try:
                payload = json.loads(row["payload"] or "{}")
            except json.JSONDecodeError:
                payload = {}
            who = row["requested_by"] or "dashboard"
            try:
                result = self._handle_command(command, payload, who, conn)
                db.finish_command(int(row["id"]), "done", result, conn)
            except Exception as exc:
                log.exception("command %s failed", command)
                db.finish_command(int(row["id"]), "failed", str(exc)[:400], conn)
                db.log_event(
                    f"Command '{command}' failed: {exc}",
                    level="alert",
                    category="system",
                    conn=conn,
                )

    def _handle_command(
        self, command: str, payload: dict[str, Any], who: str, conn: sqlite3.Connection
    ) -> str:
        if command == "start":
            db.kv_set(RUN_KEY, True, conn)
            db.log_event(f"Bot started by {who}.", category="system", conn=conn)
            return "started"

        if command == "stop":
            db.kv_set(RUN_KEY, False, conn)
            db.log_event(f"Bot stopped by {who}.", level="warn", category="system", conn=conn)
            return "stopped"

        if command == "kill_switch":
            risk.engage_kill_switch(payload.get("reason", "manual"), who, conn)
            return "kill switch engaged"

        if command == "release_kill_switch":
            risk.release_kill_switch(who, conn)
            return "kill switch released"

        if command == "reset_circuit":
            risk.reset_circuit(who, conn)
            return "circuit breaker reset"

        if command == "clear_reconcile_halt":
            from .recovery import clear_reconcile_halt

            clear_reconcile_halt(who, conn)
            return "reconciliation halt cleared"

        if command == "close_position":
            return self._manual_close(int(payload["position_id"]), who, conn)

        if command == "refresh_universe":
            stats = self.universe.refresh(conn=conn)
            return f"universe refreshed: {stats.passed} tokens"

        if command == "set_shadow":
            overrides = payload.get("overrides") or {}
            db.kv_set(SHADOW_KEY, overrides, conn)
            self.build_instances()
            return f"shadow overrides set ({len(overrides)} keys)"

        if command == "historical_pull":
            self._start_pull(payload, conn)
            return "historical pull started"

        if command == "run_backtest":
            self._start_backtest(payload, conn)
            return "backtest started"

        return f"unknown command: {command}"

    def _manual_close(self, position_id: int, who: str, conn: sqlite3.Connection) -> str:
        for inst in self.instances:
            pos = conn.execute(
                "SELECT * FROM positions WHERE id = ? AND instance = ? AND status = 'open'",
                (position_id, inst.name),
            ).fetchone()
            if pos is None:
                continue
            price_info = self.scanner.last_prices.get(pos["mint"])
            price = price_info.price if price_info else None
            if price is None:
                fetched = self.clients.jupiter.prices([pos["mint"]], priority="high")
                info = fetched.get(pos["mint"])
                price = info.price if info else float(pos["entry_price"])
            self._close(
                inst, pos, price, f"manual close by {who}", conn, manual=True
            )
            return f"closed position {position_id}"
        return f"position {position_id} is not open"

    def _start_pull(self, payload: dict[str, Any], conn: sqlite3.Connection) -> None:
        mints = payload.get("mints") or self.universe.mints
        days = int(payload.get("days") or self.cfg["backtest_lookback_days"])

        def worker() -> None:
            try:
                self.store.run_initial_pull(mints, days=days)
            except Exception:
                log.exception("historical pull failed")

        threading.Thread(target=worker, name="historical-pull", daemon=True).start()

    def _start_backtest(self, payload: dict[str, Any], conn: sqlite3.Connection) -> None:
        if self._backtest_hook is None:
            raise RuntimeError("backtest is not wired up in this process")
        days = int(payload.get("days") or self.cfg["backtest_lookback_days"])
        threading.Thread(
            target=self._backtest_hook, args=(days,), name="backtest", daemon=True
        ).start()

    # ------------------------------------------------------------------
    # housekeeping
    # ------------------------------------------------------------------
    def housekeeping(self) -> None:
        now = time.monotonic()
        prices = self.scanner.last_prices

        if now - self._last_equity_record > 60:
            for inst in self.instances:
                eq = inst.portfolio.record_equity(prices)
                if inst is self.primary:
                    risk.check_daily_drawdown(inst.name, eq, inst.cfg(self.cfg))
            self._last_equity_record = now

        if now - self._last_prune > 3600:
            deleted = self.store.prune()
            self._last_prune = now
            if any(deleted.values()):
                log.info("pruned %s", deleted)

        self._maybe_daily_backtest()
        self._check_safety_alerts()

    def _maybe_daily_backtest(self) -> None:
        """Fire the scheduled backtest once per day, at the configured hour."""
        if not self.cfg["backtest_daily_enabled"] or self._backtest_hook is None:
            return
        now = time.gmtime()
        if now.tm_hour != int(self.cfg["backtest_daily_hour_utc"]):
            return
        today = time.strftime("%Y-%m-%d", now)
        if db.kv_get(LAST_BACKTEST_KEY) == today:
            return
        db.kv_set(LAST_BACKTEST_KEY, today)
        db.log_event("Scheduled daily backtest starting.", category="system")
        threading.Thread(
            target=self._backtest_hook,
            args=(int(self.cfg["backtest_lookback_days"]),),
            name="daily-backtest",
            daemon=True,
        ).start()

    def _check_safety_alerts(self) -> None:
        rejections = self.safety.rejection_streak(3600)
        if rejections >= 25 and self._cycle % 200 == 0:
            db.log_event(
                f"{rejections} Rug Check rejections in the last hour - the universe filters "
                "may be admitting too many low-quality tokens.",
                level="warn",
                category="safety",
            )

    def _report_scan_budget(self, *, force: bool = False) -> None:
        """Surface it when the configured cadence outruns the API allowance."""
        budget = self.scanner.budget(
            len(self.universe.tokens), available_rps=self.clients.jupiter_bucket.effective_rate
        )
        db.kv_set(
            "scan_budget",
            {
                "universe": budget.universe_size,
                "hot": budget.hot_size,
                "required_rps": round(budget.required_rps, 3),
                "available_rps": round(budget.available_rps, 3),
                "sustainable": budget.sustainable,
                "message": budget.message(),
            },
        )
        now = time.monotonic()
        if not budget.sustainable and (force or now - self._last_budget_warning > 3600):
            self._last_budget_warning = now
            db.log_event(budget.message(), level="warn", category="system")

    def heartbeat(self) -> None:
        db.kv_set(
            HEARTBEAT_KEY,
            {
                "ts": db.now(),
                "cycle": self._cycle,
                "pid": __import__("os").getpid(),
                "mode": self.cfg["trading_mode"],
                "instances": [i.name for i in self.instances],
            },
        )
        if self._cycle % 30 == 0:
            db.kv_set(
                STATE_KEY,
                {
                    "running": self._running,
                    "trading_enabled": self.trading_enabled(),
                    "universe": len(self.universe.tokens),
                    "scanner": self.scanner.stats(),
                    "limits": self.clients.stats(),
                    "coverage": self.store.coverage(),
                },
            )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _apply_config(self) -> None:
        self.cfg = self.config.as_dict()
        self.clients.apply_config(self.config)
        for component in (self.store, self.universe, self.scanner, self.safety):
            component.update_config(self.cfg)
        for inst in self.instances:
            inst.portfolio.update_config(self.cfg)
            inst.executor.update_config(self.cfg)
        self._candle_cache.clear()
        db.log_event("Settings reloaded from config.json.", category="system")
        self._report_scan_budget(force=True)

    def _candles(self, mint: str) -> pd.DataFrame:
        """Indicator-computed candles for one token, cached for the current bar.

        Recomputing indicators for every token on every 1s poll would be the one
        thing in this loop heavy enough to matter on a single vCPU. The cache key
        is the current bar, so it invalidates exactly when new data can exist.
        """
        bar = db.now() // self.store.target_seconds
        cached = self._candle_cache.get(mint)
        if cached and cached[0] == bar:
            return cached[1]

        df = self.store.candles_for(
            mint, limit=int(self.cfg["min_candles_required"]) * 2, conn=db.connect()
        )
        df = self.store.drop_forming_bar(df)
        if not df.empty:
            df = compute(df, self.cfg)
        self._candle_cache[mint] = (bar, df)

        if len(self._candle_cache) > 600:  # bound memory on a 2GB box
            for key in list(self._candle_cache)[:200]:
                self._candle_cache.pop(key, None)
        return df

    @staticmethod
    def _atr(df: pd.DataFrame) -> float:
        if df.empty or "atr" not in df.columns:
            return 0.0
        snap = snapshot_at(df, -1)
        return snap.atr if snap else 0.0

    @staticmethod
    def _atr_pct(df: pd.DataFrame) -> float:
        if df.empty or "atr_pct" not in df.columns:
            return 0.0
        snap = snapshot_at(df, -1)
        return snap.atr_pct if snap else 0.0
