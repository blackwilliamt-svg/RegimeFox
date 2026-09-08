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

from . import db, drift, paramsync, review, risk, wfmc
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


class _EmptyLibraryStore:
    """Regime-scoped selection's fallback when the local WFMC store can't be
    opened - always reports no entries, never raises."""

    def nearest_regime_entries(self, *_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        return []

    def regime_cluster_entries(self, *_args: Any, **_kwargs: Any) -> dict[int, dict[str, Any]]:
        return {}

    def get_regime_model(self, *_args: Any, **_kwargs: Any) -> dict[str, Any] | None:
        return None


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

        self.store = DataStore(self.clients.binance, self.cfg)
        self.universe = UniverseBuilder(self.clients.binance, self.clients.jupiter, self.cfg)
        self.scanner = Scanner(self.clients.jupiter, self.cfg)
        self.safety = SafetyGate(self.clients.rugcheck, self.cfg)

        self.reviewer = review.EntryGate(self.cfg)

        self.instances: list[InstanceRunner] = []
        self.primary: InstanceRunner | None = None

        self._stop = threading.Event()
        self._last_drift_check = 0.0
        self._last_promotion_check = 0.0
        self._running = False
        self._cycle = 0
        self._last_equity_record = 0.0
        self._last_prune = 0.0
        self._last_budget_warning = 0.0
        self._candle_cache: dict[str, tuple[int, pd.DataFrame]] = {}
        # (instance, mint) -> the regime-scoped fingerprint currently in
        # effect for new entries, or None - tracked only so a change of
        # choice logs once rather than every cycle it stays the same.
        self._regime_selection: dict[tuple[str, str], str | None] = {}
        self._library_store: Any = None  # lazily opened once, not per entry check
        # (instance, mint) -> the blend detail _regime_scoped_cfg computed
        # for the entry decision currently in flight, if fuzzy blending
        # (step 4) applied - _open() reads and clears this when it records
        # the actual position, so it can only ever attach to the trade it
        # was computed for.
        self._active_blend_detail: dict[tuple[str, str], dict[str, Any]] = {}
        self._backtest_hook = None  # set by run_worker so a daily run can fire
        # Off by default so constructing an Engine for a test or a script never
        # spawns a real WFMC run purely because the wall clock happens to match
        # the configured hour; run_worker.py turns this on for the real process.
        self._wfmc_hook_enabled = False

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

        gate = risk.can_open_position(cfg=cfg, conn=conn)
        if not gate.allowed:
            return

        throttle = self._throttle(inst, cfg, conn)
        if throttle:
            return

        # No fixed position-count cap: how many positions may be open at once
        # is decided by the total-deployed budget below, re-read every
        # iteration so capital committed earlier in this same scan counts
        # against it immediately.
        max_deployed_pct = float(cfg["max_total_deployed_pct"])
        for mint, price_info in prices.items():
            balance = inst.portfolio.balance(conn)
            deployed = inst.portfolio.deployed_usd(conn)
            wallet_usd = balance + deployed
            if deployed >= wallet_usd * max_deployed_pct:
                break
            if inst.portfolio.position_for(mint, conn) is not None:
                continue
            token = self.universe.tokens.get(mint)
            if token is None:
                continue
            if self._cooling_off(inst, mint, cfg, conn):
                continue

            # 4. Safety gate runs BEFORE the signal is evaluated.
            verdict = self.safety.check(mint, token=token, conn=conn)
            if not verdict.passed:
                self.scanner.cool(mint)
                continue

            df = self._candles(mint)
            if df.empty:
                continue

            # Correlation first: it is cheap, it is a hard refusal, and it also
            # feeds the sizing below, so computing it once up front saves doing
            # the work twice.
            open_positions = inst.portfolio.open_positions(conn)
            open_book, corr = self._exposure(df, open_positions, cfg)
            if corr is not None and not corr.allowed:
                db.log_event(
                    f"Skipped {token.symbol or mint[:8]}: {corr.reason}",
                    category="risk",
                    instance=inst.name,
                    mint=mint,
                    conn=conn,
                )
                continue

            sizing = risk.size_position(
                wallet_usd=wallet_usd,
                liquidity_usd=token.liquidity,
                price=price_info.price,
                atr_pct=self._atr_pct(df),
                cfg=cfg,
                deployed_usd=deployed,
                open_book=open_book,
            )
            if not sizing.ok:
                continue

            # Regime-scoped promoted sets (gap-closure item 5): a symbol with
            # its own validated per-regime library entries trades *those* for
            # this one new-entry decision when the current regime is a close
            # enough match - never for sizing/risk/correlation above, and
            # never retroactively for a position already open (those keep
            # whatever stop/target their own entry fixed).
            entry_cfg = self._regime_scoped_cfg(inst, mint, token.symbol or mint, cfg, df, conn)

            signal = evaluate_entry(
                df,
                entry_cfg,
                mint=mint,
                liquidity_usd=token.liquidity,
                intended_size_usd=sizing.size_usd,
                precomputed=True,
            )
            if signal.interest:
                self.scanner.mark_interesting(mint)
            if not signal.ok:
                continue

            # 5. The entry review gate has the last word on borderline entries.
            decision = self._review(inst, cfg, token, signal, sizing, conn)
            if decision is not None and not decision.approve:
                db.log_event(
                    f"Review declined {token.symbol or mint[:8]}: {decision.rationale}",
                    category="signal",
                    instance=inst.name,
                    mint=mint,
                    detail=decision.as_dict(),
                    conn=conn,
                )
                self.scanner.cool(mint)
                continue

            self._open(
                inst, cfg, token, price_info, signal, sizing, tier, conn,
                decision=decision,
            )

    # ------------------------------------------------------------------
    # Overtrading brakes (spec 5)
    # ------------------------------------------------------------------
    def _throttle(
        self, inst: InstanceRunner, cfg: dict[str, Any], conn: sqlite3.Connection
    ) -> str:
        """Refuse to look for entries at all when the day's budget is spent.

        Overtrading is the failure mode the spec names: 244 trades in seven days
        cost $449 in fees. The regime and confluence gates raise the bar for an
        individual entry; this caps the total regardless of how good each one
        looked in isolation.
        """
        cap = int(cfg.get("max_trades_per_day", 0))
        if cap <= 0:
            return ""
        day_start = (db.now() // 86400) * 86400
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM positions WHERE instance = ? AND entry_ts >= ?",
            (inst.name, day_start),
        ).fetchone()
        taken = int(row["n"] or 0)
        if taken >= cap:
            if self._cycle % 600 == 0:
                db.log_event(
                    f"{inst.name} has opened {taken} positions today, at the "
                    f"{cap}-trade cap. No further entries until the UTC day rolls.",
                    category="risk",
                    instance=inst.name,
                    conn=conn,
                )
            return f"daily cap of {cap} trades reached"

        gap = int(cfg.get("min_seconds_between_entries", 0))
        if gap > 0:
            row = conn.execute(
                "SELECT MAX(entry_ts) AS t FROM positions WHERE instance = ?",
                (inst.name,),
            ).fetchone()
            last = int(row["t"] or 0)
            if last and db.now() - last < gap:
                return "still inside the minimum gap between entries"
        return ""

    def _cooling_off(
        self, inst: InstanceRunner, mint: str, cfg: dict[str, Any], conn: sqlite3.Connection
    ) -> bool:
        """Has this token been traded too recently to try again?

        Re-entering the same token minutes after exiting it is the shape
        overtrading usually takes: the signal that fired once tends to keep
        firing while the bar prints.
        """
        gap = int(cfg.get("min_seconds_between_entries_same_mint", 0))
        if gap <= 0:
            return False
        row = conn.execute(
            "SELECT MAX(entry_ts) AS t FROM positions WHERE instance = ? AND mint = ?",
            (inst.name, mint),
        ).fetchone()
        last = int(row["t"] or 0)
        return bool(last and db.now() - last < gap)

    # ------------------------------------------------------------------
    # Correlation-aware exposure (spec 4.4)
    # ------------------------------------------------------------------
    def _exposure(
        self,
        df: pd.DataFrame,
        open_positions: list[sqlite3.Row],
        cfg: dict[str, Any],
    ) -> tuple[list[risk.OpenExposure], risk.GateResult | None]:
        """The open book as the sizing maths sees it, plus the hard gate result.

        The correlation is computed once and used twice: to refuse a position
        that is merely riding an open one, and to shrink one that is partly
        riding it. The gate is the cliff; the sizing is the slope.
        """
        if not open_positions:
            return [], None

        lookback = int(cfg["correlation_lookback"])
        candidate = df["close"].tolist()[-lookback:]
        held_closes: dict[str, list[float]] = {}
        book: list[risk.OpenExposure] = []

        for position in open_positions:
            held = self._candles(position["mint"])
            closes = held["close"].tolist()[-lookback:] if not held.empty else []
            correlation = risk.correlation(candidate, closes) if len(closes) >= 10 else None
            if closes and len(closes) >= 10:
                held_closes[position["mint"]] = closes
            book.append(
                risk.OpenExposure(
                    size_usd=float(position["size_usd"]),
                    atr_pct=self._atr_pct(held),
                    # An uncomputable correlation is treated as fully correlated:
                    # sizing down on a token we cannot measure is the safe error.
                    correlation=1.0 if correlation is None else float(correlation),
                )
            )

        gate = (
            risk.correlation_gate(candidate, held_closes, cfg) if held_closes else None
        )
        return book, gate

    def _get_library_store(self) -> Any:
        """The local WFMC library store, opened once and reused - a fresh
        sqlite3 connection per entry candidate would be needless overhead in
        a loop that already runs on one shared vCPU. If it can't be opened
        (not yet on disk on a fresh install, or genuinely unopenable), an
        always-empty stand-in is cached instead, so a failure is diagnosed
        once rather than retried every cycle."""
        if self._library_store is None:
            try:
                from .wfmc import DAILY_STORE_PATH
                from solopt.store import RunStore

                self._library_store = RunStore(DAILY_STORE_PATH)
            except Exception:
                self._library_store = _EmptyLibraryStore()
        return self._library_store

    def _regime_scoped_cfg(
        self,
        inst: "InstanceRunner",
        mint: str,
        symbol: str,
        cfg: dict[str, Any],
        df: pd.DataFrame,
        conn: sqlite3.Connection,
    ) -> dict[str, Any]:
        """This one entry decision's effective config.

        Two mechanisms, tried in order, either falling back to `cfg`
        unchanged:

        1. Blended (fuzzy-regime section, step 4, off by default): every
           regime this coin has its own promoted set for contributes,
           weighted by current fuzzy membership - a smooth transition
           between regimes rather than a hard switch. Needs regime
           discovery and per-regime walk-forward to have actually run for
           this coin; when they have not, falls through to (2).
        2. Nearest-match (gap-closure item 5): the single closest
           continuous-regime library entry, hard-switched.

        Either way, logs a plain-language event only when the *choice*
        changes for this mint (not every cycle it stays the same), and
        records blend weights (if any) for _open() to attach to the entry
        snapshot - the "log the blend weights alongside each trade decision"
        explainability requirement.
        """
        snap = snapshot_at(df, -1)
        key = (inst.name, mint)

        if cfg.get("fuzzy_regime_blend_enabled", False) and snap is not None:
            from .regime_classify import classify_current_regime

            store = self._get_library_store()
            membership = classify_current_regime(symbol, snap, store=store)
            if membership is not None:
                entries = store.regime_cluster_entries(symbol)
                blend = paramsync.blend_regime_params(cfg, membership, entries)
                if blend.applied:
                    self._active_blend_detail[key] = blend.as_dict()
                    self._note_regime_choice(
                        inst, mint, symbol, f"blend:{sorted(blend.weights)}", blend.reason, conn,
                        switched_msg=f"{symbol}: now blending {len(blend.weights)} regime(s) "
                                     f"for new entries ({blend.reason}).",
                    )
                    return blend.params
        self._active_blend_detail.pop(key, None)

        current_regime = snap.efficiency if snap is not None else None
        selection = paramsync.select_regime_scoped_params(
            cfg, symbol, current_regime, store=self._get_library_store()
        )
        self._note_regime_choice(
            inst, mint, symbol, selection.fingerprint if selection.applied else None,
            selection.reason, conn,
            switched_msg=f"{symbol}: switched to a regime-scoped parameter set "
                         f"({selection.reason}) for new entries.",
            reverted_msg=f"{symbol}: reverted to the global parameter set for new entries "
                         f"({selection.reason}).",
            detail={"fingerprint": selection.fingerprint, "distance": selection.distance},
        )
        return selection.params

    def _note_regime_choice(
        self,
        inst: "InstanceRunner",
        mint: str,
        symbol: str,
        chosen: Any,
        reason: str,
        conn: sqlite3.Connection,
        *,
        switched_msg: str,
        reverted_msg: str | None = None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        """Log only on an actual change of choice for this mint - shared by
        both the blended and nearest-match paths in _regime_scoped_cfg so
        neither one spams the feed every cycle it stays the same."""
        key = (inst.name, mint)
        previous = self._regime_selection.get(key)
        if chosen == previous:
            return
        self._regime_selection[key] = chosen
        if chosen is not None:
            db.log_event(
                switched_msg, category="system", instance=inst.name, mint=mint,
                detail=detail, conn=conn,
            )
        elif previous is not None and reverted_msg is not None:
            db.log_event(reverted_msg, category="system", instance=inst.name, mint=mint, conn=conn)

    # ------------------------------------------------------------------
    # The entry review gate
    # ------------------------------------------------------------------
    def _review(
        self,
        inst: InstanceRunner,
        cfg: dict[str, Any],
        token: Any,
        signal: Any,
        sizing: Any,
        conn: sqlite3.Connection,
    ) -> Any:
        """Put an entry to the review gate. Returns None when it is switched off."""
        if self.reviewer is None or not cfg.get("entry_gate_enabled", True):
            return None
        day_start = (db.now() // 86400) * 86400
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM positions WHERE instance = ? AND entry_ts >= ?",
            (inst.name, day_start),
        ).fetchone()

        request = review.EntryRequest(
            mint=token.mint,
            symbol=token.symbol,
            instance=inst.name,
            price=signal.price,
            size_usd=sizing.size_usd,
            liquidity_usd=token.liquidity,
            rr=signal.rr,
            strength=signal.strength,
            entry_reason=signal.summary(),
            snapshot=signal.snapshot.to_dict() if signal.snapshot else {},
            market=review.build_market_context(conn=conn),
            trades_today=int(row["n"] or 0),
        )
        return self.reviewer.review_entry(request, conn=conn)

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
        *,
        decision: Any = None,
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
        if decision is not None:
            reason += f"; review: {decision.rationale}"

        snapshot_dict = signal.snapshot.to_dict() if signal.snapshot else {}
        # Fuzzy-regime section, step 4's explainability requirement: the
        # blend weights that produced this entry's effective parameters, if
        # any applied - popped, not just read, so a decision that does not
        # convert into a trade never leaves a stale blend attached to the
        # next one for this mint.
        blend = self._active_blend_detail.pop((inst.name, token.mint), None)
        if blend is not None:
            snapshot_dict["regime_blend"] = blend

        inst.portfolio.open_position(
            mint=token.mint,
            symbol=token.symbol,
            fill=fill,
            stop=stop,
            target=target,
            rr=signal.rr,
            entry_reason=reason,
            snapshot=snapshot_dict,
            review=decision.as_dict() if decision is not None else None,
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
            # The review gate decides ride-versus-lock-in per trade, so the
            # trailing distance can differ from the configured default for this
            # position and only this position.
            pos_cfg = cfg
            override = position.get("trail_override_atr")
            if override:
                pos_cfg = {**cfg, "trailing_distance_atr": float(override)}

            exit_signal = evaluate_exit(
                position, price, df if not df.empty else None, pos_cfg,
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
            trail = update_trailing_stop(position, price, atr_value, pos_cfg)
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

        if command == "run_wfmc_daily":
            threading.Thread(
                target=lambda: wfmc.run_daily(self.cfg, self.store),
                name="wfmc-daily-manual", daemon=True,
            ).start()
            return "daily walk-forward run started"

        if command == "run_wfmc_monthly":
            threading.Thread(
                target=lambda: wfmc.run_monthly(self.cfg, self.store, self.config.secrets),
                name="wfmc-monthly-manual", daemon=True,
            ).start()
            return "monthly RunPod retest started"

        if command == "run_runpod_benchmark":
            threading.Thread(
                target=lambda: wfmc.run_benchmark(self.cfg, self.store, self.config.secrets),
                name="runpod-benchmark-manual", daemon=True,
            ).start()
            return "RunPod GPU-tier benchmark started"

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
        pairs = self.store.pair_map(conn)
        wanted = payload.get("mints")
        if wanted:
            pairs = {m: p for m, p in pairs.items() if m in wanted}
        months = int(payload.get("months") or self.cfg["bulk_backfill_months"])

        def worker() -> None:
            try:
                self.store.run_initial_pull(pairs, months=months)
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
        self._maybe_daily_wfmc()
        self._maybe_monthly_wfmc()
        self._check_safety_alerts()
        self._check_drift(now)
        self._check_promotion(now)

    def _check_drift(self, now: float) -> None:
        """Compare live/paper performance against backtest expectation (4.5)."""
        interval = float(self.cfg.get("drift_check_seconds", 900))
        if now - self._last_drift_check < interval:
            return
        self._last_drift_check = now
        try:
            drift.check_all(self.cfg)
        except Exception as exc:
            log.warning("drift check failed: %s", exc)

    def _check_promotion(self, now: float) -> None:
        """Consider promoting the shadow parameter set to live.

        Bundles no longer arrive by being pulled from anywhere - the daily
        WFMC run installs one directly, and a RunPod worker's report lands via
        the dashboard's ingest endpoint - so this is promotion-only now.
        Guarded: a promotion that cannot be evaluated must never take the
        trading loop down with it.
        """
        interval = float(self.cfg.get("promotion_check_interval_seconds", 3600))
        if now - self._last_promotion_check < interval:
            return
        self._last_promotion_check = now

        try:
            verdict = paramsync.evaluate_promotion(self.cfg)
            if verdict.promoted:
                paramsync.promote(self.config, verdict)
                self._apply_config()
                self.build_instances()
        except Exception as exc:
            log.warning("promotion check failed: %s", exc)

    def _maybe_daily_wfmc(self) -> None:
        """Fire the daily on-droplet walk-forward/Monte Carlo run (spec 5)."""
        if not self._wfmc_hook_enabled or not self.cfg.get("wfmc_daily_enabled", True):
            return
        now = time.gmtime()
        if now.tm_hour != int(self.cfg.get("wfmc_daily_hour_utc", 3)):
            return
        today = time.strftime("%Y-%m-%d", now)
        if db.kv_get(wfmc.LAST_DAILY_KEY) == today:
            return
        db.kv_set(wfmc.LAST_DAILY_KEY, today)
        db.log_event("Daily walk-forward/Monte Carlo run starting.", category="system")

        def worker() -> None:
            try:
                self.store.run_daily_incremental(self.store.pair_map())
            except Exception:
                log.warning("daily incremental candle pull failed", exc_info=True)
            try:
                wfmc.run_daily(self.cfg, self.store)
            except Exception:
                log.exception("daily WFMC run failed")
                db.log_event(
                    "Daily walk-forward run failed; see the worker log.",
                    level="alert", category="system",
                )

        threading.Thread(target=worker, name="wfmc-daily", daemon=True).start()

    def _maybe_monthly_wfmc(self) -> None:
        """Fire the monthly RunPod-orchestrated full retest (spec 5)."""
        if not self._wfmc_hook_enabled or not self.cfg.get("wfmc_monthly_enabled", False):
            return
        now = time.gmtime()
        if now.tm_mday != int(self.cfg.get("wfmc_monthly_day_utc", 1)):
            return
        if now.tm_hour != int(self.cfg.get("wfmc_monthly_hour_utc", 3)):
            return
        month = time.strftime("%Y-%m", now)
        if db.kv_get(wfmc.LAST_MONTHLY_KEY) == month:
            return
        db.kv_set(wfmc.LAST_MONTHLY_KEY, month)
        db.log_event("Monthly RunPod walk-forward retest starting.", category="system")

        def worker() -> None:
            try:
                wfmc.run_monthly(self.cfg, self.store, self.config.secrets)
            except Exception:
                log.exception("monthly WFMC run failed")
                db.log_event(
                    "Monthly RunPod retest failed; see the worker log.",
                    level="alert", category="system",
                )

        threading.Thread(target=worker, name="wfmc-monthly", daemon=True).start()

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
        self.reviewer.update_config(self.cfg)
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
