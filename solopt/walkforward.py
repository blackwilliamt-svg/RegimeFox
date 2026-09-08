"""Walk-forward optimization.

Parameters are tuned on an in-sample window, tested on the next window the
search has never seen, and the whole thing rolls forward. That structure is the
entire point: any parameter set can be made to look good on data it was fitted
to, so the only number worth reading is what it did on data it was not.

The acceptance thresholds come straight from the spec and are deliberately
unforgiving:

* a window needs at least 30 closed trades before it counts at all - a window
  that fired six trades is a coin-flip wearing a percentage sign;
* at least 70% of counted out-of-sample windows must be profitable;
* if the standard deviation of window returns exceeds their mean, the set is
  flagged OVERFIT. A strategy that makes 2% on average by making 20% once and
  losing 6% four times is not a strategy, it is a lottery ticket with a backtest.

A run is resumable at window granularity, and re-running a *completed* run starts
fresh against whatever data has been pulled since, rather than resuming something
that already finished.
"""
from __future__ import annotations

import logging
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from .arrays import Backend, Throttle, get_backend
from .engine import (
    PortfolioSettings,
    VectorEngine,
    cache_requirements,
    trades_to_records,
)
from .frames import Frames
from .indicators import build_cache
from .params import ParamSpace, Search, SearchState

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Window:
    """One optimize-then-validate pair."""

    index: int
    is_from: int
    is_to: int
    oos_from: int
    oos_to: int

    @property
    def is_days(self) -> float:
        return (self.is_to - self.is_from) / 86400.0

    @property
    def oos_days(self) -> float:
        return (self.oos_to - self.oos_from) / 86400.0

    def label(self) -> str:
        return f"{_day(self.is_from)} to {_day(self.is_to)}"


def _day(ts: int) -> str:
    return time.strftime("%Y-%m-%d", time.gmtime(int(ts)))


@dataclass
class WindowResult:
    index: int
    status: str = "done"
    params: dict[str, Any] | None = None
    is_metrics: dict[str, Any] = field(default_factory=dict)
    oos_metrics: dict[str, Any] = field(default_factory=dict)
    search: dict[str, Any] = field(default_factory=dict)
    counted: bool = False
    profitable: bool = False
    note: str = ""
    trades: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index, "status": self.status, "params": self.params,
            "is_metrics": self.is_metrics, "oos_metrics": self.oos_metrics,
            "search": self.search, "counted": self.counted,
            "profitable": self.profitable, "note": self.note, "trades": self.trades,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WindowResult":
        return cls(**data)


@dataclass(frozen=True)
class WalkForwardConfig:
    """Window geometry, thresholds, and how hard the search is allowed to work."""

    in_sample_days: int = 60
    out_of_sample_days: int = 15
    step_days: int = 15

    min_trades_per_window: int = 30
    min_profitable_window_pct: float = 0.70

    max_evaluations: int = 2000
    batch_size: int = 64
    plateau_rounds: int = 4
    plateau_epsilon: float = 0.005
    seed: int = 20260903

    # Overtrading is the known failure mode: a prior backtest fired 244 trades in
    # seven days and paid $449 in fees for the privilege. A set that churns past
    # this rate is discounted rather than banned, so the search can still find it
    # if it is genuinely worth the cost.
    trades_per_day_cap: float = 8.0
    drawdown_floor: float = 0.05

    # Per-coin parameters need enough of that coin's own trades to be trusted.
    per_symbol_min_trades: int = 60
    per_symbol_margin: float = 0.20
    per_symbol_max_evaluations: int = 400
    per_symbol_enabled: bool = True

    workers: int = 0                 # 0 = one per core, minus one for the desktop
    utilization_pct: float = 100.0
    memory_budget_mb: int = 512
    warmup_bars: int = 0             # 0 = derive from the search space

    # Persistent library (gap-closure item 4): the share of each window's
    # first search batch seeded from combinations that already proved
    # themselves in an earlier run, rather than starting cold from
    # DEFAULT_GRID every time. The rest of that batch, and every later
    # round's refinement, is unaffected.
    library_seed_fraction: float = 0.3

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WalkForwardConfig":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class WalkForwardResult:
    """The verdict on one run."""

    windows: list[WindowResult] = field(default_factory=list)
    counted_windows: int = 0
    profitable_windows: int = 0
    profitable_share: float = 0.0
    mean_window_return: float = 0.0
    stdev_window_return: float = 0.0
    overfit: bool = False
    fragile: bool = False
    fragile_windows: list[str] = field(default_factory=list)
    walk_forward_efficiency: float = 0.0
    aggregate_is_return: float = 0.0
    aggregate_oos_return: float = 0.0
    best_params: dict[str, Any] | None = None
    per_symbol_params: dict[str, dict[str, Any]] = field(default_factory=dict)
    accepted: bool = False
    reasons: list[str] = field(default_factory=list)
    elapsed_minutes: float = 0.0

    def oos_trades(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for window in self.windows:
            out.extend(window.trades)
        out.sort(key=lambda t: t["exit_ts"])
        return out

    def summary(self) -> dict[str, Any]:
        return {
            "windows": len(self.windows),
            "counted_windows": self.counted_windows,
            "profitable_windows": self.profitable_windows,
            "profitable_share": round(self.profitable_share, 4),
            "mean_window_return": round(self.mean_window_return, 6),
            "stdev_window_return": round(self.stdev_window_return, 6),
            "overfit": self.overfit,
            "fragile": self.fragile,
            "fragile_windows": self.fragile_windows,
            "walk_forward_efficiency": round(self.walk_forward_efficiency, 4),
            "aggregate_is_return": round(self.aggregate_is_return, 6),
            "aggregate_oos_return": round(self.aggregate_oos_return, 6),
            "best_params": self.best_params,
            "per_symbol_params": self.per_symbol_params,
            "accepted": self.accepted,
            "reasons": self.reasons,
            "elapsed_minutes": round(self.elapsed_minutes, 2),
            "trades": sum(len(w.trades) for w in self.windows),
        }


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
def objective_score(
    metrics: dict[str, Any], days: float, config: WalkForwardConfig
) -> float:
    """Return per unit of drawdown, discounted for churn.

    Raw return picks the set that got luckiest; return divided by the drawdown it
    took to get there picks the set an operator could actually hold. The churn
    discount then makes the search pay for trade frequency, which is what stops
    it rediscovering the 35-trades-a-day behaviour that produced the fee bill.
    """
    trades = int(metrics.get("trades", 0))
    if trades < config.min_trades_per_window:
        return -math.inf
    drawdown = max(float(metrics.get("max_drawdown", 0.0)), config.drawdown_floor)
    score = float(metrics.get("total_return", 0.0)) / drawdown
    rate = trades / max(days, 1.0)
    if rate > config.trades_per_day_cap > 0:
        # The discount has to move the score *down* whichever side of zero it
        # sits: scaling a negative score toward zero would make churn look like
        # an improvement, which is precisely the behaviour being penalised.
        factor = config.trades_per_day_cap / rate
        score = score * factor if score > 0 else score / factor
    return score


def plan_windows(
    first_ts: int, last_ts: int, config: WalkForwardConfig
) -> list[Window]:
    """Roll optimize-then-validate pairs across the whole history."""
    is_span = int(config.in_sample_days) * 86400
    oos_span = int(config.out_of_sample_days) * 86400
    step = max(1, int(config.step_days)) * 86400

    windows: list[Window] = []
    is_from = int(first_ts)
    index = 0
    while is_from + is_span + oos_span <= int(last_ts) + 1:
        is_to = is_from + is_span
        windows.append(Window(index, is_from, is_to, is_to, is_to + oos_span))
        index += 1
        is_from += step
    return windows


def derive_warmup(space: ParamSpace, settings: PortfolioSettings) -> int:
    """Bars of lead-in a window needs before its indicators mean anything."""
    longest = 0
    for name in ("ema_slow", "volume_spike_lookback", "regime_lookback", "rsi_period", "atr_period"):
        values = space.values.get(name) or space.fixed.get(name)
        if isinstance(values, (list, tuple)) and values:
            longest = max(longest, int(max(values)))
        elif isinstance(values, (int, float)):
            longest = max(longest, int(values))
    multiple = max(settings.confluence_multiples) if settings.confluence_multiples else 1
    return int(max(settings.min_candles_required, longest * 3, longest * multiple) + 5)


# --------------------------------------------------------------------------
# One window's work - importable at module level so it can run in a subprocess
# --------------------------------------------------------------------------
_WORKER: dict[str, Any] = {}


def _init_worker(
    frames: Frames,
    settings: PortfolioSettings,
    config: WalkForwardConfig,
    space_values: dict[str, list[Any]],
    space_fixed: dict[str, Any],
    warmup: int,
    seed_pool: Sequence[dict[str, Any]] = (),
) -> None:  # pragma: no cover - exercised only via the process pool
    _WORKER.update(
        frames=frames,
        settings=settings,
        config=config,
        space=ParamSpace(values=space_values, fixed=space_fixed),
        warmup=warmup,
        seed_pool=seed_pool,
        engine=VectorEngine(
            get_backend(),
            Throttle(config.utilization_pct),
            memory_budget_mb=config.memory_budget_mb,
        ),
    )


def _worker_window(window: Window) -> dict[str, Any]:  # pragma: no cover - subprocess
    return evaluate_window(
        window,
        _WORKER["frames"],
        _WORKER["space"],
        _WORKER["settings"],
        _WORKER["config"],
        _WORKER["engine"],
        _WORKER["warmup"],
        seed_pool=_WORKER.get("seed_pool", ()),
    ).as_dict()


def evaluate_window(
    window: Window,
    frames: Frames,
    space: ParamSpace,
    settings: PortfolioSettings,
    config: WalkForwardConfig,
    engine: VectorEngine,
    warmup: int,
    *,
    on_batch: Callable[[int, SearchState], None] | None = None,
    seed_pool: Sequence[dict[str, Any]] = (),
) -> WindowResult:
    """Search the in-sample window, then test the winner out of sample."""
    result = WindowResult(index=window.index)

    is_frames = frames.window(window.is_from, window.is_to, warmup_bars=warmup)
    if is_frames.n_bars == 0 or not is_frames.tradeable.any():
        result.status = "skipped"
        result.note = "no eligible bars in the in-sample window"
        return result

    search = Search(
        space=space,
        max_evaluations=config.max_evaluations,
        batch_size=config.batch_size,
        plateau_rounds=config.plateau_rounds,
        plateau_epsilon=config.plateau_epsilon,
        seed=config.seed + window.index,
        seed_pool=seed_pool,
        seed_fraction=config.library_seed_fraction,
    )

    cache = _shared_cache(is_frames, space, settings, config)
    for batch in search.batches():
        run = engine.run(is_frames, batch, settings, cache=cache)
        scored = [
            (combo, objective_score(metrics, window.is_days, config))
            for combo, metrics in zip(batch, run.metrics)
        ]
        search.observe(scored)
        if on_batch is not None:
            on_batch(window.index, search.state)

    result.search = search.state.as_dict()
    if not search.state.best_combo:
        result.status = "skipped"
        result.note = (
            f"no in-sample combination produced the {config.min_trades_per_window} "
            "trades a window needs to count"
        )
        return result

    result.params = search.state.best_combo
    is_run = engine.run(is_frames, [result.params], settings)
    result.is_metrics = is_run.metrics[0]

    oos_frames = frames.window(window.oos_from, window.oos_to, warmup_bars=warmup)
    if oos_frames.n_bars == 0 or not oos_frames.tradeable.any():
        result.status = "skipped"
        result.note = "no eligible bars in the out-of-sample window"
        return result

    oos_run = engine.run(oos_frames, [result.params], settings)
    result.oos_metrics = oos_run.metrics[0]
    result.trades = trades_to_records(oos_run.trades, oos_run.symbols)
    result.counted = result.oos_metrics["trades"] >= config.min_trades_per_window
    result.profitable = result.counted and result.oos_metrics["total_return"] > 0
    if not result.counted:
        result.note = (
            f"{result.oos_metrics['trades']} out-of-sample trades, below the "
            f"{config.min_trades_per_window}-trade floor"
        )
    return result


def _shared_cache(
    frames: Frames,
    space: ParamSpace,
    settings: PortfolioSettings,
    config: WalkForwardConfig,
) -> Any:
    """Precompute every series the whole space needs, if it fits in budget.

    Building one cache per window instead of one per batch is the difference
    between recomputing the same EMA forty times and computing it once. When the
    space is too wide for that to fit, the engine falls back to building exactly
    what each batch needs.
    """
    try:
        requirements = cache_requirements(space.full(), settings)
    except MemoryError:  # pragma: no cover - a pathologically wide space
        return None
    series = sum(len(v) for v in requirements.values()) + len(requirements.get("atr", []))
    estimated = series * frames.n_symbols * frames.n_bars * 4
    if estimated > config.memory_budget_mb * 1024 * 1024:
        log.info(
            "space cache would need %.0f MB; letting each batch build its own",
            estimated / 1e6,
        )
        return None
    return build_cache(frames, requirements)


# --------------------------------------------------------------------------
# The run
# --------------------------------------------------------------------------
class WalkForward:
    """Drives windows, aggregates them, and decides whether the result stands."""

    def __init__(
        self,
        space: ParamSpace,
        settings: PortfolioSettings,
        config: WalkForwardConfig,
        *,
        backend: Backend | None = None,
        feed: Any = None,
        store: Any = None,
        run_id: int = 0,
    ) -> None:
        self.space = space
        self.settings = settings
        self.config = config
        self.backend = backend or get_backend()
        self.throttle = Throttle(config.utilization_pct, self.backend)
        self.engine = VectorEngine(
            self.backend, self.throttle, memory_budget_mb=config.memory_budget_mb
        )
        self.feed = feed
        self.store = store
        self.run_id = run_id
        self.seed_pool = self._fetch_seed_pool()

    def _fetch_seed_pool(self) -> list[dict[str, Any]]:
        """Top library entries (gap-closure item 4), market-wide only - a
        window's own per-symbol seeding happens where per-symbol params are
        searched (see :meth:`_per_symbol`), not here. `store` may be a bare
        RunStore substitute in a test or an early call site that predates the
        library; either way, no library just means no seeding, not an error.
        """
        fetch = getattr(self.store, "top_library_entries", None)
        if not callable(fetch) or self.config.library_seed_fraction <= 0:
            return []
        try:
            n = max(1, self.config.batch_size)
            return [e["params"] for e in fetch(n, symbol=None)]
        except Exception:
            log.debug("library seed pool unavailable", exc_info=True)
            return []

    # ------------------------------------------------------------------
    def run(self, frames: Frames, *, resume: bool = True) -> WalkForwardResult:
        started = time.monotonic()
        ts = frames.ts[frames.mask]
        if ts.size == 0:
            raise ValueError("the panel contains no bars to walk forward over")

        windows = plan_windows(int(ts.min()), int(ts.max()), self.config)
        warmup = self.config.warmup_bars or derive_warmup(self.space, self.settings)
        if self.feed:
            self.feed.planned(
                len(windows), self.config.in_sample_days, self.config.out_of_sample_days
            )
        if not windows:
            raise ValueError(
                f"history covers {(int(ts.max()) - int(ts.min())) / 86400:.0f} days, "
                f"too short for a {self.config.in_sample_days}+"
                f"{self.config.out_of_sample_days}-day window"
            )
        if self.store:
            self.store.plan_windows(self.run_id, windows)

        done: dict[int, WindowResult] = {}
        if resume and self.store:
            already = self.store.completed_windows(self.run_id)
            for row in self.store.windows(self.run_id):
                if int(row["idx"]) in already:
                    done[int(row["idx"])] = WindowResult(
                        index=int(row["idx"]),
                        status=row["status"],
                        params=row["params"] or None,
                        is_metrics=row["is_metrics"] or {},
                        oos_metrics=row["oos_metrics"] or {},
                        search=row["search"] or {},
                        counted=bool(row["counted"]),
                        profitable=bool(row["profitable"]),
                        note=row["note"] or "",
                        trades=[],
                    )
            if done and self.feed:
                self.feed.say(
                    f"Resuming: {len(done)} of {len(windows)} windows were already "
                    "finished by an earlier run and will not be repeated."
                )
            for index in list(done):
                done[index].trades = [
                    t for t in self.store.oos_trades(self.run_id)
                    if int(t["window_idx"]) == index
                ]

        pending = [w for w in windows if w.index not in done]
        for result in self._execute(pending, frames, warmup, len(windows)):
            done[result.index] = result
            if self.store:
                self.store.record_window(self.run_id, result)

        ordered = [done[w.index] for w in windows if w.index in done]
        outcome = self.aggregate(ordered)
        outcome.elapsed_minutes = (time.monotonic() - started) / 60.0

        if outcome.best_params and self.config.per_symbol_enabled:
            outcome.per_symbol_params = self._per_symbol(frames, outcome, warmup)
        self._narrate(outcome)
        return outcome

    # ------------------------------------------------------------------
    def _execute(
        self, windows: Sequence[Window], frames: Frames, warmup: int, total: int
    ):
        if not windows:
            return
        workers = self.config.workers or max(1, _cpu_count() - 1)
        if workers <= 1 or len(windows) == 1:
            for window in windows:
                if self.feed:
                    self.feed.searching(
                        window.index, total, _day(window.is_from), _day(window.is_to),
                        min(self.space.size(), self.config.max_evaluations),
                    )
                yield self._one(window, frames, warmup)
            return

        # Windows are independent, so the orchestration layer is where multi-core
        # pays: each process gets its own copy of the panel once and then chews
        # through whole windows without talking to the others.
        if self.feed:
            self.feed.say(
                f"Running {len(windows)} windows across {workers} CPU workers at a "
                f"{self.config.utilization_pct:.0f}% utilization cap."
            )
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(
                frames, self.settings, self.config, self.space.values,
                self.space.fixed, warmup, self.seed_pool,
            ),
        ) as pool:
            futures = {pool.submit(_worker_window, w): w for w in windows}
            for future in as_completed(futures):
                window = futures[future]
                try:
                    yield WindowResult.from_dict(future.result())
                except Exception as exc:
                    log.exception("window %s failed", window.index)
                    if self.feed:
                        self.feed.warn(f"Window {window.index + 1} failed: {exc}")
                    yield WindowResult(
                        index=window.index, status="failed", note=str(exc)[:300]
                    )

    def _one(self, window: Window, frames: Frames, warmup: int) -> WindowResult:
        def on_batch(index: int, state: SearchState) -> None:
            if self.feed and state.rounds % 5 == 0 and state.best_combo:
                self.feed.say(
                    f"Window {index + 1}: {state.evaluated:,} combinations tried, "
                    f"best score {state.best_score:.2f} so far."
                )

        result = evaluate_window(
            window, frames, self.space, self.settings, self.config,
            self.engine, warmup, on_batch=on_batch, seed_pool=self.seed_pool,
        )
        if self.feed:
            if result.status == "skipped":
                self.feed.skipped(result.index, result.note)
            else:
                self.feed.in_sample(result.index, result.params or {}, result.is_metrics)
                self.feed.out_of_sample(result.index, result.oos_metrics, result.counted)
        return result

    # ------------------------------------------------------------------
    def aggregate(self, windows: Sequence[WindowResult]) -> WalkForwardResult:
        """Apply the acceptance thresholds to a set of finished windows."""
        outcome = WalkForwardResult(windows=list(windows))
        counted = [w for w in windows if w.counted]
        outcome.counted_windows = len(counted)
        outcome.profitable_windows = sum(1 for w in counted if w.profitable)
        outcome.profitable_share = (
            outcome.profitable_windows / len(counted) if counted else 0.0
        )

        returns = np.array(
            [float(w.oos_metrics.get("total_return", 0.0)) for w in counted],
            dtype=np.float64,
        )
        if returns.size:
            outcome.mean_window_return = float(returns.mean())
            # Sample standard deviation: with a handful of windows the population
            # form understates the spread, which is the wrong way to be wrong
            # when the number gates an overfit flag.
            outcome.stdev_window_return = float(returns.std(ddof=1)) if returns.size > 1 else 0.0
            outcome.overfit = outcome.stdev_window_return > outcome.mean_window_return

        outcome.aggregate_is_return = float(
            sum(float(w.is_metrics.get("total_return", 0.0)) for w in counted)
        )
        outcome.aggregate_oos_return = float(returns.sum()) if returns.size else 0.0
        outcome.walk_forward_efficiency = self._efficiency(counted, outcome)

        outcome.best_params = self._consensus(counted)

        reasons: list[str] = []
        if not counted:
            reasons.append(
                f"no window reached the {self.config.min_trades_per_window}-trade floor"
            )
        if counted and outcome.profitable_share < self.config.min_profitable_window_pct:
            reasons.append(
                f"only {outcome.profitable_share * 100:.0f}% of windows were profitable "
                f"out of sample, below the "
                f"{self.config.min_profitable_window_pct * 100:.0f}% threshold"
            )
        if outcome.overfit:
            reasons.append(
                f"flagged overfit: window returns vary by "
                f"{outcome.stdev_window_return * 100:.1f}% around a "
                f"{outcome.mean_window_return * 100:.1f}% mean"
            )
        outcome.reasons = reasons
        outcome.accepted = not reasons and outcome.best_params is not None
        return outcome

    def _efficiency(
        self, counted: Sequence[WindowResult], outcome: WalkForwardResult
    ) -> float:
        """Out-of-sample return per day divided by in-sample return per day.

        Normalising by days matters: in-sample windows are four times longer
        here, so comparing raw totals would report an efficiency of roughly 0.25
        for a strategy that generalised perfectly.
        """
        is_days = self.config.in_sample_days * len(counted)
        oos_days = self.config.out_of_sample_days * len(counted)
        if not counted or is_days <= 0 or oos_days <= 0:
            return 0.0
        is_rate = outcome.aggregate_is_return / is_days
        oos_rate = outcome.aggregate_oos_return / oos_days
        if is_rate <= 0:
            return 0.0
        return float(oos_rate / is_rate)

    @staticmethod
    def _consensus(counted: Sequence[WindowResult]) -> dict[str, Any] | None:
        """The parameter set to carry forward: the most-agreed-on winner.

        Taking the single best window's parameters would promote whichever window
        got luckiest. Taking the set that won most often - breaking ties on
        out-of-sample return - promotes the one that keeps working.
        """
        if not counted:
            return None
        tally: dict[tuple, list[Any]] = {}
        for window in counted:
            if not window.params:
                continue
            key = tuple(sorted(window.params.items()))
            entry = tally.setdefault(key, [0, 0.0, window.params])
            entry[0] += 1
            entry[1] += float(window.oos_metrics.get("total_return", 0.0))
        if not tally:
            return None
        best = max(tally.values(), key=lambda e: (e[0], e[1]))
        return dict(best[2])

    # ------------------------------------------------------------------
    def _per_symbol(
        self, frames: Frames, outcome: WalkForwardResult, warmup: int
    ) -> dict[str, dict[str, Any]]:
        """Tune per coin where that coin has enough of its own trade history.

        A coin with 200 trades behind it can support its own parameters. A coin
        with nine cannot, and giving it some anyway is how a walk-forward run
        quietly turns into per-coin curve fitting - so those fall back to the
        global set, which is the honest answer.
        """
        counts: dict[str, int] = {}
        for window in outcome.windows:
            for trade in window.trades:
                counts[trade["symbol"]] = counts.get(trade["symbol"], 0) + 1

        eligible = [
            s for s, n in counts.items() if n >= self.config.per_symbol_min_trades
        ]
        if not eligible:
            if self.feed:
                self.feed.say(
                    "No coin has enough of its own trade history for per-coin "
                    f"parameters (the floor is {self.config.per_symbol_min_trades} "
                    "trades); every coin uses the global set."
                )
            return {}

        if self.feed:
            self.feed.say(
                f"{len(eligible)} coin(s) have enough history to try per-coin "
                "parameters; the rest keep the global set."
            )

        config = WalkForwardConfig(
            **{
                **self.config.as_dict(),
                "max_evaluations": self.config.per_symbol_max_evaluations,
                "per_symbol_enabled": False,
                "workers": 1,
            }
        )
        adopted: dict[str, dict[str, Any]] = {}
        for symbol in sorted(eligible):
            index = frames.symbols.index(symbol)
            solo = _single_symbol(frames, index)
            baseline = self._symbol_baseline(outcome, symbol)
            runner = WalkForward(
                self.space, self.settings, config, backend=self.backend
            )
            try:
                solo_outcome = runner.run(solo, resume=False)
            except ValueError:
                continue
            if not solo_outcome.accepted or not solo_outcome.best_params:
                continue
            gain = solo_outcome.aggregate_oos_return - baseline
            if baseline > 0 and gain < abs(baseline) * self.config.per_symbol_margin:
                continue
            if baseline <= 0 and solo_outcome.aggregate_oos_return <= 0:
                continue
            adopted[symbol] = solo_outcome.best_params
            if self.feed:
                self.feed.say(
                    f"{symbol}: its own parameters beat the global set out of sample "
                    f"({solo_outcome.aggregate_oos_return * 100:+.1f}% against "
                    f"{baseline * 100:+.1f}%); adopting them for this coin only."
                )
        return adopted

    @staticmethod
    def _symbol_baseline(outcome: WalkForwardResult, symbol: str) -> float:
        """What the global set actually earned on this coin, as a return share."""
        pnl = 0.0
        size = 0.0
        for window in outcome.windows:
            for trade in window.trades:
                if trade["symbol"] == symbol:
                    pnl += trade["pnl_usd"]
                    size += trade["size_usd"]
        return pnl / size if size > 0 else 0.0

    # ------------------------------------------------------------------
    def _narrate(self, outcome: WalkForwardResult) -> None:
        if not self.feed:
            return
        self.feed.consistency(
            outcome.profitable_windows,
            outcome.counted_windows,
            self.config.min_profitable_window_pct,
        )
        if outcome.counted_windows:
            self.feed.overfit(
                outcome.mean_window_return, outcome.stdev_window_return, outcome.overfit
            )
            self.feed.efficiency(
                outcome.walk_forward_efficiency,
                outcome.aggregate_is_return,
                outcome.aggregate_oos_return,
            )
        for reason in outcome.reasons:
            self.feed.warn(f"Not accepted: {reason}.")


def _single_symbol(frames: Frames, index: int) -> Frames:
    """A one-symbol view, for per-coin tuning."""
    sl = slice(index, index + 1)
    return Frames(
        symbols=[frames.symbols[index]],
        ts=frames.ts[sl],
        open=frames.open[sl],
        high=frames.high[sl],
        low=frames.low[sl],
        close=frames.close[sl],
        volume=frames.volume[sl],
        mask=frames.mask[sl],
        eligible=frames.eligible[sl],
        counts=frames.counts[sl],
        grid_pos=frames.grid_pos[sl],
        seconds=frames.seconds,
        schema=frames.schema,
        liquidity=None if frames.liquidity is None else frames.liquidity[sl],
    )


def _cpu_count() -> int:
    import os

    return os.cpu_count() or 1
