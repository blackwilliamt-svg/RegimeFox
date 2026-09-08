"""Monte Carlo resampling of walk-forward out-of-sample trades.

This is not another backtest. It consumes exactly the trades the walk-forward
run already produced on data it had never seen, and asks a different question:
how much of that result was the *order* things happened in, and how much would
survive execution behaving as badly as it sometimes does?

Two sources of randomness, both from the spec:

* **Trade order.** With position size a fraction of current equity, order decides
  the shape of the equity path: a losing streak early compounds differently from
  the same streak late, and the worst drawdown moves a long way with it.
* **Execution variance.** Slippage and fees are sampled per trade from what the
  live and paper executors *actually* recorded, not from the constants the
  backtest assumed. A strategy whose edge is thinner than its real fill costs
  looks fine against an assumed 0.25% and falls apart against the observed
  distribution, and that is the failure this catches.

**On resampling with replacement.** Reordering alone cannot produce a
terminal-return distribution: under fixed-fraction sizing the equity path is a
cumulative product, and a product does not care what order its terms come in.
Permuting 112 trades gives 112 different drawdown paths and exactly one terminal
value, so a "5th-percentile terminal return" built that way would report the
spread of the execution noise and nothing else. Drawing the trades *with
replacement* - the standard bootstrap - keeps the ordering randomness the spec
asks for and adds the sampling question that makes the tail meaningful: what if
the next hundred trades are a different hundred draws from the same strategy?
That is the default; ``resample="permutation"`` restores pure reordering for
anyone who wants the drawdown-only view.

The number that matters downstream is the 5%-worst-case maximum drawdown, which
feeds position sizing - not the single historical drawdown, which is one draw
from this distribution with no claim to being the representative one.
"""
from __future__ import annotations

import heapq
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

log = logging.getLogger(__name__)

DEFAULT_ITERATIONS = 5000
HISTOGRAM_BINS = 40


@dataclass(frozen=True, slots=True)
class ExecutionProfile:
    """Per-side execution cost, in percent of notional.

    Built from observed fills where there are enough of them, and from the
    configured constants otherwise. ``source`` says which, because a Monte Carlo
    run against assumed constants is a weaker claim than one against measured
    fills and the dashboard should not present them as the same thing.
    """

    slippage_mean_pct: float = 0.25
    slippage_stdev_pct: float = 0.10
    fee_mean_pct: float = 0.25
    fee_stdev_pct: float = 0.02
    samples: int = 0
    source: str = "assumed"

    # Sampled costs are clamped to this multiple of the mean. Without a ceiling a
    # fat-tailed draw can produce a negative fill price, which is not pessimism,
    # it is nonsense.
    max_multiple: float = 4.0

    @classmethod
    def from_fills(
        cls,
        slippage_pct: Sequence[float],
        fee_pct: Sequence[float],
        *,
        min_samples: int = 30,
        fallback: "ExecutionProfile | None" = None,
    ) -> "ExecutionProfile":
        slip = np.asarray([s for s in slippage_pct if math.isfinite(s)], dtype=np.float64)
        fee = np.asarray([f for f in fee_pct if math.isfinite(f)], dtype=np.float64)
        if slip.size < min_samples or fee.size < min_samples:
            base = fallback or cls()
            return cls(
                slippage_mean_pct=base.slippage_mean_pct,
                slippage_stdev_pct=base.slippage_stdev_pct,
                fee_mean_pct=base.fee_mean_pct,
                fee_stdev_pct=base.fee_stdev_pct,
                samples=int(min(slip.size, fee.size)),
                source="assumed",
            )
        return cls(
            slippage_mean_pct=float(slip.mean()),
            slippage_stdev_pct=float(slip.std(ddof=1)),
            fee_mean_pct=float(fee.mean()),
            fee_stdev_pct=float(fee.std(ddof=1)),
            samples=int(min(slip.size, fee.size)),
            source="observed",
        )

    @classmethod
    def assumed(cls, slippage_pct: float, fee_pct: float) -> "ExecutionProfile":
        # With no observations, the spread is a guess. A quarter of the mean is
        # deliberately generous: understating execution variance is the mistake
        # that makes a marginal strategy look safe.
        return cls(
            slippage_mean_pct=float(slippage_pct),
            slippage_stdev_pct=float(slippage_pct) * 0.25,
            fee_mean_pct=float(fee_pct),
            fee_stdev_pct=float(fee_pct) * 0.05,
            samples=0,
            source="assumed",
        )

    def round_trip_pct(self) -> float:
        """Expected round-trip cost as a fraction of notional."""
        return 2.0 * (self.slippage_mean_pct + self.fee_mean_pct) / 100.0

    def sample_round_trip(
        self, shape: tuple[int, ...], rng: np.random.Generator
    ) -> np.ndarray:
        """Draw round-trip costs as fractions of notional."""
        slip = rng.normal(self.slippage_mean_pct, self.slippage_stdev_pct, shape)
        fee = rng.normal(self.fee_mean_pct, self.fee_stdev_pct, shape)
        slip = np.clip(slip, 0.0, self.slippage_mean_pct * self.max_multiple)
        fee = np.clip(fee, 0.0, max(self.fee_mean_pct * self.max_multiple, 1e-9))
        return 2.0 * (slip + fee) / 100.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "slippage_mean_pct": round(self.slippage_mean_pct, 4),
            "slippage_stdev_pct": round(self.slippage_stdev_pct, 4),
            "fee_mean_pct": round(self.fee_mean_pct, 4),
            "fee_stdev_pct": round(self.fee_stdev_pct, 4),
            "samples": self.samples,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExecutionProfile":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in (data or {}).items() if k in known})


@dataclass
class MonteCarloResult:
    iterations: int = 0
    trades: int = 0
    median_return: float = 0.0
    mean_return: float = 0.0
    stdev_return: float = 0.0
    best_return: float = 0.0
    worst_return: float = 0.0
    p5_return: float = 0.0
    p95_return: float = 0.0
    median_max_drawdown: float = 0.0
    p5_max_drawdown: float = 0.0
    historical_return: float = 0.0
    historical_max_drawdown: float = 0.0
    probability_of_loss: float = 0.0
    risk_of_ruin: float = 0.0
    resample: str = "bootstrap"
    execution: dict[str, Any] = field(default_factory=dict)
    drawdown_histogram: dict[str, Any] = field(default_factory=dict)
    return_histogram: dict[str, Any] = field(default_factory=dict)
    # The worst-K simulated equity paths by max drawdown, as a fraction of
    # starting balance (spec 6a: plotted explicitly, not just the single
    # 5%-worst-case number). Worst first.
    worst_equity_paths: list[list[float]] = field(default_factory=list)
    worst_path_drawdowns: list[float] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "iterations": self.iterations,
            "trades": self.trades,
            "median_return": round(self.median_return, 6),
            "mean_return": round(self.mean_return, 6),
            "stdev_return": round(self.stdev_return, 6),
            "best_return": round(self.best_return, 6),
            "worst_return": round(self.worst_return, 6),
            "p5_return": round(self.p5_return, 6),
            "p95_return": round(self.p95_return, 6),
            "median_max_drawdown": round(self.median_max_drawdown, 6),
            "p5_max_drawdown": round(self.p5_max_drawdown, 6),
            "historical_return": round(self.historical_return, 6),
            "historical_max_drawdown": round(self.historical_max_drawdown, 6),
            "probability_of_loss": round(self.probability_of_loss, 4),
            "risk_of_ruin": round(self.risk_of_ruin, 4),
            "resample": self.resample,
            "execution": self.execution,
        }

    def histograms(self) -> dict[str, Any]:
        return {"drawdown": self.drawdown_histogram, "return": self.return_histogram}

    def worst_paths(self) -> dict[str, Any]:
        """The worst-K equity curves, for the dashboard's detailed Monte Carlo view."""
        return {
            "paths": self.worst_equity_paths,
            "drawdowns": [round(d, 6) for d in self.worst_path_drawdowns],
        }


def _histogram(values: np.ndarray, bins: int = HISTOGRAM_BINS) -> dict[str, Any]:
    if values.size == 0:
        return {"edges": [], "counts": [], "total": 0}
    counts, edges = np.histogram(values, bins=bins)
    return {
        "edges": [round(float(e), 6) for e in edges],
        "counts": [int(c) for c in counts],
        "total": int(values.size),
    }


def simulate(
    trades: Sequence[dict[str, Any]],
    *,
    starting_balance: float,
    execution: ExecutionProfile,
    assumed_round_trip_pct: float,
    iterations: int = DEFAULT_ITERATIONS,
    position_fraction: float | None = None,
    seed: int = 20260903,
    ruin_threshold: float = 0.5,
    resample: str = "bootstrap",
) -> MonteCarloResult:
    """Resample ``trades`` and report the distribution of outcomes.

    Each trade becomes a return on the notional it deployed. Replaying those in a
    random order with size held at a constant fraction of current equity turns
    the equity path into a cumulative product, which is why thousands of
    iterations cost three array operations rather than a loop.
    """
    result = MonteCarloResult(iterations=0, trades=len(trades))
    if not trades or starting_balance <= 0:
        return result

    pnl = np.asarray([float(t["pnl_usd"]) for t in trades], dtype=np.float64)
    size = np.asarray([float(t["size_usd"]) for t in trades], dtype=np.float64)
    usable = size > 0
    pnl, size = pnl[usable], size[usable]
    if pnl.size == 0:
        return result
    result.trades = int(pnl.size)

    returns = pnl / size
    if position_fraction is None:
        # Reproduce the sizing the run actually used, so the resampled paths are
        # about ordering and execution rather than about a different bet size.
        position_fraction = float(np.median(size) / starting_balance)
    fraction = float(min(max(position_fraction, 1e-4), 1.0))

    rng = np.random.default_rng(seed)
    n = returns.size
    iterations = max(1, int(iterations))

    # The recorded returns already carry the backtest's assumed costs; strip
    # those out before adding a sampled draw, or the costs are charged twice.
    stripped = returns + assumed_round_trip_pct

    # Each iteration needs several (n+1)-wide float64 rows live at once. Chunking
    # keeps that inside a fixed budget however many trades the run produced,
    # rather than allocating iterations x trades and hoping.
    per_iteration_bytes = max(1, (n + 1) * 8 * 6)
    chunk = max(1, min(iterations, (256 * 1024 * 1024) // per_iteration_bytes))

    max_drawdown = np.empty(iterations, dtype=np.float64)
    terminal = np.empty(iterations, dtype=np.float64)

    # A bounded min-heap of the worst-K simulated paths by max drawdown (spec
    # 6a wants the actual curves plotted, not just the 5%-worst-case number).
    # Kept as a running top-K across chunks rather than regenerated afterward,
    # since re-deriving a specific iteration's draw from the RNG stream after
    # the fact is not something `default_rng` supports cheaply.
    worst_k = 10
    worst_heap: list[tuple[float, int, np.ndarray]] = []
    tiebreak = 0

    for lo in range(0, iterations, chunk):
        hi = min(iterations, lo + chunk)
        rows = hi - lo
        if resample == "permutation":
            order = np.argsort(rng.random((rows, n)), axis=1)
        else:
            order = rng.integers(0, n, size=(rows, n))
        drawn = stripped[order] - execution.sample_round_trip((rows, n), rng)

        growth = 1.0 + fraction * drawn
        # A single trade cannot lose more than the position it deployed; below
        # zero the equity path would flip sign, which no exit could produce.
        np.clip(growth, 0.0, None, out=growth)

        equity = np.empty((rows, n + 1), dtype=np.float64)
        equity[:, 0] = starting_balance
        np.cumprod(growth, axis=1, out=equity[:, 1:])
        equity[:, 1:] *= starting_balance

        peak = np.maximum.accumulate(equity, axis=1)
        drawdown = np.where(peak > 0, (peak - equity) / peak, 0.0)
        row_drawdown = drawdown.max(axis=1)
        max_drawdown[lo:hi] = row_drawdown
        terminal[lo:hi] = equity[:, -1] / starting_balance - 1.0

        normalized = equity / starting_balance
        for i in range(rows):
            dd = float(row_drawdown[i])
            entry = (dd, tiebreak, normalized[i].copy())
            tiebreak += 1
            if len(worst_heap) < worst_k:
                heapq.heappush(worst_heap, entry)
            elif dd > worst_heap[0][0]:
                heapq.heapreplace(worst_heap, entry)

    result.iterations = iterations
    result.worst_path_drawdowns = [dd for dd, _, _ in sorted(worst_heap, key=lambda e: -e[0])]
    result.worst_equity_paths = [
        row.tolist() for _, _, row in sorted(worst_heap, key=lambda e: -e[0])
    ]
    result.median_return = float(np.median(terminal))
    result.mean_return = float(terminal.mean())
    result.stdev_return = float(terminal.std(ddof=1)) if terminal.size > 1 else 0.0
    result.best_return = float(terminal.max())
    result.worst_return = float(terminal.min())
    result.p5_return = float(np.percentile(terminal, 5))
    result.p95_return = float(np.percentile(terminal, 95))
    result.median_max_drawdown = float(np.median(max_drawdown))
    # "5th-percentile drawdown" means the 5%-worst case, which is the 95th
    # percentile of the drawdown magnitudes. The literal 5th percentile would be
    # the mildest outcome, which is exactly the wrong number to size against.
    result.p5_max_drawdown = float(np.percentile(max_drawdown, 95))
    result.probability_of_loss = float((terminal < 0).mean())
    result.risk_of_ruin = float((max_drawdown >= ruin_threshold).mean())

    historical = starting_balance * np.cumprod(1.0 + fraction * returns)
    historical = np.concatenate(([starting_balance], historical))
    hist_peak = np.maximum.accumulate(historical)
    result.historical_return = float(historical[-1] / starting_balance - 1.0)
    result.historical_max_drawdown = float(
        np.max(np.where(hist_peak > 0, (hist_peak - historical) / hist_peak, 0.0))
    )

    result.resample = "permutation" if resample == "permutation" else "bootstrap"
    result.execution = execution.as_dict()
    result.drawdown_histogram = _histogram(max_drawdown)
    result.return_histogram = _histogram(terminal)
    return result


def from_walk_forward(
    outcome: Any,
    *,
    starting_balance: float,
    execution: ExecutionProfile,
    assumed_round_trip_pct: float,
    iterations: int = DEFAULT_ITERATIONS,
    seed: int = 20260903,
) -> MonteCarloResult:
    """Run the resampling over a walk-forward result's out-of-sample trades."""
    return simulate(
        outcome.oos_trades(),
        starting_balance=starting_balance,
        execution=execution,
        assumed_round_trip_pct=assumed_round_trip_pct,
        iterations=iterations,
        seed=seed,
    )
