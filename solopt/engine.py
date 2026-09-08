"""The vectorized backtest engine.

This is the foundation everything else sits on, and it is built around one
decision: the unit of work is *the whole search*, not one backtest. Signals and
trade outcomes are computed as batched array operations across every symbol and
every parameter combination at once. There is no per-combination loop over
candles anywhere in here.

Four stages, in order:

1. **Group by indicator signature.** Combinations that share every
   indicator *period* share every indicator *series*, so they are evaluated
   together against one set of ``[symbols, bars]`` arrays and differ only in the
   scalar thresholds they compare against.
2. **Entry masks.** One ``[combos, symbols, bars]`` boolean per group, built by
   broadcasting per-combination thresholds against the shared series.
3. **Bounded forward scan.** Entries are sparse, so the mask collapses to a flat
   candidate list and the exit search runs as ``H`` array operations over that
   list - ``H`` being the maximum hold in bars. Every candidate's trade advances
   one bar per iteration, together, whatever combination or symbol it belongs to.
4. **Portfolio sweep.** Concurrency and sizing are genuinely path-dependent:
   whether a signal becomes a trade depends on what is already open. That sweep
   is sequential per combination, but it walks the (few thousand) surviving
   candidates rather than the (millions of) bars, which is why it is not the
   bottleneck.

Stage 3 is where the GPU earns its place: the candidate list is long, the state
update is pure elementwise arithmetic, and there is no data dependency between
candidates.
"""
from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from .arrays import Backend, Throttle, chunks, get_backend
from .frames import Frames
from .indicators import IndicatorCache, build_cache
from .params import (
    IND_ADX,
    IND_BBANDS,
    IND_EMA_CROSS,
    IND_MACD,
    IND_MOMENTUM,
    IND_RSI,
    IND_STOCHASTIC,
    IND_VOLUME_SPIKE,
    IND_VWAP,
    LEGACY_INDICATOR_MASK,
)
from .schema import CostModel

log = logging.getLogger(__name__)

# Exit reasons, kept as small integers so a trade log stays a numeric array.
EXIT_STOP = 0
EXIT_TRAIL = 1
EXIT_TARGET = 2
EXIT_MAXHOLD = 3
EXIT_INVALIDATED = 4
EXIT_WINDOW_END = 5

EXIT_NAMES = {
    EXIT_STOP: "hard stop hit",
    EXIT_TRAIL: "trailing stop hit",
    EXIT_TARGET: "take-profit target reached",
    EXIT_MAXHOLD: "max hold reached",
    EXIT_INVALIDATED: "signal invalidated",
    EXIT_WINDOW_END: "window ended",
}

# A hold longer than this many bars is not a swing trade, it is a buy-and-hold
# with extra steps; the forward scan refuses to allocate for it.
MAX_HORIZON_BARS = 2048

# Indicator periods that define a signature group. Two combinations agreeing on
# all of these read the identical series and can share one pass. Extended for
# the indicator-combination search (gap-closure item 3) with every new
# indicator's own period(s) - indicator_mask/indicator_min_agree/bb_std are
# per-combo scalars applied by broadcasting, same as volume_spike_multiple
# already is, so they do not need to be part of the signature.
SIGNATURE_KEYS = (
    "ema_fast", "ema_slow", "rsi_period", "atr_period",
    "volume_spike_lookback", "momentum_candles", "regime_lookback",
    "macd_fast", "macd_slow", "macd_signal",
    "bb_period", "stoch_k_period", "stoch_d_period", "adx_period", "vwap_period",
)


@dataclass(frozen=True, slots=True)
class PortfolioSettings:
    """Everything that is a risk control rather than a tunable.

    These are deliberately *not* in the search space. The optimizer may decide
    when to enter; it may not decide to risk more of the wallet than the spec's
    hard caps allow. There is deliberately no position-*count* ceiling here -
    how many positions are open at once falls out of the total-deployed cap,
    mirroring ``solbot.risk.size_position`` exactly.
    """

    starting_balance: float = 1000.0
    max_total_deployed_pct: float = 0.90
    max_position_pct_of_wallet: float = 0.45
    max_position_pct_of_liquidity: float = 0.01
    min_position_usd: float = 10.0
    min_candles_required: int = 40

    volatility_target_atr_pct: float = 0.03
    volatility_size_floor: float = 0.35

    correlation_lookback: int = 60
    correlation_max: float = 0.80

    costs: CostModel = field(default_factory=CostModel)

    # 4.2 - the aggregate timeframes an entry must agree with, as multiples of
    # the base bar. Empty disables the check.
    confluence_multiples: tuple[int, ...] = (3, 6)

    # 4.4 - portfolio sizing. "flat" reproduces the shipped behaviour;
    # "portfolio" solves for the weight that holds portfolio volatility at
    # `portfolio_vol_target`, and shrinks that target when the Monte Carlo
    # 5th-percentile drawdown says the strategy is riskier than it looks.
    sizing_mode: str = "flat"
    # The budget for the book as a whole: two uncorrelated positions at the 45%
    # cap and the 3% target ATR. A single position stays governed by the flat
    # volatility scalar, so the two modes diverge only once a second, correlated
    # position is open.
    portfolio_vol_target: float = 0.019
    drawdown_tolerance: float = 0.25
    drawdown_p5: float = 0.0

    @property
    def fee_pct(self) -> float:
        return self.costs.fee_pct

    @property
    def slippage_pct(self) -> float:
        return self.costs.slippage_pct

    def as_dict(self) -> dict[str, Any]:
        return {
            "starting_balance": self.starting_balance,
            "max_total_deployed_pct": self.max_total_deployed_pct,
            "max_position_pct_of_wallet": self.max_position_pct_of_wallet,
            "max_position_pct_of_liquidity": self.max_position_pct_of_liquidity,
            "min_position_usd": self.min_position_usd,
            "min_candles_required": self.min_candles_required,
            "volatility_target_atr_pct": self.volatility_target_atr_pct,
            "volatility_size_floor": self.volatility_size_floor,
            "correlation_lookback": self.correlation_lookback,
            "correlation_max": self.correlation_max,
            "confluence_multiples": list(self.confluence_multiples),
            "sizing_mode": self.sizing_mode,
            "portfolio_vol_target": self.portfolio_vol_target,
            "drawdown_tolerance": self.drawdown_tolerance,
            "drawdown_p5": self.drawdown_p5,
            "costs": self.costs.as_dict(),
        }


# --------------------------------------------------------------------------
# Trade log
# --------------------------------------------------------------------------
TRADE_DTYPE = np.dtype(
    [
        ("combo", np.int32),
        ("symbol", np.int32),
        ("entry_ts", np.int64),
        ("exit_ts", np.int64),
        ("entry_price", np.float64),
        ("exit_price", np.float64),
        ("size_usd", np.float64),
        ("qty", np.float64),
        ("pnl_usd", np.float64),
        ("fees_usd", np.float64),
        ("r_multiple", np.float32),
        ("exit_reason", np.int8),
        ("bars_held", np.int32),
    ]
)


def empty_trades() -> np.ndarray:
    return np.zeros(0, dtype=TRADE_DTYPE)


def summarise(trades: np.ndarray, starting_balance: float) -> dict[str, Any]:
    """Metrics for one combination's trade list.

    The equity curve is rebuilt from realised PnL in exit order, so drawdown is
    measured on closed trades. Marking open positions to market would produce a
    deeper, more flattering-to-quote number that no exit ever realised.
    """
    n = int(trades.shape[0])
    if n == 0:
        return {
            "trades": 0, "wins": 0, "losses": 0, "win_rate": 0.0,
            "profit_factor": 0.0, "total_pnl": 0.0, "total_fees": 0.0,
            "total_return": 0.0, "max_drawdown": 0.0, "expectancy": 0.0,
            "avg_r": 0.0, "ending_balance": starting_balance,
        }

    order = np.argsort(trades["exit_ts"], kind="stable")
    pnl = trades["pnl_usd"][order]
    wins = pnl > 0
    gross_win = float(pnl[wins].sum())
    gross_loss = float(-pnl[~wins].sum())

    equity = starting_balance + np.cumsum(pnl)
    peak = np.maximum.accumulate(np.concatenate(([starting_balance], equity)))
    drawdown = np.where(peak > 0, (peak - np.concatenate(([starting_balance], equity))) / peak, 0.0)

    return {
        "trades": n,
        "wins": int(wins.sum()),
        "losses": int((~wins).sum()),
        "win_rate": float(wins.mean()),
        "profit_factor": (gross_win / gross_loss) if gross_loss > 0 else (
            float("inf") if gross_win > 0 else 0.0
        ),
        "total_pnl": float(pnl.sum()),
        "total_fees": float(trades["fees_usd"].sum()),
        "total_return": float(pnl.sum() / starting_balance) if starting_balance else 0.0,
        "max_drawdown": float(drawdown.max()),
        "expectancy": float(pnl.mean()),
        "avg_r": float(np.nanmean(trades["r_multiple"])) if n else 0.0,
        "ending_balance": float(equity[-1]),
    }


@dataclass
class EngineResult:
    """Per-combination trades and metrics from one engine pass."""

    combos: list[dict[str, Any]]
    trades: np.ndarray = field(default_factory=empty_trades)
    metrics: list[dict[str, Any]] = field(default_factory=list)
    candidates_considered: int = 0
    bars_evaluated: int = 0
    symbols: list[str] = field(default_factory=list)
    seconds_elapsed: float = 0.0

    def trades_for(self, combo_index: int) -> np.ndarray:
        return self.trades[self.trades["combo"] == combo_index]

    def best(self, key: str = "total_return") -> int:
        if not self.metrics:
            return -1
        values = [m.get(key, 0.0) for m in self.metrics]
        return int(np.argmax(values))


# --------------------------------------------------------------------------
# Correlation, mirroring solbot.risk.correlation
# --------------------------------------------------------------------------
def pearson_returns(a: np.ndarray, b: np.ndarray) -> float | None:
    """Pearson correlation of two close series' returns, or None if unusable."""
    n = min(len(a), len(b))
    if n < 10:
        return None
    x = np.asarray(a[-n:], dtype=np.float64)
    y = np.asarray(b[-n:], dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        rx = np.diff(x) / np.where(x[:-1] == 0, np.nan, x[:-1])
        ry = np.diff(y) / np.where(y[:-1] == 0, np.nan, y[:-1])
    keep = ~(np.isnan(rx) | np.isnan(ry))
    rx, ry = rx[keep], ry[keep]
    if len(rx) < 8 or rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


class CloseHistory:
    """Trailing closes per symbol, addressed by wall-clock time."""

    __slots__ = ("_ts", "_close", "_counts")

    def __init__(self, frames: Frames) -> None:
        self._ts = frames.ts
        self._close = frames.close
        self._counts = frames.counts

    def tail(self, symbol: int, ts: int, lookback: int) -> np.ndarray:
        """The last ``lookback`` closes this symbol printed at or before ``ts``."""
        count = int(self._counts[symbol])
        if count == 0:
            return np.zeros(0, dtype=np.float32)
        end = int(np.searchsorted(self._ts[symbol, :count], int(ts), side="right"))
        start = max(0, end - int(lookback))
        return self._close[symbol, start:end]


# --------------------------------------------------------------------------
# Sizing (spec 4.4)
# --------------------------------------------------------------------------
def volatility_scalar(atr_pct: float, target: float, floor: float) -> float:
    if atr_pct <= 0 or target <= 0:
        return 1.0
    return float(max(floor, min(1.0, target / atr_pct)))


def portfolio_weight(
    *,
    candidate_vol: float,
    open_weights: Sequence[float],
    open_vols: Sequence[float],
    correlations: Sequence[float],
    target_vol: float,
    max_weight: float,
) -> float:
    """Largest weight for a new position that keeps portfolio volatility at target.

    For a portfolio already holding weights ``w_i`` with volatilities ``s_i``,
    adding weight ``w`` at volatility ``s`` gives

        sigma_p^2 = existing + 2*w*s*sum(rho_i*w_i*s_i) + w^2*s^2

    which is a quadratic in ``w``. Solving it and clamping to the hard cap is
    what turns "scale by correlation and volatility" into an actual number: a
    candidate that moves with what is already open gets a smaller slice, because
    its marginal contribution to portfolio risk is larger, not because a
    heuristic said so.
    """
    if candidate_vol <= 0 or target_vol <= 0:
        return max_weight

    existing_var = 0.0
    for i, w_i in enumerate(open_weights):
        s_i = open_vols[i]
        existing_var += (w_i * s_i) ** 2
        for j in range(i + 1, len(open_weights)):
            existing_var += 2.0 * w_i * open_weights[j] * s_i * open_vols[j]
    cross = sum(
        correlations[i] * open_weights[i] * open_vols[i] for i in range(len(open_weights))
    )

    a = candidate_vol**2
    b = 2.0 * candidate_vol * cross
    c = existing_var - target_vol**2
    if c >= 0:
        # The book is already at or over budget; nothing new fits.
        return 0.0
    disc = b * b - 4.0 * a * c
    if disc <= 0:
        return 0.0
    root = (-b + math.sqrt(disc)) / (2.0 * a)
    return float(max(0.0, min(max_weight, root)))


def drawdown_scalar(p5_drawdown: float, tolerance: float, floor: float = 0.25) -> float:
    """Shrink the risk budget when the Monte Carlo tail is worse than tolerated.

    The historical backtest drawdown is one draw from a distribution; the 5th
    percentile of the resampled distribution is the one worth sizing against.
    Below tolerance this returns 1.0 and changes nothing.
    """
    if p5_drawdown <= 0 or tolerance <= 0 or p5_drawdown <= tolerance:
        return 1.0
    return float(max(floor, tolerance / p5_drawdown))


# A combo dict predating the indicator-combination search (gap-closure item
# 3) - an older stored bundle, a hand-built test fixture - simply lacks these
# keys. Reading them with these defaults reproduces the legacy hard-AND of
# volume spike + momentum + RSI + EMA cross exactly, the same convention
# solbot.config's DEFAULTS and cache_requirements() already use.
INDICATOR_SEARCH_DEFAULTS: dict[str, Any] = {
    "indicator_mask": LEGACY_INDICATOR_MASK,
    "indicator_min_agree": 4,
    "macd_fast": 12, "macd_slow": 26, "macd_signal": 9,
    "bb_period": 20, "bb_std": 2.0, "bb_bullish_pct": 0.5,
    "stoch_k_period": 14, "stoch_d_period": 3, "stoch_overbought": 80.0,
    "adx_period": 14, "adx_min": 20.0,
    "vwap_period": 20,
}


# --------------------------------------------------------------------------
# Combination grouping
# --------------------------------------------------------------------------
@dataclass(slots=True)
class SignatureGroup:
    signature: tuple
    indices: list[int]
    combos: list[dict[str, Any]]

    def column(self, key: str, dtype: Any = np.float32) -> np.ndarray:
        default = INDICATOR_SEARCH_DEFAULTS.get(key)
        if default is None:
            return np.asarray([c[key] for c in self.combos], dtype=dtype)
        return np.asarray([c.get(key, default) for c in self.combos], dtype=dtype)


def _signature_value(combo: dict[str, Any], key: str) -> int:
    if key in combo:
        return int(combo[key])
    return int(INDICATOR_SEARCH_DEFAULTS[key])


def group_by_signature(combos: Sequence[dict[str, Any]]) -> list[SignatureGroup]:
    buckets: dict[tuple, list[int]] = defaultdict(list)
    for i, combo in enumerate(combos):
        buckets[tuple(_signature_value(combo, k) for k in SIGNATURE_KEYS)].append(i)
    return [
        SignatureGroup(sig, idx, [combos[i] for i in idx])
        for sig, idx in buckets.items()
    ]


def cache_requirements(
    combos: Sequence[dict[str, Any]], settings: PortfolioSettings
) -> dict[str, list[Any]]:
    """Indicator series the whole batch needs, including the confluence pairs.

    A combo missing one of the indicator-combination-search keys (gap-closure
    item 3) is read with the legacy default for it - the same convention
    solbot.config's DEFAULTS use, so an older combo dict (a test, a stored
    bundle from before this search space existed) still resolves to something
    the cache can serve rather than a KeyError.
    """
    req: dict[str, set] = {
        "ema": set(), "rsi": set(), "atr": set(), "vol_ratio": set(),
        "momentum": set(), "efficiency": set(),
        "macd_line": set(), "macd_signal": set(),
        "bbands": set(), "stoch_k": set(), "stoch_d": set(), "adx": set(), "vwap": set(),
    }
    mtf: set[tuple[int, int, int]] = set()
    for combo in combos:
        req["ema"].update((int(combo["ema_fast"]), int(combo["ema_slow"])))
        req["rsi"].add(int(combo["rsi_period"]))
        req["atr"].add(int(combo["atr_period"]))
        req["vol_ratio"].add(int(combo["volume_spike_lookback"]))
        req["momentum"].add(int(combo["momentum_candles"]))
        req["efficiency"].add(int(combo["regime_lookback"]))
        for multiple in settings.confluence_multiples:
            mtf.add((int(multiple), int(combo["ema_fast"]), int(combo["ema_slow"])))

        d = INDICATOR_SEARCH_DEFAULTS
        mask = int(combo.get("indicator_mask", d["indicator_mask"]))
        macd_fast = int(combo.get("macd_fast", d["macd_fast"]))
        macd_slow = int(combo.get("macd_slow", d["macd_slow"]))
        macd_signal = int(combo.get("macd_signal", d["macd_signal"]))
        if mask & IND_MACD:
            req["macd_line"].add((macd_fast, macd_slow))
            req["macd_signal"].add((macd_fast, macd_slow, macd_signal))
        if mask & IND_BBANDS:
            req["bbands"].add(int(combo.get("bb_period", d["bb_period"])))
        if mask & IND_STOCHASTIC:
            k_period = int(combo.get("stoch_k_period", d["stoch_k_period"]))
            req["stoch_k"].add(k_period)
            req["stoch_d"].add((k_period, int(combo.get("stoch_d_period", d["stoch_d_period"]))))
        if mask & IND_ADX:
            req["adx"].add(int(combo.get("adx_period", d["adx_period"])))
        if mask & IND_VWAP:
            req["vwap"].add(int(combo.get("vwap_period", d["vwap_period"])))

    out: dict[str, list[Any]] = {k: sorted(v) for k, v in req.items()}
    out["mtf"] = sorted(mtf)
    return out


# --------------------------------------------------------------------------
# The engine
# --------------------------------------------------------------------------
class VectorEngine:
    """Batched backtests across symbols and parameter combinations."""

    def __init__(
        self,
        backend: Backend | None = None,
        throttle: Throttle | None = None,
        *,
        memory_budget_mb: int = 512,
    ) -> None:
        self.backend = backend or get_backend()
        self.throttle = throttle or Throttle(100.0, self.backend)
        self.memory_budget = int(memory_budget_mb) * 1024 * 1024

    # ------------------------------------------------------------------
    def run(
        self,
        frames: Frames,
        combos: Sequence[dict[str, Any]],
        settings: PortfolioSettings,
        *,
        cache: IndicatorCache | None = None,
    ) -> EngineResult:
        import time as _time

        started = _time.monotonic()
        combos = list(combos)
        result = EngineResult(combos=combos, symbols=list(frames.symbols))
        if not combos or frames.n_symbols == 0 or frames.n_bars == 0:
            result.metrics = [summarise(empty_trades(), settings.starting_balance) for _ in combos]
            return result

        cache = cache or build_cache(frames, cache_requirements(combos, settings))
        history = CloseHistory(frames)
        result.bars_evaluated = int(frames.mask.sum())

        collected: list[np.ndarray] = []
        per_combo: dict[int, list[np.ndarray]] = defaultdict(list)

        for group in group_by_signature(combos):
            with self.throttle.segment():
                candidates = self._candidates(frames, cache, group, settings)
            result.candidates_considered += int(candidates["combo"].shape[0])
            if candidates["combo"].shape[0] == 0:
                continue
            with self.throttle.segment():
                resolved = self._resolve_exits(frames, cache, group, settings, candidates)
            for combo_index in group.indices:
                subset = _select(resolved, resolved["combo"] == combo_index)
                if subset["combo"].shape[0]:
                    per_combo[combo_index].append(subset)

        for combo_index, chunks_ in per_combo.items():
            merged = {
                key: np.concatenate([c[key] for c in chunks_])
                for key in chunks_[0]
            }
            trades = self._sweep(
                frames, history, combos[combo_index], settings, merged, combo_index
            )
            if trades.shape[0]:
                collected.append(trades)

        result.trades = (
            np.concatenate(collected) if collected else empty_trades()
        )
        result.metrics = [
            summarise(result.trades_for(i), settings.starting_balance)
            for i in range(len(combos))
        ]
        result.seconds_elapsed = _time.monotonic() - started
        return result

    # ------------------------------------------------------------------
    # Stage 2: entry masks -> candidate list
    # ------------------------------------------------------------------
    def _candidates(
        self,
        frames: Frames,
        cache: IndicatorCache,
        group: SignatureGroup,
        settings: PortfolioSettings,
    ) -> dict[str, np.ndarray]:
        (
            ema_fast, ema_slow, rsi_period, atr_period, vol_lookback, mom_bars, er_bars,
            macd_fast, macd_slow, macd_signal_span,
            bb_period, stoch_k_period, stoch_d_period, adx_period, vwap_period,
        ) = group.signature
        ef = cache.get(("ema", ema_fast))
        es = cache.get(("ema", ema_slow))
        rsi = cache.get(("rsi", rsi_period))
        atr = cache.get(("atr", atr_period))
        atr_pct = cache.get(("atr_pct", atr_period))
        vol_ratio = cache.get(("vol_ratio", vol_lookback))
        momentum = cache.get(("momentum", mom_bars))
        consec = cache.get(("consec_up", mom_bars))
        efficiency = cache.get(("efficiency", er_bars))

        # Structural conditions every entry needs regardless of which
        # indicators are active - ATR sizes the stop, so a signal with no ATR
        # reading cannot be traded no matter what the vote gate decides.
        base = (
            frames.tradeable
            & frames.warmup_split(settings.min_candles_required)
            & (np.nan_to_num(atr, nan=0.0) > 0)
        )
        if settings.confluence_multiples:
            agree = np.zeros(frames.close.shape, dtype=np.int8)
            for multiple in settings.confluence_multiples:
                agree += cache.get(("mtf", (int(multiple), ema_fast, ema_slow))).astype(np.int8)
        else:
            agree = None

        n_combos = len(group.indices)
        spike = group.column("volume_spike_multiple")
        mom_min = group.column("momentum_min_pct")
        rsi_max = group.column("rsi_max_entry")
        trend_er = group.column("regime_trend_er")
        chop_atr = group.column("regime_chop_atr_pct")
        allowed = group.column("regime_allowed", np.int32)
        need_agree = group.column("confluence_required", np.int8)

        # --- indicator-combination search (gap-closure item 3) ------------
        # Fetch each new indicator's shared (group-signature-level) series
        # only when some combo in this group actually votes on it - the cache
        # was only ever populated for the (period) combinations that at least
        # one combo's mask requested (see cache_requirements).
        ind_mask = group.column("indicator_mask", np.int32)
        min_agree = group.column("indicator_min_agree", np.int8)
        wants = lambda bit: bool(int((ind_mask & bit).any()))  # noqa: E731

        macd_line = macd_sig = None
        if wants(IND_MACD):
            macd_line = cache.get(("macd_line", (macd_fast, macd_slow)))
            macd_sig = cache.get(("macd_signal", (macd_fast, macd_slow, macd_signal_span)))

        bb_mid = bb_std_series = None
        if wants(IND_BBANDS):
            bb_mid = cache.get(("bb_mid", bb_period))
            bb_std_series = cache.get(("bb_std", bb_period))

        stoch_k = stoch_d = None
        if wants(IND_STOCHASTIC):
            stoch_k = cache.get(("stoch_k", stoch_k_period))
            stoch_d = cache.get(("stoch_d", (stoch_k_period, stoch_d_period)))

        adx_line = plus_di = minus_di = None
        if wants(IND_ADX):
            adx_line = cache.get(("adx", adx_period))
            plus_di = cache.get(("plus_di", adx_period))
            minus_di = cache.get(("minus_di", adx_period))

        vwap = None
        if wants(IND_VWAP):
            vwap = cache.get(("vwap", vwap_period))

        bb_std_mult = group.column("bb_std")
        bb_bullish_pct = group.column("bb_bullish_pct")
        stoch_overbought = group.column("stoch_overbought")
        adx_min = group.column("adx_min")

        # Chunk the combination axis so the boolean block stays inside budget.
        # More vote arrays than the legacy four now, so a more generous
        # per-combo estimate than the old "6".
        per_combo_bytes = max(1, frames.n_symbols * frames.n_bars * 12)
        step = max(1, min(n_combos, self.memory_budget // per_combo_bytes))

        parts: list[dict[str, np.ndarray]] = []
        for lo, hi in chunks(n_combos, step):
            sl = slice(lo, hi)
            width = hi - lo

            def active(bit: int) -> np.ndarray:
                return ((ind_mask[sl] & bit) != 0)[:, None, None]

            agree_count = np.zeros((width, frames.n_symbols, frames.n_bars), dtype=np.int8)
            agree_count += (
                active(IND_VOLUME_SPIKE) & (vol_ratio[None] >= spike[sl][:, None, None])
            ).astype(np.int8)
            agree_count += (
                active(IND_MOMENTUM)
                & (momentum[None] >= mom_min[sl][:, None, None])
                & consec[None]
            ).astype(np.int8)
            agree_count += (active(IND_RSI) & (rsi[None] <= rsi_max[sl][:, None, None])).astype(
                np.int8
            )
            agree_count += (active(IND_EMA_CROSS) & (ef > es)[None]).astype(np.int8)
            if macd_line is not None:
                agree_count += (active(IND_MACD) & (macd_line > macd_sig)[None]).astype(np.int8)
            if bb_mid is not None:
                with np.errstate(divide="ignore", invalid="ignore"):
                    upper = bb_mid[None] + bb_std_mult[sl][:, None, None] * bb_std_series[None]
                    lower = bb_mid[None] - bb_std_mult[sl][:, None, None] * bb_std_series[None]
                    span = upper - lower
                    percent_b = np.where(span == 0.0, np.nan, (frames.close[None] - lower) / span)
                bb_vote = percent_b > bb_bullish_pct[sl][:, None, None]
                agree_count += (active(IND_BBANDS) & np.nan_to_num(bb_vote, nan=False)).astype(
                    np.int8
                )
            if stoch_k is not None:
                stoch_vote = (stoch_k > stoch_d)[None] & (
                    stoch_k[None] < stoch_overbought[sl][:, None, None]
                )
                agree_count += (active(IND_STOCHASTIC) & np.nan_to_num(stoch_vote, nan=False)).astype(
                    np.int8
                )
            if adx_line is not None:
                adx_vote = (adx_line[None] >= adx_min[sl][:, None, None]) & (
                    plus_di > minus_di
                )[None]
                agree_count += (active(IND_ADX) & np.nan_to_num(adx_vote, nan=False)).astype(
                    np.int8
                )
            if vwap is not None:
                vwap_vote = (frames.close[None] > vwap[None])
                agree_count += (active(IND_VWAP) & np.nan_to_num(vwap_vote, nan=False)).astype(
                    np.int8
                )

            mask = base[None, :, :] & (agree_count >= min_agree[sl][:, None, None])
            # 4.1 - regime gate. NaN efficiency means not enough history to
            # judge, which classifies as chop so the gate errs toward silence.
            trending = efficiency[None] >= trend_er[sl][:, None, None]
            volatile = atr_pct[None] > chop_atr[sl][:, None, None]
            regime = np.where(trending, 0, np.where(volatile, 2, 1)).astype(np.int8)
            regime = np.where(np.isnan(efficiency)[None], np.int8(2), regime)
            mask &= ((allowed[sl][:, None, None] >> regime) & 1).astype(bool)
            # 4.2 - multi-timeframe confluence.
            if agree is not None:
                mask &= agree[None] >= need_agree[sl][:, None, None]

            local, symbol, bar = np.nonzero(mask)
            if local.size == 0:
                continue
            parts.append(
                {
                    "combo": np.asarray(
                        [group.indices[lo + i] for i in local], dtype=np.int32
                    ),
                    "local": (local + lo).astype(np.int32),
                    "symbol": symbol.astype(np.int32),
                    "bar": bar.astype(np.int32),
                }
            )

        if not parts:
            return {k: np.zeros(0, dtype=np.int32) for k in ("combo", "local", "symbol", "bar")}
        return {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}

    # ------------------------------------------------------------------
    # Stage 3: bounded forward scan
    # ------------------------------------------------------------------
    def _resolve_exits(
        self,
        frames: Frames,
        cache: IndicatorCache,
        group: SignatureGroup,
        settings: PortfolioSettings,
        candidates: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Walk every candidate trade forward one bar at a time, together."""
        ema_fast, ema_slow, rsi_period, atr_period, vol_lookback, mom_bars = group.signature[:6]
        ef = cache.get(("ema", ema_fast))
        es = cache.get(("ema", ema_slow))
        atr = np.nan_to_num(cache.get(("atr", atr_period)), nan=0.0)
        atr_pct = np.nan_to_num(cache.get(("atr_pct", atr_period)), nan=0.0)
        vol_ratio = cache.get(("vol_ratio", vol_lookback))
        momentum = cache.get(("momentum", mom_bars))

        sym = candidates["symbol"]
        bar = candidates["bar"]
        local = candidates["local"]

        stop_mult = group.column("stop_atr_mult")[local]
        rr_min = group.column("rr_min")[local]
        rr_max = group.column("rr_max")[local]
        spike = group.column("volume_spike_multiple")[local]
        mom_min = group.column("momentum_min_pct")[local]
        activate_r = group.column("trailing_activate_r")[local]
        trail_atr = group.column("trailing_distance_atr")[local]
        inval_bars = group.column("signal_invalidation_bars", np.int32)[local]
        max_hold_s = group.column("max_hold_minutes", np.int64)[local] * 60

        price = frames.close[sym, bar].astype(np.float64)
        entry_ts = frames.ts[sym, bar]
        atr_entry = np.maximum(atr[sym, bar].astype(np.float64), price * 0.002)
        stop_distance = atr_entry * stop_mult

        # Signal strength blends volume conviction with momentum, each capped so
        # one runaway input cannot dominate - identical to the live _levels().
        vol_component = np.clip((vol_ratio[sym, bar] - spike) / np.maximum(spike, 1e-9), 0.0, 1.0)
        mom_component = np.clip(
            momentum[sym, bar] / np.maximum(mom_min * 3.0, 1e-9), 0.0, 1.0
        )
        strength = np.clip(0.6 * vol_component + 0.4 * mom_component, 0.0, 1.0)
        rr = np.clip(rr_min + (rr_max - rr_min) * strength, rr_min, rr_max)

        slip = settings.slippage_pct / 2.0 / 100.0     # expected, not worst case
        entry_price = price * (1.0 + slip)
        # The stop and target re-anchor on the filled price, not the quoted one;
        # otherwise slippage silently shrinks the real R multiple.
        risk_per_unit = stop_distance
        hard_stop = entry_price - risk_per_unit
        take_profit = entry_price + risk_per_unit * rr
        cost_pct = (settings.fee_pct * 2.0 + settings.slippage_pct) / 100.0
        breakeven = entry_price * (1.0 + cost_pct)

        horizon = int(
            min(
                MAX_HORIZON_BARS,
                max(1, math.ceil(float(max_hold_s.max()) / max(1, frames.seconds)) + 2),
            )
        )

        n = sym.shape[0]
        counts = frames.counts[sym].astype(np.int64)
        alive = np.ones(n, dtype=bool)
        trailing = np.zeros(n, dtype=np.float64)
        armed = np.zeros(n, dtype=bool)
        high_water = entry_price.copy()
        invalidations = np.zeros(n, dtype=np.int32)
        exit_bar = np.full(n, -1, dtype=np.int32)
        exit_price = np.zeros(n, dtype=np.float64)
        exit_reason = np.full(n, EXIT_WINDOW_END, dtype=np.int8)

        for k in range(1, horizon + 1):
            if not alive.any():
                break
            here = (bar + k).astype(np.int64)
            in_range = here < counts
            # Running out of data closes the trade where it last traded.
            ended = alive & ~in_range
            if ended.any():
                last = np.maximum(counts[ended] - 1, 0).astype(np.int64)
                exit_bar[ended] = last
                exit_price[ended] = frames.close[sym[ended], last]
                exit_reason[ended] = EXIT_WINDOW_END
                alive &= in_range

            if not alive.any():
                break
            idx = np.flatnonzero(alive)
            s_i, b_i = sym[idx], here[idx]
            bar_low = frames.low[s_i, b_i].astype(np.float64)

            # 1. Intrabar stop. If the bar's low pierced the stop, that is where
            #    the exit happened, not at the close.
            level = np.maximum(hard_stop[idx], trailing[idx])
            struck = bar_low <= level
            if struck.any():
                hit = idx[struck]
                exit_bar[hit] = b_i[struck]
                exit_price[hit] = level[struck]
                exit_reason[hit] = np.where(
                    armed[hit] & (trailing[hit] >= hard_stop[hit]), EXIT_TRAIL, EXIT_STOP
                )
                alive[hit] = False

            still = np.flatnonzero(alive)
            if still.size == 0:
                continue
            s_i, b_i = sym[still], here[still]
            close_i = frames.close[s_i, b_i].astype(np.float64)

            # 2. Take profit, evaluated on the close: a wick through the target
            #    is not a fill the live bot would have got.
            won = close_i >= take_profit[still]
            if won.any():
                hit = still[won]
                exit_bar[hit] = b_i[won]
                exit_price[hit] = close_i[won]
                exit_reason[hit] = EXIT_TARGET
                alive[hit] = False

            still = np.flatnonzero(alive)
            if still.size == 0:
                continue
            s_i, b_i = sym[still], here[still]
            close_i = frames.close[s_i, b_i].astype(np.float64)

            # 3. Max hold, measured in wall-clock seconds rather than bars so a
            #    data gap cannot extend a hold past its limit.
            timed_out = (frames.ts[s_i, b_i] - entry_ts[still]) >= max_hold_s[still]
            if timed_out.any():
                hit = still[timed_out]
                exit_bar[hit] = b_i[timed_out]
                exit_price[hit] = close_i[timed_out]
                exit_reason[hit] = EXIT_MAXHOLD
                alive[hit] = False

            still = np.flatnonzero(alive)
            if still.size == 0:
                continue
            s_i, b_i = sym[still], here[still]
            close_i = frames.close[s_i, b_i].astype(np.float64)

            # 4. Signal invalidation - two of three broken, for N bars running.
            broken = (
                (ef[s_i, b_i] < es[s_i, b_i]).astype(np.int8)
                + (np.nan_to_num(vol_ratio[s_i, b_i], nan=0.0) < 1.0).astype(np.int8)
                + (np.nan_to_num(momentum[s_i, b_i], nan=0.0) < 0.0).astype(np.int8)
            )
            faded = broken >= 2
            invalidations[still] = np.where(faded, invalidations[still] + 1, 0)
            done = faded & (invalidations[still] >= inval_bars[still])
            if done.any():
                hit = still[done]
                exit_bar[hit] = b_i[done]
                exit_price[hit] = close_i[done]
                exit_reason[hit] = EXIT_INVALIDATED
                alive[hit] = False

            still = np.flatnonzero(alive)
            if still.size == 0:
                continue
            s_i, b_i = sym[still], here[still]
            close_i = frames.close[s_i, b_i].astype(np.float64)

            # 5. Advance the trail. It only arms once the move covers round-trip
            #    costs, so a position is never stopped at a nominal breakeven
            #    that is really a loss after fees.
            high_water[still] = np.maximum(high_water[still], close_i)
            r_multiple = (close_i - entry_price[still]) / np.maximum(
                risk_per_unit[still], 1e-12
            )
            arm_now = (
                ~armed[still]
                & (r_multiple >= activate_r[still])
                & (close_i > breakeven[still])
            )
            if arm_now.any():
                hit = still[arm_now]
                armed[hit] = True
                trailing[hit] = breakeven[hit]

            live_trail = np.flatnonzero(alive & armed)
            if live_trail.size:
                atr_here = atr[sym[live_trail], (bar[live_trail] + k).astype(np.int64)]
                atr_here = np.where(
                    atr_here > 0, atr_here, frames.close[sym[live_trail], (bar[live_trail] + k)] * 0.01
                ).astype(np.float64)
                distance = np.maximum(
                    atr_here * trail_atr[live_trail], entry_price[live_trail] * 0.002
                )
                candidate = high_water[live_trail] - distance
                floor = np.maximum(breakeven[live_trail], trailing[live_trail])
                trailing[live_trail] = np.maximum(trailing[live_trail], np.maximum(candidate, floor))

        # Anything still open when the horizon ran out exits at the last bar it
        # reached; with the horizon derived from max hold this is rare.
        if alive.any():
            last = np.minimum(bar[alive] + horizon, counts[alive] - 1).astype(np.int64)
            exit_bar[alive] = last
            exit_price[alive] = frames.close[sym[alive], last]
            exit_reason[alive] = EXIT_WINDOW_END

        exit_ts = frames.ts[sym, exit_bar.astype(np.int64)]
        return {
            "combo": candidates["combo"],
            "symbol": sym,
            "bar": bar,
            "entry_ts": entry_ts,
            "exit_ts": exit_ts,
            "entry_price": entry_price,
            "exit_price": exit_price * (1.0 - slip),
            "raw_exit": exit_price,
            "exit_reason": exit_reason,
            "risk_per_unit": risk_per_unit,
            "atr_pct": atr_pct[sym, bar].astype(np.float64),
            "bars_held": (exit_bar - bar).astype(np.int32),
        }

    # ------------------------------------------------------------------
    # Stage 4: portfolio sweep
    # ------------------------------------------------------------------
    def _sweep(
        self,
        frames: Frames,
        history: CloseHistory,
        combo: dict[str, Any],
        settings: PortfolioSettings,
        resolved: dict[str, np.ndarray],
        combo_index: int,
    ) -> np.ndarray:
        """Turn signals into trades under the concurrency and sizing caps.

        Whether a signal becomes a trade depends on what is already open, so this
        is the one genuinely sequential step. It walks candidate signals in
        wall-clock order - a few thousand of them - rather than bars.
        """
        order = np.argsort(resolved["entry_ts"], kind="stable")
        n = order.shape[0]
        if n == 0:
            return empty_trades()

        entry_ts = resolved["entry_ts"][order]
        exit_ts = resolved["exit_ts"][order]
        symbols = resolved["symbol"][order]
        entry_price = resolved["entry_price"][order]
        exit_price = resolved["exit_price"][order]
        reasons = resolved["exit_reason"][order]
        risk = resolved["risk_per_unit"][order]
        atr_pct = resolved["atr_pct"][order]
        bars_held = resolved["bars_held"][order]
        entry_bar = resolved["bar"][order]

        fee_pct = settings.fee_pct / 100.0
        max_deployed_pct = float(settings.max_total_deployed_pct)
        wallet_cap = float(settings.max_position_pct_of_wallet)
        liq_cap = float(settings.max_position_pct_of_liquidity)
        lookback = int(settings.correlation_lookback)
        dd_scale = drawdown_scalar(settings.drawdown_p5, settings.drawdown_tolerance)
        target_vol = settings.portfolio_vol_target * dd_scale

        balance = float(settings.starting_balance)
        # Each open position: [exit_ts, size, symbol, atr_pct, proceeds]
        open_positions: list[dict[str, Any]] = []
        out: list[tuple] = []

        def realise(until_ts: int) -> None:
            nonlocal balance
            still_open = []
            for pos in open_positions:
                if pos["exit_ts"] <= until_ts:
                    balance += pos["proceeds"]
                    out.append(pos["record"])
                else:
                    still_open.append(pos)
            open_positions[:] = still_open

        for i in range(n):
            now = int(entry_ts[i])
            realise(now)

            symbol = int(symbols[i])
            if any(p["symbol"] == symbol for p in open_positions):
                continue

            deployed = sum(p["size"] for p in open_positions)
            wallet = balance + deployed
            if deployed >= wallet * max_deployed_pct:
                continue
            tradeable = wallet
            base_cap = tradeable * wallet_cap
            # Never deploy past the total-deployed cap, so the reserve the spec
            # requires actually stays untouched. There is no cap on how many
            # positions make up that total.
            remaining = max(0.0, tradeable * max_deployed_pct - deployed)
            size = min(base_cap, remaining)

            # A bundle without a liquidity surface cannot police pool depth. That
            # is a gap in the data, not a licence to size freely, so it is
            # reported in the run feed rather than silently ignored.
            liquidity = (
                float(frames.liquidity[symbol, entry_bar[i]])
                if frames.liquidity is not None
                else 0.0
            )
            if liquidity > 0:
                size = min(size, liquidity * liq_cap)

            candidate_vol = float(atr_pct[i])

            # Correlations are computed once and used twice: the hard gate
            # refuses a position that is merely riding an open one, and the
            # portfolio budget shrinks one that is partly riding it. Mirrors
            # solbot.engine._exposure exactly.
            correlations: list[float] = []
            if open_positions:
                candidate_closes = history.tail(symbol, now, lookback)
                for pos in open_positions:
                    rho = pearson_returns(
                        candidate_closes, history.tail(pos["symbol"], now, lookback)
                    )
                    # An uncomputable correlation counts as fully correlated:
                    # sizing down on what we cannot measure is the safe error.
                    correlations.append(1.0 if rho is None else float(rho))
                if max(abs(c) for c in correlations) >= settings.correlation_max:
                    continue

            # The token's own volatility scales the size in both modes;
            # portfolio mode adds a second cap on top and never lifts this one.
            size *= volatility_scalar(
                candidate_vol,
                settings.volatility_target_atr_pct,
                settings.volatility_size_floor,
            )
            if settings.sizing_mode == "portfolio" and target_vol > 0:
                weight = portfolio_weight(
                    candidate_vol=candidate_vol,
                    open_weights=(
                        [p["size"] / wallet for p in open_positions] if wallet > 0 else []
                    ),
                    open_vols=[p["atr_pct"] for p in open_positions],
                    correlations=correlations,
                    target_vol=target_vol,
                    max_weight=wallet_cap,
                )
                size = min(size, tradeable * weight)

            if size < settings.min_position_usd or size > balance:
                continue

            fill = float(entry_price[i])
            entry_fee = size * fee_pct
            qty = (size - entry_fee) / fill if fill > 0 else 0.0
            gross = qty * float(exit_price[i])
            exit_fee = gross * fee_pct
            proceeds = gross - exit_fee
            pnl = proceeds - size
            r_multiple = (
                (float(exit_price[i]) - fill) / float(risk[i]) if risk[i] > 0 else 0.0
            )

            balance -= size
            open_positions.append(
                {
                    "exit_ts": int(exit_ts[i]),
                    "size": size,
                    "symbol": symbol,
                    "atr_pct": candidate_vol,
                    "proceeds": proceeds,
                    "record": (
                        combo_index, symbol, now, int(exit_ts[i]), fill,
                        float(exit_price[i]), size, qty, pnl,
                        entry_fee + exit_fee, r_multiple, int(reasons[i]),
                        int(bars_held[i]),
                    ),
                }
            )

        realise(2**62)
        if not out:
            return empty_trades()
        return np.asarray(out, dtype=TRADE_DTYPE)


def _select(data: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    return {key: value[mask] for key, value in data.items()}


def trades_to_records(
    trades: np.ndarray, symbols: Sequence[str]
) -> list[dict[str, Any]]:
    """Trade rows as dicts, for JSON hand-off and the Monte Carlo module."""
    out = []
    for row in trades:
        out.append(
            {
                "combo": int(row["combo"]),
                "symbol": symbols[int(row["symbol"])] if symbols else int(row["symbol"]),
                "entry_ts": int(row["entry_ts"]),
                "exit_ts": int(row["exit_ts"]),
                "entry_price": float(row["entry_price"]),
                "exit_price": float(row["exit_price"]),
                "size_usd": float(row["size_usd"]),
                "pnl_usd": float(row["pnl_usd"]),
                "fees_usd": float(row["fees_usd"]),
                "r_multiple": float(row["r_multiple"]),
                "exit_reason": EXIT_NAMES.get(int(row["exit_reason"]), "unknown"),
                "bars_held": int(row["bars_held"]),
            }
        )
    return out
