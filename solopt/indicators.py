"""Batched indicator math over ``[symbols, bars]`` blocks.

Two rules shape everything here.

**Parity with production.** Every function reproduces the pandas implementation
in ``solbot.indicators`` bar for bar, including its warm-up conventions - the
Wilder smoothing that only becomes valid after ``period`` observations, the
volume average that excludes the current bar, the RSI that reads 100 rather than
NaN before it has enough history. ``tests/test_vector_parity.py`` asserts this
against the live code on random data. If the two ever diverge the optimizer stops
being evidence about the rules that actually trade.

**Compute once per distinct value, not once per combination.** A grid with 200
parameter combinations rarely has 200 distinct EMA spans; it has four or five.
So each family is computed as a single stacked pass over the distinct values it
uses - one loop over bars for all of them at once - and combinations then index
into the result. The recursive filters (EMA, Wilder) are the only place a bar
loop survives, and it is a loop over *bars* with every symbol and every distinct
period advancing together inside it, which is the vectorization that matters.

Recursive filters stay on the host even when a GPU is present: their inner step
is tiny and serial, so per-kernel launch latency would dominate. The GPU earns
its keep on the next stage, where the combination axis makes the arrays large.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

from .arrays import Backend
from .frames import Frames

NAN = np.float32("nan")


# --------------------------------------------------------------------------
# Recursive primitives
# --------------------------------------------------------------------------
def ewm_stack(
    x: np.ndarray,
    alphas: Sequence[float],
    *,
    start: int = 0,
    min_index: Sequence[int] | None = None,
) -> np.ndarray:
    """Exponentially weighted mean for several alphas at once.

    Mirrors ``Series.ewm(alpha=..., adjust=False)``: the recursion is seeded with
    the first observation rather than with zero, so early bars are not dragged
    toward an arbitrary origin.

    ``start`` is the index carrying the first observation (1 for series derived
    from ``diff``, which has no value at index 0). ``min_index[p]`` is the first
    index whose output is defined for alpha ``p``, standing in for pandas'
    ``min_periods``; earlier bars come back NaN.
    """
    x = np.asarray(x, dtype=np.float32)
    n_symbols, n_bars = x.shape
    a = np.asarray(alphas, dtype=np.float32).reshape(-1, 1)
    n_alpha = a.shape[0]

    out = np.full((n_alpha, n_symbols, n_bars), NAN, dtype=np.float32)
    if n_bars <= start:
        return out

    prev = np.repeat(x[None, :, start], n_alpha, axis=0).astype(np.float32)
    out[:, :, start] = prev
    one_minus = (1.0 - a).astype(np.float32)
    for t in range(start + 1, n_bars):
        prev = a * x[None, :, t] + one_minus * prev
        out[:, :, t] = prev

    if min_index is not None:
        idx = np.arange(n_bars, dtype=np.int32)[None, None, :]
        out = np.where(idx < np.asarray(min_index, np.int32)[:, None, None], NAN, out)
    return out


def ema_stack(close: np.ndarray, spans: Sequence[int]) -> np.ndarray:
    """``[len(spans), symbols, bars]`` EMAs. No warm-up mask - pandas has none."""
    alphas = [2.0 / (max(1, int(s)) + 1.0) for s in spans]
    return ewm_stack(close, alphas, start=0)


def rsi_stack(close: np.ndarray, periods: Sequence[int]) -> np.ndarray:
    """Wilder RSI, including production's "no losses yet reads 100" convention."""
    close = np.asarray(close, dtype=np.float32)
    n_symbols, n_bars = close.shape
    periods = [max(2, int(p)) for p in periods]

    delta = np.zeros_like(close)
    if n_bars > 1:
        delta[:, 1:] = close[:, 1:] - close[:, :-1]
    gain = np.clip(delta, 0.0, None)
    loss = np.clip(-delta, 0.0, None)

    alphas = [1.0 / p for p in periods]
    # Index 0 holds no observation (diff is undefined there), so the recursion
    # starts at 1 and the p-th observation lands on index p.
    mins = [p for p in periods]
    avg_gain = ewm_stack(gain, alphas, start=1, min_index=mins)
    avg_loss = ewm_stack(loss, alphas, start=1, min_index=mins)

    with np.errstate(divide="ignore", invalid="ignore"):
        rs = avg_gain / np.where(avg_loss == 0.0, NAN, avg_loss)
        rsi = 100.0 - (100.0 / (1.0 + rs))
    defined = ~np.isnan(avg_loss) & (avg_loss != 0.0)
    return np.where(defined, np.nan_to_num(rsi, nan=50.0), 100.0).astype(np.float32)


def true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """Bar 0 has no previous close, so it falls back to the bar's own range."""
    tr = (high - low).astype(np.float32)
    if high.shape[1] > 1:
        prev = close[:, :-1]
        tr[:, 1:] = np.maximum(
            high[:, 1:] - low[:, 1:],
            np.maximum(np.abs(high[:, 1:] - prev), np.abs(low[:, 1:] - prev)),
        )
    return tr


def atr_stack(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, periods: Sequence[int]
) -> np.ndarray:
    periods = [max(2, int(p)) for p in periods]
    tr = true_range(high, low, close)
    return ewm_stack(
        tr, [1.0 / p for p in periods], start=0, min_index=[p - 1 for p in periods]
    )


# --------------------------------------------------------------------------
# Rolling / shifted primitives
# --------------------------------------------------------------------------
def _prefix_sum(x: np.ndarray) -> np.ndarray:
    """``out[:, k] = sum(x[:, :k])`` with a leading zero column."""
    out = np.zeros((x.shape[0], x.shape[1] + 1), dtype=np.float64)
    np.cumsum(x, axis=1, out=out[:, 1:])
    return out


def volume_ratio_stack(volume: np.ndarray, lookbacks: Sequence[int]) -> np.ndarray:
    """Volume against its own trailing average, current bar excluded.

    Excluding the current bar is the point of the rule: comparing a bar against
    an average that already contains it dilutes exactly the spike being looked
    for, and the dilution grows as the lookback shrinks.
    """
    volume = np.asarray(volume, dtype=np.float32)
    n_symbols, n_bars = volume.shape
    cum = _prefix_sum(volume)
    idx = np.arange(n_bars, dtype=np.int64)

    out = np.full((len(lookbacks), n_symbols, n_bars), NAN, dtype=np.float32)
    for i, raw in enumerate(lookbacks):
        lookback = max(2, int(raw))
        lo = np.maximum(0, idx - lookback)
        total = cum[:, idx] - cum[:, lo]
        count = np.minimum(idx, lookback).astype(np.float64)
        enough = count >= max(3, lookback // 2)
        with np.errstate(divide="ignore", invalid="ignore"):
            avg = np.where(enough[None, :] & (count[None, :] > 0), total / np.where(count == 0, 1, count), np.nan)
            out[i] = (volume / np.where(avg == 0.0, np.nan, avg)).astype(np.float32)
    return out


def shift_stack(close: np.ndarray, bars: Sequence[int]) -> np.ndarray:
    """``close`` moved forward by each of ``bars``; leading positions are NaN."""
    close = np.asarray(close, dtype=np.float32)
    n_bars = close.shape[1]
    out = np.full((len(bars), close.shape[0], n_bars), NAN, dtype=np.float32)
    for i, raw in enumerate(bars):
        b = max(1, int(raw))
        if b < n_bars:
            out[i, :, b:] = close[:, :-b]
    return out


def momentum_stack(close: np.ndarray, bars: Sequence[int]) -> np.ndarray:
    base = shift_stack(close, bars)
    with np.errstate(divide="ignore", invalid="ignore"):
        return ((close[None, :, :] - base) / np.where(base == 0.0, NAN, base)).astype(
            np.float32
        )


def consecutive_up_stack(close: np.ndarray, bars: Sequence[int]) -> np.ndarray:
    """True where the last ``n`` bars each closed above the one before.

    The "not one noisy print" half of the momentum rule: a single violent green
    candle does not qualify on its own.
    """
    close = np.asarray(close, dtype=np.float32)
    n_symbols, n_bars = close.shape
    up = np.zeros((n_symbols, n_bars), dtype=np.float32)
    if n_bars > 1:
        up[:, 1:] = (close[:, 1:] > close[:, :-1]).astype(np.float32)
    cum = _prefix_sum(up)
    idx = np.arange(n_bars, dtype=np.int64)

    out = np.zeros((len(bars), n_symbols, n_bars), dtype=bool)
    for i, raw in enumerate(bars):
        b = max(1, int(raw))
        lo = np.maximum(0, idx + 1 - b)
        total = cum[:, idx + 1] - cum[:, lo]
        out[i] = (total >= b) & (idx >= b - 1)[None, :]
    return out


# --------------------------------------------------------------------------
# Regime detection (spec 4.1)
# --------------------------------------------------------------------------
TRENDING, RANGING, CHOPPY = 0, 1, 2
REGIME_NAMES = {TRENDING: "trending", RANGING: "ranging", CHOPPY: "choppy"}


def efficiency_ratio_stack(close: np.ndarray, lookbacks: Sequence[int]) -> np.ndarray:
    """Kaufman efficiency ratio: net travel divided by gross travel.

    A market that moves 10% in a straight line and one that moves 10% net after
    whipsawing 40% look identical to a momentum rule and completely different to
    a trader. The ratio separates them with one number and no extra parameters,
    which is why it is the regime input rather than a tuned oscillator.
    """
    close = np.asarray(close, dtype=np.float32)
    n_symbols, n_bars = close.shape
    step = np.zeros((n_symbols, n_bars), dtype=np.float32)
    if n_bars > 1:
        step[:, 1:] = np.abs(close[:, 1:] - close[:, :-1])
    cum = _prefix_sum(step)
    idx = np.arange(n_bars, dtype=np.int64)

    out = np.full((len(lookbacks), n_symbols, n_bars), NAN, dtype=np.float32)
    for i, raw in enumerate(lookbacks):
        n = max(2, int(raw))
        lo = np.maximum(0, idx - n)
        gross = (cum[:, idx + 1] - cum[:, lo + 1]).astype(np.float32)
        net = np.full((n_symbols, n_bars), NAN, dtype=np.float32)
        if n < n_bars:
            net[:, n:] = np.abs(close[:, n:] - close[:, :-n])
        with np.errstate(divide="ignore", invalid="ignore"):
            out[i] = net / np.where(gross == 0.0, NAN, gross)
    return out


def classify_regime(
    efficiency: np.ndarray, atr_pct: np.ndarray, *, trend_er: float, chop_atr_pct: float
) -> np.ndarray:
    """Map efficiency and volatility onto trending / ranging / choppy.

    Directional efficiency splits trend from no-trend; volatility then splits
    no-trend into an orderly range (safe to fade, unsafe to breakout-trade) and
    genuine chop (unsafe for everything, which is where the overtrading in the
    prior backtest came from).
    """
    trending = efficiency >= np.float32(trend_er)
    volatile = atr_pct > np.float32(chop_atr_pct)
    out = np.where(trending, TRENDING, np.where(volatile, CHOPPY, RANGING))
    # An undefined efficiency reading means not enough history to judge; treat
    # that as chop so the gate errs toward not trading.
    return np.where(np.isnan(efficiency), CHOPPY, out).astype(np.int8)


# --------------------------------------------------------------------------
# Multi-timeframe confluence (spec 4.2)
# --------------------------------------------------------------------------
def aggregate_trend(
    close: np.ndarray, multiple: int, fast: int, slow: int
) -> np.ndarray:
    """Whether the ``multiple``-bar aggregate timeframe is in an uptrend.

    The aggregate closing at bar ``k*m - 1`` is the higher-timeframe candle; a
    base bar reads the last aggregate that has *closed*, never the one still
    forming. Reading the forming bar is the classic multi-timeframe mistake: it
    repaints, and a backtest that repaints reports trades the live bot could not
    have taken.
    """
    close = np.asarray(close, dtype=np.float32)
    n_symbols, n_bars = close.shape
    m = max(1, int(multiple))
    out = np.zeros((n_symbols, n_bars), dtype=bool)
    n_agg = n_bars // m
    if n_agg < 2:
        return out

    agg_close = close[:, m - 1 :: m][:, :n_agg]
    emas = ema_stack(agg_close, [fast, slow])
    agg_up = emas[0] > emas[1]

    # Base bar t reads aggregate ((t + 1) // m) - 1, i.e. the last closed one.
    idx = ((np.arange(n_bars, dtype=np.int64) + 1) // m) - 1
    have = idx >= 0
    gathered = agg_up[:, np.clip(idx, 0, n_agg - 1)]
    return gathered & have[None, :]


# --------------------------------------------------------------------------
# The cache
# --------------------------------------------------------------------------
@dataclass
class IndicatorCache:
    """Every distinct indicator series a parameter grid needs, computed once."""

    series: dict[tuple, np.ndarray] = field(default_factory=dict)
    n_symbols: int = 0
    n_bars: int = 0

    def get(self, key: tuple) -> np.ndarray:
        try:
            return self.series[key]
        except KeyError:
            raise KeyError(f"indicator {key} was not requested when the cache was built") from None

    def nbytes(self) -> int:
        return sum(int(a.nbytes) for a in self.series.values())

    def to_backend(self, backend: Backend) -> "IndicatorCache":
        """Move every series onto the compute device."""
        if not backend.on_gpu:
            return self
        moved = {k: backend.asarray(v) for k, v in self.series.items()}
        return IndicatorCache(moved, self.n_symbols, self.n_bars)


def _distinct(values: Iterable[Any]) -> list:
    out: list = []
    for v in values:
        if v not in out:
            out.append(v)
    return out


def build_cache(frames: Frames, requirements: dict[str, Iterable[Any]]) -> IndicatorCache:
    """Compute every series named in ``requirements``.

    ``requirements`` maps a family name to the distinct parameter values the grid
    uses, e.g. ``{"ema": [9, 12], "rsi": [14], "mtf": [(3, 9, 21)]}``.
    """
    close, high, low, volume = frames.close, frames.high, frames.low, frames.volume
    cache = IndicatorCache(n_symbols=frames.n_symbols, n_bars=frames.n_bars)

    def add(family: str, values: list, stack: np.ndarray) -> None:
        for i, value in enumerate(values):
            cache.series[(family, value)] = stack[i]

    spans = _distinct(int(v) for v in requirements.get("ema", ()))
    if spans:
        add("ema", spans, ema_stack(close, spans))

    rsi_periods = _distinct(int(v) for v in requirements.get("rsi", ()))
    if rsi_periods:
        add("rsi", rsi_periods, rsi_stack(close, rsi_periods))

    atr_periods = _distinct(int(v) for v in requirements.get("atr", ()))
    if atr_periods:
        stack = atr_stack(high, low, close, atr_periods)
        add("atr", atr_periods, stack)
        with np.errstate(divide="ignore", invalid="ignore"):
            pct = stack / np.where(close == 0.0, NAN, close)[None, :, :]
        add("atr_pct", atr_periods, pct.astype(np.float32))

    lookbacks = _distinct(int(v) for v in requirements.get("vol_ratio", ()))
    if lookbacks:
        add("vol_ratio", lookbacks, volume_ratio_stack(volume, lookbacks))

    mom_bars = _distinct(int(v) for v in requirements.get("momentum", ()))
    if mom_bars:
        add("momentum", mom_bars, momentum_stack(close, mom_bars))
        add("consec_up", mom_bars, consecutive_up_stack(close, mom_bars))

    er_bars = _distinct(int(v) for v in requirements.get("efficiency", ()))
    if er_bars:
        add("efficiency", er_bars, efficiency_ratio_stack(close, er_bars))

    for spec in _distinct(tuple(v) for v in requirements.get("mtf", ())):
        multiple, fast, slow = spec
        cache.series[("mtf", spec)] = aggregate_trend(close, multiple, fast, slow)

    return cache
