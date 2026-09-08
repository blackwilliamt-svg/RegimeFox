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
# Indicator-combination search (gap-closure item 3) - MACD, Bollinger,
# Stochastic, ADX, VWAP. Parity with solbot.indicators is asserted bar for
# bar in tests/test_vector_parity.py; see that module's docstring for why
# this is the constraint that matters most in the whole optimizer.
# --------------------------------------------------------------------------
def _rolling_extreme_stack(
    x: np.ndarray, windows: Sequence[int], *, mode: str
) -> np.ndarray:
    """``[len(windows), symbols, bars]`` causal rolling max/min - pandas'
    ``rolling(window).max()``/``.min()`` semantics (NaN before the window is
    filled).

    Block-decomposition per symbol row: prefix and suffix extrema within
    blocks of ``window``, then one elementwise combine - the same algorithm
    ``solopt.stress._rolling_max`` uses for its (forward-looking) window,
    adapted here for a trailing one via an index shift, and batched over
    symbols and several window sizes at once.
    """
    # float64 internally: stochastic's %K subtracts two nearly-equal prices, and
    # float32's ~7 significant digits is not enough headroom for that
    # cancellation to stay inside the parity test's tolerance.
    x = np.asarray(x, dtype=np.float64)
    n_symbols, n_bars = x.shape
    fill = -np.inf if mode == "max" else np.inf
    reduce = np.maximum if mode == "max" else np.minimum
    out = np.full((len(windows), n_symbols, n_bars), NAN, dtype=np.float32)
    for wi, raw in enumerate(windows):
        w = max(1, int(raw))
        if w == 1:
            out[wi] = x
            continue
        if w > n_bars:
            continue
        pad = (-n_bars) % w
        padded = np.concatenate(
            [x, np.full((n_symbols, pad), fill, dtype=np.float64)], axis=1
        )
        blocks = padded.reshape(n_symbols, -1, w)
        prefix = reduce.accumulate(blocks, axis=2)
        suffix = reduce.accumulate(blocks[:, :, ::-1], axis=2)[:, :, ::-1]
        prefix_flat = prefix.reshape(n_symbols, -1)
        suffix_flat = suffix.reshape(n_symbols, -1)
        n_out = n_bars - w + 1
        forward = reduce(suffix_flat[:, :n_out], prefix_flat[:, w - 1 : w - 1 + n_out])
        out[wi, :, w - 1 :] = forward
    return out


def _ema_stack_f64(close: np.ndarray, spans: Sequence[int]) -> np.ndarray:
    """Same recursion as :func:`ema_stack`, kept in float64 throughout.

    MACD's line is the *difference* of two EMAs that are usually close in
    magnitude - float32's ~7 significant digits leave too little headroom for
    that cancellation once both spans exceed a few hundred bars of history,
    which the module-wide float32 storage convention elsewhere in this file
    is fine with only because nothing else here subtracts two near-equal
    large numbers.
    """
    x = np.asarray(close, dtype=np.float64)
    n_symbols, n_bars = x.shape
    alphas = np.asarray(
        [2.0 / (max(1, int(s)) + 1.0) for s in spans], dtype=np.float64
    ).reshape(-1, 1)
    n_alpha = alphas.shape[0]
    out = np.empty((n_alpha, n_symbols, n_bars), dtype=np.float64)
    if n_bars == 0:
        return out
    prev = np.repeat(x[None, :, 0], n_alpha, axis=0)
    out[:, :, 0] = prev
    one_minus = 1.0 - alphas
    for t in range(1, n_bars):
        prev = alphas * x[None, :, t] + one_minus * prev
        out[:, :, t] = prev
    return out


def macd_line_stack(close: np.ndarray, pairs: Sequence[tuple[int, int]]) -> np.ndarray:
    """``[len(pairs), symbols, bars]`` MACD line for each distinct (fast, slow)."""
    fasts = [int(f) for f, _ in pairs]
    slows = [int(s) for _, s in pairs]
    ef = _ema_stack_f64(close, fasts)
    es = _ema_stack_f64(close, slows)
    return (ef - es).astype(np.float32)


def macd_signal_stack(macd_line: np.ndarray, spans: Sequence[int]) -> np.ndarray:
    """``[len(spans), symbols, bars]`` signal line for one (fast, slow) pair's
    MACD line - ``ema(macd_line, signal)`` with no warm-up mask, matching
    ``solbot.indicators.macd``'s ``ema()`` call exactly. `macd_line` is 2D
    ``[symbols, bars]``, one distinct (fast, slow) pair at a time."""
    alphas = [2.0 / (max(1, int(s)) + 1.0) for s in spans]
    return ewm_stack(macd_line, alphas, start=0)


def bollinger_stack(
    close: np.ndarray, periods: Sequence[int]
) -> tuple[np.ndarray, np.ndarray]:
    """``[len(periods), symbols, bars]`` rolling mean and population std
    (ddof=0), computed from prefix sums of x and x^2 in O(1) per period."""
    close = np.asarray(close, dtype=np.float64)  # variance needs the precision
    n_symbols, n_bars = close.shape
    cum = _prefix_sum(close)
    cum2 = _prefix_sum(close * close)
    idx = np.arange(n_bars, dtype=np.int64)

    mean_out = np.full((len(periods), n_symbols, n_bars), NAN, dtype=np.float32)
    std_out = np.full((len(periods), n_symbols, n_bars), NAN, dtype=np.float32)
    for i, raw in enumerate(periods):
        p = max(2, int(raw))
        lo = idx - p
        enough = idx >= p - 1
        lo_c = np.maximum(lo, -1) + 1
        total = cum[:, idx + 1] - cum[:, lo_c]
        total2 = cum2[:, idx + 1] - cum2[:, lo_c]
        mean = total / p
        var = np.maximum(total2 / p - mean * mean, 0.0)
        mean_out[i] = np.where(enough[None, :], mean, np.nan).astype(np.float32)
        std_out[i] = np.where(enough[None, :], np.sqrt(var), np.nan).astype(np.float32)
    return mean_out, std_out


def stochastic_k_stack(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, k_periods: Sequence[int]
) -> np.ndarray:
    """``[len(k_periods), symbols, bars]`` %K."""
    close64 = np.asarray(close, dtype=np.float64)
    lowest = _rolling_extreme_stack(low, k_periods, mode="min").astype(np.float64)
    highest = _rolling_extreme_stack(high, k_periods, mode="max").astype(np.float64)
    span = highest - lowest
    with np.errstate(divide="ignore", invalid="ignore"):
        k = 100.0 * (close64[None, :, :] - lowest) / np.where(span == 0.0, NAN, span)
    return k.astype(np.float32)


def stochastic_d_stack(k: np.ndarray, d_periods: Sequence[int]) -> np.ndarray:
    """``[len(d_periods), symbols, bars]`` %D - a plain rolling mean of one
    %K series, via the same prefix-sum trick as the Bollinger mean."""
    n_symbols, n_bars = k.shape
    valid = ~np.isnan(k)
    filled = np.where(valid, k, 0.0).astype(np.float64)
    cum = _prefix_sum(filled)
    cum_n = _prefix_sum(valid.astype(np.float64))
    idx = np.arange(n_bars, dtype=np.int64)

    out = np.full((len(d_periods), n_symbols, n_bars), NAN, dtype=np.float32)
    for i, raw in enumerate(d_periods):
        p = max(1, int(raw))
        lo = np.maximum(idx - p, -1) + 1
        total = cum[:, idx + 1] - cum[:, lo]
        count = cum_n[:, idx + 1] - cum_n[:, lo]
        enough = (idx >= p - 1) & (count >= p)
        with np.errstate(divide="ignore", invalid="ignore"):
            mean = total / np.where(count == 0, np.nan, count)
        out[i] = np.where(enough[None, :], mean, np.nan).astype(np.float32)
    return out


def adx_stack(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, periods: Sequence[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``[len(periods), symbols, bars]`` ADX, +DI, -DI - Wilder's definition,
    matching ``solbot.indicators.adx`` bar for bar."""
    high = np.asarray(high, dtype=np.float32)
    low = np.asarray(low, dtype=np.float32)
    n_symbols, n_bars = high.shape

    up_move = np.zeros_like(high)
    down_move = np.zeros_like(low)
    if n_bars > 1:
        up_move[:, 1:] = high[:, 1:] - high[:, :-1]
        down_move[:, 1:] = -(low[:, 1:] - low[:, :-1])
    plus_dm = np.where((up_move > down_move) & (up_move > 0.0), up_move, 0.0).astype(np.float32)
    minus_dm = np.where((down_move > up_move) & (down_move > 0.0), down_move, 0.0).astype(
        np.float32
    )
    tr = true_range(high, low, close)

    periods = [max(2, int(p)) for p in periods]
    alphas = [1.0 / p for p in periods]
    mins = [p - 1 for p in periods]  # matches atr_stack's warm-up convention

    smoothed_tr = ewm_stack(tr, alphas, start=0, min_index=mins)
    smoothed_plus = ewm_stack(plus_dm, alphas, start=0, min_index=mins)
    smoothed_minus = ewm_stack(minus_dm, alphas, start=0, min_index=mins)

    with np.errstate(divide="ignore", invalid="ignore"):
        plus_di = 100.0 * smoothed_plus / np.where(smoothed_tr == 0.0, NAN, smoothed_tr)
        minus_di = 100.0 * smoothed_minus / np.where(smoothed_tr == 0.0, NAN, smoothed_tr)
        di_sum = plus_di + minus_di
        dx = 100.0 * np.abs(plus_di - minus_di) / np.where(di_sum == 0.0, NAN, di_sum)

    # Wilder-smooth DX per distinct period. dx carries NaN through the warm-up,
    # which ewm_stack (adjust=False) propagates from the first NaN onward - so
    # each period's DX must be smoothed against its own min_index, one call per
    # period rather than one batched call across all of them.
    adx_out = np.full((len(periods), n_symbols, n_bars), NAN, dtype=np.float32)
    for i, (alpha, p) in enumerate(zip(alphas, periods)):
        dx_p = np.where(np.isnan(dx[i]), NAN, dx[i])
        smoothed = ewm_stack(np.nan_to_num(dx_p, nan=0.0), [alpha], start=p - 1)[0]
        adx_out[i] = np.where(np.arange(n_bars)[None, :] < (2 * p - 2), NAN, smoothed)

    return adx_out, plus_di.astype(np.float32), minus_di.astype(np.float32)


def vwap_stack(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray,
    periods: Sequence[int],
) -> np.ndarray:
    """``[len(periods), symbols, bars]`` rolling volume-weighted average price."""
    typical = ((high + low + close) / 3.0).astype(np.float64)
    volume = np.asarray(volume, dtype=np.float64)
    n_symbols, n_bars = typical.shape
    cum_pv = _prefix_sum(typical * volume)
    cum_v = _prefix_sum(volume)
    idx = np.arange(n_bars, dtype=np.int64)

    out = np.full((len(periods), n_symbols, n_bars), NAN, dtype=np.float32)
    for i, raw in enumerate(periods):
        p = max(2, int(raw))
        lo = np.maximum(idx - p, -1) + 1
        enough = idx >= p - 1
        num = cum_pv[:, idx + 1] - cum_pv[:, lo]
        den = cum_v[:, idx + 1] - cum_v[:, lo]
        with np.errstate(divide="ignore", invalid="ignore"):
            val = num / np.where(den == 0.0, np.nan, den)
        out[i] = np.where(enough[None, :], val, np.nan).astype(np.float32)
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

    # --- indicator-combination search (gap-closure item 3) ----------------
    macd_pairs = _distinct(tuple(int(x) for x in v) for v in requirements.get("macd_line", ()))
    if macd_pairs:
        add("macd_line", macd_pairs, macd_line_stack(close, macd_pairs))
    for fast, slow, signal in _distinct(
        tuple(int(x) for x in v) for v in requirements.get("macd_signal", ())
    ):
        line = cache.series.get(("macd_line", (fast, slow)))
        if line is None:
            line = macd_line_stack(close, [(fast, slow)])[0]
        cache.series[("macd_signal", (fast, slow, signal))] = macd_signal_stack(
            line, [signal]
        )[0]

    bb_periods = _distinct(int(v) for v in requirements.get("bbands", ()))
    if bb_periods:
        mean, std = bollinger_stack(close, bb_periods)
        add("bb_mid", bb_periods, mean)
        add("bb_std", bb_periods, std)

    stoch_k_periods = _distinct(int(v) for v in requirements.get("stoch_k", ()))
    if stoch_k_periods:
        add("stoch_k", stoch_k_periods, stochastic_k_stack(high, low, close, stoch_k_periods))
    for k_period, d_period in _distinct(
        tuple(int(x) for x in v) for v in requirements.get("stoch_d", ())
    ):
        k = cache.series.get(("stoch_k", k_period))
        if k is None:
            k = stochastic_k_stack(high, low, close, [k_period])[0]
        cache.series[("stoch_d", (k_period, d_period))] = stochastic_d_stack(k, [d_period])[0]

    adx_periods = _distinct(int(v) for v in requirements.get("adx", ()))
    if adx_periods:
        adx_v, plus_v, minus_v = adx_stack(high, low, close, adx_periods)
        add("adx", adx_periods, adx_v)
        add("plus_di", adx_periods, plus_v)
        add("minus_di", adx_periods, minus_v)

    vwap_periods = _distinct(int(v) for v in requirements.get("vwap", ()))
    if vwap_periods:
        add("vwap", vwap_periods, vwap_stack(high, low, close, volume, vwap_periods))

    return cache
