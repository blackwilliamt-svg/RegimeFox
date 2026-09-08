"""Vectorization correctness safeguards.

The gap-closure spec calls these "critical, required": the whole optimizer
rests on the vectorized ``[symbols, bars]`` engine computing exactly what a
scalar, per-bar implementation would, and the parity tests elsewhere in this
suite (``test_vector_parity.py``) already check that against pandas. What
they do not check is the *shape* of the vectorization itself - four separate
ways a batched implementation can quietly diverge from a scalar one even
while every individual formula is correct:

1. **Look-ahead leakage** - a rolling/recursive routine that (by an indexing
   bug) reads a bar it has not "reached" yet. A live bot never has next
   week's candle; a backtest that accidentally does will produce numbers no
   strategy can actually earn.
2. **Time misalignment** - symbols packed into one rectangular block at
   different real lengths (different listing dates, different history) must
   not bleed into each other just because they now share a bars axis.
3. **Reference-implementation cross-check** - an independent, deliberately
   *not* vectorized loop, to catch a bug that a from-the-same-assumptions
   NumPy rewrite could reproduce identically.
4. **Memory duplication** - a parameter grid with many combinations but few
   *distinct* indicator settings must compute (and store) each distinct
   series once, not once per combination; see ``solopt.indicators``' "compute
   once per distinct value" rule.
"""
from __future__ import annotations

import numpy as np
import pytest

from solopt import indicators as ind
from solopt.frames import frames_from_series
from solopt.schema import CRYPTO

SEED = 2024


def _ohlcv(n: int, n_symbols: int = 1, seed: int = SEED):
    rng = np.random.default_rng(seed)
    close = np.cumprod(1 + rng.normal(0, 0.01, (n_symbols, n)), axis=1).astype(np.float32) * 100.0
    high = (close * (1 + np.abs(rng.normal(0, 0.004, (n_symbols, n))))).astype(np.float32)
    low = (close * (1 - np.abs(rng.normal(0, 0.004, (n_symbols, n))))).astype(np.float32)
    volume = np.abs(rng.normal(1000.0, 400.0, (n_symbols, n))).astype(np.float32)
    return close, high, low, volume


# --------------------------------------------------------------------------
# 1. Look-ahead-leakage: an adversarial test
# --------------------------------------------------------------------------
# Every function below is documented as causal (its output at bar i depends
# only on bars <= i). The test proves that empirically rather than trusting
# the docstring: recompute with every bar at and after `CUTOFF` replaced by
# adversarial values - reordered *and* blown up in magnitude, either of which
# would perturb the output at bar i < CUTOFF if the implementation peeked
# ahead - and require the untouched prefix to come back byte-for-byte
# identical. `MARGIN` stays clear of `CUTOFF` itself so a correct routine's
# own window (e.g. the last `period` bars before the cutoff) is never
# straddling the mutated region.
N = 260
CUTOFF = 160
MARGIN = 40  # >= the largest period any case below uses


def _mutate_future(arr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = arr.copy()
    tail = out[..., CUTOFF:]
    width = tail.shape[-1]
    perm = rng.permutation(width)
    out[..., CUTOFF:] = tail[..., perm] * rng.uniform(5.0, 50.0, size=width).astype(out.dtype)
    return out


def _assert_prefix_unaffected(before: np.ndarray, after: np.ndarray) -> None:
    hi = CUTOFF - MARGIN
    np.testing.assert_array_equal(before[..., :hi], after[..., :hi])


LEAKAGE_CASES = {
    "ema": lambda c, h, l, v: ind.ema_stack(c, [9, 21]),
    "rsi": lambda c, h, l, v: ind.rsi_stack(c, [14]),
    "atr": lambda c, h, l, v: ind.atr_stack(h, l, c, [14]),
    "volume_ratio": lambda c, h, l, v: ind.volume_ratio_stack(v, [20]),
    "momentum": lambda c, h, l, v: ind.momentum_stack(c, [3]),
    "consecutive_up": lambda c, h, l, v: ind.consecutive_up_stack(c, [3]).astype(np.float32),
    "efficiency_ratio": lambda c, h, l, v: ind.efficiency_ratio_stack(c, [20]),
    "macd_line": lambda c, h, l, v: ind.macd_line_stack(c, [(12, 26)]),
    "macd_signal": lambda c, h, l, v: ind.macd_signal_stack(
        ind.macd_line_stack(c, [(12, 26)])[0], [9]
    ),
    "bollinger_mean": lambda c, h, l, v: ind.bollinger_stack(c, [20])[0],
    "bollinger_std": lambda c, h, l, v: ind.bollinger_stack(c, [20])[1],
    "volume_zscore": lambda c, h, l, v: ind.volume_zscore_stack(v, [20]),
    "stochastic_k": lambda c, h, l, v: ind.stochastic_k_stack(h, l, c, [14]),
    "stochastic_d": lambda c, h, l, v: ind.stochastic_d_stack(
        ind.stochastic_k_stack(h, l, c, [14])[0], [3]
    ),
    "adx": lambda c, h, l, v: ind.adx_stack(h, l, c, [14])[0],
    "plus_di": lambda c, h, l, v: ind.adx_stack(h, l, c, [14])[1],
    "minus_di": lambda c, h, l, v: ind.adx_stack(h, l, c, [14])[2],
    "vwap": lambda c, h, l, v: ind.vwap_stack(h, l, c, v, [20]),
}


@pytest.mark.parametrize("name", sorted(LEAKAGE_CASES))
def test_no_look_ahead_leakage(name):
    close, high, low, volume = _ohlcv(N, n_symbols=3)
    rng = np.random.default_rng(SEED + 1)
    m_close = _mutate_future(close, rng)
    m_high = _mutate_future(high, rng)
    m_low = _mutate_future(low, rng)
    m_volume = np.abs(_mutate_future(volume, rng))  # volume must stay non-negative

    fn = LEAKAGE_CASES[name]
    before = fn(close, high, low, volume)
    after = fn(m_close, m_high, m_low, m_volume)

    _assert_prefix_unaffected(before, after)
    # Sanity check the harness itself: the mutation must actually have
    # changed *something* downstream, or a broken test could pass vacuously.
    assert not np.array_equal(
        np.nan_to_num(before[..., CUTOFF:]), np.nan_to_num(after[..., CUTOFF:])
    )


# --------------------------------------------------------------------------
# 2. Time misalignment: symbols packed at different real lengths
# --------------------------------------------------------------------------
def test_packing_symbols_of_different_length_does_not_perturb_either():
    """A newly-listed coin with 60 bars of history and an old one with 300
    share one [2, 300] block once packed (the short row zero-padded on the
    right). Batching them together must produce the exact same indicator
    values for each symbol as computing that symbol alone - if the padding
    of one row ever leaked into another's math, or a bar landed in the wrong
    column, this is where it would show up."""
    long_close, long_high, long_low, long_volume = _ohlcv(300, seed=SEED + 10)
    short_close, short_high, short_low, short_volume = _ohlcv(60, seed=SEED + 11)

    def _rows(n, close, high, low, volume):
        ts = np.arange(n, dtype=np.int64) * 300
        return np.stack([ts, close[0], high[0], low[0], close[0], volume[0]])

    series = {
        "OLD": _rows(300, long_close, long_high, long_low, long_volume),
        "NEW": _rows(60, short_close, short_high, short_low, short_volume),
    }
    packed = frames_from_series(series, seconds=300, schema=CRYPTO)
    assert packed.symbols == ["NEW", "OLD"]  # frames_from_series sorts names

    solo_old = frames_from_series(
        {"OLD": series["OLD"]}, seconds=300, schema=CRYPTO
    )
    solo_new = frames_from_series(
        {"NEW": series["NEW"]}, seconds=300, schema=CRYPTO
    )

    old_row = packed.symbols.index("OLD")
    new_row = packed.symbols.index("NEW")

    packed_ema = ind.ema_stack(packed.close, [9, 21])
    packed_rsi = ind.rsi_stack(packed.close, [14])
    solo_old_ema = ind.ema_stack(solo_old.close, [9, 21])
    solo_old_rsi = ind.rsi_stack(solo_old.close, [14])
    solo_new_ema = ind.ema_stack(solo_new.close, [9, 21])
    solo_new_rsi = ind.rsi_stack(solo_new.close, [14])

    # OLD occupies all 300 columns; NEW is real for only its first 60 and
    # zero-padded after - compare each symbol over its own valid columns.
    np.testing.assert_array_equal(packed_ema[:, old_row, :], solo_old_ema[:, 0, :])
    np.testing.assert_array_equal(packed_rsi[:, old_row, :], solo_old_rsi[:, 0, :])
    np.testing.assert_array_equal(packed_ema[:, new_row, :60], solo_new_ema[:, 0, :])
    np.testing.assert_array_equal(packed_rsi[:, new_row, :60], solo_new_rsi[:, 0, :])

    # And the mask says exactly what it should: real for NEW's first 60
    # columns, padding after.
    assert packed.mask[new_row, :60].all()
    assert not packed.mask[new_row, 60:].any()
    assert packed.mask[old_row, :].all()


# --------------------------------------------------------------------------
# 3. Reference-implementation cross-check: independent, non-vectorized loops
# --------------------------------------------------------------------------
def _ref_ema(close: np.ndarray, span: int) -> np.ndarray:
    """Plain Python loop, one bar at a time - no NumPy broadcasting tricks
    that could reproduce a bug ``ewm_stack`` also has."""
    alpha = np.float32(2.0 / (span + 1.0))
    one_minus = np.float32(1.0 - alpha)
    out = np.empty(len(close), dtype=np.float32)
    out[0] = close[0]
    for t in range(1, len(close)):
        out[t] = alpha * close[t] + one_minus * out[t - 1]
    return out


def _ref_rsi(close: np.ndarray, period: int) -> np.ndarray:
    """Wilder RSI via a hand-rolled loop, matching solbot.indicators.rsi's
    documented warm-up convention (100 before any loss has been seen)."""
    n = len(close)
    alpha = 1.0 / period
    avg_gain = 0.0
    avg_loss = 0.0
    out = np.full(n, 100.0, dtype=np.float32)
    for t in range(1, n):
        delta = float(close[t]) - float(close[t - 1])
        gain = max(delta, 0.0)
        loss = max(-delta, 0.0)
        if t == 1:
            avg_gain, avg_loss = gain, loss
        else:
            avg_gain = alpha * gain + (1 - alpha) * avg_gain
            avg_loss = alpha * loss + (1 - alpha) * avg_loss
        if t >= period:
            if avg_loss == 0.0:
                out[t] = 100.0
            else:
                rs = avg_gain / avg_loss
                out[t] = 100.0 - 100.0 / (1.0 + rs)
    return out


def _ref_bollinger_mean_std(close: np.ndarray, period: int):
    n = len(close)
    mean_out = np.full(n, np.nan, dtype=np.float64)
    std_out = np.full(n, np.nan, dtype=np.float64)
    for t in range(period - 1, n):
        window = close[t - period + 1 : t + 1].astype(np.float64)
        mean_out[t] = window.mean()
        std_out[t] = window.std(ddof=0)
    return mean_out, std_out


def test_ema_matches_an_independent_loop_reference():
    close, _, _, _ = _ohlcv(200, seed=SEED + 20)
    vectorized = ind.ema_stack(close, [12])[0, 0]
    reference = _ref_ema(close[0], 12)
    np.testing.assert_allclose(vectorized, reference, rtol=1e-4, atol=1e-4)


def test_rsi_matches_an_independent_loop_reference():
    close, _, _, _ = _ohlcv(200, seed=SEED + 21)
    vectorized = ind.rsi_stack(close, [14])[0, 0]
    reference = _ref_rsi(close[0], 14)
    np.testing.assert_allclose(vectorized, reference, rtol=1e-3, atol=1e-2)


def test_bollinger_matches_an_independent_loop_reference():
    close, _, _, _ = _ohlcv(150, seed=SEED + 22)
    mean_v, std_v = ind.bollinger_stack(close, [20])
    mean_ref, std_ref = _ref_bollinger_mean_std(close[0], 20)
    np.testing.assert_allclose(mean_v[0, 0], mean_ref, rtol=1e-5, atol=1e-4, equal_nan=True)
    np.testing.assert_allclose(std_v[0, 0], std_ref, rtol=1e-4, atol=1e-4, equal_nan=True)


# --------------------------------------------------------------------------
# 4. Memory duplication: distinct-value count, not combination count
# --------------------------------------------------------------------------
def test_indicator_series_are_computed_once_per_distinct_value_not_per_combo(monkeypatch):
    """A grid of many combinations sharing a handful of distinct EMA/RSI
    settings must call the underlying batched routine once (covering every
    distinct value in one stacked pass, per solopt.indicators' own "compute
    once per distinct value" rule) and store one series per distinct value -
    not once per combination, which is what would happen if a caller ever
    looped ``combo -> build_cache`` instead of batching the whole grid first.
    """
    import solopt.indicators as ind_mod
    from solopt.engine import PortfolioSettings, cache_requirements
    from solopt.params import ParamSpace

    calls = {"ema": 0, "rsi": 0}
    real_ema, real_rsi = ind_mod.ema_stack, ind_mod.rsi_stack

    def counting_ema(*a, **kw):
        calls["ema"] += 1
        return real_ema(*a, **kw)

    def counting_rsi(*a, **kw):
        calls["rsi"] += 1
        return real_rsi(*a, **kw)

    monkeypatch.setattr(ind_mod, "ema_stack", counting_ema)
    monkeypatch.setattr(ind_mod, "rsi_stack", counting_rsi)

    rng = np.random.default_rng(SEED + 30)
    combos = ParamSpace().sample(300, rng)
    assert len(combos) > 50  # the grid is rich enough for this to be meaningful

    close, high, low, volume = _ohlcv(120, n_symbols=2, seed=SEED + 31)
    from solopt.frames import Frames
    from solopt.schema import CRYPTO

    frames = Frames(
        symbols=["AAA", "BBB"],
        ts=(np.arange(120, dtype=np.int64) * 300)[None, :].repeat(2, axis=0),
        open=close, high=high, low=low, close=close, volume=volume,
        mask=np.ones((2, 120), dtype=bool),
        eligible=np.ones((2, 120), dtype=bool),
        counts=np.full(2, 120, dtype=np.int32),
        grid_pos=(np.arange(120, dtype=np.int32))[None, :].repeat(2, axis=0),
        seconds=300, schema=CRYPTO,
    )

    settings = PortfolioSettings()
    reqs = cache_requirements(combos, settings)
    distinct_ema = len(reqs["ema"])
    distinct_rsi = len(reqs["rsi"])
    assert distinct_ema < len(combos)  # the whole point of the grid: few distinct values

    cache = ind_mod.build_cache(frames, reqs)

    # One batched call per distinct *need* - the top-level EMA request, plus
    # one more per distinct multi-timeframe (confluence) spec, since each of
    # those needs its own EMA over its own aggregated series - regardless of
    # how many of the 300 combinations shared those values.
    assert calls["ema"] == 1 + len(reqs["mtf"])
    assert calls["ema"] < len(combos)
    assert calls["rsi"] == 1

    ema_keys = [k for k in cache.series if k[0] == "ema"]
    rsi_keys = [k for k in cache.series if k[0] == "rsi"]
    assert len(ema_keys) == distinct_ema
    assert len(rsi_keys) == distinct_rsi

    # And the cache's footprint tracks the distinct-value count, not the
    # combination count: given the exact same requirements (same distinct
    # values across every family), rebuilding with 10x fewer combinations
    # that happen to produce them must not shrink it - the footprint is a
    # function of what was asked for, never of how many combos asked for it.
    small_combos = combos[: max(5, len(combos) // 10)]
    small_reqs = cache_requirements(small_combos, settings)
    if small_reqs == reqs:
        small_cache = ind_mod.build_cache(frames, small_reqs)
        assert small_cache.nbytes() == cache.nbytes()
    else:
        pytest.skip("this seed's smaller sub-grid happened to drop a distinct value")
