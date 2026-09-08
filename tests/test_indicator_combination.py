"""Gap-closure item 3: indicator-combination search.

The optimizer needs to choose *which* of nine indicators drives an entry, not
just their periods/thresholds. This pins the pandas reference side: the five
new indicator functions in solbot/indicators.py, the vote gate on Snapshot,
and that the legacy default (mask/min_agree unset) reproduces the exact old
hard-AND of volume spike + momentum + RSI + EMA cross.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from solbot.indicators import (
    ALL_INDICATOR_BITS,
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
    Snapshot,
    adx,
    bollinger_bands,
    compute,
    macd,
    percent_b,
    popcount,
    snapshot_at,
    stochastic,
    vwap,
)
from solbot.strategy import evaluate_entry

from test_strategy import core_only, make_candles, with_spike


# --------------------------------------------------------------------------
# the five new indicator functions
# --------------------------------------------------------------------------
def _trending_up_series(n: int = 80) -> pd.Series:
    return pd.Series(np.linspace(1.0, 2.0, n))


def test_macd_reads_bullish_in_a_clean_uptrend():
    close = _trending_up_series()
    line, signal = macd(close, 12, 26, 9)
    assert line.iloc[-1] > signal.iloc[-1]


def test_bollinger_percent_b_is_near_one_at_a_fresh_high():
    close = _trending_up_series()
    _mid, upper, lower = bollinger_bands(close, 20, 2.0)
    pb = percent_b(close, upper, lower)
    assert pb.iloc[-1] > 0.5


def test_bollinger_bands_are_nan_before_the_window_fills():
    close = _trending_up_series(10)
    _mid, upper, lower = bollinger_bands(close, 20, 2.0)
    assert upper.isna().all()


def test_stochastic_k_is_high_at_a_fresh_high():
    df = make_candles(60, drift=0.01)
    k, d = stochastic(df, 14, 3)
    assert k.iloc[-1] > 50.0


def test_adx_and_di_directional_in_a_clean_uptrend():
    df = make_candles(80, drift=0.01, seed=3)
    adx_line, plus_di, minus_di = adx(df, 14)
    assert plus_di.iloc[-1] > minus_di.iloc[-1]
    assert adx_line.iloc[-1] >= 0.0


def test_vwap_sits_below_price_in_a_clean_uptrend():
    df = make_candles(60, drift=0.01)
    v = vwap(df, 20)
    assert df["close"].iloc[-1] > v.iloc[-1]


# --------------------------------------------------------------------------
# the vote gate
# --------------------------------------------------------------------------
def _snap(cfg: dict, *, spike: bool = True) -> Snapshot:
    df = with_spike(make_candles(80)) if spike else make_candles(80)
    data = compute(df, cfg)
    return snapshot_at(data, -1)


def test_legacy_mask_requires_all_four_original_checks(settings):
    snap = _snap(core_only(settings), spike=False)
    gate = snap.indicator_gate(core_only(settings))
    assert gate.active_count == 4
    assert gate.min_agree == 4
    assert not gate.ok   # flat data: at least volume/momentum must fail


def test_popcount():
    assert popcount(0) == 0
    assert popcount(LEGACY_INDICATOR_MASK) == 4
    assert popcount(ALL_INDICATOR_BITS) == 9


def test_a_single_indicator_with_min_agree_one_can_pass_alone(settings):
    """Only MACD active, one vote required - an entry that only a legacy
    hard-AND config would reject can now fire on MACD alone."""
    cfg = {
        **core_only(settings),
        "indicator_mask": IND_MACD,
        "indicator_min_agree": 1,
    }
    df = with_spike(make_candles(80))
    data = compute(df, cfg)
    snap = snapshot_at(data, -1)
    gate = snap.indicator_gate(cfg)
    assert gate.active_count == 1
    assert gate.votes["macd"] is True
    assert gate.ok


def test_inactive_indicators_are_not_reported_in_entry_reasons(settings):
    """An indicator outside the mask never shows up in passed/reasons - only
    what actually governed the decision does."""
    cfg = {
        **core_only(settings),
        "indicator_mask": IND_VOLUME_SPIKE | IND_MOMENTUM,
        "indicator_min_agree": 2,
    }
    df = make_candles(80)   # flat - nothing fires
    signal = evaluate_entry(df, cfg, mint="X", liquidity_usd=500_000)
    assert not signal.ok
    joined = " ".join(signal.reasons)
    assert "RSI" not in joined
    assert "EMA" not in joined
    assert "MACD" not in joined


def test_entry_fires_via_new_indicators_alone_when_legacy_ones_are_excluded(settings):
    """A combo that only trusts ADX + VWAP, needing both, can still fire a
    real entry through evaluate_entry end-to-end."""
    cfg = {
        **core_only(settings),
        "indicator_mask": IND_ADX | IND_VWAP,
        "indicator_min_agree": 2,
        "adx_min": 0.0,   # a short synthetic series won't build much ADX magnitude
    }
    df = with_spike(make_candles(80, drift=0.01), pump=0.01, vol_mult=1.0)
    signal = evaluate_entry(df, cfg, mint="X", liquidity_usd=500_000)
    assert signal.ok, signal.reasons


def test_indicator_min_agree_exceeding_active_count_is_rejected(settings):
    from solbot.config import ConfigError, validate

    with pytest.raises(ConfigError, match="indicator_min_agree"):
        validate(
            {"indicator_mask": IND_VOLUME_SPIKE | IND_RSI, "indicator_min_agree": 3},
            settings,
        )


def test_macd_fast_must_be_shorter_than_slow(settings):
    from solbot.config import ConfigError, validate

    with pytest.raises(ConfigError, match="macd_fast"):
        validate({"macd_fast": 30, "macd_slow": 26}, settings)
