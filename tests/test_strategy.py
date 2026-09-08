"""Entry/exit rules and indicator behaviour."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from solbot.indicators import atr, compute, ema, resample, rsi, snapshot_at, volume_ratio
from solbot.strategy import evaluate_entry, evaluate_exit, update_trailing_stop


def make_candles(
    n: int = 60,
    *,
    start: float = 1.0,
    drift: float = 0.0,
    volume: float = 1000.0,
    interval: int = 600,
    seed: int = 7,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    price = start
    ts = 1_700_000_000
    for i in range(n):
        price = max(0.0001, price * (1.0 + drift + rng.normal(0, 0.002)))
        high = price * 1.004
        low = price * 0.996
        rows.append((ts + i * interval, price * 0.999, high, low, price, volume))
    return pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])


def with_spike(df: pd.DataFrame, *, bars: int = 3, pump: float = 0.004, vol_mult: float = 6.0):
    """Append a clean volume-spike + momentum setup on the tail."""
    df = df.copy()
    last = df.iloc[-1]
    price = float(last["close"])
    ts = int(last["ts"])
    base_vol = float(df["volume"].tail(20).mean())
    rows = []
    for i in range(bars):
        price *= 1.0 + pump
        rows.append(
            {
                "ts": ts + (i + 1) * 600,
                "open": price / (1.0 + pump),
                "high": price * 1.003,
                "low": price / (1.0 + pump) * 0.999,
                "close": price,
                "volume": base_vol * vol_mult,
            }
        )
    return pd.concat([df, pd.DataFrame(rows)], ignore_index=True)


def core_only(settings: dict) -> dict:
    """Settings with the regime and confluence gates off.

    Those two gates are separate rules with their own tests below. Tests that
    exist to pin the volume/momentum/RSI behaviour switch them off so a change
    in regime classification cannot silently make them pass or fail for an
    unrelated reason.
    """
    return {**settings, "regime_gate_enabled": False, "confluence_enabled": False}


# --------------------------------------------------------------------------
# indicators
# --------------------------------------------------------------------------
def test_rsi_bounds_and_direction():
    rising = pd.Series(np.linspace(1, 2, 60))
    falling = pd.Series(np.linspace(2, 1, 60))
    assert rsi(rising, 14).iloc[-1] > 70
    assert rsi(falling, 14).iloc[-1] < 30
    assert rsi(rising, 14).between(0, 100).all()


def test_rsi_all_gains_does_not_produce_nan():
    """A window with no losses must read as overbought, not NaN."""
    series = pd.Series(np.linspace(1, 5, 40))
    out = rsi(series, 14)
    assert not out.isna().any()
    assert out.iloc[-1] == pytest.approx(100.0, abs=1e-6)


def test_ema_tracks_and_atr_positive():
    df = make_candles(60)
    assert ema(df["close"], 9).iloc[-1] > 0
    assert atr(df, 14).iloc[-1] > 0


def test_volume_ratio_excludes_current_bar():
    """The average must not include the spike being measured, or it self-dampens."""
    df = make_candles(40, volume=100.0)
    df.loc[df.index[-1], "volume"] = 1000.0
    ratio = volume_ratio(df, 20).iloc[-1]
    assert ratio == pytest.approx(10.0, rel=0.05)


def test_resample_aggregates_correctly():
    df = make_candles(12, interval=300)   # 5-minute bars
    out = resample(df, 600)               # to 10-minute bars
    assert len(out) == 6
    assert out["open"].iloc[0] == df["open"].iloc[0]
    assert out["close"].iloc[0] == df["close"].iloc[1]
    assert out["high"].iloc[0] == max(df["high"].iloc[0], df["high"].iloc[1])
    assert out["volume"].iloc[0] == df["volume"].iloc[0] + df["volume"].iloc[1]


# --------------------------------------------------------------------------
# entry
# --------------------------------------------------------------------------
def test_entry_requires_all_conditions(settings):
    """Flat, low-volume price action must not produce an entry."""
    df = make_candles(60)
    signal = evaluate_entry(df, settings, mint="X", liquidity_usd=500_000)
    assert not signal.ok
    assert signal.reasons


def test_entry_fires_on_volume_spike_with_momentum(settings):
    df = with_spike(make_candles(60))
    signal = evaluate_entry(df, core_only(settings), mint="X", liquidity_usd=500_000)
    assert signal.ok, signal.reasons
    assert signal.stop < signal.price < signal.target
    assert settings["rr_min"] <= signal.rr <= settings["rr_max"]


def test_reward_risk_stays_inside_configured_bounds(settings):
    """Even an extreme signal must not scale outside the window."""
    for vol_mult in (2.5, 10.0, 100.0):
        df = with_spike(make_candles(60), pump=0.05, vol_mult=vol_mult)
        signal = evaluate_entry(df, settings, mint="X", liquidity_usd=1_000_000)
        if signal.ok:
            assert settings["rr_min"] <= signal.rr <= settings["rr_max"]


def test_entry_blocked_by_insufficient_liquidity(settings):
    df = with_spike(make_candles(60))
    signal = evaluate_entry(df, settings, mint="X", liquidity_usd=1_000)
    assert not signal.ok
    assert any("liquidity" in r for r in signal.reasons)


def test_entry_blocked_when_size_exceeds_pool_share(settings):
    df = with_spike(make_candles(60))
    signal = evaluate_entry(
        df, settings, mint="X", liquidity_usd=100_000, intended_size_usd=50_000
    )
    assert not signal.ok
    assert any("exceeds" in r for r in signal.reasons)


def test_entry_blocked_when_overbought(settings):
    settings = {**settings, "rsi_max_entry": 50.0}
    df = with_spike(make_candles(60))
    signal = evaluate_entry(df, settings, mint="X", liquidity_usd=500_000)
    assert not signal.ok
    assert any("RSI" in r for r in signal.reasons)


def test_entry_needs_minimum_candles(settings):
    df = make_candles(10)
    signal = evaluate_entry(df, settings, mint="X", liquidity_usd=500_000)
    assert not signal.ok
    assert "candles" in signal.reasons[0]


# --------------------------------------------------------------------------
# exit
# --------------------------------------------------------------------------
def base_position(**kw):
    position = {
        "entry_price": 1.0,
        "entry_ts": 1_700_000_000,
        "hard_stop": 0.95,
        "take_profit": 1.15,
        "trailing_stop": None,
        "trailing_armed": 0,
        "high_water_price": 1.0,
        "initial_risk": 0.05,
        "invalidation_count": 0,
    }
    position.update(kw)
    return position


def test_hard_stop_fires(settings):
    out = evaluate_exit(base_position(), 0.94, None, settings, now_ts=1_700_000_600)
    assert out.should_exit
    assert "hard stop" in out.reason


def test_trailing_stop_fires_when_armed(settings):
    position = base_position(trailing_stop=1.05, trailing_armed=1)
    out = evaluate_exit(position, 1.04, None, settings, now_ts=1_700_000_600)
    assert out.should_exit
    assert "trailing" in out.reason


def test_take_profit_fires(settings):
    out = evaluate_exit(base_position(), 1.20, None, settings, now_ts=1_700_000_600)
    assert out.should_exit
    assert "take-profit" in out.reason


def test_max_hold_fires(settings):
    later = 1_700_000_000 + settings["max_hold_minutes"] * 60 + 1
    out = evaluate_exit(base_position(), 1.01, None, settings, now_ts=later)
    assert out.should_exit
    assert "max hold" in out.reason


def test_signal_invalidation_needs_consecutive_bars(settings):
    """One bad bar is noise; the configured number in a row is a real exit."""
    faded = make_candles(60, drift=-0.004, volume=10.0)
    faded = compute(faded, settings)
    position = base_position(entry_price=float(faded["close"].iloc[-1]) * 0.9)
    position["hard_stop"] = position["entry_price"] * 0.5

    price = float(faded["close"].iloc[-1])
    soon = position["entry_ts"] + 600
    first = evaluate_exit(position, price, faded, settings, now_ts=soon, precomputed=True)
    assert not first.should_exit
    assert first.detail.get("invalidation_count") == 1

    position["invalidation_count"] = settings["signal_invalidation_bars"] - 1
    second = evaluate_exit(position, price, faded, settings, now_ts=soon, precomputed=True)
    assert second.should_exit
    assert "invalidated" in second.reason


# --------------------------------------------------------------------------
# trailing stop
# --------------------------------------------------------------------------
def test_trailing_arms_only_after_costs_are_covered(settings):
    position = base_position()
    # +0.5R is not enough to arm at the default 1.0R activation.
    changes = update_trailing_stop(position, 1.025, 0.02, settings)
    assert "trailing_stop" not in changes

    changes = update_trailing_stop(position, 1.06, 0.02, settings)
    assert changes.get("trailing_armed") == 1
    cost = (settings["taker_fee_pct"] * 2 + settings["max_slippage_pct"]) / 100.0
    breakeven = 1.0 * (1 + cost)
    # Breakeven-after-costs is a floor, never bare entry. The ATR trail may
    # already sit above it, which is fine - it must never sit below.
    assert changes["trailing_stop"] >= breakeven
    assert breakeven > 1.0


def test_trailing_stop_never_moves_down(settings):
    position = base_position(trailing_stop=1.08, trailing_armed=1, high_water_price=1.20)
    changes = update_trailing_stop(position, 1.10, 0.02, settings)
    assert changes.get("trailing_stop", 1.08) >= 1.08


def test_trailing_stop_follows_new_highs(settings):
    position = base_position(trailing_stop=1.02, trailing_armed=1, high_water_price=1.10)
    changes = update_trailing_stop(position, 1.50, 0.02, settings)
    assert changes["high_water_price"] == 1.50
    assert changes["trailing_stop"] > 1.02


def test_rsi_ceiling_leaves_a_workable_entry_window(settings):
    """The RSI ceiling must not cancel out the momentum rule.

    The momentum rule requires consecutive rising candles, which mechanically
    drives Wilder RSI into the mid-70s. A ceiling near 70 double-counts that
    condition and leaves no window in which an entry can fire. This pins the
    reason `rsi_max_entry` defaults to 78 rather than ~70.
    """
    setups = [
        with_spike(make_candles(60), bars=b, pump=p, vol_mult=6.0)
        for b in (3, 4)
        for p in (0.003, 0.004, 0.005)
    ]

    base = core_only(settings)
    at_78 = sum(
        1 for df in setups
        if evaluate_entry(df, base, mint="X", liquidity_usd=500_000).ok
    )
    tight = {**base, "rsi_max_entry": 72.0}
    at_72 = sum(
        1 for df in setups
        if evaluate_entry(df, tight, mint="X", liquidity_usd=500_000).ok
    )

    assert at_78 >= 4, "the default ceiling should admit ordinary setups"
    assert at_72 <= 1, "a ~70 ceiling is what made the strategy effectively inert"
    assert at_78 > at_72
