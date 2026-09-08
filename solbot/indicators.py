"""Indicator math.

Pure functions over a pandas frame with columns ts/open/high/low/close/volume.
Nothing here touches the network or the database, which is what lets the
backtest and the live engine share one implementation - if they diverged, the
backtest would stop being evidence about the live rules.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Sequence

import numpy as np
import pandas as pd

COLUMNS = ("ts", "open", "high", "low", "close", "volume")


def to_frame(rows: Sequence[Any]) -> pd.DataFrame:
    """Build a frame from sqlite3.Row objects or plain tuples."""
    if isinstance(rows, pd.DataFrame):
        df = rows.copy()
    else:
        data = [tuple(r[c] for c in COLUMNS) if hasattr(r, "keys") else tuple(r) for r in rows]
        df = pd.DataFrame(data, columns=list(COLUMNS))
    if df.empty:
        return df
    for c in COLUMNS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["close"]).sort_values("ts").reset_index(drop=True)
    df["ts"] = df["ts"].astype("int64")
    return df


def resample(df: pd.DataFrame, target_seconds: int) -> pd.DataFrame:
    """Aggregate candles up to a longer timeframe.

    Candle history is stored at a fixed 1-minute base interval (spec 3b), so
    the configured trading timeframe (5-15 minutes) is always built by rolling
    that base up here rather than trading a timeframe that merely resembles
    the configured one.
    """
    if df.empty:
        return df
    bucket = (df["ts"] // target_seconds) * target_seconds
    grouped = df.groupby(bucket, sort=True)
    out = pd.DataFrame(
        {
            "ts": grouped["ts"].first().index.astype("int64"),
            "open": grouped["open"].first().to_numpy(),
            "high": grouped["high"].max().to_numpy(),
            "low": grouped["low"].min().to_numpy(),
            "close": grouped["close"].last().to_numpy(),
            "volume": grouped["volume"].sum().to_numpy(),
        }
    ).reset_index(drop=True)
    return out


# --------------------------------------------------------------------------
# primitives
# --------------------------------------------------------------------------
def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=max(1, int(period)), adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    period = max(2, int(period))
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    # Wilder smoothing
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    # No losses in the window means a maximally overbought reading, not NaN.
    return out.where(avg_loss.notna() & (avg_loss != 0.0), 100.0).fillna(50.0)


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    ranges = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    period = max(2, int(period))
    return true_range(df).ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def volume_ratio(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """Current volume against its own trailing average, excluding the current bar."""
    lookback = max(2, int(lookback))
    avg = df["volume"].shift(1).rolling(lookback, min_periods=max(3, lookback // 2)).mean()
    return df["volume"] / avg.replace(0.0, np.nan)


def momentum_pct(df: pd.DataFrame, bars: int) -> pd.Series:
    bars = max(1, int(bars))
    base = df["close"].shift(bars)
    return (df["close"] - base) / base.replace(0.0, np.nan)


def consecutive_up(df: pd.DataFrame, bars: int) -> pd.Series:
    """True where the last `bars` candles each closed above the previous close.

    This is the 'not one noisy print' half of the momentum rule - a single
    violent green candle does not qualify on its own.
    """
    bars = max(1, int(bars))
    up = (df["close"] > df["close"].shift(1)).astype(float)
    return up.rolling(bars).sum() >= bars


# --------------------------------------------------------------------------
# Regime detection (spec 4.1)
# --------------------------------------------------------------------------
TRENDING, RANGING, CHOPPY = 0, 1, 2
REGIME_NAMES = {TRENDING: "trending", RANGING: "ranging", CHOPPY: "choppy"}
REGIME_BITS = {TRENDING: 1, RANGING: 2, CHOPPY: 4}


def efficiency_ratio(series: pd.Series, lookback: int = 20) -> pd.Series:
    """Net travel over gross travel - Kaufman's efficiency ratio.

    A market that moves 10% in a straight line and one that moves 10% net after
    whipsawing 40% look identical to a momentum rule and completely different to
    a trader. This separates them with one number and no extra thresholds, which
    is why it is the regime input rather than a tuned oscillator.
    """
    lookback = max(2, int(lookback))
    step = series.diff().abs().fillna(0.0)
    gross = step.rolling(lookback).sum()
    net = (series - series.shift(lookback)).abs()
    return net / gross.replace(0.0, np.nan)


def classify_regime(
    efficiency: pd.Series, atr_pct: pd.Series, *, trend_er: float, chop_atr_pct: float
) -> pd.Series:
    """Map efficiency and volatility onto trending / ranging / choppy.

    Directional efficiency splits trend from no-trend; volatility then splits
    no-trend into an orderly range and genuine chop. An efficiency reading that
    is not yet defined counts as chop, so the gate errs toward standing aside.
    """
    trending = efficiency >= float(trend_er)
    volatile = atr_pct > float(chop_atr_pct)
    out = np.where(trending, TRENDING, np.where(volatile, CHOPPY, RANGING))
    out = np.where(efficiency.isna().to_numpy(), CHOPPY, out)
    return pd.Series(out.astype(np.int8), index=efficiency.index)


def regime_allowed(regime: int, allowed_mask: int) -> bool:
    return bool(int(allowed_mask) & REGIME_BITS.get(int(regime), 0))


# --------------------------------------------------------------------------
# Multi-timeframe confluence (spec 4.2)
# --------------------------------------------------------------------------
def aggregate_trend(
    series: pd.Series, multiple: int, fast: int, slow: int
) -> pd.Series:
    """Is the ``multiple``-bar aggregate timeframe in an uptrend?

    The aggregate closing at bar ``k*m - 1`` is the higher-timeframe candle, and
    a base bar reads the last aggregate that has *closed* - never the one still
    forming. Reading the forming bar is the classic multi-timeframe mistake: it
    repaints, and a backtest that repaints reports entries the live bot could
    never have taken.
    """
    multiple = max(1, int(multiple))
    n = len(series)
    out = pd.Series(np.zeros(n, dtype=bool), index=series.index)
    n_agg = n // multiple
    if n_agg < 2:
        return out

    agg_close = series.iloc[multiple - 1 :: multiple].iloc[:n_agg].reset_index(drop=True)
    agg_up = (ema(agg_close, fast) > ema(agg_close, slow)).to_numpy()

    idx = ((np.arange(n, dtype=np.int64) + 1) // multiple) - 1
    have = idx >= 0
    out.iloc[:] = agg_up[np.clip(idx, 0, n_agg - 1)] & have
    return out


def confluence_count(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.Series:
    """How many configured aggregate timeframes agree the trend is up."""
    multiples = cfg.get("confluence_timeframes") or []
    total = pd.Series(np.zeros(len(df), dtype=np.int8), index=df.index)
    for multiple in multiples:
        total = total + aggregate_trend(
            df["close"], int(multiple), int(cfg["ema_fast"]), int(cfg["ema_slow"])
        ).astype(np.int8)
    return total


@dataclass(slots=True)
class Snapshot:
    """Indicator state at one bar - stored with a position to justify the entry."""

    ts: int
    close: float
    ema_fast: float
    ema_slow: float
    rsi: float
    atr: float
    atr_pct: float
    volume: float
    volume_ratio: float
    momentum_pct: float
    consecutive_up: bool
    bars: int
    efficiency: float = 0.0
    regime: int = CHOPPY
    confluence: int = 0

    @property
    def regime_name(self) -> str:
        return REGIME_NAMES.get(int(self.regime), "unknown")

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["regime_name"] = self.regime_name
        return out


def compute(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Attach every indicator column the strategy reads."""
    if df.empty:
        return df
    out = df.copy()
    out["ema_fast"] = ema(out["close"], cfg["ema_fast"])
    out["ema_slow"] = ema(out["close"], cfg["ema_slow"])
    out["rsi"] = rsi(out["close"], cfg["rsi_period"])
    out["atr"] = atr(out, cfg["atr_period"])
    out["atr_pct"] = out["atr"] / out["close"].replace(0.0, np.nan)
    out["vol_ratio"] = volume_ratio(out, cfg["volume_spike_lookback"])
    out["momentum"] = momentum_pct(out, cfg["momentum_candles"])
    out["consec_up"] = consecutive_up(out, cfg["momentum_candles"])
    out["efficiency"] = efficiency_ratio(out["close"], cfg.get("regime_lookback", 20))
    out["regime"] = classify_regime(
        out["efficiency"],
        out["atr_pct"],
        trend_er=cfg.get("regime_trend_er", 0.35),
        chop_atr_pct=cfg.get("regime_chop_atr_pct", 0.03),
    )
    out["confluence"] = confluence_count(out, cfg)
    return out


def snapshot_at(df: pd.DataFrame, index: int = -1) -> Snapshot | None:
    """Freeze the indicator state at one bar. `df` must already be computed."""
    if df.empty:
        return None
    try:
        row = df.iloc[index]
    except IndexError:
        return None

    def val(name: str, default: float = 0.0) -> float:
        v = row.get(name)
        try:
            f = float(v)
        except (TypeError, ValueError):
            return default
        return default if np.isnan(f) else f

    return Snapshot(
        ts=int(row["ts"]),
        close=val("close"),
        ema_fast=val("ema_fast"),
        ema_slow=val("ema_slow"),
        rsi=val("rsi", 50.0),
        atr=val("atr"),
        atr_pct=val("atr_pct"),
        volume=val("volume"),
        volume_ratio=val("vol_ratio"),
        momentum_pct=val("momentum"),
        consecutive_up=bool(row.get("consec_up", False)),
        bars=len(df),
        efficiency=val("efficiency"),
        regime=int(row.get("regime", CHOPPY)),
        confluence=int(row.get("confluence", 0)),
    )
