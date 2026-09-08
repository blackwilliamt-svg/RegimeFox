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


def macd(series: pd.Series, fast: int, slow: int, signal: int) -> tuple[pd.Series, pd.Series]:
    """MACD line and its signal line - the fast/slow EMA spread, smoothed again."""
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    return macd_line, signal_line


def bollinger_bands(
    series: pd.Series, period: int, num_std: float
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Mid/upper/lower bands. Population std (ddof=0) - consistent with the
    vectorized twin, which computes it from a prefix sum rather than a
    per-window pass, and NaN before the window is filled either way."""
    period = max(2, int(period))
    mid = series.rolling(period, min_periods=period).mean()
    std = series.rolling(period, min_periods=period).std(ddof=0)
    upper = mid + num_std * std
    lower = mid - num_std * std
    return mid, upper, lower


def percent_b(series: pd.Series, upper: pd.Series, lower: pd.Series) -> pd.Series:
    """Where price sits inside the bands: 0 at the lower band, 1 at the upper."""
    span = (upper - lower).replace(0.0, np.nan)
    return (series - lower) / span


def stochastic(
    df: pd.DataFrame, k_period: int, d_period: int
) -> tuple[pd.Series, pd.Series]:
    """%K (close's position in the recent high/low range) and its %D average."""
    k_period = max(2, int(k_period))
    d_period = max(1, int(d_period))
    lowest_low = df["low"].rolling(k_period, min_periods=k_period).min()
    highest_high = df["high"].rolling(k_period, min_periods=k_period).max()
    span = (highest_high - lowest_low).replace(0.0, np.nan)
    k = 100.0 * (df["close"] - lowest_low) / span
    d = k.rolling(d_period, min_periods=d_period).mean()
    return k, d


def adx(df: pd.DataFrame, period: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Wilder's ADX, +DI and -DI: trend strength and its direction.

    Directional movement is the larger of the up-move / down-move, zeroed out
    when the other direction is larger or the move is negative - the standard
    Wilder definition, not a simplified variant.
    """
    period = max(2, int(period))
    up_move = df["high"].diff()
    down_move = -df["low"].diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0.0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0.0), 0.0)

    tr = true_range(df)
    alpha = 1.0 / period
    smoothed_tr = tr.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    smoothed_plus = plus_dm.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    smoothed_minus = minus_dm.ewm(alpha=alpha, adjust=False, min_periods=period).mean()

    with np.errstate(divide="ignore", invalid="ignore"):
        plus_di = 100.0 * smoothed_plus / smoothed_tr.replace(0.0, np.nan)
        minus_di = 100.0 * smoothed_minus / smoothed_tr.replace(0.0, np.nan)
        di_sum = (plus_di + minus_di).replace(0.0, np.nan)
        dx = 100.0 * (plus_di - minus_di).abs() / di_sum
    adx_line = dx.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    return adx_line, plus_di, minus_di


def vwap(df: pd.DataFrame, period: int) -> pd.Series:
    """Rolling volume-weighted average price over the trailing `period` bars.

    A crypto pair trades continuously, so there is no session boundary to reset
    a running VWAP against - a rolling window is the honest equivalent.
    """
    period = max(2, int(period))
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = typical * df["volume"]
    num = pv.rolling(period, min_periods=period).sum()
    den = df["volume"].rolling(period, min_periods=period).sum().replace(0.0, np.nan)
    return num / den


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


# --------------------------------------------------------------------------
# Indicator-combination search (gap-closure item 3)
#
# Nine independent entry-signal contributors the walk-forward optimizer can
# include or exclude per parameter bundle: the five the spec named already
# (volume spike, momentum, RSI, EMA cross - ATR is structural, it sizes the
# stop rather than voting on entry) plus five more popular indicators. Entry
# requires `indicator_min_agree` of the *active* (indicator_mask) votes to be
# bullish - the same AND/vote mechanism confluence_required already uses for
# higher-timeframe agreement, not a new one.
#
# The legacy default (mask = the first four bits, min_agree = 4) reproduces
# exactly the pre-item-3 hard-AND of volume+momentum+RSI+EMA, so a config
# that has never touched these two keys behaves identically to before.
# --------------------------------------------------------------------------
IND_VOLUME_SPIKE = 1 << 0
IND_MOMENTUM = 1 << 1
IND_RSI = 1 << 2
IND_EMA_CROSS = 1 << 3
IND_MACD = 1 << 4
IND_BBANDS = 1 << 5
IND_STOCHASTIC = 1 << 6
IND_ADX = 1 << 7
IND_VWAP = 1 << 8

INDICATOR_BITS: dict[str, int] = {
    "volume_spike": IND_VOLUME_SPIKE,
    "momentum": IND_MOMENTUM,
    "rsi": IND_RSI,
    "ema_cross": IND_EMA_CROSS,
    "macd": IND_MACD,
    "bbands": IND_BBANDS,
    "stochastic": IND_STOCHASTIC,
    "adx": IND_ADX,
    "vwap": IND_VWAP,
}
ALL_INDICATOR_BITS = sum(INDICATOR_BITS.values())
LEGACY_INDICATOR_MASK = IND_VOLUME_SPIKE | IND_MOMENTUM | IND_RSI | IND_EMA_CROSS


def popcount(mask: int) -> int:
    return bin(int(mask) & 0xFFFFFFFF).count("1")


@dataclass(slots=True)
class IndicatorGateResult:
    """Which of the active (indicator_mask) indicators agreed - the explicit
    pass/fail detail an explainability panel or a test can inspect, rather
    than just a bool."""

    ok: bool
    active_count: int
    min_agree: int
    agreed: list[str]
    disagreed: list[str]
    votes: dict[str, bool]


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
    macd_line: float = 0.0
    macd_signal: float = 0.0
    bb_percent: float = 0.5
    stoch_k: float = 50.0
    stoch_d: float = 50.0
    adx: float = 0.0
    plus_di: float = 0.0
    minus_di: float = 0.0
    vwap: float = 0.0

    @property
    def regime_name(self) -> str:
        return REGIME_NAMES.get(int(self.regime), "unknown")

    def indicator_votes(self, cfg: dict[str, Any]) -> dict[str, bool]:
        """Bullish/bearish reading of each of the nine combinable indicators,
        independent of whether `indicator_mask` currently includes it - the
        gate below is what applies the mask, this is what makes every vote
        inspectable (e.g. for a dashboard explainability panel)."""
        return {
            "volume_spike": self.volume_ratio >= float(cfg["volume_spike_multiple"]),
            "momentum": (
                self.momentum_pct >= float(cfg["momentum_min_pct"]) and self.consecutive_up
            ),
            "rsi": self.rsi <= float(cfg["rsi_max_entry"]),
            "ema_cross": self.ema_fast > self.ema_slow,
            "macd": self.macd_line > self.macd_signal,
            "bbands": self.bb_percent > float(cfg.get("bb_bullish_pct", 0.5)),
            "stochastic": (
                self.stoch_k > self.stoch_d
                and self.stoch_k < float(cfg.get("stoch_overbought", 80.0))
            ),
            "adx": self.adx >= float(cfg.get("adx_min", 20.0)) and self.plus_di > self.minus_di,
            "vwap": self.close > self.vwap,
        }

    def indicator_gate(self, cfg: dict[str, Any]) -> "IndicatorGateResult":
        """Apply `indicator_mask` / `indicator_min_agree` to the votes above."""
        mask = int(cfg.get("indicator_mask", LEGACY_INDICATOR_MASK))
        min_agree = int(cfg.get("indicator_min_agree", popcount(LEGACY_INDICATOR_MASK)))
        votes = self.indicator_votes(cfg)
        active = {name: v for name, v in votes.items() if mask & INDICATOR_BITS[name]}
        agree = [name for name, v in active.items() if v]
        disagree = [name for name in active if name not in agree]
        return IndicatorGateResult(
            ok=len(agree) >= min_agree,
            active_count=len(active),
            min_agree=min_agree,
            agreed=agree,
            disagreed=disagree,
            votes=votes,
        )

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

    macd_line, macd_signal = macd(
        out["close"],
        int(cfg.get("macd_fast", 12)), int(cfg.get("macd_slow", 26)),
        int(cfg.get("macd_signal", 9)),
    )
    out["macd_line"], out["macd_signal"] = macd_line, macd_signal

    _bb_mid, bb_upper, bb_lower = bollinger_bands(
        out["close"], int(cfg.get("bb_period", 20)), float(cfg.get("bb_std", 2.0))
    )
    out["bb_percent"] = percent_b(out["close"], bb_upper, bb_lower)

    stoch_k, stoch_d = stochastic(
        out, int(cfg.get("stoch_k_period", 14)), int(cfg.get("stoch_d_period", 3))
    )
    out["stoch_k"], out["stoch_d"] = stoch_k, stoch_d

    adx_line, plus_di, minus_di = adx(out, int(cfg.get("adx_period", 14)))
    out["adx"], out["plus_di"], out["minus_di"] = adx_line, plus_di, minus_di

    out["vwap"] = vwap(out, int(cfg.get("vwap_period", 20)))
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
        macd_line=val("macd_line"),
        macd_signal=val("macd_signal"),
        bb_percent=val("bb_percent", 0.5),
        stoch_k=val("stoch_k", 50.0),
        stoch_d=val("stoch_d", 50.0),
        adx=val("adx"),
        plus_di=val("plus_di"),
        minus_di=val("minus_di"),
        vwap=val("vwap"),
    )
