"""Entry and exit rules - pure technical analysis, no sentiment or news.

Entry requires all four spec conditions to hold at once:

1. a volume spike well above the token's *own* rolling average,
2. price momentum confirmed across several candles rather than one noisy print,
3. enough liquidity pool depth for the intended position size,
4. a clean Rug Check result (evaluated by :mod:`solbot.safety` before this runs).

Exit checks three independent conditions on every cycle, any one of which
closes the position: the hard stop, the trailing stop once it is armed, and
signal invalidation - the last being the main defence against capital sitting
idle in sideways chop.

Everything here is a pure function of a candle frame plus settings, so the
backtest and the live engine cannot drift apart.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .indicators import Snapshot, compute, snapshot_at


@dataclass(slots=True)
class EntrySignal:
    ok: bool
    mint: str = ""
    reasons: list[str] = field(default_factory=list)   # why it failed
    passed: list[str] = field(default_factory=list)    # which rules fired
    snapshot: Snapshot | None = None
    price: float = 0.0
    stop: float = 0.0
    target: float = 0.0
    rr: float = 0.0
    risk_per_unit: float = 0.0
    strength: float = 0.0     # 0-1, drives volatility-scaled sizing
    interest: bool = False    # early characteristics -> escalate to the hot tier

    def summary(self) -> str:
        return "; ".join(self.passed) if self.ok else "; ".join(self.reasons)


@dataclass(slots=True)
class ExitSignal:
    should_exit: bool
    reason: str = ""
    detail: dict[str, Any] = field(default_factory=dict)


def _finite(x: Any, default: float = 0.0) -> float:
    try:
        f = float(x)
    except (TypeError, ValueError):
        return default
    return default if (np.isnan(f) or np.isinf(f)) else f


# --------------------------------------------------------------------------
# Entry
# --------------------------------------------------------------------------
def evaluate_entry(
    df: pd.DataFrame,
    cfg: dict[str, Any],
    *,
    mint: str = "",
    liquidity_usd: float = 0.0,
    intended_size_usd: float = 0.0,
    precomputed: bool = False,
) -> EntrySignal:
    """Decide whether this token is a buy on the most recent closed candle."""
    if df is None or df.empty:
        return EntrySignal(False, mint, ["no candle data"])
    if len(df) < cfg["min_candles_required"]:
        return EntrySignal(
            False, mint, [f"only {len(df)} candles, need {cfg['min_candles_required']}"]
        )

    data = df if precomputed else compute(df, cfg)
    snap = snapshot_at(data, -1)
    if snap is None or snap.close <= 0:
        return EntrySignal(False, mint, ["indicator snapshot unavailable"])

    reasons: list[str] = []
    passed: list[str] = []

    # --- 1. volume spike --------------------------------------------------
    vol_ratio = _finite(snap.volume_ratio)
    vol_needed = float(cfg["volume_spike_multiple"])
    if vol_ratio >= vol_needed:
        passed.append(f"volume {vol_ratio:.1f}x its {cfg['volume_spike_lookback']}-bar average")
    else:
        reasons.append(f"volume only {vol_ratio:.1f}x average (need {vol_needed:.1f}x)")

    # --- 2. momentum confirmed across candles -----------------------------
    mom = _finite(snap.momentum_pct)
    mom_needed = float(cfg["momentum_min_pct"])
    trend_ok = snap.ema_fast > snap.ema_slow
    if mom >= mom_needed and snap.consecutive_up and trend_ok:
        passed.append(
            f"momentum +{mom * 100:.2f}% over {cfg['momentum_candles']} rising candles, "
            f"EMA{cfg['ema_fast']} above EMA{cfg['ema_slow']}"
        )
    else:
        if mom < mom_needed:
            reasons.append(f"momentum {mom * 100:.2f}% below {mom_needed * 100:.2f}%")
        if not snap.consecutive_up:
            reasons.append(f"not {cfg['momentum_candles']} consecutive rising candles")
        if not trend_ok:
            reasons.append("fast EMA below slow EMA")

    # --- overbought guard -------------------------------------------------
    if snap.rsi > float(cfg["rsi_max_entry"]):
        reasons.append(f"RSI {snap.rsi:.0f} above {cfg['rsi_max_entry']:.0f} (blow-off top)")
    else:
        passed.append(f"RSI {snap.rsi:.0f}")

    # --- 3. liquidity depth for the intended size -------------------------
    max_by_liq = liquidity_usd * float(cfg["max_position_pct_of_liquidity"])
    if liquidity_usd <= 0:
        reasons.append("liquidity unknown")
    elif liquidity_usd < float(cfg["min_liquidity_usd"]):
        reasons.append(
            f"liquidity ${liquidity_usd:,.0f} below floor ${cfg['min_liquidity_usd']:,.0f}"
        )
    elif intended_size_usd > 0 and intended_size_usd > max_by_liq:
        reasons.append(
            f"intended ${intended_size_usd:,.0f} exceeds "
            f"{cfg['max_position_pct_of_liquidity'] * 100:.1f}% of ${liquidity_usd:,.0f} pool"
        )
    else:
        passed.append(f"pool depth ${liquidity_usd:,.0f}")

    # --- "of interest": early characteristics worth a faster poll ---------
    # Deliberately looser than the entry gate. The point is to escalate a token
    # to the 1s tier while the signal is still forming, so the entry window is
    # not missed while it is being confirmed.
    interest = bool(
        vol_ratio >= max(1.2, vol_needed * 0.6)
        and (mom > 0 or snap.ema_fast > snap.ema_slow)
    )

    atr_val = _finite(snap.atr)
    if atr_val <= 0:
        reasons.append("ATR unavailable")

    if reasons:
        return EntrySignal(
            False, mint, reasons, passed, snap, snap.close, interest=interest
        )

    # --- stop / target sizing --------------------------------------------
    stop, target, rr, strength = _levels(snap, cfg)
    if stop <= 0 or stop >= snap.close:
        return EntrySignal(False, mint, ["computed stop is not below entry"], passed, snap)

    return EntrySignal(
        ok=True,
        mint=mint,
        reasons=[],
        passed=passed,
        snapshot=snap,
        price=snap.close,
        stop=stop,
        target=target,
        rr=rr,
        risk_per_unit=snap.close - stop,
        strength=strength,
        interest=True,
    )


def _levels(snap: Snapshot, cfg: dict[str, Any]) -> tuple[float, float, float, float]:
    """Hard stop, take-profit, reward:risk ratio and signal strength.

    The stop sits an ATR multiple below entry so it scales with the token's own
    volatility. The reward:risk ratio is then scaled *within* the configured
    bounds by how strong the signal is - never outside them, as the spec
    requires.
    """
    price = snap.close
    atr_val = max(_finite(snap.atr), price * 0.002)  # floor: 0.2% of price

    rr_min = float(cfg["rr_min"])
    rr_max = float(cfg["rr_max"])

    # Strength blends volume conviction with momentum, each capped so one
    # runaway input cannot dominate.
    vol_component = min(
        1.0,
        max(0.0, (_finite(snap.volume_ratio) - cfg["volume_spike_multiple"]))
        / max(cfg["volume_spike_multiple"], 1e-9),
    )
    mom_component = min(
        1.0, max(0.0, _finite(snap.momentum_pct)) / max(float(cfg["momentum_min_pct"]) * 3, 1e-9)
    )
    strength = max(0.0, min(1.0, 0.6 * vol_component + 0.4 * mom_component))

    rr = rr_min + (rr_max - rr_min) * strength
    rr = max(rr_min, min(rr_max, rr))

    stop_distance = atr_val * 1.2
    stop = price - stop_distance
    target = price + stop_distance * rr
    return stop, target, rr, strength


# --------------------------------------------------------------------------
# Exit
# --------------------------------------------------------------------------
def evaluate_exit(
    position: dict[str, Any],
    price: float,
    df: pd.DataFrame | None,
    cfg: dict[str, Any],
    *,
    now_ts: int,
    precomputed: bool = False,
) -> ExitSignal:
    """Three independent conditions; the first that fires closes the position."""
    entry = float(position["entry_price"])
    hard_stop = float(position["hard_stop"])
    trailing = position.get("trailing_stop")
    trailing = float(trailing) if trailing is not None else None

    if price <= 0:
        return ExitSignal(False)

    # --- 1. hard stop -----------------------------------------------------
    if price <= hard_stop:
        return ExitSignal(
            True,
            "hard stop hit",
            {"price": price, "stop": hard_stop},
        )

    # --- 2. trailing stop (only once armed) -------------------------------
    if trailing is not None and price <= trailing:
        return ExitSignal(
            True,
            "trailing stop hit",
            {"price": price, "trailing_stop": trailing},
        )

    # --- take profit ------------------------------------------------------
    take_profit = float(position.get("take_profit") or 0.0)
    if take_profit > 0 and price >= take_profit:
        return ExitSignal(
            True, "take-profit target reached", {"price": price, "target": take_profit}
        )

    # --- max hold ---------------------------------------------------------
    max_hold = int(cfg["max_hold_minutes"]) * 60
    held = now_ts - int(position["entry_ts"])
    if max_hold > 0 and held >= max_hold:
        return ExitSignal(
            True, f"max hold of {cfg['max_hold_minutes']}m reached", {"held_seconds": held}
        )

    # --- 3. signal invalidation ------------------------------------------
    if df is not None and not df.empty and len(df) >= cfg["min_candles_required"]:
        data = df if precomputed else compute(df, cfg)
        snap = snapshot_at(data, -1)
        if snap is not None:
            faded = _signal_faded(snap, cfg)
            if faded:
                count = int(position.get("invalidation_count", 0)) + 1
                if count >= int(cfg["signal_invalidation_bars"]):
                    return ExitSignal(
                        True,
                        f"signal invalidated ({faded})",
                        {"bars": count, "detail": faded},
                    )
                return ExitSignal(False, "", {"invalidation_count": count, "detail": faded})
            return ExitSignal(False, "", {"invalidation_count": 0})

    return ExitSignal(False)


def _signal_faded(snap: Snapshot, cfg: dict[str, Any]) -> str:
    """Describe how the entry thesis broke, or '' if it still holds."""
    problems = []
    if snap.ema_fast < snap.ema_slow:
        problems.append("trend flipped")
    if _finite(snap.volume_ratio) < 1.0:
        problems.append("volume dried up")
    if _finite(snap.momentum_pct) < 0:
        problems.append("momentum negative")
    # Two of three is a faded signal; one alone is noise.
    return ", ".join(problems) if len(problems) >= 2 else ""


def update_trailing_stop(
    position: dict[str, Any], price: float, atr_value: float, cfg: dict[str, Any]
) -> dict[str, Any]:
    """Advance the trailing stop. Returns the fields that changed.

    The trail only arms once the move covers round-trip costs, so a position is
    never stopped out at a nominal 'breakeven' that is actually a loss after
    fees and slippage.
    """
    entry = float(position["entry_price"])
    risk = float(position.get("initial_risk") or 0.0)
    if risk <= 0:
        return {}

    changes: dict[str, Any] = {}
    high_water = max(float(position.get("high_water_price") or entry), price)
    if high_water != position.get("high_water_price"):
        changes["high_water_price"] = high_water

    armed = bool(position.get("trailing_armed"))
    r_multiple = (price - entry) / risk

    # Round-trip cost as a fraction of entry: both fees plus assumed slippage.
    cost_pct = (float(cfg["taker_fee_pct"]) * 2 + float(cfg["max_slippage_pct"])) / 100.0
    breakeven = entry * (1.0 + cost_pct)

    if not armed and r_multiple >= float(cfg["trailing_activate_r"]) and price > breakeven:
        armed = True
        changes["trailing_armed"] = 1
        changes["trailing_stop"] = breakeven
        changes["_armed_now"] = True

    if armed:
        distance = max(atr_value * float(cfg["trailing_distance_atr"]), entry * 0.002)
        candidate = high_water - distance
        floor = max(breakeven, float(position.get("trailing_stop") or 0.0))
        new_stop = max(candidate, floor)
        if new_stop > float(position.get("trailing_stop") or 0.0):
            changes["trailing_stop"] = new_stop

    return changes
