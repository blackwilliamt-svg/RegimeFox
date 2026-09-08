"""The five strategy enhancements, on the droplet side.

Regime gating and multi-timeframe confluence decide whether an entry is allowed
to fire at all; correlation-aware sizing decides how large it is; drift
monitoring decides whether any of it is still working. Each is tested for the
thing it is supposed to prevent, not just for running without error.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from solbot import db, drift, risk
from solbot.indicators import (
    CHOPPY,
    RANGING,
    TRENDING,
    aggregate_trend,
    classify_regime,
    compute,
    efficiency_ratio,
    regime_allowed,
    snapshot_at,
)
from solbot.strategy import evaluate_entry
from test_strategy import make_candles, with_spike


# --------------------------------------------------------------------------
# 4.1 regime detection
# --------------------------------------------------------------------------
def test_efficiency_ratio_separates_a_trend_from_a_whipsaw():
    """The point of the measure: same net move, very different journeys."""
    n = 60
    straight = pd.Series(np.linspace(100.0, 110.0, n))
    zigzag = pd.Series(
        [100.0 + (10.0 * i / (n - 1)) + (3.0 if i % 2 else -3.0) for i in range(n)]
    )

    assert straight.iloc[-1] == pytest.approx(zigzag.iloc[-1], rel=0.05)
    assert efficiency_ratio(straight, 20).iloc[-1] > 0.9
    assert efficiency_ratio(zigzag, 20).iloc[-1] < 0.3


def test_regime_classification_covers_all_three_states():
    efficiency = pd.Series([0.8, 0.1, 0.1, np.nan])
    atr_pct = pd.Series([0.02, 0.01, 0.09, 0.01])
    got = classify_regime(efficiency, atr_pct, trend_er=0.35, chop_atr_pct=0.03)

    assert list(got) == [TRENDING, RANGING, CHOPPY, CHOPPY]


def test_an_undefined_efficiency_reading_counts_as_chop():
    """Not enough history to judge must mean "stand aside", not "trade freely"."""
    got = classify_regime(
        pd.Series([np.nan]), pd.Series([0.001]), trend_er=0.35, chop_atr_pct=0.03
    )
    assert got.iloc[0] == CHOPPY


def test_regime_bitmask_gates_the_states_it_should():
    assert regime_allowed(TRENDING, 3) and regime_allowed(RANGING, 3)
    assert not regime_allowed(CHOPPY, 3), "the default must exclude chop"
    assert regime_allowed(CHOPPY, 7)
    assert not regime_allowed(RANGING, 1)


def test_regime_gate_blocks_an_otherwise_valid_entry(settings):
    """A textbook setup in a regime the operator excluded must not fire."""
    df = with_spike(make_candles(60))
    permissive = {
        **settings, "confluence_enabled": False, "regime_gate_enabled": True,
        "regime_allowed": 7,
    }
    assert evaluate_entry(df, permissive, mint="X", liquidity_usd=500_000).ok

    snapshot = snapshot_at(compute(df, permissive), -1)
    excluded = {**permissive, "regime_allowed": 7 & ~(1 << snapshot.regime)}
    signal = evaluate_entry(df, excluded, mint="X", liquidity_usd=500_000)

    assert not signal.ok
    assert any("not an allowed regime" in r for r in signal.reasons)


# --------------------------------------------------------------------------
# 4.2 multi-timeframe confluence
# --------------------------------------------------------------------------
def test_aggregate_trend_never_reads_a_forming_higher_timeframe_bar():
    """Reading the bar still forming would repaint and inflate the backtest."""
    rising = pd.Series(np.linspace(100.0, 200.0, 90))
    trend = aggregate_trend(rising, 6, 3, 6)

    # Nothing is known until the first aggregate bar has closed.
    assert not trend.iloc[:5].any()
    assert trend.iloc[-1]

    # Changing only the bars inside the currently-forming aggregate must not
    # move any earlier reading.
    tampered = rising.copy()
    tampered.iloc[-3:] = 1.0
    assert (aggregate_trend(tampered, 6, 3, 6).iloc[:-3] == trend.iloc[:-3]).all()


def test_confluence_gate_blocks_a_spike_the_higher_timeframes_do_not_confirm(settings):
    """A three-bar pump on a flat base is exactly what this is meant to refuse."""
    df = with_spike(make_candles(60))
    cfg = {
        **settings, "regime_gate_enabled": False, "confluence_enabled": True,
        "confluence_timeframes": [3, 6], "confluence_required": 1,
    }
    signal = evaluate_entry(df, cfg, mint="X", liquidity_usd=500_000)

    assert not signal.ok
    assert any("higher timeframes agree" in r for r in signal.reasons)

    # With the requirement lifted the same setup fires, so it is the gate doing
    # the work rather than some unrelated rule rejecting the fixture.
    lifted = evaluate_entry(
        df, {**cfg, "confluence_required": 0}, mint="X", liquidity_usd=500_000
    )
    assert lifted.ok, lifted.reasons


def test_confluence_requirement_cannot_exceed_the_timeframes_configured(cfg):
    from solbot.config import ConfigError

    with pytest.raises(ConfigError, match="could ever satisfy"):
        cfg.update({"confluence_timeframes": "3", "confluence_required": 2})


# --------------------------------------------------------------------------
# 4.4 correlation-aware portfolio sizing
# --------------------------------------------------------------------------
def test_portfolio_mode_matches_flat_mode_for_a_single_position(settings):
    """Switching modes must change nothing until a second position is open.

    The portfolio volatility target is calibrated so a lone position at the
    target ATR gets exactly the flat-mode size; anything else would make the new
    mode a silent risk change rather than a refinement.
    """
    for atr_pct in (0.015, 0.03, 0.06, 0.12):
        flat = risk.size_position(
            wallet_usd=1000.0, liquidity_usd=1e7, price=1.0, atr_pct=atr_pct,
            cfg={**settings, "sizing_mode": "flat"},
        )
        portfolio = risk.size_position(
            wallet_usd=1000.0, liquidity_usd=1e7, price=1.0, atr_pct=atr_pct,
            cfg={**settings, "sizing_mode": "portfolio"},
        )
        assert portfolio.size_usd == pytest.approx(flat.size_usd, rel=0.02), (
            f"modes diverge at ATR {atr_pct}"
        )


def test_a_correlated_second_position_is_sized_down(settings):
    cfg = {**settings, "sizing_mode": "portfolio"}
    open_at = 450.0

    independent = risk.size_position(
        wallet_usd=1000.0, liquidity_usd=1e7, price=1.0, atr_pct=0.03, cfg=cfg,
        deployed_usd=open_at,
        open_book=[risk.OpenExposure(size_usd=open_at, atr_pct=0.03, correlation=0.0)],
    )
    correlated = risk.size_position(
        wallet_usd=1000.0, liquidity_usd=1e7, price=1.0, atr_pct=0.03, cfg=cfg,
        deployed_usd=open_at,
        open_book=[risk.OpenExposure(size_usd=open_at, atr_pct=0.03, correlation=0.9)],
    )

    assert correlated.size_usd < independent.size_usd, (
        "a position riding the open one adds more portfolio risk and must be smaller"
    )


def test_an_uncomputable_correlation_is_treated_as_fully_correlated(settings):
    """Sizing down on a token we cannot measure is the safe direction to err."""
    cfg = {**settings, "sizing_mode": "portfolio"}
    unknown = risk.size_position(
        wallet_usd=1000.0, liquidity_usd=1e7, price=1.0, atr_pct=0.03, cfg=cfg,
        deployed_usd=450.0,
        open_book=[risk.OpenExposure(450.0, 0.03, correlation=1.0)],
    )
    known_independent = risk.size_position(
        wallet_usd=1000.0, liquidity_usd=1e7, price=1.0, atr_pct=0.03, cfg=cfg,
        deployed_usd=450.0,
        open_book=[risk.OpenExposure(450.0, 0.03, correlation=0.0)],
    )
    assert unknown.size_usd < known_independent.size_usd


def test_hard_caps_still_bind_in_portfolio_mode(settings):
    """The new mode may only size down. The spec's caps are not negotiable."""
    cfg = {**settings, "sizing_mode": "portfolio", "portfolio_vol_target": 10.0}
    result = risk.size_position(
        wallet_usd=1000.0, liquidity_usd=1e7, price=1.0, atr_pct=0.0001, cfg=cfg
    )
    assert result.size_usd <= 1000.0 * settings["max_position_pct_of_wallet"] + 1e-9


def test_monte_carlo_tail_shrinks_the_risk_budget(settings):
    """The resampled 5% drawdown feeds sizing; the historical one does not.

    The shrink halves the *portfolio volatility budget*, which then becomes the
    binding cap - so the resulting size is the halved budget divided by the
    token's volatility, not half of whatever the previous size happened to be.
    """
    base = {
        **settings, "sizing_mode": "portfolio", "drawdown_tolerance": 0.25,
        "portfolio_vol_target": 0.019,
    }
    untested = risk.size_position(
        wallet_usd=1000.0, liquidity_usd=1e7, price=1.0, atr_pct=0.06, cfg=base
    )
    alarming = risk.size_position(
        wallet_usd=1000.0, liquidity_usd=1e7, price=1.0, atr_pct=0.06,
        cfg={**base, "monte_carlo_p5_drawdown": 0.50},
    )

    assert alarming.size_usd < untested.size_usd
    assert alarming.size_usd == pytest.approx(1000.0 * (0.019 * 0.5) / 0.06, rel=1e-6)
    assert alarming.capped_by == "portfolio_volatility"
    assert any("worst-case drawdown" in r for r in alarming.reasons)


def test_the_tail_scales_the_budget_proportionally_once_it_binds(settings):
    """Where the budget is the binding cap, twice the tail is half the size."""
    base = {
        **settings, "sizing_mode": "portfolio", "drawdown_tolerance": 0.25,
        "portfolio_vol_target": 0.019,
    }
    mild = risk.size_position(
        wallet_usd=1000.0, liquidity_usd=1e7, price=1.0, atr_pct=0.06,
        cfg={**base, "monte_carlo_p5_drawdown": 0.50},
    )
    severe = risk.size_position(
        wallet_usd=1000.0, liquidity_usd=1e7, price=1.0, atr_pct=0.06,
        cfg={**base, "monte_carlo_p5_drawdown": 1.00},
    )

    assert mild.capped_by == severe.capped_by == "portfolio_volatility"
    assert severe.size_usd == pytest.approx(mild.size_usd * 0.5, rel=1e-6)


def test_drawdown_scalar_is_inert_below_tolerance():
    assert risk.drawdown_scalar(0.0, 0.25) == 1.0     # never measured
    assert risk.drawdown_scalar(0.20, 0.25) == 1.0    # inside tolerance
    assert risk.drawdown_scalar(0.50, 0.25) == pytest.approx(0.5)
    assert risk.drawdown_scalar(5.0, 0.25) == 0.25    # floored, never zero


# --------------------------------------------------------------------------
# 4.5 drift monitoring
# --------------------------------------------------------------------------
def seed_backtest(conn, *, win_rate: float, avg_win: float, avg_loss: float) -> None:
    conn.execute(
        "INSERT INTO backtest_runs(ts, trades, win_rate, profit_factor, max_drawdown, "
        "total_return, avg_win, avg_loss) VALUES (?,?,?,?,?,?,?,?)",
        (db.now(), 120, win_rate, 1.8, 0.1, 0.2, avg_win, avg_loss),
    )


def seed_trades(conn, instance: str, pnls: list[float]) -> None:
    now = db.now()
    for i, pnl in enumerate(pnls):
        conn.execute(
            "INSERT INTO trades(instance, mint, symbol, entry_ts, entry_price, exit_ts, "
            "exit_price, qty, size_usd, proceeds_usd, fees_usd, pnl_usd, pnl_pct) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                instance, "M", "M", now - 3600 * (i + 2), 1.0, now - 3600 * (i + 1),
                1.0, 100.0, 100.0, 100.0 + pnl, 0.5, pnl, pnl / 100.0,
            ),
        )


def test_drift_is_insufficient_without_a_backtest_to_compare_against(workspace, settings):
    seed_trades(workspace["conn"], "paper", [5.0] * 40)
    sample = drift.measure("paper", settings, conn=workspace["conn"])

    assert sample.status == drift.STATUS_INSUFFICIENT
    assert "No backtest" in sample.message


def test_drift_is_insufficient_with_too_few_live_trades(workspace, settings):
    conn = workspace["conn"]
    seed_backtest(conn, win_rate=0.55, avg_win=10.0, avg_loss=-5.0)
    seed_trades(conn, "paper", [5.0] * 3)

    sample = drift.measure("paper", settings, conn=conn)
    assert sample.status == drift.STATUS_INSUFFICIENT


def test_matching_performance_reads_as_ok(workspace, settings):
    conn = workspace["conn"]
    seed_backtest(conn, win_rate=0.50, avg_win=10.0, avg_loss=-5.0)
    # Expectancy 0.5*10 + 0.5*-5 = 2.50; reproduce it exactly.
    seed_trades(conn, "paper", [10.0, -5.0] * 20)

    sample = drift.measure("paper", settings, conn=conn)
    assert sample.status == drift.STATUS_OK
    assert sample.live_expectancy == pytest.approx(2.5)


def test_a_collapsed_edge_is_flagged_as_drifting(workspace, settings):
    """The failure mode this exists to catch: still winning, earning far less."""
    conn = workspace["conn"]
    seed_backtest(conn, win_rate=0.50, avg_win=10.0, avg_loss=-5.0)
    seed_trades(conn, "paper", [1.0, -0.9] * 20)

    sample = drift.measure("paper", settings, conn=conn)
    assert sample.status == drift.STATUS_DRIFTING
    assert "drifting" in sample.message


def test_a_collapsed_win_rate_is_flagged_as_drifting(workspace, settings):
    conn = workspace["conn"]
    seed_backtest(conn, win_rate=0.60, avg_win=10.0, avg_loss=-5.0)
    seed_trades(conn, "paper", [10.0] + [-5.0] * 39)

    sample = drift.measure("paper", settings, conn=conn)
    assert sample.status == drift.STATUS_DRIFTING
    assert abs(sample.win_rate_gap) > settings["drift_win_rate_tolerance"]


def test_drift_samples_are_recorded_and_a_status_change_logs_once(workspace, settings):
    conn = workspace["conn"]
    seed_backtest(conn, win_rate=0.50, avg_win=10.0, avg_loss=-5.0)
    seed_trades(conn, "paper", [1.0, -0.9] * 20)

    drift.record(drift.measure("paper", settings, conn=conn), conn)
    drift.record(drift.measure("paper", settings, conn=conn), conn)

    rows = conn.execute("SELECT * FROM drift_samples").fetchall()
    assert len(rows) == 2, "every measurement is stored"

    events = conn.execute(
        "SELECT * FROM events WHERE category = 'system'"
    ).fetchall()
    assert len(events) == 1, "but only the transition is announced"
