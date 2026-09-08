"""Position sizing, correlation, circuit breaker, kill switch."""
from __future__ import annotations

import numpy as np
import pytest

from solbot import db, risk


# --------------------------------------------------------------------------
# sizing
# --------------------------------------------------------------------------
def test_size_capped_by_wallet_percentage(settings):
    out = risk.size_position(
        wallet_usd=1000.0, liquidity_usd=10_000_000.0, price=1.0,
        atr_pct=settings["volatility_target_atr_pct"], cfg=settings,
    )
    assert out.ok
    assert out.size_usd == pytest.approx(450.0)   # 45% of 1000
    assert out.capped_by == "wallet"


def test_size_capped_by_pool_depth(settings):
    """1% of a shallow pool must beat the wallet percentage."""
    out = risk.size_position(
        wallet_usd=100_000.0, liquidity_usd=60_000.0, price=1.0,
        atr_pct=settings["volatility_target_atr_pct"], cfg=settings,
    )
    assert out.capped_by == "liquidity"
    assert out.size_usd == pytest.approx(600.0)   # 1% of 60k


def test_volatility_scales_size_down(settings):
    """A token twice as volatile as target gets roughly half the size."""
    calm = risk.size_position(
        wallet_usd=1000.0, liquidity_usd=10_000_000.0, price=1.0,
        atr_pct=settings["volatility_target_atr_pct"], cfg=settings,
    )
    wild = risk.size_position(
        wallet_usd=1000.0, liquidity_usd=10_000_000.0, price=1.0,
        atr_pct=settings["volatility_target_atr_pct"] * 2, cfg=settings,
    )
    assert wild.size_usd == pytest.approx(calm.size_usd * 0.5, rel=0.01)
    assert wild.volatility_scalar == pytest.approx(0.5, rel=0.01)


def test_volatility_scalar_respects_floor(settings):
    scalar = risk.volatility_scalar(atr_pct=10.0, cfg=settings)
    assert scalar == settings["volatility_size_floor"]


def test_gas_reserve_is_excluded_from_sizing(settings):
    """The reserve comes off the top - a position can never dip into it."""
    settings = {**settings, "gas_reserve_sol": 1.0}
    out = risk.size_position(
        wallet_usd=1000.0, liquidity_usd=10_000_000.0, price=1.0,
        atr_pct=settings["volatility_target_atr_pct"], cfg=settings,
        sol_balance=5.0, sol_price=100.0,
    )
    # $100 reserved, so 45% of $900, not of $1000.
    assert out.size_usd == pytest.approx(405.0)


def test_refuses_when_entire_balance_is_reserve(settings):
    settings = {**settings, "gas_reserve_sol": 1.0}
    out = risk.size_position(
        wallet_usd=50.0, liquidity_usd=10_000_000.0, price=1.0,
        atr_pct=0.03, cfg=settings, sol_balance=1.0, sol_price=100.0,
    )
    assert not out.ok
    assert out.capped_by == "gas_reserve"


def test_deployment_never_exceeds_the_total_deployed_cap(settings):
    """A second position cannot exceed what the total-deployed budget has left.

    There is no position-count ceiling (spec 4): the budget is a percentage of
    the wallet, not a fixed number of slots.
    """
    out = risk.size_position(
        wallet_usd=1000.0, liquidity_usd=10_000_000.0, price=1.0,
        atr_pct=settings["volatility_target_atr_pct"], cfg=settings,
        deployed_usd=450.0,
    )
    assert out.size_usd <= 450.0
    total = 450.0 + out.size_usd
    assert total <= 1000.0 * settings["max_total_deployed_pct"] + 1e-6


def test_a_third_and_fourth_position_can_still_open_if_budget_remains(settings):
    """No count ceiling: sizing keeps succeeding as long as the budget has room."""
    cfg = {**settings, "max_position_pct_of_wallet": 0.30, "max_total_deployed_pct": 0.90}
    deployed = 0.0
    opened = 0
    for _ in range(6):
        out = risk.size_position(
            wallet_usd=1000.0, liquidity_usd=10_000_000.0, price=1.0,
            atr_pct=cfg["volatility_target_atr_pct"], cfg=cfg, deployed_usd=deployed,
        )
        if not out.ok:
            break
        deployed += out.size_usd
        opened += 1
    assert opened >= 3, "a 30%-per-position cap under a 90% total should allow at least 3"
    assert deployed <= 1000.0 * 0.90 + 1e-6


def test_rejects_size_below_minimum(settings):
    out = risk.size_position(
        wallet_usd=10.0, liquidity_usd=10_000_000.0, price=1.0,
        atr_pct=0.03, cfg=settings,
    )
    assert not out.ok
    assert "minimum" in out.reasons[-1]


# --------------------------------------------------------------------------
# correlation
# --------------------------------------------------------------------------
def test_correlation_detects_identical_series():
    series = list(np.cumprod(1 + np.random.default_rng(1).normal(0, 0.01, 80)))
    assert risk.correlation(series, series) == pytest.approx(1.0, abs=1e-6)


def test_correlation_gate_blocks_a_duplicate_move(settings):
    rng = np.random.default_rng(3)
    base = list(np.cumprod(1 + rng.normal(0, 0.01, 80)))
    gate = risk.correlation_gate(base, {"MINT_A": base}, settings)
    assert not gate.allowed
    assert "correlation" in gate.reason


def test_correlation_gate_allows_independent_moves(settings):
    rng = np.random.default_rng(4)
    a = list(np.cumprod(1 + rng.normal(0, 0.01, 200)))
    b = list(np.cumprod(1 + rng.normal(0, 0.01, 200)))
    gate = risk.correlation_gate(a, {"MINT_B": b}, settings)
    assert gate.allowed


def test_correlation_returns_none_on_short_series():
    assert risk.correlation([1, 2, 3], [1, 2, 3]) is None


# --------------------------------------------------------------------------
# circuit breaker + kill switch
# --------------------------------------------------------------------------
def test_circuit_trips_after_consecutive_losses(workspace, settings):
    conn = workspace["conn"]
    for _ in range(settings["circuit_consecutive_losses"] - 1):
        state = risk.record_trade_result(-10.0, settings, conn)
        assert not state["tripped"]
    state = risk.record_trade_result(-10.0, settings, conn)
    assert state["tripped"]
    assert risk.circuit_tripped(conn)


def test_a_win_resets_the_loss_streak(workspace, settings):
    conn = workspace["conn"]
    risk.record_trade_result(-10.0, settings, conn)
    risk.record_trade_result(-10.0, settings, conn)
    state = risk.record_trade_result(+5.0, settings, conn)
    assert state["consecutive_losses"] == 0
    assert not state.get("tripped")


def test_circuit_requires_manual_reset(workspace, settings):
    conn = workspace["conn"]
    for _ in range(settings["circuit_consecutive_losses"]):
        risk.record_trade_result(-10.0, settings, conn)
    assert risk.circuit_tripped(conn)

    # A subsequent win must NOT auto-resume trading.
    risk.record_trade_result(+50.0, settings, conn)
    assert risk.circuit_tripped(conn)

    risk.reset_circuit("tester", conn)
    assert not risk.circuit_tripped(conn)


def test_daily_drawdown_trips_the_breaker(workspace, settings):
    conn = workspace["conn"]
    risk.check_daily_drawdown("paper", 1000.0, settings, conn)     # sets the peak
    assert not risk.circuit_tripped(conn)
    tripped = risk.check_daily_drawdown("paper", 880.0, settings, conn)  # -12%
    assert tripped
    assert risk.circuit_tripped(conn)


def test_drawdown_peak_survives_a_restart(workspace, settings):
    """The high-water mark is persisted, so a restart cannot reset it."""
    conn = workspace["conn"]
    risk.check_daily_drawdown("paper", 1000.0, settings, conn)
    anchor = db.kv_get(risk.KEY_DAY_ANCHOR, {}, conn)
    assert anchor["peak"] == 1000.0
    # Simulate a restart reading the persisted anchor rather than starting fresh.
    tripped = risk.check_daily_drawdown("paper", 890.0, settings, conn)
    assert tripped


def test_kill_switch_blocks_new_positions(workspace, settings):
    conn = workspace["conn"]
    gate = risk.can_open_position(cfg=settings, conn=conn)
    assert gate.allowed

    risk.engage_kill_switch("testing", "tester", conn)
    gate = risk.can_open_position(cfg=settings, conn=conn)
    assert not gate.allowed
    assert "kill switch" in gate.reason

    risk.release_kill_switch("tester", conn)
    assert risk.can_open_position(cfg=settings, conn=conn).allowed


def test_open_position_gate_has_no_position_count_ceiling(workspace, settings):
    """Concurrency is bounded by the total-deployed budget, not a fixed count."""
    gate = risk.can_open_position(cfg=settings, conn=workspace["conn"])
    assert gate.allowed
