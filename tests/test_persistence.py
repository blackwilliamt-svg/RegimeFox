"""State persistence, crash recovery and on-chain reconciliation.

The spec's hardest requirement: the bot must never lose track of an open
position because the process died or the droplet rebooted, and it must halt
rather than guess when the database and the wallet disagree.
"""
from __future__ import annotations

import pytest

from solbot import db, risk
from solbot.datastore import DataStore
from solbot.execution import Fill
from solbot.portfolio import Portfolio
from solbot.recovery import (
    KEY_RECONCILE_HALT, _compare_wallet, clear_reconcile_halt, recover, reconcile_halted,
)

from fakes import fake_clients, token


MINT = "TokenMintAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"


def open_a_position(portfolio: Portfolio, conn, *, price: float = 1.0, qty: float = 100.0):
    fill = Fill(True, "buy", MINT, price=price, qty=qty, gross_usd=price * qty,
                fee_usd=0.25, slippage_pct=0.1)
    return portfolio.open_position(
        mint=MINT, symbol="TEST", fill=fill, stop=price * 0.95, target=price * 1.15,
        rr=3.0, entry_reason="test entry", snapshot={"rsi": 60}, conn=conn,
    )


# --------------------------------------------------------------------------
# position lifecycle
# --------------------------------------------------------------------------
def test_open_position_persists_immediately(workspace, settings):
    conn = workspace["conn"]
    p = Portfolio("paper", settings)
    start = p.balance(conn)

    position_id = open_a_position(p, conn)

    row = conn.execute("SELECT * FROM positions WHERE id = ?", (position_id,)).fetchone()
    assert row["status"] == "open"
    assert row["mint"] == MINT
    assert row["hard_stop"] == pytest.approx(0.95)
    assert row["initial_risk"] == pytest.approx(0.05)
    assert p.balance(conn) == pytest.approx(start - 100.0)


def test_every_stop_advance_is_written_through(workspace, settings):
    conn = workspace["conn"]
    p = Portfolio("paper", settings)
    position_id = open_a_position(p, conn)

    p.update_position(position_id, {"trailing_stop": 1.02, "trailing_armed": 1}, conn)

    row = conn.execute("SELECT * FROM positions WHERE id = ?", (position_id,)).fetchone()
    assert row["trailing_stop"] == pytest.approx(1.02)
    assert row["trailing_armed"] == 1


def test_close_writes_a_journal_row_and_credits_the_balance(workspace, settings):
    conn = workspace["conn"]
    p = Portfolio("paper", settings)
    position_id = open_a_position(p, conn)
    position = conn.execute("SELECT * FROM positions WHERE id = ?", (position_id,)).fetchone()

    fill = Fill(True, "sell", MINT, price=1.20, qty=100.0, gross_usd=120.0, fee_usd=0.3)
    closed = p.close_position(position, fill, "take-profit target reached", conn=conn)

    assert closed.pnl_usd == pytest.approx(120.0 - 0.3 - 100.0)
    trade = conn.execute("SELECT * FROM trades WHERE position_id = ?", (position_id,)).fetchone()
    assert trade is not None
    assert trade["exit_reason"] == "take-profit target reached"
    assert trade["fees_usd"] == pytest.approx(0.55)   # entry 0.25 + exit 0.30
    assert p.open_count(conn) == 0


def test_manual_close_is_tagged_in_the_journal(workspace, settings):
    conn = workspace["conn"]
    p = Portfolio("paper", settings)
    position_id = open_a_position(p, conn)
    position = conn.execute("SELECT * FROM positions WHERE id = ?", (position_id,)).fetchone()

    fill = Fill(True, "sell", MINT, price=1.10, qty=100.0, gross_usd=110.0, fee_usd=0.3)
    p.close_position(position, fill, "manual close by tester", manual=True, conn=conn)

    trade = conn.execute("SELECT * FROM trades WHERE position_id = ?", (position_id,)).fetchone()
    assert trade["manual"] == 1


def test_performance_metrics(workspace, settings):
    conn = workspace["conn"]
    p = Portfolio("paper", settings)
    for exit_price in (1.30, 0.90, 1.20, 0.95):
        position_id = open_a_position(p, conn)
        position = conn.execute(
            "SELECT * FROM positions WHERE id = ?", (position_id,)
        ).fetchone()
        fill = Fill(True, "sell", MINT, price=exit_price, qty=100.0,
                    gross_usd=exit_price * 100, fee_usd=0.25)
        p.close_position(position, fill, "test", conn=conn)

    perf = p.performance(conn=conn)
    assert perf["trades"] == 4
    assert perf["wins"] == 2
    assert perf["losses"] == 2
    assert perf["win_rate"] == pytest.approx(0.5)
    assert perf["profit_factor"] > 0
    assert perf["total_fees"] == pytest.approx(4 * 0.5)


# --------------------------------------------------------------------------
# crash recovery
# --------------------------------------------------------------------------
def test_recovery_reloads_open_positions(workspace, settings):
    """A restart must resume managing what was open, not start clean."""
    conn = workspace["conn"]
    p = Portfolio("paper", settings)
    open_a_position(p, conn)

    clients = fake_clients(prices={MINT: 1.10})
    store = DataStore(clients.birdeye, settings)

    report = recover(portfolio=p, clients=clients, store=store, cfg=settings, conn=conn)

    assert report.positions_recovered == 1
    assert not report.halted
    assert p.open_count(conn) == 1


def test_recovery_recomputes_stops_from_price_moved_while_down(workspace, settings):
    """A position that ran up while the process was dead gets its trail advanced."""
    conn = workspace["conn"]
    p = Portfolio("paper", settings)
    position_id = open_a_position(p, conn, price=1.0)

    before = conn.execute("SELECT * FROM positions WHERE id = ?", (position_id,)).fetchone()
    assert before["trailing_stop"] is None

    clients = fake_clients(prices={MINT: 1.40})   # +8R while offline
    store = DataStore(clients.birdeye, settings)
    report = recover(portfolio=p, clients=clients, store=store, cfg=settings, conn=conn)

    after = conn.execute("SELECT * FROM positions WHERE id = ?", (position_id,)).fetchone()
    assert report.stops_updated == 1
    assert after["trailing_armed"] == 1
    assert after["trailing_stop"] is not None
    assert after["trailing_stop"] > before["entry_price"]
    assert after["high_water_price"] == pytest.approx(1.40)


def test_recovery_on_a_clean_start_is_a_no_op(workspace, settings):
    conn = workspace["conn"]
    p = Portfolio("paper", settings)
    clients = fake_clients()
    store = DataStore(clients.birdeye, settings)

    report = recover(portfolio=p, clients=clients, store=store, cfg=settings, conn=conn)
    assert report.positions_recovered == 0
    assert not report.halted


def test_halts_are_re_announced_after_a_restart(workspace, settings):
    """A restart must never silently clear a triggered halt."""
    conn = workspace["conn"]
    risk.engage_kill_switch("testing", "tester", conn)
    for _ in range(settings["circuit_consecutive_losses"]):
        risk.record_trade_result(-10.0, settings, conn)

    p = Portfolio("paper", settings)
    clients = fake_clients()
    store = DataStore(clients.birdeye, settings)
    recover(portfolio=p, clients=clients, store=store, cfg=settings, conn=conn)

    # Both are still in force after the "restart".
    assert risk.kill_switch_engaged(conn)
    assert risk.circuit_tripped(conn)

    messages = [r["message"] for r in db.recent_events(50, conn=conn)]
    assert any("Kill switch is still engaged" in m for m in messages)
    assert any("Circuit breaker is still tripped" in m for m in messages)


# --------------------------------------------------------------------------
# on-chain reconciliation
# --------------------------------------------------------------------------
def test_reconciliation_passes_when_wallet_matches(workspace, settings):
    rows = [{"mint": MINT, "qty": 100.0, "symbol": "TEST"}]
    problems = _compare_wallet(rows, {"tokens": {MINT: 100.0}, "sol": 1.0}, settings)
    assert problems == []


def test_reconciliation_flags_a_missing_balance(workspace, settings):
    rows = [{"mint": MINT, "qty": 100.0, "symbol": "TEST"}]
    problems = _compare_wallet(rows, {"tokens": {}, "sol": 1.0}, settings)
    assert len(problems) == 1
    assert "wallet holds none" in problems[0]


def test_reconciliation_tolerates_dust_drift(workspace, settings):
    """Rounding drift is normal; only a material gap is a disagreement."""
    rows = [{"mint": MINT, "qty": 100.0, "symbol": "TEST"}]
    assert _compare_wallet(rows, {"tokens": {MINT: 100.5}, "sol": 1.0}, settings) == []
    assert _compare_wallet(rows, {"tokens": {MINT: 120.0}, "sol": 1.0}, settings) != []


def test_reconciliation_flags_an_untracked_holding(workspace, settings):
    problems = _compare_wallet([], {"tokens": {MINT: 50.0}, "sol": 1.0}, settings)
    assert len(problems) == 1
    assert "untracked" in problems[0]


def test_live_mismatch_halts_trading(workspace, settings):
    """The spec: do not guess - halt, alert, require manual review."""
    conn = workspace["conn"]
    settings = {**settings, "trading_mode": "live"}
    p = Portfolio("live", settings)
    open_a_position(p, conn)

    class Executor:
        def wallet_snapshot(self):
            return {"pubkey": "Wallet111", "sol": 1.0, "tokens": {}}  # holds nothing

    clients = fake_clients(prices={MINT: 1.0})
    store = DataStore(clients.birdeye, settings)
    report = recover(
        portfolio=p, clients=clients, store=store, cfg=settings,
        executor=Executor(), conn=conn,
    )

    assert report.halted
    assert report.discrepancies
    assert reconcile_halted(conn)

    clear_reconcile_halt("tester", conn)
    assert not reconcile_halted(conn)


def test_live_reconciliation_passes_when_consistent(workspace, settings):
    conn = workspace["conn"]
    settings = {**settings, "trading_mode": "live"}
    p = Portfolio("live", settings)
    open_a_position(p, conn, qty=100.0)

    class Executor:
        def wallet_snapshot(self):
            return {"pubkey": "Wallet111", "sol": 1.0, "tokens": {MINT: 100.0}}

    clients = fake_clients(prices={MINT: 1.0})
    store = DataStore(clients.birdeye, settings)
    report = recover(
        portfolio=p, clients=clients, store=store, cfg=settings,
        executor=Executor(), conn=conn,
    )
    assert not report.halted
    assert report.reconciled


# --------------------------------------------------------------------------
# instance isolation
# --------------------------------------------------------------------------
def test_instances_keep_separate_balances_and_positions(workspace, settings):
    conn = workspace["conn"]
    live = Portfolio("live", settings)
    paper = Portfolio("paper", settings)

    open_a_position(live, conn)

    assert live.open_count(conn) == 1
    assert paper.open_count(conn) == 0
    assert paper.balance(conn) == pytest.approx(settings["paper_starting_balance"])
    assert live.balance(conn) < paper.balance(conn)
