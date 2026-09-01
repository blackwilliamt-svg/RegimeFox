"""Backtest simulation, survivorship-bias handling, and the export formats."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from solbot import db, exports
from solbot.backtest import Backtester
from solbot.datastore import DataStore
from solbot.execution import Fill
from solbot.portfolio import Portfolio
from solbot.universe import historical_basket

from fakes import fake_clients

DEAD = "DeadTokenMintAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
ALIVE = "AliveTokenMintAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"


def seed_candles(conn, mint: str, *, bars: int = 300, interval: str = "5m",
                 trend: float = 0.0, seed: int = 1, end: int | None = None):
    """Write synthetic candles with periodic volume spikes so entries can fire."""
    rng = np.random.default_rng(seed)
    step = 300
    end = end or db.now()
    start = end - bars * step
    price = 1.0
    rows = []
    for i in range(bars):
        drift = trend + rng.normal(0, 0.003)
        # A spike every 40 bars, gentle enough to clear the RSI ceiling.
        spike = (i % 40) in (0, 1, 2)
        if spike:
            drift = abs(drift) + 0.004
        price = max(1e-6, price * (1 + drift))
        volume = 5000.0 * (6.0 if spike else 1.0)
        rows.append(
            (start + i * step, price * 0.999, price * 1.004, price * 0.996, price, volume)
        )
    db.upsert_candles(mint, interval, rows, conn=conn)
    return rows


# --------------------------------------------------------------------------
# survivorship bias
# --------------------------------------------------------------------------
def test_basket_includes_tokens_that_later_died(workspace, settings):
    """The whole point: a token present only in old snapshots must still appear."""
    conn = workspace["conn"]
    old_day = time.strftime("%Y-%m-%d", time.gmtime(db.now() - 30 * 86400))
    today = time.strftime("%Y-%m-%d", time.gmtime())

    conn.execute(
        "INSERT INTO universe_history(day, mint, symbol, liquidity_usd, volume_24h_usd) "
        "VALUES (?,?,?,?,?)", (old_day, DEAD, "DEAD", 100_000.0, 200_000.0)
    )
    conn.execute(
        "INSERT INTO universe_history(day, mint, symbol, liquidity_usd, volume_24h_usd) "
        "VALUES (?,?,?,?,?)", (today, ALIVE, "ALIVE", 100_000.0, 200_000.0)
    )
    # Only the survivor is in today's universe.
    conn.execute(
        "INSERT INTO universe(mint, symbol, updated_at) VALUES (?,?,?)",
        (ALIVE, "ALIVE", db.now()),
    )

    basket = historical_basket(old_day, today, conn=conn)
    assert DEAD in basket, "a token that died was dropped - that is survivorship bias"
    assert ALIVE in basket


def test_basket_falls_back_to_today_when_no_history(workspace, settings):
    conn = workspace["conn"]
    conn.execute(
        "INSERT INTO universe(mint, symbol, updated_at) VALUES (?,?,?)",
        (ALIVE, "ALIVE", db.now()),
    )
    clients = fake_clients()
    tester = Backtester(DataStore(clients.birdeye, settings), settings)
    assert tester.basket(30, conn) == [ALIVE]


# --------------------------------------------------------------------------
# simulation
# --------------------------------------------------------------------------
def test_backtest_runs_and_reports_metrics(workspace, settings):
    conn = workspace["conn"]
    seed_candles(conn, ALIVE, bars=400, trend=0.0005)
    conn.execute(
        "INSERT INTO universe(mint, symbol, liquidity_usd, updated_at) VALUES (?,?,?,?)",
        (ALIVE, "ALIVE", 1_000_000.0, db.now()),
    )

    clients = fake_clients()
    tester = Backtester(DataStore(clients.birdeye, settings), settings)
    result = tester.run(days=40, mints=[ALIVE], conn=conn)

    assert result.tokens_with_data == 1
    assert result.bars_evaluated > 0
    summary = result.summary()
    assert set(summary) >= {"trades", "win_rate", "profit_factor", "max_drawdown"}
    assert 0.0 <= result.win_rate <= 1.0
    assert result.ending_balance > 0


def test_backtest_respects_the_concurrent_position_cap(workspace, settings):
    """Simulating tokens in isolation would let it hold more than the live bot can."""
    conn = workspace["conn"]
    mints = []
    for i in range(5):
        mint = f"Mint{i}" + "X" * 38
        seed_candles(conn, mint, bars=400, trend=0.0006, seed=i + 2)
        conn.execute(
            "INSERT INTO universe(mint, symbol, liquidity_usd, updated_at) VALUES (?,?,?,?)",
            (mint, f"T{i}", 1_000_000.0, db.now()),
        )
        mints.append(mint)

    clients = fake_clients()
    tester = Backtester(DataStore(clients.birdeye, settings), settings)
    result = tester.run(days=40, mints=mints, conn=conn)

    # Reconstruct concurrency from the trade timeline.
    events = []
    for t in result.trades:
        events.append((t.entry_ts, 1))
        events.append((t.exit_ts, -1))
    events.sort()
    concurrent = peak = 0
    for _, delta in events:
        concurrent += delta
        peak = max(peak, concurrent)
    assert peak <= settings["max_concurrent_positions"]


def test_backtest_charges_fees_and_slippage(workspace, settings):
    """A simulation that ignores costs is the main way paper flatters itself."""
    conn = workspace["conn"]
    seed_candles(conn, ALIVE, bars=400, trend=0.0005)
    conn.execute(
        "INSERT INTO universe(mint, symbol, liquidity_usd, updated_at) VALUES (?,?,?,?)",
        (ALIVE, "ALIVE", 1_000_000.0, db.now()),
    )
    clients = fake_clients()
    tester = Backtester(DataStore(clients.birdeye, settings), settings)
    result = tester.run(days=40, mints=[ALIVE], conn=conn)
    if result.trades:
        assert result.total_fees > 0
        assert all(t.fees_usd > 0 for t in result.trades)


def test_backtest_with_no_data_returns_empty(workspace, settings):
    clients = fake_clients()
    tester = Backtester(DataStore(clients.birdeye, settings), settings)
    result = tester.run(days=30, mints=["NoSuchMint"], conn=workspace["conn"])
    assert result.trades == []
    assert result.tokens_with_data == 0


# --------------------------------------------------------------------------
# exports
# --------------------------------------------------------------------------
def make_trade(conn, instance: str, pnl: float, *, manual: bool = False, hold: int = 3600):
    now = db.now()
    conn.execute(
        "INSERT INTO trades(instance, mint, symbol, entry_ts, entry_price, exit_ts, "
        "exit_price, qty, size_usd, proceeds_usd, fees_usd, pnl_usd, pnl_pct, "
        "entry_reason, exit_reason, manual, hold_seconds) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (instance, ALIVE, "ALIVE", now - hold, 1.0, now, 1.0 + pnl / 100.0, 100.0,
         100.0, 100.0 + pnl, 0.5, pnl, pnl / 100.0, "volume spike", "target hit",
         1 if manual else 0, hold),
    )


def test_trades_csv_has_a_row_per_trade(workspace, settings):
    conn = workspace["conn"]
    make_trade(conn, "live", 10.0)
    make_trade(conn, "paper", -5.0)

    body = exports.trades_csv(exports.fetch_trades(conn=conn))
    lines = body.strip().splitlines()
    assert len(lines) == 3    # header + 2
    assert "instance" in lines[0]
    assert "live" in body and "paper" in body


def test_tax_summary_excludes_paper_and_shadow(workspace, settings):
    """Paper and shadow fills are not taxable events."""
    conn = workspace["conn"]
    make_trade(conn, "live", 100.0)
    make_trade(conn, "paper", 5000.0)
    make_trade(conn, "shadow", 9000.0)

    summary = exports.tax_summary(conn=conn)
    assert summary.trades == 1
    assert summary.realized_gain == pytest.approx(100.0)
    assert summary.proceeds == pytest.approx(200.0)


def test_tax_summary_splits_short_and_long_term(workspace, settings):
    conn = workspace["conn"]
    make_trade(conn, "live", 50.0, hold=3600)                  # short
    make_trade(conn, "live", 80.0, hold=400 * 86400)           # long

    summary = exports.tax_summary(conn=conn)
    assert summary.short_term_gain == pytest.approx(50.0)
    assert summary.long_term_gain == pytest.approx(80.0)


def test_tax_csv_states_the_exclusion(workspace, settings):
    conn = workspace["conn"]
    make_trade(conn, "live", 25.0)
    body = exports.tax_csv(exports.tax_summary(conn=conn))
    assert "LIVE trades only" in body
    assert "not taxable events" in body
    assert "Net realized gain/loss" in body


def test_manual_trades_are_distinguishable_in_the_export(workspace, settings):
    conn = workspace["conn"]
    make_trade(conn, "live", 10.0, manual=True)
    make_trade(conn, "live", 10.0, manual=False)
    body = exports.trades_csv(exports.fetch_trades(conn=conn))
    rows = body.strip().splitlines()[1:]
    flags = sorted(r.split(",")[15] for r in rows)
    assert flags == ["no", "yes"]
