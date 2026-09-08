"""The entry review gate: a deterministic rules engine, no model in the loop.

There is no network path here to test a timeout or fallback for - the gate
either runs the rules engine or is switched off entirely. These tests confirm
it produces a decision with no network call involved anywhere.
"""
from __future__ import annotations

import json

import pytest

from solbot import daily_review, db, review


def make_request(**overrides) -> review.EntryRequest:
    base = dict(
        mint="MintAAA",
        symbol="AAA",
        instance="paper",
        price=1.5,
        size_usd=400.0,
        liquidity_usd=5_000_000.0,
        rr=3.0,
        strength=0.8,
        entry_reason="volume 4.0x average; momentum +1.2%",
        snapshot={
            "volume_ratio": 4.0, "momentum_pct": 0.012, "rsi": 68.0,
            "atr_pct": 0.02, "efficiency": 0.6, "regime": 0, "confluence": 2,
        },
        market=review.MarketContext(breadth=0.72, index_return=0.01, tokens=40),
        trades_today=0,
    )
    base.update(overrides)
    return review.EntryRequest(**base)


def gate(settings, **overrides) -> review.EntryGate:
    cfg = {**settings, "entry_gate_enabled": True}
    cfg.update(overrides)
    return review.EntryGate(cfg)


# --------------------------------------------------------------------------
# Market context
# --------------------------------------------------------------------------
def test_market_tone_reads_breadth_and_direction():
    firm = review.MarketContext(breadth=0.7, index_return=0.02, tokens=40)
    soft = review.MarketContext(breadth=0.25, index_return=-0.03, tokens=40)
    mixed = review.MarketContext(breadth=0.5, index_return=0.0, tokens=40)
    thin = review.MarketContext(breadth=0.9, index_return=0.05, tokens=2)

    assert firm.tone == "firm"
    assert soft.tone == "soft"
    assert mixed.tone == "mixed"
    assert thin.tone == "unknown", "a handful of tokens is not a market read"


def test_market_context_is_built_from_stored_ticks(workspace):
    conn = workspace["conn"]
    now = db.now()
    for i, (start, end) in enumerate([(1.0, 1.1), (1.0, 1.2), (1.0, 0.9)] * 4):
        mint = f"M{i}"
        conn.execute(
            "INSERT INTO price_ticks(mint, ts, price) VALUES (?,?,?)",
            (mint, now - 1800, start),
        )
        conn.execute(
            "INSERT INTO price_ticks(mint, ts, price) VALUES (?,?,?)",
            (mint, now - 60, end),
        )

    context = review.build_market_context(conn=conn, force=True)
    assert context.tokens == 12
    assert context.breadth == pytest.approx(8 / 12)
    assert context.index_return > 0


# --------------------------------------------------------------------------
# The deterministic rules engine
# --------------------------------------------------------------------------
def test_rules_approve_a_strong_signal_in_a_firm_market(settings):
    decision = review.RulesProxy(settings).decide(make_request())

    assert decision.approve
    assert decision.conviction > 0.55
    assert decision.source == review.SOURCE_RULES
    assert decision.rationale


def test_rules_refuse_everything_in_a_soft_market(settings):
    """Rubric clause 1: judge the whole market, not just the token."""
    decision = review.RulesProxy(settings).decide(
        make_request(market=review.MarketContext(breadth=0.2, index_return=-0.04, tokens=40))
    )

    assert not decision.approve
    assert "soft" in decision.rationale


def test_rules_raise_the_bar_as_the_day_fills_up(settings):
    """Rubric clause 2: overtrading is the failure mode."""
    proxy = review.RulesProxy({**settings, "max_trades_per_day": 8})
    fresh = proxy.decide(make_request(trades_today=0))
    busy = proxy.decide(make_request(trades_today=7))

    assert fresh.conviction > busy.conviction
    assert fresh.approve and not busy.approve


def test_rules_decide_ride_versus_lock_in_per_trade(settings):
    """Rubric clause 4: no fixed answer, decided from the metrics."""
    proxy = review.RulesProxy(settings)

    trending = proxy.decide(make_request())
    assert trending.exit_style == review.STYLE_RIDE
    assert trending.trailing_distance_atr > settings["trailing_distance_atr"]

    choppy = proxy.decide(
        make_request(
            snapshot={**make_request().snapshot, "regime": 2, "efficiency": 0.05},
            market=review.MarketContext(breadth=0.5, index_return=0.0, tokens=40),
        )
    )
    assert choppy.exit_style == review.STYLE_LOCK_IN
    assert choppy.trailing_distance_atr < settings["trailing_distance_atr"]


def test_rules_reward_higher_timeframe_agreement(settings):
    proxy = review.RulesProxy({**settings, "confluence_timeframes": [3, 6]})
    agreeing = proxy.decide(make_request())
    alone = proxy.decide(
        make_request(snapshot={**make_request().snapshot, "confluence": 0})
    )
    assert agreeing.conviction > alone.conviction


# --------------------------------------------------------------------------
# The gate itself: no network call anywhere in it
# --------------------------------------------------------------------------
def test_gate_produces_a_decision_with_no_network_call(workspace, settings):
    g = gate(settings)
    decision = g.review_entry(make_request(), conn=workspace["conn"])

    assert decision.source == review.SOURCE_RULES
    assert decision.approve
    rows = workspace["conn"].execute("SELECT * FROM entry_reviews").fetchall()
    assert len(rows) == 1, "every decision is recorded"


def test_gate_disabled_approves_everything_without_the_rules_engine(workspace, settings):
    g = review.EntryGate({**settings, "entry_gate_enabled": False})
    decision = g.review_entry(make_request(), conn=workspace["conn"])

    assert decision.approve
    assert decision.source == review.SOURCE_DISABLED
    rows = workspace["conn"].execute("SELECT * FROM entry_reviews").fetchall()
    assert len(rows) == 1


def test_a_weak_signal_still_goes_through_the_rules_engine(workspace, settings):
    """There is no API call to skip for a weak signal any more - it is scored
    by the same free, instant rules either way."""
    g = gate(settings, entry_gate_min_strength=0.5)
    weak = make_request(
        strength=0.05,
        snapshot={**make_request().snapshot, "regime": 2, "confluence": 0},
        market=review.MarketContext(breadth=0.5, index_return=0.0, tokens=40),
    )
    decision = g.review_entry(weak, conn=workspace["conn"])

    assert decision.source == review.SOURCE_RULES
    assert not decision.approve


def test_update_config_reaches_the_underlying_rules_engine(settings):
    g = review.EntryGate(settings)
    g.update_config({**settings, "max_trades_per_day": 1})
    assert g.rules.cfg["max_trades_per_day"] == 1


# --------------------------------------------------------------------------
# The daily review
# --------------------------------------------------------------------------
def test_rules_proposer_targets_overtrading_first(settings):
    summary = {
        "trades": 244, "win_rate": 0.42, "profit_factor": 1.1,
        "max_drawdown": 0.12, "total_fees": 449.0, "total_pnl": 300.0,
    }
    outcome = daily_review.propose_from_rules(summary, settings, days=7)

    keys = [p.key for p in outcome.proposals]
    assert "volume_spike_multiple" in keys, "the volume bar is the direct brake"
    assert outcome.proposals[0].proposed > settings["volume_spike_multiple"]
    assert "trades a day" in outcome.assessment


def test_rules_proposer_stays_quiet_on_a_healthy_backtest(settings):
    summary = {
        "trades": 40, "win_rate": 0.55, "profit_factor": 2.1,
        "max_drawdown": 0.08, "total_fees": 20.0, "total_pnl": 400.0,
    }
    outcome = daily_review.propose_from_rules(summary, settings, days=30)

    assert outcome.proposals == []
    assert "Nothing in the results argues for a change" in outcome.assessment


def test_proposals_outside_the_bounds_are_refused(settings):
    proposals = [
        daily_review.Proposal("rr_min", 2.0, 99.0, "far too high"),
        daily_review.Proposal("paper_starting_balance", 1000.0, 1e9, "not adjustable"),
        daily_review.Proposal("volume_spike_multiple", 2.0, 2.5, "reasonable"),
    ]
    kept, rejected = daily_review.filter_valid(proposals, settings)

    assert [p.key for p in kept] == ["volume_spike_multiple"]
    assert len(rejected) == 2


def test_an_improvement_is_shadowed_and_never_applied_to_live(workspace, settings):
    conn = workspace["conn"]
    before = {
        "trades": 244, "win_rate": 0.42, "profit_factor": 1.1,
        "max_drawdown": 0.20, "total_fees": 449.0, "total_pnl": 300.0,
    }
    after = {"trades": 60, "total_return": 0.35, "max_drawdown": 0.10}

    outcome = daily_review.run_daily_review(
        cfg=settings, summary=before, days=7,
        backtest=lambda overrides: after, conn=conn,
    )

    assert outcome.applied
    assert outcome.accepted
    commands = conn.execute("SELECT * FROM commands").fetchall()
    assert len(commands) == 1
    assert commands[0]["command"] == "set_shadow", "shadow only, never live"

    payload = json.loads(commands[0]["payload"])
    assert payload["overrides"], "the shadow instance receives the proposed set"


def test_a_change_that_does_not_improve_is_discarded(workspace, settings):
    """"It backtested worse" is the whole point of running it before shadowing."""
    conn = workspace["conn"]
    before = {
        "trades": 244, "win_rate": 0.42, "profit_factor": 1.1,
        "max_drawdown": 0.20, "total_fees": 449.0, "total_pnl": 300.0, "total_return": 0.30,
    }
    after = {"trades": 60, "total_return": 0.10, "max_drawdown": 0.25}

    outcome = daily_review.run_daily_review(
        cfg=settings, summary=before, days=7,
        backtest=lambda overrides: after, conn=conn,
    )

    assert not outcome.applied
    assert conn.execute("SELECT COUNT(*) AS n FROM commands").fetchone()["n"] == 0
    assert "did not backtest better" in outcome.note


def test_more_return_bought_with_much_more_drawdown_is_not_an_improvement():
    before = {"total_return": 0.20, "max_drawdown": 0.10, "trades": 50}
    after = {"total_return": 0.22, "max_drawdown": 0.30, "trades": 50}
    assert not daily_review._improves(before, after)


def test_nothing_is_shadowed_without_a_backtest_to_prove_it(workspace, settings):
    before = {
        "trades": 244, "win_rate": 0.42, "profit_factor": 1.1,
        "max_drawdown": 0.20, "total_fees": 449.0, "total_pnl": 300.0,
    }
    outcome = daily_review.run_daily_review(
        cfg=settings, summary=before, days=7, backtest=None,
        conn=workspace["conn"],
    )

    assert not outcome.applied
    assert "nothing was shadowed unproven" in outcome.note
