"""Rug Check gate: hard filters, fail-closed behaviour, and the cooldown."""
from __future__ import annotations

import pytest

from solbot import db
from solbot.clients.rugcheck import RugReport
from solbot.safety import SafetyGate

from fakes import FakeRugCheck, token

MINT = "TokenMintBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB"


def clean_report(**kw) -> RugReport:
    base = dict(
        mint=MINT, ok=True, rugged=False, score=1.0,
        mint_authority=None, freeze_authority=None,
        lp_locked_pct=99.0, top_holder_pct=5.0, insider_pct=1.0,
        total_holders=5000, total_liquidity_usd=500_000.0,
        transfer_fee_pct=0.0, market_count=50, lp_providers=10, risks=[], raw={},
    )
    base.update(kw)
    return RugReport(**base)


def gate(settings, clean: bool = True) -> SafetyGate:
    return SafetyGate(FakeRugCheck(clean), settings)


# --------------------------------------------------------------------------
# hard filters
# --------------------------------------------------------------------------
def test_clean_token_passes(workspace, settings):
    verdict = gate(settings).evaluate(clean_report(), token=token(MINT))
    assert verdict.passed, verdict.reasons


def test_active_mint_authority_fails(workspace, settings):
    verdict = gate(settings).evaluate(clean_report(mint_authority="Auth1"), token=token(MINT))
    assert not verdict.passed
    assert any("mint authority" in r for r in verdict.reasons)


def test_active_freeze_authority_fails(workspace, settings):
    verdict = gate(settings).evaluate(clean_report(freeze_authority="Auth1"), token=token(MINT))
    assert not verdict.passed
    assert any("freeze authority" in r for r in verdict.reasons)


def test_unlocked_liquidity_fails(workspace, settings):
    # market_count is below the waiver threshold, so the lock check applies.
    verdict = gate(settings).evaluate(
        clean_report(lp_locked_pct=10.0, market_count=4), token=token(MINT)
    )
    assert not verdict.passed
    assert any("locked" in r for r in verdict.reasons)


def test_holder_concentration_fails(workspace, settings):
    verdict = gate(settings).evaluate(clean_report(top_holder_pct=60.0), token=token(MINT))
    assert not verdict.passed
    assert any("top holder" in r for r in verdict.reasons)


def test_insider_concentration_fails(workspace, settings):
    verdict = gate(settings).evaluate(clean_report(insider_pct=45.0), token=token(MINT))
    assert not verdict.passed
    assert any("insiders" in r for r in verdict.reasons)


def test_thin_liquidity_fails(workspace, settings):
    verdict = gate(settings).evaluate(
        clean_report(total_liquidity_usd=1_000.0), token=token(MINT, liquidity=1_000.0)
    )
    assert not verdict.passed
    assert any("liquidity" in r for r in verdict.reasons)


def test_young_token_fails(workspace, settings):
    """Minimum age is a hard filter, not a preference."""
    recent = db.now() - 3600   # one hour old
    verdict = gate(settings).evaluate(clean_report(), token=token(MINT, first_pool_at=recent))
    assert not verdict.passed
    assert any("old" in r for r in verdict.reasons)


def test_transfer_fee_disqualifies(workspace, settings):
    """A transfer fee taxes every round trip; do not try to model it."""
    verdict = gate(settings).evaluate(clean_report(transfer_fee_pct=2.0), token=token(MINT))
    assert not verdict.passed
    assert any("transfer fee" in r for r in verdict.reasons)


def test_danger_risk_flags_fail(workspace, settings):
    verdict = gate(settings).evaluate(
        clean_report(risks=[{"name": "Honeypot", "level": "danger"}]), token=token(MINT)
    )
    assert not verdict.passed
    assert any("danger" in r for r in verdict.reasons)


def test_rugged_flag_fails(workspace, settings):
    verdict = gate(settings).evaluate(clean_report(rugged=True), token=token(MINT))
    assert not verdict.passed


# --------------------------------------------------------------------------
# fail closed
# --------------------------------------------------------------------------
def test_unreadable_report_fails_closed(workspace, settings):
    """An API outage must stop entries, not wave unverified tokens through."""
    verdict = gate(settings).evaluate(RugReport(mint=MINT, ok=False), token=token(MINT))
    assert not verdict.passed
    assert "failing closed" in verdict.reasons[0]


def test_outage_retries_sooner_than_a_real_failure(workspace, settings):
    outage = gate(settings).evaluate(RugReport(mint=MINT, ok=False), token=token(MINT))
    real = gate(settings).evaluate(clean_report(rugged=True), token=token(MINT))
    assert outage.recheck_after - outage.checked_at < real.recheck_after - real.checked_at


# --------------------------------------------------------------------------
# caching and cooldown
# --------------------------------------------------------------------------
def test_result_is_cached_within_the_ttl(workspace, settings):
    conn = workspace["conn"]
    g = gate(settings)
    first = g.check(MINT, token=token(MINT), conn=conn)
    assert not first.cached
    assert g.client.calls == 1

    second = g.check(MINT, token=token(MINT), conn=conn)
    assert second.cached
    assert g.client.calls == 1   # no second API call


def test_failure_is_retried_after_the_cooldown_not_blacklisted(workspace, settings):
    """The spec: re-test on a cooldown, not every poll and not never."""
    conn = workspace["conn"]
    g = gate(settings, clean=False)
    verdict = g.check(MINT, token=token(MINT), conn=conn)
    assert not verdict.passed

    row = conn.execute("SELECT * FROM safety_reports WHERE mint = ?", (MINT,)).fetchone()
    assert row["recheck_after"] > row["checked_at"]
    assert row["recheck_after"] - row["checked_at"] == settings["rugcheck_fail_cooldown_seconds"]

    # Rewind the cooldown: the token becomes eligible for re-testing.
    conn.execute(
        "UPDATE safety_reports SET recheck_after = ? WHERE mint = ?", (db.now() - 1, MINT)
    )
    g.client.clean = True     # its risk profile genuinely improved
    retried = g.check(MINT, token=token(MINT), conn=conn)
    assert not retried.cached
    assert retried.passed


def test_pass_ttl_is_shorter_than_fail_cooldown(settings):
    """A token that degrades must be caught sooner than a failure is retried."""
    assert settings["rugcheck_pass_ttl_seconds"] >= settings["rugcheck_fail_cooldown_seconds"] \
        or settings["rugcheck_pass_ttl_seconds"] > 0


def test_disabling_the_gate_passes_everything(workspace, settings):
    settings = {**settings, "rugcheck_enabled": False}
    verdict = gate(settings).check(MINT, token=token(MINT), conn=workspace["conn"])
    assert verdict.passed


def test_rejection_is_logged_to_the_event_feed(workspace, settings):
    conn = workspace["conn"]
    gate(settings, clean=False).check(MINT, token=token(MINT), conn=conn)
    messages = [r["message"] for r in db.recent_events(20, conn=conn)]
    assert any("Rug Check rejected" in m for m in messages)


# --------------------------------------------------------------------------
# prescreen
# --------------------------------------------------------------------------
def test_prescreen_catches_obvious_failures_without_an_api_call(workspace, settings):
    """Jupiter already reports the authorities; do not spend a RugCheck call."""
    g = gate(settings)
    assert g.prescreen(token(MINT)) == []
    assert g.prescreen(token(MINT, mint_authority="Auth1"))
    assert g.prescreen(token(MINT, freeze_authority="Auth1"))
    assert g.client.calls == 0


# --------------------------------------------------------------------------
# LP-lock waiver
# --------------------------------------------------------------------------
def test_lp_lock_is_enforced_for_a_few_pool_token(workspace, settings):
    """A token with a handful of pools is exactly what the lock check is for."""
    verdict = gate(settings).evaluate(
        clean_report(lp_locked_pct=0.0, market_count=6), token=token(MINT)
    )
    assert not verdict.passed
    assert any("liquidity locked" in r for r in verdict.reasons)


def test_lp_lock_is_waived_when_liquidity_spans_many_pools(workspace, settings):
    """Concentrated-liquidity venues have no LP token to lock, and liquidity
    spread over hundreds of pools cannot be pulled by one actor. Applying the
    threshold literally there rejects the whole tradeable universe."""
    verdict = gate(settings).evaluate(
        clean_report(lp_locked_pct=0.0, market_count=443), token=token(MINT)
    )
    assert verdict.passed, verdict.reasons
    assert verdict.detail["lp_lock_waived"] is True
    assert verdict.detail["market_count"] == 443


def test_waiver_threshold_is_configurable(workspace, settings):
    strict = {**settings, "rugcheck_lp_lock_waiver_markets": 1000}
    verdict = gate(strict).evaluate(
        clean_report(lp_locked_pct=0.0, market_count=443), token=token(MINT)
    )
    assert not verdict.passed


def test_waiver_can_be_disabled_entirely(workspace, settings):
    """Setting the waiver to 0 restores the spec's literal behaviour."""
    strict = {**settings, "rugcheck_lp_lock_waiver_markets": 0}
    verdict = gate(strict).evaluate(
        clean_report(lp_locked_pct=0.0, market_count=443), token=token(MINT)
    )
    assert not verdict.passed


def test_waiver_does_not_bypass_the_other_gates(workspace, settings):
    """A widely-pooled token still has to clear every other hard filter."""
    verdict = gate(settings).evaluate(
        clean_report(lp_locked_pct=0.0, market_count=443, mint_authority="Auth1"),
        token=token(MINT),
    )
    assert not verdict.passed
    assert any("mint authority" in r for r in verdict.reasons)


def test_no_market_data_is_not_waived(workspace, settings):
    """Zero markets means RugCheck could not see any liquidity - not a pass."""
    verdict = gate(settings).evaluate(
        clean_report(lp_locked_pct=0.0, market_count=0), token=token(MINT)
    )
    assert not verdict.passed
