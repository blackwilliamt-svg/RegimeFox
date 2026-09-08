"""Token bucket, priority reserve, and the scan-cadence budget.

Rate limits are the binding constraint on this bot, so these behaviours matter
as much as the strategy: the previous generation of this bot starved its price
poll because the safety scanner drained a shared budget.
"""
from __future__ import annotations

import time

import pytest

from solbot.ratelimit import TokenBucket
from solbot.scanner import Scanner

from fakes import FakeJupiter


# --------------------------------------------------------------------------
# token bucket
# --------------------------------------------------------------------------
def test_bucket_starts_empty_not_full():
    """Starting full would dump a whole window's budget on the first cycle."""
    bucket = TokenBucket(10.0, burst=5, reserve=0.0)
    assert not bucket.try_acquire(1.0)


def test_bucket_refills_at_the_configured_rate():
    bucket = TokenBucket(20.0, burst=5, reserve=0.0)
    time.sleep(0.15)   # ~3 tokens
    assert bucket.try_acquire(1.0)


def test_burst_is_capped_separately_from_rate():
    bucket = TokenBucket(100.0, burst=2, reserve=0.0)
    time.sleep(0.2)    # would be 20 tokens at the raw rate
    assert bucket.try_acquire(1.0)
    assert bucket.try_acquire(1.0)
    assert not bucket.try_acquire(1.0)   # burst ceiling, not the rate, binds


def test_low_priority_cannot_drain_the_reserve():
    """A scan must never starve an entry or exit quote."""
    bucket = TokenBucket(50.0, burst=3, reserve=1.0)
    time.sleep(0.1)
    acquired = 0
    while bucket.try_acquire(1.0, priority="low"):
        acquired += 1
        if acquired > 10:
            break
    # Headroom remains for a high-priority caller.
    assert bucket.try_acquire(1.0, priority="high")


def test_high_priority_may_use_the_reserve():
    bucket = TokenBucket(50.0, burst=2, reserve=2.0)
    time.sleep(0.08)
    assert not bucket.try_acquire(1.0, priority="low")
    assert bucket.try_acquire(1.0, priority="high")


def test_penalise_backs_off_after_429():
    bucket = TokenBucket(10.0, burst=5, reserve=0.0)
    time.sleep(0.2)
    before = bucket.effective_rate
    bucket.penalise(seconds=5.0, factor=0.5)
    assert bucket.effective_rate == pytest.approx(before * 0.5)
    assert bucket.stats()["rate_limited"] == 1
    # The bucket is emptied AND the refill clock restarts, so credit accrued
    # before the 429 is not handed straight back.
    assert not bucket.try_acquire(1.0)


def test_acquire_times_out_rather_than_blocking_forever():
    bucket = TokenBucket(0.1, burst=1, reserve=0.0)
    assert not bucket.acquire(1.0, timeout=0.05)


def test_configure_updates_limits_live():
    bucket = TokenBucket(1.0, burst=1, reserve=0.0)
    bucket.configure(100.0, 10)
    time.sleep(0.05)
    assert bucket.try_acquire(1.0)


# --------------------------------------------------------------------------
# scan cadence budget
# --------------------------------------------------------------------------
def test_batching_is_capped_at_the_api_limit(settings):
    """Jupiter takes 50 ids per call, not the 100 older integrations assumed."""
    jupiter = FakeJupiter()
    assert jupiter.price_calls_for(50) == 1
    assert jupiter.price_calls_for(51) == 2
    assert jupiter.price_calls_for(300) == 6


def test_budget_reports_an_unsustainable_cadence(settings):
    """A 300-token universe at 1 rps cannot also serve a 1s hot tier."""
    scanner = Scanner(FakeJupiter(), settings)
    for i in range(10):
        scanner.mark_interesting(f"mint{i}")

    budget = scanner.budget(universe_size=300, available_rps=1.0)
    assert budget.broad_calls_per_sweep == 6
    assert budget.hot_calls_per_poll == 1
    # 6 calls / 15s + 1 call / 1s = 1.4 rps required
    assert budget.required_rps == pytest.approx(1.4, abs=0.01)
    assert not budget.sustainable
    assert "exceeds the API budget" in budget.message()


def test_budget_is_sustainable_on_the_developer_tier(settings):
    scanner = Scanner(FakeJupiter(), settings)
    for i in range(10):
        scanner.mark_interesting(f"mint{i}")
    budget = scanner.budget(universe_size=300, available_rps=10.0)
    assert budget.sustainable
    assert "sustainable" in budget.message()


# --------------------------------------------------------------------------
# hot list
# --------------------------------------------------------------------------
def test_hot_list_respects_its_ceiling(settings):
    settings = {**settings, "hot_list_max": 3}
    scanner = Scanner(FakeJupiter(), settings)
    for i in range(10):
        scanner.mark_interesting(f"mint{i}")
    assert len(scanner.hot) == 3


def test_hot_list_evicts_the_stalest_entry(settings):
    settings = {**settings, "hot_list_max": 2}
    scanner = Scanner(FakeJupiter(), settings)
    scanner.mark_interesting("old")
    time.sleep(0.01)
    scanner.mark_interesting("mid")
    time.sleep(0.01)
    scanner.mark_interesting("new")
    assert "old" not in scanner.hot
    assert "new" in scanner.hot


def test_expire_hot_frees_stale_tokens(settings):
    scanner = Scanner(FakeJupiter(), settings)
    scanner.mark_interesting("stale")
    assert scanner.expire_hot(max_age_seconds=-1) == ["stale"]
    assert scanner.hot == []


def test_open_positions_stay_pinned_hot(settings):
    """A held token needs per-second exit checks regardless of its signal."""
    scanner = Scanner(FakeJupiter(), settings)
    scanner.pin(["held1", "held2"])
    assert set(scanner.hot) == {"held1", "held2"}


def test_scanner_uses_only_jupiter(settings):
    """Neither scan tier may touch Binance - it is reserved for candle history."""
    jupiter = FakeJupiter(prices={"a": 1.0, "b": 2.0})
    scanner = Scanner(jupiter, settings)
    result = scanner.scan_broad(["a", "b"])
    assert set(result.prices) == {"a", "b"}
    assert jupiter.calls == 1
