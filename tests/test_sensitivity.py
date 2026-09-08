"""Parameter sensitivity: aggregating the persistent library's kept
combinations by each tunable axis's own value.
"""
from __future__ import annotations

from solopt.sensitivity import compute_sensitivity


def _entry(**params) -> dict:
    performance = params.pop("performance", None) or {
        "walk_forward_efficiency": params.pop("wfe", 0.5),
        "mean_window_return": params.pop("ret", 0.02),
    }
    return {"params": params, "performance": performance}


def test_too_few_entries_returns_nothing():
    entries = [_entry(ema_fast=9, wfe=0.5), _entry(ema_fast=12, wfe=0.6)]
    result = compute_sensitivity(entries, min_entries=3)
    assert result["axes"] == {}
    assert result["ranked"] == []
    assert result["entries_used"] == 2


def test_an_axis_with_only_one_observed_value_is_excluded():
    entries = [
        _entry(ema_fast=9, ema_slow=21, wfe=0.5),
        _entry(ema_fast=9, ema_slow=34, wfe=0.6),
        _entry(ema_fast=9, ema_slow=50, wfe=0.4),
    ]
    result = compute_sensitivity(entries, min_entries=3)
    assert "ema_fast" not in result["axes"]   # only ever 9
    assert "ema_slow" in result["axes"]        # 21, 34, 50


def test_buckets_and_ranks_by_score_range():
    entries = [
        _entry(rsi_period=14, wfe=0.8, ret=0.05),
        _entry(rsi_period=14, wfe=0.7, ret=0.04),
        _entry(rsi_period=21, wfe=0.2, ret=0.01),
        _entry(rsi_period=21, wfe=0.1, ret=0.005),
        _entry(atr_period=14, wfe=0.5, ret=0.02),
        _entry(atr_period=21, wfe=0.5, ret=0.02),
        # atr_period's mean score is identical across its two values -
        # rsi_period's is not, so rsi_period must rank first.
    ]
    result = compute_sensitivity(entries, min_entries=3)

    assert result["ranked"][0] == "rsi_period"
    rsi = result["axes"]["rsi_period"]
    assert rsi["best_value"] == 14
    assert len(rsi["buckets"]) == 2
    fourteen = next(b for b in rsi["buckets"] if b["value"] == 14)
    assert fourteen["count"] == 2

    if "atr_period" in result["axes"]:
        assert result["axes"]["atr_period"]["score_range"] == 0.0


def test_continuous_values_are_bucketed_to_two_decimal_places():
    entries = [
        _entry(stop_atr_mult=1.001, wfe=0.5),
        _entry(stop_atr_mult=1.004, wfe=0.6),
        _entry(stop_atr_mult=2.0, wfe=0.2),
    ]
    result = compute_sensitivity(entries, min_entries=3)
    buckets = result["axes"]["stop_atr_mult"]["buckets"]
    values = sorted(b["value"] for b in buckets)
    assert values == [1.0, 2.0]
    one = next(b for b in buckets if b["value"] == 1.0)
    assert one["count"] == 2   # 1.001 and 1.004 fall in the same bucket


def test_only_tunable_axes_are_considered():
    entries = [
        _entry(ema_fast=9, not_a_real_axis=123, wfe=0.5),
        _entry(ema_fast=12, not_a_real_axis=456, wfe=0.6),
        _entry(ema_fast=21, not_a_real_axis=789, wfe=0.3),
    ]
    result = compute_sensitivity(entries, min_entries=3)
    assert "not_a_real_axis" not in result["axes"]
    assert "ema_fast" in result["axes"]


def test_empty_input_is_handled_gracefully():
    result = compute_sensitivity([])
    assert result == {"axes": {}, "ranked": [], "entries_used": 0}
