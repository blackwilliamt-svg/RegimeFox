"""Parameter bundles, shadow installation, and the auto-promotion gate.

The hand-off repository is untrusted input: it comes from another machine, and
the droplet has to decide for itself whether a set is worth running. So the
tests here are mostly about refusal - a bundle that fails its own gates, carries
an out-of-bounds parameter, or has not earned the live slot yet.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pytest

from solbot import db, paramsync


def good_params() -> dict:
    return {
        "volume_spike_multiple": 2.6,
        "momentum_min_pct": 0.012,
        "ema_fast": 9,
        "ema_slow": 21,
        "stop_atr_mult": 1.4,
        "rr_min": 2.0,
        "rr_max": 4.0,
    }


def bundle(**overrides) -> dict:
    payload = {
        "schema": 1,
        "generated_at": int(time.time()),
        "global": good_params(),
        "per_symbol": {},
        "risk": {"p5_max_drawdown": 0.18, "recommended_portfolio_vol_target": 0.017},
        "evidence": {
            "walk_forward": {
                "counted_windows": 8, "profitable_windows": 7,
                "walk_forward_efficiency": 0.62,
            }
        },
        "gates": {
            "accepted": True, "walk_forward_accepted": True, "overfit": False,
            "fragile": False, "fragile_windows": [], "reasons": [],
            "shadow_min_days": 15,
        },
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(payload.get(key), dict):
            payload[key] = {**payload[key], **value}
        else:
            payload[key] = value
    return payload


# --------------------------------------------------------------------------
# Bundle validation
# --------------------------------------------------------------------------
def test_a_clean_bundle_is_accepted(settings):
    check = paramsync.check_bundle(bundle(), settings)

    assert check.ok
    assert check.overrides["volume_spike_multiple"] == pytest.approx(2.6)
    assert check.fingerprint


def test_a_fragile_bundle_is_refused(settings):
    """A crash-replay failure blocks promotion however good the windows were."""
    check = paramsync.check_bundle(
        bundle(gates={"fragile": True, "fragile_windows": ["crash 2024-03-12"]}),
        settings,
    )

    assert not check.ok
    assert "fragile" in check.reason
    assert "crash 2024-03-12" in check.reason


def test_an_overfit_bundle_is_refused(settings):
    check = paramsync.check_bundle(bundle(gates={"overfit": True}), settings)
    assert not check.ok
    assert "overfit" in check.reason


def test_a_bundle_the_optimizer_did_not_accept_is_refused(settings):
    check = paramsync.check_bundle(
        bundle(gates={"accepted": False, "reasons": ["only 40% of windows profitable"]}),
        settings,
    )
    assert not check.ok
    assert "40%" in check.reason


def test_an_out_of_bounds_parameter_is_stripped_not_trusted(settings):
    """The repository is another machine's output, not an authority."""
    check = paramsync.check_bundle(
        bundle(**{"global": {**good_params(), "rr_min": 99.0}}), settings
    )

    assert "rr_min" in " ".join(check.rejected)
    assert "rr_min" not in check.overrides
    assert check.ok, "the rest of a mostly-valid set still stands"


def test_an_unknown_parameter_is_ignored(settings):
    check = paramsync.check_bundle(
        bundle(**{"global": {**good_params(), "drop_tables": 1}}), settings
    )
    assert "drop_tables" not in check.overrides
    assert any("drop_tables" in r for r in check.rejected)


def test_an_internally_inconsistent_set_is_refused_whole(settings):
    """Individually valid, jointly impossible - caught before it reaches shadow."""
    check = paramsync.check_bundle(
        bundle(**{"global": {**good_params(), "ema_fast": 34, "ema_slow": 21}}), settings
    )
    assert not check.ok
    assert "inconsistent" in check.reason


def test_an_unknown_schema_version_is_refused():
    with pytest.raises(paramsync.BundleError, match="schema 99"):
        paramsync.validate_schema({"schema": 99})


def test_a_non_object_payload_is_refused():
    with pytest.raises(paramsync.BundleError, match="not an object"):
        paramsync.validate_schema([1, 2, 3])  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# accept_bundle - the unified entrypoint for both the daily run and a RunPod
# worker's report (spec 5): no git repository or hand-off path in between.
# --------------------------------------------------------------------------
def test_accept_bundle_installs_a_valid_set_from_either_source(workspace, settings):
    conn = workspace["conn"]
    result = paramsync.accept_bundle(bundle(), settings, source="daily-droplet", conn=conn)

    assert result["accepted"]
    assert result["installed"]
    row = conn.execute("SELECT * FROM param_bundles").fetchone()
    assert row["status"] == paramsync.STATUS_SHADOW
    assert row["source_commit"] == "daily-droplet"


def test_accept_bundle_refuses_a_bad_schema_without_touching_the_database(workspace, settings):
    with pytest.raises(paramsync.BundleError):
        paramsync.accept_bundle({"schema": 7}, settings, source="runpod:1.2.3.4", conn=workspace["conn"])
    assert workspace["conn"].execute(
        "SELECT COUNT(*) AS n FROM param_bundles"
    ).fetchone()["n"] == 0


def test_accept_bundle_records_but_does_not_install_a_failing_set(workspace, settings):
    conn = workspace["conn"]
    result = paramsync.accept_bundle(
        bundle(gates={"overfit": True}), settings, source="daily-droplet", conn=conn
    )
    assert not result["accepted"]
    assert not result["installed"]
    row = conn.execute("SELECT * FROM param_bundles").fetchone()
    assert row["status"] == paramsync.STATUS_REJECTED


# --------------------------------------------------------------------------
# Installation
# --------------------------------------------------------------------------
def test_installing_a_bundle_carries_the_monte_carlo_tail_across(workspace, settings):
    """Parameters without their risk figure would leave sizing miscalibrated."""
    conn = workspace["conn"]
    payload = bundle()
    check = paramsync.check_bundle(payload, settings)
    paramsync.store_bundle(payload, check, "abc123", conn)
    paramsync.install_shadow(check, payload, conn)

    overrides = db.kv_get(paramsync.SHADOW_KEY, {}, conn)
    assert overrides["monte_carlo_p5_drawdown"] == pytest.approx(0.18)
    assert overrides["portfolio_vol_target"] == pytest.approx(0.017)

    row = conn.execute("SELECT * FROM param_bundles").fetchone()
    assert row["status"] == paramsync.STATUS_SHADOW
    assert row["shadow_since"]

    commands = conn.execute("SELECT * FROM commands").fetchall()
    assert [c["command"] for c in commands] == ["set_shadow"]


def test_a_new_bundle_supersedes_the_one_in_shadow(workspace, settings):
    conn = workspace["conn"]
    first = bundle()
    check_one = paramsync.check_bundle(first, settings)
    paramsync.store_bundle(first, check_one, "aaa", conn)
    paramsync.install_shadow(check_one, first, conn)

    second = bundle(**{"global": {**good_params(), "stop_atr_mult": 1.9}})
    check_two = paramsync.check_bundle(second, settings)
    paramsync.store_bundle(second, check_two, "bbb", conn)
    paramsync.install_shadow(check_two, second, conn)

    statuses = {
        r["fingerprint"]: r["status"]
        for r in conn.execute("SELECT fingerprint, status FROM param_bundles")
    }
    assert statuses[check_one.fingerprint] == paramsync.STATUS_SUPERSEDED
    assert statuses[check_two.fingerprint] == paramsync.STATUS_SHADOW


def test_the_same_bundle_is_only_stored_once(workspace, settings):
    conn = workspace["conn"]
    payload = bundle()
    check = paramsync.check_bundle(payload, settings)

    assert paramsync.store_bundle(payload, check, "aaa", conn) is True
    assert paramsync.store_bundle(payload, check, "aaa", conn) is False
    assert conn.execute("SELECT COUNT(*) AS n FROM param_bundles").fetchone()["n"] == 1


# --------------------------------------------------------------------------
# Promotion
# --------------------------------------------------------------------------
def seed_shadow(conn, days_ago: float) -> str:
    payload = bundle()
    fingerprint = paramsync.fingerprint(payload)
    conn.execute(
        "INSERT INTO param_bundles(fingerprint, received_at, payload, status, shadow_since) "
        "VALUES (?,?,?,?,?)",
        (
            fingerprint, db.now(), json.dumps(payload), paramsync.STATUS_SHADOW,
            db.now() - int(days_ago * 86400),
        ),
    )
    return fingerprint


def seed_trades(conn, instance: str, pnls, since: int) -> None:
    for i, pnl in enumerate(pnls):
        conn.execute(
            "INSERT INTO trades(instance, mint, symbol, entry_ts, entry_price, exit_ts, "
            "exit_price, qty, size_usd, proceeds_usd, fees_usd, pnl_usd, pnl_pct) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                instance, "M", "M", since + i * 60, 1.0, since + i * 120, 1.0,
                100.0, 100.0, 100.0 + pnl, 0.5, pnl, pnl / 100.0,
            ),
        )


def test_promotion_waits_out_the_full_shadow_period(workspace, settings):
    conn = workspace["conn"]
    seed_shadow(conn, days_ago=5)

    verdict = paramsync.evaluate_promotion(settings, conn)
    assert not verdict.promoted
    assert "of the required 15 clean days" in verdict.reason


def test_promotion_needs_enough_shadow_trades(workspace, settings):
    conn = workspace["conn"]
    since = db.now() - 20 * 86400
    seed_shadow(conn, days_ago=20)
    seed_trades(conn, "shadow", [5.0] * 4, since)

    verdict = paramsync.evaluate_promotion(settings, conn)
    assert not verdict.promoted
    assert "below the 30" in verdict.reason


def test_a_marginal_edge_is_not_promoted(workspace, settings):
    """"Decisive margin, not a marginal edge" - the spec's words, as a number."""
    conn = workspace["conn"]
    since = db.now() - 20 * 86400
    seed_shadow(conn, days_ago=20)
    rng = np.random.default_rng(2)
    live = list(rng.normal(2.0, 8.0, 60))
    shadow = list(rng.normal(2.2, 8.0, 60))   # 10% better, well inside the noise
    seed_trades(conn, "paper", live, since)
    seed_trades(conn, "shadow", shadow, since)

    verdict = paramsync.evaluate_promotion(settings, conn)
    assert not verdict.promoted
    assert verdict.margin < settings["shadow_min_margin"]


def test_a_decisive_edge_is_promoted(workspace, settings, cfg):
    conn = workspace["conn"]
    since = db.now() - 20 * 86400
    seed_shadow(conn, days_ago=20)
    rng = np.random.default_rng(4)
    seed_trades(conn, "paper", list(rng.normal(1.0, 3.0, 80)), since)
    seed_trades(conn, "shadow", list(rng.normal(6.0, 3.0, 80)), since)

    verdict = paramsync.evaluate_promotion(settings, conn)
    assert verdict.promoted, verdict.reason
    assert verdict.confidence >= settings["shadow_min_confidence"]
    assert verdict.margin >= settings["shadow_min_margin"]

    db.kv_set(paramsync.SHADOW_KEY, {"volume_spike_multiple": 2.6}, conn)
    applied = paramsync.promote(cfg, verdict, conn)

    assert applied["volume_spike_multiple"][1] == pytest.approx(2.6)
    assert cfg["volume_spike_multiple"] == pytest.approx(2.6)
    assert db.kv_get(paramsync.SHADOW_KEY, {}, conn) == {}, "shadow is cleared after"
    assert conn.execute(
        "SELECT status FROM param_bundles"
    ).fetchone()["status"] == paramsync.STATUS_PROMOTED


def test_a_drifting_shadow_set_is_not_promoted(workspace, settings):
    """Fifteen days that stop meeting expectation are not fifteen clean days."""
    conn = workspace["conn"]
    seed_shadow(conn, days_ago=20)
    conn.execute(
        "INSERT INTO backtest_runs(ts, trades, win_rate, profit_factor, max_drawdown, "
        "total_return, avg_win, avg_loss) VALUES (?,?,?,?,?,?,?,?)",
        (db.now(), 200, 0.60, 2.5, 0.08, 0.4, 20.0, -5.0),
    )
    # Recent enough to be inside the drift window, and far below expectation.
    recent = db.now() - 3 * 86400
    seed_trades(conn, "shadow", [0.4, -0.3] * 30, recent)
    seed_trades(conn, "paper", [0.1] * 40, recent)

    verdict = paramsync.evaluate_promotion(settings, conn)
    assert not verdict.promoted
    assert "drifted" in verdict.reason


def test_promotion_can_be_switched_off(workspace, settings):
    conn = workspace["conn"]
    seed_shadow(conn, days_ago=40)
    verdict = paramsync.evaluate_promotion({**settings, "auto_promote_enabled": False}, conn)

    assert not verdict.promoted
    assert "switched off" in verdict.reason


def test_every_promotion_decision_is_recorded_including_refusals(workspace, settings):
    conn = workspace["conn"]
    since = db.now() - 20 * 86400
    seed_shadow(conn, days_ago=20)
    seed_trades(conn, "shadow", [1.0] * 40, since)
    seed_trades(conn, "paper", [5.0] * 40, since)

    paramsync.evaluate_promotion(settings, conn)
    rows = conn.execute("SELECT * FROM promotions").fetchall()

    assert len(rows) == 1
    assert rows[0]["promoted"] == 0
    assert rows[0]["reason"]


def test_bootstrap_confidence_separates_a_real_edge_from_noise():
    rng = np.random.default_rng(1)
    noise_a = list(rng.normal(1.0, 10.0, 60))
    noise_b = list(rng.normal(1.1, 10.0, 60))
    assert paramsync.bootstrap_confidence(noise_b, noise_a) < 0.9

    clear = list(rng.normal(8.0, 2.0, 60))
    weak = list(rng.normal(1.0, 2.0, 60))
    assert paramsync.bootstrap_confidence(clear, weak) > 0.95


def test_bootstrap_needs_a_sample_before_it_will_say_anything():
    assert paramsync.bootstrap_confidence([1.0, 2.0], [0.0, 0.0]) == 0.0


# --------------------------------------------------------------------------
# regime-scoped promoted sets (gap-closure item 5)
# --------------------------------------------------------------------------
class _FakeLibraryStore:
    """Stands in for solopt.store.RunStore's regime lookup, so this can be
    tested without a real wfmc.db on disk."""

    def __init__(self, entries: dict[str, list[dict]] | None = None) -> None:
        self.entries = entries or {}
        self.calls: list[tuple[str, float]] = []

    def nearest_regime_entries(self, symbol, target, *, n=1):
        self.calls.append((symbol, target))
        rows = sorted(
            self.entries.get(symbol, []),
            key=lambda e: abs(e["symbol_regime_score"] - target),
        )
        return rows[:n]


def _regime_entry(fingerprint, score, **params):
    return {"fingerprint": fingerprint, "symbol_regime_score": score, "params": params}


def test_selects_the_nearest_regime_entry_and_merges_its_params(settings):
    store = _FakeLibraryStore({"AAA": [
        _regime_entry("far", 0.9, volume_spike_multiple=9.0),
        _regime_entry("near", 0.42, volume_spike_multiple=3.5),
    ]})
    result = paramsync.select_regime_scoped_params(settings, "AAA", 0.40, store=store)

    assert result.applied is True
    assert result.fingerprint == "near"
    assert result.params["volume_spike_multiple"] == 3.5
    # everything else from the global config passes through untouched
    assert result.params["rr_min"] == settings["rr_min"]


def test_falls_back_when_the_nearest_entry_is_too_far(settings):
    store = _FakeLibraryStore({"AAA": [_regime_entry("far", 0.95, volume_spike_multiple=9.0)]})
    result = paramsync.select_regime_scoped_params(
        {**settings, "regime_scoped_max_distance": 0.1}, "AAA", 0.40, store=store
    )

    assert result.applied is False
    assert result.params is settings or result.params.get("volume_spike_multiple") == settings["volume_spike_multiple"]
    assert "beyond" in result.reason


def test_falls_back_when_the_symbol_has_no_library_entries(settings):
    store = _FakeLibraryStore({})
    result = paramsync.select_regime_scoped_params(settings, "ZZZ", 0.40, store=store)

    assert result.applied is False
    assert result.params == settings


def test_falls_back_when_disabled(settings):
    store = _FakeLibraryStore({"AAA": [_regime_entry("near", 0.40, volume_spike_multiple=3.5)]})
    cfg = {**settings, "regime_scoped_promotion_enabled": False}
    result = paramsync.select_regime_scoped_params(cfg, "AAA", 0.40, store=store)

    assert result.applied is False
    assert store.calls == []   # never even queries the library when disabled


def test_falls_back_when_the_current_regime_score_is_unavailable(settings):
    store = _FakeLibraryStore({"AAA": [_regime_entry("near", 0.40, volume_spike_multiple=3.5)]})
    result = paramsync.select_regime_scoped_params(settings, "AAA", None, store=store)

    assert result.applied is False
    assert result.reason == "regime score unavailable"


def test_a_library_lookup_failure_falls_back_rather_than_raising(settings):
    class BrokenStore:
        def nearest_regime_entries(self, *a, **kw):
            raise RuntimeError("disk full")

    result = paramsync.select_regime_scoped_params(settings, "AAA", 0.4, store=BrokenStore())
    assert result.applied is False
    assert result.params == settings


# --------------------------------------------------------------------------
# blended live strategy application (fuzzy-regime section, step 4)
# --------------------------------------------------------------------------
class _Membership:
    def __init__(self, percentages: dict[int, float]) -> None:
        self.percentages = percentages


def _cluster_entry(**params) -> dict:
    return {"params": params}


def test_blend_weighted_averages_a_continuous_parameter(settings):
    membership = _Membership({0: 0.6, 1: 0.4})
    entries = {
        0: _cluster_entry(volume_spike_multiple=2.0),
        1: _cluster_entry(volume_spike_multiple=4.0),
    }
    result = paramsync.blend_regime_params(settings, membership, entries)

    assert result.applied is True
    assert result.params["volume_spike_multiple"] == pytest.approx(0.6 * 2.0 + 0.4 * 4.0)
    assert result.weights == {0: pytest.approx(0.6), 1: pytest.approx(0.4)}


def test_blend_normalizes_weights_over_only_the_available_clusters(settings):
    """Membership carries a third cluster (0.3) with no promoted set - its
    weight must not just vanish, the other two must pick it up."""
    membership = _Membership({0: 0.5, 1: 0.2, 2: 0.3})
    entries = {0: _cluster_entry(rr_min=2.0), 1: _cluster_entry(rr_min=3.0)}

    result = paramsync.blend_regime_params(settings, membership, entries)

    assert result.weights[0] == pytest.approx(0.5 / 0.7)
    assert result.weights[1] == pytest.approx(0.2 / 0.7)
    assert 2 not in result.weights


def test_blend_takes_the_dominant_clusters_value_for_a_bitmask_setting(settings):
    membership = _Membership({0: 0.9, 1: 0.1})
    entries = {
        0: _cluster_entry(indicator_mask=15),
        1: _cluster_entry(indicator_mask=511),
    }
    result = paramsync.blend_regime_params(settings, membership, entries)
    assert result.params["indicator_mask"] == 15   # the 0.9-weight cluster's value, not an average


def test_blend_rounds_an_int_period_rather_than_leaving_it_fractional(settings):
    membership = _Membership({0: 0.5, 1: 0.5})
    entries = {0: _cluster_entry(ema_fast=9), 1: _cluster_entry(ema_fast=10)}
    result = paramsync.blend_regime_params(settings, membership, entries)
    assert isinstance(result.params["ema_fast"], int)


def test_blend_falls_back_with_no_per_regime_entries(settings):
    result = paramsync.blend_regime_params(settings, _Membership({0: 1.0}), {})
    assert result.applied is False
    assert result.params == settings


def test_blend_falls_back_when_membership_is_zero_in_every_available_cluster(settings):
    membership = _Membership({0: 0.0, 1: 0.0})
    entries = {0: _cluster_entry(rr_min=2.0), 1: _cluster_entry(rr_min=3.0)}
    result = paramsync.blend_regime_params(settings, membership, entries)
    assert result.applied is False


def test_blend_leaves_a_key_no_cluster_provides_untouched(settings):
    membership = _Membership({0: 1.0})
    entries = {0: _cluster_entry(volume_spike_multiple=5.0)}
    result = paramsync.blend_regime_params(settings, membership, entries)
    assert result.params["rr_min"] == settings["rr_min"]   # untouched, not zeroed
