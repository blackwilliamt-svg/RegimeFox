"""Gap-closure items 4 and 5: the persistent parameter library.

Item 4: validated combinations survive a run and seed the next search's
initial candidate pool rather than starting from DEFAULT_GRID every time.
Item 5: entries carry continuous regime tags (market-wide and per-symbol) so
a live selection can find the nearest regime match - see test_paramsync.py /
test_wfmc.py for the promotion-time and live-engine consumers of this.
"""
from __future__ import annotations

import numpy as np
import pytest

from solopt.frames import frames_from_series
from solopt.params import DEFAULT_GRID, ParamSpace, Search
from solopt.pipeline import _regime_scores, _update_library
from solopt.promotion import ParameterBundle
from solopt.schema import CRYPTO
from solopt.store import LibraryEntry, RunStore, _library_score, _recency_weight, now
from solopt.walkforward import WalkForwardResult


def _entry(fp: str, *, efficiency: float, mean_return: float = 0.0, symbol=None, **kw) -> LibraryEntry:
    return LibraryEntry(
        fingerprint=fp,
        params={"volume_spike_multiple": 2.0, "indicator_mask": 15},
        performance={"walk_forward_efficiency": efficiency, "mean_window_return": mean_return},
        symbol=symbol,
        **kw,
    )


def test_upsert_inserts_a_new_entry(tmp_path):
    store = RunStore(tmp_path / "wfmc.db")
    created = store.upsert_library(_entry("fp1", efficiency=0.5))

    assert created is True
    row = store.library_entry("fp1")
    assert row is not None
    assert row["params"]["volume_spike_multiple"] == 2.0
    assert row["times_seen"] == 1


def test_upsert_keeps_the_better_performing_entry_per_fingerprint(tmp_path):
    store = RunStore(tmp_path / "wfmc.db")
    store.upsert_library(_entry("fp1", efficiency=0.3))
    changed = store.upsert_library(_entry("fp1", efficiency=0.7))

    assert changed is True
    row = store.library_entry("fp1")
    assert row["performance"]["walk_forward_efficiency"] == 0.7
    assert row["times_seen"] == 2


def test_upsert_does_not_overwrite_a_better_incumbent(tmp_path):
    store = RunStore(tmp_path / "wfmc.db")
    store.upsert_library(_entry("fp1", efficiency=0.9))
    changed = store.upsert_library(_entry("fp1", efficiency=0.2))

    assert changed is False
    row = store.library_entry("fp1")
    assert row["performance"]["walk_forward_efficiency"] == 0.9
    assert row["times_seen"] == 2   # still recorded as a repeat appearance


def test_top_library_entries_ranks_by_score_descending(tmp_path):
    store = RunStore(tmp_path / "wfmc.db")
    store.upsert_library(_entry("low", efficiency=0.1))
    store.upsert_library(_entry("high", efficiency=0.9))
    store.upsert_library(_entry("mid", efficiency=0.5))

    top2 = store.top_library_entries(2)
    assert [e["fingerprint"] for e in top2] == ["high", "mid"]


def test_top_library_entries_separates_market_wide_from_per_symbol(tmp_path):
    store = RunStore(tmp_path / "wfmc.db")
    store.upsert_library(_entry("global1", efficiency=0.8))
    store.upsert_library(_entry("aaa1", efficiency=0.8, symbol="AAA"))

    market_wide = store.top_library_entries(10)
    per_symbol = store.top_library_entries(10, symbol="AAA")

    assert [e["fingerprint"] for e in market_wide] == ["global1"]
    assert [e["fingerprint"] for e in per_symbol] == ["aaa1"]


def test_recency_weighting_favours_a_recently_seen_entry_over_a_stale_better_one(tmp_path):
    store = RunStore(tmp_path / "wfmc.db")
    store.upsert_library(_entry("stale_best", efficiency=1.0))
    # Backdate it far past the half-life so its recency weight collapses.
    store.conn.execute(
        "UPDATE library SET last_seen_at = ? WHERE fingerprint = ?",
        (now() - 120 * 86400, "stale_best"),
    )
    store.upsert_library(_entry("fresh_good", efficiency=0.6))

    top1 = store.top_library_entries(1, half_life_days=30.0)
    assert top1[0]["fingerprint"] == "fresh_good"


def test_nearest_regime_entries_orders_by_distance_not_performance(tmp_path):
    store = RunStore(tmp_path / "wfmc.db")
    store.upsert_library(
        _entry("far_but_better", efficiency=0.9, symbol="AAA", symbol_regime_score=0.9)
    )
    store.upsert_library(
        _entry("close_but_worse", efficiency=0.2, symbol="AAA", symbol_regime_score=0.42)
    )

    nearest = store.nearest_regime_entries("AAA", target_regime_score=0.40, n=1)
    assert nearest[0]["fingerprint"] == "close_but_worse"


def test_nearest_regime_entries_is_empty_for_an_unseen_symbol(tmp_path):
    store = RunStore(tmp_path / "wfmc.db")
    store.upsert_library(_entry("aaa1", efficiency=0.5, symbol="AAA", symbol_regime_score=0.3))

    assert store.nearest_regime_entries("BBB", target_regime_score=0.3) == []


def test_prune_library_drops_the_lowest_scored_entries_first(tmp_path):
    store = RunStore(tmp_path / "wfmc.db")
    for i in range(5):
        store.upsert_library(_entry(f"fp{i}", efficiency=i / 10.0))

    deleted = store.prune_library(keep=3)

    assert deleted == 2
    remaining = {r["fingerprint"] for r in store.conn.execute("SELECT fingerprint FROM library")}
    assert remaining == {"fp2", "fp3", "fp4"}   # the three highest-efficiency entries


def test_prune_library_is_a_noop_under_the_cap(tmp_path):
    store = RunStore(tmp_path / "wfmc.db")
    store.upsert_library(_entry("fp1", efficiency=0.5))

    assert store.prune_library(keep=500) == 0


def test_library_score_prefers_efficiency_then_mean_return():
    assert _library_score({"walk_forward_efficiency": 0.5, "mean_window_return": 0.01}) > (
        _library_score({"walk_forward_efficiency": 0.4, "mean_window_return": 100.0})
    )


def test_recency_weight_decays_toward_zero_but_never_negative():
    fresh = _recency_weight(now(), now_ts=now())
    old = _recency_weight(now() - 365 * 86400, now_ts=now())
    assert fresh == 1.0
    assert 0.0 < old < 0.01


# --------------------------------------------------------------------------
# seeding a new search from the library (item 4)
# --------------------------------------------------------------------------
def _rng():
    return np.random.default_rng(1234)


def test_sample_seeded_prioritises_the_pool_before_filling_with_random():
    space = ParamSpace()
    pool = [
        {**{n: space.values[n][0] for n in space.names}, "ema_fast": 9, "ema_slow": 34}
    ]
    batch = space.sample_seeded(10, _rng(), seed_pool=pool, seed_fraction=1.0)

    assert any(c["ema_fast"] == 9 and c["ema_slow"] == 34 for c in batch)
    assert len(batch) == 10   # still filled up to `count` even with a pool of one


def test_sample_seeded_ignores_axes_the_pool_entry_does_not_have():
    space = ParamSpace()
    incomplete = {"ema_fast": 9}   # missing every other tunable axis
    batch = space.sample_seeded(5, _rng(), seed_pool=[incomplete], seed_fraction=0.2)

    assert len(batch) == 5
    # The seeded combo (if it survived is_valid) falls back to each missing
    # axis's own first grid value rather than raising.
    seeded = [c for c in batch if c["ema_fast"] == 9]
    if seeded:
        assert seeded[0]["ema_slow"] == DEFAULT_GRID["ema_slow"][0]


def test_sample_seeded_with_zero_fraction_behaves_like_plain_sample():
    space = ParamSpace()
    batch = space.sample_seeded(5, _rng(), seed_pool=[{"ema_fast": 9}], seed_fraction=0.0)
    assert len(batch) == 5


def test_search_only_seeds_the_very_first_batch():
    space = ParamSpace()
    pool = [{**{n: space.values[n][0] for n in space.names}, "ema_fast": 9, "ema_slow": 34}]
    search = Search(
        space=space, max_evaluations=40, batch_size=8, refine_after=1,
        seed_pool=pool, seed_fraction=1.0,
    )

    batches = search.batches()
    first = next(batches)
    assert any(c["ema_fast"] == 9 and c["ema_slow"] == 34 for c in first)
    search.observe([(c, 1.0) for c in first])

    second = next(batches)   # round 1: refine_after=1 means neighbours() now, not the pool
    assert len(second) > 0   # just proving the search keeps going past the seeded round


# --------------------------------------------------------------------------
# the library actually gets written to (item 4), tagged with regime (item 5)
# --------------------------------------------------------------------------
def _synthetic_frames(seed: int = 3, n: int = 200):
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, n))
    rows = np.vstack(
        [
            np.arange(n) * 300,
            np.concatenate(([close[0]], close[:-1])),
            close * 1.003,
            close * 0.997,
            close,
            np.abs(rng.lognormal(8, 0.5, n)),
        ]
    )
    return frames_from_series({"AAA": rows, "BBB": rows * 1.01}, seconds=300, schema=CRYPTO)


def test_regime_scores_returns_a_market_wide_and_a_per_symbol_reading():
    frames = _synthetic_frames()
    market, per_symbol = _regime_scores(frames, lookback=20)

    assert market is not None
    assert set(per_symbol) == {"AAA", "BBB"}
    assert all(np.isfinite(v) for v in per_symbol.values())


def test_update_library_writes_a_market_wide_and_per_symbol_entries(tmp_path):
    store = RunStore(tmp_path / "wfmc.db")
    frames = _synthetic_frames()
    outcome = WalkForwardResult(
        best_params={"ema_fast": 9, "ema_slow": 21, "regime_lookback": 20},
        per_symbol_params={"AAA": {"ema_fast": 12}},
        walk_forward_efficiency=0.6,
        mean_window_return=0.02,
    )
    bundle = ParameterBundle(global_params=outcome.best_params)

    _update_library(store, bundle, outcome, frames, run_id=1)

    market_wide = store.top_library_entries(10)
    per_symbol = store.top_library_entries(10, symbol="AAA")
    assert len(market_wide) == 1
    assert market_wide[0]["params"]["ema_fast"] == 9
    assert len(per_symbol) == 1
    assert per_symbol[0]["params"]["ema_fast"] == 12   # override applied on top of global
    assert per_symbol[0]["symbol_regime_score"] is not None
    assert per_symbol[0]["market_regime_score"] == market_wide[0]["market_regime_score"]


def test_update_library_is_a_noop_when_the_store_has_no_upsert_method(tmp_path):
    class BareStore:
        pass

    frames = _synthetic_frames()
    outcome = WalkForwardResult(best_params={"ema_fast": 9, "ema_slow": 21})
    bundle = ParameterBundle(global_params=outcome.best_params)

    _update_library(BareStore(), bundle, outcome, frames, run_id=1)  # must not raise


def test_update_library_does_nothing_without_a_winning_combo(tmp_path):
    store = RunStore(tmp_path / "wfmc.db")
    frames = _synthetic_frames()
    outcome = WalkForwardResult(best_params=None)
    bundle = ParameterBundle(global_params={})

    _update_library(store, bundle, outcome, frames, run_id=1)

    assert store.top_library_entries(10) == []
