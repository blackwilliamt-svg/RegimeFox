"""Fuzzy-regime section, step 2: per-coin, per-regime walk-forward.

A synthetic two-regime market (a clean trending half and a genuinely choppy
half) so the fuzzy clustering has real structure, and so a window drawn from
each half should land predominantly in a different cluster - letting these
tests assert on which windows get selected, not just that something runs.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from solopt.engine import PortfolioSettings
from solopt.frames import frames_from_series
from solopt.params import ParamSpace
from solopt.regime_discovery import discover_coin_regimes
from solopt.regime_walkforward import (
    regime_membership_series,
    run_and_store_all_regimes,
    run_regime_scoped_search,
    windows_for_regime,
)
from solopt.schema import CRYPTO, CostModel
from solopt.store import RunStore
from solopt.walkforward import WalkForwardConfig, plan_windows

SECONDS = 300  # 5-minute bars


def _two_regime_series(n_days: int = 200, seed: int = 1) -> np.ndarray:
    """~n_days of 5-minute bars, alternating a trending week with a choppy
    week, so the coin's own fuzzy clustering has two real regimes to find."""
    bars_per_day = 86400 // SECONDS
    n = n_days * bars_per_day
    rng = np.random.default_rng(seed)
    closes = [100.0]
    week_bars = 7 * bars_per_day
    for i in range(n - 1):
        trending = (i // week_bars) % 2 == 0
        step = (0.0006 + rng.normal(0, 0.0002)) if trending else rng.normal(0, 0.004)
        closes.append(max(0.01, closes[-1] * (1 + step)))
    close = np.array(closes)
    high = close * (1 + np.abs(rng.normal(0, 0.002, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.002, n)))
    volume = np.abs(rng.lognormal(8, 0.5, n))
    start = int(time.time()) - n * SECONDS
    ts = start + np.arange(n) * SECONDS
    return np.vstack([ts, np.concatenate(([close[0]], close[:-1])), high, low, close, volume])


@pytest.fixture(scope="module")
def two_regime_frames():
    return frames_from_series({"AAA": _two_regime_series(200)}, seconds=SECONDS, schema=CRYPTO)


@pytest.fixture(scope="module")
def two_regime_model(two_regime_frames):
    model = discover_coin_regimes(two_regime_frames, "AAA", k_range=(2,), min_samples=200, seed=1)
    assert model is not None
    return model


def _small_space() -> ParamSpace:
    space = ParamSpace()
    # Shrink to a handful of combinations so the search finishes fast - these
    # tests are about window selection and storage, not search quality.
    space.values = {
        "volume_spike_multiple": [1.5, 2.0],
        "momentum_min_pct": [0.004],
        "rsi_max_entry": [78.0],
        "ema_fast": [9],
        "ema_slow": [21],
        "rsi_period": [14],
        "atr_period": [14],
        "volume_spike_lookback": [20],
        "momentum_candles": [3],
        "stop_atr_mult": [1.2],
        "rr_min": [2.0],
        "rr_max": [4.0],
        "trailing_activate_r": [1.0],
        "trailing_distance_atr": [1.5],
        "signal_invalidation_bars": [2],
        "max_hold_minutes": [240],
        "regime_lookback": [20],
        "regime_trend_er": [0.3],
        "regime_chop_atr_pct": [0.03],
        "regime_allowed": [7],
        "confluence_required": [0],
    }
    return space


def _portfolio() -> PortfolioSettings:
    return PortfolioSettings(
        starting_balance=1000.0, max_total_deployed_pct=0.9, max_position_pct_of_wallet=0.45,
        max_position_pct_of_liquidity=0.01, min_position_usd=10.0, min_candles_required=40,
        volatility_target_atr_pct=0.03, volatility_size_floor=0.35,
        correlation_lookback=60, correlation_max=0.8,
        costs=CostModel(fee_pct=0.25, slippage_pct=0.5),
        confluence_multiples=(), sizing_mode="flat",
    )


def _wf_config() -> WalkForwardConfig:
    return WalkForwardConfig(
        in_sample_days=30, out_of_sample_days=7, step_days=7,
        max_evaluations=20, batch_size=8, workers=1, min_trades_per_window=0,
        library_seed_fraction=0.0,
    )


# --------------------------------------------------------------------------
# membership / window filtering
# --------------------------------------------------------------------------
def test_regime_membership_series_has_one_column_per_cluster(two_regime_frames, two_regime_model):
    membership = regime_membership_series(two_regime_frames, "AAA", two_regime_model)
    assert membership.shape == (two_regime_frames.n_bars, two_regime_model.n_clusters)
    valid = ~np.isnan(membership).any(axis=1)
    assert np.allclose(membership[valid].sum(axis=1), 1.0, atol=1e-6)


def test_windows_for_regime_returns_a_subset_of_all_windows(two_regime_frames, two_regime_model):
    idx = two_regime_frames.symbols.index("AAA")
    ts = two_regime_frames.ts[idx][two_regime_frames.mask[idx]]
    all_windows = plan_windows(int(ts.min()), int(ts.max()), _wf_config())
    assert all_windows

    kept = windows_for_regime(two_regime_frames, "AAA", two_regime_model, 0, all_windows)
    assert len(kept) <= len(all_windows)
    assert all(w in all_windows for w in kept)


def test_the_two_clusters_select_different_windows(two_regime_frames, two_regime_model):
    """With two genuinely different regimes in the data, cluster 0 and
    cluster 1 should not both dominate the exact same windows."""
    idx = two_regime_frames.symbols.index("AAA")
    ts = two_regime_frames.ts[idx][two_regime_frames.mask[idx]]
    all_windows = plan_windows(int(ts.min()), int(ts.max()), _wf_config())

    kept_0 = {w.index for w in windows_for_regime(two_regime_frames, "AAA", two_regime_model, 0, all_windows)}
    kept_1 = {w.index for w in windows_for_regime(two_regime_frames, "AAA", two_regime_model, 1, all_windows)}
    assert kept_0 != kept_1


# --------------------------------------------------------------------------
# the search + storage loop
# --------------------------------------------------------------------------
def test_run_regime_scoped_search_returns_none_for_a_cluster_with_too_few_windows(
    two_regime_frames, two_regime_model
):
    outcome = run_regime_scoped_search(
        two_regime_frames, "AAA", two_regime_model, cluster_id=0,
        space=_small_space(), portfolio=_portfolio(), wf_config=_wf_config(),
        min_windows=10_000,  # impossible to reach - forces the "too few" path
    )
    assert outcome is None


def test_run_and_store_all_regimes_covers_every_cluster(tmp_path, two_regime_frames, two_regime_model):
    store = RunStore(tmp_path / "wfmc.db")
    seen = []

    outcomes = run_and_store_all_regimes(
        two_regime_frames, "AAA", two_regime_model,
        space=_small_space(), portfolio=_portfolio(), wf_config=_wf_config(),
        store=store, min_windows=1, on_progress=lambda c, n: seen.append((c, n)),
    )

    assert seen == [(0, two_regime_model.n_clusters), (1, two_regime_model.n_clusters)]
    # Every accepted outcome must have been written to the library under its
    # own cluster id.
    for cluster_id, outcome in outcomes.items():
        if outcome.accepted:
            entries = store.regime_cluster_entries("AAA")
            assert cluster_id in entries


def test_a_progress_callback_that_raises_does_not_abort_the_run(tmp_path, two_regime_frames, two_regime_model):
    store = RunStore(tmp_path / "wfmc.db")

    def broken_progress(cluster_id, n):
        raise RuntimeError("dashboard feed offline")

    outcomes = run_and_store_all_regimes(
        two_regime_frames, "AAA", two_regime_model,
        space=_small_space(), portfolio=_portfolio(), wf_config=_wf_config(),
        store=store, min_windows=1, on_progress=broken_progress,
    )
    assert isinstance(outcomes, dict)   # completed despite the broken callback
