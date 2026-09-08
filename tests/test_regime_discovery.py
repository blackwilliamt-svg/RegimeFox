"""Fuzzy-regime section, step 1: per-coin regime discovery from a Frames panel."""
from __future__ import annotations

import numpy as np
import pytest

from solopt.frames import frames_from_series
from solopt.regime_discovery import (
    CoinRegimeModel,
    MIN_SAMPLES_PER_COIN,
    compute_features,
    discover_all,
    discover_coin_regimes,
)
from solopt.schema import CRYPTO


def _trending_choppy_series(n: int = 300, seed: int = 1) -> np.ndarray:
    """A market that alternates between a clean trend and genuine chop, so
    fuzzy clustering has real structure to find rather than noise."""
    rng = np.random.default_rng(seed)
    closes = [100.0]
    for i in range(n - 1):
        if (i // 50) % 2 == 0:
            step = 0.004 + rng.normal(0, 0.0008)   # trending leg
        else:
            step = rng.normal(0, 0.006)             # choppy leg
        closes.append(max(0.01, closes[-1] * (1 + step)))
    close = np.array(closes)
    high = close * (1 + np.abs(rng.normal(0, 0.003, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.003, n)))
    volume = np.abs(rng.lognormal(8, 0.6, n))
    return np.vstack([np.arange(n) * 300, np.concatenate(([close[0]], close[:-1])), high, low, close, volume])


def _panel(**series):
    return frames_from_series(series, seconds=300, schema=CRYPTO)


def test_compute_features_has_the_expected_shape():
    frames = _panel(AAA=_trending_choppy_series(300))
    features = compute_features(frames)
    assert features.shape == (1, 300, 3)


def test_a_coin_with_enough_history_gets_a_model():
    frames = _panel(AAA=_trending_choppy_series(400))
    model = discover_coin_regimes(frames, "AAA", min_samples=200, seed=1)

    assert model is not None
    assert isinstance(model, CoinRegimeModel)
    assert 4 <= model.n_clusters <= 6
    assert model.n_samples >= 200


def test_a_coin_with_too_little_history_returns_none_not_a_bad_model():
    frames = _panel(AAA=_trending_choppy_series(50))
    model = discover_coin_regimes(frames, "AAA", min_samples=MIN_SAMPLES_PER_COIN)
    assert model is None


def test_an_unknown_symbol_returns_none():
    frames = _panel(AAA=_trending_choppy_series(400))
    assert discover_coin_regimes(frames, "ZZZ") is None


def test_discover_all_only_includes_coins_with_enough_history():
    frames = _panel(
        AAA=_trending_choppy_series(400, seed=1),
        BBB=_trending_choppy_series(50, seed=2),   # too short
    )
    models = discover_all(frames, min_samples=200)
    assert set(models) == {"AAA"}


def test_discover_all_computes_features_once_not_per_symbol(monkeypatch):
    """Discovery for a whole panel should compute the feature stack once and
    slice it per coin - not recompute ADX/ATR/volume-zscore for every symbol,
    which is exactly the 'compute once per distinct value' rule the rest of
    solopt.indicators already follows."""
    frames = _panel(
        AAA=_trending_choppy_series(400, seed=1),
        BBB=_trending_choppy_series(400, seed=2),
        CCC=_trending_choppy_series(400, seed=3),
    )
    calls = {"n": 0}
    import solopt.regime_discovery as rd

    real_compute = rd.compute_features

    def counting_compute(*a, **kw):
        calls["n"] += 1
        return real_compute(*a, **kw)

    monkeypatch.setattr(rd, "compute_features", counting_compute)
    discover_all(frames, min_samples=200)
    assert calls["n"] == 1


def test_model_round_trips_through_as_dict():
    frames = _panel(AAA=_trending_choppy_series(400))
    model = discover_coin_regimes(frames, "AAA", min_samples=200, seed=2)

    restored = CoinRegimeModel.from_dict(model.as_dict())
    assert restored.symbol == model.symbol
    assert restored.n_clusters == model.n_clusters
    assert np.allclose(restored.centroids, model.centroids)
    assert np.allclose(restored.scaler.mean, model.scaler.mean)
    assert np.allclose(restored.scaler.std, model.scaler.std)
