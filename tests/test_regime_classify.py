"""Fuzzy-regime section, step 1 (persistence) and step 3 (live classification)."""
from __future__ import annotations

import numpy as np
import pytest

from solbot.indicators import Snapshot
from solbot.regime_classify import classify_current_regime, feature_vector_from_snapshot
from solopt.fuzzy import Standardizer, discover_best_k
from solopt.regime_discovery import CoinRegimeModel
from solopt.store import RunStore


def _fake_model(symbol: str) -> CoinRegimeModel:
    rng = np.random.default_rng(1)
    X = np.vstack(
        [
            rng.normal([10.0, 0.01, -1.0], 1.0, size=(50, 3)),
            rng.normal([40.0, 0.05, 1.5], 1.0, size=(50, 3)),
        ]
    )
    scaler = Standardizer.fit(X)
    result = discover_best_k(scaler.transform(X), k_range=(2,), seed=1)
    return CoinRegimeModel.from_result(symbol, scaler, result, n_samples=X.shape[0])


def _snap(adx: float, atr_pct: float, volume_zscore: float) -> Snapshot:
    return Snapshot(
        ts=0, close=1.0, ema_fast=1.0, ema_slow=1.0, rsi=50.0, atr=0.01, atr_pct=atr_pct,
        volume=100.0, volume_ratio=1.0, momentum_pct=0.0, consecutive_up=False, bars=100,
        adx=adx, volume_zscore=volume_zscore,
    )


# --------------------------------------------------------------------------
# persistence (store.py)
# --------------------------------------------------------------------------
def test_save_and_get_regime_model_round_trips(tmp_path):
    store = RunStore(tmp_path / "wfmc.db")
    model = _fake_model("AAA")

    store.save_regime_model("AAA", model.as_dict(), run_id=7)
    stored = store.get_regime_model("AAA")

    assert stored is not None
    assert stored["run_id"] == 7
    restored = CoinRegimeModel.from_dict(stored["model"])
    assert np.allclose(restored.centroids, model.centroids)


def test_save_regime_model_replaces_the_previous_one_wholesale(tmp_path):
    store = RunStore(tmp_path / "wfmc.db")
    store.save_regime_model("AAA", _fake_model("AAA").as_dict(), run_id=1)

    newer = _fake_model("AAA")
    store.save_regime_model("AAA", newer.as_dict(), run_id=2)

    stored = store.get_regime_model("AAA")
    assert stored["run_id"] == 2
    assert np.allclose(
        np.asarray(stored["model"]["centroids"]), newer.centroids
    )


def test_get_regime_model_is_none_for_an_unseen_symbol(tmp_path):
    store = RunStore(tmp_path / "wfmc.db")
    assert store.get_regime_model("ZZZ") is None


def test_list_regime_models_lists_every_symbol(tmp_path):
    store = RunStore(tmp_path / "wfmc.db")
    store.save_regime_model("AAA", _fake_model("AAA").as_dict())
    store.save_regime_model("BBB", _fake_model("BBB").as_dict())

    listed = {r["symbol"] for r in store.list_regime_models()}
    assert listed == {"AAA", "BBB"}


# --------------------------------------------------------------------------
# live classification (regime_classify.py)
# --------------------------------------------------------------------------
def test_feature_vector_from_snapshot_matches_discovery_order():
    snap = _snap(adx=25.0, atr_pct=0.02, volume_zscore=0.5)
    assert feature_vector_from_snapshot(snap) == (25.0, 0.02, 0.5)


def test_classify_current_regime_returns_none_without_a_stored_model(tmp_path):
    store = RunStore(tmp_path / "wfmc.db")
    result = classify_current_regime("AAA", _snap(20.0, 0.02, 0.0), store=store)
    assert result is None


def test_classify_current_regime_matches_membership_for_a_point_near_a_centroid(tmp_path):
    store = RunStore(tmp_path / "wfmc.db")
    model = _fake_model("AAA")
    store.save_regime_model("AAA", model.as_dict())

    # A feature vector close to the first training blob's raw location.
    snap = _snap(adx=10.0, atr_pct=0.01, volume_zscore=-1.0)
    result = classify_current_regime("AAA", snap, store=store)

    assert result is not None
    assert result.symbol == "AAA"
    assert result.n_clusters == 2
    assert sum(result.percentages.values()) == pytest.approx(1.0, abs=1e-6)
    # It should be dominated by whichever cluster is nearest that blob.
    assert max(result.percentages.values()) > 0.5


def test_classify_current_regime_percentages_sum_to_one(tmp_path):
    store = RunStore(tmp_path / "wfmc.db")
    store.save_regime_model("AAA", _fake_model("AAA").as_dict())

    result = classify_current_regime("AAA", _snap(25.0, 0.03, 0.2), store=store)
    assert sum(result.percentages.values()) == pytest.approx(1.0, abs=1e-6)


def test_classify_current_regime_returns_none_on_a_nan_feature(tmp_path):
    store = RunStore(tmp_path / "wfmc.db")
    store.save_regime_model("AAA", _fake_model("AAA").as_dict())

    snap = _snap(adx=float("nan"), atr_pct=0.02, volume_zscore=0.0)
    assert classify_current_regime("AAA", snap, store=store) is None


def test_classify_current_regime_falls_back_silently_on_a_broken_store(tmp_path):
    class BrokenStore:
        def get_regime_model(self, symbol):
            raise RuntimeError("disk full")

    result = classify_current_regime("AAA", _snap(20.0, 0.02, 0.0), store=BrokenStore())
    assert result is None


def test_dominant_cluster_picks_the_highest_membership():
    from solbot.regime_classify import RegimeMembership

    membership = RegimeMembership(
        symbol="AAA", percentages={0: 0.2, 1: 0.7, 2: 0.1}, n_clusters=3,
        feature_vector=(1.0, 2.0, 3.0),
    )
    assert membership.dominant_cluster() == 1
