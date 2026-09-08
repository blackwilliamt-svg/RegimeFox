"""Fuzzy-regime section: the fuzzy c-means core, pure NumPy (no sklearn).

Correctness first (membership sums to 1, clusters land near the true blobs
they were sampled from, live classification agrees with training-time
membership), then the standardizer and discover_best_k's model selection.
"""
from __future__ import annotations

import numpy as np
import pytest

from solopt.fuzzy import (
    Standardizer,
    discover_best_k,
    fuzzy_cmeans,
    membership_for,
    membership_stack,
)


def _three_blobs(seed: int = 0, n_per_blob: int = 60) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centers = np.array([[0.0, 0.0], [10.0, 10.0], [10.0, -10.0]])
    blobs = [rng.normal(c, 0.5, size=(n_per_blob, 2)) for c in centers]
    return np.vstack(blobs), centers


def test_membership_sums_to_one_for_every_sample():
    X, _ = _three_blobs()
    result = fuzzy_cmeans(X, 3, seed=1)
    sums = result.membership.sum(axis=1)
    assert np.allclose(sums, 1.0, atol=1e-6)


def test_membership_is_between_zero_and_one():
    X, _ = _three_blobs()
    result = fuzzy_cmeans(X, 3, seed=1)
    assert (result.membership >= 0.0).all()
    assert (result.membership <= 1.0).all()


def test_centroids_land_near_the_true_blob_centers():
    X, centers = _three_blobs()
    result = fuzzy_cmeans(X, 3, seed=2)

    # Each true center should have a discovered centroid within striking
    # distance - order is not guaranteed, so match each to its nearest.
    for center in centers:
        dist = np.linalg.norm(result.centroids - center[None, :], axis=1)
        assert dist.min() < 1.5


def test_a_point_deep_inside_one_blob_is_mostly_one_membership():
    X, centers = _three_blobs()
    result = fuzzy_cmeans(X, 3, seed=3)

    # The point closest to the first true center should be dominated by
    # whichever discovered cluster is nearest that center.
    nearest_centroid_to_true = np.argmin(
        np.linalg.norm(result.centroids - centers[0][None, :], axis=1)
    )
    point_idx = np.argmin(np.linalg.norm(X - centers[0][None, :], axis=1))
    assert result.membership[point_idx, nearest_centroid_to_true] > 0.9


def test_a_point_exactly_between_two_clusters_is_split_roughly_evenly():
    # Two well-separated 1-D clusters; a point exactly at the midpoint has no
    # reason to favour either.
    left = np.random.default_rng(4).normal(-5.0, 0.3, size=(40, 1))
    right = np.random.default_rng(5).normal(5.0, 0.3, size=(40, 1))
    X = np.vstack([left, right])
    result = fuzzy_cmeans(X, 2, seed=6)

    midpoint = np.array([[0.0]])
    u = membership_for(midpoint[0], result.centroids)
    assert abs(u[0] - u[1]) < 0.15


def test_fuzzy_cmeans_rejects_more_clusters_than_samples():
    X = np.zeros((3, 2))
    with pytest.raises(ValueError):
        fuzzy_cmeans(X, 5)


def test_membership_for_matches_training_membership_on_a_training_point():
    """The live classification path (step 3) must agree with what discovery
    (step 1) already computed for the same point - two different code paths
    computing the same thing is exactly where they could quietly diverge."""
    X, _ = _three_blobs()
    result = fuzzy_cmeans(X, 3, seed=7)

    for i in (0, 25, 80, 150):
        live = membership_for(X[i], result.centroids)
        assert np.allclose(live, result.membership[i], atol=0.05)


def test_membership_for_puts_full_weight_on_an_exact_centroid_match():
    centroids = np.array([[0.0, 0.0], [10.0, 10.0], [10.0, -10.0]])
    u = membership_for(centroids[1], centroids)
    assert u[1] == pytest.approx(1.0)
    assert u[0] == pytest.approx(0.0)
    assert u[2] == pytest.approx(0.0)


# --------------------------------------------------------------------------
# standardization
# --------------------------------------------------------------------------
def test_standardizer_produces_zero_mean_unit_std():
    X = np.array([[1.0, 100.0], [2.0, 200.0], [3.0, 300.0], [4.0, 400.0]])
    scaler = Standardizer.fit(X)
    scaled = scaler.transform(X)
    assert np.allclose(scaled.mean(axis=0), 0.0, atol=1e-9)
    assert np.allclose(scaled.std(axis=0), 1.0, atol=1e-9)


def test_standardizer_does_not_divide_by_zero_for_a_constant_feature():
    X = np.array([[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]])
    scaler = Standardizer.fit(X)
    scaled = scaler.transform(X)
    assert np.isfinite(scaled).all()
    assert np.allclose(scaled[:, 0], 0.0)   # the constant column stays at 0, not NaN/inf


def test_standardizer_round_trips_through_a_dict():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    scaler = Standardizer.fit(X)
    restored = Standardizer.from_dict(scaler.as_dict())
    assert np.allclose(restored.transform(X), scaler.transform(X))


# --------------------------------------------------------------------------
# model selection
# --------------------------------------------------------------------------
def test_discover_best_k_picks_a_k_within_the_candidate_range():
    X, _ = _three_blobs()
    result = discover_best_k(X, k_range=(2, 3, 4, 5), seed=8)
    assert result.centroids.shape[0] in (2, 3, 4, 5)


def test_discover_best_k_skips_a_k_larger_than_the_sample_count():
    X = np.random.default_rng(9).normal(size=(4, 2))
    result = discover_best_k(X, k_range=(2, 3, 10))
    assert result.centroids.shape[0] in (2, 3)


# --------------------------------------------------------------------------
# vectorized membership (many points at once)
# --------------------------------------------------------------------------
def test_membership_stack_matches_membership_for_pointwise():
    X, _ = _three_blobs()
    result = fuzzy_cmeans(X, 3, seed=10)

    stack = membership_stack(X, result.centroids)
    for i in (0, 40, 90, 150):
        pointwise = membership_for(X[i], result.centroids)
        assert np.allclose(stack[i], pointwise, atol=1e-9)


def test_membership_stack_rows_sum_to_one():
    X, _ = _three_blobs()
    result = fuzzy_cmeans(X, 3, seed=11)
    stack = membership_stack(X, result.centroids)
    assert np.allclose(stack.sum(axis=1), 1.0, atol=1e-6)
