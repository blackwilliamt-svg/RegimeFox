"""Fuzzy c-means clustering, pure NumPy.

Per-coin regime discovery (the fuzzy-regime section) needs soft clustering -
a coin sits 60% in one regime and 30% in another, not hard-assigned to one -
and solopt's one hard dependency is NumPy (see the module docstring in
indicators.py): scikit-learn or scikit-fuzzy would be the obvious library
choice, but pulling in a dependency the rest of this package deliberately
avoids for one feature is the wrong trade. Bezdek's fuzzy c-means is a short,
well-understood iterative algorithm, and implementing it directly keeps that
"NumPy only" guarantee true for every caller, including a RunPod worker image
that otherwise has nothing to install beyond numpy itself.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(slots=True)
class FuzzyCMeansResult:
    """One clustering's outcome. `membership[i, k]` is how much sample `i`
    belongs to cluster `k`, in [0, 1], summing to 1 across `k`."""

    centroids: np.ndarray       # [n_clusters, n_features]
    membership: np.ndarray      # [n_samples, n_clusters]
    n_iter: int
    fpc: float                  # fuzzy partition coefficient, 1/k (max fuzz) .. 1 (hard)


def fuzzy_cmeans(
    X: np.ndarray,
    n_clusters: int,
    *,
    m: float = 2.0,
    max_iter: int = 150,
    tol: float = 1e-5,
    seed: int = 0,
) -> FuzzyCMeansResult:
    """Bezdek's fuzzy c-means.

    `X` is `[n_samples, n_features]`, already standardized by the caller -
    clustering on raw features whose scales differ by orders of magnitude
    (ADX ~0-100, ATR% ~0-0.1) would let whichever feature has the largest
    numeric range dominate every distance regardless of how informative it
    actually is.

    `m` is the fuzziness exponent (2.0 is the standard default: m -> 1 is
    hard k-means, m -> infinity is maximally fuzzy). Converges when the
    membership matrix stops moving by more than `tol`, or after `max_iter`
    rounds either way.
    """
    X = np.asarray(X, dtype=np.float64)
    n_samples, _n_features = X.shape
    if n_samples < n_clusters:
        raise ValueError(
            f"{n_samples} samples cannot form {n_clusters} clusters"
        )

    rng = np.random.default_rng(seed)
    u = rng.random((n_samples, n_clusters))
    u /= u.sum(axis=1, keepdims=True)

    centroids = np.zeros((n_clusters, X.shape[1]), dtype=np.float64)
    power = 2.0 / (m - 1.0)
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        um = u ** m
        weight = um.sum(axis=0)
        centroids = (um.T @ X) / np.maximum(weight[:, None], 1e-12)

        dist = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
        dist = np.maximum(dist, 1e-12)
        # u_ik = 1 / sum_j (d_ik / d_ij)^(2/(m-1))
        ratio = dist[:, :, None] / dist[:, None, :]
        new_u = 1.0 / np.power(ratio, power).sum(axis=2)

        diff = float(np.linalg.norm(new_u - u))
        u = new_u
        if diff < tol:
            break

    fpc = float(np.sum(u ** 2) / n_samples)
    return FuzzyCMeansResult(centroids=centroids, membership=u, n_iter=n_iter, fpc=fpc)


def discover_best_k(
    X: np.ndarray,
    k_range: Sequence[int] = (4, 5, 6),
    **kwargs,
) -> FuzzyCMeansResult:
    """Run fuzzy c-means for each candidate cluster count and keep the one
    with the highest fuzzy partition coefficient - the standard simple
    validity measure for FCM (Bezdek 1974): higher means the partition is
    less ambiguous, i.e. the discovered regimes are more distinct rather than
    the algorithm just spreading membership thin over an ill-fitting k.
    """
    best: FuzzyCMeansResult | None = None
    for k in k_range:
        if X.shape[0] < k:
            continue
        result = fuzzy_cmeans(X, k, **kwargs)
        if best is None or result.fpc > best.fpc:
            best = result
    if best is None:
        raise ValueError(
            f"not enough samples ({X.shape[0]}) for the smallest candidate k in {k_range!r}"
        )
    return best


def membership_for(
    x: np.ndarray, centroids: np.ndarray, *, m: float = 2.0
) -> np.ndarray:
    """Live classification (fuzzy-regime section, step 3): membership degrees
    of one feature vector `x` `[n_features]` against stored `centroids`
    `[n_clusters, n_features]` - the exact same formula fuzzy_cmeans converges
    with, applied once rather than iterated, since the centroids are already
    fixed from discovery.
    """
    x = np.asarray(x, dtype=np.float64)
    centroids = np.asarray(centroids, dtype=np.float64)
    dist = np.linalg.norm(centroids - x[None, :], axis=1)

    on_centroid = dist < 1e-9
    if on_centroid.any():
        u = np.zeros(len(centroids), dtype=np.float64)
        u[np.argmax(on_centroid)] = 1.0
        return u

    power = 2.0 / (m - 1.0)
    inv = 1.0 / np.power(dist, power)
    return inv / inv.sum()


def membership_stack(
    X: np.ndarray, centroids: np.ndarray, *, m: float = 2.0
) -> np.ndarray:
    """Vectorized :func:`membership_for` over many points at once -
    `X` is `[n_samples, n_features]`, returns `[n_samples, n_clusters]`.

    Used to classify a whole bar history at once (per-regime walk-forward,
    fuzzy-regime section step 2, needs every historical bar's membership to
    pick which windows belong to a regime) rather than one live point at a
    time (step 3's membership_for).
    """
    X = np.asarray(X, dtype=np.float64)
    centroids = np.asarray(centroids, dtype=np.float64)
    dist = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)  # [n, k]

    out = np.empty_like(dist)
    on_centroid = dist < 1e-9
    any_on_centroid = on_centroid.any(axis=1)

    power = 2.0 / (m - 1.0)
    safe = np.maximum(dist, 1e-12)
    inv = 1.0 / np.power(safe, power)
    out = inv / inv.sum(axis=1, keepdims=True)

    if any_on_centroid.any():
        hard = on_centroid.astype(np.float64)
        hard /= np.maximum(hard.sum(axis=1, keepdims=True), 1e-12)
        out = np.where(any_on_centroid[:, None], hard, out)
    return out


@dataclass(slots=True)
class Standardizer:
    """Per-feature mean/std, fit once at discovery time and reused for every
    later live classification - classifying against centroids computed in
    standardized space only makes sense if new points are standardized the
    same way, not re-fit against whatever the live window happens to contain.
    """

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, X: np.ndarray) -> "Standardizer":
        X = np.asarray(X, dtype=np.float64)
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std = np.where(std < 1e-9, 1.0, std)  # a constant feature must not divide by ~0
        return cls(mean=mean, std=std)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, dtype=np.float64) - self.mean) / self.std

    def as_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, data: dict) -> "Standardizer":
        return cls(mean=np.asarray(data["mean"], dtype=np.float64), std=np.asarray(data["std"], dtype=np.float64))
