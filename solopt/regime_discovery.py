"""Per-coin fuzzy regime discovery (fuzzy-regime section, step 1).

Trend strength (ADX), volatility (ATR%), and volume behaviour (a rolling
z-score) computed per bar, then clustered independently *per coin* rather
than market-wide: a large-cap's "choppy" and a fresh, thinly-traded token's
"choppy" sit at completely different absolute feature values, and pooling
them into one clustering would just teach the model where the market-cap
split is, not what a regime transition looks like for either coin on its own.

This is meant to run as part of the monthly RunPod job (the README's existing
"part of the monthly RunPod job" framing) or the manual trigger (step 5) -
never continuously; a coin's regimes do not need re-discovering every day the
way its parameters might.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .frames import Frames
from .fuzzy import FuzzyCMeansResult, Standardizer, discover_best_k
from .indicators import adx_stack, atr_stack, volume_zscore_stack

FEATURE_NAMES = ("adx", "atr_pct", "volume_zscore")

# Below this many valid (finite, tradeable) bars a coin's own clustering is
# not statistically meaningful - the fuzzy-regime section's own "small-cap
# coins with short histories may need to fall back... until they accumulate
# enough data" note. ~200 bars is a low bar deliberately: the monthly job
# would rather discover a coarse model early and refine it next month than
# withhold one indefinitely from every young listing.
MIN_SAMPLES_PER_COIN = 200
DEFAULT_K_RANGE: tuple[int, ...] = (4, 5, 6)


@dataclass(slots=True)
class CoinRegimeModel:
    """One coin's discovered regimes: where the clusters sit in standardized
    feature space, and the scaler needed to put a new live feature vector
    into that same space before classifying it."""

    symbol: str
    scaler: Standardizer
    centroids: np.ndarray   # [n_clusters, n_features]
    fpc: float
    n_samples: int

    @property
    def n_clusters(self) -> int:
        return int(self.centroids.shape[0])

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "feature_names": list(FEATURE_NAMES),
            "scaler": self.scaler.as_dict(),
            "centroids": self.centroids.tolist(),
            "fpc": round(self.fpc, 6),
            "n_clusters": self.n_clusters,
            "n_samples": self.n_samples,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CoinRegimeModel":
        return cls(
            symbol=data["symbol"],
            scaler=Standardizer.from_dict(data["scaler"]),
            centroids=np.asarray(data["centroids"], dtype=np.float64),
            fpc=float(data.get("fpc", 0.0)),
            n_samples=int(data.get("n_samples", 0)),
        )

    @classmethod
    def from_result(
        cls, symbol: str, scaler: Standardizer, result: FuzzyCMeansResult, n_samples: int
    ) -> "CoinRegimeModel":
        return cls(
            symbol=symbol, scaler=scaler, centroids=result.centroids,
            fpc=result.fpc, n_samples=n_samples,
        )


def compute_features(
    frames: Frames,
    *,
    adx_period: int = 14,
    atr_period: int = 14,
    vol_zscore_period: int = 20,
) -> np.ndarray:
    """``[symbols, bars, 3]`` - ADX, ATR%, volume z-score, in `FEATURE_NAMES`
    order. Reuses the same vectorized stacks the indicator-combination search
    (gap-closure item 3) already computes once per distinct period."""
    adx, _plus, _minus = adx_stack(frames.high, frames.low, frames.close, [adx_period])
    atr = atr_stack(frames.high, frames.low, frames.close, [atr_period])[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        atr_pct = atr / np.where(frames.close == 0.0, np.nan, frames.close)
    vol_z = volume_zscore_stack(frames.volume, [vol_zscore_period])[0]
    return np.stack([adx[0], atr_pct.astype(np.float32), vol_z], axis=-1)


def discover_coin_regimes(
    frames: Frames,
    symbol: str,
    *,
    k_range: Sequence[int] = DEFAULT_K_RANGE,
    min_samples: int = MIN_SAMPLES_PER_COIN,
    seed: int = 0,
    features: np.ndarray | None = None,
) -> CoinRegimeModel | None:
    """Discover one coin's own regimes from its own tradeable history.

    Returns None - never raises - when there is not enough history yet;
    the caller (a coin with too few bars) falls back to a market-wide or
    default blend until this coin accumulates enough data, per the
    fuzzy-regime section's own note.
    """
    try:
        idx = frames.symbols.index(symbol)
    except ValueError:
        return None

    feats = features if features is not None else compute_features(frames)
    coin_features = feats[idx]  # [bars, 3]
    valid = frames.mask[idx] & np.all(np.isfinite(coin_features), axis=1)
    X = coin_features[valid]
    if X.shape[0] < min_samples:
        return None

    scaler = Standardizer.fit(X)
    scaled = scaler.transform(X)
    result = discover_best_k(scaled, k_range=k_range, seed=seed)
    return CoinRegimeModel.from_result(symbol, scaler, result, n_samples=int(X.shape[0]))


def discover_all(
    frames: Frames,
    *,
    k_range: Sequence[int] = DEFAULT_K_RANGE,
    min_samples: int = MIN_SAMPLES_PER_COIN,
    seed: int = 0,
) -> dict[str, CoinRegimeModel]:
    """Every coin in `frames` with enough history for its own model - the
    feature stack is computed once for the whole panel and sliced per coin,
    not recomputed per symbol."""
    features = compute_features(frames)
    out: dict[str, CoinRegimeModel] = {}
    for symbol in frames.symbols:
        model = discover_coin_regimes(
            frames, symbol, k_range=k_range, min_samples=min_samples, seed=seed,
            features=features,
        )
        if model is not None:
            out[symbol] = model
    return out
