"""Live fuzzy regime classification (fuzzy-regime section, step 3).

Reads the per-coin model solopt.regime_discovery discovered and persisted
(step 1), standardizes the current live feature vector the same way, and
computes membership percentages against that coin's own centroids - the
same formula solopt.fuzzy.fuzzy_cmeans converges with, applied once instead
of iterated, since the centroids are already fixed.

solbot imports solopt only lazily and only here (and in paramsync.py /
engine.py for the persistent library) - the live trading loop has no reason
to import solopt's CuPy/NumPy array machinery at module load time.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd

from .indicators import Snapshot, compute

log = logging.getLogger(__name__)


@dataclass(slots=True)
class RegimeMembership:
    """One symbol's current fuzzy membership - the dashboard (step 3's
    'expose these live percentages... for visibility/debugging') and the
    blended strategy application (step 4) both read this."""

    symbol: str
    percentages: dict[int, float]   # cluster index -> membership, sums to 1
    n_clusters: int
    feature_vector: tuple[float, float, float]  # (adx, atr_pct, volume_zscore)

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "percentages": {str(k): round(v, 4) for k, v in self.percentages.items()},
            "n_clusters": self.n_clusters,
            "feature_vector": list(self.feature_vector),
        }

    def dominant_cluster(self) -> int | None:
        if not self.percentages:
            return None
        return max(self.percentages, key=self.percentages.get)


def feature_vector_from_snapshot(snap: Snapshot) -> tuple[float, float, float]:
    """The same three features, in the same order, solopt.regime_discovery
    clusters on: ADX, ATR%, volume z-score."""
    return (float(snap.adx), float(snap.atr_pct), float(snap.volume_zscore))


def classify_current_regime(
    symbol: str, snap: Snapshot, *, store: Any = None
) -> RegimeMembership | None:
    """This symbol's current fuzzy membership, or None when it has no
    discovered model yet (too little history, or discovery has simply never
    run for it) - never raises, a missing model is an ordinary case the
    caller falls back from, not a failure.
    """
    from solopt.fuzzy import Standardizer, membership_for

    try:
        if store is None:
            from .wfmc import DAILY_STORE_PATH
            from solopt.store import RunStore

            store = RunStore(DAILY_STORE_PATH)
        stored = store.get_regime_model(symbol)
    except Exception:
        log.debug("regime model lookup failed for %s", symbol, exc_info=True)
        return None

    if stored is None:
        return None

    model = stored["model"]
    try:
        scaler = Standardizer.from_dict(model["scaler"])
        centroids = model["centroids"]
    except (KeyError, TypeError):
        log.warning("regime model for %s is malformed, ignoring it", symbol)
        return None

    features = feature_vector_from_snapshot(snap)
    if not all(f == f for f in features):  # NaN check without importing numpy/math here
        return None

    import numpy as np

    scaled = scaler.transform(np.asarray([features]))[0]
    u = membership_for(scaled, np.asarray(centroids, dtype=np.float64))

    return RegimeMembership(
        symbol=symbol,
        percentages={i: float(v) for i, v in enumerate(u)},
        n_clusters=len(centroids),
        feature_vector=features,
    )


def classify_series(
    df: pd.DataFrame, model_dict: dict[str, Any], cfg: dict[str, Any], *, precomputed: bool = False
) -> list[dict[str, Any]]:
    """Fuzzy membership at every bar of `df`, not just the last one - the
    fuzzy-regime section's step 6: overlaying a coin's regime history on its
    price chart needs a reading per historical bar, not only the live one
    classify_current_regime answers.

    Returns one entry per bar with a valid feature vector - `{"ts", "dominant",
    "percentages"}` - skipping bars still inside indicator warm-up (rather
    than a None placeholder, which would just push the "no data yet" handling
    onto every caller).
    """
    if df.empty:
        return []

    from solopt.fuzzy import Standardizer, membership_stack

    try:
        scaler = Standardizer.from_dict(model_dict["scaler"])
        centroids_list = model_dict["centroids"]
    except (KeyError, TypeError):
        log.warning("regime model is malformed, cannot classify a series")
        return []

    data = df if precomputed else compute(df, cfg)
    if data.empty:
        return []

    features = data[["adx", "atr_pct", "volume_zscore"]].to_numpy(dtype=float)
    valid = ~pd.isna(features).any(axis=1)
    if not valid.any():
        return []

    scaled = scaler.transform(features[valid])
    import numpy as np

    membership = membership_stack(scaled, np.asarray(centroids_list, dtype=np.float64))

    ts_valid = data["ts"].to_numpy()[valid]
    out: list[dict[str, Any]] = []
    for i in range(membership.shape[0]):
        percentages = {j: float(v) for j, v in enumerate(membership[i])}
        dominant = max(percentages, key=percentages.get)
        out.append({"ts": int(ts_valid[i]), "dominant": dominant, "percentages": percentages})
    return out
