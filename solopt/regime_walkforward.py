"""Per-coin, per-regime walk-forward optimization (fuzzy-regime section,
step 2).

For each coin, for each of its own discovered fuzzy regimes (step 1), the
search runs against that coin alone and only over the historical windows
where that regime actually had high membership for it - a distinct promoted
parameter set (and, once the indicator-combination search's shape choice is
in play, potentially a distinct strategy shape) per coin per regime. Results
land in the same persistent library gap-closure item 4 already writes to,
tagged with `symbol` and `regime_cluster_id` (see solopt/store.py's
LibraryEntry) rather than a new table - the sharing gap-closure item 5
already established between these features continues here.
"""
from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np

from .arrays import Backend, get_backend
from .engine import PortfolioSettings
from .frames import Frames
from .fuzzy import membership_stack
from .params import ParamSpace
from .promotion import ParameterBundle
from .regime_discovery import CoinRegimeModel, compute_features
from .store import LibraryEntry
from .walkforward import (
    WalkForward,
    WalkForwardConfig,
    WalkForwardResult,
    Window,
    plan_windows,
)

log = logging.getLogger(__name__)

# A window's out-of-sample span must average at least this much membership in
# the target cluster to count as "belonging" to that regime - the fuzzy-
# regime section's own "high membership" phrasing, made concrete. Not 0.5
# (a bare plurality among 4-6 clusters is not "high"); comfortably above
# an even split.
DEFAULT_MEMBERSHIP_THRESHOLD = 0.5
MIN_WINDOWS_FOR_A_REGIME = 2


def regime_membership_series(
    frames: Frames, symbol: str, model: CoinRegimeModel
) -> np.ndarray:
    """``[bars]`` this coin's own membership in each of its clusters over its
    whole history - NaN wherever the bar isn't valid (padding, or a feature
    that couldn't be computed). Shape ``[bars, n_clusters]``.
    """
    idx = frames.symbols.index(symbol)
    features = compute_features(frames)[idx]  # [bars, 3]
    valid = frames.mask[idx] & np.all(np.isfinite(features), axis=1)

    out = np.full((features.shape[0], model.n_clusters), np.nan, dtype=np.float64)
    if valid.any():
        scaled = model.scaler.transform(features[valid])
        out[valid] = membership_stack(scaled, model.centroids)
    return out


def windows_for_regime(
    frames: Frames,
    symbol: str,
    model: CoinRegimeModel,
    cluster_id: int,
    all_windows: Sequence[Window],
    *,
    threshold: float = DEFAULT_MEMBERSHIP_THRESHOLD,
) -> list[Window]:
    """Which of `all_windows` this coin's regime `cluster_id` actually
    dominated during the out-of-sample span - the "only historical windows
    where that regime had high membership" restriction."""
    idx = frames.symbols.index(symbol)
    membership = regime_membership_series(frames, symbol, model)[:, cluster_id]
    ts = frames.ts[idx]

    kept = []
    for w in all_windows:
        in_oos = (ts >= w.oos_from) & (ts < w.oos_to)
        values = membership[in_oos]
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        if float(np.mean(values)) >= threshold:
            kept.append(w)
    return kept


def run_regime_scoped_search(
    frames: Frames,
    symbol: str,
    model: CoinRegimeModel,
    cluster_id: int,
    *,
    space: ParamSpace,
    portfolio: PortfolioSettings,
    wf_config: WalkForwardConfig,
    backend: Backend | None = None,
    threshold: float = DEFAULT_MEMBERSHIP_THRESHOLD,
    min_windows: int = MIN_WINDOWS_FOR_A_REGIME,
) -> WalkForwardResult | None:
    """Walk-forward for one coin, restricted to its own bars and to the
    windows its regime `cluster_id` actually dominated.

    Returns None (never raises) when there simply isn't enough regime-
    specific history to search - too few qualifying windows is exactly the
    "small-cap coins... may need to fall back... until they accumulate
    enough data" case the fuzzy-regime section calls out, just measured per
    regime instead of per coin.
    """
    idx = frames.symbols.index(symbol)
    coin_ts = frames.ts[idx][frames.mask[idx]]
    if coin_ts.size == 0:
        return None
    all_windows = plan_windows(int(coin_ts.min()), int(coin_ts.max()), wf_config)
    if not all_windows:
        return None

    qualifying = windows_for_regime(
        frames, symbol, model, cluster_id, all_windows, threshold=threshold
    )
    if len(qualifying) < min_windows:
        return None

    coin_frames = frames.select_symbols([symbol])
    walker = WalkForward(space, portfolio, wf_config, backend=backend or get_backend())
    warmup = wf_config.warmup_bars or 0

    results = []
    for window in qualifying:
        results.append(walker._one(window, coin_frames, warmup))
    outcome = walker.aggregate(results)
    return outcome


def update_library_for_regime(
    store: Any,
    symbol: str,
    cluster_id: int,
    model: CoinRegimeModel,
    outcome: WalkForwardResult,
    *,
    run_id: int | None = None,
) -> bool:
    """Persist a regime-scoped search's result, if it passed the walk-forward
    thresholds - the exact same acceptance bar run_pipeline._update_library
    already applies for the market-wide/per-symbol case, just keyed by
    cluster instead of (or alongside) a continuous regime score."""
    if not outcome.accepted or not outcome.best_params:
        return False

    fingerprint = ParameterBundle(global_params=dict(outcome.best_params)).fingerprint()
    entry = LibraryEntry(
        fingerprint=fingerprint,
        params=dict(outcome.best_params),
        performance=outcome.summary(),
        symbol=symbol,
        regime_cluster_id=int(cluster_id),
        run_id=run_id,
    )
    return bool(store.upsert_library(entry))


def run_and_store_all_regimes(
    frames: Frames,
    symbol: str,
    model: CoinRegimeModel,
    *,
    space: ParamSpace,
    portfolio: PortfolioSettings,
    wf_config: WalkForwardConfig,
    store: Any,
    backend: Backend | None = None,
    threshold: float = DEFAULT_MEMBERSHIP_THRESHOLD,
    min_windows: int = MIN_WINDOWS_FOR_A_REGIME,
    run_id: int | None = None,
    on_progress: Any = None,
) -> dict[int, WalkForwardResult]:
    """Search and (if accepted) store every one of this coin's discovered
    regimes in turn - the per-coin loop the monthly job / manual trigger
    (step 5) drives. `on_progress(cluster_id, n_clusters)` fires before each
    regime starts, for the dashboard's plain-language feed.
    """
    outcomes: dict[int, WalkForwardResult] = {}
    for cluster_id in range(model.n_clusters):
        if on_progress is not None:
            try:
                on_progress(cluster_id, model.n_clusters)
            except Exception:
                log.debug("regime progress callback failed", exc_info=True)

        outcome = run_regime_scoped_search(
            frames, symbol, model, cluster_id,
            space=space, portfolio=portfolio, wf_config=wf_config,
            backend=backend, threshold=threshold, min_windows=min_windows,
        )
        if outcome is None:
            continue
        outcomes[cluster_id] = outcome
        update_library_for_regime(store, symbol, cluster_id, model, outcome, run_id=run_id)

    return outcomes
