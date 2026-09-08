"""Parameter sensitivity: which values of each tunable axis the persistent
library's kept combinations correlate with stronger walk-forward performance.

This is not a systematic single-axis sweep within one search - the search
itself does not persist every combination it evaluates, only the ones that
cleared the walk-forward gates (``solopt.store``'s ``library`` table,
gap-closure item 4). So this reads across everything the bot has ever kept
market-wide: group every library entry by one axis's own value, and average
that axis's own robustness score (the same walk-forward-efficiency-led
number the library already ranks its seed pool by) across every kept combo
that used it. Real evidence, just a coarser lens than a dedicated sweep -
an axis the library has only ever seen at one value says nothing about its
sensitivity, and is left out rather than reported as flat.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Sequence

from .params import TUNABLE
from .store import _library_score

MIN_DISTINCT_VALUES = 2
MIN_ENTRIES = 3


def _bucket_key(value: Any) -> Any:
    """Continuous axes bucketed to 2 decimal places, so "1.231" and "1.234"
    from two runs' slightly different sampling do not each get their own
    single-sample bucket."""
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return round(value, 2)
    return value


def compute_sensitivity(
    entries: Sequence[dict[str, Any]],
    *,
    min_distinct_values: int = MIN_DISTINCT_VALUES,
    min_entries: int = MIN_ENTRIES,
) -> dict[str, Any]:
    """``entries`` are library rows - each with a ``params`` dict and a
    ``performance`` dict (e.g. from ``RunStore.top_library_entries`` or a
    direct query of every market-wide row).

    Returns ``{"axes": {axis: {"buckets": [...], "score_range": r,
    "best_value": v}}, "ranked": [axis, ...]}`` - axes with at least
    ``min_distinct_values`` distinct observed values, ranked by
    ``score_range`` descending (the axes whose value the library's own
    outcomes actually swung on, most first).
    """
    if len(entries) < min_entries:
        return {"axes": {}, "ranked": [], "entries_used": len(entries)}

    by_axis: dict[str, dict[Any, list[float]]] = defaultdict(lambda: defaultdict(list))
    for entry in entries:
        params = entry.get("params") or {}
        score = _library_score(entry.get("performance") or {})
        for axis in TUNABLE:
            if axis not in params:
                continue
            by_axis[axis][_bucket_key(params[axis])].append(score)

    axes: dict[str, Any] = {}
    for axis, buckets in by_axis.items():
        if len(buckets) < min_distinct_values:
            continue
        rows = []
        for value, scores in buckets.items():
            rows.append(
                {
                    "value": value,
                    "count": len(scores),
                    "mean_score": round(sum(scores) / len(scores), 4),
                    "min_score": round(min(scores), 4),
                    "max_score": round(max(scores), 4),
                }
            )
        rows.sort(key=lambda r: (isinstance(r["value"], str), r["value"]))
        best = max(rows, key=lambda r: r["mean_score"])
        means = [r["mean_score"] for r in rows]
        axes[axis] = {
            "buckets": rows,
            "score_range": round(max(means) - min(means), 4),
            "best_value": best["value"],
            "best_mean_score": best["mean_score"],
        }

    ranked = sorted(axes, key=lambda a: axes[a]["score_range"], reverse=True)
    return {"axes": axes, "ranked": ranked, "entries_used": len(entries)}
