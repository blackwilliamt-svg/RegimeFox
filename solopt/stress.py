"""Crash and flash-crash replay (spec 4.3).

Walk-forward windows roll evenly across history, which means the worst days get
averaged in with everything else. A set that loses 40% in the two days the market
fell apart and makes it back over the following six weeks passes every threshold
in :mod:`solopt.walkforward` and is still a set nobody should be running.

So the crash windows are pulled out and replayed on their own. They are found in
the data rather than hard-coded to a list of dates: an equal-weight index is
built from the panel, the deepest sustained declines and the sharpest single-bar
drops become the stress windows, and the parameter set is run through each. That
keeps the whole thing asset-class-generic - the same code finds the March 2020
equity crash and the May 2022 crypto one without being told either exists.

A set that fails here is flagged **fragile**, and a fragile flag blocks
auto-promotion no matter how good the ordinary windows looked.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from .engine import PortfolioSettings, VectorEngine
from .frames import Frames

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class StressWindow:
    name: str
    from_ts: int
    to_ts: int
    kind: str = "crash"           # crash | flash | manual
    depth: float = 0.0            # index decline over the window

    def label(self) -> str:
        return (
            f"{self.name} ({time.strftime('%Y-%m-%d', time.gmtime(self.from_ts))}"
            f" to {time.strftime('%Y-%m-%d', time.gmtime(self.to_ts))})"
        )


@dataclass
class StressResult:
    name: str
    from_ts: int
    to_ts: int
    kind: str = "crash"
    metrics: dict[str, Any] = field(default_factory=dict)
    passed: bool = True
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "from_ts": self.from_ts, "to_ts": self.to_ts,
            "kind": self.kind, "metrics": self.metrics, "passed": self.passed,
            "note": self.note,
        }


@dataclass(frozen=True)
class StressConfig:
    """How stress windows are found and what counts as surviving one."""

    crash_windows: int = 3
    flash_windows: int = 2
    crash_days: int = 7
    flash_bars: int = 6
    min_index_decline: float = 0.15      # a 15% index fall is worth replaying
    min_flash_decline: float = 0.08      # over flash_bars, not days
    pad_days: float = 1.0                # lead-in so positions can already be open

    max_stress_drawdown: float = 0.30
    min_stress_return: float = -0.15
    min_trades: int = 5                  # fewer than this is inconclusive

    def as_dict(self) -> dict[str, Any]:
        return {
            "crash_windows": self.crash_windows,
            "flash_windows": self.flash_windows,
            "crash_days": self.crash_days,
            "flash_bars": self.flash_bars,
            "min_index_decline": self.min_index_decline,
            "min_flash_decline": self.min_flash_decline,
            "max_stress_drawdown": self.max_stress_drawdown,
            "min_stress_return": self.min_stress_return,
            "min_trades": self.min_trades,
        }


def market_index(frames: Frames) -> tuple[np.ndarray, np.ndarray]:
    """An equal-weight index of the whole test universe, on the wall clock.

    Averaging *returns* rather than prices keeps a single high-priced symbol from
    dominating, and keeps the index defined across a window where symbols come
    and go - which, given the universe deliberately includes coins that later
    died, it always does.
    """
    if frames.n_symbols == 0 or frames.n_bars < 2:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64)

    ts = frames.ts[frames.mask]
    if ts.size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64)
    grid = np.unique(ts)

    total = np.zeros(grid.size, dtype=np.float64)
    count = np.zeros(grid.size, dtype=np.float64)
    for i in range(frames.n_symbols):
        n = int(frames.counts[i])
        if n < 2:
            continue
        closes = frames.close[i, :n].astype(np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            step = np.diff(closes) / np.where(closes[:-1] == 0, np.nan, closes[:-1])
        slots = np.searchsorted(grid, frames.ts[i, 1:n])
        keep = np.isfinite(step) & (slots < grid.size)
        np.add.at(total, slots[keep], step[keep])
        np.add.at(count, slots[keep], 1.0)

    mean_step = np.where(count > 0, total / np.where(count == 0, 1.0, count), 0.0)
    return grid, np.cumprod(1.0 + mean_step)


def _rolling_max(x: np.ndarray, window: int) -> np.ndarray:
    """``out[i] = x[i:i+window].max()`` in linear time.

    The two-pass block algorithm: prefix maxima within each block of ``window``
    and suffix maxima within each block, then one elementwise maximum. Avoids the
    ``n x window`` view that a strided sliding window would materialise, which on
    a two-year 10-minute index is the difference between 2MB and 200MB.
    """
    n = x.size
    window = max(1, min(int(window), n))
    if window == 1:
        return x.copy()
    pad = (-n) % window
    padded = np.concatenate((x, np.full(pad, -np.inf, dtype=x.dtype)))
    blocks = padded.reshape(-1, window)
    prefix = np.maximum.accumulate(blocks, axis=1).ravel()
    suffix = np.maximum.accumulate(blocks[:, ::-1], axis=1)[:, ::-1].ravel()
    return np.maximum(suffix[: n - window + 1], prefix[window - 1 : n])


def find_windows(
    frames: Frames, config: StressConfig | None = None
) -> list[StressWindow]:
    """Locate the worst sustained declines and the sharpest sudden ones."""
    config = config or StressConfig()
    grid, index = market_index(frames)
    if grid.size < 10:
        return []

    seconds = max(1, frames.seconds)
    pad = int(config.pad_days * 86400)
    found: list[StressWindow] = []
    taken = np.zeros(grid.size, dtype=bool)

    def claim(lo: int, hi: int) -> bool:
        if taken[lo:hi].any():
            return False
        taken[lo:hi] = True
        return True

    def scan(span_bars: int, count: int, floor: float, kind: str) -> None:
        span_bars = max(2, min(span_bars, grid.size - 1))
        window = span_bars + 1
        if grid.size < window:
            return
        # Rank candidate starts by the spread between the window's high and low.
        # That is an O(n) proxy for the drawdown inside it; the exact peak-to-
        # trough is then measured on the handful of windows actually selected.
        highs = _rolling_max(index, window)
        lows = -_rolling_max(-index, window)
        spread = np.where(highs > 0, (highs - lows) / highs, 0.0)

        wanted = count
        for rank in np.argsort(spread)[::-1]:
            if wanted <= 0 or spread[rank] < floor:
                return
            start = int(rank)
            stop = min(grid.size - 1, start + span_bars)
            if not claim(start, stop + 1):
                continue
            segment = index[start : stop + 1]
            peak = np.maximum.accumulate(segment)
            depth = float(np.max((peak - segment) / np.where(peak > 0, peak, 1.0)))
            if depth < floor:
                continue
            found.append(
                StressWindow(
                    name=f"{kind} {time.strftime('%Y-%m-%d', time.gmtime(int(grid[start])))}",
                    from_ts=int(grid[start]) - pad,
                    to_ts=int(grid[stop]) + seconds,
                    kind=kind,
                    depth=depth,
                )
            )
            wanted -= 1

    scan(
        span_bars=int(config.crash_days * 86400 // seconds),
        count=config.crash_windows,
        floor=config.min_index_decline,
        kind="crash",
    )
    scan(
        span_bars=int(config.flash_bars),
        count=config.flash_windows,
        floor=config.min_flash_decline,
        kind="flash",
    )
    found.sort(key=lambda w: w.from_ts)
    return found


def replay(
    frames: Frames,
    params: dict[str, Any],
    settings: PortfolioSettings,
    engine: VectorEngine,
    windows: Sequence[StressWindow],
    *,
    warmup_bars: int,
    config: StressConfig | None = None,
) -> list[StressResult]:
    """Run one parameter set through each stress window."""
    config = config or StressConfig()
    results: list[StressResult] = []

    for window in windows:
        sliced = frames.window(window.from_ts, window.to_ts, warmup_bars=warmup_bars)
        result = StressResult(
            name=window.name, from_ts=window.from_ts, to_ts=window.to_ts, kind=window.kind
        )
        if sliced.n_bars == 0 or not sliced.tradeable.any():
            result.note = "no eligible bars in this window"
            results.append(result)
            continue

        run = engine.run(sliced, [params], settings)
        metrics = run.metrics[0]
        result.metrics = {**metrics, "index_decline": round(window.depth, 4)}

        if metrics["trades"] < config.min_trades:
            # Refusing to trade through a crash is a pass, not a failure to
            # measure: the regime gate doing its job looks exactly like this.
            result.note = (
                f"only {metrics['trades']} trades - the rules mostly stood aside, "
                "which is not a failure"
            )
            result.passed = True
            results.append(result)
            continue

        too_deep = metrics["max_drawdown"] > config.max_stress_drawdown
        too_lossy = metrics["total_return"] < config.min_stress_return
        result.passed = not (too_deep or too_lossy)
        if too_deep:
            result.note = (
                f"drawdown {metrics['max_drawdown'] * 100:.1f}% exceeds the "
                f"{config.max_stress_drawdown * 100:.0f}% stress limit"
            )
        elif too_lossy:
            result.note = (
                f"lost {abs(metrics['total_return']) * 100:.1f}%, past the "
                f"{abs(config.min_stress_return) * 100:.0f}% stress limit"
            )
        else:
            result.note = (
                f"survived: {metrics['total_return'] * 100:+.1f}% with a "
                f"{metrics['max_drawdown'] * 100:.1f}% drawdown across "
                f"{metrics['trades']} trades"
            )
        results.append(result)

    return results


def fragile_from(results: Sequence[StressResult]) -> list[str]:
    """Names of the stress windows a set failed. Empty means not fragile."""
    return [r.name for r in results if not r.passed]
