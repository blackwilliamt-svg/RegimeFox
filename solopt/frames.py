"""Compacting an aligned panel into per-symbol bar sequences.

The panel from :mod:`solopt.dataset` is laid out on a shared wall-clock grid,
which is what the portfolio-level concurrency cap needs. Indicators, though, must
see exactly what the live bot sees: a frame of that symbol's own candles with the
gaps closed up, because that is what ``DataStore.candles_for`` returns. Computing
a rolling volume average over grid slots where the symbol never printed would
give the optimizer a different answer than production for the same history.

So this module compacts: every symbol's valid bars are packed to the left of a
rectangular ``[symbols, bars]`` block, with a mask marking the padding tail and
``grid_pos`` remembering where each bar sat on the shared clock. Indicators run
on the packed block (gap-free, exactly like the live frame); the trade simulator
uses ``ts`` and ``grid_pos`` to put fills back on the wall clock so two symbols
cannot both take the last position slot at the same instant.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .dataset import Panel
from .schema import Schema


@dataclass(slots=True)
class Frames:
    """Left-packed per-symbol candles plus the masks the engine reads."""

    symbols: list[str]
    ts: np.ndarray            # int64[S, R]  bar open time, 0 in the padding tail
    open: np.ndarray          # float32[S, R]
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    mask: np.ndarray          # bool[S, R]   True where a real bar sits
    eligible: np.ndarray      # bool[S, R]   passed the universe floors that day
    counts: np.ndarray        # int32[S]     bars per symbol
    grid_pos: np.ndarray      # int32[S, R]  index back into the shared grid
    seconds: int
    schema: Schema
    liquidity: np.ndarray | None = None

    @property
    def n_symbols(self) -> int:
        return len(self.symbols)

    @property
    def n_bars(self) -> int:
        return int(self.close.shape[1])

    @property
    def tradeable(self) -> np.ndarray:
        return self.mask & self.eligible

    def warmup_split(self, warmup_bars: int) -> np.ndarray:
        """Bars that are past the indicator warm-up, per symbol.

        Every symbol is packed to the left, so the same column index means
        "the n-th bar this symbol printed" for all of them - which is what the
        warm-up rule is actually about.
        """
        idx = np.arange(self.n_bars, dtype=np.int32)[None, :]
        return idx >= int(warmup_bars)

    def slice_bars(self, lo: int, hi: int) -> "Frames":
        """Restrict to a column range. Counts are re-derived from the mask."""
        lo = max(0, int(lo))
        hi = min(self.n_bars, int(hi))
        mask = self.mask[:, lo:hi]
        return Frames(
            symbols=list(self.symbols),
            ts=self.ts[:, lo:hi],
            open=self.open[:, lo:hi],
            high=self.high[:, lo:hi],
            low=self.low[:, lo:hi],
            close=self.close[:, lo:hi],
            volume=self.volume[:, lo:hi],
            mask=mask,
            eligible=self.eligible[:, lo:hi],
            counts=mask.sum(axis=1).astype(np.int32),
            grid_pos=self.grid_pos[:, lo:hi],
            seconds=self.seconds,
            schema=self.schema,
            liquidity=None if self.liquidity is None else self.liquidity[:, lo:hi],
        )

    def window(self, start_ts: int, end_ts: int, *, warmup_bars: int = 0) -> "Frames":
        """The walk-forward window primitive: bars in ``[start_ts, end_ts)``.

        ``warmup_bars`` extends the slice *backwards* so recursive indicators
        enter the window already seeded. Without it the first bars of every
        window would carry warm-up artefacts and each window would be scored on
        a slightly different strategy than the one that actually trades.
        """
        # Symbols are packed independently, so the column that holds a given
        # timestamp differs per symbol; take the widest span that covers them all.
        inside = self.mask & (self.ts >= int(start_ts)) & (self.ts < int(end_ts))
        if not inside.any():
            return self.slice_bars(0, 0)
        cols = np.flatnonzero(inside.any(axis=0))
        lo = max(0, int(cols[0]) - max(0, int(warmup_bars)))
        hi = int(cols[-1]) + 1
        out = self.slice_bars(lo, hi)
        # Bars pulled in only as warm-up must never produce a trade in this
        # window; the engine reads `in_window` for that.
        out.eligible = out.eligible & (out.ts >= int(start_ts)) & (out.ts < int(end_ts))
        return out

    def select_symbols(self, symbols: Sequence[str]) -> "Frames":
        """Restrict to a subset of symbols, in the given order - the
        fuzzy-regime section's per-coin walk-forward (step 2) runs the search
        against one coin alone, not the whole panel, so its promoted set is
        genuinely that coin's own rather than a market-wide consensus. Each
        row is already left-packed per symbol, so selecting rows disturbs
        nothing about that; only the symbol axis shrinks.
        """
        idx = [self.symbols.index(s) for s in symbols]
        return Frames(
            symbols=list(symbols),
            ts=self.ts[idx],
            open=self.open[idx],
            high=self.high[idx],
            low=self.low[idx],
            close=self.close[idx],
            volume=self.volume[idx],
            mask=self.mask[idx],
            eligible=self.eligible[idx],
            counts=self.counts[idx],
            grid_pos=self.grid_pos[idx],
            seconds=self.seconds,
            schema=self.schema,
            liquidity=None if self.liquidity is None else self.liquidity[idx],
        )

    def coverage(self) -> dict[str, Any]:
        real = self.mask.sum()
        ts = self.ts[self.mask]
        return {
            "symbols": self.n_symbols,
            "columns": self.n_bars,
            "bars": int(real),
            "first_ts": int(ts.min()) if real else 0,
            "last_ts": int(ts.max()) if real else 0,
            "median_bars_per_symbol": int(np.median(self.counts)) if self.n_symbols else 0,
        }


def compact(panel: Panel) -> Frames:
    """Pack every symbol's valid bars to the left of a rectangular block."""
    valid = panel.valid
    n_symbols, n_grid = valid.shape
    counts = valid.sum(axis=1).astype(np.int32)
    width = int(counts.max()) if n_symbols and counts.size else 0

    # A stable argsort of ``~valid`` lists each row's valid columns first, in
    # order, then the invalid ones - i.e. the packing permutation, for free.
    order = np.argsort(~valid, axis=1, kind="stable")[:, :width]
    rows = np.arange(n_symbols)[:, None]
    mask = np.arange(width, dtype=np.int32)[None, :] < counts[:, None]

    def take(field: np.ndarray) -> np.ndarray:
        return np.where(mask, field[rows, order], np.float32(0.0)).astype(np.float32)

    return Frames(
        symbols=list(panel.symbols),
        ts=np.where(mask, panel.ts[order], 0).astype(np.int64),
        open=take(panel.open),
        high=take(panel.high),
        low=take(panel.low),
        close=take(panel.close),
        volume=take(panel.volume),
        mask=mask,
        eligible=panel.eligible[rows, order] & mask,
        counts=counts,
        grid_pos=order.astype(np.int32),
        seconds=panel.seconds,
        schema=panel.schema,
        liquidity=(
            np.where(mask, panel.liquidity[rows, order], np.float32(0.0)).astype(np.float32)
            if panel.liquidity is not None
            else None
        ),
    )


def frames_from_series(
    series: dict[str, np.ndarray],
    *,
    seconds: int,
    schema: Schema,
    eligible: dict[str, np.ndarray] | None = None,
) -> Frames:
    """Build frames straight from per-symbol ``(6, n)`` arrays.

    Used by the tests and by callers that already have gap-free series and do
    not need the shared-grid alignment step.
    """
    names = sorted(series)
    counts = np.array([series[s].shape[1] for s in names], dtype=np.int32)
    width = int(counts.max()) if counts.size else 0
    shape = (len(names), width)

    ts = np.zeros(shape, dtype=np.int64)
    fields = {f: np.zeros(shape, dtype=np.float32) for f in
              ("open", "high", "low", "close", "volume")}
    mask = np.zeros(shape, dtype=bool)
    elig = np.zeros(shape, dtype=bool)

    for i, name in enumerate(names):
        rows = series[name]
        n = rows.shape[1]
        ts[i, :n] = rows[0].astype(np.int64)
        for j, f in enumerate(("open", "high", "low", "close", "volume"), start=1):
            fields[f][i, :n] = rows[j]
        mask[i, :n] = True
        if eligible is not None and name in eligible:
            elig[i, :n] = eligible[name][:n]
        else:
            elig[i, :n] = True

    return Frames(
        symbols=names,
        ts=ts,
        mask=mask,
        eligible=elig,
        counts=counts,
        grid_pos=np.tile(np.arange(width, dtype=np.int32), (len(names), 1)),
        seconds=seconds,
        schema=schema,
        **fields,
    )
