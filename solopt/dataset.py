"""Loading bulk history into the aligned panel the engine works on.

The engine wants one rectangular block of memory per OHLCV field, shaped
``[symbols, bars]`` on a shared time grid, so every operation can be a single
array op across the whole universe. Getting there is this module's whole job:
read whatever the droplet sent, roll the 1-minute base up to the timeframe under
test, and lay the result onto a common grid with two masks alongside it.

The two masks are what keep the results honest:

* ``valid`` marks bars where the symbol actually printed. Gaps are left as gaps.
  Forward-filling a missing bar invents a flat candle, and a flat candle reads as
  "volume dried up, momentum zero" - a phantom signal the live bot could never
  have seen.
* ``eligible`` marks bars where the symbol passed the universe floors *on that
  historical day*, rebuilt from snapshots. A symbol that rugged in month two is
  eligible in month one and not afterwards, which is precisely the population
  the live bot was choosing from at the time. Building the basket from what is
  tradeable today would quietly delete every failure from the record.
"""
from __future__ import annotations

import csv
import gzip
import io
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .schema import CANONICAL_COLUMNS, Schema, get_schema

log = logging.getLogger(__name__)

SNAPSHOT_FILE = "universe_snapshots.csv"
MANIFEST_FILE = "manifest.json"


@dataclass(slots=True)
class Panel:
    """Aligned OHLCV for the whole test universe at one timeframe."""

    symbols: list[str]
    ts: np.ndarray                 # int64[T], bar open times on a uniform grid
    open: np.ndarray               # float32[S, T]
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    valid: np.ndarray              # bool[S, T]
    eligible: np.ndarray           # bool[S, T]
    seconds: int
    schema: Schema = field(default_factory=get_schema)
    liquidity: np.ndarray | None = None   # float32[S, T], quote currency

    # ------------------------------------------------------------------
    @property
    def n_symbols(self) -> int:
        return len(self.symbols)

    @property
    def n_bars(self) -> int:
        return int(self.ts.shape[0])

    @property
    def tradeable(self) -> np.ndarray:
        """Bars the live bot could actually have acted on."""
        return self.valid & self.eligible

    def span(self) -> tuple[int, int]:
        if self.n_bars == 0:
            return 0, 0
        return int(self.ts[0]), int(self.ts[-1])

    def bar_index(self, ts: int) -> int:
        """Index of the first bar at or after ``ts``."""
        return int(np.searchsorted(self.ts, int(ts), side="left"))

    def slice(self, start_ts: int, end_ts: int) -> "Panel":
        """A view over ``[start_ts, end_ts)`` - the walk-forward window primitive."""
        lo = self.bar_index(start_ts)
        hi = self.bar_index(end_ts)
        return self.slice_index(lo, hi)

    def slice_index(self, lo: int, hi: int) -> "Panel":
        lo = max(0, int(lo))
        hi = min(self.n_bars, int(hi))
        return Panel(
            symbols=list(self.symbols),
            ts=self.ts[lo:hi],
            open=self.open[:, lo:hi],
            high=self.high[:, lo:hi],
            low=self.low[:, lo:hi],
            close=self.close[:, lo:hi],
            volume=self.volume[:, lo:hi],
            valid=self.valid[:, lo:hi],
            eligible=self.eligible[:, lo:hi],
            seconds=self.seconds,
            schema=self.schema,
            liquidity=None if self.liquidity is None else self.liquidity[:, lo:hi],
        )

    def drop_empty_symbols(self, min_bars: int = 1) -> "Panel":
        """Remove symbols with too little data in this window to evaluate."""
        keep = np.flatnonzero(self.valid.sum(axis=1) >= max(1, int(min_bars)))
        if keep.size == self.n_symbols:
            return self
        return Panel(
            symbols=[self.symbols[i] for i in keep],
            ts=self.ts,
            open=self.open[keep],
            high=self.high[keep],
            low=self.low[keep],
            close=self.close[keep],
            volume=self.volume[keep],
            valid=self.valid[keep],
            eligible=self.eligible[keep],
            seconds=self.seconds,
            schema=self.schema,
            liquidity=None if self.liquidity is None else self.liquidity[keep],
        )

    def coverage(self) -> dict[str, Any]:
        first, last = self.span()
        total = self.valid.size or 1
        return {
            "symbols": self.n_symbols,
            "bars": self.n_bars,
            "timeframe_seconds": self.seconds,
            "first_ts": first,
            "last_ts": last,
            "days": round((last - first) / 86400.0, 1) if last else 0.0,
            "density": round(float(self.valid.sum()) / total, 4),
            "eligible_density": round(float(self.tradeable.sum()) / total, 4),
        }


# --------------------------------------------------------------------------
# Reading one symbol's file
# --------------------------------------------------------------------------
def _read_rows(path: Path, schema: Schema) -> np.ndarray:
    """Return a structured ``(ts, o, h, l, c, v)`` array from one file.

    Parquet is preferred for size; gzipped CSV is the fallback so a bundle stays
    readable without pyarrow installed.
    """
    if path.suffix == ".parquet":
        try:
            import pyarrow.parquet as pq  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on the host
            raise RuntimeError(
                f"{path.name} is Parquet but pyarrow is not installed; "
                "install pyarrow or request the CSV bundle"
            ) from exc
        table = pq.read_table(path)
        rename = schema.rename_map()
        data = {}
        for name in table.column_names:
            data[rename.get(name, name)] = np.asarray(table.column(name))
        missing = [c for c in CANONICAL_COLUMNS if c not in data]
        if missing:
            raise ValueError(f"{path.name} is missing column(s): {', '.join(missing)}")
        return _stack(
            data["ts"], data["open"], data["high"], data["low"], data["close"],
            data["volume"],
        )

    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="") as fh:  # type: ignore[operator]
        reader = csv.DictReader(fh)
        rename = schema.rename_map()
        cols: dict[str, list[float]] = {c: [] for c in CANONICAL_COLUMNS}
        for row in reader:
            mapped = {rename.get(k, k): v for k, v in row.items()}
            try:
                for c in CANONICAL_COLUMNS:
                    cols[c].append(float(mapped[c]))
            except (KeyError, TypeError, ValueError):
                continue  # a malformed row is dropped, never guessed at
    return _stack(*(np.asarray(cols[c], dtype=np.float64) for c in CANONICAL_COLUMNS))


def _stack(ts: Any, o: Any, h: Any, l: Any, c: Any, v: Any) -> np.ndarray:
    out = np.empty((6, len(ts)), dtype=np.float64)
    for i, col in enumerate((ts, o, h, l, c, v)):
        out[i] = np.asarray(col, dtype=np.float64)
    # Sort and de-duplicate: bundles are concatenations of incremental pulls, so
    # the same bar can legitimately appear twice with identical values.
    order = np.argsort(out[0], kind="stable")
    out = out[:, order]
    if out.shape[1]:
        keep = np.concatenate(([True], np.diff(out[0]) != 0))
        out = out[:, keep]
    return out


def rollup(rows: np.ndarray, target_seconds: int) -> np.ndarray:
    """Aggregate base-interval rows up to ``target_seconds``.

    Open is the first print in the bucket, high/low the extremes, close the last,
    volume the sum - the standard roll-up, done here rather than pulled from the
    source so every timeframe under test is provably the same data.
    """
    if rows.shape[1] == 0:
        return rows
    buckets = (rows[0] // target_seconds).astype(np.int64) * target_seconds
    edges = np.flatnonzero(np.concatenate(([True], np.diff(buckets) != 0)))
    starts = edges
    stops = np.concatenate((edges[1:], [rows.shape[1]]))

    out = np.empty((6, starts.size), dtype=np.float64)
    out[0] = buckets[starts]
    out[1] = rows[1][starts]
    out[4] = rows[4][stops - 1]
    for i, (a, b) in enumerate(zip(starts, stops)):
        out[2, i] = rows[2, a:b].max()
        out[3, i] = rows[3, a:b].min()
        out[5, i] = rows[5, a:b].sum()
    return out


# --------------------------------------------------------------------------
# Universe snapshots -> the eligibility mask
# --------------------------------------------------------------------------
@dataclass(slots=True)
class Snapshots:
    """What passed the filters on each historical day."""

    by_day: dict[str, dict[str, float]] = field(default_factory=dict)

    @property
    def days(self) -> list[str]:
        return sorted(self.by_day)

    def symbols(self) -> set[str]:
        out: set[str] = set()
        for row in self.by_day.values():
            out.update(row)
        return out


def load_snapshots(path: Path, schema: Schema) -> Snapshots:
    """Read ``day,symbol,liquidity,volume_24h`` and apply the floors."""
    snaps = Snapshots()
    if not path.exists():
        return snaps
    with path.open("rt", newline="") as fh:
        for row in csv.DictReader(fh):
            day = (row.get("day") or "").strip()
            symbol = (row.get("symbol") or row.get("mint") or "").strip()
            if not day or not symbol:
                continue
            try:
                liquidity = float(row.get("liquidity") or row.get("liquidity_usd") or 0.0)
                volume = float(row.get("volume_24h") or row.get("volume_24h_usd") or 0.0)
            except (TypeError, ValueError):
                continue
            if liquidity < schema.min_liquidity or volume < schema.min_volume_24h:
                continue
            snaps.by_day.setdefault(day, {})[symbol] = liquidity
    return snaps


def _day_index(ts: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Map each bar to an index into a sorted list of ``YYYY-MM-DD`` days."""
    if ts.size == 0:
        return np.zeros(0, dtype=np.int32), []
    day_start = (ts // 86400).astype(np.int64)
    uniq, inverse = np.unique(day_start, return_inverse=True)
    labels = [time.strftime("%Y-%m-%d", time.gmtime(int(d) * 86400)) for d in uniq]
    return inverse.astype(np.int32), labels


# --------------------------------------------------------------------------
# Bundle loading
# --------------------------------------------------------------------------
def discover(directory: Path, timeframe_seconds: int | None = None) -> dict[str, Path]:
    """``{symbol: path}`` for every data file in a bundle directory.

    Files are named ``<symbol>__<interval>.<ext>``; the interval suffix lets a
    bundle carry several timeframes side by side without a manifest.
    """
    found: dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if path.name in (SNAPSHOT_FILE, MANIFEST_FILE) or path.is_dir():
            continue
        if path.suffix not in (".parquet", ".gz", ".csv"):
            continue
        stem = path.name.split(".")[0]
        symbol, _, interval = stem.partition("__")
        if timeframe_seconds and interval and interval != _interval_label(timeframe_seconds):
            continue
        found.setdefault(symbol, path)
    return found


def _interval_label(seconds: int) -> str:
    if seconds % 86400 == 0:
        return f"{seconds // 86400}d"
    if seconds % 3600 == 0:
        return f"{seconds // 3600}h"
    return f"{seconds // 60}m"


def load_panel(
    directory: str | Path,
    *,
    timeframe_seconds: int,
    schema: Schema | None = None,
    symbols: Sequence[str] | None = None,
    since: int | None = None,
    until: int | None = None,
    max_symbols: int | None = None,
) -> Panel:
    """Read a bundle directory into an aligned :class:`Panel`."""
    directory = Path(directory)
    schema = schema or get_schema()
    if not directory.exists():
        raise FileNotFoundError(f"data bundle {directory} does not exist")

    files = discover(directory, timeframe_seconds=None)
    if symbols is not None:
        wanted = set(symbols)
        files = {s: p for s, p in files.items() if s in wanted}
    if not files:
        raise ValueError(f"no data files found in {directory}")

    snaps = load_snapshots(directory / SNAPSHOT_FILE, schema)
    if snaps.by_day:
        # Restrict to symbols that were eligible at some point. A file for a
        # symbol that never cleared the floors is data we should not be trading.
        eligible_symbols = snaps.symbols()
        files = {s: p for s, p in files.items() if s in eligible_symbols} or files

    ordered = sorted(files)
    if max_symbols:
        ordered = ordered[: int(max_symbols)]

    # Pass one: read and roll up, tracking the global time span as we go.
    series: dict[str, np.ndarray] = {}
    lo_ts, hi_ts = None, None
    for symbol in ordered:
        try:
            rows = _read_rows(files[symbol], schema)
        except Exception as exc:
            log.warning("skipping %s: %s", symbol, exc)
            continue
        if rows.shape[1] == 0:
            continue
        rows = rollup(rows, timeframe_seconds)
        if since is not None:
            rows = rows[:, rows[0] >= since]
        if until is not None:
            rows = rows[:, rows[0] < until]
        if rows.shape[1] == 0:
            continue
        series[symbol] = rows
        first, last = int(rows[0, 0]), int(rows[0, -1])
        lo_ts = first if lo_ts is None else min(lo_ts, first)
        hi_ts = last if hi_ts is None else max(hi_ts, last)

    if not series or lo_ts is None or hi_ts is None:
        raise ValueError(f"no usable candles in {directory} for the requested window")

    grid = np.arange(lo_ts, hi_ts + timeframe_seconds, timeframe_seconds, dtype=np.int64)
    n_symbols, n_bars = len(series), grid.size
    shape = (n_symbols, n_bars)

    fields = {
        name: np.zeros(shape, dtype=np.float32)
        for name in ("open", "high", "low", "close", "volume")
    }
    valid = np.zeros(shape, dtype=bool)

    names = sorted(series)
    for i, symbol in enumerate(names):
        rows = series[symbol]
        idx = ((rows[0] - lo_ts) // timeframe_seconds).astype(np.int64)
        inside = (idx >= 0) & (idx < n_bars)
        idx = idx[inside]
        fields["open"][i, idx] = rows[1][inside]
        fields["high"][i, idx] = rows[2][inside]
        fields["low"][i, idx] = rows[3][inside]
        fields["close"][i, idx] = rows[4][inside]
        fields["volume"][i, idx] = rows[5][inside]
        valid[i, idx] = True

    eligible, liquidity = _eligibility(names, grid, snaps)

    panel = Panel(
        symbols=names,
        ts=grid,
        open=fields["open"],
        high=fields["high"],
        low=fields["low"],
        close=fields["close"],
        volume=fields["volume"],
        valid=valid & (fields["close"] > 0),
        eligible=eligible,
        seconds=timeframe_seconds,
        schema=schema,
        liquidity=liquidity,
    )
    log.info("loaded panel: %s", panel.coverage())
    return panel


def _eligibility(
    symbols: Sequence[str], grid: np.ndarray, snaps: Snapshots
) -> tuple[np.ndarray, np.ndarray | None]:
    """Build the point-in-time eligibility mask and the liquidity surface."""
    shape = (len(symbols), grid.size)
    if not snaps.by_day:
        # No snapshots means we cannot prove point-in-time membership. Say so
        # loudly rather than quietly testing against today's survivors.
        log.warning(
            "no %s in the bundle; every symbol is treated as eligible throughout, "
            "which reintroduces survivorship bias until snapshots are supplied",
            SNAPSHOT_FILE,
        )
        return np.ones(shape, dtype=bool), None

    bar_day, labels = _day_index(grid)
    index = {s: i for i, s in enumerate(symbols)}
    per_day = np.zeros((len(symbols), len(labels)), dtype=bool)
    liq_day = np.zeros((len(symbols), len(labels)), dtype=np.float32)
    for j, label in enumerate(labels):
        for symbol, liquidity in snaps.by_day.get(label, {}).items():
            i = index.get(symbol)
            if i is not None:
                per_day[i, j] = True
                liq_day[i, j] = liquidity
    return per_day[:, bar_day], liq_day[:, bar_day]


# --------------------------------------------------------------------------
# Writing a bundle (used by the droplet exporter and by the tests)
# --------------------------------------------------------------------------
def write_symbol_csv(
    path: Path, rows: Iterable[Sequence[float]], *, gzipped: bool = True
) -> int:
    """Write one symbol's candles as (optionally gzipped) canonical CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(CANONICAL_COLUMNS)
    count = 0
    for row in rows:
        writer.writerow([int(row[0]), *(float(x) for x in row[1:6])])
        count += 1
    payload = buffer.getvalue().encode()
    if gzipped:
        path.write_bytes(gzip.compress(payload, compresslevel=6))
    else:
        path.write_bytes(payload)
    return count


def write_manifest(directory: Path, payload: dict[str, Any]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / MANIFEST_FILE).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def bundle_fingerprint(directory: str | Path) -> str:
    """Identity of a data bundle, so a resumed run is matched to its data.

    Built from each file's name, size and modification time rather than its
    content: hashing gigabytes of candles on every run start would cost more
    than the run saves, and a changed size or mtime is exactly the signal that
    the bundle is no longer the one a checkpoint was built against.
    """
    import hashlib

    directory = Path(directory)
    digest = hashlib.sha256()
    if not directory.exists():
        return "missing"
    for path in sorted(directory.iterdir()):
        if path.is_dir() or path.name == MANIFEST_FILE:
            continue
        stat = path.stat()
        digest.update(f"{path.name}:{stat.st_size}:{int(stat.st_mtime)}".encode())
    return digest.hexdigest()[:16]
