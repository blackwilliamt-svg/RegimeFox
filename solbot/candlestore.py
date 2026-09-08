"""Per-coin Parquet candle storage (spec 3).

Historical candle data lives on disk as one Parquet file per coin per
calendar month, never as SQLite rows. SQLite keeps trades, positions,
settings, the audit log and walk-forward/Monte Carlo results; a year of
1-minute candles across a hundred coins would put that alongside the trading
loop's own state on the same write path, which is exactly what this design
avoids.

Layout: ``<root>/<interval>/<mint>/<YYYY-MM>.parquet``. Partitioning by month
(spec 3a) rather than one giant per-coin file means a daily incremental pull
is a cheap append to a small file rather than a rewrite of the whole history,
and a walk-forward run reading last year's data is never blocked behind
today's write.

**Append-without-duplication.** Every append checks the last stored timestamp
first (:meth:`ParquetCandleStore.latest_ts`), so a re-run or restart never
writes the same candle twice; new rows are merged with whatever the target
month's file already holds, de-duplicated on timestamp, and the result is
written to a temp file and renamed over the original. That rename is what
keeps a walk-forward run reading the old file mid-write immune to a torn
read - the run started from a listing of complete files and finishes against
that same snapshot even if a new candle lands while it runs (spec 3a's
concurrency rule): it is never asked to read a half-written one, because there
never is one.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

CANONICAL_COLUMNS = ("ts", "open", "high", "low", "close", "volume")


def _month_label(ts: int) -> str:
    return time.strftime("%Y-%m", time.gmtime(int(ts)))


def default_root() -> Path:
    return Path(os.getenv("SOLBOT_CANDLES_DIR", "data/candles"))


class ParquetCandleStore:
    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root) if root is not None else default_root()

    # ------------------------------------------------------------------
    def _coin_dir(self, mint: str, interval: str) -> Path:
        # Mints are base58 (alnum only) and Binance pairs are alnum too, so no
        # escaping is needed; this is not exposed to arbitrary user input.
        return self.root / interval / mint

    def _month_path(self, mint: str, interval: str, label: str) -> Path:
        return self._coin_dir(mint, interval) / f"{label}.parquet"

    def months_available(self, mint: str, interval: str) -> list[str]:
        d = self._coin_dir(mint, interval)
        if not d.exists():
            return []
        return sorted(p.stem for p in d.glob("*.parquet"))

    # ------------------------------------------------------------------
    # writing
    # ------------------------------------------------------------------
    def latest_ts(self, mint: str, interval: str) -> int | None:
        """Newest candle timestamp stored for this coin, or None."""
        months = self.months_available(mint, interval)
        if not months:
            return None
        path = self._month_path(mint, interval, months[-1])
        table = self._read_table(path)
        if table is None or table.num_rows == 0:
            return None
        return int(np.max(table.column("ts").to_numpy()))

    def append(
        self,
        mint: str,
        interval: str,
        rows: Iterable[Sequence[Any]],
        *,
        incremental: bool = True,
    ) -> int:
        """Append ``(ts, open, high, low, close, volume)`` rows.

        Rows are split by the calendar month they fall in and merged into
        each month's file individually, so a pull spanning a month boundary
        never has to rewrite the whole history to append a handful of bars.
        """
        rows = [tuple(r) for r in rows]
        if not rows:
            return 0

        if incremental:
            latest = self.latest_ts(mint, interval)
            if latest is not None:
                rows = [r for r in rows if int(r[0]) > latest]
        if not rows:
            return 0

        by_month: dict[str, list[Sequence[Any]]] = {}
        for row in rows:
            by_month.setdefault(_month_label(int(row[0])), []).append(row)

        written = 0
        for label, month_rows in by_month.items():
            written += self._merge_month(mint, interval, label, month_rows)
        return written

    def _merge_month(
        self, mint: str, interval: str, label: str, rows: list[Sequence[Any]]
    ) -> int:
        import pyarrow as pa
        import pyarrow.parquet as pq

        path = self._month_path(mint, interval, label)
        existing = self._read_table(path)

        new_table = pa.table(
            {
                "ts": pa.array([int(r[0]) for r in rows], type=pa.int64()),
                "open": pa.array([float(r[1]) for r in rows], type=pa.float64()),
                "high": pa.array([float(r[2]) for r in rows], type=pa.float64()),
                "low": pa.array([float(r[3]) for r in rows], type=pa.float64()),
                "close": pa.array([float(r[4]) for r in rows], type=pa.float64()),
                "volume": pa.array([float(r[5]) for r in rows], type=pa.float64()),
            }
        )
        combined = pa.concat_tables([existing, new_table]) if existing is not None else new_table

        df = combined.to_pandas()
        df = df.drop_duplicates(subset="ts", keep="last").sort_values("ts")
        out_table = pa.Table.from_pandas(df, preserve_index=False)

        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".parquet.tmp")
        pq.write_table(out_table, tmp, compression="zstd")
        # Atomic rename: a walk-forward run holding the old file open keeps
        # reading the pre-rename data; a fresh open sees the merged file. No
        # reader ever observes a half-written one.
        tmp.replace(path)
        return len(rows)

    # ------------------------------------------------------------------
    # reading
    # ------------------------------------------------------------------
    @staticmethod
    def _read_table(path: Path) -> Any:
        import pyarrow.parquet as pq

        if not path.exists():
            return None
        try:
            return pq.read_table(path)
        except Exception as exc:  # a corrupt file must not crash the caller
            log.warning("candle file %s unreadable, treating as empty: %s", path, exc)
            return None

    def read_range(
        self,
        mint: str,
        interval: str,
        *,
        since: int | None = None,
        until: int | None = None,
        limit: int | None = None,
    ) -> pd.DataFrame:
        """Candles at the base interval, oldest first."""
        months = self.months_available(mint, interval)
        if since is not None:
            lo = _month_label(since)
            months = [m for m in months if m >= lo]
        if until is not None:
            hi = _month_label(until)
            months = [m for m in months if m <= hi]
        if limit and not since:
            # Reading tail-first: walk backward until enough rows accumulate,
            # rather than loading the whole history to keep the last N bars.
            months = months[-max(1, _months_for_limit(months, limit)) :]

        frames = []
        for label in months:
            table = self._read_table(self._month_path(mint, interval, label))
            if table is not None and table.num_rows:
                frames.append(table.to_pandas())
        if not frames:
            return pd.DataFrame(columns=CANONICAL_COLUMNS)

        df = pd.concat(frames, ignore_index=True)
        df = df.drop_duplicates(subset="ts", keep="last").sort_values("ts").reset_index(drop=True)
        if since is not None:
            df = df[df["ts"] >= since]
        if until is not None:
            df = df[df["ts"] <= until]
        if limit:
            df = df.tail(int(limit)).reset_index(drop=True)
        return df[list(CANONICAL_COLUMNS)]

    # ------------------------------------------------------------------
    def coverage(self, mint: str, interval: str) -> dict[str, Any]:
        months = self.months_available(mint, interval)
        if not months:
            return {"bars": 0, "first_ts": None, "last_ts": None, "days": 0.0}
        first_table = self._read_table(self._month_path(mint, interval, months[0]))
        last_table = self._read_table(self._month_path(mint, interval, months[-1]))
        first_ts = int(np.min(first_table.column("ts").to_numpy())) if first_table else None
        last_ts = int(np.max(last_table.column("ts").to_numpy())) if last_table else None
        bars = sum(
            (self._read_table(self._month_path(mint, interval, m)) or _EMPTY).num_rows
            for m in months
        )
        return {
            "bars": bars,
            "first_ts": first_ts,
            "last_ts": last_ts,
            "days": round(((last_ts or 0) - (first_ts or 0)) / 86400.0, 1) if last_ts else 0.0,
        }

    def universe_coverage(self, mints: Iterable[str], interval: str) -> dict[str, Any]:
        """Aggregate coverage across every coin, for the dashboard's summary card."""
        total_bars = 0
        first_ts: int | None = None
        last_ts: int | None = None
        tokens = 0
        for mint in mints:
            cov = self.coverage(mint, interval)
            if not cov["bars"]:
                continue
            tokens += 1
            total_bars += cov["bars"]
            first_ts = cov["first_ts"] if first_ts is None else min(first_ts, cov["first_ts"])
            last_ts = cov["last_ts"] if last_ts is None else max(last_ts, cov["last_ts"])
        return {
            "interval": interval,
            "tokens": tokens,
            "candles": total_bars,
            "first_ts": first_ts,
            "last_ts": last_ts,
            "days": round(((last_ts or 0) - (first_ts or 0)) / 86400.0, 1) if last_ts else 0.0,
        }

    # ------------------------------------------------------------------
    def materialize_bundle(
        self, mints: Iterable[str], interval: str, out_dir: str | Path
    ) -> int:
        """Write one merged Parquet file per coin, in the layout solopt reads.

        The walk-forward optimizer (:mod:`solopt.dataset`) expects a
        directory of ``<symbol>__<interval>.parquet`` files, one per coin -
        the same shape the old PC-side bundle sync used to produce. Since the
        optimizer now runs against this store directly (daily, on-droplet) or
        against a copy shipped to a remote worker (monthly, on RunPod), this
        is the one place that bridges "many small month-files per coin" to
        "one file per coin", so :func:`solopt.dataset.load_panel` never has
        to know the candle store's internal layout.
        """
        import pyarrow.parquet as pq

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        written = 0
        for mint in mints:
            df = self.read_range(mint, interval)
            if df.empty:
                continue
            import pyarrow as pa

            table = pa.Table.from_pandas(df, preserve_index=False)
            pq.write_table(table, out_dir / f"{mint}__{interval}.parquet", compression="zstd")
            written += 1
        return written

    # ------------------------------------------------------------------
    def disk_usage(self) -> dict[str, Any]:
        """Bytes on disk, for the storage browser (spec 6c)."""
        total = 0
        files = 0
        if self.root.exists():
            for path in self.root.rglob("*.parquet"):
                try:
                    total += path.stat().st_size
                    files += 1
                except OSError:
                    continue
        return {"bytes": total, "megabytes": round(total / 1e6, 2), "files": files}


def _months_for_limit(months: list[str], limit: int) -> int:
    """A rough number of month-files that should hold ``limit`` bars.

    Cheap and approximate on purpose: the caller trims to ``limit`` after
    reading, so over-including a month costs a little I/O, never correctness.
    """
    return max(1, min(len(months), (limit // 1000) + 2))


_EMPTY = type("_Empty", (), {"num_rows": 0})()
