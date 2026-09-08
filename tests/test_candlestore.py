"""Per-coin Parquet candle storage (spec 3): append-without-duplication,
month partitioning, and reading back what was written."""
from __future__ import annotations

import os

import pytest

from solbot.candlestore import ParquetCandleStore

MINT = "TestMintAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"


def _rows(start_ts: int, count: int, step: int = 60):
    return [
        (start_ts + i * step, 1.0 + i * 0.001, 1.02, 0.98, 1.0 + i * 0.001, 500.0)
        for i in range(count)
    ]


def test_append_then_read_round_trips(tmp_path):
    store = ParquetCandleStore(tmp_path / "candles")
    rows = _rows(1_700_000_000, 100)
    written = store.append(MINT, "1m", rows)

    assert written == 100
    df = store.read_range(MINT, "1m")
    assert len(df) == 100
    assert list(df.columns) == ["ts", "open", "high", "low", "close", "volume"]
    assert df["ts"].is_monotonic_increasing


def test_append_never_duplicates_a_candle(tmp_path):
    store = ParquetCandleStore(tmp_path / "candles")
    rows = _rows(1_700_000_000, 50)
    store.append(MINT, "1m", rows)
    # Re-running the same pull (a restart, say) must not double the rows.
    store.append(MINT, "1m", rows)

    df = store.read_range(MINT, "1m")
    assert len(df) == 50


def test_incremental_append_only_writes_the_gap(tmp_path):
    store = ParquetCandleStore(tmp_path / "candles")
    store.append(MINT, "1m", _rows(1_700_000_000, 50))
    latest = store.latest_ts(MINT, "1m")
    assert latest == 1_700_000_000 + 49 * 60

    # Overlapping request: only the truly new bars should be written.
    written = store.append(MINT, "1m", _rows(1_700_000_000 + 40 * 60, 20))
    assert written == 10
    assert len(store.read_range(MINT, "1m")) == 60


def test_data_spanning_a_month_boundary_splits_into_two_files(tmp_path):
    store = ParquetCandleStore(tmp_path / "candles")
    # 2024-01-31 23:00 UTC through 2024-02-01 01:00 UTC, hourly bars.
    start = 1_706_734_800  # 2024-01-31 23:00:00 UTC
    store.append(MINT, "1h", _rows(start, 4, step=3600))

    months = store.months_available(MINT, "1h")
    assert months == ["2024-01", "2024-02"]
    assert len(store.read_range(MINT, "1h")) == 4


@pytest.mark.skipif(
    os.name == "nt",
    reason="atomic rename-over-an-open-handle is a POSIX guarantee; the "
    "deployment target is a Linux droplet, not Windows",
)
def test_concurrent_read_sees_a_consistent_snapshot(tmp_path):
    """A reader that opened the file before an append must not see a torn write.

    The merge writes to a temp file and renames over the original, so an
    already-open file handle keeps reading the pre-rename bytes.
    """
    store = ParquetCandleStore(tmp_path / "candles")
    store.append(MINT, "1m", _rows(1_700_000_000, 30))
    path = store._month_path(MINT, "1m", "2023-11")

    handle = path.open("rb")
    try:
        before = handle.read()
        store.append(MINT, "1m", _rows(1_700_000_000 + 30 * 60, 30))
        handle.seek(0)
        after_same_handle = handle.read()
        assert before == after_same_handle, "the renamed-over file must not mutate the old inode"
    finally:
        handle.close()
    assert len(store.read_range(MINT, "1m")) == 60


def test_missing_coin_reads_back_empty(tmp_path):
    store = ParquetCandleStore(tmp_path / "candles")
    df = store.read_range("NeverSeenMint", "1m")
    assert df.empty
    assert store.latest_ts("NeverSeenMint", "1m") is None


def test_coverage_and_disk_usage(tmp_path):
    store = ParquetCandleStore(tmp_path / "candles")
    store.append(MINT, "1m", _rows(1_700_000_000, 100))

    cov = store.coverage(MINT, "1m")
    assert cov["bars"] == 100
    assert cov["first_ts"] == 1_700_000_000

    usage = store.disk_usage()
    assert usage["files"] == 1
    assert usage["bytes"] > 0
