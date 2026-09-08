"""Droplet-side WFMC orchestration (spec 5): bundle materialization, the
focused daily search space, and the daily run's wiring into paramsync.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from solbot import db, paramsync, wfmc
from solbot.candlestore import ParquetCandleStore
from solbot.datastore import DataStore
from solbot.runpod import RunPodClient

MINT_A = "WfmcMintAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
MINT_B = "WfmcMintBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB"


def _seed_universe(conn, mints_pairs: dict[str, str]) -> None:
    for mint, pair in mints_pairs.items():
        conn.execute(
            "INSERT INTO universe(mint, symbol, binance_pair, updated_at) VALUES (?,?,?,?)",
            (mint, pair[:-4], pair, db.now()),
        )
        day = time.strftime("%Y-%m-%d", time.gmtime())
        conn.execute(
            "INSERT INTO universe_history(day, mint, symbol, liquidity_usd, volume_24h_usd) "
            "VALUES (?,?,?,?,?)",
            (day, mint, pair[:-4], 5_000_000.0, 2_000_000.0),
        )


def _seed_minutes(mint: str, days: int, *, seed: int = 1) -> None:
    rng = np.random.default_rng(seed)
    total = days * 1440
    end = int(time.time())
    start = end - total * 60
    price = 1.0
    rows = []
    for i in range(total):
        drift = rng.normal(0, 0.0015)
        spike = (i % 1600) < 3
        if spike:
            drift = abs(drift) + 0.003
        price = max(1e-6, price * (1 + drift))
        rows.append((start + i * 60, price * 0.999, price * 1.004, price * 0.996, price,
                     5000.0 * (6.0 if spike else 1.0)))
    ParquetCandleStore().append(mint, "1m", rows)


# --------------------------------------------------------------------------
# Bundle materialization
# --------------------------------------------------------------------------
def test_materialize_bundle_writes_one_parquet_file_per_coin_and_snapshots(workspace, tmp_path):
    conn = workspace["conn"]
    _seed_universe(conn, {MINT_A: "AAAUSDT", MINT_B: "BBBUSDT"})
    _seed_minutes(MINT_A, days=2)
    _seed_minutes(MINT_B, days=2, seed=2)

    out_dir = wfmc.materialize_bundle(
        ParquetCandleStore(), [MINT_A, MINT_B], out_dir=tmp_path / "bundle", conn=conn
    )

    files = sorted(p.name for p in out_dir.glob("*.parquet"))
    assert files == [f"{MINT_A}__1m.parquet", f"{MINT_B}__1m.parquet"]
    assert (out_dir / "universe_snapshots.csv").exists()
    lines = (out_dir / "universe_snapshots.csv").read_text().strip().splitlines()
    assert lines[0] == "day,symbol,liquidity,volume_24h"
    assert len(lines) == 3  # header + one row per coin


def test_materialize_bundle_skips_a_coin_with_no_candles(workspace, tmp_path):
    conn = workspace["conn"]
    _seed_universe(conn, {MINT_A: "AAAUSDT"})
    # No candle data written for MINT_A at all.
    out_dir = wfmc.materialize_bundle(ParquetCandleStore(), [MINT_A], out_dir=tmp_path / "b", conn=conn)
    assert list(out_dir.glob("*.parquet")) == []


# --------------------------------------------------------------------------
# The focused grid
# --------------------------------------------------------------------------
def test_narrow_grid_trims_each_axis_to_its_neighbourhood():
    base = {"rr_min": [1.5, 2.0, 2.5, 3.0], "ema_fast": [6, 9, 12]}
    narrowed = wfmc._narrow_grid({"rr_min": 2.5, "ema_fast": 12}, base)

    assert narrowed["rr_min"] == [2.0, 2.5, 3.0]
    assert narrowed["ema_fast"] == [9, 12]   # already at the edge


def test_narrow_grid_keeps_the_full_axis_when_the_center_is_off_grid():
    base = {"rr_min": [1.5, 2.0, 2.5]}
    narrowed = wfmc._narrow_grid({"rr_min": 99.0}, base)
    assert narrowed["rr_min"] == [1.5, 2.0, 2.5]


def test_focused_space_falls_back_to_the_default_grid_with_no_prior_run(workspace, settings):
    from solopt.params import DEFAULT_GRID

    space = wfmc.focused_space(settings, workspace["conn"])
    assert space.values == DEFAULT_GRID


def test_focused_space_narrows_once_a_center_is_remembered(workspace, settings):
    conn = workspace["conn"]
    from solopt.params import DEFAULT_GRID

    center = {k: v[0] for k, v in DEFAULT_GRID.items()}
    db.kv_set(wfmc.FOCUSED_GRID_KEY, center, conn)

    space = wfmc.focused_space(settings, conn)
    for key, options in DEFAULT_GRID.items():
        assert len(space.values[key]) <= len(options)


# --------------------------------------------------------------------------
# next_run_id - collision-free across daily runs and RunPod batches
# --------------------------------------------------------------------------
def test_next_run_id_is_monotonic_and_never_repeats(workspace):
    conn = workspace["conn"]
    ids = [wfmc.next_run_id(conn) for _ in range(5)]
    assert ids == sorted(ids)
    assert len(set(ids)) == 5


# --------------------------------------------------------------------------
# run_daily - short-circuits and failure wiring
# --------------------------------------------------------------------------
def test_run_daily_short_circuits_with_no_routed_universe(workspace, settings):
    store = DataStore(binance=None, cfg=settings)
    result = wfmc.run_daily(settings, store)
    assert not result["ran"]
    assert "no routed universe" in result["reason"]


def test_run_daily_reports_a_failure_when_there_is_not_enough_history(workspace, settings):
    """Real wiring, real failure: too little data for even one window.

    Cheap to run (a couple of days of candles) precisely because it is
    expected to fail before doing any real search - WalkForward.run refuses a
    span shorter than one in-sample+out-of-sample window.
    """
    conn = workspace["conn"]
    _seed_universe(conn, {MINT_A: "AAAUSDT"})
    _seed_minutes(MINT_A, days=3)

    store = DataStore(binance=None, cfg=settings)
    with pytest.raises(ValueError, match="too short"):
        wfmc.run_daily(settings, store, conn=conn)

    row = conn.execute("SELECT * FROM optimizer_feed").fetchall()
    assert any("failed" in (r["message"] or "").lower() for r in row)


# --------------------------------------------------------------------------
# run_monthly - RunPod orchestration wiring, fully mocked
# --------------------------------------------------------------------------
class _Secrets:
    runpod_api_key = "test-key"
    runpod_s3_access_key = ""
    runpod_s3_secret_key = ""
    bulk_data_token = "worker-token"


def test_run_monthly_requires_a_runpod_api_key(workspace, settings):
    store = DataStore(binance=None, cfg=settings)

    class NoKey:
        runpod_api_key = ""

    result = wfmc.run_monthly(settings, store, NoKey(), conn=workspace["conn"])
    assert not result["ran"]
    assert "RUNPOD_API_KEY" in result["reason"]


def test_run_monthly_launches_a_batch_per_chunk_and_verifies_teardown(workspace, settings):
    from tests.test_runpod import FakeTransport

    conn = workspace["conn"]
    _seed_universe(conn, {MINT_A: "AAAUSDT", MINT_B: "BBBUSDT"})
    _seed_minutes(MINT_A, days=1)
    _seed_minutes(MINT_B, days=1, seed=3)

    transport = FakeTransport()
    client = RunPodClient(
        api_key="test-key", transport=transport,
        poll_interval_seconds=0.01, max_poll_seconds=0.05,
    )
    store = DataStore(binance=None, cfg={**settings, "runpod_batch_size": 1})

    result = wfmc.run_monthly(
        {**settings, "runpod_batch_size": 1}, store, _Secrets(), conn=conn, runpod_client=client,
    )

    assert result["ran"]
    assert len(result["jobs"]) == 2   # one batch per coin, batch size 1
    assert result["teardown"]["clean"] is True
    assert not transport.pods and not transport.volumes

    events = conn.execute("SELECT message FROM events ORDER BY id").fetchall()
    assert any("teardown check" in e["message"].lower() for e in events)
