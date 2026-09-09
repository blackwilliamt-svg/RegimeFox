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


# --------------------------------------------------------------------------
# GPU-tier benchmark (gap-closure item 7)
# --------------------------------------------------------------------------
def test_run_benchmark_requires_a_runpod_api_key(workspace, settings):
    store = DataStore(binance=None, cfg=settings)

    class NoKey:
        runpod_api_key = ""

    result = wfmc.run_benchmark(settings, store, NoKey(), conn=workspace["conn"])
    assert not result["ran"]
    assert "RUNPOD_API_KEY" in result["reason"]


def test_run_benchmark_requires_a_routed_universe(workspace, settings):
    store = DataStore(binance=None, cfg=settings)
    result = wfmc.run_benchmark(settings, store, _Secrets(), conn=workspace["conn"])
    assert not result["ran"]
    assert "universe" in result["reason"]


def test_run_benchmark_reports_and_stores_results_without_touching_runpod_gpu_type(
    workspace, settings
):
    from tests.test_runpod import FakeTransport

    conn = workspace["conn"]
    _seed_universe(conn, {MINT_A: "AAAUSDT"})
    _seed_minutes(MINT_A, days=1)

    transport = FakeTransport()
    transport.gpu_prices = {"TIER-CHEAP": 0.3, "TIER-PRICEY": 1.2}
    orig_create_pod = None

    client = RunPodClient(
        api_key="test-key", transport=transport,
        poll_interval_seconds=0.01, max_poll_seconds=0.05,
    )
    orig_create_pod = client.create_pod

    def create_pod_and_arm(*a, **kw):
        pod = orig_create_pod(*a, **kw)
        transport.pod_polls_until_stopped[pod["id"]] = 1
        return pod

    client.create_pod = create_pod_and_arm
    store = DataStore(binance=None, cfg=settings)

    original_gpu_type = settings["runpod_gpu_type"]
    result = wfmc.run_benchmark(
        settings, store, _Secrets(), conn=conn, runpod_client=client,
        tiers=["TIER-CHEAP", "TIER-PRICEY"], max_seconds=0.05,
    )

    assert result["ran"]
    assert len(result["results"]) == 2
    assert all(r["ok"] for r in result["results"])
    assert result["recommendation"]   # a recommendation string, not applied

    # Never touches the live setting - only surfaces a recommendation.
    assert settings["runpod_gpu_type"] == original_gpu_type

    stored = db.kv_get(wfmc.RUNPOD_BENCHMARK_KEY, conn=conn)
    assert stored["results"][0]["gpu_type"] == "TIER-CHEAP"
    assert stored["recommendation"]

    events = conn.execute("SELECT message, level FROM events ORDER BY id").fetchall()
    assert any("benchmark finished" in e["message"].lower() for e in events)


def test_run_benchmark_flags_a_tier_that_failed_to_tear_down_cleanly(workspace, settings):
    from tests.test_runpod import FakeTransport, RunPodError

    conn = workspace["conn"]
    _seed_universe(conn, {MINT_A: "AAAUSDT"})
    _seed_minutes(MINT_A, days=1)

    transport = FakeTransport()
    transport.gpu_prices = {"TIER-A": 0.5}
    client = RunPodClient(
        api_key="test-key", transport=transport,
        poll_interval_seconds=0.01, max_poll_seconds=0.02,
    )
    store = DataStore(binance=None, cfg=settings)

    def fail_delete(volume_id: str) -> None:
        raise RunPodError("simulated failure")

    client.delete_network_volume = fail_delete

    result = wfmc.run_benchmark(
        settings, store, _Secrets(), conn=conn, runpod_client=client, tiers=["TIER-A"],
        max_seconds=0.05,
    )

    assert result["ran"]
    assert result["results"][0]["teardown_clean"] is False

    events = conn.execute("SELECT message, level FROM events ORDER BY id").fetchall()
    assert any(
        "not cleanly torn down" in e["message"].lower() and e["level"] == "alert"
        for e in events
    )


# --------------------------------------------------------------------------
# Settings fix-up section 4, item 2: live status/progress for the benchmark -
# it launches real billed pods and can take a while, so "Run benchmark now"
# needs something to show while it's in flight, the same bar()-polling
# pattern run_regime_pass/run_monthly already use (REGIME_PASS_JOB etc.).
# --------------------------------------------------------------------------
def test_run_benchmark_reports_failed_progress_when_no_api_key(workspace, settings):
    store = DataStore(binance=None, cfg=settings)

    class NoKey:
        runpod_api_key = ""

    wfmc.run_benchmark(settings, store, NoKey(), conn=workspace["conn"])

    progress = db.get_progress(wfmc.RUNPOD_BENCHMARK_JOB, conn=workspace["conn"])
    assert progress["status"] == "failed"
    assert "RUNPOD_API_KEY" in progress["message"]


def test_run_benchmark_reports_failed_progress_when_no_routed_universe(workspace, settings):
    store = DataStore(binance=None, cfg=settings)
    wfmc.run_benchmark(settings, store, _Secrets(), conn=workspace["conn"])

    progress = db.get_progress(wfmc.RUNPOD_BENCHMARK_JOB, conn=workspace["conn"])
    assert progress["status"] == "failed"
    assert "universe" in progress["message"]


def test_run_benchmark_reports_progress_per_tier_and_a_done_summary_at_the_end(
    workspace, settings, monkeypatch
):
    from tests.test_runpod import FakeTransport

    conn = workspace["conn"]
    _seed_universe(conn, {MINT_A: "AAAUSDT"})
    _seed_minutes(MINT_A, days=1)

    transport = FakeTransport()
    transport.gpu_prices = {"TIER-CHEAP": 0.3, "TIER-PRICEY": 1.2}
    client = RunPodClient(
        api_key="test-key", transport=transport,
        poll_interval_seconds=0.01, max_poll_seconds=0.05,
    )
    orig_create_pod = client.create_pod

    def create_pod_and_arm(*a, **kw):
        pod = orig_create_pod(*a, **kw)
        transport.pod_polls_until_stopped[pod["id"]] = 1
        return pod

    client.create_pod = create_pod_and_arm
    store = DataStore(binance=None, cfg=settings)

    seen: list[dict] = []
    original_set_progress = db.set_progress

    def spy(job, **kwargs):
        if job == wfmc.RUNPOD_BENCHMARK_JOB:
            seen.append(dict(kwargs))
        return original_set_progress(job, **kwargs)

    monkeypatch.setattr("solbot.wfmc.db.set_progress", spy)

    result = wfmc.run_benchmark(
        settings, store, _Secrets(), conn=conn, runpod_client=client,
        tiers=["TIER-CHEAP", "TIER-PRICEY"], max_seconds=0.05,
    )
    assert result["ran"]

    statuses = [s["status"] for s in seen]
    assert statuses[0] == "running"      # reported at start, before any tier
    assert statuses[-1] == "done"        # a final done/failed state at the end
    assert any(
        s["status"] == "running" and "tier 1 of 2" in s.get("message", "") for s in seen
    )
    assert any(
        s["status"] == "running" and "tier 2 of 2" in s.get("message", "") for s in seen
    )

    final = db.get_progress(wfmc.RUNPOD_BENCHMARK_JOB, conn=conn)
    assert final["status"] == "done"
    assert final["done"] == final["total"] == 2
    assert "benchmark finished" in final["message"].lower()


def test_run_benchmark_reports_failed_progress_when_it_could_not_start(workspace, settings):
    from solbot.runpod import RunPodError

    conn = workspace["conn"]
    _seed_universe(conn, {MINT_A: "AAAUSDT"})
    _seed_minutes(MINT_A, days=1)

    class ExplodingClient:
        def benchmark_tiers(self, *a, **kw):
            raise RunPodError("simulated launch failure")

    store = DataStore(binance=None, cfg=settings)
    result = wfmc.run_benchmark(
        settings, store, _Secrets(), conn=conn, runpod_client=ExplodingClient(),
        tiers=["TIER-A"],
    )

    assert not result["ran"]
    progress = db.get_progress(wfmc.RUNPOD_BENCHMARK_JOB, conn=conn)
    assert progress["status"] == "failed"
    assert "simulated launch failure" in progress["message"]


# --------------------------------------------------------------------------
# Fuzzy-regime section, step 5: manual trigger + progress - dispatched to
# RunPod exactly like run_monthly, never run in-process on the droplet.
# --------------------------------------------------------------------------
def test_run_regime_pass_requires_a_runpod_api_key(workspace, settings):
    store = DataStore(binance=None, cfg=settings)

    class NoKey:
        runpod_api_key = ""

    result = wfmc.run_regime_pass(settings, store, NoKey(), conn=workspace["conn"])
    assert not result["ran"]
    assert "RUNPOD_API_KEY" in result["reason"]

    progress = db.get_progress(wfmc.REGIME_PASS_JOB, conn=workspace["conn"])
    assert progress["status"] == "failed"


def test_run_regime_pass_requires_a_routed_universe(workspace, settings):
    store = DataStore(binance=None, cfg=settings)
    result = wfmc.run_regime_pass(settings, store, _Secrets(), conn=workspace["conn"])
    assert not result["ran"]
    assert "universe" in result["reason"]

    progress = db.get_progress(wfmc.REGIME_PASS_JOB, conn=workspace["conn"])
    assert progress["status"] == "failed"


def test_run_regime_pass_launches_a_runpod_batch_per_chunk_and_verifies_teardown(workspace, settings):
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

    result = wfmc.run_regime_pass(
        {**settings, "runpod_batch_size": 1}, store, _Secrets(), conn=conn, runpod_client=client,
    )

    assert result["ran"]
    assert len(result["jobs"]) == 2   # one batch per coin, batch size 1
    assert result["teardown"]["clean"] is True
    assert not transport.pods and not transport.volumes

    # The job kind (and this pass's own knobs) are signalled to the worker
    # through its environment, the same mechanism the benchmark path already
    # uses for SOLOPT_BENCHMARK - no separate dispatch endpoint needed.
    pod_calls = [c for c in transport.calls if c[0] == "POST" and c[1] == "/pods"]
    assert len(pod_calls) == 2
    for _, _, body in pod_calls:
        # A plain {key: value} object - RunPod's PodCreateInput schema, not
        # a list of {key, value} pairs (a real pod-creation call's own 400
        # is what caught that mismatch).
        env = body["env"]
        assert env["SOLOPT_JOB_KIND"] == "regime-pass"
        assert env["SOLOPT_REGIME_K_RANGE"] == "4,5,6"

    progress = db.get_progress(wfmc.REGIME_PASS_JOB, conn=conn)
    assert progress["status"] == "done"
    assert progress["done"] == progress["total"] == 2

    events = conn.execute("SELECT message FROM events ORDER BY id").fetchall()
    assert any(
        "regime" in e["message"].lower() and "runpod" in e["message"].lower() for e in events
    )
    assert any("teardown check" in e["message"].lower() for e in events)


def test_run_regime_pass_records_a_launch_failure_without_aborting_the_others(workspace, settings):
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

    orig_create_pod = client.create_pod
    state = {"calls": 0}

    def flaky_create_pod(*a, **kw):
        state["calls"] += 1
        if state["calls"] == 1:
            from solbot.runpod import RunPodError

            raise RunPodError("simulated launch failure")
        return orig_create_pod(*a, **kw)

    client.create_pod = flaky_create_pod

    result = wfmc.run_regime_pass(
        {**settings, "runpod_batch_size": 1}, store, _Secrets(), conn=conn, runpod_client=client,
    )

    assert result["ran"]
    assert len(result["jobs"]) == 1
    assert len(result["errors"]) == 1
