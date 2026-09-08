"""Walk-forward/Monte Carlo orchestration on the droplet (spec 5).

Two cadences, two places the search actually runs, one reusable engine
(:mod:`solopt.pipeline`) underneath both:

* **Daily**, on this droplet's own CPU, against a *focused* set of
  parameter combinations retained from the last monthly run - narrow enough
  to finish in minutes on one core, so it does not meaningfully compete with
  the trading loop for the vCPU.
* **Monthly**, on a RunPod GPU worker this droplet spins up, ships data to,
  and tears down again - a full sweep across the entire parameter space,
  wide enough to catch anything the narrowed daily search misses.

There is no PC in this picture and no git hand-off repository. A finished
bundle goes straight into :func:`solbot.paramsync.accept_bundle` - directly,
in-process, for the daily run; over the ``/api/optimizer/bundle`` endpoint,
for a RunPod worker reporting back.
"""
from __future__ import annotations

import csv
import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any

from . import db, paramsync
from .candlestore import ParquetCandleStore

log = logging.getLogger(__name__)

LAST_DAILY_KEY = "wfmc_last_daily_day"
LAST_MONTHLY_KEY = "wfmc_last_monthly_month"
FOCUSED_GRID_KEY = "wfmc_focused_center"
RUN_ID_COUNTER_KEY = "wfmc_next_run_id"
TEARDOWN_LOG_KEY = "wfmc_last_teardown_check"

DAILY_BUNDLE_DIR = "data/wfmc_bundle"
DAILY_STORE_PATH = "data/wfmc.db"
MONTHLY_STORE_PATH = "data/wfmc_monthly.db"


def next_run_id(conn: sqlite3.Connection | None = None) -> int:
    """A run id unique across every daily run and every RunPod batch ever.

    ``optimizer_runs.run_id`` is shared by however many separate local
    stores are involved (this droplet's own daily store, and a fresh one per
    RunPod worker) - handing out the number here, once, per run, is what
    keeps two workers' "run 1" from colliding in that table.
    """
    conn = conn or db.connect()
    current = int(db.kv_get(RUN_ID_COUNTER_KEY, 0, conn) or 0) + 1
    db.kv_set(RUN_ID_COUNTER_KEY, current, conn)
    return current


class _LocalFeedSink:
    """Delivers feed lines straight into solbot's own tables - no HTTP.

    Used only by the daily run: it is already the same process and the same
    database, so posting to itself over the network would be pure overhead.
    """

    def __init__(self, conn: sqlite3.Connection | None = None) -> None:
        self.conn = conn or db.connect()

    def emit(self, run_id: int, line: dict[str, Any]) -> None:
        self.conn.execute(
            "INSERT INTO optimizer_runs(run_id, started_at, updated_at, status) "
            "VALUES (?,?,?, 'running') ON CONFLICT(run_id) DO UPDATE SET "
            "updated_at = excluded.updated_at",
            (run_id, db.now(), db.now()),
        )
        self.conn.execute(
            "INSERT INTO optimizer_feed(run_id, remote_id, ts, level, message, detail) "
            "VALUES (?,?,?,?,?,?) ON CONFLICT(run_id, remote_id) DO NOTHING",
            (
                run_id, int(line.get("id", 0)), int(line.get("ts", db.now())),
                line.get("level", "info"), str(line.get("message", ""))[:2000],
                json.dumps(line.get("detail"), default=str) if line.get("detail") is not None else None,
            ),
        )

    def flush(self) -> None:
        return None


def _write_snapshots_csv(path: Path, conn: sqlite3.Connection) -> None:
    """``universe_snapshots.csv`` from ``universe_history`` - point-in-time
    membership, exactly what :mod:`solopt.dataset` needs for eligibility."""
    rows = conn.execute(
        "SELECT day, mint, liquidity_usd, volume_24h_usd FROM universe_history ORDER BY day, mint"
    ).fetchall()
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["day", "symbol", "liquidity", "volume_24h"])
        for r in rows:
            writer.writerow([r["day"], r["mint"], r["liquidity_usd"] or 0.0, r["volume_24h_usd"] or 0.0])


def materialize_bundle(
    candles: ParquetCandleStore, mints: list[str], *, out_dir: str | Path, conn: sqlite3.Connection | None = None
) -> Path:
    """Rebuild the solopt-readable bundle directory from the Parquet store."""
    conn = conn or db.connect()
    out_dir = Path(out_dir)
    candles.materialize_bundle(mints, candles_interval(), out_dir)
    _write_snapshots_csv(out_dir / "universe_snapshots.csv", conn)
    return out_dir


def candles_interval() -> str:
    from .datastore import BASE_INTERVAL

    return BASE_INTERVAL


# --------------------------------------------------------------------------
# Focused grid (the daily search space)
# --------------------------------------------------------------------------
def _narrow_grid(center: dict[str, Any], base_grid: dict[str, list[Any]]) -> dict[str, list[Any]]:
    """A grid narrowed to the neighbourhood of ``center`` on each axis.

    Keeps the daily search "focused" (spec 5): each axis is trimmed to the
    values in the base grid immediately around the monthly winner, rather
    than searching the whole space again every day.
    """
    narrowed: dict[str, list[Any]] = {}
    for key, options in base_grid.items():
        value = center.get(key)
        if value in options:
            idx = options.index(value)
            lo, hi = max(0, idx - 1), min(len(options), idx + 2)
            narrowed[key] = sorted(set(options[lo:hi]))
        else:
            narrowed[key] = list(options)
    return narrowed


def focused_space(cfg: dict[str, Any], conn: sqlite3.Connection | None = None):
    from solopt.params import DEFAULT_GRID, ParamSpace

    conn = conn or db.connect()
    center = db.kv_get(FOCUSED_GRID_KEY, None, conn)
    values = _narrow_grid(center, DEFAULT_GRID) if center else dict(DEFAULT_GRID)
    return ParamSpace(values=values)


def _remember_focus(best_params: dict[str, Any] | None, conn: sqlite3.Connection) -> None:
    if best_params:
        db.kv_set(FOCUSED_GRID_KEY, best_params, conn)


# --------------------------------------------------------------------------
# Daily on-droplet run
# --------------------------------------------------------------------------
def run_daily(
    cfg: dict[str, Any],
    store: Any,
    *,
    conn: sqlite3.Connection | None = None,
    wf_config_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The daily incremental walk-forward/Monte Carlo run, on this droplet's CPU.

    ``store`` is the running engine's :class:`solbot.datastore.DataStore` -
    reused so the same Parquet candle store and universe are the ones tested.
    ``wf_config_overrides`` exists for tests, which need a search cheap enough
    to run in seconds; production never passes it.
    """
    from solopt.arrays import get_backend
    from solopt.dataset import load_panel
    from solopt.engine import PortfolioSettings
    from solopt.feed import RunFeed
    from solopt.frames import compact
    from solopt.montecarlo import ExecutionProfile
    from solopt.params import ParamSpace
    from solopt.pipeline import run_pipeline
    from solopt.schema import CostModel, get_schema
    from solopt.store import RunStore
    from solopt.walkforward import WalkForwardConfig

    conn = conn or db.connect()
    mints = list(store.pair_map(conn))
    if not mints:
        return {"ran": False, "reason": "no routed universe tokens to test"}

    out_dir = materialize_bundle(store.candles, mints, out_dir=DAILY_BUNDLE_DIR, conn=conn)
    schema = get_schema("crypto")
    timeframe_seconds = int(cfg["candle_minutes"]) * 60

    panel = load_panel(out_dir, timeframe_seconds=timeframe_seconds, schema=schema)
    frames = compact(panel)
    coverage = panel.coverage()

    space: ParamSpace = focused_space(cfg, conn)
    wf_config = WalkForwardConfig(
        **{
            "in_sample_days": 45, "out_of_sample_days": 10, "step_days": 10,
            "max_evaluations": 300, "batch_size": 32, "workers": 1,
            "utilization_pct": 80.0,
            "library_seed_fraction": float(cfg.get("library_seed_fraction", 0.3)),
            **(wf_config_overrides or {}),
        }
    )
    portfolio = PortfolioSettings(
        starting_balance=float(cfg["paper_starting_balance"]),
        max_total_deployed_pct=float(cfg["max_total_deployed_pct"]),
        max_position_pct_of_wallet=float(cfg["max_position_pct_of_wallet"]),
        max_position_pct_of_liquidity=float(cfg["max_position_pct_of_liquidity"]),
        min_position_usd=float(cfg["min_position_usd"]),
        min_candles_required=int(cfg["min_candles_required"]),
        volatility_target_atr_pct=float(cfg["volatility_target_atr_pct"]),
        volatility_size_floor=float(cfg["volatility_size_floor"]),
        correlation_lookback=int(cfg["correlation_lookback"]),
        correlation_max=float(cfg["correlation_max"]),
        costs=CostModel(fee_pct=float(cfg["taker_fee_pct"]), slippage_pct=float(cfg["max_slippage_pct"])),
        confluence_multiples=tuple(cfg.get("confluence_timeframes") or ()),
        sizing_mode=str(cfg.get("sizing_mode", "portfolio")),
        portfolio_vol_target=float(cfg.get("portfolio_vol_target", 0.019)),
        drawdown_tolerance=float(cfg.get("drawdown_tolerance", 0.25)),
    )

    local_store = RunStore(DAILY_STORE_PATH)
    bundle_id = time.strftime("daily-%Y-%m-%d", time.gmtime())
    run_id = local_store.start_run(
        config=wf_config.as_dict(), settings=portfolio.as_dict(),
        space={"values": space.values}, bundle=bundle_id, coverage=coverage,
        label=bundle_id,
    )
    report_id = next_run_id(conn)
    feed = RunFeed(local_store, run_id, report_run_id=report_id, sinks=[_LocalFeedSink(conn)])
    feed.loaded(coverage)

    samples = db.execution_samples(conn=conn)
    execution = ExecutionProfile.from_fills(
        [s for s, _ in samples], [f for _, f in samples],
        fallback=ExecutionProfile.assumed(portfolio.slippage_pct / 2.0, portfolio.fee_pct),
    )

    backend = get_backend(prefer_gpu=False)
    try:
        result = run_pipeline(
            frames, space=space, portfolio=portfolio, wf_config=wf_config,
            backend=backend, feed=feed, store=local_store, run_id=run_id,
            resume=True, coverage=coverage, execution=execution,
            monte_carlo_iterations=2000, run_meta={"bundle": bundle_id, "kind": "daily"},
        )
    except Exception as exc:
        feed.alert(f"Daily run failed: {exc}")
        local_store.finish_run(run_id, "failed", {"error": str(exc)[:500]})
        raise

    summary = dict(result.summary)
    summary["accepted"] = result.accepted
    feed.finished(summary)
    local_store.finish_run(run_id, "done", summary)

    outcome_accept = None
    if result.accepted:
        _remember_focus(result.outcome.best_params, conn)
        outcome_accept = paramsync.accept_bundle(
            result.bundle.as_dict(), cfg, source="daily-droplet", conn=conn
        )
    else:
        feed.warn(f"Nothing accepted: {result.verdict}.")

    monte_carlo_payload = {
        **result.monte_carlo.summary(), **result.monte_carlo.histograms(),
        "drawdown_histogram": result.monte_carlo.drawdown_histogram,
        **result.monte_carlo.worst_paths(),
    }
    conn.execute(
        "UPDATE optimizer_runs SET status='done', label=?, coverage=?, summary=?, "
        "monte_carlo=?, stress=? WHERE run_id = ?",
        (
            bundle_id, json.dumps(coverage, default=float), json.dumps(summary, default=float),
            json.dumps(monte_carlo_payload, default=float),
            json.dumps([r.as_dict() for r in result.stress], default=float),
            report_id,
        ),
    )
    return {"ran": True, "run_id": report_id, "summary": summary, "accepted": outcome_accept}


# --------------------------------------------------------------------------
# Monthly RunPod-orchestrated run
# --------------------------------------------------------------------------
def run_monthly(
    cfg: dict[str, Any],
    store: Any,
    secrets: Any,
    *,
    conn: sqlite3.Connection | None = None,
    runpod_client: Any = None,
) -> dict[str, Any]:
    """Orchestrate the monthly full-space retest on RunPod.

    Materializes the whole bundle, splits it into coin batches, ships each
    batch to its own RunPod GPU worker, waits for them to report back
    (they POST to ``/api/optimizer/bundle``/``/api/optimizer/feed`` like any
    other reporting client), then tears every worker and volume down and
    verifies the teardown - logged either way.
    """
    from .runpod import RunPodClient, RunPodError

    conn = conn or db.connect()
    if not secrets.runpod_api_key:
        return {"ran": False, "reason": "RUNPOD_API_KEY is not configured"}

    mints = list(store.pair_map(conn))
    if not mints:
        return {"ran": False, "reason": "no routed universe tokens to test"}

    batch_size = int(cfg.get("runpod_batch_size", 75))
    batches = [mints[i : i + batch_size] for i in range(0, len(mints), batch_size)]

    client = runpod_client or RunPodClient(secrets.runpod_api_key)
    jobs: list[dict[str, Any]] = []
    errors: list[str] = []

    for index, batch in enumerate(batches):
        run_id = next_run_id(conn)
        try:
            job = client.launch_batch(
                batch_index=index,
                mints=batch,
                candles=store.candles,
                interval=candles_interval(),
                gpu_type=str(cfg.get("runpod_gpu_type") or "NVIDIA RTX 4090"),
                report_run_id=run_id,
                droplet_url=str(cfg.get("runpod_callback_url") or ""),
                droplet_token=getattr(secrets, "bulk_data_token", ""),
                s3_access_key=getattr(secrets, "runpod_s3_access_key", ""),
                s3_secret_key=getattr(secrets, "runpod_s3_secret_key", ""),
            )
            jobs.append({"run_id": run_id, **job})
        except RunPodError as exc:
            errors.append(f"batch {index}: {exc}")
            log.warning("RunPod batch %d failed to launch: %s", index, exc)

    db.log_event(
        f"Monthly RunPod retest launched {len(jobs)}/{len(batches)} batch(es) across "
        f"{len(mints)} coins."
        + (f" Failures: {'; '.join(errors)}" if errors else ""),
        level="warn" if errors else "info",
        category="system",
        detail={"jobs": jobs, "errors": errors},
        conn=conn,
    )

    teardown = client.verify_teardown(jobs)
    db.kv_set(TEARDOWN_LOG_KEY, {"ts": db.now(), **teardown}, conn)
    db.log_event(
        "RunPod teardown check: "
        + ("clean - no worker or volume left running." if teardown.get("clean")
           else f"NOT CLEAN - {teardown.get('detail')}"),
        level="info" if teardown.get("clean") else "alert",
        category="system",
        detail=teardown,
        conn=conn,
    )
    return {"ran": True, "jobs": jobs, "errors": errors, "teardown": teardown}


# --------------------------------------------------------------------------
# GPU-tier benchmark (gap-closure item 7)
# --------------------------------------------------------------------------
RUNPOD_BENCHMARK_KEY = "runpod_benchmark_results"


def run_benchmark(
    cfg: dict[str, Any],
    store: Any,
    secrets: Any,
    *,
    conn: sqlite3.Connection | None = None,
    runpod_client: Any = None,
    tiers: list[str] | None = None,
    max_seconds: float | None = None,
) -> dict[str, Any]:
    """Benchmark 2-3 GPU tiers on a small representative slice and report
    cost-per-run for each - on demand (`manage.py benchmark-runpod`, or the
    dashboard button next to the RunPod settings), never automatically.

    Stores the result and a recommendation under RUNPOD_BENCHMARK_KEY for the
    settings page to show; it never changes `runpod_gpu_type` itself - the
    operator confirms that from the numbers, same as every other setting.
    """
    from .runpod import BENCHMARK_TIMEOUT_SECONDS, DEFAULT_BENCHMARK_TIERS, RunPodClient, RunPodError

    conn = conn or db.connect()
    if not secrets.runpod_api_key:
        return {"ran": False, "reason": "RUNPOD_API_KEY is not configured"}

    mints = list(store.pair_map(conn))
    if not mints:
        return {"ran": False, "reason": "no routed universe tokens to test"}

    client = runpod_client or RunPodClient(secrets.runpod_api_key)
    run_id_start = next_run_id(conn)

    db.log_event(
        f"RunPod GPU-tier benchmark started: {', '.join(tiers or DEFAULT_BENCHMARK_TIERS)}.",
        category="system", conn=conn,
    )

    try:
        results = client.benchmark_tiers(
            tiers,
            mints=mints,
            candles=store.candles,
            interval=candles_interval(),
            report_run_id_start=run_id_start,
            droplet_url=str(cfg.get("runpod_callback_url") or ""),
            droplet_token=getattr(secrets, "bulk_data_token", ""),
            s3_access_key=getattr(secrets, "runpod_s3_access_key", ""),
            s3_secret_key=getattr(secrets, "runpod_s3_secret_key", ""),
            max_seconds=BENCHMARK_TIMEOUT_SECONDS if max_seconds is None else max_seconds,
        )
    except RunPodError as exc:
        db.log_event(
            f"RunPod GPU-tier benchmark could not start: {exc}",
            level="alert", category="system", conn=conn,
        )
        return {"ran": False, "reason": str(exc)}

    recommendation = _recommend_tier(results)
    payload = {
        "ts": db.now(),
        "results": [r.as_dict() for r in results],
        "recommendation": recommendation,
    }
    db.kv_set(RUNPOD_BENCHMARK_KEY, payload, conn)

    failed = [r.gpu_type for r in results if not r.ok]
    dirty = [r.gpu_type for r in results if r.teardown_clean is False]
    db.log_event(
        "RunPod GPU-tier benchmark finished: "
        + "; ".join(
            f"{r.gpu_type} {r.elapsed_seconds:.0f}s"
            + (f" (${r.cost_usd:.4f})" if r.cost_usd is not None else " (cost unknown)")
            for r in results
        )
        + (f". Recommendation: {recommendation}." if recommendation else "")
        + (f" FAILED: {', '.join(failed)}." if failed else "")
        + (f" NOT CLEANLY TORN DOWN: {', '.join(dirty)}." if dirty else ""),
        level="alert" if (failed or dirty) else "info",
        category="system",
        detail=payload,
        conn=conn,
    )
    return {"ran": True, **payload}


def _recommend_tier(results: list[Any]) -> str:
    """Cheapest $/1000-combinations among tiers that actually completed and
    tore down cleanly - a recommendation to show, never applied automatically."""
    candidates = [
        r for r in results
        if r.ok and r.teardown_clean is not False and r.cost_per_1000_combinations is not None
    ]
    if candidates:
        best = min(candidates, key=lambda r: r.cost_per_1000_combinations)
        return (
            f"{best.gpu_type} (${best.cost_per_1000_combinations:.4f} per 1,000 combinations)"
        )
    # No combination counts made it back (e.g. the report round-trip did not
    # complete in this benchmark window) - fall back to raw $/hr among tiers
    # that at least completed cleanly, rather than recommending nothing.
    priced = [r for r in results if r.ok and r.teardown_clean is not False and r.price_per_hour]
    if priced:
        best = min(priced, key=lambda r: r.price_per_hour)
        return f"{best.gpu_type} (${best.price_per_hour:.2f}/hr - no combination count to compare cost-per-run)"
    return ""


# --------------------------------------------------------------------------
# Fuzzy-regime section, step 5: manual trigger + progress for the full
# per-coin regime discovery + per-regime walk-forward pass.
# --------------------------------------------------------------------------
REGIME_PASS_JOB = "regime_pass"


def run_regime_pass(
    cfg: dict[str, Any],
    store: Any,
    *,
    conn: sqlite3.Connection | None = None,
    wf_config_overrides: dict[str, Any] | None = None,
    k_range: tuple[int, ...] = (4, 5, 6),
    membership_threshold: float | None = None,
) -> dict[str, Any]:
    """Discover every coin's own fuzzy regimes, then walk-forward each one -
    on demand (the dashboard button, or `manage.py`), independent of the
    monthly automatic retest. Reports progress two ways at once: a percent-
    complete bar under REGIME_PASS_JOB (db.get_progress, the same mechanism
    the historical pull and daily backtest already use), and a plain-
    language feed through the existing walk-forward run/feed tables the
    dashboard's walk-forward tab already polls - "now discovering regimes
    for X", "now optimizing the trending regime for X", per step 5's own
    wording.
    """
    from solopt.dataset import load_panel
    from solopt.engine import PortfolioSettings
    from solopt.feed import RunFeed
    from solopt.frames import compact
    from solopt.regime_discovery import MIN_SAMPLES_PER_COIN, compute_features, discover_coin_regimes
    from solopt.regime_walkforward import DEFAULT_MEMBERSHIP_THRESHOLD, run_and_store_all_regimes
    from solopt.schema import CostModel, get_schema
    from solopt.store import RunStore
    from solopt.walkforward import WalkForwardConfig

    conn = conn or db.connect()
    mints = list(store.pair_map(conn))
    if not mints:
        db.set_progress(REGIME_PASS_JOB, status="failed", message="no routed universe tokens", conn=conn)
        return {"ran": False, "reason": "no routed universe tokens to test"}

    db.set_progress(
        REGIME_PASS_JOB, status="running", done=0, total=len(mints),
        message="loading candle history", conn=conn,
    )

    out_dir = materialize_bundle(store.candles, mints, out_dir=DAILY_BUNDLE_DIR, conn=conn)
    schema = get_schema("crypto")
    timeframe_seconds = int(cfg["candle_minutes"]) * 60
    panel = load_panel(out_dir, timeframe_seconds=timeframe_seconds, schema=schema)
    frames = compact(panel)

    space = focused_space(cfg, conn)
    wf_config = WalkForwardConfig(
        **{
            "in_sample_days": 45, "out_of_sample_days": 10, "step_days": 10,
            "max_evaluations": 150, "batch_size": 32, "workers": 1,
            "utilization_pct": 80.0,
            "library_seed_fraction": float(cfg.get("library_seed_fraction", 0.3)),
            **(wf_config_overrides or {}),
        }
    )
    portfolio = PortfolioSettings(
        starting_balance=float(cfg["paper_starting_balance"]),
        max_total_deployed_pct=float(cfg["max_total_deployed_pct"]),
        max_position_pct_of_wallet=float(cfg["max_position_pct_of_wallet"]),
        max_position_pct_of_liquidity=float(cfg["max_position_pct_of_liquidity"]),
        min_position_usd=float(cfg["min_position_usd"]),
        min_candles_required=int(cfg["min_candles_required"]),
        volatility_target_atr_pct=float(cfg["volatility_target_atr_pct"]),
        volatility_size_floor=float(cfg["volatility_size_floor"]),
        correlation_lookback=int(cfg["correlation_lookback"]),
        correlation_max=float(cfg["correlation_max"]),
        costs=CostModel(fee_pct=float(cfg["taker_fee_pct"]), slippage_pct=float(cfg["max_slippage_pct"])),
        confluence_multiples=tuple(cfg.get("confluence_timeframes") or ()),
        sizing_mode=str(cfg.get("sizing_mode", "portfolio")),
        portfolio_vol_target=float(cfg.get("portfolio_vol_target", 0.019)),
        drawdown_tolerance=float(cfg.get("drawdown_tolerance", 0.25)),
    )

    local_store = RunStore(DAILY_STORE_PATH)
    bundle_id = time.strftime("regime-pass-%Y-%m-%d-%H%M", time.gmtime())
    run_id = local_store.start_run(
        config=wf_config.as_dict(), settings=portfolio.as_dict(),
        space={"values": space.values}, bundle=bundle_id, coverage=panel.coverage(),
        label=bundle_id,
    )
    report_id = next_run_id(conn)
    feed = RunFeed(local_store, run_id, report_run_id=report_id, sinks=[_LocalFeedSink(conn)])
    feed.say(f"Regime pass starting: {len(mints)} coin(s) in the routed universe.")

    all_features = compute_features(frames)  # once for the whole panel, not per coin
    threshold = DEFAULT_MEMBERSHIP_THRESHOLD if membership_threshold is None else membership_threshold

    totals = {"coins_modeled": 0, "coins_skipped": 0, "regimes_accepted": 0, "regimes_tried": 0}
    total_coins = len(mints)
    for i, mint in enumerate(mints):
        label = mint[:8]
        db.set_progress(
            REGIME_PASS_JOB, status="running", done=i, total=total_coins,
            message=f"discovering regimes for {label}", conn=conn,
        )
        feed.say(f"Now discovering regimes for {label}...")

        model = discover_coin_regimes(
            frames, mint, k_range=k_range, min_samples=MIN_SAMPLES_PER_COIN, features=all_features,
        )
        if model is None:
            totals["coins_skipped"] += 1
            feed.say(f"{label}: not enough history yet - skipped.")
            continue

        totals["coins_modeled"] += 1
        local_store.save_regime_model(mint, model.as_dict(), run_id=run_id)
        feed.say(
            f"{label}: discovered {model.n_clusters} regime(s) from {model.n_samples} bars "
            f"(partition coefficient {model.fpc:.2f})."
        )

        def on_progress(cluster_id: int, n_clusters: int, *, _label: str = label, _i: int = i) -> None:
            db.set_progress(
                REGIME_PASS_JOB, status="running", done=_i, total=total_coins,
                message=f"optimizing regime {cluster_id + 1}/{n_clusters} for {_label}",
                conn=conn,
            )
            feed.say(f"Now optimizing regime {cluster_id + 1}/{n_clusters} for {_label}...")

        outcomes = run_and_store_all_regimes(
            frames, mint, model,
            space=space, portfolio=portfolio, wf_config=wf_config, store=local_store,
            threshold=threshold, run_id=run_id, on_progress=on_progress,
        )
        totals["regimes_tried"] += len(outcomes)
        accepted = sum(1 for o in outcomes.values() if o.accepted)
        totals["regimes_accepted"] += accepted
        feed.say(
            f"{label}: {accepted} of {len(outcomes)} tested regime(s) accepted and promoted."
            if outcomes else f"{label}: no regime had enough dominated windows to search yet."
        )

    db.set_progress(
        REGIME_PASS_JOB, status="done", done=total_coins, total=total_coins,
        message="finished", conn=conn,
    )
    feed.say(
        f"Regime pass finished: {totals['coins_modeled']} coin(s) modeled "
        f"({totals['coins_skipped']} skipped for too little history), "
        f"{totals['regimes_accepted']} of {totals['regimes_tried']} tested regime(s) promoted."
    )
    local_store.finish_run(run_id, "done", totals)
    return {"ran": True, "run_id": report_id, **totals}
