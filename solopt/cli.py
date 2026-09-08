"""Command line for the optimizer - the RunPod monthly-retest worker's entry point.

    python -m solopt run                 walk-forward, Monte Carlo, stress, report
    python -m solopt status              what the last run concluded
    python -m solopt feed --follow       tail the plain-language run feed

The daily on-droplet run does not go through this CLI at all - it calls
:mod:`solopt.pipeline` directly, in-process, since it already has local access
to the droplet's database and Parquet files (see :mod:`solbot.wfmc`). This
entry point exists for the one case that genuinely runs somewhere else: a
RunPod GPU worker, which gets its data shipped to it before ``run`` starts and
reports back to the droplet over HTTP as it goes.

``run`` is the whole pipeline in one command because the stages are not
independently useful: a walk-forward result without the Monte Carlo tail has no
number to size with, and neither is safe to report without the crash replay.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from .arrays import get_backend
from .dataset import bundle_fingerprint, load_panel
from .engine import PortfolioSettings
from .feed import BufferedSink, RunFeed
from .frames import compact
from .montecarlo import ExecutionProfile
from .params import DEFAULT_GRID, ParamSpace
from .pipeline import run_pipeline
from .report import DropletClient, ReportError
from .schema import CostModel, get_schema
from .store import RunStore
from .stress import StressConfig
from .walkforward import WalkForwardConfig

log = logging.getLogger("solopt")

DEFAULT_CONFIG = "solopt.json"


@dataclass
class Settings:
    """Everything the optimizer needs, from one JSON file plus the environment."""

    droplet_url: str = ""
    droplet_token: str = ""
    data_dir: str = "data/bundle"
    store_path: str = "data/solopt.db"
    # A globally-unique id for the droplet's dashboard to track this run by,
    # distinct from this worker's own local run id (see solopt.feed.RunFeed).
    # 0 means "use the local id" - fine for a single, long-lived optimizer,
    # not for several concurrent RunPod batch workers each with a fresh store.
    report_run_id: int = 0

    asset_class: str = "crypto"
    base_interval: str = "1m"
    timeframe_minutes: int = 10

    utilization_pct: float = 100.0
    memory_budget_mb: int = 512
    workers: int = 0
    prefer_gpu: bool = True

    grid: dict[str, list[Any]] = field(default_factory=lambda: dict(DEFAULT_GRID))
    walk_forward: dict[str, Any] = field(default_factory=dict)
    portfolio: dict[str, Any] = field(default_factory=dict)
    stress: dict[str, Any] = field(default_factory=dict)
    monte_carlo_iterations: int = 5000
    push_feed: bool = True

    @classmethod
    def load(cls, path: str | Path | None) -> "Settings":
        data: dict[str, Any] = {}
        target = Path(path or os.getenv("SOLOPT_CONFIG", DEFAULT_CONFIG))
        if target.exists():
            data = json.loads(target.read_text(encoding="utf-8"))
        elif path:
            raise SystemExit(f"config file {target} does not exist")

        known = {f for f in cls.__dataclass_fields__}
        settings = cls(**{k: v for k, v in data.items() if k in known})
        # The token is a credential; the environment wins so it need not be
        # written into a file that might end up in a container image.
        settings.droplet_url = os.getenv("SOLOPT_DROPLET_URL", settings.droplet_url)
        settings.droplet_token = os.getenv("SOLOPT_TOKEN", settings.droplet_token)
        if os.getenv("SOLOPT_UTILIZATION"):
            settings.utilization_pct = float(os.environ["SOLOPT_UTILIZATION"])
        if os.getenv("SOLOPT_REPORT_RUN_ID"):
            settings.report_run_id = int(os.environ["SOLOPT_REPORT_RUN_ID"])
        return settings

    def client(self) -> DropletClient:
        if not self.droplet_url:
            raise SystemExit(
                "no droplet URL configured. Set droplet_url in the config file or "
                "SOLOPT_DROPLET_URL in the environment."
            )
        return DropletClient(self.droplet_url, self.droplet_token)

    def portfolio_settings(self, drawdown_p5: float = 0.0) -> PortfolioSettings:
        schema = get_schema(self.asset_class)
        overrides = dict(self.portfolio)
        costs = CostModel(
            fee_pct=float(overrides.pop("fee_pct", schema.costs.fee_pct)),
            slippage_pct=float(overrides.pop("slippage_pct", schema.costs.slippage_pct)),
        )
        confluence = tuple(overrides.pop("confluence_multiples", (3, 6)))
        known = {f for f in PortfolioSettings.__dataclass_fields__}
        return PortfolioSettings(
            costs=costs,
            confluence_multiples=confluence,
            drawdown_p5=drawdown_p5,
            **{k: v for k, v in overrides.items() if k in known},
        )

    def wf_config(self) -> WalkForwardConfig:
        return WalkForwardConfig.from_dict(
            {
                "utilization_pct": self.utilization_pct,
                "memory_budget_mb": self.memory_budget_mb,
                "workers": self.workers,
                **self.walk_forward,
            }
        )

    def space(self) -> ParamSpace:
        return ParamSpace(values={k: list(v) for k, v in self.grid.items()})


# --------------------------------------------------------------------------
def cmd_run(args: argparse.Namespace, settings: Settings) -> int:
    directory = Path(args.data_dir or settings.data_dir)
    schema = get_schema(settings.asset_class)
    timeframe = int(args.timeframe or settings.timeframe_minutes) * 60

    store = RunStore(settings.store_path)
    bundle_id = bundle_fingerprint(directory)

    space = settings.space()
    wf_config = settings.wf_config()
    portfolio = settings.portfolio_settings()

    existing = store.resumable_run(bundle_id) if not args.fresh else None
    if existing:
        run_id = int(existing["id"])
        print(f"Resuming run {run_id} against the same data bundle.")
    else:
        run_id = 0

    sinks = []
    client = None
    if settings.push_feed and settings.droplet_url:
        try:
            client = settings.client()
            sinks.append(BufferedSink(send=lambda rid, lines: client.push_feed(rid, lines)))
        except SystemExit:
            client = None

    print(f"Loading {directory} at {timeframe // 60}-minute candles…")
    panel = load_panel(
        directory,
        timeframe_seconds=timeframe,
        schema=schema,
        max_symbols=args.max_symbols,
        since=int(time.time()) - args.days * 86400 if args.days else None,
    )
    frames = compact(panel)
    coverage = panel.coverage()

    if not run_id:
        run_id = store.start_run(
            config=wf_config.as_dict(),
            settings=portfolio.as_dict(),
            space={"values": space.values},
            bundle=bundle_id,
            coverage=coverage,
            label=args.label or "",
        )
    feed = RunFeed(store, run_id, report_run_id=settings.report_run_id, sinks=sinks)
    feed.loaded(coverage)

    backend = get_backend(prefer_gpu=settings.prefer_gpu and not args.cpu)
    feed.say(
        f"Compute backend: {backend.name} on {backend.device}, "
        f"capped at {settings.utilization_pct:.0f}% utilization."
    )

    execution = _execution_profile(client, portfolio)
    stress_config = StressConfig(**settings.stress) if settings.stress else StressConfig()

    try:
        result = run_pipeline(
            frames,
            space=space, portfolio=portfolio, wf_config=wf_config, backend=backend,
            feed=feed, store=store, run_id=run_id, resume=not args.fresh,
            coverage=coverage, execution=execution,
            monte_carlo_iterations=settings.monte_carlo_iterations,
            stress_config=stress_config,
            run_meta={"bundle": bundle_id, "elapsed_minutes": 0.0},
        )
    except Exception as exc:
        feed.alert(f"Run failed: {exc}")
        store.finish_run(run_id, "failed", {"error": str(exc)[:500]})
        feed.flush()
        raise

    summary = dict(result.summary)
    summary["accepted"] = result.accepted
    summary["windows_detail"] = _window_rows(store, run_id)
    feed.finished(summary)

    if result.accepted and not args.no_report and client:
        try:
            response = client.push_bundle(result.bundle.as_dict())
            feed.say(
                f"Parameter set {result.bundle.fingerprint()} reported to the droplet: "
                f"{response.get('reason', 'accepted') if isinstance(response, dict) else 'sent'}."
            )
            summary["reported"] = True
        except ReportError as exc:
            feed.alert(f"Could not report the bundle to the droplet: {exc}")
            summary["report_error"] = str(exc)[:300]
    elif result.accepted and not client:
        feed.warn(
            "No droplet configured to report to; the bundle was not delivered anywhere."
        )
    elif not result.accepted:
        feed.warn(f"Nothing reported: {result.verdict}.")

    store.finish_run(run_id, "done", summary)
    if client:
        try:
            client.push_run(
                feed.report_run_id,
                {
                    "status": "done",
                    "label": args.label or "",
                    "coverage": coverage,
                    "summary": summary,
                    "monte_carlo": {
                        **result.monte_carlo.summary(), **result.monte_carlo.histograms(),
                        "drawdown_histogram": result.monte_carlo.drawdown_histogram,
                        **result.monte_carlo.worst_paths(),
                    },
                    "stress": [r.as_dict() for r in result.stress],
                },
            )
        except Exception as exc:
            log.warning("could not push the run summary: %s", exc)
    feed.flush()

    _print_verdict(summary, result.monte_carlo, result.stress)
    return 0 if result.accepted else 2


def _print_verdict(
    summary: dict[str, Any], monte: Any, stress: Sequence[Any]
) -> None:
    """A readable close-out. The full record is in the store and the bundle.

    Dumping the summary as JSON meant truncating it mid-object, which is worse
    than not printing it: `solopt status` and `solopt feed` exist for the detail.
    """
    def pct(value: Any, digits: int = 1) -> str:
        try:
            return f"{float(value) * 100:+.{digits}f}%"
        except (TypeError, ValueError):
            return "n/a"

    line = "─" * 62
    print(f"\n{line}")
    print(f"  Walk-forward run {summary.get('fingerprint', '')}")
    print(line)
    print(
        f"  windows        {summary.get('profitable_windows', 0)}/"
        f"{summary.get('counted_windows', 0)} profitable out of sample"
    )
    print(f"  efficiency     {summary.get('walk_forward_efficiency', 0):.2f}")
    print(
        f"  window return  {pct(summary.get('mean_window_return'), 2)} mean, "
        f"{pct(summary.get('stdev_window_return'), 2)} std dev"
    )
    print(f"  trades         {summary.get('trades', 0)} out of sample")
    if monte.iterations:
        print(
            f"  monte carlo    median {pct(monte.median_return)}, "
            f"5% worst {pct(monte.p5_return)} at a "
            f"{monte.p5_max_drawdown * 100:.1f}% drawdown "
            f"({monte.execution.get('source', 'assumed')} costs)"
        )
    failed = [r.name for r in stress if not r.passed]
    if stress:
        print(
            f"  crash replay   {len(stress) - len(failed)}/{len(stress)} survived"
            + (f" — FAILED {', '.join(failed)}" if failed else "")
        )

    flags = []
    if summary.get("overfit"):
        flags.append("OVERFIT")
    if summary.get("fragile"):
        flags.append("FRAGILE")
    if flags:
        print(f"  flags          {', '.join(flags)}")

    print(line)
    if summary.get("accepted"):
        print(f"  PROMOTABLE — {summary.get('verdict', '')}")
        if summary.get("reported"):
            print("  reported to the droplet")
    else:
        print(f"  NOT PROMOTABLE — {summary.get('verdict', '')}")
    print(f"{line}\n  solopt status   for the full record")
    print("  solopt feed     for what it found along the way\n")


def _window_rows(store: RunStore, run_id: int) -> list[dict[str, Any]]:
    """Per-window rows for the dashboard's pass/fail table."""
    def day(ts: Any) -> str:
        return time.strftime("%Y-%m-%d", time.gmtime(int(ts or 0)))

    rows = []
    for window in store.windows(run_id):
        rows.append(
            {
                "index": window["idx"],
                "is_label": f"{day(window['is_from'])} → {day(window['is_to'])}",
                "oos_label": f"{day(window['oos_from'])} → {day(window['oos_to'])}",
                "status": window["status"],
                "counted": bool(window["counted"]),
                "profitable": bool(window["profitable"]),
                "note": window["note"],
                "oos_metrics": window["oos_metrics"] or {},
            }
        )
    return rows


def _execution_profile(
    client: DropletClient | None, portfolio: PortfolioSettings
) -> ExecutionProfile:
    """Observed execution from the droplet, or the configured constants.

    Falling back is fine; pretending the fallback is measured is not, which is
    why the profile carries its own ``source``.
    """
    assumed = ExecutionProfile.assumed(
        portfolio.slippage_pct / 2.0, portfolio.fee_pct
    )
    if client is None:
        return assumed
    try:
        payload = client.execution_profile()
    except Exception as exc:
        log.info("using assumed execution costs (%s)", exc)
        return assumed
    profile = ExecutionProfile.from_dict(payload or {})
    return profile if profile.source == "observed" else assumed


def cmd_status(args: argparse.Namespace, settings: Settings) -> int:
    store = RunStore(settings.store_path)
    run = store.get_run(args.run) if args.run else store.latest_run()
    if not run:
        print("No runs recorded yet.")
        return 1
    windows = store.windows(int(run["id"]))
    monte = store.monte_carlo(int(run["id"]))
    print(json.dumps(
        {
            "run": {k: run[k] for k in ("id", "status", "started_at", "finished_at", "label")},
            "coverage": run.get("coverage"),
            "summary": run.get("summary"),
            "windows": [
                {
                    "idx": w["idx"], "status": w["status"], "counted": bool(w["counted"]),
                    "profitable": bool(w["profitable"]),
                    "oos_return": (w["oos_metrics"] or {}).get("total_return"),
                    "oos_trades": (w["oos_metrics"] or {}).get("trades"),
                    "note": w["note"],
                }
                for w in windows
            ],
            "monte_carlo": (monte or {}).get("summary"),
            "stress": store.stress(int(run["id"])),
        },
        indent=2,
        default=float,
    ))
    return 0


def cmd_feed(args: argparse.Namespace, settings: Settings) -> int:
    store = RunStore(settings.store_path)
    run = store.get_run(args.run) if args.run else store.latest_run()
    if not run:
        print("No runs recorded yet.")
        return 1
    run_id = int(run["id"])
    since = 0
    while True:
        for line in store.feed(run_id, since_id=since, limit=500):
            since = max(since, int(line["id"]))
            stamp = time.strftime("%H:%M:%S", time.localtime(line["ts"]))
            marker = {"warn": "!", "alert": "!!"}.get(line["level"], " ")
            print(f"{stamp} {marker:<2} {line['message']}")
        if not args.follow:
            return 0
        run = store.get_run(run_id) or run
        if run["status"] != "running":
            return 0
        time.sleep(2.0)


# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="solopt", description=__doc__)
    parser.add_argument("--config", help=f"config file (default {DEFAULT_CONFIG})")
    parser.add_argument("--verbose", "-v", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="walk-forward, Monte Carlo, crash replay, report")
    run.add_argument("--data-dir")
    run.add_argument("--timeframe", type=int, help="candle minutes to test")
    run.add_argument("--days", type=int, help="only use the last N days of history")
    run.add_argument("--max-symbols", type=int)
    run.add_argument("--label", help="a name for this run")
    run.add_argument("--fresh", action="store_true", help="ignore any checkpoint")
    run.add_argument("--cpu", action="store_true", help="force the NumPy backend")
    run.add_argument("--no-report", action="store_true", help="do not report the bundle")
    run.set_defaults(func=cmd_run)

    status = sub.add_parser("status", help="what the last run concluded")
    status.add_argument("--run", type=int)
    status.set_defaults(func=cmd_status)

    feed = sub.add_parser("feed", help="tail the plain-language run feed")
    feed.add_argument("--run", type=int)
    feed.add_argument("--follow", "-f", action="store_true")
    feed.set_defaults(func=cmd_feed)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    settings = Settings.load(args.config)
    return int(args.func(args, settings) or 0)
