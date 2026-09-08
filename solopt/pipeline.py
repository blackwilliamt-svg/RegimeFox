"""The reusable walk-forward -> Monte Carlo -> stress -> bundle pipeline.

This is the core spec 5 asks to be invoked from two places rather than one:
locally on the droplet's CPU for the daily incremental run
(:func:`solbot.wfmc.run_daily`), and inside a RunPod GPU worker for the
monthly full retest (:func:`solopt.cli.cmd_run`). Neither caller needs to
know how a window is searched or how the Monte Carlo tail is computed - they
differ only in where the data comes from, which backend runs it, where the
feed lines go, and what happens with the finished bundle.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Any

from .arrays import Backend, Throttle, get_backend
from .engine import PortfolioSettings, VectorEngine
from .feed import RunFeed
from .frames import Frames
from .montecarlo import ExecutionProfile, MonteCarloResult, from_walk_forward
from .params import ParamSpace
from .promotion import ParameterBundle, build_bundle, promotable
from .stress import StressConfig, StressResult, find_windows, replay
from .walkforward import WalkForward, WalkForwardConfig, WalkForwardResult, derive_warmup

log = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    outcome: WalkForwardResult
    monte_carlo: MonteCarloResult
    stress: list[StressResult]
    bundle: ParameterBundle
    summary: dict[str, Any]
    accepted: bool
    verdict: str


def run_pipeline(
    frames: Frames,
    *,
    space: ParamSpace,
    portfolio: PortfolioSettings,
    wf_config: WalkForwardConfig,
    backend: Backend | None = None,
    feed: RunFeed,
    store: Any,
    run_id: int,
    resume: bool = True,
    coverage: dict[str, Any],
    execution: ExecutionProfile | None = None,
    monte_carlo_iterations: int = 5000,
    stress_config: StressConfig | None = None,
    run_meta: dict[str, Any] | None = None,
) -> PipelineResult:
    """Walk-forward, then Monte Carlo, then crash replay, then a bundle.

    Runs and persists window-by-window through ``store``/``feed`` exactly as
    a standalone CLI run would, so a caller embedding this (the daily droplet
    run) gets identical resumability and narration for free.
    """
    backend = backend or get_backend()

    walker = WalkForward(
        space, portfolio, wf_config, backend=backend, feed=feed, store=store, run_id=run_id
    )
    outcome = walker.run(frames, resume=resume)

    execution = execution or ExecutionProfile.assumed(
        portfolio.slippage_pct / 2.0, portfolio.fee_pct
    )
    monte = from_walk_forward(
        outcome,
        starting_balance=portfolio.starting_balance,
        execution=execution,
        assumed_round_trip_pct=portfolio.costs.round_trip_pct(),
        iterations=monte_carlo_iterations,
    )
    if monte.iterations:
        feed.monte_carlo(monte.summary())
        store.save_monte_carlo(run_id, monte.iterations, monte.summary(), monte.histograms())
    else:
        feed.warn("No out-of-sample trades to resample; Monte Carlo skipped.")

    stress_results: list[StressResult] = []
    if outcome.best_params:
        stress_cfg = stress_config or StressConfig()
        windows = find_windows(frames, stress_cfg)
        if windows:
            feed.say(
                "Replaying the worst stretches in the data: "
                + ", ".join(w.label() for w in windows) + "."
            )
            engine = VectorEngine(
                backend, Throttle(wf_config.utilization_pct, backend),
                memory_budget_mb=wf_config.memory_budget_mb,
            )
            sized = replace(portfolio, drawdown_p5=monte.p5_max_drawdown)
            stress_results = replay(
                frames, outcome.best_params, sized, engine, windows,
                warmup_bars=wf_config.warmup_bars or derive_warmup(space, portfolio),
                config=stress_cfg,
            )
            store.save_stress(run_id, stress_results)
            failures = [r.name for r in stress_results if not r.passed]
            outcome.fragile = bool(failures)
            outcome.fragile_windows = failures
            feed.fragile(failures)
        else:
            feed.say("No window in this history was severe enough to use as a crash test.")

    bundle = build_bundle(
        outcome,
        monte_carlo=monte if monte.iterations else None,
        stress=stress_results,
        coverage=coverage,
        run_meta={"run_id": run_id, **(run_meta or {})},
        base_vol_target=portfolio.portfolio_vol_target,
    )
    ok, why = promotable(bundle)
    summary = {
        **outcome.summary(), "monte_carlo": monte.summary(),
        "promotable": ok, "verdict": why, "fingerprint": bundle.fingerprint(),
    }
    return PipelineResult(
        outcome=outcome, monte_carlo=monte, stress=stress_results,
        bundle=bundle, summary=summary, accepted=ok, verdict=why,
    )
