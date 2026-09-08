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
from typing import Any, Callable, Sequence

import numpy as np

from .arrays import Backend, Throttle, get_backend
from .engine import PortfolioSettings, VectorEngine
from .feed import RunFeed
from .frames import Frames
from .indicators import efficiency_ratio_stack
from .montecarlo import ExecutionProfile, MonteCarloResult, from_walk_forward
from .params import ParamSpace
from .promotion import ParameterBundle, build_bundle, promotable
from .store import LibraryEntry
from .stress import StressConfig, StressResult, find_windows, replay
from .walkforward import WalkForward, WalkForwardConfig, WalkForwardResult, derive_warmup

log = logging.getLogger(__name__)


def _regime_scores(frames: Frames, lookback: int) -> tuple[float | None, dict[str, float]]:
    """Continuous regime tags at the end of the panel (gap-closure item 5):
    the market-wide efficiency ratio averaged across every symbol, and each
    symbol's own. Reuses efficiency_ratio_stack rather than a separate
    measure - the same signal the regime gate itself reads, just captured at
    validation time instead of live."""
    if frames.n_bars == 0 or frames.n_symbols == 0:
        return None, {}
    er = efficiency_ratio_stack(frames.close, [max(2, int(lookback))])[0]  # [symbols, bars]
    last = er[:, -1]
    market = float(np.nanmean(last)) if np.isfinite(last).any() else None
    per_symbol = {
        sym: float(last[i])
        for i, sym in enumerate(frames.symbols)
        if np.isfinite(last[i])
    }
    return market, per_symbol


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
    if ok:
        _update_library(store, bundle, outcome, frames, run_id)
    return PipelineResult(
        outcome=outcome, monte_carlo=monte, stress=stress_results,
        bundle=bundle, summary=summary, accepted=ok, verdict=why,
    )


# --------------------------------------------------------------------------
# Timeframe search (spec: non-standard candle-timeframe search)
# --------------------------------------------------------------------------
# The base candle a strategy trades is itself a parameter, but not one
# ``ParamSpace``/``TUNABLE`` can hold: it changes which *panel* gets loaded
# (a different rollup of the 1-minute base, per ``solopt.dataset.rollup``),
# not a value read out of an already-built one. So it is searched as an outer
# loop over whole ``run_pipeline`` calls rather than an axis inside one -
# every candidate gets its own full walk-forward/Monte Carlo/stress
# evaluation on its own panel, and the results are compared on equal terms.
# Deliberately not limited to the "standard" 5/10/15-minute candles a human
# would reach for first: 7 and 13 land off any moving-average's usual period,
# and 20 sits just past the live bot's previous ceiling (widened alongside
# this to admit a winning 20-minute set - see solbot/config.py's SPEC).
TIMEFRAME_SEARCH_CANDIDATES: tuple[int, ...] = (7, 10, 13, 15, 20)


def _timeframe_rank(item: tuple[int, "PipelineResult"]) -> tuple[int, float, float]:
    """Higher is better. Accepted runs always outrank rejected ones; among
    ties, the out-of-sample return actually earned (not a ratio, which a
    thinly-traded high timeframe can post a flattering one of on noise)
    breaks it, then walk-forward efficiency as a second tiebreaker."""
    _, result = item
    outcome = result.outcome
    return (
        1 if outcome.accepted else 0,
        float(outcome.aggregate_oos_return),
        float(outcome.walk_forward_efficiency),
    )


def run_pipeline_over_timeframes(
    build_frames: Callable[[int], Frames],
    timeframe_minutes: Sequence[int] = TIMEFRAME_SEARCH_CANDIDATES,
    *,
    run_id_for: Callable[[int], int],
    feed: RunFeed,
    **run_pipeline_kwargs: Any,
) -> tuple[int, PipelineResult, dict[int, PipelineResult]]:
    """Run the full pipeline once per candidate timeframe and keep the best.

    ``build_frames(minutes)`` loads that candidate's panel (a caller-supplied
    closure so this module stays ignorant of where candles come from - the
    droplet's Parquet store, a RunPod data bundle, or a test fixture).
    ``run_id_for(minutes)`` mints this candidate's own run id, since each is
    tracked (and independently resumable) in ``store`` exactly like a
    standalone run. Every other keyword is forwarded to :func:`run_pipeline`
    unchanged for every candidate (``space``, ``portfolio``, ``wf_config``,
    ``store``, ...); pass neither ``run_id``, ``run_meta`` nor ``coverage``
    here - ``run_id``/``run_meta`` are supplied per-candidate, and
    ``coverage`` is derived from each candidate's own ``Frames.coverage()``
    (a 7-minute panel and a 20-minute panel of the same history do not share
    one bar count or day span, so one caller-supplied value could not
    describe both).

    A candidate whose panel comes back empty (too little history rolled up
    to that timeframe) is skipped rather than failing the whole search. The
    winner's ``candle_minutes`` is written into its own ``best_params`` (and
    therefore its bundle's ``global`` params) so promoting the winning set
    actually switches the live candle timeframe, not just its indicator
    settings - the same ``EDITABLE``/``SPEC`` bounds check every other
    promoted parameter goes through on the droplet side.
    """
    base_run_meta = dict(run_pipeline_kwargs.pop("run_meta", None) or {})
    run_pipeline_kwargs.pop("coverage", None)
    results: dict[int, PipelineResult] = {}
    for minutes in timeframe_minutes:
        frames = build_frames(int(minutes))
        if frames.n_bars == 0 or frames.n_symbols == 0:
            feed.say(f"Skipping the {minutes}-minute candle: no history rolls up to it.")
            continue
        feed.say(f"Searching the {minutes}-minute candle timeframe.")
        result = run_pipeline(
            frames,
            feed=feed,
            run_id=run_id_for(int(minutes)),
            coverage=frames.coverage(),
            run_meta={"candle_minutes": int(minutes), **base_run_meta},
            **run_pipeline_kwargs,
        )
        results[int(minutes)] = result

    if not results:
        raise ValueError("no candidate timeframe produced a usable panel")

    best_minutes, best_result = max(results.items(), key=_timeframe_rank)
    if best_result.outcome.best_params is not None:
        best_result.outcome.best_params = {
            **best_result.outcome.best_params, "candle_minutes": best_minutes,
        }
        best_result.bundle.global_params = dict(best_result.outcome.best_params)
        best_result.summary["best_params"] = best_result.outcome.best_params
    feed.say(
        f"Timeframe search: {best_minutes}-minute candles won "
        f"({', '.join(f'{m}m={results[m].outcome.aggregate_oos_return:.4f}' for m in sorted(results))})."
    )
    return best_minutes, best_result, results


def _update_library(
    store: Any, bundle: ParameterBundle, outcome: WalkForwardResult, frames: Frames, run_id: int
) -> None:
    """Persist this run's winning combination(s) (gap-closure item 4) - a
    market-wide entry always, plus one per-symbol entry per coin the
    walk-forward found its own parameters for, each tagged with both the
    market-wide and that coin's own regime score at validation time
    (gap-closure item 5). `store` predating the library (a bare test double)
    simply does not get one - upserting is best-effort, never fatal to the
    run that produced the evidence.
    """
    upsert = getattr(store, "upsert_library", None)
    if not callable(upsert) or not outcome.best_params:
        return
    try:
        lookback = int(outcome.best_params.get("regime_lookback", 20))
        market_score, per_symbol_scores = _regime_scores(frames, lookback)
        performance = outcome.summary()

        upsert(
            LibraryEntry(
                fingerprint=bundle.fingerprint(),
                params=dict(outcome.best_params),
                performance=performance,
                market_regime_score=market_score,
                run_id=run_id,
            )
        )
        for symbol, overrides in outcome.per_symbol_params.items():
            effective = {**outcome.best_params, **overrides}
            fp = ParameterBundle(global_params=effective).fingerprint()
            upsert(
                LibraryEntry(
                    fingerprint=fp,
                    params=effective,
                    per_symbol=dict(overrides),
                    performance=performance,
                    symbol=symbol,
                    market_regime_score=market_score,
                    symbol_regime_score=per_symbol_scores.get(symbol),
                    run_id=run_id,
                )
            )
    except Exception:
        log.debug("library update failed", exc_info=True)
