"""Non-standard candle-timeframe search: solopt.pipeline.run_pipeline_over_timeframes.

The base candle a strategy trades is not a ``ParamSpace`` axis - it selects
which panel gets built, not a value read out of one already built - so it is
searched as an outer loop over whole ``run_pipeline`` calls. These tests
exercise that orchestration directly (which timeframe wins, how a promoted
winner's candle_minutes gets wired into its bundle, what happens when a
candidate's history does not reach that far back) rather than re-running the
walk-forward search itself, which is already covered in test_optimizer.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

from solopt.frames import Frames, frames_from_series
from solopt.promotion import ParameterBundle
from solopt.schema import CRYPTO
from solopt.walkforward import WalkForwardResult


class _NullFeed:
    def say(self, message: str, *, level: str = "info", detail: Any = None) -> None:
        pass


def _empty_frames(seconds: int) -> Frames:
    return frames_from_series({}, seconds=seconds, schema=CRYPTO)


def _tiny_frames(seconds: int, bars: int = 10) -> Frames:
    rng = np.random.default_rng(1)
    close = 100 + np.cumsum(rng.normal(0, 0.1, bars))
    row = np.vstack(
        [np.arange(bars) * seconds, close, close, close, close, np.full(bars, 1000.0)]
    )
    return frames_from_series({"AAA": row}, seconds=seconds, schema=CRYPTO)


def _fake_result(
    *, accepted: bool, oos_return: float, efficiency: float,
    best_params: dict[str, Any] | None = None,
) -> "PipelineResult":
    from solopt.pipeline import PipelineResult

    params = dict(best_params or {"ema_fast": 9, "ema_slow": 21})
    outcome = WalkForwardResult(
        accepted=accepted,
        aggregate_oos_return=oos_return,
        walk_forward_efficiency=efficiency,
        best_params=dict(params),
    )
    bundle = ParameterBundle(global_params=dict(params))
    summary = {**outcome.summary(), "monte_carlo": {}, "promotable": accepted, "verdict": ""}
    return PipelineResult(
        outcome=outcome, monte_carlo=None, stress=[], bundle=bundle,
        summary=summary, accepted=accepted, verdict="",
    )


@dataclass
class _FakeRunPipeline:
    """Stands in for solopt.pipeline.run_pipeline: returns a canned result
    keyed by the candle_minutes the caller asked to search, so the test
    verifies the orchestration (which candidate wins, how it is patched)
    without re-running a real walk-forward search per candidate."""

    by_minutes: dict[int, "PipelineResult"]
    calls: list[int] = field(default_factory=list)

    def __call__(self, frames, *, run_meta, **kwargs):
        minutes = run_meta["candle_minutes"]
        self.calls.append(minutes)
        return self.by_minutes[minutes]


def test_picks_the_accepted_candidate_over_a_higher_return_rejected_one(monkeypatch):
    import solopt.pipeline as pipeline

    fake = _FakeRunPipeline(
        by_minutes={
            7: _fake_result(accepted=False, oos_return=0.50, efficiency=0.9),
            10: _fake_result(accepted=True, oos_return=0.10, efficiency=0.5),
        }
    )
    monkeypatch.setattr(pipeline, "run_pipeline", fake)

    winner, result, results = pipeline.run_pipeline_over_timeframes(
        lambda minutes: _tiny_frames(minutes * 60),
        (7, 10),
        run_id_for=lambda m: m,
        feed=_NullFeed(),
    )

    assert winner == 10
    assert result.accepted
    assert set(results) == {7, 10}
    assert sorted(fake.calls) == [7, 10]


def test_among_accepted_candidates_the_larger_oos_return_wins(monkeypatch):
    import solopt.pipeline as pipeline

    fake = _FakeRunPipeline(
        by_minutes={
            7: _fake_result(accepted=True, oos_return=0.08, efficiency=0.9),
            13: _fake_result(accepted=True, oos_return=0.22, efficiency=0.3),
            20: _fake_result(accepted=True, oos_return=0.15, efficiency=0.6),
        }
    )
    monkeypatch.setattr(pipeline, "run_pipeline", fake)

    winner, result, _results = pipeline.run_pipeline_over_timeframes(
        lambda minutes: _tiny_frames(minutes * 60),
        (7, 13, 20),
        run_id_for=lambda m: m,
        feed=_NullFeed(),
    )

    assert winner == 13
    assert result.outcome.aggregate_oos_return == pytest.approx(0.22)


def test_the_winning_candle_minutes_is_written_into_the_promoted_bundle(monkeypatch):
    import solopt.pipeline as pipeline

    fake = _FakeRunPipeline(
        by_minutes={
            10: _fake_result(
                accepted=True, oos_return=0.10, efficiency=0.5,
                best_params={"ema_fast": 9, "ema_slow": 21},
            ),
        }
    )
    monkeypatch.setattr(pipeline, "run_pipeline", fake)

    winner, result, _results = pipeline.run_pipeline_over_timeframes(
        lambda minutes: _tiny_frames(minutes * 60),
        (10,),
        run_id_for=lambda m: m,
        feed=_NullFeed(),
    )

    # Global params (what a bundle's "global" section carries downstream to
    # solbot.paramsync.check_bundle) must carry the winning timeframe, not
    # just the run's own summary metadata - that is what actually switches
    # the live candle timeframe on promotion.
    assert result.bundle.global_params["candle_minutes"] == 10
    assert result.outcome.best_params["candle_minutes"] == 10
    assert result.summary["best_params"]["candle_minutes"] == 10
    # And the rest of the winning combo's own parameters must survive intact.
    assert result.bundle.global_params["ema_fast"] == 9


def test_a_candidate_with_no_history_at_that_timeframe_is_skipped_not_failed(monkeypatch):
    import solopt.pipeline as pipeline

    fake = _FakeRunPipeline(
        by_minutes={10: _fake_result(accepted=True, oos_return=0.05, efficiency=0.4)}
    )
    monkeypatch.setattr(pipeline, "run_pipeline", fake)

    def build(minutes: int) -> Frames:
        return _empty_frames(minutes * 60) if minutes == 20 else _tiny_frames(minutes * 60)

    winner, _result, results = pipeline.run_pipeline_over_timeframes(
        build, (10, 20), run_id_for=lambda m: m, feed=_NullFeed(),
    )

    assert winner == 10
    assert set(results) == {10}   # the 20-minute candidate never even reached run_pipeline
    assert fake.calls == [10]


def test_every_candidate_empty_raises_rather_than_silently_returning_nothing():
    import solopt.pipeline as pipeline

    with pytest.raises(ValueError):
        pipeline.run_pipeline_over_timeframes(
            lambda minutes: _empty_frames(minutes * 60),
            (7, 10, 13),
            run_id_for=lambda m: m,
            feed=_NullFeed(),
        )


def test_run_id_for_and_run_meta_are_threaded_per_candidate(monkeypatch):
    import solopt.pipeline as pipeline

    seen_run_ids: list[int] = []
    seen_run_meta: list[dict[str, Any]] = []

    def fake_run_pipeline(frames, *, run_id, run_meta, **kwargs):
        seen_run_ids.append(run_id)
        seen_run_meta.append(dict(run_meta))
        return _fake_result(accepted=True, oos_return=0.01, efficiency=0.1)

    monkeypatch.setattr(pipeline, "run_pipeline", fake_run_pipeline)

    pipeline.run_pipeline_over_timeframes(
        lambda minutes: _tiny_frames(minutes * 60),
        (7, 10),
        run_id_for=lambda m: 1000 + m,
        feed=_NullFeed(),
        run_meta={"bundle": "monthly-2026-09"},
    )

    assert sorted(seen_run_ids) == [1007, 1010]
    for meta in seen_run_meta:
        assert meta["bundle"] == "monthly-2026-09"
        assert meta["candle_minutes"] in (7, 10)


def test_default_candidates_include_the_non_standard_intervals():
    from solopt.pipeline import TIMEFRAME_SEARCH_CANDIDATES

    for minutes in (7, 13, 20):
        assert minutes in TIMEFRAME_SEARCH_CANDIDATES


def test_candle_minutes_spec_bound_admits_every_search_candidate():
    """A winning 20-minute set must actually be promotable - if the live
    bot's own bounds rejected it, searching for it would be a dead end."""
    from solbot.config import SPEC
    from solopt.pipeline import TIMEFRAME_SEARCH_CANDIDATES

    _typ, low, high = SPEC["candle_minutes"]
    for minutes in TIMEFRAME_SEARCH_CANDIDATES:
        assert low <= minutes <= high
