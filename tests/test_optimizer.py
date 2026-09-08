"""Walk-forward thresholds, the search's stopping rules, Monte Carlo and stress.

These cover the parts of the optimizer that decide whether a parameter set is
allowed to exist: the trade floor, the 70% consistency bar, the overfit flag,
the plateau stop, and the resampled tail that ends up sizing positions.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from solopt.engine import PortfolioSettings
from solopt.frames import frames_from_series
from solopt.montecarlo import ExecutionProfile, simulate
from solopt.params import ParamSpace, Search, is_valid
from solopt.schema import CRYPTO
from solopt.stress import StressConfig, _rolling_max, find_windows, market_index
from solopt.walkforward import (
    WalkForward,
    WalkForwardConfig,
    WindowResult,
    objective_score,
    plan_windows,
)


def make_frames(symbols: int = 4, bars: int = 3000, seconds: int = 600, seed: int = 3):
    rng = np.random.default_rng(seed)
    start = 1_700_000_000
    series = {}
    for i in range(symbols):
        ret = rng.normal(0.0, 0.008, bars)
        close = 50 * np.cumprod(1 + ret)
        series[f"C{i:02d}"] = np.vstack(
            [
                start + np.arange(bars) * seconds,
                np.concatenate(([close[0]], close[:-1])),
                close * (1 + np.abs(rng.normal(0, 0.004, bars))),
                close * (1 - np.abs(rng.normal(0, 0.004, bars))),
                close,
                np.abs(rng.lognormal(9, 0.6, bars)),
            ]
        )
    frames = frames_from_series(series, seconds=seconds, schema=CRYPTO)
    frames.liquidity = np.full(frames.close.shape, 5e6, dtype=np.float32)
    return frames


def window_result(index: int, oos_return: float, trades: int = 40) -> WindowResult:
    return WindowResult(
        index=index,
        params={"volume_spike_multiple": 2.0},
        is_metrics={"total_return": 0.1, "trades": 60},
        oos_metrics={
            "total_return": oos_return,
            "trades": trades,
            "win_rate": 0.5,
            "max_drawdown": 0.05,
        },
        counted=trades >= 30,
        profitable=trades >= 30 and oos_return > 0,
    )


def walker(**overrides) -> WalkForward:
    config = WalkForwardConfig(**overrides)
    return WalkForward(ParamSpace(), PortfolioSettings(), config)


# --------------------------------------------------------------------------
# Window planning
# --------------------------------------------------------------------------
def test_windows_roll_forward_without_overlapping_out_of_sample():
    config = WalkForwardConfig(in_sample_days=60, out_of_sample_days=15, step_days=15)
    windows = plan_windows(0, 200 * 86400, config)

    assert windows, "a 200-day history should support several windows"
    for window in windows:
        assert window.is_to == window.oos_from, "validation must start where tuning ends"
        assert window.is_days == 60
        assert window.oos_days == 15
    # Stepping by the out-of-sample length means each window validates on data
    # the previous one never saw.
    for earlier, later in zip(windows, windows[1:]):
        assert later.oos_from >= earlier.oos_to


def test_a_history_too_short_for_one_window_is_refused():
    config = WalkForwardConfig(in_sample_days=60, out_of_sample_days=15)
    assert plan_windows(0, 30 * 86400, config) == []

    with pytest.raises(ValueError, match="too short"):
        walker().run(make_frames(bars=200))


# --------------------------------------------------------------------------
# Acceptance thresholds
# --------------------------------------------------------------------------
def test_a_window_below_the_trade_floor_does_not_count_either_way():
    outcome = walker().aggregate(
        [window_result(0, 0.05, trades=40), window_result(1, -0.05, trades=12)]
    )
    assert outcome.counted_windows == 1
    assert outcome.profitable_windows == 1
    # The losing window is excluded because it never reached 30 trades, not
    # because it lost - a 12-trade window is a coin flip, not evidence.
    assert outcome.profitable_share == 1.0


def consistency_complaint(outcome) -> str | None:
    """The reason mentioning the profitable-window bar, if it was raised.

    Isolated deliberately: seven wins and three losses in the same run is a wide
    spread by construction, so such a set also trips the overfit flag. Asserting
    on `accepted` here would be testing both rules and pinning neither.
    """
    return next((r for r in outcome.reasons if "profitable out of sample" in r), None)


def test_seventy_percent_of_windows_must_be_profitable():
    at_the_bar = walker().aggregate(
        [window_result(i, 0.04) for i in range(7)]
        + [window_result(i + 7, -0.02) for i in range(3)]
    )
    assert at_the_bar.profitable_share == pytest.approx(0.7)
    assert consistency_complaint(at_the_bar) is None, "70% exactly should clear the bar"

    below = walker().aggregate(
        [window_result(i, 0.04) for i in range(6)]
        + [window_result(i + 6, -0.02) for i in range(4)]
    )
    assert below.profitable_share == pytest.approx(0.6)
    assert not below.accepted
    assert "60%" in (consistency_complaint(below) or "")


def test_a_consistent_profitable_run_is_accepted():
    """The one combination that clears every gate at once."""
    steady = [window_result(i, 0.03 + 0.001 * (i % 3)) for i in range(8)]
    outcome = walker().aggregate(steady)

    assert outcome.profitable_share == 1.0
    assert not outcome.overfit
    assert outcome.accepted
    assert outcome.reasons == []


def test_overfit_flag_fires_when_returns_swing_more_than_they_average():
    """A set that makes its money in one window is a lottery ticket."""
    spiky = [
        window_result(0, 0.40), window_result(1, -0.06), window_result(2, -0.05),
        window_result(3, -0.04), window_result(4, 0.02),
    ]
    outcome = walker().aggregate(spiky)
    assert outcome.stdev_window_return > outcome.mean_window_return
    assert outcome.overfit
    assert not outcome.accepted
    assert any("overfit" in r for r in outcome.reasons)

    steady = [window_result(i, 0.03 + 0.002 * (i % 2)) for i in range(6)]
    assert not walker().aggregate(steady).overfit


def test_walk_forward_efficiency_normalises_for_window_length():
    """In-sample windows are four times longer, so raw totals would mislead.

    A strategy that generalised perfectly would report an efficiency near 0.25
    if the totals were divided directly; per-day normalisation reports 1.0.
    """
    config = WalkForwardConfig(in_sample_days=60, out_of_sample_days=15)
    runner = WalkForward(ParamSpace(), PortfolioSettings(), config)
    windows = []
    for i in range(4):
        result = window_result(i, 0.025)
        result.is_metrics = {"total_return": 0.10, "trades": 60}
        windows.append(result)

    outcome = runner.aggregate(windows)
    assert outcome.walk_forward_efficiency == pytest.approx(1.0, rel=1e-6)


def test_consensus_prefers_the_most_repeated_winner_not_the_luckiest_window():
    common = {"volume_spike_multiple": 2.0}
    lucky = {"volume_spike_multiple": 9.9}

    windows = []
    for i in range(3):
        result = window_result(i, 0.02)
        result.params = dict(common)
        windows.append(result)
    outlier = window_result(3, 0.90)
    outlier.params = dict(lucky)
    windows.append(outlier)

    assert walker().aggregate(windows).best_params == common


# --------------------------------------------------------------------------
# Objective
# --------------------------------------------------------------------------
def test_objective_rejects_a_window_below_the_trade_floor():
    config = WalkForwardConfig(min_trades_per_window=30)
    metrics = {"trades": 10, "total_return": 5.0, "max_drawdown": 0.01}
    assert objective_score(metrics, 30.0, config) == -math.inf


@pytest.mark.parametrize("total_return", [0.20, -0.20])
def test_churn_is_penalised_on_both_sides_of_zero(total_return):
    """Discounting a negative score toward zero would reward overtrading.

    This is the bug the trade-frequency penalty is meant to fix, so it is pinned
    in both directions: a set that churns must score worse than an identical set
    that does not, whether it made money or lost it.
    """
    config = WalkForwardConfig(min_trades_per_window=10, trades_per_day_cap=8.0)
    calm = {"trades": 80, "total_return": total_return, "max_drawdown": 0.10}
    churny = {"trades": 800, "total_return": total_return, "max_drawdown": 0.10}

    assert objective_score(churny, 10.0, config) < objective_score(calm, 10.0, config)


def test_objective_prefers_the_same_return_at_a_smaller_drawdown():
    config = WalkForwardConfig(min_trades_per_window=10)
    shallow = {"trades": 50, "total_return": 0.20, "max_drawdown": 0.08}
    deep = {"trades": 50, "total_return": 0.20, "max_drawdown": 0.30}
    assert objective_score(shallow, 30.0, config) > objective_score(deep, 30.0, config)


# --------------------------------------------------------------------------
# Search
# --------------------------------------------------------------------------
def test_search_stops_on_a_plateau_before_the_cap():
    space = ParamSpace(values={k: v for k, v in ParamSpace().values.items()})
    search = Search(space=space, max_evaluations=5000, batch_size=16, plateau_rounds=3)

    rounds = 0
    for batch in search.batches():
        rounds += 1
        # Every combination scores the same, so nothing ever improves.
        search.observe([(combo, 1.0) for combo in batch])
        assert rounds < 50, "the plateau rule should have stopped this long ago"

    assert "plateau" in search.state.stopped
    assert search.state.evaluated < 5000


def test_search_respects_the_iteration_cap_when_it_keeps_improving():
    space = ParamSpace()
    search = Search(space=space, max_evaluations=64, batch_size=16, plateau_rounds=99)

    score = 0.0
    for batch in search.batches():
        scored = []
        for combo in batch:
            score += 1.0
            scored.append((combo, score))
        search.observe(scored)

    assert search.state.evaluated <= 64
    assert "cap" in search.state.stopped or "exhausted" in search.state.stopped


def test_cross_field_rules_are_enforced_at_search_time():
    """A combination the config would refuse must never be searched.

    Otherwise the optimizer can spend hours proving out a set that cannot be
    written to config.json, and the failure surfaces at hand-off time.
    """
    assert not is_valid(
        {"ema_fast": 21, "ema_slow": 9, "rr_min": 2.0, "rr_max": 4.0,
         "momentum_candles": 3, "volume_spike_lookback": 20}
    )
    assert not is_valid(
        {"ema_fast": 9, "ema_slow": 21, "rr_min": 5.0, "rr_max": 4.0,
         "momentum_candles": 3, "volume_spike_lookback": 20}
    )
    for combo in ParamSpace().sample(50, np.random.default_rng(1)):
        assert is_valid(combo)


# --------------------------------------------------------------------------
# Monte Carlo
# --------------------------------------------------------------------------
def trades(n: int, seed: int = 5) -> list[dict]:
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        size = 400.0
        win = rng.random() < 0.45
        pnl = size * (rng.uniform(0.02, 0.09) if win else -rng.uniform(0.01, 0.04))
        out.append(
            {"pnl_usd": pnl, "size_usd": size, "exit_ts": 1_700_000_000 + i * 3600}
        )
    return out


def test_monte_carlo_reports_a_distribution_not_a_point():
    result = simulate(
        trades(150),
        starting_balance=1000.0,
        execution=ExecutionProfile.assumed(0.25, 0.25),
        assumed_round_trip_pct=0.01,
        iterations=2000,
        seed=11,
    )

    assert result.iterations == 2000
    assert result.p5_return < result.median_return < result.p95_return
    # The 5%-worst-case drawdown must be worse than the median one: it is the
    # tail, and it is the number position sizing reads.
    assert result.p5_max_drawdown > result.median_max_drawdown
    assert 0.0 <= result.probability_of_loss <= 1.0
    assert result.drawdown_histogram["total"] == 2000

    # The dashboard's numeric detail panel: best/worst simulated outcome
    # bracket every percentile between them, and the spread has a number.
    assert result.worst_return <= result.p5_return <= result.median_return
    assert result.median_return <= result.p95_return <= result.best_return
    assert result.stdev_return > 0.0
    summary = result.summary()
    for key in ("stdev_return", "best_return", "worst_return"):
        assert key in summary


def test_monte_carlo_keeps_the_worst_simulated_equity_paths():
    """Spec 6a: the actual worst-case drawdown curves, not just the number."""
    result = simulate(
        trades(150),
        starting_balance=1000.0,
        execution=ExecutionProfile.assumed(0.25, 0.25),
        assumed_round_trip_pct=0.01,
        iterations=2000,
        seed=11,
    )

    assert len(result.worst_equity_paths) == 10
    assert len(result.worst_path_drawdowns) == 10
    # Worst-first, and each path starts at 1.0 (100% of starting balance).
    assert result.worst_path_drawdowns == sorted(result.worst_path_drawdowns, reverse=True)
    assert all(path[0] == pytest.approx(1.0) for path in result.worst_equity_paths)
    # The single worst path's own max drawdown must match the headline number
    # exactly - it is one of the iterations that produced it.
    worst_path = np.asarray(result.worst_equity_paths[0])
    peak = np.maximum.accumulate(worst_path)
    dd = float(((peak - worst_path) / peak).max())
    assert dd == pytest.approx(result.worst_path_drawdowns[0], abs=1e-9)

    paths_payload = result.worst_paths()
    assert paths_payload["paths"] == result.worst_equity_paths
    assert len(paths_payload["drawdowns"]) == 10


def test_permutation_alone_leaves_terminal_return_almost_fixed():
    """Why the default resamples with replacement.

    Under fixed-fraction sizing the equity path is a cumulative product, and a
    product does not care about ordering. Permutation therefore produces one
    terminal value and many drawdown paths - useful for drawdown, useless for
    the 5th-percentile terminal return the spec asks for.
    """
    common = dict(
        starting_balance=1000.0,
        execution=ExecutionProfile(0.0, 0.0, 0.0, 0.0, samples=99, source="observed"),
        assumed_round_trip_pct=0.0,
        iterations=500,
        seed=3,
    )
    permuted = simulate(trades(120), resample="permutation", **common)
    bootstrapped = simulate(trades(120), resample="bootstrap", **common)

    permuted_spread = permuted.p95_return - permuted.p5_return
    bootstrap_spread = bootstrapped.p95_return - bootstrapped.p5_return
    assert permuted_spread == pytest.approx(0.0, abs=1e-9)
    assert bootstrap_spread > 0.01
    # Ordering still moves the drawdown, which is the half permutation is for.
    assert permuted.p5_max_drawdown > permuted.median_max_drawdown


def test_worse_execution_costs_produce_a_worse_distribution():
    cheap = ExecutionProfile(0.05, 0.01, 0.05, 0.01, samples=99, source="observed")
    dear = ExecutionProfile(0.60, 0.20, 0.30, 0.05, samples=99, source="observed")
    sample = trades(200)
    common = dict(
        starting_balance=1000.0, assumed_round_trip_pct=0.002, iterations=1500, seed=8
    )

    optimistic = simulate(sample, execution=cheap, **common)
    realistic = simulate(sample, execution=dear, **common)
    assert realistic.median_return < optimistic.median_return


def test_execution_profile_falls_back_loudly_without_enough_fills():
    profile = ExecutionProfile.from_fills([0.2] * 5, [0.25] * 5, min_samples=30)
    assert profile.source == "assumed", "5 fills must not pass as a measured model"

    measured = ExecutionProfile.from_fills(
        list(np.random.default_rng(0).normal(0.3, 0.1, 100)),
        list(np.random.default_rng(1).normal(0.25, 0.02, 100)),
        min_samples=30,
    )
    assert measured.source == "observed"
    assert measured.samples == 100


def test_monte_carlo_on_no_trades_is_empty_rather_than_wrong():
    result = simulate(
        [],
        starting_balance=1000.0,
        execution=ExecutionProfile.assumed(0.25, 0.25),
        assumed_round_trip_pct=0.01,
    )
    assert result.iterations == 0
    assert result.p5_max_drawdown == 0.0


# --------------------------------------------------------------------------
# Stress
# --------------------------------------------------------------------------
def test_rolling_max_matches_the_naive_computation():
    rng = np.random.default_rng(4)
    x = rng.normal(0, 1, 500)
    for window in (1, 7, 64, 128):
        got = _rolling_max(x, window)
        expected = np.array(
            [x[i : i + window].max() for i in range(x.size - window + 1)]
        )
        assert np.allclose(got, expected)


def test_crash_windows_are_found_in_the_data():
    """The stress windows come from the data, not a hard-coded list of dates."""
    frames = make_frames(symbols=5, bars=4000, seed=9)
    # Impose a market-wide slide on every symbol over the same stretch.
    crash_from, crash_to = 2000, 2400
    for i in range(frames.n_symbols):
        decay = np.linspace(1.0, 0.55, crash_to - crash_from)
        frames.close[i, crash_from:crash_to] *= decay
        frames.high[i, crash_from:crash_to] *= decay
        frames.low[i, crash_from:crash_to] *= decay
        frames.close[i, crash_to:] *= 0.55

    grid, index = market_index(frames)
    assert index[crash_to] < index[crash_from] * 0.8, "the fixture should show a crash"

    windows = find_windows(
        frames, StressConfig(crash_windows=2, flash_windows=1, min_index_decline=0.10)
    )
    assert windows, "a 45% market-wide fall should be found"
    crash = [w for w in windows if w.kind == "crash"]
    assert crash and crash[0].depth >= 0.10
    assert any(
        w.from_ts <= int(grid[crash_to]) and w.to_ts >= int(grid[crash_from])
        for w in windows
    ), "at least one window should cover the imposed crash"


# --------------------------------------------------------------------------
# RunStore retention (spec 6c/7) - WF/MC output is the fastest-growing
# storage component and needs its own purge policy, unlike raw candles.
# --------------------------------------------------------------------------
def _seed_run(store, *, age_days: float, status: str = "done", label: str = "") -> int:
    from solopt.store import now as store_now

    run_id = store.start_run(
        config={}, settings={}, space={"values": {}}, bundle="b", coverage={}, label=label,
    )
    finished = int(store_now() - age_days * 86400)
    store.conn.execute(
        "UPDATE runs SET status = ?, finished_at = ? WHERE id = ?",
        (status, finished, run_id),
    )
    store.append_feed(run_id, "a line")
    return run_id


def test_purge_removes_only_finished_runs_past_the_retention_window(tmp_path):
    from solopt.store import RunStore

    store = RunStore(tmp_path / "wfmc.db")
    old_id = _seed_run(store, age_days=200)
    recent_id = _seed_run(store, age_days=5)

    deleted = store.purge_older_than(180)

    assert deleted["runs"] == 1
    assert store.get_run(old_id) is None
    assert store.get_run(recent_id) is not None
    assert store.feed(recent_id)   # the recent run's feed survives


def test_purge_never_touches_a_still_running_run(tmp_path):
    from solopt.store import RunStore

    store = RunStore(tmp_path / "wfmc.db")
    stuck_id = _seed_run(store, age_days=400, status="running")

    deleted = store.purge_older_than(180)

    assert deleted["runs"] == 0
    assert store.get_run(stuck_id) is not None


def test_list_runs_pages_newest_first(tmp_path):
    from solopt.store import RunStore

    store = RunStore(tmp_path / "wfmc.db")
    ids = [_seed_run(store, age_days=i, label=f"run-{i}") for i in range(5)]

    first_page = store.list_runs(limit=2)
    assert [r["id"] for r in first_page] == sorted(ids, reverse=True)[:2]

    second_page = store.list_runs(limit=2, cursor=first_page[-1]["id"])
    assert [r["id"] for r in second_page] == sorted(ids, reverse=True)[2:4]
