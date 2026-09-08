"""The plain-language run feed.

The spec asks for "what it is currently doing, as a plain-language feed of what
it is discovering as it runs" rather than a results table, so this module is
where numbers become sentences. A percentage bar tells you the run is 40% done;
"window 7 out-of-sample: -3.1% from 33 trades, did not hold up" tells you the
strategy is not working, which is the thing worth knowing four hours in.

Lines go three places at once: the local run store (durable, survives a reboot),
the Python log (so a headless run leaves a trace), and - when configured - the
droplet, which is what puts the feed on the dashboard's walk-forward tab while
the search is still running.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

log = logging.getLogger(__name__)


class FeedSink(Protocol):
    """Somewhere a feed line can be delivered."""

    def emit(self, run_id: int, line: dict[str, Any]) -> None: ...

    def flush(self) -> None: ...


@dataclass
class RunFeed:
    """Writes feed lines, and turns metrics into sentences.

    ``run_id`` addresses this run in ``store`` (a local, single-machine
    concern: window resumability, the checkpoint). ``report_run_id`` is what
    gets handed to the sinks - the id the *dashboard* knows this run by. They
    are the same number by default, but a caller orchestrating several
    concurrent workers against their own throwaway local stores (the monthly
    RunPod retest, batched by coin subset) needs those local ids kept apart
    from a single globally-unique id per batch, or two workers' "run 1" would
    collide in the droplet's own run table.
    """

    store: Any
    run_id: int
    report_run_id: int = 0
    sinks: list[FeedSink] = field(default_factory=list)
    echo: bool = True

    def __post_init__(self) -> None:
        self._lock = threading.Lock()
        if not self.report_run_id:
            self.report_run_id = self.run_id

    # ------------------------------------------------------------------
    def say(self, message: str, *, level: str = "info", detail: Any = None) -> None:
        with self._lock:
            line_id = self.store.append_feed(
                self.run_id, message, level=level, detail=detail
            )
        if self.echo:
            log.log(
                logging.WARNING if level in ("warn", "alert") else logging.INFO, "%s", message
            )
        payload = {
            "id": line_id, "ts": int(time.time()), "level": level,
            "message": message, "detail": detail,
        }
        for sink in self.sinks:
            try:
                sink.emit(self.report_run_id, payload)
            except Exception as exc:  # a dashboard being down must not stop a run
                log.debug("feed sink %s failed: %s", type(sink).__name__, exc)

    def warn(self, message: str, **kwargs: Any) -> None:
        self.say(message, level="warn", **kwargs)

    def alert(self, message: str, **kwargs: Any) -> None:
        self.say(message, level="alert", **kwargs)

    def flush(self) -> None:
        for sink in self.sinks:
            try:
                sink.flush()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Sentence builders. Kept here rather than at the call sites so the voice of
    # the feed stays consistent as more stages are added.
    # ------------------------------------------------------------------
    def loaded(self, coverage: dict[str, Any]) -> None:
        self.say(
            f"Loaded {coverage.get('symbols', 0)} coins and "
            f"{coverage.get('bars', 0):,} bars covering "
            f"{coverage.get('days', 0)} days at "
            f"{coverage.get('timeframe_seconds', 0) // 60}-minute candles.",
            detail=coverage,
        )

    def planned(self, windows: int, is_days: int, oos_days: int) -> None:
        self.say(
            f"Planned {windows} walk-forward windows: tune on {is_days} days, "
            f"then test on the next {oos_days} days it has never seen, then roll forward."
        )

    def searching(self, index: int, total: int, is_from: str, is_to: str, combos: int) -> None:
        self.say(
            f"Window {index + 1} of {total}: searching up to {combos:,} parameter "
            f"combinations on {is_from} to {is_to}."
        )

    def in_sample(self, index: int, params: dict[str, Any], metrics: dict[str, Any]) -> None:
        self.say(
            f"Window {index + 1} in-sample best: {metrics['total_return'] * 100:+.1f}% "
            f"from {metrics['trades']} trades with a "
            f"{metrics['max_drawdown'] * 100:.1f}% drawdown — "
            f"{describe(params)}.",
            detail={"params": params, "metrics": metrics},
        )

    def out_of_sample(self, index: int, metrics: dict[str, Any], counted: bool) -> None:
        if not counted:
            self.warn(
                f"Window {index + 1} out-of-sample: only {metrics['trades']} trades, "
                f"below the floor — this window does not count either way."
            )
            return
        verdict = "held up" if metrics["total_return"] > 0 else "did not hold up"
        self.say(
            f"Window {index + 1} out-of-sample: {metrics['total_return'] * 100:+.1f}% "
            f"from {metrics['trades']} trades, {metrics['win_rate'] * 100:.0f}% win rate. "
            f"It {verdict}.",
            level="info" if metrics["total_return"] > 0 else "warn",
            detail=metrics,
        )

    def skipped(self, index: int, reason: str) -> None:
        self.warn(f"Window {index + 1} skipped: {reason}.")

    def efficiency(self, wfe: float, is_return: float, oos_return: float) -> None:
        if is_return <= 0:
            self.warn(
                "Walk-forward efficiency is undefined — the in-sample windows did not "
                "make money in aggregate, so there is nothing for out-of-sample to keep."
            )
            return
        self.say(
            f"Walk-forward efficiency {wfe:.2f}: out-of-sample kept "
            f"{wfe * 100:.0f}% of what in-sample promised "
            f"({oos_return * 100:+.1f}% against {is_return * 100:+.1f}%).",
            level="info" if wfe >= 0.5 else "warn",
        )

    def consistency(self, profitable: int, counted: int, threshold: float) -> None:
        if counted == 0:
            self.warn("No window produced enough trades to count. Nothing has been proven.")
            return
        share = profitable / counted
        met = share >= threshold
        self.say(
            f"{profitable} of {counted} out-of-sample windows were profitable "
            f"({share * 100:.0f}%) — {'clears' if met else 'below'} the "
            f"{threshold * 100:.0f}% bar.",
            level="info" if met else "warn",
        )

    def overfit(self, mean: float, stdev: float, flagged: bool) -> None:
        if flagged:
            self.alert(
                f"OVERFIT: window returns swing more than they average "
                f"(standard deviation {stdev * 100:.1f}% against a mean of "
                f"{mean * 100:.1f}%). This parameter set is fitted to the past, "
                f"not to the market."
            )
        else:
            self.say(
                f"Return consistency is acceptable: standard deviation "
                f"{stdev * 100:.1f}% against a mean of {mean * 100:.1f}%."
            )

    def fragile(self, failures: list[str]) -> None:
        if failures:
            self.alert(
                "FRAGILE: this set passed the normal windows but failed the crash "
                "replay on " + ", ".join(failures) + ". Auto-promotion is blocked."
            )
        else:
            self.say("Crash replay passed: the set survived every stress window tested.")

    def monte_carlo(self, summary: dict[str, Any]) -> None:
        self.say(
            f"Monte Carlo over {summary['iterations']:,} resamples: median return "
            f"{summary['median_return'] * 100:+.1f}%, and in the worst 5% of orderings "
            f"a {summary['p5_max_drawdown'] * 100:.1f}% drawdown with a "
            f"{summary['p5_return'] * 100:+.1f}% return. Position sizing uses the "
            f"{summary['p5_max_drawdown'] * 100:.1f}% figure, not the backtest's "
            f"{summary['historical_max_drawdown'] * 100:.1f}%.",
            detail=summary,
        )

    def finished(self, summary: dict[str, Any]) -> None:
        self.say(
            f"Run finished in {summary.get('elapsed_minutes', 0):.1f} minutes. "
            + (
                "The winning set met every threshold and has been written to the "
                "hand-off repo."
                if summary.get("accepted")
                else "No parameter set met every threshold; nothing was promoted."
            ),
            level="info" if summary.get("accepted") else "warn",
            detail=summary,
        )


def describe(params: dict[str, Any]) -> str:
    """A parameter set in words, for the feed and the daily review."""
    if not params:
        return "no parameters"
    bits = [
        f"volume {params.get('volume_spike_multiple', 0):.1f}x over "
        f"{params.get('volume_spike_lookback', 0)} bars",
        f"momentum {float(params.get('momentum_min_pct', 0)) * 100:.1f}% across "
        f"{params.get('momentum_candles', 0)} candles",
        f"EMA {params.get('ema_fast', 0)}/{params.get('ema_slow', 0)}",
        f"stop {params.get('stop_atr_mult', 0):.1f} ATR",
        f"reward:risk {params.get('rr_min', 0):.1f}-{params.get('rr_max', 0):.1f}",
        f"trail arms at {params.get('trailing_activate_r', 0):.1f}R",
    ]
    allowed = int(params.get("regime_allowed", 7))
    names = [n for bit, n in ((1, "trending"), (2, "ranging"), (4, "choppy")) if allowed & bit]
    bits.append("trades in " + (", ".join(names) if names else "no") + " markets")
    required = int(params.get("confluence_required", 0))
    if required:
        bits.append(f"needs {required} higher timeframe(s) to agree")
    return "; ".join(bits)


# --------------------------------------------------------------------------
# Sinks
# --------------------------------------------------------------------------
@dataclass
class CallbackSink:
    """Delivers lines to a Python callable - used by the CLI and the tests."""

    callback: Callable[[int, dict[str, Any]], None]

    def emit(self, run_id: int, line: dict[str, Any]) -> None:
        self.callback(run_id, line)

    def flush(self) -> None:
        return None


@dataclass
class BufferedSink:
    """Batches lines and hands them to ``send`` when the batch fills or ages.

    One HTTP round trip per feed line would make the run's pace depend on the
    droplet's latency. Batching keeps the dashboard within a few seconds of live
    without ever blocking the search on the network.
    """

    send: Callable[[int, list[dict[str, Any]]], None]
    batch_size: int = 20
    max_age_seconds: float = 5.0

    def __post_init__(self) -> None:
        self._buffer: list[dict[str, Any]] = []
        self._run_id = 0
        self._oldest = 0.0
        self._lock = threading.Lock()

    def emit(self, run_id: int, line: dict[str, Any]) -> None:
        with self._lock:
            if run_id != self._run_id and self._buffer:
                self._drain()
            self._run_id = run_id
            if not self._buffer:
                self._oldest = time.monotonic()
            self._buffer.append(line)
            due = (
                len(self._buffer) >= self.batch_size
                or (time.monotonic() - self._oldest) >= self.max_age_seconds
                or line.get("level") in ("warn", "alert")
            )
            if due:
                self._drain()

    def flush(self) -> None:
        with self._lock:
            self._drain()

    def _drain(self) -> None:
        if not self._buffer:
            return
        batch, self._buffer = self._buffer, []
        try:
            self.send(self._run_id, batch)
        except Exception as exc:
            log.debug("feed batch dropped: %s", exc)
