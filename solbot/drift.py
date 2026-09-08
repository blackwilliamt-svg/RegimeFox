"""Live-versus-backtest drift monitoring (spec 4.5).

A backtest is a claim about what the rules will do. Live and paper trading are
the measurement. Drift is the gap between them, and it is the single most useful
early warning the bot has: strategies do not usually fail with a bang, they fail
by quietly winning less often than they used to while every individual trade
still looks reasonable.

Two comparisons run continuously, over a rolling window:

* **Win rate**, compared in absolute points. A strategy expected to win 55% of
  the time that is winning 38% is describing a different market than the one it
  was tuned on.
* **Expectancy per trade**, compared in relative terms. This is the one that
  catches cost drift - the win rate holds, the average win shrinks, and fees
  quietly eat the edge. It is the failure mode the fee bill in the earlier
  backtest was a symptom of.

Nothing here halts trading. Drift is information for the operator and for the
daily review; the circuit breaker and the kill switch remain the things that
stop the bot, and they trip on realised losses rather than on a statistic.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from typing import Any

from . import db
from .portfolio import Portfolio

log = logging.getLogger(__name__)

STATUS_OK = "ok"
STATUS_WATCH = "watch"
STATUS_DRIFTING = "drifting"
STATUS_INSUFFICIENT = "insufficient"

# Ranked worst-first so a change in either direction can be described.
SEVERITY = {STATUS_INSUFFICIENT: 0, STATUS_OK: 1, STATUS_WATCH: 2, STATUS_DRIFTING: 3}

LAST_STATUS_KEY = "drift_status:{instance}"


@dataclass(slots=True)
class Expectation:
    """What the most recent backtest said to expect."""

    win_rate: float = 0.0
    expectancy: float = 0.0
    profit_factor: float = 0.0
    trades: int = 0
    ts: int = 0
    source: str = "none"

    @property
    def usable(self) -> bool:
        return self.trades > 0 and self.source != "none"

    def as_dict(self) -> dict[str, Any]:
        return {
            "win_rate": round(self.win_rate, 4),
            "expectancy": round(self.expectancy, 4),
            "profit_factor": round(self.profit_factor, 4),
            "trades": self.trades,
            "ts": self.ts,
            "source": self.source,
        }


@dataclass
class DriftSample:
    instance: str
    window_days: int
    status: str = STATUS_INSUFFICIENT
    live_trades: int = 0
    live_win_rate: float = 0.0
    live_expectancy: float = 0.0
    live_profit_factor: float = 0.0
    expected: Expectation = field(default_factory=Expectation)
    win_rate_gap: float = 0.0
    expectancy_ratio: float = 0.0
    message: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "instance": self.instance,
            "window_days": self.window_days,
            "status": self.status,
            "live_trades": self.live_trades,
            "live_win_rate": round(self.live_win_rate, 4),
            "live_expectancy": round(self.live_expectancy, 4),
            "live_profit_factor": round(self.live_profit_factor, 4),
            "expected": self.expected.as_dict(),
            "win_rate_gap": round(self.win_rate_gap, 4),
            "expectancy_ratio": round(self.expectancy_ratio, 4),
            "message": self.message,
        }


def latest_expectation(conn: sqlite3.Connection | None = None) -> Expectation:
    """Read the newest backtest run as the expectation to measure against.

    Expectancy is rebuilt from the win rate and the average win/loss rather than
    read directly, because that is the form the live side reports and comparing
    two differently-derived numbers is how a monitor ends up crying wolf.
    """
    conn = conn or db.connect()
    row = conn.execute(
        "SELECT * FROM backtest_runs WHERE trades > 0 ORDER BY ts DESC LIMIT 1"
    ).fetchone()
    if row is None:
        return Expectation()

    win_rate = float(row["win_rate"] or 0.0)
    avg_win = float(row["avg_win"] or 0.0)
    avg_loss = float(row["avg_loss"] or 0.0)
    return Expectation(
        win_rate=win_rate,
        expectancy=win_rate * avg_win + (1.0 - win_rate) * avg_loss,
        profit_factor=float(row["profit_factor"] or 0.0),
        trades=int(row["trades"] or 0),
        ts=int(row["ts"]),
        source="backtest",
    )


def measure(
    instance: str,
    cfg: dict[str, Any],
    *,
    expectation: Expectation | None = None,
    conn: sqlite3.Connection | None = None,
) -> DriftSample:
    """Compare one instance's recent trading against backtest expectation."""
    conn = conn or db.connect()
    window_days = int(cfg.get("drift_window_days", 14))
    sample = DriftSample(instance=instance, window_days=window_days)

    expected = expectation if expectation is not None else latest_expectation(conn)
    sample.expected = expected

    since = db.now() - window_days * 86400
    live = Portfolio(instance, cfg).performance(since=since, conn=conn)
    sample.live_trades = int(live.get("trades", 0))
    sample.live_win_rate = float(live.get("win_rate", 0.0))
    sample.live_expectancy = float(live.get("expectancy", 0.0))
    profit_factor = live.get("profit_factor", 0.0)
    sample.live_profit_factor = (
        0.0 if profit_factor in (None, float("inf")) else float(profit_factor)
    )

    minimum = int(cfg.get("drift_min_trades", 20))
    if not expected.usable:
        sample.status = STATUS_INSUFFICIENT
        sample.message = (
            "No backtest has run yet, so there is no expectation to compare against."
        )
        return sample
    if sample.live_trades < minimum:
        sample.status = STATUS_INSUFFICIENT
        sample.message = (
            f"{sample.live_trades} {instance} trades in {window_days} days - "
            f"{minimum} are needed before a comparison means anything."
        )
        return sample

    sample.win_rate_gap = sample.live_win_rate - expected.win_rate
    if expected.expectancy != 0:
        sample.expectancy_ratio = sample.live_expectancy / abs(expected.expectancy)
        if expected.expectancy < 0:
            # A backtest that expected to lose money is not a bar to clear;
            # treat any live result as its own story rather than a ratio.
            sample.expectancy_ratio = 0.0 if sample.live_expectancy < 0 else 1.0
    else:
        sample.expectancy_ratio = 1.0 if sample.live_expectancy >= 0 else 0.0

    win_tol = float(cfg.get("drift_win_rate_tolerance", 0.15))
    exp_tol = float(cfg.get("drift_expectancy_tolerance", 0.40))

    win_breach = abs(sample.win_rate_gap) > win_tol
    exp_breach = sample.expectancy_ratio < (1.0 - exp_tol)
    win_watch = abs(sample.win_rate_gap) > win_tol / 2.0
    exp_watch = sample.expectancy_ratio < (1.0 - exp_tol / 2.0)

    if win_breach or exp_breach:
        sample.status = STATUS_DRIFTING
    elif win_watch or exp_watch:
        sample.status = STATUS_WATCH
    else:
        sample.status = STATUS_OK
    sample.message = _describe(sample)
    return sample


def _describe(sample: DriftSample) -> str:
    expected = sample.expected
    head = (
        f"{sample.instance} over {sample.window_days}d: "
        f"{sample.live_win_rate * 100:.0f}% win rate against a backtested "
        f"{expected.win_rate * 100:.0f}%, "
        f"${sample.live_expectancy:.2f} per trade against ${expected.expectancy:.2f}"
    )
    if sample.status == STATUS_OK:
        return head + " - tracking expectation."
    if sample.status == STATUS_WATCH:
        return head + " - starting to diverge; worth watching."
    return (
        head
        + f" - drifting. Live is keeping {sample.expectancy_ratio * 100:.0f}% of the "
        "expected edge; the rules may have been tuned on a market that has moved on."
    )


def record(
    sample: DriftSample, conn: sqlite3.Connection | None = None
) -> None:
    """Persist a sample, and log an event when the status actually changes."""
    conn = conn or db.connect()
    conn.execute(
        "INSERT INTO drift_samples(ts, instance, window_days, live_trades, "
        "live_win_rate, live_expectancy, live_profit_factor, expected_win_rate, "
        "expected_expectancy, expected_profit_factor, status, detail) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            db.now(), sample.instance, sample.window_days, sample.live_trades,
            sample.live_win_rate, sample.live_expectancy, sample.live_profit_factor,
            sample.expected.win_rate, sample.expected.expectancy,
            sample.expected.profit_factor, sample.status,
            json.dumps(sample.as_dict()),
        ),
    )

    key = LAST_STATUS_KEY.format(instance=sample.instance)
    previous = db.kv_get(key, STATUS_OK, conn)
    if previous == sample.status:
        return
    db.kv_set(key, sample.status, conn)

    worsened = SEVERITY.get(sample.status, 0) > SEVERITY.get(previous, 0)
    if sample.status == STATUS_INSUFFICIENT:
        return  # not worth an event either way
    db.log_event(
        sample.message,
        level="warn" if sample.status == STATUS_DRIFTING else "info",
        category="system",
        instance=sample.instance,
        detail=sample.as_dict(),
        conn=conn,
    )
    if worsened and sample.status == STATUS_DRIFTING:
        log.warning("drift on %s: %s", sample.instance, sample.message)


def check_all(
    cfg: dict[str, Any],
    instances: tuple[str, ...] = ("live", "paper", "shadow"),
    conn: sqlite3.Connection | None = None,
) -> list[DriftSample]:
    """Measure and record every instance that has traded."""
    conn = conn or db.connect()
    expectation = latest_expectation(conn)
    out: list[DriftSample] = []
    for instance in instances:
        row = conn.execute(
            "SELECT 1 FROM trades WHERE instance = ? LIMIT 1", (instance,)
        ).fetchone()
        if row is None:
            continue
        sample = measure(instance, cfg, expectation=expectation, conn=conn)
        record(sample, conn)
        out.append(sample)
    return out


def recent(
    instance: str | None = None,
    *,
    limit: int = 100,
    conn: sqlite3.Connection | None = None,
) -> list[dict[str, Any]]:
    """Samples for the dashboard's drift panel, oldest first."""
    conn = conn or db.connect()
    sql = "SELECT * FROM drift_samples"
    args: list[Any] = []
    if instance:
        sql += " WHERE instance = ?"
        args.append(instance)
    sql += " ORDER BY id DESC LIMIT ?"
    args.append(int(limit))
    rows = conn.execute(sql, args).fetchall()
    out = []
    for row in reversed(rows):
        item = dict(row)
        try:
            item["detail"] = json.loads(item.get("detail") or "{}")
        except json.JSONDecodeError:
            item["detail"] = {}
        out.append(item)
    return out
