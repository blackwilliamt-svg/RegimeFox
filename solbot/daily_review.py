"""The daily review: propose parameter adjustments, prove them, shadow them.

After the scheduled backtest finishes, the results are reviewed and a small set
of parameter adjustments is proposed by a deterministic set of rules - there is
no model in this loop, and never was anything for it to fall back from. Three
rules govern what happens next, and none of them are negotiable:

* **Nothing is applied to live.** A proposal goes into the shadow instance,
  which trades on paper alongside the real one. Promotion to live is a separate
  decision made by :mod:`solbot.paramsync` after fifteen clean days.
* **Every change is proved before it is shadowed.** The proposed set is
  backtested over the same window as the current one and both numbers are
  logged, so "this change helped" is a measurement rather than an assertion.
  A proposal that backtests worse than what it replaces is discarded.
* **Only known parameters, only inside their bounds.** Proposals are filtered
  against the config's own validation before they go anywhere near a backtest.
  A review that suggested a 400% position size would be refused by the same code
  that refuses it from the settings page.

This is the daily, on-droplet half of parameter tuning. The walk-forward
optimizer (``solopt``, invoked by :mod:`solbot.wfmc`) is what does real
parameter *search*; this module only nudges a handful of thresholds based on
what the daily backtest actually showed, and proves each nudge before shadowing
it.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from . import db
from .config import ConfigError, validate
from .review import SOURCE_RULES

log = logging.getLogger(__name__)

# The only parameters a review may touch. Deliberately narrower than the full
# editable set: risk caps, API budgets and the trading mode are the operator's,
# not the reviewer's.
ADJUSTABLE = (
    "volume_spike_multiple",
    "volume_spike_lookback",
    "momentum_candles",
    "momentum_min_pct",
    "rsi_max_entry",
    "ema_fast",
    "ema_slow",
    "stop_atr_mult",
    "rr_min",
    "rr_max",
    "trailing_activate_r",
    "trailing_distance_atr",
    "signal_invalidation_bars",
    "max_hold_minutes",
    "regime_trend_er",
    "regime_chop_atr_pct",
    "regime_allowed",
    "confluence_required",
    "max_trades_per_day",
)


@dataclass
class Proposal:
    key: str
    current: Any
    proposed: Any
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "from": self.current,
            "to": self.proposed,
            "reason": self.reason,
        }


@dataclass
class ReviewOutcome:
    source: str = SOURCE_RULES
    assessment: str = ""
    confidence: float = 0.0
    proposals: list[Proposal] = field(default_factory=list)
    accepted: list[Proposal] = field(default_factory=list)
    before: dict[str, Any] = field(default_factory=dict)
    after: dict[str, Any] = field(default_factory=dict)
    applied: bool = False
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "assessment": self.assessment,
            "confidence": round(self.confidence, 3),
            "proposals": [p.as_dict() for p in self.proposals],
            "accepted": [p.as_dict() for p in self.accepted],
            "before": self.before,
            "after": self.after,
            "applied": self.applied,
            "note": self.note,
        }


# --------------------------------------------------------------------------
# The deterministic proposer
# --------------------------------------------------------------------------
def propose_from_rules(
    summary: dict[str, Any], cfg: dict[str, Any], *, days: int
) -> ReviewOutcome:
    """Adjustments from the results alone.

    Each rule targets one specific failure the numbers can actually show. It
    deliberately proposes small steps: a review that moves a threshold by 40% is
    not tuning, it is guessing, and the walk-forward optimizer is the thing that
    does real parameter search.
    """
    outcome = ReviewOutcome(source=SOURCE_RULES)
    trades = int(summary.get("trades", 0))
    if trades < 10:
        outcome.assessment = (
            f"Only {trades} trades over {days} days - too few to draw a conclusion "
            "from, so nothing is proposed."
        )
        return outcome

    per_day = trades / max(days, 1)
    win_rate = float(summary.get("win_rate", 0.0))
    profit_factor = summary.get("profit_factor")
    profit_factor = float(profit_factor) if profit_factor else 0.0
    drawdown = float(summary.get("max_drawdown", 0.0))
    fees = float(summary.get("total_fees", 0.0))
    pnl = float(summary.get("total_pnl", 0.0))

    notes: list[str] = []
    cap = int(cfg.get("max_trades_per_day", 8))

    # 1. Overtrading: raise the volume bar before anything else. It is the rule
    #    that most directly reduces the number of entries.
    if per_day > cap:
        notes.append(f"{per_day:.0f} trades a day against a cap of {cap}")
        outcome.proposals.append(
            Proposal(
                "volume_spike_multiple",
                cfg["volume_spike_multiple"],
                round(float(cfg["volume_spike_multiple"]) * 1.15, 3),
                f"{per_day:.0f} trades a day exceeds the {cap} cap; a higher volume "
                "bar is the most direct way to take fewer, better setups.",
            )
        )
        if int(cfg.get("confluence_required", 0)) < len(
            cfg.get("confluence_timeframes") or []
        ):
            outcome.proposals.append(
                Proposal(
                    "confluence_required",
                    cfg["confluence_required"],
                    int(cfg["confluence_required"]) + 1,
                    "Requiring one more timeframe to agree filters the weakest "
                    "entries without touching the core rules.",
                )
            )

    # 2. Fees eating the edge: the scalping failure mode in numbers.
    if pnl > 0 and fees > pnl * 0.5:
        notes.append(f"fees are ${fees:,.0f} against ${pnl:,.0f} of profit")
        outcome.proposals.append(
            Proposal(
                "rr_min",
                cfg["rr_min"],
                round(min(float(cfg["rr_max"]), float(cfg["rr_min"]) + 0.25), 3),
                f"Fees of ${fees:,.0f} against ${pnl:,.0f} profit means the average "
                "win is too small to be worth its round trip; demand more reward "
                "per unit of risk.",
            )
        )

    # 3. Getting stopped out too often: the stop may be inside the noise.
    if win_rate < 0.35 and profit_factor < 1.2:
        notes.append(f"{win_rate * 100:.0f}% win rate with a {profit_factor:.2f} profit factor")
        outcome.proposals.append(
            Proposal(
                "stop_atr_mult",
                cfg["stop_atr_mult"],
                round(float(cfg["stop_atr_mult"]) * 1.15, 3),
                f"A {win_rate * 100:.0f}% win rate at a {profit_factor:.2f} profit "
                "factor suggests stops are sitting inside normal noise; widening "
                "them slightly should cut the premature exits.",
            )
        )

    # 4. Drawdown past tolerance: stop riding losers so long.
    if drawdown > float(cfg.get("drawdown_tolerance", 0.25)):
        notes.append(f"{drawdown * 100:.0f}% drawdown")
        outcome.proposals.append(
            Proposal(
                "signal_invalidation_bars",
                cfg["signal_invalidation_bars"],
                max(1, int(cfg["signal_invalidation_bars"]) - 1),
                f"A {drawdown * 100:.0f}% drawdown means losing positions are held "
                "too long; exit on invalidation a bar sooner.",
            )
        )

    outcome.proposals = outcome.proposals[:3]
    if not outcome.proposals:
        outcome.assessment = (
            f"{trades} trades over {days} days at {win_rate * 100:.0f}% win rate and "
            f"a {profit_factor:.2f} profit factor, {per_day:.1f} trades a day. "
            "Nothing in the results argues for a change."
        )
        outcome.confidence = 0.5
        return outcome

    outcome.assessment = (
        f"{trades} trades over {days} days at {win_rate * 100:.0f}% win rate and a "
        f"{profit_factor:.2f} profit factor. Concerns: " + "; ".join(notes) + "."
    )
    outcome.confidence = min(0.8, 0.4 + 0.15 * len(outcome.proposals))
    return outcome


# --------------------------------------------------------------------------
# Running the review
# --------------------------------------------------------------------------
def filter_valid(
    proposals: Sequence[Proposal], cfg: dict[str, Any]
) -> tuple[list[Proposal], list[str]]:
    """Drop proposals the config itself would refuse, and say why.

    Validation runs against the state the config *would* be in with every
    surviving proposal applied, so a pair that is individually fine but jointly
    contradictory - ``ema_fast`` above ``ema_slow``, say - is caught here rather
    than at write time.
    """
    kept: list[Proposal] = []
    rejected: list[str] = []
    for proposal in proposals:
        if proposal.key not in ADJUSTABLE:
            rejected.append(f"{proposal.key}: not an adjustable parameter")
            continue
        candidate = {p.key: p.proposed for p in kept}
        candidate[proposal.key] = proposal.proposed
        try:
            validate(candidate, cfg)
        except ConfigError as exc:
            rejected.append(f"{proposal.key}: {exc}")
            continue
        if str(cfg.get(proposal.key)) == str(proposal.proposed):
            rejected.append(f"{proposal.key}: already at that value")
            continue
        kept.append(proposal)
    return kept, rejected


def run_daily_review(
    *,
    cfg: dict[str, Any],
    summary: dict[str, Any],
    days: int,
    backtest: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    conn: sqlite3.Connection | None = None,
) -> ReviewOutcome:
    """Review a finished backtest and shadow anything that measurably improves it.

    ``backtest`` re-runs the same window with a set of overrides and returns its
    summary. Without it the review still proposes and logs, but nothing is
    shadowed - an unproven change is not worth running even on paper.
    """
    conn = conn or db.connect()
    outcome = propose_from_rules(summary, cfg, days=days)

    outcome.before = summary
    kept, rejected = filter_valid(outcome.proposals, cfg)
    if rejected:
        outcome.note = (outcome.note + " " if outcome.note else "") + (
            "Refused: " + "; ".join(rejected)
        )

    if not kept:
        _log_review(outcome, days, conn)
        return outcome

    if backtest is None:
        outcome.note = (
            (outcome.note + " " if outcome.note else "")
            + "No backtest hook available, so nothing was shadowed unproven."
        )
        outcome.proposals = kept
        _log_review(outcome, days, conn)
        return outcome

    overrides = {p.key: p.proposed for p in kept}
    try:
        outcome.after = backtest(overrides)
    except Exception as exc:
        outcome.note = (
            (outcome.note + " " if outcome.note else "")
            + f"The proposed set could not be backtested ({exc}); nothing was shadowed."
        )
        _log_review(outcome, days, conn)
        return outcome

    if _improves(outcome.before, outcome.after):
        outcome.accepted = kept
        db.enqueue_command(
            "set_shadow", {"overrides": overrides}, requested_by="daily-review", conn=conn
        )
        outcome.applied = True
    else:
        outcome.note = (
            (outcome.note + " " if outcome.note else "")
            + "The proposed set did not backtest better than the current one, so it "
            "was discarded rather than shadowed."
        )

    _log_review(outcome, days, conn)
    return outcome


def _improves(before: dict[str, Any], after: dict[str, Any]) -> bool:
    """Is the proposed set actually better, on the terms the spec cares about?

    Return alone is not enough: a set that earns 2% more while doubling the
    drawdown is worse. Both matter, and neither is allowed to get materially
    worse in exchange for the other.
    """
    if int(after.get("trades", 0)) < 5:
        return False
    before_return = float(before.get("total_return", 0.0))
    after_return = float(after.get("total_return", 0.0))
    before_dd = max(float(before.get("max_drawdown", 0.0)), 1e-6)
    after_dd = max(float(after.get("max_drawdown", 0.0)), 1e-6)

    if after_return <= before_return:
        return False
    if after_dd > before_dd * 1.15:
        return False
    # Return per unit of drawdown has to improve, not just raw return.
    return (after_return / after_dd) > (before_return / before_dd)


def _log_review(outcome: ReviewOutcome, days: int, conn: sqlite3.Connection) -> None:
    conn.execute(
        "INSERT INTO entry_reviews(ts, kind, decision, conviction, detail) "
        "VALUES (?,?,?,?,?)",
        (
            db.now(), "daily",
            "applied" if outcome.applied else "no_change",
            outcome.confidence, json.dumps(outcome.as_dict(), default=float),
        ),
    )

    if outcome.applied:
        changes = ", ".join(
            f"{p.key} {p.current} -> {p.proposed}" for p in outcome.accepted
        )
        message = (
            f"Daily review shadowed {len(outcome.accepted)} change(s): {changes}. "
            f"Backtest over {days}d improved from "
            f"{float(outcome.before.get('total_return', 0)) * 100:+.2f}% return / "
            f"{float(outcome.before.get('max_drawdown', 0)) * 100:.1f}% drawdown to "
            f"{float(outcome.after.get('total_return', 0)) * 100:+.2f}% / "
            f"{float(outcome.after.get('max_drawdown', 0)) * 100:.1f}%. "
            "Running in shadow only; live is untouched."
        )
        level = "warn"
    else:
        message = (
            f"Daily review ({outcome.source}): {outcome.assessment} "
            f"{outcome.note}".strip()
        )
        level = "info"

    db.log_event(
        message, level=level, category="system", detail=outcome.as_dict(), conn=conn
    )
