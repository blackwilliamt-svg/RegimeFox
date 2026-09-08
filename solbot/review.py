"""The entry review gate: a deterministic rules engine, no model in the loop.

One job now, not three. Borderline and high-conviction entries are put to the
gate before they execute. The rubric is in :data:`RUBRIC` and is deliberately
short: judge the whole market rather than the single token, favour strong
signals over frequent ones, and prefer capturing a large move with gains locked
in over scalping. The ride-versus-lock-in call is made per trade from the
metrics, not fixed one way, because it genuinely depends on whether the move
has room left.

There used to be a Claude API call sitting on top of this with a deterministic
proxy standing in for it on routine days. That model layer is gone: trade
approval, like everything else in the bot, is decided entirely by this
rules-based logic backed by walk-forward/Monte Carlo evidence. The rubric
itself did not change - it is now simply the only thing enforcing it, with no
network call, no timeout path, and no calibration drift to track.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from typing import Any

from . import db
from .indicators import REGIME_NAMES, TRENDING

DECISION_APPROVE = "approve"
DECISION_REJECT = "reject"

STYLE_RIDE = "ride"
STYLE_LOCK_IN = "lock_in"

SOURCE_RULES = "rules"
SOURCE_DISABLED = "disabled"

MARKET_CACHE_KEY = "market_context"
MARKET_TTL_SECONDS = 300


RUBRIC = """The rules-based entry gate applies this rubric to every borderline or \
high-conviction entry, after the bot's own volume, momentum, trend, liquidity \
depth, market regime, multi-timeframe agreement and rug-pull safety checks have \
already passed:

1. Judge the broad crypto market, not just the token. A strong single-token \
signal into a market that is selling off broadly is a worse trade than the same \
signal into a firm market. Breadth and the index move matter as much as the \
token's own metrics.
2. Favour only strong, high-conviction signals. The bot's known failure mode is \
overtrading: a previous configuration took 244 trades in seven days and paid \
449 USD in fees for them. Rejecting a mediocre setup costs one missed trade; \
approving it costs fees plus the position slot. When in doubt, reject.
3. Prioritise capturing larger moves with gains locked in over scalping small \
increments. A trade that needs everything to go right for a small gain is not \
worth the slot.
4. Decide the ride-versus-lock-in trade-off per trade, from the metrics. There \
is no fixed answer. Room to run - a trending regime, a firm market, agreement \
across timeframes, volatility that is high but not extreme - argues for riding \
with a wider trailing stop. A ranging or choppy regime, a soft market, a signal \
that has already extended, or unusually high volatility argues for locking \
gains in with a tighter trail."""


# --------------------------------------------------------------------------
# Broad market context
# --------------------------------------------------------------------------
@dataclass
class MarketContext:
    """What the whole tradeable universe is doing, not just one token."""

    breadth: float = 0.5           # share of the universe up over the window
    index_return: float = 0.0      # equal-weight mean return over the window
    tokens: int = 0
    window_minutes: int = 60
    median_atr_pct: float = 0.0
    trending_share: float = 0.0    # share of the universe classified trending
    ts: int = 0

    @property
    def tone(self) -> str:
        """A one-word read, used by the gate and shown in the feed."""
        if self.tokens < 5:
            return "unknown"
        if self.breadth >= 0.60 and self.index_return >= 0.0:
            return "firm"
        if self.breadth <= 0.40 or self.index_return <= -0.015:
            return "soft"
        return "mixed"

    def as_dict(self) -> dict[str, Any]:
        return {
            "breadth": round(self.breadth, 4),
            "index_return": round(self.index_return, 5),
            "tokens": self.tokens,
            "window_minutes": self.window_minutes,
            "median_atr_pct": round(self.median_atr_pct, 5),
            "trending_share": round(self.trending_share, 4),
            "tone": self.tone,
            "ts": self.ts,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MarketContext":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in (data or {}).items() if k in known})


def build_market_context(
    *,
    window_minutes: int = 60,
    conn: sqlite3.Connection | None = None,
    force: bool = False,
) -> MarketContext:
    """Breadth and index move across the universe, from the tick history.

    Built from price ticks the scanner is already collecting rather than from a
    fresh API call, and cached for five minutes: this is a description of the
    market, not of the token, and it does not change between two entries a
    minute apart.
    """
    conn = conn or db.connect()
    now = db.now()
    if not force:
        cached = db.kv_get(MARKET_CACHE_KEY, None, conn)
        if cached and now - int(cached.get("ts", 0)) < MARKET_TTL_SECONDS:
            return MarketContext.from_dict(cached)

    since = now - window_minutes * 60
    rows = conn.execute(
        "SELECT mint, "
        "       (SELECT price FROM price_ticks p2 WHERE p2.mint = p1.mint "
        "        AND p2.ts >= ? ORDER BY p2.ts ASC LIMIT 1) AS first_price, "
        "       (SELECT price FROM price_ticks p3 WHERE p3.mint = p1.mint "
        "        ORDER BY p3.ts DESC LIMIT 1) AS last_price "
        "FROM (SELECT DISTINCT mint FROM price_ticks WHERE ts >= ?) p1",
        (since, since),
    ).fetchall()

    returns = []
    for row in rows:
        first, last = row["first_price"], row["last_price"]
        if first and last and float(first) > 0:
            returns.append(float(last) / float(first) - 1.0)

    context = MarketContext(window_minutes=window_minutes, ts=now, tokens=len(returns))
    if returns:
        context.breadth = sum(1 for r in returns if r > 0) / len(returns)
        context.index_return = sum(returns) / len(returns)
    db.kv_set(MARKET_CACHE_KEY, context.as_dict(), conn)
    return context


# --------------------------------------------------------------------------
# Request / decision
# --------------------------------------------------------------------------
@dataclass
class EntryRequest:
    """Everything the gate is allowed to see about one candidate entry."""

    mint: str
    symbol: str = ""
    instance: str = "paper"
    price: float = 0.0
    size_usd: float = 0.0
    liquidity_usd: float = 0.0
    rr: float = 0.0
    strength: float = 0.0
    entry_reason: str = ""
    snapshot: dict[str, Any] = field(default_factory=dict)
    market: MarketContext = field(default_factory=MarketContext)
    trades_today: int = 0

    def as_dict(self) -> dict[str, Any]:
        snap = self.snapshot or {}
        return {
            "token": {
                "symbol": self.symbol or self.mint[:8],
                "price": round(float(self.price), 8),
                "liquidity_usd": round(float(self.liquidity_usd), 0),
                "intended_size_usd": round(float(self.size_usd), 2),
            },
            "signal": {
                "reward_risk": round(float(self.rr), 2),
                "strength": round(float(self.strength), 3),
                "volume_vs_average": round(float(snap.get("volume_ratio") or 0.0), 2),
                "momentum_pct": round(float(snap.get("momentum_pct") or 0.0) * 100, 2),
                "rsi": round(float(snap.get("rsi") or 0.0), 1),
                "atr_pct": round(float(snap.get("atr_pct") or 0.0) * 100, 2),
                "efficiency_ratio": round(float(snap.get("efficiency") or 0.0), 3),
                "regime": REGIME_NAMES.get(int(snap.get("regime", 2)), "unknown"),
                "higher_timeframes_agreeing": int(snap.get("confluence") or 0),
                "why": self.entry_reason[:400],
            },
            "market": self.market.as_dict(),
            "context": {"trades_already_today": self.trades_today},
        }


@dataclass
class GateDecision:
    approve: bool = False
    conviction: float = 0.0
    exit_style: str = STYLE_LOCK_IN
    trailing_distance_atr: float | None = None
    rationale: str = ""
    source: str = SOURCE_RULES

    @property
    def decision(self) -> str:
        return DECISION_APPROVE if self.approve else DECISION_REJECT

    def as_dict(self) -> dict[str, Any]:
        return {
            "approve": self.approve,
            "conviction": round(self.conviction, 3),
            "exit_style": self.exit_style,
            "trailing_distance_atr": self.trailing_distance_atr,
            "rationale": self.rationale,
            "source": self.source,
        }


# --------------------------------------------------------------------------
# The rules engine
# --------------------------------------------------------------------------
class RulesProxy:
    """A deterministic reading of the rubric. Free, instant, and auditable.

    Each rubric clause becomes a term with a fixed weight, so a decision can
    always be explained by which term dominated.
    """

    #: Weights sum to 1.0. Market tone is weighted as heavily as the token's own
    #: momentum because clause 1 of the rubric says it should be.
    WEIGHTS = {
        "strength": 0.25,
        "market": 0.25,
        "regime": 0.20,
        "confluence": 0.15,
        "reward_risk": 0.15,
    }

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg

    def update_config(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg

    def decide(self, request: EntryRequest) -> GateDecision:
        cfg = self.cfg
        snap = request.snapshot or {}
        market = request.market
        regime = int(snap.get("regime", 2))
        atr_pct = float(snap.get("atr_pct") or 0.0)

        terms: dict[str, float] = {}
        terms["strength"] = _clamp(float(request.strength))
        terms["market"] = {"firm": 1.0, "mixed": 0.5, "soft": 0.0, "unknown": 0.45}[
            market.tone
        ]
        terms["regime"] = {0: 1.0, 1: 0.5, 2: 0.0}.get(regime, 0.0)

        frames = cfg.get("confluence_timeframes") or []
        agreeing = int(snap.get("confluence") or 0)
        terms["confluence"] = _clamp(agreeing / len(frames)) if frames else 0.5

        rr_min, rr_max = float(cfg["rr_min"]), float(cfg["rr_max"])
        span = max(rr_max - rr_min, 1e-9)
        terms["reward_risk"] = _clamp((float(request.rr) - rr_min) / span)

        conviction = sum(self.WEIGHTS[k] * v for k, v in terms.items())

        # Clause 2: overtrading is the failure mode. Every trade already taken
        # today raises the bar for the next one.
        cap = int(cfg.get("max_trades_per_day", 8))
        if cap > 0 and request.trades_today:
            conviction *= max(0.4, 1.0 - 0.6 * (request.trades_today / cap))

        approve = conviction >= 0.55 and market.tone != "soft"

        # Clause 4: decided per trade from the metrics, not fixed.
        room_to_run = (
            regime == TRENDING
            and market.tone == "firm"
            and (not frames or agreeing >= max(1, len(frames) - 1))
            and atr_pct <= float(cfg.get("regime_chop_atr_pct", 0.03)) * 1.5
        )
        style = STYLE_RIDE if room_to_run else STYLE_LOCK_IN
        base_trail = float(cfg["trailing_distance_atr"])
        trail = base_trail * (1.5 if style == STYLE_RIDE else 0.7)

        top = max(terms, key=lambda k: self.WEIGHTS[k] * terms[k])
        weakest = min(terms, key=lambda k: terms[k])
        if approve:
            rationale = (
                f"{market.tone} market, {REGIME_NAMES.get(regime, 'unknown')} regime; "
                f"{top.replace('_', ' ')} carried it. "
                + (
                    "Room to run, so trail wide and let it work."
                    if style == STYLE_RIDE
                    else "Not enough confirmation to ride it; tighten the trail and bank the move."
                )
            )
        else:
            rationale = (
                f"Conviction {conviction:.2f} is short of the bar - "
                f"{weakest.replace('_', ' ')} is the weak link"
                + (" and the broad market is soft." if market.tone == "soft" else ".")
            )

        return GateDecision(
            approve=approve,
            conviction=conviction,
            exit_style=style,
            trailing_distance_atr=round(trail, 3),
            rationale=rationale,
            source=SOURCE_RULES,
        )


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    try:
        return float(max(low, min(high, float(value))))
    except (TypeError, ValueError):
        return low


# --------------------------------------------------------------------------
# The gate
# --------------------------------------------------------------------------
class EntryGate:
    """Runs every candidate entry through the rules engine and records it.

    No network call, no timeout, no fallback path: the rules engine either runs
    or the gate is switched off entirely (``entry_gate_enabled``), in which case
    every signal that already passed the strategy's own checks is approved
    without comment.
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.rules = RulesProxy(cfg)

    def update_config(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.rules.update_config(cfg)

    @property
    def enabled(self) -> bool:
        return bool(self.cfg.get("entry_gate_enabled", True))

    def review_entry(
        self, request: EntryRequest, *, conn: sqlite3.Connection | None = None
    ) -> GateDecision:
        conn = conn or db.connect()
        if not self.enabled:
            decision = GateDecision(approve=True, source=SOURCE_DISABLED,
                                     rationale="entry review gate is disabled")
            self._record(request, decision, conn)
            return decision

        # A weak signal is not worth the gate's time: the rubric would reject it
        # anyway.
        if request.strength < float(self.cfg.get("entry_gate_min_strength", 0.0)):
            decision = self.rules.decide(request)
            self._record(request, decision, conn)
            return decision

        decision = self.rules.decide(request)
        self._record(request, decision, conn)
        return decision

    def _record(
        self, request: EntryRequest, decision: GateDecision, conn: sqlite3.Connection
    ) -> None:
        # The full request (signal readings, the strategy's own plain-language
        # `why`, token/liquidity context) alongside the gate's verdict - the
        # explainability dashboard's whole point is showing what was actually
        # seen, not just what was decided.
        conn.execute(
            "INSERT INTO entry_reviews(ts, kind, instance, mint, decision, "
            "conviction, exit_style, detail) VALUES (?,?,?,?,?,?,?,?)",
            (
                db.now(), "gate", request.instance, request.mint,
                decision.decision, decision.conviction, decision.exit_style,
                json.dumps({"decision": decision.as_dict(), **request.as_dict()}),
            ),
        )
