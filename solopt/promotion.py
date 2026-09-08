"""Turning an accepted run into a parameter bundle the droplet can act on.

The bundle is the only thing that crosses from the optimizer PC to the trading
droplet, and it deliberately carries its own evidence: the walk-forward summary,
the Monte Carlo distribution, and every stress window's result travel with the
parameters. That is what lets the droplet refuse a bundle on its own terms
instead of trusting whatever produced it, and what lets the dashboard explain a
promotion after the fact rather than just announcing one.

Nothing here promotes anything to live. The optimizer's authority ends at
"this set is worth running in shadow"; the fifteen clean days and the decisive
margin are judged on the droplet, against the live instance it would replace.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from . import __version__
from .montecarlo import MonteCarloResult
from .stress import StressResult
from .walkforward import WalkForwardResult

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1
BUNDLE_FILENAME = "current.json"
HISTORY_DIRNAME = "history"

# Position sizing is fed the Monte Carlo tail, not the historical drawdown. This
# is the share of the account the operator is willing to see underwater in the
# 5%-worst case; the sizing target is scaled down until the tail fits inside it.
DEFAULT_DRAWDOWN_TOLERANCE = 0.25


@dataclass
class ParameterBundle:
    """What travels through the hand-off repository."""

    global_params: dict[str, Any]
    per_symbol_params: dict[str, dict[str, Any]] = field(default_factory=dict)
    evidence: dict[str, Any] = field(default_factory=dict)
    risk: dict[str, Any] = field(default_factory=dict)
    gates: dict[str, Any] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)
    run: dict[str, Any] = field(default_factory=dict)
    generated_at: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA_VERSION,
            "generated_at": self.generated_at or int(time.time()),
            "generator": f"solopt {__version__}",
            "run": self.run,
            "data": self.data,
            "global": self.global_params,
            "per_symbol": self.per_symbol_params,
            "evidence": self.evidence,
            "risk": self.risk,
            "gates": self.gates,
        }

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), indent=2, sort_keys=True, default=float)

    def fingerprint(self) -> str:
        """Stable identity for this parameter set, ignoring the evidence."""
        import hashlib

        payload = json.dumps(
            {"global": self.global_params, "per_symbol": self.per_symbol_params},
            sort_keys=True,
            default=float,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


def recommended_vol_target(
    base_target: float, p5_drawdown: float, tolerance: float = DEFAULT_DRAWDOWN_TOLERANCE
) -> float:
    """Shrink the portfolio volatility target until the 5% tail fits tolerance.

    A run whose worst-5% drawdown lands at 40% against a 25% tolerance gets its
    volatility budget multiplied by 25/40. The historical drawdown does not enter
    into it: it is one sample from the distribution this number describes.
    """
    if p5_drawdown <= 0 or tolerance <= 0 or p5_drawdown <= tolerance:
        return float(base_target)
    return float(base_target * max(0.25, tolerance / p5_drawdown))


def build_bundle(
    outcome: WalkForwardResult,
    *,
    monte_carlo: MonteCarloResult | None = None,
    stress: Sequence[StressResult] = (),
    coverage: dict[str, Any] | None = None,
    run_meta: dict[str, Any] | None = None,
    base_vol_target: float = 0.03,
    drawdown_tolerance: float = DEFAULT_DRAWDOWN_TOLERANCE,
    shadow_min_days: int = 15,
    shadow_min_margin: float = 0.25,
) -> ParameterBundle:
    """Assemble the bundle, with the gates already evaluated."""
    fragile_names = [r.name for r in stress if not r.passed]
    p5 = monte_carlo.p5_max_drawdown if monte_carlo else 0.0

    bundle = ParameterBundle(
        global_params=dict(outcome.best_params or {}),
        per_symbol_params={k: dict(v) for k, v in outcome.per_symbol_params.items()},
        generated_at=int(time.time()),
        run=dict(run_meta or {}),
        data=dict(coverage or {}),
        evidence={
            "walk_forward": outcome.summary(),
            "monte_carlo": monte_carlo.summary() if monte_carlo else None,
            "stress": [r.as_dict() for r in stress],
        },
        risk={
            "p5_max_drawdown": round(p5, 6),
            "drawdown_tolerance": drawdown_tolerance,
            "recommended_portfolio_vol_target": round(
                recommended_vol_target(base_vol_target, p5, drawdown_tolerance), 6
            ),
            "sizing_mode": "portfolio",
        },
        gates={
            "accepted": bool(outcome.accepted) and not fragile_names,
            "walk_forward_accepted": bool(outcome.accepted),
            "overfit": bool(outcome.overfit),
            "fragile": bool(fragile_names),
            "fragile_windows": fragile_names,
            "reasons": list(outcome.reasons)
            + (
                [f"failed the crash replay on {', '.join(fragile_names)}"]
                if fragile_names
                else []
            ),
            "shadow_min_days": int(shadow_min_days),
            "shadow_min_margin": float(shadow_min_margin),
        },
    )
    return bundle


def promotable(bundle: ParameterBundle) -> tuple[bool, str]:
    """Whether this bundle may be offered to the droplet's shadow instance."""
    gates = bundle.gates
    if not bundle.global_params:
        return False, "the run produced no parameter set"
    if gates.get("fragile"):
        return False, (
            "fragile: failed the crash replay on "
            + ", ".join(gates.get("fragile_windows", []))
        )
    if gates.get("overfit"):
        return False, "flagged overfit by the walk-forward thresholds"
    if not gates.get("walk_forward_accepted"):
        reasons = gates.get("reasons") or ["did not meet the walk-forward thresholds"]
        return False, "; ".join(reasons)
    return True, "meets every walk-forward, Monte Carlo and stress gate"
