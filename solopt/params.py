"""The parameter search space, and how the search decides it is finished.

Two things live here. :class:`ParamSpace` describes what may be tuned and turns
that into batches of concrete combinations. :class:`Search` drives the sampling
and holds the stopping rule the spec asks for: a fixed iteration cap, or early
exit once the best result stops improving.

The plateau rule matters more than the cap. A search that keeps going after the
objective has flattened is not finding a better strategy, it is finding a luckier
arrangement of the same noise - which is the mechanism that produces an overfit
parameter set. Stopping when improvement dies is a cheap defence against that,
and the walk-forward thresholds are the expensive one.
"""
from __future__ import annotations

import itertools
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence

import numpy as np

log = logging.getLogger(__name__)

# Nine indicators the search can independently include/exclude from an
# entry's vote (gap-closure item 3). Mirrors solbot.indicators.INDICATOR_BITS
# exactly - duplicated rather than imported because solopt imports no solbot
# (see the layout note in README.md); tests/test_vector_parity.py's
# test_indicator_bits_mirror_solbot catches the two ever drifting apart.
IND_VOLUME_SPIKE = 1 << 0
IND_MOMENTUM = 1 << 1
IND_RSI = 1 << 2
IND_EMA_CROSS = 1 << 3
IND_MACD = 1 << 4
IND_BBANDS = 1 << 5
IND_STOCHASTIC = 1 << 6
IND_ADX = 1 << 7
IND_VWAP = 1 << 8
ALL_INDICATOR_BITS = 0x1FF
LEGACY_INDICATOR_MASK = IND_VOLUME_SPIKE | IND_MOMENTUM | IND_RSI | IND_EMA_CROSS


def popcount(mask: int) -> int:
    return bin(int(mask) & 0xFFFFFFFF).count("1")


# Everything the optimizer is allowed to tune. Anything not listed here is a
# risk control or a venue constraint, and is deliberately not up for search.
TUNABLE: dict[str, tuple[type, float, float]] = {
    "volume_spike_multiple": (float, 1.1, 20.0),
    "volume_spike_lookback": (int, 5, 200),
    "momentum_candles": (int, 2, 10),
    "momentum_min_pct": (float, 0.001, 0.20),
    "rsi_period": (int, 2, 50),
    "rsi_max_entry": (float, 50.0, 95.0),
    "ema_fast": (int, 2, 50),
    "ema_slow": (int, 3, 200),
    "atr_period": (int, 2, 50),
    "stop_atr_mult": (float, 0.3, 5.0),
    "rr_min": (float, 1.5, 4.0),
    "rr_max": (float, 2.0, 6.0),
    "trailing_activate_r": (float, 0.1, 5.0),
    "trailing_distance_atr": (float, 0.2, 10.0),
    "signal_invalidation_bars": (int, 1, 10),
    "max_hold_minutes": (int, 10, 10080),
    "regime_lookback": (int, 5, 200),
    "regime_trend_er": (float, 0.05, 0.95),
    "regime_chop_atr_pct": (float, 0.001, 0.50),
    "regime_allowed": (int, 1, 7),          # bitmask: 1 trending | 2 ranging | 4 choppy
    "confluence_required": (int, 0, 4),
    # --- indicator-combination search (gap-closure item 3) ----------------
    "indicator_mask": (int, 0, ALL_INDICATOR_BITS),
    "indicator_min_agree": (int, 0, 9),
    "macd_fast": (int, 2, 100),
    "macd_slow": (int, 3, 200),
    "macd_signal": (int, 1, 100),
    "bb_period": (int, 2, 200),
    "bb_std": (float, 0.5, 5.0),
    "stoch_k_period": (int, 2, 200),
    "stoch_d_period": (int, 1, 50),
    "adx_period": (int, 2, 100),
    "adx_min": (float, 0.0, 80.0),
    "vwap_period": (int, 2, 500),
}

# Sensible starting grid. Coarse on purpose: a fine grid over sixteen axes is a
# curve-fitting machine, and the walk-forward windows are what earn the right to
# narrow it later.
DEFAULT_GRID: dict[str, list[Any]] = {
    "volume_spike_multiple": [1.8, 2.2, 3.0, 4.0],
    "volume_spike_lookback": [20, 40],
    "momentum_candles": [3, 4],
    "momentum_min_pct": [0.006, 0.010, 0.016],
    "rsi_period": [14],
    "rsi_max_entry": [72.0, 78.0, 84.0],
    "ema_fast": [9, 12],
    "ema_slow": [21, 34],
    "atr_period": [14],
    "stop_atr_mult": [1.0, 1.2, 1.6],
    "rr_min": [2.0, 2.5],
    "rr_max": [4.0, 5.0],
    "trailing_activate_r": [1.0, 1.5],
    "trailing_distance_atr": [1.5, 2.2],
    "signal_invalidation_bars": [2, 3],
    "max_hold_minutes": [240, 480],
    "regime_lookback": [20],
    "regime_trend_er": [0.30, 0.40],
    "regime_chop_atr_pct": [0.03],
    "regime_allowed": [1, 3],
    "confluence_required": [1, 2],
    # A curated set of masks rather than a uniform sample of the 512 possible
    # ones: the legacy four, the legacy four plus one new indicator at a time,
    # a combo of only new indicators, and every indicator active. Between
    # those and indicator_min_agree, is_valid() (below) prunes any pairing
    # that could never fire (min_agree above the mask's own popcount).
    "indicator_mask": [
        LEGACY_INDICATOR_MASK,
        LEGACY_INDICATOR_MASK | IND_MACD,
        LEGACY_INDICATOR_MASK | IND_ADX | IND_VWAP,
        IND_MACD | IND_BBANDS | IND_STOCHASTIC,
        ALL_INDICATOR_BITS,
    ],
    "indicator_min_agree": [1, 2, 3, 4],
    "macd_fast": [8, 12],
    "macd_slow": [21, 26],
    "macd_signal": [9],
    "bb_period": [14, 20],
    "bb_std": [1.5, 2.0, 2.5],
    "stoch_k_period": [9, 14],
    "stoch_d_period": [3],
    "adx_period": [14],
    "adx_min": [15.0, 20.0, 25.0],
    "vwap_period": [20],
}


class InvalidCombo(ValueError):
    """A sampled combination violates a cross-parameter rule."""


def is_valid(combo: dict[str, Any]) -> bool:
    """Cross-field rules, mirroring the live config's validation.

    A combination that could never be saved to ``config.json`` must never be
    promoted either, so the same relationships are enforced at search time
    rather than discovered at hand-off time.
    """
    if combo["ema_fast"] >= combo["ema_slow"]:
        return False
    if combo["rr_min"] > combo["rr_max"]:
        return False
    if combo["momentum_candles"] > combo["volume_spike_lookback"]:
        return False
    if "macd_fast" in combo and "macd_slow" in combo:
        if combo["macd_fast"] >= combo["macd_slow"]:
            return False
    if "indicator_mask" in combo and "indicator_min_agree" in combo:
        if combo["indicator_min_agree"] > popcount(combo["indicator_mask"]):
            return False
    return True


@dataclass
class ParamSpace:
    """Named axes and their candidate values."""

    values: dict[str, list[Any]] = field(default_factory=lambda: dict(DEFAULT_GRID))
    fixed: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        unknown = set(self.values) - set(TUNABLE)
        if unknown:
            raise ValueError(f"not tunable: {', '.join(sorted(unknown))}")
        for name, options in self.values.items():
            if not options:
                raise ValueError(f"axis {name} has no candidate values")

    @property
    def names(self) -> list[str]:
        return sorted(self.values)

    def size(self) -> int:
        total = 1
        for options in self.values.values():
            total *= len(options)
        return total

    def full(self) -> list[dict[str, Any]]:
        """Every valid combination. Only sane when :meth:`size` is small."""
        names = self.names
        out = []
        for point in itertools.product(*(self.values[n] for n in names)):
            combo = {**self.fixed, **dict(zip(names, point))}
            if is_valid(combo):
                out.append(combo)
        return out

    def sample(self, count: int, rng: np.random.Generator) -> list[dict[str, Any]]:
        """Random combinations, de-duplicated and rule-checked."""
        names = self.names
        seen: set[tuple] = set()
        out: list[dict[str, Any]] = []
        # Rejection sampling can stall if most of the space is invalid; give up
        # after a bounded number of tries rather than spinning.
        for _ in range(count * 20):
            if len(out) >= count:
                break
            point = tuple(
                self.values[n][int(rng.integers(len(self.values[n])))] for n in names
            )
            if point in seen:
                continue
            seen.add(point)
            combo = {**self.fixed, **dict(zip(names, point))}
            if is_valid(combo):
                out.append(combo)
        return out

    def neighbours(
        self, combo: dict[str, Any], rng: np.random.Generator, count: int
    ) -> list[dict[str, Any]]:
        """Combinations one step away on one axis - the refinement phase.

        Moving a single axis at a time is what makes the result interpretable:
        when a refined set beats the coarse one, it is clear which knob did it.
        """
        names = self.names
        out: list[dict[str, Any]] = []
        seen = {tuple(combo[n] for n in names)}
        for _ in range(count * 20):
            if len(out) >= count:
                break
            axis = names[int(rng.integers(len(names)))]
            options = self.values[axis]
            try:
                here = options.index(combo[axis])
            except ValueError:
                here = 0
            step = 1 if rng.random() < 0.5 else -1
            nxt = min(len(options) - 1, max(0, here + step))
            if nxt == here:
                continue
            candidate = dict(combo)
            candidate[axis] = options[nxt]
            key = tuple(candidate[n] for n in names)
            if key in seen or not is_valid(candidate):
                continue
            seen.add(key)
            out.append(candidate)
        return out

    def requirements(self, combos: Sequence[dict[str, Any]]) -> dict[str, list[Any]]:
        """The distinct indicator parameters this set of combinations needs."""
        def distinct(key: str) -> list[Any]:
            return sorted({c[key] for c in combos})

        return {
            "ema": sorted({c["ema_fast"] for c in combos} | {c["ema_slow"] for c in combos}),
            "rsi": distinct("rsi_period"),
            "atr": distinct("atr_period"),
            "vol_ratio": distinct("volume_spike_lookback"),
            "momentum": distinct("momentum_candles"),
            "efficiency": distinct("regime_lookback"),
        }


# --------------------------------------------------------------------------
# Search driver
# --------------------------------------------------------------------------
@dataclass
class SearchState:
    """Where a search has got to. Serialised into the resume checkpoint."""

    evaluated: int = 0
    best_score: float = -math.inf
    best_combo: dict[str, Any] | None = None
    flat_rounds: int = 0
    rounds: int = 0
    stopped: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "evaluated": self.evaluated,
            "best_score": None if self.best_score == -math.inf else round(self.best_score, 6),
            "best_combo": self.best_combo,
            "rounds": self.rounds,
            "flat_rounds": self.flat_rounds,
            "stopped": self.stopped,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SearchState":
        return cls(
            evaluated=int(data.get("evaluated", 0)),
            best_score=(
                -math.inf if data.get("best_score") is None else float(data["best_score"])
            ),
            best_combo=data.get("best_combo"),
            flat_rounds=int(data.get("flat_rounds", 0)),
            rounds=int(data.get("rounds", 0)),
            stopped=data.get("stopped", ""),
        )


@dataclass
class Search:
    """Batched search with a cap and a plateau stop."""

    space: ParamSpace
    max_evaluations: int = 2000
    batch_size: int = 64
    plateau_rounds: int = 4
    plateau_epsilon: float = 0.005     # relative improvement that counts as progress
    refine_after: int = 3              # rounds of random sampling before refining
    seed: int = 20260903
    state: SearchState = field(default_factory=SearchState)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed + self.state.rounds)

    def batches(self) -> Iterator[list[dict[str, Any]]]:
        """Yield combinations to evaluate until the cap or the plateau stops us.

        The caller feeds results back through :meth:`observe`; the generator
        reads the updated state before deciding whether to yield again.
        """
        exhaustive = self.space.size() <= self.max_evaluations
        pool = self.space.full() if exhaustive else []
        cursor = 0

        while self.state.evaluated < self.max_evaluations:
            if self.state.flat_rounds >= self.plateau_rounds:
                self.state.stopped = (
                    f"plateau: no better than {self.state.best_score:.4f} for "
                    f"{self.state.flat_rounds} rounds"
                )
                return

            remaining = self.max_evaluations - self.state.evaluated
            take = min(self.batch_size, remaining)
            if exhaustive:
                if cursor >= len(pool):
                    self.state.stopped = "search space exhausted"
                    return
                batch = pool[cursor : cursor + take]
                cursor += len(batch)
            elif self.state.rounds >= self.refine_after and self.state.best_combo:
                batch = self.space.neighbours(self.state.best_combo, self._rng, take)
                if len(batch) < take:
                    batch += self.space.sample(take - len(batch), self._rng)
            else:
                batch = self.space.sample(take, self._rng)

            if not batch:
                self.state.stopped = "no further combinations available"
                return
            yield batch

        self.state.stopped = f"reached the {self.max_evaluations}-combination cap"

    def observe(self, scored: Sequence[tuple[dict[str, Any], float]]) -> bool:
        """Record a batch's results. Returns True if the best improved."""
        self.state.rounds += 1
        self.state.evaluated += len(scored)
        improved = False
        for combo, score in scored:
            if not math.isfinite(score):
                continue
            threshold = self.state.best_score
            if threshold == -math.inf:
                gain = math.inf
            else:
                gain = (score - threshold) / max(abs(threshold), 1e-9)
            if score > threshold and (threshold == -math.inf or gain >= self.plateau_epsilon):
                improved = True
            if score > self.state.best_score:
                self.state.best_score = score
                self.state.best_combo = dict(combo)
        self.state.flat_rounds = 0 if improved else self.state.flat_rounds + 1
        return improved
