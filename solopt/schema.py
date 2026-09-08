"""Asset-class description: field names, bar cadence, and the cost model.

The engine is deliberately generic. It works on "symbols" and "bars"; it has no
idea whether a symbol is a Solana mint, a NYSE ticker or a currency pair. Adding
an asset class later means adding a :class:`Schema` here and a loader that emits
the same column names, not touching the engine.

The one thing that genuinely differs between asset classes and *cannot* be
abstracted away is the cost model. A crypto AMM charges a percentage per side
and slips against pool depth; an equity broker charges per share against a
spread; forex is quoted in pips. So :class:`CostModel` carries all three terms
and each schema fills in the ones that apply.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

# Canonical column names. Loaders rename whatever the source calls these.
CANONICAL_COLUMNS = ("ts", "open", "high", "low", "close", "volume")


@dataclass(frozen=True, slots=True)
class CostModel:
    """Round-trip execution cost, expressed per side.

    ``fee_pct`` and ``slippage_pct`` are percentages of notional (0.25 = 0.25%).
    ``fee_per_unit`` is a per-share/per-contract commission for asset classes
    that charge that way; it is zero for crypto.
    """

    fee_pct: float = 0.25
    slippage_pct: float = 0.25
    fee_per_unit: float = 0.0
    min_fee: float = 0.0

    def side_cost_pct(self) -> float:
        return (self.fee_pct + self.slippage_pct) / 100.0

    def round_trip_pct(self) -> float:
        return 2.0 * self.side_cost_pct()

    def as_dict(self) -> dict[str, float]:
        return {
            "fee_pct": self.fee_pct,
            "slippage_pct": self.slippage_pct,
            "fee_per_unit": self.fee_per_unit,
            "min_fee": self.min_fee,
        }


@dataclass(frozen=True, slots=True)
class Schema:
    """How one asset class's data is shaped and what trading it costs."""

    asset_class: str = "crypto"
    quote_currency: str = "USD"

    # The interval actually stored on disk. Everything longer is rolled up from
    # it rather than pulled separately, so a 5m and a 15m test are guaranteed to
    # be describing the same underlying prints.
    base_seconds: int = 60

    # Bars per calendar day that can exist. 24/7 markets fill every slot; a
    # session-bound market leaves gaps, which the loader marks invalid rather
    # than forward-filling (a forward-filled bar produces phantom signals).
    continuous: bool = True
    session_minutes: int = 1440

    columns: tuple[str, ...] = CANONICAL_COLUMNS
    source_columns: dict[str, str] = field(default_factory=dict)

    costs: CostModel = field(default_factory=CostModel)

    # Universe eligibility floors, in quote currency. Used to rebuild the
    # point-in-time universe from snapshots; never applied to today's data.
    min_liquidity: float = 1_000_000.0
    min_volume_24h: float = 500_000.0

    def rename_map(self) -> dict[str, str]:
        """``{source name: canonical name}`` for the loader."""
        return {src: dst for dst, src in self.source_columns.items() if src}

    def bars_per(self, seconds: int) -> int:
        """How many base bars roll up into one bar of ``seconds``."""
        if seconds % self.base_seconds:
            raise ValueError(
                f"{seconds}s is not a whole multiple of the {self.base_seconds}s base bar"
            )
        return seconds // self.base_seconds

    def with_costs(self, **kwargs: Any) -> "Schema":
        return replace(self, costs=replace(self.costs, **kwargs))

    def as_dict(self) -> dict[str, Any]:
        return {
            "asset_class": self.asset_class,
            "quote_currency": self.quote_currency,
            "base_seconds": self.base_seconds,
            "continuous": self.continuous,
            "session_minutes": self.session_minutes,
            "costs": self.costs.as_dict(),
            "min_liquidity": self.min_liquidity,
            "min_volume_24h": self.min_volume_24h,
        }


# --------------------------------------------------------------------------
# Built-ins. Crypto is the only one wired end-to-end today; the other two exist
# so the generic path is exercised rather than merely claimed.
# --------------------------------------------------------------------------
CRYPTO = Schema(
    asset_class="crypto",
    base_seconds=60,
    continuous=True,
    session_minutes=1440,
    costs=CostModel(fee_pct=0.25, slippage_pct=0.25),
    min_liquidity=1_000_000.0,
    min_volume_24h=500_000.0,
)

EQUITY = Schema(
    asset_class="equity",
    base_seconds=60,
    continuous=False,
    session_minutes=390,          # 09:30-16:00 ET regular hours
    costs=CostModel(fee_pct=0.0, slippage_pct=0.02, fee_per_unit=0.005, min_fee=1.0),
    min_liquidity=0.0,
    min_volume_24h=1_000_000.0,
)

FOREX = Schema(
    asset_class="fx",
    quote_currency="USD",
    base_seconds=60,
    continuous=False,
    session_minutes=1440,         # 24 hours a day, five days a week
    costs=CostModel(fee_pct=0.0, slippage_pct=0.01),
    min_liquidity=0.0,
    min_volume_24h=0.0,
)

BUILTIN: dict[str, Schema] = {s.asset_class: s for s in (CRYPTO, EQUITY, FOREX)}


def get_schema(name: str = "crypto") -> Schema:
    try:
        return BUILTIN[name]
    except KeyError:
        raise ValueError(
            f"unknown asset class {name!r}; known: {', '.join(sorted(BUILTIN))}"
        ) from None
