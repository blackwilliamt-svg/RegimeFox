"""API clients and the factory that wires them to config + rate limits."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .. import db
from ..config import Config
from ..ratelimit import MonthlyBudget, TokenBucket
from .base import ApiError, BudgetExhausted, HttpClient, RateLimited
from .birdeye import BirdeyeClient, Candle, nearest_interval
from .jupiter import JupiterClient, PriceInfo, Quote, TokenInfo, SOL_MINT, USDC_MINT
from .rugcheck import RugCheckClient, RugReport
from .solana_rpc import Congestion, SolanaRpcClient

__all__ = [
    "ApiError",
    "BudgetExhausted",
    "RateLimited",
    "HttpClient",
    "BirdeyeClient",
    "Candle",
    "nearest_interval",
    "JupiterClient",
    "PriceInfo",
    "Quote",
    "TokenInfo",
    "SOL_MINT",
    "USDC_MINT",
    "RugCheckClient",
    "RugReport",
    "SolanaRpcClient",
    "Congestion",
    "Clients",
    "build_clients",
]


@dataclass
class Clients:
    """The four API clients plus their shared limiters."""

    jupiter: JupiterClient
    birdeye: BirdeyeClient
    rugcheck: RugCheckClient
    rpc: SolanaRpcClient
    jupiter_bucket: TokenBucket
    birdeye_bucket: TokenBucket
    rugcheck_bucket: TokenBucket
    rpc_bucket: TokenBucket
    birdeye_budget: MonthlyBudget

    def apply_config(self, cfg: Config) -> None:
        """Re-apply rate limits after a live settings change."""
        # Reserve one token of Jupiter headroom so an entry/exit quote is never
        # stuck behind scan traffic.
        self.jupiter_bucket.configure(
            cfg["jupiter_rps"], cfg["jupiter_burst"], reserve=1.0
        )
        self.birdeye_bucket.configure(cfg["birdeye_rps"], 2, reserve=0.0)
        self.rugcheck_bucket.configure(cfg["rugcheck_rps"], 3, reserve=0.0)
        self.birdeye_budget.configure(cfg["birdeye_monthly_cu_budget"])

    def apply_keys(self, *, jupiter: str | None = None, birdeye: str | None = None,
                   rugcheck: str | None = None) -> None:
        if jupiter is not None:
            self.jupiter.set_api_key(jupiter)
        if birdeye is not None:
            self.birdeye.set_api_key(birdeye)
        if rugcheck is not None:
            self.rugcheck.set_api_key(rugcheck)

    def stats(self) -> dict[str, Any]:
        return {
            "jupiter": self.jupiter_bucket.stats(),
            "birdeye": self.birdeye_bucket.stats(),
            "rugcheck": self.rugcheck_bucket.stats(),
            "rpc": self.rpc_bucket.stats(),
            "birdeye_budget": self.birdeye_budget.stats(),
        }

    def close(self) -> None:
        for c in (self.jupiter, self.birdeye, self.rugcheck, self.rpc):
            c.close()


def build_clients(cfg: Config, *, track_budget: bool = True) -> Clients:
    """Construct every client from settings + environment secrets."""
    secrets = cfg.secrets

    jupiter_bucket = TokenBucket(
        cfg["jupiter_rps"], cfg["jupiter_burst"], reserve=1.0, name="jupiter"
    )
    birdeye_bucket = TokenBucket(cfg["birdeye_rps"], 2, reserve=0.0, name="birdeye")
    rugcheck_bucket = TokenBucket(cfg["rugcheck_rps"], 3, reserve=0.0, name="rugcheck")
    rpc_bucket = TokenBucket(5.0, 5, reserve=1.0, name="rpc")

    spent = db.api_usage("birdeye")["units"] if track_budget else 0
    budget = MonthlyBudget(cfg["birdeye_monthly_cu_budget"], spent=spent)

    def record_spend(units: int) -> None:
        if track_budget:
            try:
                db.add_api_usage("birdeye", units)
            except Exception:
                pass

    return Clients(
        jupiter=JupiterClient(jupiter_bucket, secrets.jupiter_api_key),
        birdeye=BirdeyeClient(
            birdeye_bucket, secrets.birdeye_api_key, budget, on_spend=record_spend
        ),
        rugcheck=RugCheckClient(rugcheck_bucket, secrets.rugcheck_api_key),
        rpc=SolanaRpcClient(rpc_bucket, secrets.solana_rpc_url),
        jupiter_bucket=jupiter_bucket,
        birdeye_bucket=birdeye_bucket,
        rugcheck_bucket=rugcheck_bucket,
        rpc_bucket=rpc_bucket,
        birdeye_budget=budget,
    )
