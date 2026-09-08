"""API clients and the factory that wires them to config + rate limits."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..config import Config
from ..ratelimit import TokenBucket
from .base import ApiError, HttpClient, RateLimited
from .binance import BinanceAsset, BinanceClient, Candle
from .jito import JitoClient
from .jupiter import JupiterClient, PriceInfo, Quote, TokenInfo, SOL_MINT, USDC_MINT
from .rugcheck import RugCheckClient, RugReport
from .solana_rpc import Congestion, SolanaRpcClient

__all__ = [
    "ApiError",
    "RateLimited",
    "HttpClient",
    "BinanceAsset",
    "BinanceClient",
    "Candle",
    "JitoClient",
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
    """The API clients plus their shared limiters."""

    jupiter: JupiterClient
    rugcheck: RugCheckClient
    rpc: SolanaRpcClient
    binance: BinanceClient
    jito: JitoClient
    jupiter_bucket: TokenBucket
    rugcheck_bucket: TokenBucket
    rpc_bucket: TokenBucket
    binance_bucket: TokenBucket
    jito_bucket: TokenBucket

    def apply_config(self, cfg: Config) -> None:
        """Re-apply rate limits after a live settings change."""
        # Reserve one token of Jupiter headroom so an entry/exit quote is never
        # stuck behind scan traffic.
        self.jupiter_bucket.configure(
            cfg["jupiter_rps"], cfg["jupiter_burst"], reserve=1.0
        )
        self.rugcheck_bucket.configure(cfg["rugcheck_rps"], 3, reserve=0.0)
        self.binance_bucket.configure(cfg["binance_rps"], 3, reserve=0.0)

    def apply_keys(self, *, jupiter: str | None = None, rugcheck: str | None = None) -> None:
        if jupiter is not None:
            self.jupiter.set_api_key(jupiter)
        if rugcheck is not None:
            self.rugcheck.set_api_key(rugcheck)

    def stats(self) -> dict[str, Any]:
        return {
            "jupiter": self.jupiter_bucket.stats(),
            "rugcheck": self.rugcheck_bucket.stats(),
            "rpc": self.rpc_bucket.stats(),
            "binance": self.binance_bucket.stats(),
            "jito": self.jito_bucket.stats(),
        }

    def close(self) -> None:
        for c in (self.jupiter, self.rugcheck, self.rpc, self.binance, self.jito):
            c.close()


def build_clients(cfg: Config) -> Clients:
    """Construct every client from settings + environment secrets."""
    secrets = cfg.secrets

    jupiter_bucket = TokenBucket(
        cfg["jupiter_rps"], cfg["jupiter_burst"], reserve=1.0, name="jupiter"
    )
    rugcheck_bucket = TokenBucket(cfg["rugcheck_rps"], 3, reserve=0.0, name="rugcheck")
    rpc_bucket = TokenBucket(5.0, 5, reserve=1.0, name="rpc")
    # Binance is keyless and generously weight-limited for this bot's call
    # volume: a top-coin ranking pull once per universe refresh, and a klines
    # pull once per coin per day for the incremental candle backfill.
    binance_bucket = TokenBucket(cfg["binance_rps"], 3, reserve=0.0, name="binance")
    # Only ever called on the live-trading path, at most once per swap, so a
    # generous burst with no sustained-rate concern is fine. A short timeout
    # keeps a slow/unreachable block engine from stalling a trade for long
    # before LiveExecutor falls back to plain submission.
    jito_bucket = TokenBucket(5.0, 5, reserve=0.0, name="jito")

    return Clients(
        jupiter=JupiterClient(jupiter_bucket, secrets.jupiter_api_key),
        rugcheck=RugCheckClient(rugcheck_bucket, secrets.rugcheck_api_key),
        rpc=SolanaRpcClient(rpc_bucket, secrets.solana_rpc_url),
        binance=BinanceClient(binance_bucket),
        jito=JitoClient(jito_bucket, timeout=8.0, max_retries=1),
        jupiter_bucket=jupiter_bucket,
        rugcheck_bucket=rugcheck_bucket,
        rpc_bucket=rpc_bucket,
        binance_bucket=binance_bucket,
        jito_bucket=jito_bucket,
    )
