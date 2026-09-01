"""Jupiter API client - live prices, the token universe, and swap execution.

Endpoints verified against the live API on 2026-08-31:

    price     GET  https://api.jup.ag/price/v3?ids=<up to 50 mints>
    tokens    GET  https://api.jup.ag/tokens/v2/{category}/{interval}?limit=N
    search    GET  https://api.jup.ag/tokens/v2/search?query=<mint|symbol>
    order     GET  https://api.jup.ag/swap/v2/order
    execute   POST https://api.jup.ag/swap/v2/execute

Three things that differ from older Jupiter integrations and from the build
spec, all of which matter:

* ``lite-api.jup.ag`` was retired on 31 Jan 2026. Everything is on
  ``api.jup.ag`` now, authenticated with an ``x-api-key`` header.
* **The Price API accepts at most 50 mints per call, not 100.** The scan loop
  batches at 50.
* The old ``/quote`` + ``/swap`` pair is replaced by ``/order`` -> sign ->
  ``/execute``. Calling ``/order`` *without* a ``taker`` returns a quote with no
  transaction attached, which is exactly what paper mode and the pre-trade
  slippage check want. Passing ``taker`` returns a base64 transaction plus the
  ``requestId`` that ``/execute`` needs.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterable

from ..ratelimit import TokenBucket
from .base import ApiError, HttpClient, chunked

log = logging.getLogger(__name__)

# The API rejects more than this per call.
PRICE_BATCH_LIMIT = 50

USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
SOL_MINT = "So11111111111111111111111111111111111111112"


@dataclass(slots=True)
class PriceInfo:
    mint: str
    price: float
    decimals: int
    block_id: int = 0
    price_change_24h: float = 0.0
    liquidity: float = 0.0


@dataclass(slots=True)
class TokenInfo:
    """A universe candidate as Jupiter describes it."""

    mint: str
    symbol: str = ""
    name: str = ""
    decimals: int = 9
    price: float = 0.0
    liquidity: float = 0.0
    volume_24h: float = 0.0
    mcap: float = 0.0
    holder_count: int = 0
    organic_score: float = 0.0
    organic_label: str = ""
    is_verified: bool = False
    mint_authority: str | None = None
    freeze_authority: str | None = None
    top_holders_pct: float = 0.0
    first_pool_at: int | None = None
    tags: list[str] = field(default_factory=list)


@dataclass(slots=True)
class Quote:
    """Normalised view of a /order response."""

    input_mint: str
    output_mint: str
    in_amount: int
    out_amount: int
    min_out_amount: int
    price_impact_pct: float
    slippage_bps: int
    router: str
    swap_type: str
    request_id: str | None
    transaction: str | None
    in_usd: float
    out_usd: float
    priority_fee_lamports: int
    signature_fee_lamports: int
    raw: dict[str, Any]

    @property
    def implied_slippage_pct(self) -> float:
        """Round-trip value lost on this fill, as a positive percentage.

        Derived from the USD value in vs. out, so it captures price impact,
        the platform fee and the route's spread in one number - which is what
        the spec's slippage gate should actually be measuring.
        """
        if self.in_usd <= 0:
            return abs(self.price_impact_pct) * 100.0
        return max(0.0, (self.in_usd - self.out_usd) / self.in_usd * 100.0)


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _iso_to_epoch(value: Any) -> int | None:
    if not value or not isinstance(value, str):
        return None
    from datetime import datetime

    try:
        return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp())
    except ValueError:
        return None


class JupiterClient(HttpClient):
    provider = "jupiter"
    base_url = "https://api.jup.ag"

    def __init__(self, bucket: TokenBucket, api_key: str = "", **kw: Any) -> None:
        headers = {"x-api-key": api_key} if api_key else {}
        super().__init__(bucket, headers=headers, **kw)
        self.api_key = api_key

    def set_api_key(self, api_key: str) -> None:
        self.api_key = api_key
        self.set_header("x-api-key", api_key or None)

    # ------------------------------------------------------------------
    # Prices - the live feed for both scan tiers
    # ------------------------------------------------------------------
    def prices(
        self, mints: Iterable[str], *, priority: str = "normal"
    ) -> dict[str, PriceInfo]:
        """Batched USD prices. Tokens without a reliable price are simply absent."""
        unique = list(dict.fromkeys(m for m in mints if m))
        out: dict[str, PriceInfo] = {}
        for batch in chunked(unique, PRICE_BATCH_LIMIT):
            try:
                data = self.get(
                    "/price/v3", params={"ids": ",".join(batch)}, priority=priority
                )
            except ApiError as exc:
                log.warning("price batch of %d failed: %s", len(batch), exc)
                continue
            if not isinstance(data, dict):
                continue
            for mint, row in data.items():
                if not isinstance(row, dict):
                    continue
                price = _to_float(row.get("usdPrice"))
                if price <= 0:
                    continue
                out[mint] = PriceInfo(
                    mint=mint,
                    price=price,
                    decimals=_to_int(row.get("decimals"), 9),
                    block_id=_to_int(row.get("blockId")),
                    price_change_24h=_to_float(row.get("priceChange24h")),
                    liquidity=_to_float(row.get("liquidity")),
                )
        return out

    def price_calls_for(self, token_count: int) -> int:
        """How many requests one sweep of `token_count` tokens costs."""
        if token_count <= 0:
            return 0
        return -(-token_count // PRICE_BATCH_LIMIT)  # ceil

    # ------------------------------------------------------------------
    # Token universe
    # ------------------------------------------------------------------
    def top_tokens(
        self,
        category: str = "toptraded",
        interval: str = "24h",
        limit: int = 100,
        *,
        priority: str = "low",
    ) -> list[TokenInfo]:
        """Ranked token list used to build the scan universe.

        Categories: ``toptraded``, ``toporganicscore``, ``toptrending``.
        The response carries liquidity, 24h volume, holder count and the mint /
        freeze authorities, so one call seeds both the liquidity/volume floor
        and a cheap first-pass safety screen before RugCheck is consulted.
        """
        data = self.get(
            f"/tokens/v2/{category}/{interval}",
            params={"limit": max(1, min(int(limit), 100))},
            priority=priority,
        )
        return [self._parse_token(t) for t in data or [] if isinstance(t, dict)]

    def search(self, query: str, *, priority: str = "low") -> list[TokenInfo]:
        data = self.get("/tokens/v2/search", params={"query": query}, priority=priority)
        return [self._parse_token(t) for t in data or [] if isinstance(t, dict)]

    @staticmethod
    def _parse_token(t: dict[str, Any]) -> TokenInfo:
        stats = t.get("stats24h") or {}
        audit = t.get("audit") or {}
        first_pool = t.get("firstPool") or {}
        volume = _to_float(stats.get("buyVolume")) + _to_float(stats.get("sellVolume"))
        created = _iso_to_epoch(first_pool.get("createdAt")) or _iso_to_epoch(
            t.get("createdAt")
        )
        return TokenInfo(
            mint=t.get("id", ""),
            symbol=t.get("symbol", "") or "",
            name=t.get("name", "") or "",
            decimals=_to_int(t.get("decimals"), 9),
            price=_to_float(t.get("usdPrice")),
            liquidity=_to_float(t.get("liquidity")),
            volume_24h=volume,
            mcap=_to_float(t.get("mcap")),
            holder_count=_to_int(t.get("holderCount")),
            organic_score=_to_float(t.get("organicScore")),
            organic_label=t.get("organicScoreLabel", "") or "",
            is_verified=bool(t.get("isVerified")),
            mint_authority=t.get("mintAuthority"),
            freeze_authority=t.get("freezeAuthority"),
            top_holders_pct=_to_float(audit.get("topHoldersPercentage")),
            first_pool_at=created,
            tags=list(t.get("tags") or []),
        )

    # ------------------------------------------------------------------
    # Swap
    # ------------------------------------------------------------------
    def order(
        self,
        input_mint: str,
        output_mint: str,
        amount: int,
        *,
        slippage_bps: int,
        taker: str | None = None,
        priority_fee_lamports: int | None = None,
        priority: str = "high",
    ) -> Quote:
        """Get a quote, and a signable transaction when `taker` is supplied.

        Quotes for the pre-trade slippage gate omit `taker` - cheaper, and it
        avoids the "Missing associated token account" error for a token the
        wallet has never held.
        """
        params: dict[str, Any] = {
            "inputMint": input_mint,
            "outputMint": output_mint,
            "amount": str(int(amount)),
            "slippageBps": int(slippage_bps),
        }
        if taker:
            params["taker"] = taker
        if priority_fee_lamports is not None:
            params["prioritizationFeeLamports"] = int(priority_fee_lamports)

        data = self.get("/swap/v2/order", params=params, priority=priority)
        if not isinstance(data, dict):
            raise ApiError("jupiter: malformed order response", provider=self.provider)
        if data.get("error"):
            raise ApiError(
                f"jupiter order: {data.get('errorMessage') or data['error']} "
                f"(code {data.get('errorCode')})",
                provider=self.provider,
            )

        in_amt = _to_int(data.get("inAmount"))
        out_amt = _to_int(data.get("outAmount"))
        return Quote(
            input_mint=data.get("inputMint", input_mint),
            output_mint=data.get("outputMint", output_mint),
            in_amount=in_amt,
            out_amount=out_amt,
            min_out_amount=_to_int(data.get("otherAmountThreshold"), out_amt),
            price_impact_pct=_to_float(data.get("priceImpactPct")),
            slippage_bps=_to_int(data.get("slippageBps"), slippage_bps),
            router=data.get("router", "") or "",
            swap_type=data.get("swapType", "") or "",
            request_id=data.get("requestId"),
            transaction=data.get("transaction"),
            in_usd=_to_float(data.get("inUsdValue")),
            out_usd=_to_float(data.get("outUsdValue")),
            priority_fee_lamports=_to_int(data.get("prioritizationFeeLamports")),
            signature_fee_lamports=_to_int(data.get("signatureFeeLamports")),
            raw=data,
        )

    def execute(self, signed_transaction_b64: str, request_id: str) -> dict[str, Any]:
        """Submit a signed transaction through Jupiter's managed landing.

        This endpoint has its own rate-limit bucket upstream, but it is still
        marked high priority locally so an exit never waits behind scan traffic.
        """
        data = self.post(
            "/swap/v2/execute",
            json_body={
                "signedTransaction": signed_transaction_b64,
                "requestId": request_id,
            },
            priority="high",
        )
        if not isinstance(data, dict):
            raise ApiError("jupiter: malformed execute response", provider=self.provider)
        return data

    def ping(self) -> bool:
        """Cheap key validation for the settings page's rotation check."""
        self.get("/price/v3", params={"ids": SOL_MINT}, priority="high")
        return True
