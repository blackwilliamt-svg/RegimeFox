"""RugCheck client - the pre-trade safety gate.

Verified against the live API on 2026-08-31; the report endpoints are keyless.

    GET https://api.rugcheck.xyz/v1/tokens/{mint}/report
    GET https://api.rugcheck.xyz/v1/tokens/{mint}/report/summary

The full report carries everything the spec's safety filter needs:
``mintAuthority`` and ``freezeAuthority`` (null once revoked), per-market
``lp.lpLockedPct``, ``topHolders[]`` with ``pct`` and an ``insider`` flag,
``totalMarketLiquidity``, a ``risks[]`` list, and a ``rugged`` boolean.

Note RugCheck's score runs *upward with risk*: 1 is clean, high numbers are bad.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from ..ratelimit import TokenBucket
from .base import ApiError, HttpClient

log = logging.getLogger(__name__)


@dataclass(slots=True)
class RugReport:
    """Only the fields the safety gate reasons about, plus the raw payload."""

    mint: str
    ok: bool = True                       # False when the report itself failed to load
    rugged: bool = False
    score: float = 0.0                    # higher = riskier
    mint_authority: str | None = None
    freeze_authority: str | None = None
    lp_locked_pct: float = 0.0
    top_holder_pct: float = 0.0
    top10_pct: float = 0.0
    insider_pct: float = 0.0
    total_holders: int = 0
    total_liquidity_usd: float = 0.0
    transfer_fee_pct: float = 0.0
    market_count: int = 0
    lp_providers: int = 0
    risks: list[dict[str, Any]] = field(default_factory=list)
    detected_at: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def mint_authority_revoked(self) -> bool:
        return not self.mint_authority

    @property
    def freeze_authority_revoked(self) -> bool:
        return not self.freeze_authority

    def danger_risks(self) -> list[str]:
        return [
            r.get("name", "unknown")
            for r in self.risks
            if str(r.get("level", "")).lower() in {"danger", "critical", "high"}
        ]


def _f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class RugCheckClient(HttpClient):
    provider = "rugcheck"
    base_url = "https://api.rugcheck.xyz"

    def __init__(self, bucket: TokenBucket, api_key: str = "", **kw: Any) -> None:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        super().__init__(bucket, headers=headers, **kw)
        self.api_key = api_key

    def set_api_key(self, api_key: str) -> None:
        self.api_key = api_key
        self.set_header("Authorization", f"Bearer {api_key}" if api_key else None)

    def report(self, mint: str, *, priority: str = "normal") -> RugReport:
        """Full report. Returns ``ok=False`` rather than raising on API failure.

        The caller treats an unreadable report as a failed check - the gate
        fails closed, so a RugCheck outage stops new entries instead of waving
        unverified tokens through.
        """
        try:
            data = self.get(f"/v1/tokens/{mint}/report", priority=priority)
        except ApiError as exc:
            log.warning("rugcheck report failed for %s: %s", mint, exc)
            return RugReport(mint=mint, ok=False, risks=[{"name": str(exc), "level": "danger"}])

        if not isinstance(data, dict):
            return RugReport(mint=mint, ok=False)

        markets = data.get("markets") or []
        # Weight LP-lock by pool size: one tiny locked pool should not vouch for
        # a token whose real liquidity sits in an unlocked one.
        #
        # Note this reads near-zero for most established tokens, and that is
        # accurate rather than a bug: concentrated-liquidity venues (Orca
        # Whirlpool, Meteora DLMM, Raydium CLMM) have no fungible LP token to
        # burn or lock, so there is nothing for RugCheck to measure. The safety
        # gate handles that case explicitly - see SafetyGate.evaluate.
        locked_usd = 0.0
        total_usd = 0.0
        for m in markets:
            lp = m.get("lp") or {}
            base_usd = _f(lp.get("baseUSD")) + _f(lp.get("quoteUSD"))
            if base_usd <= 0:
                base_usd = _f(lp.get("lpLockedUSD"))
            total_usd += base_usd
            locked_usd += base_usd * (_f(lp.get("lpLockedPct")) / 100.0)
        lp_locked_pct = (locked_usd / total_usd * 100.0) if total_usd > 0 else 0.0
        if not markets:
            lp_locked_pct = _f(data.get("lpLockedPct"))

        holders = data.get("topHolders") or []
        top_holder = max((_f(h.get("pct")) for h in holders), default=0.0)
        top10 = sum(_f(h.get("pct")) for h in holders[:10])
        insider = sum(_f(h.get("pct")) for h in holders if h.get("insider"))

        transfer_fee = data.get("transferFee") or {}

        return RugReport(
            mint=mint,
            ok=True,
            rugged=bool(data.get("rugged")),
            score=_f(data.get("score_normalised", data.get("score"))),
            mint_authority=data.get("mintAuthority"),
            freeze_authority=data.get("freezeAuthority"),
            lp_locked_pct=lp_locked_pct,
            top_holder_pct=top_holder,
            top10_pct=top10,
            insider_pct=insider,
            total_holders=int(_f(data.get("totalHolders"))),
            total_liquidity_usd=_f(data.get("totalMarketLiquidity")),
            transfer_fee_pct=_f(transfer_fee.get("pct")),
            market_count=len(markets),
            lp_providers=int(_f(data.get("totalLPProviders"))),
            risks=[r for r in (data.get("risks") or []) if isinstance(r, dict)],
            detected_at=str(data.get("detectedAt") or ""),
            raw=data,
        )

    def summary(self, mint: str, *, priority: str = "low") -> dict[str, Any]:
        """Cheap variant: score, risks and lpLockedPct only."""
        return self.get(f"/v1/tokens/{mint}/report/summary", priority=priority)

    def ping(self) -> bool:
        self.summary("So11111111111111111111111111111111111111112", priority="high")
        return True
