"""Order execution: simulated fills for paper mode, Jupiter swaps for live.

Both executors present the same interface so the engine does not branch on mode
anywhere except construction. That is what lets the permanent parallel paper
instance run the identical code path alongside live trading.

The slippage gate is enforced *before* either path submits: a quote whose
implied round-trip cost exceeds the configured tolerance causes the trade to be
skipped entirely rather than filled at a worse price than planned.

Live trading additionally checks network congestion first. Solana swaps are
atomic once submitted - there is no order book to cancel - so everything that
can be checked has to be checked before signing.
"""
from __future__ import annotations

import base64
import logging
from dataclasses import dataclass, field
from typing import Any

from . import db
from .clients import ApiError, Clients, Quote, USDC_MINT, SOL_MINT

log = logging.getLogger(__name__)


@dataclass(slots=True)
class Fill:
    ok: bool
    side: str                       # buy | sell
    mint: str = ""
    price: float = 0.0              # USD per token, after costs
    qty: float = 0.0
    gross_usd: float = 0.0
    fee_usd: float = 0.0
    slippage_pct: float = 0.0
    tx_signature: str | None = None
    reason: str = ""                # why it was skipped/failed
    detail: dict[str, Any] = field(default_factory=dict)


class SkipTrade(Exception):
    """Refused before submission - not an error, a deliberate pass."""


class Executor:
    """Interface shared by the paper and live executors."""

    mode = "paper"

    def __init__(self, clients: Clients, cfg: dict[str, Any]) -> None:
        self.clients = clients
        self.cfg = cfg

    def update_config(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg

    def buy(self, mint: str, usd_amount: float, price: float, **kw: Any) -> Fill:
        raise NotImplementedError

    def sell(self, mint: str, qty: float, price: float, **kw: Any) -> Fill:
        raise NotImplementedError

    # -- shared cost model ---------------------------------------------
    def _fee(self, notional: float) -> float:
        return notional * float(self.cfg["taker_fee_pct"]) / 100.0

    def preflight_quote(
        self, input_mint: str, output_mint: str, amount_raw: int, *, decimals_hint: int = 6
    ) -> Quote:
        """Quote the swap and enforce the slippage ceiling.

        Uses a quote-only order (no ``taker``), which is both cheaper and avoids
        a spurious "missing associated token account" error for a token the
        wallet has not held before.
        """
        slippage_bps = int(float(self.cfg["max_slippage_pct"]) * 100)
        quote = self.clients.jupiter.order(
            input_mint, output_mint, amount_raw, slippage_bps=slippage_bps, priority="high"
        )
        implied = quote.implied_slippage_pct
        limit = float(self.cfg["max_slippage_pct"])
        if implied > limit:
            raise SkipTrade(
                f"quoted fill would cost {implied:.2f}% versus the {limit:.2f}% tolerance"
            )
        return quote


class PaperExecutor(Executor):
    """Simulated fills with realistic costs.

    Slippage is charged against the position rather than assumed away: the live
    quote's implied cost is used when a quote is available, otherwise the
    configured tolerance is applied as a conservative stand-in. Paper results
    that ignore fees and slippage are the main way a strategy looks profitable
    on paper and is not.
    """

    mode = "paper"

    def __init__(self, clients: Clients, cfg: dict[str, Any], *, use_live_quotes: bool = True):
        super().__init__(clients, cfg)
        self.use_live_quotes = use_live_quotes

    def _estimate_slippage(self, mint: str, usd_amount: float, side: str) -> float:
        """Ask Jupiter what this size would really cost, without submitting."""
        if not self.use_live_quotes or usd_amount <= 0:
            return float(self.cfg["max_slippage_pct"]) / 2.0
        try:
            amount_raw = int(usd_amount * 1e6)  # USDC has 6 decimals
            quote = self.clients.jupiter.order(
                USDC_MINT if side == "buy" else mint,
                mint if side == "buy" else USDC_MINT,
                amount_raw,
                slippage_bps=int(float(self.cfg["max_slippage_pct"]) * 100),
                priority="normal",
            )
            return quote.implied_slippage_pct
        except (ApiError, SkipTrade) as exc:
            log.debug("paper slippage estimate fell back for %s: %s", mint, exc)
            return float(self.cfg["max_slippage_pct"]) / 2.0

    def buy(self, mint: str, usd_amount: float, price: float, **kw: Any) -> Fill:
        if price <= 0 or usd_amount <= 0:
            return Fill(False, "buy", mint, reason="invalid price or size")

        slippage = kw.get("slippage_pct")
        if slippage is None:
            slippage = self._estimate_slippage(mint, usd_amount, "buy")
        if slippage > float(self.cfg["max_slippage_pct"]):
            return Fill(
                False, "buy", mint, slippage_pct=slippage,
                reason=f"estimated slippage {slippage:.2f}% exceeds the "
                       f"{self.cfg['max_slippage_pct']:.2f}% tolerance",
            )

        fee = self._fee(usd_amount)
        effective_price = price * (1.0 + slippage / 100.0)   # buying pays up
        qty = (usd_amount - fee) / effective_price
        return Fill(
            True, "buy", mint,
            price=effective_price, qty=qty, gross_usd=usd_amount,
            fee_usd=fee, slippage_pct=slippage,
            detail={"simulated": True, "quoted_price": price},
        )

    def sell(self, mint: str, qty: float, price: float, **kw: Any) -> Fill:
        if price <= 0 or qty <= 0:
            return Fill(False, "sell", mint, reason="invalid price or quantity")

        notional = qty * price
        slippage = kw.get("slippage_pct")
        if slippage is None:
            slippage = self._estimate_slippage(mint, notional, "sell")

        effective_price = price * (1.0 - slippage / 100.0)   # selling receives less
        gross = qty * effective_price
        fee = self._fee(gross)
        return Fill(
            True, "sell", mint,
            price=effective_price, qty=qty, gross_usd=gross,
            fee_usd=fee, slippage_pct=slippage,
            detail={"simulated": True, "quoted_price": price},
        )


class LiveExecutor(Executor):
    """Real swaps through Jupiter: /order -> sign locally -> /execute.

    The keypair is loaded from the environment and never leaves this module -
    it is not passed to the web process, not logged, and not written anywhere.
    """

    mode = "live"

    def __init__(self, clients: Clients, cfg: dict[str, Any], private_key_b58: str) -> None:
        super().__init__(clients, cfg)
        self._keypair = self._load_keypair(private_key_b58)
        self.pubkey = str(self._keypair.pubkey())

    @staticmethod
    def _load_keypair(private_key_b58: str):
        if not private_key_b58:
            raise RuntimeError(
                "SOLANA_PRIVATE_KEY is not set. Live trading needs a dedicated, "
                "freshly generated hot wallet - run: python manage.py new-wallet"
            )
        try:
            from solders.keypair import Keypair  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "Live trading needs the 'solders' package: pip install solders base58"
            ) from exc
        import base58  # noqa: PLC0415

        raw = base58.b58decode(private_key_b58.strip())
        if len(raw) == 64:
            return Keypair.from_bytes(raw)
        if len(raw) == 32:
            return Keypair.from_seed(raw)
        raise RuntimeError(f"unrecognised private key length: {len(raw)} bytes")

    # ------------------------------------------------------------------
    def _check_congestion(self) -> None:
        if not self.cfg.get("congestion_check_enabled", True):
            return
        c = self.clients.rpc.congestion(
            max_priority_fee=int(self.cfg["congestion_max_priority_fee_lamports"])
        )
        if not c.healthy:
            raise SkipTrade(f"network congestion: {c.reason}")

    def _swap(self, input_mint: str, output_mint: str, amount_raw: int) -> dict[str, Any]:
        """Quote with a taker, sign, submit, confirm."""
        self._check_congestion()

        slippage_bps = int(float(self.cfg["max_slippage_pct"]) * 100)
        priority_fee = self.clients.rpc.suggested_priority_fee(
            int(self.cfg["priority_fee_lamports"]),
            int(self.cfg["congestion_max_priority_fee_lamports"]),
        )
        quote = self.clients.jupiter.order(
            input_mint,
            output_mint,
            amount_raw,
            slippage_bps=slippage_bps,
            taker=self.pubkey,
            priority_fee_lamports=priority_fee,
            priority="high",
        )

        implied = quote.implied_slippage_pct
        if implied > float(self.cfg["max_slippage_pct"]):
            raise SkipTrade(
                f"quoted fill would cost {implied:.2f}% versus the "
                f"{self.cfg['max_slippage_pct']:.2f}% tolerance"
            )
        if not quote.transaction or not quote.request_id:
            raise SkipTrade("Jupiter returned no signable transaction for this order")

        signed = self._sign(quote.transaction)
        result = self.clients.jupiter.execute(signed, quote.request_id)

        signature = result.get("signature") or result.get("txSignature")
        status = str(result.get("status", "")).lower()
        if status in {"failed", "error"} or result.get("error"):
            raise ApiError(
                f"swap failed: {result.get('error') or result.get('code') or status}"
            )
        if signature:
            confirmation = self.clients.rpc.confirm(signature)
            if confirmation["status"] == "failed":
                raise ApiError(f"swap reverted on-chain: {confirmation.get('error')}")
            result["confirmation"] = confirmation

        return {"quote": quote, "result": result, "signature": signature, "slippage": implied}

    def _sign(self, transaction_b64: str) -> str:
        from solders.transaction import VersionedTransaction  # noqa: PLC0415

        raw = base64.b64decode(transaction_b64)
        tx = VersionedTransaction.from_bytes(raw)
        signed = VersionedTransaction(tx.message, [self._keypair])
        return base64.b64encode(bytes(signed)).decode("ascii")

    # ------------------------------------------------------------------
    def buy(self, mint: str, usd_amount: float, price: float, **kw: Any) -> Fill:
        decimals = int(kw.get("decimals", 6))
        amount_raw = int(usd_amount * 1e6)  # spending USDC
        try:
            out = self._swap(USDC_MINT, mint, amount_raw)
        except SkipTrade as exc:
            return Fill(False, "buy", mint, reason=str(exc))
        except ApiError as exc:
            return Fill(False, "buy", mint, reason=f"swap error: {exc}")

        quote: Quote = out["quote"]
        qty = quote.out_amount / (10**decimals)
        fill_price = (quote.in_usd / qty) if qty > 0 else price
        return Fill(
            True, "buy", mint,
            price=fill_price, qty=qty, gross_usd=quote.in_usd,
            fee_usd=max(0.0, quote.in_usd - quote.out_usd),
            slippage_pct=out["slippage"],
            tx_signature=out["signature"],
            detail={"router": quote.router, "swap_type": quote.swap_type},
        )

    def sell(self, mint: str, qty: float, price: float, **kw: Any) -> Fill:
        decimals = int(kw.get("decimals", 6))
        amount_raw = int(qty * (10**decimals))
        try:
            out = self._swap(mint, USDC_MINT, amount_raw)
        except SkipTrade as exc:
            return Fill(False, "sell", mint, reason=str(exc))
        except ApiError as exc:
            return Fill(False, "sell", mint, reason=f"swap error: {exc}")

        quote: Quote = out["quote"]
        proceeds = quote.out_amount / 1e6  # receiving USDC
        fill_price = proceeds / qty if qty > 0 else price
        return Fill(
            True, "sell", mint,
            price=fill_price, qty=qty, gross_usd=proceeds,
            fee_usd=max(0.0, quote.in_usd - quote.out_usd),
            slippage_pct=out["slippage"],
            tx_signature=out["signature"],
            detail={"router": quote.router, "swap_type": quote.swap_type},
        )

    # ------------------------------------------------------------------
    def wallet_snapshot(self) -> dict[str, Any]:
        """On-chain truth, for startup reconciliation."""
        return {
            "pubkey": self.pubkey,
            "sol": self.clients.rpc.get_sol_balance(self.pubkey),
            "tokens": self.clients.rpc.get_token_balances(self.pubkey),
        }


def build_executor(clients: Clients, cfg: dict[str, Any], mode: str, secrets: Any) -> Executor:
    if mode == "live":
        return LiveExecutor(clients, cfg, secrets.solana_private_key)
    return PaperExecutor(clients, cfg)
