"""Solana JSON-RPC client.

Used for three things the spec calls for, none of which Jupiter answers:

* **Congestion awareness** - recent prioritization-fee levels and slot health,
  checked before submitting so entries are skipped or delayed during abnormal
  congestion rather than landing at a terrible fee.
* **Startup reconciliation** - the authoritative on-chain SOL and SPL balances,
  compared against what SQLite believes before the engine resumes trading.
* **Confirmation** - polling a signature after a swap is submitted.

A public RPC endpoint is fine here: the bot trades 5-15 minute candles, so it
does not need the dedicated low-latency infrastructure an arbitrage bot would.
"""
from __future__ import annotations

import logging
import statistics
import time
from dataclasses import dataclass
from typing import Any

from ..ratelimit import TokenBucket
from .base import ApiError, HttpClient

log = logging.getLogger(__name__)

LAMPORTS_PER_SOL = 1_000_000_000
TOKEN_PROGRAM = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
TOKEN_2022_PROGRAM = "TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb"


@dataclass(slots=True)
class Congestion:
    median_priority_fee: float      # micro-lamports per compute unit
    p75_priority_fee: float
    samples: int
    healthy: bool
    reason: str = ""


class SolanaRpcClient(HttpClient):
    provider = "solana_rpc"

    def __init__(
        self, bucket: TokenBucket, rpc_url: str = "https://api.mainnet-beta.solana.com", **kw: Any
    ) -> None:
        self.base_url = rpc_url
        super().__init__(bucket, headers={"Content-Type": "application/json"}, **kw)
        self.rpc_url = rpc_url
        self._id = 0

    def set_rpc_url(self, url: str) -> None:
        self.rpc_url = url

    def _call(self, method: str, params: list[Any] | None = None, *, priority: str = "normal") -> Any:
        self._id += 1
        body = {"jsonrpc": "2.0", "id": self._id, "method": method, "params": params or []}
        data = self.post(self.rpc_url, json_body=body, priority=priority)
        if not isinstance(data, dict):
            raise ApiError(f"rpc {method}: malformed response", provider=self.provider)
        if "error" in data:
            err = data["error"]
            raise ApiError(
                f"rpc {method}: {err.get('message', err)}", provider=self.provider
            )
        return data.get("result")

    # ------------------------------------------------------------------
    def get_sol_balance(self, pubkey: str) -> float:
        result = self._call("getBalance", [pubkey], priority="high")
        lamports = (result or {}).get("value", 0) if isinstance(result, dict) else 0
        return lamports / LAMPORTS_PER_SOL

    def get_token_balances(self, owner: str) -> dict[str, float]:
        """{mint: ui_amount} across both token programs."""
        balances: dict[str, float] = {}
        for program in (TOKEN_PROGRAM, TOKEN_2022_PROGRAM):
            try:
                result = self._call(
                    "getTokenAccountsByOwner",
                    [owner, {"programId": program}, {"encoding": "jsonParsed"}],
                    priority="high",
                )
            except ApiError as exc:
                log.warning("token account lookup failed (%s): %s", program, exc)
                continue
            for acct in (result or {}).get("value", []):
                try:
                    info = acct["account"]["data"]["parsed"]["info"]
                    amount = info["tokenAmount"]["uiAmount"]
                    if amount:
                        balances[info["mint"]] = balances.get(info["mint"], 0.0) + float(amount)
                except (KeyError, TypeError, ValueError):
                    continue
        return balances

    # ------------------------------------------------------------------
    def congestion(
        self, *, max_priority_fee: int, sample_accounts: list[str] | None = None
    ) -> Congestion:
        """Read recent prioritization fees to decide whether it is safe to submit.

        ``getRecentPrioritizationFees`` returns roughly the last 150 slots. A
        network under load shows the median climbing sharply; when it exceeds
        the configured ceiling the engine skips or delays the entry rather than
        overpaying to land a trade it can wait on.
        """
        try:
            result = self._call(
                "getRecentPrioritizationFees",
                [sample_accounts or []],
                priority="normal",
            )
        except ApiError as exc:
            # Unknown network state is not a reason to trade blind.
            return Congestion(0.0, 0.0, 0, healthy=False, reason=f"rpc unavailable: {exc}")

        fees = [
            float(r.get("prioritizationFee", 0))
            for r in (result or [])
            if isinstance(r, dict)
        ]
        fees = [f for f in fees if f >= 0]
        if not fees:
            return Congestion(0.0, 0.0, 0, healthy=True, reason="no fee samples")

        fees.sort()
        median = statistics.median(fees)
        p75 = fees[int(len(fees) * 0.75)] if len(fees) > 3 else median
        healthy = p75 <= max_priority_fee
        return Congestion(
            median_priority_fee=median,
            p75_priority_fee=p75,
            samples=len(fees),
            healthy=healthy,
            reason=""
            if healthy
            else f"75th-percentile priority fee {p75:,.0f} exceeds ceiling {max_priority_fee:,}",
        )

    def suggested_priority_fee(self, floor: int, ceiling: int) -> int:
        c = self.congestion(max_priority_fee=ceiling)
        if c.samples == 0:
            return floor
        return int(max(floor, min(ceiling, c.p75_priority_fee * 1.15)))

    def get_latest_blockhash(self) -> str:
        """The recent blockhash a locally-built transaction needs to be
        signable - used for the Jito tip transaction, which Jupiter's /order
        never returns because we build and sign it ourselves."""
        result = self._call(
            "getLatestBlockhash", [{"commitment": "confirmed"}], priority="high"
        )
        blockhash = (result or {}).get("value", {}).get("blockhash") if isinstance(result, dict) else None
        if not blockhash:
            raise ApiError("rpc getLatestBlockhash: malformed response", provider=self.provider)
        return str(blockhash)

    # ------------------------------------------------------------------
    def confirm(self, signature: str, *, timeout: float = 90.0, poll: float = 2.0) -> dict[str, Any]:
        """Poll until the signature confirms, fails, or the timeout elapses."""
        deadline = time.monotonic() + timeout
        last: dict[str, Any] = {"status": "pending", "signature": signature}
        while time.monotonic() < deadline:
            try:
                result = self._call(
                    "getSignatureStatuses", [[signature], {"searchTransactionHistory": True}],
                    priority="high",
                )
            except ApiError as exc:
                last = {"status": "unknown", "signature": signature, "error": str(exc)}
                time.sleep(poll)
                continue
            value = ((result or {}).get("value") or [None])[0]
            if value:
                if value.get("err"):
                    return {"status": "failed", "signature": signature, "error": value["err"]}
                conf = value.get("confirmationStatus")
                if conf in ("confirmed", "finalized"):
                    return {"status": "confirmed", "signature": signature, "level": conf}
            time.sleep(poll)
        return {**last, "status": "timeout"}

    def ping(self) -> bool:
        self._call("getHealth", priority="high")
        return True
