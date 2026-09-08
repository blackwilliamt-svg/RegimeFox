"""Jito Block Engine client - MEV-protected bundle submission for live swaps.

Verified against docs.jito.wtf on 2026-09-07.

    POST https://mainnet.block-engine.jito.wtf/api/v1/bundles   (sendBundle)
    POST https://mainnet.block-engine.jito.wtf/api/v1/bundles   (getTipAccounts)

A bundle is up to 5 signed, base64-encoded transactions submitted atomically
to the same validator slot through Jito's private relay rather than the
public mempool - the point being a sandwich bot never sees the swap before it
lands. One transaction in the bundle must transfer lamports to one of Jito's
designated tip accounts or the whole bundle is dropped; see
:func:`solbot.execution.LiveExecutor._build_tip_transaction`.
"""
from __future__ import annotations

import logging
from typing import Any

from ..ratelimit import TokenBucket
from .base import ApiError, HttpClient

log = logging.getLogger(__name__)

# Straight from Jito's docs - the fallback used when getTipAccounts can't be
# reached. A stale-but-valid tip account beats blocking a trade on this call.
STATIC_TIP_ACCOUNTS = [
    "96gYZGLnJYVFmbjzopPSU6QiEV5fGqZNyN9nmNhvrZU5",
    "HFqU5x63VTqvQss8hp11i4wVV8bD44PvwucfZ2bU7gRe",
    "Cw8CFyM9FkoMi7K7Crf6HNQqf4uEMzpKw6QNghXLvLkY",
    "ADaUMid9yfUytqMBgopwjb2DTLSokTSzL1zt6iGPaS49",
    "DfXygSm4jCyNCybVYYK6DwvWqjKee8pbDmJGcLWNDXjh",
    "ADuUkR4vqLUMWXxW9gh6D6L8pMSawimctcNZ5pGwDcEt",
    "DttWaMuVvTiduZRnguLF7jNxTgiMBZ1hyAumKUiL2KRL",
    "3AVi9Tg9Uo68tJfuvoKvqKNWKkC5wPdSSdeBnizKZ6jT",
]

MIN_TIP_LAMPORTS = 1000
MAX_BUNDLE_SIZE = 5


class JitoClient(HttpClient):
    provider = "jito"
    base_url = "https://mainnet.block-engine.jito.wtf"

    def __init__(self, bucket: TokenBucket, **kw: Any) -> None:
        super().__init__(bucket, **kw)
        self._tip_accounts_cache: list[str] = []

    def tip_accounts(self) -> list[str]:
        """The current tip account set. Falls back to the static list from
        Jito's docs if the live call fails or hasn't been made yet."""
        if self._tip_accounts_cache:
            return self._tip_accounts_cache
        try:
            data = self.post(
                "/api/v1/bundles",
                json_body={
                    "jsonrpc": "2.0", "id": 1, "method": "getTipAccounts", "params": [],
                },
                priority="high",
            )
            accounts = data.get("result") if isinstance(data, dict) else None
            if isinstance(accounts, list) and accounts:
                self._tip_accounts_cache = [str(a) for a in accounts]
                return self._tip_accounts_cache
        except ApiError as exc:
            log.warning("jito: getTipAccounts failed, using the static list: %s", exc)
        return list(STATIC_TIP_ACCOUNTS)

    def send_bundle(self, signed_transactions_b64: list[str]) -> str:
        """Submit up to 5 signed, base64-encoded transactions as one atomic
        bundle. Returns the bundle id. Raises ApiError on failure or timeout -
        the caller is expected to fall back to plain submission."""
        if not signed_transactions_b64:
            raise ApiError("jito: cannot submit an empty bundle", provider=self.provider)
        if len(signed_transactions_b64) > MAX_BUNDLE_SIZE:
            raise ApiError(
                f"jito: a bundle may hold at most {MAX_BUNDLE_SIZE} transactions",
                provider=self.provider,
            )

        data = self.post(
            "/api/v1/bundles",
            json_body={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "sendBundle",
                "params": [signed_transactions_b64, {"encoding": "base64"}],
            },
            priority="high",
        )
        if not isinstance(data, dict) or "result" not in data:
            raise ApiError(
                f"jito: malformed sendBundle response: {data}", provider=self.provider
            )
        if data.get("error"):
            raise ApiError(
                f"jito: sendBundle rejected: {data['error']}", provider=self.provider
            )
        return str(data["result"])

    def set_api_key(self, key: str) -> None:
        """No-op: Jito's public bundle endpoint is keyless. Kept for symmetry
        with the other clients' rotation interface."""
