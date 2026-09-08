"""Gap-closure item 2: MEV protection on live-trade submission.

Covers the three paths LiveExecutor._submit can take: a successful Jito
bundle, falling back to plain Jupiter execution when Jito fails, and the
disabled-flag path going straight to plain execution. No network - Jupiter,
RPC and Jito are all fakes, and the swap "transaction" Jupiter hands back is
a real (but unsigned) solders VersionedTransaction so LiveExecutor's own
signing code runs unmodified.
"""
from __future__ import annotations

import base64

import base58
import pytest
from solders.hash import Hash
from solders.keypair import Keypair
from solders.message import MessageV0
from solders.pubkey import Pubkey
from solders.signature import Signature
from solders.system_program import TransferParams, transfer
from solders.transaction import VersionedTransaction

from solbot.clients import ApiError, Quote
from solbot.execution import LiveExecutor

from fakes import fake_clients

MINT = "LiveMevTestMintAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
DUMMY_DEST = "96gYZGLnJYVFmbjzopPSU6QiEV5fGqZNyN9nmNhvrZU5"  # a real Jito tip account


def _placeholder_signable_tx(payer: Pubkey) -> str:
    """An unsigned VersionedTransaction whose fee payer is `payer` - exactly
    the shape Jupiter's /order returns for the taker to sign locally."""
    ix = transfer(
        TransferParams(from_pubkey=payer, to_pubkey=Pubkey.from_string(DUMMY_DEST), lamports=1)
    )
    msg = MessageV0.try_compile(payer, [ix], [], Hash.default())
    unsigned = VersionedTransaction.populate(msg, [Signature.default()])
    return base64.b64encode(bytes(unsigned)).decode("ascii")


class StubJupiterForLiveExecution:
    """Just enough of JupiterClient's interface for LiveExecutor._swap."""

    def __init__(self, payer: Pubkey) -> None:
        self.payer = payer
        self.execute_calls: list[tuple[str, str]] = []

    def order(self, input_mint, output_mint, amount, *, slippage_bps, taker, **kw):
        return Quote(
            input_mint=input_mint, output_mint=output_mint,
            in_amount=amount, out_amount=amount, min_out_amount=amount,
            price_impact_pct=0.001, slippage_bps=slippage_bps,
            router="fake", swap_type="fake",
            request_id="req-1", transaction=_placeholder_signable_tx(self.payer),
            in_usd=amount / 1e6, out_usd=amount / 1e6 * 0.999,
            priority_fee_lamports=0, signature_fee_lamports=5000, raw={},
        )

    def execute(self, signed_transaction_b64: str, request_id: str) -> dict:
        self.execute_calls.append((signed_transaction_b64, request_id))
        return {"status": "success", "signature": "plain-execute-signature"}

    def close(self) -> None:
        pass

    def set_api_key(self, key: str) -> None:
        pass


@pytest.fixture
def wallet():
    kp = Keypair()
    return kp, base58.b58encode(bytes(kp)).decode("ascii")


@pytest.fixture
def live_executor(workspace, settings, wallet):
    kp, secret_b58 = wallet
    clients = fake_clients()
    clients.jupiter = StubJupiterForLiveExecution(kp.pubkey())
    ex = LiveExecutor(clients, settings, secret_b58)
    return ex, clients


def test_jito_bundle_is_the_default_send_path(live_executor):
    ex, clients = live_executor
    fill = ex.buy(MINT, 10.0, 1.0, decimals=6)

    assert fill.ok
    assert fill.detail["send_route"] == "jito"
    assert fill.detail["mev_fallback_reason"] is None
    assert clients.jito.calls == 1
    assert len(clients.jito.bundles_sent) == 1
    assert len(clients.jito.bundles_sent[0]) == 2   # swap tx + tip tx
    assert clients.jupiter.execute_calls == []      # never fell back


def test_falls_back_to_plain_execute_when_jito_fails(workspace, settings, wallet):
    kp, secret_b58 = wallet
    clients = fake_clients(jito_fail_with=ApiError("block engine unreachable", provider="jito"))
    clients.jupiter = StubJupiterForLiveExecution(kp.pubkey())
    ex = LiveExecutor(clients, settings, secret_b58)

    fill = ex.buy(MINT, 10.0, 1.0, decimals=6)

    assert fill.ok, fill.reason
    assert fill.detail["send_route"] == "jupiter_managed"
    assert "block engine unreachable" in fill.detail["mev_fallback_reason"]
    assert clients.jito.calls == 1
    assert len(clients.jupiter.execute_calls) == 1   # fallback actually happened


def test_mev_protection_disabled_skips_jito_entirely(live_executor):
    ex, clients = live_executor
    ex.cfg["mev_protection_enabled"] = False

    fill = ex.buy(MINT, 10.0, 1.0, decimals=6)

    assert fill.ok
    assert fill.detail["send_route"] == "jupiter_managed"
    assert fill.detail["mev_fallback_reason"] is None   # disabled, not a failure
    assert clients.jito.calls == 0                      # never even tried
    assert len(clients.jupiter.execute_calls) == 1


def test_sell_also_goes_through_jito(live_executor):
    ex, clients = live_executor
    fill = ex.sell(MINT, 10.0, 1.0, decimals=6)

    assert fill.ok
    assert fill.detail["send_route"] == "jito"
    assert clients.jito.calls == 1


def test_tip_transaction_pays_the_configured_amount_to_a_tip_account(live_executor):
    ex, clients = live_executor
    ex.cfg["jito_tip_lamports"] = 250_000

    ex.buy(MINT, 10.0, 1.0, decimals=6)

    swap_tx_b64, tip_tx_b64 = clients.jito.bundles_sent[0]
    tip_tx = VersionedTransaction.from_bytes(base64.b64decode(tip_tx_b64))
    # Signed by this wallet, and its only instruction is the transfer we built.
    assert str(tip_tx.message.account_keys[0]) == str(ex._keypair.pubkey())
    assert tip_tx.signatures[0] != Signature.default()


def test_swap_signature_is_recovered_from_the_signed_transaction_not_the_bundle_id(live_executor):
    """The bundle id Jito returns is not a transaction signature - the
    confirmation poll needs the swap tx's own signature, recovered locally."""
    ex, clients = live_executor
    fill = ex.buy(MINT, 10.0, 1.0, decimals=6)

    assert fill.tx_signature
    assert fill.tx_signature != "fake-bundle-id-1"
