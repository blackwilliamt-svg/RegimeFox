"""In-memory stand-ins for the API clients, so tests never touch the network."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from solbot.clients import BinanceAsset, Clients, PriceInfo, TokenInfo
from solbot.clients.rugcheck import RugReport
from solbot.ratelimit import TokenBucket


class FakeJupiter:
    def __init__(self, prices: dict[str, float] | None = None,
                 tokens: list[TokenInfo] | None = None) -> None:
        self._prices = dict(prices or {})
        self._tokens = list(tokens or [])
        self.calls = 0

    def set_price(self, mint: str, price: float) -> None:
        self._prices[mint] = price

    def prices(self, mints, *, priority: str = "normal") -> dict[str, PriceInfo]:
        self.calls += 1
        out = {}
        for m in mints:
            if m in self._prices:
                out[m] = PriceInfo(mint=m, price=self._prices[m], decimals=6, liquidity=500_000)
        return out

    def price_calls_for(self, n: int) -> int:
        return max(0, -(-n // 50))

    def top_tokens(self, category: str = "toptraded", interval: str = "24h",
                   limit: int = 100, *, priority: str = "low") -> list[TokenInfo]:
        self.calls += 1
        return list(self._tokens)

    def search(self, query: str, *, priority: str = "low") -> list[TokenInfo]:
        self.calls += 1
        q = query.strip().upper()
        return [t for t in self._tokens if q in t.symbol.strip().upper()]

    def order(self, *a: Any, **kw: Any):
        raise NotImplementedError

    def close(self) -> None:
        pass

    def set_api_key(self, key: str) -> None:
        pass


class FakeBinance:
    """Stands in for Binance.US's top-coin ranking and candle history (spec 2/3)."""

    def __init__(
        self,
        assets: list[BinanceAsset] | None = None,
        *,
        klines_fn: Any = None,
    ) -> None:
        self._assets = list(assets or [])
        # (pair, since, until) -> list[Candle]; default is "nothing on record",
        # which is what most callers of this fake actually want.
        self._klines_fn = klines_fn or (lambda pair, since, until: [])
        self.calls = 0
        self.klines_calls: list[tuple[str, int, int]] = []

    def top_bases(self, limit: int = 100) -> list[BinanceAsset]:
        self.calls += 1
        ranked = sorted(self._assets, key=lambda a: a.quote_volume_24h, reverse=True)
        return ranked[: max(0, int(limit))]

    def klines_range(self, pair: str, *, since: int, until: int):
        self.calls += 1
        self.klines_calls.append((pair, since, until))
        return self._klines_fn(pair, since, until)

    def close(self) -> None:
        pass

    def ping(self) -> bool:
        return True


def binance_asset(symbol: str, volume_24h: float = 1_000_000.0, **kw: Any) -> BinanceAsset:
    return BinanceAsset(
        symbol=symbol, quote_volume_24h=volume_24h, pair=f"{symbol}USDT", **kw
    )


class FakeRugCheck:
    def __init__(self, clean: bool = True) -> None:
        self.clean = clean
        self.calls = 0

    def report(self, mint: str, *, priority: str = "normal") -> RugReport:
        self.calls += 1
        if self.clean:
            return RugReport(
                mint=mint, ok=True, rugged=False, score=1.0,
                mint_authority=None, freeze_authority=None,
                lp_locked_pct=99.0, top_holder_pct=5.0, insider_pct=1.0,
                total_holders=5000, total_liquidity_usd=500_000.0,
                risks=[], raw={},
            )
        return RugReport(
            mint=mint, ok=True, rugged=True, score=80.0,
            mint_authority="SomeAuthority", freeze_authority="SomeAuthority",
            lp_locked_pct=0.0, top_holder_pct=70.0, insider_pct=50.0,
            total_holders=12, total_liquidity_usd=1_000.0,
            risks=[{"name": "Mint authority enabled", "level": "danger"}], raw={},
        )

    def summary(self, mint: str, *, priority: str = "low") -> dict[str, Any]:
        return {}

    def close(self) -> None:
        pass

    def set_api_key(self, key: str) -> None:
        pass


class FakeRpc:
    def __init__(self, sol: float = 1.0, tokens: dict[str, float] | None = None,
                 healthy: bool = True) -> None:
        self.sol = sol
        self.tokens = dict(tokens or {})
        self.healthy = healthy
        self.blockhash = "11111111111111111111111111111111"  # solders Hash.default()

    def get_sol_balance(self, pubkey: str) -> float:
        return self.sol

    def get_token_balances(self, owner: str) -> dict[str, float]:
        return dict(self.tokens)

    def congestion(self, *, max_priority_fee: int, sample_accounts=None):
        from solbot.clients.solana_rpc import Congestion

        return Congestion(1000.0, 2000.0, 100, healthy=self.healthy,
                          reason="" if self.healthy else "congested")

    def suggested_priority_fee(self, floor: int, ceiling: int) -> int:
        return floor

    def get_latest_blockhash(self) -> str:
        return self.blockhash

    def confirm(self, signature: str, **kw: Any) -> dict[str, Any]:
        return {"status": "confirmed", "signature": signature}

    def close(self) -> None:
        pass


class FakeJito:
    """Stands in for the Jito Block Engine. `fail_with` lets a test force the
    fallback-to-plain-RPC path without touching the network."""

    def __init__(self, fail_with: Exception | None = None) -> None:
        self.fail_with = fail_with
        self.bundles_sent: list[list[str]] = []
        self.calls = 0

    def tip_accounts(self) -> list[str]:
        from solbot.clients.jito import STATIC_TIP_ACCOUNTS

        return list(STATIC_TIP_ACCOUNTS)

    def send_bundle(self, signed_transactions_b64: list[str]) -> str:
        self.calls += 1
        if self.fail_with is not None:
            raise self.fail_with
        self.bundles_sent.append(list(signed_transactions_b64))
        return "fake-bundle-id-" + str(self.calls)

    def close(self) -> None:
        pass

    def set_api_key(self, key: str) -> None:
        pass


def fake_clients(
    *,
    prices: dict[str, float] | None = None,
    tokens: list[TokenInfo] | None = None,
    binance_assets: list[BinanceAsset] | None = None,
    clean_safety: bool = True,
    sol: float = 1.0,
    wallet_tokens: dict[str, float] | None = None,
    jito_fail_with: Exception | None = None,
) -> Clients:
    bucket = lambda name: TokenBucket(1000.0, 100, reserve=0.0, name=name)
    return Clients(
        jupiter=FakeJupiter(prices, tokens),
        rugcheck=FakeRugCheck(clean_safety),
        rpc=FakeRpc(sol, wallet_tokens),
        binance=FakeBinance(binance_assets),
        jito=FakeJito(jito_fail_with),
        jupiter_bucket=bucket("jupiter"),
        rugcheck_bucket=bucket("rugcheck"),
        rpc_bucket=bucket("rpc"),
        binance_bucket=bucket("binance"),
        jito_bucket=bucket("jito"),
    )


def token(mint: str, symbol: str = "TEST", **kw: Any) -> TokenInfo:
    base = dict(
        mint=mint, symbol=symbol, name=symbol, decimals=6, price=1.0,
        liquidity=500_000.0, volume_24h=1_000_000.0, mcap=5_000_000.0,
        holder_count=5000, organic_score=80.0, organic_label="high",
        is_verified=True, mint_authority=None, freeze_authority=None,
        top_holders_pct=10.0, first_pool_at=1_600_000_000, tags=["verified"],
    )
    base.update(kw)
    return TokenInfo(**base)
