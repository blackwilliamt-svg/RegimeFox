"""In-memory stand-ins for the API clients, so tests never touch the network."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from solbot.clients import Clients, PriceInfo, TokenInfo
from solbot.clients.rugcheck import RugReport
from solbot.ratelimit import MonthlyBudget, TokenBucket


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

    def order(self, *a: Any, **kw: Any):
        raise NotImplementedError

    def close(self) -> None:
        pass

    def set_api_key(self, key: str) -> None:
        pass


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

    def confirm(self, signature: str, **kw: Any) -> dict[str, Any]:
        return {"status": "confirmed", "signature": signature}

    def close(self) -> None:
        pass


class FakeBirdeye:
    def __init__(self) -> None:
        self.budget = MonthlyBudget(0)

    def ohlcv_range(self, *a: Any, **kw: Any):
        return []

    def calls_needed(self, interval: str, days: int) -> int:
        return 1

    def close(self) -> None:
        pass

    def set_api_key(self, key: str) -> None:
        pass


def fake_clients(
    *,
    prices: dict[str, float] | None = None,
    tokens: list[TokenInfo] | None = None,
    clean_safety: bool = True,
    sol: float = 1.0,
    wallet_tokens: dict[str, float] | None = None,
) -> Clients:
    bucket = lambda name: TokenBucket(1000.0, 100, reserve=0.0, name=name)
    return Clients(
        jupiter=FakeJupiter(prices, tokens),
        birdeye=FakeBirdeye(),
        rugcheck=FakeRugCheck(clean_safety),
        rpc=FakeRpc(sol, wallet_tokens),
        jupiter_bucket=bucket("jupiter"),
        birdeye_bucket=bucket("birdeye"),
        rugcheck_bucket=bucket("rugcheck"),
        rpc_bucket=bucket("rpc"),
        birdeye_budget=MonthlyBudget(0),
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
