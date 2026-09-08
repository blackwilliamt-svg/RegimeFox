"""Building the tradeable universe.

The universe is Binance's top ``binance_top_n`` coins by 24h quote volume -
established, large-cap coins, not a pure liquidity/volume floor over the whole
Jupiter token list. That Binance-derived shortlist is then intersected with
whatever of those coins Jupiter can actually route a swap for on Solana
(native SOL, or a wrapped/bridged version), and only *then* does the
liquidity/volume floor apply, as a filter on that shortlist rather than as the
mechanism that built it.

A large-cap coin frequently is not tradeable on Solana at all, or trades there
only through a wrapped representation whose ticker does not match the
originating chain's (wrapped Bitcoin is ``WBTC``, not ``BTC``). Matching is
therefore done by symbol with a small alias table for the handful of coins
where that mapping is not the identity, then narrowed to the most liquid,
verified candidate Jupiter returns for that query - the same signal the
previous liquidity-floor universe already used to judge a token, just applied
to a shortlist instead of to the whole list.

Every refresh also writes a row per token into ``universe_history``. That table
is what the backtest builds its basket from - a point-in-time record of what
passed the filters on each day, including tokens that later died. Rebuilding a
historical basket from what is tradeable *today* would silently drop every token
that rugged or got delisted, which is exactly the survivorship bias the spec
warns about.
"""
from __future__ import annotations

import logging
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any, Iterable

from . import db
from .clients import ApiError, BinanceAsset, BinanceClient, JupiterClient, TokenInfo

log = logging.getLogger(__name__)

# Never trade the quote assets themselves.
EXCLUDED = {
    "So11111111111111111111111111111111111111112",   # SOL
    "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",  # USDC
    "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB",  # USDT
}

# Symbol aliases for coins whose Solana-routable representation does not carry
# the same ticker as the coin Binance ranks. Checked before the identity
# match; native SOL is deliberately absent since Jupiter's own symbol for it
# is already "SOL".
SYMBOL_ALIASES: dict[str, tuple[str, ...]] = {
    "BTC": ("WBTC", "BTC"),
    "ETH": ("WETH", "ETH"),
    "DOGE": ("WDOGE", "DOGE"),
    "XRP": ("WXRP", "XRP"),
    "LTC": ("WLTC", "LTC"),
    "BCH": ("WBCH", "BCH"),
    "BNB": ("WBNB", "BNB"),
}


@dataclass(slots=True)
class UniverseStats:
    considered: int = 0             # Binance top-N candidates
    routable: int = 0               # of those, matched to a Solana mint
    passed: int = 0
    rejected_not_routable: int = 0
    rejected_liquidity: int = 0
    rejected_volume: int = 0
    rejected_excluded: int = 0
    api_calls: int = 0
    refreshed_at: int = 0


class UniverseBuilder:
    def __init__(
        self, binance: BinanceClient, jupiter: JupiterClient, cfg: dict[str, Any]
    ) -> None:
        self.binance = binance
        self.jupiter = jupiter
        self.cfg = cfg
        self._tokens: dict[str, TokenInfo] = {}
        self._last_refresh: float = 0.0
        self.stats = UniverseStats()

    def update_config(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg

    @property
    def tokens(self) -> dict[str, TokenInfo]:
        return self._tokens

    @property
    def mints(self) -> list[str]:
        return list(self._tokens)

    def due(self) -> bool:
        return (time.monotonic() - self._last_refresh) >= float(
            self.cfg["universe_refresh_seconds"]
        )

    # ------------------------------------------------------------------
    def refresh(
        self, *, conn: sqlite3.Connection | None = None, record_history: bool = True
    ) -> UniverseStats:
        """Re-pull Binance's top coins, route them onto Solana, apply the floors."""
        conn = conn or db.connect()
        stats = UniverseStats(refreshed_at=db.now())

        try:
            top = self.binance.top_bases(limit=int(self.cfg["binance_top_n"]))
            stats.api_calls += 1
        except ApiError as exc:
            log.warning("universe: binance top-coin pull failed: %s", exc)
            top = []
        stats.considered = len(top)

        routed = self._route_to_jupiter(top, stats)
        stats.routable = len(routed)

        kept = self.filter_tokens(routed.values(), stats)
        kept.sort(key=lambda t: t.volume_24h, reverse=True)
        kept = kept[: int(self.cfg["universe_max_tokens"])]
        stats.passed = len(kept)

        self._tokens = {t.mint: t for t in kept}
        self._last_refresh = time.monotonic()
        self.stats = stats

        self._persist(kept, conn)
        if record_history:
            self._record_history(kept, conn)

        db.log_event(
            f"Universe refreshed: {stats.passed} tradeable of {stats.considered} of "
            f"Binance's top coins considered ({stats.routable} routable on Solana; "
            f"floors: ${self.cfg['min_liquidity_usd']:,.0f} liquidity / "
            f"${self.cfg['min_volume_24h_usd']:,.0f} 24h volume)",
            category="system",
            detail={
                "passed": stats.passed,
                "considered": stats.considered,
                "routable": stats.routable,
                "rejected_not_routable": stats.rejected_not_routable,
                "rejected_liquidity": stats.rejected_liquidity,
                "rejected_volume": stats.rejected_volume,
            },
            conn=conn,
        )
        return stats

    # ------------------------------------------------------------------
    def _route_to_jupiter(
        self, assets: list[BinanceAsset], stats: UniverseStats
    ) -> dict[str, TokenInfo]:
        """Find the best Solana-routable match for each Binance top-coin.

        "Routable" here means Jupiter's own token list carries it - the same
        precondition the live scanner and executor already require of every
        token they touch. A coin large enough for Binance's top list but with
        no Solana market at all (or with a wrapped market too thin to be worth
        Jupiter listing) is not routable, and is dropped rather than guessed at.
        """
        routed: dict[str, TokenInfo] = {}
        for asset in assets:
            queries = SYMBOL_ALIASES.get(asset.symbol, (asset.symbol,))
            match: TokenInfo | None = None
            for query in queries:
                try:
                    candidates = self.jupiter.search(query)
                    stats.api_calls += 1
                except ApiError as exc:
                    log.debug("universe: jupiter search %r failed: %s", query, exc)
                    continue
                match = self._best_candidate(asset.symbol, query, candidates)
                if match is not None:
                    break
            if match is None:
                stats.rejected_not_routable += 1
                continue
            match.binance_pair = asset.pair
            routed[match.mint] = match
        return routed

    @staticmethod
    def _best_candidate(
        base_symbol: str, query: str, candidates: Iterable[TokenInfo]
    ) -> TokenInfo | None:
        """The most liquid verified token whose ticker actually matches.

        Jupiter's search is a substring/fuzzy match, so it is filtered back
        down to an exact (case-insensitive) ticker match before anything else
        is judged - a search for "SOL" returning a token merely named
        "SOLDIER" is not a match, it is noise.
        """
        exact = [
            c for c in candidates
            if c.mint and c.symbol.strip().upper() == query.strip().upper()
        ]
        if not exact:
            return None
        verified = [c for c in exact if c.is_verified]
        pool = verified or exact
        return max(pool, key=lambda c: c.liquidity)

    def filter_tokens(
        self, tokens: Iterable[TokenInfo], stats: UniverseStats | None = None
    ) -> list[TokenInfo]:
        stats = stats or UniverseStats()
        min_liq = float(self.cfg["min_liquidity_usd"])
        min_vol = float(self.cfg["min_volume_24h_usd"])
        kept: list[TokenInfo] = []
        for t in tokens:
            if not t.mint or t.mint in EXCLUDED:
                stats.rejected_excluded += 1
                continue
            if t.liquidity < min_liq:
                stats.rejected_liquidity += 1
                continue
            if t.volume_24h < min_vol:
                stats.rejected_volume += 1
                continue
            kept.append(t)
        return kept

    # ------------------------------------------------------------------
    def _persist(self, tokens: list[TokenInfo], conn: sqlite3.Connection) -> None:
        now = db.now()
        conn.executemany(
            "INSERT INTO universe(mint, symbol, name, decimals, liquidity_usd, "
            "volume_24h_usd, mcap, holder_count, organic_score, is_verified, "
            "first_pool_at, binance_pair, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(mint) DO UPDATE SET symbol=excluded.symbol, name=excluded.name, "
            "decimals=excluded.decimals, liquidity_usd=excluded.liquidity_usd, "
            "volume_24h_usd=excluded.volume_24h_usd, mcap=excluded.mcap, "
            "holder_count=excluded.holder_count, organic_score=excluded.organic_score, "
            "is_verified=excluded.is_verified, first_pool_at=excluded.first_pool_at, "
            "binance_pair=excluded.binance_pair, updated_at=excluded.updated_at",
            [
                (
                    t.mint, t.symbol, t.name, t.decimals, t.liquidity, t.volume_24h,
                    t.mcap, t.holder_count, t.organic_score, 1 if t.is_verified else 0,
                    t.first_pool_at, t.binance_pair, now,
                )
                for t in tokens
            ],
        )
        # Drop entries that have fallen out of the universe entirely.
        if tokens:
            placeholders = ",".join("?" * len(tokens))
            conn.execute(
                f"DELETE FROM universe WHERE mint NOT IN ({placeholders})",
                [t.mint for t in tokens],
            )

    @staticmethod
    def _record_history(tokens: list[TokenInfo], conn: sqlite3.Connection) -> None:
        day = time.strftime("%Y-%m-%d", time.gmtime())
        conn.executemany(
            "INSERT INTO universe_history(day, mint, symbol, liquidity_usd, volume_24h_usd) "
            "VALUES (?,?,?,?,?) ON CONFLICT(day, mint) DO UPDATE SET "
            "liquidity_usd=excluded.liquidity_usd, volume_24h_usd=excluded.volume_24h_usd",
            [(day, t.mint, t.symbol, t.liquidity, t.volume_24h) for t in tokens],
        )

    # ------------------------------------------------------------------
    def load_persisted(self, conn: sqlite3.Connection | None = None) -> dict[str, TokenInfo]:
        """Restore the last universe after a restart, so scanning can start at once."""
        conn = conn or db.connect()
        rows = conn.execute("SELECT * FROM universe").fetchall()
        self._tokens = {
            r["mint"]: TokenInfo(
                mint=r["mint"],
                symbol=r["symbol"] or "",
                name=r["name"] or "",
                decimals=int(r["decimals"] or 9),
                liquidity=float(r["liquidity_usd"] or 0.0),
                volume_24h=float(r["volume_24h_usd"] or 0.0),
                mcap=float(r["mcap"] or 0.0),
                holder_count=int(r["holder_count"] or 0),
                organic_score=float(r["organic_score"] or 0.0),
                is_verified=bool(r["is_verified"]),
                first_pool_at=r["first_pool_at"],
                binance_pair=r["binance_pair"] or "",
            )
            for r in rows
        }
        return self._tokens


def historical_basket(
    day_from: str, day_to: str, *, conn: sqlite3.Connection | None = None
) -> list[str]:
    """Every mint that passed the filters at any point in the window.

    Deliberately a union over the period, and deliberately *not* intersected
    with today's universe: the tokens that dropped out because they died are
    the ones that keep the backtest honest.
    """
    conn = conn or db.connect()
    rows = conn.execute(
        "SELECT DISTINCT mint FROM universe_history WHERE day BETWEEN ? AND ?",
        (day_from, day_to),
    ).fetchall()
    return [r["mint"] for r in rows]
