"""Building the tradeable universe.

The universe is derived from a liquidity floor and a volume floor rather than a
fixed token count, so it grows and shrinks with what is actually tradeable
instead of being pinned to an arbitrary number like 20 or 50. The defaults
($50k liquidity / $75k 24h volume) intentionally exclude micro-caps, which move
too erratically for a rules-based technical approach.

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
from dataclasses import dataclass
from typing import Any, Iterable

from . import db
from .clients import ApiError, JupiterClient, TokenInfo

log = logging.getLogger(__name__)

# Pulled from several rankings so the universe is not just whatever is loudest.
CATEGORIES = (("toptraded", "24h"), ("toporganicscore", "24h"))

# Never trade the quote assets themselves.
EXCLUDED = {
    "So11111111111111111111111111111111111111112",   # SOL
    "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",  # USDC
    "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB",  # USDT
}


@dataclass(slots=True)
class UniverseStats:
    considered: int = 0
    passed: int = 0
    rejected_liquidity: int = 0
    rejected_volume: int = 0
    rejected_excluded: int = 0
    api_calls: int = 0
    refreshed_at: int = 0


class UniverseBuilder:
    def __init__(self, jupiter: JupiterClient, cfg: dict[str, Any]) -> None:
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
        """Re-pull the rankings, apply the floors, and persist the result."""
        conn = conn or db.connect()
        stats = UniverseStats(refreshed_at=db.now())
        candidates: dict[str, TokenInfo] = {}

        for category, interval in CATEGORIES:
            try:
                found = self.jupiter.top_tokens(
                    category=category, interval=interval, limit=100
                )
                stats.api_calls += 1
            except ApiError as exc:
                log.warning("universe: %s/%s failed: %s", category, interval, exc)
                continue
            for t in found:
                if t.mint and t.mint not in candidates:
                    candidates[t.mint] = t

        stats.considered = len(candidates)
        kept = self.filter_tokens(candidates.values(), stats)

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
            f"Universe refreshed: {stats.passed} tradeable of {stats.considered} considered "
            f"(floors: ${self.cfg['min_liquidity_usd']:,.0f} liquidity / "
            f"${self.cfg['min_volume_24h_usd']:,.0f} 24h volume)",
            category="system",
            detail={
                "passed": stats.passed,
                "considered": stats.considered,
                "rejected_liquidity": stats.rejected_liquidity,
                "rejected_volume": stats.rejected_volume,
            },
            conn=conn,
        )
        return stats

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
            "first_pool_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(mint) DO UPDATE SET symbol=excluded.symbol, name=excluded.name, "
            "decimals=excluded.decimals, liquidity_usd=excluded.liquidity_usd, "
            "volume_24h_usd=excluded.volume_24h_usd, mcap=excluded.mcap, "
            "holder_count=excluded.holder_count, organic_score=excluded.organic_score, "
            "is_verified=excluded.is_verified, first_pool_at=excluded.first_pool_at, "
            "updated_at=excluded.updated_at",
            [
                (
                    t.mint, t.symbol, t.name, t.decimals, t.liquidity, t.volume_24h,
                    t.mcap, t.holder_count, t.organic_score, 1 if t.is_verified else 0,
                    t.first_pool_at, now,
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
