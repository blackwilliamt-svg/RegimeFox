"""Binance.US client - top-coin ranking and candle history, not a trading venue.

The bot never trades on Binance; Solana/Jupiter remains the only execution
path. Binance is used for two things, both read-only:

1. **Top-coin ranking** (spec 2) - "what are the ~100 largest, most
   established coins right now", a question Jupiter's own token rankings
   cannot answer on their own, since they only describe what is liquid *on
   Solana*, not what is large in the market as a whole.
2. **Candle history** (spec 3b) - once a coin is routed onto Solana, its
   indicators are computed from Binance's deep centralized order-book data
   rather than from a Solana DEX aggregator's thinner pricing, which is what
   Birdeye/GeckoTerminal would otherwise have to supply for a token whose
   real liquidity mostly lives elsewhere.

This talks to **Binance.US** (``api.binance.us``), not global Binance
(``api.binance.com``). Global Binance geo-blocks US-origin traffic with an
HTTP 451 under its terms of service "b. Eligibility" clause - a droplet
hosted in a US region gets refused outright. Binance.US mirrors the same
REST API shape (``/api/v3/ticker/24hr``, ``/api/v3/klines``), keyless, but
with a far smaller listed universe (~150 coins vs. global Binance's
thousands) and USD as a first-class quote asset alongside USDT/USDC.

Binance.US does not publish a documented equivalent of global Binance's
``data.binance.vision`` monthly zip archives, so the one-time bulk backfill
(spec 3b) pages the ordinary klines endpoint month by month instead of
downloading pre-built archives - more requests, but still a one-time job
comfortably inside ``binance_rps``.

Endpoints used:

    GET https://api.binance.us/api/v3/ticker/24hr      top-coin ranking
    GET https://api.binance.us/api/v3/klines            candles (incremental and bulk)

Both are keyless and generously weight-limited for the call volume this bot
makes (a ranking pull once per universe refresh, klines pulls rate-limited
through the same token bucket as everything else).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from .base import ApiError, HttpClient

log = logging.getLogger(__name__)

# Binance's kline cap per call.
MAX_KLINES_PER_CALL = 1000
KLINE_INTERVAL = "1m"

# Preferred quote assets, in order. A base asset is ranked on the first of
# these it trades against. Binance.US quotes heavily in both USDT (150+
# pairs) and USD (50+ pairs, fiat-settled) - USDT is checked first since it
# covers more of the listed universe, but USD-only bases would be missed
# entirely without it in this list.
QUOTE_PRIORITY = ("USDT", "USD", "USDC")

# Never treat a stablecoin (or a fiat-pegged token) itself as a "top coin" -
# Binance's volume ranking is dominated by stablecoin pairs, and none of them
# are a token this bot should be scanning for a technical-analysis entry.
STABLE_BASES = {
    "USDT", "USDC", "USD", "TUSD", "DAI", "USDP", "PYUSD", "USDD",
    "EUR", "GBP", "UST", "USTC",
}

# Leveraged/derivative tokens ("BTCUP", "ETHDOWN", ...) track a multiple of the
# underlying's move rather than being the coin itself, and would otherwise
# rank highly purely from being frequently churned.
LEVERAGED_SUFFIXES = ("UP", "DOWN", "BULL", "BEAR")


@dataclass(slots=True)
class Candle:
    ts: int         # unix seconds, bar open
    open: float
    high: float
    low: float
    close: float
    volume: float

    def as_row(self) -> tuple[int, float, float, float, float, float]:
        return (self.ts, self.open, self.high, self.low, self.close, self.volume)


@dataclass(slots=True)
class BinanceAsset:
    """One ranked base asset, as Binance's 24h ticker describes it."""

    symbol: str            # base asset ticker, e.g. "BTC", "SOL", "DOGE"
    quote_volume_24h: float
    last_price: float = 0.0
    price_change_pct_24h: float = 0.0
    trade_count_24h: int = 0
    pair: str = ""          # the Binance symbol actually ranked, e.g. "BTCUSDT"


class BinanceClient(HttpClient):
    provider = "binance"
    base_url = "https://api.binance.us"

    def top_bases(self, limit: int = 100) -> list[BinanceAsset]:
        """The ``limit`` largest non-stable base assets by 24h quote volume."""
        data = self.get("/api/v3/ticker/24hr", priority="low")
        if not isinstance(data, list):
            raise ApiError("binance: malformed ticker/24hr response", provider=self.provider)

        best: dict[str, BinanceAsset] = {}
        for row in data:
            if not isinstance(row, dict):
                continue
            symbol = str(row.get("symbol", ""))
            base, quote = _split_symbol(symbol)
            if not base or quote not in QUOTE_PRIORITY:
                continue
            if base in STABLE_BASES:
                continue
            if any(base.endswith(suffix) for suffix in LEVERAGED_SUFFIXES):
                continue

            volume = _to_float(row.get("quoteVolume"))
            existing = best.get(base)
            # Prefer the higher-priority quote asset; within the same quote,
            # prefer more volume (a base can appear more than once against the
            # same quote only via a data glitch, but ties are handled cheaply).
            if existing is not None:
                existing_rank = QUOTE_PRIORITY.index(_split_symbol(existing.pair)[1])
                this_rank = QUOTE_PRIORITY.index(quote)
                if this_rank > existing_rank:
                    continue
                if this_rank == existing_rank and volume <= existing.quote_volume_24h:
                    continue

            best[base] = BinanceAsset(
                symbol=base,
                quote_volume_24h=volume,
                last_price=_to_float(row.get("lastPrice")),
                price_change_pct_24h=_to_float(row.get("priceChangePercent")),
                trade_count_24h=int(row.get("count") or 0),
                pair=symbol,
            )

        ranked = sorted(best.values(), key=lambda a: a.quote_volume_24h, reverse=True)
        return ranked[: max(0, int(limit))]

    def ping(self) -> bool:
        self.get("/api/v3/ping", priority="high")
        return True

    # ------------------------------------------------------------------
    # candle history (spec 3b)
    # ------------------------------------------------------------------
    def klines(
        self, pair: str, *, start_ms: int, end_ms: int, limit: int = MAX_KLINES_PER_CALL
    ) -> list[Candle]:
        """One page of 1-minute candles for ``pair`` (e.g. "BTCUSDT")."""
        data = self.get(
            "/api/v3/klines",
            params={
                "symbol": pair,
                "interval": KLINE_INTERVAL,
                "startTime": int(start_ms),
                "endTime": int(end_ms),
                "limit": max(1, min(int(limit), MAX_KLINES_PER_CALL)),
            },
            priority="low",
        )
        if not isinstance(data, list):
            raise ApiError(f"binance: malformed klines response for {pair}", provider=self.provider)
        out: list[Candle] = []
        for row in data:
            try:
                out.append(
                    Candle(
                        ts=int(row[0]) // 1000,
                        open=float(row[1]),
                        high=float(row[2]),
                        low=float(row[3]),
                        close=float(row[4]),
                        volume=float(row[7]),  # quote-asset volume, USD-comparable
                    )
                )
            except (IndexError, TypeError, ValueError):
                continue
        return out

    def klines_range(self, pair: str, *, since: int, until: int) -> list[Candle]:
        """Page through a window at 1-minute resolution, respecting the 1000-row cap."""
        out: list[Candle] = []
        cursor_ms = int(since) * 1000
        end_ms = int(until) * 1000
        step_ms = MAX_KLINES_PER_CALL * 60 * 1000
        while cursor_ms < end_ms:
            chunk_end = min(cursor_ms + step_ms, end_ms)
            page = self.klines(pair, start_ms=cursor_ms, end_ms=chunk_end)
            if not page:
                cursor_ms = chunk_end
                continue
            out.extend(page)
            cursor_ms = max(page[-1].ts * 1000 + 60_000, chunk_end)
        seen: dict[int, Candle] = {c.ts: c for c in out}
        return [seen[k] for k in sorted(seen)]


def _split_symbol(symbol: str) -> tuple[str, str]:
    """``"BTCUSDT" -> ("BTC", "USDT")``, trying each known quote in turn."""
    for quote in QUOTE_PRIORITY:
        if symbol.endswith(quote) and len(symbol) > len(quote):
            return symbol[: -len(quote)], quote
    return "", ""


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
