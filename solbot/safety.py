"""Rug Check safety gate.

Runs before any entry signal is *evaluated*, not after - a token that fails here
never reaches the strategy, so no scan budget is spent reasoning about something
that cannot be traded.

Results are cached in ``safety_reports`` with an explicit ``recheck_after``
timestamp. A failure is not a permanent blacklist: it is retried on a cooldown
(4h by default) so a token whose risk profile genuinely improves - liquidity
later locked, mint authority revoked - can become eligible again, without
burning API calls re-checking it every poll cycle. Passes carry a shorter TTL
so a token that *degrades* is caught too.

The gate fails closed. An unreadable RugCheck response counts as a failure, so
an API outage stops new entries rather than waving unverified tokens through.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any

from . import db
from .clients import RugCheckClient, RugReport, TokenInfo

log = logging.getLogger(__name__)


@dataclass(slots=True)
class SafetyVerdict:
    mint: str
    passed: bool
    reasons: list[str] = field(default_factory=list)
    checked_at: int = 0
    recheck_after: int = 0
    cached: bool = False
    score: float = 0.0
    detail: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        return "clean" if self.passed else "; ".join(self.reasons)


class SafetyGate:
    def __init__(self, client: RugCheckClient, cfg: dict[str, Any]) -> None:
        self.client = client
        self.cfg = cfg

    def update_config(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg

    # ------------------------------------------------------------------
    def check(
        self,
        mint: str,
        *,
        token: TokenInfo | None = None,
        force: bool = False,
        conn: sqlite3.Connection | None = None,
    ) -> SafetyVerdict:
        conn = conn or db.connect()
        if not self.cfg.get("rugcheck_enabled", True):
            return SafetyVerdict(mint, True, ["rug check disabled in settings"])

        now = db.now()
        if not force:
            cached = self._cached(mint, now, conn)
            if cached is not None:
                return cached

        report = self.client.report(mint)
        verdict = self.evaluate(report, token=token, now=now)
        self._store(verdict, report, conn)

        if not verdict.passed:
            db.log_event(
                f"Rug Check rejected {token.symbol if token else mint[:8]}: "
                f"{verdict.summary()}",
                level="warn",
                category="safety",
                mint=mint,
                detail={"reasons": verdict.reasons},
                conn=conn,
            )
        return verdict

    # ------------------------------------------------------------------
    def evaluate(
        self, report: RugReport, *, token: TokenInfo | None = None, now: int | None = None
    ) -> SafetyVerdict:
        """Apply every hard filter. Pure function - the backtest uses it too."""
        cfg = self.cfg
        now = now or db.now()
        reasons: list[str] = []

        if not report.ok:
            reasons.append("rug check report unavailable (failing closed)")
            return SafetyVerdict(
                report.mint,
                False,
                reasons,
                checked_at=now,
                # Retry an outage sooner than a genuine failure.
                recheck_after=now + min(900, int(cfg["rugcheck_fail_cooldown_seconds"])),
                score=report.score,
            )

        if report.rugged:
            reasons.append("flagged as rugged")

        if cfg["rugcheck_require_mint_revoked"] and not report.mint_authority_revoked:
            reasons.append("mint authority still active")

        if cfg["rugcheck_require_freeze_revoked"] and not report.freeze_authority_revoked:
            reasons.append("freeze authority still active")

        # --- liquidity locked -------------------------------------------
        # The spec's intent is "the deployer must not be able to pull the rug".
        # A literal LP-lock percentage only measures that on venues with a
        # fungible LP token to burn. Modern Solana liquidity mostly sits in
        # concentrated-liquidity pools (Orca Whirlpool, Meteora DLMM, Raydium
        # CLMM) where no such token exists, so RugCheck reports ~0% locked for
        # essentially every established token - applying the threshold
        # literally rejects the entire tradeable universe.
        #
        # Market count is the honest proxy for the same question: liquidity
        # spread across many independent pools cannot be withdrawn by one
        # actor, whereas a token with a handful of pools is exactly the case
        # the lock was meant to catch. Verified against live data 2026-08-31:
        # established tokens show 69-459 markets, a fresh pump.fun token 6.
        lp_waiver = int(cfg.get("rugcheck_lp_lock_waiver_markets", 20))
        lp_lock_waived = report.market_count >= lp_waiver > 0
        if not lp_lock_waived and report.lp_locked_pct < float(cfg["rugcheck_min_lp_locked_pct"]):
            reasons.append(
                f"only {report.lp_locked_pct:.0f}% of liquidity locked "
                f"(need {cfg['rugcheck_min_lp_locked_pct']:.0f}%) across "
                f"{report.market_count} pool(s)"
            )

        if report.top_holder_pct > float(cfg["rugcheck_max_top_holder_pct"]):
            reasons.append(
                f"top holder holds {report.top_holder_pct:.0f}% "
                f"(max {cfg['rugcheck_max_top_holder_pct']:.0f}%)"
            )

        if report.insider_pct > float(cfg["rugcheck_max_insider_pct"]):
            reasons.append(
                f"insiders hold {report.insider_pct:.0f}% "
                f"(max {cfg['rugcheck_max_insider_pct']:.0f}%)"
            )

        liquidity = report.total_liquidity_usd or (token.liquidity if token else 0.0)
        if liquidity < float(cfg["rugcheck_min_liquidity_usd"]):
            reasons.append(
                f"liquidity ${liquidity:,.0f} below the "
                f"${cfg['rugcheck_min_liquidity_usd']:,.0f} floor"
            )

        # A transfer fee silently taxes every round trip; treat any as
        # disqualifying rather than trying to model it.
        if report.transfer_fee_pct > 0:
            reasons.append(f"token charges a {report.transfer_fee_pct:.2f}% transfer fee")

        danger = report.danger_risks()
        if danger:
            reasons.append(f"rug check danger flags: {', '.join(danger[:4])}")

        age_hours = self._age_hours(report, token, now)
        min_age = float(cfg["rugcheck_min_token_age_hours"])
        if age_hours is None:
            if min_age > 0:
                reasons.append("token age unknown")
        elif age_hours < min_age:
            reasons.append(f"token is {age_hours:.0f}h old (need {min_age:.0f}h)")

        passed = not reasons
        ttl = int(
            cfg["rugcheck_pass_ttl_seconds"] if passed else cfg["rugcheck_fail_cooldown_seconds"]
        )
        return SafetyVerdict(
            mint=report.mint,
            passed=passed,
            reasons=reasons,
            checked_at=now,
            recheck_after=now + ttl,
            score=report.score,
            detail={
                "lp_locked_pct": round(report.lp_locked_pct, 2),
                "lp_lock_waived": lp_lock_waived,
                "market_count": report.market_count,
                "lp_providers": report.lp_providers,
                "top_holder_pct": round(report.top_holder_pct, 2),
                "top10_pct": round(report.top10_pct, 2),
                "insider_pct": round(report.insider_pct, 2),
                "holders": report.total_holders,
                "liquidity_usd": round(liquidity, 2),
                "mint_authority_revoked": report.mint_authority_revoked,
                "freeze_authority_revoked": report.freeze_authority_revoked,
                "age_hours": round(age_hours, 1) if age_hours is not None else None,
                "score": report.score,
            },
        )

    @staticmethod
    def _age_hours(
        report: RugReport, token: TokenInfo | None, now: int
    ) -> float | None:
        if token and token.first_pool_at:
            return max(0.0, (now - token.first_pool_at) / 3600.0)
        created = (report.raw.get("token") or {}).get("createdAt") if report.raw else None
        if isinstance(created, (int, float)) and created > 0:
            ts = float(created)
            if ts > 1e12:  # milliseconds
                ts /= 1000.0
            return max(0.0, (now - ts) / 3600.0)
        return None

    # ------------------------------------------------------------------
    def _cached(
        self, mint: str, now: int, conn: sqlite3.Connection
    ) -> SafetyVerdict | None:
        row = conn.execute(
            "SELECT * FROM safety_reports WHERE mint = ?", (mint,)
        ).fetchone()
        if row is None or now >= int(row["recheck_after"]):
            return None
        try:
            reasons = json.loads(row["reasons"] or "[]")
        except json.JSONDecodeError:
            reasons = []
        try:
            detail = json.loads(row["detail"] or "{}")
        except json.JSONDecodeError:
            detail = {}
        return SafetyVerdict(
            mint=mint,
            passed=bool(row["passed"]),
            reasons=reasons,
            checked_at=int(row["checked_at"]),
            recheck_after=int(row["recheck_after"]),
            cached=True,
            score=float(row["score"] or 0.0),
            detail=detail,
        )

    @staticmethod
    def _store(verdict: SafetyVerdict, report: RugReport, conn: sqlite3.Connection) -> None:
        conn.execute(
            "INSERT INTO safety_reports(mint, passed, score, reasons, detail, checked_at, "
            "recheck_after) VALUES (?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(mint) DO UPDATE SET passed=excluded.passed, score=excluded.score, "
            "reasons=excluded.reasons, detail=excluded.detail, checked_at=excluded.checked_at, "
            "recheck_after=excluded.recheck_after",
            (
                verdict.mint,
                1 if verdict.passed else 0,
                verdict.score,
                json.dumps(verdict.reasons),
                json.dumps(verdict.detail),
                verdict.checked_at,
                verdict.recheck_after,
            ),
        )

    # ------------------------------------------------------------------
    def prescreen(self, token: TokenInfo) -> list[str]:
        """Free checks from the Jupiter token record, before spending a RugCheck call.

        Jupiter already reports the mint and freeze authorities and top-holder
        concentration, so an obviously disqualified token is dropped without
        touching RugCheck's budget at all.
        """
        cfg = self.cfg
        out: list[str] = []
        if cfg["rugcheck_require_mint_revoked"] and token.mint_authority:
            out.append("mint authority still active")
        if cfg["rugcheck_require_freeze_revoked"] and token.freeze_authority:
            out.append("freeze authority still active")
        if token.top_holders_pct > float(cfg["rugcheck_max_top_holder_pct"]) * 2:
            out.append(f"top holders hold {token.top_holders_pct:.0f}%")
        if token.first_pool_at:
            age_h = (db.now() - token.first_pool_at) / 3600.0
            if age_h < float(cfg["rugcheck_min_token_age_hours"]):
                out.append(f"token is {age_h:.0f}h old")
        return out

    def rejection_streak(self, window_seconds: int = 3600, conn: sqlite3.Connection | None = None) -> int:
        """Recent rejection count - the alerting rule watches this."""
        conn = conn or db.connect()
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM safety_reports WHERE passed = 0 AND checked_at > ?",
            (db.now() - window_seconds,),
        ).fetchone()
        return int(row["n"]) if row else 0
