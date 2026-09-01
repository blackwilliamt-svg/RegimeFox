"""CSV and tax exports.

Two distinct products:

* **Trade history** - every trade from every instance, with entry/exit reasons
  and the manual flag, for analysis.
* **Tax summary** - *live trades only*. Paper and shadow trades are excluded
  because they are not real taxable events; including them would inflate the
  numbers with fills that never happened. Output is totalled per token and for
  the period as a whole, so it can be handed to a preparer or imported without
  the user reconstructing anything from the raw log.
"""
from __future__ import annotations

import csv
import io
import sqlite3
import time
from dataclasses import dataclass
from typing import Any, Iterable

from . import db

TRADE_COLUMNS = [
    "id", "instance", "symbol", "mint", "entry_time", "entry_price", "exit_time",
    "exit_price", "qty", "size_usd", "proceeds_usd", "fees_usd", "pnl_usd",
    "pnl_pct", "hold_minutes", "manual", "entry_reason", "exit_reason",
]


def _fmt_ts(ts: Any) -> str:
    if not ts:
        return ""
    return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(int(ts)))


def fetch_trades(
    *,
    instance: str | None = None,
    since: int | None = None,
    until: int | None = None,
    conn: sqlite3.Connection | None = None,
) -> list[sqlite3.Row]:
    conn = conn or db.connect()
    sql = "SELECT * FROM trades WHERE 1=1"
    args: list[Any] = []
    if instance:
        sql += " AND instance = ?"
        args.append(instance)
    if since:
        sql += " AND exit_ts >= ?"
        args.append(int(since))
    if until:
        sql += " AND exit_ts <= ?"
        args.append(int(until))
    return conn.execute(sql + " ORDER BY exit_ts", args).fetchall()


def trades_csv(rows: Iterable[sqlite3.Row]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(TRADE_COLUMNS)
    for r in rows:
        writer.writerow(
            [
                r["id"], r["instance"], r["symbol"] or "", r["mint"],
                _fmt_ts(r["entry_ts"]), f"{float(r['entry_price']):.10g}",
                _fmt_ts(r["exit_ts"]), f"{float(r['exit_price']):.10g}",
                f"{float(r['qty']):.10g}", f"{float(r['size_usd']):.2f}",
                f"{float(r['proceeds_usd']):.2f}", f"{float(r['fees_usd']):.4f}",
                f"{float(r['pnl_usd']):.4f}", f"{float(r['pnl_pct']) * 100:.4f}",
                round(int(r["hold_seconds"] or 0) / 60.0, 1),
                "yes" if r["manual"] else "no",
                (r["entry_reason"] or "").replace("\n", " "),
                (r["exit_reason"] or "").replace("\n", " "),
            ]
        )
    return buf.getvalue()


# --------------------------------------------------------------------------
# Tax summary - live trades only
# --------------------------------------------------------------------------
@dataclass
class TaxSummary:
    period_from: str
    period_to: str
    trades: int
    proceeds: float
    cost_basis: float
    fees: float
    realized_gain: float
    short_term_gain: float
    long_term_gain: float
    by_token: list[dict[str, Any]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "period_from": self.period_from,
            "period_to": self.period_to,
            "trades": self.trades,
            "proceeds": round(self.proceeds, 2),
            "cost_basis": round(self.cost_basis, 2),
            "fees": round(self.fees, 2),
            "realized_gain": round(self.realized_gain, 2),
            "short_term_gain": round(self.short_term_gain, 2),
            "long_term_gain": round(self.long_term_gain, 2),
            "by_token": self.by_token,
        }


LONG_TERM_SECONDS = 365 * 86400


def tax_summary(
    *, year: int | None = None, since: int | None = None, until: int | None = None,
    conn: sqlite3.Connection | None = None,
) -> TaxSummary:
    """Totals realized gains and losses across live trades for a period.

    Paper and shadow instances are excluded - they are not taxable events.
    """
    conn = conn or db.connect()
    if year is not None:
        since = int(time.mktime((year, 1, 1, 0, 0, 0, 0, 1, 0)))
        until = int(time.mktime((year + 1, 1, 1, 0, 0, 0, 0, 1, 0))) - 1

    rows = fetch_trades(instance="live", since=since, until=until, conn=conn)

    by_token: dict[str, dict[str, Any]] = {}
    proceeds = cost = fees = short = long = 0.0

    for r in rows:
        size = float(r["size_usd"])
        got = float(r["proceeds_usd"])
        fee = float(r["fees_usd"])
        pnl = float(r["pnl_usd"])
        held = int(r["hold_seconds"] or 0)

        proceeds += got
        cost += size
        fees += fee
        if held >= LONG_TERM_SECONDS:
            long += pnl
        else:
            short += pnl

        key = r["mint"]
        entry = by_token.setdefault(
            key,
            {
                "symbol": r["symbol"] or "",
                "mint": key,
                "trades": 0,
                "proceeds": 0.0,
                "cost_basis": 0.0,
                "fees": 0.0,
                "realized_gain": 0.0,
            },
        )
        entry["trades"] += 1
        entry["proceeds"] += got
        entry["cost_basis"] += size
        entry["fees"] += fee
        entry["realized_gain"] += pnl

    for entry in by_token.values():
        for k in ("proceeds", "cost_basis", "fees", "realized_gain"):
            entry[k] = round(entry[k], 2)

    return TaxSummary(
        period_from=_fmt_ts(since) or (str(rows[0]["exit_ts"]) if rows else ""),
        period_to=_fmt_ts(until) or _fmt_ts(rows[-1]["exit_ts"] if rows else None),
        trades=len(rows),
        proceeds=proceeds,
        cost_basis=cost,
        fees=fees,
        realized_gain=proceeds - cost,
        short_term_gain=short,
        long_term_gain=long,
        by_token=sorted(by_token.values(), key=lambda d: d["realized_gain"], reverse=True),
    )


TAX_COLUMNS = [
    "symbol", "mint", "trades", "proceeds_usd", "cost_basis_usd",
    "fees_usd", "realized_gain_usd",
]


def tax_csv(summary: TaxSummary) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow([f"Tax summary - LIVE trades only ({summary.period_from} to {summary.period_to})"])
    writer.writerow([])
    writer.writerow(TAX_COLUMNS)
    for row in summary.by_token:
        writer.writerow(
            [
                row["symbol"], row["mint"], row["trades"], f"{row['proceeds']:.2f}",
                f"{row['cost_basis']:.2f}", f"{row['fees']:.2f}",
                f"{row['realized_gain']:.2f}",
            ]
        )
    writer.writerow([])
    writer.writerow(["TOTALS"])
    writer.writerow(["Trades", summary.trades])
    writer.writerow(["Gross proceeds", f"{summary.proceeds:.2f}"])
    writer.writerow(["Cost basis", f"{summary.cost_basis:.2f}"])
    writer.writerow(["Fees paid", f"{summary.fees:.2f}"])
    writer.writerow(["Net realized gain/loss", f"{summary.realized_gain:.2f}"])
    writer.writerow(["  of which short-term (<1yr)", f"{summary.short_term_gain:.2f}"])
    writer.writerow(["  of which long-term (>=1yr)", f"{summary.long_term_gain:.2f}"])
    writer.writerow([])
    writer.writerow(
        ["Note: paper and shadow-mode trades are excluded - they are not taxable events."]
    )
    return buf.getvalue()
