"""SQLite storage.

One file holds everything: historical candles, the trade journal, live position
state, the event feed, auth records, and the command queue the dashboard uses to
talk to the worker.

Two design points worth knowing before editing:

* **The worker and the web app are separate processes** and share *only* this
  database. The worker owns all trading; the dashboard never touches a position
  directly, it enqueues a row in ``commands`` that the worker drains. That keeps
  a single writer on the trading path.
* **WAL mode** is on so the dashboard can read while the worker writes.

``instance`` distinguishes the three parallel engines the spec requires:
``live``, ``paper`` (the permanent parallel comparison), and ``shadow`` (a
candidate rule set running alongside before promotion).
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

INSTANCES = ("live", "paper", "shadow")

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA foreign_keys=ON;

-- Historical OHLCV pulled from Birdeye, plus candles rolled up from live ticks.
CREATE TABLE IF NOT EXISTS candles (
    mint        TEXT    NOT NULL,
    interval    TEXT    NOT NULL,          -- e.g. '10m'
    ts          INTEGER NOT NULL,          -- unix seconds, candle open
    open        REAL    NOT NULL,
    high        REAL    NOT NULL,
    low         REAL    NOT NULL,
    close       REAL    NOT NULL,
    volume      REAL    NOT NULL,          -- USD volume
    source      TEXT    NOT NULL DEFAULT 'birdeye',
    PRIMARY KEY (mint, interval, ts)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS idx_candles_ts ON candles(ts);

-- Rolling live prices; used for correlation checks and the live chart tail.
CREATE TABLE IF NOT EXISTS price_ticks (
    mint     TEXT    NOT NULL,
    ts       INTEGER NOT NULL,
    price    REAL    NOT NULL,
    PRIMARY KEY (mint, ts)
) WITHOUT ROWID;

-- The tradeable universe as it stands right now.
CREATE TABLE IF NOT EXISTS universe (
    mint            TEXT PRIMARY KEY,
    symbol          TEXT,
    name            TEXT,
    decimals        INTEGER,
    liquidity_usd   REAL,
    volume_24h_usd  REAL,
    mcap            REAL,
    holder_count    INTEGER,
    organic_score   REAL,
    is_verified     INTEGER DEFAULT 0,
    first_pool_at   INTEGER,
    updated_at      INTEGER NOT NULL
);

-- Point-in-time record of what passed the filters on a given day. The backtest
-- builds its basket from THIS, not from what is tradeable today - that is what
-- keeps survivorship bias out of the results.
CREATE TABLE IF NOT EXISTS universe_history (
    day             TEXT    NOT NULL,      -- 'YYYY-MM-DD' UTC
    mint            TEXT    NOT NULL,
    symbol          TEXT,
    liquidity_usd   REAL,
    volume_24h_usd  REAL,
    passed_safety   INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (day, mint)
) WITHOUT ROWID;

-- Rug Check results, cached so we re-test on a cooldown rather than every poll.
CREATE TABLE IF NOT EXISTS safety_reports (
    mint            TEXT PRIMARY KEY,
    passed          INTEGER NOT NULL,
    score           REAL,
    reasons         TEXT,                  -- JSON list of failure strings
    detail          TEXT,                  -- JSON snapshot of the raw checks
    checked_at      INTEGER NOT NULL,
    recheck_after   INTEGER NOT NULL       -- unix seconds; the cooldown gate
);

-- Open and closed positions. Written on EVERY change so a crash cannot lose one.
CREATE TABLE IF NOT EXISTS positions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    instance            TEXT    NOT NULL,
    mint                TEXT    NOT NULL,
    symbol              TEXT,
    status              TEXT    NOT NULL,  -- open | closed
    entry_price         REAL    NOT NULL,
    entry_ts            INTEGER NOT NULL,
    qty                 REAL    NOT NULL,  -- token units
    size_usd            REAL    NOT NULL,
    hard_stop           REAL    NOT NULL,
    take_profit         REAL    NOT NULL,
    trailing_stop       REAL,              -- NULL until armed
    trailing_armed      INTEGER NOT NULL DEFAULT 0,
    high_water_price    REAL,
    initial_risk        REAL    NOT NULL,  -- entry - hard_stop, i.e. 1R
    rr_target           REAL    NOT NULL,
    entry_reason        TEXT,
    entry_snapshot      TEXT,              -- JSON indicator state that justified it
    invalidation_count  INTEGER NOT NULL DEFAULT 0,
    entry_fee_usd       REAL    NOT NULL DEFAULT 0,
    entry_tx            TEXT,
    exit_price          REAL,
    exit_ts             INTEGER,
    exit_reason         TEXT,
    exit_fee_usd        REAL DEFAULT 0,
    exit_tx             TEXT,
    pnl_usd             REAL,
    pnl_pct             REAL,
    manual              INTEGER NOT NULL DEFAULT 0,
    updated_at          INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_pos_open   ON positions(instance, status);
CREATE INDEX IF NOT EXISTS idx_pos_exit   ON positions(instance, exit_ts);

-- Immutable journal row per completed trade; the CSV and tax exports read this.
CREATE TABLE IF NOT EXISTS trades (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    position_id     INTEGER,
    instance        TEXT    NOT NULL,
    mint            TEXT    NOT NULL,
    symbol          TEXT,
    entry_ts        INTEGER NOT NULL,
    entry_price     REAL    NOT NULL,
    exit_ts         INTEGER NOT NULL,
    exit_price      REAL    NOT NULL,
    qty             REAL    NOT NULL,
    size_usd        REAL    NOT NULL,
    proceeds_usd    REAL    NOT NULL,
    fees_usd        REAL    NOT NULL,
    pnl_usd         REAL    NOT NULL,
    pnl_pct         REAL    NOT NULL,
    entry_reason    TEXT,
    exit_reason     TEXT,
    manual          INTEGER NOT NULL DEFAULT 0,
    hold_seconds    INTEGER,
    FOREIGN KEY (position_id) REFERENCES positions(id)
);
CREATE INDEX IF NOT EXISTS idx_trades_exit ON trades(instance, exit_ts);

-- Plain-language activity feed shown on the dashboard.
CREATE TABLE IF NOT EXISTS events (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    ts        INTEGER NOT NULL,
    level     TEXT    NOT NULL,           -- info | warn | alert
    category  TEXT    NOT NULL,           -- signal | trade | safety | risk | system | auth
    instance  TEXT,
    mint      TEXT,
    message   TEXT    NOT NULL,
    detail    TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts DESC);

-- Account balance over time, per instance; drives the equity curve.
CREATE TABLE IF NOT EXISTS equity (
    instance  TEXT    NOT NULL,
    ts        INTEGER NOT NULL,
    balance   REAL    NOT NULL,
    equity    REAL    NOT NULL,           -- balance + unrealised
    PRIMARY KEY (instance, ts)
) WITHOUT ROWID;

-- Durable engine state: kill switch, circuit breaker, heartbeat, run flag.
-- These MUST survive a restart so a reboot cannot silently clear a halt.
CREATE TABLE IF NOT EXISTS kv (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    updated_at INTEGER NOT NULL
);

-- Dashboard -> worker queue. The web process never trades directly.
CREATE TABLE IF NOT EXISTS commands (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at   INTEGER NOT NULL,
    command      TEXT    NOT NULL,
    payload      TEXT,
    status       TEXT    NOT NULL DEFAULT 'pending',  -- pending|done|failed
    result       TEXT,
    handled_at   INTEGER,
    requested_by TEXT
);
CREATE INDEX IF NOT EXISTS idx_cmd_pending ON commands(status, id);

-- Every settings change, timestamped, with who made it.
CREATE TABLE IF NOT EXISTS settings_audit (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    ts        INTEGER NOT NULL,
    username  TEXT,
    key       TEXT NOT NULL,
    old_value TEXT,
    new_value TEXT
);

-- Long-running job progress (initial Birdeye pull, daily backtest).
CREATE TABLE IF NOT EXISTS job_progress (
    job        TEXT PRIMARY KEY,
    status     TEXT NOT NULL,             -- idle|running|done|failed|cancelled
    done       INTEGER NOT NULL DEFAULT 0,
    total      INTEGER NOT NULL DEFAULT 0,
    message    TEXT,
    started_at INTEGER,
    updated_at INTEGER
);

-- Birdeye compute-unit spend, bucketed by month, so the free tier's budget
-- is visible before it is blown rather than after.
CREATE TABLE IF NOT EXISTS api_budget (
    provider  TEXT NOT NULL,
    month     TEXT NOT NULL,              -- 'YYYY-MM'
    units     INTEGER NOT NULL DEFAULT 0,
    calls     INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (provider, month)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS backtest_runs (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    ts             INTEGER NOT NULL,
    from_ts        INTEGER,
    to_ts          INTEGER,
    trades         INTEGER,
    win_rate       REAL,
    profit_factor  REAL,
    max_drawdown   REAL,
    total_return   REAL,
    avg_win        REAL,
    avg_loss       REAL,
    params         TEXT,
    detail         TEXT
);

-- Dashboard auth. Password is argon2/bcrypt hashed; TOTP secret sits alongside.
CREATE TABLE IF NOT EXISTS users (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    username       TEXT UNIQUE NOT NULL,
    password_hash  TEXT NOT NULL,
    totp_secret    TEXT,
    totp_confirmed INTEGER NOT NULL DEFAULT 0,
    backup_codes   TEXT,                  -- JSON list of hashed one-time codes
    created_at     INTEGER NOT NULL,
    last_login     INTEGER
);

CREATE TABLE IF NOT EXISTS login_attempts (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    ts         INTEGER NOT NULL,
    username   TEXT,
    ip         TEXT,
    success    INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_login_ts ON login_attempts(ts);

-- API keys rotated from the dashboard, encrypted at rest. The environment
-- remains the default source; a row here overrides it once set.
CREATE TABLE IF NOT EXISTS api_keys (
    provider    TEXT PRIMARY KEY,
    ciphertext  TEXT NOT NULL,
    last4       TEXT,
    updated_at  INTEGER NOT NULL,
    updated_by  TEXT
);
"""

_local = threading.local()


def db_path() -> Path:
    return Path(os.getenv("SOLBOT_DB", "data/solbot.db"))


def connect(path: str | Path | None = None) -> sqlite3.Connection:
    """Per-thread connection. SQLite objects are not safe to share across threads."""
    target = str(path or db_path())
    cached = getattr(_local, "conns", None)
    if cached is None:
        cached = {}
        _local.conns = cached
    conn = cached.get(target)
    if conn is None:
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(target, timeout=30.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=30000")
        cached[target] = conn
    return conn


def init_db(path: str | Path | None = None) -> sqlite3.Connection:
    conn = connect(path)
    conn.executescript(SCHEMA)
    return conn


def close_all() -> None:
    for conn in getattr(_local, "conns", {}).values():
        try:
            conn.close()
        except sqlite3.Error:
            pass
    _local.conns = {}


@contextmanager
def transaction(conn: sqlite3.Connection | None = None) -> Iterator[sqlite3.Connection]:
    """Explicit transaction; commits on success, rolls back on any exception."""
    conn = conn or connect()
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except Exception:
        conn.execute("ROLLBACK")
        raise
    else:
        conn.execute("COMMIT")


def now() -> int:
    return int(time.time())


# --------------------------------------------------------------------------
# key/value state
# --------------------------------------------------------------------------
def kv_get(key: str, default: Any = None, conn: sqlite3.Connection | None = None) -> Any:
    conn = conn or connect()
    row = conn.execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
    if row is None:
        return default
    try:
        return json.loads(row["value"])
    except json.JSONDecodeError:
        return row["value"]


def kv_set(key: str, value: Any, conn: sqlite3.Connection | None = None) -> None:
    conn = conn or connect()
    conn.execute(
        "INSERT INTO kv(key, value, updated_at) VALUES (?, ?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value, "
        "updated_at = excluded.updated_at",
        (key, json.dumps(value), now()),
    )


def kv_all(prefix: str = "", conn: sqlite3.Connection | None = None) -> dict[str, Any]:
    conn = conn or connect()
    rows = conn.execute(
        "SELECT key, value FROM kv WHERE key LIKE ?", (f"{prefix}%",)
    ).fetchall()
    out: dict[str, Any] = {}
    for r in rows:
        try:
            out[r["key"]] = json.loads(r["value"])
        except json.JSONDecodeError:
            out[r["key"]] = r["value"]
    return out


# --------------------------------------------------------------------------
# events / journal
# --------------------------------------------------------------------------
def log_event(
    message: str,
    *,
    level: str = "info",
    category: str = "system",
    instance: str | None = None,
    mint: str | None = None,
    detail: Any = None,
    conn: sqlite3.Connection | None = None,
) -> None:
    conn = conn or connect()
    conn.execute(
        "INSERT INTO events(ts, level, category, instance, mint, message, detail) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            now(),
            level,
            category,
            instance,
            mint,
            message,
            json.dumps(detail) if detail is not None else None,
        ),
    )


def recent_events(
    limit: int = 200,
    *,
    since_id: int | None = None,
    category: str | None = None,
    conn: sqlite3.Connection | None = None,
) -> list[sqlite3.Row]:
    conn = conn or connect()
    sql = "SELECT * FROM events WHERE 1=1"
    args: list[Any] = []
    if since_id is not None:
        sql += " AND id > ?"
        args.append(since_id)
    if category:
        sql += " AND category = ?"
        args.append(category)
    sql += " ORDER BY id DESC LIMIT ?"
    args.append(limit)
    return conn.execute(sql, args).fetchall()


# --------------------------------------------------------------------------
# commands (dashboard -> worker)
# --------------------------------------------------------------------------
def enqueue_command(
    command: str,
    payload: dict[str, Any] | None = None,
    *,
    requested_by: str | None = None,
    conn: sqlite3.Connection | None = None,
) -> int:
    conn = conn or connect()
    cur = conn.execute(
        "INSERT INTO commands(created_at, command, payload, requested_by) "
        "VALUES (?, ?, ?, ?)",
        (now(), command, json.dumps(payload or {}), requested_by),
    )
    return int(cur.lastrowid)


def pending_commands(conn: sqlite3.Connection | None = None) -> list[sqlite3.Row]:
    conn = conn or connect()
    return conn.execute(
        "SELECT * FROM commands WHERE status = 'pending' ORDER BY id"
    ).fetchall()


def finish_command(
    cmd_id: int,
    status: str,
    result: str | None = None,
    conn: sqlite3.Connection | None = None,
) -> None:
    conn = conn or connect()
    conn.execute(
        "UPDATE commands SET status = ?, result = ?, handled_at = ? WHERE id = ?",
        (status, result, now(), cmd_id),
    )


# --------------------------------------------------------------------------
# job progress
# --------------------------------------------------------------------------
def set_progress(
    job: str,
    *,
    status: str,
    done: int = 0,
    total: int = 0,
    message: str = "",
    conn: sqlite3.Connection | None = None,
) -> None:
    conn = conn or connect()
    ts = now()
    conn.execute(
        "INSERT INTO job_progress(job, status, done, total, message, started_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(job) DO UPDATE SET status=excluded.status, done=excluded.done, "
        "total=excluded.total, message=excluded.message, updated_at=excluded.updated_at, "
        "started_at=CASE WHEN excluded.status='running' AND job_progress.status!='running' "
        "THEN excluded.started_at ELSE job_progress.started_at END",
        (job, status, done, total, message, ts, ts),
    )


def get_progress(job: str, conn: sqlite3.Connection | None = None) -> dict[str, Any]:
    conn = conn or connect()
    row = conn.execute("SELECT * FROM job_progress WHERE job = ?", (job,)).fetchone()
    if row is None:
        return {"job": job, "status": "idle", "done": 0, "total": 0, "message": ""}
    d = dict(row)
    total = d.get("total") or 0
    d["percent"] = round(100.0 * (d.get("done") or 0) / total, 1) if total else 0.0
    return d


# --------------------------------------------------------------------------
# API budget accounting
# --------------------------------------------------------------------------
def add_api_usage(
    provider: str, units: int, *, conn: sqlite3.Connection | None = None
) -> None:
    conn = conn or connect()
    month = time.strftime("%Y-%m", time.gmtime())
    conn.execute(
        "INSERT INTO api_budget(provider, month, units, calls) VALUES (?, ?, ?, 1) "
        "ON CONFLICT(provider, month) DO UPDATE SET units = units + excluded.units, "
        "calls = calls + 1",
        (provider, month, units),
    )


def api_usage(provider: str, conn: sqlite3.Connection | None = None) -> dict[str, int]:
    conn = conn or connect()
    month = time.strftime("%Y-%m", time.gmtime())
    row = conn.execute(
        "SELECT units, calls FROM api_budget WHERE provider = ? AND month = ?",
        (provider, month),
    ).fetchone()
    return {"units": row["units"] if row else 0, "calls": row["calls"] if row else 0}


# --------------------------------------------------------------------------
# candles
# --------------------------------------------------------------------------
def upsert_candles(
    mint: str,
    interval: str,
    rows: Iterable[Sequence[Any]],
    *,
    source: str = "birdeye",
    conn: sqlite3.Connection | None = None,
) -> int:
    """rows: iterable of (ts, open, high, low, close, volume)."""
    conn = conn or connect()
    payload = [(mint, interval, int(r[0]), *[float(x) for x in r[1:6]], source) for r in rows]
    if not payload:
        return 0
    conn.executemany(
        "INSERT INTO candles(mint, interval, ts, open, high, low, close, volume, source) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(mint, interval, ts) DO UPDATE SET "
        "open=excluded.open, high=excluded.high, low=excluded.low, "
        "close=excluded.close, volume=excluded.volume",
        payload,
    )
    return len(payload)


def load_candles(
    mint: str,
    interval: str,
    *,
    limit: int | None = None,
    since: int | None = None,
    until: int | None = None,
    conn: sqlite3.Connection | None = None,
) -> list[sqlite3.Row]:
    conn = conn or connect()
    sql = "SELECT ts, open, high, low, close, volume FROM candles WHERE mint=? AND interval=?"
    args: list[Any] = [mint, interval]
    if since is not None:
        sql += " AND ts >= ?"
        args.append(int(since))
    if until is not None:
        sql += " AND ts <= ?"
        args.append(int(until))
    if limit:
        sql += " ORDER BY ts DESC LIMIT ?"
        args.append(int(limit))
        rows = conn.execute(sql, args).fetchall()
        return list(reversed(rows))
    sql += " ORDER BY ts"
    return conn.execute(sql, args).fetchall()


def latest_candle_ts(
    mint: str, interval: str, conn: sqlite3.Connection | None = None
) -> int | None:
    conn = conn or connect()
    row = conn.execute(
        "SELECT MAX(ts) AS t FROM candles WHERE mint = ? AND interval = ?", (mint, interval)
    ).fetchone()
    return int(row["t"]) if row and row["t"] is not None else None


# --------------------------------------------------------------------------
# retention - keeps the database small enough for a 2GB droplet
# --------------------------------------------------------------------------
def prune(
    *,
    candle_days: int,
    tick_hours: int,
    event_days: int,
    conn: sqlite3.Connection | None = None,
) -> dict[str, int]:
    conn = conn or connect()
    t = now()
    deleted = {}
    cur = conn.execute("DELETE FROM candles WHERE ts < ?", (t - candle_days * 86400,))
    deleted["candles"] = cur.rowcount
    cur = conn.execute("DELETE FROM price_ticks WHERE ts < ?", (t - tick_hours * 3600,))
    deleted["price_ticks"] = cur.rowcount
    cur = conn.execute("DELETE FROM events WHERE ts < ?", (t - event_days * 86400,))
    deleted["events"] = cur.rowcount
    cur = conn.execute(
        "DELETE FROM commands WHERE status != 'pending' AND created_at < ?",
        (t - 7 * 86400,),
    )
    deleted["commands"] = cur.rowcount
    cur = conn.execute("DELETE FROM login_attempts WHERE ts < ?", (t - 30 * 86400,))
    deleted["login_attempts"] = cur.rowcount
    return deleted


def vacuum(conn: sqlite3.Connection | None = None) -> None:
    conn = conn or connect()
    conn.execute("VACUUM")
