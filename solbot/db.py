"""SQLite storage.

Everything *except* bulk candle history: the trade journal, live position
state, the event feed, auth records, walk-forward/Monte Carlo run results, and
the command queue the dashboard uses to talk to the worker. Candle history
lives as per-coin Parquet files instead (spec 3; see solbot.candlestore) -
a year of 1-minute candles across a hundred coins has no business going
through the same write path as the trading loop's own state.

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
from typing import Any, Iterator, Sequence

INSTANCES = ("live", "paper", "shadow")

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA foreign_keys=ON;

-- Candle history is NOT stored here (spec 3): it lives as per-coin Parquet
-- files under solbot.candlestore, one file per coin per month, read directly
-- by both the live bot and the walk-forward optimizer. Routing it through
-- SQLite would put a year of 1-minute candles across a hundred coins through
-- the same write path as the trading loop's own state.

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
    -- The Binance trading pair this mint was routed from (spec 2), e.g.
    -- "BTCUSDT" - candle history is pulled from Binance under this symbol.
    binance_pair    TEXT,
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
    -- Set per trade by the review gate: ride the move, or lock the gain in.
    exit_style          TEXT,
    trail_override_atr  REAL,
    review_source       TEXT,
    review_conviction   REAL,
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
-- Every parameter change, whether made by the operator from the dashboard or
-- by the bot itself (a promoted walk-forward bundle) - spec 6b. `username` is
-- "auto-promotion" for a bot-made change; `reason` names the run/evidence
-- that triggered it, and is blank for an ordinary manual edit.
CREATE TABLE IF NOT EXISTS settings_audit (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    ts        INTEGER NOT NULL,
    username  TEXT,
    key       TEXT NOT NULL,
    old_value TEXT,
    new_value TEXT,
    reason    TEXT
);

-- Long-running job progress (historical candle pull, daily backtest).
CREATE TABLE IF NOT EXISTS job_progress (
    job        TEXT PRIMARY KEY,
    status     TEXT NOT NULL,             -- idle|running|done|failed|cancelled
    done       INTEGER NOT NULL DEFAULT 0,
    total      INTEGER NOT NULL DEFAULT 0,
    message    TEXT,
    started_at INTEGER,
    updated_at INTEGER
);

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

-- Every fill, with what it actually cost. The Monte Carlo module resamples
-- slippage and fees from THESE rather than from the configured constants, so a
-- strategy whose edge is thinner than its real fill costs is caught.
CREATE TABLE IF NOT EXISTS execution_fills (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            INTEGER NOT NULL,
    instance      TEXT    NOT NULL,
    mint          TEXT    NOT NULL,
    side          TEXT    NOT NULL,        -- buy | sell
    notional_usd  REAL    NOT NULL,
    slippage_pct  REAL    NOT NULL,
    fee_usd       REAL    NOT NULL,
    fee_pct       REAL    NOT NULL,
    simulated     INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_fills_ts ON execution_fills(ts DESC);

-- Walk-forward runs happening on the optimizer PC, mirrored here so the
-- dashboard can show a live feed of a search running on another machine.
CREATE TABLE IF NOT EXISTS optimizer_runs (
    run_id       INTEGER PRIMARY KEY,      -- the optimizer's own run id
    started_at   INTEGER NOT NULL,
    updated_at   INTEGER NOT NULL,
    status       TEXT    NOT NULL,         -- running|done|failed
    label        TEXT,
    coverage     TEXT,
    summary      TEXT,
    monte_carlo  TEXT,
    stress       TEXT,
    notified     INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS optimizer_feed (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id    INTEGER NOT NULL,
    remote_id INTEGER NOT NULL,            -- the line's id on the optimizer
    ts        INTEGER NOT NULL,
    level     TEXT    NOT NULL,
    message   TEXT    NOT NULL,
    detail    TEXT,
    UNIQUE (run_id, remote_id)
);
CREATE INDEX IF NOT EXISTS idx_optfeed ON optimizer_feed(run_id, id);

-- Parameter bundles pulled from the hand-off repository.
CREATE TABLE IF NOT EXISTS param_bundles (
    fingerprint   TEXT PRIMARY KEY,
    received_at   INTEGER NOT NULL,
    generated_at  INTEGER,
    source_commit TEXT,
    payload       TEXT    NOT NULL,        -- the whole bundle JSON
    status        TEXT    NOT NULL,        -- pending|shadow|promoted|rejected|superseded
    shadow_since  INTEGER,
    installed_at  INTEGER,
    note          TEXT
);

-- Auto-promotion decisions, kept whether they promoted or declined; a refusal
-- is as much a part of the record as a promotion.
CREATE TABLE IF NOT EXISTS promotions (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ts           INTEGER NOT NULL,
    fingerprint  TEXT,
    promoted     INTEGER NOT NULL,
    reason       TEXT    NOT NULL,
    shadow_days  REAL,
    margin       REAL,
    confidence   REAL,
    detail       TEXT
);

-- Live/paper performance against backtest expectation (spec 4.5).
CREATE TABLE IF NOT EXISTS drift_samples (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    ts                    INTEGER NOT NULL,
    instance              TEXT    NOT NULL,
    window_days           INTEGER NOT NULL,
    live_trades           INTEGER NOT NULL,
    live_win_rate         REAL,
    live_expectancy       REAL,
    live_profit_factor    REAL,
    expected_win_rate     REAL,
    expected_expectancy   REAL,
    expected_profit_factor REAL,
    status                TEXT    NOT NULL,   -- ok|watch|drifting|insufficient
    detail                TEXT
);
CREATE INDEX IF NOT EXISTS idx_drift_ts ON drift_samples(instance, ts DESC);

-- Every entry-gate and daily-review decision, from the deterministic rules
-- engine. There is no model in this loop to compare against, so this is a
-- record of what the rules decided and why, not a calibration log.
CREATE TABLE IF NOT EXISTS entry_reviews (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    ts             INTEGER NOT NULL,
    kind           TEXT    NOT NULL,       -- gate | daily
    instance       TEXT,
    mint           TEXT,
    decision       TEXT    NOT NULL,       -- approve | reject | applied | no_change
    conviction     REAL,
    exit_style     TEXT,                   -- ride | lock_in
    detail         TEXT
);
CREATE INDEX IF NOT EXISTS idx_reviews_ts ON entry_reviews(ts DESC);

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


# Columns added to existing tables after the first release. ``CREATE TABLE IF
# NOT EXISTS`` cannot add a column to a table that already exists, so these are
# applied separately; an already-migrated database sees no writes at all.
MIGRATIONS: dict[str, dict[str, str]] = {
    "positions": {
        # Set by the review gate, per trade: whether this position is being
        # ridden for a larger move or trailed tightly to lock the gain in.
        "exit_style": "TEXT",
        "trail_override_atr": "REAL",
        "review_source": "TEXT",
        "review_conviction": "REAL",
    },
    "universe": {
        "binance_pair": "TEXT",
    },
    "settings_audit": {
        "reason": "TEXT",
    },
}


def migrate(conn: sqlite3.Connection) -> list[str]:
    """Add any columns this version expects but an older database lacks."""
    applied: list[str] = []
    for table, columns in MIGRATIONS.items():
        try:
            existing = {
                row["name"] for row in conn.execute(f"PRAGMA table_info({table})")
            }
        except sqlite3.Error:
            continue
        if not existing:
            continue
        for name, decl in columns.items():
            if name in existing:
                continue
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")
            applied.append(f"{table}.{name}")
    return applied


def init_db(path: str | Path | None = None) -> sqlite3.Connection:
    conn = connect(path)
    conn.executescript(SCHEMA)
    applied = migrate(conn)
    if applied:
        import logging

        logging.getLogger(__name__).info("schema migrated: %s", ", ".join(applied))
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
# execution fills - the observed cost model the Monte Carlo resamples
# --------------------------------------------------------------------------
def record_fill(
    *,
    instance: str,
    mint: str,
    side: str,
    notional_usd: float,
    slippage_pct: float,
    fee_usd: float,
    simulated: bool,
    conn: sqlite3.Connection | None = None,
) -> None:
    conn = conn or connect()
    fee_pct = (fee_usd / notional_usd * 100.0) if notional_usd > 0 else 0.0
    conn.execute(
        "INSERT INTO execution_fills(ts, instance, mint, side, notional_usd, "
        "slippage_pct, fee_usd, fee_pct, simulated) VALUES (?,?,?,?,?,?,?,?,?)",
        (
            now(), instance, mint, side, float(notional_usd), float(slippage_pct),
            float(fee_usd), fee_pct, 1 if simulated else 0,
        ),
    )


def execution_samples(
    *,
    days: int = 90,
    instances: Sequence[str] = ("live", "paper"),
    limit: int = 5000,
    conn: sqlite3.Connection | None = None,
) -> list[tuple[float, float]]:
    """``(slippage_pct, fee_pct)`` per side, newest first."""
    conn = conn or connect()
    placeholders = ",".join("?" * len(instances))
    rows = conn.execute(
        f"SELECT slippage_pct, fee_pct FROM execution_fills "
        f"WHERE ts >= ? AND instance IN ({placeholders}) AND notional_usd > 0 "
        f"ORDER BY ts DESC LIMIT ?",
        (now() - int(days) * 86400, *instances, int(limit)),
    ).fetchall()
    return [(float(r["slippage_pct"]), float(r["fee_pct"])) for r in rows]


# --------------------------------------------------------------------------
# retention - keeps the database small enough for a 2GB droplet
#
# Candle history is never pruned (spec 3a): it lives in Parquet, kept
# indefinitely, and is not touched here at all.
# --------------------------------------------------------------------------
def prune(
    *,
    tick_hours: int,
    event_days: int,
    wfmc_days: int | None = None,
    conn: sqlite3.Connection | None = None,
) -> dict[str, int]:
    conn = conn or connect()
    t = now()
    deleted = {}
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

    # The fill history is the Monte Carlo's execution model, so it is kept for a
    # full year rather than the shorter event retention - a thin sample there
    # quietly turns a measured cost model back into an assumed one.
    cur = conn.execute("DELETE FROM execution_fills WHERE ts < ?", (t - 365 * 86400,))
    deleted["execution_fills"] = cur.rowcount
    cur = conn.execute(
        "DELETE FROM drift_samples WHERE ts < ?", (t - event_days * 86400,)
    )
    deleted["drift_samples"] = cur.rowcount
    cur = conn.execute(
        "DELETE FROM entry_reviews WHERE ts < ?", (t - event_days * 86400,)
    )
    deleted["entry_reviews"] = cur.rowcount
    # Feed lines for finished runs age out; a running run's feed is never touched.
    # Walk-forward/Monte Carlo output (spec 6c/7): the fastest-growing storage
    # component, so it gets its own retention window rather than riding on the
    # general event retention. A run still "running" is never touched
    # regardless of age.
    wfmc_cutoff = t - int(wfmc_days if wfmc_days is not None else event_days) * 86400
    old_runs = [
        int(r["run_id"]) for r in conn.execute(
            "SELECT run_id FROM optimizer_runs WHERE status != 'running' AND updated_at < ?",
            (wfmc_cutoff,),
        ).fetchall()
    ]
    deleted["optimizer_feed"] = 0
    deleted["optimizer_runs"] = 0
    if old_runs:
        placeholders = ",".join("?" * len(old_runs))
        cur = conn.execute(
            f"DELETE FROM optimizer_feed WHERE run_id IN ({placeholders})", old_runs
        )
        deleted["optimizer_feed"] = cur.rowcount
        cur = conn.execute(
            f"DELETE FROM optimizer_runs WHERE run_id IN ({placeholders})", old_runs
        )
        deleted["optimizer_runs"] = cur.rowcount

    # A rejected or superseded bundle is only useful for as long as its run
    # is; the active shadow/promoted one is never touched by age alone.
    cur = conn.execute(
        "DELETE FROM param_bundles WHERE status IN ('rejected', 'superseded') "
        "AND received_at < ?",
        (wfmc_cutoff,),
    )
    deleted["param_bundles"] = cur.rowcount
    return deleted


def vacuum(conn: sqlite3.Connection | None = None) -> None:
    conn = conn or connect()
    conn.execute("VACUUM")
