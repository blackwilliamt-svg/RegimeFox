"""Run state for the optimizer: checkpoints, windows, trades, feed lines.

A walk-forward run over a full history takes hours. It must survive the
process being restarted, the search being interrupted, or a window failing,
and pick up where it left off rather than starting over - so every window's
result is committed the moment it completes, and a resumed run replays
nothing.

This is a separate SQLite file from the droplet's trading database
(``data/wfmc.db`` next to ``data/solbot.db``). There is no PC and no git
hand-off repository anywhere in this picture: the daily incremental run lives
entirely inside the droplet's own trading worker process, in-process CPU only,
and the monthly full retest runs on a RunPod GPU worker the droplet itself
provisions, ships candle data to over HTTP, and tears down afterward -
verified, not assumed. Both call :func:`solopt.pipeline.run_pipeline`, which
writes here through this same RunStore either way; nothing here ever
communicates with anything but that one process's own filesystem.
"""
from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS runs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at   INTEGER NOT NULL,
    finished_at  INTEGER,
    status       TEXT    NOT NULL,        -- running|done|failed|cancelled
    label        TEXT,
    config       TEXT    NOT NULL,        -- JSON WalkForwardConfig
    settings     TEXT    NOT NULL,        -- JSON PortfolioSettings
    space        TEXT    NOT NULL,        -- JSON ParamSpace
    bundle       TEXT,                    -- data bundle path/fingerprint
    coverage     TEXT,                    -- JSON panel coverage
    summary      TEXT                     -- JSON WalkForwardResult summary
);

-- One row per in-sample/out-of-sample window. Written on completion, which is
-- what makes a run resumable at window granularity.
CREATE TABLE IF NOT EXISTS windows (
    run_id       INTEGER NOT NULL,
    idx          INTEGER NOT NULL,
    is_from      INTEGER NOT NULL,
    is_to        INTEGER NOT NULL,
    oos_from     INTEGER NOT NULL,
    oos_to       INTEGER NOT NULL,
    status       TEXT    NOT NULL,        -- pending|done|skipped|failed
    params       TEXT,                    -- JSON winning combination
    is_metrics   TEXT,
    oos_metrics  TEXT,
    search       TEXT,                    -- JSON SearchState
    counted      INTEGER NOT NULL DEFAULT 0,
    profitable   INTEGER NOT NULL DEFAULT 0,
    note         TEXT,
    updated_at   INTEGER NOT NULL,
    PRIMARY KEY (run_id, idx)
) WITHOUT ROWID;

-- Out-of-sample trades, kept because the Monte Carlo module resamples exactly
-- these and nothing else.
CREATE TABLE IF NOT EXISTS oos_trades (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id       INTEGER NOT NULL,
    window_idx   INTEGER NOT NULL,
    symbol       TEXT    NOT NULL,
    entry_ts     INTEGER NOT NULL,
    exit_ts      INTEGER NOT NULL,
    entry_price  REAL    NOT NULL,
    exit_price   REAL    NOT NULL,
    size_usd     REAL    NOT NULL,
    pnl_usd      REAL    NOT NULL,
    fees_usd     REAL    NOT NULL,
    r_multiple   REAL,
    exit_reason  TEXT,
    bars_held    INTEGER
);
CREATE INDEX IF NOT EXISTS idx_oos_run ON oos_trades(run_id, exit_ts);

-- The plain-language feed the dashboard renders.
CREATE TABLE IF NOT EXISTS feed (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id       INTEGER NOT NULL,
    ts           INTEGER NOT NULL,
    level        TEXT    NOT NULL,
    message      TEXT    NOT NULL,
    detail       TEXT
);
CREATE INDEX IF NOT EXISTS idx_feed_run ON feed(run_id, id);

CREATE TABLE IF NOT EXISTS monte_carlo (
    run_id       INTEGER PRIMARY KEY,
    ts           INTEGER NOT NULL,
    iterations   INTEGER NOT NULL,
    summary      TEXT    NOT NULL,        -- JSON percentiles
    histogram    TEXT    NOT NULL         -- JSON drawdown distribution
);

CREATE TABLE IF NOT EXISTS stress (
    run_id       INTEGER NOT NULL,
    name         TEXT    NOT NULL,
    from_ts      INTEGER NOT NULL,
    to_ts        INTEGER NOT NULL,
    metrics      TEXT    NOT NULL,
    passed       INTEGER NOT NULL,
    PRIMARY KEY (run_id, name)
) WITHOUT ROWID;

-- Persistent library of validated combinations (gap-closure item 4), shared
-- with regime-scoped promotion (item 5): every combo a run's walk-forward
-- thresholds accepted, deduped by fingerprint (keep the best-performing entry
-- per fingerprint), so a future search can seed from what already worked
-- instead of starting from DEFAULT_GRID every time. `symbol` NULL means a
-- market-wide/global entry; a per-symbol entry additionally carries that
-- coin's own regime score at validation time alongside the market-wide one -
-- both continuous (item 5's regime_trend_er / efficiency-ratio measures), not
-- hard buckets.
CREATE TABLE IF NOT EXISTS library (
    fingerprint         TEXT    PRIMARY KEY,
    symbol              TEXT,                  -- NULL = global/market-wide
    params              TEXT    NOT NULL,       -- JSON: indicator set + parameters
    per_symbol          TEXT,                   -- JSON per-symbol overrides, if any
    performance         TEXT    NOT NULL,       -- JSON: walk-forward summary metrics
    market_regime_score REAL,                   -- continuous efficiency ratio, whole panel
    symbol_regime_score REAL,                   -- continuous efficiency ratio, this symbol only
    run_id              INTEGER,
    first_seen_at       INTEGER NOT NULL,
    last_seen_at        INTEGER NOT NULL,
    times_seen          INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_library_symbol ON library(symbol);
CREATE INDEX IF NOT EXISTS idx_library_last_seen ON library(last_seen_at);

-- Per-coin fuzzy regime models (fuzzy-regime section, step 1): one row per
-- symbol, replaced wholesale on rediscovery (the monthly job or the manual
-- trigger, step 5) rather than versioned - live classification (step 3)
-- always wants the latest model, and the discovery run that produced it is
-- already on record in `runs` if the history matters.
CREATE TABLE IF NOT EXISTS regime_models (
    symbol       TEXT PRIMARY KEY,
    model        TEXT NOT NULL,     -- JSON: CoinRegimeModel.as_dict()
    run_id       INTEGER,
    discovered_at INTEGER NOT NULL
);
"""


def connect(path: str | Path) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=30.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def now() -> int:
    return int(time.time())


def _json(value: Any) -> str:
    return json.dumps(value, default=float)


def _load(raw: Any, default: Any = None) -> Any:
    if raw in (None, ""):
        return default
    try:
        return json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return default


# --------------------------------------------------------------------------
# Library (gap-closure items 4 and 5)
# --------------------------------------------------------------------------
@dataclass
class LibraryEntry:
    """A combination worth remembering: it passed the walk-forward thresholds
    at least once. `regime` tags (item 5) are continuous efficiency-ratio
    readings, not bucket labels - see solopt/indicators.py's
    efficiency_ratio_stack, which this reuses rather than reinventing a
    separate regime measure."""

    fingerprint: str
    params: dict[str, Any]
    performance: dict[str, Any]
    symbol: str | None = None
    per_symbol: dict[str, Any] | None = None
    market_regime_score: float | None = None
    symbol_regime_score: float | None = None
    run_id: int | None = None


def _library_score(performance: dict[str, Any]) -> float:
    """Rank combinations by the same headline robustness number the
    dashboard already shows: walk-forward efficiency, tie-broken by mean
    window return - not total return, so a longer run doesn't automatically
    outrank a shorter one that used its trades better."""
    efficiency = float(performance.get("walk_forward_efficiency") or 0.0)
    mean_return = float(performance.get("mean_window_return") or 0.0)
    return efficiency * 1000.0 + mean_return


def _recency_weight(last_seen_at: int, *, half_life_days: float = 30.0, now_ts: int | None = None) -> float:
    """Exponential decay: an entry not re-validated in a while counts for
    less when seeding a new search, without being deleted outright - it may
    still be the best evidence available for a regime that hasn't recurred
    recently."""
    reference = now_ts if now_ts is not None else now()
    age_days = max(0.0, (reference - int(last_seen_at)) / 86400.0)
    return 0.5 ** (age_days / max(1e-6, half_life_days))


class RunStore:
    """Everything one optimizer run needs to persist, and to resume from."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.conn = connect(self.path)

    def close(self) -> None:
        self.conn.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            yield self.conn
        except Exception:
            self.conn.execute("ROLLBACK")
            raise
        else:
            self.conn.execute("COMMIT")

    # ------------------------------------------------------------------
    # runs
    # ------------------------------------------------------------------
    def start_run(
        self,
        *,
        config: dict[str, Any],
        settings: dict[str, Any],
        space: dict[str, Any],
        bundle: str,
        coverage: dict[str, Any],
        label: str = "",
    ) -> int:
        cur = self.conn.execute(
            "INSERT INTO runs(started_at, status, label, config, settings, space, "
            "bundle, coverage) VALUES (?,?,?,?,?,?,?,?)",
            (
                now(), "running", label, _json(config), _json(settings),
                _json(space), bundle, _json(coverage),
            ),
        )
        return int(cur.lastrowid)

    def finish_run(self, run_id: int, status: str, summary: dict[str, Any]) -> None:
        self.conn.execute(
            "UPDATE runs SET status = ?, finished_at = ?, summary = ? WHERE id = ?",
            (status, now(), _json(summary), run_id),
        )

    def get_run(self, run_id: int) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            return None
        out = dict(row)
        for key in ("config", "settings", "space", "coverage", "summary"):
            out[key] = _load(out.get(key), {})
        return out

    def latest_run(self, status: str | None = None) -> dict[str, Any] | None:
        sql = "SELECT id FROM runs"
        args: list[Any] = []
        if status:
            sql += " WHERE status = ?"
            args.append(status)
        sql += " ORDER BY id DESC LIMIT 1"
        row = self.conn.execute(sql, args).fetchone()
        return self.get_run(int(row["id"])) if row else None

    def resumable_run(self, fingerprint: str) -> dict[str, Any] | None:
        """A run against the same data that stopped before finishing.

        A completed run is deliberately *not* resumable: the spec says starting
        again after a full completion kicks off a fresh run against the latest
        pulled data, which is the only way new candles ever get tested.
        """
        row = self.conn.execute(
            "SELECT id FROM runs WHERE status = 'running' AND bundle = ? "
            "ORDER BY id DESC LIMIT 1",
            (fingerprint,),
        ).fetchone()
        return self.get_run(int(row["id"])) if row else None

    # ------------------------------------------------------------------
    # windows
    # ------------------------------------------------------------------
    def plan_windows(self, run_id: int, windows: Sequence[Any]) -> None:
        self.conn.executemany(
            "INSERT INTO windows(run_id, idx, is_from, is_to, oos_from, oos_to, "
            "status, updated_at) VALUES (?,?,?,?,?,?, 'pending', ?) "
            "ON CONFLICT(run_id, idx) DO NOTHING",
            [
                (run_id, w.index, w.is_from, w.is_to, w.oos_from, w.oos_to, now())
                for w in windows
            ],
        )

    def completed_windows(self, run_id: int) -> set[int]:
        rows = self.conn.execute(
            "SELECT idx FROM windows WHERE run_id = ? AND status IN ('done','skipped')",
            (run_id,),
        ).fetchall()
        return {int(r["idx"]) for r in rows}

    def record_window(self, run_id: int, result: Any) -> None:
        with self.transaction() as conn:
            conn.execute(
                "UPDATE windows SET status = ?, params = ?, is_metrics = ?, "
                "oos_metrics = ?, search = ?, counted = ?, profitable = ?, "
                "note = ?, updated_at = ? WHERE run_id = ? AND idx = ?",
                (
                    result.status,
                    _json(result.params),
                    _json(result.is_metrics),
                    _json(result.oos_metrics),
                    _json(result.search),
                    1 if result.counted else 0,
                    1 if result.profitable else 0,
                    result.note,
                    now(),
                    run_id,
                    result.index,
                ),
            )
            conn.execute(
                "DELETE FROM oos_trades WHERE run_id = ? AND window_idx = ?",
                (run_id, result.index),
            )
            if result.trades:
                conn.executemany(
                    "INSERT INTO oos_trades(run_id, window_idx, symbol, entry_ts, "
                    "exit_ts, entry_price, exit_price, size_usd, pnl_usd, fees_usd, "
                    "r_multiple, exit_reason, bars_held) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    [
                        (
                            run_id, result.index, t["symbol"], t["entry_ts"], t["exit_ts"],
                            t["entry_price"], t["exit_price"], t["size_usd"], t["pnl_usd"],
                            t["fees_usd"], t.get("r_multiple"), t.get("exit_reason"),
                            t.get("bars_held"),
                        )
                        for t in result.trades
                    ],
                )

    def windows(self, run_id: int) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM windows WHERE run_id = ? ORDER BY idx", (run_id,)
        ).fetchall()
        out = []
        for row in rows:
            d = dict(row)
            for key in ("params", "is_metrics", "oos_metrics", "search"):
                d[key] = _load(d.get(key), {})
            out.append(d)
        return out

    def oos_trades(self, run_id: int) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM oos_trades WHERE run_id = ? ORDER BY exit_ts, id", (run_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # feed
    # ------------------------------------------------------------------
    def append_feed(
        self, run_id: int, message: str, *, level: str = "info", detail: Any = None
    ) -> int:
        cur = self.conn.execute(
            "INSERT INTO feed(run_id, ts, level, message, detail) VALUES (?,?,?,?,?)",
            (run_id, now(), level, message, _json(detail) if detail is not None else None),
        )
        return int(cur.lastrowid)

    def feed(self, run_id: int, *, since_id: int = 0, limit: int = 200) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM feed WHERE run_id = ? AND id > ? ORDER BY id LIMIT ?",
            (run_id, since_id, limit),
        ).fetchall()
        out = []
        for row in rows:
            d = dict(row)
            d["detail"] = _load(d.get("detail"))
            out.append(d)
        return out

    # ------------------------------------------------------------------
    # derived results
    # ------------------------------------------------------------------
    def save_monte_carlo(
        self, run_id: int, iterations: int, summary: dict[str, Any], histogram: dict[str, Any]
    ) -> None:
        self.conn.execute(
            "INSERT INTO monte_carlo(run_id, ts, iterations, summary, histogram) "
            "VALUES (?,?,?,?,?) ON CONFLICT(run_id) DO UPDATE SET ts=excluded.ts, "
            "iterations=excluded.iterations, summary=excluded.summary, "
            "histogram=excluded.histogram",
            (run_id, now(), iterations, _json(summary), _json(histogram)),
        )

    def monte_carlo(self, run_id: int) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM monte_carlo WHERE run_id = ?", (run_id,)
        ).fetchone()
        if row is None:
            return None
        return {
            "run_id": run_id,
            "ts": int(row["ts"]),
            "iterations": int(row["iterations"]),
            "summary": _load(row["summary"], {}),
            "histogram": _load(row["histogram"], {}),
        }

    def save_stress(self, run_id: int, results: Sequence[Any]) -> None:
        self.conn.executemany(
            "INSERT INTO stress(run_id, name, from_ts, to_ts, metrics, passed) "
            "VALUES (?,?,?,?,?,?) ON CONFLICT(run_id, name) DO UPDATE SET "
            "metrics=excluded.metrics, passed=excluded.passed",
            [
                (run_id, r.name, r.from_ts, r.to_ts, _json(r.metrics), 1 if r.passed else 0)
                for r in results
            ],
        )

    # ------------------------------------------------------------------
    # retention (spec 6c/7) - WF/MC output is the fastest-growing storage
    # component and, unlike raw candles, does need a purge policy.
    # ------------------------------------------------------------------
    def purge_older_than(self, days: int) -> dict[str, int]:
        """Delete finished runs (and everything under them) past ``days`` old.

        A run still ``running`` is never touched regardless of age - only
        finished_at (or started_at, if it somehow never finished) ages a run
        out, so an interrupted-but-still-"running" row is not silently lost.
        """
        cutoff = now() - int(days) * 86400
        old_ids = [
            int(r["id"]) for r in self.conn.execute(
                "SELECT id FROM runs WHERE status != 'running' "
                "AND COALESCE(finished_at, started_at) < ?",
                (cutoff,),
            ).fetchall()
        ]
        if not old_ids:
            return {"runs": 0, "windows": 0, "oos_trades": 0, "feed": 0}

        deleted = {"runs": 0, "windows": 0, "oos_trades": 0, "feed": 0}
        placeholders = ",".join("?" * len(old_ids))
        with self.transaction() as conn:
            for table, key in (
                ("windows", "run_id"), ("oos_trades", "run_id"), ("feed", "run_id"),
                ("monte_carlo", "run_id"), ("stress", "run_id"),
            ):
                cur = conn.execute(f"DELETE FROM {table} WHERE {key} IN ({placeholders})", old_ids)
                deleted[table] = deleted.get(table, 0) + cur.rowcount
            cur = conn.execute(f"DELETE FROM runs WHERE id IN ({placeholders})", old_ids)
            deleted["runs"] = cur.rowcount
        return deleted

    def list_runs(
        self, *, limit: int = 50, cursor: int | None = None
    ) -> list[dict[str, Any]]:
        """Newest-first run summaries, for the storage browser (spec 6c)."""
        sql = "SELECT id, started_at, finished_at, status, label, bundle FROM runs"
        args: list[Any] = []
        if cursor:
            sql += " WHERE id < ?"
            args.append(cursor)
        sql += " ORDER BY id DESC LIMIT ?"
        args.append(int(limit))
        return [dict(r) for r in self.conn.execute(sql, args).fetchall()]

    def stress(self, run_id: int) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM stress WHERE run_id = ? ORDER BY from_ts", (run_id,)
        ).fetchall()
        out = []
        for row in rows:
            d = dict(row)
            d["metrics"] = _load(d["metrics"], {})
            d["passed"] = bool(d["passed"])
            out.append(d)
        return out

    # ------------------------------------------------------------------
    # library (gap-closure items 4 and 5)
    # ------------------------------------------------------------------
    def upsert_library(self, entry: LibraryEntry) -> bool:
        """Insert, or update only if this run's evidence outperforms what's
        already on record for the same fingerprint. Either way the entry's
        `last_seen_at`/`times_seen` advance - a repeat appearance is itself
        evidence, even when it did not beat the incumbent.

        Returns True if the stored parameters/performance actually changed.
        """
        with self.transaction() as conn:
            row = conn.execute(
                "SELECT performance FROM library WHERE fingerprint = ?", (entry.fingerprint,)
            ).fetchone()
            ts = now()
            if row is None:
                conn.execute(
                    "INSERT INTO library(fingerprint, symbol, params, per_symbol, "
                    "performance, market_regime_score, symbol_regime_score, run_id, "
                    "first_seen_at, last_seen_at, times_seen) VALUES (?,?,?,?,?,?,?,?,?,?,1)",
                    (
                        entry.fingerprint, entry.symbol, _json(entry.params),
                        _json(entry.per_symbol) if entry.per_symbol else None,
                        _json(entry.performance), entry.market_regime_score,
                        entry.symbol_regime_score, entry.run_id, ts, ts,
                    ),
                )
                return True

            incumbent = _library_score(_load(row["performance"], {}))
            challenger = _library_score(entry.performance)
            if challenger > incumbent:
                conn.execute(
                    "UPDATE library SET symbol = ?, params = ?, per_symbol = ?, "
                    "performance = ?, market_regime_score = ?, symbol_regime_score = ?, "
                    "run_id = ?, last_seen_at = ?, times_seen = times_seen + 1 "
                    "WHERE fingerprint = ?",
                    (
                        entry.symbol, _json(entry.params),
                        _json(entry.per_symbol) if entry.per_symbol else None,
                        _json(entry.performance), entry.market_regime_score,
                        entry.symbol_regime_score, entry.run_id, ts, entry.fingerprint,
                    ),
                )
                return True

            conn.execute(
                "UPDATE library SET last_seen_at = ?, times_seen = times_seen + 1 "
                "WHERE fingerprint = ?",
                (ts, entry.fingerprint),
            )
            return False

    def _library_row(self, row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        d["params"] = _load(d.get("params"), {})
        d["per_symbol"] = _load(d.get("per_symbol"))
        d["performance"] = _load(d.get("performance"), {})
        return d

    def library_entry(self, fingerprint: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM library WHERE fingerprint = ?", (fingerprint,)
        ).fetchone()
        return self._library_row(row) if row is not None else None

    def top_library_entries(
        self, n: int, *, symbol: str | None = None, half_life_days: float = 30.0
    ) -> list[dict[str, Any]]:
        """The best `n` entries by recency-weighted performance - a search's
        seed pool (item 4). `symbol=None` returns market-wide entries only;
        pass a symbol for that coin's own per-symbol entries (item 5)."""
        if symbol is None:
            rows = self.conn.execute("SELECT * FROM library WHERE symbol IS NULL").fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM library WHERE symbol = ?", (symbol,)
            ).fetchall()

        ts = now()
        scored = [
            (
                _library_score(_load(r["performance"], {}))
                * _recency_weight(int(r["last_seen_at"]), half_life_days=half_life_days, now_ts=ts),
                r,
            )
            for r in rows
        ]
        scored.sort(key=lambda t: t[0], reverse=True)
        out = []
        for score, row in scored[: max(0, int(n))]:
            d = self._library_row(row)
            d["score"] = score
            out.append(d)
        return out

    def nearest_regime_entries(
        self, symbol: str, target_regime_score: float, *, n: int = 1
    ) -> list[dict[str, Any]]:
        """This symbol's own library entries ordered by nearest regime match
        (item 5) - a similarity search in continuous regime space, not a
        threshold boundary. Falls back to nothing (an empty list) rather than
        guessing when this symbol has no library entries yet; the caller is
        expected to fall back to the existing global/per-coin set."""
        rows = self.conn.execute(
            "SELECT * FROM library WHERE symbol = ? AND symbol_regime_score IS NOT NULL",
            (symbol,),
        ).fetchall()
        scored = [
            (abs(float(r["symbol_regime_score"]) - float(target_regime_score)), r) for r in rows
        ]
        scored.sort(key=lambda t: t[0])
        return [self._library_row(row) for _distance, row in scored[: max(0, int(n))]]

    def prune_library(self, *, keep: int = 500) -> int:
        """Cap the library at `keep` entries, dropping the lowest-scored
        (recency-weighted) ones first - the same 'grow, then bound' pattern
        purge_older_than uses for run history, sized by count rather than age
        since a library entry has no natural expiry the way a run does."""
        total = int(self.conn.execute("SELECT COUNT(*) AS n FROM library").fetchone()["n"])
        overflow = total - max(0, int(keep))
        if overflow <= 0:
            return 0

        rows = self.conn.execute(
            "SELECT fingerprint, performance, last_seen_at FROM library"
        ).fetchall()
        ts = now()
        scored = sorted(
            rows,
            key=lambda r: _library_score(_load(r["performance"], {}))
            * _recency_weight(int(r["last_seen_at"]), now_ts=ts),
        )
        doomed = [r["fingerprint"] for r in scored[:overflow]]
        with self.transaction() as conn:
            placeholders = ",".join("?" * len(doomed))
            cur = conn.execute(
                f"DELETE FROM library WHERE fingerprint IN ({placeholders})", doomed
            )
            return cur.rowcount

    # ------------------------------------------------------------------
    # per-coin fuzzy regime models (fuzzy-regime section, steps 1 and 3)
    # ------------------------------------------------------------------
    def save_regime_model(
        self, symbol: str, model_dict: dict[str, Any], *, run_id: int | None = None
    ) -> None:
        """Replace this symbol's regime model wholesale - rediscovery
        supersedes whatever centroids were there before, it does not merge
        with them."""
        self.conn.execute(
            "INSERT INTO regime_models(symbol, model, run_id, discovered_at) "
            "VALUES (?,?,?,?) ON CONFLICT(symbol) DO UPDATE SET "
            "model=excluded.model, run_id=excluded.run_id, "
            "discovered_at=excluded.discovered_at",
            (symbol, _json(model_dict), run_id, now()),
        )

    def get_regime_model(self, symbol: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM regime_models WHERE symbol = ?", (symbol,)
        ).fetchone()
        if row is None:
            return None
        return {
            "symbol": row["symbol"],
            "model": _load(row["model"], {}),
            "run_id": row["run_id"],
            "discovered_at": int(row["discovered_at"]),
        }

    def list_regime_models(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT symbol, run_id, discovered_at FROM regime_models ORDER BY symbol"
        ).fetchall()
        return [dict(r) for r in rows]
