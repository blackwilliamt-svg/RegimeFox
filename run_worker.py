#!/usr/bin/env python3
"""Worker entry point - the trading loop.

Run under systemd as ``solbot-worker``. This is the only process that trades;
the dashboard talks to it through the SQLite command queue.

    python run_worker.py
"""
from __future__ import annotations

import logging
import os
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from solbot import db
from solbot.backtest import run_and_store
from solbot.config import get_config
from solbot.engine import Engine
from solbot.lock import AlreadyRunning, ProcessLock

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def setup_logging() -> None:
    level = os.getenv("SOLBOT_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=level, format=LOG_FORMAT, stream=sys.stdout)
    # httpx logs every request at INFO; far too chatty for a 1s poll loop.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def main() -> int:
    setup_logging()
    log = logging.getLogger("solbot.worker")

    config = get_config()
    db.init_db()

    lock_path = Path(os.getenv("SOLBOT_LOCK", "data/worker.lock"))
    try:
        lock = ProcessLock(lock_path)
        lock.acquire()
    except AlreadyRunning as exc:
        log.error("%s", exc)
        db.log_event(
            f"Worker refused to start: {exc}", level="alert", category="system"
        )
        return 1

    engine = Engine(config)

    # Wire the backtest in so the daily schedule and the dashboard button work.
    def run_backtest(days: int) -> None:
        try:
            run_and_store(engine.store, engine.cfg, days=days)
        except Exception:
            log.exception("backtest failed")

    engine._backtest_hook = run_backtest

    def handle_signal(signum: int, _frame: object) -> None:
        log.info("received signal %s, shutting down", signum)
        engine.stop()

    signal.signal(signal.SIGINT, handle_signal)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, handle_signal)

    try:
        engine.run()
    except Exception:
        log.exception("worker crashed")
        db.log_event(
            "Worker crashed - systemd will restart it. Kill-switch and circuit-breaker "
            "state persist across restarts, so a crash cannot clear a halt.",
            level="alert",
            category="system",
        )
        return 1
    finally:
        lock.release()
        engine.clients.close()
        db.close_all()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
