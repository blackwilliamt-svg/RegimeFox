"""Backfill timing/batch-trigger revision: the monthly RunPod retest fires
once the *whole* historical backfill has landed, never per-coin - see
Engine._maybe_trigger_runpod_after_backfill's docstring for why "once for
the freshly-backfilled universe" is the property that actually matters here
(run_monthly already batches every routed coin into one set of workers;
what this guards is the trigger itself never firing more than once per
backfill, regardless of how many tokens were in it).
"""
from __future__ import annotations

import time

import pytest

from solbot.config import Config
from solbot.datastore import PullReport
from solbot.engine import Engine

from fakes import binance_asset, fake_clients, token

MINTS = [f"Mint{i:03d}AAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"[:44] for i in range(5)]


@pytest.fixture
def engine(workspace):
    cfg = Config(workspace["config"])
    clients = fake_clients(
        prices={m: 1.0 for m in MINTS},
        tokens=[token(m, f"SYM{i}", liquidity=800_000.0, volume_24h=2_000_000.0)
                for i, m in enumerate(MINTS)],
        binance_assets=[binance_asset(f"SYM{i}") for i in range(len(MINTS))],
    )
    eng = Engine(cfg, clients)
    eng.build_instances()
    return eng


def _clean_report(tokens: int) -> PullReport:
    return PullReport(tokens_requested=tokens, tokens_done=tokens, candles_written=1000)


def test_does_nothing_when_monthly_runpod_is_not_enabled(engine, monkeypatch):
    engine.cfg["wfmc_monthly_enabled"] = False
    calls = []
    monkeypatch.setattr("solbot.wfmc.run_monthly", lambda *a, **kw: calls.append(1))

    engine._maybe_trigger_runpod_after_backfill(_clean_report(len(MINTS)), len(MINTS))
    time.sleep(0.05)  # the trigger itself is synchronous; only run_monthly is threaded

    assert calls == []


def test_fires_exactly_once_for_the_whole_backfill_not_per_coin(engine, monkeypatch):
    engine.cfg["wfmc_monthly_enabled"] = True
    calls = []
    monkeypatch.setattr(
        "solbot.wfmc.run_monthly", lambda *a, **kw: calls.append(1)
    )

    engine._maybe_trigger_runpod_after_backfill(_clean_report(len(MINTS)), len(MINTS))
    time.sleep(0.2)  # run_monthly fires on its own daemon thread

    assert calls == [1]   # once, regardless of how many coins were backfilled


def test_skips_a_backfill_that_stopped_early(engine, monkeypatch):
    engine.cfg["wfmc_monthly_enabled"] = True
    calls = []
    monkeypatch.setattr("solbot.wfmc.run_monthly", lambda *a, **kw: calls.append(1))

    report = PullReport(tokens_requested=5, tokens_done=2, stopped_early="cancelled")
    engine._maybe_trigger_runpod_after_backfill(report, 5)
    time.sleep(0.05)

    assert calls == []


def test_skips_a_backfill_that_wrote_nothing(engine, monkeypatch):
    engine.cfg["wfmc_monthly_enabled"] = True
    calls = []
    monkeypatch.setattr("solbot.wfmc.run_monthly", lambda *a, **kw: calls.append(1))

    report = PullReport(tokens_requested=5, tokens_done=0)
    engine._maybe_trigger_runpod_after_backfill(report, 5)
    time.sleep(0.05)

    assert calls == []


def test_a_failing_runpod_trigger_does_not_raise_into_the_backfill_worker(engine, monkeypatch):
    engine.cfg["wfmc_monthly_enabled"] = True

    def boom(*a, **kw):
        raise RuntimeError("RunPod is unreachable")

    monkeypatch.setattr("solbot.wfmc.run_monthly", boom)

    # Must not raise - it runs on its own daemon thread and only ever logs.
    engine._maybe_trigger_runpod_after_backfill(_clean_report(len(MINTS)), len(MINTS))
    time.sleep(0.1)
