"""Gap-closure item 5: regime-scoped promoted sets in the live engine.

The selection logic itself (nearest match, fallback, disabled) is pinned in
tests/test_paramsync.py against solbot.paramsync.select_regime_scoped_params
directly. This covers the live engine's actual wiring: that a new entry
decision genuinely uses the regime-scoped params when one qualifies, that
switching logs a plain-language event exactly once per change (not every
cycle), and that it never reaches the exit/position-management path.
"""
from __future__ import annotations

import json

import pytest

from solbot import db
from solbot.config import Config
from solbot.engine import Engine
from solbot import strategy as strategy_module

from fakes import binance_asset, fake_clients, token
from test_backtest_and_exports import seed_candles

MINT = "RegimeScopedTestMintAAAAAAAAAAAAAAAAAAAAAAAA"


@pytest.fixture
def engine(workspace, monkeypatch):
    cfg = Config(workspace["config"])
    clients = fake_clients(
        prices={MINT: 1.0},
        tokens=[token(MINT, "REGM", liquidity=800_000.0, volume_24h=2_000_000.0)],
        binance_assets=[binance_asset("REGM")],
    )
    eng = Engine(cfg, clients)
    eng.build_instances()
    return eng


class _FixedStore:
    """A library store whose nearest match is fixed by the test - close
    enough to always qualify, so the sole variable is whether an entry is
    even present for this symbol.

    Regime/library lookups are keyed by mint, not ticker - see
    Engine._regime_scoped_cfg's docstring - so `_entry_mint` here must be the
    same mint the test's positions are opened against, not a ticker.
    """

    def __init__(self, entry: dict | None) -> None:
        self.entry = entry
        self.calls = 0

    def nearest_regime_entries(self, mint, target, *, n=1):
        self.calls += 1
        if self.entry is None or mint != self.entry.get("_entry_mint"):
            return []
        return [
            {
                "fingerprint": self.entry["fingerprint"],
                "symbol_regime_score": target,   # always exactly on target -> distance 0
                "params": self.entry["params"],
            }
        ]


def test_a_qualifying_regime_scoped_set_is_used_for_the_entry_decision(engine, workspace, monkeypatch):
    seed_candles(MINT, bars=400, trend=0.0)
    engine.startup()

    store = _FixedStore(
        {"_entry_mint": MINT, "fingerprint": "fp-live-1", "params": {"volume_spike_multiple": 999.0}}
    )
    monkeypatch.setattr(engine, "_get_library_store", lambda: store)

    captured: list[dict] = []
    real_evaluate_entry = strategy_module.evaluate_entry

    def spy(df, cfg, **kw):
        captured.append(cfg)
        return real_evaluate_entry(df, cfg, **kw)

    monkeypatch.setattr("solbot.engine.evaluate_entry", spy)

    prices = engine.clients.jupiter.prices([MINT])
    engine.look_for_entries(prices, tier="broad")

    assert captured, "evaluate_entry was never called"
    assert captured[-1]["volume_spike_multiple"] == 999.0
    assert store.calls >= 1


def test_switching_logs_an_event_only_on_change_not_every_cycle(engine, workspace, monkeypatch):
    seed_candles(MINT, bars=400, trend=0.0)
    engine.startup()
    conn = workspace["conn"]

    store = _FixedStore(
        {"_entry_mint": MINT, "fingerprint": "fp-live-2", "params": {"volume_spike_multiple": 999.0}}
    )
    monkeypatch.setattr(engine, "_get_library_store", lambda: store)
    monkeypatch.setattr(
        "solbot.engine.evaluate_entry",
        lambda df, cfg, **kw: strategy_module.evaluate_entry(df, cfg, **kw),
    )

    prices = engine.clients.jupiter.prices([MINT])
    engine.look_for_entries(prices, tier="broad")
    engine.look_for_entries(prices, tier="broad")   # same selection again

    switched = [
        e for e in db.recent_events(limit=50, conn=conn)
        if "switched to a regime-scoped parameter set" in e["message"]
    ]
    assert len(switched) == 1   # not one per cycle


def test_reverting_to_the_global_set_also_logs_once(engine, workspace, monkeypatch):
    seed_candles(MINT, bars=400, trend=0.0)
    engine.startup()
    conn = workspace["conn"]

    store = _FixedStore(
        {"_entry_mint": MINT, "fingerprint": "fp-live-3", "params": {"volume_spike_multiple": 999.0}}
    )
    monkeypatch.setattr(engine, "_get_library_store", lambda: store)

    prices = engine.clients.jupiter.prices([MINT])
    engine.look_for_entries(prices, tier="broad")

    store.entry = None   # the library no longer has an entry for this symbol
    engine.look_for_entries(prices, tier="broad")

    reverted = [
        e for e in db.recent_events(limit=50, conn=conn)
        if "reverted to the global parameter set" in e["message"]
    ]
    assert len(reverted) == 1


def test_regime_scoped_selection_is_never_consulted_when_managing_open_positions():
    """Structural guard: _regime_scoped_cfg must only ever be reached from
    the entry path, never from position management - an open position's
    stop/target were fixed at its own entry and must stay that way regardless
    of how the regime-scoped choice moves afterward."""
    import inspect

    from solbot.engine import Engine

    manage_source = inspect.getsource(Engine._manage_instance)
    assert "_regime_scoped_cfg" not in manage_source
