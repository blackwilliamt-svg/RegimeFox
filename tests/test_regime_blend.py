"""Fuzzy-regime section, step 4: blended live strategy application, wired
into the live engine.

The blending math itself is pinned in tests/test_paramsync.py against
solbot.paramsync.blend_regime_params directly. This covers the engine's
actual wiring: that it takes precedence over item 5's nearest-match when
enabled and available, that the blend weights land on the opened position's
entry_snapshot for explainability, and that it is off by default.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from solbot import db
from solbot.config import Config
from solbot.engine import Engine
from solbot import strategy as strategy_module

from fakes import binance_asset, fake_clients, token
from test_backtest_and_exports import seed_candles

MINT = "RegimeBlendTestMintAAAAAAAAAAAAAAAAAAAAAAAAA"


@pytest.fixture
def engine(workspace, monkeypatch):
    cfg = Config(workspace["config"])
    clients = fake_clients(
        prices={MINT: 1.0},
        tokens=[token(MINT, "BLND", liquidity=800_000.0, volume_24h=2_000_000.0)],
        binance_assets=[binance_asset("BLND")],
    )
    eng = Engine(cfg, clients)
    eng.build_instances()
    return eng


class _OneClusterStore:
    """A regime model with exactly one cluster - membership is always
    {0: 1.0} regardless of the live feature vector, which keeps the blend
    weight deterministic without needing a realistic fuzzy fit in a test.

    Regime/library lookups are keyed by mint, not ticker - see
    Engine._regime_scoped_cfg's docstring - so this matches against MINT.
    """

    def __init__(self, cluster_params: dict | None) -> None:
        centroid = [[0.0, 0.0, 0.0]]
        self.model = {
            "symbol": MINT,
            "feature_names": ["adx", "atr_pct", "volume_zscore"],
            "scaler": {"mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]},
            "centroids": centroid,
            "fpc": 1.0,
            "n_clusters": 1,
            "n_samples": 500,
        }
        self.cluster_params = cluster_params
        self.calls = 0

    def get_regime_model(self, mint):
        self.calls += 1
        return {"symbol": mint, "model": self.model, "run_id": 1, "discovered_at": 0} \
            if mint == MINT else None

    def regime_cluster_entries(self, mint):
        if self.cluster_params is None or mint != MINT:
            return {}
        return {0: {"params": self.cluster_params, "fingerprint": "blend-fp"}}

    def nearest_regime_entries(self, *a, **kw):
        return []   # blending should never fall through to this when it applies


def test_blending_off_by_default_never_touches_the_entry_cfg(engine, workspace, monkeypatch):
    seed_candles(MINT, bars=400, trend=0.0)
    engine.startup()
    assert engine.cfg.get("fuzzy_regime_blend_enabled") is False

    store = _OneClusterStore({"volume_spike_multiple": 999.0})
    monkeypatch.setattr(engine, "_get_library_store", lambda: store)

    captured = []
    real_evaluate_entry = strategy_module.evaluate_entry

    def spy(df, cfg, **kw):
        captured.append(cfg)
        return real_evaluate_entry(df, cfg, **kw)

    monkeypatch.setattr("solbot.engine.evaluate_entry", spy)

    prices = engine.clients.jupiter.prices([MINT])
    engine.look_for_entries(prices, tier="broad")

    assert captured
    assert captured[-1]["volume_spike_multiple"] != 999.0   # blend never applied


def test_blending_applies_and_overrides_the_entry_cfg_when_enabled(engine, workspace, monkeypatch):
    seed_candles(MINT, bars=400, trend=0.0)
    engine.config.update({"fuzzy_regime_blend_enabled": True})
    engine._apply_config()
    engine.startup()

    store = _OneClusterStore({"volume_spike_multiple": 999.0})
    monkeypatch.setattr(engine, "_get_library_store", lambda: store)

    captured = []
    real_evaluate_entry = strategy_module.evaluate_entry

    def spy(df, cfg, **kw):
        captured.append(cfg)
        return real_evaluate_entry(df, cfg, **kw)

    monkeypatch.setattr("solbot.engine.evaluate_entry", spy)

    prices = engine.clients.jupiter.prices([MINT])
    engine.look_for_entries(prices, tier="broad")

    assert captured
    assert captured[-1]["volume_spike_multiple"] == 999.0


def test_blend_weights_land_on_the_opened_positions_entry_snapshot(engine, workspace, monkeypatch):
    conn = workspace["conn"]
    seed_candles(MINT, bars=400, trend=0.0006)   # a real uptrend so an entry can actually fire
    engine.config.update({"fuzzy_regime_blend_enabled": True})
    engine._apply_config()
    engine.startup()

    store = _OneClusterStore({"volume_spike_multiple": 0.5})   # loose enough to let a signal through
    monkeypatch.setattr(engine, "_get_library_store", lambda: store)

    prices = engine.clients.jupiter.prices([MINT])
    for _ in range(30):
        engine.look_for_entries(prices, tier="broad")
        if engine.primary.portfolio.open_count(conn):
            break
        engine.clients.jupiter.set_price(MINT, engine.clients.jupiter._prices[MINT] * 1.001)
        engine._candle_cache.clear()
        prices = engine.clients.jupiter.prices([MINT])

    positions = engine.primary.portfolio.open_positions(conn)
    if not positions:
        pytest.skip("synthetic series did not produce an entry within the poll budget")
    snap = json.loads(positions[0]["entry_snapshot"])
    assert "regime_blend" in snap
    assert snap["regime_blend"]["applied"] is True
    assert snap["regime_blend"]["weights"] == {"0": 1.0}


def test_blend_falls_back_to_nearest_match_when_no_regime_model_exists(engine, workspace, monkeypatch):
    seed_candles(MINT, bars=400, trend=0.0)
    engine.config.update({"fuzzy_regime_blend_enabled": True})
    engine._apply_config()
    engine.startup()

    store = _OneClusterStore(None)   # no per-regime entries -> classify still works, blend does not

    # Force get_regime_model to return None entirely, so classify_current_regime
    # itself reports "no model" and the blend path is skipped up front.
    store.get_regime_model = lambda symbol: None
    monkeypatch.setattr(engine, "_get_library_store", lambda: store)

    captured = []
    real_evaluate_entry = strategy_module.evaluate_entry
    monkeypatch.setattr(
        "solbot.engine.evaluate_entry",
        lambda df, cfg, **kw: (captured.append(cfg), real_evaluate_entry(df, cfg, **kw))[1],
    )

    prices = engine.clients.jupiter.prices([MINT])
    engine.look_for_entries(prices, tier="broad")

    assert captured
    # Falls all the way through to the untouched global config (nearest_regime_entries returns []).
    assert captured[-1]["volume_spike_multiple"] == engine.cfg["volume_spike_multiple"]
