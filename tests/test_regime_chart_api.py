"""Fuzzy-regime section, step 6: regime history on the dashboard's coin chart.

Reuses the existing /api/candles/<mint> endpoint (gap-closure item 6's own
principle: extend, don't add a parallel one) with an opt-in ?regime=1 param,
rather than a new route.
"""
from __future__ import annotations

import numpy as np
import pyotp
import pytest

from solbot import db
from solbot.candlestore import ParquetCandleStore


@pytest.fixture
def app(workspace, monkeypatch):
    monkeypatch.setenv("FLASK_SECRET_KEY", "x" * 64)
    monkeypatch.setenv("SOLBOT_INSECURE_COOKIES", "1")
    from solbot.config import reset_config_for_tests
    from solbot.web import create_app

    reset_config_for_tests(workspace["config"])
    application = create_app()
    application.config["TESTING"] = True
    return application


@pytest.fixture
def client(app):
    from solbot.web.auth import confirm_totp, create_user

    secret, _codes = create_user("trader", "correct-horse-battery")
    confirm_totp("trader")
    c = app.test_client()
    resp = c.post("/login", data={
        "username": "trader",
        "password": "correct-horse-battery",
        "token": pyotp.TOTP(secret).now(),
    })
    assert resp.status_code == 302
    return c


MINT = "RegimeChartMintAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"


def _seed_candles(mint: str, bars: int = 400) -> None:
    rng = np.random.default_rng(3)
    price = 1.0
    rows = []
    ts = 1_700_000_000
    for i in range(bars):
        price = max(1e-6, price * (1 + rng.normal(0, 0.004)))
        volume = max(50.0, rng.lognormal(np.log(1000.0), 0.5))
        rows.append((ts + i * 600, price * 0.999, price * 1.004, price * 0.996, price, volume))
    ParquetCandleStore().append(mint, "1m", rows)


def _seed_regime_model(mint: str, tmp_path, monkeypatch) -> None:
    import solbot.wfmc as wfmc
    from solopt.fuzzy import Standardizer, discover_best_k
    from solopt.regime_discovery import CoinRegimeModel
    from solopt.store import RunStore

    monkeypatch.setattr(wfmc, "DAILY_STORE_PATH", str(tmp_path / "wfmc.db"))
    rng = np.random.default_rng(1)
    X = np.vstack(
        [
            rng.normal([10.0, 0.01, -1.0], 1.0, size=(50, 3)),
            rng.normal([40.0, 0.05, 1.5], 1.0, size=(50, 3)),
        ]
    )
    scaler = Standardizer.fit(X)
    result = discover_best_k(scaler.transform(X), k_range=(2,), seed=1)
    model = CoinRegimeModel.from_result(mint, scaler, result, n_samples=X.shape[0])

    store = RunStore(tmp_path / "wfmc.db")
    store.save_regime_model(mint, model.as_dict())


def test_regime_is_absent_by_default(client, workspace, tmp_path, monkeypatch):
    _seed_candles(MINT)
    _seed_regime_model(MINT, tmp_path, monkeypatch)

    resp = client.get(f"/api/candles/{MINT}?limit=100")
    assert resp.status_code == 200
    assert "regime" not in resp.get_json()


def test_regime_overlay_included_when_requested(client, workspace, tmp_path, monkeypatch):
    _seed_candles(MINT)
    _seed_regime_model(MINT, tmp_path, monkeypatch)

    resp = client.get(f"/api/candles/{MINT}?limit=100&regime=1")
    assert resp.status_code == 200
    body = resp.get_json()
    assert "regime" in body
    assert body["regime"]["n_clusters"] == 2
    assert body["regime"]["history"]
    assert body["regime"]["current"] is not None
    assert sum(body["regime"]["current"].values()) == pytest.approx(1.0, abs=1e-3)


def test_regime_overlay_is_none_without_a_stored_model(client, workspace, tmp_path, monkeypatch):
    import solbot.wfmc as wfmc

    monkeypatch.setattr(wfmc, "DAILY_STORE_PATH", str(tmp_path / "wfmc.db"))
    _seed_candles(MINT)

    resp = client.get(f"/api/candles/{MINT}?limit=100&regime=1")
    assert resp.status_code == 200
    assert resp.get_json()["regime"] is None


def test_regime_overlay_never_500s_when_the_local_store_is_unreachable(client, workspace, monkeypatch):
    import solbot.wfmc as wfmc

    # A path that cannot possibly resolve to an openable sqlite file.
    monkeypatch.setattr(wfmc, "DAILY_STORE_PATH", "Z:\\nonexistent\\wfmc.db")
    _seed_candles(MINT)

    resp = client.get(f"/api/candles/{MINT}?limit=100&regime=1")
    assert resp.status_code == 200
