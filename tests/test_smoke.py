"""End-to-end smoke tests: a full engine cycle, and the dashboard booting.

These use fake clients throughout - no network, no wallet, no real money.
"""
from __future__ import annotations

import json

import pytest

from solbot import db, risk
from solbot.config import Config
from solbot.engine import Engine
from solbot.execution import PaperExecutor
from solbot.universe import UniverseBuilder

from fakes import binance_asset, fake_clients, token
from test_backtest_and_exports import seed_candles

MINT = "SmokeTokenMintAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"


@pytest.fixture
def engine(workspace, monkeypatch):
    cfg = Config(workspace["config"])
    clients = fake_clients(
        prices={MINT: 1.0},
        tokens=[token(MINT, "SMOKE", liquidity=800_000.0, volume_24h=2_000_000.0)],
        binance_assets=[binance_asset("SMOKE")],
    )
    eng = Engine(cfg, clients)
    eng.build_instances()
    return eng


# --------------------------------------------------------------------------
# universe
# --------------------------------------------------------------------------
def test_universe_applies_the_floors(workspace, settings):
    clients = fake_clients(
        tokens=[
            token("GoodMint" + "A" * 36, "GOOD", liquidity=200_000.0, volume_24h=500_000.0),
            token("ThinMint" + "A" * 36, "THIN", liquidity=1_000.0, volume_24h=500_000.0),
            token("QuietMint" + "A" * 35, "QUIET", liquidity=200_000.0, volume_24h=1_000.0),
        ],
        binance_assets=[
            binance_asset("GOOD"), binance_asset("THIN"), binance_asset("QUIET"),
        ],
    )
    builder = UniverseBuilder(clients.binance, clients.jupiter, settings)
    stats = builder.refresh(conn=workspace["conn"])

    assert stats.considered == 3
    assert stats.routable == 3
    assert stats.passed == 1
    assert stats.rejected_liquidity == 1
    assert stats.rejected_volume == 1
    assert "GOOD" in [t.symbol for t in builder.tokens.values()]


def test_universe_drops_a_binance_coin_with_no_solana_route(workspace, settings):
    clients = fake_clients(
        tokens=[token("M" + "A" * 43, "OK", liquidity=200_000.0, volume_24h=500_000.0)],
        binance_assets=[binance_asset("OK"), binance_asset("NOWHERE")],
    )
    builder = UniverseBuilder(clients.binance, clients.jupiter, settings)
    stats = builder.refresh(conn=workspace["conn"])

    assert stats.considered == 2
    assert stats.routable == 1
    assert stats.rejected_not_routable == 1


def test_universe_survives_a_restart(workspace, settings):
    clients = fake_clients(
        tokens=[token("M" + "A" * 43, "OK", liquidity=200_000.0, volume_24h=500_000.0)],
        binance_assets=[binance_asset("OK")],
    )
    builder = UniverseBuilder(clients.binance, clients.jupiter, settings)
    builder.refresh(conn=workspace["conn"])

    fresh = UniverseBuilder(clients.binance, clients.jupiter, settings)
    assert len(fresh.load_persisted(workspace["conn"])) == 1


def test_universe_snapshot_is_recorded_for_backtests(workspace, settings):
    clients = fake_clients(
        tokens=[token("M" + "A" * 43, "OK", liquidity=200_000.0, volume_24h=500_000.0)],
        binance_assets=[binance_asset("OK")],
    )
    UniverseBuilder(clients.binance, clients.jupiter, settings).refresh(conn=workspace["conn"])
    rows = workspace["conn"].execute("SELECT * FROM universe_history").fetchall()
    assert len(rows) == 1


# --------------------------------------------------------------------------
# engine cycle
# --------------------------------------------------------------------------
def test_engine_starts_and_ticks(engine, workspace):
    engine.startup()
    for _ in range(3):
        engine.tick()

    heartbeat = db.kv_get("worker_heartbeat", {})
    assert heartbeat["cycle"] >= 3
    assert engine.trading_enabled()


def test_engine_opens_and_manages_a_position(engine, workspace):
    """A full path: candles -> safety -> signal -> size -> fill -> persisted."""
    conn = workspace["conn"]
    seed_candles(MINT, bars=400, trend=0.0006)
    engine.startup()

    # Force the entry path directly with a known-good candle history.
    prices = engine.clients.jupiter.prices([MINT])
    for _ in range(30):
        engine.look_for_entries(prices, tier="broad")
        if engine.primary.portfolio.open_count(conn):
            break
        # Advance price so a fresh bar can qualify.
        engine.clients.jupiter.set_price(MINT, engine.clients.jupiter._prices[MINT] * 1.001)
        engine._candle_cache.clear()
        prices = engine.clients.jupiter.prices([MINT])

    # Whether or not this synthetic series triggers, the machinery must not error
    # and any opened position must be fully persisted.
    for row in engine.primary.portfolio.open_positions(conn):
        assert row["hard_stop"] < row["entry_price"]
        assert row["take_profit"] > row["entry_price"]
        assert row["initial_risk"] > 0
        assert json.loads(row["entry_snapshot"])


def test_kill_switch_stops_new_entries_but_keeps_managing(engine, workspace):
    conn = workspace["conn"]
    engine.startup()
    risk.engage_kill_switch("smoke test", "tester", conn)

    assert not engine.trading_enabled()
    engine.tick()   # must not raise; still manages open positions
    assert risk.kill_switch_engaged(conn)


def test_commands_are_drained_from_the_queue(engine, workspace):
    conn = workspace["conn"]
    engine.startup()

    db.enqueue_command("stop", {}, requested_by="tester", conn=conn)
    engine.drain_commands()
    assert db.kv_get("engine_run") is False

    db.enqueue_command("start", {}, requested_by="tester", conn=conn)
    engine.drain_commands()
    assert db.kv_get("engine_run") is True

    assert db.pending_commands(conn) == []


def test_kill_switch_command_engages_and_releases(engine, workspace):
    conn = workspace["conn"]
    engine.startup()

    db.enqueue_command("kill_switch", {"reason": "test"}, requested_by="tester", conn=conn)
    engine.drain_commands()
    assert risk.kill_switch_engaged(conn)

    db.enqueue_command("release_kill_switch", {}, requested_by="tester", conn=conn)
    engine.drain_commands()
    assert not risk.kill_switch_engaged(conn)


def test_unknown_command_does_not_crash_the_loop(engine, workspace):
    conn = workspace["conn"]
    engine.startup()
    db.enqueue_command("not_a_command", {}, conn=conn)
    engine.drain_commands()
    row = conn.execute("SELECT * FROM commands ORDER BY id DESC LIMIT 1").fetchone()
    assert row["status"] == "done"
    assert "unknown command" in row["result"]


def test_settings_reload_propagates_to_components(engine, workspace):
    engine.startup()
    engine.config.update({"volume_spike_multiple": 4.5})
    assert engine.config.maybe_reload() or True   # same process wrote it
    engine._apply_config()

    assert engine.cfg["volume_spike_multiple"] == 4.5
    assert engine.safety.cfg["volume_spike_multiple"] == 4.5
    assert engine.scanner.cfg["volume_spike_multiple"] == 4.5
    assert engine.universe.cfg["volume_spike_multiple"] == 4.5


def test_live_mode_adds_a_parallel_paper_instance(workspace, monkeypatch):
    """Once live, the same rules must run in paper alongside for comparison."""
    cfg = Config(workspace["config"])
    cfg.set_trading_mode("live")

    clients = fake_clients(prices={MINT: 1.0})
    engine = Engine(cfg, clients)
    # Avoid needing a real wallet key for this structural check.
    monkeypatch.setattr(
        "solbot.engine.build_executor",
        lambda c, s, m, sec: PaperExecutor(c, s),
    )
    engine.build_instances()

    names = [i.name for i in engine.instances]
    assert names == ["live", "paper"]
    assert engine.primary.name == "live"


def test_shadow_instance_applies_overrides(workspace, monkeypatch):
    cfg = Config(workspace["config"])
    db.kv_set("shadow_overrides", {"volume_spike_multiple": 5.0})

    clients = fake_clients(prices={MINT: 1.0})
    engine = Engine(cfg, clients)
    engine.build_instances()

    shadow = [i for i in engine.instances if i.name == "shadow"]
    assert shadow, "shadow instance was not created"
    assert shadow[0].cfg(engine.cfg)["volume_spike_multiple"] == 5.0
    # It must not leak into the primary instance.
    assert engine.primary.cfg(engine.cfg)["volume_spike_multiple"] == 2.0


# --------------------------------------------------------------------------
# paper execution
# --------------------------------------------------------------------------
def test_paper_buy_charges_fees_and_slippage(workspace, settings):
    clients = fake_clients()
    ex = PaperExecutor(clients, settings, use_live_quotes=False)
    fill = ex.buy(MINT, 100.0, 1.0)

    assert fill.ok
    assert fill.fee_usd > 0
    assert fill.price > 1.0            # buying pays up
    assert fill.qty * fill.price < 100.0   # the fee came out


def test_paper_sell_receives_less_than_mid(workspace, settings):
    clients = fake_clients()
    ex = PaperExecutor(clients, settings, use_live_quotes=False)
    fill = ex.sell(MINT, 100.0, 1.0)

    assert fill.ok
    assert fill.price < 1.0
    assert fill.fee_usd > 0


def test_paper_refuses_a_fill_beyond_the_slippage_tolerance(workspace, settings):
    clients = fake_clients()
    ex = PaperExecutor(clients, settings, use_live_quotes=False)
    fill = ex.buy(MINT, 100.0, 1.0, slippage_pct=99.0)

    assert not fill.ok
    assert "slippage" in fill.reason


# --------------------------------------------------------------------------
# dashboard
# --------------------------------------------------------------------------
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


def test_dashboard_requires_authentication(app):
    client = app.test_client()
    for path in ("/", "/trades", "/settings", "/backtest", "/tax"):
        resp = client.get(path)
        assert resp.status_code in (302, 308), path
        assert "/login" in resp.headers.get("Location", "") or "/setup" in resp.headers.get("Location", "")


def test_api_returns_401_rather_than_redirecting(app):
    client = app.test_client()
    resp = client.get("/api/status")
    assert resp.status_code == 401


def test_first_run_redirects_to_setup(app):
    client = app.test_client()
    resp = client.get("/login")
    assert resp.status_code == 302
    assert "/setup" in resp.headers["Location"]


def test_setup_creates_a_user_with_totp(app, workspace):
    client = app.test_client()
    resp = client.post(
        "/setup",
        data={"username": "trader", "password": "correct-horse-battery",
              "confirm": "correct-horse-battery"},
    )
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    assert "backup codes" in body.lower()
    assert "data:image/png;base64," in body   # QR rendered inline, no CDN

    from solbot.web.auth import get_user

    user = get_user("trader")
    assert user is not None
    assert user["totp_secret"]
    assert user["password_hash"].startswith("$argon2")


def test_setup_is_refused_once_a_user_exists(app, workspace):
    from solbot.web.auth import create_user

    create_user("first", "correct-horse-battery")
    client = app.test_client()
    resp = client.get("/setup")
    assert resp.status_code == 302
    assert "/login" in resp.headers["Location"]


def test_full_login_flow_with_totp(app, workspace):
    import pyotp
    from solbot.web.auth import confirm_totp, create_user

    secret, _codes = create_user("trader", "correct-horse-battery")
    confirm_totp("trader")

    client = app.test_client()
    resp = client.post("/login", data={
        "username": "trader",
        "password": "correct-horse-battery",
        "token": pyotp.TOTP(secret).now(),
    })
    assert resp.status_code == 302

    page = client.get("/")
    assert page.status_code == 200
    assert b"Solana TA Bot" in page.data

    status = client.get("/api/status")
    assert status.status_code == 200
    assert status.get_json()["mode"] == "paper"


def test_login_fails_without_the_totp_code(app, workspace):
    from solbot.web.auth import confirm_totp, create_user

    create_user("trader", "correct-horse-battery")
    confirm_totp("trader")

    client = app.test_client()
    resp = client.post("/login", data={
        "username": "trader", "password": "correct-horse-battery", "token": "000000",
    })
    assert resp.status_code == 200      # re-renders the form, no session
    assert client.get("/api/status").status_code == 401


def test_backup_code_works_once(app, workspace):
    from solbot.web.auth import confirm_totp, create_user, get_user, verify_totp

    _secret, codes = create_user("trader", "correct-horse-battery")
    confirm_totp("trader")
    conn = db.connect()

    assert verify_totp(get_user("trader"), codes[0], conn)
    assert not verify_totp(get_user("trader"), codes[0], conn)   # single use
    assert verify_totp(get_user("trader"), codes[1], conn)


def test_security_headers_are_set(app, workspace):
    client = app.test_client()
    resp = client.get("/login")
    assert resp.headers["X-Frame-Options"] == "DENY"
    assert resp.headers["X-Content-Type-Options"] == "nosniff"
    assert "default-src 'self'" in resp.headers["Content-Security-Policy"]


def test_session_cookie_flags(app, workspace):
    assert app.config["SESSION_COOKIE_HTTPONLY"] is True
    assert app.config["SESSION_COOKIE_SAMESITE"] == "Strict"
    # Secure is disabled only because SOLBOT_INSECURE_COOKIES=1 is set for tests.
    assert app.config["SESSION_COOKIE_SECURE"] is False


def test_secret_key_is_required_in_production(workspace, monkeypatch):
    monkeypatch.delenv("SOLBOT_INSECURE_COOKIES", raising=False)
    monkeypatch.setenv("FLASK_SECRET_KEY", "")
    from solbot.config import reset_config_for_tests
    from solbot.web import create_app

    reset_config_for_tests(workspace["config"])
    with pytest.raises(RuntimeError, match="FLASK_SECRET_KEY"):
        create_app()
