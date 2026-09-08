"""Settings validation, persistence and hot reload."""
from __future__ import annotations

import json
import time

import pytest

from solbot.config import Config, ConfigError, DEFAULTS, SPEC


def test_defaults_load(cfg):
    assert cfg["trading_mode"] == "paper"
    assert cfg["max_total_deployed_pct"] == 0.90
    assert cfg["paper_starting_balance"] == 1000.0


def test_every_spec_key_has_a_default():
    missing = set(SPEC) - set(DEFAULTS)
    assert not missing, f"SPEC keys without a default: {missing}"


def test_paper_is_the_only_starting_mode():
    """Live must be reached explicitly, never by default."""
    assert DEFAULTS["trading_mode"] == "paper"


def test_update_persists_and_reloads(cfg, workspace):
    cfg.update({"max_slippage_pct": 1.25})
    assert cfg["max_slippage_pct"] == 1.25

    stored = json.loads(workspace["config"].read_text(encoding="utf-8"))
    assert stored["max_slippage_pct"] == 1.25

    fresh = Config(workspace["config"])
    assert fresh["max_slippage_pct"] == 1.25


def test_out_of_bounds_is_rejected(cfg):
    with pytest.raises(ConfigError, match="must be <="):
        cfg.update({"max_slippage_pct": 99.0})
    assert cfg["max_slippage_pct"] == DEFAULTS["max_slippage_pct"]


def test_absurd_reward_risk_window_is_rejected(cfg):
    """The spec's example of a change that must not be accepted."""
    with pytest.raises(ConfigError):
        cfg.update({"rr_min": 3.5, "rr_max": 2.0})


def test_ema_ordering_is_enforced(cfg):
    with pytest.raises(ConfigError, match="ema_fast"):
        cfg.update({"ema_fast": 50, "ema_slow": 20})


def test_over_deployment_is_rejected(cfg):
    """A single position cannot be allowed to exceed the total-deployed cap."""
    with pytest.raises(ConfigError, match="total-deployed"):
        cfg.update({"max_position_pct_of_wallet": 0.5, "max_total_deployed_pct": 0.4})


def test_hot_scan_cannot_be_slower_than_broad(cfg):
    with pytest.raises(ConfigError, match="hot scan"):
        cfg.update({"hot_scan_seconds": 60, "broad_scan_seconds": 15})


def test_trading_mode_is_not_dashboard_editable(cfg):
    with pytest.raises(ConfigError, match="not editable"):
        cfg.update({"trading_mode": "live"})
    # The explicit path still works.
    cfg.set_trading_mode("live")
    assert cfg.is_live


def test_unknown_keys_are_rejected(cfg):
    with pytest.raises(ConfigError, match="unknown setting"):
        cfg.update({"not_a_real_setting": 1})


def test_price_batch_cannot_exceed_the_api_limit(cfg):
    """Jupiter rejects more than 50 ids per call."""
    with pytest.raises(ConfigError):
        cfg.update({"jupiter_price_batch_size": 100})


def test_maybe_reload_picks_up_external_edits(cfg, workspace):
    assert cfg["volume_spike_multiple"] == 2.0
    time.sleep(0.01)
    workspace["config"].write_text(
        json.dumps({**cfg.as_dict(), "volume_spike_multiple": 3.5}), encoding="utf-8"
    )
    assert cfg.maybe_reload()
    assert cfg["volume_spike_multiple"] == 3.5
    assert not cfg.maybe_reload()   # no second reload without another change


def test_corrupt_config_falls_back_to_defaults(workspace):
    workspace["config"].write_text("{ this is not json", encoding="utf-8")
    cfg = Config(workspace["config"])
    assert cfg["max_total_deployed_pct"] == DEFAULTS["max_total_deployed_pct"]


def test_bad_stored_value_falls_back_for_that_key_only(workspace):
    workspace["config"].write_text(
        json.dumps({"max_slippage_pct": 999.0, "candle_minutes": 12}), encoding="utf-8"
    )
    cfg = Config(workspace["config"])
    assert cfg["max_slippage_pct"] == DEFAULTS["max_slippage_pct"]   # rejected
    assert cfg["candle_minutes"] == 12                                # accepted


def test_booleans_coerce_from_form_strings(cfg):
    cfg.update({"congestion_check_enabled": "false"})
    assert cfg["congestion_check_enabled"] is False
    cfg.update({"congestion_check_enabled": "on"})
    assert cfg["congestion_check_enabled"] is True


def test_update_reports_only_actual_changes(cfg):
    applied = cfg.update({"max_slippage_pct": DEFAULTS["max_slippage_pct"]})
    assert applied == {}
    applied = cfg.update({"max_slippage_pct": 0.75})
    assert applied == {"max_slippage_pct": (DEFAULTS["max_slippage_pct"], 0.75)}


def test_ensure_on_disk_creates_defaults_file_when_missing(cfg, workspace):
    assert not workspace["config"].exists()
    created = cfg.ensure_on_disk()
    assert created is True
    assert workspace["config"].exists()

    stored = json.loads(workspace["config"].read_text(encoding="utf-8"))
    assert stored["trading_mode"] == DEFAULTS["trading_mode"]
    assert stored["max_total_deployed_pct"] == DEFAULTS["max_total_deployed_pct"]


def test_ensure_on_disk_is_a_noop_when_a_file_already_exists(cfg, workspace):
    cfg.update({"max_slippage_pct": 0.75})
    written_at = workspace["config"].stat().st_mtime

    created = cfg.ensure_on_disk()

    assert created is False
    assert workspace["config"].stat().st_mtime == written_at
    stored = json.loads(workspace["config"].read_text(encoding="utf-8"))
    assert stored["max_slippage_pct"] == 0.75  # not clobbered back to defaults
