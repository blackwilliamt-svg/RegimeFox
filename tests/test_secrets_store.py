"""RunPod API key added to the dashboard's rotatable providers (settings
fix-up section 4, item 1): same masked-display/encrypted-at-rest/env-
fallback behaviour every other provider already gets, plus the standalone
resolver wfmc.py's orchestration functions use (rather than routing
through effective_keys(), which needs every other provider's field present
on the ``secrets`` object too - see resolve_runpod_key's own docstring).
"""
from __future__ import annotations

from solbot import secrets_store


def _config(workspace, monkeypatch, *, runpod_env: str = "", encryption_key: str = "e" * 64):
    monkeypatch.setenv("SECRET_ENCRYPTION_KEY", encryption_key)
    if runpod_env:
        monkeypatch.setenv("RUNPOD_API_KEY", runpod_env)
    else:
        monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    from solbot.config import Config

    return Config(workspace["config"])


def test_runpod_is_a_rotatable_provider():
    assert "runpod" in secrets_store.PROVIDERS


def test_key_status_reports_runpod_from_env_like_every_other_provider(workspace, monkeypatch):
    config = _config(workspace, monkeypatch, runpod_env="env-runpod-key-value")
    statuses = {
        s.provider: s for s in secrets_store.key_status(config.secrets, config.secrets.secret_encryption_key)
    }
    assert "runpod" in statuses
    assert statuses["runpod"].configured
    assert statuses["runpod"].source == "env"
    assert statuses["runpod"].masked.endswith("alue")  # last 4 chars of "env-runpod-key-value"


def test_key_status_reports_runpod_unconfigured_when_neither_env_nor_dashboard_has_it(
    workspace, monkeypatch
):
    config = _config(workspace, monkeypatch, runpod_env="")
    statuses = {
        s.provider: s for s in secrets_store.key_status(config.secrets, config.secrets.secret_encryption_key)
    }
    assert statuses["runpod"].configured is False
    assert statuses["runpod"].source == "none"


def test_key_status_reports_runpod_from_dashboard_once_rotated(workspace, monkeypatch):
    config = _config(workspace, monkeypatch, runpod_env="env-key")
    secrets_store.store_key("runpod", "rotated-dashboard-key", config.secrets.secret_encryption_key)
    statuses = {
        s.provider: s for s in secrets_store.key_status(config.secrets, config.secrets.secret_encryption_key)
    }
    assert statuses["runpod"].source == "dashboard"


def test_resolve_runpod_key_prefers_a_rotated_dashboard_value_over_env(workspace, monkeypatch):
    config = _config(workspace, monkeypatch, runpod_env="env-key")
    secrets_store.store_key("runpod", "rotated-key", config.secrets.secret_encryption_key)
    assert secrets_store.resolve_runpod_key(config.secrets) == "rotated-key"


def test_resolve_runpod_key_falls_back_to_env_when_nothing_is_stored(workspace, monkeypatch):
    config = _config(workspace, monkeypatch, runpod_env="env-key")
    assert secrets_store.resolve_runpod_key(config.secrets) == "env-key"


def test_resolve_runpod_key_is_empty_when_neither_source_has_one(workspace, monkeypatch):
    config = _config(workspace, monkeypatch, runpod_env="")
    assert secrets_store.resolve_runpod_key(config.secrets) == ""


def test_resolve_runpod_key_never_attributeerrors_on_a_minimal_secrets_double(workspace, monkeypatch):
    """wfmc.py's tests exercise run_monthly/run_benchmark/run_regime_pass
    against tiny doubles that define only runpod_api_key, not every field
    effective_keys() would need - resolve_runpod_key must not require them."""
    monkeypatch.setenv("SECRET_ENCRYPTION_KEY", "e" * 64)

    class OnlyRunpodField:
        runpod_api_key = "bare-double-key"

    assert secrets_store.resolve_runpod_key(OnlyRunpodField()) == "bare-double-key"

    class NothingAtAll:
        pass

    assert secrets_store.resolve_runpod_key(NothingAtAll()) == ""
