"""manage.py's config.json/db-must-exist-before-systemd-bind-mount fix."""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import manage  # noqa: E402  (repo-root script, not a package)
from solbot.config import reset_config_for_tests  # noqa: E402


@pytest.fixture
def manage_module(workspace):
    """Point manage.py's `get_config()` singleton at this test's isolated
    SOLBOT_CONFIG rather than reloading solbot.config (reloading would mint a
    second, distinct ConfigError class and break every other test module's
    `pytest.raises(ConfigError)`)."""
    reset_config_for_tests(workspace["config"])
    return manage


def test_init_db_creates_config_json_when_missing(manage_module, workspace):
    assert not workspace["config"].exists()
    rc = manage_module.cmd_init_db(None)
    assert rc == 0
    assert workspace["config"].exists()


def test_init_db_does_not_clobber_an_existing_config(manage_module, workspace):
    workspace["config"].write_text('{"max_slippage_pct": 0.75}', encoding="utf-8")
    manage_module.cmd_init_db(None)

    stored = json.loads(workspace["config"].read_text(encoding="utf-8"))
    assert stored["max_slippage_pct"] == 0.75


def test_check_deploy_warns_when_config_is_missing(manage_module, workspace, capsys):
    rc = manage_module.cmd_check_deploy(None)
    assert rc == 1
    out = capsys.readouterr().out
    assert "config.json" in out


def test_check_deploy_passes_after_init_db(manage_module, workspace, capsys):
    manage_module.cmd_init_db(None)
    rc = manage_module.cmd_check_deploy(None)
    assert rc == 0
    out = capsys.readouterr().out
    assert "[ok]" in out


class _Args:
    def __init__(self, tiers=None):
        self.tiers = tiers


def test_benchmark_runpod_prints_results_and_never_touches_the_setting(
    manage_module, workspace, capsys, monkeypatch
):
    """gap-closure item 7: the CLI reports what run_benchmark found and is
    explicit that runpod_gpu_type is not changed automatically."""
    manage_module.cmd_init_db(None)

    def fake_run_benchmark(cfg, store, secrets, *, tiers=None, **kw):
        assert tiers == ["TIER-A", "TIER-B"]
        return {
            "ran": True,
            "results": [
                {
                    "gpu_type": "TIER-A", "ok": True, "elapsed_seconds": 12.3,
                    "cost_usd": 0.01, "cost_per_1000_combinations": 0.5,
                    "teardown_clean": True, "error": "",
                },
            ],
            "recommendation": "TIER-A ($0.5000 per 1,000 combinations)",
        }

    monkeypatch.setattr("solbot.wfmc.run_benchmark", fake_run_benchmark)

    rc = manage_module.cmd_benchmark_runpod(_Args(tiers="TIER-A,TIER-B"))

    assert rc == 0
    out = capsys.readouterr().out
    assert "TIER-A" in out
    assert "Recommendation" in out
    assert "does NOT change runpod_gpu_type automatically" in out


def test_benchmark_runpod_reports_failure_to_run(manage_module, workspace, capsys, monkeypatch):
    manage_module.cmd_init_db(None)
    monkeypatch.setattr(
        "solbot.wfmc.run_benchmark",
        lambda cfg, store, secrets, **kw: {"ran": False, "reason": "RUNPOD_API_KEY is not configured"},
    )

    rc = manage_module.cmd_benchmark_runpod(_Args())

    assert rc == 1
    assert "did not run" in capsys.readouterr().out
