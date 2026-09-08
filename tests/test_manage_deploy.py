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
