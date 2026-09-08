"""Shared fixtures. Every test runs against a throwaway database and config."""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """Isolated DB + config file, wired through the environment."""
    db_file = tmp_path / "test.db"
    cfg_file = tmp_path / "config.json"
    candles_dir = tmp_path / "candles"
    monkeypatch.setenv("SOLBOT_DB", str(db_file))
    monkeypatch.setenv("SOLBOT_CONFIG", str(cfg_file))
    monkeypatch.setenv("SOLBOT_CANDLES_DIR", str(candles_dir))

    from solbot import db as db_module

    db_module.close_all()
    conn = db_module.init_db(db_file)
    yield {
        "dir": tmp_path, "db": db_file, "config": cfg_file, "conn": conn,
        "candles_dir": candles_dir,
    }
    db_module.close_all()


@pytest.fixture
def cfg(workspace):
    from solbot.config import Config

    return Config(workspace["config"])


@pytest.fixture
def settings(cfg):
    return cfg.as_dict()
