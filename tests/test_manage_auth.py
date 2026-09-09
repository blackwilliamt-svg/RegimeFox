"""manage.py's credential-change commands: change-password and rename-user.

solbot.web.auth's own update_password()/rename_user() helpers are exercised
directly too, since the manage.py commands are thin wrappers around them.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import manage  # noqa: E402  (repo-root script, not a package)
from solbot.config import reset_config_for_tests  # noqa: E402
from solbot.web.auth import (  # noqa: E402
    create_user, get_user, hash_password, rename_user, update_password,
    verify_password,
)


@pytest.fixture
def manage_module(workspace):
    reset_config_for_tests(workspace["config"])
    return manage


def _make_user(username: str = "trader", password: str = "correct-horse-battery"):
    return create_user(username, password)


# --------------------------------------------------------------------------
# solbot.web.auth.update_password / rename_user
# --------------------------------------------------------------------------
def test_update_password_changes_the_hash_and_leaves_totp_untouched(workspace):
    secret, codes = _make_user()

    update_password("trader", hash_password("a-brand-new-password"))

    row = get_user("trader")
    assert verify_password(row["password_hash"], "a-brand-new-password")
    assert not verify_password(row["password_hash"], "correct-horse-battery")
    assert row["totp_secret"] == secret
    assert row["backup_codes"] is not None
    import json
    assert len(json.loads(row["backup_codes"])) == len(codes)


def test_rename_user_moves_the_username_and_carries_everything_else(workspace):
    secret, codes = _make_user("oldname", "correct-horse-battery")
    before = get_user("oldname")

    rename_user("oldname", "newname")

    assert get_user("oldname") is None
    after = get_user("newname")
    assert after is not None
    assert after["password_hash"] == before["password_hash"]
    assert after["totp_secret"] == secret
    assert after["backup_codes"] == before["backup_codes"]


def test_rename_user_lowercases_and_strips_both_names(workspace):
    _make_user("oldname", "correct-horse-battery")
    rename_user("  OldName  ", "  NewName  ")
    assert get_user("oldname") is None
    assert get_user("newname") is not None


# --------------------------------------------------------------------------
# manage.py change-password
# --------------------------------------------------------------------------
def test_change_password_updates_the_hash(manage_module, workspace, monkeypatch, capsys):
    _make_user()
    prompts = iter(["a-new-strong-password", "a-new-strong-password"])
    monkeypatch.setattr("getpass.getpass", lambda *_a, **_kw: next(prompts))

    rc = manage_module.cmd_change_password(argparse.Namespace(username="trader"))

    assert rc == 0
    assert "Password updated for trader." in capsys.readouterr().out
    row = get_user("trader")
    assert verify_password(row["password_hash"], "a-new-strong-password")


def test_change_password_leaves_totp_and_backup_codes_alone(manage_module, workspace, monkeypatch):
    secret, codes = _make_user()
    before = get_user("trader")
    prompts = iter(["a-new-strong-password", "a-new-strong-password"])
    monkeypatch.setattr("getpass.getpass", lambda *_a, **_kw: next(prompts))

    manage_module.cmd_change_password(argparse.Namespace(username="trader"))

    after = get_user("trader")
    assert after["totp_secret"] == secret
    assert after["backup_codes"] == before["backup_codes"]


def test_change_password_rejects_a_mismatched_confirmation(manage_module, workspace, monkeypatch, capsys):
    _make_user()
    before = get_user("trader")["password_hash"]
    prompts = iter(["a-new-strong-password", "a-different-password"])
    monkeypatch.setattr("getpass.getpass", lambda *_a, **_kw: next(prompts))

    rc = manage_module.cmd_change_password(argparse.Namespace(username="trader"))

    assert rc == 1
    assert "passwords do not match" in capsys.readouterr().out
    assert get_user("trader")["password_hash"] == before   # untouched


def test_change_password_rejects_a_too_short_password(manage_module, workspace, monkeypatch, capsys):
    _make_user()
    before = get_user("trader")["password_hash"]
    prompts = iter(["short", "short"])
    monkeypatch.setattr("getpass.getpass", lambda *_a, **_kw: next(prompts))

    rc = manage_module.cmd_change_password(argparse.Namespace(username="trader"))

    assert rc == 1
    assert "12+ characters" in capsys.readouterr().out
    assert get_user("trader")["password_hash"] == before   # untouched


def test_change_password_on_a_nonexistent_user_fails_cleanly(manage_module, workspace, capsys):
    rc = manage_module.cmd_change_password(argparse.Namespace(username="ghost"))

    assert rc == 1
    assert "no such user: ghost" in capsys.readouterr().out


# --------------------------------------------------------------------------
# manage.py rename-user
# --------------------------------------------------------------------------
def test_rename_user_command_renames_and_reports_success(manage_module, workspace, capsys):
    _make_user("oldname", "correct-horse-battery")

    rc = manage_module.cmd_rename_user(
        argparse.Namespace(old_username="oldname", new_username="newname")
    )

    assert rc == 0
    assert "Renamed oldname to newname." in capsys.readouterr().out
    assert get_user("oldname") is None
    assert get_user("newname") is not None


def test_rename_user_command_refuses_when_the_target_name_is_taken(manage_module, workspace, capsys):
    _make_user("alice", "correct-horse-battery-1")
    _make_user("bob", "correct-horse-battery-2")

    rc = manage_module.cmd_rename_user(
        argparse.Namespace(old_username="alice", new_username="bob")
    )

    assert rc == 1
    assert "a user named bob already exists" in capsys.readouterr().out
    # Neither account was touched.
    assert get_user("alice") is not None
    assert get_user("bob") is not None


def test_rename_user_command_refuses_a_nonexistent_source_user(manage_module, workspace, capsys):
    rc = manage_module.cmd_rename_user(
        argparse.Namespace(old_username="ghost", new_username="somebody")
    )

    assert rc == 1
    assert "no such user: ghost" in capsys.readouterr().out
    assert get_user("somebody") is None
