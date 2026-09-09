"""Encrypted storage for API keys rotated from the dashboard.

The environment stays the default source of every key. When one is rotated from
the settings page it is encrypted with a key derived from
``SECRET_ENCRYPTION_KEY`` and stored in ``api_keys``, which then takes
precedence. Nothing here ever handles the wallet private key - that is loaded
directly from the environment inside the executor and never reaches the web
process.

Reads always return the *masked* form (last four characters) unless the caller
explicitly asks to decrypt, so a template cannot accidentally render a key.
"""
from __future__ import annotations

import base64
import hashlib
import logging
import sqlite3
from dataclasses import dataclass
from typing import Any

from . import db

log = logging.getLogger(__name__)

# "bulk_data" is the bearer token a RunPod worker presents to report a run's
# progress back. "runpod" is this droplet's own RunPod API key, used to
# orchestrate the monthly retest / GPU-tier benchmark / fuzzy-regime pass.
# Every provider is rotatable from the settings page and falls back to the
# environment.
PROVIDERS = ("jupiter", "rugcheck", "bulk_data", "runpod")


class SecretsUnavailable(RuntimeError):
    """SECRET_ENCRYPTION_KEY is missing or unusable."""


def _fernet(encryption_key: str):
    if not encryption_key:
        raise SecretsUnavailable(
            "SECRET_ENCRYPTION_KEY is not set; API keys cannot be rotated from the "
            "dashboard until it is. Generate one with: "
            'python -c "import secrets;print(secrets.token_hex(32))"'
        )
    try:
        from cryptography.fernet import Fernet  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise SecretsUnavailable(
            "the 'cryptography' package is required for dashboard key rotation"
        ) from exc
    digest = hashlib.sha256(encryption_key.encode("utf-8")).digest()
    return Fernet(base64.urlsafe_b64encode(digest))


def mask(value: str) -> str:
    if not value:
        return ""
    if len(value) <= 4:
        return "•" * len(value)
    return "•" * 8 + value[-4:]


@dataclass(slots=True)
class KeyStatus:
    provider: str
    configured: bool
    masked: str
    source: str          # env | dashboard | none
    updated_at: int | None = None
    updated_by: str | None = None


def store_key(
    provider: str,
    value: str,
    encryption_key: str,
    *,
    updated_by: str = "dashboard",
    conn: sqlite3.Connection | None = None,
) -> None:
    conn = conn or db.connect()
    token = _fernet(encryption_key).encrypt(value.encode("utf-8")).decode("ascii")
    conn.execute(
        "INSERT INTO api_keys(provider, ciphertext, last4, updated_at, updated_by) "
        "VALUES (?,?,?,?,?) ON CONFLICT(provider) DO UPDATE SET "
        "ciphertext=excluded.ciphertext, last4=excluded.last4, "
        "updated_at=excluded.updated_at, updated_by=excluded.updated_by",
        (provider, token, value[-4:], db.now(), updated_by),
    )


def read_key(
    provider: str, encryption_key: str, conn: sqlite3.Connection | None = None
) -> str | None:
    """Decrypt a stored key. Returns None if none is stored or it cannot be read."""
    conn = conn or db.connect()
    row = conn.execute(
        "SELECT ciphertext FROM api_keys WHERE provider = ?", (provider,)
    ).fetchone()
    if row is None:
        return None
    try:
        return _fernet(encryption_key).decrypt(row["ciphertext"].encode("ascii")).decode("utf-8")
    except SecretsUnavailable:
        raise
    except Exception:
        log.warning("stored %s key could not be decrypted; check SECRET_ENCRYPTION_KEY", provider)
        return None


def delete_key(provider: str, conn: sqlite3.Connection | None = None) -> None:
    conn = conn or db.connect()
    conn.execute("DELETE FROM api_keys WHERE provider = ?", (provider,))


def _env_keys(secrets: Any) -> dict[str, str]:
    """The environment-supplied value for each provider, empty where unset."""
    return {
        "jupiter": secrets.jupiter_api_key,
        "rugcheck": secrets.rugcheck_api_key,
        "bulk_data": getattr(secrets, "bulk_data_token", ""),
        "runpod": getattr(secrets, "runpod_api_key", ""),
    }


def effective_keys(secrets: Any, encryption_key: str, conn: sqlite3.Connection | None = None) -> dict[str, str]:
    """Resolve every provider key: a stored rotation wins over the environment."""
    conn = conn or db.connect()
    env = _env_keys(secrets)
    out = dict(env)
    for provider in PROVIDERS:
        try:
            stored = read_key(provider, encryption_key, conn)
        except SecretsUnavailable:
            stored = None
        if stored:
            out[provider] = stored
    return out


def resolve_runpod_key(secrets: Any, conn: sqlite3.Connection | None = None) -> str:
    """RunPod's rotated-or-env key - the same "dashboard rotation wins over
    the environment" rule every other provider gets (effective_keys()),
    but standalone rather than routed through it: wfmc.py's monthly-run /
    benchmark / regime-pass orchestration only ever wants this one
    provider, and is exercised in tests against lightweight ``secrets``
    doubles that intentionally don't define jupiter/rugcheck/bulk_data's
    fields too - effective_keys() would AttributeError on those.
    """
    conn = conn or db.connect()
    encryption_key = getattr(secrets, "secret_encryption_key", "") or ""
    try:
        stored = read_key("runpod", encryption_key, conn)
    except SecretsUnavailable:
        stored = None
    return stored or (getattr(secrets, "runpod_api_key", "") or "")


def key_status(
    secrets: Any, encryption_key: str, conn: sqlite3.Connection | None = None
) -> list[KeyStatus]:
    """What the settings page shows: masked values and where each one came from."""
    conn = conn or db.connect()
    env = _env_keys(secrets)
    rows = {
        r["provider"]: r
        for r in conn.execute("SELECT * FROM api_keys").fetchall()
    }
    out: list[KeyStatus] = []
    for provider in PROVIDERS:
        row = rows.get(provider)
        if row is not None:
            out.append(
                KeyStatus(
                    provider=provider,
                    configured=True,
                    masked=mask("x" * 8 + (row["last4"] or "")),
                    source="dashboard",
                    updated_at=row["updated_at"],
                    updated_by=row["updated_by"],
                )
            )
        elif env.get(provider):
            out.append(
                KeyStatus(provider, True, mask(env[provider]), "env")
            )
        else:
            out.append(KeyStatus(provider, False, "", "none"))
    return out
