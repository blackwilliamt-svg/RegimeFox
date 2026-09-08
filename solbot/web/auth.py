"""Dashboard authentication: password + TOTP two-factor.

The dashboard can halt trading, close positions, edit risk parameters and rotate
API keys, so every route except the login screen itself requires a session. That
is enforced by a ``before_request`` hook rather than a decorator per view - a
route added later is protected by default instead of by remembering.

Passwords are hashed with argon2id. TOTP is app-based via ``pyotp`` (Google
Authenticator / Authy compatible) - no SMS, no third-party service, no phone
number. Backup codes are single-use and stored hashed, so a lost authenticator
does not lock the user out of their own kill switch.
"""
from __future__ import annotations

import base64
import hmac
import io
import json
import logging
import secrets
import sqlite3
import time
from functools import wraps
from typing import Any, Callable

import pyotp
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerifyMismatchError
from flask import (
    Blueprint, current_app, flash, g, redirect, render_template, request,
    session, url_for,
)

from .. import db

log = logging.getLogger(__name__)

bp = Blueprint("auth", __name__)
_hasher = PasswordHasher()

BACKUP_CODE_COUNT = 10
# Routes reachable without a session. Everything else is denied by default.
PUBLIC_ENDPOINTS = {"auth.login", "auth.setup", "static", "auth.logout"}


# --------------------------------------------------------------------------
# user records
# --------------------------------------------------------------------------
def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(stored_hash: str, password: str) -> bool:
    try:
        return _hasher.verify(stored_hash, password)
    except (VerifyMismatchError, InvalidHashError, Exception):
        return False


def user_count(conn: sqlite3.Connection | None = None) -> int:
    conn = conn or db.connect()
    return int(conn.execute("SELECT COUNT(*) AS n FROM users").fetchone()["n"])


def get_user(username: str, conn: sqlite3.Connection | None = None) -> sqlite3.Row | None:
    conn = conn or db.connect()
    return conn.execute(
        "SELECT * FROM users WHERE username = ?", (username.strip().lower(),)
    ).fetchone()


def create_user(
    username: str, password: str, conn: sqlite3.Connection | None = None
) -> tuple[str, list[str]]:
    """Create a user. Returns (totp_secret, backup_codes) - shown exactly once."""
    conn = conn or db.connect()
    username = username.strip().lower()
    if not username or len(password) < 12:
        raise ValueError("username is required and the password must be 12+ characters")
    if get_user(username, conn) is not None:
        raise ValueError(f"user {username!r} already exists")

    totp_secret = pyotp.random_base32()
    codes = [secrets.token_hex(4) for _ in range(BACKUP_CODE_COUNT)]
    hashed = json.dumps([hash_password(c) for c in codes])

    conn.execute(
        "INSERT INTO users(username, password_hash, totp_secret, totp_confirmed, "
        "backup_codes, created_at) VALUES (?,?,?,0,?,?)",
        (username, hash_password(password), totp_secret, hashed, db.now()),
    )
    db.log_event(
        f"Dashboard user {username!r} created.", category="auth", conn=conn
    )
    return totp_secret, codes


def confirm_totp(username: str, conn: sqlite3.Connection | None = None) -> None:
    conn = conn or db.connect()
    conn.execute(
        "UPDATE users SET totp_confirmed = 1 WHERE username = ?", (username.lower(),)
    )


def regenerate_backup_codes(
    username: str, conn: sqlite3.Connection | None = None
) -> list[str]:
    conn = conn or db.connect()
    codes = [secrets.token_hex(4) for _ in range(BACKUP_CODE_COUNT)]
    conn.execute(
        "UPDATE users SET backup_codes = ? WHERE username = ?",
        (json.dumps([hash_password(c) for c in codes]), username.lower()),
    )
    db.log_event(
        f"Backup codes regenerated for {username!r}.", level="warn", category="auth", conn=conn
    )
    return codes


def _consume_backup_code(
    user: sqlite3.Row, code: str, conn: sqlite3.Connection
) -> bool:
    try:
        hashes = json.loads(user["backup_codes"] or "[]")
    except json.JSONDecodeError:
        return False
    for i, h in enumerate(hashes):
        if verify_password(h, code.strip()):
            hashes.pop(i)
            conn.execute(
                "UPDATE users SET backup_codes = ? WHERE id = ?",
                (json.dumps(hashes), user["id"]),
            )
            db.log_event(
                f"Backup code used for {user['username']!r}; {len(hashes)} remain.",
                level="warn",
                category="auth",
                conn=conn,
            )
            return True
    return False


def verify_totp(user: sqlite3.Row, token: str, conn: sqlite3.Connection) -> bool:
    token = (token or "").strip().replace(" ", "")
    if not token:
        return False
    secret = user["totp_secret"]
    if secret and pyotp.TOTP(secret).verify(token, valid_window=1):
        return True
    return _consume_backup_code(user, token, conn)


def provisioning_uri(username: str, secret: str) -> str:
    return pyotp.TOTP(secret).provisioning_uri(
        name=username, issuer_name="Solana TA Bot"
    )


def qr_data_uri(uri: str) -> str:
    """Inline QR image so enrollment needs no external asset request."""
    import qrcode  # noqa: PLC0415

    img = qrcode.make(uri)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


# --------------------------------------------------------------------------
# rate limiting
# --------------------------------------------------------------------------
def record_attempt(
    username: str | None, ip: str | None, success: bool, conn: sqlite3.Connection | None = None
) -> None:
    conn = conn or db.connect()
    conn.execute(
        "INSERT INTO login_attempts(ts, username, ip, success) VALUES (?,?,?,?)",
        (db.now(), (username or "").lower(), ip, 1 if success else 0),
    )


def too_many_attempts(
    ip: str | None, cfg: dict[str, Any], conn: sqlite3.Connection | None = None
) -> bool:
    conn = conn or db.connect()
    window = int(cfg["login_rate_limit_window_seconds"])
    limit = int(cfg["login_rate_limit_attempts"])
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM login_attempts WHERE success = 0 AND ip = ? AND ts > ?",
        (ip, db.now() - window),
    ).fetchone()
    return int(row["n"]) >= limit


# --------------------------------------------------------------------------
# session handling
# --------------------------------------------------------------------------
def login_session(username: str) -> None:
    session.clear()
    session["username"] = username
    session["authenticated_at"] = int(time.time())
    session.permanent = True


def current_user() -> str | None:
    return session.get("username")


def session_expired(cfg: dict[str, Any]) -> bool:
    ts = session.get("authenticated_at")
    if not ts:
        return True
    return (time.time() - float(ts)) > int(cfg["session_timeout_minutes"]) * 60


def login_required(fn: Callable) -> Callable:
    @wraps(fn)
    def wrapper(*args: Any, **kw: Any):
        if not current_user():
            return redirect(url_for("auth.login", next=request.path))
        return fn(*args, **kw)

    return wrapper


def install_guard(app: Any) -> None:
    """Deny every request without a session, except the login screen itself."""

    @app.before_request
    def _guard():  # type: ignore[misc]
        cfg = current_app.config["SOLBOT_CONFIG"].as_dict()
        endpoint = request.endpoint or ""

        if endpoint in PUBLIC_ENDPOINTS:
            return None
        if endpoint.startswith("static"):
            return None
        # The optimizer endpoints are for a headless client on another machine
        # with no browser and no TOTP device. They enforce their own bearer-token
        # auth in solbot.web.optimizer.token_required and are exempt from the
        # session guard rather than from authentication.
        if endpoint.startswith("optimizer."):
            return None

        if not current_user():
            if request.path.startswith("/api/"):
                return {"error": "authentication required"}, 401
            return redirect(url_for("auth.login", next=request.path))

        if session_expired(cfg):
            session.clear()
            if request.path.startswith("/api/"):
                return {"error": "session expired"}, 401
            flash("Session expired - please sign in again.", "warning")
            return redirect(url_for("auth.login"))

        # Sliding expiry: activity keeps the session alive.
        session["authenticated_at"] = int(time.time())
        g.username = current_user()
        return None


# --------------------------------------------------------------------------
# routes
# --------------------------------------------------------------------------
@bp.route("/setup", methods=["GET", "POST"])
def setup():
    """First-run enrollment. Refuses once a user exists."""
    if user_count() > 0:
        flash("A dashboard user already exists. Use manage.py to add another.", "error")
        return redirect(url_for("auth.login"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        confirm = request.form.get("confirm", "")
        if password != confirm:
            flash("Passwords do not match.", "error")
            return render_template("setup.html")
        try:
            secret, codes = create_user(username, password)
        except ValueError as exc:
            flash(str(exc), "error")
            return render_template("setup.html")

        uri = provisioning_uri(username, secret)
        return render_template(
            "setup_2fa.html",
            username=username,
            secret=secret,
            qr=qr_data_uri(uri),
            codes=codes,
        )

    return render_template("setup.html")


@bp.route("/setup/confirm", methods=["POST"])
def setup_confirm():
    username = request.form.get("username", "").strip().lower()
    token = request.form.get("token", "")
    user = get_user(username)
    if user is None:
        flash("Unknown user.", "error")
        return redirect(url_for("auth.login"))
    if not verify_totp(user, token, db.connect()):
        flash("That code did not verify. Check the clock on your phone and try again.", "error")
        uri = provisioning_uri(username, user["totp_secret"])
        return render_template(
            "setup_2fa.html", username=username, secret=user["totp_secret"],
            qr=qr_data_uri(uri), codes=None,
        )
    confirm_totp(username)
    flash("Two-factor authentication enabled. Please sign in.", "success")
    return redirect(url_for("auth.login"))


@bp.route("/login", methods=["GET", "POST"])
def login():
    if user_count() == 0:
        return redirect(url_for("auth.setup"))

    cfg = current_app.config["SOLBOT_CONFIG"].as_dict()
    conn = db.connect()
    ip = request.headers.get("X-Forwarded-For", request.remote_addr or "").split(",")[0].strip()

    if request.method == "POST":
        if too_many_attempts(ip, cfg, conn):
            db.log_event(
                f"Login rate limit hit from {ip}.", level="alert", category="auth", conn=conn
            )
            flash("Too many failed attempts. Try again later.", "error")
            return render_template("login.html"), 429

        username = request.form.get("username", "").strip().lower()
        password = request.form.get("password", "")
        token = request.form.get("token", "")

        user = get_user(username, conn)
        # Always run a verification so a missing user and a wrong password take
        # comparable time.
        password_ok = (
            verify_password(user["password_hash"], password)
            if user
            else verify_password(hash_password("dummy"), "wrong")
        )
        totp_ok = bool(user) and verify_totp(user, token, conn)

        if user and password_ok and totp_ok:
            record_attempt(username, ip, True, conn)
            conn.execute(
                "UPDATE users SET last_login = ? WHERE id = ?", (db.now(), user["id"])
            )
            login_session(username)
            db.log_event(
                f"Dashboard login: {username} from {ip}.", category="auth", conn=conn
            )
            nxt = request.args.get("next") or url_for("dashboard.index")
            # Only ever redirect within this app.
            if not nxt.startswith("/") or nxt.startswith("//"):
                nxt = url_for("dashboard.index")
            return redirect(nxt)

        record_attempt(username, ip, False, conn)
        db.log_event(
            f"Failed dashboard login for {username!r} from {ip}.",
            level="warn",
            category="auth",
            conn=conn,
        )
        flash("Invalid credentials.", "error")

    return render_template("login.html")


@bp.route("/logout")
def logout():
    username = current_user()
    session.clear()
    if username:
        db.log_event(f"Dashboard logout: {username}.", category="auth")
    return redirect(url_for("auth.login"))
