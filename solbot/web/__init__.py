"""Flask application factory for the dashboard.

The web process is deliberately read-mostly. It reads state the worker wrote and
enqueues commands the worker executes; it never opens or closes a position
itself. That keeps a single writer on the trading path and means a crash or a
restart of the dashboard cannot affect an open position.

Session cookies are Secure, HttpOnly and SameSite=Strict. Secure is on by
default because the deployment terminates TLS at nginx; set
``SOLBOT_INSECURE_COOKIES=1`` only for local development over plain HTTP.
"""
from __future__ import annotations

import logging
import os
import secrets
from datetime import timedelta
from typing import Any

from flask import Flask

from .. import db
from ..config import Config, get_config

log = logging.getLogger(__name__)


def create_app(config: Config | None = None) -> Flask:
    cfg = config or get_config()
    db.init_db()

    app = Flask(__name__, template_folder="templates", static_folder="static")

    secret = cfg.secrets.flask_secret_key
    if not secret:
        if os.getenv("SOLBOT_INSECURE_COOKIES") == "1":
            secret = secrets.token_hex(32)
            log.warning("FLASK_SECRET_KEY unset; using an ephemeral development key")
        else:
            raise RuntimeError(
                "FLASK_SECRET_KEY is not set. Generate one with: "
                'python -c "import secrets;print(secrets.token_hex(32))"'
            )

    insecure = os.getenv("SOLBOT_INSECURE_COOKIES") == "1"
    app.config.update(
        SECRET_KEY=secret,
        SOLBOT_CONFIG=cfg,
        SESSION_COOKIE_SECURE=not insecure,
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Strict",
        SESSION_COOKIE_NAME="solbot_session",
        PERMANENT_SESSION_LIFETIME=timedelta(minutes=int(cfg["session_timeout_minutes"])),
        MAX_CONTENT_LENGTH=256 * 1024,
        JSON_SORT_KEYS=False,
        TEMPLATES_AUTO_RELOAD=False,
    )

    @app.template_filter("ts")
    def _ts(value: Any) -> str:
        """Unix seconds -> readable UTC. Templates render times, never raw epochs."""
        if not value:
            return "—"
        import time as _time

        return _time.strftime("%Y-%m-%d %H:%M", _time.gmtime(int(value)))

    @app.template_filter("money")
    def _money(value: Any) -> str:
        try:
            v = float(value)
        except (TypeError, ValueError):
            return "—"
        return f"-${abs(v):,.2f}" if v < 0 else f"${v:,.2f}"

    from . import api, auth, routes

    app.register_blueprint(auth.bp)
    app.register_blueprint(routes.bp)
    app.register_blueprint(api.bp, url_prefix="/api")
    auth.install_guard(app)

    @app.after_request
    def security_headers(response):  # type: ignore[misc]
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        # The dashboard ships its own JS/CSS; nothing is loaded from a CDN.
        response.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; "
            "script-src 'self'; connect-src 'self'; frame-ancestors 'none'",
        )
        if not insecure:
            response.headers.setdefault(
                "Strict-Transport-Security", "max-age=31536000; includeSubDomains"
            )
        return response

    @app.teardown_appcontext
    def _close(_exc: BaseException | None) -> None:
        # Connections are per-thread and reused; nothing to close per request.
        return None

    return app
