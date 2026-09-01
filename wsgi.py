"""WSGI entry point for gunicorn.

    gunicorn --workers 2 --bind 127.0.0.1:8090 wsgi:app

Port 8090 is chosen to stay clear of the freight lead-gen app already running on
the droplet. See deploy/DEPLOY.md for the systemd unit and the nginx front end.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from solbot.web import create_app

app = create_app()

if __name__ == "__main__":
    # Development only. Production runs gunicorn behind nginx with TLS.
    app.run(host="127.0.0.1", port=8090, debug=False)
