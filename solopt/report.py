"""Reporting a run's progress back to the droplet, over HTTP.

The optimizer no longer pulls bulk history from anywhere (spec 5) - the daily
run reads the droplet's own Parquet files directly, in-process, and the
monthly RunPod run has its data shipped to it before it starts. What both
still need, when they run somewhere other than directly against the droplet's
own database, is a way to say what they are finding as they go and to hand
over the finished bundle. That is all this module does.

Built on the standard library alone, same reasoning as its predecessor: a
worker container should need nothing beyond NumPy (and CuPy, optionally) to
run the search itself, and a small reporting client is not worth a dependency
the container has to keep in step with the droplet's.
"""
from __future__ import annotations

import gzip
import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Sequence

log = logging.getLogger(__name__)

USER_AGENT = "solopt-report"


class ReportError(RuntimeError):
    """The droplet refused a report or could not be reached."""


@dataclass
class DropletClient:
    """Authenticated client for the droplet's optimizer-ingest endpoints."""

    base_url: str
    token: str
    timeout: float = 30.0
    retries: int = 3

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")
        if not self.token:
            raise ValueError("a bearer token is required; generate one on the droplet")

    def _request(self, method: str, path: str, *, body: Any = None) -> Any:
        url = f"{self.base_url}{path}"
        data = json.dumps(body).encode() if body is not None else None
        headers = {
            "Authorization": f"Bearer {self.token}",
            "User-Agent": USER_AGENT,
            "Accept-Encoding": "gzip",
            "Content-Type": "application/json",
        }

        last: Exception | None = None
        for attempt in range(self.retries + 1):
            request = urllib.request.Request(url, data=data, headers=headers, method=method)
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    payload = response.read()
                    if response.headers.get("Content-Encoding") == "gzip":
                        payload = gzip.decompress(payload)
                    return json.loads(payload.decode()) if payload else {}
            except urllib.error.HTTPError as exc:
                detail = exc.read()[:400].decode(errors="replace")
                if exc.code in (401, 403):
                    raise ReportError(
                        f"the droplet rejected the token ({exc.code}): {detail}"
                    ) from exc
                last = ReportError(f"HTTP {exc.code} from {path}: {detail}")
            except urllib.error.URLError as exc:
                last = ReportError(f"could not reach {self.base_url}: {exc.reason}")
            if attempt < self.retries:
                time.sleep(min(10.0, 2.0**attempt))
        raise last or ReportError("request failed")

    def push_feed(self, run_id: int, lines: Sequence[dict[str, Any]]) -> None:
        self._request(
            "POST", "/api/optimizer/feed", body={"run_id": run_id, "lines": list(lines)}
        )

    def push_run(self, run_id: int, payload: dict[str, Any]) -> None:
        self._request("POST", "/api/optimizer/run", body={"run_id": run_id, **payload})

    def push_bundle(self, bundle_payload: dict[str, Any]) -> dict[str, Any]:
        """Submit a finished, promotable bundle for the droplet to accept.

        The droplet re-validates it against its own gates and bounds (see
        ``solbot.paramsync.accept_bundle``) rather than trusting it - this
        call only delivers it.
        """
        return self._request("POST", "/api/optimizer/bundle", body=bundle_payload)

    def push_regime_model(self, run_id: int, symbol: str, model: dict[str, Any]) -> None:
        """Hand a coin's freshly-discovered fuzzy regime model to the droplet's
        own regime-model table - the same one the live trading loop reads from
        (see ``solbot.store``'s reuse of :class:`solopt.store.RunStore`)."""
        self._request(
            "POST", "/api/optimizer/regime-model",
            body={"run_id": run_id, "symbol": symbol, "model": model},
        )

    def push_regime_library(self, entry_payload: dict[str, Any]) -> None:
        """Hand one accepted per-coin, per-regime walk-forward result to the
        droplet's library - keyed by ``symbol`` + ``regime_cluster_id``, same
        shape as :class:`solopt.store.LibraryEntry`."""
        self._request("POST", "/api/optimizer/regime-library", body=entry_payload)

    def execution_profile(self) -> dict[str, Any]:
        return self._request("GET", "/api/optimizer/execution")
