"""HttpClient: the rate-limited HTTP plumbing shared by every API client
(fix-up: full-pull mid-pair hang traced to `timeout` being accepted by
request() but never actually forwarded to the underlying httpx call)."""
from __future__ import annotations

from solbot.clients.base import HttpClient
from solbot.ratelimit import TokenBucket


class _FakeResponse:
    status_code = 200

    def json(self):
        return {"ok": True}


def _client() -> HttpClient:
    return HttpClient(TokenBucket(rate=1000.0, burst=1000, reserve=0.0))


def test_request_passes_a_supplied_timeout_through_to_httpx(monkeypatch):
    """timeout was dead code: request() accepted it but never forwarded it
    to self._client.request(...), so no caller could actually raise or
    lower the read timeout for a specific call - it silently always used
    the client's own constructor default regardless of what was asked."""
    client = _client()
    captured: dict = {}

    def fake_request(method, path, **kw):
        captured.update(kw)
        return _FakeResponse()

    monkeypatch.setattr(client._client, "request", fake_request)
    client.get("/ping", timeout=5.0)

    assert captured.get("timeout") == 5.0


def test_request_omits_the_timeout_kwarg_when_the_caller_does_not_supply_one(monkeypatch):
    """httpx treats an explicit timeout=None as 'no timeout at all', not
    'use the client's default' - so an ordinary caller that doesn't ask for
    a specific timeout must not have None passed through on its behalf,
    which would silently turn every such call unbounded."""
    client = _client()
    captured: dict = {}

    def fake_request(method, path, **kw):
        captured.update(kw)
        return _FakeResponse()

    monkeypatch.setattr(client._client, "request", fake_request)
    client.get("/ping")

    assert "timeout" not in captured
