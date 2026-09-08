"""Shared HTTP plumbing for the API clients.

Every outbound call goes through :meth:`HttpClient.request`, which applies the
provider's token bucket, retries transient failures with jittered backoff, and
feeds HTTP 429 responses back into the bucket so the next cycle slows down
instead of hammering.
"""
from __future__ import annotations

import logging
import random
import threading
import time
from typing import Any, Mapping

import httpx

from ..ratelimit import TokenBucket

log = logging.getLogger(__name__)

RETRY_STATUS = {408, 425, 429, 500, 502, 503, 504}


class ApiError(RuntimeError):
    """A call failed in a way the caller is expected to handle."""

    def __init__(self, message: str, *, status: int | None = None, provider: str = ""):
        super().__init__(message)
        self.status = status
        self.provider = provider


class RateLimited(ApiError):
    """The provider returned 429 and the retry budget was exhausted."""


class HttpClient:
    """Thin wrapper over httpx.Client with a rate limiter attached."""

    provider = "http"
    base_url = ""

    def __init__(
        self,
        bucket: TokenBucket,
        *,
        timeout: float = 20.0,
        max_retries: int = 3,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        self.bucket = bucket
        self.max_retries = max_retries
        self._lock = threading.Lock()
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout, connect=10.0),
            headers={"User-Agent": "solana-ta-bot/1.0", **(headers or {})},
            follow_redirects=True,
        )

    def set_header(self, name: str, value: str | None) -> None:
        with self._lock:
            if value:
                self._client.headers[name] = value
            else:
                self._client.headers.pop(name, None)

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass

    def __enter__(self) -> "HttpClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    def request(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        json_body: Any = None,
        priority: str = "normal",
        cost: float = 1.0,
        timeout: float | None = None,
    ) -> Any:
        """Rate-limited JSON request. Raises ApiError on unrecoverable failure."""
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            if not self.bucket.acquire(cost=cost, priority=priority, timeout=timeout):
                raise RateLimited(
                    f"{self.provider}: local rate limiter timed out",
                    provider=self.provider,
                )
            try:
                resp = self._client.request(method, path, params=params, json=json_body)
            except httpx.HTTPError as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                self._sleep_backoff(attempt)
                continue

            if resp.status_code == 429:
                retry_after = _parse_retry_after(resp.headers.get("Retry-After"))
                self.bucket.penalise(seconds=retry_after or 30.0)
                last_error = RateLimited(
                    f"{self.provider}: rate limited (429)",
                    status=429,
                    provider=self.provider,
                )
                log.warning("%s rate limited; backing off %.0fs", self.provider, retry_after or 30.0)
                if attempt >= self.max_retries:
                    break
                time.sleep(min(retry_after or 2.0, 10.0))
                continue

            if resp.status_code in RETRY_STATUS:
                last_error = ApiError(
                    f"{self.provider}: HTTP {resp.status_code}",
                    status=resp.status_code,
                    provider=self.provider,
                )
                if attempt >= self.max_retries:
                    break
                self._sleep_backoff(attempt)
                continue

            if resp.status_code >= 400:
                raise ApiError(
                    f"{self.provider}: HTTP {resp.status_code} {resp.text[:200]}",
                    status=resp.status_code,
                    provider=self.provider,
                )

            try:
                return resp.json()
            except ValueError as exc:
                raise ApiError(
                    f"{self.provider}: response was not JSON", provider=self.provider
                ) from exc

        if isinstance(last_error, ApiError):
            raise last_error
        raise ApiError(
            f"{self.provider}: request failed after {self.max_retries + 1} attempts "
            f"({last_error})",
            provider=self.provider,
        )

    def get(self, path: str, **kw: Any) -> Any:
        return self.request("GET", path, **kw)

    def post(self, path: str, **kw: Any) -> Any:
        return self.request("POST", path, **kw)

    @staticmethod
    def _sleep_backoff(attempt: int) -> None:
        time.sleep(min(8.0, (2**attempt) * 0.5) * (0.7 + random.random() * 0.6))


def _parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None


def chunked(items: list[Any], size: int) -> list[list[Any]]:
    size = max(1, int(size))
    return [items[i : i + size] for i in range(0, len(items), size)]
