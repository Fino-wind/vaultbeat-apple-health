"""Transient-failure retries in VaultbeatCloudClient._request.

Drives the REAL request path through httpx.MockTransport (the `transport`
test seam) — not a fake client — so what is under test is the exact retry
loop production runs.

Why this exists: the box's uplink goes through a proxy that intermittently
kills large responses (`ProxyError: 503`, Known Bug since 2026-07-29, hit
2/17 kinds on check_payload_contract that day) and nothing retried — a
single flap failed the whole read.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest

from vaultbeat_mcp_local.client import VaultbeatCloudClient, VaultbeatCloudError


def make_client(handler: Any) -> VaultbeatCloudClient:
    client = VaultbeatCloudClient(
        "https://example.test/functions/v1",
        transport=httpx.MockTransport(handler),
    )
    # Retries sleep between attempts; zero it so the suite stays fast.
    client.RETRY_BACKOFF_SECONDS = 0.0  # type: ignore[misc]
    return client


class FlakyThenOK:
    """Fails the first `failures` attempts, then answers 200."""

    def __init__(self, failures: int, exception: Exception | None = None, status: int = 503):
        self.failures = failures
        self.exception = exception
        self.status = status
        self.attempts = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.attempts += 1
        if self.attempts <= self.failures:
            if self.exception is not None:
                raise self.exception
            return httpx.Response(self.status, json={"error": "gateway"})
        return httpx.Response(200, json={"envelopes": []})


def test_proxy_error_is_retried_and_the_call_succeeds() -> None:
    handler = FlakyThenOK(1, exception=httpx.ProxyError("503 Service Unavailable"))
    client = make_client(handler)
    result = asyncio.run(client.sync("token"))
    assert result == []
    assert handler.attempts == 2


def test_gateway_503_is_retried_and_the_call_succeeds() -> None:
    handler = FlakyThenOK(2, status=503)
    client = make_client(handler)
    result = asyncio.run(client.sync("token"))
    assert result == []
    assert handler.attempts == 3  # two retries spent, third answered


def test_exhausted_transient_retries_surface_as_cloud_error() -> None:
    handler = FlakyThenOK(99, exception=httpx.ProxyError("503 Service Unavailable"))
    client = make_client(handler)
    with pytest.raises(VaultbeatCloudError) as exc_info:
        asyncio.run(client.sync("token"))
    assert "3 attempts" in str(exc_info.value)
    assert handler.attempts == 3


def test_exhausted_5xx_retries_return_the_last_response_as_a_normal_error() -> None:
    # A 503 on the FINAL attempt is not swallowed into the retry-exhausted
    # message — it flows through _decode_response like any other HTTP error,
    # keeping the error= / request_id= detail contract.
    handler = FlakyThenOK(99, status=503)
    client = make_client(handler)
    with pytest.raises(VaultbeatCloudError) as exc_info:
        asyncio.run(client.sync("token"))
    assert "HTTP 503" in str(exc_info.value)
    assert handler.attempts == 3


def test_a_plain_4xx_is_never_retried() -> None:
    # 401/400/409 are OUR answers (auth, validation, conflicts) — retrying
    # them cannot change the outcome and would triple every real failure.
    handler = FlakyThenOK(99, status=401)
    client = make_client(handler)
    with pytest.raises(VaultbeatCloudError):
        asyncio.run(client.sync("token"))
    assert handler.attempts == 1


def test_read_timeout_is_never_retried() -> None:
    # A timeout means the full 90s read budget was spent; paying it up to
    # three times on an edge cold start is worse than failing honestly.
    handler = FlakyThenOK(99, exception=httpx.ReadTimeout("read budget spent"))
    client = make_client(handler)
    with pytest.raises(httpx.ReadTimeout):
        asyncio.run(client.sync("token"))
    assert handler.attempts == 1


def test_write_path_retries_too() -> None:
    # The shared _request helper serves the write endpoints as well; a POST
    # retry is safe because every write is an idempotent upsert by blob id.
    handler = FlakyThenOK(1, exception=httpx.ProxyError("flap"))
    client = make_client(handler)
    result = asyncio.run(
        client.write_note_blob("token", blob={"id": "note-x"}, envelopes=[])
    )
    assert result.get("envelopes") == [] or isinstance(result, dict)
    assert handler.attempts == 2
