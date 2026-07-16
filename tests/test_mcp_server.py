from __future__ import annotations

import asyncio
import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from tether_mcp_local.mcp_server import (
    StaticBearerASGIMiddleware,
    _is_loopback,
    _serve_streamable_http,
    run_mcp_server,
)
from tether_mcp_local.store import ConfigStore


def _drive_http(app: Any, scope: dict[str, Any]) -> list[dict[str, Any]]:
    """Run an ASGI app through one request, returning the messages it sends."""

    sent: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    asyncio.run(app(scope, receive, send))
    return sent


class _RecordingApp:
    """Inner ASGI app that records the scope types it was actually reached for."""

    def __init__(self) -> None:
        self.scopes: list[str] = []

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        self.scopes.append(scope.get("type"))
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})


def test_middleware_rejects_missing_authorization() -> None:
    inner = _RecordingApp()
    mw = StaticBearerASGIMiddleware(inner, "s3cret-token")

    sent = _drive_http(mw, {"type": "http", "headers": []})

    assert inner.scopes == []  # inner is never reached
    assert sent[0]["type"] == "http.response.start"
    assert sent[0]["status"] == 401
    body = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    assert json.loads(body) == {"error": "unauthorized"}


def test_middleware_rejects_wrong_token() -> None:
    inner = _RecordingApp()
    mw = StaticBearerASGIMiddleware(inner, "s3cret-token")

    sent = _drive_http(mw, {"type": "http", "headers": [(b"authorization", b"Bearer wrong")]})

    assert inner.scopes == []
    assert sent[0]["status"] == 401


def test_middleware_allows_correct_token() -> None:
    inner = _RecordingApp()
    mw = StaticBearerASGIMiddleware(inner, "s3cret-token")

    sent = _drive_http(
        mw, {"type": "http", "headers": [(b"authorization", b"Bearer s3cret-token")]}
    )

    assert inner.scopes == ["http"]  # reached inner
    assert sent[0]["status"] == 200


def test_middleware_forwards_non_http_scopes() -> None:
    # lifespan must pass through untouched so the MCP session manager still starts.
    inner = _RecordingApp()
    mw = StaticBearerASGIMiddleware(inner, "s3cret-token")

    async def receive() -> dict[str, Any]:
        return {"type": "lifespan.startup"}

    async def send(message: dict[str, Any]) -> None:
        return None

    asyncio.run(mw({"type": "lifespan"}, receive, send))

    assert inner.scopes == ["lifespan"]  # forwarded despite no auth header


@pytest.mark.parametrize(
    "host",
    ["127.0.0.1", "127.0.0.5", "::1", "localhost", "LOCALHOST", "  127.0.0.1  "],
)
def test_is_loopback_true(host: str) -> None:
    assert _is_loopback(host) is True


@pytest.mark.parametrize(
    "host",
    ["0.0.0.0", "::", "192.168.1.10", "10.0.0.1", "example.com", "tether.local"],
)
def test_is_loopback_false(host: str) -> None:
    assert _is_loopback(host) is False


def test_serve_streamable_http_refuses_non_loopback_without_token() -> None:
    with pytest.raises(RuntimeError, match="generate-token"):
        _serve_streamable_http(object(), host="0.0.0.0", port=8000, token=None, allow_remote=False)


def test_serve_streamable_http_requires_allow_remote_for_non_loopback() -> None:
    with pytest.raises(RuntimeError, match="allow-remote"):
        _serve_streamable_http(
            object(), host="0.0.0.0", port=8000, token="a-token", allow_remote=False
        )


class _FakeMCPApp:
    def streamable_http_app(self) -> str:
        return "INNER_ASGI"


def test_serve_streamable_http_loopback_serves_unwrapped(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}
    import uvicorn

    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: captured.update(app=app, kw=kw))

    _serve_streamable_http(
        _FakeMCPApp(), host="127.0.0.1", port=8000, token=None, allow_remote=False
    )

    assert captured["app"] == "INNER_ASGI"  # no token => unwrapped inner app
    assert captured["kw"]["host"] == "127.0.0.1"
    assert captured["kw"]["port"] == 8000


def test_serve_streamable_http_wraps_with_bearer_when_token(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}
    import uvicorn

    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: captured.update(app=app))

    _serve_streamable_http(
        _FakeMCPApp(), host="0.0.0.0", port=8000, token="a-token", allow_remote=True
    )

    assert isinstance(captured["app"], StaticBearerASGIMiddleware)


def test_run_mcp_server_stdio_uses_mcp_run(monkeypatch: Any, tmp_path: Path) -> None:
    captured: dict[str, Any] = {}

    class FakeFastMCP:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def tool(self) -> Any:
            def decorator(function: Any) -> Any:
                return function

            return decorator

        def streamable_http_app(self) -> str:
            captured["http_app_called"] = True
            return "X"

        def run(self, **kwargs: Any) -> None:
            captured["run_kwargs"] = kwargs

    fake_module = types.ModuleType("mcp.server.fastmcp")
    fake_module.FastMCP = FakeFastMCP
    monkeypatch.setitem(sys.modules, "mcp.server.fastmcp", fake_module)

    run_mcp_server(ConfigStore(tmp_path / "config.json"), transport="stdio")

    assert captured["run_kwargs"] == {"transport": "stdio"}
    assert "http_app_called" not in captured  # stdio path never touches the http surface


def test_run_mcp_server_registers_water_and_menstrual_tools(
    monkeypatch: Any, tmp_path: Path
) -> None:
    registered: list[str] = []

    class FakeFastMCP:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def tool(self) -> Any:
            def decorator(function: Any) -> Any:
                registered.append(function.__name__)
                return function

            return decorator

        def run(self, **kwargs: Any) -> None:
            pass

    fake_module = types.ModuleType("mcp.server.fastmcp")
    fake_module.FastMCP = FakeFastMCP
    monkeypatch.setitem(sys.modules, "mcp.server.fastmcp", fake_module)

    run_mcp_server(ConfigStore(tmp_path / "config.json"), transport="stdio")

    assert "tether_sync_sleep" in registered
    assert "get_partner_water_intake" in registered
    assert "get_partner_menstrual_cycle" in registered
