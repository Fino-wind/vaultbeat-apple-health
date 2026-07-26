from __future__ import annotations

import asyncio
import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from vaultbeat_mcp_local.mcp_server import (
    StaticBearerASGIMiddleware,
    _is_loopback,
    _serve_streamable_http,
    run_mcp_server,
)
from vaultbeat_mcp_local.store import ConfigStore


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
    ["0.0.0.0", "::", "192.168.1.10", "10.0.0.1", "example.com", "vaultbeat.local"],
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

    assert "vaultbeat_sync_sleep" in registered
    assert "get_water_intake" in registered
    assert "get_menstrual_cycle" in registered


def test_get_hrv_falls_back_to_raw_when_hourly_is_empty(
    monkeypatch: Any, tmp_path: Path
) -> None:
    """An app too old to write hourly buckets must still get its HRV.

    `hrv_hourly` only exists from iOS build 77 (2026-07-22). Serving the empty
    hourly result to an App Store user on 1.2.0 would report "no HRV data" for
    an account full of raw HRV — a regression against 0.1.2, which read raw
    unconditionally. App and MCP versions drift permanently (users update them
    independently), so the aggregate degrades to the kind it aggregates.
    """
    tools: dict[str, Any] = {}

    class FakeFastMCP:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def tool(self) -> Any:
            def decorator(function: Any) -> Any:
                tools[function.__name__] = function
                return function

            return decorator

        def run(self, **kwargs: Any) -> None:
            pass

    fake_module = types.ModuleType("mcp.server.fastmcp")
    fake_module.FastMCP = FakeFastMCP
    monkeypatch.setitem(sys.modules, "mcp.server.fastmcp", fake_module)

    hourly_empty = {"records": [], "count": 0, "average_sdnn_ms": None}
    raw_present = {
        "records": [{"sdnn_ms": 42.0}],
        "count": 1,
        "average_sdnn_ms": 42.0,
    }

    async def fake_hourly(**_: Any) -> dict[str, Any]:
        return dict(hourly_empty)

    async def fake_raw(**_: Any) -> dict[str, Any]:
        return dict(raw_present)

    monkeypatch.setattr(
        "vaultbeat_mcp_local.service.VaultbeatLocalService.hrv_hourly_records",
        lambda self, **kw: fake_hourly(**kw),
    )
    monkeypatch.setattr(
        "vaultbeat_mcp_local.service.VaultbeatLocalService.hrv_records",
        lambda self, **kw: fake_raw(**kw),
    )

    run_mcp_server(ConfigStore(tmp_path / "config.json"), transport="stdio")
    get_hrv = tools["get_hrv"]

    # Default (hourly) with no hourly data → raw, clearly labelled.
    result = asyncio.run(get_hrv())
    assert result["count"] == 1
    assert result["granularity"] == "raw"
    assert "2026-07-22" in result["granularity_note"]

    # Explicit raw is untouched by the fallback path.
    result = asyncio.run(get_hrv(granularity="raw"))
    assert result["count"] == 1
    assert "granularity_note" not in result


def test_get_hrv_prefers_hourly_when_available(monkeypatch: Any, tmp_path: Path) -> None:
    """The fallback must not fire when hourly data exists (no silent downgrade)."""
    tools: dict[str, Any] = {}

    class FakeFastMCP:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def tool(self) -> Any:
            def decorator(function: Any) -> Any:
                tools[function.__name__] = function
                return function

            return decorator

        def run(self, **kwargs: Any) -> None:
            pass

    fake_module = types.ModuleType("mcp.server.fastmcp")
    fake_module.FastMCP = FakeFastMCP
    monkeypatch.setitem(sys.modules, "mcp.server.fastmcp", fake_module)

    async def fake_hourly(**_: Any) -> dict[str, Any]:
        return {"records": [{"avg_sdnn_ms": 40.0}], "count": 1, "average_sdnn_ms": 40.0}

    async def fake_raw(**_: Any) -> dict[str, Any]:
        raise AssertionError("raw must not be queried when hourly has data")

    monkeypatch.setattr(
        "vaultbeat_mcp_local.service.VaultbeatLocalService.hrv_hourly_records",
        lambda self, **kw: fake_hourly(**kw),
    )
    monkeypatch.setattr(
        "vaultbeat_mcp_local.service.VaultbeatLocalService.hrv_records",
        lambda self, **kw: fake_raw(**kw),
    )

    run_mcp_server(ConfigStore(tmp_path / "config.json"), transport="stdio")
    result = asyncio.run(tools["get_hrv"]())

    assert result["count"] == 1
    assert "granularity_note" not in result


def _capture_tools(monkeypatch: Any) -> dict[str, Any]:
    """Register the MCP surface against a fake FastMCP and return the tools."""
    tools: dict[str, Any] = {}

    class FakeFastMCP:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def tool(self) -> Any:
            def decorator(function: Any) -> Any:
                tools[function.__name__] = function
                return function

            return decorator

        def run(self, **kwargs: Any) -> None:
            pass

    fake_module = types.ModuleType("mcp.server.fastmcp")
    fake_module.FastMCP = FakeFastMCP
    monkeypatch.setitem(sys.modules, "mcp.server.fastmcp", fake_module)
    return tools


def test_empty_newer_kind_explains_itself(monkeypatch: Any, tmp_path: Path) -> None:
    """An empty result for a post-1.2.0 kind must carry its own reason.

    Otherwise `{"sessions": []}` is indistinguishable from "never trained" and
    from "install is broken", and the agent has no way to tell the user which.
    """
    tools = _capture_tools(monkeypatch)

    async def empty_strength(**_: Any) -> dict[str, Any]:
        return {"sessions": [], "errors": []}

    monkeypatch.setattr(
        "vaultbeat_mcp_local.service.VaultbeatLocalService.strength_summary",
        lambda self, **kw: empty_strength(**kw),
    )
    run_mcp_server(ConfigStore(tmp_path / "config.json"), transport="stdio")

    result = asyncio.run(tools["get_strength_log"]())
    assert result["sessions"] == []
    assert "2026-07-19" in result["hint"]
    assert "vaultbeat_doctor" in result["hint"]
    # The hint must offer the permission cause and its recovery path, not just
    # "update your app". A HealthKit READ denial is invisible to the app — it
    # reports success and delivers nothing — so an agent relaying this has no
    # other way to suggest the one action that fixes it.
    assert "Apple Health access" in result["hint"]
    assert "not been recorded yet" in result["hint"]
    # Add-only: the original keys survive untouched.
    assert result["errors"] == []


def test_populated_result_is_left_completely_alone(monkeypatch: Any, tmp_path: Path) -> None:
    """With data present, the response must be byte-identical to the service's.

    Payload shape is the contract layer no server can validate, so the
    annotation helper must be provably inert on the happy path.
    """
    tools = _capture_tools(monkeypatch)
    payload = {"sessions": [{"exercises": [], "total_volume_kg": 100}], "errors": []}

    async def full_strength(**_: Any) -> dict[str, Any]:
        return dict(payload)

    monkeypatch.setattr(
        "vaultbeat_mcp_local.service.VaultbeatLocalService.strength_summary",
        lambda self, **kw: full_strength(**kw),
    )
    run_mcp_server(ConfigStore(tmp_path / "config.json"), transport="stdio")

    result = asyncio.run(tools["get_strength_log"]())
    assert result == payload
    assert "hint" not in result


def test_vaultbeat_doctor_is_exposed_as_a_tool(monkeypatch: Any, tmp_path: Path) -> None:
    """The diagnostic must be reachable by the agent, not just from a terminal.

    It lived only in the CLI until 2026-07-25, so an agent facing an empty
    result had no way to find out why — the answer existed in a room it could
    not enter.
    """
    tools = _capture_tools(monkeypatch)

    async def fake_doctor(self: Any) -> dict[str, Any]:
        return {"ok": True, "checks": [], "capabilities": {"available": True}}

    monkeypatch.setattr(
        "vaultbeat_mcp_local.service.VaultbeatLocalService.doctor", fake_doctor
    )
    run_mcp_server(ConfigStore(tmp_path / "config.json"), transport="stdio")

    assert "vaultbeat_doctor" in tools
    result = asyncio.run(tools["vaultbeat_doctor"]())
    assert result["capabilities"]["available"] is True
