from __future__ import annotations

import asyncio
import json
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
    def streamable_http_app(self, **kwargs: Any) -> str:
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

        def tool(self, *args: Any, **kwargs: Any) -> Any:
            def decorator(function: Any) -> Any:
                return function

            return decorator

        def streamable_http_app(self, **kwargs: Any) -> str:
            captured["http_app_called"] = True
            return "X"

        def run(self, **kwargs: Any) -> None:
            captured["run_kwargs"] = kwargs

        def add_prompt(self, prompt: Any) -> None:
            # Prompts are registered on the same object as tools. A fake that
            # cannot hold one is no longer a fake of FastMCP — it fails with an
            # AttributeError from inside `register_prompts`, which reads as a
            # bug in the code under test. What prompts CONTAIN is asserted
            # against the real server in `test_prompts.py`; here they only have
            # to be accepted.
            pass

    # Patch the ATTRIBUTE, not the whole module: `run_mcp_server` reads
    # `FastMCP` at call time, and replacing `mcp.server.fastmcp` wholesale left
    # it a non-package, so any sibling the code imports (`prompts.base`) became
    # unimportable while the stub was in place.
    monkeypatch.setattr("mcp.server.mcpserver.MCPServer", FakeFastMCP)

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

        def tool(self, *args: Any, **kwargs: Any) -> Any:
            def decorator(function: Any) -> Any:
                registered.append(function.__name__)
                return function

            return decorator

        def run(self, **kwargs: Any) -> None:
            pass

        def add_prompt(self, prompt: Any) -> None:
            # Prompts are registered on the same object as tools. A fake that
            # cannot hold one is no longer a fake of FastMCP — it fails with an
            # AttributeError from inside `register_prompts`, which reads as a
            # bug in the code under test. What prompts CONTAIN is asserted
            # against the real server in `test_prompts.py`; here they only have
            # to be accepted.
            pass

    # Patch the ATTRIBUTE, not the whole module: `run_mcp_server` reads
    # `FastMCP` at call time, and replacing `mcp.server.fastmcp` wholesale left
    # it a non-package, so any sibling the code imports (`prompts.base`) became
    # unimportable while the stub was in place.
    monkeypatch.setattr("mcp.server.mcpserver.MCPServer", FakeFastMCP)

    run_mcp_server(ConfigStore(tmp_path / "config.json"), transport="stdio")

    assert "get_sleep_nights" in registered
    assert "vaultbeat_sync_sleep" not in registered, "retired in 0.9.0: get_sleep_nights reads, fresh=True syncs"
    assert "get_metric" in registered
    assert "get_intraday" in registered
    assert "get_menstrual_cycle" in registered
    # Folded into `get_metric` on 2026-09-23. Asserted absent so a merge that
    # re-registers one of them (a stale branch, a copy-paste) cannot sit beside
    # the generic tool and give an agent two answers for one number.
    for retired in (
        "get_water_intake", "get_weight_trend", "get_activity", "get_resting_hr",
        "get_hrv", "get_wrist_temp", "get_basal_energy", "get_total_energy_burned",
        "get_vo2max", "get_mindfulness",
    ):
        assert retired not in registered, retired


def test_get_intraday_hrv_falls_back_to_raw_when_hourly_is_empty(
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

        def tool(self, *args: Any, **kwargs: Any) -> Any:
            def decorator(function: Any) -> Any:
                tools[function.__name__] = function
                return function

            return decorator

        def run(self, **kwargs: Any) -> None:
            pass

        def add_prompt(self, prompt: Any) -> None:
            # Prompts are registered on the same object as tools. A fake that
            # cannot hold one is no longer a fake of FastMCP — it fails with an
            # AttributeError from inside `register_prompts`, which reads as a
            # bug in the code under test. What prompts CONTAIN is asserted
            # against the real server in `test_prompts.py`; here they only have
            # to be accepted.
            pass

    # Patch the ATTRIBUTE, not the whole module: `run_mcp_server` reads
    # `FastMCP` at call time, and replacing `mcp.server.fastmcp` wholesale left
    # it a non-package, so any sibling the code imports (`prompts.base`) became
    # unimportable while the stub was in place.
    monkeypatch.setattr("mcp.server.mcpserver.MCPServer", FakeFastMCP)

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
    get_intraday = tools["get_intraday"]

    # Default (hourly) with no hourly data → raw, clearly labelled.
    result = asyncio.run(get_intraday())
    assert result["count"] == 1
    assert result["granularity"] == "raw"
    assert "2026-07-22" in result["granularity_note"]

    # Explicit raw is untouched by the fallback path.
    result = asyncio.run(get_intraday(granularity="raw"))
    assert result["count"] == 1
    assert "granularity_note" not in result


def test_get_intraday_hrv_prefers_hourly_when_available(monkeypatch: Any, tmp_path: Path) -> None:
    """The fallback must not fire when hourly data exists (no silent downgrade)."""
    tools: dict[str, Any] = {}

    class FakeFastMCP:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def tool(self, *args: Any, **kwargs: Any) -> Any:
            def decorator(function: Any) -> Any:
                tools[function.__name__] = function
                return function

            return decorator

        def run(self, **kwargs: Any) -> None:
            pass

        def add_prompt(self, prompt: Any) -> None:
            # Prompts are registered on the same object as tools. A fake that
            # cannot hold one is no longer a fake of FastMCP — it fails with an
            # AttributeError from inside `register_prompts`, which reads as a
            # bug in the code under test. What prompts CONTAIN is asserted
            # against the real server in `test_prompts.py`; here they only have
            # to be accepted.
            pass

    # Patch the ATTRIBUTE, not the whole module: `run_mcp_server` reads
    # `FastMCP` at call time, and replacing `mcp.server.fastmcp` wholesale left
    # it a non-package, so any sibling the code imports (`prompts.base`) became
    # unimportable while the stub was in place.
    monkeypatch.setattr("mcp.server.mcpserver.MCPServer", FakeFastMCP)

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
    result = asyncio.run(tools["get_intraday"]())

    assert result["count"] == 1
    assert "granularity_note" not in result


def _capture_tools(monkeypatch: Any) -> dict[str, Any]:
    """Register the MCP surface against a fake FastMCP and return the tools."""
    tools: dict[str, Any] = {}

    class FakeFastMCP:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def tool(self, *args: Any, **kwargs: Any) -> Any:
            def decorator(function: Any) -> Any:
                tools[function.__name__] = function
                return function

            return decorator

        def run(self, **kwargs: Any) -> None:
            pass

        def add_prompt(self, prompt: Any) -> None:
            # Prompts are registered on the same object as tools. A fake that
            # cannot hold one is no longer a fake of FastMCP — it fails with an
            # AttributeError from inside `register_prompts`, which reads as a
            # bug in the code under test. What prompts CONTAIN is asserted
            # against the real server in `test_prompts.py`; here they only have
            # to be accepted.
            pass

    # Patch the ATTRIBUTE, not the whole module: `run_mcp_server` reads
    # `FastMCP` at call time, and replacing `mcp.server.fastmcp` wholesale left
    # it a non-package, so any sibling the code imports (`prompts.base`) became
    # unimportable while the stub was in place.
    monkeypatch.setattr("mcp.server.mcpserver.MCPServer", FakeFastMCP)
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


def test_an_empty_window_is_not_reported_as_an_empty_account(monkeypatch: Any, tmp_path: Path) -> None:
    """Review R3 (2026-10-03): a food window with nothing logged said the account had no food.

    The account had months of food; the `since` / `until` window simply held
    none of it. And a refused call (`invalid_since`) got the same sentence
    stapled to its error. Both are Invariant 57: one empty value, several
    causes, and the hint named the wrong one.
    """
    tools = _capture_tools(monkeypatch)
    replies: list[dict[str, Any]] = [
        {"days": [], "coverage": {"days_covered": 0, "total_available": 41,
                                  "oldest_available": "2026-07-20"}, "errors": []},
        {"error": "invalid_since", "requested": "8月1日", "message": "…"},
    ]

    async def food(**_: Any) -> dict[str, Any]:
        return replies.pop(0)

    monkeypatch.setattr(
        "vaultbeat_mcp_local.service.VaultbeatLocalService.food_summary",
        lambda self, **kw: food(**kw),
    )
    run_mcp_server(ConfigStore(tmp_path / "config.json"), transport="stdio")

    windowed = asyncio.run(tools["get_food_log"](since="2026-09-01", until="2026-09-14"))
    assert "on this account" not in windowed["hint"]
    assert "outside it" in windowed["hint"] and "2026-07-20" in windowed["hint"]

    refused = asyncio.run(tools["get_food_log"](since="8月1日"))
    assert refused["error"] == "invalid_since"
    assert "hint" not in refused


def test_an_empty_since_keeps_the_two_week_default(monkeypatch: Any, tmp_path: Path) -> None:
    """Review R10: `since=""` skipped the default and read the whole food history."""
    tools = _capture_tools(monkeypatch)
    calls: list[dict[str, Any]] = []

    async def food(**kw: Any) -> dict[str, Any]:
        calls.append(kw)
        return {"days": [{"local_date": "2026-10-01"}], "errors": []}

    monkeypatch.setattr(
        "vaultbeat_mcp_local.service.VaultbeatLocalService.food_summary",
        lambda self, **kw: food(**kw),
    )
    run_mcp_server(ConfigStore(tmp_path / "config.json"), transport="stdio")

    asyncio.run(tools["get_food_log"](since="", until=""))
    assert calls[-1]["limit"] == 14
    assert calls[-1]["since"] is None and calls[-1]["until"] is None


def test_a_since_is_not_cut_by_the_default_limit(monkeypatch: Any, tmp_path: Path) -> None:
    """Release gate round 4 on #9: `get_sleep_nights(since="2025-11-01")` came back
    starting in June, because the default 120 still applied — while the tool told
    agents to use `since` rather than guess a limit. Same rule for sleep detail."""
    tools = _capture_tools(monkeypatch)
    calls: dict[str, list[dict[str, Any]]] = {"nights": [], "detail": []}

    async def nights(**kw: Any) -> dict[str, Any]:
        calls["nights"].append(kw)
        return {"nights": [], "count": 0, "errors": []}

    async def detail(**kw: Any) -> dict[str, Any]:
        calls["detail"].append(kw)
        return {"nights": [], "count": 0, "errors": []}

    monkeypatch.setattr("vaultbeat_mcp_local.service.VaultbeatLocalService.sleep_nights",
                        lambda self, **kw: nights(**kw))
    monkeypatch.setattr("vaultbeat_mcp_local.service.VaultbeatLocalService.sleep_detail_records",
                        lambda self, **kw: detail(**kw))
    run_mcp_server(ConfigStore(tmp_path / "config.json"), transport="stdio")

    asyncio.run(tools["get_sleep_nights"]())
    asyncio.run(tools["get_sleep_nights"](since="2025-11-01"))
    asyncio.run(tools["get_sleep_nights"](since="2025-11-01", limit=10))
    asyncio.run(tools["get_sleep_nights"](since=""))
    assert [c["limit"] for c in calls["nights"]] == [120, None, 10, 120]

    asyncio.run(tools["get_sleep_detail"]())
    asyncio.run(tools["get_sleep_detail"](since="2026-09-14", until="2026-09-14"))
    asyncio.run(tools["get_sleep_detail"](until="2026-09-14", limit=3))
    assert [c["limit"] for c in calls["detail"]] == [2, None, 3]


def test_narrow_hints_name_only_real_parameters_and_values(monkeypatch: Any, tmp_path: Path) -> None:
    """A `result_too_large` hint is followed to the letter, so every parameter it
    names must exist on that tool and every value it suggests must be accepted.
    `aggregation="mean"` shipped in one (the valid value is `avg`) and an agent
    obeying it got an error instead of a smaller answer (`e34b8d6`)."""
    import inspect
    import re

    from vaultbeat_mcp_local.mcp_server import _NARROW_HINTS
    from vaultbeat_mcp_local.service import _AGGREGATIONS, _GRANULARITIES

    tools = _capture_tools(monkeypatch)
    run_mcp_server(ConfigStore(tmp_path / "config.json"), transport="stdio")
    allowed_values = {"aggregation": set(_AGGREGATIONS), "granularity": set(_GRANULARITIES)}

    for tool_name, hint in _NARROW_HINTS.items():
        assert tool_name in tools, f"{tool_name} is not a registered tool"
        params = set(inspect.signature(tools[tool_name]).parameters)
        for name in re.findall(r"`([a-z_]+)(?:=|`)", hint):
            assert name in params, f"{tool_name}'s hint names `{name}`, which it does not take"
        for name, value in re.findall(r'`([a-z_]+)="([a-z]+)"', hint):
            assert value in allowed_values[name], f"{tool_name}: {name}={value!r} is refused"
        for value in re.findall(r'`"([a-z]+)"`', hint):  # a bare alternative value: `"month"`
            assert any(value in values for values in allowed_values.values()), value


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


# ── Tool metadata: title + annotations ──────────────────────────────────────


def _capture_tool_meta(monkeypatch: Any) -> dict[str, dict[str, Any]]:
    """Register the MCP surface and return {tool_name: {title, annotations}}.

    Distinct from `_capture_tools`, which throws the kwargs away — here the
    kwargs ARE the thing under test.
    """
    meta: dict[str, dict[str, Any]] = {}

    class FakeFastMCP:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            meta.setdefault("__server__", {})["name"] = args[0] if args else None

        def tool(self, *args: Any, **kwargs: Any) -> Any:
            def decorator(function: Any) -> Any:
                meta[function.__name__] = dict(kwargs)
                return function

            return decorator

        def run(self, **kwargs: Any) -> None:
            pass

        def add_prompt(self, prompt: Any) -> None:
            # Prompts are registered on the same object as tools. A fake that
            # cannot hold one is no longer a fake of FastMCP — it fails with an
            # AttributeError from inside `register_prompts`, which reads as a
            # bug in the code under test. What prompts CONTAIN is asserted
            # against the real server in `test_prompts.py`; here they only have
            # to be accepted.
            pass

    # Patch the ATTRIBUTE, not the whole module: `run_mcp_server` reads
    # `FastMCP` at call time, and replacing `mcp.server.fastmcp` wholesale left
    # it a non-package, so any sibling the code imports (`prompts.base`) became
    # unimportable while the stub was in place.
    monkeypatch.setattr("mcp.server.mcpserver.MCPServer", FakeFastMCP)
    return meta


def test_every_tool_carries_a_title_and_annotations(monkeypatch: Any, tmp_path: Path) -> None:
    """Annotations travel in `list_tools`, so a client decides whether to prompt
    for confirmation BEFORE running anything. A tool with none defaults, per the
    MCP spec, to the most alarming reading (`destructiveHint` defaults to true) —
    so an unannotated read tool is not merely undescribed, it is mis-described.
    """
    meta = _capture_tool_meta(monkeypatch)
    run_mcp_server(ConfigStore(tmp_path / "config.json"), transport="stdio")

    tools = {k: v for k, v in meta.items() if not k.startswith("__")}
    missing_title = sorted(k for k, v in tools.items() if not v.get("title"))
    missing_annotations = sorted(k for k, v in tools.items() if v.get("annotations") is None)

    assert missing_title == []
    assert missing_annotations == []
    # Titles are what a human picks from in a tool list; two identical ones make
    # the pair unpickable. The sleep pair is the live example — one goes deep on
    # a night, the other spans weeks.
    titles = [v["title"] for v in tools.values()]
    assert len(titles) == len(set(titles))


def test_read_tools_are_annotated_read_only(monkeypatch: Any, tmp_path: Path) -> None:
    meta = _capture_tool_meta(monkeypatch)
    run_mcp_server(ConfigStore(tmp_path / "config.json"), transport="stdio")

    for name in ("vaultbeat_doctor", "get_sleep_detail", "get_food_log", "get_metric", "get_intraday"):
        annotations = meta[name]["annotations"]
        assert annotations.read_only_hint is True, name
        assert annotations.destructive_hint is False, name


def test_only_the_doctor_reaches_outside_this_system(monkeypatch: Any, tmp_path: Path) -> None:
    """`openWorldHint` says a tool talks to an unbounded external world.

    Everything else here speaks only to the owner's own Supabase project, which
    is a closed system from the caller's point of view; `vaultbeat_doctor` is the
    one tool that asks PyPI a question. If a second tool ever needs this, that is
    a fact worth noticing rather than a line to relax.
    """
    meta = _capture_tool_meta(monkeypatch)
    run_mcp_server(ConfigStore(tmp_path / "config.json"), transport="stdio")

    open_world = sorted(
        name
        for name, kwargs in meta.items()
        if not name.startswith("__") and kwargs["annotations"].open_world_hint
    )
    assert open_world == ["vaultbeat_doctor"]


#: Arguments for the read tools that cannot be called bare.
#:
#: Everything else takes only defaulted parameters and is invoked with none, so
#: this table stays empty for new tools by default — a tool lands here only
#: because it has a REQUIRED argument, which for the analysis trio is deliberate
#: (there is no sensible default quantity to trend, and defaulting one would let
#: a typo'd series silently return a different metric).
_READ_TOOL_ARGS: dict[str, dict[str, Any]] = {
    "get_metric_trend": {"series": "resting_hr", "days": 14},
    "compare_metric_periods": {"series": "resting_hr", "days": 7},
    "correlate_metric_series": {
        "series_a": "resting_hr",
        "series_b": "sleep_minutes",
        "days": 14,
    },
    "get_metric": {"days": 7},
}


def test_every_tool_that_returns_coverage_says_so_in_its_docstring(
    monkeypatch: Any, tmp_path: Path
) -> None:
    """A docstring is the tool's API to the agent, and `coverage` is invisible
    without one: an agent only learns the field exists by reading a response,
    which is after it has already decided what to quote (Invariant 64).

    Rule-based on both sides, so a new read tool cannot ship half the contract:
    returning `coverage` without naming it, or naming it without returning it.
    Demo mode supplies real-shaped results with no network, no pairing and no
    key; the doctor is skipped because it is the one tool that leaves this
    system, and writes are skipped because they need arguments.
    """
    monkeypatch.setenv("VAULTBEAT_DEMO", "1")
    captured: dict[str, tuple[Any, Any]] = {}

    class FakeFastMCP:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def tool(self, *args: Any, **kwargs: Any) -> Any:
            def decorator(function: Any) -> Any:
                captured[function.__name__] = (function, kwargs.get("annotations"))
                return function

            return decorator

        def run(self, **kwargs: Any) -> None:
            pass

        def add_prompt(self, prompt: Any) -> None:
            pass

    monkeypatch.setattr("mcp.server.mcpserver.MCPServer", FakeFastMCP)
    run_mcp_server(ConfigStore(tmp_path / "config.json"), transport="stdio")

    covered: list[str] = []
    for name, (function, hints) in captured.items():
        if not hints.read_only_hint or hints.open_world_hint:
            continue
        result = function(**_READ_TOOL_ARGS.get(name, {}))
        if asyncio.iscoroutine(result):
            result = asyncio.run(result)
        # `get_metric` carries one block PER SERIES rather than one at the top:
        # series in one reply genuinely differ in extent, and a single block
        # would have to lie about all but one of them.
        per_series = result.get("metrics")
        returns_coverage = "coverage" in result or (
            isinstance(per_series, list)
            and bool(per_series)
            and all("coverage" in m for m in per_series if "error" not in m)
        )
        documents_coverage = "coverage.days_covered" in (function.__doc__ or "")
        assert returns_coverage == documents_coverage, (
            f"{name}: returns coverage={returns_coverage}, documents it={documents_coverage}"
        )
        if returns_coverage:
            covered.append(name)

    # Sanity that demo mode actually produced results rather than the loop
    # trivially agreeing on "neither": the two ends of the read surface.
    assert "get_sleep_nights" in covered
    assert "get_metric" in covered
    assert "vaultbeat_doctor" not in covered


def test_append_tools_are_annotated_non_destructive_and_entry_tools_are_not(
    monkeypatch: Any, tmp_path: Path
) -> None:
    """The whole point of the split, in one assertion.

    `merge=True` and `log_food_append` do exactly the same thing — the
    difference is that an ARGUMENT cannot be annotated. A tool with both modes
    has to be published as destructive (worst case), so the only way an agent
    can be told "this call cannot delete anything" is a separate tool.
    """
    meta = _capture_tool_meta(monkeypatch)
    run_mcp_server(ConfigStore(tmp_path / "config.json"), transport="stdio")

    for entry, append in (
        ("log_food_entry", "log_food_append"),
        ("log_strength_entry", "log_strength_append"),
        ("log_note", "log_note_append"),
    ):
        assert meta[entry]["annotations"].destructive_hint is True, entry
        assert meta[append]["annotations"].destructive_hint is False, append
        assert meta[append]["annotations"].read_only_hint is False, append


# ── Append tools ─────────────────────────────────────────────────────────────


def test_append_tools_are_registered(monkeypatch: Any, tmp_path: Path) -> None:
    tools = _capture_tools(monkeypatch)
    run_mcp_server(ConfigStore(tmp_path / "config.json"), transport="stdio")

    for name in ("log_food_append", "log_strength_append", "log_note_append"):
        assert name in tools


@pytest.mark.parametrize(
    ("append_tool", "service_method", "payload"),
    [
        ("log_food_append", "log_food_entry", {"date": "2026-08-20", "meals": [{"items": []}]}),
        (
            "log_strength_append",
            "log_strength_entry",
            {"date": "2026-08-20", "exercises": [{"name": "卧推", "sets": []}]},
        ),
        ("log_note_append", "log_note", {"text": "恶心", "kind": "general"}),
    ],
)
def test_append_tool_is_the_entry_tool_with_merge_true(
    monkeypatch: Any,
    tmp_path: Path,
    append_tool: str,
    service_method: str,
    payload: dict[str, Any],
) -> None:
    """The zero-duplication guarantee, mechanically.

    The append tools are one line each — a fixed call into the same service
    method — and nothing but this test stops someone growing a second
    implementation behind one of them. If that ever happens, the merge/note
    kwargs stop matching and this goes red.
    """
    tools = _capture_tools(monkeypatch)
    seen: dict[str, Any] = {}

    async def spy(self: Any, **kwargs: Any) -> dict[str, Any]:
        seen.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(
        f"vaultbeat_mcp_local.service.VaultbeatLocalService.{service_method}", spy
    )
    run_mcp_server(ConfigStore(tmp_path / "config.json"), transport="stdio")

    asyncio.run(tools[append_tool](**payload))

    assert seen["merge"] is True
    if service_method == "log_note":
        # log_note's text IS the note; there is no separate replace-only field
        # to withhold, so the append tool forwards no `note` kwarg at all.
        assert "note" not in seen
    else:
        assert seen["note"] is None
    for key, value in payload.items():
        assert seen[key] == value


def test_append_tools_cannot_touch_the_replace_only_note(
    monkeypatch: Any, tmp_path: Path
) -> None:
    """`note` is replace-only even under `merge=True` — `_resolve_note("x", old)`
    returns "x" and the old note is gone.

    So a tool published as `destructiveHint=False` must not expose it, or the one
    field that was actually lost in the 2026-07-27 incident ('腿日 + 腹肌') stays
    deletable through the tool whose whole promise is that nothing can be. Not
    exposing it makes that a property of the signature rather than of a docstring
    nobody has to obey.
    """
    import inspect

    tools = _capture_tools(monkeypatch)
    run_mcp_server(ConfigStore(tmp_path / "config.json"), transport="stdio")

    for name in ("log_food_append", "log_strength_append"):
        assert "note" not in inspect.signature(tools[name]).parameters, name


# ── Demo mode ────────────────────────────────────────────────────────────────


def test_demo_mode_watermarks_every_tool_result(monkeypatch: Any, tmp_path: Path) -> None:
    """A pasted tool result has to identify itself as synthetic.

    The description prefix covers the live session; this covers everything that
    outlives it — a screenshot, a bug report, a payload quoted into a document
    three weeks later. Neither one substitutes for the other.
    """
    monkeypatch.setenv("VAULTBEAT_DEMO", "1")
    tools = _capture_tools(monkeypatch)

    async def fake_food(self: Any, **_: Any) -> dict[str, Any]:
        return {"days": [{"date": "2026-08-18"}], "day_count": 1}

    monkeypatch.setattr(
        "vaultbeat_mcp_local.service.VaultbeatLocalService.food_summary", fake_food
    )
    run_mcp_server(ConfigStore(tmp_path / "config.json"), transport="stdio")

    result = asyncio.run(tools["get_food_log"]())
    assert result["demo_mode"] is True
    assert "SYNTHETIC" in result["demo_warning"]
    # Add-only: the real payload survives untouched.
    assert result["day_count"] == 1


def test_demo_mode_prefixes_every_tool_description(monkeypatch: Any, tmp_path: Path) -> None:
    """The warning has to be in the DESCRIPTION, not only in results.

    A description is read once and stays in context for the session; a
    per-result banner has to survive the agent paraphrasing three calls into one
    sentence. The prefix is what stops "your HRV was 42" being said about a
    person who does not exist.
    """
    monkeypatch.setenv("VAULTBEAT_DEMO", "1")
    tools = _capture_tools(monkeypatch)
    run_mcp_server(ConfigStore(tmp_path / "config.json"), transport="stdio")

    for name, function in tools.items():
        assert (function.__doc__ or "").startswith("⚠️ DEMO MODE"), name
        # The original text must still be there — the prefix prepends, it does
        # not replace, so everything the tool said about its own arguments and
        # its own traps survives.
        assert len(function.__doc__ or "") > 200, name


def test_demo_mode_keeps_the_real_signature(monkeypatch: Any, tmp_path: Path) -> None:
    """FastMCP builds each tool's JSON schema from `inspect.signature(fn)`.

    `functools.wraps` sets `__wrapped__`, which `signature` follows — without it
    every wrapped tool would publish `(*args, **kwargs)` and clients would lose
    every parameter. Cosmetic-looking, load-bearing.
    """
    import inspect

    monkeypatch.setenv("VAULTBEAT_DEMO", "1")
    tools = _capture_tools(monkeypatch)
    run_mcp_server(ConfigStore(tmp_path / "config.json"), transport="stdio")

    params = inspect.signature(tools["get_intraday"]).parameters
    assert sorted(params) == ["fresh", "granularity", "limit", "series"]


def test_demo_mode_renames_the_server(monkeypatch: Any, tmp_path: Path) -> None:
    """serverInfo is the one label a client shows without calling anything."""
    monkeypatch.setenv("VAULTBEAT_DEMO", "1")
    meta = _capture_tool_meta(monkeypatch)
    run_mcp_server(ConfigStore(tmp_path / "config.json"), transport="stdio")

    assert "DEMO" in meta["__server__"]["name"]


def test_without_demo_the_tools_carry_no_demo_marks(
    monkeypatch: Any, tmp_path: Path
) -> None:
    """Production must be free of every DEMO mark — absent, not inert.

    Until vb-016 this asserted the tools were not wrapped at all. That stopped
    being true when the trial-expiry `access_note` (a deliberate production
    feature) moved onto the same registration choke point, so the assertion
    narrowed to what still must hold: no demo description prefix, no demo
    fields in a real result, and the published signature stays the real
    function's — the access wrapper uses `functools.wraps`, which is what
    FastMCP's schema builder follows, so `__wrapped__` existing is now correct
    rather than a leak.
    """
    import inspect

    monkeypatch.delenv("VAULTBEAT_DEMO", raising=False)
    tools = _capture_tools(monkeypatch)
    run_mcp_server(ConfigStore(tmp_path / "config.json"), transport="stdio")

    for name, function in tools.items():
        assert not (function.__doc__ or "").startswith("⚠️ DEMO MODE"), name

    result = asyncio.run(tools["get_metric"](series="steps", days=1))
    assert "demo_mode" not in result
    assert "demo_warning" not in result

    params = inspect.signature(tools["get_intraday"]).parameters
    assert sorted(params) == ["fresh", "granularity", "limit", "series"]


def test_destructive_titles_name_their_consequence(monkeypatch: Any, tmp_path: Path) -> None:
    """A `destructiveHint` tool's title must say what it destroys.

    Titles and annotations travel together in `list_tools`, so a client shows
    the title next to the confirmation prompt the hint triggered — and a title
    that describes the tool as harmless is what produces reflexive approval.
    `vaultbeat_poll_binding` was called "Check pairing status" while its success
    branch replaces server_id, rotates the server token, rewrites the owner
    identity and can clear the decrypted cache; the annotation was right and the
    title pointed the other way.

    Written as a rule rather than as two string comparisons on purpose: it is
    the NEXT destructive tool that needs catching, and a test naming today's two
    cannot do that. House style is `verb (consequence)`, so the parenthesis is
    the machine-checkable part of it.
    """
    meta = _capture_tool_meta(monkeypatch)
    run_mcp_server(ConfigStore(tmp_path / "config.json"), transport="stdio")

    offenders = [
        (name, kwargs["title"])
        for name, kwargs in meta.items()
        if name != "__server__"
        and kwargs["annotations"].destructive_hint
        and "(" not in kwargs["title"]
    ]
    assert not offenders, (
        "destructive tools whose title does not name the consequence: "
        f"{offenders}. Use `verb (consequence)`, e.g. 'Log food (replaces day)'."
    )

    # The pairing pair that prompted this rule left the tool surface in 0.9.0;
    # the rule itself is what now guards the write tools.
    for retired in ("vaultbeat_status", "vaultbeat_start_binding", "vaultbeat_poll_binding"):
        assert retired not in meta, retired

def test_reads_default_to_me_and_partner_is_opt_in(monkeypatch: Any, tmp_path: Path) -> None:
    """Owner, 2026-09-23: 「默认读取自己的。自由加上伴侣才读伴侣的」.

    Demo data is a paired account (demo0001 = the account that paired, demo0002 =
    the partner, who shares sleep). A default read must be ONE person — the
    user — and `partner=True` must be the other one, never both.
    """
    import inspect

    monkeypatch.setenv("VAULTBEAT_DEMO", "1")
    tools = _capture_tools(monkeypatch)
    run_mcp_server(ConfigStore(tmp_path / "config.json"), transport="stdio")

    def whose(result: dict[str, Any]) -> set[str]:
        # Nights carry no owner field; demo envelope ids end "-o" (owner) / "-p" (partner).
        return {str(n["envelope_id"]).rsplit("-", 1)[-1] for n in result.get("nights") or []}

    mine = asyncio.run(tools["get_sleep_detail"](limit=5))
    theirs = asyncio.run(tools["get_sleep_detail"](limit=5, partner=True))
    assert whose(mine) == {"o"}
    assert whose(theirs) == {"p"}
    assert "mixed_owners" not in mine and "mixed_owners" not in theirs

    # A shared kind blends nothing by default: get_metric answers instead of refusing.
    weight = asyncio.run(tools["get_metric"](series="weight_kg", days=7))
    assert "error" not in weight["metrics"][0]

    # A never-shared kind read for the partner explains itself as "not shared".
    hr = asyncio.run(tools["get_metric"](series="resting_hr", days=7, partner=True))
    assert hr["metrics"][0]["points"]["rows"] == []
    assert "share" in hr["metrics"][0]["hint"]

    # Symptoms can be shared, so they follow the same rule: one person per read.
    def symptom_owners(result: dict[str, Any]) -> set[str]:
        return {str(o.get("owner_user_id", ""))[:8] for o in result.get("owners") or []}

    assert symptom_owners(asyncio.run(tools["get_symptoms"]())) == {"demo0001"}
    assert symptom_owners(asyncio.run(tools["get_symptoms"](partner=True))) == {"demo0002"}

    # Kinds that are never shared do not offer the option at all.
    for name in ("get_workouts", "get_intraday"):
        assert "partner" not in inspect.signature(tools[name]).parameters, name
    for name in tools:
        assert "owner" not in inspect.signature(tools[name]).parameters, name


def test_body_composition_water_and_mindfulness_detail_survive_the_tool_merge(
    monkeypatch: Any, tmp_path: Path
) -> None:
    """0.9.0 folded the per-kind read tools into `get_metric`, and SERIES only
    carried `weight_kg` for body, `intake_liters` for water and `total_minutes`
    for mindfulness — so body fat / BMI / lean mass (the reason to own a body-fat
    scale), refills, container size and session counts became unreadable over
    MCP. Each must come back as a real series with points."""

    monkeypatch.setenv("VAULTBEAT_DEMO", "1")
    tools = _capture_tools(monkeypatch)
    run_mcp_server(ConfigStore(tmp_path / "config.json"), transport="stdio")

    names = [
        "body_fat_percent", "bmi", "lean_body_mass_kg",
        "water_refills", "water_container_liters", "mindfulness_sessions",
    ]
    result = asyncio.run(tools["get_metric"](series=names, days=14))
    by_name = {m["series"]: m for m in result["metrics"]}
    for name in names:
        entry = by_name[name]
        assert "error" not in entry, (name, entry)
        assert entry["points"], f"{name}: no points from demo data that has the field"

    listed = {row["series"] for row in asyncio.run(tools["list_metric_series"]())["series"]}
    assert set(names) <= listed

    # State measurements refuse `sum`, same as weight.
    fat_sum = asyncio.run(tools["get_metric"](series="body_fat_percent", days=14, aggregation="sum"))
    agg = fat_sum["metrics"][0]["aggregate"]
    assert agg["value"] is None and "state" in agg["reason"]


def test_partner_empty_hint_names_every_shareable_type() -> None:
    """It said only sleep/cycle/water/weight could be shared, so an empty partner
    `get_symptoms` / `get_notes` was explained as "that type is never shared" —
    both are shareable through the partner's "share with partner's AI" switch."""

    from vaultbeat_mcp_local.service import PARTNER_EMPTY_HINT

    for word in ("sleep", "water", "weight", "cycle", "symptoms", "notes"):
        assert word in PARTNER_EMPTY_HINT, word


def test_notes_and_symptoms_warn_when_the_binding_does_not_know_whose_it_is() -> None:
    """Pre-2026 bindings carry no owner id, so both readers return everyone's
    rows for `partner=true` and `false` alike. That must be said, not implied."""

    from vaultbeat_mcp_local.mcp_server import _owner_unknown_note

    class _Stub:
        def __init__(self, me: str | None) -> None:
            self._me = me

        def person_owner(self, *, partner: bool = False) -> str | None:
            return self._me

    full = {"coverage": {"days_covered": 3}, "kinds": []}
    warned = _owner_unknown_note(dict(full), _Stub(None), True)  # type: ignore[arg-type]
    assert warned["owner_unknown"] is True
    assert "ignored" in warned["warning"] and "re-pair" in warned["warning"]

    # Known identity, or nothing returned: untouched.
    assert "owner_unknown" not in _owner_unknown_note(dict(full), _Stub("me-id"), True)  # type: ignore[arg-type]
    empty = {"coverage": {"days_covered": 0}}
    assert _owner_unknown_note(dict(empty), _Stub(None), True) == empty  # type: ignore[arg-type]
    # Never overwrites a warning that is already there.
    kept = _owner_unknown_note({**full, "warning": "mine"}, _Stub(None), False)  # type: ignore[arg-type]
    assert kept["warning"] == "mine" and kept["owner_unknown"] is True


def test_partner_measured_nothing_is_not_reported_as_not_shared(monkeypatch: Any, tmp_path: Path) -> None:
    """A partner who shares weight but has no body-fat scale returns weight rows
    with no body-fat values. That is "not measured", and the "not shared" hint
    would point the user at a sharing switch that is already on."""

    from vaultbeat_mcp_local.service import PARTNER_EMPTY_HINT, VaultbeatLocalService

    async def weight_only(self: Any, *, limit: Any = None, owner: Any = None, fresh: bool = False) -> dict[str, Any]:
        return {"days": [{"day_start_date": "2026-09-20", "local_date": "2026-09-20",
                          "weight_kg": 40.0, "body_fat_percent": None, "owner_user_id": "p"}]}

    async def nothing(self: Any, *, limit: Any = None, owner: Any = None, fresh: bool = False) -> dict[str, Any]:
        return {"days": []}

    svc = VaultbeatLocalService(ConfigStore(tmp_path / "config.json"))
    monkeypatch.setattr(VaultbeatLocalService, "weight_trend_summary", weight_only)
    r = asyncio.run(svc.metric_values(series="body_fat_percent", days=7, owner="!me"))
    assert r["metrics"][0].get("hint") != PARTNER_EMPTY_HINT

    monkeypatch.setattr(VaultbeatLocalService, "weight_trend_summary", nothing)
    r = asyncio.run(svc.metric_values(series="body_fat_percent", days=7, owner="!me"))
    assert r["metrics"][0].get("hint") == PARTNER_EMPTY_HINT


def test_sleep_nights_is_a_compact_table_of_every_night(monkeypatch: Any, tmp_path: Path) -> None:
    monkeypatch.setenv("VAULTBEAT_DEMO", "1")
    tools = _capture_tools(monkeypatch)
    run_mcp_server(ConfigStore(tmp_path / "config.json"), transport="stdio")

    full = asyncio.run(tools["get_sleep_nights"](limit=400))
    # In demo mode the watermark appends a `synthetic` column (2026-10-02).
    data_columns = [c for c in full["columns"] if c != "synthetic"]
    assert data_columns[:5] == ["date", "dow", "src", "bed", "wake"] and data_columns[-1] == "flag"
    assert full["rows"] and all(len(r) == len(full["columns"]) for r in full["rows"])
    assert "legend" in full and "coverage" in full
    dates = [r[0] for r in full["rows"]]
    assert dates == sorted(dates, reverse=True)

    cut = dates[len(dates) // 2]
    recent = asyncio.run(tools["get_sleep_nights"](since=cut))
    assert [r[0] for r in recent["rows"]] == [d for d in dates if d >= cut]
    assert recent["coverage"]["days_covered"] == len(recent["rows"]), "coverage must describe the rows returned"

    bad = asyncio.run(tools["get_sleep_nights"](since="2026-8-1"))
    assert bad["error"] == "invalid_since"

    theirs = asyncio.run(tools["get_sleep_nights"](partner=True))
    assert theirs["rows"] and theirs["rows"] != full["rows"][: len(theirs["rows"])]


def test_sleep_night_flags_show_every_night_get_metric_leaves_out() -> None:
    """An agent averaging the table's `asleep` column itself must see the same
    exclusions `get_metric` applies. Motion-inferred nights were visible only
    through `src`, and flags were exclusive, so a daytime stage-less night showed
    one of its two reasons (2026-09-25 code-map scan)."""

    from vaultbeat_mcp_local.mcp_server import _sleep_nights_table

    nights = [
        {"local_date": "2026-09-04", "motion_inferred": True},
        {"local_date": "2026-09-03", "daytime_main_sleep": True, "no_stage_detail": True},
        # Unworn implies no stages; saying both would be noise, not information.
        {"local_date": "2026-09-02", "is_in_bed_only": True, "no_stage_detail": True},
        {"local_date": "2026-09-01"},
    ]
    table = _sleep_nights_table({"nights": nights}, since=None)
    flags = [row[-1] for row in table["rows"]]
    assert flags == ["motion_inferred", "daytime,no_stages", "unworn", None]
    assert "motion_inferred" in table["legend"]["flag"]


def test_sleep_series_are_reachable_through_get_metric(monkeypatch: Any, tmp_path: Path) -> None:
    monkeypatch.setenv("VAULTBEAT_DEMO", "1")
    tools = _capture_tools(monkeypatch)
    run_mcp_server(ConfigStore(tmp_path / "config.json"), transport="stdio")

    names = ["sleep_minutes", "bedtime_minutes", "deep_sleep_minutes", "awakenings", "sleep_hr_mean"]
    out = asyncio.run(tools["get_metric"](series=names, days=14, aggregation="avg", granularity="week"))
    got = {m["series"]: m for m in out["metrics"]}
    assert set(got) == set(names)
    assert all("error" not in m for m in got.values()), got


def test_tools_reject_arguments_they_do_not_declare() -> None:
    """`partner=true` on a tool without that parameter, or the removed `owner=`,
    must fail loudly. The SDK used to drop them before the call, so the tool
    answered with the USER's records and nothing said the request was ignored."""
    from mcp.server.mcpserver import MCPServer
    from mcp.server.mcpserver.exceptions import ToolError

    from vaultbeat_mcp_local.mcp_server import _forbid_unknown_arguments

    _forbid_unknown_arguments()
    server = MCPServer("t")

    @server.tool()
    async def get_food_log(limit: int = 90) -> dict[str, Any]:
        return {"limit": limit}

    async def call(arguments: dict[str, Any]) -> Any:
        return await server.call_tool("get_food_log", arguments)

    assert asyncio.run(call({"limit": 3})) is not None
    for extra in ({"partner": True}, {"partner": "true"}, {"owner": "partner"}, {"owner": "me"}):
        with pytest.raises(ToolError, match="Extra inputs are not permitted"):
            asyncio.run(call({"limit": 3, **extra}))

    # `partner=false` asks for what this tool returns anyway — the user's own
    # data — so it is dropped rather than refused (review R12 on #9).
    for own in ({"partner": False}, {"partner": "false"}):
        assert asyncio.run(call({"limit": 3, **own})) is not None

    # …and on a tool that HAS `partner`, false still reaches the tool.
    @server.tool()
    async def get_notes(partner: bool = True) -> dict[str, Any]:
        return {"partner": partner}

    reply = asyncio.run(server.call_tool("get_notes", {"partner": False}))
    assert '"partner":false' in str(reply).replace(" ", "")

    # A second call wraps the SDK's step, not the previous wrapper.
    _forbid_unknown_arguments()
    assert asyncio.run(call({"limit": 3, "partner": False})) is not None


def test_a_result_too_large_for_a_client_says_how_to_ask_for_less() -> None:
    from vaultbeat_mcp_local.mcp_server import MAX_RESULT_BYTES, _bounded

    small = {"days": [{"local_date": "2026-10-01"}]}
    assert _bounded(small, "get_food_log") is small

    huge = {"days": [{"note": "x" * 1000} for _ in range(MAX_RESULT_BYTES // 1000 + 5)]}
    answer = _bounded(huge, "get_food_log")
    assert answer["error"] == "result_too_large"
    assert answer["result_bytes"] > MAX_RESULT_BYTES
    assert "`limit`" in answer["message"]

    # Bytes, not characters: CJK text is ~1 token per character at 3 bytes each,
    # so the same character count weighs three times as much.
    cjk = {"note": "饭" * (MAX_RESULT_BYTES // 3 + 10)}
    assert _bounded(cjk, "get_notes")["error"] == "result_too_large"


def test_dict_results_go_out_as_compact_json() -> None:
    """The SDK's `indent=2` was half of a year of sleep nights on the wire."""
    from mcp.server.mcpserver import MCPServer

    from vaultbeat_mcp_local.mcp_server import _compact_tool_results

    _compact_tool_results()
    _compact_tool_results()  # idempotent: must not wrap the wrapper
    server = MCPServer("t")

    @server.tool()
    async def get_rows() -> dict[str, Any]:
        return {"columns": ["date", "value"], "rows": [["2026-10-01", 1.5], ["2026-10-02", 2.0]]}

    result = asyncio.run(server.call_tool("get_rows", {}))
    text = result.content[0].text
    assert "\n" not in text
    assert json.loads(text) == {"columns": ["date", "value"], "rows": [["2026-10-01", 1.5], ["2026-10-02", 2.0]]}


def test_get_food_log_drops_its_default_limit_when_given_a_window(monkeypatch: Any, tmp_path: Path) -> None:
    """Two weeks is the default page, but a `since`/`until` window is the bound
    the caller chose — cutting it to the newest 14 days would quietly answer a
    different question (GitHub #9)."""
    from vaultbeat_mcp_local.service import VaultbeatLocalService

    seen: list[dict[str, Any]] = []

    async def fake_food_summary(self: Any, **kwargs: Any) -> dict[str, Any]:
        seen.append(kwargs)
        return {"days": [{"local_date": "2026-08-01"}]}

    monkeypatch.setattr(VaultbeatLocalService, "food_summary", fake_food_summary)
    monkeypatch.setenv("VAULTBEAT_DEMO", "1")
    tools = _capture_tools(monkeypatch)
    run_mcp_server(ConfigStore(tmp_path / "config.json"), transport="stdio")

    asyncio.run(tools["get_food_log"]())
    asyncio.run(tools["get_food_log"](since="2026-07-01", until="2026-08-31"))
    asyncio.run(tools["get_food_log"](since="2026-07-01", limit=5))

    assert [(s["limit"], s["since"], s["until"]) for s in seen] == [
        (14, None, None),
        (None, "2026-07-01", "2026-08-31"),
        (5, "2026-07-01", None),
    ]
