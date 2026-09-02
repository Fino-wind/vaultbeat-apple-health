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

        def tool(self, *args: Any, **kwargs: Any) -> Any:
            def decorator(function: Any) -> Any:
                return function

            return decorator

        def streamable_http_app(self) -> str:
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
    monkeypatch.setattr("mcp.server.fastmcp.FastMCP", FakeFastMCP)

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
    monkeypatch.setattr("mcp.server.fastmcp.FastMCP", FakeFastMCP)

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
    monkeypatch.setattr("mcp.server.fastmcp.FastMCP", FakeFastMCP)

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
    monkeypatch.setattr("mcp.server.fastmcp.FastMCP", FakeFastMCP)

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
    monkeypatch.setattr("mcp.server.fastmcp.FastMCP", FakeFastMCP)
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
    monkeypatch.setattr("mcp.server.fastmcp.FastMCP", FakeFastMCP)
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

    for name in ("vaultbeat_status", "get_sleep_detail", "get_food_log", "get_hrv"):
        annotations = meta[name]["annotations"]
        assert annotations.readOnlyHint is True, name
        assert annotations.destructiveHint is False, name


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
        if not name.startswith("__") and kwargs["annotations"].openWorldHint
    )
    assert open_world == ["vaultbeat_doctor"]


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

    monkeypatch.setattr("mcp.server.fastmcp.FastMCP", FakeFastMCP)
    run_mcp_server(ConfigStore(tmp_path / "config.json"), transport="stdio")

    covered: list[str] = []
    for name, (function, hints) in captured.items():
        if not hints.readOnlyHint or hints.openWorldHint:
            continue
        result = function()
        if asyncio.iscoroutine(result):
            result = asyncio.run(result)
        returns_coverage = "coverage" in result
        documents_coverage = "coverage.days_covered" in (function.__doc__ or "")
        assert returns_coverage == documents_coverage, (
            f"{name}: returns coverage={returns_coverage}, documents it={documents_coverage}"
        )
        if returns_coverage:
            covered.append(name)

    # Sanity that demo mode actually produced results rather than the loop
    # trivially agreeing on "neither": the two ends of the read surface.
    assert "vaultbeat_sync_sleep" in covered
    assert "get_mindfulness" in covered
    assert "vaultbeat_status" not in covered


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
        assert meta[entry]["annotations"].destructiveHint is True, entry
        assert meta[append]["annotations"].destructiveHint is False, append
        assert meta[append]["annotations"].readOnlyHint is False, append


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


def test_demo_mode_wraps_sync_tools_too(monkeypatch: Any, tmp_path: Path) -> None:
    """Two of these tools are plain `def`, not `async def`.

    One sync wrapper around a coroutine function would hand FastMCP a coroutine
    object as the result — the tool would "succeed" and return something
    unserialisable — so the wrapper has to branch on the shape.
    `vaultbeat_start_binding` is the sync one that returns a plain dict.
    """
    monkeypatch.setenv("VAULTBEAT_DEMO", "1")
    tools = _capture_tools(monkeypatch)
    run_mcp_server(ConfigStore(tmp_path / "config.json"), transport="stdio")

    result = tools["vaultbeat_start_binding"]()
    assert not asyncio.iscoroutine(result)
    assert result["demo_mode"] is True
    assert "poll_id" in result


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

    params = inspect.signature(tools["get_hrv"]).parameters
    assert sorted(params) == ["fresh", "granularity", "limit", "owner"]


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

    result = tools["vaultbeat_status"]()
    assert "demo_mode" not in result
    assert "demo_warning" not in result

    params = inspect.signature(tools["get_hrv"]).parameters
    assert sorted(params) == ["fresh", "granularity", "limit", "owner"]


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
        and kwargs["annotations"].destructiveHint
        and "(" not in kwargs["title"]
    ]
    assert not offenders, (
        "destructive tools whose title does not name the consequence: "
        f"{offenders}. Use `verb (consequence)`, e.g. 'Log food (replaces day)'."
    )

    # And the two that prompted the rule say the right thing specifically —
    # the rule above accepts any parenthesis, which is deliberately loose.
    assert meta["vaultbeat_poll_binding"]["title"] == "Finish pairing (replaces this binding)"
    assert meta["vaultbeat_start_binding"]["title"] == "Start pairing (invalidates any open QR)"
