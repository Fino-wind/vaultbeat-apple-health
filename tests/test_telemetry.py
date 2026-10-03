"""Usage telemetry: what it sends, what it must never send, and when it is off."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from mcp import Client
from mcp.server.mcpserver import MCPServer

import vaultbeat_mcp_local.telemetry as telemetry_module
from vaultbeat_mcp_local.telemetry import PostHogMiddleware, Telemetry, _Sender

SECRET = "83.7kg-MENSTRUAL-2026-09-01-private-note"


class _Capture(_Sender):
    def __init__(self) -> None:
        super().__init__(post=lambda batch: None)
        self.events: list[dict[str, Any]] = []

    def put(self, event: dict[str, Any]) -> None:  # synchronous, no thread
        self.events.append(event)


def _server(sender: _Capture, monkeypatch: pytest.MonkeyPatch, account: str | None = "acct-uuid") -> MCPServer:
    monkeypatch.setattr(telemetry_module, "telemetry_disabled_reason", lambda *, demo: None)
    telemetry = Telemetry(account_id=lambda: account, demo=False, transport="stdio", sender=sender)
    server = MCPServer("t", middleware=[PostHogMiddleware(telemetry)])

    @server.tool()
    def get_metric(series: str, partner: bool = False) -> dict[str, Any]:
        return {"value": SECRET, "series": series}

    @server.tool()
    def soft_fail(note: str) -> dict[str, Any]:
        return {"error": "mixed_owners", "message": SECRET}

    @server.tool()
    def free_text_error(note: str) -> dict[str, Any]:
        return {"error": f"could not parse {SECRET}"}

    @server.tool()
    def boom(note: str) -> dict[str, Any]:
        raise ValueError(SECRET)

    return server


def _run(server: MCPServer, calls: list[tuple[str, dict[str, Any]]]) -> None:
    async def go() -> None:
        async with Client(server) as client:
            for name, args in calls:
                await client.call_tool(name, args)

    asyncio.run(go())


def test_tool_calls_are_recorded_without_arguments_or_results(monkeypatch: pytest.MonkeyPatch) -> None:
    sender = _Capture()
    _run(_server(sender, monkeypatch), [("get_metric", {"series": SECRET, "partner": True})])

    blob = json.dumps(sender.events)
    assert SECRET not in blob, "an argument or result value reached telemetry"
    by_event = {e["event"]: e for e in sender.events}
    # The in-process 2.x client speaks 2026-07-28, which has no `initialize`,
    # so `mcp_client_connected` is not expected here (see the middleware).
    call = by_event["mcp_tool_called"]
    assert call["distinct_id"] == "acct-uuid"
    props = call["properties"]
    assert props["tool"] == "get_metric" and props["ok"] is True and props["partner"] is True
    assert props["app"] == "vaultbeat-mcp" and props["duration_bucket"]
    assert props["client_name"], "which AI client is calling is the point of the session event"


def test_errors_carry_codes_we_wrote_but_never_free_text(monkeypatch: pytest.MonkeyPatch) -> None:
    sender = _Capture()
    _run(
        _server(sender, monkeypatch),
        [("soft_fail", {"note": SECRET}), ("free_text_error", {"note": SECRET}), ("boom", {"note": SECRET})],
    )
    assert SECRET not in json.dumps(sender.events)
    calls = {e["properties"]["tool"]: e["properties"] for e in sender.events if e["event"] == "mcp_tool_called"}
    assert calls["soft_fail"]["ok"] is False and calls["soft_fail"]["error_code"] == "mixed_owners"
    assert "error_code" not in calls["free_text_error"], "free text must not pass as a code"
    assert calls["boom"]["ok"] is False


def test_nothing_is_sent_for_an_unpaired_machine(monkeypatch: pytest.MonkeyPatch) -> None:
    sender = _Capture()
    _run(_server(sender, monkeypatch, account=None), [("get_metric", {"series": "steps"})])
    assert sender.events == []


@pytest.mark.parametrize(
    "env, value",
    [("VAULTBEAT_TELEMETRY", "0"), ("VAULTBEAT_TELEMETRY", "off"), ("DO_NOT_TRACK", "1")],
)
def test_opt_out_switches(monkeypatch: pytest.MonkeyPatch, env: str, value: str) -> None:
    monkeypatch.delenv("VAULTBEAT_TELEMETRY", raising=False)
    monkeypatch.delenv("DO_NOT_TRACK", raising=False)
    monkeypatch.delitem(__import__("sys").modules, "pytest", raising=False)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    assert telemetry_module.telemetry_disabled_reason(demo=False) is None
    monkeypatch.setenv(env, value)
    assert telemetry_module.telemetry_disabled_reason(demo=False) is not None


def test_demo_and_pytest_are_always_off(monkeypatch: pytest.MonkeyPatch) -> None:
    assert telemetry_module.telemetry_disabled_reason(demo=False) == "running under pytest"
    monkeypatch.delitem(__import__("sys").modules, "pytest", raising=False)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    assert telemetry_module.telemetry_disabled_reason(demo=True) == "demo mode"


def test_a_failing_endpoint_never_reaches_the_caller() -> None:
    def explode(batch: list[dict[str, Any]]) -> None:
        raise RuntimeError("network down")

    sender = _Sender(post=explode)
    sender._queue.put_nowait({"event": "x"})
    sender.flush(0.5)  # must not raise


def test_an_event_in_flight_at_exit_is_not_lost() -> None:
    """The last call of a session must survive the process exiting right after it."""

    import threading
    import time as _time

    sent: list[str] = []
    gate = threading.Event()

    def slow_post(batch: list[dict[str, Any]]) -> None:
        gate.wait(0.3)  # the network is slow when the client disconnects
        sent.extend(e["event"] for e in batch)

    sender = _Sender(post=slow_post)
    sender.put({"event": "last_call"})
    _time.sleep(0.05)  # the daemon thread has taken it and is mid-send
    sender.flush(2.0)
    assert sent == ["last_call"]


def test_doctor_reports_state_only_not_the_off_switch(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """The opt-out lives in the README and privacy page, not in every diagnosis
    an agent reads (and relays)."""

    from vaultbeat_mcp_local.mcp_server import run_mcp_server
    from vaultbeat_mcp_local.store import ConfigStore

    captured: dict[str, Any] = {}
    monkeypatch.setattr(MCPServer, "run", lambda self, **kw: captured.__setitem__("mcp", self))
    monkeypatch.setenv("VAULTBEAT_DEMO", "1")
    run_mcp_server(ConfigStore(tmp_path / "config.json"), transport="stdio")

    async def go() -> Any:
        async with Client(captured["mcp"]) as client:
            return await client.call_tool("vaultbeat_doctor", {})

    report = asyncio.run(go()).structured_content
    assert report["telemetry"] == {"enabled": False}
    assert "VAULTBEAT_TELEMETRY" not in json.dumps(report)
