from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import tether_mcp_local.cli as cli
from tether_mcp_local.mcp_server import _normalize_http_path, _normalize_transport, run_mcp_server
from tether_mcp_local.service import TetherLocalService
from tether_mcp_local.store import ConfigStore


def test_serve_defaults_to_stdio_transport(monkeypatch: Any, tmp_path: Path) -> None:
    captured: dict[str, Any] = {}

    def fake_run_mcp_server(store: ConfigStore, **kwargs: Any) -> None:
        captured["store"] = store
        captured.update(kwargs)

    # `serve` imports run_mcp_server lazily at call time, so the patch
    # must land on the defining module, not on cli.
    import tether_mcp_local.mcp_server as mcp_server_module

    monkeypatch.setattr(mcp_server_module, "run_mcp_server", fake_run_mcp_server)

    exit_code = cli.main(["--config", str(tmp_path / "config.json"), "serve"])

    assert exit_code == 0
    assert isinstance(captured["store"], ConfigStore)
    assert captured["store"].path == tmp_path / "config.json"
    assert captured["transport"] == "stdio"
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 8000
    assert captured["path"] == "/mcp"
    assert captured["json_response"] is True
    assert captured["stateless_http"] is True


def test_serve_http_transport_options_are_forwarded(monkeypatch: Any, tmp_path: Path) -> None:
    captured: dict[str, Any] = {}

    def fake_run_mcp_server(store: ConfigStore, **kwargs: Any) -> None:
        captured["store"] = store
        captured.update(kwargs)

    # `serve` imports run_mcp_server lazily at call time, so the patch
    # must land on the defining module, not on cli.
    import tether_mcp_local.mcp_server as mcp_server_module

    monkeypatch.setattr(mcp_server_module, "run_mcp_server", fake_run_mcp_server)

    exit_code = cli.main(
        [
            "--config",
            str(tmp_path / "config.json"),
            "serve",
            "--transport",
            "http",
            "--host",
            "0.0.0.0",
            "--port",
            "9000",
            "--path",
            "custom-mcp",
            "--sse-response",
            "--stateful-http",
        ]
    )

    assert exit_code == 0
    assert isinstance(captured["store"], ConfigStore)
    assert captured["transport"] == "http"
    assert captured["host"] == "0.0.0.0"
    assert captured["port"] == 9000
    assert captured["path"] == "custom-mcp"
    assert captured["json_response"] is False
    assert captured["stateless_http"] is False


def test_http_transport_alias_and_path_are_normalized() -> None:
    assert _normalize_transport("http") == "streamable-http"
    assert _normalize_transport("streamable-http") == "streamable-http"
    assert _normalize_http_path("custom-mcp") == "/custom-mcp"
    assert _normalize_http_path("") == "/mcp"


def test_run_mcp_server_configures_http_transport_on_fastmcp_init(
    monkeypatch: Any, tmp_path: Path
) -> None:
    captured: dict[str, Any] = {}

    class FakeFastMCP:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            captured["init_args"] = args
            captured["init_kwargs"] = kwargs

        def tool(self) -> Any:
            def decorator(function: Any) -> Any:
                return function

            return decorator

        def streamable_http_app(self) -> Any:
            captured["streamable_http_app_called"] = True
            return "ASGI_SENTINEL"

        def run(self, **kwargs: Any) -> None:
            captured["run_kwargs"] = kwargs

    fake_fastmcp_module = types.ModuleType("mcp.server.fastmcp")
    fake_fastmcp_module.FastMCP = FakeFastMCP
    monkeypatch.setitem(sys.modules, "mcp.server.fastmcp", fake_fastmcp_module)

    import uvicorn

    monkeypatch.setattr(
        uvicorn,
        "run",
        lambda app, **kwargs: captured.update(uvicorn_app=app, uvicorn_kwargs=kwargs),
    )

    # Loopback host keeps the fail-closed gate satisfied so this test stays focused
    # on FastMCP init wiring; non-loopback/token behaviour lives in test_mcp_server.
    run_mcp_server(
        ConfigStore(tmp_path / "config.json"),
        transport="http",
        host="127.0.0.1",
        port=9000,
        path="custom-mcp",
        json_response=False,
        stateless_http=False,
    )

    assert captured["init_args"] == ("Tether Local Sleep",)
    assert captured["init_kwargs"]["host"] == "127.0.0.1"
    assert captured["init_kwargs"]["port"] == 9000
    assert captured["init_kwargs"]["streamable_http_path"] == "/custom-mcp"
    assert captured["init_kwargs"]["json_response"] is False
    assert captured["init_kwargs"]["stateless_http"] is False
    # HTTP transport is now served by uvicorn over streamable_http_app();
    # mcp.run() is reserved for the stdio path only.
    assert captured["streamable_http_app_called"] is True
    assert captured["uvicorn_app"] == "ASGI_SENTINEL"  # loopback + no token => unwrapped
    assert captured["uvicorn_kwargs"]["host"] == "127.0.0.1"
    assert captured["uvicorn_kwargs"]["port"] == 9000
    assert "run_kwargs" not in captured


def test_resolve_http_token_prefers_env_over_config(monkeypatch: Any, tmp_path: Path) -> None:
    store = ConfigStore(tmp_path / "config.json")
    store.ensure_initialized()
    store.update(http_token="config-token")

    monkeypatch.setenv("TETHER_MCP_HTTP_TOKEN", "env-token")
    assert cli._resolve_http_token(store) == "env-token"  # env wins over stored config

    monkeypatch.delenv("TETHER_MCP_HTTP_TOKEN", raising=False)
    assert cli._resolve_http_token(store) == "config-token"  # falls back to config

    empty = ConfigStore(tmp_path / "empty.json")
    empty.ensure_initialized()
    assert cli._resolve_http_token(empty) is None  # neither env nor config set


def test_water_subcommand_prints_summary(
    monkeypatch: Any, tmp_path: Path, capsys: Any
) -> None:
    async def fake_summary(
        self: TetherLocalService, *, limit: int | None = None, fresh: bool = False
    ) -> dict[str, Any]:
        return {"day_count": 1, "average_daily_intake_liters": 3.0, "errors": []}

    monkeypatch.setattr(TetherLocalService, "water_intake_summary", fake_summary)

    exit_code = cli.main(["--config", str(tmp_path / "config.json"), "water", "--limit", "5"])

    assert exit_code == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["average_daily_intake_liters"] == 3.0


def test_menstrual_subcommand_prints_sensitivity_note_and_summary(
    monkeypatch: Any, tmp_path: Path, capsys: Any
) -> None:
    async def fake_summary(
        self: TetherLocalService, *, limit: int | None = None, fresh: bool = False
    ) -> dict[str, Any]:
        return {"day_count": 0, "predicted_next_period_start_date": None, "errors": []}

    monkeypatch.setattr(TetherLocalService, "menstrual_cycle_summary", fake_summary)

    exit_code = cli.main(["--config", str(tmp_path / "config.json"), "menstrual"])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "sensitive" in out  # privacy note is surfaced to the operator
    json_start = out.index("{")
    printed = json.loads(out[json_start:])
    assert printed["predicted_next_period_start_date"] is None


def test_water_subcommand_returns_3_on_decode_errors(
    monkeypatch: Any, tmp_path: Path
) -> None:
    async def fake_summary(
        self: TetherLocalService, *, limit: int | None = None, fresh: bool = False
    ) -> dict[str, Any]:
        return {"day_count": 0, "errors": ["env-1: TetherCryptoError"]}

    monkeypatch.setattr(TetherLocalService, "water_intake_summary", fake_summary)

    exit_code = cli.main(["--config", str(tmp_path / "config.json"), "water"])

    assert exit_code == 3  # mirrors `sync`'s nonzero exit when records failed to decode
