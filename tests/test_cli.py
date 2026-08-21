from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

import vaultbeat_mcp_local.cli as cli
from vaultbeat_mcp_local.mcp_server import _normalize_http_path, _normalize_transport, run_mcp_server
from vaultbeat_mcp_local.service import VaultbeatLocalService
from vaultbeat_mcp_local.store import ConfigStore


def test_serve_defaults_to_stdio_transport(monkeypatch: Any, tmp_path: Path) -> None:
    captured: dict[str, Any] = {}

    def fake_run_mcp_server(store: ConfigStore, **kwargs: Any) -> None:
        captured["store"] = store
        captured.update(kwargs)

    # `serve` imports run_mcp_server lazily at call time, so the patch
    # must land on the defining module, not on cli.
    import vaultbeat_mcp_local.mcp_server as mcp_server_module

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
    import vaultbeat_mcp_local.mcp_server as mcp_server_module

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

        def tool(self, *args: Any, **kwargs: Any) -> Any:
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

    # Pins the INTENT, not the string: this name is what every MCP client shows in
    # its server list, and it must not claim the server is sleep-only (it serves 18
    # kinds). Renamed 2026-08-11 from "Vaultbeat Local Sleep" — asserting the exact
    # new label would just re-freeze the copy and block the next honest rename.
    server_name = captured["init_args"][0]
    assert "Vaultbeat" in server_name
    assert "Sleep" not in server_name, "the name must not describe this as a sleep-only server"
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

    monkeypatch.setenv("VAULTBEAT_MCP_HTTP_TOKEN", "env-token")
    assert cli._resolve_http_token(store) == "env-token"  # env wins over stored config

    monkeypatch.delenv("VAULTBEAT_MCP_HTTP_TOKEN", raising=False)
    assert cli._resolve_http_token(store) == "config-token"  # falls back to config

    empty = ConfigStore(tmp_path / "empty.json")
    empty.ensure_initialized()
    assert cli._resolve_http_token(empty) is None  # neither env nor config set


def test_water_subcommand_prints_summary(
    monkeypatch: Any, tmp_path: Path, capsys: Any
) -> None:
    async def fake_summary(
        self: VaultbeatLocalService, *, limit: int | None = None, owner: str | None = None, fresh: bool = False
    ) -> dict[str, Any]:
        return {"day_count": 1, "average_daily_intake_liters": 3.0, "errors": []}

    monkeypatch.setattr(VaultbeatLocalService, "water_intake_summary", fake_summary)

    exit_code = cli.main(["--config", str(tmp_path / "config.json"), "water", "--limit", "5"])

    assert exit_code == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["average_daily_intake_liters"] == 3.0


def test_menstrual_subcommand_prints_sensitivity_note_and_summary(
    monkeypatch: Any, tmp_path: Path, capsys: Any
) -> None:
    async def fake_summary(
        self: VaultbeatLocalService, *, limit: int | None = None, owner: str | None = None, fresh: bool = False
    ) -> dict[str, Any]:
        return {"day_count": 0, "predicted_next_period_start_date": None, "errors": []}

    monkeypatch.setattr(VaultbeatLocalService, "menstrual_cycle_summary", fake_summary)

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
        self: VaultbeatLocalService, *, limit: int | None = None, owner: str | None = None, fresh: bool = False
    ) -> dict[str, Any]:
        return {"day_count": 0, "errors": ["env-1: VaultbeatCryptoError"]}

    monkeypatch.setattr(VaultbeatLocalService, "water_intake_summary", fake_summary)

    exit_code = cli.main(["--config", str(tmp_path / "config.json"), "water"])

    assert exit_code == 3  # mirrors `sync`'s nonzero exit when records failed to decode


def test_doctor_returns_1_and_prints_fail_when_unbound(tmp_path: Path, capsys: Any) -> None:
    exit_code = cli.main(["--config", str(tmp_path / "config.json"), "doctor"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "[FAIL] config" in captured.out
    assert "bind" in captured.out


def test_notes_kind_accepts_every_readable_kind() -> None:
    """`--kind` must cover the kinds this server itself writes.

    2026-07-27: choices were the iOS pair (sleep/menstrual) only, so the
    mood/general notes `log_note` creates were unfilterable from the CLI —
    argparse rejected them with exit code 2.
    """
    parser = cli.build_parser()

    for kind in ("sleep", "menstrual", "mood", "general"):
        args = parser.parse_args(["notes", "--kind", kind])
        assert args.kind == kind


def test_notes_kind_still_rejects_an_unknown_kind(capsys: Any) -> None:
    parser = cli.build_parser()

    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["notes", "--kind", "not-a-kind"])

    assert excinfo.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_cli_error_names_the_exception_type_when_str_is_empty(
    monkeypatch: Any, tmp_path: Path, capsys: Any
) -> None:
    """`httpx.ReadTimeout` stringifies to "" — the most common real failure here
    (Supabase edge cold starts) produced a bare `error:` with zero diagnostic
    content. Three consecutive cold-backup runs failed that way on 2026-07-27 and
    the cause could only be found by bypassing this handler entirely."""

    class SilentFailure(Exception):
        def __str__(self) -> str:
            return ""

    async def boom(
        self: VaultbeatLocalService, *, limit: int | None = None, owner: str | None = None, fresh: bool = False
    ) -> dict[str, Any]:
        raise SilentFailure

    monkeypatch.setattr(VaultbeatLocalService, "water_intake_summary", boom)

    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--config", str(tmp_path / "config.json"), "water"])

    assert excinfo.value.code == 1
    assert "SilentFailure" in capsys.readouterr().err


def test_cli_error_keeps_the_message_when_there_is_one(
    monkeypatch: Any, tmp_path: Path, capsys: Any
) -> None:
    """Naming the type must not cost the message — both are printed."""

    async def boom(
        self: VaultbeatLocalService, *, limit: int | None = None, owner: str | None = None, fresh: bool = False
    ) -> dict[str, Any]:
        raise ValueError("config is unreadable")

    monkeypatch.setattr(VaultbeatLocalService, "water_intake_summary", boom)

    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--config", str(tmp_path / "config.json"), "water"])

    assert excinfo.value.code == 1
    err = capsys.readouterr().err
    assert "ValueError" in err
    assert "config is unreadable" in err


def test_bind_can_always_render_a_qr_code(capsys: pytest.CaptureFixture[str]) -> None:
    """`bind` must produce something scannable without extra installs.

    qrcode was an optional `[qr]` extra until 0.3.10, so `bind` normally printed a
    raw JSON payload plus "install the extra and run me again" — at the exact
    moment the user has a phone in their hand and nothing to point it at. Worse,
    running `bind` again mints a NEW pollID, so following that advice invalidates
    anything already scanned. The owner walked into it on 2026-08-11 while
    testing the first-time bind path as a user would.

    This asserts the import is reachable and the renderer actually emits a QR,
    which is what stops the dependency from being "tidied" back into an extra.
    """

    from vaultbeat_mcp_local.cli import _print_qr

    _print_qr('{"pollID":"abc","publicKeyBase64":"k","serverName":"n"}')

    out = capsys.readouterr().out
    assert "install the qr extra" not in out
    # print_ascii uses half-block glyphs; any of them means a real code was drawn.
    assert any(ch in out for ch in "▀▄█"), "no QR was rendered"
