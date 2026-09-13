from __future__ import annotations

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


def test_doctor_returns_1_and_prints_fail_when_unbound(tmp_path: Path, capsys: Any) -> None:
    exit_code = cli.main(["--config", str(tmp_path / "config.json"), "doctor"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "[FAIL] config" in captured.out
    assert "bind" in captured.out


# ── One exit for health data (2026-09-12) ───────────────────────────────────
#
# Invariant 81 (health-data-has-one-exit). Two tests, because the regression has
# two shapes and either alone would let the other through: a command can be added
# back, and the funnel that lets a command print health data can be re-added.


def test_no_subcommand_can_print_health_data() -> None:
    """The subcommand set is exactly the six that pair, diagnose, or serve.

    Asserted as an EXACT set rather than a blacklist of the fifteen that were
    removed: a blacklist passes for `get_sleep`, `sleep2`, or anything else
    named differently, which is precisely how a removed capability comes back.

    Adding a genuinely new control command is meant to fail here — it is one
    line to update, and the failure is the prompt to ask whether the new command
    reads health data. That question is the whole point of this test.
    """

    parser = cli.build_parser()
    action = next(a for a in parser._subparsers._group_actions if hasattr(a, "choices"))

    assert set(action.choices) == {"init", "bind", "poll", "status", "doctor", "serve"}


def test_the_cli_has_no_health_data_funnel() -> None:
    """The helpers that existed to print decrypted health data stay gone.

    `_emit_decrypted` was THE place health data left the CLI, with `_health_json`
    deciding what it said and `_wrote_message` describing the 0600 file it could
    write. The set above would still pass if one of these came back attached to,
    say, `doctor` — so this asserts the machinery, not just the menu.

    ⚠️ `_print_json` is deliberately NOT in this list and must not be added: it
    prints config status and the doctor report, neither of which is health data.
    """

    for gone in ("_emit_decrypted", "_health_json", "_wrote_message", "_warn_demo_on_stderr"):
        assert not hasattr(cli, gone), (
            f"`{gone}` is back — health data has one exit, and it is the MCP protocol "
            "(Invariant 81 (health-data-has-one-exit))"
        )

    assert hasattr(cli, "_print_json"), "control commands still need their own printer"


def test_cli_error_names_the_exception_type_when_str_is_empty(
    monkeypatch: Any, tmp_path: Path, capsys: Any
) -> None:
    """`httpx.ReadTimeout` stringifies to "" — the most common real failure here
    (Supabase edge cold starts) produced a bare `error:` with zero diagnostic
    content. Three consecutive cold-backup runs failed that way on 2026-07-27 and
    the cause could only be found by bypassing this handler entirely.

    Carried by `status` since 2026-09-12: this asserts `main`'s except branch,
    not any one subcommand, and its old carrier (`water`) was one of the fifteen
    data subcommands removed that day. A surviving command is the durable
    choice — the handler under test is shared by all of them.
    """

    class SilentFailure(Exception):
        def __str__(self) -> str:
            return ""

    def boom(self: VaultbeatLocalService) -> dict[str, Any]:
        raise SilentFailure

    monkeypatch.setattr(VaultbeatLocalService, "status", boom)

    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--config", str(tmp_path / "config.json"), "status"])

    assert excinfo.value.code == 1
    assert "SilentFailure" in capsys.readouterr().err


def test_cli_error_keeps_the_message_when_there_is_one(
    monkeypatch: Any, tmp_path: Path, capsys: Any
) -> None:
    """Naming the type must not cost the message — both are printed."""

    def boom(self: VaultbeatLocalService) -> dict[str, Any]:
        raise ValueError("config is unreadable")

    monkeypatch.setattr(VaultbeatLocalService, "status", boom)

    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--config", str(tmp_path / "config.json"), "status"])

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


# ── Demo watermark on the CLI exit (2026-08-27) ─────────────────────────────
#
# The MCP exit has carried the synthetic-data stamp since demo mode shipped; the
# CLI exit did not, and the CLI is the one whose output gets redirected into a
# file, pasted into an issue, or handed to a second agent. These assert the
# CLI half of Invariant 61 (demo-is-a-boundary-not-a-flag).


# ── bind success copy (2026-09-06) ──────────────────────────────────────────


def test_bind_success_does_not_promise_access_the_plan_does_not_grant() -> None:
    """The sentence that was false every time it printed, from 09-03 to 09-06.

    It read "Full access to your health data is on until <date>". The trial
    unlocks the AI INTERFACE; Invariant 72 clamps the UPLOAD WINDOW to 7 days
    for `.trial` exactly as for `.free`, so the two facts had to stop being one
    sentence. Worse than occasionally wrong: `trial_ends_at` is non-nil only
    when a fresh trial clock starts, and that is suppressed for grandfathered,
    already-trialling and paid accounts — i.e. it printed if and only if the
    reader was clamped.
    """

    unlock = cli.TRIAL_UNLOCK_LINE.format(date="2026-09-09")
    assert "2026-09-09" in unlock
    assert "full access" not in unlock.lower(), (
        "an unqualified access claim is what made this line false; say what is unlocked"
    )
    assert "health data" not in unlock.lower(), (
        "the trial unlocks the interface, not the data window — naming the data here "
        "is exactly the merge that produced the falsehood"
    )


def test_bind_success_states_the_seven_day_window_as_a_boundary() -> None:
    """A user who asks for a month and gets a week must be able to tell why.

    Without the reason the product looks broken at its own success moment, and
    the obvious next moves (re-sync, re-pair) can never help.
    """

    window = cli.UPLOAD_WINDOW_LINE.lower()
    assert "7 days" in window
    assert "plan boundary" in window, "it has to say this is a plan, not a pending sync"
    assert "not a sync" in window
