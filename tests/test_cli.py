from __future__ import annotations

import json
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
    captured = capsys.readouterr()
    # The notice belongs on stderr and the document on stdout. Asserted as two
    # separate facts on purpose: the old version of this test read the notice
    # out of stdout and then did `out.index("{")` to step over it, which is the
    # bug written down as a test — `| jq` had no such escape hatch.
    assert "sensitive" in captured.err  # still shown to a human at a terminal
    assert "sensitive" not in captured.out  # and never in the document
    printed = json.loads(captured.out)  # parses WHOLE, no slicing
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


# ── Demo watermark on the CLI exit (2026-08-27) ─────────────────────────────
#
# The MCP exit has carried the synthetic-data stamp since demo mode shipped; the
# CLI exit did not, and the CLI is the one whose output gets redirected into a
# file, pasted into an issue, or handed to a second agent. These assert the
# CLI half of Invariant 61 (demo-is-a-boundary-not-a-flag).


def _data_subcommands() -> list[str]:
    """Every subcommand that emits health data, derived from the parser.

    Keyed off `--output`, which is exactly the set that goes through
    `_emit_decrypted` — control subcommands (`init` / `bind` / `poll` /
    `status` / `doctor` / `serve`) have no such flag. Derived rather than typed
    out so a data subcommand written next is covered on the day it is written,
    not on the day someone notices. `_read_tool_names` in `test_demo.py` exists
    for the same reason on the MCP side.
    """

    parser = cli.build_parser()
    action = next(a for a in parser._subparsers._group_actions if hasattr(a, "choices"))
    return sorted(
        name
        for name, sub in action.choices.items()
        if any("--output" in act.option_strings for act in sub._actions)
    )


@pytest.fixture
def demo_cli(monkeypatch: Any) -> None:
    """Demo mode on, memoized dataset dropped — mirrors `test_demo.py::demo_on`."""

    import vaultbeat_mcp_local.demo as demo_module

    monkeypatch.setenv(demo_module.DEMO_ENV, "1")
    demo_module.reset_cache()


def _stdout_payload(out: str) -> list[tuple[str, Any]]:
    """Parse stdout as a JSON document, preserving key order.

    Parses the WHOLE string rather than slicing from the first brace, which
    makes this the guard for "stdout is exactly one JSON document" across every
    data subcommand: anything printed alongside the payload fails here.

    It used to slice, because `menstrual` / `notes` / `symptoms` printed a prose
    sensitivity notice to stdout ahead of the payload and this docstring called
    that "a pre-existing wart this change neither introduces nor fixes". The
    slice is what let it survive — the notice broke `… | jq` for those three
    kinds while every test stepped politely around it. The notice now goes to
    stderr, so no slicing is needed and none should be reintroduced.
    """

    return json.loads(out, object_pairs_hook=list)


@pytest.mark.parametrize("subcommand", _data_subcommands())
def test_every_data_subcommand_leads_its_payload_with_the_warning(
    subcommand: str, tmp_path: Path, capsys: Any, demo_cli: None
) -> None:
    """`demo_warning` must be the FIRST key of what the CLI prints.

    First key, not merely present: `demo_mode: true` is a flag a reader has to
    already know to look for, while the banner is a sentence that acts on one
    who does not — and this payload's likeliest reader is a diff, an issue
    comment or another agent, none of which was told what it is looking at.

    Asserted on the serialised text via `object_pairs_hook`, never on a dict:
    a dict is an intermediate nobody outside this process sees, and the thing
    under test is precisely that `_health_json` stops sorting keys once the
    payload is stamped (sorted, `count` wins and the banner lands mid-document,
    behind an array long enough for a truncated paste to lose it).
    """

    assert cli.main(["--config", str(tmp_path / "config.json"), subcommand, "--limit", "2"]) == 0

    pairs = _stdout_payload(capsys.readouterr().out)
    assert pairs[0][0] == "demo_warning", f"{subcommand}: first key was {pairs[0][0]!r}"
    assert str(pairs[0][1]).startswith("[SYNTHETIC DEMO DATA]")
    assert dict(pairs)["demo_mode"] is True


@pytest.mark.parametrize("subcommand", _data_subcommands())
def test_every_data_subcommand_stamps_the_output_file_too(
    subcommand: str, tmp_path: Path, capsys: Any, demo_cli: None
) -> None:
    """`--output` is the branch that matters most and the easier one to forget.

    A file outlives the session that could have explained it, which is the whole
    case Invariant 61 exists for. It used to be a second `json.dumps` in a
    second function, so the stamp would have had to be remembered twice.
    """

    out_path = tmp_path / f"{subcommand}.json"
    args = ["--config", str(tmp_path / "config.json"), subcommand, "--limit", "2", "--output", str(out_path)]
    assert cli.main(args) == 0

    pairs = json.loads(out_path.read_text(encoding="utf-8"), object_pairs_hook=list)
    assert pairs[0][0] == "demo_warning", f"{subcommand}: file led with {pairs[0][0]!r}"

    # And the sentence about the file must not claim a decryption that never
    # happened — telling an operator to treat synthetic output as sensitive is
    # the same defect pointing the other way.
    printed = capsys.readouterr().out
    assert "SYNTHETIC" in printed
    assert "DECRYPTED" not in printed


def test_the_banner_line_goes_to_stderr_so_stdout_stays_a_json_document(
    tmp_path: Path, capsys: Any, demo_cli: None
) -> None:
    """Two markers, two channels, on purpose.

    stdout carries the in-band stamp because that is what survives `> out.json`;
    stderr carries the human sentence because a person with a terminal in front
    of them will not notice a key in the middle of 200 lines. Printing the
    sentence to stdout instead would break every consumer that parses this —
    `doctor` may print `[DEMO]` to stdout only because its human rendering is
    not JSON.
    """

    assert cli.main(["--config", str(tmp_path / "config.json"), "activity", "--limit", "1"]) == 0

    captured = capsys.readouterr()
    assert captured.out.lstrip().startswith("{")
    json.loads(captured.out)  # must parse whole, with nothing prepended
    assert captured.err.startswith("[SYNTHETIC DEMO DATA]")


def test_demo_output_is_byte_identical_across_runs(
    tmp_path: Path, capsys: Any, demo_cli: None
) -> None:
    """Dropping `sort_keys` for stamped payloads must not cost determinism.

    Sorting bought stability of key order across future code EDITS, which is
    worth something for a real export used as a diff baseline; run-to-run
    identity comes from the seeded generator and fixed dict construction, and is
    what makes two people comparing a demo payload in a bug report meaningful.
    That is the property being traded away, and this is the one being kept.
    """

    argv = ["--config", str(tmp_path / "config.json"), "sleep", "--limit", "5"]
    assert cli.main(argv) == 0
    first = capsys.readouterr().out
    assert cli.main(argv) == 0
    assert capsys.readouterr().out == first


def test_a_real_run_is_untouched_by_any_of_this(
    monkeypatch: Any, tmp_path: Path, capsys: Any
) -> None:
    """No stamp, keys still sorted, and the file sentence unchanged.

    The inverse failure is quieter than the one this change fixes but no more
    acceptable: labelling a real reading synthetic would teach a user to ignore
    the label that matters.
    """

    async def fake_summary(
        self: VaultbeatLocalService, *, limit: int | None = None, owner: str | None = None, fresh: bool = False
    ) -> dict[str, Any]:
        return {"zulu": 1, "alpha": 2, "rows": [{"a": 1}], "errors": []}

    monkeypatch.setattr(VaultbeatLocalService, "water_intake_summary", fake_summary)
    out_path = tmp_path / "real.json"
    args = ["--config", str(tmp_path / "config.json"), "water", "--output", str(out_path)]
    assert cli.main(args) == 0

    text = out_path.read_text(encoding="utf-8")
    # `object_pairs_hook` recurses, so it is used only for the ordering check —
    # reading a nested row out of it would compare a list of pairs to a dict.
    order = [key for key, _ in json.loads(text, object_pairs_hook=list)]
    assert order == ["alpha", "errors", "rows", "zulu"], "keys must stay sorted"
    written = json.loads(text)
    assert "demo_warning" not in written
    assert written["rows"] == [{"a": 1}], "rows must not gain `synthetic`"

    captured = capsys.readouterr()
    assert "DECRYPTED" in captured.out and "SYNTHETIC" not in captured.out
    assert captured.err == "", "no banner on a real run"
