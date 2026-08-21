"""Demo mode, asserted at the WIRE — one test per way synthetic data could leak.

Demo mode (`VAULTBEAT_DEMO=1`) serves a synthetic dataset so a reviewer with no
iPhone can exercise every read tool. That convenience creates exactly one new
class of harm: a number that came from `demo.py` being read, quoted, screenshotted
or filed as if it were somebody's health record. Every test here defends one
route by which that could happen, plus the anti-rot guard that keeps the demo
covering the whole product as the product grows.

WHY THIS FILE GOES THROUGH THE REAL `FastMCP`
---------------------------------------------
`tests/test_service.py` and `tests/test_mcp_server.py` already cover demo mode at
the service layer and against a fake FastMCP. Neither can answer the question
that matters here — **does the watermark survive into what a client actually
receives?** A tool result crosses the wire in TWO independent copies:
`content[0].text` (the JSON a text-only client renders) and `structuredContent`
(the parsed object a structured client reads). A stamp present in one and absent
from the other is a leak for half the clients in existence, and the fake FastMCP
used elsewhere produces neither field — it hands back the raw return value. So
these tests build the real server, call through the low-level `CallToolRequest`
handler, and read the `CallToolResult` a client would.

TWO HONEST FACTS ABOUT DEMO MODE, VERIFIED HERE RATHER THAN ASSUMED
-------------------------------------------------------------------
Both were found by running the tools, not by reading the code, and both would
otherwise make a naive test either wrong or flaky:

1. **The binding tools are not disabled by demo mode, and should not be.**
   `vaultbeat_start_binding` really does generate a keypair and write
   `config.json`; `vaultbeat_poll_binding` really does call the cloud. That is
   correct — demo mode is a data source, not a sandbox, and someone who decides
   to pair while evaluating must be able to. It means the "writes nothing to
   disk" guarantee is about the READ tools, and this file says so explicitly
   rather than quietly excluding two tools from a loop.

2. **`vaultbeat_poll_binding` raises without a binding session**, demo or not
   ("No active binding session; call `vaultbeat_start_binding` first"). That is
   its real contract, not a demo defect, so the all-tools sweep starts a session
   first instead of pretending the raise is a bug.

WHAT IS DELIBERATELY NOT ASSERTED
----------------------------------
No test here pins a tool COUNT (29 today). The sweep iterates whatever
`list_tools()` returns, so a new tool is covered the day it lands; a hardcoded
number would instead need remembering, and the one thing this file is about is
not relying on anyone remembering.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

import mcp.server.fastmcp as fastmcp_module
import mcp.types as mcp_types
import vaultbeat_mcp_local.demo as demo_module
import vaultbeat_mcp_local.service as service_module
from vaultbeat_mcp_local.client import PollBindingResult
from vaultbeat_mcp_local.demo import (
    DEMO_BANNER,
    DEMO_ENV,
    demo_records,
    missing_kinds,
    stale_kinds,
)
from vaultbeat_mcp_local.mcp_server import run_mcp_server
from vaultbeat_mcp_local.service import KNOWN_METRIC_TYPES, VaultbeatLocalService
from vaultbeat_mcp_local.store import ConfigStore

# ── Test scaffolding ─────────────────────────────────────────────────────────

#: Arguments for the tools that declare required parameters.
#:
#: Not a convenience table — `_assert_args_cover_every_tool` cross-checks it
#: against each tool's published `inputSchema`, so a new required parameter (or a
#: whole new write tool) fails here with a message naming it rather than showing
#: up as a confusing validation error inside one sweep.
#:
#: ⚠️ `sets` and `meals` must be NON-EMPTY: each log method validates its own
#: arguments BEFORE reaching the demo guard (deliberately — the tool's real
#: contract stays observable in demo mode), so `sets: []` raises a genuine
#: ValueError and the call never reaches the refusal this file is testing.
_ARGS: dict[str, dict[str, Any]] = {
    "log_food_entry": {"date": "2026-08-20", "meals": [{"items": [{"food": "香蕉"}]}]},
    "log_food_append": {"date": "2026-08-20", "meals": [{"items": [{"food": "香蕉"}]}]},
    "log_strength_entry": {
        "date": "2026-08-20",
        "exercises": [{"name": "卧推", "sets": [{"weightKg": 30, "reps": 8}]}],
    },
    "log_strength_append": {
        "date": "2026-08-20",
        "exercises": [{"name": "卧推", "sets": [{"weightKg": 30, "reps": 8}]}],
    },
    "log_note": {"text": "恶心"},
    "log_note_append": {"text": "恶心"},
    "log_weight_entry": {"weight_kg": 72.5},
}

#: Tools that write to the cloud on a real install, mapped to the service method
#: whose demo guard actually refuses them. In demo mode every one must refuse.
#:
#: SEVEN tools, FOUR methods — `log_*_append` is the same method called with
#: `merge=True`, published as a separate tool because an ARGUMENT cannot be
#: annotated non-destructive. They share the refusal, so asserting only the four
#: methods would leave three published tools unproven at the layer a client sees.
#:
#: ⚠️ The mapping is not cosmetic: the refusal's `tool` field reports the METHOD
#: (`log_food_entry`) even when the caller invoked `log_food_append`, because
#: `_demo_write_refusal` is called from inside the service. Verified 2026-08-20,
#: recorded rather than asserted-away — an agent that called the append tool is
#: told a different tool name back, which is mildly confusing but not a leak.
_WRITE_TOOLS: dict[str, str] = {
    "log_food_entry": "log_food_entry",
    "log_food_append": "log_food_entry",
    "log_strength_entry": "log_strength_entry",
    "log_strength_append": "log_strength_entry",
    "log_note": "log_note",
    "log_note_append": "log_note",
    "log_weight_entry": "log_weight_entry",
}

#: The synthetic-data sentence travels under ONE key everywhere: `demo_warning`.
#:
#: It used to travel under two — `demo_status()` alone said `demo_banner`, so a
#: consumer written against `demo_warning` (which is what `cli.py`'s doctor
#: renderer reads, and what every other surface emits) silently found nothing on
#: `vaultbeat_status` alone.
#:
#: ⚠️ Precisely: this did NOT mean `cli.py status` lost a banner LINE. That
#: handler is `_print_json(...)` and has no banner-printing code at all under
#: either key — a separate gap, deliberately left for its own change. What the
#: split cost was one key name that three surfaces agree on and a fourth did
#: not. Stating it exactly because the looser version ("status printed no
#: banner") is true for an unrelated reason and would send the next reader to
#: fix the wrong thing. Unified 2026-08-20; this tuple
#: stays a tuple only so `_banner` keeps its shape, and the single entry is now
#: itself an assertion — a regression that reintroduces `demo_banner` makes
#: `test_the_banner_travels_under_exactly_one_key` go red rather than silently
#: passing through a fallback.
_BANNER_KEYS = ("demo_warning",)


def _banner(structured: dict[str, Any]) -> str:
    return str(next((structured[k] for k in _BANNER_KEYS if k in structured), ""))


#: Tools that only read. The disk- and determinism-guarantees are scoped to these
#: — see honest fact 1 in the module docstring for why the binding pair is out.
#: Derived at runtime from the registered tool list, never typed out here.
def _read_tool_names(server: Any) -> list[str]:
    names = sorted(tool.name for tool in asyncio.run(server.list_tools()))
    return [n for n in names if n.startswith("get_") or n in ("vaultbeat_sync_sleep",)]


class _RecordingCloud:
    """A cloud client that records every call and performs none.

    Recording rather than raising: a fake that explodes proves only that the
    happy path avoided it, while a recorder can be asked the question this file
    actually cares about — *which* calls happened. `writes` being empty is then
    positive evidence, and `calls` containing `poll_binding` in the sweep is what
    gives that emptiness teeth (a fake nothing reaches records nothing either).
    """

    #: Shared across instances: the service builds a fresh client per call
    #: (`self._cloud_client or VaultbeatCloudClient(...)`), so per-instance lists
    #: would scatter the record across objects the test never sees.
    calls: list[str] = []
    writes: list[str] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    @classmethod
    def reset(cls) -> None:
        cls.calls = []
        cls.writes = []

    def _record(self, name: str) -> None:
        type(self).calls.append(name)

    def _record_write(self, name: str) -> None:
        type(self).calls.append(name)
        type(self).writes.append(name)

    async def poll_binding(self, poll_id: str) -> PollBindingResult:
        self._record("poll_binding")
        return PollBindingResult(status="pending")

    async def sync(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        self._record("sync")
        return []

    async def sync_digest(self, *args: Any, **kwargs: Any) -> Any:
        self._record("sync_digest")
        return None

    async def sync_catalog(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        self._record("sync_catalog")
        return []

    async def sync_blobs(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        self._record("sync_blobs")
        return []

    async def write_strength_blob(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        self._record_write("write_strength_blob")
        return {}

    async def write_food_blob(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        self._record_write("write_food_blob")
        return {}

    async def write_body_blob(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        self._record_write("write_body_blob")
        return {}

    async def write_note_blob(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        self._record_write("write_note_blob")
        return {}

    async def report_decrypt_failures(self, *args: Any, **kwargs: Any) -> None:
        self._record("report_decrypt_failures")


@pytest.fixture
def demo_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """Turn demo mode on for one test, and drop the memoized dataset.

    `demo._CACHE` is module-level and outlives a test. Clearing it on the way IN
    means each test exercises generation rather than whatever a previous test
    happened to leave behind — which is the difference between asserting that the
    generator is deterministic and asserting that a dict is a dict.
    """

    monkeypatch.setenv(DEMO_ENV, "1")
    demo_module.reset_cache()


@pytest.fixture
def recording_cloud(monkeypatch: pytest.MonkeyPatch) -> type[_RecordingCloud]:
    """Replace the real cloud client class the service reaches for.

    `run_mcp_server` gives no seam to inject a client, so the substitution happens
    at the name `service._client()` resolves — `VaultbeatCloudClient` in the
    service module's namespace. Without this, `vaultbeat_poll_binding` makes a
    real HTTPS request to production Supabase, which would make this suite slow,
    offline-hostile and non-credential-free.
    """

    _RecordingCloud.reset()
    monkeypatch.setattr(service_module, "VaultbeatCloudClient", _RecordingCloud)
    return _RecordingCloud


def _build_server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """Construct the real FastMCP server without letting it block on a transport.

    `run_mcp_server` ends in `mcp.run(transport="stdio")`, which never returns.
    Patching `FastMCP.run` to capture `self` is the only hook: the object is
    created inside that function and never handed back.
    """

    captured: dict[str, Any] = {}

    def fake_run(self: Any, **kwargs: Any) -> None:
        captured["mcp"] = self

    monkeypatch.setattr(fastmcp_module.FastMCP, "run", fake_run)
    run_mcp_server(ConfigStore(tmp_path / "config.json"), transport="stdio")
    return captured["mcp"]


def _call(server: Any, name: str, arguments: dict[str, Any] | None = None) -> Any:
    """Invoke one tool the way a client does, returning the `CallToolResult`.

    Goes through the low-level `CallToolRequest` handler rather than
    `FastMCP.call_tool`, because only the handler builds the result object that
    carries BOTH `content` and `structuredContent` — the two copies this file
    exists to check. `FastMCP.call_tool` returns a bare tuple with no
    `isError`, so a failing tool would surface as a raised exception rather than
    the error result a client would actually see.
    """

    handler = server._mcp_server.request_handlers[mcp_types.CallToolRequest]
    request = mcp_types.CallToolRequest(
        method="tools/call",
        params=mcp_types.CallToolRequestParams(name=name, arguments=arguments or {}),
    )
    return asyncio.run(handler(request)).root


def _assert_identical(first: bytes, second: bytes, label: str) -> None:
    """Compare two large payloads and fail SMALL.

    A bare `assert first == second` on multi-megabyte strings does not fail — it
    HANGS. pytest rewrites the comparison and builds a character diff, which on a
    1.7 MB pair spins at 100% CPU for minutes (measured 2026-08-20 while mutation
    -testing this very file: the run had to be killed, and in CI it would be a
    timeout with no diff attached).

    A determinism test whose failure mode is a hang is worse than no test: nobody
    waits it out, and the one thing it owed you — WHERE the two runs diverged —
    is exactly what never arrives. So the equality check happens outside an
    `assert` (no rewriting), and the report is a hash plus a bounded window
    around the first differing byte.
    """

    if first == second:
        return
    digests = (hashlib.sha256(first).hexdigest()[:12], hashlib.sha256(second).hexdigest()[:12])
    limit = min(len(first), len(second))
    at = next((i for i in range(limit) if first[i] != second[i]), limit)
    window = slice(max(0, at - 60), at + 60)
    raise AssertionError(
        f"{label} differs between runs — demo output must be byte-identical.\n"
        f"  sha256[:12]  {digests[0]} vs {digests[1]}\n"
        f"  lengths      {len(first)} vs {len(second)}\n"
        f"  first diff at byte {at}\n"
        f"  run A ...{first[window]!r}...\n"
        f"  run B ...{second[window]!r}..."
    )


def _assert_args_cover_every_tool(server: Any) -> None:
    """Fail with a name when `_ARGS` falls behind the published schemas."""

    gaps: list[str] = []
    for tool in asyncio.run(server.list_tools()):
        required = set(tool.inputSchema.get("required", []))
        supplied = set(_ARGS.get(tool.name, {}))
        if not required <= supplied:
            gaps.append(f"{tool.name} needs {sorted(required - supplied)}")
    assert gaps == [], "add these to _ARGS in tests/test_demo.py: " + "; ".join(gaps)


# ── 1. Every tool answers, with no account ───────────────────────────────────


def test_every_registered_tool_answers_in_demo_mode_without_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, demo_on: None, recording_cloud: Any
) -> None:
    """The product claim, at the wire: no iPhone, no key, no config — everything answers.

    A tool that errors in demo mode reads to an evaluator as a broken product,
    and they have no way to tell that apart from a broken demo. So this sweeps
    EVERY registered tool rather than a curated list — a new tool is covered the
    day it is added.

    `vaultbeat_start_binding` runs first on purpose: `vaultbeat_poll_binding`
    legitimately refuses without a session (see honest fact 2), and starting one
    is what makes the sweep test demo mode instead of re-testing that contract.
    """

    server = _build_server(tmp_path, monkeypatch)
    _assert_args_cover_every_tool(server)

    names = sorted(tool.name for tool in asyncio.run(server.list_tools()))
    # start_binding first, everything else alphabetically.
    ordered = sorted(names, key=lambda n: (n != "vaultbeat_start_binding", n))

    failures: list[str] = []
    for name in ordered:
        result = _call(server, name, _ARGS.get(name))
        if result.isError:
            detail = result.content[0].text if result.content else "<no content>"
            failures.append(f"{name}: {detail[:160]}")

    assert failures == [], "tools errored in demo mode:\n" + "\n".join(failures)

    # Teeth for the write assertions elsewhere in this file: the fake is not
    # merely unreached scenery — one read genuinely arrived at it.
    assert "poll_binding" in recording_cloud.calls
    assert recording_cloud.writes == []


# ── 2. The watermark reaches BOTH copies of every result ─────────────────────


def test_watermark_is_present_in_both_wire_copies_of_every_tool_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, demo_on: None, recording_cloud: Any
) -> None:
    """`content[0].text` AND `structuredContent` — a stamp in one is a leak for the other.

    Clients read one or the other, not both: a text-only client renders the JSON
    blob, a structured client parses the object. Marking only the object leaves
    every pasted transcript unmarked; marking only the text leaves every
    programmatic consumer unmarked. Both, or it is not watermarked.

    `demo_warning` is checked alongside `demo_mode` because a boolean survives
    summarisation badly — the sentence is what a human reading a screenshot
    actually sees.
    """

    server = _build_server(tmp_path, monkeypatch)
    names = sorted(tool.name for tool in asyncio.run(server.list_tools()))
    ordered = sorted(names, key=lambda n: (n != "vaultbeat_start_binding", n))

    unmarked_structured: list[str] = []
    unmarked_text: list[str] = []
    for name in ordered:
        result = _call(server, name, _ARGS.get(name))
        assert not result.isError, name

        structured = result.structuredContent
        if not (isinstance(structured, dict) and structured.get("demo_mode") is True):
            unmarked_structured.append(name)
        elif "SYNTHETIC" not in _banner(structured):
            unmarked_structured.append(f"{name} (no banner sentence)")

        text = result.content[0].text if result.content else ""
        if '"demo_mode": true' not in text or "SYNTHETIC" not in text:
            unmarked_text.append(name)

    assert unmarked_structured == []
    assert unmarked_text == []


def test_write_refusals_carry_the_watermark_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, demo_on: None, recording_cloud: Any
) -> None:
    """The refusal is the result most likely to be pasted into a bug report.

    It is also the one an agent is most likely to act on — and the action it
    would reach for on an unmarked "this failed" is re-pairing the machine, the
    most destructive thing available to it (Invariant 54). The stamp is what
    stops that, so it is asserted on the refusal specifically rather than left to
    the sweep above.
    """

    server = _build_server(tmp_path, monkeypatch)

    for name in _WRITE_TOOLS:
        result = _call(server, name, _ARGS[name])
        assert not result.isError, f"{name} must REFUSE, not raise"

        structured = result.structuredContent
        assert isinstance(structured, dict), name
        assert structured["demo_mode"] is True, name
        assert "SYNTHETIC" in _banner(structured), name

        text = result.content[0].text
        assert '"demo_mode": true' in text, name
        assert "SYNTHETIC" in text, name


# ── 3. Writes refuse, and nothing leaves the machine ─────────────────────────


def test_write_tools_refuse_and_no_write_ever_reaches_the_cloud(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, demo_on: None, recording_cloud: Any
) -> None:
    """Two separate claims, and the second is the one worth a fake for.

    "The tool returned a refusal" says nothing about whether a request went out
    before it — an implementation that POSTed and then reported failure would
    pass a return-value-only assertion. So the cloud client is recorded and
    asserted empty, which is a statement about the wire rather than about the
    answer.

    Returned, never raised: a raise becomes a ToolError that bypasses the
    watermarking wrapper entirely, stripping the one fact the caller most needs.
    """

    server = _build_server(tmp_path, monkeypatch)

    for name, refusing_method in _WRITE_TOOLS.items():
        result = _call(server, name, _ARGS[name])
        structured = result.structuredContent

        assert result.isError is False, f"{name} raised instead of refusing"
        assert structured["ok"] is False, name
        assert structured["error"] == "demo_mode_is_read_only", name
        assert structured["tool"] == refusing_method, name
        # Names demo mode in the refusal, so an agent does not read "not bound"
        # and go re-pair a machine that is working exactly as configured.
        assert DEMO_ENV in structured["detail"], name

    assert recording_cloud.writes == []
    assert recording_cloud.calls == []


# ── 4. The demo covers the whole product — the anti-rot guard ────────────────


def test_demo_covers_exactly_the_kinds_the_read_path_accepts(demo_on: None) -> None:
    """THE guard in this file. When kind #18 lands and the demo does not follow, this goes red.

    `check_metric_type_contract.py` cross-checks `KNOWN_METRIC_TYPES` against the
    registration sites it can discover — and it cannot see `demo.py`. A hand-typed
    kind list here would be one more copy AND the one nothing watches
    (Invariant 18), so the demo derives its coverage from the imported set and
    this test asserts both directions:

    · a known kind with no builder → that tool returns empty in the demo, which
      reads as a broken feature to the one audience evaluating the product;
    · a builder for a retired kind → the demo fabricates records for something
      the read path has stopped accepting.

    Then the third assertion, which the two set comparisons cannot make: a
    builder that exists but returns nothing satisfies both and still ships a
    hole. Coverage is about DATA, not about keys.
    """

    assert missing_kinds() == frozenset(), (
        "these metric kinds have no demo dataset; add a builder in demo.py: "
        f"{sorted(missing_kinds())}"
    )
    assert stale_kinds() == frozenset(), (
        "these demo builders are for kinds the read path no longer accepts: "
        f"{sorted(stale_kinds())}"
    )

    empty = sorted(kind for kind in KNOWN_METRIC_TYPES if not demo_records(kind))
    assert empty == [], f"demo builders produced no records for: {empty}"


def test_the_advertised_kind_list_matches_what_is_actually_served(demo_on: None) -> None:
    """`demo_status()["demo_kinds"]` is published in `status` and `doctor`.

    It is a second list, written for humans, and a second list is a second thing
    that can drift. Pinning it to the same source means the sentence a reviewer
    reads cannot disagree with the data they get.
    """

    advertised = demo_module.demo_status()["demo_kinds"]
    assert advertised == sorted(KNOWN_METRIC_TYPES)


# ── 5. Determinism ───────────────────────────────────────────────────────────


def test_two_independent_services_produce_byte_identical_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, demo_on: None, recording_cloud: Any
) -> None:
    """Byte-identical, not merely equivalent — otherwise demo output is unusable as evidence.

    Two people comparing a pasted result, or a bug report filed against a demo
    run, are only meaningful if the same command produces the same bytes
    everywhere. That requires a fixed seed AND a fixed anchor day: a
    `datetime.now()` anywhere in the generator, or an unseeded `random`, or a
    salted `hash()`, breaks it in a way no per-field assertion would catch.

    `reset_cache()` between the two runs is load-bearing. Without it the second
    service reads the first one's memoized objects and this test asserts that a
    dict is itself — green, and proving nothing about the generator.

    Scoped to the read tools deliberately: `vaultbeat_start_binding` mints a
    fresh keypair every call and is *supposed* to differ.
    """

    def render(directory: Path) -> bytes:
        server = _build_server(directory, monkeypatch)
        payload = {
            name: _call(server, name).structuredContent for name in _read_tool_names(server)
        }
        return json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()

    first = render(tmp_path / "run-a")
    demo_module.reset_cache()
    second = render(tmp_path / "run-b")

    _assert_identical(first, second, "aggregated read-tool output")
    # Non-vacuous: a generator that returned nothing would also be "identical".
    assert len(first) > 100_000, len(first)


def test_the_generator_itself_is_deterministic_across_a_cache_reset(demo_on: None) -> None:
    """The same property one layer down, where a regression would actually land.

    The test above goes through 18 tools' aggregation; this one compares the raw
    records, so a change in the generator is not masked by a summariser that
    happens to round it away.
    """

    def render() -> bytes:
        demo_module.reset_cache()
        payload = [record.to_dict() for record in demo_records()]
        return json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()

    _assert_identical(render(), render(), "raw demo records")

    # Per-kind seeding: one kind's slice must not depend on which other kinds
    # were generated first, or `demo_records("sleep")` and the sleep part of
    # `demo_records()` would disagree.
    def render_sleep(*, alone: bool) -> bytes:
        demo_module.reset_cache()
        records = demo_records("sleep") if alone else [
            r for r in demo_records() if r.metric_type == "sleep"
        ]
        payload = [record.to_dict() for record in records]
        return json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()

    _assert_identical(
        render_sleep(alone=True), render_sleep(alone=False), "the sleep slice"
    )


# ── 6. A full read pass touches no byte on disk ──────────────────────────────


def test_a_full_read_pass_writes_nothing_to_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, demo_on: None, recording_cloud: Any
) -> None:
    """Structural, not careful: the demo branch sits ABOVE `cache.load`/`cache.save`.

    A demo run that seeded the plaintext cache would contaminate the next REAL
    run of the same install with synthetic records — silently, and in the one
    file whose whole job is to hold decrypted health data. The guarantee is
    architectural rather than a rule anyone follows, and this asserts it end to
    end at the tool layer.

    Snapshots paths, bytes AND mtime_ns, so a rewrite with identical content
    fails too. Scoped to the read tools: the binding pair writes `config.json` by
    design (honest fact 1 in the module docstring).
    """

    config_path = tmp_path / "config.json"
    cache_directory = VaultbeatLocalService(ConfigStore(config_path)).cache.directory
    # A sentinel gives the assertion teeth: with an empty tree, "unchanged" is
    # also what a broken snapshot function reports.
    cache_directory.mkdir(parents=True, exist_ok=True)
    sentinel = cache_directory / "sentinel.json"
    sentinel.write_text('{"real": "plaintext"}', encoding="utf-8")

    def snapshot() -> dict[str, tuple[int, bytes]]:
        return {
            path.relative_to(tmp_path).as_posix(): (path.stat().st_mtime_ns, path.read_bytes())
            for path in sorted(tmp_path.rglob("*"))
            if path.is_file()
        }

    before = snapshot()
    assert "cache/sentinel.json" in before  # the teeth

    server = _build_server(tmp_path, monkeypatch)
    for name in _read_tool_names(server):
        assert not _call(server, name).isError, name
    for name in ("vaultbeat_status", "vaultbeat_doctor"):
        assert not _call(server, name).isError, name

    assert snapshot() == before
    assert not config_path.exists()


# ── 7. `status` never claims to be paired ────────────────────────────────────


def test_status_still_reports_unbound_in_demo_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, demo_on: None, recording_cloud: Any
) -> None:
    """The one lie that would matter, asserted at the wire.

    An agent that believes it is bound will try to write, and will read every
    synthetic number afterwards as its owner's real health record. `demo_mode`
    rides ON TOP of the real facts; `initialized` and `bound` keep reporting the
    actual config, which here is neither.

    `next_step` is checked too — it is what an evaluator reads to find out how to
    get real data, and demo mode must not replace it with reassurance.
    """

    server = _build_server(tmp_path, monkeypatch)
    status = _call(server, "vaultbeat_status").structuredContent

    assert status["demo_mode"] is True
    assert status["bound"] is False
    assert status["initialized"] is False
    assert status["demo_write_tools_available"] is False
    assert "bind" in status["next_step"]


def test_the_real_binding_facts_win_over_anything_the_demo_block_says(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, demo_on: None
) -> None:
    """Precedence, pinned — because today it holds for a reason that could evaporate.

    `status()` merges `demo_status()` with the real config, and the safety of
    that merge rests on TWO things: the demo dict is spread first, AND it happens
    to contain no `bound` / `initialized` key. Either alone is enough today, which
    is why neither is load-bearing enough for anyone to notice breaking it — a
    mutation that reversed the spread order changed no observable behaviour
    (verified 2026-08-20), so the ordering is currently protected by an absence
    rather than by a rule.

    This forces the question: a demo block that DOES claim to be bound must lose.
    Whichever of the two properties someone removes, this goes red.
    """

    real = demo_module.demo_status()
    # Property one, asserted directly: the demo dict does not carry these keys at
    # all. Cheap, and the monkeypatch below cannot see it — patching `demo_status`
    # replaces exactly the thing a "demo_status grew a bound key" regression would
    # change (verified 2026-08-20: that mutation survives the merge assertion).
    assert "bound" not in real
    assert "initialized" not in real

    # Property two: even if it did, the real config must still win.
    # `**_` rather than a bare `lambda:` — `demo_status` takes a keyword-only
    # `bound` since 2026-08-20, and a signature-exact stub would have to be
    # updated again the next time it grows one. Swallowing kwargs keeps this
    # test about PRECEDENCE, which is what it is for.
    monkeypatch.setattr(
        demo_module,
        "demo_status",
        lambda **_: {**real, "bound": True, "initialized": True},
    )

    server = _build_server(tmp_path, monkeypatch)
    status = _call(server, "vaultbeat_status").structuredContent

    assert status["bound"] is False, "a demo block must never be able to claim a binding"
    assert status["initialized"] is False


# ── 8. `doctor` reports only what it checked ─────────────────────────────────


def test_doctor_does_not_report_a_round_trip_it_never_made(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, demo_on: None, recording_cloud: Any
) -> None:
    """A green tick is evidence, and demo mode has none to offer about the network.

    The tempting shape is to keep the real check list and mark `data_roundtrip`
    OK — the report then looks healthy while no request was made, and a reader
    takes that tick as evidence about a path nobody exercised. So the skipped
    checks move OUT of the OK/FAIL list a client renders, into `not_checked`,
    each saying why.

    Asserted as absence from `checks`, not merely presence in `not_checked`:
    a version that reported it in both places would pass the weaker test while
    still showing the reader a green tick.
    """

    server = _build_server(tmp_path, monkeypatch)
    report = _call(server, "vaultbeat_doctor").structuredContent

    assert report["demo_mode"] is True
    checked = {check["name"] for check in report["checks"]}
    assert checked == {"demo_mode", "demo_dataset"}

    for skipped in ("data_roundtrip", "cloud_reachable", "bound", "identity_key", "config"):
        assert skipped not in checked, f"{skipped} must not appear as a check"
        assert skipped in report["not_checked"], skipped

    assert "NOT ATTEMPTED" in report["not_checked"]["data_roundtrip"]
    # Derived from the demo dataset through the same code path the real report
    # uses, so this is a genuine per-kind count rather than a second hand-written
    # answer that could disagree with the data.
    assert report["capabilities"]["kinds_without_data"] == []
    # It also must not claim a key it does not have.
    assert "nowhere" in report["scope"]["private_key_location"].lower()
    assert recording_cloud.calls == []


# ── 9. Demo mode is decided ONCE, and cannot change under a running server ───


def test_the_wrapper_and_the_service_share_one_demo_decision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, demo_on: None
) -> None:
    """Whether a tool is WRAPPED and whether the service SERVES demo data are
    two switches, and they used to be read from the environment independently.

    Nothing kept them in step, and they controlled opposite halves of the same
    lie: the wrapper owns the watermark, the description prefix and the server
    name, while the service owns the actual records. This asserts they come from
    one value, so no future flip can move one without the other.
    """

    import vaultbeat_mcp_local.mcp_server as mcp_module

    built: list[Any] = []

    class _Spy(service_module.VaultbeatLocalService):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            built.append(self)

    # `mcp_server` does `from ... import VaultbeatLocalService`, so the name to
    # replace is the one in ITS namespace, not the service module's.
    monkeypatch.setattr(mcp_module, "VaultbeatLocalService", _Spy)
    server = _build_server(tmp_path, monkeypatch)

    assert len(built) == 1, "exactly one service is constructed per server"
    service = built[0]

    # One value, two consumers. The wrapper is only installed when demo mode is
    # on, so "is this tool wrapped" and "does the service serve demo data" must
    # agree — that agreement is what a mid-session flip used to break.
    tool_fn = server._tool_manager.get_tool("get_activity").fn
    assert hasattr(tool_fn, "__wrapped__"), "demo mode must install the wrapper"
    assert service._demo is True

    result = _call(server, "get_activity").structuredContent
    assert result["demo_mode"] is True


def test_flipping_demo_on_mid_session_refuses_instead_of_serving_unmarked_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Direction A: started real, `VAULTBEAT_DEMO` set afterwards.

    Before 2026-08-20 this served SYNTHETIC RECORDS THROUGH AN UNWATERMARKED
    WRAPPER — owner id `demo0001-…`, no banner, no description prefix, while
    `status` and `doctor` went on correctly reporting demo_mode. The surface
    that leaves the session was the one that had stopped saying so.
    """

    service = service_module.VaultbeatLocalService(
        ConfigStore(tmp_path / "config.json"), demo=False
    )
    monkeypatch.setenv(DEMO_ENV, "1")

    with pytest.raises(RuntimeError) as excinfo:
        asyncio.run(service.sync_decrypted_records(metric_type="activity"))

    # The variable is NAMED: a bare "state changed" leaves the reader hunting.
    assert DEMO_ENV in str(excinfo.value)
    assert "restart" in str(excinfo.value).lower()


def test_flipping_demo_off_mid_session_refuses_instead_of_mislabelling_real_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Direction B, and the worse of the two on a bound machine.

    The wrapper stays frozen ON while the service starts serving REAL records,
    so a successful real write comes back stamped `demo_mode: true` +
    "[SYNTHETIC DEMO DATA]". An agent reading that concludes nothing was written
    — and either tells the user so, or retries, duplicating a real health record.
    """

    monkeypatch.setenv(DEMO_ENV, "1")
    service = service_module.VaultbeatLocalService(ConfigStore(tmp_path / "config.json"))
    assert service._demo is True

    monkeypatch.delenv(DEMO_ENV)
    with pytest.raises(RuntimeError) as excinfo:
        asyncio.run(service.sync_decrypted_records(metric_type="activity"))
    assert DEMO_ENV in str(excinfo.value)


def test_two_services_with_different_modes_coexist_in_one_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The constructor argument IS the test-isolation mechanism.

    A process-level latch would have frozen demo mode for the whole interpreter,
    so the first test to touch it would decide for every test after it and the
    suite's result would depend on collection order. This is the property that
    made the latch the wrong answer, so it is worth pinning rather than merely
    having been argued.
    """

    real = service_module.VaultbeatLocalService(ConfigStore(tmp_path / "real.json"), demo=False)
    fake = service_module.VaultbeatLocalService(ConfigStore(tmp_path / "fake.json"), demo=True)

    assert real._demo is False
    assert fake._demo is True

    records, _ = asyncio.run(fake.sync_decrypted_records(metric_type="activity"))
    assert records, "the demo service should serve synthetic records"
    assert str(records[0].owner_user_id).startswith("demo")
    # And the real one is untouched by the other's existence: unbound, so it
    # raises rather than quietly borrowing the demo dataset.
    with pytest.raises(Exception):
        asyncio.run(real.sync_decrypted_records(metric_type="activity"))


# ── 10. The warning is the FIRST thing in the payload ───────────────────────


_ALL_DEMO_TOOLS = (
    "vaultbeat_status",
    "vaultbeat_doctor",
    "get_activity",
    "get_sleep_detail",
    "get_basal_energy",
    "get_total_energy_burned",
    "log_note",
)


@pytest.mark.parametrize("tool_name", _ALL_DEMO_TOOLS)
def test_every_demo_result_leads_with_the_warning_sentence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, demo_on: None, tool_name: str
) -> None:
    """`demo_warning` must be the first key of the SERIALISED payload.

    Asserted on `content[0].text`, not on the dict: the dict is an intermediate
    nobody outside this process sees, and a test that checked it would keep
    passing if the serialisation ever stopped preserving order. `json.dumps`
    preserves insertion order today, which is exactly the fact under test.

    First key matters because `demo_mode: true` is a flag a reader has to
    already know to look for, while the banner is a sentence that acts on one
    who does not — and four separate surfaces build this block by hand
    (`_watermark_demo`, `_demo_write_refusal`, `doctor()`, `demo_status()`), so
    "we put it first" was true of one of them and not the others.

    ⚠️ Scope: this is the MCP wire, where the SDK serialises with `json.dumps`
    and insertion order survives. It is NOT a universal guarantee — `cli.py`'s
    `_print_json` sorts keys, so the CLI shows the banner alphabetically. That
    is fine (a human reading a terminal sees the whole payload), and it is the
    reason this asserts through the MCP handler rather than anywhere cheaper.
    """

    server = _build_server(tmp_path, monkeypatch)
    args = {"kind": "general", "text": "x"} if tool_name == "log_note" else {}
    result = _call(server, tool_name, args)

    pairs = json.loads(result.content[0].text, object_pairs_hook=list)
    assert pairs[0][0] == "demo_warning", f"{tool_name}: first key was {pairs[0][0]!r}"
    assert str(pairs[0][1]).startswith("[SYNTHETIC DEMO DATA]")


def test_the_banner_travels_under_exactly_one_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, demo_on: None
) -> None:
    """`demo_banner` is gone. `cli.py` renders `demo_warning`, so a second name
    is not a cosmetic split — it is a surface whose banner never prints."""

    server = _build_server(tmp_path, monkeypatch)
    for tool_name in ("vaultbeat_status", "vaultbeat_doctor", "get_activity"):
        structured = _call(server, tool_name).structuredContent
        assert "demo_banner" not in structured, tool_name
        assert "demo_warning" in structured, tool_name


# ── 11. A row lifted out of the payload still identifies itself ─────────────


def test_every_read_tool_marks_the_rows_it_returns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, demo_on: None
) -> None:
    """Generalised on purpose — naming the two known offenders would not cover
    the third aggregating tool somebody writes next.

    The top-level banner covers a result quoted whole; it does not survive one
    row being lifted out. Most tools self-identify via `owner_user_id`
    (`demo0001-…`), but `get_basal_energy` and `get_total_energy_burned` build
    fresh dicts and drop that field, so their rows were indistinguishable from
    real ones.
    """

    server = _build_server(tmp_path, monkeypatch)
    checked = 0
    for tool_name in (
        "get_basal_energy",
        "get_total_energy_burned",
        "get_activity",
        "get_sleep_detail",
    ):
        structured = _call(server, tool_name).structuredContent
        for key, value in structured.items():
            if not isinstance(value, list) or not value:
                continue
            rows = [row for row in value if isinstance(row, dict)]
            if not rows:
                continue
            checked += 1
            for row in rows:
                assert row.get("synthetic") is True, f"{tool_name}.{key} row unmarked"
    assert checked >= 4, "expected to have checked several row lists"


def test_row_marking_leaves_everything_else_exactly_as_it_was(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, demo_on: None
) -> None:
    """Add-only, one level deep, and blind to non-dict rows.

    `errors` is a list of strings and must not be touched; scalars must not be
    touched; and nested lists (a night's `stages`) are deliberately NOT recursed
    into — a stage array is not something anyone quotes as a standalone health
    fact, so marking it is noise.
    """

    from vaultbeat_mcp_local.mcp_server import _mark_demo_rows

    payload = {
        "count": 3,
        "average": 1.5,
        "errors": ["env-1: parse_failed"],
        "rows": [{"a": 1}, {"b": 2}],
        "nested": [{"outer": 1, "inner": [{"deep": True}]}],
    }
    marked = _mark_demo_rows(payload)

    assert marked["count"] == 3
    assert marked["average"] == 1.5
    assert marked["errors"] == ["env-1: parse_failed"]
    assert marked["rows"] == [{"a": 1, "synthetic": True}, {"b": 2, "synthetic": True}]
    assert marked["nested"][0]["inner"] == [{"deep": True}], "must not recurse"
    assert payload["rows"] == [{"a": 1}, {"b": 2}], "input must not be mutated"


# ── 12. The wording forbids silent persistence ──────────────────────────────


def test_demo_wording_tells_the_agent_not_to_persist_silently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, demo_on: None
) -> None:
    """A disclosure in the chat is gone in a week; an unlabelled fake row in the
    user's own notes is not.

    Both surfaces are asserted because they have different jobs and different
    budgets: the description is read once and can afford to be long, the banner
    rides every result and has to stay short. Asserting a verb AND a destination
    noun, so a future compression back to "Say so in any answer" goes red.
    """

    from vaultbeat_mcp_local.mcp_server import _DEMO_DOC_PREFIX

    assert "DO NOT" in _DEMO_DOC_PREFIX
    assert any(word in _DEMO_DOC_PREFIX for word in ("note", "file", "journal"))
    assert "[SYNTHETIC DEMO DATA]" in _DEMO_DOC_PREFIX

    assert DEMO_BANNER.startswith("[SYNTHETIC DEMO DATA]")
    assert "Do not save" in DEMO_BANNER

    # And the short one really does ride a result, rather than only existing.
    server = _build_server(tmp_path, monkeypatch)
    assert "Do not save" in _banner(_call(server, "get_activity").structuredContent)


# ── 13. `status` survives an unreadable key, and says where numbers come from ─


def _unreadable_key_config(tmp_path: Path) -> ConfigStore:
    """A config file that EXISTS but whose private key cannot be resolved.

    The autouse `fake_keychain` fixture makes the keyring an empty dict, so
    writing the JSON with no key material anywhere is enough — `load()` raises
    ConfigError, which is the real production shape (keyring moved, file
    deleted, env var not forwarded).
    """

    path = tmp_path / "config.json"
    path.write_text(
        json.dumps({"server_name": "Mac", "public_key_base64": "AAAA", "api_base_url": "x"}),
        encoding="utf-8",
    )
    return ConfigStore(path)


def test_status_survives_a_config_whose_key_cannot_be_read(tmp_path: Path) -> None:
    """`status` is the FIRST thing a stuck user runs, and it used to raise.

    Uncaught, the ConfigError escaped `_watermark_demo` (which only wraps
    returns) and surfaced as a bare ToolError — no binding facts, no demo
    stamp, nothing. `doctor` had already been hardened for exactly this on the
    grounds that a diagnostic may not decline to produce a diagnosis; `status`
    is on the same footing and was missed.
    """

    service = service_module.VaultbeatLocalService(_unreadable_key_config(tmp_path), demo=False)
    result = service.status()  # must not raise

    assert result["config_error"]
    assert result["bound"] is False
    # TRUE, and load-bearing: `initialized: False` would invite `bind`, and
    # minting a fresh identity is the one action that turns an unreadable key
    # into permanently unreadable data.
    assert result["initialized"] is True
    assert "not run `bind`" in result["next_step"] or "DO NOT" in result["next_step"]
    assert "bind" in result["next_step"]


def test_status_does_not_hand_a_demo_user_the_key_recovery_essay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, demo_on: None
) -> None:
    """Demo mode has no key and no encrypted records, so "delete this and your
    history becomes permanently unreadable" has no referent — it is a warning
    about a loss that cannot happen, handed to someone evaluating the product."""

    service = service_module.VaultbeatLocalService(_unreadable_key_config(tmp_path))
    result = service.status()

    blob = json.dumps(result)
    assert "permanently unreadable" not in blob
    assert "DO NOT DELETE" not in blob
    assert result["demo_mode"] is True
    assert DEMO_ENV in result["config_error"]


def test_status_says_where_the_numbers_came_from_even_on_a_bound_machine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Demo mode reads no config, so it runs happily on a PAIRED machine.

    There `bound: true` is a true answer to the wrong question — the binding is
    real, intact, and being bypassed — and the old `demo_note` made it worse by
    saying binding "still requires a real paired account", which reads as
    confirmation that the stored binding is in use. Every prior test asserted
    `bound is False` because every fixture happened to be unbound, so this
    branch had never been exercised at all.
    """

    store = ConfigStore(tmp_path / "config.json")
    store.ensure_initialized(server_name="Mac Studio", api_base_url="https://api.test")
    # Bind by writing the two fields `is_bound` reads, rather than driving a
    # faked poll: this test is about what `status` REPORTS for a bound machine,
    # and a fake cloud would only add a way for the fixture to stop being bound.
    store.update(server_id="srv-abc123", server_token="tok", bound_at="2026-08-01T00:00:00Z")
    assert store.load().is_bound, "fixture must actually be bound"

    setup = service_module.VaultbeatLocalService(store, demo=False)
    real_status = setup.status()
    assert real_status["data_source"] == "account"
    assert "demo_mode" not in real_status

    monkeypatch.setenv(DEMO_ENV, "1")
    demo_module.reset_cache()
    demo_service = service_module.VaultbeatLocalService(store)
    result = demo_service.status()

    assert result["bound"] is True, "the binding is real and must keep reporting so"
    assert result["demo_mode"] is True
    assert result["data_source"] == "synthetic"
    # The note must say the binding is being bypassed, not that it is required.
    assert "BYPASS" in result["demo_note"].upper()


def test_the_demo_decision_is_read_once_and_passed_down(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`run_mcp_server` must READ the environment once and hand the answer to the
    service — not let the service read it again for itself.

    Two independent reads agree on any ordinary startup, which is exactly why
    this needed a test written against something other than an ordinary startup:
    a mutation removing `demo=demo_active` from the constructor call left the
    whole suite green (verified 2026-08-20), because both reads saw the same
    variable microseconds apart. The property is real anyway — it is what makes
    the wrapper and the service structurally incapable of disagreeing — so it is
    pinned here with a `demo_enabled` that answers True once and False after.

    Under "read once, pass down" the service still gets True. Under two reads it
    gets False while the tools are already wrapped for a demo: the exact split
    that served synthetic records through an unwatermarked wrapper.
    """

    import vaultbeat_mcp_local.mcp_server as mcp_module

    calls = {"n": 0}

    def flip_flop() -> bool:
        calls["n"] += 1
        return calls["n"] == 1

    # Patched in BOTH namespaces so the counter is shared: `mcp_server` binds the
    # name at import, the service resolves it from the demo module per call.
    monkeypatch.setattr(mcp_module, "demo_enabled", flip_flop)
    monkeypatch.setattr(demo_module, "demo_enabled", flip_flop)

    built: list[Any] = []

    class _Spy(service_module.VaultbeatLocalService):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            built.append(self)

    monkeypatch.setattr(mcp_module, "VaultbeatLocalService", _Spy)
    server = _build_server(tmp_path, monkeypatch)

    assert calls["n"] >= 2, "expected the stub to be consulted more than once"
    wrapper_installed = hasattr(server._tool_manager.get_tool("get_activity").fn, "__wrapped__")
    assert wrapper_installed is True, "the first read said demo; the tools must be wrapped"
    assert built[0]._demo is True, (
        "the service re-read the environment instead of being handed the decision"
    )
