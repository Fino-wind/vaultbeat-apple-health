"""The `prompts/list` + `prompts/get` library.

Most of these assert PROPERTIES of the library rather than its contents, because
the contents will grow and the properties are what has to survive that. The one
that matters most is `test_every_prompt_carries_the_style_rules`: the whole
reason the safety layer is a concatenated constant is that a paragraph retyped
per entry rots per entry, and a test that only checked the constant's text would
not notice an entry that forgot to append it.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any

import mcp.types as mcp_types
import pytest

import vaultbeat_mcp_local.demo as demo_module
from vaultbeat_mcp_local.prompts import (
    ABSENCE,
    DEMO_PREFIX,
    PROMPTS,
    STYLE,
    render_prompt,
)

from mcp.server.fastmcp import FastMCP

from vaultbeat_mcp_local.mcp_server import run_mcp_server
from vaultbeat_mcp_local.store import ConfigStore


def _build_server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """Construct the real FastMCP server without letting it block on a transport.

    `run_mcp_server` ends in `mcp.run(transport="stdio")`, which never returns,
    and the instance is created inside that function and never handed back —
    patching `run` to capture `self` is the only hook.

    ⚠️ `test_demo.py` has the same six lines. Deliberately not shared: `tests/`
    is not a package (a cross-module import fails outright), and moving it to
    `conftest.py` would be a shared-file edit for a scaffold, not for a
    judgement. If a third file needs it, that is the moment to lift it.
    """

    captured: dict[str, Any] = {}
    monkeypatch.setattr(FastMCP, "run", lambda self, **kwargs: captured.__setitem__("mcp", self))
    run_mcp_server(ConfigStore(tmp_path / "config.json"), transport="stdio")
    return captured["mcp"]


# ── Shape of the library ────────────────────────────────────────────────────


def test_names_are_unique_and_client_safe() -> None:
    names = [p.name for p in PROMPTS]
    assert len(names) == len(set(names))
    for name in names:
        assert re.fullmatch(r"[a-z][a-z0-9_]*", name), name


def test_every_prompt_carries_the_style_rules() -> None:
    """The point of a shared constant is that nothing opts out of it silently.

    An entry that forgets `STYLE` is not a cosmetic gap: it is a prompt that
    invites a model to grade someone's sleep, and it looks exactly like the
    others in `prompts/list`.
    """

    for prompt in PROMPTS:
        assert STYLE in prompt.template, f"{prompt.name} does not append STYLE"


def test_the_absence_rule_reaches_every_prompt_that_reads_a_kind() -> None:
    """`ABSENCE` is required wherever an empty result is possible.

    The two exemptions are stated by name so adding a third is a decision
    somebody makes on purpose: `log_from_conversation` writes rather than reads,
    and `why_is_this_empty` IS the absence conversation — appending the rule
    there would have it recite its own instructions back at itself.
    """

    exempt = {"log_from_conversation", "why_is_this_empty"}
    for prompt in PROMPTS:
        has = ABSENCE in prompt.template
        if prompt.name in exempt:
            assert not has, f"{prompt.name} is exempt but appends ABSENCE"
        else:
            assert has, f"{prompt.name} reads data but does not append ABSENCE"


def test_prompts_never_name_a_tool_that_does_not_exist() -> None:
    """A prompt is the one surface that tells an agent what to call.

    A wrong name here costs a tool call and a confused apology, and unlike a
    docstring nobody reads it back against the code. Checked against the
    registered `def`s rather than a hand-kept list, for the reason
    `check_metric_type_contract.py` exists.
    """

    source = Path(__file__).resolve().parents[1] / "src" / "vaultbeat_mcp_local" / "mcp_server.py"
    defined = set(re.findall(r"^    (?:async )?def ([a-z0-9_]+)\(", source.read_text(), re.M))

    for prompt in PROMPTS:
        for referenced in re.findall(r"`((?:get|log|vaultbeat)_[a-z0-9_]+)`", prompt.template):
            assert referenced in defined, f"{prompt.name} names non-existent tool {referenced}"


def test_no_prompt_hands_the_model_a_threshold_to_judge_by() -> None:
    """No bands, grades or verdicts invented in the prompt layer.

    A number that enters here is indistinguishable downstream from one that came
    out of the data, and "good HRV" is not ours to define. The reference ranges
    in `get_vo2max`'s docstring are a different thing: those are cited, in the
    tool that owns the measurement, not asserted at a model as a rule.
    """

    banned = ("optimal", "unhealthy", "should be above", "should be below", "out of range", "score of")
    for prompt in PROMPTS:
        lowered = prompt.template.lower()
        for phrase in banned:
            assert phrase not in lowered, f"{prompt.name} contains judgement phrase {phrase!r}"


def test_every_declared_argument_appears_in_its_template() -> None:
    """And no template holds a placeholder that was never declared.

    The second half is the one that bites: an undeclared placeholder can never
    be filled, so it survives `render_prompt` into the model's context as
    literal text.
    """

    for prompt in PROMPTS:
        declared = {a.name for a in prompt.arguments}
        used = set(re.findall(r"\{\{([a-z_]+)\}\}", prompt.template))
        assert declared == used, f"{prompt.name}: declared {declared}, used {used}"


# ── Rendering ───────────────────────────────────────────────────────────────


def test_a_missing_argument_becomes_readable_text_not_a_hole() -> None:
    """A surviving `{{days}}` is not a neutral failure.

    The model reads it as literal content and either quotes it back or invents a
    binding for it — and both outcomes look like the prompt worked. Every
    argument is optional, so this is the ordinary path, not the error one.
    """

    for prompt in PROMPTS:
        rendered = render_prompt(prompt, {})
        assert "{{" not in rendered, f"{prompt.name} left a placeholder unfilled"
        assert "}}" not in rendered, f"{prompt.name} left a placeholder unfilled"


def test_an_omitted_argument_leaves_a_grammatical_sentence() -> None:
    """The reason fallbacks are per-argument rather than one generic sentence.

    A single generic fallback has to fit every slot it is dropped into, and it
    does not: the first draft rendered "…looked out of date(unspecified context
    — pick a sensible one)". Correct by the rule, unreadable in place, and the
    reader it misleads is a language model.

    Checked crudely but usefully — no doubled spaces, no space before a comma or
    full stop, no empty parentheses — because those are what a badly-fitted
    substitution actually produces.
    """

    for prompt in PROMPTS:
        rendered = render_prompt(prompt, {})
        for defect in ("  ", " .", " ,", "()", "date(", "the  "):
            assert defect not in rendered, f"{prompt.name}: {defect!r} after substitution"


@pytest.mark.parametrize("blank", ["", "   ", None])
def test_a_blank_argument_is_treated_as_missing(blank: Any) -> None:
    """An empty string is a client sending nothing, not a value of nothing."""

    prompt = next(p for p in PROMPTS if p.arguments and p.arguments[0].fallback)
    argument = prompt.arguments[0]
    assert argument.fallback in render_prompt(prompt, {argument.name: blank})


def test_a_supplied_argument_is_substituted_verbatim() -> None:
    prompt = next(p for p in PROMPTS if p.name == "sleep_review")
    rendered = render_prompt(prompt, {"nights": "30"})
    assert "the 30 nights" in rendered
    assert prompt.arguments[0].fallback not in rendered


# ── Wired into the real server ──────────────────────────────────────────────


def _list_prompts(server: Any) -> list[mcp_types.Prompt]:
    handler = server._mcp_server.request_handlers[mcp_types.ListPromptsRequest]
    request = mcp_types.ListPromptsRequest(method="prompts/list")
    return list(asyncio.run(handler(request)).root.prompts)


def _get_prompt(server: Any, name: str, arguments: dict[str, str] | None = None) -> str:
    handler = server._mcp_server.request_handlers[mcp_types.GetPromptRequest]
    request = mcp_types.GetPromptRequest(
        method="prompts/get",
        params=mcp_types.GetPromptRequestParams(name=name, arguments=arguments),
    )
    result = asyncio.run(handler(request)).root
    return "".join(
        m.content.text for m in result.messages if isinstance(m.content, mcp_types.TextContent)
    )


def test_the_server_actually_lists_them(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Through the protocol handler, not through our own table.

    Asserting on `PROMPTS` would keep passing if `register_prompts` were never
    called — which is the state this whole file exists to leave behind.
    """

    listed = _list_prompts(_build_server(tmp_path, monkeypatch))
    assert {p.name for p in listed} == {p.name for p in PROMPTS}
    for entry in listed:
        assert entry.description
        for argument in entry.arguments or []:
            assert argument.required is False


def test_prompts_get_returns_the_rendered_text(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    server = _build_server(tmp_path, monkeypatch)
    text = _get_prompt(server, "energy_balance", {"days": "21"})
    assert "over the 21 days" in text
    assert STYLE.strip() in text


def test_prompts_get_fills_omitted_arguments(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A client is allowed to omit every argument; none of ours are required."""

    text = _get_prompt(_build_server(tmp_path, monkeypatch), "energy_balance")
    assert "{{" not in text
    assert "over the last 14 days" in text


def test_demo_mode_says_so_in_the_prompt_itself(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`prompts/list` is readable before a single tool is called.

    Invariant 61 (demo-is-a-boundary-not-a-flag) asks that every surface say it
    is synthetic; a prompt that reads as a real instruction while the server
    underneath is a demo is one more place the label can be lost.
    """

    monkeypatch.setenv(demo_module.DEMO_ENV, "1")
    demo_module.reset_cache()
    text = _get_prompt(_build_server(tmp_path, monkeypatch), "daily_brief")
    assert text.startswith(DEMO_PREFIX)
    assert "[SYNTHETIC DEMO DATA]" in text


def test_a_real_server_does_not_prefix_the_prompts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    text = _get_prompt(_build_server(tmp_path, monkeypatch), "daily_brief")
    assert not text.startswith(DEMO_PREFIX)
    assert "SYNTHETIC" not in text
