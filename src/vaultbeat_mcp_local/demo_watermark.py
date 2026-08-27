"""Stamp the synthetic-data marker onto a demo result — for BOTH exits.

This server has two ways a health payload leaves the process, and they share no
code below the service layer:

    MCP tool  →  _demo_wrap  →  FastMCP  →  the agent
    CLI       →  _emit_decrypted        →  stdout, or a 0600 file

Until 2026-08-27 the marker existed only on the first one. `mcp_server.py` owned
`_watermark_demo`, so the CLI could not reach it without importing a sibling
transport — and `cli.py` deliberately keeps `mcp_server` behind a lazy import
inside `handle_serve` so that the MCP SDK's import chain is not paid by every
`vaultbeat-mcp sleep`. (`handle_serve`'s own comment puts that chain at ~1.4s;
this file does not restate the figure, having measured only the structural
half — importing `cli` loads no `mcp.*` module, and this module keeps it that
way.) So the marker did not spread; it stayed where it was written. `vaultbeat-mcp --demo sleep` emitted a payload whose only tell was the
`demo0001-` owner prefix, i.e. a tell that only works on a reader who already
knows the answer.

That is Invariant 61 (demo-is-a-boundary-not-a-flag) failing in the direction it
warns about — "every answer says it is synthetic" has to fall out of WHERE the
substitution happens, and one of the two exits was outside the boundary. It is
also the shape Invariant 58 (one-funnel-per-event) describes exactly: the
judgement did have a single home, just not one both callers could reach.

⚠️ Why this is its own module rather than a section of `demo.py`, stated
honestly: `demo.py` is the natural home — it already owns `DEMO_BANNER`,
`DEMO_ENV`, the id prefixes and `demo_enabled()`, it imports no MCP SDK, and
merging this file into it would cost nothing and remove a hop. It was kept
separate only because this change landed alongside concurrent edits to that
file, and an append there risked a silent clobber. **Folding it in is correct
whenever someone is next editing `demo.py` anyway** — the import site count is
two.

Nothing here reads the environment. Callers decide whether demo mode is on and
say so by calling; that keeps this file a pure formatter and keeps the decision
where it can be seen (`run_mcp_server` freezes it once at startup, `cli.py` asks
`demo_enabled()` at the one point a payload is serialised).
"""

from __future__ import annotations

from typing import Any

from vaultbeat_mcp_local.demo import DEMO_BANNER

__all__ = ["mark_demo_rows", "watermark_demo_result"]


def watermark_demo_result(result: Any) -> Any:
    """Stamp a synthetic-data marker onto a tool or subcommand result.

    Add-only and non-destructive, same discipline as `_annotate_if_empty`: it
    never reads, edits or drops an existing key. The marker goes FIRST in the
    dict so it survives a client that truncates a long payload from the end.

    🔴 `demo_warning` is the FIRST key, ahead of the boolean. The result is
    serialised with `json.dumps`, which preserves insertion order, so the first
    key is literally the first thing in the text a reader receives — and
    `"demo_mode": true` is a flag someone has to already know to look for,
    while the banner is a sentence that acts on a reader who does not. Ordering
    is free; which of the two goes first is not.

    ⚠️ That ordering guarantee is a property of the SERIALISER, not of this
    function, and the two exits disagree about it by default: the MCP SDK dumps
    with insertion order, while `cli.py` dumps with `sort_keys=True`, under
    which `count` beats `demo_warning` to the front. `cli.py` turns sorting off
    for stamped payloads for exactly this reason — see `_health_json` there. A
    future caller that sorts is not wrong, but it has silently downgraded the
    marker to something you have to go looking for.

    A result that already carries `demo_mode` (`vaultbeat_status`,
    `vaultbeat_doctor`, `_demo_write_refusal`, which build a richer block of
    their own) is returned untouched rather than double-stamped — those order
    their own keys the same way.
    """

    if not isinstance(result, dict) or "demo_mode" in result:
        return result
    return {
        "demo_warning": DEMO_BANNER,
        "demo_mode": True,
        **mark_demo_rows(result),
    }


def mark_demo_rows(result: dict[str, Any]) -> dict[str, Any]:
    """Tag each row of a top-level list-of-dicts with `synthetic: True`.

    The top-level banner covers a result quoted whole; it does NOT survive a row
    being lifted out of it. Most read tools self-identify anyway because their
    rows carry `owner_user_id`, which in demo mode reads `demo0001-…`, but the
    two aggregating tools build brand-new dicts (`{"day": …, "basal_kcal": …}`)
    and drop that field on the way, so `get_basal_energy` and
    `get_total_energy_burned` were the only two whose rows were, taken alone,
    indistinguishable from real ones (verified 2026-08-20: 17 of 19 read tools
    self-identify, those two did not).

    Done HERE rather than in those two methods on purpose — one site, so a third
    aggregating tool is covered on the day it is written rather than on the day
    someone notices. It is also structurally absent from a real run: the callers
    only reach this while demo mode is on.

    One level deep, deliberately. Recursing would reach into `sleep`'s per-night
    `stages` and similar, which is noise for no gain — a stage array is not
    something anyone lifts out and quotes as a health fact. Non-dict rows
    (`errors` is a list of strings) are left alone.
    """

    marked: dict[str, Any] = {}
    for key, value in result.items():
        if isinstance(value, list) and any(isinstance(row, dict) for row in value):
            marked[key] = [
                {**row, "synthetic": True} if isinstance(row, dict) else row
                for row in value
            ]
        else:
            marked[key] = value
    return marked
