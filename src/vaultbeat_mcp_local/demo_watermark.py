"""Stamp the synthetic-data marker onto a demo result.

🔀 **There is ONE exit now (2026-09-12, Invariant 81 (health-data-has-one-exit))**:

    MCP tool  →  _demo_wrap  →  MCPServer  →  the agent

The CLI half of this module's reason for existing is gone with the fifteen data
subcommands — `_emit_decrypted` no longer exists, so no health payload leaves
this process except through a tool. **The history below is kept because it is
the argument for that removal, not a description of the code.**

Until 2026-08-27 there were two exits and the marker existed only on the first.
`mcp_server.py` owned `_watermark_demo`, so the CLI could not reach it without
importing a sibling transport — and `cli.py` deliberately keeps `mcp_server`
behind a lazy import inside `handle_serve` so that the MCP SDK's import chain is
not paid by every invocation. (`handle_serve`'s own comment puts that chain at
~1.4s; this file does not restate the figure, having measured only the
structural half — importing `cli` loads no `mcp.*` module, and this module keeps
it that way.) So the marker did not spread; it stayed where it was written, and
`--demo sleep` emitted a payload whose only tell was the `demo0001-` owner
prefix, i.e. a tell that only works on a reader who already knows the answer.

That is Invariant 61 (demo-is-a-boundary-not-a-flag) failing in the direction it
warns about — "every answer says it is synthetic" has to fall out of WHERE the
substitution happens, and one of the two exits was outside the boundary. It is
also the shape Invariant 58 (one-funnel-per-event) describes exactly: the
judgement did have a single home, just not one both callers could reach.

🔑 **Worth keeping in view: that defect was fixed by extracting this module, and
the fix held for sixteen days until the second exit was deleted outright.** The
watermark was one of three things that had to be remembered separately for the
CLI path (the others: the stdout-vs-stderr split of Invariant 71
(cli-data-stdout-is-json-only), and the `--output` file's wording). Each was
found and fixed on its own. Removing the exit removed the category.

⚠️ Why this is its own module rather than a section of `demo.py`, stated
honestly: `demo.py` is the natural home — it already owns `DEMO_BANNER`,
`DEMO_ENV`, the id prefixes and `demo_enabled()`, it imports no MCP SDK, and
merging this file into it would cost nothing and remove a hop. It was kept
separate only because this change landed alongside concurrent edits to that
file, and an append there risked a silent clobber. **Folding it in is correct
whenever someone is next editing `demo.py` anyway** — the production import
site count is one (`mcp_server.py`), down from two when `cli.py` still
serialised health payloads.

Nothing here reads the environment. Callers decide whether demo mode is on and
say so by calling; that keeps this file a pure formatter and keeps the decision
where it can be seen (`run_mcp_server` freezes it once at startup; `cli.py` no
longer calls `demo_enabled()` at all — it only sets `DEMO_ENV` in `main()`).
"""

from __future__ import annotations

from typing import Any, TypeGuard

from vaultbeat_mcp_local.demo import DEMO_BANNER

__all__ = ["mark_demo_rows", "watermark_demo_result"]


def watermark_demo_result(result: Any) -> Any:
    """Stamp a synthetic-data marker onto a tool result.

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
    function. Since 0.7.4 there is only one exit for health data and it is the
    MCP SDK, which dumps in insertion order — but `cli.py`'s remaining
    `_print_json` still passes `sort_keys=True`, under which `demo_warning`
    would sink into the middle of a payload. Nothing stamped reaches that
    printer today; a future caller that routes one through it is not wrong, but
    it has silently downgraded the marker to something you have to go looking
    for.

    A result that already carries `demo_mode` (`vaultbeat_doctor` and
    `_demo_write_refusal`, which build a richer block of their own — the old
    `vaultbeat_status` tool did too until 0.9.0 folded it into doctor's
    `binding`) is returned untouched rather than double-stamped — those order
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

    if _is_table(result):
        # `get_intraday` is itself one table (`columns` / `rows` at the top).
        result = _mark_table(result)
    marked: dict[str, Any] = {}
    for key, value in result.items():
        if isinstance(value, list) and any(isinstance(row, dict) for row in value):
            marked[key] = [_mark_row(row) if isinstance(row, dict) else row for row in value]
        else:
            marked[key] = value
    return marked


def _is_table(value: Any) -> TypeGuard[dict[str, Any]]:
    return isinstance(value, dict) and isinstance(value.get("columns"), list) and isinstance(value.get("rows"), list)


def _mark_table(table: dict[str, Any]) -> dict[str, Any]:
    """A `{columns, rows}` table (2026-10-02, the tool results that went tabular to
    fit a client): the mark is a `synthetic` column, so a row lifted out of it
    still carries the flag, as a dict row would."""
    if "synthetic" in table["columns"]:
        return table
    return {
        **table,
        "columns": [*table["columns"], "synthetic"],
        "rows": [[*row, True] if isinstance(row, list) else row for row in table["rows"]],
    }


#: The one place rows sit a level down: `get_metric` returns a list of series,
#: and each series holds the per-day `points` (or `buckets`) an agent actually
#: quotes. Those are health values lifted out one at a time, unlike a sleep
#: stage array, so they get the mark too. Named keys rather than recursion, for
#: the same noise reason the docstring above gives.
_NESTED_ROW_KEYS = ("points", "buckets")


def _mark_row(row: dict[str, Any]) -> dict[str, Any]:
    out = {**row, "synthetic": True}
    for key in _NESTED_ROW_KEYS:
        inner = out.get(key)
        if isinstance(inner, list):
            out[key] = [{**r, "synthetic": True} if isinstance(r, dict) else r for r in inner]
        elif _is_table(inner):
            out[key] = _mark_table(inner)
    return out
