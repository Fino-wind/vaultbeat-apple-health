from __future__ import annotations

import functools
from datetime import date as _date
import hmac
import inspect
import ipaddress
import json
from typing import Any, Callable, TypeVar, cast

from vaultbeat_mcp_local import __version__
from vaultbeat_mcp_local.app_paths import HEALTH_ACCESS, RESYNC
from vaultbeat_mcp_local.demo import demo_enabled
from vaultbeat_mcp_local.demo_watermark import watermark_demo_result
from vaultbeat_mcp_local.prompts import register_prompts, server_instructions
from vaultbeat_mcp_local.service import PARTNER_EMPTY_HINT, VaultbeatLocalService
from vaultbeat_mcp_local.store import ConfigStore

_F = TypeVar("_F", bound=Callable[..., Any])


# ── Tool annotations ─────────────────────────────────────────────────────────
#
# `ToolAnnotations` is the MCP spec's per-tool behaviour hint block. It travels
# in `list_tools` and is what lets a client decide whether a call needs a
# confirmation prompt BEFORE running it — a static promise about the tool, not a
# per-call one, which is why anything with two modes has to be annotated for its
# worst mode (see `log_food_entry` vs `log_food_append`).
#
# The import is function-local on purpose. `run_mcp_server` deliberately imports
# MCPServer lazily so a missing SDK produces a sentence instead of a traceback; a
# module-level `from mcp.types import ...` here would raise first and make that
# whole guard dead code.


#: Shipped with the series catalog. Says the one thing the list itself cannot:
#: that the ABSENT kinds are absent on purpose, so an agent does not read a short
#: list as a gap in the product and go looking for `get_workout_trend`.
_SERIES_CATALOG_NOTE = (
    "These are the quantities that are genuinely one number per day, which is what "
    "trend / compare / correlate need. Sleep is here as numbers (duration, bedtime, "
    "wake, stage minutes and shares, awakenings, sleep vitals); the night-by-night "
    "picture — naps, recording source, unworn or stage-less nights — is "
    "`get_sleep_nights`. Kinds that are not here at all (workouts, strength sets, "
    "food, notes, symptoms, menstrual cycle) are richer than one number and are read "
    "with their own `get_*` tool — their absence from this list is a design choice, "
    "not missing data."
)


def _read_only_tool(*, open_world: bool = False) -> Any:
    """A tool that only reads. Safe to call, safe to repeat, changes nothing.

    `idempotentHint=True` is about EFFECT, not about the answer: a later call
    may return newer numbers because the account moved on, but nothing changed
    because this tool ran. Some of these do touch local state — the plaintext
    cache, `last_sync_at`, a decrypt-failure report — and that is still
    read-only in the sense a client cares about: no user data is created,
    modified or destroyed. Annotating them all `False` for that would collapse
    the read/write distinction to zero signal, which is the one thing these
    hints exist to carry.

    `open_world=True` for the single tool that talks to a host outside this
    system (`vaultbeat_doctor` asks PyPI for the current version).
    """

    from mcp.types import ToolAnnotations

    return ToolAnnotations(
        read_only_hint=True,
        destructive_hint=False,
        idempotent_hint=True,
        open_world_hint=open_world,
    )


def _mutating_tool(*, destructive: bool, idempotent: bool = False) -> Any:
    """A tool that changes something — the health record, or this binding.

    `destructive=True` means a call CAN remove or overwrite something that was
    already there. It is the spec's worst-case flag, so a tool with both a
    replace and an append mode gets `True` and the append-only sibling gets
    `False` — that split is the entire reason `log_*_append` exists as separate
    tools rather than a `merge=True` argument, because an argument cannot be
    annotated.
    """

    from mcp.types import ToolAnnotations

    return ToolAnnotations(
        read_only_hint=False,
        destructive_hint=destructive,
        idempotent_hint=idempotent,
        open_world_hint=False,
    )


# ── Demo mode ────────────────────────────────────────────────────────────────

#: Prepended to every tool description while demo mode is on.
#:
#: Deliberately in the DESCRIPTION and not only in the results: a description is
#: read once and stays in the model's context for the whole session, while a
#: per-result banner has to survive summarisation, truncation and the agent
#: paraphrasing three tool calls into one sentence. Both are used — this one so
#: the agent never forgets, the result stamp so a copy-pasted payload still
#: identifies itself.
_DEMO_DOC_PREFIX = (
    "⚠️ DEMO MODE — every number this tool returns is SYNTHETIC, generated locally, "
    "and belongs to no real person. Say so in any answer built on it, and DO NOT "
    "quietly persist it: if you are asked to write these numbers into a note, file, "
    "journal, spreadsheet, database, calendar, health app or another MCP server, say "
    "first that they are synthetic and get confirmation — and if you do write them, "
    "label the entry [SYNTHETIC DEMO DATA] in the file itself. A disclosure in this "
    "chat is gone in a week; an unlabelled fake row in the user's own notes is not. "
    "Nothing is fetched from the cloud and nothing is decrypted."
)


# `_watermark_demo` / `_mark_demo_rows` used to live here. They moved to
# `demo_watermark.py` on 2026-08-27 because the CLI's data subcommands NEEDED the
# same stamp and could not import this module — `cli.py` keeps `mcp_server` behind
# a lazy import inside `handle_serve` precisely so the MCP SDK's import chain stays
# off the data path (verified: importing `cli` loads no `mcp.*` module). A
# judgement with one home that only one of two callers can reach is the
# Invariant 58 (one-funnel-per-event) failure, not the fix.
#
# ⚠️ Past tense since 0.7.4: those subcommands are gone (Invariant 81
# (health-data-has-one-exit)), so this module is the only importer left. The
# split is KEPT anyway — the lazy-import property it protects is still real, and
# re-inlining it would have to be undone by the next caller that needs the stamp
# outside the SDK's import chain.


def _demo_wrap(function: _F) -> _F:
    """Wrap one tool so its output is watermarked and its docstring says DEMO.

    Two shapes, because a tool may be either: plain `def` or `async def` (all
    current tools are async; the sync branch stays for the next one that is not).
    A single sync wrapper around a coroutine function would hand MCPServer a
    coroutine object as the tool's result — the tool would "succeed" and return
    something unserialisable.

    `functools.wraps` is load-bearing beyond cosmetics: MCPServer builds each
    tool's JSON schema from `inspect.signature(fn)`, which follows the
    `__wrapped__` attribute wraps sets — so the published schema stays the real
    function's, not `(*args, **kwargs)`.
    """

    if inspect.iscoroutinefunction(function):

        @functools.wraps(function)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            return watermark_demo_result(await function(*args, **kwargs))

        _prefix_doc(async_wrapper, function)
        return cast(_F, async_wrapper)

    @functools.wraps(function)
    def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        return watermark_demo_result(function(*args, **kwargs))

    _prefix_doc(sync_wrapper, function)
    return cast(_F, sync_wrapper)


def _prefix_doc(wrapper: Any, original: Any) -> None:
    """Put the demo warning at the top of the description MCPServer will publish."""

    doc = inspect.getdoc(original) or ""
    wrapper.__doc__ = f"{_DEMO_DOC_PREFIX}\n\n{doc}" if doc else _DEMO_DOC_PREFIX


# ── Trial-expiry note (vb-016) ───────────────────────────────────────────────


def _annotate_access(result: Any, service: VaultbeatLocalService) -> Any:
    """Attach a one-line trial-expiry heads-up to a tool result, add-only.

    Same discipline as `_annotate_if_empty` / `watermark_demo_result`: never reads,
    edits or drops an existing key, and does nothing at all outside the last
    24 hours of a pairing-time trial deadline. The sentence is generated
    entirely client-side (Anti-pattern 23) from a timestamp this machine
    already holds — the point is that the agent can say "your trial ends
    tomorrow" BEFORE the first notice the user gets is a mid-conversation
    refusal.

    Results that already carry an `access` block are left alone — they say the
    same thing with more context. Since 0.9.0 the only such result is
    `vaultbeat_doctor`, and its block sits one level down at `binding.access`
    (the old `vaultbeat_status` tool, which had it at the top, became that
    `binding` key); checking only the top level made doctor print the expiry
    twice in the trial's last 24 hours. `access_note_if_expiring` reads an
    in-process stash set by the bound call the tool just made, so this costs no
    config or Keychain I/O.
    """

    if not isinstance(result, dict) or "access" in result or "access_note" in result:
        return result
    binding = result.get("binding")
    if isinstance(binding, dict) and "access" in binding:
        return result
    note = service.access_note_if_expiring()
    if note is None:
        return result
    return {**result, "access_note": note}


def _access_wrap(function: _F, service: VaultbeatLocalService) -> _F:
    """Wrap one tool so its dict results pick up the trial-expiry note.

    Mirrors `_demo_wrap`'s two shapes (this server has both sync and async
    tools) including `functools.wraps`, which MCPServer's schema builder relies
    on. Applied to every tool at the registration choke point rather than per
    call site — an annotation added at 29 call sites is 29 chances to forget
    the one that matters.
    """

    if inspect.iscoroutinefunction(function):

        @functools.wraps(function)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            return _annotate_access(await function(*args, **kwargs), service)

        return cast(_F, async_wrapper)

    @functools.wraps(function)
    def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        return _annotate_access(function(*args, **kwargs), service)

    return cast(_F, sync_wrapper)


#: `get_sleep_nights` columns: (output name, row field). Order is the order an
#: analyst reads a night in — when, how long, how it was built, how it held.
_SLEEP_NIGHT_COLUMNS: tuple[tuple[str, str], ...] = (
    ("date", "local_date"), ("dow", "weekday"), ("src", "source"), ("bed", "bedtime"), ("wake", "wake_time"),
    ("asleep", "asleep_minutes"), ("deep", "deep_minutes"), ("rem", "rem_minutes"),
    ("core", "core_minutes"), ("awake", "awake_minutes"), ("deep_pct", "deep_percent"),
    ("rem_pct", "rem_percent"), ("wakeups", "awakenings"), ("longest_bout", "longest_sleep_bout_minutes"),
    ("hr", "sleep_hr_mean"), ("hr_min", "sleep_hr_min"), ("rr", "sleep_rr_mean"),
    ("other", "other_sleep"), ("segs", "sleep_segments"),
)

_OTHER_COL = [col for col, _field in _SLEEP_NIGHT_COLUMNS].index("other")


def _other_sleep_label(segments: Any) -> str | None:
    """`[{bedtime, wake_time, asleep_minutes}]` → "12:26-13:48 82m; ..." or None."""

    if not segments:
        return None
    parts = []
    for seg in segments:
        start = str(seg.get("bedtime") or "")[11:16]
        end = str(seg.get("wake_time") or "")[11:16]
        parts.append(f"{start}-{end} {seg.get('asleep_minutes')}m")
    return "; ".join(parts)


_SLEEP_NIGHT_LEGEND = {
    "date": "the local day the sleep ENDED on (a night from 23:40 Mon to 07:10 Tue is Tue)",
    "src": "who scored the night (`apple` = Apple's own sleep tracking; `motion-inferred` = "
           "guessed from phone motion, no wearable; others name the band or app). "
           "Different sources use different algorithms — see `sources`",
    "bed / wake": "local clock time of the night's MAIN sleep — occasionally a daytime one, "
                  "when that was the longest sleep of the day",
    "asleep, deep, rem, core, awake": "minutes; `awake` is time scored awake inside the sleep",
    "deep_pct / rem_pct": "share of `asleep`, 0-100",
    "wakeups": "awake intervals between first and last asleep sample (waking for the day not counted)",
    "longest_bout": "longest run of sleep with no awake interval, minutes",
    "hr / hr_min / rr": "heart rate mean / min (bpm) and respiratory rate (breaths/min) while asleep; "
                        "null when too few samples were taken that night to stand for it",
    "other": "other measured sleep that day, NOT counted in `asleep`: naps and the other half of a "
             "broken night, as \"HH:MM-HH:MM Nm\". Add it to `asleep` for the day's total sleep",
    "segs": "how many separate sleeps that day (1 = one unbroken main sleep)",
    "flag": "comma-separated, a night can carry several. "
            "`unworn` = the Watch was not worn, sleep was never measured — NOT zero sleep; "
            "`motion_inferred` = guessed from phone motion, runs long; `get_metric` leaves these "
            "nights out of every sleep average, so leave them out of any average you compute here; "
            "`no_stages` = total known but no stage breakdown (Apple does not stage a short sleep, "
            "and some sources never do), so stage columns are null; "
            "`short` = a main sleep under 3 h with no stages — a nap or the part of a night the Watch "
            "caught, which look the same; `get_metric` leaves it out of `sleep_minutes`, "
            "`sleep_24h_minutes` and the bedtime/wake/midpoint series; "
            "`daytime` = the day's only sleep began after 08:00 and ended the same day — a daytime "
            "or evening nap, not a night: `get_metric` leaves it out of every nightly series and "
            "lists it, and it stays here as a row. Naps on a day that also had a night are in `other`",
    "null": "not measured. Never read a null as zero",
}


def _sleep_nights_table(summary: dict[str, Any], since: str | None) -> dict[str, Any]:
    """`sleep_nights` rows → one header plus one short row per night.

    Named columns once, values after: a year of nights is ~40k characters this
    way against ~200k as objects and ~500k through `get_sleep_detail`, which
    is the difference between an agent reading the whole history and reading
    the last fortnight while believing it read everything.
    """

    nights = summary.get("nights") or []
    rows = []
    for n in nights:
        # Not exclusive, and every flag that `get_metric` excludes on is shown:
        # an agent averaging the `asleep` column itself must be able to see the
        # nights `get_metric` left out, or the two answers disagree with nothing
        # on the page saying why (2026-09-25: motion-inferred nights were only
        # visible through `src`, and a daytime stage-less night showed one flag).
        flags = [
            name for name, on in (
                ("unworn", n.get("is_in_bed_only")),
                ("motion_inferred", n.get("motion_inferred")),
                ("daytime", n.get("daytime_main_sleep")),
                ("short", n.get("short_unstaged")),
                ("no_stages", n.get("no_stage_detail") and not n.get("is_in_bed_only")),
            ) if on
        ]
        flag = ",".join(flags) or None
        row = [n.get(field) for _col, field in _SLEEP_NIGHT_COLUMNS]
        row[_OTHER_COL] = _other_sleep_label(n.get("other_sleep"))
        rows.append(row + [flag])
    out: dict[str, Any] = {
        "columns": [col for col, _field in _SLEEP_NIGHT_COLUMNS] + ["flag"],
        "rows": rows,
        "count": len(rows),
        "legend": _SLEEP_NIGHT_LEGEND,
    }
    if since:
        out["since"] = since
    for key, value in summary.items():
        if key not in ("nights", "count"):
            out[key] = value
    return out


def _rows_table(rows: list[Any]) -> dict[str, Any]:
    """`[{date, value, partial?}, ...]` → `{columns, rows}`: each key once, then values.

    A month of every series as objects was 80k characters on the demo account,
    two thirds of it the same three key names repeated per day; as a table the
    same answer fits a client's per-result cap. Columns are the union of the
    rows' keys in first-seen order, so an optional flag (`partial`) is a column
    that is `null` on the days without it.
    """
    columns: list[str] = []
    for row in rows:
        if isinstance(row, dict):
            columns.extend(key for key in row if key not in columns)
    return {
        "columns": columns,
        "rows": [[row.get(column) for column in columns] for row in rows if isinstance(row, dict)],
    }


def _metric_tables(result: dict[str, Any]) -> dict[str, Any]:
    """`get_metric`'s per-series `points` / `buckets` lists, as `_rows_table` tables."""
    metrics = result.get("metrics")
    if not isinstance(metrics, list):
        return result
    tabled = []
    for entry in metrics:
        if isinstance(entry, dict):
            entry = {
                key: _rows_table(value) if key in ("points", "buckets") and isinstance(value, list) else value
                for key, value in entry.items()
            }
        tabled.append(entry)
    return {**result, "metrics": tabled}


#: `get_intraday` columns. The rest of a record repeats per row what the row
#: already says (`record_id`, the UTC `date`, `local_date`, `avg_sdnn_ms` beside an
#: equal `sdnn_ms`, the owner): 280 characters a row against about 40.
_INTRADAY_COLUMNS = ("local_time", "sdnn_ms", "sample_count")


def _intraday_table(result: dict[str, Any]) -> dict[str, Any]:
    """`get_intraday` records → `{columns, rows}`, the owner stated once."""
    records = result.get("records")
    if not isinstance(records, list):
        return result
    dicts = [r for r in records if isinstance(r, dict)]
    columns = [c for c in _INTRADAY_COLUMNS if any(c in r for r in dicts)]
    out = {key: value for key, value in result.items() if key != "records"}
    if dicts and dicts[0].get("owner_user_id"):
        out["owner_user_id"] = dicts[0]["owner_user_id"]
    out["columns"] = columns
    out["rows"] = [[r.get(c) for c in columns] for r in dicts]
    return out


def _partner_note(result: dict[str, Any], partner: bool) -> dict[str, Any]:
    """Explain an empty PARTNER read as what it almost always is: not shared.

    Add-only, like `_annotate_if_empty`. Without it an empty partner result reads
    as "she did not sleep" / "no cycle recorded" — a claim about a person's body
    made from what is really a privacy toggle in their app.
    """

    if not partner or not isinstance(result, dict):
        return result
    coverage = result.get("coverage")
    if isinstance(coverage, dict) and coverage.get("days_covered") == 0:
        result.setdefault("partner_note", PARTNER_EMPTY_HINT)
    return result


def _owner_unknown_note(
    result: dict[str, Any], service: VaultbeatLocalService, partner: bool
) -> dict[str, Any]:
    """Say so when a notes/symptoms read could not be split into "you" and "partner".

    Both readers select a person by comparing against the account that paired
    this machine (Invariant 88 (partner-is-not-me)). A pre-2026 binding never
    recorded that account, so the comparison has nothing to compare with and
    both readers fall back to returning EVERYONE's rows — identically for
    `partner=true` and `false`. The other readers had `_attach_owner_guard`'s
    `mixed_owners` warning for that case; these two had nothing, so an agent
    asking for the partner's notes got the user's own notes back as the
    partner's, with no signal anywhere. Notes are the sharper half: a note the
    user's AI recorded ABOUT the partner lives in the user's own account, so even
    a single-owner result is ambiguous there.

    Add-only, and quiet on an empty result — an empty read mixes no one.
    """

    if not isinstance(result, dict) or service.person_owner() is not None:
        return result
    coverage = result.get("coverage")
    if isinstance(coverage, dict) and coverage.get("days_covered") == 0:
        return result
    result["owner_unknown"] = True
    result.setdefault(
        "warning",
        (
            "This pairing does not record whose account it is (it predates owner "
            "tracking), so this result holds EVERYONE's entries — yours and your "
            "partner's — and `partner` "
            + ("was ignored" if partner else "could not narrow it to you")
            + ". Do not attribute any entry to a person from this call alone; "
            "re-pair this machine from the Vaultbeat app to fix it."
        ),
    )
    return result


def _empty_window_hint(result: dict[str, Any], kind: str) -> str | None:
    """The hint for a read whose WINDOW was empty on an account that has data.

    `_annotate_if_empty` asks this first. Until 2026-10-03 it said "No food data
    on this account" for a fortnight with nothing logged, on an account with
    months of food (review R3 on #9). None when the read held no data at all.
    """
    coverage = result.get("coverage")
    if not isinstance(coverage, dict) or not (coverage.get("total_available") or 0) > 0:
        return None
    oldest = coverage.get("oldest_available")
    return (
        f"Nothing in the window you asked for, but this account does have {kind} "
        f"data outside it"
        + (f" (its records start on {oldest})" if oldest else "")
        + ". Move or widen the window; an empty window is not a missing record type."
    )


def _annotate_if_empty(result: dict[str, Any], kind: str, rows_key: str) -> dict[str, Any]:
    """Explain an empty result for a kind that older apps never wrote.

    An agent calling `get_strength_log` on an account whose app predates
    strength logging gets `{"sessions": []}` — indistinguishable from "you never
    trained" and from "something is broken". There is no other signal anywhere:
    no error, no warning. This attaches the reason to the result itself so the
    agent can relay it without having to know that a separate diagnostic exists.

    ADD-ONLY, deliberately: it introduces a `hint` key and never reads, edits or
    removes an existing field, and it does nothing at all when there IS data.
    Payload shape is the one contract layer no server can validate (see
    scripts/ci/check_payload_contract.py), so anything touching a response has to be additive.
    """
    if result.get(rows_key):
        return result
    if "error" in result:
        # A refused call (`invalid_since`, …) returned no rows because it read
        # nothing; telling the reader the account has no data would answer a
        # question the call never asked (Invariant 57).
        return result
    window_hint = _empty_window_hint(result, kind)
    if window_hint:
        result.setdefault("hint", window_hint)
        return result

    since = VaultbeatLocalService.KIND_MIN_APP_RELEASE.get(kind)
    if since is None:
        # A kind that has existed since 1.2.0, so "your app is too old" cannot be
        # the reason and naming it would send the reader down a dead end. The
        # other causes still apply though — and the first one especially, because
        # sleep is what a new user asks for first and a freshly paired server has
        # barely any of it yet. Before this, those kinds returned a bare empty
        # result with no explanation at all.
        result.setdefault(
            "hint",
            f"No {kind} data came back. This server cannot tell why — it only sees "
            f"that no rows arrived — but in the order worth checking: (1) this "
            f"server was paired recently and the history has not finished sealing "
            f"for it; every server gets its own encrypted copy, so a new one starts "
            f"nearly empty and fills in. Open the app and tap {RESYNC}, then "
            f"retry in a few minutes. "
            f"(2) Apple Health access for it was never granted — a read denial is "
            f"invisible to the app, so it looks identical to having no data; "
            f"recover via {HEALTH_ACCESS}. (3) it "
            f"genuinely has not been recorded yet. Run `vaultbeat-apple-health doctor` or "
            f"call the vaultbeat_doctor tool for a full report.",
        )
        return result

    result.setdefault(
        "hint",
        f"No {kind} data on this account. Four causes produce an identical empty "
        f"result and this server cannot tell them apart — it only sees that no rows "
        f"arrived, in the order worth checking: (1) this server was paired recently "
        f"and the history has not finished sealing for it — every server gets its own "
        f"encrypted copy, so a new one starts nearly empty and fills in; open the app "
        f"and tap {RESYNC}, then retry "
        f"in a few minutes; (2) the iOS app predates this data type, which needs a "
        f"build from {since} or later; (3) Apple Health access for it was never "
        f"granted — a read denial is invisible to the app, so this looks the same as "
        f"having no data, and the recovery path is {HEALTH_ACCESS} in the app, "
        f"which re-presents the permission sheet; (4) it "
        f"genuinely has not been recorded yet. Run `vaultbeat-apple-health doctor` or call the "
        f"vaultbeat_doctor tool for a full report.",
    )
    return result


_LOOPBACK_HOSTNAMES = {"localhost"}


def _normalize_transport(transport: str) -> str:
    normalized = transport.strip().lower()
    if normalized == "http":
        return "streamable-http"
    if normalized in {"stdio", "streamable-http"}:
        return normalized
    raise ValueError(f"Unsupported MCP transport: {transport}")


def _normalize_http_path(path: str) -> str:
    normalized = path.strip()
    if not normalized:
        return "/mcp"
    if not normalized.startswith("/"):
        return f"/{normalized}"
    return normalized


def _is_loopback(host: str) -> bool:
    """True only for hosts unreachable from other machines.

    Wildcard binds (0.0.0.0, ::) listen on every interface and are treated as
    non-loopback. Any hostname other than "localhost" that does not parse as an
    IP is conservatively treated as non-loopback (default-deny).
    """

    candidate = host.strip().lower()
    if candidate in _LOOPBACK_HOSTNAMES:
        return True
    try:
        return ipaddress.ip_address(candidate).is_loopback
    except ValueError:
        return False


class StaticBearerASGIMiddleware:
    """Pure-ASGI gate requiring ``Authorization: Bearer <token>`` on http requests.

    Non-http scopes (notably ``lifespan``, which starts the MCP session manager,
    and websocket) are forwarded verbatim so the wrapped Starlette app behaves
    exactly as if unwrapped. The token is compared in constant time.
    """

    def __init__(self, app: Any, token: str) -> None:
        self._app = app
        self._expected = f"Bearer {token}".encode("latin-1")

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self._app(scope, receive, send)
            return

        headers = dict(scope.get("headers") or [])
        provided = headers.get(b"authorization")
        if provided is None or not hmac.compare_digest(provided, self._expected):
            body = json.dumps({"error": "unauthorized"}).encode("utf-8")
            await send(
                {
                    "type": "http.response.start",
                    "status": 401,
                    "headers": [
                        (b"content-type", b"application/json"),
                        (b"content-length", str(len(body)).encode("latin-1")),
                        (b"www-authenticate", b'Bearer realm="vaultbeat-mcp-local"'),
                    ],
                }
            )
            await send({"type": "http.response.body", "body": body})
            return

        await self._app(scope, receive, send)


def _forbid_unknown_arguments() -> None:
    """Make every tool reject an argument it does not declare.

    🔴 The SDK's argument model ignores extras by default, so the call reached
    the tool with them silently dropped. `partner=true` on one of the five tools
    that only read the user's own data — or the `owner=` that 0.9.0 removed,
    on any tool — returned the USER's records with nothing saying the request
    was not honoured, and an agent would present them as the partner's
    (pre-release review, 2026-10-02). With `extra="forbid"` the call fails with
    "partner: Extra inputs are not permitted" instead, and the tool's input
    schema says `additionalProperties: false`.

    Set on the SDK's base class before any tool is registered: each tool's
    argument model is created from it at registration. Pinned by
    `test_tools_reject_arguments_they_do_not_declare`, which fails if an SDK
    upgrade moves or renames the base.

    One extra is dropped instead of refused: `partner=false` on a tool that has
    no `partner`. It asks for exactly what that tool returns — the user's own
    data — and an agent passing the same flag to every call it makes would
    otherwise fail on the five own-data tools for no reason (review R12 on #9).
    Dropped before validation, in the SDK's own pre-parse step; idempotent.
    """
    from mcp.server.mcpserver.utilities import func_metadata

    base = func_metadata.ArgModelBase
    base.model_config = {**base.model_config, "extra": "forbid"}

    meta = func_metadata.FuncMetadata
    original = getattr(meta.pre_parse_json, "__wrapped__", meta.pre_parse_json)

    def pre_parse_json(self: Any, data: dict[str, Any]) -> dict[str, Any]:
        asks_for_own = data.get("partner") is False or data.get("partner") == "false"
        if asks_for_own and "partner" not in self.arg_model.model_fields:
            data = {key: value for key, value in data.items() if key != "partner"}
        return original(self, data)

    pre_parse_json.__wrapped__ = original  # type: ignore[attr-defined]
    meta.pre_parse_json = pre_parse_json  # type: ignore[method-assign]


def _compact_tool_results() -> None:
    """Send dict results as compact JSON instead of the SDK's `indent=2`.

    The SDK pretty-prints every dict a tool returns, and on these results — a
    list of rows, each a small object — the newlines and indentation were half
    the payload: a year of `get_sleep_nights` measured 94k characters on the
    wire against 48k compact (pre-release review, 2026-10-02), and clients cap a
    tool result at about 25k tokens. Values are converted exactly as before
    (`pydantic_core.to_json`, `fallback=str`); only the whitespace goes.

    Idempotent: a second call wraps the original converter, not the wrapper.
    """
    import pydantic_core
    from mcp.server.mcpserver.utilities import func_metadata

    original = getattr(func_metadata._convert_to_content, "__wrapped__", func_metadata._convert_to_content)

    def convert(result: Any) -> Any:
        if isinstance(result, dict):
            result = pydantic_core.to_json(result, fallback=str).decode()
        return original(result)

    convert.__wrapped__ = original  # type: ignore[attr-defined]
    func_metadata._convert_to_content = convert


#: Above this many UTF-8 BYTES, a result is replaced by an answer saying how to
#: ask for less. Claude Code rejects a tool result over 25,000 tokens
#: (MAX_MCP_OUTPUT_TOKENS). Bytes, not characters, because bytes / 3 tracks tokens
#: for both kinds of text these results carry: ASCII JSON runs about 3 characters
#: (= bytes) a token, and CJK about one character (= 3 bytes) a token. A month of
#: one account's food log was 54k characters but 82k bytes — about 27k tokens,
#: which a character count would have let through (2026-10-02). 60,000 bytes is
#: ~20k tokens, leaving the client's framing room.
MAX_RESULT_BYTES = 60_000

#: What to narrow, per tool, in a `result_too_large` answer.
_NARROW_HINTS: dict[str, str] = {
    "get_food_log": "a smaller `limit` (days, newest first), or a narrower `since`/`until` window",
    "get_metric": (
        "fewer `series` (name the ones you need), fewer `days` or a `since`/`until` window, "
        "or `aggregation=\"avg\"` with `granularity=\"week\"` or `\"month\"`"
    ),
    "get_sleep_nights": "a later `since`, or a smaller `limit`",
    "get_sleep_detail": "a smaller `limit` or a narrower `since` / `until`, without `include_timeline`",
    "get_intraday": "a smaller `limit`",
    "get_workouts": "a smaller `limit`",
    "get_strength_log": "a smaller `limit` or `limit_days`",
    "get_notes": "a smaller `limit`, or a `target_kind`",
    "get_symptoms": "a smaller `limit`",
    "get_menstrual_cycle": "a smaller `limit`",
}


def _bounded(result: Any, tool_name: str) -> Any:
    """`result`, or a `result_too_large` answer when it would not fit a client."""
    if not isinstance(result, dict):
        return result
    import pydantic_core

    size = len(pydantic_core.to_json(result, fallback=str))
    if size <= MAX_RESULT_BYTES:
        return result
    hint = _NARROW_HINTS.get(tool_name, "a smaller `limit` or a shorter date range")
    return {
        "error": "result_too_large",
        "result_bytes": size,
        "limit_bytes": MAX_RESULT_BYTES,
        "message": (
            f"This answer would be about {size // 3:,} tokens ({size:,} bytes) — more than an MCP "
            f"client accepts in one tool result. Nothing is wrong with the data. "
            f"Ask again with {hint}."
        ),
    }


def _bound_wrap(function: _F) -> _F:
    """Outermost wrapper: no argument combination sends a client more than
    it accepts (`_bounded`). Outside the demo watermark, which adds a field to
    every row and so has to be inside the measurement."""
    name = function.__name__

    if inspect.iscoroutinefunction(function):

        @functools.wraps(function)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            return _bounded(await function(*args, **kwargs), name)

        return cast(_F, async_wrapper)

    @functools.wraps(function)
    def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        return _bounded(function(*args, **kwargs), name)

    return cast(_F, sync_wrapper)


def run_mcp_server(
    store: ConfigStore | None = None,
    *,
    transport: str = "stdio",
    host: str = "127.0.0.1",
    port: int = 8000,
    path: str = "/mcp",
    json_response: bool = True,
    stateless_http: bool = True,
    token: str | None = None,
    allow_remote: bool = False,
) -> None:
    try:
        from mcp.server.mcpserver import MCPServer
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "The MCP SDK is not installed. Install with `pip install -e ./mcp-local-server`."
        ) from error

    selected_transport = _normalize_transport(transport)
    _forbid_unknown_arguments()
    _compact_tool_results()

    # Read ONCE, here, rather than per call. A tool whose annotations say
    # "read-only" while its body has quietly switched to synthetic data mid-session
    # is worse than either state on its own, and the description prefix is fixed at
    # registration anyway — so the whole server is either a demo or it is not.
    #
    # 🔴 That was the INTENT from the start; it became true on 2026-08-20. The
    # service used to re-read the environment on every call, so this constant
    # only ever froze half the server: flip VAULTBEAT_DEMO on after startup and
    # the service began serving synthetic records while this value — and with it
    # every result watermark, every description prefix and the displayed server
    # name — stayed on "real". Verified: a read tool returned owner id
    # demo0001-… with no banner and no prefix, i.e. the ONE surface that gets
    # copied out of the session was the one that had stopped saying so.
    #
    # Going the other way was structurally impossible, which is why this is
    # frozen rather than live: the SDK captures each tool's description at
    # registration (mutating `__doc__` afterwards changes nothing — verified)
    # and clients cache serverInfo from `initialize`. So "all frozen" is the
    # only self-consistent state available, and it is now passed DOWN rather
    # than re-derived, so the two halves cannot disagree by construction.
    demo_active = demo_enabled()
    service = VaultbeatLocalService(store or ConfigStore(), demo=demo_active)

    # This name and version are what a user SEES: every MCP client lists the server
    # by its serverInfo. It said "Vaultbeat Local Sleep" until 2026-08-11 — a name
    # from when sleep was the only kind — while at that point already serving 26
    # tools across 18 kinds, so the listing undersold the product to the one audience
    # already looking at it. (That 26 is the count on the day it was fixed, not a
    # figure to keep current — `grep -cE '^    @tool' mcp_server.py` is. Anchored,
    # because an unanchored pattern matches this very sentence and reports N+1.)
    #
    # The demo suffix rides on the same string for the same reason: it is the one
    # label a client shows without anyone calling anything.
    # `instructions` rides in the `initialize` response — the ONE channel every
    # MCP client receives without asking. It exists because `STYLE` and `ABSENCE`
    # did not reach most agents: those are appended to prompts, and `prompts/*`
    # is opt-in, while the common shape is initialize → tools/list → call. See
    # `prompts.server_instructions` for why the text concatenates them instead of
    # paraphrasing.
    #
    # `version` is passed explicitly: when it was absent (v1's FastMCP took none),
    # the SDK reported ITS OWN package version, so clients showed "1.29.0" (the mcp
    # SDK) for a user who had just installed vaultbeat-mcp 0.3.x — and a wrong
    # version makes a stale install indistinguishable from a current one, which is
    # exactly the question `doctor` exists to answer. v1 needed a reach into the
    # private `_mcp_server` to set it; SDK 2.x takes it in the constructor.
    #
    # Transport settings (path, json_response, stateless_http) are NOT constructor
    # arguments since SDK 2.x — they go to `streamable_http_app()` in
    # `_serve_streamable_http`, the only place they were ever used.
    #
    # Usage telemetry rides as SDK middleware, beside the SDK's own OTel one:
    # it sees every `initialize` and `tools/call` — tool name, outcome, and which
    # AI client is calling — without touching a single tool body. What it may and
    # may not send is spelled out at the top of `telemetry.py`.
    from vaultbeat_mcp_local.telemetry import PostHogMiddleware, Telemetry

    def _account_id() -> str | None:
        config = service.store.load()
        return config.owner_user_id if config else None

    telemetry = Telemetry(account_id=_account_id, demo=demo_active, transport=selected_transport)
    mcp = MCPServer(
        "Vaultbeat Health [DEMO — SYNTHETIC DATA]" if demo_active else "Vaultbeat Health",
        instructions=server_instructions(demo=demo_active),
        version=__version__,
        middleware=[PostHogMiddleware(telemetry)],
    )

    def tool(*, title: str, annotations: Any) -> Callable[[_F], _F]:
        """Register one tool — the single place demo mode can reach every tool.

        A choke point, not a convenience wrapper (Invariant 58): the alternative
        is 29 call sites each remembering to watermark, which is 29 chances to
        forget and no way to notice the one that did.

        Patching `mcp.call_tool` after construction does NOT work and was tried
        (on SDK 1.x: the constructor bound the handler into the low-level server,
        so a later reassignment was never consulted; 2.x likewise registers its
        `tools/call` handler at construction). Decorating on the way in is the
        hook that does not depend on SDK internals.
        """

        def decorator(function: _F) -> _F:
            # Access note innermost (it inspects the plain result), demo
            # watermark outermost. In demo mode the note is inert anyway —
            # the stash it reads is only ever set by a bound call, which demo
            # mode returns before reaching.
            prepared = _access_wrap(function, service)
            if demo_active:
                # A tool with a disk or network write in its BODY must refuse by
                # never running, not be watermarked after it ran — that was
                # `_demo_block_wrap`, removed in 0.9.0 together with the two
                # pairing tools that were its only users (code-map Invariant 61
                # keeps the lesson; git has the mechanism).
                prepared = _demo_wrap(prepared)
            prepared = _bound_wrap(prepared)
            return cast(_F, mcp.tool(title=title, annotations=annotations)(prepared))

        return decorator

    # Prompts. `demo_active` is handed down rather than re-read for the reason
    # recorded above the MCPServer construction: every surface that can say "demo"
    # has to learn it from the same frozen value, or they disagree mid-session.
    #
    # Registered here rather than beside `mcp.run` so that "what this server
    # exposes" reads as one block. Nothing below depends on it.
    register_prompts(mcp, demo=demo_active)

    @tool(title="Run diagnostics", annotations=_read_only_tool(open_world=True))
    async def vaultbeat_doctor() -> dict[str, Any]:
        """Diagnose this Vaultbeat MCP install, and report which data types are unavailable.

        Call this when a read tool returns nothing, or when anything fails, before
        telling the user their data is missing. Two distinct things come back:

        `checks` — the install/binding chain: config present, keypair usable,
        binding valid, cloud reachable, a real record decryptable end to end.
        A failure here means the setup is broken, not that data is absent.

        `capabilities` — which metric kinds actually have data, and which empty
        ones are explained by an older iOS app (with the release date each
        needs). An empty kind is NOT proof the user never recorded it: their app
        may predate the feature entirely. A kind under `kinds_not_checked` could
        not be counted this run; it is neither empty nor full.

        `scope` — what this report does NOT cover. Everything here can pass and
        the setup still be broken on the client side: this server is a subprocess
        of your MCP client, so it cannot read the client's config file, cannot see
        which environment variables the client forwarded, and cannot tell whether
        it was launched with the arguments the user believes. A green report never
        clears the client. `scope.env_overrides_received` lists which Vaultbeat
        variables actually arrived — check that before guessing at the client's
        environment.

        `binding` — the local pairing state (which account, which server, whether
        demo mode is serving synthetic data). No keys or tokens.

        This runs a cloud round trip, so it takes a moment.
        """

        report = await service.doctor()
        # Was its own `vaultbeat_status` tool until 0.9.0. The binding state is
        # part of the answer to "why is nothing working", so it rides here.
        report["binding"] = service.status()
        # State only, deliberately (owner, 2026-09-23). An earlier version also
        # carried a "what it sends" line and the opt-out instruction, and every
        # diagnosis would put that in front of the agent, which tends to relay it
        # to the user unprompted. What is sent and how to turn it off are
        # disclosed where a person goes to read about it — the README's
        # Telemetry section and the privacy page — not recited on each run.
        report["telemetry"] = {"enabled": telemetry.enabled}
        return report

    @tool(title="Symptoms", annotations=_read_only_tool())
    async def get_symptoms(
        limit: int = 120, partner: bool = False, fresh: bool = False
    ) -> dict[str, Any]:
        """Decrypt recent symptoms locally, grouped by data owner — both sources.

        SENSITIVE. Each entry in `owners` carries `owner_user_id` and two lists:
        · `days` — symptoms imported from Apple Health (cramps, headache, fatigue,
          coughing…), one row per day with each sample's severity
          (mild/moderate/severe/present/…). These reach this server only when the
          user opted in on iOS — their own AI toggle, or the partner-AI toggle for
          a partner's. `symptom_counts` counts logged DAYS per type.
        · `reported` — episodes a person reported themselves, through
          `log_symptom` or the Vaultbeat app: `entry_id`, `symptom_type`,
          `display_name` (their own words), `severity`, `local_date`, `onset_at` /
          `end_at` (`end_recorded: false` means no end was logged — ask, do not
          assume it is still going), `duration_minutes`, `body_location`,
          `triggers`, `note`. `reported_counts` counts EPISODES per type.
        A type that exists in Apple Health is spelled the same in both lists
        (`healthkit_type: true`), so "headache" from either source is one symptom.
        To correct or remove a reported entry use `update_symptom` /
        `delete_symptom` with its `entry_id`; imported days are edited in Apple Health.
        Stays on-device, never re-exported.

        Returns YOUR symptoms by default; `partner=true` returns your partner's
        Apple Health `days` only (if they share symptoms with you from their own
        app). A partner's `reported` episodes never reach you: the app and
        `log_symptom` seal them for their owner's own AI alone, so an empty
        `reported` under `partner=true` means "not shareable", not "none logged".

        Carries a `coverage` block: quote `coverage.days_covered` (distinct days, not
        the row count) and `coverage.span_days` beside any average or trend.
        `coverage.window_satisfied: false` alone does NOT mean a short history: it is
        also false when older days exist beyond your window. 🔴 Before saying how far
        back someone's data goes, read `coverage.more_available`: `true` means this
        server can decrypt days OLDER than `first_day` that your `limit` left behind —
        re-read with a larger `limit`, or quote `coverage.oldest_available` as the real
        start of their history. Never report a `limit`-shaped window as the extent of
        their data.
        """

        return _owner_unknown_note(
            _partner_note(
                await service.symptom_summary(
                    limit=limit, fresh=fresh, owner=service.person_owner(partner=partner)
                ),
                partner,
            ),
            service,
            partner,
        )

    @tool(title="Notes", annotations=_read_only_tool())
    async def get_notes(
        limit: int = 120,
        target_kind: str | None = None,
        partner: bool = False,
        fresh: bool = False,
    ) -> dict[str, Any]:
        """Decrypt recent free-text notes (day annotations) locally.

        SENSITIVE free text. Each note carries `owner_user_id` (who wrote it),
        `target_kind`, and `target_date` (the local day it annotates) — join
        against the same-day metric data for pattern analysis. Kinds:
        "sleep" | "menstrual" are written manually in the iOS app by either
        partner (e.g. "昨晚舍友很吵" on a sleep day); "mood" | "general" are
        agent-authored via `log_note`. Pass target_kind to filter.
        Stays on-device, never re-exported.

        Returns notes about YOU by default. `partner=true` returns notes about your
        partner: the ones they wrote themselves and shared, plus the ones your AI
        recorded about them (`about: "partner"`, written with
        `log_note_append(partner=true)`).

        Carries a `coverage` block: quote `coverage.days_covered` (distinct days, not
        the row count) and `coverage.span_days` beside any average or trend.
        `coverage.window_satisfied: false` alone does NOT mean a short history: it is
        also false when older days exist beyond your window. 🔴 Before saying how far
        back someone's data goes, read `coverage.more_available`: `true` means this
        server can decrypt days OLDER than `first_day` that your `limit` left behind —
        re-read with a larger `limit`, or quote `coverage.oldest_available` as the real
        start of their history. Never report a `limit`-shaped window as the extent of
        their data.
        """

        return _owner_unknown_note(
            _partner_note(
                await service.notes_summary(limit=limit, target_kind=target_kind, fresh=fresh, partner=partner),
                partner,
            ),
            service,
            partner,
        )

    @tool(title="Strength log", annotations=_read_only_tool())
    async def get_strength_log(
        limit: int = 120, limit_days: int | None = None, fresh: bool = False
    ) -> dict[str, Any]:
        """Decrypt recent strength-training sessions locally (newest first).

        Exercise-level detail HealthKit's workout type cannot carry: each
        session lists exercises with their sets (weightKg × reps), an optional
        session note, and `total_volume_kg` (Σ weight × reps). Logged manually
        in Vaultbeat; owner's own sessions only — strength has no partner
        fan-out. Join `date` against sleep/HRV/weight for training-load
        analysis. Pass limit_days to cap how many sessions return.
        `duplicate_sessions`, when present, lists stored entries that only repeat
        another session of the same day; they are already left out of `sessions`
        and `session_count`, so do not add their volume back.

        Carries a `coverage` block: quote `coverage.days_covered` (distinct days, not
        the row count) and `coverage.span_days` beside any average or trend.
        `coverage.window_satisfied: false` alone does NOT mean a short history: it is
        also false when older days exist beyond your window. 🔴 Before saying how far
        back someone's data goes, read `coverage.more_available`: `true` means this
        server can decrypt days OLDER than `first_day` that your `limit` left behind —
        re-read with a larger `limit`, or quote `coverage.oldest_available` as the real
        start of their history. Never report a `limit`-shaped window as the extent of
        their data.
        """

        return _annotate_if_empty(
            await service.strength_summary(limit=limit, limit_days=limit_days, fresh=fresh),
            "strength",
            "sessions",
        )

    @tool(title="Log strength session (replaces day)", annotations=_mutating_tool(destructive=True))
    async def log_strength_entry(
        date: str,
        exercises: list[dict[str, Any]],
        note: str | None = None,
        merge: bool = False,
    ) -> dict[str, Any]:
        """Log one strength-training session on the owner's behalf (agent write).

        ⚠️ THIS TOOL DELETES. The supplied `exercises` become the day's ENTIRE
        session — any exercise you don't re-send is silently deleted. If you meant
        to ADD to a day rather than replace it, STOP and call
        `log_strength_append` instead; it cannot delete anything.

        (`merge=True` still does the same thing as `log_strength_append` and
        keeps working for callers that already use it. New callers should use
        the separate tool: which one you called is visible to the owner, a flag
        buried in the arguments is not.)

        The result carries `replaced_exercises` — the names this call deleted.
        If that list is non-empty and you did not intend to replace the day, you
        just destroyed those exercises; re-send them with `merge=True`.

        `note=None` LEAVES THE EXISTING NOTE ALONE (pass `note=""` to clear it).

        `date` is the LOCAL calendar day the session happened, "YYYY-MM-DD".
        `exercises` is `[{"name": "卧推", "sets": [{"weightKg": 30, "reps": 8}, ...]}, ...]`.
        Encrypted end-to-end before it ever leaves this machine — this server
        never sends plaintext. Requires a bind made after this feature shipped
        (carries owner_user_id/owner_public_key_base64/owner_device_id from
        the pairing handshake); an older bind must re-pair by running
        `uvx vaultbeat-apple-health@latest bind` in a terminal.
        """

        return await service.log_strength_entry(
            date=date, exercises=exercises, note=note, merge=merge
        )

    @tool(title="Add to strength log", annotations=_mutating_tool(destructive=False))
    async def log_strength_append(
        date: str,
        exercises: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Add exercises to a session WITHOUT touching what is already logged (agent write).

        This tool cannot delete anything you did not send. Your `exercises` are
        appended to the day's existing session; an exercise whose `name` matches an
        existing one gets its sets appended to it. This is the right tool for "log
        the set I forgot" or for logging a session in installments while the owner
        is still in the gym — which is how strength data actually arrives.

        Reach for `log_strength_entry` ONLY when you intend the supplied exercises
        to become the day's ENTIRE session and everything else to be deleted.

        `date` is the LOCAL calendar day the session happened, "YYYY-MM-DD".
        `exercises` is `[{"name": "卧推", "sets": [{"weightKg": 30, "reps": 8}, ...]}, ...]`.

        This tool deliberately cannot set the session note: that field is
        replace-only, and a tool that promises to delete nothing must not carry an
        exception. Use `log_strength_entry` to change it.

        The result is the same shape `log_strength_entry` returns.
        `replaced_exercises` is always `[]` here — that empty list is the receipt
        that this call deleted nothing. Encrypted end-to-end before it ever leaves
        this machine. Requires a bind made after the agent write path shipped; an
        older bind must re-pair by running `uvx vaultbeat-apple-health@latest bind`
        in a terminal.
        """

        return await service.log_strength_entry(
            date=date, exercises=exercises, note=None, merge=True
        )

    @tool(title="Food log", annotations=_read_only_tool())
    async def get_food_log(
        limit: int | None = None,
        limit_days: int | None = None,
        fresh: bool = False,
        since: str | None = None,
        until: str | None = None,
    ) -> dict[str, Any]:
        """Decrypt daily food-intake logs locally (newest first).

        `limit` counts LOGGED days: by default the newest 14 days that have a
        log — not a calendar fortnight. Days with nothing logged are skipped, so
        the 14 can reach further back; `coverage.span_days` and
        `coverage.days_missing_in_span` say how far. `since` / `until`
        ("YYYY-MM-DD", local days, both inclusive) read a calendar window
        instead — "what did I eat in the first week of August" — and then
        `limit` applies only if you pass it. Each day's `date` is its local
        calendar day (the same as `local_date`, and the form `log_food_entry`
        and `get_strength_log` use). A logged day is roughly 0.5-1k
        tokens (more with notes, and Chinese text counts more), so a month can
        already be past what a client takes in one result — it then comes back
        as `result_too_large`; ask for fewer days or a narrower window.

        Each day carries `meals`, each meal a list of `items` with `food` (name),
        optional free-text `portion` ("1 根" / "300g" / "小份"), optional
        per-item/per-meal `note`, and — when the logging agent estimated them —
        optional structured nutrition numbers (`kcal`, `proteinGrams`,
        `fatGrams`, `carbGrams`). Items without those fields need analysis-time
        estimation from name + portion; items with them can be summed directly.
        Owner's own days only. Pass limit_days to cap how many days return.

        Carries a `coverage` block: quote `coverage.days_covered` (distinct days, not
        the row count) and `coverage.span_days` beside any average or trend.
        `coverage.window_satisfied: false` alone does NOT mean a short history: it is
        also false when older days exist beyond your window. 🔴 Before saying how far
        back someone's data goes, read `coverage.more_available`: `true` means this
        server can decrypt days OLDER than `first_day` that your `limit` left behind —
        re-read with a larger `limit`, or quote `coverage.oldest_available` as the real
        start of their history. Never report a `limit`-shaped window as the extent of
        their data.
        """

        # An empty string is "no bound", the same as omitting it. Tested against
        # None alone, `since=""` skipped the two-week default here and then read
        # as no bound downstream — the whole history in one result (review R10).
        since, until = since or None, until or None
        if limit is None and since is None and until is None:
            limit = 14
        return _annotate_if_empty(
            await service.food_summary(
                limit=limit, limit_days=limit_days, fresh=fresh, since=since, until=until
            ),
            "food",
            "days",
        )

    @tool(title="Log food (replaces day)", annotations=_mutating_tool(destructive=True))
    async def log_food_entry(
        date: str, meals: list[dict[str, Any]], note: str | None = None, merge: bool = False
    ) -> dict[str, Any]:
        """Log one day's food intake on the owner's behalf (agent write).

        ⚠️ THIS TOOL DELETES. The supplied `meals` become the day's ENTIRE log —
        any meal you don't re-send is silently deleted. If you meant to ADD to a
        day rather than replace it, STOP and call `log_food_append` instead; it
        cannot delete anything.

        (`merge=True` still does the same thing as `log_food_append` and keeps
        working for callers that already use it. New callers should use the
        separate tool: which one you called is visible to the owner, a flag
        buried in the arguments is not.)

        The result carries `replaced_meals` — the meals this call deleted. If
        that list is non-empty and you did not intend to replace the day, you
        just destroyed them; re-send them with `merge=True`.

        `note=None` LEAVES THE EXISTING NOTE ALONE in both modes (pass
        `note=""` to clear it).

        `date` is the LOCAL calendar day, "YYYY-MM-DD".
        `meals` is a list of `{name?, timeOfDay?, items: [...], note?}` where each
        item is `{food, portion?, note?, kcal?, proteinGrams?, fatGrams?, carbGrams?}`,
        e.g. `[{"name": "lunch", "items": [{"food": "香蕉", "portion": "1 根", "kcal": 105}]}]`.
        Everything but `food` is optional so a rushed "just log 香蕉" still works;
        when you DO estimate nutrition at logging time, put the numbers in the
        structured fields (snake_case aliases like `protein_g` are accepted) —
        they persist for later sessions instead of being re-guessed each read.

        ESTIMATING FROM A PHOTO: look for something of known size in the frame
        first — a utensil, a hand, a coin, the rim of a standard plate — and
        calibrate the portion against it. With no such reference an image cannot
        settle portion size, and portion size is what the whole estimate rests
        on. In that case say so in your reply and give a range rather than a
        precise-looking number. These values are persisted and summed into daily
        totals later, so a confident "650 kcal" that is wrong does more damage
        than "roughly 500-700, nothing in frame to judge size by" — the first
        silently poisons a week of trends, the second invites a correction.
        [keep the photo-estimation paragraph above in sync with log_food_append's copy]
        Encrypted end-to-end before it ever leaves this machine.
        """

        return await service.log_food_entry(date=date, meals=meals, note=note, merge=merge)

    @tool(title="Add to food log", annotations=_mutating_tool(destructive=False))
    async def log_food_append(
        date: str,
        meals: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Add meals to a day WITHOUT touching what is already logged (agent write).

        This tool cannot delete anything you did not send. Your `meals` are appended
        to whatever the day already holds; a meal whose `name` matches an existing
        meal gets its items appended to that meal. This is the right tool for "log
        the snack I forgot" / "add dinner to today" — which is almost every
        follow-up write of the day.

        Reach for `log_food_entry` ONLY when you intend the supplied meals to become
        the day's ENTIRE log and everything else to be deleted.

        `date` is the LOCAL calendar day, "YYYY-MM-DD".
        `meals` is a list of `{name?, timeOfDay?, items: [...], note?}` where each
        item is `{food, portion?, note?, kcal?, proteinGrams?, fatGrams?, carbGrams?}`,
        e.g. `[{"name": "lunch", "items": [{"food": "香蕉", "portion": "1 根", "kcal": 105}]}]`.
        Everything but `food` is optional so a rushed "just log 香蕉" still works;
        when you DO estimate nutrition at logging time, put the numbers in the
        structured fields (snake_case aliases like `protein_g` are accepted) — they
        persist for later sessions instead of being re-guessed each read.

        ESTIMATING FROM A PHOTO: look for something of known size in the frame
        first — a utensil, a hand, a coin, the rim of a standard plate — and
        calibrate the portion against it. With no such reference an image cannot
        settle portion size, and portion size is what the whole estimate rests on.
        In that case say so in your reply and give a range rather than a
        precise-looking number. These values are persisted and summed into daily
        totals later, so a confident "650 kcal" that is wrong does more damage than
        "roughly 500-700, nothing in frame to judge size by" — the first silently
        poisons a week of trends, the second invites a correction.
        [keep the photo-estimation paragraph above in sync with log_food_entry's copy]

        This tool deliberately cannot set the day's note: that field is
        replace-only, and a tool that promises to delete nothing must not carry an
        exception. Use `log_food_entry` to change it.

        The result is the same shape `log_food_entry` returns. `replaced_meals` is
        always `[]` here — that empty list is the receipt that this call deleted
        nothing. Encrypted end-to-end before it ever leaves this machine.
        """

        return await service.log_food_entry(date=date, meals=meals, note=None, merge=True)

    @tool(title="Log note (replaces note)", annotations=_mutating_tool(destructive=True))
    async def log_note(
        text: str,
        kind: str = "general",
        date: str | None = None,
        merge: bool = False,
        partner: bool = False,
    ) -> dict[str, Any]:
        """Log a free-text note on the owner's behalf (agent write).

        For narratives that belong next to the metric data instead of in chat
        history: `kind="mood"` for emotional state ("为什么今天情绪低落"),
        `kind="general"` for day events worth joining against sleep/HRV later.
        (sleep/menstrual notes stay iOS-authored — this tool refuses them.)

        ⚠️ THIS TOOL DELETES. There is one note per (kind, day), and your `text`
        becomes its ENTIRE contents — anything already written for that kind+day
        is silently deleted. If you meant to ADD to a day rather than replace it,
        STOP and call `log_note_append` instead; it cannot delete anything.

        (`merge=True` still does the same thing as `log_note_append` and keeps
        working for callers that already use it. New callers should use the
        separate tool: which one you called is visible to the owner, a flag
        buried in the arguments is not.)

        The result carries `replaced_text` — the note this call deleted. If it is
        non-null and you did not intend to replace, you just destroyed that text;
        re-send it with `merge=True`.

        Symptoms do NOT go here: use `log_symptom`, one call per symptom. A
        note is free text nothing can be matched against; a symptom entry keeps
        its type, severity and timing so it can later be lined up with sleep,
        food and heart rate. A note is for what surrounds it (mood, events, how
        a day went).

        `date` = LOCAL calendar day "YYYY-MM-DD" (default today). Read back via
        `get_notes` (optionally `target_kind="mood"`/`"general"`). Encrypted
        end-to-end before it ever leaves this machine.

        `partner=true` records a note ABOUT your partner ("she had a stomach ache
        this morning"). It is stored in YOUR account and sealed to you and this
        machine only — your partner's account is never touched, and it never
        mixes into your own notes: plain reads leave it out, `get_notes(partner=true)`
        returns it. Default `false` = a note about you.
        """

        return await service.log_note(
            text=text, kind=kind, date=date, merge=merge, partner=partner
        )

    @tool(title="Add to note", annotations=_mutating_tool(destructive=False))
    async def log_note_append(
        text: str,
        kind: str = "general",
        date: str | None = None,
        partner: bool = False,
    ) -> dict[str, Any]:
        """Add a line to a day's note WITHOUT erasing what is already there (agent write).

        There is one note per (kind, day). This tool appends your `text` to it on a
        new line and cannot delete what is already written.

        The second note of a day is the normal case, not the exception — a
        morning event and an evening one — and `log_note` would replace the
        first with the second. Symptoms do NOT go here: use `log_symptom`, which
        keeps type, severity and timing as fields a later analysis can match.

        Reach for `log_note` ONLY when you intend your text to become the note's
        ENTIRE contents and whatever is there now to be deleted (e.g. correcting
        something you yourself wrote a minute ago).

        `kind="general"` for day events worth joining against sleep/HRV later,
        `kind="mood"` for emotional state. (sleep/menstrual notes stay iOS-authored
        — this tool refuses them.) `date` = LOCAL calendar day "YYYY-MM-DD"
        (default today). Read back via `get_notes`.

        The result is the same shape `log_note` returns. `replaced_text` is always
        `null` here — that null is the receipt that this call deleted nothing.
        Encrypted end-to-end before it ever leaves this machine.

        `partner=true` records a note ABOUT your partner ("she had a stomach ache
        this morning"). It is stored in YOUR account and sealed to you and this
        machine only — your partner's account is never touched, and it never
        mixes into your own notes: plain reads leave it out, `get_notes(partner=true)`
        returns it. Default `false` = a note about you.
        """

        return await service.log_note(
            text=text, kind=kind, date=date, merge=True, partner=partner
        )

    @tool(title="Log symptom", annotations=_mutating_tool(destructive=False))
    async def log_symptom(
        symptom_type: str,
        severity: str = "unspecified",
        onset_at: str | None = None,
        end_at: str | None = None,
        date: str | None = None,
        display_name: str | None = None,
        body_location: str | None = None,
        triggers: list[str] | None = None,
        note: str | None = None,
    ) -> dict[str, Any]:
        """Record one symptom the person reports — one call per symptom (agent write).

        This is where "我肚子疼" / "headache since lunch" / "拉肚子了" goes, not a
        note: each episode is kept as fields (type, severity, timing, place,
        suspected causes) so a later read can line it up with sleep, food, heart
        rate and wrist temperature. Two symptoms at once are two calls. This tool
        only ever adds; it cannot change or delete anything already recorded.

        `symptom_type`: a short English token. Use the Apple Health name when one
        fits — abdominal_cramps, bloating, constipation, diarrhea, heartburn,
        nausea, vomiting, headache, dizziness, fatigue, fever, chills, coughing,
        sore_throat, runny_nose, lower_back_pain, chest_tightness_or_pain,
        shortness_of_breath, acne, night_sweats… — so it merges with what Apple
        Health imports; otherwise a plain descriptive one (rectal_bleeding,
        stomach_spasm, eye_strain). Any spelling is folded to one form.
        `display_name`: the person's own words, e.g. "便血" / "胃痉挛".

        `severity`: mild | moderate | severe | unspecified. Use `unspecified`
        unless they said how bad it is — never grade it for them.

        `onset_at`: when it began, ISO 8601 with their UTC offset, e.g.
        "2026-09-30T10:49+08:00". Omit it only for something happening NOW (it
        defaults to the current time). If they named a day but no time
        ("昨天头疼"), pass that day as a bare "2026-09-29" — do not invent an hour.
        `end_at`: when it eased, if they said; add it later with `update_symptom`.
        `date`: the local day to file it under, only when it differs from the
        onset's own day.
        `body_location`, `note`: free text, only what they said.
        `triggers`: what THEY suspect caused it, as short tokens
        (["bbq_dinner", "spicy_food"], ["sleep_deprivation"]). Record their
        suspicion; do not add causes of your own.

        Returns `entry_id` (keep it for `update_symptom` / `delete_symptom`) and
        the entry as read back. Read everything with `get_symptoms`. Sealed for
        the owner and this machine only, end to end, before it leaves here.
        """

        return await service.log_symptom(
            symptom_type=symptom_type,
            severity=severity,
            onset_at=onset_at,
            end_at=end_at,
            date=date,
            display_name=display_name,
            body_location=body_location,
            triggers=triggers,
            note=note,
        )

    @tool(
        title="Update symptom (overwrites given fields)",
        annotations=_mutating_tool(destructive=True),
    )
    async def update_symptom(
        entry_id: str,
        severity: str | None = None,
        end_at: str | None = None,
        onset_at: str | None = None,
        date: str | None = None,
        symptom_type: str | None = None,
        display_name: str | None = None,
        body_location: str | None = None,
        triggers: list[str] | None = None,
        note: str | None = None,
    ) -> dict[str, Any]:
        """Change fields of one reported symptom (agent write) — e.g. it eased.

        The common call is `update_symptom(entry_id, end_at="…")` when the person
        says it has stopped, or a new `severity` when it got worse. Only the
        fields you pass change; omitted ones keep their value. Passing a field
        OVERWRITES it: `note` replaces the old note rather than adding to it, and
        `triggers` replaces the whole list — read the entry first and send the
        combined list if you mean to add one. `""` clears a text field or a
        mistaken `end_at`; `[]` clears triggers.

        `entry_id` comes from `log_symptom` or from `get_symptoms`' `reported`
        list. Symptoms imported from Apple Health cannot be edited here.

        Returns `previous` — the old value of every field this call changed — and
        the entry as read back.
        """

        return await service.update_symptom(
            entry_id=entry_id,
            symptom_type=symptom_type,
            severity=severity,
            onset_at=onset_at,
            end_at=end_at,
            date=date,
            display_name=display_name,
            body_location=body_location,
            triggers=triggers,
            note=note,
        )

    @tool(title="Delete symptom (erases the entry)", annotations=_mutating_tool(destructive=True))
    async def delete_symptom(entry_id: str) -> dict[str, Any]:
        """Erase one reported symptom that was logged by mistake (agent write).

        Only for an entry that should never have existed — a duplicate, the wrong
        person, a symptom they did not have. A symptom that simply ended is NOT
        deleted: record its end with `update_symptom(entry_id, end_at=…)`.

        The entry's content is overwritten in the cloud, so it disappears from
        `get_symptoms` and from the app. The result carries `deleted_entry` — the
        full entry as it was — so a mistaken delete can be logged again with
        `log_symptom`. Symptoms imported from Apple Health cannot be deleted here.
        """

        return await service.delete_symptom(entry_id=entry_id)

    # Found by the generalised rule in `test_destructive_titles_name_their_
    # consequence`, not by anyone auditing this line: its own docstring says
    # "re-logging the same day overwrites", so it is a same-day replace exactly
    # like the three tools that already say so in their titles. Third offender
    # of a class that had been described as two.
    @tool(title="Log weight (replaces day)", annotations=_mutating_tool(destructive=True))
    async def log_weight_entry(weight_kg: float, date: str | None = None) -> dict[str, Any]:
        """Log the owner's weight (kg) on their behalf (agent write, 2026-07-21).

        `weight_kg`: kilograms (positive, ≤500). `date`: LOCAL calendar day
        "YYYY-MM-DD" (default = today). Same-day upsert-in-place semantics
        (dayID = "body-{dayStart.epoch}") — re-logging the same day overwrites.
        Encrypted end-to-end before it ever leaves this machine.

        Written data always lands in Vaultbeat cloud + MCP (visible to
        `get_metric` series "weight_kg"). Whether it also reaches Apple Health depends on an
        iOS setting, OFF by default: the MCP tab → "Allow AI to update Apple
        Health" (app 1.2.8 and earlier: Settings → Data & AI). When it is on,
        weigh-ins logged here sync back into Apple Health on the next app sync.

        ⚠️ If the owner wants this number in the Apple Health app, tell them to
        turn that toggle ON — do NOT tell them to re-enter it by hand in the
        Vaultbeat weight card. Logging it in both places produces two entries
        for the same day from different sources and corrupts the trend line.
        (Before 2026-07-28 this docstring said propagation was impossible and
        instructed exactly that manual double-entry; the toggle shipped
        2026-07-22.)
        """

        return await service.log_weight_entry(weight_kg=weight_kg, date=date)

    @tool(title="Menstrual cycle", annotations=_read_only_tool())
    async def get_menstrual_cycle(
        limit: int = 90, partner: bool = False, fresh: bool = False
    ) -> dict[str, Any]:
        """Decrypt recent menstrual cycle data locally and predict the next period.

        SENSITIVE: menstrual data only reaches this server if the user explicitly opted
        in on iOS; it stays on-device and is never re-exported. Returns recent samples plus
        a next-period prediction. Reads YOUR cycle by default; if the cycle being
        asked about is your partner's, pass `partner=true` (it reaches this server
        only if they share it).

        Carries a `coverage` block: quote `coverage.days_covered` (distinct days, not
        the row count) and `coverage.span_days` beside any average or trend.
        `coverage.window_satisfied: false` alone does NOT mean a short history: it is
        also false when older days exist beyond your window. 🔴 Before saying how far
        back someone's data goes, read `coverage.more_available`: `true` means this
        server can decrypt days OLDER than `first_day` that your `limit` left behind —
        re-read with a larger `limit`, or quote `coverage.oldest_available` as the real
        start of their history. Never report a `limit`-shaped window as the extent of
        their data.
        """

        return _partner_note(
            await service.menstrual_cycle_summary(
                limit=limit, owner=service.person_owner(partner=partner), fresh=fresh
            ),
            partner,
        )

    # Pairing is not a tool (removed 0.9.0, owner 2026-09-23): it happens once,
    # at install time, through `uvx vaultbeat-apple-health bind` in a terminal —
    # the same place every other MCP server does its setup. Two tools that sat in
    # every agent's tool list forever for a one-time step were noise, and the QR
    # they relayed often did not render in an agent transcript anyway. An
    # unpaired read raises `ConfigError` carrying `PAIRING_GUIDANCE`, so the
    # agent still learns exactly what to tell the user at the moment it matters.

    @tool(title="Sleep stage detail", annotations=_read_only_tool())
    async def get_sleep_detail(
        limit: int | None = None,
        partner: bool = False,
        fresh: bool = False,
        include_timeline: bool = False,
        since: str | None = None,
        until: str | None = None,
    ) -> dict[str, Any]:
        """Per-night sleep stages with per-stage HR/RR — depth on a few nights.

        Which sleep tool to use:
        · THIS one for going deep on one or two specific nights (stage bands,
          per-stage vitals). It defaults to 2 nights because each is ~3k
          characters; see the SIZE note below.
        · `get_sleep_nights` for anything spanning time — "how did I sleep
          this week/month/year", patterns, worst nights. One compact row per
          night, a whole year in one read. Reach for it whenever the question
          is about a period rather than a night, and do not conclude from THIS
          tool's two rows that only two nights exist.
        · `get_metric` with the sleep series (`bedtime_minutes`,
          `deep_sleep_minutes`, `awakenings`, ...) for averages computed for you.

        (This used to call itself "the primary tool for detailed sleep analysis",
        which read as "use this one for sleep" and handed back two days to
        anyone who asked how their week went.)

        Returns `stage_intervals` (contiguous stage bands with start/end),
        `stage_minutes`, and `stage_vitals` (per-stage HR/RR min/mean/max). Reads
        YOUR nights by default; `partner=true` reads your partner's shared sleep.

        `limit` = the newest N nights, 2 by default. To read a particular night,
        pass its date as `since` and `until` ("YYYY-MM-DD", the local day the
        night ENDS on — the `local_date` / `date` column of `get_sleep_nights`);
        a wider `since` / `until` reads every night in that window, and `limit`
        then applies only if you pass it.

        ⚠️ SIZE: each night is ~3k characters as returned (a week ~21k). Setting
        `include_timeline=True` adds the raw per-sample array (hr, rr, stage,
        time) — about 13k characters PER NIGHT, which will overflow a typical
        25k-token client budget after 2-3 nights. Ask for it only when you need
        sample-level vitals (e.g. "when exactly did HR spike"); `stage_vitals`
        already answers per-stage questions. Raise `limit` for trends, but keep
        it low whenever `include_timeline=True`.

        ⚠️ `is_in_bed_only: true` means sleep was NEVER MEASURED that night (the
        Watch wasn't worn) — NOT that the person slept zero. `duration_label`
        reads "no sleep data" and `in_bed_minutes` holds the time actually
        recorded in bed. Report it as "no sleep data (in bed ~Xh)", never as
        "slept 0 hours".

        Carries a `coverage` block: quote `coverage.days_covered` (distinct days, not
        the row count) and `coverage.span_days` beside any average or trend.
        `coverage.window_satisfied: false` alone does NOT mean a short history: it is
        also false when older days exist beyond your window. 🔴 Before saying how far
        back someone's data goes, read `coverage.more_available`: `true` means this
        server can decrypt days OLDER than `first_day` that your `limit` left behind —
        re-read with a larger `limit`, or quote `coverage.oldest_available` as the real
        start of their history. Never report a `limit`-shaped window as the extent of
        their data.
        """

        since, until = since or None, until or None
        if limit is None and since is None and until is None:
            limit = 2
        return _partner_note(
            await service.sleep_detail_records(
                limit=limit,
                owner=service.person_owner(partner=partner),
                fresh=fresh,
                include_timeline=include_timeline,
                since=since,
                until=until,
            ),
            partner,
        )

    @tool(title="Sleep, every night", annotations=_read_only_tool())
    async def get_sleep_nights(
        limit: int | None = None, since: str | None = None, partner: bool = False, fresh: bool = False,
    ) -> dict[str, Any]:
        """Every night as one compact row — the tool for reading a sleep HISTORY.

        One row per night, newest first: date, weekday, bed and wake time, time
        asleep, deep / REM / core / awake minutes, deep and REM share, number of
        awakenings, longest unbroken stretch of sleep, heart and breathing rate
        while asleep, and any OTHER sleep that day (naps, the second half of a
        broken night) — so a day's total sleep is `asleep` plus `other`. Column names come once in `columns`; each row is the
        values in that order, so a whole year is ~40k characters. Read it when
        the question is about a period or a pattern: weekends vs weekdays,
        drifting bedtimes, which nights were worst, what changed after a date.

        Which sleep tool:
        · THIS one for any span of time.
        · `get_metric` with the sleep series (`sleep_minutes`, `bedtime_minutes`,
          `deep_sleep_minutes`, `awakenings`, ...) when you want the arithmetic —
          weekly/monthly averages computed for you, from these same rows.
        · `get_sleep_detail` for one or two nights in depth (stage bands,
          per-stage vitals).

        `limit` = newest N nights (default 120; pass 400 for everything, and
        check `coverage.more_available`). `since` = "YYYY-MM-DD" keeps every
        night on or after that day — use it for "last month" rather than
        guessing a limit; with `since`, `limit` applies only if you pass it.

        Read `legend` once. Five flags, comma-separated when a night has several:
        `unworn` means sleep was never measured that night (not zero sleep),
        `motion_inferred` means phone motion guessed it (leave it out of averages,
        as `get_metric` does), `daytime` means the day's only sleep was a daytime
        nap (real, but not a night — out of `get_metric`'s nightly series, still
        a row here), `short` means a main sleep under 3 h with
        no stages — a nap or a partly recorded night, left out of `get_metric`'s
        sleep-duration and clock series — and `no_stages` means the total is
        known but the stage columns are null. A null anywhere is "not measured",
        never zero. Reads YOUR nights; `partner=true` reads your partner's
        shared sleep.

        Carries a `coverage` block: quote `coverage.days_covered` beside any
        average, and read `coverage.more_available` before saying how far back
        the history goes.
        """

        # With a `since`, the default 120 used to cut the window anyway: since
        # 2025-11-01 returned nights from June on, the instruction above being
        # "use since rather than guessing a limit" (#9, release gate round 4).
        since = since or None
        if limit is None and since is None:
            limit = 120
        if since is not None:
            # Compared as a string against zero-padded dates, so "2026-8-1"
            # would silently select the wrong nights rather than fail.
            try:
                since = _date.fromisoformat(since).isoformat()
            except ValueError:
                return {
                    "error": "invalid_since",
                    "requested": since,
                    "message": 'Pass `since` as "YYYY-MM-DD", e.g. "2026-08-01".',
                }
        summary = await service.sleep_nights(
            limit=limit, since=since,
            owner=service.person_owner(partner=partner),
            fresh=fresh,
        )
        return _partner_note(_sleep_nights_table(summary, since), partner)

    @tool(title="Workouts", annotations=_read_only_tool())
    async def get_workouts(limit: int = 90, fresh: bool = False) -> dict[str, Any]:
        """Decrypt recent workout sessions (type, duration, calories, distance).
        Reads YOUR workouts; workouts are never shared between partners, so this
        tool has no partner option.

        Carries a `coverage` block: quote `coverage.days_covered` (distinct days, not
        the row count) and `coverage.span_days` beside any average or trend.
        `coverage.window_satisfied: false` alone does NOT mean a short history: it is
        also false when older days exist beyond your window. 🔴 Before saying how far
        back someone's data goes, read `coverage.more_available`: `true` means this
        server can decrypt days OLDER than `first_day` that your `limit` left behind —
        re-read with a larger `limit`, or quote `coverage.oldest_available` as the real
        start of their history. Never report a `limit`-shaped window as the extent of
        their data.
        """

        return await service.workout_records(limit=limit, owner=service.person_owner(), fresh=fresh)

    @tool(title="Health profile", annotations=_read_only_tool())
    async def get_user_profile(fresh: bool = False) -> dict[str, Any]:
        """Read YOUR health profile: biological sex, age, date of birth, height.

        Use it before anything that depends on sex, age or height — basal
        metabolic rate, heart-rate zones, VO2 max bands, BMI. It is never shared
        between partners, so this tool has no partner option.

        `profile.sex_source` says where the sex came from: "chosen" (the person
        picked it in the app) or "apple_health" (a field in Apple Health they may
        never have looked at). Every field may be null; `profile` itself is null
        when nothing was uploaded, and the reply's `note` explains why that
        cannot be narrowed down. Never fill a null in with a guess.

        A profile is not a series: its `coverage.days_covered` is 0 by
        construction, and the reply carries no upload time — the stored row is
        replaced in place on every edit, so nothing here says how current the
        values are.
        """

        return await service.user_profile(owner=service.person_owner(), fresh=fresh)

    # ── Daily metrics: one generic reader ──────────────────────────────────
    #
    # Replaced ten per-kind tools (`get_resting_hr`, `get_activity`, `get_hrv`,
    # …) on 2026-09-23. Every one of them was a thin shell over one service
    # method, and every new data source would have added another shell, another
    # prompt reference and another README row. The kinds that are not one number
    # per day keep their own tools above — flattening a workout to "duration"
    # would answer a question nobody asked while hiding the ones they did.

    @tool(title="Daily health metrics", annotations=_read_only_tool())
    async def get_metric(
        series: str | list[str] | None = None,
        days: int = 30,
        aggregation: str = "none",
        granularity: str = "day",
        partner: bool = False,
        fresh: bool = False,
        since: str | None = None,
        until: str | None = None,
    ) -> dict[str, Any]:
        """Read any one-number-per-day health series: steps, resting HR, HRV, weight, …

        `series` takes ONE name, a LIST of names, or nothing (= every series).
        Names come from `list_metric_series`, which also says how much data backs
        each — call it first rather than guessing. Activity is five series
        (steps, active_energy, exercise_minutes, stand_minutes, distance_meters);
        pass them as a list to get the whole day in one call.

        `days` = the newest N days THAT HAVE DATA, per series (not N calendar days).
        For a CALENDAR window — "last month", "this semester", "July" — pass
        `since` / `until` ("YYYY-MM-DD", either or both) instead; `days` is then
        ignored and the reply names the `range` it used.

        `aggregation` is computed here, not by you: `avg` / `sum` / `min` / `max` /
        `latest` over the window, or `none` (default) for the per-day points only.
        `sum` is refused on state measurements (resting HR, weight, VO2max) — it
        has no meaning there. `granularity` = `day` (default), `week` (ISO weeks)
        or `month`; week/month return buckets aggregated with `aggregation`
        (`avg` when `none`), each with `days_with_data` beside `days_in_period`.
        `period_in_progress: true` marks the current, unfinished week/month and
        `clipped_by_window: true` the oldest one your `days` cut through — never
        compare either against a full period as if it were one.
        `granularity` = `weekday` groups the window by day of the week (Mon..Sun)
        — the direct answer to "weekends vs weekdays" and to social jet lag. Read
        its `weekday_note`: sleep is dated by the morning it ends, so `Sat` is the
        Friday-night sleep.

        Honesty the reply already does for you, so do not undo it:
        · On `cumulative` series (steps, energy, water) today is still accruing —
          it is marked `partial: true` and excluded from every aggregate.
        · `excluded_days` names days dropped because their data was short (basal
          energy with the Watch off the wrist, TDEE without basal), with reasons.
        · Missing days are absent, never zero.

        Each series' `points` (or `buckets`) is a table: `columns` names the
        fields once and every entry of `rows` is one day (or period) in that
        order — e.g. `columns: ["date", "value", "partial"]`. The flags above are
        columns; a day without one has `null` there.

        Each series carries its own `coverage` block: quote `coverage.days_covered`
        and `coverage.span_days` beside any number. 🔴 Before saying how far back
        someone's data goes, read `coverage.more_available`: `true` means older days
        exist that your `days` left behind — ask for more, or quote
        `coverage.oldest_available`. Never report a window as the extent of their data.

        Reads YOUR data by default.
        Pass `partner=true` for your partner's — of these series only sleep, water
        and weight (with body composition) can be shared, so every other series
        comes back empty for a partner by design.

        Sleep is here too — duration, timing (`bedtime_minutes`, `wake_minutes`,
        `sleep_midpoint_minutes`), stages, awakenings, heart/breathing rate while
        asleep; the nights behind those numbers are `get_sleep_nights`. Richer
        kinds have their own tools: workouts, strength, food, notes, symptoms, cycle,
        and the sex / age / height profile (`get_user_profile`).
        Samples within a day (HRV spikes) → `get_intraday`. Trend / period
        comparison / correlation → `get_metric_trend`, `compare_metric_periods`,
        `correlate_metric_series`.
        """

        return _metric_tables(
            await service.metric_values(
                series=series,
                days=days,
                aggregation=aggregation,
                granularity=granularity,
                owner=service.person_owner(partner=partner),
                fresh=fresh,
                since=since,
                until=until,
            )
        )

    @tool(title="Intraday samples", annotations=_read_only_tool())
    async def get_intraday(
        series: str = "hrv_sdnn",
        granularity: str = "hourly",
        limit: int = 168,
        fresh: bool = False,
    ) -> dict[str, Any]:
        """Samples WITHIN a day, for the series that record several per day (HRV).

        `granularity`:
        - `"hourly"` (default) — one row per hour that HAS samples (UTC hour
          boundaries, shown in local time), the mean SDNN of that hour and its
          `sample_count`. Right for "how did my HRV move through the day".
        - `"raw"` — one row per sample, newest first.

        The Watch measures HRV a handful of times a day, not continuously (more
        during a Breathe session), so most days have about 8-14 rows either way.
        `limit` counts ROWS, not days: the default 168 is roughly two to three
        weeks, not one. How far back the history goes depends on what this
        phone uploaded, not on a fixed window — read `coverage.oldest_available`
        and `coverage.more_available`. For one number per day use `get_metric`.
        Reads YOUR data only: HRV is never shared between partners.

        `rows` is a table in `columns` order (`local_time`, `sdnn_ms`, plus
        `sample_count` for hourly). Carries a `coverage` block: quote
        `coverage.days_covered` (distinct days, not the row count) beside any number.
        """

        return _intraday_table(
            await service.intraday_values(
                series=series, granularity=granularity, limit=limit, owner=service.person_owner(), fresh=fresh
            )
        )

    # ── Analysis ───────────────────────────────────────────────────────────
    #
    # Three tools, one `series` parameter each, instead of a trend tool per kind:
    # the arithmetic is identical across kinds, and a per-kind family would be
    # 45 tools that all have to be kept in step. `list_metric_series` is what
    # makes the parameter guessable — an agent reads the catalog, it does not
    # guess a name and get an error.
    #
    # 🔑 They exist because an agent asked for a trend WILL produce one, and a
    # Pearson coefficient computed token-by-token is the least reliable number
    # an LLM emits. Moving the arithmetic here does not make the analysis
    # better-founded; it makes it DETERMINISTIC and identical between two
    # sessions asking the same question — which is the property the raw-rows
    # route cannot have. Interpretation stays out (see `analysis.py`).

    @tool(title="List analysable series", annotations=_read_only_tool())
    async def list_metric_series(
        partner: bool = False, fresh: bool = False
    ) -> dict[str, Any]:
        """Every series, with how much data actually backs each one.

        Call this BEFORE guessing a `series` name, and before concluding a kind is
        empty. Each row carries `rows` / `first_date` / `last_date` / `latest`, so
        "does this person track VO2max at all" is answered here in one call instead
        of by reading the kind and getting nothing back.

        `rows` = the days that HAVE a value for that series — counted the way
        `get_metric` reads it, so `rows: 2` means `get_metric` can return at most 2
        points. A kind's series can differ: a partner with 14 weigh-ins may have
        a BMI on only 2 of those days.

        `cumulative: true` marks the series whose daily value ACCRUES over the day
        (steps, energy, water). On those the newest value is routinely a partial day
        and must not be compared against completed ones — that is what makes "steps
        are down today" wrong at 9am. Measurements of a state (resting HR, VO2max)
        carry `false` and need no such care.

        Sleep appears here as numbers (duration, timing, stages, awakenings,
        vitals); read the nights themselves with `get_sleep_nights`. Kinds with a
        richer shape (workouts, strength sets, food, notes, symptoms, cycle) are
        deliberately absent — flattening them to one number per day would answer a
        question you did not ask; read them with their own tool.
        """

        return {
            "series": await service.series_overview(owner=service.person_owner(partner=partner), fresh=fresh),
            "note": _SERIES_CATALOG_NOTE,
        }

    @tool(title="Metric trend", annotations=_read_only_tool())
    async def get_metric_trend(
        series: str, days: int = 30, partner: bool = False, fresh: bool = False,
        since: str | None = None, until: str | None = None,
    ) -> dict[str, Any]:
        """Least-squares trend for one daily series: slope per day, endpoints, spread.

        `series` is a name from `list_metric_series`. `days` selects the newest N days
        THAT HAVE DATA, not the last N calendar days — compare `n_days` with
        `span_days` to see whether the history is dense or sparse. `since` /
        `until` ("YYYY-MM-DD") fit a calendar window instead.

        Returns arithmetic only. `slope_per_day` is in the series' own unit per day and
        carries no threshold, band or verdict; fewer than 3 days returns a null slope
        with a reason rather than a number fitted to noise.

        Carries a `coverage` block over the days the arithmetic used: quote
        `coverage.days_covered` and `coverage.span_days` beside any number here.
        `coverage.window_satisfied: false` alone does NOT mean a short history — it is
        also false when older days exist beyond the window; `coverage.more_available`
        tells the two apart.
        """

        return await service.metric_trend(
            series=series, days=days, owner=service.person_owner(partner=partner), fresh=fresh,
            since=since, until=until,
        )

    @tool(title="Compare two periods", annotations=_read_only_tool())
    async def compare_metric_periods(
        series: str, days: int = 7, partner: bool = False, fresh: bool = False,
        period: str | None = None, baseline: str | None = None,
    ) -> dict[str, Any]:
        """Compare two periods of one series: the newest `days` vs the `days` before, or two named windows.

        To compare CALENDAR periods — "this semester vs the summer", "before and
        after I started training" — pass `period` and `baseline`, each
        "YYYY-MM-DD..YYYY-MM-DD" (an open end is allowed: "2026-09-01.."). `days`
        is then ignored. Without them, the rule below applies.

        "Before" means the next-oldest days WITH DATA, so a gap makes the earlier
        window older rather than emptier — read `previous.first_day` / `previous.last_day`
        to see which period you actually got, and quote them.

        Returns both windows' mean/median/min/max and the differences. It does not say
        which window is better; that depends on the metric and on the person.

        Carries a `coverage` block over the days the arithmetic used: quote
        `coverage.days_covered` and `coverage.span_days` beside any number here.
        `coverage.window_satisfied: false` alone does NOT mean a short history — it is
        also false when older days exist beyond the window; `coverage.more_available`
        tells the two apart.
        """

        return await service.metric_compare_periods(
            series=series, days=days, owner=service.person_owner(partner=partner), fresh=fresh,
            period=period, baseline=baseline,
        )

    @tool(title="Correlate two series", annotations=_read_only_tool())
    async def correlate_metric_series(
        series_a: str, series_b: str, days: int = 30, partner: bool = False, fresh: bool = False,
        since: str | None = None, until: str | None = None, lag_days: int = 0,
    ) -> dict[str, Any]:
        """Pearson r between two daily series, over days that have BOTH recorded.

        `lag_days` pairs `series_a` on each day with `series_b` that many days
        LATER (negative = earlier): "does a short night show up in HRV the day
        after" is sleep vs HRV with lag_days=1. Sleep is already dated by the
        morning it ends, so lag 0 is "last night vs today". `since` / `until`
        ("YYYY-MM-DD") restrict to a calendar window.

        Days missing on either side are dropped, never interpolated and never read as
        zero, so `n_pairs` is usually smaller than either series — quote it with `r`.
        Fewer than 3 shared days returns null with a reason: any two points are
        perfectly collinear, so a coefficient there is an artefact.

        The result carries a `caveat` field about causation. Repeat its substance in
        your answer, and do not translate r into a word like "strong".

        Carries a `coverage` block over the days the arithmetic used: quote
        `coverage.days_covered` and `coverage.span_days` beside any number here.
        `coverage.window_satisfied: false` alone does NOT mean a short history — it is
        also false when older days exist beyond the window; `coverage.more_available`
        tells the two apart.
        """

        return await service.metric_correlate(
            series_a=series_a, series_b=series_b, days=days, owner=service.person_owner(partner=partner),
            fresh=fresh, since=since, until=until, lag_days=lag_days,
        )

    if selected_transport == "stdio":
        mcp.run(transport="stdio")
        return

    _serve_streamable_http(
        mcp,
        host=host,
        port=port,
        token=token,
        allow_remote=allow_remote,
        path=_normalize_http_path(path),
        json_response=json_response,
        stateless_http=stateless_http,
    )


def _serve_streamable_http(
    mcp: Any,
    *,
    host: str,
    port: int,
    token: str | None,
    allow_remote: bool,
    path: str = "/mcp",
    json_response: bool = True,
    stateless_http: bool = True,
) -> None:
    """Fail closed before binding a network-reachable socket, then gate with the token."""

    if not _is_loopback(host):
        if not token:
            raise RuntimeError(
                f"Refusing to bind {host}: HTTP transport on a non-loopback address exposes "
                "decrypted sleep data. Run `serve --generate-token` (or set VAULTBEAT_MCP_HTTP_TOKEN), "
                "or bind 127.0.0.1."
            )
        if not allow_remote:
            raise RuntimeError(
                f"Refusing to bind {host}: non-loopback exposure must be confirmed with "
                "--allow-remote. Front it with TLS (a reverse proxy) before exposing beyond a trusted LAN."
            )

    import uvicorn  # transitive dep of mcp; imported lazily so the stdio path never needs it

    # `host` is forwarded because the SDK keys its DNS-rebinding protection on it
    # (auto-on for loopback) — the same behaviour v1 derived from the constructor.
    inner = mcp.streamable_http_app(
        streamable_http_path=path,
        json_response=json_response,
        stateless_http=stateless_http,
        host=host,
    )
    app: Any = StaticBearerASGIMiddleware(inner, token) if token else inner
    uvicorn.run(app, host=host, port=port, log_level="info")
