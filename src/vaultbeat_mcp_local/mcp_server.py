from __future__ import annotations

import functools
import hmac
import inspect
import ipaddress
import json
from typing import Any, Callable, TypeVar, cast

from vaultbeat_mcp_local import __version__
from vaultbeat_mcp_local.demo import DEMO_BANNER, demo_enabled
from vaultbeat_mcp_local.service import VaultbeatLocalService
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
# FastMCP lazily so a missing SDK produces a sentence instead of a traceback; a
# module-level `from mcp.types import ...` here would raise first and make that
# whole guard dead code.


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
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=open_world,
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
        readOnlyHint=False,
        destructiveHint=destructive,
        idempotentHint=idempotent,
        openWorldHint=False,
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


def _watermark_demo(result: Any) -> Any:
    """Stamp a synthetic-data marker onto a tool result.

    Add-only and non-destructive, same discipline as `_annotate_if_empty`: it
    never reads, edits or drops an existing key. The marker goes FIRST in the
    dict so it survives a client that truncates a long payload from the end.

    🔴 `demo_warning` is the FIRST key, ahead of the boolean. The result is
    serialised with `json.dumps`, which preserves insertion order, so the first
    key is literally the first thing in the text an agent receives — and
    `"demo_mode": true` is a flag someone has to already know to look for,
    while the banner is a sentence that acts on a reader who does not. Ordering
    is free; which of the two goes first is not.

    A result that already carries `demo_mode` (`vaultbeat_status`,
    `vaultbeat_doctor`, which build a richer block of their own) is returned
    untouched rather than double-stamped — those two order their own keys the
    same way.
    """

    if not isinstance(result, dict) or "demo_mode" in result:
        return result
    return {
        "demo_warning": DEMO_BANNER,
        "demo_mode": True,
        **_mark_demo_rows(result),
    }


def _mark_demo_rows(result: dict[str, Any]) -> dict[str, Any]:
    """Tag each row of a top-level list-of-dicts with `synthetic: True`.

    The top-level banner covers a result quoted whole; it does NOT survive a row
    being lifted out of it. Most read tools self-identify anyway because their
    rows carry `owner_user_id`, which in demo mode reads `demo0001-…` — but the
    two aggregating tools build brand-new dicts (`{"day": …, "basal_kcal": …}`)
    and drop that field on the way, so `get_basal_energy` and
    `get_total_energy_burned` were the only two whose rows were, taken alone,
    indistinguishable from real ones (verified 2026-08-20: 17 of 19 read tools
    self-identify, those two did not).

    Done HERE rather than in those two methods on purpose — one site, so a third
    aggregating tool is covered on the day it is written rather than on the day
    someone notices. It is also structurally absent from a real run: the wrapper
    is only installed when demo mode is on.

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


def _demo_wrap(function: _F) -> _F:
    """Wrap one tool so its output is watermarked and its docstring says DEMO.

    Two shapes, because this server has both: `vaultbeat_status` and
    `vaultbeat_start_binding` are plain `def`, everything else is `async def`.
    A single sync wrapper around a coroutine function would hand FastMCP a
    coroutine object as the tool's result — the tool would "succeed" and return
    something unserialisable.

    `functools.wraps` is load-bearing beyond cosmetics: FastMCP builds each
    tool's JSON schema from `inspect.signature(fn)`, which follows the
    `__wrapped__` attribute wraps sets — so the published schema stays the real
    function's, not `(*args, **kwargs)`.
    """

    if inspect.iscoroutinefunction(function):

        @functools.wraps(function)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            return _watermark_demo(await function(*args, **kwargs))

        _prefix_doc(async_wrapper, function)
        return cast(_F, async_wrapper)

    @functools.wraps(function)
    def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        return _watermark_demo(function(*args, **kwargs))

    _prefix_doc(sync_wrapper, function)
    return cast(_F, sync_wrapper)


def _prefix_doc(wrapper: Any, original: Any) -> None:
    """Put the demo warning at the top of the description FastMCP will publish."""

    doc = inspect.getdoc(original) or ""
    wrapper.__doc__ = f"{_DEMO_DOC_PREFIX}\n\n{doc}" if doc else _DEMO_DOC_PREFIX


# ── Trial-expiry note (vb-016) ───────────────────────────────────────────────


def _annotate_access(result: Any, service: VaultbeatLocalService) -> Any:
    """Attach a one-line trial-expiry heads-up to a tool result, add-only.

    Same discipline as `_annotate_if_empty` / `_watermark_demo`: never reads,
    edits or drops an existing key, and does nothing at all outside the last
    24 hours of a pairing-time trial deadline. The sentence is generated
    entirely client-side (Anti-pattern 23) from a timestamp this machine
    already holds — the point is that the agent can say "your trial ends
    tomorrow" BEFORE the first notice the user gets is a mid-conversation
    refusal.

    Results that already carry an `access` block (`vaultbeat_status`,
    `vaultbeat_doctor`) are left alone — they say the same thing with more
    context. `access_note_if_expiring` reads an in-process stash set by the
    bound call the tool just made, so this costs no config or Keychain I/O.
    """

    if not isinstance(result, dict) or "access" in result or "access_note" in result:
        return result
    note = service.access_note_if_expiring()
    if note is None:
        return result
    return {**result, "access_note": note}


def _access_wrap(function: _F, service: VaultbeatLocalService) -> _F:
    """Wrap one tool so its dict results pick up the trial-expiry note.

    Mirrors `_demo_wrap`'s two shapes (this server has both sync and async
    tools) including `functools.wraps`, which FastMCP's schema builder relies
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


def _annotate_if_empty(result: dict[str, Any], kind: str, rows_key: str) -> dict[str, Any]:
    """Explain an empty result for a kind that older apps never wrote.

    An agent calling `get_strength_log` on an account whose app predates
    strength logging gets `{"sessions": []}` — indistinguishable from "you never
    trained" and from "something is broken". There is no other signal anywhere:
    no error, no warning. This attaches the reason to the result itself so the
    agent can relay it without having to know that a separate diagnostic exists.

    ADD-ONLY, deliberately: it introduces a `hint` key and never reads, edits or
    removes an existing field, and it does nothing at all when there IS data.
    Payload shape is the one contract layer no server can validate (see AGENTS.md
    § app↔MCP coupling), so anything touching a response has to be additive.
    """
    if result.get(rows_key):
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
            f"nearly empty and fills in. Open the app and tap Settings → Data & AI "
            f"→ 'Re-sync all health data to AI', then retry in a few minutes. "
            f"(2) Apple Health access for it was never granted — a read denial is "
            f"invisible to the app, so it looks identical to having no data; "
            f"recover via Settings → Data & AI → 'Apple Health access'. (3) it "
            f"genuinely has not been recorded yet. Run `vaultbeat-mcp doctor` or "
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
        f"and tap Settings → Data & AI → 'Re-sync all health data to AI', then retry "
        f"in a few minutes; (2) the iOS app predates this data type, which needs a "
        f"build from {since} or later; (3) Apple Health access for it was never "
        f"granted — a read denial is invisible to the app, so this looks the same as "
        f"having no data, and the recovery path is the app's Settings → Data & AI → "
        f"'Apple Health access' row, which re-presents the permission sheet; (4) it "
        f"genuinely has not been recorded yet. Run `vaultbeat-mcp doctor` or call the "
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
        from mcp.server.fastmcp import FastMCP
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "The MCP SDK is not installed. Install with `pip install -e ./mcp-local-server`."
        ) from error

    selected_transport = _normalize_transport(transport)

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
    mcp = FastMCP(
        "Vaultbeat Health [DEMO — SYNTHETIC DATA]" if demo_active else "Vaultbeat Health",
        host=host,
        port=port,
        streamable_http_path=_normalize_http_path(path),
        json_response=json_response,
        stateless_http=stateless_http,
    )

    # FastMCP takes no `version`, and when the inner server's is None the SDK falls
    # back to reporting ITS OWN package version — so clients showed "1.29.0" (the mcp
    # SDK) for a user who had just installed vaultbeat-mcp 0.3.x. Wrong version numbers
    # are worse than absent ones: they make a stale install indistinguishable from a
    # current one, which is exactly the question `doctor` exists to answer.
    # Guarded because it reaches past the public API — a future SDK that renames this
    # attribute costs a cosmetic version string, never a working server.
    inner_server = getattr(mcp, "_mcp_server", None)
    if inner_server is not None and hasattr(inner_server, "version"):
        inner_server.version = __version__

    def tool(*, title: str, annotations: Any) -> Callable[[_F], _F]:
        """Register one tool — the single place demo mode can reach every tool.

        A choke point, not a convenience wrapper (Invariant 58): the alternative
        is 29 call sites each remembering to watermark, which is 29 chances to
        forget and no way to notice the one that did.

        Patching `mcp.call_tool` after construction does NOT work and was tried:
        `FastMCP.__init__` calls `_setup_handlers`, which binds the bound method
        into the low-level server at construction time, so a later reassignment
        is never consulted. Decorating on the way in is the only hook that
        exists.
        """

        def decorator(function: _F) -> _F:
            # Access note innermost (it inspects the plain result), demo
            # watermark outermost. In demo mode the note is inert anyway —
            # the stash it reads is only ever set by a bound call, which demo
            # mode returns before reaching.
            prepared = _access_wrap(function, service)
            if demo_active:
                prepared = _demo_wrap(prepared)
            return cast(_F, mcp.tool(title=title, annotations=annotations)(prepared))

        return decorator

    @tool(title="Connection status", annotations=_read_only_tool())
    def vaultbeat_status() -> dict[str, Any]:
        """Return local Vaultbeat binding state without exposing private keys or server tokens."""

        return service.status()

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
        may predate the feature entirely.

        `scope` — what this report does NOT cover. Everything here can pass and
        the setup still be broken on the client side: this server is a subprocess
        of your MCP client, so it cannot read the client's config file, cannot see
        which environment variables the client forwarded, and cannot tell whether
        it was launched with the arguments the user believes. A green report never
        clears the client. `scope.env_overrides_received` lists which Vaultbeat
        variables actually arrived — check that before guessing at the client's
        environment.

        This runs a cloud round trip, so it is slower than `vaultbeat_status`
        (local config only) — prefer status for a quick liveness check.
        """

        return await service.doctor()

    @tool(title="Sleep history", annotations=_read_only_tool())
    async def vaultbeat_sync_sleep(
        limit: int = 50, owner: str | None = None, fresh: bool = False
    ) -> dict[str, Any]:
        """Fetch encrypted Vaultbeat sleep records, decrypt them locally, and return
        per-day primary session summaries matching the iOS app's display.

        Returns `daily_summary` (one primary session per local date, selected by
        iOS priority: Watch > iPhone > inBedOnly) and `sessions` (all raw records).
        The `limit` controls how many raw blobs are fetched; 50 covers ~2-3 weeks.
        Use `owner` prefix to filter by person (e.g. "a1a1" for one user,
        "b2b2" for the other) — without it, both partners' data is mixed and
        per-day selection may pick the wrong person's session.

        ⚠️ `is_in_bed_only: true` means sleep was NEVER MEASURED that night (the
        Watch wasn't worn) — NOT that the person slept zero. On those nights
        `total_sleep_minutes` is 0, `duration_label` reads "no sleep data", and
        `in_bed_minutes` holds the time actually recorded in bed. Report such a
        night as "no sleep data (in bed ~Xh)", never as "slept 0 hours".

        Results are served from a short-lived local cache (default 10 min);
        pass fresh=True to force a cloud round trip.
        """

        return _annotate_if_empty(
            await service.sleep_records(limit=limit, owner=owner, fresh=fresh),
            "sleep",
            "sessions",
        )

    @tool(title="Water intake", annotations=_read_only_tool())
    async def get_water_intake(
        limit: int = 30, owner: str | None = None, fresh: bool = False
    ) -> dict[str, Any]:
        """Decrypt recent daily water intake locally and compute the average.

        Returns one entry per day (newest first) with refill count, container volume, and
        derived intake in liters, plus `average_daily_intake_liters` over the window.
        Use `owner` prefix to filter by person (e.g. "a1a1" or "b2b2").
        Each record carries `owner_user_id` to identify whose data it is.
        """

        return _annotate_if_empty(
            await service.water_intake_summary(limit=limit, owner=owner, fresh=fresh),
            "water",
            "days",
        )

    @tool(title="Weight trend", annotations=_read_only_tool())
    async def get_weight_trend(
        limit: int = 90, goal_kg: float | None = None, owner: str | None = None, fresh: bool = False
    ) -> dict[str, Any]:
        """Decrypt recent body-weight records locally and compute the trend.

        Returns one entry per day (newest first, kilograms) plus latest/average/min/max,
        the OLS weekly rate (kg/week), and — when `goal_kg` is given — the distance to goal.
        Use `owner` prefix to filter by person (e.g. "a1a1" or "b2b2").
        Each record carries `owner_user_id` to identify whose data it is.
        """

        return _annotate_if_empty(
            await service.weight_trend_summary(
                limit=limit, goal_kg=goal_kg, owner=owner, fresh=fresh
            ),
            "body",
            "days",
        )

    @tool(title="Symptoms", annotations=_read_only_tool())
    async def get_symptoms(limit: int = 120, fresh: bool = False) -> dict[str, Any]:
        """Decrypt recent HealthKit symptom days locally, grouped by data owner.

        SENSITIVE: symptom data (cramps, headache, fatigue, coughing…) only reaches
        this server when a user explicitly opted in on iOS — their own AI toggle for
        their own data, or the partner-AI toggle for a partner's data. Both partners
        can track symptoms, so each entry in `owners` carries `owner_user_id` plus
        per-type counts and day-by-day samples with severity
        (mild/moderate/severe/present/…). Stays on-device, never re-exported.
        """

        return await service.symptom_summary(limit=limit, fresh=fresh)

    @tool(title="Notes", annotations=_read_only_tool())
    async def get_notes(limit: int = 120, target_kind: str | None = None, fresh: bool = False) -> dict[str, Any]:
        """Decrypt recent free-text notes (day annotations) locally.

        SENSITIVE free text. Each note carries `owner_user_id` (who wrote it),
        `target_kind`, and `target_date` (the local day it annotates) — join
        against the same-day metric data for pattern analysis. Kinds:
        "sleep" | "menstrual" are written manually in the iOS app by either
        partner (e.g. "昨晚舍友很吵" on a sleep day); "mood" | "general" are
        agent-authored via `log_note`. Pass target_kind to filter.
        Stays on-device, never re-exported.
        """

        return await service.notes_summary(limit=limit, target_kind=target_kind, fresh=fresh)

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
        the pairing handshake); an older bind must re-pair by calling
        `vaultbeat_start_binding` then `vaultbeat_poll_binding`.
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
        older bind must re-pair via `vaultbeat_start_binding` then
        `vaultbeat_poll_binding`.
        """

        return await service.log_strength_entry(
            date=date, exercises=exercises, note=None, merge=True
        )

    @tool(title="Food log", annotations=_read_only_tool())
    async def get_food_log(
        limit: int = 90, limit_days: int | None = None, fresh: bool = False
    ) -> dict[str, Any]:
        """Decrypt recent daily food-intake logs locally (newest first).

        Each day carries `meals`, each meal a list of `items` with `food` (name),
        optional free-text `portion` ("1 根" / "300g" / "小份"), optional
        per-item/per-meal `note`, and — when the logging agent estimated them —
        optional structured nutrition numbers (`kcal`, `proteinGrams`,
        `fatGrams`, `carbGrams`). Items without those fields need analysis-time
        estimation from name + portion; items with them can be summed directly.
        Owner's own days only. Pass limit_days to cap how many days return.
        """

        return _annotate_if_empty(
            await service.food_summary(limit=limit, limit_days=limit_days, fresh=fresh),
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
        text: str, kind: str = "general", date: str | None = None, merge: bool = False
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

        For symptoms, use `log_note_append`. Discomfort shows up in installments
        across a day (nausea at noon, dizziness at night), so the second write
        of the day is the normal case, not the exception — and this tool would
        replace the morning's entry with the evening's.

        `date` = LOCAL calendar day "YYYY-MM-DD" (default today). Read back via
        `get_notes` (optionally `target_kind="mood"`/`"general"`). Encrypted
        end-to-end before it ever leaves this machine.
        """

        return await service.log_note(text=text, kind=kind, date=date, merge=merge)

    @tool(title="Add to note", annotations=_mutating_tool(destructive=False))
    async def log_note_append(
        text: str,
        kind: str = "general",
        date: str | None = None,
    ) -> dict[str, Any]:
        """Add a line to a day's note WITHOUT erasing what is already there (agent write).

        There is one note per (kind, day). This tool appends your `text` to it on a
        new line and cannot delete what is already written.

        USE THIS FOR SYMPTOMS. Discomfort arrives in installments across a day —
        nausea at noon, dizziness at night — so the second write of the day is the
        normal case, not the exception. `log_note` would replace the morning's
        entry with the evening's.

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
        """

        return await service.log_note(text=text, kind=kind, date=date, merge=True)

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
        `get_weight_trend`). Whether it also reaches Apple Health depends on an
        iOS setting: Settings → Data & AI → "Allow AI to update Apple Health"
        (OFF by default). When it is on, weigh-ins logged here sync back into
        Apple Health on the next app sync.

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
        limit: int = 60, owner: str | None = None, fresh: bool = False
    ) -> dict[str, Any]:
        """Decrypt recent menstrual cycle data locally and predict the next period.

        SENSITIVE: menstrual data only reaches this server if the user explicitly opted
        in on iOS; it stays on-device and is never re-exported. Returns recent samples plus
        a next-period prediction. Use `owner` prefix to filter by person.
        """

        return await service.menstrual_cycle_summary(limit=limit, owner=owner, fresh=fresh)

    # Same family, milder: since 2026-08-11 this leaves credentials alone (see
    # its docstring), but it still overwrites `poll_id` — which invalidates a QR
    # code already on screen, the exact hazard `vaultbeat_poll_binding` warns
    # about. The hint stays True; the title now says what for.
    @tool(
        title="Start pairing (invalidates any open QR)",
        annotations=_mutating_tool(destructive=True),
    )
    def vaultbeat_start_binding(server_name: str = "Local AI Server") -> dict[str, Any]:
        """Initialize a binding session: generates a keypair (if needed) and returns a
        QR payload that the user scans in the Vaultbeat iOS app to authorize this AI server.

        ⚠️ If the user says they cannot see the QR code, believe them. Many
        terminals and most agent transcripts drop the block characters it is
        drawn with, so the payload can reach you intact while their screen shows
        a blank gap — do not assert that it is there. Have them run
        `uvx vaultbeat-mcp bind` in a real terminal instead.

        Returns `qr_payload_json` — a JSON string the AI should render as a QR code
        for the user to scan, plus `poll_id` to pass to `vaultbeat_poll_binding`.
        After the user scans, call `vaultbeat_poll_binding` to complete authorization.
        The iOS path is Settings → Data & AI → Connect an AI server.

        Opening a session does NOT disturb an existing binding: the current
        credentials keep working until a scan actually lands, and are only
        replaced at that moment. (Before 2026-08-11 this call wiped them up
        front, so an agent "just re-checking" silently unbound its owner.)
        """

        session = service.start_binding(server_name=server_name)
        return {
            "poll_id": session.poll_id,
            "qr_payload": session.qr_payload,
            "qr_payload_json": session.qr_payload_json,
        }

    # "Check pairing status" until 2026-08-20, which pointed the opposite way to
    # this tool's own `destructiveHint: True` — and the annotation is right. Its
    # SUCCESS branch is the write: `store.update` replaces server_id, server_token
    # (rotated even when the identity is unchanged), the owner identity and
    # bound_at, and on a new identity also clears last_sync_at and the decrypted
    # cache. A title that says "Check" invites the reflexive approval that a
    # destructive hint exists to prevent. House style is `verb (consequence)`,
    # as in "Log food (replaces day)".
    @tool(
        title="Finish pairing (replaces this binding)",
        annotations=_mutating_tool(destructive=True),
    )
    async def vaultbeat_poll_binding() -> dict[str, Any]:
        """Check whether the user has scanned the QR code and authorized this server.

        Call this after `vaultbeat_start_binding`. THREE possible `status` values:

        · "pending"  — the pairing is alive and simply has not been scanned yet.
                       Keep polling with short delays.
        · "bound"    — success. The server can now decrypt health data.
        · "expired"  — TERMINAL. Stop polling; no amount of retrying recovers it.
                       Run `uvx vaultbeat-mcp bind` for a fresh QR code.

        🔴 "pending" is positive evidence that nothing is wrong. The endpoint
        looks the pairing row up by pollID and answers "expired" when it is gone,
        so a long run of "pending" means the row is still there and nobody has
        scanned — NOT that it went stale while you waited.

        That distinction decides what you tell the user, and getting it backwards
        is destructive: re-running `bind` mints a NEW pollID, which invalidates
        the QR they are looking at — so "just run bind again" turns a pairing
        that was one scan away from working into one that cannot complete. Only
        do it on "expired".

        So while it stays "pending", the useful action is to get the code
        scanned, not to restart anything. Connecting is open on every plan, so
        there is no tier to check; the scanner is at Settings → Data & AI →
        "Connect an AI server". Do not send them to check the network, reinstall
        this server, or run diagnostics — none of those are implicated.

        If they cannot see a QR code at all, it is your output that failed, not
        their phone — see `vaultbeat_start_binding`.
        """

        result = await service.poll_once()
        return {
            "status": result.status,
            "server_id": result.server_id,
            "owner_user_id": result.owner_user_id,
        }

    @tool(title="Sleep stage detail", annotations=_read_only_tool())
    async def get_sleep_detail(
        limit: int = 2,
        owner: str | None = None,
        fresh: bool = False,
        include_timeline: bool = False,
    ) -> dict[str, Any]:
        """Per-night sleep stages with per-stage HR/RR — depth on a few nights.

        Which sleep tool to use:
        · THIS one for going deep on one or two specific nights (stage bands,
          per-stage vitals). It defaults to 2 nights because each is ~1-2k
          characters; see the SIZE note below.
        · `vaultbeat_sync_sleep` for anything spanning time — "how did I sleep
          this week/month", trends, averages. Its default of 50 covers ~2-3
          weeks. Reach for it whenever the question is about a period rather
          than a night, and do not conclude from THIS tool's two rows that only
          two nights exist.

        (This used to call itself "the primary tool for detailed sleep analysis",
        which read as "use this one for sleep" and handed back two days to
        anyone who asked how their week went.)

        Returns `stage_intervals` (contiguous stage bands with start/end),
        `stage_minutes`, and `stage_vitals` (per-stage HR/RR min/mean/max). Use
        `owner` prefix to filter by person (e.g. "a1a1" for linyou, "b2b2" for
        partner).

        ⚠️ SIZE: each night is ~1-2k characters as returned. Setting
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
        """

        return await service.sleep_detail_records(
            limit=limit, owner=owner, fresh=fresh, include_timeline=include_timeline
        )

    @tool(title="Activity rings", annotations=_read_only_tool())
    async def get_activity(limit: int = 30, owner: str | None = None, fresh: bool = False) -> dict[str, Any]:
        """Decrypt recent daily activity rings (steps, active energy kcal, exercise minutes,
        stand hours, distance km). One entry per day, newest first.
        Use `owner` prefix to filter by person. Each record carries `owner_user_id`."""

        return _annotate_if_empty(
            await service.activity_summary(limit=limit, owner=owner, fresh=fresh),
            "activity",
            "days",
        )

    @tool(title="Resting heart rate", annotations=_read_only_tool())
    async def get_resting_hr(limit: int = 30, owner: str | None = None, fresh: bool = False) -> dict[str, Any]:
        """Decrypt recent resting heart rate samples (bpm). Returns per-day records
        plus average over the window. Use `owner` prefix to filter by person."""

        return await service.resting_hr_records(limit=limit, owner=owner, fresh=fresh)

    @tool(title="Workouts", annotations=_read_only_tool())
    async def get_workouts(limit: int = 30, owner: str | None = None, fresh: bool = False) -> dict[str, Any]:
        """Decrypt recent workout sessions (type, duration, calories, distance).
        Use `owner` prefix to filter by person."""

        return await service.workout_records(limit=limit, owner=owner, fresh=fresh)

    @tool(title="Heart rate variability", annotations=_read_only_tool())
    async def get_hrv(
        limit: int = 30,
        owner: str | None = None,
        fresh: bool = False,
        granularity: str = "hourly",
    ) -> dict[str, Any]:
        """Decrypt recent HRV (SDNN in ms) — returns records plus average over the window.

        `granularity` selects between two backing kinds:
        - `"hourly"` (default) — routes to `hrv_hourly` kind: one bucket per
          UTC hour (arithmetic mean of every raw SDNN sample in the hour).
          **30-day rolling window**, ≤720 records, includes `sample_count`
          per bucket. Records also carry a `sdnn_ms` alias equal to the
          hourly mean, so callers migrating from the pre-build-77 raw
          default keep working without a field rename. Right for trend /
          aggregate queries — SAVES CONTEXT vs raw.
        - `"raw"` — routes to `hrv` kind: one record per SDNN sample (Apple
          Watch emits every 5-15min). **3-day rolling window**; older raw
          history lives in prior-recipient envelopes plus the
          `VaultbeatHistoryBackfillCoordinator`-driven historical push
          (advances 30d/24h on device wake-ups, up to 5 years). Use for
          spike-precision questions (e.g. "HRV during the 3 minutes I
          opened a stressful message"). Note: single-day count is often
          30-100+ records.

        ⚠️ `average_sdnn_ms` from the two granularities is **NOT directly
        comparable** — they observe different windows (3d vs 30d) and, on
        the raw side, also include legacy per-sample blobs from before
        build 77. Use hourly for "what has my HRV been lately?" trend
        answers; use raw only when you need per-sample precision inside
        the last ~3 days. The equivalence claim in previous doc versions
        was retracted 2026-07-22 after an adversarial review pointed out
        the window mismatch.

        Use `owner` prefix to filter by person (e.g. `"a1a1"` / `"b2b2"`).
        """

        if granularity == "raw":
            return await service.hrv_records(limit=limit, owner=owner, fresh=fresh)

        # Default + explicit "hourly" → aggregate kind, but fall back to raw
        # when it yields nothing.
        #
        # `hrv_hourly` only started being written by iOS build 77 (2026-07-22).
        # An App Store user on 1.2.0 has plenty of raw HRV and zero hourly
        # buckets, so serving the empty hourly result would report "no HRV
        # data" to someone whose HRV data is sitting right there — a straight
        # regression against 0.1.2, where get_hrv read raw unconditionally.
        # Version skew between app and MCP is normal and permanent (users
        # update the two independently), so the aggregate kind degrades to the
        # kind it aggregates instead of pretending the data is absent.
        hourly = await service.hrv_hourly_records(limit=limit, owner=owner, fresh=fresh)
        if hourly.get("records"):
            return hourly

        raw = await service.hrv_records(limit=limit, owner=owner, fresh=fresh)
        if not raw.get("records"):
            return hourly  # genuinely no HRV at all — keep the hourly shape

        raw["granularity"] = "raw"
        raw["granularity_note"] = (
            "Requested hourly averages, but this account has none — hourly HRV "
            "requires the iOS app from 2026-07-22 or later. Returned raw "
            "per-sample HRV instead; the numbers are the same measurements, "
            "just not hour-averaged."
        )
        return raw

    @tool(title="Wrist temperature", annotations=_read_only_tool())
    async def get_wrist_temp(limit: int = 30, owner: str | None = None, fresh: bool = False) -> dict[str, Any]:
        """Decrypt recent sleeping wrist temperature samples — ABSOLUTE °C.

        ⚠️ These are absolute skin temperatures (~35.5-36.5 °C), NOT baseline
        deltas — the legacy `temperature_delta_celsius` field name is a wire-
        contract misnomer (kept for compatibility; prefer the honest twin
        `wrist_temperature_celsius`). For cycle analysis, derive the deviation
        yourself: reading minus that person's rolling baseline. One sample per
        night. Use `owner` prefix to filter."""

        return await service.wrist_temp_records(limit=limit, owner=owner, fresh=fresh)

    @tool(title="Basal energy (BMR)", annotations=_read_only_tool())
    async def get_basal_energy(limit: int | None = None, owner: str | None = None, fresh: bool = False) -> dict[str, Any]:
        """Decrypt recent basal-energy-burned samples (Apple Watch BMR estimate, kcal).
        Watch typically emits hourly samples; unlimited limit + daily aggregation
        returns per-day BMR (~1500-2000 kcal for active young adults) + average.
        Use `owner` prefix to filter by person.

        READ `hours_covered` BEFORE QUOTING ANY SINGLE DAY. Basal arrives as one
        blob per hour, so a day the Watch spent off the wrist comes back as a
        real-looking row that is short in exact proportion — 883 kcal at 12 of
        24 hours is half a day of data, NOT a collapsed metabolism. Rows with
        `incomplete: true` are already excluded from `average_daily_basal_kcal`
        (`average_over_days` is its denominator); if you quote such a day, say
        how many hours it covers."""

        return _annotate_if_empty(
            await service.basal_energy_records(limit=limit, owner=owner, fresh=fresh),
            "basal_energy",
            "daily",
        )

    @tool(title="Total energy burned (TDEE)", annotations=_read_only_tool())
    async def get_total_energy_burned(days: int = 7, owner: str | None = None, fresh: bool = False) -> dict[str, Any]:
        """TDEE (total daily energy expenditure) = basal + active per day, last N days.

        The truthful daily calorie burn from Watch's actual measurements — not
        a formula. Diet targets need to aim BELOW this to lose weight (e.g.
        eating avg_tdee - 500 = ~0.5 kg/week loss). Returns per-day breakdown
        {day, basal_kcal, active_kcal, total_kcal, basal_missing, partial,
        basal_hours_covered, basal_hours_expected, basal_incomplete} + average
        TDEE. Three kinds of day are excluded from the average and each is
        listed with its reason in `average_excluded_days`: today (`partial`,
        still accumulating), days with no basal data (`basal_missing`), and
        days whose Watch coverage was short (`basal_incomplete` — e.g. 16 of 24
        hours). A short day's kcal is low in proportion to the hours it missed,
        so including it drags the average down and, since the error is
        one-directional, never cancels out. Quote `average_tdee_kcal` for diet
        targets, and if you quote a single day, check `basal_incomplete` first.
        Use `owner` prefix to filter by person."""

        return await service.total_energy_burned(days=days, owner=owner, fresh=fresh)

    @tool(title="VO₂ max", annotations=_read_only_tool())
    async def get_vo2max(limit: int = 30, owner: str | None = None, fresh: bool = False) -> dict[str, Any]:
        """Decrypt recent VO2Max samples (Apple Watch cardiorespiratory fitness).
        Unit: mL/(kg·min); higher = better. Male 20-29 reference: <35 poor,
        35-42 fair, 42-46 good, 46-50 excellent, 50+ superior. Returns
        newest-first records plus latest / peak / trough / average over the
        window. Use `owner` prefix to filter by person. VO2Max is sparse (Watch
        computes it during outdoor brisk walk/run bouts, days apart), so a
        limit of 30 usually covers many months."""

        return _annotate_if_empty(
            await service.vo2max_records(limit=limit, owner=owner, fresh=fresh),
            "vo2max",
            "records",
        )

    @tool(title="Mindfulness", annotations=_read_only_tool())
    async def get_mindfulness(limit: int = 30, owner: str | None = None, fresh: bool = False) -> dict[str, Any]:
        """Decrypt recent daily mindfulness summaries (session count, total minutes).
        Use `owner` prefix to filter by person."""

        return await service.mindfulness_summary(limit=limit, owner=owner, fresh=fresh)

    if selected_transport == "stdio":
        mcp.run(transport="stdio")
        return

    _serve_streamable_http(mcp, host=host, port=port, token=token, allow_remote=allow_remote)


def _serve_streamable_http(
    mcp: Any,
    *,
    host: str,
    port: int,
    token: str | None,
    allow_remote: bool,
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

    inner = mcp.streamable_http_app()
    app: Any = StaticBearerASGIMiddleware(inner, token) if token else inner
    uvicorn.run(app, host=host, port=port, log_level="info")
