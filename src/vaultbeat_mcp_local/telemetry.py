"""Usage telemetry for the MCP server → PostHog (owner, 2026-09-23: 「加posthog」).

WHY IT EXISTS. Until this module the only observable trace of an agent using
Vaultbeat was an edge-function log line per sync: which account pulled data,
never which tool, from which client, or whether it failed. The one question
the business turns on — "do paying users actually use the MCP, and where does
it break for them" — had no data at all (the owner's standing rule: a failure the user
can see must have a PostHog event).

WHAT IT MAY SEND — the same red line as the app (`docs/code-map.md`
§ 埋点隐私红线), restated because this code runs on the user's own machine:

  ✅ tool name · ok / error · error CLASS name · our own short error codes
     (`mixed_owners`, `invalid_aggregation` — constants we wrote, never text)
     · duration bucket · whether `partner` was set · client product name and
     version (e.g. "claude-code") · protocol version · package / SDK version ·
     OS family.
  ❌ tool ARGUMENTS and RESULTS of any kind · exception messages (they can
     quote user input) · hostnames, paths, locale, timezone · any id except
     the Supabase account id this machine was paired to — the same
     `distinct_id` the iOS app already uses, so MCP usage lands on the person
     who pays, and nothing new identifies anyone.

OFF when: `VAULTBEAT_TELEMETRY` is 0/false/off/no · `DO_NOT_TRACK` is set to a
truthy value (the community convention) · demo mode · the machine is not
paired (no account id — there is nothing honest to attribute an event to, and
minting an install id would be a new identifier needing its own argument) ·
running under pytest.

HOW IT MUST BEHAVE. It is observation, never a dependency: events go through a
bounded in-memory queue drained by a daemon thread, every failure is
swallowed, a full queue drops events rather than blocking, and nothing is
ever written to stdout — on the stdio transport stdout IS the protocol, and a
stray byte there corrupts the session.
"""

from __future__ import annotations

import atexit
import os
import platform
import queue
import re
import sys
import threading
import time
from datetime import datetime, timezone
from collections.abc import Mapping
from typing import Any, Callable

from vaultbeat_mcp_local import __version__

#: Public client key of the Vaultbeat PostHog project (234261) — the same one
#: compiled into the iOS app. `phc_` keys are write-only ingestion keys that
#: ship in every client by design; this is not a secret.
POSTHOG_KEY = "phc_rhMivZUSTSmK6XEsJz3CF8MHuCsp6RfhoB5TzM7jjRSG"
#: EU region, same as the app (GDPR; the owner's own EU plans).
POSTHOG_BATCH_URL = "https://eu.i.posthog.com/batch/"

OPT_OUT_ENV = "VAULTBEAT_TELEMETRY"
_FALSY = {"0", "false", "off", "no"}
_ERROR_CODE = re.compile(r"^[a-z][a-z0-9_]{0,48}$")
_QUEUE_MAX = 500


def telemetry_disabled_reason(*, demo: bool) -> str | None:
    """Why telemetry is off, or None when it is on."""

    if os.environ.get(OPT_OUT_ENV, "").strip().lower() in _FALSY:
        return f"{OPT_OUT_ENV} is off"
    dnt = os.environ.get("DO_NOT_TRACK", "").strip().lower()
    if dnt and dnt not in _FALSY:
        return "DO_NOT_TRACK is set"
    if demo:
        return "demo mode"
    if "pytest" in sys.modules or os.environ.get("PYTEST_CURRENT_TEST"):
        return "running under pytest"
    return None


def duration_bucket(ms: float) -> str:
    if ms < 200:
        return "<200ms"
    if ms < 1000:
        return "200-1000ms"
    if ms < 3000:
        return "1-3s"
    if ms < 10000:
        return "3-10s"
    return ">10s"


class _Sender:
    """Bounded queue + one daemon thread POSTing batches. Never raises."""

    def __init__(self, post: Callable[[list[dict[str, Any]]], None] | None = None) -> None:
        self._queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=_QUEUE_MAX)
        self._post = post or _http_post
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._idle = threading.Event()  # set = there is work in the queue
        self._inflight = False

    def put(self, event: dict[str, Any]) -> None:
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            return
        with self._lock:
            self._idle.set()
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name="vaultbeat-telemetry", daemon=True)
                self._thread.start()
                atexit.register(self.flush, 2.0)

    def _drain(self) -> list[dict[str, Any]]:
        batch: list[dict[str, Any]] = []
        while len(batch) < 50:
            try:
                batch.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return batch

    def _send(self, batch: list[dict[str, Any]]) -> None:
        if not batch:
            return
        try:
            self._post(batch)
        except Exception:  # noqa: BLE001 — observation must never break the server
            pass

    def _run(self) -> None:
        # 🔴 No "wait a second to batch" here, and the in-flight flag is set
        # BEFORE the event leaves the queue. The first version slept 1s holding
        # an event it had already taken out of the queue; when the client
        # disconnected, `flush` at exit found an empty queue and returned, the
        # daemon thread died mid-sleep, and the LAST call of every session was
        # lost — measured 2026-09-23 against PostHog: two sessions, both ending
        # in `vaultbeat_doctor`, neither doctor call arrived. The last call is
        # disproportionately the one that failed, i.e. the one worth having.
        while True:
            self._idle.wait()
            with self._lock:
                self._inflight = True
                batch = self._drain()
            try:
                self._send(batch)
            finally:
                with self._lock:
                    self._inflight = False
                    if self._queue.empty():
                        self._idle.clear()

    def flush(self, timeout: float = 2.0) -> None:
        """Send what is queued, and wait for a send already in flight, within *timeout*."""

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                busy = self._inflight
                batch = [] if busy else self._drain()
            if batch:
                self._send(batch)
            elif not busy:
                return
            else:
                time.sleep(0.05)


DEBUG_ENV = "VAULTBEAT_TELEMETRY_DEBUG"


def _http_post(batch: list[dict[str, Any]]) -> None:
    import httpx

    # Transparency switch: every event is printed to STDERR before it is sent,
    # so anyone can see exactly what leaves their machine. Never stdout — on
    # the stdio transport that is the protocol stream.
    debug = os.environ.get(DEBUG_ENV, "").strip().lower() not in ("", *_FALSY)
    if debug:
        import json

        for event in batch:
            print(f"[vaultbeat telemetry] {json.dumps(event, ensure_ascii=False)}", file=sys.stderr, flush=True)
    response = httpx.post(POSTHOG_BATCH_URL, json={"api_key": POSTHOG_KEY, "batch": batch}, timeout=3.0)
    if debug:
        print(f"[vaultbeat telemetry] -> HTTP {response.status_code}", file=sys.stderr, flush=True)


class Telemetry:
    """Builds whitelisted events and hands them to the sender."""

    def __init__(
        self,
        *,
        account_id: Callable[[], str | None],
        demo: bool,
        transport: str,
        sender: _Sender | None = None,
    ) -> None:
        self._account_id = account_id
        self._disabled = telemetry_disabled_reason(demo=demo)
        self._sender = sender or _Sender()
        self._base = {
            "app": "vaultbeat-mcp",
            "surface": "mcp",
            "$lib": "vaultbeat-mcp",
            "mcp_package_version": __version__,
            "mcp_sdk_version": _sdk_version(),
            "transport": transport,
            "os": platform.system().lower(),
        }

    @property
    def enabled(self) -> bool:
        return self._disabled is None

    def capture(self, event: str, properties: dict[str, Any]) -> None:
        if self._disabled is not None:
            return
        try:
            distinct_id = self._account_id()
        except Exception:  # noqa: BLE001
            return
        if not distinct_id:
            return
        self._sender.put(
            {
                "event": event,
                "distinct_id": distinct_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "properties": {**self._base, **properties},
            }
        )


def _sdk_version() -> str:
    try:
        from importlib.metadata import version

        return version("mcp")
    except Exception:  # noqa: BLE001
        return "unknown"


def _client_properties(ctx: Any) -> dict[str, Any]:
    """Client product name/version and protocol version, when the SDK has them.

    On `initialize` the session does not hold the client info yet (the handler
    that stores it runs inside `call_next`), so it is read from the request's
    own `clientInfo` — the first version read the session and every
    `mcp_client_connected` arrived without a name.
    """

    props: dict[str, Any] = {}
    try:
        props["protocol_version"] = str(ctx.protocol_version)
    except Exception:  # noqa: BLE001
        pass
    name = version = None
    try:
        raw = ctx.params.get("clientInfo") if isinstance(ctx.params, Mapping) else None
        if isinstance(raw, Mapping):
            name, version = raw.get("name"), raw.get("version")
        else:
            params = ctx.session.client_params
            info = getattr(params, "client_info", None) if params is not None else None
            if info is not None:
                name, version = getattr(info, "name", None), getattr(info, "version", None)
    except Exception:  # noqa: BLE001
        pass
    if name:
        props["client_name"] = str(name)[:64]
    if version:
        props["client_version"] = str(version)[:32]
    return props


def _outcome(result: Any) -> dict[str, Any]:
    """ok / error for a tool result, with OUR error code when there is one."""

    is_error = getattr(result, "is_error", None)
    structured = getattr(result, "structured_content", None)
    if isinstance(result, dict):
        is_error = result.get("isError", is_error)
        structured = result.get("structuredContent", structured)
    out: dict[str, Any] = {"ok": not bool(is_error)}
    if isinstance(structured, dict):
        code = structured.get("error")
        # Only identifier-shaped codes we wrote ourselves; any free text is dropped.
        if isinstance(code, str) and _ERROR_CODE.match(code):
            out["ok"] = False
            out["error_code"] = code
    return out


class PostHogMiddleware:
    """SDK 2.x `ServerMiddleware`: `mcp_client_connected` per `initialize`, `mcp_tool_called` per call."""

    def __init__(self, telemetry: Telemetry) -> None:
        self._telemetry = telemetry

    async def __call__(self, ctx: Any, call_next: Any) -> Any:
        if not self._telemetry.enabled or ctx.method not in ("initialize", "tools/call"):
            return await call_next(ctx)

        started = time.monotonic()
        try:
            result = await call_next(ctx)
        except Exception as error:
            if ctx.method == "tools/call":
                self._emit_tool(ctx, started, {"ok": False, "error_type": type(error).__name__})
            raise
        if ctx.method == "initialize":
            # Only 2025-era clients (every real one as of 2026-09) send
            # `initialize`; the 2026-07-28 protocol has no handshake at all. So
            # this event answers "connected but never called a tool" for today's
            # clients, and every `mcp_tool_called` carries the client name anyway.
            self._telemetry.capture("mcp_client_connected", _client_properties(ctx))
        else:
            self._emit_tool(ctx, started, _outcome(result))
        return result

    def _emit_tool(self, ctx: Any, started: float, outcome: dict[str, Any]) -> None:
        params = ctx.params if isinstance(ctx.params, Mapping) else {}
        name = params.get("name")
        raw_args = params.get("arguments")
        arguments = raw_args if isinstance(raw_args, Mapping) else {}
        props: dict[str, Any] = {
            "tool": name if isinstance(name, str) and _ERROR_CODE.match(name) else "unknown",
            "duration_bucket": duration_bucket((time.monotonic() - started) * 1000),
            **outcome,
            **_client_properties(ctx),
        }
        if "partner" in arguments:
            props["partner"] = bool(arguments.get("partner"))
        self._telemetry.capture("mcp_tool_called", props)
