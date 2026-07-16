from __future__ import annotations

import hmac
import ipaddress
import json
from typing import Any

from tether_mcp_local.service import TetherLocalService
from tether_mcp_local.store import ConfigStore


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
                        (b"www-authenticate", b'Bearer realm="tether-mcp-local"'),
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

    service = TetherLocalService(store or ConfigStore())
    selected_transport = _normalize_transport(transport)
    mcp = FastMCP(
        "Tether Local Sleep",
        host=host,
        port=port,
        streamable_http_path=_normalize_http_path(path),
        json_response=json_response,
        stateless_http=stateless_http,
    )

    @mcp.tool()
    def tether_status() -> dict[str, Any]:
        """Return local Tether binding state without exposing private keys or server tokens."""

        return service.status()

    @mcp.tool()
    async def tether_sync_sleep(
        limit: int = 50, owner: str | None = None, fresh: bool = False
    ) -> dict[str, Any]:
        """Fetch encrypted Tether sleep records, decrypt them locally, and return
        per-day primary session summaries matching the iOS app's display.

        Returns `daily_summary` (one primary session per local date, selected by
        iOS priority: Watch > iPhone > inBedOnly) and `sessions` (all raw records).
        The `limit` controls how many raw blobs are fetched; 50 covers ~2-3 weeks.
        Use `owner` prefix to filter by person (e.g. "dce9" for one user,
        "f835" for the other) — without it, both partners' data is mixed and
        per-day selection may pick the wrong person's session.
        Results are served from a short-lived local cache (default 10 min);
        pass fresh=True to force a cloud round trip.
        """

        return await service.sleep_records(limit=limit, owner=owner, fresh=fresh)

    @mcp.tool()
    async def get_water_intake(
        limit: int = 30, owner: str | None = None, fresh: bool = False
    ) -> dict[str, Any]:
        """Decrypt recent daily water intake locally and compute the average.

        Returns one entry per day (newest first) with refill count, container volume, and
        derived intake in liters, plus `average_daily_intake_liters` over the window.
        Use `owner` prefix to filter by person (e.g. "dce9" or "f835").
        Each record carries `owner_user_id` to identify whose data it is.
        """

        return await service.water_intake_summary(limit=limit, owner=owner, fresh=fresh)

    @mcp.tool()
    async def get_weight_trend(
        limit: int = 90, goal_kg: float | None = None, owner: str | None = None, fresh: bool = False
    ) -> dict[str, Any]:
        """Decrypt recent body-weight records locally and compute the trend.

        Returns one entry per day (newest first, kilograms) plus latest/average/min/max,
        the OLS weekly rate (kg/week), and — when `goal_kg` is given — the distance to goal.
        Use `owner` prefix to filter by person (e.g. "dce9" or "f835").
        Each record carries `owner_user_id` to identify whose data it is.
        """

        return await service.weight_trend_summary(limit=limit, goal_kg=goal_kg, owner=owner, fresh=fresh)

    @mcp.tool()
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

    @mcp.tool()
    async def get_notes(limit: int = 120, target_kind: str | None = None, fresh: bool = False) -> dict[str, Any]:
        """Decrypt recent free-text notes (sleep/menstrual day annotations) locally.

        SENSITIVE free text written manually in Tether by either partner — e.g.
        "昨晚舍友很吵" on a sleep day, or a period-day remark. Each note carries
        `owner_user_id` (who wrote it), `target_kind` ("sleep" | "menstrual"),
        and `target_date` (the local day it annotates) — join against the
        same-day metric data for pattern analysis. Pass target_kind to filter.
        Stays on-device, never re-exported.
        """

        return await service.notes_summary(limit=limit, target_kind=target_kind, fresh=fresh)

    @mcp.tool()
    async def get_menstrual_cycle(
        limit: int = 60, owner: str | None = None, fresh: bool = False
    ) -> dict[str, Any]:
        """Decrypt recent menstrual cycle data locally and predict the next period.

        SENSITIVE: menstrual data only reaches this server if the user explicitly opted
        in on iOS; it stays on-device and is never re-exported. Returns recent samples plus
        a next-period prediction. Use `owner` prefix to filter by person.
        """

        return await service.menstrual_cycle_summary(limit=limit, owner=owner, fresh=fresh)

    @mcp.tool()
    def tether_start_binding(server_name: str = "Local AI Server") -> dict[str, Any]:
        """Initialize a binding session: generates a keypair (if needed) and returns a
        QR payload that the user scans in the Tether iOS app to authorize this AI server.

        Returns `qr_payload_json` — a JSON string the AI should render as a QR code
        for the user to scan, plus `poll_id` to pass to `tether_poll_binding`.
        After the user scans, call `tether_poll_binding` to complete authorization.
        """

        session = service.start_binding(server_name=server_name)
        return {
            "poll_id": session.poll_id,
            "qr_payload": session.qr_payload,
            "qr_payload_json": session.qr_payload_json,
        }

    @mcp.tool()
    async def tether_poll_binding() -> dict[str, Any]:
        """Check whether the user has scanned the QR code and authorized this server.

        Call this after `tether_start_binding`. Returns `status`: "pending" (user hasn't
        scanned yet — wait and retry) or "bound" (success — server is now authorized
        and can decrypt health data). Polls once; call repeatedly with short delays
        until status is "bound" or you decide to time out.
        """

        result = await service.poll_once()
        return {
            "status": result.status,
            "server_id": result.server_id,
            "owner_user_id": result.owner_user_id,
        }

    @mcp.tool()
    async def get_sleep_detail(
        limit: int = 5, owner: str | None = None, fresh: bool = False
    ) -> dict[str, Any]:
        """Per-night timeline: each HR/RR sample tagged with the concurrent sleep stage.

        Returns chronological `timeline` array (hr, rr, stage, time), `stage_intervals`
        (contiguous stage bands with start/end), `stage_minutes`, and `stage_vitals`
        (per-stage HR/RR min/mean/max). Use `owner` prefix to filter by person
        (e.g. "dce9" for linyou, "f835" for partner). This is the primary tool for
        detailed sleep analysis — richer than tether_sync_sleep.
        """

        return await service.sleep_detail_records(limit=limit, owner=owner, fresh=fresh)

    @mcp.tool()
    async def get_activity(limit: int = 30, owner: str | None = None, fresh: bool = False) -> dict[str, Any]:
        """Decrypt recent daily activity rings (steps, active energy kcal, exercise minutes,
        stand hours, distance km). One entry per day, newest first.
        Use `owner` prefix to filter by person. Each record carries `owner_user_id`."""

        return await service.activity_summary(limit=limit, owner=owner, fresh=fresh)

    @mcp.tool()
    async def get_resting_hr(limit: int = 30, owner: str | None = None, fresh: bool = False) -> dict[str, Any]:
        """Decrypt recent resting heart rate samples (bpm). Returns per-day records
        plus average over the window. Use `owner` prefix to filter by person."""

        return await service.resting_hr_records(limit=limit, owner=owner, fresh=fresh)

    @mcp.tool()
    async def get_workouts(limit: int = 30, owner: str | None = None, fresh: bool = False) -> dict[str, Any]:
        """Decrypt recent workout sessions (type, duration, calories, distance).
        Use `owner` prefix to filter by person."""

        return await service.workout_records(limit=limit, owner=owner, fresh=fresh)

    @mcp.tool()
    async def get_hrv(limit: int = 30, owner: str | None = None, fresh: bool = False) -> dict[str, Any]:
        """Decrypt recent HRV (SDNN in ms) samples. Returns per-day records
        plus average over the window. Use `owner` prefix to filter by person."""

        return await service.hrv_records(limit=limit, owner=owner, fresh=fresh)

    @mcp.tool()
    async def get_wrist_temp(limit: int = 30, owner: str | None = None, fresh: bool = False) -> dict[str, Any]:
        """Decrypt recent sleeping wrist temperature samples (°C baseline deviation).
        Returns per-day records plus average over the window. Use `owner` prefix to filter."""

        return await service.wrist_temp_records(limit=limit, owner=owner, fresh=fresh)

    @mcp.tool()
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
                "decrypted sleep data. Run `serve --generate-token` (or set TETHER_MCP_HTTP_TOKEN), "
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
