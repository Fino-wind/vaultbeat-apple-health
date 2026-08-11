from __future__ import annotations

import hmac
import ipaddress
import json
from typing import Any

from vaultbeat_mcp_local.service import VaultbeatLocalService
from vaultbeat_mcp_local.store import ConfigStore


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
    since = VaultbeatLocalService.KIND_MIN_APP_RELEASE.get(kind)
    if since is None or result.get(rows_key):
        return result
    result.setdefault(
        "hint",
        f"No {kind} data on this account. Three causes produce an identical empty "
        f"result and this server cannot tell them apart — it only sees that no rows "
        f"arrived: (1) the iOS app predates this data type, which needs a build from "
        f"{since} or later; (2) Apple Health access for it was never granted — a read "
        f"denial is invisible to the app, so this looks the same as having no data, and "
        f"the recovery path is the app's Settings → Data & AI → 'Apple Health access' "
        f"row, which re-presents the permission sheet; (3) it genuinely has not been "
        f"recorded yet. Run `vaultbeat-mcp doctor` or call the vaultbeat_doctor tool "
        f"for a full report.",
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

    service = VaultbeatLocalService(store or ConfigStore())
    selected_transport = _normalize_transport(transport)
    mcp = FastMCP(
        "Vaultbeat Local Sleep",
        host=host,
        port=port,
        streamable_http_path=_normalize_http_path(path),
        json_response=json_response,
        stateless_http=stateless_http,
    )

    @mcp.tool()
    def vaultbeat_status() -> dict[str, Any]:
        """Return local Vaultbeat binding state without exposing private keys or server tokens."""

        return service.status()

    @mcp.tool()
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

        This runs a cloud round trip, so it is slower than `vaultbeat_status`
        (local config only) — prefer status for a quick liveness check.
        """

        return await service.doctor()

    @mcp.tool()
    async def vaultbeat_sync_sleep(
        limit: int = 50, owner: str | None = None, fresh: bool = False
    ) -> dict[str, Any]:
        """Fetch encrypted Vaultbeat sleep records, decrypt them locally, and return
        per-day primary session summaries matching the iOS app's display.

        Returns `daily_summary` (one primary session per local date, selected by
        iOS priority: Watch > iPhone > inBedOnly) and `sessions` (all raw records).
        The `limit` controls how many raw blobs are fetched; 50 covers ~2-3 weeks.
        Use `owner` prefix to filter by person (e.g. "dce9" for one user,
        "f835" for the other) — without it, both partners' data is mixed and
        per-day selection may pick the wrong person's session.

        ⚠️ `is_in_bed_only: true` means sleep was NEVER MEASURED that night (the
        Watch wasn't worn) — NOT that the person slept zero. On those nights
        `total_sleep_minutes` is 0, `duration_label` reads "no sleep data", and
        `in_bed_minutes` holds the time actually recorded in bed. Report such a
        night as "no sleep data (in bed ~Xh)", never as "slept 0 hours".

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

    @mcp.tool()
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

    @mcp.tool()
    async def log_strength_entry(
        date: str,
        exercises: list[dict[str, Any]],
        note: str | None = None,
        merge: bool = False,
    ) -> dict[str, Any]:
        """Log one strength-training session on the owner's behalf (agent write).

        ⚠️ DEFAULT IS REPLACE-THE-WHOLE-DAY: with `merge=False`, the supplied
        `exercises` become the day's ENTIRE session — any exercise you don't
        re-send is silently deleted. To ADD an exercise or a forgotten set to a
        day that already has one, pass `merge=True`: your exercises are appended
        to the existing session (an exercise whose `name` matches an existing one
        gets its sets appended to it), and nothing already logged can be lost.

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

    @mcp.tool()
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

    @mcp.tool()
    async def log_food_entry(
        date: str, meals: list[dict[str, Any]], note: str | None = None, merge: bool = False
    ) -> dict[str, Any]:
        """Log one day's food intake on the owner's behalf (agent write).

        ⚠️ DEFAULT IS REPLACE-THE-WHOLE-DAY: with `merge=False`, the supplied
        `meals` become the day's ENTIRE log — any meal you don't re-send is
        silently deleted. To ADD a meal/snack to a day that already has
        entries, pass `merge=True`: your meals are appended to the existing
        day (a meal whose `name` matches an existing meal gets its items
        appended to that meal) and nothing already logged can be lost.

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
        Encrypted end-to-end before it ever leaves this machine.
        """

        return await service.log_food_entry(date=date, meals=meals, note=note, merge=merge)

    @mcp.tool()
    async def log_note(
        text: str, kind: str = "general", date: str | None = None, merge: bool = False
    ) -> dict[str, Any]:
        """Log a free-text note on the owner's behalf (agent write).

        For narratives that belong next to the metric data instead of in chat
        history: `kind="mood"` for emotional state ("为什么今天情绪低落"),
        `kind="general"` for day events worth joining against sleep/HRV later.
        (sleep/menstrual notes stay iOS-authored — this tool refuses them.)

        ⚠️ DEFAULT IS REPLACE-THE-WHOLE-NOTE: there is one note per (kind, day),
        and with `merge=False` your `text` becomes its ENTIRE contents — anything
        already written for that kind+day is silently deleted. To ADD to a day
        that already has a note, pass `merge=True`: your text is appended on a
        new line and nothing already logged can be lost.

        The result carries `replaced_text` — the note this call deleted. If it is
        non-null and you did not intend to replace, you just destroyed that text;
        re-send it with `merge=True`.

        Use `merge=True` for symptoms. Discomfort shows up in installments
        across a day (nausea at noon, dizziness at night), so the second write
        of the day is the normal case, not the exception.

        `date` = LOCAL calendar day "YYYY-MM-DD" (default today). Read back via
        `get_notes` (optionally `target_kind="mood"`/`"general"`). Encrypted
        end-to-end before it ever leaves this machine.
        """

        return await service.log_note(text=text, kind=kind, date=date, merge=merge)

    @mcp.tool()
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
    def vaultbeat_start_binding(server_name: str = "Local AI Server") -> dict[str, Any]:
        """Initialize a binding session: generates a keypair (if needed) and returns a
        QR payload that the user scans in the Vaultbeat iOS app to authorize this AI server.

        ⚠️ REQUIRES VAULTBEAT PRO on the owner's iPhone (monthly, yearly or
        lifetime — the AI agent interface is the Pro tier). Mention this when you
        show the QR code: without Pro that row opens a paywall instead of the
        scanner, so they never reach a scannable state.

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

    @mcp.tool()
    async def vaultbeat_poll_binding() -> dict[str, Any]:
        """Check whether the user has scanned the QR code and authorized this server.

        Call this after `vaultbeat_start_binding`. Returns `status`: "pending" (user hasn't
        scanned yet — wait and retry) or "bound" (success — server is now authorized
        and can decrypt health data). Polls once; call repeatedly with short delays
        until status is "bound" or you decide to time out.

        ⚠️ Stuck on "pending" forever? Ask them to read the subtitle under
        Settings → Data & AI → "Connect an AI server" — that line answers it
        without tapping anything. "Vaultbeat Pro unlocks the AI agent interface"
        means the tier is the whole problem: binding needs Pro, and the iOS app
        creates no pending record at all until they have it. Do not send them to
        re-scan, check the network, reinstall this server, or run diagnostics;
        nothing on this side is broken. "Scan the QR code shown by your local MCP
        server" means the tier is fine and they simply have not scanned yet.
        """

        result = await service.poll_once()
        return {
            "status": result.status,
            "server_id": result.server_id,
            "owner_user_id": result.owner_user_id,
        }

    @mcp.tool()
    async def get_sleep_detail(
        limit: int = 2,
        owner: str | None = None,
        fresh: bool = False,
        include_timeline: bool = False,
    ) -> dict[str, Any]:
        """Per-night sleep stages with per-stage HR/RR. The primary tool for detailed
        sleep analysis — richer than vaultbeat_sync_sleep.

        Returns `stage_intervals` (contiguous stage bands with start/end),
        `stage_minutes`, and `stage_vitals` (per-stage HR/RR min/mean/max). Use
        `owner` prefix to filter by person (e.g. "dce9" for linyou, "f835" for
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

        Use `owner` prefix to filter by person (e.g. `"dce9"` / `"f835"`).
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

    @mcp.tool()
    async def get_wrist_temp(limit: int = 30, owner: str | None = None, fresh: bool = False) -> dict[str, Any]:
        """Decrypt recent sleeping wrist temperature samples — ABSOLUTE °C.

        ⚠️ These are absolute skin temperatures (~35.5-36.5 °C), NOT baseline
        deltas — the legacy `temperature_delta_celsius` field name is a wire-
        contract misnomer (kept for compatibility; prefer the honest twin
        `wrist_temperature_celsius`). For cycle analysis, derive the deviation
        yourself: reading minus that person's rolling baseline. One sample per
        night. Use `owner` prefix to filter."""

        return await service.wrist_temp_records(limit=limit, owner=owner, fresh=fresh)

    @mcp.tool()
    async def get_basal_energy(limit: int | None = None, owner: str | None = None, fresh: bool = False) -> dict[str, Any]:
        """Decrypt recent basal-energy-burned samples (Apple Watch BMR estimate, kcal).
        Watch typically emits hourly samples; unlimited limit + daily aggregation
        returns per-day BMR (~1500-2000 kcal for active young adults) + average.
        Use `owner` prefix to filter by person."""

        return _annotate_if_empty(
            await service.basal_energy_records(limit=limit, owner=owner, fresh=fresh),
            "basal_energy",
            "daily",
        )

    @mcp.tool()
    async def get_total_energy_burned(days: int = 7, owner: str | None = None, fresh: bool = False) -> dict[str, Any]:
        """TDEE (total daily energy expenditure) = basal + active per day, last N days.

        The truthful daily calorie burn from Watch's actual measurements — not
        a formula. Diet targets need to aim BELOW this to lose weight (e.g.
        eating avg_tdee - 500 = ~0.5 kg/week loss). Returns per-day breakdown
        {day, basal_kcal, active_kcal, total_kcal, basal_missing, partial} +
        average TDEE. Today's row is flagged `partial` (still accumulating)
        and excluded from the average — quote complete days for targets.
        Use `owner` prefix to filter by person."""

        return await service.total_energy_burned(days=days, owner=owner, fresh=fresh)

    @mcp.tool()
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
