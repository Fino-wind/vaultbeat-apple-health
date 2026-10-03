from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import secrets
import time
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone, tzinfo
from collections.abc import Awaitable, Callable
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from typing import Any, Protocol, TypeVar

from vaultbeat_mcp_local.analysis import (
    SERIES,
    compare_periods,
    correlate,
    daily_series,
    excluded_days as series_excluded_days,
    lookup as series_lookup,
    series_catalog,
    trend,
)
from vaultbeat_mcp_local.app_paths import AUTHORIZED_SERVERS, CONNECT_SERVER, HEALTH_ACCESS, RESYNC
from vaultbeat_mcp_local.cache import LocalRecordCache
from vaultbeat_mcp_local.client import (
    _ISO_TIMESTAMP,
    _REQUEST_ID,
    PollBindingResult,
    VaultbeatBlobOwnerConflictError,
    VaultbeatCloudClient,
    VaultbeatCloudError,
    VaultbeatTrialExpiredError,
    VaultbeatUnsupportedMetricError,
    server_token,
)
from vaultbeat_mcp_local.crypto import (
    RecipientKey,
    VaultbeatCryptoError,
    VaultbeatDekMismatchError,
    decode_json_payload,
    decrypt_blob_payload,
    encrypt_blob_payload,
)
from vaultbeat_mcp_local.store import (
    DEFAULT_API_BASE_URL,
    PAIRING_GUIDANCE,
    ConfigError,
    ConfigStore,
    LocalServerConfig,
    now_iso,
)


_LOG = logging.getLogger("vaultbeat_mcp_local.service")

_T = TypeVar("_T")

# Health kinds carried in encrypted_sleep_blobs.metric_type. Decryption is identical
# for every kind (Curve25519 ECDH + HKDF-SHA256 + AES-GCM); only the post-decrypt JSON
# decode/aggregate differs. "sleep" stays the historical default for legacy blobs that
# predate metric_type tagging.
METRIC_SLEEP = "sleep"
METRIC_WATER = "water"
METRIC_MENSTRUAL = "menstrual"
METRIC_BODY = "body"
METRIC_ACTIVITY = "activity"
METRIC_RESTING_HR = "resting_hr"
METRIC_WORKOUT = "workout"
METRIC_MINDFULNESS = "mindfulness"
METRIC_HRV = "hrv"
METRIC_HRV_HOURLY = "hrv_hourly"
METRIC_WRIST_TEMP = "wrist_temp"
METRIC_SYMPTOM = "symptom"
METRIC_NOTE = "note"
METRIC_STRENGTH = "strength"
METRIC_FOOD = "food"
METRIC_VO2MAX = "vo2max"
METRIC_BASAL_ENERGY = "basal_energy"
# One record per user: biological sex, date of birth, height (GitHub #14).
METRIC_PROFILE = "profile"

# Every metric kind this layer understands. Doubles as the safety gate for
# anything derived from a caller-supplied metric_type (cache file names, the
# edge query parameter): membership here means the value is a known enum
# token, not free text.
KNOWN_METRIC_TYPES = frozenset(
    {
        METRIC_SLEEP,
        METRIC_WATER,
        METRIC_MENSTRUAL,
        METRIC_BODY,
        METRIC_ACTIVITY,
        METRIC_RESTING_HR,
        METRIC_WORKOUT,
        METRIC_MINDFULNESS,
        METRIC_HRV,
        METRIC_HRV_HOURLY,
        METRIC_WRIST_TEMP,
        METRIC_SYMPTOM,
        METRIC_NOTE,
        METRIC_STRENGTH,
        METRIC_FOOD,
        METRIC_VO2MAX,
        METRIC_BASAL_ENERGY,
        METRIC_PROFILE,
    }
)

# ── Basal-energy day completeness ────────────────────────────────────────────
#
# `basal_energy` is uploaded as ONE blob per UTC hour bucket (Invariant 26), so
# a fully-worn, fully-synced day holds 24 of them and the count IS the coverage.
# Anything below the threshold means the Watch was off the wrist or had not
# synced — NOT that the person's metabolism dropped.
#
# 22 (≥91.7% coverage) is picked from the real distribution, not by feel. Over
# 655 days of the owner's history the shape is strongly bimodal — 598 days at
# 24 buckets, then a clean gap, then a long tail at ≤20 — so every threshold in
# 18..22 flags exactly the same days on a 14- or 30-day window. Given a flat
# region, take its top edge: <24 and <23 would also flag days short by a single
# bucket (~80 kcal, ~4%), and a flag that fires on noise is a flag people learn
# to ignore. <16 was rejected outright: it misses 2026-08-19 (16 buckets,
# 1221 kcal), which is precisely the day that must not enter an average.
#
# 22 also survives a move to a DST timezone, which is not hypothetical here —
# a local calendar day is 23 or 25 UTC hours on the two changeover days, and
# <23 would raise a false alarm on one of them every year.
_BASAL_HOURS_PER_DAY = 24
_BASAL_MIN_HOURS_COVERED = 22

# Note target kinds (mirrors iOS VaultbeatNoteTargetKind). Unknown kinds are
# accepted as-is so a newer app adding a kind doesn't brick older decoders.
NOTE_TARGET_KINDS = frozenset({"sleep", "menstrual"})

# Agent-authored note kinds — the only ones `log_note` will WRITE (iOS owns
# sleep/menstrual). Reading is a different question: every kind above and here
# can show up in a `get_notes` result, so a read-side filter must accept the
# union (2026-07-27: the CLI's `notes --kind` offered only the iOS pair, so
# there was no way to filter for the kinds this server itself writes).
AGENT_NOTE_KINDS = frozenset({"mood", "general"})

# Everything `notes_summary` / `notes --kind` may legitimately be asked to keep.
READABLE_NOTE_KINDS = NOTE_TARGET_KINDS | AGENT_NOTE_KINDS

# String forms of the three HK category-value enums the iOS reader maps
# (HKCategoryValueSeverity / HKCategoryValuePresence / HKCategoryValueAppetiteChanges).
# Keep in sync with VaultbeatSymptomHealthKitReader.mapValue.
SYMPTOM_SEVERITY_VALUES = frozenset(
    {
        "unspecified",
        "notPresent",
        "mild",
        "moderate",
        "severe",
        "present",
        "noChange",
        "decreased",
        "increased",
    }
)

# ── Self-reported symptom entries (2026-09-30, GitHub #3) ─────────────────────
#
# A second payload shape under metric_type "symptom". The HealthKit import is one
# blob per (device, day) carrying `dayID` + `samples`; a REPORTED entry is one blob
# per episode carrying `entryID`, written by `log_symptom` or by the app's own
# symptom card. The key that tells them apart is `entryID` — nothing else — so a
# reader never has to guess. Sharing the kind rather than adding one keeps the
# DB CHECK, the mcp-sync whitelist and every other metric-kind registry untouched
# (Invariant 18), and the two shapes land in one `get_symptoms` answer.
#
# Only the four values a person can actually report. HealthKit's presence and
# appetite enums (present / notPresent / noChange / …) stay on the import side:
# logging a symptom already says it is present, and "notPresent" is not an entry.
SYMPTOM_ENTRY_SEVERITIES = frozenset({"unspecified", "mild", "moderate", "severe"})

# HealthKit's symptom identifiers, as the iOS importer writes them
# (`VaultbeatSymptomHealthKitReader.catalog`). A reported type that names one of
# these is stored under the SAME spelling, so "abdominal_cramps" logged by an
# agent and an Apple Health `abdominalCramps` sample count as one symptom.
HEALTHKIT_SYMPTOM_TYPES = frozenset(
    {
        "abdominalCramps", "bloating", "constipation", "diarrhea", "heartburn",
        "nausea", "vomiting", "appetiteChanges", "chills", "dizziness", "fainting",
        "fatigue", "fever", "generalizedBodyAche", "hotFlashes",
        "chestTightnessOrPain", "coughing", "rapidPoundingOrFlutteringHeartbeat",
        "shortnessOfBreath", "skippedHeartbeat", "wheezing", "lowerBackPain",
        "headache", "memoryLapse", "moodChanges", "lossOfSmell", "lossOfTaste",
        "runnyNose", "sinusCongestion", "soreThroat", "breastPain", "pelvicPain",
        "vaginalDryness", "acne", "drySkin", "hairLoss", "nightSweats",
        "sleepChanges", "bladderIncontinence",
    }
)
_HEALTHKIT_SYMPTOM_BY_KEY = {name.lower(): name for name in HEALTHKIT_SYMPTOM_TYPES}

# Bounds on free text. Generous for a person describing their body, tight enough
# that a runaway agent cannot grow one entry past the edge's ciphertext ceiling.
_SYMPTOM_TEXT_MAX = 2_000
_SYMPTOM_LABEL_MAX = 120
_SYMPTOM_TRIGGERS_MAX = 20

# Menstrual flow enum (mirrors the iOS HKCategoryValueVaginalBloodFlow mapping).
MENSTRUAL_FLOW_VALUES = frozenset({"unspecified", "light", "medium", "heavy", "none"})

class CloudClientProtocol(Protocol):
    async def poll_binding(self, poll_id: str) -> PollBindingResult: ...

    async def sync(
        self, server_token: str, *, metric_type: str | None = None
    ) -> list[dict[str, Any]]: ...

    # Catalog mode (2026-07-29). Test doubles that predate it can omit these —
    # the service only calls them when a stored digest exists, and every failure
    # path degrades to `sync`.
    async def sync_digest(
        self, server_token: str, *, metric_type: str | None = None
    ) -> tuple[dict[str, Any] | None, list[dict[str, Any]] | None]: ...

    async def sync_catalog(
        self, server_token: str, *, metric_type: str | None = None
    ) -> list[dict[str, Any]] | None: ...

    async def sync_blobs(
        self, server_token: str, *, blob_ids: list[str], metric_type: str | None = None
    ) -> list[dict[str, Any]]: ...

    async def write_strength_blob(
        self, server_token: str, *, blob: dict[str, Any], envelopes: list[dict[str, Any]]
    ) -> dict[str, Any]: ...

    async def write_food_blob(
        self, server_token: str, *, blob: dict[str, Any], envelopes: list[dict[str, Any]]
    ) -> dict[str, Any]: ...

    async def write_body_blob(
        self, server_token: str, *, blob: dict[str, Any], envelopes: list[dict[str, Any]]
    ) -> dict[str, Any]: ...

    async def write_note_blob(
        self, server_token: str, *, blob: dict[str, Any], envelopes: list[dict[str, Any]]
    ) -> dict[str, Any]: ...

    async def write_symptom_blob(
        self, server_token: str, *, blob: dict[str, Any], envelopes: list[dict[str, Any]]
    ) -> dict[str, Any]: ...

    async def report_decrypt_failures(self, server_token: str, *, items: list[dict[str, str]]) -> None: ...


@dataclass(frozen=True)
class BindingSession:
    poll_id: str
    qr_payload: dict[str, str]
    qr_payload_json: str
    config: LocalServerConfig


@dataclass(frozen=True)
class DecryptedRecord:
    envelope_id: str
    blob_id: str
    metric_type: str | None
    created_at: str | None
    payload: Any
    # Blob owner (whose data this is). Needed because this server holds envelopes
    # for BOTH partners' blobs — e.g. symptoms are tracked by both people, and a
    # summary that can't tell them apart is useless. None until the mcp-sync edge
    # function that returns owner_user_id is deployed (older responses lack it).
    owner_user_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "envelope_id": self.envelope_id,
            "blob_id": self.blob_id,
            "metric_type": self.metric_type,
            "created_at": self.created_at,
            "owner_user_id": self.owner_user_id,
            "payload": self.payload,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> DecryptedRecord:
        """Inverse of `to_dict` — used to rehydrate cache entries.

        🔴 The same shapes `_decrypt_row` enforces, applied again here (review
        V10, 2026-10-03): a cache written before R5 holds the server's columns
        as they came, and replayed them verbatim for as long as the kind's
        digest did not change. Re-checking costs nothing and does not depend
        on the cache ever being refreshed.
        """

        kind = raw.get("metric_type")
        return cls(
            envelope_id=_safe_row_id(raw.get("envelope_id", "")),
            blob_id=_safe_row_id(raw.get("blob_id", "")),
            metric_type=kind if kind in KNOWN_METRIC_TYPES else None,
            created_at=_safe_shaped(raw.get("created_at"), _INSTANT),
            payload=raw.get("payload"),
            owner_user_id=_safe_shaped(raw.get("owner_user_id"), _UUID),
        )


@dataclass(frozen=True)
class WaterDay:
    """One day's water intake decoded from a metric_type="water" blob."""

    day_id: str
    day_start_date: str
    container_volume_liters: float
    refill_count: int
    owner_user_id: str | None = None

    @property
    def intake_liters(self) -> float:
        """Daily intake = number of refills * that day's container volume."""

        return self.refill_count * self.container_volume_liters

    def to_dict(self) -> dict[str, Any]:
        return {
            "day_id": self.day_id,
            "day_start_date": self.day_start_date,
            **_local_date_fields(self.day_start_date),
            "container_volume_liters": self.container_volume_liters,
            "refill_count": self.refill_count,
            "intake_liters": self.intake_liters,
            "owner_user_id": self.owner_user_id,
        }


@dataclass(frozen=True)
class BodyDay:
    """One day's body metrics decoded from a metric_type="body" blob.

    Body weight is shared bidirectionally by default (like sleep, unlike menstrual's
    explicit opt-in). Storage is always kilograms; unit conversion (jin/lb) happens
    only in presentation layers.

    Composition (fat / BMI / lean mass) is populated only when a smart scale wrote
    those samples into Apple Health and iOS read them. They were null on every blob
    until 2026-08-05 — not by design, but because no iOS reader asked HealthKit for
    them while this decoder had parsed them since day one. `body_fat_percent` is
    0–100, converted once on the iOS read edge from HealthKit's native 0–1.
    """

    day_id: str
    day_start_date: str
    weight_kg: float
    body_fat_percent: float | None
    bmi: float | None
    lean_body_mass_kg: float | None = None
    owner_user_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "day_id": self.day_id,
            "day_start_date": self.day_start_date,
            **_local_date_fields(self.day_start_date),
            "weight_kg": self.weight_kg,
            "body_fat_percent": self.body_fat_percent,
            "bmi": self.bmi,
            "lean_body_mass_kg": self.lean_body_mass_kg,
            "owner_user_id": self.owner_user_id,
        }


@dataclass(frozen=True)
class MenstrualSample:
    start_date: str
    end_date: str
    flow: str

    def to_dict(self) -> dict[str, Any]:
        return {"start_date": self.start_date, "end_date": self.end_date, "flow": self.flow}


@dataclass(frozen=True)
class MenstrualDay:
    """One day's menstrual samples decoded from a metric_type="menstrual" blob.

    Menstrual data is sensitive: it only reaches this server when the user explicitly
    opted in on iOS, and never leaves the device beyond locally-decrypted tool results.
    """

    day_id: str
    day_start_date: str
    samples: list[MenstrualSample]
    owner_user_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "day_id": self.day_id,
            "day_start_date": self.day_start_date,
            **_local_date_fields(self.day_start_date),
            "samples": [sample.to_dict() for sample in self.samples],
            "owner_user_id": self.owner_user_id,
        }


@dataclass(frozen=True)
class ActivityDay:
    """One calendar day's activity rings decoded from a metric_type="activity" blob."""

    day_id: str
    day_start_date: str
    step_count: int
    active_energy_kcal: float
    exercise_minutes: int
    stand_minutes: int
    distance_meters: float | None
    owner_user_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "day_id": self.day_id,
            "day_start_date": self.day_start_date,
            **_local_date_fields(self.day_start_date),
            "step_count": self.step_count,
            "active_energy_kcal": self.active_energy_kcal,
            "exercise_minutes": self.exercise_minutes,
            "stand_minutes": self.stand_minutes,
            "distance_meters": self.distance_meters,
            "owner_user_id": self.owner_user_id,
        }


@dataclass(frozen=True)
class RestingHrRecord:
    """One resting heart rate sample decoded from a metric_type="resting_hr" blob."""

    record_id: str
    date: str
    bpm: float
    owner_user_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "date": self.date,
            **_local_date_fields(self.date),
            "bpm": self.bpm,
            "owner_user_id": self.owner_user_id,
        }


@dataclass(frozen=True)
class WorkoutRecord:
    """One workout session decoded from a metric_type="workout" blob."""

    workout_id: str
    activity_type: str
    start_date: str
    end_date: str
    duration_seconds: float
    active_kcal: float | None
    distance_meters: float | None
    owner_user_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "workout_id": self.workout_id,
            "activity_type": self.activity_type,
            "start_date": self.start_date,
            **_local_date_fields(self.start_date, with_time=True),
            "end_date": self.end_date,
            "duration_seconds": self.duration_seconds,
            "active_kcal": self.active_kcal,
            "distance_meters": self.distance_meters,
            "owner_user_id": self.owner_user_id,
        }


@dataclass(frozen=True)
class MindfulnessDay:
    """One calendar day's mindfulness summary decoded from a metric_type="mindfulness" blob."""

    day_id: str
    day_start_date: str
    session_count: int
    total_minutes: float
    owner_user_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "day_id": self.day_id,
            "day_start_date": self.day_start_date,
            **_local_date_fields(self.day_start_date),
            "session_count": self.session_count,
            "total_minutes": self.total_minutes,
            "owner_user_id": self.owner_user_id,
        }


@dataclass(frozen=True)
class HRVRecord:
    """One HRV (SDNN) sample decoded from a metric_type="hrv" blob."""

    record_id: str
    date: str
    sdnn_ms: float
    owner_user_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "date": self.date,
            **_local_date_fields(self.date, with_time=True),
            "sdnn_ms": self.sdnn_ms,
            "owner_user_id": self.owner_user_id,
        }


@dataclass(frozen=True)
class HRVHourlyBucket:
    """One hourly-averaged HRV bucket decoded from a metric_type="hrv_hourly" blob.

    Companion aggregate kind to raw HRV — 30-day rolling window, one blob
    per UTC hour, `avg_sdnn_ms` is the arithmetic mean of every raw SDNN
    sample whose midpoint fell inside the hour (via HKStatisticsCollection
    Query `.discreteAverage` on iOS). `sample_count` lets consumers weight
    buckets when computing a longer-window average or detect low-
    confidence hours (sample_count == 1 = a single 5-min reading).
    """

    record_id: str
    date: str
    avg_sdnn_ms: float
    sample_count: int
    owner_user_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        # `sdnn_ms` is a back-compat alias for callers migrating from the
        # pre-build-77 `get_hrv` default that returned raw per-sample records.
        # Value is identical to `avg_sdnn_ms` — semantically it is the arithmetic
        # mean of every raw SDNN sample in this hour, which is the closest
        # single-value analogue to the old raw kind's per-sample sdnn_ms.
        # Any agent/skill/prompt/jq pipeline that read `records[].sdnn_ms`
        # under the old default keeps working; new callers should prefer
        # `avg_sdnn_ms` (self-documenting) plus `sample_count` for weighting.
        # Adversarial review 2026-07-22 caught the silent-rename regression.
        return {
            "record_id": self.record_id,
            "date": self.date,
            **_local_date_fields(self.date, with_time=True),
            "avg_sdnn_ms": self.avg_sdnn_ms,
            "sdnn_ms": self.avg_sdnn_ms,
            "sample_count": self.sample_count,
            "owner_user_id": self.owner_user_id,
        }


@dataclass(frozen=True)
class WristTempRecord:
    """One sleeping wrist temperature sample decoded from a metric_type="wrist_temp" blob.

    ⚠️ Field-name lie inherited from the wire contract: iOS named the payload
    field `temperatureDeltaCelsius`, but `appleSleepingWristTemperature` is an
    ABSOLUTE skin temperature (observed 35.5-36.5 °C), not a baseline delta —
    the iOS reader stores `sample.quantity.doubleValue(for: .degreeCelsius())`
    verbatim (confirmed 2026-07-24 against live data + HealthKit docs). The
    wire name cannot change without a two-sided migration, so the output keeps
    the legacy key for compatibility and adds an honestly-named twin. Baseline
    deviation must be DERIVED (reading minus the person's rolling baseline),
    which is exactly what the ovulation detector does with day-to-day shifts.
    """

    record_id: str
    date: str
    temperature_delta_celsius: float
    owner_user_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "date": self.date,
            **_local_date_fields(self.date),
            # Honest name first; legacy misnomer kept for back-compat.
            "wrist_temperature_celsius": self.temperature_delta_celsius,
            "temperature_delta_celsius": self.temperature_delta_celsius,
            "owner_user_id": self.owner_user_id,
        }


@dataclass(frozen=True)
class BasalEnergyRecord:
    """One basal-energy-burned sample decoded from a metric_type="basal_energy" blob.

    Watch estimates BMR from age/sex/height/weight + observed HR patterns,
    typically emitting one sample per hour (or more granular). Unit: kcal.
    Sum over a day = daily BMR contribution (typically 1500-2000 kcal for
    active young adults).
    """

    record_id: str
    date: str
    kcal: float
    owner_user_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "date": self.date,
            **_local_date_fields(self.date, with_time=True),
            "kcal": self.kcal,
            "owner_user_id": self.owner_user_id,
        }


@dataclass(frozen=True)
class VO2MaxRecord:
    """One VO2Max sample decoded from a metric_type="vo2max" blob.

    Unit: mL O2 · kg⁻¹ · min⁻¹ (the SI unit Apple Watch reports; iOS
    HKUnit(from: "mL/kg*min")). Higher = better cardiorespiratory fitness.
    Reference bands (male, 20-29): <35 poor · 35-42 fair · 42-46 good ·
    46-50 excellent · 50+ superior.
    """

    record_id: str
    date: str
    vo2_max_ml_kg_min: float
    owner_user_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "date": self.date,
            **_local_date_fields(self.date, with_time=True),
            "vo2_max_ml_kg_min": self.vo2_max_ml_kg_min,
            "owner_user_id": self.owner_user_id,
        }


@dataclass(frozen=True)
class ProfileRecord:
    """The Health Profile decoded from a metric_type="profile" blob (GitHub #14).

    Every field may be None: Apple Health returns the same nothing for "not
    set" and "not allowed", and the app omits a field rather than guess.
    `sex_source` is "chosen" (the person picked it in the app) or
    "apple_health" (a stored field they may never have looked at).
    """

    record_id: str
    biological_sex: str | None
    sex_source: str | None
    date_of_birth: str | None
    height_cm: float | None
    owner_user_id: str | None = None

    def age_on(self, today: date) -> int | None:
        if not self.date_of_birth:
            return None
        try:
            born = date.fromisoformat(self.date_of_birth)
        except ValueError:
            return None
        if born > today:
            return None
        return today.year - born.year - ((today.month, today.day) < (born.month, born.day))

    def to_dict(self, *, today: date) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "biological_sex": self.biological_sex,
            "sex_source": self.sex_source,
            "date_of_birth": self.date_of_birth,
            "age": self.age_on(today),
            "height_cm": self.height_cm,
            "owner_user_id": self.owner_user_id,
        }


@dataclass(frozen=True)
class NoteRecord:
    """One free-text annotation pinned to (target_kind, local day), decoded from a
    metric_type="note" blob.

    Dual-source: both partners write notes from their own devices (e.g. she
    annotates her own cycle day, he annotates the same day from his side), so
    `owner_user_id` says who wrote it. Sensitive free text — decoded locally,
    never re-exported.
    """

    note_id: str
    target_kind: str
    target_date: str
    text: str
    created_at: str | None
    updated_at: str | None
    owner_user_id: str | None
    about: str = "self"
    """"self" or "partner". A note the user's AI wrote ABOUT the partner lives in
    the USER's account (sealed to the user + this server, never to the partner),
    so the writer alone cannot say whose body it describes. Missing on every note
    written before 2026-09-23, which were all about their writer."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "note_id": self.note_id,
            "about": self.about,
            "target_kind": self.target_kind,
            # The day the note is ABOUT, as a local calendar day — the form
            # `log_note` takes. The wire value is the UTC instant of local
            # midnight, which read as the previous day east of UTC (2026-10-02).
            "target_date": _local_date_fields(self.target_date).get("local_date", self.target_date),
            **_local_date_fields(self.target_date),
            "text": self.text,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "owner_user_id": self.owner_user_id,
        }


@dataclass(frozen=True)
class StrengthRecord:
    """One strength-training session pinned to a local day, decoded from a
    metric_type="strength" blob — exercise-level detail (movement, weight,
    sets × reps) that HealthKit's workout type cannot carry. Owner's own AI
    only in v1 (no partner fan-out)."""

    entry_id: str
    date: str
    exercises: list[dict[str, Any]]
    note: str | None
    created_at: str | None
    updated_at: str | None
    owner_user_id: str | None

    @property
    def total_volume_kg(self) -> float:
        """Σ weight × reps across every set — the one number lifters compare."""
        total = 0.0
        for exercise in self.exercises:
            for one_set in exercise.get("sets", []):
                try:
                    total += float(one_set["weightKg"]) * int(one_set["reps"])
                except (KeyError, TypeError, ValueError):
                    continue
        return total

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            # The training DAY, local — the form `log_strength_entry` takes. The
            # wire value is the UTC instant of local midnight (2026-10-02).
            "date": _local_date_fields(self.date).get("local_date", self.date),
            **_local_date_fields(self.date),
            "exercises": self.exercises,
            "note": self.note,
            "total_volume_kg": round(self.total_volume_kg, 1),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "owner_user_id": self.owner_user_id,
        }


@dataclass(frozen=True)
class FoodRecord:
    """One day's food-intake log, decoded from a metric_type="food" blob.

    Structure mirrors strength on purpose (per-day entry with a list of meals,
    each meal with a list of items) so the two features share test/UI patterns.
    Nutrition estimation is intentionally NOT recorded here — the AI derives
    kcal/protein/carbs at analysis time from the item name + portion using its
    own commonsense, keeping data-entry friction minimal (the deciding factor
    for a log the owner has to feed every day). Owner's own AI only in v1
    (no partner fan-out)."""

    entry_id: str
    date: str
    meals: list[dict[str, Any]]
    note: str | None
    created_at: str | None
    updated_at: str | None
    owner_user_id: str | None

    @property
    def total_item_count(self) -> int:
        """Σ items across all meals — a lightweight "how much did I eat today"
        proxy that costs no schema. Not calories."""
        total = 0
        for meal in self.meals:
            items = meal.get("items")
            if isinstance(items, list):
                total += len(items)
        return total

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "date": self.date,
            **_local_date_fields(self.date),
            "meals": self.meals,
            "note": self.note,
            "total_item_count": self.total_item_count,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "owner_user_id": self.owner_user_id,
        }


@dataclass(frozen=True)
class SymptomSample:
    """One HealthKit symptom category sample (e.g. abdominalCramps @ moderate)."""

    symptom_type: str
    severity: str
    start_date: str
    end_date: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "symptom_type": self.symptom_type,
            "severity": self.severity,
            "start_date": self.start_date,
            "end_date": self.end_date,
        }


@dataclass(frozen=True)
class SymptomDay:
    """One day's symptom samples decoded from a metric_type="symptom" blob.

    Symptom data is sensitive: it only reaches this server when the user (or their
    partner, for partner-AI sharing) explicitly opted in on iOS. `owner_user_id`
    distinguishes whose symptoms these are — both partners can track this kind.
    """

    day_id: str
    day_start_date: str
    owner_user_id: str | None
    samples: list[SymptomSample]

    def to_dict(self) -> dict[str, Any]:
        return {
            "day_id": self.day_id,
            "day_start_date": self.day_start_date,
            **_local_date_fields(self.day_start_date),
            "owner_user_id": self.owner_user_id,
            "samples": [sample.to_dict() for sample in self.samples],
        }


@dataclass(frozen=True)
class SymptomEntry:
    """One self-reported symptom episode (GitHub #3), decoded from a
    metric_type="symptom" blob that carries `entryID`.

    Written by `log_symptom` or by the app. Unlike a HealthKit `SymptomDay`, it
    is one blob per episode, so an episode that starts at 23:30 and eases the
    next morning stays one record. `local_date` is the day the WRITER filed it
    under — carried in the payload for the reason `log_weight_entry` gives: a
    date string has no timezone for two machines to disagree about.

    `onset_at` is None when only the day is known ("昨天头疼"), which is honest
    where a fabricated midnight would not be. `deleted` marks a tombstone: the
    blob that once held the entry now holds nothing but its id.
    """

    entry_id: str
    symptom_type: str
    severity: str
    local_date: str
    onset_at: str | None
    end_at: str | None
    display_name: str | None
    body_location: str | None
    triggers: tuple[str, ...]
    note: str | None
    created_at: str | None
    updated_at: str | None
    owner_user_id: str | None
    deleted: bool = False

    def sort_key(self) -> str:
        """Business order for Invariant 38's cut: the filed day, then the onset."""

        return f"{self.local_date}|{self.onset_at or ''}"

    def to_dict(self) -> dict[str, Any]:
        duration: int | None = None
        if self.onset_at and self.end_at:
            try:
                delta = _parse_iso8601(self.end_at) - _parse_iso8601(self.onset_at)
                duration = max(int(delta.total_seconds() // 60), 0)
            except ValueError:
                duration = None
        onset_local = _local_date_fields(self.onset_at, with_time=True).get("local_time")
        end_local = _local_date_fields(self.end_at, with_time=True).get("local_time")
        return {
            "entry_id": self.entry_id,
            "symptom_type": self.symptom_type,
            "healthkit_type": self.symptom_type in HEALTHKIT_SYMPTOM_TYPES,
            "display_name": self.display_name,
            "severity": self.severity,
            "local_date": self.local_date,
            "onset_at": self.onset_at,
            "onset_local_time": onset_local,
            "end_at": self.end_at,
            "end_local_time": end_local,
            # Deliberately not "ongoing": a missing end means nobody logged one,
            # which is what was OBSERVED. Whether it is still going is a question
            # to ask the person, not a state to print.
            "end_recorded": self.end_at is not None,
            "duration_minutes": duration,
            "body_location": self.body_location,
            "triggers": list(self.triggers),
            "note": self.note,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "owner_user_id": self.owner_user_id,
        }


def _require_mapping(payload: Any, metric: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise VaultbeatCryptoError(f"{metric} payload must be a JSON object")
    return payload


def parse_water_day(payload: Any, *, owner_user_id: str | None = None) -> WaterDay:
    """Decode a decrypted water blob into a typed WaterDay (no aggregation)."""

    data = _require_mapping(payload, METRIC_WATER)
    refill_events = data.get("refillEvents")
    if not isinstance(refill_events, list):
        raise VaultbeatCryptoError("water payload is missing refillEvents list")
    container_volume = data.get("containerVolumeLiters")
    if not isinstance(container_volume, (int, float)) or isinstance(container_volume, bool):
        raise VaultbeatCryptoError("water payload is missing containerVolumeLiters")
    return WaterDay(
        day_id=str(data["dayID"]),
        day_start_date=str(data["dayStartDate"]),
        container_volume_liters=float(container_volume),
        refill_count=len(refill_events),
        owner_user_id=owner_user_id,
    )


def _optional_number(data: dict[str, Any], key: str, metric: str) -> float | None:
    """A numeric-or-null field; a present-but-non-numeric value is a contract violation."""

    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise VaultbeatCryptoError(f"{metric} payload has non-numeric {key}")
    return float(value)


def parse_body_day(payload: Any, *, owner_user_id: str | None = None) -> BodyDay:
    """Decode a decrypted body blob into a typed BodyDay (no aggregation).

    Wire contract mirrors iOS VaultbeatBodySharedCloudPayload:
    {dayID, dayStartDate, weightKg, bodyFatPercent, bmi, leanBodyMassKg} —
    weightKg required (kg); the three composition fields are nullable and absent
    on any blob written before the field existed (add-only, 2026-08-05).
    """

    data = _require_mapping(payload, METRIC_BODY)
    weight = data.get("weightKg")
    if not isinstance(weight, (int, float)) or isinstance(weight, bool):
        raise VaultbeatCryptoError("body payload is missing weightKg")
    return BodyDay(
        day_id=str(data["dayID"]),
        day_start_date=str(data["dayStartDate"]),
        weight_kg=float(weight),
        body_fat_percent=_optional_number(data, "bodyFatPercent", METRIC_BODY),
        bmi=_optional_number(data, "bmi", METRIC_BODY),
        lean_body_mass_kg=_optional_number(data, "leanBodyMassKg", METRIC_BODY),
        owner_user_id=owner_user_id,
    )


def parse_activity_day(payload: Any, *, owner_user_id: str | None = None) -> ActivityDay:
    """Decode a decrypted activity blob into a typed ActivityDay.

    Wire contract mirrors iOS VaultbeatActivitySharedCloudPayload:
    {dayID, dayStartDate, stepCount, activeEnergyKcal, exerciseMinutes, standMinutes, distanceMeters}.
    """

    data = _require_mapping(payload, METRIC_ACTIVITY)
    step_count = data.get("stepCount", 0)
    if not isinstance(step_count, (int, float)) or isinstance(step_count, bool):
        raise VaultbeatCryptoError("activity payload has non-numeric stepCount")
    active_energy = data.get("activeEnergyKcal", 0)
    if not isinstance(active_energy, (int, float)) or isinstance(active_energy, bool):
        raise VaultbeatCryptoError("activity payload has non-numeric activeEnergyKcal")
    exercise_minutes = data.get("exerciseMinutes", 0)
    stand_minutes = data.get("standMinutes", 0)
    return ActivityDay(
        day_id=str(data["dayID"]),
        day_start_date=str(data["dayStartDate"]),
        step_count=int(step_count),
        active_energy_kcal=float(active_energy),
        exercise_minutes=int(exercise_minutes),
        stand_minutes=int(stand_minutes),
        distance_meters=_optional_number(data, "distanceMeters", METRIC_ACTIVITY),
        owner_user_id=owner_user_id,
    )


def parse_resting_hr_record(payload: Any, *, owner_user_id: str | None = None) -> RestingHrRecord:
    """Decode a decrypted resting_hr blob into a typed RestingHrRecord.

    Wire contract mirrors iOS VaultbeatRestingHeartRateSharedCloudPayload:
    {dayID, dayStartDate, restingHeartRateBPM}.
    """

    data = _require_mapping(payload, METRIC_RESTING_HR)
    bpm = data.get("restingHeartRateBPM")
    if not isinstance(bpm, (int, float)) or isinstance(bpm, bool):
        raise VaultbeatCryptoError("resting_hr payload is missing restingHeartRateBPM")
    return RestingHrRecord(
        record_id=str(data["dayID"]),
        date=str(data["dayStartDate"]),
        bpm=float(bpm),
        owner_user_id=owner_user_id,
    )


def parse_workout_record(payload: Any, *, owner_user_id: str | None = None) -> WorkoutRecord:
    """Decode a decrypted workout blob into a typed WorkoutRecord.

    Wire contract mirrors iOS VaultbeatWorkoutSharedCloudPayload:
    {workoutID, activityType, startDate, endDate, durationSeconds, activeKcal, distanceMeters}.
    """

    data = _require_mapping(payload, METRIC_WORKOUT)
    duration = data.get("durationSeconds")
    if not isinstance(duration, (int, float)) or isinstance(duration, bool):
        raise VaultbeatCryptoError("workout payload is missing durationSeconds")
    return WorkoutRecord(
        workout_id=str(data["workoutID"]),
        activity_type=str(data.get("activityType", "Other")),
        start_date=str(data["startDate"]),
        end_date=str(data["endDate"]),
        duration_seconds=float(duration),
        active_kcal=_optional_number(data, "activeKcal", METRIC_WORKOUT),
        distance_meters=_optional_number(data, "distanceMeters", METRIC_WORKOUT),
        owner_user_id=owner_user_id,
    )


def parse_mindfulness_day(payload: Any, *, owner_user_id: str | None = None) -> MindfulnessDay:
    """Decode a decrypted mindfulness blob into a typed MindfulnessDay.

    Wire contract mirrors iOS VaultbeatMindfulnessSharedCloudPayload:
    {dayID, dayStartDate, sessionCount, totalMinutes}.
    """

    data = _require_mapping(payload, METRIC_MINDFULNESS)
    session_count = data.get("sessionCount", 0)
    total_minutes = data.get("totalMinutes", 0.0)
    if not isinstance(total_minutes, (int, float)) or isinstance(total_minutes, bool):
        raise VaultbeatCryptoError("mindfulness payload has non-numeric totalMinutes")
    return MindfulnessDay(
        day_id=str(data["dayID"]),
        day_start_date=str(data["dayStartDate"]),
        session_count=int(session_count),
        total_minutes=float(total_minutes),
        owner_user_id=owner_user_id,
    )


def parse_hrv_record(payload: Any, *, owner_user_id: str | None = None) -> HRVRecord:
    """Decode a decrypted hrv blob into a typed HRVRecord.

    Wire contract mirrors iOS VaultbeatHRVSharedCloudPayload:
    {dayID, dayStartDate, sdnnMilliseconds}.
    """

    data = _require_mapping(payload, METRIC_HRV)
    sdnn = data.get("sdnnMilliseconds")
    if not isinstance(sdnn, (int, float)) or isinstance(sdnn, bool):
        raise VaultbeatCryptoError("hrv payload is missing sdnnMilliseconds")
    return HRVRecord(
        record_id=str(data["dayID"]),
        date=str(data["dayStartDate"]),
        sdnn_ms=float(sdnn),
        owner_user_id=owner_user_id,
    )


def parse_hrv_hourly_record(payload: Any, *, owner_user_id: str | None = None) -> HRVHourlyBucket:
    """Decode a decrypted hrv_hourly blob into a typed HRVHourlyBucket.

    Wire contract mirrors iOS VaultbeatHRVHourlySharedCloudPayload:
    {hourID, hourStartDate, avgSdnnMilliseconds, sampleCount}.
    """

    data = _require_mapping(payload, METRIC_HRV_HOURLY)
    avg = data.get("avgSdnnMilliseconds")
    if not isinstance(avg, (int, float)) or isinstance(avg, bool):
        raise VaultbeatCryptoError("hrv_hourly payload is missing avgSdnnMilliseconds")
    count = data.get("sampleCount")
    if not isinstance(count, int) or isinstance(count, bool):
        raise VaultbeatCryptoError("hrv_hourly payload is missing sampleCount")
    return HRVHourlyBucket(
        record_id=str(data["hourID"]),
        date=str(data["hourStartDate"]),
        avg_sdnn_ms=float(avg),
        sample_count=int(count),
        owner_user_id=owner_user_id,
    )


def parse_wrist_temp_record(payload: Any, *, owner_user_id: str | None = None) -> WristTempRecord:
    """Decode a decrypted wrist_temp blob into a typed WristTempRecord.

    Wire contract mirrors iOS VaultbeatWristTemperatureSharedCloudPayload:
    {dayID, dayStartDate, temperatureDeltaCelsius}.
    """

    data = _require_mapping(payload, METRIC_WRIST_TEMP)
    delta = data.get("temperatureDeltaCelsius")
    if not isinstance(delta, (int, float)) or isinstance(delta, bool):
        raise VaultbeatCryptoError("wrist_temp payload is missing temperatureDeltaCelsius")
    return WristTempRecord(
        record_id=str(data["dayID"]),
        date=str(data["dayStartDate"]),
        temperature_delta_celsius=float(delta),
        owner_user_id=owner_user_id,
    )


def parse_basal_energy_record(payload: Any, *, owner_user_id: str | None = None) -> BasalEnergyRecord:
    """Decode a decrypted basal_energy blob into a typed BasalEnergyRecord.

    Wire contract mirrors iOS VaultbeatBasalEnergySharedCloudPayload:
    {sampleID, sampleStartDate, basalEnergyKcal}.
    """

    data = _require_mapping(payload, METRIC_BASAL_ENERGY)
    value = data.get("basalEnergyKcal")
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise VaultbeatCryptoError("basal_energy payload is missing basalEnergyKcal")
    return BasalEnergyRecord(
        record_id=str(data["sampleID"]),
        date=str(data["sampleStartDate"]),
        kcal=float(value),
        owner_user_id=owner_user_id,
    )


def parse_vo2max_record(payload: Any, *, owner_user_id: str | None = None) -> VO2MaxRecord:
    """Decode a decrypted vo2max blob into a typed VO2MaxRecord.

    Wire contract mirrors iOS VaultbeatVO2MaxSharedCloudPayload:
    {sampleID, sampleStartDate, vo2MaxMlKgMin}.
    """

    data = _require_mapping(payload, METRIC_VO2MAX)
    value = data.get("vo2MaxMlKgMin")
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise VaultbeatCryptoError("vo2max payload is missing vo2MaxMlKgMin")
    return VO2MaxRecord(
        record_id=str(data["sampleID"]),
        date=str(data["sampleStartDate"]),
        vo2_max_ml_kg_min=float(value),
        owner_user_id=owner_user_id,
    )


def parse_profile_record(payload: Any, *, owner_user_id: str | None = None) -> ProfileRecord:
    """Decode a decrypted profile blob into a typed ProfileRecord.

    Wire contract mirrors iOS VaultbeatHealthProfileSharedCloudPayload:
    {profileID, biologicalSex?, sexSource?, dateOfBirth? ("YYYY-MM-DD"), heightCm?}.
    """

    data = _require_mapping(payload, METRIC_PROFILE)
    if "profileID" not in data:
        raise VaultbeatCryptoError("profile payload is missing profileID")
    sex = data.get("biologicalSex")
    if sex not in (None, "female", "male", "other"):
        raise VaultbeatCryptoError(f"profile payload has an unknown biologicalSex: {sex!r}")
    height = data.get("heightCm")
    if height is not None and (not isinstance(height, (int, float)) or isinstance(height, bool)):
        raise VaultbeatCryptoError("profile payload heightCm is not a number")
    birth = data.get("dateOfBirth")
    return ProfileRecord(
        record_id=str(data["profileID"]),
        biological_sex=sex,
        sex_source=data.get("sexSource") if isinstance(data.get("sexSource"), str) else None,
        date_of_birth=str(birth) if birth else None,
        height_cm=float(height) if height is not None else None,
        owner_user_id=owner_user_id,
    )


def parse_menstrual_day(payload: Any, *, owner_user_id: str | None = None) -> MenstrualDay:
    """Decode a decrypted menstrual blob into a typed MenstrualDay (no prediction)."""

    data = _require_mapping(payload, METRIC_MENSTRUAL)
    raw_samples = data.get("samples")
    if not isinstance(raw_samples, list):
        raise VaultbeatCryptoError("menstrual payload is missing samples list")
    samples: list[MenstrualSample] = []
    for raw in raw_samples:
        if not isinstance(raw, dict):
            raise VaultbeatCryptoError("menstrual sample must be a JSON object")
        flow = str(raw.get("flow", "unspecified"))
        if flow not in MENSTRUAL_FLOW_VALUES:
            raise VaultbeatCryptoError(f"menstrual sample has unknown flow value: {flow}")
        samples.append(
            MenstrualSample(
                start_date=str(raw["startDate"]),
                end_date=str(raw["endDate"]),
                flow=flow,
            )
        )
    return MenstrualDay(
        day_id=str(data["dayID"]),
        day_start_date=str(data["dayStartDate"]),
        samples=samples,
        owner_user_id=owner_user_id,
    )


def parse_symptom_day(payload: Any, *, owner_user_id: str | None = None) -> SymptomDay:
    """Decode a decrypted symptom blob into a typed SymptomDay.

    Wire contract mirrors iOS VaultbeatSymptomSharedCloudPayload:
    {dayID, dayStartDate, samples: [{symptomType, severity, startDate, endDate}]}.
    An unknown severity string is a contract violation (the iOS mapper only emits
    SYMPTOM_SEVERITY_VALUES); unknown symptomType strings are accepted as-is so a
    newer app adding a type doesn't brick older decoders.
    """

    data = _require_mapping(payload, METRIC_SYMPTOM)
    raw_samples = data.get("samples")
    if not isinstance(raw_samples, list):
        raise VaultbeatCryptoError("symptom payload is missing samples list")
    samples: list[SymptomSample] = []
    for raw in raw_samples:
        if not isinstance(raw, dict):
            raise VaultbeatCryptoError("symptom sample must be a JSON object")
        severity = str(raw.get("severity", "unspecified"))
        if severity not in SYMPTOM_SEVERITY_VALUES:
            raise VaultbeatCryptoError(f"symptom sample has unknown severity value: {severity}")
        symptom_type = raw.get("symptomType")
        if not isinstance(symptom_type, str) or not symptom_type:
            raise VaultbeatCryptoError("symptom sample is missing symptomType")
        samples.append(
            SymptomSample(
                symptom_type=symptom_type,
                severity=severity,
                start_date=str(raw["startDate"]),
                end_date=str(raw["endDate"]),
            )
        )
    return SymptomDay(
        day_id=str(data["dayID"]),
        day_start_date=str(data["dayStartDate"]),
        owner_user_id=owner_user_id,
        samples=samples,
    )


def is_symptom_entry_payload(payload: Any) -> bool:
    """True for a self-reported entry, False for a HealthKit day blob.

    `entryID` is the one discriminator (see SYMPTOM_ENTRY_SEVERITIES' header).
    """

    return isinstance(payload, dict) and "entryID" in payload


def _optional_text(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def parse_symptom_entry(payload: Any, *, owner_user_id: str | None = None) -> SymptomEntry:
    """Decode a self-reported symptom blob into a typed SymptomEntry.

    Wire contract (camelCase, shared with iOS `VaultbeatSymptomEntryPayload`):
    {entryID, symptomType, severity, localDate, onsetAt?, endAt?, displayName?,
    bodyLocation?, triggers?, note?, createdAt?, updatedAt?}; a tombstone is
    {entryID, deleted: true, updatedAt}. Optional fields tolerate absence so an
    older or newer writer still decodes; `severity` is held to the four values a
    writer may emit, like `parse_symptom_day` holds the HealthKit enum.
    """

    data = _require_mapping(payload, METRIC_SYMPTOM)
    entry_id = data.get("entryID")
    if not isinstance(entry_id, str) or not entry_id:
        raise VaultbeatCryptoError("symptom entry is missing entryID")
    updated_at = _optional_text(data.get("updatedAt"))
    created_at = _optional_text(data.get("createdAt"))
    if data.get("deleted") is True:
        return SymptomEntry(
            entry_id=entry_id,
            symptom_type="",
            severity="unspecified",
            local_date="",
            onset_at=None,
            end_at=None,
            display_name=None,
            body_location=None,
            triggers=(),
            note=None,
            created_at=created_at,
            updated_at=updated_at,
            owner_user_id=owner_user_id,
            deleted=True,
        )
    symptom_type = data.get("symptomType")
    if not isinstance(symptom_type, str) or not symptom_type:
        raise VaultbeatCryptoError("symptom entry is missing symptomType")
    severity = str(data.get("severity", "unspecified"))
    if severity not in SYMPTOM_ENTRY_SEVERITIES:
        raise VaultbeatCryptoError(f"symptom entry has unknown severity value: {severity}")
    onset_at = _optional_text(data.get("onsetAt"))
    local_date = _optional_text(data.get("localDate"))
    if local_date is None:
        # A writer that left the day out still said when it began; file it under
        # that instant's local day rather than dropping the record.
        local_date = _local_date_fields(onset_at).get("local_date")
    if not local_date:
        raise VaultbeatCryptoError("symptom entry has neither localDate nor onsetAt")
    raw_triggers = data.get("triggers")
    triggers = tuple(
        t.strip() for t in (raw_triggers if isinstance(raw_triggers, list) else [])
        if isinstance(t, str) and t.strip()
    )
    return SymptomEntry(
        entry_id=entry_id,
        symptom_type=symptom_type,
        severity=severity,
        local_date=local_date,
        onset_at=onset_at,
        end_at=_optional_text(data.get("endAt")),
        display_name=_optional_text(data.get("displayName")),
        body_location=_optional_text(data.get("bodyLocation")),
        triggers=triggers,
        note=_optional_text(data.get("note")),
        created_at=created_at,
        updated_at=updated_at,
        owner_user_id=owner_user_id,
    )


def normalize_symptom_type(raw: Any) -> str:
    """One spelling per symptom, so reported entries and HealthKit samples join.

    A HealthKit type is matched ignoring case and separators ("abdominal_cramps",
    "Abdominal Cramps", "abdominalcramps" → "abdominalCramps"). Anything else is
    folded into the same camelCase HealthKit uses ("rectal_bleeding" →
    "rectalBleeding"). Non-ASCII is refused rather than stored: the type is the
    join key across months of records, and "便血" today, "便中带血" next week
    would be two symptoms. The person's own words go in `display_name`.
    """

    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("symptom_type must be a non-empty string")
    text = raw.strip()
    if not text.isascii():
        raise ValueError(
            f"symptom_type must be an English token such as 'rectal_bleeding' or "
            f"'headache', not {text!r}; put the person's own words in display_name"
        )
    key = re.sub(r"[^a-z0-9]", "", text.lower())
    if not key:
        raise ValueError(f"symptom_type {text!r} has no letters or digits")
    if key in _HEALTHKIT_SYMPTOM_BY_KEY:
        return _HEALTHKIT_SYMPTOM_BY_KEY[key]
    # Split on separators AND on existing camelCase humps, then re-join as camelCase.
    words: list[str] = [
        w.lower()
        for chunk in re.split(r"[^A-Za-z0-9]+", text)
        for w in re.findall(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|[0-9]+", chunk)
    ]
    if not words:
        raise ValueError(f"symptom_type {text!r} has no letters or digits")
    if len(key) > 64:
        raise ValueError("symptom_type is too long for a type token (max 64 letters)")
    return words[0] + "".join(w.capitalize() for w in words[1:])


def _symptom_wire_instant(value: datetime) -> str:
    """UTC, whole seconds, trailing Z — the form the app's own encoder writes, so
    an entry the app re-saves reads back byte-for-byte in the same shape."""

    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_symptom_time(value: str, field: str) -> tuple[datetime | None, date | None]:
    """`(instant, None)` for a date-time, `(None, day)` for a bare 'YYYY-MM-DD'.

    A date-time without an offset is read in this machine's timezone — the
    module-wide assumption `_local_calendar_day` states.
    """

    text = value.strip()
    if len(text) == 10 and text[4] == "-" and text[7] == "-":
        try:
            return None, date.fromisoformat(text)
        except ValueError as error:
            raise ValueError(f"{field} {value!r} is not a real calendar day") from error
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00").replace("z", "+00:00"))
    except ValueError as error:
        raise ValueError(
            f"{field} must be ISO 8601 such as '2026-09-30T10:49+08:00' (or a bare "
            f"'2026-09-30' when only the day is known), got {value!r}"
        ) from error
    if parsed.tzinfo is None:
        # A naive time is this machine's wall clock ON THAT DATE (see `_local_midnight`).
        parsed = parsed.astimezone()
    return parsed, None


def _bounded_text(value: Any, field: str, limit: int) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    text = value.strip()
    if len(text) > limit:
        raise ValueError(f"{field} is longer than {limit} characters")
    return text or None


def _validate_symptom_fields(
    *,
    symptom_type: Any,
    severity: Any,
    onset_at: Any,
    end_at: Any,
    day: Any,
    display_name: Any,
    body_location: Any,
    triggers: Any,
    note: Any,
    default_onset: datetime | None,
) -> dict[str, Any]:
    """Normalise one entry's fields into the camelCase wire shape, or raise.

    `default_onset` fills a missing onset (a fresh log means "now"); an update
    passes None because the existing onset is already in `onset_at`.
    """

    kind = normalize_symptom_type(symptom_type)
    level = severity.strip().lower() if isinstance(severity, str) else ""
    if level not in SYMPTOM_ENTRY_SEVERITIES:
        raise ValueError(
            f"severity must be one of {sorted(SYMPTOM_ENTRY_SEVERITIES)}, got {severity!r} "
            "(use 'unspecified' when the person did not say how bad it is)"
        )

    onset_instant: datetime | None = None
    onset_day: date | None = None
    if isinstance(onset_at, str) and onset_at.strip():
        onset_instant, onset_day = _parse_symptom_time(onset_at, "onset_at")
    elif onset_at is not None and not isinstance(onset_at, str):
        raise ValueError("onset_at must be a string")
    elif default_onset is not None:
        onset_instant = default_onset

    end_instant: datetime | None = None
    if isinstance(end_at, str) and end_at.strip():
        end_instant, _end_day = _parse_symptom_time(end_at, "end_at")
        if end_instant is None:
            raise ValueError("end_at needs a time of day, not just a date")
    elif end_at is not None and not isinstance(end_at, str):
        raise ValueError("end_at must be a string")

    if isinstance(day, str) and day.strip():
        try:
            local_day = date.fromisoformat(day.strip())
        except ValueError as error:
            raise ValueError(f"date must be 'YYYY-MM-DD', got {day!r}") from error
    elif onset_instant is not None:
        # The wall-clock day in the offset the caller wrote (or this machine's,
        # for a naive time) — the person's day, not UTC's.
        local_day = onset_instant.date()
    elif onset_day is not None:
        local_day = onset_day
    else:
        raise ValueError("give onset_at (when it began) or date (the day it happened)")

    now = datetime.now(timezone.utc)
    if onset_instant is not None and onset_instant > now + timedelta(minutes=5):
        raise ValueError("onset_at is in the future")
    if end_instant is not None and end_instant > now + timedelta(minutes=5):
        raise ValueError("end_at is in the future")
    if onset_instant is not None and end_instant is not None and end_instant < onset_instant:
        raise ValueError("end_at is before onset_at")

    if triggers is None:
        raw_triggers: list[Any] = []
    elif isinstance(triggers, str):
        raw_triggers = [triggers]
    elif isinstance(triggers, (list, tuple)):
        raw_triggers = list(triggers)
    else:
        raise ValueError("triggers must be a list of short strings")
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in raw_triggers:
        text = _bounded_text(item, "each trigger", _SYMPTOM_LABEL_MAX)
        if text and text.casefold() not in seen:
            seen.add(text.casefold())
            cleaned.append(text)
    if len(cleaned) > _SYMPTOM_TRIGGERS_MAX:
        raise ValueError(f"at most {_SYMPTOM_TRIGGERS_MAX} triggers per entry")

    return {
        "symptomType": kind,
        "severity": level,
        "localDate": local_day.isoformat(),
        "onsetAt": _symptom_wire_instant(onset_instant) if onset_instant else None,
        "endAt": _symptom_wire_instant(end_instant) if end_instant else None,
        "displayName": _bounded_text(display_name, "display_name", _SYMPTOM_LABEL_MAX),
        "bodyLocation": _bounded_text(body_location, "body_location", _SYMPTOM_LABEL_MAX),
        "triggers": cleaned,
        "note": _bounded_text(note, "note", _SYMPTOM_TEXT_MAX),
    }


def _symptom_entry_payload(
    entry_id: str, fields: dict[str, Any], *, created_at: str, updated_at: str
) -> dict[str, Any]:
    """The plaintext of one reported entry. Empty optionals are left out so the
    wire carries only what someone actually said."""

    return {
        "entryID": entry_id,
        **{key: value for key, value in fields.items() if value not in (None, [])},
        "createdAt": created_at,
        "updatedAt": updated_at,
    }


def parse_note(payload: Any, *, owner_user_id: str | None = None) -> NoteRecord:
    """Decode a decrypted note blob into a typed NoteRecord.

    Wire contract mirrors iOS VaultbeatNoteCloudPayload:
    {noteID, targetKind, targetDate, text, createdAt, updatedAt}. text and a
    non-empty targetKind are required; timestamps are tolerated missing so a
    payload written by a newer/older writer still decodes.
    """

    data = _require_mapping(payload, METRIC_NOTE)
    text = data.get("text")
    if not isinstance(text, str) or not text.strip():
        raise VaultbeatCryptoError("note payload is missing text")
    target_kind = data.get("targetKind")
    if not isinstance(target_kind, str) or not target_kind:
        raise VaultbeatCryptoError("note payload is missing targetKind")
    return NoteRecord(
        note_id=str(data["noteID"]),
        target_kind=target_kind,
        target_date=str(data["targetDate"]),
        text=text,
        created_at=(str(data["createdAt"]) if data.get("createdAt") is not None else None),
        updated_at=(str(data["updatedAt"]) if data.get("updatedAt") is not None else None),
        owner_user_id=owner_user_id,
        about="partner" if data.get("about") == "partner" else "self",
    )


def parse_strength(payload: Any, *, owner_user_id: str | None = None) -> StrengthRecord:
    """Decode a decrypted strength blob into a typed StrengthRecord.

    Wire contract mirrors iOS VaultbeatStrengthCloudPayload:
    {entryID, date, exercises: [{name, sets: [{weightKg, reps}]}], note?,
    createdAt, updatedAt}. A non-empty exercises list is required; timestamps
    and note are tolerated missing.
    """

    data = _require_mapping(payload, METRIC_STRENGTH)
    exercises = data.get("exercises")
    if not isinstance(exercises, list) or not exercises:
        raise VaultbeatCryptoError("strength payload is missing exercises")
    cleaned: list[dict[str, Any]] = []
    for exercise in exercises:
        if not isinstance(exercise, dict):
            raise VaultbeatCryptoError("strength exercise is not a mapping")
        name = exercise.get("name")
        if not isinstance(name, str) or not name.strip():
            raise VaultbeatCryptoError("strength exercise is missing name")
        sets = exercise.get("sets")
        if not isinstance(sets, list):
            raise VaultbeatCryptoError("strength exercise is missing sets")
        cleaned.append({"name": name, "sets": sets})
    note = data.get("note")
    return StrengthRecord(
        entry_id=str(data["entryID"]),
        date=str(data["date"]),
        exercises=cleaned,
        note=(str(note) if isinstance(note, str) and note.strip() else None),
        created_at=(str(data["createdAt"]) if data.get("createdAt") is not None else None),
        updated_at=(str(data["updatedAt"]) if data.get("updatedAt") is not None else None),
        owner_user_id=owner_user_id,
    )


def _strength_number_key(value: Any) -> str:
    """A set's number as the same text whichever writer wrote it.

    🔴 The iOS encoder writes a whole Double without its fraction (`39`) and the
    agent path writes a float (`39.0`), so `str()` of the two never matched and
    the copy this exists to collapse was counted twice for every whole weight
    (review V3, 2026-10-03 — the tests wrote `39.0` on both sides).
    """
    if isinstance(value, bool):
        return repr(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return repr(value)
    return repr(number) if math.isfinite(number) else repr(value)


def _strength_exercise_key(exercise: dict[str, Any]) -> tuple[str, tuple[tuple[str, str], ...]]:
    """An exercise as a comparable value: its name (case-folded, as
    `_merge_strength_exercises` compares it) and every set, in order."""

    sets = tuple(
        (_strength_number_key(s.get("weightKg")), _strength_number_key(s.get("reps")))
        if isinstance(s, dict) else (repr(s), "")
        for s in exercise.get("sets") or []
    )
    return str(exercise.get("name") or "").strip().casefold(), sets


def _split_same_day_copies(
    sessions: list[StrengthRecord],
) -> tuple[list[StrengthRecord], list[tuple[StrengthRecord, str]]]:
    """Separate sessions that are a COPY of another one logged the same day.

    Both writers keep one session per local day — the app's store is keyed by
    day and `log_strength_entry` reuses the day's entry id — so two entry ids
    on one day only arise when a writer mints an id for a day whose other
    entry it has not seen yet. Measured 2026-10-03 (release gate G3): an
    agent's 02:37 write held one exercise; the phone, not having pulled it,
    started its own entry for that day at 02:57 holding the same exercise and
    more. Both blobs are real rows, so the read returned both: the day counted
    as two sessions and its 1872 kg lat pulldown was summed twice.

    A session is dropped from the totals only when EVERY exercise in it —
    name and every set — already appears in a session of the same person and
    day that was touched more recently. A session holding anything of its own
    is kept: a genuine second workout must never be swallowed to make a count
    look tidy. Dropped sessions are returned, never erased (Invariant 41).
    """

    groups: dict[tuple[str, str], list[StrengthRecord]] = {}
    for entry in sessions:
        day = _local_date_fields(entry.date).get("local_date", entry.date[:10])
        groups.setdefault((entry.owner_user_id or "", str(day)), []).append(entry)

    kept: list[StrengthRecord] = []
    copies: list[tuple[StrengthRecord, str]] = []
    for group in groups.values():
        if len(group) == 1:
            kept.extend(group)
            continue
        group.sort(key=lambda e: e.updated_at or e.created_at or "", reverse=True)
        day_kept: list[StrengthRecord] = []
        for entry in group:
            keys = Counter(_strength_exercise_key(x) for x in entry.exercises)
            # A multiset, not a set: [E, E] is two exercises' worth of work and
            # is not a copy of [E] (review V3).
            holder = next(
                (k for k in day_kept
                 if keys <= Counter(_strength_exercise_key(x) for x in k.exercises)),
                None,
            )
            if holder is not None and keys:
                copies.append((entry, holder.entry_id))
            else:
                day_kept.append(entry)
        kept.extend(day_kept)
    return kept, copies


def summarize_strength(entries: list[StrengthRecord], *, limit_days: int | None = None) -> dict[str, Any]:
    """Recent strength sessions, newest day first, with per-session volume.

    Dedup by entry_id (newest updated_at wins — edits upsert the same blob id),
    then set aside any session that only repeats another one logged the same
    day (`_split_same_day_copies`): it is listed under `duplicate_sessions` and
    kept out of `sessions` and `session_count`.
    Pass `limit_days` to keep only the most recent N sessions after dedup.
    """

    by_id: dict[str, StrengthRecord] = {}
    for entry in entries:
        existing = by_id.get(entry.entry_id)
        new_key = entry.updated_at or entry.created_at or ""
        old_key = existing.updated_at or existing.created_at or "" if existing else ""
        if existing is None or new_key >= old_key:
            by_id[entry.entry_id] = entry

    kept, copies = _split_same_day_copies(list(by_id.values()))
    ordered = sorted(kept, key=lambda e: e.date, reverse=True)
    if limit_days is not None:
        ordered = ordered[:limit_days]
    shown_days = {_local_date_fields(e.date).get("local_date") for e in ordered}

    summary: dict[str, Any] = {
        "session_count": len(ordered),
        "sessions": [entry.to_dict() for entry in ordered],
    }
    listed = [
        {
            "entry_id": copy.entry_id,
            "local_date": _local_date_fields(copy.date).get("local_date"),
            "duplicate_of": holder,
            "total_volume_kg": round(copy.total_volume_kg, 1),
        }
        for copy, holder in sorted(copies, key=lambda c: c[0].date, reverse=True)
        if _local_date_fields(copy.date).get("local_date") in shown_days
    ]
    if listed:
        summary["duplicate_sessions"] = listed
        summary["duplicate_note"] = (
            "These entries only repeat exercises (same name, same sets) already in "
            "another session of the same day, so they are left out of `sessions`, "
            "`session_count` and any volume you add up. Each is still stored, so "
            "nothing was lost; it is a second copy, not a second workout."
        )
    return summary


def parse_food(payload: Any, *, owner_user_id: str | None = None) -> FoodRecord:
    """Decode a decrypted food blob into a typed FoodRecord.

    Wire contract mirrors iOS VaultbeatFoodCloudPayload:
    {entryID, date, meals: [{name?, timeOfDay?, items: [{food, portion?, note?}]}], note?,
    createdAt, updatedAt}. A non-empty meals list is required; per-item
    portion/note and per-meal name/timeOfDay are all optional (the recording
    friction we're minimizing is real — you can just write down "香蕉" and
    have the AI figure out kcal later).
    """

    data = _require_mapping(payload, METRIC_FOOD)
    meals = data.get("meals")
    if not isinstance(meals, list) or not meals:
        raise VaultbeatCryptoError("food payload is missing meals")
    cleaned: list[dict[str, Any]] = []
    for meal in meals:
        if not isinstance(meal, dict):
            raise VaultbeatCryptoError("food meal is not a mapping")
        items = meal.get("items")
        if not isinstance(items, list):
            raise VaultbeatCryptoError("food meal is missing items")
        cleaned_meal: dict[str, Any] = {"items": items}
        for optional_key in ("name", "timeOfDay", "note"):
            if optional_key in meal:
                cleaned_meal[optional_key] = meal[optional_key]
        cleaned.append(cleaned_meal)
    note = data.get("note")
    return FoodRecord(
        entry_id=str(data["entryID"]),
        date=str(data["date"]),
        meals=cleaned,
        note=(str(note) if isinstance(note, str) and note.strip() else None),
        created_at=(str(data["createdAt"]) if data.get("createdAt") is not None else None),
        updated_at=(str(data["updatedAt"]) if data.get("updatedAt") is not None else None),
        owner_user_id=owner_user_id,
    )


def summarize_food(entries: list[FoodRecord], *, limit_days: int | None = None) -> dict[str, Any]:
    """Recent food-intake logs, newest day first.

    Dedup by entry_id (newest updated_at wins — edits upsert the same blob id).
    Pass `limit_days` to keep only the most recent N days after dedup.
    """

    by_id: dict[str, FoodRecord] = {}
    for entry in entries:
        existing = by_id.get(entry.entry_id)
        new_key = entry.updated_at or entry.created_at or ""
        old_key = existing.updated_at or existing.created_at or "" if existing else ""
        if existing is None or new_key >= old_key:
            by_id[entry.entry_id] = entry

    ordered = sorted(by_id.values(), key=lambda e: e.date, reverse=True)
    if limit_days is not None:
        ordered = ordered[:limit_days]

    return {
        "day_count": len(ordered),
        "days": [entry.to_dict() for entry in ordered],
    }


def summarize_notes(notes: list[NoteRecord], *, target_kind: str | None = None) -> dict[str, Any]:
    """Recent notes grouped by target kind, each carrying its writer.

    Dedup by note_id (newest updated_at wins — edits upsert the same blob id)
    and sort newest target day first. Pass `target_kind` to keep only one kind
    (e.g. just cycle notes when analysing a period).
    """

    by_id: dict[str, NoteRecord] = {}
    for note in notes:
        if target_kind is not None and note.target_kind != target_kind:
            continue
        existing = by_id.get(note.note_id)
        # Missing updatedAt falls back to createdAt so a timestampless edit from
        # a newer/older writer still competes on SOME recency signal instead of
        # always losing to any timestamped copy.
        new_key = note.updated_at or note.created_at or ""
        old_key = existing.updated_at or existing.created_at or "" if existing else ""
        if existing is None or new_key >= old_key:
            by_id[note.note_id] = note

    kinds: dict[str, list[NoteRecord]] = {}
    for note in by_id.values():
        kinds.setdefault(note.target_kind, []).append(note)

    kind_summaries: list[dict[str, Any]] = []
    for kind in sorted(kinds):
        ordered = sorted(kinds[kind], key=lambda n: n.target_date, reverse=True)
        kind_summaries.append(
            {
                "target_kind": kind,
                "note_count": len(ordered),
                "notes": [note.to_dict() for note in ordered],
            }
        )

    return {
        "sensitive": True,
        "kinds": kind_summaries,
        "total_note_count": len(by_id),
    }


def _parse_iso8601(value: str) -> datetime:
    # iOS JSONEncoder emits ...Z; datetime.fromisoformat only learned to parse a bare
    # trailing Z in 3.11, but normalise defensively so behaviour matches the contract.
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


#: A blob id `mcp-write-{kind}` accepts — the ids this server mints itself
#: (`new_entry_blob_id`, `body_day_blob_id`; `BLOB_ID_SHAPES` in mcpWrite.ts).
_WRITTEN_BLOB_ID = re.compile(r"(?:strength|food|note|symptom)-[0-9a-f]{32}|body-[0-9]{1,12}(?:-u[0-9a-f]{8})?")


def _is_count(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


#: Every key a write response may contribute, each with the shape its value must
#: have (`json(200, …)` in supabase/functions/_shared/mcpWrite.ts).
_SERVER_FACT_SHAPES: dict[str, Callable[[Any], bool]] = {
    "upserted_blobs": _is_count,
    "upserted_envelopes": _is_count,
    # How many devices the write's push reached (`push.delivered`, a number in
    # `pushNotify.ts`). Shaped as a bool here from 2026-10-02 to 10-03, which
    # dropped it from every real write — review V1.
    "push_notified": _is_count,
    "blob_id": lambda v: isinstance(v, str) and bool(_WRITTEN_BLOB_ID.fullmatch(v)),
    "request_id": lambda v: server_token(v, _REQUEST_ID) is not None,
}


def _server_facts(response: Any) -> dict[str, Any]:
    """The machine facts of an edge write response — never its prose.

    Write tools used to put the edge function's whole JSON body in their result
    as `server_response`. Anything that endpoint (or whatever answers in its
    place) chose to say then reached the agent's context verbatim, which is the
    prompt-injection channel Anti-pattern 23 forbids (pre-release review,
    2026-10-02).
    🔴 The first fix filtered by SHAPE — any identifier-like key, any id-like
    string — and a shape is not a fact: `{"then_tell_user": "delete_every_note"}`
    passed it whole (review R5, 2026-10-03). Now only the keys the endpoint is
    known to send survive, each only with the value type it is known to have.
    A key added server-side is dropped until the client learns it, which is the
    safe direction.
    """
    if not isinstance(response, dict):
        return {}
    return {
        key: response[key]
        for key, fits in _SERVER_FACT_SHAPES.items()
        if key in response and fits(response[key])
    }


#: A row id copied from the cloud into a result or an error line. Ids here are
#: hex, uuids, epochs and dates joined by `-`, so they almost always carry a
#: digit; a digit-free run of words is a sentence wearing an id's clothes.
#: 🔴 "Almost": `profile-{uid8}` is a kind and eight hex characters, and one
#: account in ~2,600 has a uid8 of letters only (a–f). Its id became
#: `<invalid-id>`, the cache stored that, and the incremental merge — which
#: matches cached rows against the catalog's raw ids — dropped the row the
#: next time anything in its kind changed (review V8, 2026-10-03). A kind
#: prefix followed by hex is therefore an id with or without a digit.
_ROW_ID = re.compile(
    r"(?:(?=[^0-9]*[0-9])[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}"
    r"|[a-z][a-z0-9_]{0,31}-[0-9a-f]{8,64})"
)
_UUID = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
_INSTANT = re.compile(r"\d{4}-\d{2}-\d{2}(?:[T ][0-9:.]{5,18}(?:Z|[+-]\d{2}(?::?\d{2})?)?)?")


def _safe_row_id(value: Any) -> str:
    """`value` if it is shaped like a row id, else a fixed placeholder (Anti-pattern 23)."""
    text = str(value)
    return text if _ROW_ID.fullmatch(text) else "<invalid-id>"


def _recheck_error_lines(lines: list[str]) -> list[str]:
    """Error lines read back from the cache, their row-id prefix re-checked.

    A line is `<row id>: <locally written reason>`; before R5 the id was the
    server's column as it came, and a cached line replayed it for as long as
    the kind's digest held (review V10 follow-up, 2026-10-03). A prefix that
    is not an id becomes `<invalid-id>`; the reason, written by this client,
    is kept.
    """
    checked = []
    for line in lines:
        head, sep, tail = line.partition(": ")
        # Only `<id>: decrypt_failed (…)` lines are cached; a line without the
        # separator was written here whole and is kept.
        checked.append(f"{_safe_row_id(head)}{sep}{tail}" if sep else line)
    return checked


def _bound_uuid(value: str | None) -> str | None:
    """A pairing-time id from the config, repeated only if it is a UUID. It
    came from the server, and a config written before 2026-10-03 may hold any
    string (review V2)."""
    return server_token(value, _UUID)


def _safe_shaped(value: Any, shape: re.Pattern[str]) -> str | None:
    return str(value) if value is not None and shape.fullmatch(str(value)) else None


#: How many kinds the catalog and `get_metric` read from the cloud at once.
#: Enough to overlap the waits, few enough to stay polite to the edge function.
_CONCURRENT_KIND_READS = 4
#: How many per-kind digests the doctor asks for at once, after one probe kind
#: has proved the token (`_kind_counts`).
_CONCURRENT_DIGESTS = 9


def _local_date_fields(iso: str | None, *, with_time: bool = False) -> dict[str, Any]:
    """Human-readable local-calendar fields for a wire timestamp.

    Wire dates are UTC instants ("2026-07-21T16:00:00Z" is local 2026-07-22
    midnight for a UTC+8 user); every consumer was doing the +8h conversion by
    hand and occasionally off-by-one-day'ing it. Emit the server-local calendar
    day — and, for intra-day kinds, the local clock time — alongside the raw
    value. Same "server runs in the phone's timezone" assumption as
    ``_local_calendar_day``. Unparseable/missing input yields {} so a record
    with a malformed date degrades to the raw fields instead of raising.
    """

    if not isinstance(iso, str) or not iso:
        return {}
    try:
        parsed = _parse_iso8601(iso)
    except ValueError:
        return {}
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    local = parsed.astimezone()  # the offset in force AT that instant — see `_local_midnight`
    fields: dict[str, Any] = {"local_date": local.date().isoformat()}
    if with_time:
        fields["local_time"] = local.strftime("%Y-%m-%dT%H:%M")
    return fields


# Shown instead of "0h00m" on in-bed-only nights so a reader (an AI, usually)
# can tell "sleep was never measured" from "measured, and it was zero"
# (2026-07-27 — 60 owner-scoped nights were reporting a literal 0h00m while the
# iOS app showed 5-12h of in-bed time for the same dates).
_NO_SLEEP_DATA_LABEL = "no sleep data"

_ERRORS_NOTE = (
    "Each entry in `errors` is ONE blob that failed to decrypt (`decrypt_failed`: "
    "usually a historical-backfill blob sealed with a stale envelope key — cosmetic) "
    "or to parse (`parse_failed`: payload written by an older/newer schema). "
    "A failed blob is skipped; every other record in this result is complete, so a "
    "handful of errors among hundreds of records is NOT a data-integrity problem."
)


def _cut_newest_by(
    items: list[_T], limit: int | None, *, key: Callable[[_T], str | None]
) -> list[_T]:
    """Keep the newest `limit` items by their OWN business date (Invariant 38).

    Exists so the rule has ONE implementation instead of four copies. Invariant 38
    has now regressed three times — 2026-07-24 (8 readers), 2026-07-27 (4 more that
    the previous fix's own "unaffected" list wrongly cleared), and 2026-07-29
    (symptom / note / strength / food, the four kinds the owner hand-logs daily).
    Every time, the fix was recorded as "I fixed these N tools" rather than as a
    reusable rule, so the next reader written or reviewed reintroduced it.

    Why cutting earlier is wrong: `created_at` is an upload-batch stamp shared by
    up to 50 blobs, and a history backfill uploads years of data in minutes, so it
    carries no information about when anything actually happened. Slicing on it
    returns "N arbitrary rows from whichever batch landed last" — and can drop the
    newest record entirely, which is the failure the owner would actually hit:
    log today's session, then ask what you trained recently, and not see it.

    A None/empty key sorts last rather than raising — a malformed date should cost
    that one record its position, not fail the whole read.
    """

    if limit is None:
        return items
    return sorted(items, key=lambda i: key(i) or "", reverse=True)[:limit]


def _cut_newest_by_with_span(
    items: list[_T], limit: int | None, *, key: Callable[[_T], str | None]
) -> tuple[list[_T], int, str | None]:
    """`_cut_newest_by`, plus the two facts the cut destroys.

    Returns `(kept, total_before_cut, oldest_date_before_cut)` — exactly the pair
    `_attach_coverage` needs for `more_available`, taken at the only moment they
    are still knowable. Feeding them from the caller instead would mean sorting
    the list a second time, and a second sort is a second chance for the two
    orderings to disagree about which row is oldest.

    The `limit is None` branch returns `items` UNSORTED, matching
    `_cut_newest_by` exactly: callers pass the result straight to a
    `summarize_*` that does its own grouping, and quietly reordering it here
    would change output that has nothing to do with this feature. The span facts
    are still computed from the sorted view, so they are correct either way.
    """

    ordered = sorted(items, key=lambda i: key(i) or "", reverse=True)
    oldest = key(ordered[-1]) if ordered else None
    if limit is None:
        return items, len(items), oldest
    return ordered[:limit], len(items), oldest


def _attach_errors(summary: dict[str, Any], errors: list[str]) -> dict[str, Any]:
    """Standard error reporting: the raw list plus, when non-empty, a note that
    explains what an error means so callers stop treating two stale-key blobs
    as a broken dataset (2026-07-23 client feedback)."""

    summary["errors"] = errors
    if errors:
        summary["errors_note"] = _ERRORS_NOTE
    return summary


#: An `owner` value starting with this selects everyone EXCEPT the id after it.
#: That is how "my partner" is expressed without this server ever storing, or an
#: agent ever having to look up, the partner's id: the account that paired this
#: machine is known from the binding, and on a paired account every other owner
#: in the data IS the partner (a pairing is exactly two people — enforced by the
#: DB trigger that caps a user at one relationship).
NOT_OWNER_MARK = "!"


def _select_owner(records: list[Any], owner: str | None) -> list[Any]:
    """The one owner filter every reader goes through."""

    if not owner:
        return records
    # Case-folded on both sides. Both ids come from Postgres `uuid` columns today
    # and are lowercase, but this repo has already shipped one bug from the same
    # uuid written in two cases (Invariant 32), and here a mismatch would not
    # error: "me" would read as empty and `partner=true` would hand back the
    # user's own rows labelled as the partner's.
    if owner.startswith(NOT_OWNER_MARK):
        me = owner[len(NOT_OWNER_MARK):].lower()
        return [r for r in records if r.owner_user_id and not r.owner_user_id.lower().startswith(me)]
    owner = owner.lower()
    return [r for r in records if r.owner_user_id and r.owner_user_id.lower().startswith(owner)]


def _attach_owner_guard(
    summary: dict[str, Any], records: list["DecryptedRecord"], owner: str | None
) -> dict[str, Any]:
    """Flag cross-user mixing when the caller did not filter by owner.

    This server holds BOTH partners' envelopes, so an unfiltered query blends
    two people's records and every aggregate (average weight, per-day sleep
    selection, weekly rate…) becomes a meaningless blend — e.g. `average_kg`
    once came out 53.79 from an ~82 kg owner and a ~40 kg partner
    (2026-07-23). Kept additive (a warning, not a hard error) for
    backward compatibility with deliberate both-people queries.
    """

    if owner:
        return summary
    owners = sorted({r.owner_user_id[:8] for r in records if r.owner_user_id})
    if len(owners) <= 1:
        return summary
    summary["mixed_owners"] = True
    summary["owner_user_id_prefixes"] = owners
    summary["warning"] = (
        "Records from MULTIPLE people are mixed in this result (owner prefixes: "
        + ", ".join(owners)
        + "). Aggregate numbers blend both people and are meaningless. This "
        "happens only when the server cannot tell whose account paired it (an "
        "old binding); re-pair from the Vaultbeat app so reads default to you "
        "and `partner=true` selects your partner."
    )
    return summary


# ---------------------------------------------------------------------------
# Coverage (Invariant 62 (coverage-before-average), generalised from hourly
# buckets to calendar days across every read tool).
# ---------------------------------------------------------------------------

_COVERAGE_DATE_KEYS = ("local_date", "day", "date")

# A bare local calendar day, already in the caller's timezone. It must NOT be
# round-tripped through `_local_date_fields`: that function reads its input as a
# UTC instant, so "2026-08-27" becomes 2026-08-27T00:00Z and lands on the 26th
# for every negative-offset timezone.
#
# ⚠️ This branch applies to `local_date` FIRST, not only to the hand-built rows.
# `_local_date_fields` emits `local_date` as a bare day, so every row that came
# from a `to_dict()` hits this line too — skipping the reparse is what keeps
# coverage agreeing with the day printed in the row beside it, rather than one
# day earlier. (`basal_energy_records.daily` / `total_energy_burned.days` under
# "day" and sleep's `daily_summary` under "date" have no `local_date` at all, so
# for those it is the only handling there is.)
#
# ⚠️ Removing it is INVISIBLE from Shanghai and from Lisbon — verified by
# deleting it and running the suite under three zones: Asia/Shanghai and
# Europe/Lisbon stay green, America/New_York fails three tests. Both zones this
# household will ever develop in are non-negative, so the guard test below pins
# a negative one explicitly instead of trusting the machine it runs on.
_BARE_DAY_LENGTH = 10

#: Rides on EVERY read result, so its length is paid on every call — it was
#: 2.9k characters until 2026-09-24, more than the data on a one-week read.
#: Condensed rule for rule: each sentence below is one thing an agent has
#: actually got wrong. Adding to it costs every tool; say it once, here.
_COVERAGE_NOTE = (
    "How much data this answer rests on. Quote `days_covered` (DISTINCT local days "
    "the numbers are computed over — not the length of any list) beside any average, "
    "trend or comparison. `days_in_payload` below `days_covered` means a display cap "
    "hid rows that are still inside the numbers: not gaps, do not re-read for them. "
    "`rows_counted` above `days_covered` means intra-day samples. `span_days` far "
    "above `days_covered` is a sparse history, not a dense one. A day missing inside "
    "the span means nothing was recorded or the limit cut it — never that nothing "
    "happened. 🔴 Read `more_available` before saying how much history exists: true "
    "means older days are decryptable right now and your `limit` left them behind — "
    "ask for more, or quote `oldest_available` as the real start; never report a "
    "limit-shaped window as all their data (`total_available` is the pre-cut count). "
    "`window_satisfied` is true only when you got everything you asked for AND "
    "nothing older is withheld. `more_available: false` with a short history really "
    "is all this server holds — a freshly paired server may still be sealing its "
    "copy, and an account with no membership or live trial uploads nothing new "
    "from app 1.2.9 (its last 7 days on app 1.2.8 and earlier). None of this is a "
    "quality judgement."
)


#: Rides on every analysis result. It says the two things an agent gets wrong
#: about these tools specifically, which is why it is not just `_COVERAGE_NOTE`:
#: the window here counts DAYS THAT HAVE DATA rather than calendar days, and the
#: output is arithmetic that deliberately stops short of a conclusion.
_SERIES_NOTE = (
    "`days` selects the newest N calendar days THAT HAVE DATA for this series, not "
    "the last N days on the calendar — for a metric the Watch samples occasionally, "
    "30 days of data can span half a year. Read `span_days` beside `n_days` to see "
    "which you got. Days with no reading are skipped, never filled in and never read "
    "as zero. Where a day holds several samples they are averaged, and "
    "`rows_consumed` is how many rows that took. These numbers are arithmetic only: "
    "no threshold, band, grade or verdict is implied by any of them."
)


_AGGREGATIONS = ("none", "avg", "sum", "min", "max", "latest")
_GRANULARITIES = ("day", "week", "month", "weekday")

#: "Every day there is", for the one reader whose window is a day count.
_ALL_DAYS = 36500

_WEEKDAY_NAMES = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")

#: Rides every weekday-bucketed series. The trap it names is the one an agent
#: will actually fall into: a night is dated by the morning it ENDS, so the
#: "Sat" bucket is Friday night's sleep — reading it as "Saturday night" puts
#: the weekend lie-in on the wrong night and inverts the social-jet-lag story.
_WEEKDAY_NOTE = (
    "Buckets are days of the week, Mon..Sun, each aggregating every day of that "
    "weekday in the window; `days_in_period` is how many of that weekday the window "
    "spans and `days_with_data` how many of them had data. Sleep is dated by the day "
    "it ENDED, so `Sat` is the Friday-night sleep and `Mon` the Sunday-night one."
)

#: Series with sub-daily samples worth reading as samples. HRV is the one kind
#: where "when exactly" questions are real (a spike during a stressful minute);
#: the others either record once a day or are only meaningful summed.
_INTRADAY_SERIES = frozenset({"hrv_sdnn"})

_METRIC_NOTE = (
    "`days` selects the newest N calendar days THAT HAVE DATA for each series, not "
    "the last N days on the calendar — quote each series' `coverage.days_covered` "
    "and `coverage.span_days`. Days with no reading are absent, never zero. Where a "
    "day holds several samples its value is their mean. On `cumulative` series the "
    "newest day may be today, still accumulating: it is marked `partial` and left out "
    "of every aggregate. `excluded_days` names days dropped because their data was "
    "short (e.g. the Watch off the wrist) — each is listed with its reason. These are "
    "numbers only; no threshold, band or verdict is implied."
)

_EMPTY_METRIC_HINT = (
    "No values for this series in the data this server can decrypt. That has "
    "several possible causes — recently paired, Apple Health access never granted, "
    "or genuinely not recorded — and this reply cannot tell them apart. Check "
    "`coverage.more_available` first, then call `vaultbeat_doctor`."
)


def _weekday_span_counts(points: dict[str, float]) -> dict[str, int]:
    """How many of each weekday the window's first..last day spans.

    The denominator for a weekday bucket: "4 of 5 Mondays had data" is a
    different finding from "4 of 4", and without it the two look the same.
    """

    counts = {name: 0 for name in _WEEKDAY_NAMES}
    if not points:
        return counts
    day = date.fromisoformat(min(points))
    last = date.fromisoformat(max(points))
    while day <= last:
        counts[_WEEKDAY_NAMES[day.weekday()]] += 1
        day += timedelta(days=1)
    return counts


def _period_key(day: str, granularity: str) -> tuple[str, str, str, int]:
    """(label, first_day, last_day, days_in_period) for the bucket holding *day*."""

    d = date.fromisoformat(day)
    if granularity == "week":
        start = d - timedelta(days=d.weekday())
        end = start + timedelta(days=6)
        iso = start.isocalendar()
        return f"{iso[0]}-W{iso[1]:02d}", start.isoformat(), end.isoformat(), 7
    start = d.replace(day=1)
    nxt = (start.replace(day=28) + timedelta(days=4)).replace(day=1)
    end = nxt - timedelta(days=1)
    return start.strftime("%Y-%m"), start.isoformat(), end.isoformat(), (nxt - start).days


def _aggregate(fn: str, points: dict[str, float], spec: Any) -> dict[str, Any]:
    """One number over a set of days, or null with the reason it was refused."""

    out: dict[str, Any] = {"fn": fn, "value": None, "days_used": len(points)}
    if not points:
        out["reason"] = "no days with data in this window"
        return out
    if fn == "sum" and not spec.cumulative:
        # Summing a state measurement (resting HR, weight, VO2max) produces a
        # number with no meaning that still LOOKS like a quantity — refuse it.
        out["reason"] = (
            f"`{spec.name}` is a measurement of a state, not an amount that accrues, "
            "so a sum of it means nothing. Use avg, min, max or latest."
        )
        return out
    ordered = sorted(points)
    values = [points[d] for d in ordered]
    if fn == "avg":
        out["value"] = round(sum(values) / len(values), 3)
    elif fn == "sum":
        out["value"] = round(sum(values), 3)
    elif fn == "min":
        low = min(ordered, key=lambda d: points[d])
        out["value"], out["day"] = round(points[low], 3), low
    elif fn == "max":
        high = max(ordered, key=lambda d: points[d])
        out["value"], out["day"] = round(points[high], 3), high
    elif fn == "latest":
        out["value"], out["day"] = round(points[ordered[-1]], 3), ordered[-1]
    return out


def _metric_entry(
    spec: Any,
    points: dict[str, float],
    raw: dict[str, Any],
    *,
    consumed: int,
    days: int,
    aggregation: str,
    granularity: str,
    today: str,
) -> dict[str, Any]:
    """One series' block in a `get_metric` reply."""

    entry: dict[str, Any] = {
        "series": spec.name,
        "unit": spec.unit,
        "cumulative": spec.cumulative,
        **({"note": spec.direction_note} if spec.direction_note else {}),
    }

    # Today on an accruing series is a partial day: shown, never aggregated.
    # Leaving it in would drag every average down in one direction only — the
    # 9am "steps are down today" mistake, made by the server on the agent's behalf.
    partial = today if spec.cumulative and today in points else None
    complete = {d: v for d, v in points.items() if d != partial}

    oldest_kept = min(points) if points else None
    excluded = [
        e for e in series_excluded_days(raw, spec)
        if oldest_kept is None or e["day"] >= oldest_kept
    ]
    if partial is not None:
        excluded.append({"day": partial, "reason": "partial_today"})
    if excluded:
        entry["excluded_days"] = sorted(excluded, key=lambda e: e["day"], reverse=True)

    if granularity == "day":
        entry["points"] = [
            {"date": d, "value": round(points[d], 3), **({"partial": True} if d == partial else {})}
            for d in sorted(points, reverse=True)
        ]
        if aggregation != "none":
            entry["aggregate"] = _aggregate(aggregation, complete, spec)
    else:
        fn = aggregation if aggregation != "none" else "avg"
        entry["bucket_fn"] = fn
        buckets: dict[str, dict[str, Any]] = {}
        members: dict[str, dict[str, float]] = {}
        if granularity == "weekday":
            span_days = _weekday_span_counts(points)
            for d, v in complete.items():
                label = _WEEKDAY_NAMES[date.fromisoformat(d).weekday()]
                members.setdefault(label, {})[d] = v
            for label, group in members.items():
                buckets[label] = {
                    "period": label, "first_day": min(group), "last_day": max(group),
                    "days_in_period": span_days[label],
                }
            order = [name for name in _WEEKDAY_NAMES if name in buckets]
            entry["weekday_note"] = _WEEKDAY_NOTE
        else:
            for d, v in complete.items():
                label, first, last, size = _period_key(d, granularity)
                buckets.setdefault(
                    label,
                    {"period": label, "first_day": first, "last_day": last, "days_in_period": size},
                )
                members.setdefault(label, {})[d] = v
            order = sorted(buckets, reverse=True)
        rows = []
        for label in order:
            agg = _aggregate(fn, members[label], spec)
            row = {**buckets[label], "value": agg["value"], "days_with_data": agg["days_used"]}
            if agg.get("reason"):
                row["reason"] = agg["reason"]
            rows.append(row)
        entry["buckets"] = rows

    entry["rows_consumed"] = consumed
    _attach_sources(entry, points, raw)
    _attach_series_coverage(entry, points, requested=days, source=raw)
    # The `days` cut lands wherever the newest-N days end, which is usually
    # mid-week / mid-month — so the OLDEST bucket holds only the tail of its
    # period. `days_with_data < days_in_period` cannot tell that apart from real
    # gaps, and a clipped week's `sum` of steps reads as a week they barely
    # walked. Flag it only when older data is known to exist beyond the window
    # AND the bucket starts before the oldest day kept.
    bucket_rows = entry.get("buckets") if granularity != "weekday" else None
    coverage = entry.get("coverage")
    if (
        bucket_rows
        and points
        and isinstance(coverage, dict)
        and coverage.get("more_available") is True
        and bucket_rows[-1]["first_day"] < min(points)
    ):
        bucket_rows[-1]["clipped_by_window"] = True
    # The other end has the same problem and hits far more often: the NEWEST
    # bucket is usually the current week / month, still running. Measured on
    # the owner's account 2026-09-23 (a Wednesday): "weekly steps, sum" put
    # 2 days into this week's bucket and printed 10,433 beside last week's
    # 42,107 — a collapse that is only the calendar.
    if bucket_rows and bucket_rows[0]["last_day"] >= today:
        bucket_rows[0]["period_in_progress"] = True
    if not points:
        entry["hint"] = _EMPTY_METRIC_HINT
    return entry


#: Attached when a PARTNER read comes back empty. The likeliest cause is not a
#: missing measurement at all: partners share per data type, and only some types
#: can reach this server at all. The list is an iOS fact, checked against the
#: sync executors' `mcpPolicy` (2026-09-23): sleep, water and body go to the
#: partner's AI by default (`.allVisible`); cycle, symptoms and notes only when
#: the partner flips that type's "share with partner's AI" switch; every other
#: kind is `.ownDevicesOnly` and never leaves their own devices. It said "sleep,
#: cycle, water and weight" until 0.9.0 — which made an empty `get_symptoms` or
#: `get_notes(partner=true)` read as "cannot be shared" when it can.
PARTNER_EMPTY_HINT = (
    "Nothing of your partner's came back for this. Partner data reaches this "
    "server only for the types they share, and only on a paired account: sleep, "
    "water and weight (including body composition) are shared by default; cycle, "
    "Apple Health symptoms and notes only if your partner turned on sharing them "
    "with your AI in their own Vaultbeat app; everything else — including "
    "symptoms they reported themselves — is never shared. Treat this as "
    "'not shared', never as 'they did not do it'."
)


def _mixed_owner_refusal(spec: Any, raw: dict[str, Any], owner: str | None) -> dict[str, Any] | None:
    """Refuse one-number-per-day arithmetic over two people's rows.

    The read tools could get away with a WARNING on a blended result because
    their rows still carried `owner_user_id` — an agent could split them. This
    layer collapses each day to one number first, so after bucketing the two
    people are physically inseparable: an 82 kg day and a 40 kg day come out as
    one 61 kg day that nobody weighed. A number that was true of neither person
    must not be produced, so it is refused before it exists.
    """

    if owner or not raw.get("mixed_owners"):
        return None
    prefixes = raw.get("owner_user_id_prefixes") or []
    return {
        "series": spec.name,
        "error": "mixed_owners",
        "owner_user_id_prefixes": prefixes,
        "message": (
            "This account holds more than one person's data and the server could not "
            "tell which one is you, "
            "so each day would average two bodies into one number that is true of "
            "neither. This server could not tell whose account paired it, so it cannot "
            "default to 'you' — re-pair from the Vaultbeat app to fix that."
        ),
    }


def _attach_series_exclusions(
    result: dict[str, Any],
    points: dict[str, float],
    raw: dict[str, Any],
    spec: Any,
    *,
    partial_today: str | None = None,
) -> None:
    """Name the days the arithmetic dropped on purpose (see `SeriesSpec.exclude_when`),
    plus today on an accruing series — same `partial_today` reason `get_metric` gives."""

    oldest = min(points) if points else None
    dropped = [
        e for e in series_excluded_days(raw, spec) if oldest is None or e["day"] >= oldest
    ]
    if partial_today is not None:
        dropped.append({"day": partial_today, "reason": "partial_today"})
    if dropped:
        result["excluded_days"] = dropped


def _fetch_was_cut(summary: dict[str, Any], spec: Any, limit: int) -> bool:
    """Did `limit` leave older data behind? Ask the read method, not the array.

    The method's own `coverage.more_available` is the authority, because only it
    knows what its `limit` counts. Counting the returned array against `limit`
    is right for the kinds that return one row per unit of `limit`, and wrong
    for basal energy: its `limit` counts HOURLY samples while its `daily` array
    holds DAYS, so 30 samples come back as 2 rows, "2 < 30" reads as "history
    ended", and a 7-day request answered with one day (found on the owner's
    real account 2026-09-23, through `get_metric`; the trend tool had the same
    short window all along). The row count stays as the fallback for a reader
    that does not say.
    """

    said = (summary.get("coverage") or {}).get("more_available")
    if isinstance(said, bool):
        return said
    return _rows_returned(summary, spec) >= limit


def _rows_returned(summary: dict[str, Any], spec: Any) -> int:
    """How many rows the read actually handed back, valid or not.

    Deliberately NOT `daily_series`' `consumed`, which counts only rows that
    survived validation. The question here is "did this response hit the limit",
    and a truncated page full of rows this layer cannot bucket still hit it.
    """

    rows = summary.get(spec.array)
    return len(rows) if isinstance(rows, list) else 0


def _attach_series_coverage(
    result: dict[str, Any],
    points: dict[str, float],
    *,
    requested: int | None,
    source: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The standard coverage block, over an already-bucketed `{day: value}` map.

    Reuses `_attach_coverage` rather than hand-rolling the fields, so an analysis
    result and a read result answer "how much data is this?" with the same keys,
    the same note, and the same meaning. That matters more here than anywhere
    else: `STYLE` tells every agent to quote `coverage.days_covered` beside a
    trend, and a trend tool that spelled it `n_days` would be the one place that
    instruction silently fails.

    The rows handed in are synthetic one-per-day stubs because the days ARE the
    rows at this layer — the real ones were collapsed by `daily_series`, and
    passing those instead would count samples where the caller asked for days.

    *source* is the underlying read tool's own summary (`_series_points`'s third
    return value), and `more_available` is INHERITED from it rather than
    recomputed. It has to be: by this layer the rows are already bucketed into
    `{day: value}`, so nothing here can still see what the fetch left behind —
    and the fetch already worked it out. Re-deriving it from `points` would only
    ever report on the slice this layer was handed, which is the bug the field
    exists to expose, reintroduced one level up.
    """

    source_coverage = (source or {}).get("coverage") or {}
    return _attach_coverage(
        result,
        rows=[{"local_date": day} for day in sorted(points)],
        requested=requested,
        unit="days",
        total_available=source_coverage.get("total_available"),
        oldest_raw=source_coverage.get("oldest_available"),
    )


def _coverage_day_of(row: Any) -> str | None:
    """The LOCAL calendar day one returned row belongs to, or None.

    Reads the row as it will be serialised rather than the typed object behind
    it, so coverage can never describe a different set than the numbers printed
    next to it: every `to_dict()` in this module emits `local_date`, and the
    three hand-built row shapes (basal `daily`, TDEE `days`, sleep
    `daily_summary`) carry a bare local day under "day" / "date".

    Both of those are ALREADY local days, which is why the bare-day branch below
    fires for the common case too and not just the hand-built rows — see the
    `_BARE_DAY_LENGTH` comment for what reparsing them would cost. Anything
    unparseable costs that one row its place in the count instead of failing the
    read, matching `_cut_newest_by`.
    """

    if not isinstance(row, dict):
        return None
    for key in _COVERAGE_DATE_KEYS:
        value = row.get(key)
        if not isinstance(value, str) or not value:
            continue
        if len(value) == _BARE_DAY_LENGTH and value[4] == "-" and value[7] == "-":
            return value
        local = _local_date_fields(value).get("local_date")
        if isinstance(local, str):
            return local
    return None


def _attach_coverage(
    summary: dict[str, Any],
    *,
    # `Any`, not `Iterable[Any]`: the call sites pass `summary["records"]` and
    # friends out of heterogeneous dict literals, which mypy infers as `object`.
    # Tightening this annotation costs eight `cast`s and buys no safety — the
    # body already tolerates any iterable and any row shape.
    rows: Any,
    requested: int | None,
    unit: str,
    cut_count: int | None = None,
    displayed: Any = None,
    total_available: int | None = None,
    oldest_raw: str | None = None,
) -> dict[str, Any]:
    """Attach a `coverage` block stating how many days this result rests on.

    ADD-ONLY: introduces one new top-level key and never reads, edits or removes
    an existing field — the same discipline `_attach_errors` /
    `_attach_owner_guard` / mcp_server's `_annotate_if_empty` follow, because
    payload shape is the one contract layer no server can validate.

    Why it exists: a read tool returns records and aggregates, and an agent that
    wants to say "this is based on 3 days" has nothing to read — the array
    length has already been cut by `limit` (Invariant 38), so counting it is
    wrong, and there is no other signal. That is the missing half of the
    2026-08-02 harm: the statistics were wrong, but what let a wrong statistic
    be acted on the same afternoon is that "how many points is this?" never
    appeared beside the answer. This cannot stop an agent reasoning badly; it
    gives a careful one something to cite.

    *rows* must be the set the summary's own aggregates are computed over — for
    every tool here that is also the set it returns, except
    `basal_energy_records`, whose `daily` list is display-capped by `day_limit`
    while its average is not (the sibling bug in Invariant 62). Pass the
    uncapped list there.

    *displayed* is the list the summary actually PRINTS, and is only needed when
    that is a display-capped subset of *rows*; it defaults to *rows*, which is
    the truth for all but one reader. It exists because with only one of the two
    sets the block could not tell them apart, and every field silently described
    the wider one: `basal_energy_records` reported `first_day` a day older than
    anything in `daily` and `days_missing_in_span: 0` beside it, so an agent was
    told a day was present, could not find it anywhere in the response, and was
    told by the note not to go looking. `days_in_payload` is the separation —
    coverage still MEASURES past the cap (Invariant 64), it now also says how
    far the printing stops short.

    *cut_count* overrides what `window_satisfied` compares against. It defaults
    to the row count, which is the honest denominator almost everywhere: when a
    summary dedups after the cut (water/weight/menstrual by dayID, symptom by
    day_id, strength/food by entry_id) the rows that survive ARE how many days
    the caller actually got, and comparing the pre-dedup count instead would
    answer "yes you got your 10 days" to a read holding 5. Erring toward
    `False` is the safe direction for a field whose whole job is to stop an
    answer being over-trusted. Only two readers override it, both because the
    row count answers a different question than the limit asked: `notes_summary`
    (`target_kind` discards kinds the caller did not ask for, which is not
    scarcity) and `basal_energy_records` (`limit` caps SAMPLES while the rows
    are days).

    *total_available* and *oldest_raw* are read one line BEFORE `limit` cuts,
    and they answer the one question this block could never answer before: **is
    there older history behind the cut?** Every other field here describes what
    came back; `oldest_available` / `total_available` / `more_available` describe
    what was there to come back. They are two scalars rather than the pre-cut
    list because Invariant 38 has already sorted it by business day descending —
    so its length and its LAST element are the whole answer, and no reader pays
    to re-serialise a list it is about to throw away. *oldest_raw* is that last
    element's own date string, in whatever shape the reader already holds
    (bare day or ISO instant); this function normalises it through the same
    `_coverage_day_of` every other field uses, so a caller cannot introduce a
    second notion of what day a row falls on.

    🔴 **Why it was added (2026-09-14, a real and silent product failure).** A
    paying user's agent called `get_activity()`, took the default `limit=30`,
    and received thirty days out of the 822 this server could decrypt. Nothing
    in the response was false and nothing was missing by its own definition —
    `days_covered: 30`, `days_missing_in_span: 0`, `window_satisfied: true` —
    so the agent reported "you only have the last month", which is the exact
    sentence a user who bought full history must never read. On the sample
    kinds it is starker still: `get_hrv(limit=45)` is six days of a 357-day
    history, and the same three fields go green. **`limit` counts rows; the
    agent is reasoning about time; and until this field existed no part of the
    payload knew the difference.** Compare with Invariant 63 (a-slow-backfill-may-be-a-wall)
    — that one warns against widening a window that is really a paywall. This
    is its mirror: a window that is really just an argument default. The
    distinguishing question is whether the data is THERE, and `more_available`
    is now literally the answer to it.

    ⚠️ **`more_available` is computed from THIS server's decryptable set**, so it
    cannot leak across the paywall: a free account has only what its app
    uploaded sealed for it — nothing new from app 1.2.9, 7 days on 1.2.8 and
    earlier (Invariant 72 (free-tier-uploads-seven-days)) — so the field reads `false`
    for exactly the reason it should — there is no more, not "we won't say".

    ⚠️ **It compares DAYS, not row counts**, and that is deliberate. A
    `limit` that lands mid-day, or a dedupe that runs after the cut, both shrink
    the row count without hiding any history; only `oldest_available < first_day`
    means an older day exists that the caller did not get. Counting rows here
    would fire `true` on every deduped read and teach agents to ignore it —
    Invariant 62's "a flag that cries wolf gets tuned out" applied to a new field.
    """

    # Materialised up front: `rows` is iterated twice below, and a caller passing
    # a generator would otherwise get days from the first pass and rows_counted=0
    # from the second — a coverage block that contradicts itself.
    row_list = list(rows)
    days = sorted({d for d in (_coverage_day_of(row) for row in row_list) if d})
    # Counted off the list that really ships rather than re-deriving it from a
    # cap: a cap read a second time is a second chance to read it wrong, which
    # is the whole shape of the bug this field exists to close.
    days_in_payload = (
        len(days)
        if displayed is None
        else len({d for d in (_coverage_day_of(row) for row in displayed) if d})
    )
    first_day = days[0] if days else None
    last_day = days[-1] if days else None

    span_days: int | None = None
    missing: int | None = None
    if first_day is not None and last_day is not None:
        try:
            span_days = (date.fromisoformat(last_day) - date.fromisoformat(first_day)).days + 1
        except ValueError:  # a row carried a day-shaped string that is not a day
            span_days = None
        if span_days is not None:
            missing = max(span_days - len(days), 0)

    counted = cut_count if cut_count is not None else len(row_list)

    # What was there BEFORE the cut. `None` throughout when the caller did not
    # say — an honest "not stated" rather than a fabricated `false`, which would
    # assert there is no more history on exactly the readers that have not been
    # taught to look.
    oldest_available = _coverage_day_of({"date": oldest_raw}) if oldest_raw else None
    more_available: bool | None = None
    if total_available is not None:
        # Strictly older, by day. See the docstring: rows lost to a mid-day cut
        # or to a post-cut dedupe are not hidden history.
        more_available = bool(
            oldest_available is not None
            and first_day is not None
            and oldest_available < first_day
        )

    summary["coverage"] = {
        "days_covered": len(days),
        "days_in_payload": days_in_payload,
        "first_day": first_day,
        "last_day": last_day,
        "span_days": span_days,
        "days_missing_in_span": missing,
        "rows_counted": counted,
        "requested": requested,
        "requested_unit": unit,
        "oldest_available": oldest_available,
        "total_available": total_available,
        "more_available": more_available,
        # Both halves, and the second half is the 2026-09-14 fix: "you got the
        # number of rows you asked for" was answering a question nobody asked
        # while reading as an all-clear. A read that leaves older days behind is
        # NOT a satisfied window, however many rows it returned.
        "window_satisfied": (
            None if requested is None else (counted >= requested and not more_available)
        ),
        "note": _COVERAGE_NOTE,
    }
    return summary


def _merge_food_meals(
    existing: list[dict[str, Any]], new: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Append-merge for ``log_food_entry(merge=True)``.

    A new meal whose (case-insensitive) name matches an existing named meal has
    its items appended to that meal; every other new meal is appended whole.
    Existing content is NEVER dropped or rewritten — the whole point of merge
    mode is that "log one forgotten snack" cannot wipe the rest of the day.
    ``existing`` comes from the decoded cloud payload and is trusted as-is
    (re-normalizing it would strip fields this normalizer doesn't know about).
    """

    merged: list[dict[str, Any]] = [
        dict(meal, items=list(meal.get("items") or [])) for meal in existing if isinstance(meal, dict)
    ]
    by_name: dict[str, dict[str, Any]] = {}
    for meal in merged:
        name = meal.get("name")
        if isinstance(name, str) and name.strip():
            by_name.setdefault(name.strip().casefold(), meal)
    for meal in new:
        name = meal.get("name")
        key = name.strip().casefold() if isinstance(name, str) and name.strip() else None
        target = by_name.get(key) if key is not None else None
        if target is not None:
            target["items"].extend(meal.get("items") or [])
            if meal.get("note") and not target.get("note"):
                target["note"] = meal["note"]
        else:
            merged.append(meal)
            if key is not None:
                by_name.setdefault(key, meal)
    return merged


def _merge_strength_exercises(
    existing: list[dict[str, Any]], new: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Append-merge for ``log_strength_entry(merge=True)``. Mirrors ``_merge_food_meals``.

    A new exercise whose (case-insensitive) name matches an existing one has its
    sets appended to that exercise; every other new exercise is appended whole.
    Existing content is NEVER dropped — "log the one set I forgot" must not wipe
    the rest of the session.

    ``existing`` comes from the decoded cloud payload and is trusted as-is: it
    was normalized when it was written, and re-normalizing would strip any field
    a future iOS version adds that this normalizer predates.
    """

    merged: list[dict[str, Any]] = [
        dict(exercise, sets=list(exercise.get("sets") or []))
        for exercise in existing
        if isinstance(exercise, dict)
    ]
    by_name: dict[str, dict[str, Any]] = {}
    for exercise in merged:
        name = exercise.get("name")
        if isinstance(name, str) and name.strip():
            by_name.setdefault(name.strip().casefold(), exercise)
    for exercise in new:
        name = exercise.get("name")
        key = name.strip().casefold() if isinstance(name, str) and name.strip() else None
        target = by_name.get(key) if key is not None else None
        if target is not None:
            target["sets"].extend(exercise.get("sets") or [])
        else:
            merged.append(exercise)
            if key is not None:
                by_name.setdefault(key, exercise)
    return merged


def _resolve_note(supplied: str | None, existing: Any) -> str | None:
    """Decide the note to persist on an upsert that rewrites a whole day.

    ``None`` means "leave the note alone" — an agent that omits the argument is
    saying nothing about the note, not asking to erase it. Passing an empty or
    whitespace-only string is the explicit way to clear it.

    Before 2026-07-28 every writer did `note.strip() if isinstance(note, str)
    and note.strip() else None`, so any call that did not re-send the note
    silently destroyed it — 07-27's session lost `'腿日 + 腹肌'` that way. The
    surrounding writes are whole-day replacements, which makes the omission
    especially easy: you resend the data you care about and the note evaporates.
    """

    if supplied is None:
        return existing if isinstance(existing, str) and existing.strip() else None
    stripped = supplied.strip()
    return stripped or None


def summarize_water_intake(days: list[WaterDay]) -> dict[str, Any]:
    """Recent daily intake plus the average over the available window.

    Average daily intake = sum over days of (refillEvents.count * that day's
    containerVolumeLiters) / number_of_days_in_window. Days are deduplicated by dayID
    (most-recent dayStartDate wins) and returned newest-first.
    """

    if not days:
        _LOG.info("water summary requested with no decoded water days available")
        return {"days": [], "average_daily_intake_liters": None, "day_count": 0}

    # Dedup by dayID: newest dayStartDate wins, last-iterated wins on an exact tie
    # — matches the iOS aggregator (VaultbeatWaterIntakeAggregator) so the AI and the
    # app agree. (In practice the upsert keeps one blob per dayID.)
    by_id: dict[str, WaterDay] = {}
    for day in days:
        existing = by_id.get(day.day_id)
        if existing is None or day.day_start_date >= existing.day_start_date:
            by_id[day.day_id] = day

    ordered = sorted(by_id.values(), key=lambda d: d.day_start_date, reverse=True)
    total = sum(day.intake_liters for day in ordered)
    average = total / len(ordered)
    return {
        "days": [day.to_dict() for day in ordered],
        "average_daily_intake_liters": average,
        "day_count": len(ordered),
    }


def summarize_weight_trend(days: list[BodyDay], *, goal_kg: float | None = None) -> dict[str, Any]:
    """Recent daily weights plus latest/average/min/max, distance to goal, and weekly rate.

    Days are deduplicated by dayID (most-recent dayStartDate wins, matching the water
    summary and the one-blob-per-dayID upsert) and returned newest-first.

    - delta_to_goal_kg = latest - goal; negative means already below the goal, which is
      good in the weight-loss framing (mirrors iOS WeightRangeAggregate.deltaToGoalKg).
    - weekly_rate_kg_per_week is the ordinary-least-squares linear-regression slope of
      weight (kg) over time, scaled to kg/week — algorithm aligned with iOS
      WeightRangeAggregate.weeklyRateKgPerWeek so the AI and the app report the same
      trend. Needs >=2 distinct timestamps; otherwise reported as None, not guessed.
    """

    if not days:
        _LOG.info("weight summary requested with no decoded body days available")
        return {
            "days": [],
            "day_count": 0,
            "latest_kg": None,
            "average_kg": None,
            "min_kg": None,
            "max_kg": None,
            "goal_kg": goal_kg,
            "delta_to_goal_kg": None,
            "weekly_rate_kg_per_week": None,
        }

    # Dedup by dayID: newest dayStartDate wins (same rule as summarize_water_intake).
    by_id: dict[str, BodyDay] = {}
    for day in days:
        existing = by_id.get(day.day_id)
        if existing is None or day.day_start_date >= existing.day_start_date:
            by_id[day.day_id] = day

    ordered = sorted(by_id.values(), key=lambda d: d.day_start_date, reverse=True)
    weights = [day.weight_kg for day in ordered]
    latest = ordered[0].weight_kg
    average = sum(weights) / len(weights)

    # OLS slope over (days since first record, kg) -> kg/day, then * 7 -> kg/week.
    # Aligned with iOS WeightRangeAggregate: same least-squares slope, same week scale.
    weekly_rate: float | None = None
    points = sorted(
        (( _parse_iso8601(day.day_start_date).timestamp(), day.weight_kg) for day in ordered),
        key=lambda p: p[0],
    )
    if len(points) >= 2:
        seconds_per_day = 86400.0
        xs = [(t - points[0][0]) / seconds_per_day for t, _ in points]
        ys = [kg for _, kg in points]
        mean_x = sum(xs) / len(xs)
        mean_y = sum(ys) / len(ys)
        var_x = sum((x - mean_x) ** 2 for x in xs)
        if var_x > 0:  # all-same-timestamp window has no defined slope
            slope_per_day = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / var_x
            weekly_rate = slope_per_day * 7.0

    return {
        "days": [day.to_dict() for day in ordered],
        "day_count": len(ordered),
        "latest_kg": latest,
        "average_kg": average,
        "min_kg": min(weights),
        "max_kg": max(weights),
        "goal_kg": goal_kg,
        "delta_to_goal_kg": (latest - goal_kg) if goal_kg is not None else None,
        "weekly_rate_kg_per_week": weekly_rate,
    }


# A new cycle begins when a bleeding day follows a gap longer than this many days from
# the previous bleeding day (contiguous bleeding belongs to one period).
_CYCLE_GAP_THRESHOLD_DAYS = 2


def _cycle_starts(days: list[MenstrualDay]) -> list[datetime]:
    """Distinct cycle-start datetimes: the first bleeding day of each cycle.

    A day counts as bleeding if it carries at least one sample with flow other than
    "none". HealthKit's "unspecified" means "flow occurred, amount not specified"
    (Apple Health's quick period log writes exactly this), so it counts as bleeding —
    mirrors the Swift `VaultbeatMenstrualFlowLevel.isBleeding`. Consecutive bleeding days
    within _CYCLE_GAP_THRESHOLD_DAYS of each other belong to the same cycle; a larger
    gap opens a new cycle. Returns starts oldest-first.
    """

    bleeding_days = sorted(
        {
            _parse_iso8601(day.day_start_date)
            for day in days
            if any(sample.flow != "none" for sample in day.samples)
        }
    )
    if not bleeding_days:
        return []

    starts = [bleeding_days[0]]
    previous = bleeding_days[0]
    for current in bleeding_days[1:]:
        if (current - previous).days > _CYCLE_GAP_THRESHOLD_DAYS:
            starts.append(current)
        previous = current
    return starts


# Cycle statistics — MUST stay logic-identical to Swift's
# VaultbeatMenstrualCycleAggregator (any change lands in both in the same commit).
_CYCLE_STATISTICS_WINDOW = 12  # most-recent gaps considered (~1 year of rhythm)
_MIN_GAPS_FOR_VARIABILITY = 3  # below this a spread estimate is noise

# Biphasic-shift ovulation detection — MUST stay logic-identical to Swift's
# VaultbeatWristTemperatureOvulationDetector. Wrist temp is `.ownDevicesOnly`, so
# this server only ever holds the OWNER's deltas — the function fires when the
# same person tracks both cycle and wrist temperature here (gender-neutral:
# whoever tracks, benefits). Threshold awaits real-cycle calibration.
_OVULATION_BASELINE_POINTS = 6
_OVULATION_SUSTAINED_POINTS = 3
_OVULATION_SUSTAINED_MAX_SPAN_DAYS = 4
_OVULATION_SHIFT_THRESHOLD_C = 0.15


def _local_calendar_day(value: datetime) -> date:
    """Floor a datetime to its LOCAL calendar day (a `date`).

    Aware datetimes (the `_parse_iso8601` output — UTC) convert to the
    server's local timezone first, matching Swift's `Calendar.current` day
    bucketing (the server runs in the same timezone as the phone); naive
    datetimes (tests) are taken at face value. Bucketing by UTC day instead
    would shift every reading a day for UTC+8 users.
    """

    if value.tzinfo is not None:
        value = value.astimezone()  # per-instant offset — see `_local_midnight`
    return value.date()


def new_entry_blob_id(kind: str) -> str:
    """A fresh blob id for an entry-shaped agent write: `{kind}-{32 hex}`.

    The same shape the iOS app mints for these kinds, and the ONLY shape
    `mcp-write-{kind}` accepts (`BLOB_ID_SHAPES` in
    supabase/functions/_shared/mcpWrite.ts, GitHub #9). Change one and every
    agent write of that kind is refused with `invalid_blob_id`;
    `scripts/ci/check_write_blob_id_shapes.py` holds the two together.
    """
    return f"{kind}-{secrets.token_hex(16)}"


def body_day_blob_id(day_start_epoch: int, owner_user_id: str | None = None) -> str:
    """The blob id of one body day: `body-{local-midnight epoch}`.

    With `owner_user_id`, the id `upsert_sleep_payload` gives the same day when
    another account already holds the plain one — `-u` plus the first eight hex
    digits of the owner's uuid. Body ids are not per-user (two people in one
    time zone share every day's epoch), so a second account's weigh-in lives
    under that remapped id, and an agent writing for it has to use it too.
    """
    base = f"body-{int(day_start_epoch)}"
    if owner_user_id is None:
        return base
    return f"{base}-u{owner_user_id.replace('-', '').lower()[:8]}"


def _local_midnight_iso(day: date) -> str:
    """UTC ISO8601 instant for local midnight of `day` — the inverse of
    ``_local_calendar_day``, used when the agent writes a brand-new entry for a
    given local calendar day. Same "server runs in the phone's timezone"
    assumption as that function; see its docstring.
    """

    return _local_midnight(day).astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _local_midnight(day: date) -> datetime:
    """Local midnight of `day`, carrying the UTC offset in force ON THAT DAY.

    🔴 Every local-time conversion here used to take its zone from
    `datetime.now().astimezone().tzinfo` — a FIXED offset, today's. In a zone
    with daylight saving that is an hour wrong for half the year: once a US
    user's clocks go back on 1 November, every summer entry's local midnight
    lands at 23:00 the day before, so a note's `target_date`, a strength or
    food day, and a cycle's start all read a day early (review R2 on #9,
    2026-10-03). A naive local datetime's `.astimezone()` asks the OS for the
    offset that applied at that moment instead, which is the same thing
    Swift's `Calendar.current` does on the phone.
    """

    return datetime(day.year, day.month, day.day).astimezone()


def _normalize_strength_exercises(exercises: Any) -> list[dict[str, Any]]:
    """Validate + coerce agent-supplied exercises into the iOS wire shape.

    Accepts `weightKg` or `weight_kg` for the common case of an agent writing
    snake_case; the OUTPUT is always camelCase (`weightKg`) to match what the
    iOS decoder and every other reader (`parse_strength`) expects.
    """

    if not isinstance(exercises, list) or not exercises:
        raise ValueError("exercises must be a non-empty list")

    normalized: list[dict[str, Any]] = []
    for exercise in exercises:
        if not isinstance(exercise, dict):
            raise ValueError("each exercise must be an object")
        name = exercise.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("each exercise needs a non-empty name")
        sets_in = exercise.get("sets")
        if not isinstance(sets_in, list) or not sets_in:
            raise ValueError(f"exercise {name!r} needs a non-empty sets list")

        sets_out: list[dict[str, Any]] = []
        for one_set in sets_in:
            if not isinstance(one_set, dict):
                raise ValueError(f"exercise {name!r} has a non-object set")
            weight = one_set.get("weightKg", one_set.get("weight_kg"))
            reps = one_set.get("reps")
            if weight is None or reps is None:
                raise ValueError(f"exercise {name!r} has a set missing weight/reps")
            try:
                weight_kg = float(weight)
                rep_count = int(reps)
            except (TypeError, ValueError) as error:
                raise ValueError(f"exercise {name!r} has a non-numeric weight/reps") from error
            if weight_kg < 0 or rep_count <= 0:
                raise ValueError(f"exercise {name!r} has an invalid weight/reps")
            sets_out.append({"weightKg": weight_kg, "reps": rep_count})

        normalized.append({"name": name.strip(), "sets": sets_out})
    return normalized


# Optional structured nutrition on a food item: wire key (camelCase, matching
# every other wire field) plus the snake_case aliases an agent will naturally
# type. Values are kcal / grams. Recording stays optional — friction-free "just
# log 香蕉" still works — but when the agent DOES estimate at logging time the
# numbers persist instead of living only in free-text portion strings that every
# later session re-parses inconsistently (2026-07-23 client feedback). iOS-safe:
# Swift's Codable ignores unknown keys; an iOS edit of the same day re-encodes
# without them (acceptable — editing a day is defined as rewriting it).
_FOOD_NUTRITION_KEYS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("kcal", ("kcal", "calories")),
    ("proteinGrams", ("proteinGrams", "protein_g", "protein_grams")),
    ("fatGrams", ("fatGrams", "fat_g", "fat_grams")),
    ("carbGrams", ("carbGrams", "carb_g", "carb_grams", "carbs_g")),
)


def _normalize_food_meals(meals: Any) -> list[dict[str, Any]]:
    """Validate + coerce agent-supplied meals into the iOS wire shape.

    Structure = `[{name?, timeOfDay?, items: [{food, portion?, note?,
    kcal?, proteinGrams?, fatGrams?, carbGrams?}], note?}]`.
    `name` (breakfast/lunch/…) and `timeOfDay` (HH:MM) are optional; `items` is
    required non-empty; per-item `food` is required; `portion` is a free-text
    (e.g. "1 根" / "300g" / "小份") because tight units would kill entry speed
    for the marginal analytical value — the AI can normalize on read. The
    nutrition fields are optional numbers (see _FOOD_NUTRITION_KEYS); anything
    NOT in the allow-list is dropped, so numbers passed under an unknown key
    would vanish silently — hence the aliases.
    """

    if not isinstance(meals, list) or not meals:
        raise ValueError("meals must be a non-empty list")

    normalized: list[dict[str, Any]] = []
    for meal in meals:
        if not isinstance(meal, dict):
            raise ValueError("each meal must be an object")
        items_in = meal.get("items")
        if not isinstance(items_in, list) or not items_in:
            raise ValueError("each meal needs a non-empty items list")

        items_out: list[dict[str, Any]] = []
        for item in items_in:
            if not isinstance(item, dict):
                raise ValueError("each meal item must be an object")
            food = item.get("food")
            if not isinstance(food, str) or not food.strip():
                raise ValueError("each meal item needs a non-empty food name")
            out_item: dict[str, Any] = {"food": food.strip()}
            portion = item.get("portion")
            if isinstance(portion, str) and portion.strip():
                out_item["portion"] = portion.strip()
            item_note = item.get("note")
            if isinstance(item_note, str) and item_note.strip():
                out_item["note"] = item_note.strip()
            for out_key, aliases in _FOOD_NUTRITION_KEYS:
                for alias in aliases:
                    value = item.get(alias)
                    if value is None:
                        continue
                    try:
                        number = float(value)
                    except (TypeError, ValueError) as error:
                        raise ValueError(
                            f"item {food!r} has a non-numeric {alias}: {value!r}"
                        ) from error
                    if number < 0:
                        raise ValueError(f"item {food!r} has a negative {alias}")
                    out_item[out_key] = number
                    break
            items_out.append(out_item)

        out_meal: dict[str, Any] = {"items": items_out}
        name = meal.get("name")
        if isinstance(name, str) and name.strip():
            out_meal["name"] = name.strip()
        time_of_day = meal.get("timeOfDay", meal.get("time_of_day"))
        if isinstance(time_of_day, str) and time_of_day.strip():
            out_meal["timeOfDay"] = time_of_day.strip()
        meal_note = meal.get("note")
        if isinstance(meal_note, str) and meal_note.strip():
            out_meal["note"] = meal_note.strip()
        normalized.append(out_meal)

    return normalized


def detect_ovulation_from_wrist_temp(
    readings: list[tuple[datetime, float]],
    cycle_start: datetime,
) -> date | None:
    """Estimated ovulation day (a local calendar `date`) or None.

    Classic 3-over-6 rule on this cycle's readings: baseline = median of the
    previous 6 readings; a shift = 3 consecutive readings all >= baseline +
    threshold, spanning < 4 calendar days; ovulation ~= the day before the
    first elevated reading. Retrospective by nature — it confirms, it does not
    forecast; the caller fuses it as `ovulation + luteal 14` (mirrors Swift's
    VaultbeatCyclePredictionCalculator / VaultbeatMenstrualCycleSummary.calibrated).
    """

    cycle_start_day = _local_calendar_day(cycle_start)
    delta_by_day: dict[date, float] = {}
    for day, delta in readings:
        day_key = _local_calendar_day(day)
        if day_key < cycle_start_day:
            continue
        delta_by_day[day_key] = delta
    series = sorted(delta_by_day.items())
    if len(series) < _OVULATION_BASELINE_POINTS + _OVULATION_SUSTAINED_POINTS:
        return None

    for index in range(_OVULATION_BASELINE_POINTS, len(series) - _OVULATION_SUSTAINED_POINTS + 1):
        baseline = _median([v for _, v in series[index - _OVULATION_BASELINE_POINTS:index]])
        if baseline is None:  # unreachable: the slice is always BASELINE_POINTS long
            continue
        run = series[index:index + _OVULATION_SUSTAINED_POINTS]
        if all(v >= baseline + _OVULATION_SHIFT_THRESHOLD_C for _, v in run) and (
            (run[-1][0] - run[0][0]).days < _OVULATION_SUSTAINED_MAX_SPAN_DAYS
        ):
            return run[0][0] - timedelta(days=1)
    return None


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2 == 0:
        return (ordered[middle - 1] + ordered[middle]) / 2
    return ordered[middle]


# Mirrors Swift's VaultbeatCyclePredictionCalculator.lutealPhaseDays.
_LUTEAL_PHASE_DAYS = 14


#: Two asleep samples closer than this are one continuous bout. Apple writes
#: stage samples back to back, but a Watch reconnect can leave a gap of a few
#: seconds; without a tolerance every such seam would read as an awakening.
_SLEEP_BOUT_GAP_SECONDS = 120


def _sleep_continuity(
    intervals: list[tuple[float, float, str]], asleep_stages: set[str],
) -> tuple[int, int]:
    """(awakenings, longest continuous sleep in minutes) for one night.

    An awakening is an `awake` interval that starts after the first asleep
    sample and before the last one ends — waking up for the day is not an
    awakening, and neither is lying awake before sleep arrives. A bout is a run
    of asleep intervals with no `awake` between them and no gap longer than
    `_SLEEP_BOUT_GAP_SECONDS`.
    """

    asleep = [iv for iv in intervals if iv[2] in asleep_stages]
    if not asleep:
        return 0, 0
    first_start = min(iv[0] for iv in asleep)
    last_end = max(iv[1] for iv in asleep)
    awakenings = sum(
        1 for start, _end, stage in intervals
        if stage == "awake" and first_start < start < last_end
    )

    longest = 0.0
    bout_start: float | None = None
    bout_end = 0.0
    for start, end, stage in sorted(intervals, key=lambda iv: iv[0]):
        if stage in asleep_stages:
            if bout_start is None or start - bout_end > _SLEEP_BOUT_GAP_SECONDS:
                bout_start = start
            bout_end = max(bout_end, end)
            longest = max(longest, bout_end - bout_start)
        elif stage == "awake":
            bout_start = None
    return awakenings, int(longest / 60)


#: Minimum asleep-stage samples before a night's heart / breathing rate is
#: reported. A real Watch night carries ~90 HR and ~35 RR samples; the night
#: that prompted this carried 2 and read "100 bpm asleep". Low enough that a
#: sparse-but-real night (a band sampling every ~90 min) still counts.
_MIN_SLEEP_HR_SAMPLES = 5
_MIN_SLEEP_RR_SAMPLES = 3


def _source_label(source_id: str) -> str:
    """A stable, human-readable name for a HealthKit sample source.

    Every `com.apple.health.<UUID>` is Apple's own sleep scoring and is named
    `apple`. The UUID is NOT a physical device: it is the identity the device
    was registered under, and re-pairing or restoring a Watch mints a new one.
    Until 2026-09-24 it was kept as `apple-xxxx`, and one Watch re-paired around
    2025-10-01 showed up as "two Apple devices" — an invented change of
    instrument, the exact misreading this label exists to prevent (owner:
    「apple设备a和b是一个啊」; the two ids' stage shares matched to the point).
    The question this answers is "same algorithm or not", and Apple is one.
    Anything else is an app's bundle id, named by its last meaningful component.
    """

    sid = source_id.strip()
    lowered = sid.lower()
    if lowered.startswith("com.apple.health."):
        return "apple"
    if lowered == "com.apple.health":
        return "apple-health-app"
    known = {"huawei": "huawei", "otterlife": "otterlife", "xiaomi": "xiaomi", "garmin": "garmin",
             "fitbit": "fitbit", "oura": "oura", "whoop": "whoop", "withings": "withings",
             "autosleep": "autosleep", "sleepcycle": "sleep-cycle", "pillow": "pillow"}
    for needle, name in known.items():
        if needle in lowered:
            return name
    parts = [p for p in lowered.split(".") if p not in ("com", "app", "ios", "net", "org", "io")]
    return parts[-1] if parts else lowered or "unknown"


def _sleep_source(samples: list[dict[str, Any]], provenance: str | None = None) -> str | None:
    """The source that wrote most of a night's asleep time.

    Samples without a `sourceID` fall back to the session's provenance:
    `motionInferred` nights were inferred from phone motion, not measured by a
    wearable (10 real nights, 2026-09-24 — long, unstaged, all while the Watch
    was not in use), which is the difference an agent most needs to see.
    """

    seconds: dict[str, float] = {}
    for sample in samples:
        sid = sample.get("sourceID")
        if not sid or not str(sample.get("stage", "")).startswith("asleep"):
            continue
        try:
            t0 = datetime.fromisoformat(sample["startDate"].replace("Z", "+00:00"))
            t1 = datetime.fromisoformat(sample["endDate"].replace("Z", "+00:00"))
        except (KeyError, TypeError, ValueError, AttributeError):
            continue
        label = _source_label(str(sid))
        seconds[label] = seconds.get(label, 0.0) + max((t1 - t0).total_seconds(), 0.0)
    if not seconds:
        if provenance == "motionInferred":
            return "motion-inferred"
        return None
    return max(seconds.items(), key=lambda kv: kv[1])[0]


def _source_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every recording source in these rows: its first and last night, and how many.

    Different sources score sleep with different algorithms — on this data a
    Huawei band read REM at ~15% of sleep and an Apple Watch at ~27% on the same
    body — so a trend that crosses from one to another is partly a change of
    instrument. Grouped per source rather than as consecutive runs, because an
    app that writes the odd night between a Watch's (OtterLife did, 27 times)
    would otherwise shatter the list into dozens of one-night runs.
    """

    by_source: dict[str, dict[str, Any]] = {}
    for row in rows:
        src = row.get("source")
        day = row.get("local_date")
        if src is None or day is None or row.get("is_in_bed_only"):
            continue
        entry = by_source.setdefault(src, {"source": src, "first_day": day, "last_day": day, "nights": 0})
        entry["first_day"] = min(entry["first_day"], day)
        entry["last_day"] = max(entry["last_day"], day)
        entry["nights"] += 1
    return sorted(by_source.values(), key=lambda e: e["first_day"])


def _parse_day_range(since: str | None, until: str | None) -> tuple[str | None, str | None, dict[str, Any] | None]:
    """Normalise a `since` / `until` pair, or return the error to hand back.

    Days are compared as strings, so "2026-8-1" would silently select the wrong
    rows rather than fail — it is normalised or refused here, never passed on.
    """

    out: list[str | None] = []
    for name, value in (("since", since), ("until", until)):
        if value is None or value == "":
            out.append(None)
            continue
        try:
            out.append(date.fromisoformat(str(value)).isoformat())
        except ValueError:
            return None, None, {
                "error": f"invalid_{name}", "requested": value,
                "message": f'Pass `{name}` as "YYYY-MM-DD", e.g. "2026-08-01".',
            }
    if out[0] and out[1] and out[0] > out[1]:
        return None, None, {"error": "invalid_range", "since": out[0], "until": out[1],
                            "message": "`since` is after `until`."}
    return out[0], out[1], None


def _parse_period(value: str, name: str) -> tuple[str | None, str | None, dict[str, Any] | None]:
    """"YYYY-MM-DD..YYYY-MM-DD" (either end may be empty) → (since, until, error)."""

    if ".." not in value:
        return None, None, {"error": f"invalid_{name}", "requested": value,
                            "message": f'Pass `{name}` as "YYYY-MM-DD..YYYY-MM-DD".'}
    left, right = value.split("..", 1)
    return _parse_day_range(left.strip() or None, right.strip() or None)


def _apply_range_coverage(result: dict[str, Any], since: str | None, until: str | None) -> None:
    """After a calendar-window read: say which window, and fix `more_available`.

    The window, not a row count, bounded this read — so "older days exist" is
    whether history starts before the window's first day, and `window_satisfied`
    (which answers "did I get the count I asked for") has no count to answer.
    """

    if not (since or until):
        return
    result["range"] = {"since": since, "until": until}
    coverage = result.get("coverage")
    if isinstance(coverage, dict):
        first = coverage.get("first_day") or since
        oldest = coverage.get("oldest_available")
        coverage["more_available"] = bool(oldest and first and oldest < first)
        coverage["window_satisfied"] = None
        coverage["requested"] = None
        coverage["requested_unit"] = "date range"


def _hoist_common_exclusions(metrics: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Move exclusions shared by every listing series to one top-level list."""

    lists: list[list[dict[str, str]]] = [m["excluded_days"] for m in metrics if m.get("excluded_days")]
    if len(lists) < 2:
        return []
    keyed = [{(e["day"], e["reason"]) for e in lst} for lst in lists]
    shared = set.intersection(*keyed)
    if not shared:
        return []
    for m in metrics:
        if m.get("excluded_days"):
            rest = [e for e in m["excluded_days"] if (e["day"], e["reason"]) not in shared]
            if rest:
                m["excluded_days"] = rest
            else:
                m.pop("excluded_days")
    return [{"day": d, "reason": r} for d, r in sorted(shared, reverse=True)]


def _attach_sources(result: dict[str, Any], points: dict[str, float], raw: dict[str, Any]) -> None:
    """Add `sources` (+ note) when the days behind `points` came from >1 source."""

    rows = raw.get("nights") if isinstance(raw, dict) else None
    if not isinstance(rows, list) or not rows or "source" not in rows[0]:
        return
    summary = _source_summary([r for r in rows if r.get("local_date") in points])
    if len(summary) > 1:
        result["sources"] = summary
        result["source_note"] = SLEEP_SOURCE_NOTE


#: Rides next to `sources` whenever more than one is present.
SLEEP_SOURCE_NOTE = (
    "More than one device or app recorded these nights. Sources score sleep with "
    "different algorithms (stage shares especially), so compare within one source, "
    "and when a trend crosses from one to another say so before concluding the "
    "person changed rather than the instrument."
)


def _sessions_overlap(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """True when two sleep sessions share any time (local `YYYY-MM-DDTHH:MM`)."""

    try:
        a0, a1 = datetime.fromisoformat(a["bedtime"]), datetime.fromisoformat(a["wake_time"])
        b0, b1 = datetime.fromisoformat(b["bedtime"]), datetime.fromisoformat(b["wake_time"])
    except (KeyError, TypeError, ValueError):
        return True  # cannot tell → treat as the same sleep; never invent a second one
    return a0 < b1 and b0 < a1


# A main sleep that began at or after 08:00 ON THE DAY IT ENDED never reached
# into the night: a daytime nap, or an evening one that ended before midnight.
# Until 2026-10-03 the window closed at 18:00, so an 18:28-20:03 nap that was
# the only sleep recorded that day became the "bedtime" of 2025-11-13 and pulled
# that month's average to ~02:52, the latest of the year (release gate G1). No
# upper bound is needed: a real night that starts in the evening ends on the
# NEXT day, so its bedtime is negative here.
_DAYTIME_START_MIN = 8 * 60


def _session_zone(session: dict[str, Any]) -> tzinfo | None:
    """The zone the phone stamped on a sleep session, or None for "use this machine's".

    🔴 Review V4 (2026-10-03): every clock time and calendar day of a night was
    converted with THIS machine's zone. On an MCP server running in UTC — a VPS,
    the documented deployment — a UTC+8 night of 23:30→07:30 became 15:30→23:30
    on one day, was flagged `daytime_main_sleep`, and bedtime, wake and midpoint
    went empty for every night. iOS has stamped each session with the assembling
    phone's zone since 1.2.5 (Invariant 70 (first-observation-wins-for-a-zone));
    it was never read.

    The identifier is preferred for clock times (it knows DST inside the night);
    the stored offset is the fallback, and older sessions carry neither. Never
    rendered: the stamp describes the phone that assembled the night, not where
    anyone slept.
    """
    identifier = session.get("timeZoneIdentifier")
    if isinstance(identifier, str) and identifier:
        try:
            return ZoneInfo(identifier)
        except (ZoneInfoNotFoundError, ValueError):
            pass
    offset = session.get("timeZoneUTCOffsetSeconds")
    if isinstance(offset, int) and not isinstance(offset, bool) and abs(offset) <= 18 * 3600:
        return timezone(timedelta(seconds=offset))
    return None


def _session_local_date(session: dict[str, Any], session_date_utc: datetime) -> str:
    """The night's calendar day. `sessionDate` IS the phone's local midnight, so
    with the stamped offset `sessionDate + offset` read as UTC is that day exactly
    (the iOS doc on `timeZoneUTCOffsetSeconds`: consumers bucketing by day use the
    number, not the identifier). Without a stamp: this machine's zone, as before."""
    offset = session.get("timeZoneUTCOffsetSeconds")
    if isinstance(offset, int) and not isinstance(offset, bool) and abs(offset) <= 18 * 3600:
        return (session_date_utc.astimezone(timezone.utc) + timedelta(seconds=offset)).strftime("%Y-%m-%d")
    return session_date_utc.astimezone().strftime("%Y-%m-%d")


def _clock_minutes(local_iso: str, midnight: datetime) -> int | None:
    """Minutes from `midnight` to a local `YYYY-MM-DDTHH:MM` — negative before it.

    Clock times cannot be averaged as clock times: 23:50 and 00:10 average to
    noon. Measured against the midnight that opens the WAKE day, a bedtime of
    23:50 is -10 and 01:19 is 79, and their mean (34, i.e. 00:34) is right.
    """

    try:
        return int((datetime.fromisoformat(local_iso) - midnight).total_seconds() // 60)
    except (ValueError, TypeError):
        return None


def _clock_label(minutes: int | float | None) -> str | None:
    if minutes is None:
        return None
    total = int(round(minutes)) % 1440
    return f"{total // 60:02d}:{total % 60:02d}"


def _night_clock(night: dict[str, Any]) -> tuple[int | None, int | None, int | None]:
    """(bedtime, wake, midpoint) in minutes from the midnight that opens the wake day."""

    wake_raw = night.get("wake_time") or ""
    try:
        midnight = datetime.fromisoformat(wake_raw).replace(hour=0, minute=0, second=0, microsecond=0)
    except (ValueError, TypeError):
        return None, None, None
    bed_min = _clock_minutes(night.get("bedtime") or "", midnight)
    wake_min = _clock_minutes(wake_raw, midnight)
    mid_min = (bed_min + wake_min) // 2 if bed_min is not None and wake_min is not None else None
    return bed_min, wake_min, mid_min


def sleep_night_row(night: dict[str, Any]) -> dict[str, Any]:
    """One night as the flat record the sleep series and `get_sleep_nights` read.

    `None` means "not measured", never zero: stage minutes are None on a night
    without stage detail (iPhone-only), and every sleep field is None on an
    in-bed-only night (the Watch was not worn). The two flags travel with the
    row so `SeriesSpec.exclude_when` can name why a night was left out.
    """

    stages = night.get("stage_minutes") or {}
    in_bed_only = bool(night.get("is_in_bed_only"))
    staged = bool(night.get("has_stage_detail")) and not in_bed_only
    asleep = None if in_bed_only else night.get("total_sleep_minutes")

    bed_min, wake_min, mid_min = _night_clock(night)

    def stage(name: str) -> int | None:
        return int(stages.get(name, 0)) if staged else None

    deep, rem = stage("asleepDeep"), stage("asleepREM")

    def share(part: int | None) -> float | None:
        if part is None or not asleep:
            return None
        return round(100.0 * part / float(asleep), 1)

    return {
        "local_date": night.get("local_date"),
        "weekday": _weekday(night.get("local_date")),
        "bedtime": _clock_label(bed_min),
        "wake_time": _clock_label(wake_min),
        "bedtime_minutes": bed_min,
        "wake_minutes": wake_min,
        "midpoint_minutes": mid_min,
        "asleep_minutes": asleep,
        "deep_minutes": deep,
        "rem_minutes": rem,
        "core_minutes": stage("asleepCore"),
        "awake_minutes": stage("awake"),
        "deep_percent": share(deep),
        "rem_percent": share(rem),
        "awakenings": night.get("awakenings") if staged else None,
        "longest_sleep_bout_minutes": night.get("longest_sleep_bout_minutes") if staged else None,
        "source": night.get("source"),
        "sleep_hr_samples": night.get("sleep_hr_samples"),
        "sleep_hr_mean": None if in_bed_only else night.get("sleep_hr_mean"),
        "sleep_hr_min": None if in_bed_only else night.get("sleep_hr_min"),
        "sleep_rr_mean": None if in_bed_only else night.get("sleep_rr_mean"),
        "other_sleep_minutes": night.get("other_sleep_minutes") or 0,
        "sleep_segments": night.get("sleep_segments"),
        # All measured sleep that day, main + naps + a split night's other half.
        # None only when nothing at all was measured.
        "total_sleep_24h_minutes": (
            (asleep or 0) + (night.get("other_sleep_minutes") or 0)
            if (asleep is not None or night.get("other_sleep_minutes")) else None
        ),
        "other_sleep": night.get("other_sleep") or [],
        "in_bed_minutes": night.get("in_bed_minutes"),
        "has_stage_detail": staged,
        "is_in_bed_only": in_bed_only,
        "no_stage_detail": not staged,
        # The day's MAIN sleep began at or after 08:00 on the day it ended — a
        # nap (daytime or evening) that was the only sleep recorded that day.
        # Real, so it stays in the table and in `sleep_minutes`; left out of the
        # clock series, where one 12:34 "bedtime" moved a whole month's average
        # past 03:30 (2026-09-24, 6 of 323 real nights), and an 18:28 one did
        # the same to November 2025 (see `_DAYTIME_START_MIN`).
        "daytime_main_sleep": bed_min is not None and bed_min >= _DAYTIME_START_MIN,
        # Guessed from the phone lying still, not measured by a wearable. Kept in
        # the table; left out of the sleep series, because stillness is not
        # sleep and the guess runs long (real data: 474 min on average against
        # 434 for the Watch nights around them).
        "motion_inferred": night.get("source") == "motion-inferred",
        "owner_user_id": night.get("owner_user_id"),
    }


def _weekday(day: Any) -> str | None:
    try:
        return datetime.strptime(str(day), "%Y-%m-%d").strftime("%a")
    except (ValueError, TypeError):
        return None


def summarize_menstrual_cycle(
    days: list[MenstrualDay],
    wrist_readings: list[tuple[datetime, float]] | None = None,
) -> dict[str, Any]:
    """Recent cycle samples plus a robust next-period prediction.

    Prediction = last cycle start + typical cycle length, where "typical" is the
    MEDIAN gap between consecutive cycle starts over the most recent
    _CYCLE_STATISTICS_WINDOW gaps — median (not mean) so a single missed logging
    month (one 56-day gap in a 28-day rhythm) cannot drag the prediction.
    `cycle_length_variability_days` is the median absolute deviation over the
    same window (needs >= _MIN_GAPS_FOR_VARIABILITY gaps). Rounding is
    int(x + 0.5) to match Swift exactly. With fewer than two distinct cycle
    starts there is no gap, so the prediction is reported as unavailable rather
    than guessed.

    `wrist_readings` (the SAME person's nightly wrist-temp deltas — the caller
    is responsible for owner matching) upgrades the prediction: a detected
    biphasic shift re-anchors it to `ovulation + luteal 14`, exactly like the
    iOS summary calibration, so the app and the AI keep agreeing on the date.
    """

    ordered = sorted(days, key=lambda d: d.day_start_date, reverse=True)
    payload: dict[str, Any] = {
        "sensitive": True,
        "days": [day.to_dict() for day in ordered],
        "day_count": len(ordered),
        "average_cycle_length_days": None,
        "cycle_length_variability_days": None,
        "last_cycle_start_date": None,
        "predicted_next_period_start_date": None,
        "detected_ovulation_date": None,
        "prediction_calibrated_by_ovulation": False,
        "prediction_note": None,
    }

    starts = _cycle_starts(days)
    if len(starts) < 2:
        _LOG.info("menstrual prediction skipped: need >=2 cycle starts, have %d", len(starts))
        payload["last_cycle_start_date"] = _local_calendar_day(starts[-1]).isoformat() if starts else None
        payload["prediction_note"] = (
            "Insufficient history to predict the next period "
            f"(need at least two recorded cycle starts, have {len(starts)})."
        )
        return payload

    gaps = [(starts[i + 1] - starts[i]).days for i in range(len(starts) - 1)]
    recent_gaps = gaps[-_CYCLE_STATISTICS_WINDOW:]
    typical = _median([float(g) for g in recent_gaps])
    if typical is None:  # unreachable: >=2 starts guarantee >=1 gap
        return payload
    rounded_length = int(typical + 0.5)
    if len(recent_gaps) >= _MIN_GAPS_FOR_VARIABILITY:
        mad = _median([abs(float(g) - typical) for g in recent_gaps])
        if mad is not None:
            payload["cycle_length_variability_days"] = int(mad + 0.5)
    last_start = starts[-1]
    predicted = last_start + timedelta(days=rounded_length)
    payload["average_cycle_length_days"] = rounded_length
    # Local calendar days, like the ovulation date below. The starts are UTC
    # instants of local midnight, and `.isoformat()` of one read as the PREVIOUS
    # day east of UTC ("2026-09-29T16:00:00+00:00" for a 09-30 start at UTC+8 —
    # pre-release review, 2026-10-02).
    payload["last_cycle_start_date"] = _local_calendar_day(last_start).isoformat()
    payload["predicted_next_period_start_date"] = _local_calendar_day(predicted).isoformat()

    if wrist_readings:
        ovulation = detect_ovulation_from_wrist_temp(wrist_readings, last_start)
        if ovulation is not None:
            calibrated = ovulation + timedelta(days=_LUTEAL_PHASE_DAYS)
            payload["detected_ovulation_date"] = ovulation.isoformat()
            payload["prediction_calibrated_by_ovulation"] = True
            payload["predicted_next_period_start_date"] = calibrated.isoformat()
            payload["prediction_note"] = (
                "Prediction anchored to this cycle's measured ovulation "
                "(wrist-temperature biphasic shift) + a 14-day luteal phase."
            )
    return payload


def summarize_symptoms(
    days: list[SymptomDay], entries: list[SymptomEntry] | None = None
) -> dict[str, Any]:
    """Recent symptom days grouped by data owner, plus per-owner type counts.

    Both partners can track symptoms, so days are grouped by `owner_user_id`
    (None → "unknown", e.g. blobs fetched before the edge function returned
    ownership). Within an owner, days dedup by day_id (newest day_start_date
    wins — matching the one-blob-per-day upsert) and sort newest-first.
    `symptom_counts` counts logged days per symptom type, skipping explicit
    "notPresent" entries so "logged as absent" doesn't inflate the tally.

    `entries` are the self-reported episodes (GitHub #3). They sit beside the
    HealthKit `days` under `reported`, never inside them: `days` / `symptom_counts`
    keep the exact shape and meaning every earlier caller relies on (logged DAYS
    per type), while `reported_counts` counts EPISODES per type — two different
    units, so they are two different fields. Tombstones never reach here.
    """

    owners: dict[str, dict[str, SymptomDay]] = {}
    for day in days:
        owner_key = day.owner_user_id or "unknown"
        by_id = owners.setdefault(owner_key, {})
        existing = by_id.get(day.day_id)
        if existing is None or day.day_start_date >= existing.day_start_date:
            by_id[day.day_id] = day

    reported: dict[str, dict[str, SymptomEntry]] = {}
    for entry in entries or []:
        owner_key = entry.owner_user_id or "unknown"
        by_entry = reported.setdefault(owner_key, {})
        prior = by_entry.get(entry.entry_id)
        if prior is None or (entry.updated_at or "") >= (prior.updated_at or ""):
            by_entry[entry.entry_id] = entry
        owners.setdefault(owner_key, {})

    owner_summaries: list[dict[str, Any]] = []
    for owner_key in sorted(owners):
        ordered = sorted(owners[owner_key].values(), key=lambda d: d.day_start_date, reverse=True)
        type_counts: dict[str, int] = {}
        for day in ordered:
            for sample in day.samples:
                if sample.severity == "notPresent":
                    continue
                type_counts[sample.symptom_type] = type_counts.get(sample.symptom_type, 0) + 1
        own_entries = sorted(
            reported.get(owner_key, {}).values(), key=lambda e: e.sort_key(), reverse=True
        )
        reported_counts: dict[str, int] = {}
        for entry in own_entries:
            reported_counts[entry.symptom_type] = reported_counts.get(entry.symptom_type, 0) + 1
        owner_summaries.append(
            {
                "owner_user_id": None if owner_key == "unknown" else owner_key,
                "day_count": len(ordered),
                "symptom_counts": dict(sorted(type_counts.items(), key=lambda kv: -kv[1])),
                "days": [day.to_dict() for day in ordered],
                "reported_count": len(own_entries),
                "reported_counts": dict(sorted(reported_counts.items(), key=lambda kv: -kv[1])),
                "reported": [entry.to_dict() for entry in own_entries],
            }
        )

    return {
        "sensitive": True,
        "owners": owner_summaries,
        "owner_count": len(owner_summaries),
        "total_day_count": sum(o["day_count"] for o in owner_summaries),
        "total_reported_count": sum(o["reported_count"] for o in owner_summaries),
    }


def _demo_write_refusal(tool: str) -> dict[str, Any]:
    """The answer every `log_*` gives while demo mode is on.

    RETURNED, never raised. An exception out of a tool becomes a ToolError,
    which a client renders as a failure and — more to the point — never passes
    through the watermarking wrapper in `mcp_server`, so the one fact the caller
    most needs (this was a demo, nothing is broken) would be the one fact
    stripped off.

    It names demo mode in the first clause on purpose. A refusal that only said
    "not bound" would send an agent to re-pair a machine that is working exactly
    as configured — and re-pairing is the most destructive thing an agent can do
    here (Invariant 54).

    Writes are the one thing demo mode cannot honestly fake: sealing a blob
    needs a real key and a real edge function, so a write tool reporting success
    would be lying about the only kind of call that changes something.
    """

    from vaultbeat_mcp_local.demo import DEMO_BANNER, DEMO_ENV

    return {
        # Banner first — the sentence acts on a reader who does not already know
        # to look for a boolean. Same ordering as `_watermark_demo`.
        "demo_warning": DEMO_BANNER,
        "demo_mode": True,
        "ok": False,
        "error": "demo_mode_is_read_only",
        "tool": tool,
        "detail": (
            f"Demo mode is on ({DEMO_ENV} is set), so nothing was written and nothing "
            f"changed. This is NOT a fault and the binding is not broken — demo mode "
            f"serves synthetic records so the read tools can be exercised without an "
            f"account, and it has no account to write to. Do not re-pair or run "
            f"diagnostics. To log real data, unset {DEMO_ENV} and pair this machine "
            f"with the Vaultbeat iOS app ({CONNECT_SERVER})."
        ),
    }


class VaultbeatLocalService:
    def __init__(
        self,
        store: ConfigStore,
        cloud_client: CloudClientProtocol | None = None,
        cache: LocalRecordCache | None = None,
        *,
        demo: bool | None = None,
    ):
        self.store = store
        self._cloud_client = cloud_client
        self._cache = cache
        # Demo mode is resolved ONCE, here, and every branch below reads
        # `self._demo` rather than the environment.
        #
        # `mcp_server.run_mcp_server` already froze its own copy at registration
        # — it has to, because a tool's description and the server's displayed
        # name are captured by the SDK at that moment and cannot be changed
        # afterwards. So the only reachable shape is "everything frozen"; when
        # this class re-read the environment per call, the two halves could
        # disagree, and the half that lies is the one that leaves the machine:
        # flipping VAULTBEAT_DEMO on after startup served SYNTHETIC RECORDS
        # THROUGH AN UNWATERMARKED WRAPPER (verified 2026-08-20 — owner id
        # demo0001-…, no banner, no description prefix), while `status` and
        # `doctor` went on correctly reporting demo_mode. The surface that gets
        # copied, quoted and pasted was the one telling the lie.
        #
        # Keyword-only, and defaulting to "ask the environment": the several
        # dozen existing constructions pass `store` and `cloud_client`
        # positionally and are unaffected, while a test can put one demo and one
        # real service side by side in one process — which a module-level latch
        # could not do, and which is why one was rejected.
        from vaultbeat_mcp_local.demo import demo_enabled

        self._demo = demo_enabled() if demo is None else demo
        # What the ENVIRONMENT said when this object was built — which is a
        # different question from which mode it is in, and conflating the two
        # broke the constructor argument the first time it was tried.
        #
        # The tripwire below asks "did the environment change under me?", so it
        # must compare against the environment's own past value. Comparing
        # against `self._demo` instead makes an explicit `demo=True` illegal
        # whenever the variable happens to be unset — i.e. it would forbid
        # exactly the two-services-in-one-process case the argument exists for,
        # while ALSO disarming itself in production, where `run_mcp_server`
        # passes the flag explicitly.
        self._demo_env_at_start = demo_enabled()
        # One cloud fetch per kind at a time; see `sync_decrypted_records`.
        self._inflight_syncs: dict[str | None, asyncio.Future[tuple[list[DecryptedRecord], list[str]]]] = {}
        # The newest fetch of each kind; only it may write the cache.
        self._sync_generation: dict[str | None, int] = {}
        # Trial-deadline snapshot from the config of the LAST successful
        # require_bound() in this process — a zero-I/O stash so the per-tool
        # `access_note` annotation (vb-016) never adds a config/Keychain read
        # of its own on top of the one every read already performs. None until
        # a bound call has run; never set in demo mode (demo returns before
        # require_bound).
        self._last_bound_trial_ends_at: str | None = None

    @property
    def cache(self) -> LocalRecordCache:
        if self._cache is None:
            self._cache = LocalRecordCache(self.store.path.parent / "cache")
        return self._cache

    def _require_bound_config(self) -> LocalServerConfig:
        """`store.require_bound()` plus the trial-snapshot stash.

        Every read and write path goes through here rather than calling the
        store directly, so `access_note_if_expiring` can answer from memory —
        the funnel discipline (Invariant 58's shape): a stash set at four of
        five call sites is a note that silently never fires on the fifth.
        """
        config = self.store.require_bound()
        self._last_bound_trial_ends_at = config.trial_ends_at
        return config

    # ── Trial-access snapshot (vb-016) ───────────────────────────────────────
    #
    # Everything here reads the trial deadline RECORDED AT PAIRING — a snapshot,
    # never a live entitlement. The one rule: these sentences may only describe
    # what was observed at bind time and must say so, because a Pro purchase
    # made in the iOS app afterwards is invisible to this machine (the cloud
    # enforces the real rule on every request; iOS asks `mcp_access_verdict`
    # for its own display). Saying less beats guessing.

    @staticmethod
    def _trial_deadline(trial_ends_at: str) -> datetime | None:
        try:
            parsed = _parse_iso8601(trial_ends_at)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed

    @classmethod
    def access_snapshot(
        cls, config: LocalServerConfig, *, now: datetime | None = None
    ) -> dict[str, Any] | None:
        """The `access` block for status/doctor, or None when there is nothing
        honest to say.

        None on an unbound config, and None when no deadline was recorded at
        pairing — mirroring the iOS Settings row, which shows nothing for a
        grandfathered/lifetime account rather than a reassuring sentence nobody
        needed. Absence of a deadline is NOT knowledge of unlimited access
        (it also covers an edge too old to report one), so no block is the
        only claim-free rendering of it.
        """
        if not config.is_bound or not config.trial_ends_at:
            return None
        moment = now or datetime.now(timezone.utc)
        deadline = cls._trial_deadline(config.trial_ends_at)
        block: dict[str, Any] = {
            "phase_at_pairing": "trial",
            # A config written before 2026-10-03 holds whatever the server sent;
            # only a timestamp-shaped value is repeated (review V2).
            "trial_ends_at": server_token(config.trial_ends_at, _ISO_TIMESTAMP),
            "recorded_at_pairing": config.bound_at,
        }
        if deadline is None:
            # Unparseable date — the caveat, never the raw string (it came from
            # the server, Anti-pattern 23), never a crash, never a day count.
            block["note"] = (
                "A trial deadline was recorded when this server paired but its "
                "date could not be parsed. The cloud "
                "enforces the real entitlement on every request; a purchase made "
                "in the iOS app since pairing is not reflected in this snapshot."
            )
            return block
        if deadline >= moment:
            days_left = max(0, (deadline.date() - moment.date()).days)
            block["days_left"] = days_left
            when = "today" if days_left == 0 else f"in {days_left} day(s)"
            block["note"] = (
                "A 3-day trial of the AI interface was running when this "
                "server paired; "
                f"it ends {deadline.date().isoformat()} ({when}). This is a "
                "snapshot from pairing time — a Pro purchase made in the iOS app "
                "since then is not visible here, and the cloud enforces the real "
                "entitlement on every request. If it is about to expire, tell the "
                "user now rather than letting the first notice be a refusal; "
                "expiry deletes nothing, and Pro in the app (Settings → "
                "Membership) restores access with no re-pairing."
            )
            return block
        block["note"] = (
            "The trial deadline recorded when this server paired "
            f"({deadline.date().isoformat()}) has passed. That snapshot does not "
            "update: if reads are working, this account has active access (for "
            "example Pro was purchased, which this machine cannot see); if they "
            "fail with a trial-expired message, that message carries the current "
            "state and the fix. Nothing is deleted either way."
        )
        return block

    def access_note_if_expiring(self, *, now: datetime | None = None) -> str | None:
        """One sentence for read/write tool results when the pairing-time trial
        deadline falls within the NEXT 24 hours — else None.

        Reads only the in-process stash (no config/Keychain I/O), so it costs
        nothing on the paths that run it after every tool call. Deliberately
        silent once the deadline has passed: the server's own refusal is the
        authoritative message then, and a stale "your trial ended" beside a
        working read would be wrong for a user who purchased Pro after pairing.
        """
        if self._demo or not self._last_bound_trial_ends_at:
            return None
        deadline = self._trial_deadline(self._last_bound_trial_ends_at)
        if deadline is None:
            return None
        moment = now or datetime.now(timezone.utc)
        if not (moment <= deadline <= moment + timedelta(hours=24)):
            return None
        return (
            "Heads-up worth passing to the user: the 3-day AI-interface trial "
            "recorded when this Vaultbeat server paired ends within the next "
            f"day ({self._last_bound_trial_ends_at}). After that, reads return "
            "a trial-expired refusal until Pro is active in the Vaultbeat iOS "
            "app (Settings → Membership) — nothing gets deleted. This is a "
            "pairing-time snapshot: if Pro was already purchased since, ignore "
            "this note."
        )

    def start_binding(
        self,
        *,
        server_name: str = "Local AI Server",
        api_base_url: str = DEFAULT_API_BASE_URL,
    ) -> BindingSession:
        config = self.store.ensure_initialized(server_name=server_name, api_base_url=api_base_url)
        poll_id = secrets.token_urlsafe(24)
        # Starting a binding session must NOT touch the existing credentials.
        # It used to null out server_id/server_token/bound_at/last_sync_at and
        # clear the cache right here, before anyone had scanned anything — so a
        # session nobody completed (the common case: the phone is in the other
        # room, the QR expires, the link blips) destroyed a working binding and
        # said nothing about it. The cloud's token_hash was never touched, so
        # the credential was only ever destroyed on THIS side, by the very
        # command a user runs to repair things. That is Invariant 54 (a), and it
        # is how fino ended up holding an all-NULL config while the server still
        # had four live rows for it.
        #
        # Replacement now happens in poll_once, at the one moment a REPLACEMENT
        # actually exists. Until then the old binding keeps working: reads
        # served, agent writes accepted, nothing lost if the scan never comes.
        config = self.store.update(
            server_name=server_name.strip() or config.server_name,
            api_base_url=api_base_url.rstrip("/") or config.api_base_url,
            poll_id=poll_id,
        )
        qr_payload = {
            "pollID": poll_id,
            "publicKeyBase64": config.public_key_base64,
            "serverName": config.server_name,
        }
        return BindingSession(
            poll_id=poll_id,
            qr_payload=qr_payload,
            qr_payload_json=json.dumps(qr_payload, separators=(",", ":"), sort_keys=True),
            config=config,
        )

    async def poll_once(self) -> PollBindingResult:
        config = self.store.load()
        if not config or not config.poll_id:
            raise RuntimeError(
                "No active binding session; run `vaultbeat-apple-health bind`, which "
                "starts one and waits for the scan."
            )

        result = await self._client(config).poll_binding(config.poll_id)
        if result.status == "bound":
            if not result.server_id or not result.server_token:
                raise RuntimeError("Cloud returned bound without server credentials")
            # This is the swap point: the old credentials are only overwritten
            # now that a replacement is in hand (see start_binding's comment).
            landed_on_new_identity = config.server_id != result.server_id
            self.store.update(
                server_id=result.server_id,
                server_token=result.server_token,
                owner_user_id=result.owner_user_id,
                owner_public_key_base64=result.owner_public_key_base64,
                owner_device_id=result.owner_device_id,
                # Bind-time SNAPSHOT of the trial deadline, overwritten on every
                # successful pairing — including with None, which means "the
                # cloud reported no deadline at THIS pairing" (grandfathered,
                # currently paid, or an older edge). Stored so status/doctor can
                # warn about an approaching expiry instead of the first notice
                # being a mid-conversation 403 (vb-016). Never a gate: the
                # server enforces the real entitlement on every request.
                trial_ends_at=result.trial_ends_at,
                poll_id=None,
                bound_at=now_iso(),
                # A different identity's sync history says nothing about this
                # one; the same identity's still does.
                **({"last_sync_at": None} if landed_on_new_identity else {}),
            )
            if landed_on_new_identity:
                # Cached plaintext is keyed by server_id, but a stale file for
                # an identity this machine no longer holds is plaintext health
                # data with no reader — clear it. Re-binding to the SAME id
                # (what the planned upsert makes the normal outcome) keeps its
                # cache: same key, same private key, same records.
                self.cache.clear()
        return result

    #: Consecutive failed polls tolerated before giving up. Five ≈ 35s of an
    #: uninterrupted outage — long enough to ride out the link's measured 2–3%
    #: blips, short enough that a genuinely dead uplink does not make the user
    #: watch a spinner for the full five minutes before being told.
    POLL_CONSECUTIVE_FAILURE_LIMIT = 5

    async def poll_until_bound(self, *, timeout_sec: int = 300, interval_sec: float = 7.0) -> PollBindingResult:
        """Poll until bound, surviving the blips this link actually has.

        A single failed poll used to end the whole bind: the exception went
        straight up, the user re-ran `bind`, and that produced a NEW pollID —
        invalidating the QR they had just scanned. On a loop that runs 5–10
        minutes at one request every 7 seconds over a link measured at ~97.3%
        (7-day `tw`), hitting at least one blip is close to certain, so the
        common path was failing for a reason that has nothing to do with
        binding.

        ⚠️ Note what this deliberately does NOT do: retry the request itself.
        `/mcp-poll-binding` is consume-on-read and lives in the client's
        NON_IDEMPOTENT_PATHS for that reason — a replay after a lost response
        would destroy the state the first attempt consumed. This retries the
        NEXT poll on the next tick, which is a different thing: if the request
        never reached the server the pending row is still there to be claimed,
        and if it did reach the server the token is gone either way and no
        amount of retrying inside one tick would have helped. (That case is now
        self-healing anyway — since the bind upsert, re-scanning lands on the
        same row instead of minting an orphan.)
        """

        import httpx

        deadline = asyncio.get_running_loop().time() + timeout_sec
        consecutive_failures = 0
        last_error: Exception | None = None
        last_result: PollBindingResult | None = None

        while True:
            try:
                result = await self.poll_once()
            except (httpx.TransportError, VaultbeatCloudError) as error:
                consecutive_failures += 1
                last_error = error
                _LOG.warning(
                    "Poll attempt failed (%d/%d consecutive): %s: %s",
                    consecutive_failures,
                    self.POLL_CONSECUTIVE_FAILURE_LIMIT,
                    type(error).__name__,
                    error,
                )
                if consecutive_failures >= self.POLL_CONSECUTIVE_FAILURE_LIMIT:
                    raise
            else:
                consecutive_failures = 0
                last_error = None
                last_result = result
                if result.status == "bound":
                    return result

            if asyncio.get_running_loop().time() >= deadline:
                if last_result is not None:
                    return last_result
                # Every single poll failed, so there is no status to report.
                # Returning a synthetic "pending" here would tell the user to
                # keep waiting for a scan when the truth is that this machine
                # never reached the server.
                raise last_error or RuntimeError(
                    "Binding timed out without a single successful poll"
                )
            await asyncio.sleep(interval_sec)

    async def sync_decrypted_records(
        self,
        *,
        limit: int | None = None,
        metric_type: str | None = None,
        fresh: bool = False,
    ) -> tuple[list[DecryptedRecord], list[str]]:
        """`_sync_decrypted_records`, with concurrent reads of one kind sharing one fetch.

        The catalog and `get_metric` read their kinds concurrently (cold, one
        kind is a multi-megabyte download — 14 MB of sleep in 13 s on one
        account — and they used to run back to back: 170 s for the catalog with
        the cache off, past a client's 60 s timeout). Concurrency must not
        download a kind twice, and it would: total energy reads basal energy
        itself. So a second read of a kind already being fetched waits for that
        fetch instead of starting its own. `fresh=True` keeps its own round trip.

        🔴 A fresh read also becomes THE fetch of its kind, and only the newest
        fetch of a kind may write the cache (review R6 on #9, 2026-10-03). Every
        write tool re-reads `fresh=True` right after writing; until then a plain
        read already in flight kept its slot, so a read arriving after the write
        joined the download that began before it, and when that download landed
        its `cache.save` overwrote the post-write cache for the whole TTL — the
        agent's own write vanished from its next ten minutes of reads. Each
        fetch now takes a generation of its kind, and `_sync_decrypted_records`
        saves only while its generation is still the newest. Waiters of the
        older fetch still get what it read: they asked before the write.
        """
        task = None if fresh else self._inflight_syncs.get(metric_type)
        if task is None:
            generation = self._sync_generation.get(metric_type, 0) + 1
            self._sync_generation[metric_type] = generation
            task = asyncio.ensure_future(
                self._sync_decrypted_records(
                    limit=None, metric_type=metric_type, fresh=fresh, generation=generation
                )
            )
            self._inflight_syncs[metric_type] = task
            key = metric_type

            def forget(done: asyncio.Future[Any]) -> None:
                if self._inflight_syncs.get(key) is done:
                    self._inflight_syncs.pop(key, None)

            task.add_done_callback(forget)
        records, errors = await asyncio.shield(task)
        return (list(records[:limit]) if limit is not None else list(records)), list(errors)

    async def _sync_decrypted_records(
        self,
        *,
        limit: int | None = None,
        metric_type: str | None = None,
        fresh: bool = False,
        generation: int | None = None,
    ) -> tuple[list[DecryptedRecord], list[str]]:
        """Fetch + decrypt this server's records, cache-first.

        `metric_type` narrows the fetch server-side (older mcp-sync deployments
        ignore the parameter, so the local filter below stays authoritative —
        the parameter is an optimization, never a correctness dependency).
        Within the cache TTL a repeat query answers from local plaintext with
        ZERO network; `fresh=True` forces a cloud round trip. The cache always
        stores the FULL result set for its key — `limit` only trims the copy
        returned to the caller. `generation` is this fetch's place in its
        kind's order (`sync_decrypted_records`); a fetch overtaken by a newer
        one returns what it read but does not write the cache.
        """

        if metric_type is not None and metric_type not in KNOWN_METRIC_TYPES:
            # Fail fast locally: an unknown value would (a) 400 on the new edge,
            # and (b) poison a cache key — e.g. "all" maps to the same file as
            # the unfiltered set. Membership check beats both.
            raise ValueError(
                f"unknown metric_type {metric_type!r}; expected one of "
                f"{', '.join(sorted(KNOWN_METRIC_TYPES))}"
            )

        # ── Demo mode ────────────────────────────────────────────────────────
        #
        # THE choke point for demo data, and the reason there is no second
        # implementation of anything: every read tool reaches this method via
        # `_records_for_metric`, and `_capability_report` calls it directly, so
        # one branch covers all of them.
        #
        # Its position is three separate guarantees, none of them incidental:
        #  · BELOW the membership check, so a bad `metric_type` still raises in
        #    demo mode — the tool's real contract stays visible to whoever is
        #    evaluating it.
        #  · ABOVE `require_bound()`, so demo mode needs no config, no server
        #    token and no private key. `status` / `doctor` keep saying "not
        #    bound", which is the truth.
        #  · ABOVE `self.cache`, so synthetic records are structurally incapable
        #    of being written into the on-disk plaintext cache. A demo run cannot
        #    contaminate a real one by construction, not by care.
        #
        # Imported here rather than at module scope: `demo` imports this module
        # for `DecryptedRecord` / `KNOWN_METRIC_TYPES`, so a top-level import
        # would be a cycle.
        from vaultbeat_mcp_local.demo import DEMO_ENV, demo_enabled, demo_sync_result

        # Tripwire, not a second source of truth. `self._demo` is the decision;
        # this compares it against the environment purely to refuse rather than
        # mislead when the two have come apart. It sits at the DATA choke point
        # because that is where both failure directions land: demo data leaving
        # through a wrapper that no longer stamps it, and real records leaving
        # under a stamp that calls them synthetic — the second being the worse
        # one, since an agent told "SYNTHETIC" about a successful real write
        # concludes nothing was written and retries.
        #
        # Unreachable in normal operation (an MCP client cannot edit an already
        # exec'd subprocess's environment), so this costs one getenv per read
        # and its message names the variable, the frozen value and the fix.
        if demo_enabled() != self._demo_env_at_start:
            raise RuntimeError(
                f"{DEMO_ENV} changed after this server started "
                f"(it was {self._demo_env_at_start}, it is now {demo_enabled()}; this "
                f"server is running in demo_mode={self._demo}). "
                "Refusing to serve: this server's tool descriptions, displayed name and "
                "result watermarks were all fixed at startup and cannot follow the "
                "change, so continuing would either hand you synthetic records with no "
                "synthetic marker, or label your real records synthetic. "
                f"Restart the server with {DEMO_ENV} set the way you want it."
            )

        if self._demo:
            return demo_sync_result(metric_type=metric_type, limit=limit)

        config = self._require_bound_config()
        server_token = config.server_token
        server_id = config.server_id or ""
        if not server_token:
            # Unreachable in practice (is_bound requires server_token), kept as
            # defence in depth — with the same two-sided guidance as the real
            # refusal above it, so no path ever prints the old tool-only text.
            raise RuntimeError(
                f"This Vaultbeat MCP server has no usable pairing. {PAIRING_GUIDANCE}"
            )

        # This fetch's place among every fetch of its kind, in any process
        # sharing this pairing (`LocalRecordCache.save`, review V5).
        fetch_started = time.time()
        if not fresh:
            cached = self.cache.load(server_id=server_id, metric_type=metric_type)
            if cached is not None:
                cached_records, cached_errors = cached
                cached_errors = _recheck_error_lines(cached_errors)
                records = [DecryptedRecord.from_dict(row) for row in cached_records]
                if limit is not None:
                    records = records[:limit]
                return records, cached_errors

        # ── Catalog path: ask what changed before asking for anything ────────
        #
        # A full fetch of one kind measured 12,108,143 bytes on 2026-07-29; the
        # digest that decides whether it is needed measured 119. Three users had
        # spent 21.877 GB of a 5 GB monthly allowance re-downloading history that
        # had not changed, and the same queries put the instance into Unhealthy
        # for 2.5 hours that day.
        #
        # `fresh=True` still comes through here on purpose: it means "do not
        # trust age", not "re-download everything". The digest re-verifies
        # against the server, which is strictly stronger than a TTL, so honouring
        # it no longer has to cost 12 MB. (It reaches this code as every read
        # tool's `fresh` argument; the `--fresh` CLI flag it was named for went
        # with the data subcommands in 0.7.4.)
        #
        # Every branch degrades to the full fetch, never to an error: an older
        # edge deployment (no catalog mode), a cache with no stored digest, an
        # unparseable catalog — all fall through with `envelope_rows = None`.
        client = self._client(config)
        envelope_rows: list[dict[str, Any]] | None = None
        reusable_records: list[dict[str, Any]] = []
        reusable_errors: list[str] = []
        server_digest: dict[str, Any] | None = None
        catalog_xmins: dict[str, str] = {}

        # Capability probe, not defensive clutter: CloudClientProtocol declares
        # the catalog trio optional, so a client object that predates it (an
        # injected double, a pinned older transport) must degrade to the full
        # fetch rather than raise AttributeError mid-read.
        supports_catalog = all(
            callable(getattr(client, name, None))
            for name in ("sync_digest", "sync_catalog", "sync_blobs")
        )

        persisted = self.cache.load_persisted(server_id=server_id, metric_type=metric_type)
        if supports_catalog and persisted is not None:
            prev_records, prev_errors, prev_digest, prev_xmins = persisted
            try:
                server_digest, legacy_envelopes = await client.sync_digest(
                    server_token, metric_type=metric_type
                )
            except VaultbeatUnsupportedMetricError:
                raise
            except VaultbeatCloudError:
                # Digest is an optimization; a failure here must not fail the
                # read. Fall through to the full fetch.
                server_digest, legacy_envelopes = None, None

            if legacy_envelopes is not None:
                # Old edge ignored `fields` and sent everything. Do not pay for
                # it twice — that response IS the data.
                envelope_rows = legacy_envelopes
            elif server_digest is not None and prev_digest is not None:
                if server_digest == prev_digest:
                    # Nothing changed. Zero further bytes.
                    records = [DecryptedRecord.from_dict(row) for row in prev_records]
                    prev_errors = _recheck_error_lines(prev_errors)
                    if self._is_newest_fetch(metric_type, generation):
                        self.cache.save(
                            prev_records,
                            server_id=server_id,
                            metric_type=metric_type,
                            errors=prev_errors,
                            digest=prev_digest,
                            blob_xmins=prev_xmins,
                            started_at=fetch_started,
                        )
                    self.store.update(last_sync_at=now_iso())
                    if limit is not None:
                        records = records[:limit]
                    return records, prev_errors

                catalog = await client.sync_catalog(server_token, metric_type=metric_type)
                if catalog is not None:
                    catalog_xmins = {
                        str(row.get("blob_id", "")): str(row.get("xmin", ""))
                        for row in catalog
                        if row.get("blob_id")
                    }
                    # Rows whose version differs, or that we have never seen.
                    # A row missing from the catalog was deleted server-side and
                    # is simply not carried forward — that is the delete path.
                    needed = [
                        blob_id
                        for blob_id, xmin in catalog_xmins.items()
                        if prev_xmins.get(blob_id) != xmin
                    ]
                    keep = {
                        blob_id
                        for blob_id, xmin in catalog_xmins.items()
                        if prev_xmins.get(blob_id) == xmin
                    }
                    reusable_records = [
                        row for row in prev_records if str(row.get("blob_id", "")) in keep
                    ]
                    # Errors belong to rows we are re-fetching, so they are
                    # recomputed below rather than carried over — keeping them
                    # would report a failure that may have just been repaired.
                    reusable_errors = []
                    envelope_rows = await client.sync_blobs(
                        server_token, blob_ids=needed, metric_type=metric_type
                    )

        try:
            if envelope_rows is None:
                envelope_rows = await client.sync(server_token, metric_type=metric_type)
        except VaultbeatUnsupportedMetricError:
            # Version skew: this MCP server knows a kind the deployed edge
            # function does not. Degrade to "this one kind is unavailable"
            # rather than raising — every other tool call still works, and the
            # agent gets a message it can relay instead of an opaque failure.
            # (2026-07-22: the opposite behaviour took ALL default HRV reads
            # down for two days.) The kind named is the one THIS call asked for,
            # not the server's echo of it (Anti-pattern 23).
            return [], [
                f"unsupported_metric:{metric_type or 'unknown'} — the Vaultbeat cloud "
                "has not been updated to serve this data type yet; every other "
                "data type is unaffected"
            ]
        records = []
        errors: list[str] = []
        # Blobs whose envelope unwrapped to a valid DEK that then failed to
        # open the ciphertext — proven damage from THIS server's vantage point,
        # collected so it can be reported to `report_decrypt_failures` after the
        # loop. Deliberately NOT every decrypt_failed entry: a plain
        # VaultbeatCryptoError also covers "not for me" (rotated key, unfinished
        # rebind), which must never trigger a report — see Invariant 34.
        undecryptable: list[dict[str, str]] = []

        for row in envelope_rows:
            try:
                records.append(self._decrypt_row(row, config))
            except VaultbeatDekMismatchError as error:
                blob = row.get("encrypted_sleep_blobs")
                if isinstance(blob, list):
                    blob = blob[0] if blob else None
                blob_id = str(row.get("blob_id", "")) if isinstance(blob, dict) else ""
                kind = blob.get("metric_type") if isinstance(blob, dict) else None
                if blob_id and isinstance(kind, str) and kind in KNOWN_METRIC_TYPES:
                    undecryptable.append({"blob_id": blob_id, "metric_type": kind})
                envelope_id = _safe_row_id(row.get("id", "<unknown>"))
                errors.append(f"{envelope_id}: decrypt_failed ({type(error).__name__})")
            except (KeyError, TypeError, VaultbeatCryptoError, ValueError) as error:
                envelope_id = _safe_row_id(row.get("id", "<unknown>"))
                # Stage-tagged so a consumer can tell "sealed with a stale key /
                # corrupt ciphertext" apart from the parse_failed entries the
                # per-metric decoders append (2026-07-23 client feedback: a bare
                # exception name gave no clue whether the error mattered).
                errors.append(f"{envelope_id}: decrypt_failed ({type(error).__name__})")

        if undecryptable:
            # Best-effort: a failed report must never surface as a read
            # failure. Nothing is lost by swallowing it here — the same blob
            # gets another chance to be reported on the next call that reaches
            # it (the cloud side upserts on blob_id, so repeat reports just
            # refresh reported_at).
            try:
                await self._client(config).report_decrypt_failures(server_token, items=undecryptable)
            except Exception:
                pass

        if metric_type is not None:
            # Defensive filter: also correct against pre-metric_type edge deploys.
            records = [r for r in records if (r.metric_type or METRIC_SLEEP) == metric_type]

        # Fold the rows we already held (verified unchanged by xmin) back in with
        # the ones just fetched. Keyed by blob_id, freshly-fetched wins, then
        # sorted so the stored order is deterministic regardless of which path
        # produced it — every downstream reader re-sorts by business date anyway
        # (Invariant 38), but a stable order keeps cache files diffable.
        record_dicts = [record.to_dict() for record in records]
        if reusable_records:
            by_blob: dict[str, dict[str, Any]] = {}
            for row in reusable_records:
                by_blob[str(row.get("blob_id", ""))] = row
            for row in record_dicts:
                by_blob[str(row.get("blob_id", ""))] = row
            record_dicts = sorted(by_blob.values(), key=lambda r: str(r.get("blob_id", "")))
            records = [DecryptedRecord.from_dict(row) for row in record_dicts]
            errors = reusable_errors + errors

        # Came through the full-fetch path, so there is no catalog yet. Pull one
        # (43 KB measured, once) and derive the digest FROM IT rather than asking
        # the server separately: same response, so the stored digest and the
        # stored per-row versions cannot disagree with each other. Without this
        # the next read would have a digest to compare but no per-row versions to
        # diff against, and would re-download everything to learn what changed.
        if supports_catalog and not catalog_xmins:
            try:
                catalog = await client.sync_catalog(server_token, metric_type=metric_type)
            except VaultbeatCloudError:
                catalog = None
            if catalog is not None:
                catalog_xmins = {
                    str(row.get("blob_id", "")): str(row.get("xmin", ""))
                    for row in catalog
                    if row.get("blob_id")
                }
                server_digest = self._digest_from_catalog(catalog_xmins)

        if self._is_newest_fetch(metric_type, generation):
            self.cache.save(
                record_dicts,
                server_id=server_id,
                metric_type=metric_type,
                errors=errors,
                digest=server_digest,
                blob_xmins=catalog_xmins,
                started_at=fetch_started,
            )
        self.store.update(last_sync_at=now_iso())
        if limit is not None:
            records = records[:limit]
        return records, errors

    async def _committed(self, metric_type: str, write: Awaitable[_T]) -> _T:
        """Await a write and, once the cloud accepted it, retire what the cache
        held for its kind (review V6, 2026-10-03).

        Every write tool re-reads `fresh=True` afterwards, but a re-read that
        failed (a catalog or blob error after the 200) made the tool report an
        error for a write that HAD landed, and left the pre-write snapshot — the
        one its own pre-write read had just saved — answering plain reads for
        the whole TTL, so an agent that retried saw nothing change. Now the
        kind's cache expires and its generation moves on the moment the write
        returns, so neither that snapshot nor a fetch begun before the write
        can answer the next read.
        """
        response = await write
        self.cache.expire(metric_type)
        self._sync_generation[metric_type] = self._sync_generation.get(metric_type, 0) + 1
        # And no plain read may join a download begun before the write: one
        # without a fresh re-read after it (`delete_symptom`) would otherwise
        # hand the deleted entry back until that download finished.
        self._inflight_syncs.pop(metric_type, None)
        return response

    def _is_newest_fetch(self, metric_type: str | None, generation: int | None) -> bool:
        """Whether a fetch may still write the cache: no newer fetch of its kind began."""
        return generation is None or self._sync_generation.get(metric_type) == generation

    @staticmethod
    def _digest_from_catalog(blob_xmins: dict[str, str]) -> dict[str, Any] | None:
        """Recompute mcp-sync's digest locally from a catalog response.

        MUST stay identical to the server's arithmetic in
        `supabase/functions/mcp-sync/index.ts` — count, max, sum over the same
        rows. A mismatch would make every subsequent digest comparison fail and
        silently degrade the client to full fetches (costly, never incorrect).
        Returns None if any xmin is unparseable, which keeps a bad catalog from
        poisoning the stored digest.
        """

        total = 0
        largest = 0
        for raw in blob_xmins.values():
            try:
                value = int(raw)
            except (TypeError, ValueError):
                return None
            total += value
            if value > largest:
                largest = value
        return {
            "count": len(blob_xmins),
            "max_xmin": str(largest),
            "sum_xmin": str(total),
        }

    async def _records_for_metric(
        self, metric_type: str, *, limit: int | None, fresh: bool = False
    ) -> tuple[list[DecryptedRecord], list[str]]:
        """Records of one metric kind, newest first.

        The limit is applied AFTER sorting by created_at descending (newest
        first) so a caller asking for 50 sleep records always gets the 50 most
        recent, regardless of envelope ID ordering from the cloud. Legacy blobs
        with a null metric_type are treated as "sleep".
        """

        kept, errors = await self.sync_decrypted_records(
            limit=None, metric_type=metric_type, fresh=fresh
        )
        kept = sorted(kept, key=lambda r: r.created_at or "", reverse=True)
        if limit is not None:
            kept = kept[:limit]
        return kept, errors

    async def sleep_detail_records(
        self, *, limit: int | None = None, fresh: bool = False,
        owner: str | None = None, include_timeline: bool = False,
    ) -> dict[str, Any]:
        """Return per-night time-aligned HR + RR + sleep stage data.

        Each vital-sign sample is tagged with the sleep stage active at that
        moment.  Output is one object per night (primary session only), sorted
        newest-first.

        *owner*: if given, only include records whose ``owner_user_id`` starts
        with this prefix — the first characters of the account's UUID, enough to
        tell two people apart (e.g. ``"a1b2c3d4"``).

        The example is synthetic on purpose. This file ships to PyPI, so a
        docstring written against whichever account was open at the time
        publishes that person's real id fragment to everyone who installs the
        package — on a product whose entire claim is that we cannot see your
        data. Two real prefixes rode here from v0.1.0 until 2026-08-21.

        *include_timeline*: the per-sample ``timeline`` array is ~80% of this
        payload — roughly 13k characters PER NIGHT, so the old ``limit=5``
        default returned ~66k characters and blew past a 25k-token MCP client
        budget at ``limit=4`` (2026-07-28). It is off by default: the derived
        ``stage_intervals`` / ``stage_minutes`` / ``stage_vitals`` answer most
        sleep questions without it. Turn it on only when you actually need
        sample-level HR/RR, and keep ``limit`` small when you do. The timeline
        is still computed either way — ``stage_vitals`` is aggregated from it.
        """

        records, errors = await self._records_for_metric(METRIC_SLEEP, limit=None, fresh=fresh)
        if owner:
            records = _select_owner(records, owner)
        # Every conversion below asks for the offset in force AT that instant,
        # never today's — see `_local_midnight` — and in the zone the phone
        # stamped on the night when it did (`_session_zone`); `zone=None` is
        # this machine's.

        def _to_local_iso(raw: str, zone: tzinfo | None = None) -> str:
            try:
                return datetime.fromisoformat(
                    raw.replace("Z", "+00:00")
                ).astimezone(zone).strftime("%Y-%m-%dT%H:%M:%S")
            except (ValueError, TypeError, AttributeError):
                return raw

        def _to_local_short(raw: str, zone: tzinfo | None = None) -> str:
            try:
                return datetime.fromisoformat(
                    raw.replace("Z", "+00:00")
                ).astimezone(zone).strftime("%Y-%m-%dT%H:%M")
            except (ValueError, TypeError, AttributeError):
                return raw

        def _stage_at(ts_utc: str, stage_intervals: list[tuple[float, float, str]]) -> str:
            try:
                t = datetime.fromisoformat(ts_utc.replace("Z", "+00:00")).timestamp()
            except (ValueError, TypeError):
                return "unknown"
            # `inBed` spans the whole night and sorts first, so first-match
            # handed EVERY sample on a night that also carries stages to
            # "inBed" and left the asleep stages' vitals empty (2026-09-24:
            # the demo's HR never reached `sleep_hr_mean`; on real data, every
            # night an iPhone or a third-party app also wrote `inBed`). A
            # sample inside a real stage belongs to that stage.
            in_bed = False
            for start_ts, end_ts, stage in stage_intervals:
                if start_ts <= t <= end_ts:
                    if stage != "inBed":
                        return stage
                    in_bed = True
            return "inBed" if in_bed else "between_stages"

        all_nights: list[dict[str, Any]] = []

        for record in records:
            try:
                payload = record.payload
                session = payload.get("session", payload)
                samples = session.get("samples", [])
                hrs = payload.get("heartRateSamples", [])
                rrs = payload.get("respiratoryRateSamples", [])

                actual_sleep_stages = {
                    "asleepCore", "asleepDeep", "asleepREM", "asleepUnspecified"
                }
                distinct_actual = {
                    s.get("stage") for s in samples
                    if s.get("stage") in actual_sleep_stages
                }
                has_stage_detail = len(distinct_actual) >= 2
                is_in_bed_only = len(distinct_actual) == 0

                zone = _session_zone(session)
                sd_raw = session.get("sessionDate", "")
                try:
                    sd_utc = datetime.fromisoformat(sd_raw.replace("Z", "+00:00"))
                    local_date = _session_local_date(session, sd_utc)
                except (ValueError, TypeError):
                    local_date = sd_raw[:10] if sd_raw else ""

                # One pass over the samples building intervals AND durations.
                # Was two near-identical loops, each swallowing malformed
                # samples with a bare `pass` — the only place in this file that
                # dropped bad data without telling anyone. Malformed samples now
                # land in `errors` like everywhere else, collapsed to one line
                # per night so a single bad blob can't flood the list
                # (2026-07-27).
                # Durations accumulate in SECONDS and truncate once per stage;
                # the old per-sample int(seconds / 60) ran ~6.8 min short per
                # night on average (2026-07-27).
                stage_intervals: list[tuple[float, float, str]] = []
                stage_seconds: dict[str, float] = {}
                malformed_samples = 0
                last_sample_error: Exception | None = None
                for s in samples:
                    try:
                        t0 = datetime.fromisoformat(s["startDate"].replace("Z", "+00:00"))
                        t1 = datetime.fromisoformat(s["endDate"].replace("Z", "+00:00"))
                        stg = s.get("stage", "unknown")
                        stage_intervals.append((t0.timestamp(), t1.timestamp(), stg))
                        secs = max((t1 - t0).total_seconds(), 0.0)
                        stage_seconds[stg] = stage_seconds.get(stg, 0.0) + secs
                    except (ValueError, TypeError, KeyError, AttributeError) as sample_error:
                        malformed_samples += 1
                        last_sample_error = sample_error
                if last_sample_error is not None:
                    errors.append(
                        f"{record.envelope_id}: parse_failed "
                        f"({malformed_samples} sleep sample(s) skipped; last: "
                        f"{type(last_sample_error).__name__}: {last_sample_error})"
                    )
                stage_minutes = {
                    stage: int(secs / 60) for stage, secs in stage_seconds.items()
                }
                total_sleep_min = sum(
                    v for k, v in stage_minutes.items() if k in actual_sleep_stages
                )
                stage_intervals.sort(key=lambda x: x[0])

                stage_intervals_out: list[dict[str, str]] = []
                for si_start, si_end, si_stage in stage_intervals:
                    stage_intervals_out.append({
                        "stage": si_stage,
                        # Same zone as bedtime / wake (review V4 follow-up):
                        # this alone stayed in the machine's, so a UTC host
                        # put the stages eight hours off the night they are in.
                        "start": datetime.fromtimestamp(si_start, tz=zone).strftime("%Y-%m-%dT%H:%M:%S"),
                        "end": datetime.fromtimestamp(si_end, tz=zone).strftime("%Y-%m-%dT%H:%M:%S"),
                    })

                raw_points: list[tuple[str, float | None, float | None]] = []
                for h in hrs:
                    raw_points.append((h.get("startDate", ""), h.get("value"), None))
                for r in rrs:
                    raw_points.append((r.get("startDate", ""), None, r.get("value")))

                raw_points.sort(key=lambda x: x[0])

                timeline: list[dict[str, Any]] = []
                last_hr: float | None = None
                last_rr: float | None = None
                stage_hr: dict[str, list[float]] = {}
                stage_rr: dict[str, list[float]] = {}
                for ts_raw, hr_val, rr_val in raw_points:
                    point_stage = _stage_at(ts_raw, stage_intervals)
                    if hr_val is not None:
                        last_hr = hr_val
                        stage_hr.setdefault(point_stage, []).append(hr_val)
                    if rr_val is not None:
                        last_rr = rr_val
                        stage_rr.setdefault(point_stage, []).append(rr_val)
                    timeline.append({
                        "time": _to_local_iso(ts_raw, zone),
                        "hr": last_hr,
                        "rr": last_rr,
                        "stage": point_stage,
                    })

                # Pre-computed per-stage vitals so downstream consumers (weak
                # local LLMs included) never have to aggregate the timeline
                # themselves.
                stage_vitals: dict[str, dict[str, float | int | None]] = {}
                for stg in set(stage_hr) | set(stage_rr):
                    hr_vals = stage_hr.get(stg, [])
                    rr_vals = stage_rr.get(stg, [])
                    stage_vitals[stg] = {
                        "hr_mean": round(sum(hr_vals) / len(hr_vals), 1) if hr_vals else None,
                        "hr_min": min(hr_vals) if hr_vals else None,
                        "hr_max": max(hr_vals) if hr_vals else None,
                        "rr_mean": round(sum(rr_vals) / len(rr_vals), 1) if rr_vals else None,
                        "rr_min": min(rr_vals) if rr_vals else None,
                        "rr_max": max(rr_vals) if rr_vals else None,
                    }

                # Night-level features computed HERE, while the raw intervals
                # and per-stage samples are still in hand, so every consumer
                # (this tool, `sleep_nights`, the sleep series) reads one
                # derivation instead of re-deriving from a payload that has
                # already had its timeline dropped (2026-09-24).
                asleep_hr = [v for k, vs in stage_hr.items() if k in actual_sleep_stages for v in vs]
                asleep_rr = [v for k, vs in stage_rr.items() if k in actual_sleep_stages for v in vs]
                night_source = _sleep_source(samples, session.get("provenance"))
                awakenings, longest_bout = _sleep_continuity(stage_intervals, actual_sleep_stages)

                all_nights.append({
                    "envelope_id": record.envelope_id,
                    "local_date": local_date,
                    "bedtime": _to_local_short(session.get("bedtime", ""), zone),
                    "wake_time": _to_local_short(session.get("wakeTime", ""), zone),
                    "total_sleep_minutes": total_sleep_min,
                    # In-bed-only nights (Watch not worn — no actual-sleep stage
                    # anywhere) keep the honest `total_sleep_minutes: 0` but must
                    # NOT read as "slept 0 hours": the label says "no sleep data",
                    # `is_in_bed_only` is surfaced, and `in_bed_minutes` carries
                    # what was measured. Mirrors the iOS assembler's F2 rule that
                    # the in-bed number "MUST be labelled 'in bed', never as
                    # sleep" (2026-07-27). This text lived on the retired
                    # `_select_primary_sessions` until 2026-09-25.
                    "duration_label": (
                        _NO_SLEEP_DATA_LABEL
                        if is_in_bed_only
                        else "{}h{:02d}m".format(*divmod(total_sleep_min, 60))
                    ),
                    # None when no in-bed time was recorded: a Watch night has
                    # stages and no `inBed` samples, and 0 read as "measured
                    # zero minutes in bed" (pre-release review, 2026-10-02).
                    "in_bed_minutes": int(stage_minutes["inBed"]) if stage_minutes.get("inBed") else None,
                    "has_stage_detail": has_stage_detail,
                    "is_in_bed_only": is_in_bed_only,
                    "stage_minutes": stage_minutes,
                    "stage_intervals": stage_intervals_out,
                    "stage_vitals": stage_vitals,
                    # A mean over a couple of samples is not a night's heart rate:
                    # 2026-07-31 reported "100 bpm asleep" from exactly 2 samples
                    # (median night: ~90). Below the floor it is None, and the
                    # count travels so a reader can see why.
                    "sleep_hr_mean": (
                        round(sum(asleep_hr) / len(asleep_hr), 1)
                        if len(asleep_hr) >= _MIN_SLEEP_HR_SAMPLES else None
                    ),
                    "sleep_hr_min": (
                        round(min(asleep_hr), 1) if len(asleep_hr) >= _MIN_SLEEP_HR_SAMPLES else None
                    ),
                    "sleep_rr_mean": (
                        round(sum(asleep_rr) / len(asleep_rr), 1)
                        if len(asleep_rr) >= _MIN_SLEEP_RR_SAMPLES else None
                    ),
                    "sleep_hr_samples": len(asleep_hr),
                    "source": night_source,
                    "awakenings": awakenings if has_stage_detail else None,
                    "longest_sleep_bout_minutes": longest_bout if has_stage_detail else None,
                    "hr_samples": len(hrs),
                    "rr_samples": len(rrs),
                    "stage_samples": len(samples),
                    "timeline": timeline,
                })
            except (KeyError, TypeError, VaultbeatCryptoError, ValueError) as error:
                errors.append(f"{record.envelope_id}: parse_failed ({type(error).__name__}: {error})")

        by_date: dict[str, list[dict[str, Any]]] = {}
        for n in all_nights:
            by_date.setdefault(n["local_date"], []).append(n)

        result_nights: list[dict[str, Any]] = []
        for day_key in sorted(by_date.keys(), reverse=True):
            candidates = by_date[day_key]
            best = max(candidates, key=lambda n: (
                not n.get("is_in_bed_only", False),
                n.get("has_stage_detail", False),
                n.get("total_sleep_minutes", 0),
                -(datetime.fromisoformat(n["bedtime"]).timestamp()
                  if n.get("bedtime") else 0),
            ))
            # Other MEASURED sleep that day that does not overlap the main one:
            # a nap, or the second half of a broken night. Until 2026-09-24
            # only the main session survived, so 18 of 323 real days lost a
            # real sleep (a 12:26-13:48 nap, a night split 01:42-02:54 +
            # 05:52-10:23). A candidate that OVERLAPS a kept one is the same
            # sleep written by another source, never a second sleep.
            kept = [best]
            others: list[dict[str, Any]] = []
            for n in sorted(candidates, key=lambda n: n.get("bedtime") or ""):
                if n is best or n.get("is_in_bed_only") or not n.get("total_sleep_minutes"):
                    continue
                if any(_sessions_overlap(n, k) for k in kept):
                    continue
                kept.append(n)
                others.append({
                    "bedtime": n.get("bedtime"),
                    "wake_time": n.get("wake_time"),
                    "asleep_minutes": n.get("total_sleep_minutes"),
                })
            best["other_sleep"] = others
            best["other_sleep_minutes"] = sum(o["asleep_minutes"] for o in others)
            best["sleep_segments"] = len(others) + (0 if best.get("is_in_bed_only") else 1)
            result_nights.append(best)

        _avail = len(result_nights)
        _oldest = _coverage_day_of(result_nights[-1]) if result_nights else None
        if limit is not None:
            result_nights = result_nights[:limit]

        if not include_timeline:
            # Drop after primary-session selection so the choice still sees every
            # field it ranks on. Safe to mutate: these dicts are rebuilt per call
            # (the cache holds decrypted records, not this summary).
            for night in result_nights:
                night.pop("timeline", None)

        summary = {
            "nights": result_nights,
            "count": len(result_nights),
            # Say so explicitly — otherwise a caller that read about `timeline`
            # in the tool description sees it missing and re-queries with
            # fresh=True chasing a sync problem that does not exist.
            "timeline_included": include_timeline,
        }
        _attach_coverage(summary, rows=result_nights, requested=limit, unit="nights", total_available=_avail, oldest_raw=_oldest)
        _attach_errors(summary, errors)
        return _attach_owner_guard(summary, records, owner)

    async def sleep_nights(
        self, *, limit: int | None = None, owner: str | None = None, fresh: bool = False,
        since: str | None = None,
    ) -> dict[str, Any]:
        """Every night as one flat row: timing, stages, continuity, vitals.

        The source for BOTH the sleep series in `get_metric` and the
        `get_sleep_nights` table, so an average and the rows it came from can
        never disagree. Built on `sleep_detail_records` (primary session per
        night, same selection as the app) with the timeline dropped — a year of
        nights fits in one read this way, which is the whole point.
        """

        detail = await self.sleep_detail_records(limit=None, owner=owner, fresh=fresh)
        rows = [sleep_night_row(n) for n in detail.get("nights", [])]
        avail = len(rows)
        oldest = rows[-1]["local_date"] if rows else None
        if since:
            # A calendar bound, filtered BEFORE coverage so the block describes
            # the rows actually returned. `more_available` then correctly says
            # older nights exist beyond `since`.
            rows = [r for r in rows if str(r.get("local_date") or "") >= since]
        if limit is not None:
            rows = rows[:limit]
        summary: dict[str, Any] = {"nights": rows, "count": len(rows)}
        sources = _source_summary(rows)
        if sources:
            summary["sources"] = sources
            if len(sources) > 1:
                summary["source_note"] = SLEEP_SOURCE_NOTE
        _attach_coverage(summary, rows=rows, requested=limit, unit="nights", total_available=avail, oldest_raw=oldest)
        errors = list(detail.get("errors") or [])
        _attach_errors(summary, errors)
        for key in ("mixed_owners", "owner_user_id_prefixes", "warning"):
            if key in detail:
                summary[key] = detail[key]
        return summary

    async def water_intake_summary(self, *, limit: int | None = None, owner: str | None = None, fresh: bool = False) -> dict[str, Any]:
        """Return recent daily water intake plus the computed average over the window."""

        records, errors = await self._records_for_metric(METRIC_WATER, limit=None, fresh=fresh)
        if owner:
            records = _select_owner(records, owner)
        days: list[WaterDay] = []
        for record in records:
            try:
                days.append(parse_water_day(record.payload, owner_user_id=record.owner_user_id))
            except (KeyError, TypeError, VaultbeatCryptoError, ValueError) as error:
                errors.append(f"{record.envelope_id}: parse_failed ({type(error).__name__}: {error})")
        # Business-time sort before the cut (see activity_summary's comment).
        # 2026-07-27: this kind still cut `records` by created_at BEFORE parsing,
        # so a backfill batch could drop the newest days — same bug the other
        # eight kinds had fixed on 2026-07-24.
        days.sort(key=lambda d: d.day_start_date, reverse=True)
        _avail, _oldest = len(days), (days[-1].day_start_date if days else None)
        if limit is not None:
            days = days[:limit]
        summary = summarize_water_intake(days)
        # From the summary's OWN rows, not from `days`: summarize_water_intake
        # dedups by dayID, so len(days) can exceed what the average divides by.
        _attach_coverage(summary, rows=summary["days"], requested=limit, unit="days", total_available=_avail, oldest_raw=_oldest)
        _attach_errors(summary, errors)
        return _attach_owner_guard(summary, records, owner)

    async def weight_trend_summary(
        self, *, limit: int | None = None, goal_kg: float | None = None, owner: str | None = None, fresh: bool = False
    ) -> dict[str, Any]:
        """Return recent body-weight days plus the computed trend over the window.

        Body weight is shared bidirectionally by default (sleep-style visibility, not
        menstrual-style opt-in). goal_kg is supplied by the caller — the goal lives in
        the owner's iOS UserDefaults (VaultbeatBodyGoalSettingsStore) and never syncs here,
        so without it the goal-distance is reported as None rather than assumed.
        """

        records, errors = await self._records_for_metric(METRIC_BODY, limit=None, fresh=fresh)
        if owner:
            records = _select_owner(records, owner)
        days: list[BodyDay] = []
        for record in records:
            try:
                days.append(parse_body_day(record.payload, owner_user_id=record.owner_user_id))
            except (KeyError, TypeError, VaultbeatCryptoError, ValueError) as error:
                errors.append(f"{record.envelope_id}: parse_failed ({type(error).__name__}: {error})")
        # Business-time sort before the cut (see activity_summary's comment).
        # 2026-07-27: reproduced on live data — get_weight_trend(limit=10,
        # owner="a1a1") silently dropped the real 2026-07-21 weigh-in (83.0 kg)
        # while keeping the older 07-10 one, because the created_at cut landed
        # before the parse.
        days.sort(key=lambda d: d.day_start_date, reverse=True)
        _avail, _oldest = len(days), (days[-1].day_start_date if days else None)
        if limit is not None:
            days = days[:limit]
        summary = summarize_weight_trend(days, goal_kg=goal_kg)
        # The trend line and `weekly_change_kg` are fitted over these rows, so
        # coverage has to describe the same set — two weigh-ins 40 days apart and
        # 40 daily ones produce an identically-shaped slope.
        _attach_coverage(summary, rows=summary["days"], requested=limit, unit="days", total_available=_avail, oldest_raw=_oldest)
        _attach_errors(summary, errors)
        return _attach_owner_guard(summary, records, owner)

    async def menstrual_cycle_summary(self, *, limit: int | None = None, owner: str | None = None, fresh: bool = False) -> dict[str, Any]:
        """Return recent menstrual cycle samples plus a simple next-period prediction.

        Menstrual blobs only arrive here when the user explicitly opted in on iOS; this
        layer never requests them differently, it just decodes whatever envelopes show
        up. The data is sensitive — it stays on-device and is never re-exported.
        """

        records, errors = await self._records_for_metric(METRIC_MENSTRUAL, limit=None, fresh=fresh)
        if owner:
            records = _select_owner(records, owner)
        if not records:
            _LOG.info("no menstrual envelopes present (likely not opted in on iOS)")
        else:
            _LOG.info("decoding %d menstrual envelope(s); sensitive, kept local", len(records))
        days: list[MenstrualDay] = []
        for record in records:
            try:
                days.append(parse_menstrual_day(record.payload, owner_user_id=record.owner_user_id))
            except (KeyError, TypeError, VaultbeatCryptoError, ValueError) as error:
                errors.append(f"{record.envelope_id}: parse_failed ({type(error).__name__}: {error})")
        # Business-time sort before the cut (see activity_summary's comment).
        # 2026-07-27: this kind still cut `records` by created_at BEFORE parsing,
        # so a backfill batch could hide the most recent cycle days — and the
        # prediction is only as good as the newest bleeding day it can see.
        days.sort(key=lambda d: d.day_start_date, reverse=True)
        _avail, _oldest = len(days), (days[-1].day_start_date if days else None)
        # 🔴 The prediction is computed over the WHOLE history; `limit` only
        # trims the days returned. It used to be computed on the cut window,
        # so the answer depended on how many rows the caller asked to see —
        # measured 2026-09-24 on real data: limit=3 misplaced the last cycle
        # start (a mid-period day read as a start) and reported "insufficient
        # history" over 48 recorded days; 10 / 20 / 90 predicted three
        # different dates. An agent lowering `limit` to save tokens should get
        # a shorter list, never a different forecast.
        menstrual_owners = {d.owner_user_id for d in days if d.owner_user_id}
        wrist_readings = await self._wrist_readings_for_owner(menstrual_owners, errors, fresh=fresh)
        summary = summarize_menstrual_cycle(days, wrist_readings=wrist_readings)
        if limit is not None:
            summary["days"] = summary["days"][:limit]
            summary["day_count"] = len(summary["days"])
        _attach_coverage(summary, rows=summary["days"], requested=limit, unit="days", total_available=_avail, oldest_raw=_oldest)
        _attach_errors(summary, errors)
        return _attach_owner_guard(summary, records, owner)

    async def _wrist_readings_for_owner(
        self, menstrual_owners: set[str], errors: list[str], *, fresh: bool = False
    ) -> list[tuple[datetime, float]] | None:
        """Wrist-temp readings for ovulation calibration — SAME OWNER only.

        Wrist temp is `.ownDevicesOnly`, so this server only ever holds the
        owner's deltas; the menstrual blobs may belong to the partner (shared
        cycle). Calibration is only honest when the cycle and the temperatures
        come from the same body: exactly one menstrual owner, and it must also
        own the wrist blobs. Any ambiguity (no owner metadata yet, mixed
        owners) → None, and the prediction stays statistical.
        """

        if len(menstrual_owners) != 1:
            return None
        cycle_owner = next(iter(menstrual_owners))
        records, wrist_errors = await self._records_for_metric(METRIC_WRIST_TEMP, limit=120, fresh=fresh)
        errors.extend(wrist_errors)
        readings: list[tuple[datetime, float]] = []
        for record in records:
            if record.owner_user_id != cycle_owner:
                continue
            try:
                parsed = parse_wrist_temp_record(record.payload, owner_user_id=record.owner_user_id)
                readings.append((_parse_iso8601(parsed.date), parsed.temperature_delta_celsius))
            except (KeyError, TypeError, VaultbeatCryptoError, ValueError) as error:
                errors.append(f"{record.envelope_id}: parse_failed ({type(error).__name__}: {error})")
        return readings or None

    async def activity_summary(self, *, limit: int | None = None, owner: str | None = None, fresh: bool = False) -> dict[str, Any]:
        """Return recent daily activity ring data (steps, energy, exercise, stand, distance)."""

        records, errors = await self._records_for_metric(METRIC_ACTIVITY, limit=None, fresh=fresh)
        if owner:
            records = _select_owner(records, owner)
        days: list[ActivityDay] = []
        for record in records:
            try:
                days.append(parse_activity_day(record.payload, owner_user_id=record.owner_user_id))
            except (KeyError, TypeError, VaultbeatCryptoError, ValueError) as error:
                errors.append(f"{record.envelope_id}: parse_failed ({type(error).__name__}: {error})")
        # Sort by the payload's own day, NOT created_at: a history backfill
        # uploads years of old records in one batch, so upload order stops
        # matching business time and a created_at cut can drop the newest days
        # (2026-07-24: mindfulness/vo2max came back visibly shuffled).
        days.sort(key=lambda d: d.day_start_date, reverse=True)
        _avail, _oldest = len(days), (days[-1].day_start_date if days else None)
        if limit is not None:
            days = days[:limit]
        summary = {"days": [d.to_dict() for d in days], "count": len(days)}
        _attach_coverage(summary, rows=summary["days"], requested=limit, unit="days", total_available=_avail, oldest_raw=_oldest)
        _attach_errors(summary, errors)
        return _attach_owner_guard(summary, records, owner)

    async def resting_hr_records(self, *, limit: int | None = None, owner: str | None = None, fresh: bool = False) -> dict[str, Any]:
        """Return recent resting heart rate samples."""

        records, errors = await self._records_for_metric(METRIC_RESTING_HR, limit=None, fresh=fresh)
        if owner:
            records = _select_owner(records, owner)
        hr_records: list[RestingHrRecord] = []
        for record in records:
            try:
                hr_records.append(parse_resting_hr_record(record.payload, owner_user_id=record.owner_user_id))
            except (KeyError, TypeError, VaultbeatCryptoError, ValueError) as error:
                errors.append(f"{record.envelope_id}: parse_failed ({type(error).__name__}: {error})")
        # Business-time sort before the cut (see activity_summary's comment).
        hr_records.sort(key=lambda r: r.date, reverse=True)
        _avail, _oldest = len(hr_records), (hr_records[-1].date if hr_records else None)
        if limit is not None:
            hr_records = hr_records[:limit]
        bpms = [r.bpm for r in hr_records]
        average_bpm = sum(bpms) / len(bpms) if bpms else None
        summary = {
            "records": [r.to_dict() for r in hr_records],
            "count": len(hr_records),
            "average_bpm": round(average_bpm, 1) if average_bpm is not None else None,
        }
        # `limit` counts SAMPLES here, so days_covered can be far below it without
        # anything being missing — `requested_unit` is what stops that reading as
        # a gap.
        _attach_coverage(summary, rows=summary["records"], requested=limit, unit="samples", total_available=_avail, oldest_raw=_oldest)
        _attach_errors(summary, errors)
        return _attach_owner_guard(summary, records, owner)

    async def workout_records(self, *, limit: int | None = None, owner: str | None = None, fresh: bool = False) -> dict[str, Any]:
        """Return recent workout sessions."""

        records, errors = await self._records_for_metric(METRIC_WORKOUT, limit=None, fresh=fresh)
        if owner:
            records = _select_owner(records, owner)
        workouts: list[WorkoutRecord] = []
        for record in records:
            try:
                workouts.append(parse_workout_record(record.payload, owner_user_id=record.owner_user_id))
            except (KeyError, TypeError, VaultbeatCryptoError, ValueError) as error:
                errors.append(f"{record.envelope_id}: parse_failed ({type(error).__name__}: {error})")
        # Business-time sort before the cut (see activity_summary's comment).
        workouts.sort(key=lambda w: w.start_date, reverse=True)
        _avail, _oldest = len(workouts), (workouts[-1].start_date if workouts else None)
        if limit is not None:
            workouts = workouts[:limit]
        total_duration = sum(w.duration_seconds for w in workouts)
        summary = {
            "workouts": [w.to_dict() for w in workouts],
            "count": len(workouts),
            "total_duration_hours": round(total_duration / 3600, 2),
        }
        _attach_coverage(summary, rows=summary["workouts"], requested=limit, unit="workouts", total_available=_avail, oldest_raw=_oldest)
        _attach_errors(summary, errors)
        return _attach_owner_guard(summary, records, owner)

    async def mindfulness_summary(self, *, limit: int | None = None, owner: str | None = None, fresh: bool = False) -> dict[str, Any]:
        """Return recent daily mindfulness data (session count and total minutes)."""

        records, errors = await self._records_for_metric(METRIC_MINDFULNESS, limit=None, fresh=fresh)
        if owner:
            records = _select_owner(records, owner)
        days: list[MindfulnessDay] = []
        for record in records:
            try:
                days.append(parse_mindfulness_day(record.payload, owner_user_id=record.owner_user_id))
            except (KeyError, TypeError, VaultbeatCryptoError, ValueError) as error:
                errors.append(f"{record.envelope_id}: parse_failed ({type(error).__name__}: {error})")
        # Business-time sort before the cut (see activity_summary's comment).
        days.sort(key=lambda d: d.day_start_date, reverse=True)
        _avail, _oldest = len(days), (days[-1].day_start_date if days else None)
        if limit is not None:
            days = days[:limit]
        total_minutes = sum(d.total_minutes for d in days)
        summary = {
            "days": [d.to_dict() for d in days],
            "count": len(days),
            "total_minutes": round(total_minutes, 1),
        }
        _attach_coverage(summary, rows=summary["days"], requested=limit, unit="days", total_available=_avail, oldest_raw=_oldest)
        _attach_errors(summary, errors)
        return _attach_owner_guard(summary, records, owner)

    async def hrv_records(self, *, limit: int | None = None, owner: str | None = None, fresh: bool = False) -> dict[str, Any]:
        """Return recent HRV (SDNN) samples."""

        records, errors = await self._records_for_metric(METRIC_HRV, limit=None, fresh=fresh)
        if owner:
            records = _select_owner(records, owner)
        hrv_list: list[HRVRecord] = []
        for record in records:
            try:
                hrv_list.append(parse_hrv_record(record.payload, owner_user_id=record.owner_user_id))
            except (KeyError, TypeError, VaultbeatCryptoError, ValueError) as error:
                errors.append(f"{record.envelope_id}: parse_failed ({type(error).__name__}: {error})")
        # Business-time sort before the cut (see activity_summary's comment).
        hrv_list.sort(key=lambda r: r.date, reverse=True)
        _avail, _oldest = len(hrv_list), (hrv_list[-1].date if hrv_list else None)
        if limit is not None:
            hrv_list = hrv_list[:limit]
        sdnns = [r.sdnn_ms for r in hrv_list]
        average_sdnn = sum(sdnns) / len(sdnns) if sdnns else None
        summary = {
            "records": [r.to_dict() for r in hrv_list],
            "count": len(hrv_list),
            "average_sdnn_ms": round(average_sdnn, 1) if average_sdnn is not None else None,
        }
        # The headline case for this whole field: "average HRV" over 3 days and over
        # 30 days are the same number of digits, and `limit` counts samples, so a
        # 100-sample read can be a single night.
        _attach_coverage(summary, rows=summary["records"], requested=limit, unit="samples", total_available=_avail, oldest_raw=_oldest)
        _attach_errors(summary, errors)
        return _attach_owner_guard(summary, records, owner)

    async def hrv_hourly_records(self, *, limit: int | None = None, owner: str | None = None, fresh: bool = False) -> dict[str, Any]:
        """Return recent hourly-averaged HRV (SDNN) buckets.

        Companion aggregate view of raw HRV — same underlying SDNN
        measurements but pre-averaged per UTC hour bucket, so a 30-day
        window returns ≤720 rows instead of the raw kind's thousands.
        This is the default MCP granularity (`get_hrv()` without
        `granularity="raw"`) — saves context and is the right shape for
        trend/aggregate queries. For 5-15min spike-precision analysis
        (e.g. "HRV during those 3 minutes when I checked my phone"),
        callers should pass `granularity="raw"` which routes to
        `hrv_records`.

        `average_sdnn_ms` here is a sample-weighted average across all
        returned buckets. Note: it is NOT directly comparable to the raw
        kind's `average_sdnn_ms` — the two averages observe different
        underlying sample pools (hourly's 30d window vs raw's 3d window)
        plus the raw side includes any legacy per-sample blobs from
        before build 77. The previous "identical number regardless of
        granularity" claim was retracted 2026-07-22 (adversarial review
        pointed out the window mismatch).
        """

        records, errors = await self._records_for_metric(METRIC_HRV_HOURLY, limit=None, fresh=fresh)
        if owner:
            records = _select_owner(records, owner)
        buckets: list[HRVHourlyBucket] = []
        for record in records:
            try:
                buckets.append(parse_hrv_hourly_record(record.payload, owner_user_id=record.owner_user_id))
            except (KeyError, TypeError, VaultbeatCryptoError, ValueError) as error:
                errors.append(f"{record.envelope_id}: parse_failed ({type(error).__name__}: {error})")
        # Business-time sort before the cut (see activity_summary's comment).
        buckets.sort(key=lambda b: b.date, reverse=True)
        _avail, _oldest = len(buckets), (buckets[-1].date if buckets else None)
        if limit is not None:
            buckets = buckets[:limit]
        # Sample-weighted average so a hour with 12 samples counts more than
        # a hour with 1 sample. Matches the raw-kind average exactly (both
        # are means over the same underlying samples). Zero-sample buckets
        # were elided by the reader, so `sample_count` is guaranteed >= 1.
        total_weight = sum(b.sample_count for b in buckets)
        weighted_sum = sum(b.avg_sdnn_ms * b.sample_count for b in buckets)
        average = (weighted_sum / total_weight) if total_weight else None
        summary = {
            "records": [b.to_dict() for b in buckets],
            "count": len(buckets),
            "total_sample_count": total_weight,
            "average_sdnn_ms": round(average, 1) if average is not None else None,
        }
        _attach_coverage(
            summary, rows=summary["records"], requested=limit, unit="hourly buckets",
            total_available=_avail,
            oldest_raw=_oldest,
        )
        _attach_errors(summary, errors)
        return _attach_owner_guard(summary, records, owner)

    async def wrist_temp_records(self, *, limit: int | None = None, owner: str | None = None, fresh: bool = False) -> dict[str, Any]:
        """Return recent sleeping wrist temperature samples."""

        records, errors = await self._records_for_metric(METRIC_WRIST_TEMP, limit=None, fresh=fresh)
        if owner:
            records = _select_owner(records, owner)
        temp_list: list[WristTempRecord] = []
        for record in records:
            try:
                temp_list.append(parse_wrist_temp_record(record.payload, owner_user_id=record.owner_user_id))
            except (KeyError, TypeError, VaultbeatCryptoError, ValueError) as error:
                errors.append(f"{record.envelope_id}: parse_failed ({type(error).__name__}: {error})")
        # Business-time sort before the cut (see activity_summary's comment).
        temp_list.sort(key=lambda r: r.date, reverse=True)
        _avail, _oldest = len(temp_list), (temp_list[-1].date if temp_list else None)
        if limit is not None:
            temp_list = temp_list[:limit]
        deltas = [r.temperature_delta_celsius for r in temp_list]
        average_delta = sum(deltas) / len(deltas) if deltas else None
        summary = {
            "records": [r.to_dict() for r in temp_list],
            "count": len(temp_list),
            "average_delta_celsius": round(average_delta, 2) if average_delta is not None else None,
        }
        _attach_coverage(summary, rows=summary["records"], requested=limit, unit="samples", total_available=_avail, oldest_raw=_oldest)
        _attach_errors(summary, errors)
        return _attach_owner_guard(summary, records, owner)

    async def basal_energy_records(
        self,
        *,
        limit: int | None = None,
        owner: str | None = None,
        fresh: bool = False,
        day_limit: int | None = None,
    ) -> dict[str, Any]:
        """Return recent basal-energy-burned samples (Watch BMR estimate, kcal).

        Watch emits many samples per day, so `limit` caps SAMPLES, not days —
        leave it None (the MCP tool's default) and let the daily aggregation
        below do the summarising. Groups by local calendar day for a daily-BMR
        view (usually 1500-2000 kcal for an active young adult). Combine with
        `get_activity`'s `active_energy_kcal` for a proper TDEE (see
        `total_energy_burned`).

        Every `daily` row carries `hours_covered` — how many of the day's 24
        hourly buckets actually arrived. A day the Watch spent on the charger
        still produces a row, just a short one, and the kcal figure is short in
        exact proportion; `incomplete` is the flag that keeps such a day out of
        any average.

        `day_limit` caps how many days `daily` RETURNS. It is a DISPLAY cap,
        never a cap on what was read, and it defaults to None (every day).
        🔴 It defaulted to 30 for the per-kind `get_basal_energy` tool, and each
        consumer that forgot to override it inherited a 30-day history: first
        `total_energy_burned` (a 90-day TDEE query reported 60 days of
        `basal_missing` for data present the whole time, Invariant 62), then —
        after that tool was folded into `get_metric` in 0.9.0 and nobody wanted
        the cap any more — the series layer, so `list_metric_series` and
        `get_metric basal_energy` showed 29 days of a two-year history even
        with `since`, while `coverage.more_available` told the agent to ask for
        more (release gate G2, 2026-10-03). A default only one caller wanted is
        a trap for every other; a caller that wants a short list passes it.
        """

        records, errors = await self._records_for_metric(METRIC_BASAL_ENERGY, limit=None, fresh=fresh)
        if owner:
            records = _select_owner(records, owner)

        parsed: list[BasalEnergyRecord] = []
        for record in records:
            try:
                parsed.append(parse_basal_energy_record(record.payload, owner_user_id=record.owner_user_id))
            except (KeyError, TypeError, VaultbeatCryptoError, ValueError) as error:
                errors.append(f"{record.envelope_id}: parse_failed ({type(error).__name__}: {error})")

        # Business-time sort before the cut (see activity_summary's comment).
        # 2026-07-27: this kind still cut `records` by created_at BEFORE parsing,
        # which on a backfill batch throws away the newest samples and makes the
        # per-day sums below silently short.
        parsed.sort(key=lambda r: r.date, reverse=True)
        _avail, _oldest = len(parsed), (parsed[-1].date if parsed else None)
        if limit is not None:
            parsed = parsed[:limit]

        # Group by LOCAL calendar day. The old `r.date[:10]` cut took the UTC
        # date prefix, which for a UTC+8 user mis-filed every sample between
        # local 00:00-08:00 (= previous UTC day) into the PREVIOUS day —
        # ~600-700 kcal of sleeping basal per night, self-cancelling on middle
        # days but visibly wrong on the edges (2026-07-24 11:40 "today" showed
        # 191.9 kcal when local 00:00-11:40 alone should hold ~900).
        # Sum kcal AND count buckets in the same pass. The count was previously
        # discarded here, and losing it at this line is what made every
        # downstream consumer unable to tell a 24-hour day from a 12-hour one:
        # both arrive as a single kcal scalar, and 883 kcal reads as a real
        # (very low) metabolism rather than as half a day of missing data.
        by_day: dict[str, list[float]] = {}
        for r in parsed:
            try:
                day_key = _local_calendar_day(_parse_iso8601(r.date)).isoformat()
            except ValueError:
                day_key = r.date[:10]  # unparseable date: degrade to old cut
            bucket = by_day.setdefault(day_key, [0.0, 0.0])
            bucket[0] += r.kcal
            bucket[1] += 1
        daily: list[dict[str, Any]] = sorted(
            [
                {
                    "day": d,
                    "basal_kcal": round(kcal, 1),
                    "hours_covered": int(hours),
                    "hours_expected": _BASAL_HOURS_PER_DAY,
                    "incomplete": int(hours) < _BASAL_MIN_HOURS_COVERED,
                }
                for d, (kcal, hours) in by_day.items()
            ],
            key=lambda x: str(x["day"]),
            reverse=True,
        )

        # Today is still accumulating, so it is short for a reason that says
        # nothing about the Watch — excluded alongside the genuinely incomplete
        # days rather than lumped in with them.
        today_key = _local_calendar_day(datetime.now(timezone.utc)).isoformat()
        latest_day: dict[str, Any] | None = daily[0] if daily else None
        full_days = [d for d in daily if not d["incomplete"] and str(d["day"]) != today_key]
        avg_daily: float | None = (
            round(sum(float(d["basal_kcal"]) for d in full_days) / len(full_days), 1)
            if full_days
            else None
        )

        summary = {
            "sample_count": len(parsed),
            "day_count": len(daily),
            "latest_day_basal_kcal": float(latest_day["basal_kcal"]) if latest_day else None,
            # The newest day is the one an agent is most likely to quote as a
            # bare number, and the one most likely to be short (Watch not synced
            # yet). It says how complete it is in the same breath.
            "latest_day_hours_covered": int(latest_day["hours_covered"]) if latest_day else None,
            "latest_day_incomplete": bool(latest_day["incomplete"]) if latest_day else None,
            "average_daily_basal_kcal": avg_daily,
            # The denominator, stated. It is NOT len(daily) and it is NOT the
            # length of the `daily` list below (that one is display-capped), so
            # without this field the two numbers look like they disagree.
            "average_over_days": len(full_days),
            "average_note": (
                f"average over {len(full_days)} complete days of {len(daily)} with any "
                f"data; excludes today and any day with fewer than "
                f"{_BASAL_MIN_HOURS_COVERED} of {_BASAL_HOURS_PER_DAY} hourly buckets. "
                f"`daily` below is capped at "
                f"{'all days' if day_limit is None else f'the newest {day_limit} days'}, "
                f"so its own average will differ from this one."
            ),
            "daily": daily if day_limit is None else daily[:day_limit],
        }
        # `daily` above is DISPLAY-capped by `day_limit` while `average_over_days`
        # is not, so coverage is computed over the uncapped list — the set the
        # averages actually rest on. Reading the capped list here would repeat the
        # sibling bug in Invariant 62, where consuming a display cap as if it were
        # the data reported 60 present days as missing.
        # ⚠️ This is the ONLY reader where the two sets differ, which is exactly why
        # it needs `displayed`: measuring past the cap is right, but saying nothing
        # about the cap left `first_day` naming a day that is nowhere in `daily` and
        # `days_missing_in_span: 0` telling the reader not to look for it. Pass the
        # list off `summary`, not `daily[:day_limit]` — the shipped list is the fact.
        # `requested`/`cut_count` describe `limit`, which caps SAMPLES, not days.
        _attach_coverage(
            summary,
            rows=daily,
            displayed=summary["daily"],
            requested=limit,
            unit="samples",
            cut_count=len(parsed),
            total_available=_avail,
            oldest_raw=_oldest,
        )
        _attach_errors(summary, errors)
        return _attach_owner_guard(summary, records, owner)

    async def total_energy_burned(
        self,
        *,
        days: int = 30,
        owner: str | None = None,
        fresh: bool = False,
    ) -> dict[str, Any]:
        """Return TDEE (basal + active) per day for the last `days` days.

        The truthful daily calorie burn — the number a diet target needs to
        aim below to lose weight. Basal from Watch's `basalEnergyBurned`
        (via `basal_energy_records`), active from `activity.active_energy_kcal`
        (via `activity_summary`). If basal is empty (Vaultbeat hadn't ingested
        the kind yet for the day, or Watch was off), the day's total is
        active-only and flagged `basal_missing=true`.

        A day can also be PARTLY covered, which is the common case and the one
        that used to be invisible: basal arrives as one blob per hour, so a
        Watch left on the charger produces a real-looking row that is short in
        exact proportion to the hours it missed. 2026-08-19 came back as
        1221 kcal (16 of 24 hours) and 2026-08-14 as 883 kcal (12 of 24) —
        both fed straight into the average, pulling it down 178 kcal/day.
        `basal_hours_covered` reports it and `basal_incomplete` flags it; such
        days stay in `days` (a caller asking for a window should see every day
        in it) and are kept out of `average_tdee_kcal`, with the reason listed
        in `average_excluded_days`.

        This matters more than a normal rounding error because there is a
        standing downstream action: a cut is set at `average_tdee_kcal - 500`.
        The contamination is one-directional — a short day can only drag the
        average DOWN — so the errors never cancel out, they accumulate into a
        deficit nobody chose.
        """

        # Pull both underlying streams in parallel — this is a compute
        # aggregation, no new envelope fetches once both caches are warm.
        # day_limit=None (now also the default, kept explicit because this is
        # where it first bit): `basal_energy_records` display-capped `daily` at 30
        # days for its old tool, and consuming that cap here reported `basal_missing`
        # for every day older than 30 — 60 of them on a days=90 query, for data
        # that was present and readable the whole time. That is an
        # Invariant 57 violation (rendering "I truncated the list" as "the Watch
        # was off"), so this asks for everything and does its own windowing.
        basal_task = asyncio.create_task(
            self.basal_energy_records(owner=owner, fresh=fresh, day_limit=None)
        )
        activity_task = asyncio.create_task(self.activity_summary(owner=owner, fresh=fresh))
        basal = await basal_task
        activity = await activity_task

        # Build lookups: day -> kcal
        basal_by_day = {d["day"]: d["basal_kcal"] for d in basal.get("daily", [])}
        basal_hours_by_day = {d["day"]: d.get("hours_covered") for d in basal.get("daily", [])}

        # activity_summary emits {"days": [{day_start_date, active_energy_kcal, ...}]}
        active_by_day: dict[str, float] = {}
        for d in activity.get("days", []):
            raw = d.get("day_start_date") or ""
            # HealthKit activity's day_start_date is an ISO instant of the local-day
            # midnight-in-UTC (e.g. "2026-07-20T16:00:00Z" == 2026-07-21 00:00 CST).
            # Convert to the caller's local calendar day for a clean join.
            try:
                dt = _parse_iso8601(raw)
                day_key = _local_calendar_day(dt).isoformat()
            except (ValueError, KeyError):
                continue
            active_by_day[day_key] = d.get("active_energy_kcal", 0.0)

        # Both upstream reads above are deliberately UNLIMITED, so this union is
        # every day either stream can account for — which makes it the honest
        # denominator for `more_available`. The `[:days]` on the next line is the
        # only cut in this tool, and it is the one the caller needs told about.
        _universe = sorted(set(basal_by_day) | set(active_by_day), reverse=True)
        _avail, _oldest = len(_universe), (_universe[-1] if _universe else None)
        all_days = _universe[:days]

        # Today is a PARTIAL day (its basal/active are still accumulating) —
        # 2026-07-24 at 11:40 it showed 213 kcal and dragged a ~2617 average
        # down to 2136, a distortion the size of an entire diet deficit. It
        # stays in `days` (callers may want the live number) but is flagged
        # and excluded from the average.
        today_key = _local_calendar_day(datetime.now(timezone.utc)).isoformat()

        out = []
        for day in all_days:
            b = basal_by_day.get(day)
            hours = basal_hours_by_day.get(day)
            a = active_by_day.get(day, 0.0)
            total = (b or 0.0) + a
            # A missing day is the most incomplete day there is (0 of 24), so it
            # sets this too — a consumer that only looks at `basal_incomplete`
            # must not accidentally treat "no data at all" as fine. `hours is
            # None` while `b` is present cannot happen (one source), but if it
            # ever did, not flagging is the safe direction: never assert
            # incompleteness we did not measure.
            incomplete = b is None or (
                hours is not None and int(hours) < _BASAL_MIN_HOURS_COVERED
            )
            out.append({
                "day": day,
                "basal_kcal": b,
                "active_kcal": round(a, 1),
                "total_kcal": round(total, 1),
                "basal_missing": b is None,
                "partial": day == today_key,
                # ADD-ONLY (2026-08-20). `basal_missing` and `partial` keep their
                # exact original meanings — "no basal row at all" and "this is
                # today" — and neither ever meant "trustworthy".
                "basal_hours_covered": hours,
                "basal_hours_expected": _BASAL_HOURS_PER_DAY,
                "basal_incomplete": incomplete,
            })

        # Say what was left out, not just how many survived. The caller is an
        # LLM that cannot see the days it did not receive, so an average of
        # 2227.6 is indistinguishable from a wrong one unless the exclusions are
        # named — the same reasoning as Invariant 41 for whole-day writers.
        excluded: list[dict[str, str]] = []
        totals_with_basal: list[float] = []
        for d in out:
            day_str = str(d["day"])
            if d["partial"]:
                excluded.append({"day": day_str, "reason": "partial"})
            elif d["basal_missing"]:
                excluded.append({"day": day_str, "reason": "basal_missing"})
            elif d["basal_incomplete"]:
                excluded.append({"day": day_str, "reason": "basal_incomplete"})
            else:
                totals_with_basal.append(float(d["total_kcal"] or 0.0))
        avg_tdee: float | None = (
            round(sum(totals_with_basal) / len(totals_with_basal), 1)
            if totals_with_basal
            else None
        )

        summary: dict[str, Any] = {
            "days_returned": len(out),
            "average_tdee_kcal": avg_tdee,
            "average_day_count": len(totals_with_basal),
            "average_excluded_days": excluded,
            "average_note": (
                f"average over {len(totals_with_basal)} complete days; excludes today "
                f"(partial, still accumulating), days with no basal data at all, and "
                f"days with fewer than {_BASAL_MIN_HOURS_COVERED} of "
                f"{_BASAL_HOURS_PER_DAY} hourly basal buckets (Watch not worn or not "
                f"synced — its kcal is short in proportion, so it would drag the "
                f"average down). Every exclusion is listed with its reason in "
                f"average_excluded_days."
            ),
            "days": out,
            "basal_errors": basal.get("errors", []),
            "activity_errors": activity.get("errors", []),
        }
        # TDEE joins two readers that each guard against blending two people,
        # but the join discarded their flags — so an unfiltered TDEE on a paired
        # account summed one person's basal with the other's activity and said
        # nothing. Carry the guard across the join.
        for part in (basal, activity):
            if part.get("mixed_owners"):
                for key in ("mixed_owners", "owner_user_id_prefixes", "warning"):
                    summary.setdefault(key, part.get(key))
        # The one reader whose parameter really is a WINDOW: `days=90` asks for 90
        # days. But `all_days` above takes the newest N days THAT HAVE DATA, so 90
        # rows can span seven months — `span_days` is what tells those apart, and
        # `window_satisfied=false` says the history is shorter than the question.
        # Note this counts days with ANY data; `average_day_count` (fewer) is the
        # divisor, and `average_excluded_days` says which ones dropped out.
        _attach_coverage(
            summary, rows=out, requested=days, unit="days",
            total_available=_avail, oldest_raw=_oldest,
        )
        return summary

    async def vo2max_records(self, *, limit: int | None = None, owner: str | None = None, fresh: bool = False) -> dict[str, Any]:
        """Return recent VO2Max samples with a peak / trough summary.

        Motivated by the 2026-11 萨武神山 备训 trend tracking (target 45+ from
        current ~39.3, health.md 训练方案段). Watch computes VO2Max periodically
        during outdoor brisk walk/run bouts — sparse samples over weeks/months,
        so newest first and no artificial gap-filling.
        """

        records, errors = await self._records_for_metric(METRIC_VO2MAX, limit=None, fresh=fresh)
        if owner:
            records = _select_owner(records, owner)
        vo2_list: list[VO2MaxRecord] = []
        for record in records:
            try:
                vo2_list.append(parse_vo2max_record(record.payload, owner_user_id=record.owner_user_id))
            except (KeyError, TypeError, VaultbeatCryptoError, ValueError) as error:
                errors.append(f"{record.envelope_id}: parse_failed ({type(error).__name__}: {error})")
        # Business-time sort before the cut — the old created_at cut only
        # happened to put the newest sample first; a backfill batch breaks
        # that (2026-07-24: vo2max came back visibly shuffled) and `latest`
        # is defined as values[0], so the sort is what makes it truthful.
        vo2_list.sort(key=lambda r: r.date, reverse=True)
        _avail, _oldest = len(vo2_list), (vo2_list[-1].date if vo2_list else None)
        if limit is not None:
            vo2_list = vo2_list[:limit]
        values = [r.vo2_max_ml_kg_min for r in vo2_list]
        latest = values[0] if values else None
        peak = max(values) if values else None
        trough = min(values) if values else None
        average = sum(values) / len(values) if values else None
        summary = {
            "records": [r.to_dict() for r in vo2_list],
            "count": len(vo2_list),
            "latest_ml_kg_min": round(latest, 1) if latest is not None else None,
            "peak_ml_kg_min": round(peak, 1) if peak is not None else None,
            "trough_ml_kg_min": round(trough, 1) if trough is not None else None,
            "average_ml_kg_min": round(average, 1) if average is not None else None,
        }
        # VO2Max is measured only during outdoor brisk bouts, so a wide span with
        # few days is NORMAL here rather than a sync failure — which is exactly
        # why the span has to be visible next to `peak` and `trough`.
        _attach_coverage(summary, rows=summary["records"], requested=limit, unit="samples", total_available=_avail, oldest_raw=_oldest)
        _attach_errors(summary, errors)
        return _attach_owner_guard(summary, records, owner)

    async def user_profile(self, *, owner: str | None = None, fresh: bool = False) -> dict[str, Any]:
        """The owner's Health Profile: biological sex, age, height (GitHub #14).

        One record per user; every device overwrites the same row, so the
        newest upload is the profile. `profile` is None when nothing was
        uploaded — an app older than the feature, Apple Health fields never
        filled in, or reading them not allowed all look the same from here.
        """

        records, errors = await self._records_for_metric(METRIC_PROFILE, limit=None, fresh=fresh)
        if owner:
            records = _select_owner(records, owner)
        parsed: list[tuple[str, ProfileRecord]] = []
        for record in records:
            try:
                parsed.append((str(record.created_at or ""), parse_profile_record(record.payload, owner_user_id=record.owner_user_id)))
            except (KeyError, TypeError, VaultbeatCryptoError, ValueError) as error:
                errors.append(f"{record.envelope_id}: parse_failed ({type(error).__name__}: {error})")
        parsed.sort(key=lambda item: item[0], reverse=True)
        newest = parsed[0][1] if parsed else None
        # #14 also asked for `is_partner_paired`. Deliberately absent: a profile
        # is own-devices-only, so nothing in THIS kind says whether a partner
        # exists. Whether a partner shares something is answered per kind, by
        # reading it with `partner=true` (doctor's `owner_prefixes` answered it
        # until 2026-10-02, at the price of downloading every record).
        # 🚫 No upload timestamp (GitHub #42). The only one available is the
        # row's `created_at`, i.e. the FIRST upload: the upsert RPC's conflict
        # branch rewrites only `ciphertext` / `encryption_version`, and this kind
        # overwrites one row per user forever. Shipped as `uploaded_at` it read
        # as "how current", which an edited profile proves false. A real
        # freshness signal needs a server-maintained column; until then, no
        # field beats a misleading one.
        profile = newest.to_dict(today=date.today()) if newest else None
        summary: dict[str, Any] = {"profile": profile}
        if newest is None:
            summary["note"] = (
                "No health profile has been uploaded. The iOS app uploads it from "
                "Settings → Health Profile once the person has allowed Apple Health "
                "to share sex, date of birth and height, or picked their sex there. "
                "An older app, fields never filled in, and access not allowed all look "
                "the same from here — do not guess which."
            )
        # Every read tool carries the block (test_every_read_tool_reports_coverage).
        # A profile is not a series, so `days_covered` is 0 by construction and
        # `total_available` counts uploaded copies, not days.
        _attach_coverage(summary, rows=[], requested=None, unit="records", total_available=len(parsed))
        _attach_errors(summary, errors)
        return _attach_owner_guard(summary, records, owner)

    async def symptom_summary(
        self, *, limit: int | None = None, fresh: bool = False, owner: str | None = None
    ) -> dict[str, Any]:
        """Return recent symptom days grouped by data owner.

        Symptom blobs only arrive when someone opted in on iOS (own AI or the
        partner-AI ladder). Sensitive — decoded locally, never re-exported.
        """

        # Invariant 38 — see the identical note in strength_summary.
        records, errors = await self._records_for_metric(METRIC_SYMPTOM, limit=None, fresh=fresh)
        if not records:
            _LOG.info("no symptom envelopes present (likely not opted in on iOS)")
        else:
            _LOG.info("decoding %d symptom envelope(s); sensitive, kept local", len(records))
        days: list[SymptomDay] = []
        entries: list[SymptomEntry] = []
        for record in records:
            try:
                if is_symptom_entry_payload(record.payload):
                    entries.append(
                        parse_symptom_entry(record.payload, owner_user_id=record.owner_user_id)
                    )
                else:
                    days.append(parse_symptom_day(record.payload, owner_user_id=record.owner_user_id))
            except (KeyError, TypeError, VaultbeatCryptoError, ValueError) as error:
                errors.append(f"{record.envelope_id}: parse_failed ({type(error).__name__}: {error})")
        # A deleted entry is a tombstone blob; it is nobody's symptom.
        entries = [e for e in entries if not e.deleted]
        # Same owner filter as every other reader (`me` or `!me` for the partner).
        days = _select_owner(days, owner)
        entries = _select_owner(entries, owner)
        days, _avail, _oldest = _cut_newest_by_with_span(days, limit, key=lambda d: d.day_start_date)
        # The two shapes are cut separately on their own business dates
        # (Invariant 38): `limit` HealthKit days AND `limit` reported episodes.
        entries, _e_avail, _e_oldest = _cut_newest_by_with_span(
            entries, limit, key=lambda e: e.sort_key()
        )
        summary = summarize_symptoms(days, entries)
        # Rows live one level down, grouped per owner. Flattened here so coverage
        # describes the days actually reported, which after the per-owner dedup is
        # also what `window_satisfied` should be measured against.
        # ⚠️ With both partners tracking, the span blends two people — read it
        # alongside each owner's own `day_count`.
        day_rows = [d for o in summary["owners"] for d in o["days"]]
        entry_rows = [e for o in summary["owners"] for e in o["reported"]]
        # Oldest day behind EITHER cut, normalised to a bare local day so the two
        # shapes compare (`_oldest` is an ISO instant, an entry's key a day).
        oldest_days = [
            d
            for d in (
                _local_date_fields(_oldest).get("local_date") if _oldest else None,
                _e_oldest.split("|", 1)[0] if _e_oldest else None,
            )
            if d
        ]
        _attach_coverage(
            summary,
            rows=day_rows + entry_rows,
            requested=limit,
            unit="days",
            # Each list was cut to `limit` on its own, so the binding count is the
            # larger of the two — summing them would call two half-full lists a
            # satisfied window.
            cut_count=max(len(day_rows), len(entry_rows)),
            total_available=_avail + _e_avail,
            oldest_raw=min(oldest_days) if oldest_days else None,
        )
        return _attach_errors(summary, errors)

    async def notes_summary(
        self,
        *,
        limit: int | None = None,
        target_kind: str | None = None,
        fresh: bool = False,
        partner: bool | None = None,
    ) -> dict[str, Any]:
        """Return recent free-text notes grouped by target kind.

        Kinds: "sleep"/"menstrual" are written manually in Vaultbeat by either
        partner; "mood"/"general" are agent-authored via `log_note`. Each note
        carries its writer (owner_user_id). Sensitive free text — decoded
        locally, never re-exported.
        """

        # Invariant 38 — see the identical note in strength_summary. Notes cut on
        # target_date (the day the note is ABOUT), matching how summarize_notes
        # orders them; created_at would reorder a backdated note to the front.
        records, errors = await self._records_for_metric(METRIC_NOTE, limit=None, fresh=fresh)
        if not records:
            _LOG.info("no note envelopes present (nothing written or not shared)")
        else:
            _LOG.info("decoding %d note envelope(s); sensitive, kept local", len(records))
        notes: list[NoteRecord] = []
        for record in records:
            try:
                notes.append(parse_note(record.payload, owner_user_id=record.owner_user_id))
            except (KeyError, TypeError, VaultbeatCryptoError, ValueError) as error:
                errors.append(f"{record.envelope_id}: parse_failed ({type(error).__name__}: {error})")
        # Whose notes. `partner=None` keeps the old everyone-grouped read for
        # internal callers. Me = notes I wrote about myself; partner = notes the
        # partner wrote (shared from their app) PLUS notes my AI wrote about
        # them, which live in my account with `about: "partner"`.
        me = self.person_owner() if partner is not None else None
        if me:
            def _mine(n: NoteRecord) -> bool:
                # Case-folded for the reason `_select_owner` gives.
                return bool(n.owner_user_id and n.owner_user_id.lower().startswith(me.lower()))

            if partner:
                notes = [n for n in notes if not _mine(n) or n.about == "partner"]
            else:
                notes = [n for n in notes if _mine(n) and n.about != "partner"]
        # Span facts come from the notes the caller ASKED FOR, not from every
        # note fetched — the one reader where those differ. `target_kind` filters
        # AFTER the cut, so measuring `more_available` over the unfiltered pool
        # reports the caller's own narrowing as hidden history: a `limit=4` read
        # of 2 general + 2 sleep notes gets both sleep notes (everything there
        # is) and would still be told older days were withheld. Same reasoning as
        # the `cut_count` override two lines below, which Invariant 64 already
        # carves out for exactly this tool — a filter is a narrower question, not
        # a shorter history.
        _pool = notes if target_kind is None else [n for n in notes if n.target_kind == target_kind]
        _avail = len(_pool)
        _oldest = min((n.target_date for n in _pool if n.target_date), default=None)
        notes = _cut_newest_by(notes, limit, key=lambda n: n.target_date)
        summary = summarize_notes(notes, target_kind=target_kind)
        # `target_kind` filters AFTER the cut, so the returned rows can be far
        # fewer than `limit` for a query that asked for one kind out of several.
        # `cut_count` is the pre-filter count, or asking for cycle notes only
        # would report "not enough data" every time.
        _attach_coverage(
            summary,
            rows=[n for k in summary["kinds"] for n in k["notes"]],
            requested=limit,
            unit="notes",
            cut_count=len(notes),
            total_available=_avail,
            oldest_raw=_oldest,
        )
        return _attach_errors(summary, errors)

    async def strength_summary(
        self, *, limit: int | None = None, limit_days: int | None = None, fresh: bool = False
    ) -> dict[str, Any]:
        """Return recent strength-training sessions with exercise-level detail.

        Logged manually in Vaultbeat (HealthKit's workout type has no per-set
        data). Owner's own sessions only — strength has no partner fan-out.
        """

        # Invariant 38: fetch EVERYTHING, then cut on the payload's own date.
        # Passing `limit` down would cut on created_at (an upload-batch stamp), and
        # a history backfill uploads years of data in minutes — so upload order and
        # business order are unrelated and the newest session can be dropped outright.
        records, errors = await self._records_for_metric(METRIC_STRENGTH, limit=None, fresh=fresh)
        if not records:
            _LOG.info("no strength envelopes present (nothing logged yet)")
        entries: list[StrengthRecord] = []
        for record in records:
            try:
                entries.append(parse_strength(record.payload, owner_user_id=record.owner_user_id))
            except (KeyError, TypeError, VaultbeatCryptoError, ValueError) as error:
                errors.append(f"{record.envelope_id}: parse_failed ({type(error).__name__}: {error})")
        entries, _avail, _oldest = _cut_newest_by_with_span(entries, limit, key=lambda e: e.date)
        summary = summarize_strength(entries, limit_days=limit_days)
        # Two caps stack: `limit` cuts blobs, then `limit_days` cuts sessions after
        # dedup — report whichever is binding. Coverage counts the sessions the
        # summary actually returned, so a `limit` of 20 blobs that dedups to 18
        # sessions reports 18 and window_satisfied=False rather than claiming the
        # window was met.
        _attach_coverage(
            summary,
            rows=summary["sessions"],
            requested=limit_days if limit_days is not None else limit,
            unit="sessions",
            total_available=_avail,
            oldest_raw=_oldest,
        )
        return _attach_errors(summary, errors)

    async def log_strength_entry(
        self,
        *,
        date: str,
        exercises: list[dict[str, Any]],
        note: str | None = None,
        merge: bool = False,
    ) -> dict[str, Any]:
        """Encrypt and upsert one strength-training session on the owner's behalf.

        `date` is the LOCAL calendar day ("YYYY-MM-DD"). If that day already has
        a session — logged by the app or by a previous agent write — this reuses
        its entryID so the upsert replaces the ciphertext in place (mirrors the
        iOS editor's "editing a day reuses the id" invariant); otherwise a fresh
        opaque entryID is minted. Sealed for the owner (readable in the app's own
        account, though the iOS app does not yet display agent-authored sessions
        — see StrengthOwnBlobPullClient) and for this MCP server (so a later
        `get_strength_log` sees it immediately).

        Two write modes, mirroring `log_food_entry` (2026-07-28):
        - `merge=False` (default, back-compat): the supplied exercises REPLACE
          the day's exercises. `replaced_exercises` in the result names every
          exercise this call removed, so a caller that did not mean to replace
          can see the damage instead of guessing.
        - `merge=True`: the supplied exercises are APPENDED to the day's existing
          ones; an exercise whose name matches an existing one gets its sets
          appended to it (see `_merge_strength_exercises`). Nothing already
          logged can be lost.

        `note=None` LEAVES THE EXISTING NOTE ALONE — pass `note=""` to clear it.
        Until 2026-07-28 omitting the note erased it, which is how 07-27's
        session silently lost `'腿日 + 腹肌'`.

        Requires a bind that carried owner identity through the handshake
        (owner_user_id / owner_public_key_base64 / owner_device_id) — a bind
        from before this feature shipped predates that and must re-bind.
        """

        from datetime import date as _date_type

        requested_day = _date_type.fromisoformat(date)
        normalized_exercises = _normalize_strength_exercises(exercises)

        # Demo mode has no account to write to — see `_demo_write_refusal`.
        # Sits BELOW the argument validation above, so a malformed call still
        # raises its real error and the tool's contract stays observable.
        if self._demo:
            return _demo_write_refusal("log_strength_entry")

        config = self._require_bound_config()
        server_token = config.server_token
        if not (
            server_token
            and config.owner_user_id
            and config.owner_public_key_base64
            and config.owner_device_id
            and config.server_id
        ):
            raise RuntimeError(
                "This bind predates the agent write path (missing owner identity/device). "
                # CLI only, on purpose: 0.9.0 removed the MCP pairing tools this
                # message used to name first (setup does not belong in the tool
                # list), so `bind` in a terminal is the one way to re-pair.
                # `vaultbeat-apple-health` is the console script since the 0.6.2
                # rename; `vaultbeat-mcp` / `vaultbeat-mcp-local` are back-compat
                # aliases (see pyproject.toml [project.scripts]).
                "Re-pair by running `uvx vaultbeat-apple-health@latest bind` in a terminal."
            )

        existing_summary = await self.strength_summary(fresh=True)
        existing_entry_id: str | None = None
        existing_created_at: str | None = None
        existing_session: dict[str, Any] | None = None
        for session in existing_summary.get("sessions", []):
            session_date_raw = session.get("date")
            if not isinstance(session_date_raw, str):
                continue
            try:
                session_day = _local_calendar_day(_parse_iso8601(session_date_raw))
            except ValueError:
                continue
            if session_day == requested_day:
                existing_entry_id = session.get("entry_id")
                existing_created_at = session.get("created_at")
                existing_session = session
                break

        replaced_exercises: list[str] = []
        if merge and existing_session is not None:
            existing_exercises = existing_session.get("exercises")
            normalized_exercises = _merge_strength_exercises(
                existing_exercises if isinstance(existing_exercises, list) else [],
                normalized_exercises,
            )
        elif existing_session is not None:
            # Replace mode on a day that already has data: report exactly what is
            # about to disappear. The caller is an LLM that cannot see the day it
            # is overwriting, and a silent whole-day replace is how 07-27's
            # session lost exercises (see `merge`'s docstring).
            replaced_exercises = [
                str(exercise.get("name"))
                for exercise in (existing_session.get("exercises") or [])
                if isinstance(exercise, dict) and exercise.get("name")
            ]

        cleaned_note = _resolve_note(
            note, existing_session.get("note") if existing_session else None
        )

        now_iso_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        entry_id = existing_entry_id or new_entry_blob_id(METRIC_STRENGTH)
        plaintext_obj = {
            "entryID": entry_id,
            "date": _local_midnight_iso(requested_day),
            "exercises": normalized_exercises,
            "note": cleaned_note,
            "createdAt": existing_created_at or now_iso_utc,
            "updatedAt": now_iso_utc,
        }
        plaintext_bytes = json.dumps(plaintext_obj, ensure_ascii=False).encode("utf-8")

        recipients = [
            RecipientKey(
                recipient_kind="owner_user",
                recipient_id=config.owner_user_id,
                public_key_base64=config.owner_public_key_base64,
            ),
            RecipientKey(
                recipient_kind="mcp_server",
                recipient_id=config.server_id,
                public_key_base64=config.public_key_base64,
            ),
        ]
        ciphertext_base64, sealed_envelopes = encrypt_blob_payload(
            plaintext=plaintext_bytes, recipients=recipients
        )

        blob = {
            "id": entry_id,
            "owner_user_id": config.owner_user_id,
            "source_device_id": config.owner_device_id,
            "metric_type": METRIC_STRENGTH,
            "encryption_version": "v1",
            "ciphertext": ciphertext_base64,
        }
        envelope_rows = [
            {
                "recipient_kind": envelope.recipient_kind,
                "recipient_id": envelope.recipient_id,
                "encrypted_data_key": envelope.encrypted_data_key_base64,
            }
            for envelope in sealed_envelopes
        ]

        server_response = await self._committed(METRIC_STRENGTH, self._client(config).write_strength_blob(
            server_token, blob=blob, envelopes=envelope_rows
        ))

        refreshed = await self.strength_summary(fresh=True)
        session = next(
            (s for s in refreshed.get("sessions", []) if s.get("entry_id") == entry_id), None
        )
        return {
            "entry_id": entry_id,
            "date": requested_day.isoformat(),
            "updated_existing_day": existing_entry_id is not None,
            "merge_mode": merge,
            # Names this call deleted (replace mode over a day that had data).
            # Empty in merge mode and on a fresh day. Surfaced because the caller
            # is an LLM with no view of the day it just overwrote.
            "replaced_exercises": replaced_exercises,
            "server_response": _server_facts(server_response),
            "session": session,
        }

    async def food_summary(
        self,
        *,
        limit: int | None = None,
        limit_days: int | None = None,
        fresh: bool = False,
        since: str | None = None,
        until: str | None = None,
    ) -> dict[str, Any]:
        """Return recent daily food-intake logs with per-day meals + items.

        Logged manually in Vaultbeat (no automatic HealthKit source — HealthKit's
        dietary types are point samples that don't survive as "what a meal actually
        was"). Owner's own days only — food has no partner fan-out in v1.

        `since` / `until` ("YYYY-MM-DD", local calendar days, both inclusive)
        select a window instead of the newest days (GitHub #9). Without them,
        reading a fortnight a month ago meant reading the whole month between,
        which a food log — about 1k tokens a day — does not fit in one result.
        """

        since, until, bad = _parse_day_range(since, until)
        if bad:
            return bad
        # Invariant 38 — see the identical note in strength_summary.
        records, errors = await self._records_for_metric(METRIC_FOOD, limit=None, fresh=fresh)
        if not records:
            _LOG.info("no food envelopes present (nothing logged yet)")
        entries: list[FoodRecord] = []
        for record in records:
            try:
                entries.append(parse_food(record.payload, owner_user_id=record.owner_user_id))
            except (KeyError, TypeError, VaultbeatCryptoError, ValueError) as error:
                errors.append(f"{record.envelope_id}: parse_failed ({type(error).__name__}: {error})")
        # The span facts come from EVERY day, before the window narrows the list,
        # so `more_available` can say that older days exist beyond `since`.
        _, _avail, _oldest = _cut_newest_by_with_span(entries, None, key=lambda e: e.date)
        if since or until:
            # By the LOCAL day the entry is about, the same day `local_date` shows;
            # the wire value is an instant (local midnight in UTC), and comparing
            # it as a string would move every UTC+ user's window by a day.
            def in_window(entry: FoodRecord) -> bool:
                day = str(_local_date_fields(entry.date).get("local_date") or "")
                if not day:
                    return False
                return (since is None or day >= since) and (until is None or day <= until)

            entries = [entry for entry in entries if in_window(entry)]
        entries, _, _ = _cut_newest_by_with_span(entries, limit, key=lambda e: e.date)
        summary = summarize_food(entries, limit_days=limit_days)
        # Same double cap as strength_summary — see the note there.
        _attach_coverage(
            summary,
            rows=summary["days"],
            requested=limit_days if limit_days is not None else limit,
            unit="days",
            total_available=_avail,
            oldest_raw=_oldest,
        )
        _apply_range_coverage(summary, since, until)
        return _attach_errors(summary, errors)

    async def log_food_entry(
        self,
        *,
        date: str,
        meals: list[dict[str, Any]],
        note: str | None = None,
        merge: bool = False,
    ) -> dict[str, Any]:
        """Encrypt and upsert one day's food-intake log on the owner's behalf.

        Same shape / invariants as `log_strength_entry`: `date` is the LOCAL
        calendar day ("YYYY-MM-DD"); an existing day (app- or agent-authored)
        reuses its entryID for an upsert-in-place edit instead of forking a new
        blob; sealed for the owner + this MCP server (no partner, own-AI only).

        Two write modes (2026-07-23 client feedback — replace-only silently ate
        every meal the caller forgot to re-send when "adding one snack"):
        - ``merge=False`` (default, back-compat): the supplied meals REPLACE the
          whole day.
        - ``merge=True``: the supplied meals are APPENDED to the day's existing
          meals (same-name meals get their items appended; see
          ``_merge_food_meals``).

        ``note=None`` keeps the existing day note in BOTH modes (2026-07-28; it
        used to be preserved only under ``merge=True``, so a replace-mode call
        that omitted the note erased it silently). Pass ``note=""`` to clear it.

        Requires a bind that carried owner identity through the handshake — a
        bind from before this feature shipped predates that and must re-bind.
        """

        from datetime import date as _date_type

        requested_day = _date_type.fromisoformat(date)
        normalized_meals = _normalize_food_meals(meals)

        # Demo mode has no account to write to — see `_demo_write_refusal`.
        # Sits BELOW the argument validation above, so a malformed call still
        # raises its real error and the tool's contract stays observable.
        if self._demo:
            return _demo_write_refusal("log_food_entry")

        config = self._require_bound_config()
        server_token = config.server_token
        if not (
            server_token
            and config.owner_user_id
            and config.owner_public_key_base64
            and config.owner_device_id
            and config.server_id
        ):
            raise RuntimeError(
                "This bind predates the agent write path (missing owner identity/device). "
                # CLI only, on purpose: 0.9.0 removed the MCP pairing tools this
                # message used to name first (setup does not belong in the tool
                # list), so `bind` in a terminal is the one way to re-pair.
                # `vaultbeat-apple-health` is the console script since the 0.6.2
                # rename; `vaultbeat-mcp` / `vaultbeat-mcp-local` are back-compat
                # aliases (see pyproject.toml [project.scripts]).
                "Re-pair by running `uvx vaultbeat-apple-health@latest bind` in a terminal."
            )

        existing_summary = await self.food_summary(fresh=True)
        existing_entry_id: str | None = None
        existing_created_at: str | None = None
        existing_day: dict[str, Any] | None = None
        for day in existing_summary.get("days", []):
            day_date_raw = day.get("date")
            if not isinstance(day_date_raw, str):
                continue
            try:
                parsed_day = _local_calendar_day(_parse_iso8601(day_date_raw))
            except ValueError:
                continue
            if parsed_day == requested_day:
                existing_entry_id = day.get("entry_id")
                existing_created_at = day.get("created_at")
                existing_day = day
                break

        replaced_meals: list[str] = []
        if merge and existing_day is not None:
            existing_meals = existing_day.get("meals")
            normalized_meals = _merge_food_meals(
                existing_meals if isinstance(existing_meals, list) else [], normalized_meals
            )
        elif existing_day is not None:
            # Replace mode over a day that already has meals: name what is about
            # to be deleted. Same reasoning as log_strength_entry — the caller
            # cannot see the day it is overwriting.
            replaced_meals = [
                str(meal.get("name") or "(unnamed)")
                for meal in (existing_day.get("meals") or [])
                if isinstance(meal, dict)
            ]

        # Note preservation is NOT merge-only (2026-07-28): omitting the note in
        # replace mode used to erase it too, which is the same silent-loss bug.
        cleaned_note = _resolve_note(note, existing_day.get("note") if existing_day else None)

        now_iso_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        entry_id = existing_entry_id or new_entry_blob_id(METRIC_FOOD)
        plaintext_obj = {
            "entryID": entry_id,
            "date": _local_midnight_iso(requested_day),
            "meals": normalized_meals,
            "note": cleaned_note,
            "createdAt": existing_created_at or now_iso_utc,
            "updatedAt": now_iso_utc,
        }
        plaintext_bytes = json.dumps(plaintext_obj, ensure_ascii=False).encode("utf-8")

        recipients = [
            RecipientKey(
                recipient_kind="owner_user",
                recipient_id=config.owner_user_id,
                public_key_base64=config.owner_public_key_base64,
            ),
            RecipientKey(
                recipient_kind="mcp_server",
                recipient_id=config.server_id,
                public_key_base64=config.public_key_base64,
            ),
        ]
        ciphertext_base64, sealed_envelopes = encrypt_blob_payload(
            plaintext=plaintext_bytes, recipients=recipients
        )

        blob = {
            "id": entry_id,
            "owner_user_id": config.owner_user_id,
            "source_device_id": config.owner_device_id,
            "metric_type": METRIC_FOOD,
            "encryption_version": "v1",
            "ciphertext": ciphertext_base64,
        }
        envelope_rows = [
            {
                "recipient_kind": envelope.recipient_kind,
                "recipient_id": envelope.recipient_id,
                "encrypted_data_key": envelope.encrypted_data_key_base64,
            }
            for envelope in sealed_envelopes
        ]

        server_response = await self._committed(METRIC_FOOD, self._client(config).write_food_blob(
            server_token, blob=blob, envelopes=envelope_rows
        ))

        refreshed = await self.food_summary(fresh=True)
        day = next(
            (d for d in refreshed.get("days", []) if d.get("entry_id") == entry_id), None
        )
        return {
            "entry_id": entry_id,
            "date": requested_day.isoformat(),
            "updated_existing_day": existing_entry_id is not None,
            "merge_mode": merge,
            # Meals this call deleted (replace mode over a day that had data).
            # Empty in merge mode and on a fresh day.
            "replaced_meals": replaced_meals,
            "server_response": _server_facts(server_response),
            "day": day,
        }

    async def log_weight_entry(
        self,
        *,
        weight_kg: float,
        date: str | None = None,
    ) -> dict[str, Any]:
        """Encrypt and upsert one weight blob on the owner's behalf (agent write).

        `weight_kg` = kilograms. `date` = LOCAL calendar day "YYYY-MM-DD" (defaults
        to today). One blob per local day (dayID = `body-{dayStart.epoch}`) so
        re-recording the same day upserts in place — same convention as the iOS
        weight card. Sealed for owner + this MCP server (own AI only).

        ⚠️ CAVEAT: agent-written weight ONLY lands in Vaultbeat cloud + MCP
        (visible to `get_metric` series "weight_kg"). It does NOT write to Apple Health
        (HealthKit is iOS-only). If the owner also wants the number in the
        iPhone Health app, they need to record it manually in the Vaultbeat
        weight card (which does the HealthKit write). Design decision:
        keeping this MCP tool light + read-only wrt HealthKit avoids the
        entire "server-triggered HealthKit push" complexity.

        Requires a bind that carried owner identity through the handshake
        (owner_user_id / owner_public_key_base64 / owner_device_id).
        """

        from datetime import date as _date_type

        requested_day = _date_type.fromisoformat(date) if date else _date_type.today()

        # Demo mode has no account to write to — see `_demo_write_refusal`.
        # Sits BELOW the argument validation above, so a malformed call still
        # raises its real error and the tool's contract stays observable.
        if self._demo:
            return _demo_write_refusal("log_weight_entry")

        config = self._require_bound_config()
        server_token = config.server_token
        if not (
            server_token
            and config.owner_user_id
            and config.owner_public_key_base64
            and config.owner_device_id
            and config.server_id
        ):
            raise RuntimeError(
                "This bind predates the agent write path (missing owner identity/device). "
                # CLI only, on purpose: 0.9.0 removed the MCP pairing tools this
                # message used to name first (setup does not belong in the tool
                # list), so `bind` in a terminal is the one way to re-pair.
                # `vaultbeat-apple-health` is the console script since the 0.6.2
                # rename; `vaultbeat-mcp` / `vaultbeat-mcp-local` are back-compat
                # aliases (see pyproject.toml [project.scripts]).
                "Re-pair by running `uvx vaultbeat-apple-health@latest bind` in a terminal."
            )

        if weight_kg <= 0 or weight_kg > 500:
            raise ValueError(f"weight_kg out of realistic range: {weight_kg}")

        # Match iOS: dayID = "body-{dayStart.epoch}", dayStart = local midnight
        # with that day's offset — a summer day written in winter must land on
        # the id the phone minted in summer, or it becomes a second row.
        day_start_local = _local_midnight(requested_day)
        day_start_epoch = int(day_start_local.timestamp())
        day_id = body_day_blob_id(day_start_epoch)

        # Invariant 41 (no-silent-day-erasure): this rewrites the day's WHOLE blob,
        # so anything the caller did not mention has to be carried over. Body
        # composition arrives from a smart scale via HealthKit and an agent has no
        # way to supply it — while those fields were always null (before iOS began
        # reading them on 2026-08-05) overwriting them was free; now a bare weight
        # log would silently wipe that day's scale reading.
        # Not wrapped in try/except on purpose, matching log_food_entry: failing
        # loudly beats writing a blob that quietly drops data.
        preserved: dict[str, float | None] = {
            "bodyFatPercent": None,
            "bmi": None,
            "leanBodyMassKg": None,
        }
        existing_summary = await self.weight_trend_summary(fresh=True, owner=config.owner_user_id)
        for day in existing_summary.get("days", []):
            if day.get("day_id") != day_id:
                continue
            preserved = {
                "bodyFatPercent": day.get("body_fat_percent"),
                "bmi": day.get("bmi"),
                "leanBodyMassKg": day.get("lean_body_mass_kg"),
            }
            break

        # Which ROW this day lives in. `day_id` above is the day's identity and
        # stays in the payload; the row may carry the remapped id instead (see
        # `body_day_blob_id`), and writing the plain one would then fail as
        # another account's — or, if nobody holds it yet, start a second row
        # for the same day. The read above was fresh, so this hits the cache.
        remapped_id = body_day_blob_id(day_start_epoch, config.owner_user_id)
        own_rows = {
            record.blob_id
            for record in (await self._records_for_metric(METRIC_BODY, limit=None))[0]
            if (record.owner_user_id or "").lower() == config.owner_user_id.lower()
        }
        blob_id = remapped_id if remapped_id in own_rows else day_id

        plaintext_obj = {
            "dayID": day_id,
            "dayStartDate": day_start_local.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
            "weightKg": float(weight_kg),
            # ── Two fields iOS's merge needs, both added 2026-08-07 ──────────
            #
            # `declaredAt` is the merge AXIS. It has to live inside the sealed
            # payload: iOS previously ordered declarations by the blob's
            # plaintext `created_at`, which any service_role holder can bump —
            # letting a server-side actor replay an old authentic weight into
            # the owner's Apple Health and have the app re-upload it as that
            # day's canonical value. Values were never forgeable; the ordering
            # was.
            #
            # `localDate` is the DAY, as the writer meant it. `dayStartDate` is
            # local midnight in THIS machine's timezone, and this machine is
            # wherever the owner put their agent — a UTC VPS logging "today"
            # for a Denver owner produces an instant that is the PREVIOUS day
            # on their phone, so the merge looked for that day's samples under
            # the wrong key and wrote past every conflict guard. A date string
            # has no timezone to disagree about.
            "declaredAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "localDate": requested_day.isoformat(),
            **preserved,
        }
        plaintext_bytes = json.dumps(plaintext_obj, ensure_ascii=False).encode("utf-8")

        recipients = [
            RecipientKey(
                recipient_kind="owner_user",
                recipient_id=config.owner_user_id,
                public_key_base64=config.owner_public_key_base64,
            ),
            RecipientKey(
                recipient_kind="mcp_server",
                recipient_id=config.server_id,
                public_key_base64=config.public_key_base64,
            ),
        ]
        ciphertext_base64, sealed_envelopes = encrypt_blob_payload(
            plaintext=plaintext_bytes, recipients=recipients
        )

        blob = {
            "id": blob_id,
            "owner_user_id": config.owner_user_id,
            "source_device_id": config.owner_device_id,
            "metric_type": METRIC_BODY,
            "encryption_version": "v1",
            "ciphertext": ciphertext_base64,
        }
        envelope_rows = [
            {
                "recipient_kind": envelope.recipient_kind,
                "recipient_id": envelope.recipient_id,
                "encrypted_data_key": envelope.encrypted_data_key_base64,
            }
            for envelope in sealed_envelopes
        ]

        try:
            server_response = await self._committed(METRIC_BODY, self._client(config).write_body_blob(
                server_token, blob=blob, envelopes=envelope_rows
            ))
        except VaultbeatBlobOwnerConflictError:
            if blob["id"] == remapped_id:
                raise
            # Another account recorded this day first, in this time zone
            # (measured 2026-10-02: 10 of 76 body rows carry the remap, across 4
            # accounts). Until this retry, an agent could not log a weight on any
            # such day at all. Same id the app would get; same envelopes.
            blob["id"] = remapped_id
            server_response = await self._committed(METRIC_BODY, self._client(config).write_body_blob(
                server_token, blob=blob, envelopes=envelope_rows
            ))

        # Verify by re-reading (cache-bypass)
        refreshed = await self.weight_trend_summary(fresh=True, limit=30)
        latest = refreshed.get("days", [{}])[0] if refreshed.get("days") else {}
        return {
            "day_id": day_id,
            "date": requested_day.isoformat(),
            "weight_kg": weight_kg,
            # Says what this write carried over rather than overwrote, so a caller
            # can see that the day's scale composition survived (Invariant 41).
            "preserved_composition": preserved,
            "server_response": _server_facts(server_response),
            "latest_after_write": latest,
        }

    async def log_note(
        self,
        *,
        text: str,
        kind: str = "general",
        date: str | None = None,
        merge: bool = False,
        partner: bool = False,
    ) -> dict[str, Any]:
        """Encrypt and upsert one agent-authored note on the owner's behalf.

        `partner=True` (2026-09-23): the note is ABOUT the partner — "she had
        diarrhoea twice this morning". It is still written into the USER's own
        account and sealed to the user + this server only; the partner's account
        is never touched and the partner never receives it. It carries
        `about: "partner"`, which is what keeps it out of the user's own reads
        and puts it into `partner=True` reads. Notes only, deliberately: iOS
        never pulls notes back, so an extra field cannot surface in the app as
        the user's own data — food, strength and body ARE pulled and merged per
        day, and a partner row there would overwrite the user's.

        Fills the gap the 2026-07-23 roadmap entry describes: "今天为什么情绪
        低落 / 发生了什么" narratives had nowhere to live in Vaultbeat (the
        `note` kind's targetKind was only ever written as sleep/menstrual by
        iOS), so they piled up in local markdown where no metric join can see
        them. `kind` is "mood" or "general" — agent-only kinds; iOS's
        VaultbeatNoteTargetKind stores the raw string on the wire and has no
        note pull path, so a new kind is wire-safe by construction. sleep and
        menstrual stay iOS-authored (writing them here would silently coexist
        with an app-authored note for the same day and confuse dedup).

        `date` = LOCAL calendar day "YYYY-MM-DD" (default today). One
        agent-authored note per (kind, local day), so the same kind+day reuses
        the noteID and rewrites that one note. Two write modes, mirroring
        `log_food_entry` / `log_strength_entry` (2026-07-28):
        - `merge=False` (default, back-compat): `text` REPLACES the day's note.
          `replaced_text` in the result carries whatever this call destroyed, so
          a caller that did not mean to replace can see it instead of guessing.
        - `merge=True`: `text` is APPENDED to the existing note for that
          (kind, day), newline-separated. Nothing already logged can be lost.

        Merge matters most for `kind="general"`, which is where symptoms land
        (the owner's standing rule: log any symptom they mention). Discomfort arrives
        in installments across a day — 恶心 at noon, 头晕 at night — so the
        second write of the day is the norm, not the exception. Until
        2026-07-28 this tool had no merge and the docstring pushed
        read-modify-write onto the caller; an agent that had just learned
        `merge=True` from the food/strength tools would reasonably assume the
        same here and silently erase the morning's symptoms.

        Sealed for owner + this MCP server (own AI only, no partner fan-out).
        """

        from datetime import date as _date_type

        if kind not in AGENT_NOTE_KINDS:
            raise ValueError(
                f"unsupported note kind {kind!r}; expected one of {sorted(AGENT_NOTE_KINDS)} "
                "(sleep/menstrual notes are iOS-authored)"
            )
        cleaned_text = text.strip() if isinstance(text, str) else ""
        if not cleaned_text:
            raise ValueError("note text must be a non-empty string")
        requested_day = _date_type.fromisoformat(date) if date else _date_type.today()

        # Demo mode has no account to write to — see `_demo_write_refusal`.
        # Sits BELOW the argument validation above, so a malformed call still
        # raises its real error and the tool's contract stays observable.
        if self._demo:
            return _demo_write_refusal("log_note")

        config = self._require_bound_config()
        server_token = config.server_token
        if not (
            server_token
            and config.owner_user_id
            and config.owner_public_key_base64
            and config.owner_device_id
            and config.server_id
        ):
            raise RuntimeError(
                "This bind predates the agent write path (missing owner identity/device). "
                # CLI only, on purpose: 0.9.0 removed the MCP pairing tools this
                # message used to name first (setup does not belong in the tool
                # list), so `bind` in a terminal is the one way to re-pair.
                # `vaultbeat-apple-health` is the console script since the 0.6.2
                # rename; `vaultbeat-mcp` / `vaultbeat-mcp-local` are back-compat
                # aliases (see pyproject.toml [project.scripts]).
                "Re-pair by running `uvx vaultbeat-apple-health@latest bind` in a terminal."
            )

        # Upsert key: an existing agent-authored note for the same (kind, day).
        existing_note_id: str | None = None
        existing_created_at: str | None = None
        existing_text: str | None = None
        # One note per (kind, day, ABOUT WHOM), and only among notes this account
        # wrote: a note about the partner must never overwrite the user's own
        # note for that day, nor append onto it.
        existing_summary = await self.notes_summary(target_kind=kind, fresh=True)
        wanted_about = "partner" if partner else "self"
        for kind_group in existing_summary.get("kinds", []):
            for note_row in kind_group.get("notes", []):
                if note_row.get("about", "self") != wanted_about:
                    continue
                if not str(note_row.get("owner_user_id") or "").startswith(config.owner_user_id):
                    continue
                raw_date = note_row.get("target_date")
                if not isinstance(raw_date, str):
                    continue
                try:
                    parsed_day = _local_calendar_day(_parse_iso8601(raw_date))
                except ValueError:
                    continue
                if parsed_day == requested_day:
                    existing_note_id = note_row.get("note_id")
                    existing_created_at = note_row.get("created_at")
                    row_text = note_row.get("text")
                    existing_text = (
                        row_text if isinstance(row_text, str) and row_text.strip() else None
                    )
                    break
            if existing_note_id:
                break

        replaced_text: str | None = None
        if merge and existing_text is not None:
            # Append, newline-separated. `parse_note` rejects an empty text, so a
            # decoded existing note always has real content to keep.
            cleaned_text = f"{existing_text}\n{cleaned_text}"
        elif existing_text is not None:
            # Replace mode over a day that already had a note: hand back exactly
            # what is being destroyed. The caller is an LLM that cannot see the
            # note it is overwriting, and silence is why this class of loss stays
            # invisible for weeks (see log_strength_entry's `replaced_exercises`).
            replaced_text = existing_text

        now_iso_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        note_id = existing_note_id or new_entry_blob_id(METRIC_NOTE)
        plaintext_obj = {
            "noteID": note_id,
            "targetKind": kind,
            "targetDate": _local_midnight_iso(requested_day),
            "text": cleaned_text,
            "createdAt": existing_created_at or now_iso_utc,
            "updatedAt": now_iso_utc,
            # Only when true: a note about oneself stays byte-identical to what
            # every earlier version wrote.
            **({"about": "partner"} if partner else {}),
        }
        plaintext_bytes = json.dumps(plaintext_obj, ensure_ascii=False).encode("utf-8")

        recipients = [
            RecipientKey(
                recipient_kind="owner_user",
                recipient_id=config.owner_user_id,
                public_key_base64=config.owner_public_key_base64,
            ),
            RecipientKey(
                recipient_kind="mcp_server",
                recipient_id=config.server_id,
                public_key_base64=config.public_key_base64,
            ),
        ]
        ciphertext_base64, sealed_envelopes = encrypt_blob_payload(
            plaintext=plaintext_bytes, recipients=recipients
        )

        blob = {
            "id": note_id,
            "owner_user_id": config.owner_user_id,
            "source_device_id": config.owner_device_id,
            "metric_type": METRIC_NOTE,
            "encryption_version": "v1",
            "ciphertext": ciphertext_base64,
        }
        envelope_rows = [
            {
                "recipient_kind": envelope.recipient_kind,
                "recipient_id": envelope.recipient_id,
                "encrypted_data_key": envelope.encrypted_data_key_base64,
            }
            for envelope in sealed_envelopes
        ]

        server_response = await self._committed(METRIC_NOTE, self._client(config).write_note_blob(
            server_token, blob=blob, envelopes=envelope_rows
        ))

        refreshed = await self.notes_summary(target_kind=kind, fresh=True)
        written = None
        for kind_group in refreshed.get("kinds", []):
            for note_row in kind_group.get("notes", []):
                if note_row.get("note_id") == note_id:
                    written = note_row
                    break
        return {
            "note_id": note_id,
            "kind": kind,
            "about": "partner" if partner else "self",
            "date": requested_day.isoformat(),
            "updated_existing_note": existing_note_id is not None,
            "merge_mode": merge,
            # The note text this call deleted (replace mode over a day that had
            # one). None in merge mode and on a fresh day. Surfaced because the
            # caller is an LLM with no view of the note it just overwrote.
            "replaced_text": replaced_text,
            "server_response": _server_facts(server_response),
            "note": written,
        }

    # ── Self-reported symptoms (GitHub #3, 2026-09-30) ──────────────────────

    def _require_agent_write_config(self) -> LocalServerConfig:
        """The bound config, provided the bind carried owner identity (Invariant 21 ①)."""

        config = self._require_bound_config()
        if not (
            config.server_token
            and config.owner_user_id
            and config.owner_public_key_base64
            and config.owner_device_id
            and config.server_id
        ):
            raise RuntimeError(
                "This bind predates the agent write path (missing owner identity/device). "
                "Re-pair by running `uvx vaultbeat-apple-health@latest bind` in a terminal."
            )
        return config

    async def _seal_and_write_symptom(
        self, config: LocalServerConfig, plaintext_obj: dict[str, Any]
    ) -> dict[str, Any]:
        """Seal one symptom entry for owner + this server and POST it.

        Same recipients as every agent write (owner_user + this mcp_server, no
        partner). The blob id IS the entry id, so rewriting an entry is an
        upsert of the same row.
        """

        plaintext_bytes = json.dumps(plaintext_obj, ensure_ascii=False).encode("utf-8")
        recipients = [
            RecipientKey(
                recipient_kind="owner_user",
                recipient_id=config.owner_user_id or "",
                public_key_base64=config.owner_public_key_base64 or "",
            ),
            RecipientKey(
                recipient_kind="mcp_server",
                recipient_id=config.server_id or "",
                public_key_base64=config.public_key_base64,
            ),
        ]
        ciphertext_base64, sealed_envelopes = encrypt_blob_payload(
            plaintext=plaintext_bytes, recipients=recipients
        )
        blob = {
            "id": plaintext_obj["entryID"],
            "owner_user_id": config.owner_user_id,
            "source_device_id": config.owner_device_id,
            "metric_type": METRIC_SYMPTOM,
            "encryption_version": "v1",
            "ciphertext": ciphertext_base64,
        }
        envelope_rows = [
            {
                "recipient_kind": envelope.recipient_kind,
                "recipient_id": envelope.recipient_id,
                "encrypted_data_key": envelope.encrypted_data_key_base64,
            }
            for envelope in sealed_envelopes
        ]
        return await self._committed(METRIC_SYMPTOM, self._client(config).write_symptom_blob(
            config.server_token or "", blob=blob, envelopes=envelope_rows
        ))

    async def _find_own_symptom_entry(
        self, entry_id: str, config: LocalServerConfig
    ) -> SymptomEntry:
        """The live (not deleted) reported entry `entry_id`, written by this account."""

        if not isinstance(entry_id, str) or not entry_id.strip():
            raise ValueError("entry_id must be a non-empty string")
        wanted = entry_id.strip()
        records, _errors = await self._records_for_metric(METRIC_SYMPTOM, limit=None, fresh=True)
        me = (config.owner_user_id or "").lower()
        for record in records:
            if not is_symptom_entry_payload(record.payload):
                continue
            try:
                entry = parse_symptom_entry(record.payload, owner_user_id=record.owner_user_id)
            except (KeyError, TypeError, VaultbeatCryptoError, ValueError):
                continue
            if entry.entry_id != wanted:
                continue
            if entry.deleted:
                raise ValueError(f"symptom entry {wanted!r} was already deleted")
            if (entry.owner_user_id or "").lower() != me:
                raise ValueError(f"symptom entry {wanted!r} belongs to another account")
            return entry
        raise ValueError(
            f"no symptom entry {wanted!r} readable by this server — read `entry_id` "
            "from get_symptoms' `reported` list (HealthKit-imported days cannot be edited here)"
        )

    async def _read_back_symptom_entry(self, entry_id: str, owner: str | None) -> dict[str, Any] | None:
        refreshed = await self.symptom_summary(fresh=True, owner=owner)
        for owner_group in refreshed.get("owners", []):
            for row in owner_group.get("reported", []):
                if row.get("entry_id") == entry_id:
                    found: dict[str, Any] = row
                    return found
        return None

    async def log_symptom(
        self,
        *,
        symptom_type: str,
        severity: str = "unspecified",
        onset_at: str | None = None,
        end_at: str | None = None,
        date: str | None = None,
        display_name: str | None = None,
        body_location: str | None = None,
        triggers: list[str] | str | None = None,
        note: str | None = None,
    ) -> dict[str, Any]:
        """Encrypt and write one self-reported symptom episode (agent write).

        One blob per episode (`symptom-` + 16 random bytes, opaque like food and
        note ids — Invariant 53), sealed for owner + this MCP server. Never
        touches a HealthKit-imported day. Validation runs before the demo and
        bind checks so a malformed call reports its real error everywhere.
        """

        now = datetime.now(timezone.utc)
        fields = _validate_symptom_fields(
            symptom_type=symptom_type,
            severity=severity,
            onset_at=onset_at,
            end_at=end_at,
            day=date,
            display_name=display_name,
            body_location=body_location,
            triggers=triggers,
            note=note,
            default_onset=now,
        )

        if self._demo:
            return _demo_write_refusal("log_symptom")
        config = self._require_agent_write_config()

        entry_id = new_entry_blob_id(METRIC_SYMPTOM)
        stamp = _symptom_wire_instant(now)
        payload = _symptom_entry_payload(entry_id, fields, created_at=stamp, updated_at=stamp)
        server_response = await self._seal_and_write_symptom(config, payload)
        return {
            "entry_id": entry_id,
            "server_response": _server_facts(server_response),
            "entry": await self._read_back_symptom_entry(entry_id, config.owner_user_id),
        }

    async def update_symptom(
        self,
        *,
        entry_id: str,
        symptom_type: str | None = None,
        severity: str | None = None,
        onset_at: str | None = None,
        end_at: str | None = None,
        date: str | None = None,
        display_name: str | None = None,
        body_location: str | None = None,
        triggers: list[str] | str | None = None,
        note: str | None = None,
    ) -> dict[str, Any]:
        """Change the given fields of one reported entry; every other field stays.

        `None` = leave as is. `""` clears an optional text field or `end_at`
        (an end logged by mistake), `[]` clears `triggers`. The result carries
        `previous` — the old value of every field this call changed — because the
        caller cannot otherwise see what it overwrote.
        """

        if self._demo:
            return _demo_write_refusal("update_symptom")
        config = self._require_agent_write_config()
        existing = await self._find_own_symptom_entry(entry_id, config)

        def _keep(new: Any, old: Any) -> Any:
            return old if new is None else (new or None)

        onset_value = onset_at if onset_at is not None else (existing.onset_at or existing.local_date)
        fields = _validate_symptom_fields(
            symptom_type=symptom_type if symptom_type is not None else existing.symptom_type,
            severity=severity if severity is not None else existing.severity,
            onset_at=onset_value,
            end_at=_keep(end_at, existing.end_at),
            # A new onset re-derives the day unless the caller also names one.
            day=date if date is not None else (None if onset_at is not None else existing.local_date),
            display_name=_keep(display_name, existing.display_name),
            body_location=_keep(body_location, existing.body_location),
            triggers=list(existing.triggers) if triggers is None else triggers,
            note=_keep(note, existing.note),
            default_onset=None,
        )

        stamp = _symptom_wire_instant(datetime.now(timezone.utc))
        payload = _symptom_entry_payload(
            existing.entry_id, fields, created_at=existing.created_at or stamp, updated_at=stamp
        )
        before = existing.to_dict()
        server_response = await self._seal_and_write_symptom(config, payload)
        after = await self._read_back_symptom_entry(existing.entry_id, config.owner_user_id)
        compared = (
            "symptom_type", "severity", "local_date", "onset_at", "end_at",
            "display_name", "body_location", "triggers", "note",
        )
        previous = {
            key: before.get(key)
            for key in compared
            if after is not None and after.get(key) != before.get(key)
        }
        return {
            "entry_id": existing.entry_id,
            "changed_fields": sorted(previous),
            "previous": previous,
            "server_response": _server_facts(server_response),
            "entry": after,
        }

    async def delete_symptom(self, *, entry_id: str) -> dict[str, Any]:
        """Erase one reported entry by overwriting its blob with a tombstone.

        There is no delete endpoint on the agent write path, and adding one
        would widen what a server token can do. Overwriting instead removes the
        content itself — the ciphertext that held the symptom is replaced by one
        holding only the id — and tells every reader, the app included, that
        the entry is gone. `deleted_entry` hands back what was erased so a
        mistaken delete can be re-logged.
        """

        if self._demo:
            return _demo_write_refusal("delete_symptom")
        config = self._require_agent_write_config()
        existing = await self._find_own_symptom_entry(entry_id, config)
        tombstone = {
            "entryID": existing.entry_id,
            "deleted": True,
            "updatedAt": _symptom_wire_instant(datetime.now(timezone.utc)),
        }
        server_response = await self._seal_and_write_symptom(config, tombstone)
        return {
            "entry_id": existing.entry_id,
            "deleted": True,
            "deleted_entry": existing.to_dict(),
            "server_response": _server_facts(server_response),
        }

    def _probe_cloud(self, api_base_url: str) -> tuple[bool, str]:
        """Reachability probe: any HTTP answer (even 401/405) proves the edge
        is reachable; only transport-level failures count as unreachable.
        Split out so tests can monkeypatch it without a network."""
        import httpx  # lazy — keep the cache-hit CLI start fast

        try:
            response = httpx.get(
                api_base_url.rstrip("/") + "/mcp-sync", timeout=10.0
            )
            return True, f"cloud answered HTTP {response.status_code}"
        except httpx.HTTPError as error:
            return False, f"{type(error).__name__}: {error}"

    def _private_key_location(self) -> str:
        """Which of the three storage layers actually holds the key right now.

        Reports the layer, never the value. Order matches `_keychain_load`, so
        this describes what a read WOULD do rather than what it once did.
        """
        from vaultbeat_mcp_local.store import PRIVATE_KEY_ENV, identity_file_path

        # The keyring branch below is a FALLBACK — it is reached whenever the env
        # var and the file are both absent, without checking that the keyring
        # actually holds anything. That is fine for a real install (a bound one
        # has a key somewhere), but in demo mode it printed "Private key: the
        # system keyring" for an install that has no key at all and decrypts
        # nothing. Small, and still a claim about where a secret lives.
        if self._demo:
            return (
                "nowhere — demo mode holds no private key and decrypts nothing. "
                "The synthetic records are generated in memory on this machine."
            )
        if os.getenv(PRIVATE_KEY_ENV, "").strip():
            return (
                f"injected via {PRIVATE_KEY_ENV} — supplied by whatever started this "
                "process; Vaultbeat never writes it to disk."
            )
        path = identity_file_path(self.store.path)
        if path.is_file():
            return (
                f"PLAINTEXT FILE at {path} (mode 0600). No system keyring was available "
                "when this key was created — normal on a headless server. Anyone who can "
                "read that file can decrypt this account's health data, so treat the "
                "machine's own security as the boundary: full-disk encryption is the "
                "meaningful protection here. To keep the key off disk entirely, set "
                f"{PRIVATE_KEY_ENV} instead."
            )
        return "the system keyring (macOS Keychain / SecretService)."

    def _scope_report(self) -> dict[str, Any]:
        """What `doctor` can see, and what it structurally cannot.

        The half most likely to be broken is the half this process cannot
        inspect. This server runs as a SUBPROCESS of the MCP client (Claude
        Code, Claude Desktop, Hermes, …): the client owns the command line,
        decides which environment variables survive into it, and holds a config
        file this process never sees. So every check can pass while the client
        is launching a different install, pointing at a different config, or
        dropping a variable it does not recognise — and a reader who takes a
        green report as "the whole setup is fine" then looks for the fault
        everywhere except where it is. Recorded cost (2026-08-12): an agent
        spent half an hour guessing environment variables and ended up reading
        the HOST's source for its env allow-list, because doctor had reported
        everything healthy.

        `env_overrides_received` is what turns a disclaimer into something
        actionable — it answers "did the client actually forward this?" without
        anyone reading anyone's source. Names and presence ONLY, never values:
        one of these is a bearer token.
        """
        import os

        from vaultbeat_mcp_local.cache import TTL_ENV, _LEGACY_TTL_ENV
        from vaultbeat_mcp_local.demo import DEMO_ENV
        from vaultbeat_mcp_local.store import CONFIG_ENV, _LEGACY_CONFIG_ENV

        # HAND-MAINTAINED. A new Vaultbeat env var is invisible here until someone
        # adds it — there is no discovery mechanism, and the whole value of this
        # block is answering "did the client forward it?" without reading anyone's
        # source. DEMO_ENV especially: it silently swaps every number this server
        # returns for a synthetic one, so "is it set, and did it survive into this
        # process?" is the first question worth being able to answer.
        names = (
            CONFIG_ENV,
            _LEGACY_CONFIG_ENV,
            TTL_ENV,
            _LEGACY_TTL_ENV,
            DEMO_ENV,
            "VAULTBEAT_MCP_HTTP_TOKEN",
            "TETHER_MCP_HTTP_TOKEN",
        )
        return {
            "covers": (
                "the Vaultbeat side on this machine only: the config file, the identity "
                "key, the binding, cloud reachability, and whether a real "
                "record decrypts end to end."
            ),
            # WHERE the decryption key lives, stated plainly. Not a footnote: on a
            # headless server this key sits in a plaintext 0600 file, and the person
            # running it has a right to know that without reading our source. Names
            # a location and who can read it — never the key itself.
            "private_key_location": self._private_key_location(),
            "does_not_cover": (
                "your MCP client's own configuration. This server runs as a subprocess of "
                "the client, so it cannot read the client's config file, cannot see which "
                "environment variables the client chose to forward, and cannot tell whether "
                "the client launched it with the arguments you think. Every check here can "
                "pass while the client is starting a different install or pointing at a "
                "different config — if the tools are missing, or the data looks like "
                "someone else's, look there next rather than at this report."
            ),
            "env_overrides_received": {
                name: bool(os.getenv(name, "").strip()) for name in names
            },
        }

    # ── Analysis over daily series (arithmetic only — see `analysis.py`) ────
    #
    # These three READ THROUGH the same per-kind methods the tools use rather
    # than querying separately, so a trend can never disagree with the rows
    # `get_<kind>` prints. Two query paths for one number is the shape that lets
    # a product report two different answers for the same question.

    #: How many rows to ask a kind for, per day of window requested.
    #:
    #: Kinds whose `limit` counts SAMPLES (resting_hr, hrv, wrist_temp, vo2max)
    #: need more rows than days or the window comes back short; kinds whose
    #: limit counts days are simply over-fetched, which costs nothing because
    #: every read is served from the same local cache. Over-fetching in one
    #: direction is recoverable, under-fetching silently truncates the window —
    #: so the asymmetric choice is deliberate.
    _SERIES_ROW_MULTIPLIER = 4
    _SERIES_ROW_FLOOR = 30

    #: How many times `_series_points` may double its ask when the rows it got
    #: back filled fewer days than were requested AND arrived at the limit.
    #:
    #: 5 doublings take the first ask (4 rows/day) to 128 rows per day, past any
    #: sampling rate Apple Health produces for these kinds. The cap exists so a
    #: kind that returns a constant number of rows regardless of `limit` — a
    #: bug, but a possible one — cannot spin here; it is not expected to bind.
    _SERIES_MAX_WIDENINGS = 5

    async def _series_points(
        self, spec: Any, *, days: int, owner: str | None, fresh: bool,
        since: str | None = None, until: str | None = None,
    ) -> tuple[dict[str, float], int, dict[str, Any]]:
        """Fetch one series and cut it to the newest `days` calendar days.

        Returns (points, rows_consumed, raw_summary). The cut happens on DAYS
        after bucketing, never on rows before it — cutting rows first is the bug
        the read tools already fixed twice (business-time sort before the cut):
        a kind that records several samples on a busy day would otherwise return
        fewer days than asked while looking like it returned the full window.
        """

        fetch = getattr(self, spec.method, None)
        if fetch is None:  # pragma: no cover — SERIES is checked by a test
            raise VaultbeatUnsupportedMetricError(spec.method)

        if since or until:
            # A calendar window: read the whole history once, keep the days in
            # range. Counting "newest N days with data" cannot express "July"
            # (2026-09-24: comparing a summer with a semester meant hand-picking
            # dates out of a table).
            summary = await fetch(limit=None, owner=owner, fresh=fresh)
            points, consumed = daily_series(summary, spec)
            kept = {
                d: v for d, v in points.items()
                if (since is None or d >= since) and (until is None or d <= until)
            }
            return kept, consumed, summary

        limit = max(days * self._SERIES_ROW_MULTIPLIER, self._SERIES_ROW_FLOOR)
        summary = await fetch(limit=limit, owner=owner, fresh=fresh)
        points, consumed = daily_series(summary, spec)

        # 🔴 Widen and re-ask while the window is SHORT and the rows arrived at
        # the ceiling. Both conditions are needed and they mean different things:
        #   · fewer days than asked, on its own, is the normal answer for a kind
        #     the Watch measures occasionally — widening then would just re-read
        #     the same rows.
        #   · rows landing exactly at `limit` is what says the server had more
        #     to give. Fewer than `limit` means the history genuinely ended.
        # Measured before this loop existed: a kind with 6 samples a day, asked
        # for 30 days against 60 days of stored history, returned 20 days with
        # `window_satisfied: false` — which is TRUE but reads as the wrong
        # thing, because the coverage note explains that flag as "that is all
        # the history this server holds … re-check after a re-sync". So the
        # trend was computed over a third of the requested span while the agent
        # was told to send the user to re-sync data that was already there.
        #
        # ⚠️ "Arrived at the ceiling" is asked of the READ METHOD first, through
        # its `coverage.more_available` (`_fetch_was_cut`): only the method knows
        # what its `limit` counts, and basal energy counts hourly samples while
        # returning days. Raw rows are only the fallback for a reader that does
        # not say — and even then never `consumed`: `daily_series` drops rows
        # with no day, a non-numeric field, a bool or a NaN, so `consumed` can
        # sit below `limit` on a response that was in fact truncated.
        #
        # `fresh` is not repeated: the first ask already refreshed the cache if
        # it was going to, and every widening after that is served locally.
        for _ in range(self._SERIES_MAX_WIDENINGS):
            if len(points) >= days or not _fetch_was_cut(summary, spec, limit):
                break
            limit *= 2
            summary = await fetch(limit=limit, owner=owner, fresh=False)
            points, consumed = daily_series(summary, spec)

        newest = sorted(points)[-days:] if days > 0 else []
        return {d: points[d] for d in newest}, consumed, summary

    async def _analysis_points(
        self, spec: Any, *, days: int, owner: str | None, fresh: bool,
        since: str | None = None, until: str | None = None,
    ) -> tuple[dict[str, float], int, dict[str, Any], str | None]:
        """`_series_points` for the analysis tools: today left out of accruing series.

        `get_metric` has always kept today off every aggregate of a cumulative
        series (it is shown, marked `partial`). Trend, compare and correlate
        read the same points and did not, so at 9am today's 800 steps sat in
        compare's "recent" window and dragged its mean down — the "steps are
        down today" sentence the partial marker exists to prevent, and a number
        that disagreed with `get_metric` over the same data. One extra day is
        fetched so dropping today still leaves `days` complete days.

        Returns the dropped day (or None) so the caller can name it in
        `excluded_days` with the same `partial_today` reason `get_metric` uses.
        """

        if since or until:
            points, consumed, raw = await self._series_points(
                spec, days=days, owner=owner, fresh=fresh, since=since, until=until
            )
            if not spec.cumulative:
                return points, consumed, raw, None
            today = _local_calendar_day(datetime.now(timezone.utc)).isoformat()
            dropped = today if today in points else None
            return {d: v for d, v in points.items() if d != today}, consumed, raw, dropped
        if not spec.cumulative:
            points, consumed, raw = await self._series_points(spec, days=days, owner=owner, fresh=fresh)
            return points, consumed, raw, None
        today = _local_calendar_day(datetime.now(timezone.utc)).isoformat()
        points, consumed, raw = await self._series_points(spec, days=days + 1, owner=owner, fresh=fresh)
        dropped = today if today in points else None
        complete = {d: v for d, v in points.items() if d != today}
        newest = sorted(complete)[-days:] if days > 0 else []
        return {d: complete[d] for d in newest}, consumed, raw, dropped

    @staticmethod
    def _unknown_series(name: str) -> dict[str, Any]:
        """A wrong series name answered with the right ones, not with a stack trace.

        Raising would reach the agent as one sentence with no way forward, and
        the most likely next move is another guess. The catalog costs a few
        hundred bytes on a path that only runs when someone is already lost.
        """

        return {
            "error": "unknown_series",
            "requested": name,
            "message": (
                f"No series named {name!r}. Pick one of `available_series` below "
                "(`list_metric_series` also says how much data backs each). Kinds that "
                "are not one number per day — workouts, strength, food, notes, symptoms, "
                "cycle, the sex/age/height profile — have their own tools; sleep nights as a table are "
                "`get_sleep_nights`."
            ),
            "available_series": series_catalog(),
        }

    async def _prefetch_series_kinds(self, names: list[str], *, owner: str | None, fresh: bool) -> None:
        """Warm the cache for every kind behind `names`, a few at a time.

        `metric_values` reads its series one by one; cold, that summed each kind's
        download (every series at once was 50 s on one account). Fetching the kinds
        concurrently first leaves that loop reading the local cache. Skipped for
        `fresh` (the loop's own fresh read would download each kind a second time)
        and for a single kind (nothing to overlap). Failures are left to the loop,
        which reports them per series.
        """
        methods = list(dict.fromkeys(spec.method for spec in map(series_lookup, names) if spec is not None))
        # With the cache off (TTL 0) a warmed kind is not kept, and the loop would
        # download it a second time.
        if fresh or len(methods) < 2 or self.cache.ttl_seconds <= 0:
            return
        gate = asyncio.Semaphore(_CONCURRENT_KIND_READS)

        async def warm(method: str) -> None:
            async with gate:
                try:
                    await getattr(self, method)(limit=None, owner=owner, fresh=False)
                except Exception:  # noqa: BLE001 — reported per series by the caller
                    pass

        await asyncio.gather(*(warm(method) for method in methods))

    async def series_overview(
        self, *, owner: str | None = None, fresh: bool = False
    ) -> list[dict[str, Any]]:
        """Every series with how much data actually backs it, not just its name.

        The catalog on its own answers "what may I ask for". It cannot answer the
        question an agent asks first — "is there anything there" — so the only way
        to find out a kind is empty used to be reading it and getting nothing back.
        That is one round trip per kind, and it is the wrong shape entirely once a
        second data source lands and the list stops being seventeen names long.

        One read per backing method, not one per series — activity alone carries
        five series out of the same response — then each series is counted with
        `daily_series`, the function `get_metric` reads it with. `rows` is
        therefore the days that have a value for THAT series, and `first_date` /
        `last_date` are its own extent. (Until 2026-10-02 every series of a kind
        reported the kind's row count and coverage dates: a partner's BMI said 14
        rows while 2 days had one.) Dates come from the read methods, which parse
        each payload for its own business date — not from `created_at`, an
        upload-batch timestamp a backfill makes meaningless (Invariant 38).
        """

        by_method: dict[str, list[Any]] = {}
        for spec in SERIES:
            by_method.setdefault(spec.method, []).append(spec)

        # Concurrently, a few at a time: each read is mostly a wait on the cloud,
        # and back to back they summed to 170 s cold (see
        # `sync_decrypted_records`). Results are consumed in catalog order.
        gate = asyncio.Semaphore(_CONCURRENT_KIND_READS)

        async def read(method: str) -> Any:
            async with gate:
                try:
                    return await getattr(self, method)(limit=None, owner=owner, fresh=fresh)
                except Exception as error:  # noqa: BLE001 — reported per row below
                    return error

        results = await asyncio.gather(*(read(method) for method in by_method))

        out: list[dict[str, Any]] = []
        for (method, specs), result in zip(by_method.items(), results, strict=True):
            if isinstance(result, Exception):
                error = result
                # One unreadable kind must not blank the whole catalog: the agent
                # still needs the other fourteen names to work with, and "this one
                # could not be read" is itself the answer for this row.
                for spec in specs:
                    out.append(
                        {
                            "series": spec.name,
                            "unit": spec.unit,
                            "cumulative": spec.cumulative,
                            "available": False,
                            "reason": str(error)[:200],
                            **({"note": spec.direction_note} if spec.direction_note else {}),
                        }
                    )
                continue

            # Counted PER SERIES with the function `get_metric` reads them with.
            # One count per kind said 14 rows for a partner's BMI and body fat
            # because 14 body days existed, while 2 had a BMI and none a body fat —
            # `get_metric` then returned 0-2 points for a series "with 14 rows"
            # (pre-release review, 2026-10-02).
            for spec in specs:
                # A binding that cannot tell its owner apart averages two people
                # into each day here exactly as `get_metric` would — which refuses
                # (Invariant 87). The catalog has to refuse too, or its `latest`
                # is a number true of neither (82.9 kg and 39.5 kg → 61.2; review
                # R7 on #9).
                if (refusal := _mixed_owner_refusal(spec, result, owner)) is not None:
                    out.append({
                        "series": spec.name, "unit": spec.unit, "cumulative": spec.cumulative,
                        "available": False, "error": refusal["error"],
                        "owner_user_id_prefixes": refusal["owner_user_id_prefixes"],
                        "reason": refusal["message"],
                    })
                    continue
                points, _consumed = daily_series(result, spec)
                days = sorted(points)
                out.append(
                    {
                        "series": spec.name,
                        "unit": spec.unit,
                        "cumulative": spec.cumulative,
                        "available": bool(days),
                        "rows": len(days),
                        "first_date": days[0] if days else None,
                        "last_date": days[-1] if days else None,
                        "latest": points[days[-1]] if days else None,
                        **({"note": spec.direction_note} if spec.direction_note else {}),
                    }
                )

        out.sort(key=lambda row: str(row.get("series")))
        return out

    async def metric_trend(
        self, *, series: str, days: int = 30, owner: str | None = None, fresh: bool = False,
        since: str | None = None, until: str | None = None,
    ) -> dict[str, Any]:
        spec = series_lookup(series)
        if spec is None:
            return self._unknown_series(series)
        since, until, bad = _parse_day_range(since, until)
        if bad:
            return bad
        points, consumed, raw, partial = await self._analysis_points(
            spec, days=days, owner=owner, fresh=fresh, since=since, until=until
        )
        if (refusal := _mixed_owner_refusal(spec, raw, owner)) is not None:
            return refusal
        result = trend(points, spec)
        result["requested_days"] = days
        result["rows_consumed"] = consumed
        _attach_series_exclusions(result, points, raw, spec, partial_today=partial)
        _attach_sources(result, points, raw)
        result["note"] = _SERIES_NOTE
        _attach_series_coverage(result, points, requested=days, source=raw)
        _apply_range_coverage(result, since, until)
        return result

    async def metric_compare_periods(
        self, *, series: str, days: int = 7, owner: str | None = None, fresh: bool = False,
        period: str | None = None, baseline: str | None = None,
    ) -> dict[str, Any]:
        """Compare the newest `days` days against the `days` immediately before them.

        "Immediately before" means the next-newest days WITH DATA, not the
        preceding calendar block — this layer never invents a day, so a gap
        makes the previous window older rather than emptier. `previous.first_day`
        and `previous.last_day` are what say which it was, and they ship on every
        response for exactly that reason.
        """

        spec = series_lookup(series)
        if spec is None:
            return self._unknown_series(series)
        if period or baseline:
            return await self._compare_explicit_periods(
                spec, period=period, baseline=baseline, owner=owner, fresh=fresh
            )
        points, consumed, raw, partial = await self._analysis_points(
            spec, days=days * 2, owner=owner, fresh=fresh
        )
        if (refusal := _mixed_owner_refusal(spec, raw, owner)) is not None:
            return refusal
        ordered = sorted(points)
        recent_days = ordered[-days:]
        previous_days = ordered[: -days][-days:] if len(ordered) > days else []
        result = compare_periods(
            {d: points[d] for d in recent_days},
            {d: points[d] for d in previous_days},
            spec,
        )
        result["requested_days_per_window"] = days
        result["rows_consumed"] = consumed
        _attach_series_exclusions(result, points, raw, spec, partial_today=partial)
        _attach_sources(result, {d: points[d] for d in recent_days + previous_days}, raw)
        result["note"] = _SERIES_NOTE
        # Over BOTH windows, and `requested` is therefore `days * 2`: the block
        # has to describe the set the comparison rests on, and a comparison
        # rests on both halves. Each half's own day count is inside its block.
        _attach_series_coverage(
            result,
            {d: points[d] for d in recent_days + previous_days},
            requested=days * 2,
            source=raw,
        )
        return result

    async def _compare_explicit_periods(
        self, spec: Any, *, period: str | None, baseline: str | None,
        owner: str | None, fresh: bool,
    ) -> dict[str, Any]:
        """Two named calendar windows — "this semester vs the summer" — side by side."""

        if not (period and baseline):
            return {"error": "invalid_periods",
                    "message": 'Pass BOTH `period` and `baseline` as "YYYY-MM-DD..YYYY-MM-DD".'}
        p_since, p_until, bad = _parse_period(period, "period")
        if bad:
            return bad
        b_since, b_until, bad = _parse_period(baseline, "baseline")
        if bad:
            return bad
        recent, consumed, raw, partial = await self._analysis_points(
            spec, days=0, owner=owner, fresh=fresh, since=p_since, until=p_until
        )
        if (refusal := _mixed_owner_refusal(spec, raw, owner)) is not None:
            return refusal
        earlier = {
            d: v for d, v in daily_series(raw, spec)[0].items()
            if (b_since is None or d >= b_since) and (b_until is None or d <= b_until)
        }
        result = compare_periods(recent, earlier, spec)
        result["period"] = {"since": p_since, "until": p_until}
        result["baseline"] = {"since": b_since, "until": b_until}
        result["rows_consumed"] = consumed
        both = {**earlier, **recent}
        _attach_series_exclusions(result, both, raw, spec, partial_today=partial)
        _attach_sources(result, both, raw)
        result["note"] = _SERIES_NOTE
        _attach_series_coverage(result, both, requested=None, source=raw)
        # The span both windows cover: an open end on EITHER side keeps it open.
        # min()/max() over the non-None ends dropped an open `until`, so
        # "2026-09-01.." vs "July-August" reported a range ending 08-31.
        span_since = None if None in (p_since, b_since) else min(p_since, b_since)  # type: ignore[type-var]
        span_until = None if None in (p_until, b_until) else max(p_until, b_until)  # type: ignore[type-var]
        _apply_range_coverage(result, span_since, span_until)
        if span_since is None and span_until is None:
            result["range"] = {"since": None, "until": None}
        return result

    async def metric_correlate(
        self,
        *,
        series_a: str,
        series_b: str,
        days: int = 30,
        owner: str | None = None,
        fresh: bool = False,
        since: str | None = None,
        until: str | None = None,
        lag_days: int = 0,
    ) -> dict[str, Any]:
        since, until, bad = _parse_day_range(since, until)
        if bad:
            return bad
        if not -30 <= lag_days <= 30:
            return {"error": "invalid_lag_days", "requested": lag_days,
                    "message": "`lag_days` must be between -30 and 30."}
        spec_a = series_lookup(series_a)
        if spec_a is None:
            return self._unknown_series(series_a)
        spec_b = series_lookup(series_b)
        if spec_b is None:
            return self._unknown_series(series_b)
        a_points, a_rows, a_raw, a_partial = await self._analysis_points(
            spec_a, days=days, owner=owner, fresh=fresh, since=since, until=until
        )
        b_points, b_rows, b_raw, b_partial = await self._analysis_points(
            spec_b, days=days + abs(lag_days), owner=owner, fresh=fresh,
            since=since,
            until=(date.fromisoformat(until) + timedelta(days=max(lag_days, 0))).isoformat()
            if until else None,
        )
        if lag_days:
            # Pair a on day D with b on day D + lag: "a short night → HRV two
            # days later" is lag_days=2. Re-keyed onto a's calendar so the
            # shared-day logic below needs no second path.
            b_points = {
                (date.fromisoformat(d) - timedelta(days=lag_days)).isoformat(): v
                for d, v in b_points.items()
            }
        for spec_x, raw_x in ((spec_a, a_raw), (spec_b, b_raw)):
            if (refusal := _mixed_owner_refusal(spec_x, raw_x, owner)) is not None:
                return refusal
        result = correlate(a_points, b_points, spec_a, spec_b)
        if partial := (a_partial or b_partial):
            result["excluded_days"] = [{"day": partial, "reason": "partial_today"}]
        result["requested_days"] = days
        result["rows_consumed"] = {"a": a_rows, "b": b_rows}
        result["note"] = _SERIES_NOTE
        # The SHARED days, not the union: those are the only days the
        # coefficient is computed over, and Invariant 62 asks that coverage
        # describe the set the numbers come from rather than the set that was
        # fetched. A union here would report 30 days behind an r built on 11.
        shared = {d: a_points[d] for d in sorted(set(a_points) & set(b_points))}
        if lag_days:
            result["lag_days"] = lag_days
            result["lag_note"] = (
                f"`{series_a}` on each day is paired with `{series_b}` {abs(lag_days)} "
                f"day(s) {'later' if lag_days > 0 else 'earlier'}."
            )
        _attach_sources(result, shared, a_raw)
        _attach_series_coverage(result, shared, requested=days, source=a_raw)
        _apply_range_coverage(result, since, until)
        return result

    # ── Generic reader: get_metric / get_intraday ──────────────────────────
    #
    # One tool for every one-number-per-day quantity instead of one tool per kind
    # (owner, 2026-09-22: more data sources are coming, and the per-kind shape
    # costs a tool, a prompt reference and a doc row for every one of them).
    # It reads through `_series_points` — the same path trend/compare/correlate
    # use — so a value printed here can never disagree with an analysis of it.

    def person_owner(self, *, partner: bool = False) -> str | None:
        """Turn "me" / "my partner" into the `owner` filter the readers take.

        Me = the account that paired this machine. Partner = every OTHER owner in
        the data (see `NOT_OWNER_MARK`). Returns None only when the binding does
        not know whose account it is (a pre-2026 binding) — the readers then fall
        back to their unfiltered behaviour, where the mixed-owner guard still
        refuses or warns rather than blending two people silently.
        """

        if self._demo:
            from vaultbeat_mcp_local.demo import DEMO_OWNER_ID

            me: str | None = DEMO_OWNER_ID
        else:
            config = self.store.load()
            me = config.owner_user_id if config else None
        if not me:
            return None
        return f"{NOT_OWNER_MARK}{me}" if partner else me

    async def total_energy_records(
        self, *, limit: int | None = None, owner: str | None = None, fresh: bool = False
    ) -> dict[str, Any]:
        """`total_energy_burned` under the `limit=` signature every series method shares.

        TDEE's own parameter is a real window (`days`), which is what `limit`
        means for the day-keyed kinds anyway; this only renames it so the series
        table can treat it like every other method.
        """

        # `None` means "all of it" for every other reader; here it used to mean
        # 30 days, so a calendar-window read of TDEE silently lost the rest.
        return await self.total_energy_burned(days=limit or _ALL_DAYS, owner=owner, fresh=fresh)

    async def metric_values(
        self,
        *,
        series: str | list[str] | None = None,
        days: int = 30,
        aggregation: str = "none",
        granularity: str = "day",
        owner: str | None = None,
        fresh: bool = False,
        since: str | None = None,
        until: str | None = None,
    ) -> dict[str, Any]:
        """Per-day values for one, several or all series, optionally aggregated server-side."""

        since, until, bad = _parse_day_range(since, until)
        if bad:
            return bad

        if aggregation not in _AGGREGATIONS:
            return {
                "error": "invalid_aggregation",
                "requested": aggregation,
                "allowed": list(_AGGREGATIONS),
            }
        if granularity not in _GRANULARITIES:
            return {
                "error": "invalid_granularity",
                "requested": granularity,
                "allowed": list(_GRANULARITIES),
            }
        if days < 1:
            return {"error": "invalid_days", "requested": days, "message": "`days` must be at least 1."}

        if series is None:
            names = [spec.name for spec in SERIES]
        elif isinstance(series, str):
            names = [series]
        else:
            names = list(dict.fromkeys(series))

        await self._prefetch_series_kinds(names, owner=owner, fresh=fresh)

        today = _local_calendar_day(datetime.now(timezone.utc)).isoformat()
        refreshed: set[str] = set()
        metrics: list[dict[str, Any]] = []
        unknown = False
        for name in names:
            spec = series_lookup(name)
            if spec is None:
                unknown = True
                metrics.append({"series": name, "error": "unknown_series"})
                continue
            # Refresh each BACKING METHOD once: activity alone backs five series,
            # and a fresh read per series would pull the same kind five times.
            want_fresh = fresh and spec.method not in refreshed
            refreshed.add(spec.method)
            try:
                points, consumed, raw = await self._series_points(
                    spec, days=days, owner=owner, fresh=want_fresh, since=since, until=until
                )
            except Exception as error:  # noqa: BLE001
                # One unreadable kind must not blank the others the agent asked for.
                metrics.append(
                    {"series": name, "error": "read_failed", "reason": str(error)[:300]}
                )
                continue
            refusal = _mixed_owner_refusal(spec, raw, owner)
            if refusal is not None:
                metrics.append(refusal)
                continue
            entry = _metric_entry(
                spec,
                points,
                raw,
                consumed=consumed,
                days=days,
                aggregation=aggregation,
                granularity=granularity,
                today=today,
            )
            _apply_range_coverage(entry, since, until)
            # "Not shared" only when the partner's kind came back with no rows at
            # all. Rows present but no points means the kind IS shared and this
            # field was never measured — body composition on a partner who weighs
            # in on a plain scale (seen on the owner's account 2026-09-23: 14 days
            # of her weight, zero of body fat). Calling that "not shared" would
            # send the user to a sharing switch that is already on.
            if (
                not points
                and owner
                and owner.startswith(NOT_OWNER_MARK)
                and _rows_returned(raw, spec) == 0
            ):
                entry["hint"] = PARTNER_EMPTY_HINT
            metrics.append(entry)

        # The coverage note is identical on every series and is ~2.5k characters;
        # repeated per series it was 40k of a 66k reply for "every series, 30
        # days" (measured on a real account 2026-09-23). Said once, here, and
        # stripped from each block — the fields stay per series because they
        # genuinely differ, the explanation of them does not.
        for entry in metrics:
            block = entry.get("coverage")
            if isinstance(block, dict):
                block.pop("note", None)
        # Same reasoning for `excluded_days`: several sleep series drop the same
        # unworn nights, and a four-series read printed one 25-day list four
        # times (2026-09-24). A day excluded for the same reason from EVERY
        # series that has a list is said once; each series keeps only its own.
        common = _hoist_common_exclusions(metrics)
        # And the source note: it is one paragraph, identical on every sleep
        # series; `sources` stays per series because the night counts differ.
        source_note = None
        for entry in metrics:
            source_note = entry.pop("source_note", None) or source_note
        result: dict[str, Any] = {
            "metrics": metrics,
            "requested_days": days,
            "aggregation": aggregation,
            "granularity": granularity,
            "note": _METRIC_NOTE,
            "coverage_note": _COVERAGE_NOTE,
        }
        if common:
            result["excluded_days_all_series"] = common
        if source_note:
            result["source_note"] = source_note
        if since or until:
            result["range"] = {"since": since, "until": until}
        if unknown:
            result["available_series"] = series_catalog()
        return result

    async def intraday_values(
        self,
        *,
        series: str = "hrv_sdnn",
        granularity: str = "hourly",
        limit: int = 168,
        owner: str | None = None,
        fresh: bool = False,
    ) -> dict[str, Any]:
        """Sub-daily samples for the kinds that record several per day."""

        if series not in _INTRADAY_SERIES:
            return {
                "error": "unknown_intraday_series",
                "requested": series,
                "available": sorted(_INTRADAY_SERIES),
                "message": (
                    "Only the listed series keep sub-daily samples. For one number per "
                    "day use `get_metric`."
                ),
            }
        if granularity not in ("hourly", "raw"):
            return {
                "error": "invalid_granularity",
                "requested": granularity,
                "allowed": ["hourly", "raw"],
            }

        if granularity == "raw":
            return await self.hrv_records(limit=limit, owner=owner, fresh=fresh)

        # Hourly first, raw as the fallback. `hrv_hourly` only started being
        # written by iOS build 77 (2026-07-22): an account on an older app has
        # plenty of raw HRV and zero hourly buckets, and serving the empty hourly
        # result would report "no HRV" to someone whose HRV is right there. App
        # and MCP versions drift independently and permanently, so the aggregate
        # kind degrades to the kind it aggregates instead of pretending absence.
        hourly = await self.hrv_hourly_records(limit=limit, owner=owner, fresh=fresh)
        if hourly.get("records"):
            return hourly
        raw = await self.hrv_records(limit=limit, owner=owner, fresh=fresh)
        if not raw.get("records"):
            return hourly
        raw["granularity"] = "raw"
        raw["granularity_note"] = (
            "Requested hourly averages, but this account has none — hourly HRV "
            "requires the iOS app from 2026-07-22 or later. Returned raw per-sample "
            "HRV instead; the numbers are the same measurements, just not hour-averaged."
        )
        return raw

    async def doctor(self) -> dict[str, Any]:
        """Aggregated self-diagnosis for the install/binding first mile
        (roadmap v1.2.1 "绑定失败自诊断"). Returns machine-readable checks;
        the CLI renders them as an [OK]/[FAIL] list with hints."""
        checks: list[dict[str, Any]] = []

        def add(name: str, ok: bool, detail: str, hint: str | None = None) -> None:
            check: dict[str, Any] = {"name": name, "ok": ok, "detail": detail}
            if hint and not ok:
                check["hint"] = hint
            checks.append(check)

        # ── Demo mode: a different report, not a decorated one ───────────────
        #
        # First statement, before `store.load()`, because a demo install usually
        # has no config at all and the real path bails out there with
        # `ok: False, "no config"` — which reads as "broken" for a mode that is
        # working exactly as designed.
        #
        # 🔴 The rule this branch exists to obey: report what was ACTUALLY
        # checked, and name what was not. The tempting shape is to keep the
        # real check list and mark `data_roundtrip` OK — the report then looks
        # healthy while no round trip was attempted, and a reader takes a green
        # tick as evidence about a network path nobody exercised. Same family as
        # the Product Positioning rule: state what was observed, never what was
        # inferred. So the skipped checks move to `not_checked`, out of the
        # OK/FAIL list a client renders, and each says why.
        from vaultbeat_mcp_local.demo import (
            DEMO_ANCHOR_DAY,
            DEMO_BANNER,
            DEMO_ENV,
            demo_records,
            missing_kinds,
            stale_kinds,
        )

        if self._demo:
            uncovered = sorted(missing_kinds())
            retired = sorted(stale_kinds())
            add(
                "demo_mode",
                True,
                f"{DEMO_ENV} is set — every number this server returns is synthetic, "
                f"generated locally, anchored at {DEMO_ANCHOR_DAY}. No account, no "
                f"network, no decryption.",
            )
            add(
                "demo_dataset",
                not uncovered and not retired,
                f"{len(demo_records())} synthetic records"
                + (f"; kinds with no dataset: {uncovered}" if uncovered else "")
                + (f"; datasets for retired kinds: {retired}" if retired else ""),
                hint="A known metric kind has no demo dataset, so its tool will "
                "return empty and read as broken. Add a builder in demo.py.",
            )
            return {
                # Banner first, ahead of even `ok` — a green `ok: true` is
                # exactly the shape a reader stops reading after, and here it
                # means "the demo dataset is intact", not "your install is fine".
                "demo_warning": DEMO_BANNER,
                "demo_mode": True,
                "ok": all(check["ok"] for check in checks),
                "checks": checks,
                # Named individually rather than summarised, because these are the
                # exact rows a reader expects to find and will otherwise hunt for.
                "not_checked": {
                    "config": "demo mode reads no config file",
                    "identity_key": "demo mode needs no private key — nothing is decrypted",
                    "cloud_reachable": "no request was made; demo mode is fully offline",
                    "bound": "demo mode is deliberately unbound",
                    "data_roundtrip": (
                        "NOT ATTEMPTED. No blob was fetched and none was decrypted, so "
                        "this report says nothing about whether the cloud path works."
                    ),
                    "client_version": (
                        "skipped so demo mode makes no network calls at all; run without "
                        f"{DEMO_ENV} to check for a newer release"
                    ),
                },
                "next_step": (
                    f"To diagnose a real install, unset {DEMO_ENV} and run this again."
                ),
                # Derived from the demo dataset by the same code path the real one
                # uses — `sync_decrypted_records` is where demo data is injected, so
                # this reports genuine per-kind counts rather than a second answer
                # written by hand.
                "capabilities": await self._capability_report(),
                "scope": self._scope_report(),
            }

        # ⚠️ load() RAISES on a config it cannot read, and that path is earlier
        # than the `if not config` bail-out below. Uncaught, it takes the whole
        # doctor down: the CLI prints one bare `error: ConfigError: ...` line,
        # zero [OK]/[FAIL] rows, and the MCP tool answers isError. So the two
        # things written specifically for this situation — the [KEY] private-key
        # location and the scope report — were unreachable at the exact moment
        # they were needed, on the command every doc names as the first stop.
        #
        # A diagnostic tool may not decline to produce a diagnosis. Its whole
        # job is to speak when everything else is failing.
        try:
            config = self.store.load()
        except ConfigError as error:
            add(
                "config", False, f"{type(error).__name__}: {error}",
                hint="The config file exists but its key material could not be "
                "resolved. Do NOT delete it — the private key is stored outside "
                "it, so deleting mints a new identity and orphans everything "
                "already encrypted. The detail above lists all three key "
                "locations and what was found in each.",
            )
            return {"ok": False, "checks": checks, "scope": self._scope_report()}

        if not config:
            add(
                "config", False, f"no config at {self.store.path}",
                hint="Run `vaultbeat-apple-health bind` to initialize and pair with the iOS app.",
            )
            # `scope` rides even the earliest bail-out: "no config" is exactly
            # when a reader is most likely to start hunting on the client side,
            # so this is the return that most needs to say where the border is.
            return {"ok": False, "checks": checks, "scope": self._scope_report()}
        add("config", True, str(self.store.path))

        add(
            "identity_key", bool(config.private_key_base64),
            "private key present" if config.private_key_base64 else "private key missing",
            hint="The Keychain entry is gone or unreadable. Re-run `vaultbeat-apple-health bind` to mint a fresh identity.",
        )

        reachable, probe_detail = self._probe_cloud(config.api_base_url)
        add(
            "cloud_reachable", reachable, probe_detail,
            hint="Check your internet connection / proxy. The cloud endpoint is "
            f"{config.api_base_url}",
        )

        add(
            "bound", config.is_bound,
            f"server_id={config.server_id}" if config.is_bound else "not bound to an iOS app",
            hint="Run `vaultbeat-apple-health bind`, then scan the QR with Vaultbeat on iOS "
            f"({CONNECT_SERVER}). Codes expire after 10 minutes — "
            "if the phone scanned but this side stayed pending, re-run bind for a fresh code.",
        )
        # Informational only: legacy bindings predate the owner-identity
        # handshake and still decrypt fine — absence must not fail the doctor.
        #
        # The second half of the legacy line is conditional on purpose. It used
        # to be printed only when `capabilities.owner_prefixes` showed two
        # people, and that list needed every record downloaded (GitHub #9), so
        # the condition is now stated instead of tested: this binding cannot
        # tell whether a partner shares data, and if one does, reads cannot
        # tell the two apart.
        add(
            "owner_identity", True,
            "owner identity received"
            if config.owner_user_id and config.owner_public_key_base64
            else "owner identity missing (legacy binding — decryption is unaffected; "
            "if a partner shares data with this account, reads cannot tell the two "
            "of you apart: row reads return both people and daily aggregates refuse. "
            "Re-pairing this machine from the Vaultbeat app fixes it)",
        )
        # Informational, always ok=True: the pairing-time trial snapshot
        # (vb-016). It never fails the doctor — the LIVE answer is the
        # data_roundtrip below, which surfaces a real trial_expired refusal
        # with its own hint; this row exists so an approaching deadline is
        # visible BEFORE the first refusal, and the wording keeps naming
        # itself a snapshot because a purchase made since pairing is
        # invisible to this machine. Absent when no deadline was recorded.
        access = self.access_snapshot(config)
        if access:
            add("access", True, access["note"])

        counts, counts_failure = await self._data_roundtrip_check(config, reachable, add)

        installed, latest, version_note = self._client_version_status()
        add(
            "client_version",
            latest is None or not self._version_is_older(installed, latest),
            version_note,
            hint=(
                f"Run `uvx --refresh vaultbeat-apple-health` (or `pip install -U vaultbeat-apple-health`) "
                f"to move from {installed} to {latest}. Older clients re-download the "
                f"entire history on every read instead of only what changed."
            ),
        )

        report: dict[str, Any] = {
            "ok": all(check["ok"] for check in checks),
            "checks": checks,
            "scope": self._scope_report(),
        }
        if config.is_bound:
            report["capabilities"] = await self._capability_report(
                config, counts=counts, failure=counts_failure
            )
        return report

    async def _data_roundtrip_check(
        self,
        config: LocalServerConfig,
        reachable: bool,
        add: Callable[..., None],
    ) -> tuple[dict[str, int | None] | None, Exception | None]:
        """The doctor's `data_roundtrip` row, and the per-kind counts it fetched.

        Returns ``(counts, failure)`` for `_capability_report`: the counts when
        they were had, else the reason they were not, so the report neither asks
        twice nor makes the same doomed requests again.
        """
        counts: dict[str, int | None] | None = None
        counts_failure: Exception | None = None
        if config.is_bound and reachable:
            try:
                # 🔴 The cost of this block was the bug (2026-10-02, GitHub #9).
                # It used to read sleep with `fresh=True` and the capability
                # report then decrypted EVERY kind just to count them: 51 MB and
                # ~100 s on a cold machine, past the 60 s many MCP clients allow
                # one tool call — on the command every guide names as the first
                # stop. Counts now come from per-kind digests (~100 bytes each)
                # and the round trip decrypts a sample of up to
                # `ROUNDTRIP_SAMPLE` records. A full read of a kind proves no
                # more about the pipe than three records of it do.
                counts = await self._kind_counts(config)
                sample = (
                    await self._roundtrip_sample(config, counts)
                    if counts is not None
                    else None
                )
                if sample is not None:
                    kind, decrypted, sampled, errors = sample
                    roundtrip_detail = (
                        f"decrypted {decrypted} of {sampled} sampled {kind} record(s)"
                        if sampled
                        else "the cloud accepted this server's token, but no records "
                        "are sealed for it yet, so decryption has not been tested"
                    )
                else:
                    # A transport without the catalog trio, or an edge too old
                    # to serve it: the old full read of one kind, which every
                    # deployment answers.
                    sleep_records, errors = await self._records_for_metric(
                        METRIC_SLEEP, limit=1, fresh=True
                    )
                    decrypted = len(sleep_records)
                    roundtrip_detail = f"decrypted {decrypted} sleep record(s)"
                if errors and not decrypted:
                    add(
                        "data_roundtrip", False,
                        f"fetch ok but decrypt failed ({errors[0]})",
                        # ⚠️ This used to open with "delete this server in the
                        # iOS app and bind again" — the most expensive of the
                        # available actions, recommended first. Deleting the row
                        # fires a BEFORE DELETE trigger that takes every envelope
                        # addressed to it (20k+ on a two-month-old binding).
                        # Re-binding alone now lands on the same row and keeps
                        # them, so deletion buys nothing.
                        #
                        # ⚠️ On how much a delete really costs — do NOT reason
                        # about this from Invariant 34. That invariant says the
                        # self-healing GC cannot repair agent-written kinds
                        # (strength/food/note) because iOS holds no fingerprint
                        # for blobs it did not write. TRUE, and irrelevant here:
                        # `VaultbeatNewServerBackfillCoordinator` does not work
                        # through fingerprints at all, it re-seals from the local
                        # JSON store — into which pull-and-merge has already
                        # folded the agent's writes. So those kinds come back in
                        # FULL after a re-bind; what does not is anything outside
                        # a kind's window (sleep/body ≤365d re-read from
                        # HealthKit, symptoms ≤120d) or already deleted from
                        # HealthKit. Two different mechanisms, opposite answers.
                        hint="The stored key can no longer decrypt your data. Re-run "
                        "`vaultbeat-apple-health bind` first — it re-binds this machine in place and "
                        "keeps everything already encrypted for it. Deleting the server in "
                        "the iOS app also works, but costs history: it destroys those keys, "
                        "and re-binding afterwards only re-seals what the phone can still "
                        "reproduce — recent Apple Health data plus your full strength / food "
                        "/ note log. Anything older than those windows is gone.",
                    )
                else:
                    add("data_roundtrip", True, roundtrip_detail)
            except VaultbeatTrialExpiredError as error:
                counts_failure = error
                # 🔴 Must precede the generic handler below, which would call
                # this a dead server token and prescribe a re-bind. That advice
                # is worse than useless here: re-binding SUCCEEDS, the trial does
                # not restart, and the next doctor run says the same thing — a
                # closed loop that never mentions the actual reason. Nothing is
                # broken and no data is lost; the fix is a purchase, not a repair.
                add(
                    "data_roundtrip", False, str(error),
                    hint="This is not a fault and nothing has been lost — your records "
                    "are intact and still encrypted to this machine. Agent access has "
                    "simply run out. Subscribe to Pro in the Vaultbeat iOS app "
                    "(Settings → Membership) and this machine resumes with no re-pairing. "
                    "Re-running `bind` will not help.",
                )
            except Exception as error:  # noqa: BLE001 — diagnostic surface, report everything
                counts_failure = error
                if isinstance(error, VaultbeatCloudError) and error.code == "rate_limited":
                    # mcp-sync limits an address only after repeated rejected
                    # tokens, so "nothing points at the pairing" would be the
                    # opposite of true here (review V7 follow-up).
                    add(
                        "data_roundtrip", False, f"{type(error).__name__}: {error}",
                        hint="The cloud is refusing this address for a while, which it "
                        "does after repeated requests with a token it does not accept. "
                        "Wait 15 minutes and run the doctor once; if it then says the "
                        "token is not accepted, re-run `vaultbeat-apple-health bind`.",
                    )
                    return counts, counts_failure
                if not (isinstance(error, VaultbeatCloudError) and error.rejects_credentials):
                    # 🔴 Review V7 (2026-10-03): since the doctor probes ONE kind
                    # before the rest (R4), a single 503, 429 or dropped
                    # connection on that probe landed here and was told "the
                    # server token is no longer accepted — re-run bind". Before
                    # R4 it took all 18 kinds failing. Only a rejected
                    # credential is a re-bind; anything else is said as itself.
                    add(
                        "data_roundtrip", False, f"{type(error).__name__}: {error}",
                        hint="This check did not complete, and nothing in the error "
                        "points at this machine's pairing: the cloud did not reject its "
                        "token. A network drop, a timeout or a server error clears on "
                        "its own — run the doctor again in a minute. Do not re-pair for "
                        "this; if it keeps failing with the same error, report it at "
                        "github.com/Fino-wind/vaultbeat-apple-health.",
                    )
                    return counts, counts_failure
                add(
                    "data_roundtrip", False, f"{type(error).__name__}: {error}",
                    # The old wording sent the user to verify the row still
                    # exists, on the assumption that a deleted row is the only
                    # way a token dies. That stopped being true when binding
                    # became an upsert: re-binding this same machine now
                    # ROTATES the token in place, so a second bind (or an old
                    # QR still on screen being scanned again) invalidates the
                    # token this config holds while the row sits there looking
                    # perfectly healthy. A user who checks and finds the server
                    # present concludes the diagnosis is wrong — so lead with
                    # the fix that covers both causes.
                    hint="The server token is no longer accepted. Re-run "
                    "`vaultbeat-apple-health bind` — this is expected if you bound this machine "
                    "again since, which replaces the old token. The server still being "
                    "listed in the iOS app does not rule this out. If binding does not "
                    f"fix it, check the server is still listed ({AUTHORIZED_SERVERS}).",
                )
        return counts, counts_failure

    # ── Client version freshness ─────────────────────────────────────────────
    #
    # 🔒 The upgrade prompt is generated HERE, from a version-number comparison.
    # PyPI supplies one string that must parse as digits; every word the user or
    # their agent reads is hardcoded above.
    #
    # This is deliberate and the alternative was rejected on 2026-07-29. The
    # obvious design — have the SERVER return a `notice` string that the client
    # passes through — would have handed whoever controls the backend a channel
    # that writes arbitrary text straight into the user's agent context. MCP tool
    # results and real user turns arrive under the same `user` role; a model
    # cannot tell them apart. And this server exposes WRITE tools
    # (log_food_entry / log_strength_entry / log_weight_entry / log_note), while
    # the agent reading them usually has other MCPs attached (filesystem, shell,
    # browser) — so the blast radius is the user's whole toolset, not this app.
    # It also directly contradicts the product's one promise: a service that
    # "cannot read your data" must not be able to put words in your AI's mouth.
    #
    # If a server-driven notice is ever genuinely needed, send an ENUM, never
    # prose: `{"client_outdated": true, "min_version": "0.2.5"}` — the worst a
    # compromised backend can then do is show a false upgrade prompt.
    _PYPI_URL = "https://pypi.org/pypi/vaultbeat-apple-health/json"

    @staticmethod
    def _version_tuple(value: str) -> tuple[int, ...]:
        import re

        return tuple(int(p) for p in re.findall(r"\d+", value)[:3])

    @classmethod
    def _version_is_older(cls, installed: str, latest: str) -> bool:
        try:
            return cls._version_tuple(installed) < cls._version_tuple(latest)
        except (TypeError, ValueError):
            return False

    def _client_version_status(self) -> tuple[str, str | None, str]:
        """(installed, latest_or_None, human_readable_detail).

        Unreachable PyPI is NOT a failure — an offline machine still has a
        perfectly working install, and a diagnostic that cries wolf about the
        network teaches people to ignore it.
        """

        import json as _json
        import os
        import urllib.error
        import urllib.request

        from vaultbeat_mcp_local import __version__ as installed

        # Injection point. Tests must not depend on the network — a unit suite
        # that reaches PyPI is slow, flaky offline, and its result changes when
        # someone publishes a release. Empty string = "PyPI unreachable", which
        # is the branch that must stay non-failing. Also lets every branch be
        # exercised once, including the one that reports being behind.
        override = os.getenv("VAULTBEAT_MCP_FAKE_LATEST")
        if override is not None:
            if not override:
                return installed, None, f"{installed} installed (could not reach PyPI to compare)"
            if self._version_is_older(installed, override):
                return installed, override, f"{installed} installed, {override} available"
            return installed, override, f"{installed} installed (latest)"

        try:
            with urllib.request.urlopen(self._PYPI_URL, timeout=5) as response:
                latest = str(_json.load(response)["info"]["version"])
        except (urllib.error.URLError, TimeoutError, KeyError, ValueError, OSError):
            return installed, None, f"{installed} installed (could not reach PyPI to compare)"

        if self._version_is_older(installed, latest):
            return installed, latest, f"{installed} installed, {latest} available"
        return installed, latest, f"{installed} installed (latest)"

    # Metric kinds that only exist from a given iOS release onward. A tool whose
    # backing kind predates the user's app returns nothing — with no error, which
    # is indistinguishable from "you have no data" unless we say so here.
    #
    # This is derived from the DATA, not from an app-reported version number.
    # Asking the app to report its version would only help users who later
    # install a build that knows to report it — useless for the people already
    # affected, and it would mean changing the payload shape, which is the one
    # thing the wire contract most wants left alone.
    KIND_MIN_APP_RELEASE: dict[str, str] = {
        "strength": "2026-07-19",
        "food": "2026-07-20",
        "vo2max": "2026-07-20",
        "basal_energy": "2026-07-21",
        "hrv_hourly": "2026-07-22",
    }

    #: Records the doctor's round trip decrypts. More than one so that a single
    #: damaged record cannot fail the check on its own (it passes when ANY
    #: decrypts, like the full read it replaced); small so it stays a probe.
    ROUNDTRIP_SAMPLE = 3

    async def _kind_counts(self, config: LocalServerConfig) -> dict[str, int | None] | None:
        """Records sealed for this server, per kind, from one digest per kind.

        A digest is ~100 bytes whatever the library size: one probe kind, then
        the rest `_CONCURRENT_KIND_READS` at a time, where decrypting everything
        to count it took ~100 s and 51 MB.

        ``None`` per kind = that kind could not be counted (a transient failure,
        an unparseable digest); the report lists it as unchecked rather than as
        empty. ``None`` overall = this transport cannot ask for digests, and the
        caller falls back to counting decrypted records.

        Raises when the probe kind fails, when EVERY kind failed, or on an
        expired trial: those are account-level answers, and the caller reports
        them as such.
        """
        client = self._client(config)
        if not callable(getattr(client, "sync_digest", None)):
            return None
        token = config.server_token or ""
        kinds = sorted(KNOWN_METRIC_TYPES)

        async def one(kind: str) -> int | None:
            try:
                digest, legacy = await client.sync_digest(token, metric_type=kind)
            except VaultbeatUnsupportedMetricError:
                # The deployed edge does not know this kind, so nothing of it
                # can be stored there: empty, not unknown.
                return 0
            if legacy is not None:
                # An edge older than the catalog sent the kind's rows instead.
                return len(legacy)
            count = digest.get("count") if isinstance(digest, dict) else None
            return count if isinstance(count, int) and count >= 0 else None

        # 🔴 One kind first, alone, and the rest a few at a time. All eighteen used
        # to go out at once, so a dead token produced eighteen 401s in a second —
        # and `mcp-sync` locks an IP out for 15 minutes after 20 of them, so the
        # agent's very next read tripped the lock and the doctor that was meant
        # to explain the failure had caused a second one (review R4 on #9). A
        # failed probe is an account- or network-level answer: it is raised
        # without asking the other seventeen the same question. Once the probe
        # has answered, the token is good and the lockout cannot be reached, so
        # the rest go `_CONCURRENT_DIGESTS` at a time — a digest is ~100 bytes,
        # and at `_CONCURRENT_KIND_READS` (4) the real doctor took 29 s against
        # 18 s fully concurrent (release gate, 2026-10-03).
        first = await one(kinds[0])
        gate = asyncio.Semaphore(_CONCURRENT_DIGESTS)

        async def gated(kind: str) -> int | None:
            async with gate:
                return await one(kind)

        rest = await asyncio.gather(*(gated(kind) for kind in kinds[1:]), return_exceptions=True)
        results: list[int | None | BaseException] = [first, *rest]
        failures = [r for r in results if isinstance(r, BaseException)]
        for failure in failures:
            if isinstance(failure, VaultbeatTrialExpiredError):
                raise failure
        if failures and len(failures) == len(results):
            raise failures[0]
        return {
            kind: None if isinstance(result, BaseException) else result
            for kind, result in zip(kinds, results)
        }

    async def _roundtrip_sample(
        self, config: LocalServerConfig, counts: dict[str, int | None]
    ) -> tuple[str, int, int, list[str]] | None:
        """Fetch and decrypt up to `ROUNDTRIP_SAMPLE` records of one kind.

        Returns ``(kind, decrypted, sampled, errors)``; ``sampled == 0`` when no
        kind has records. ``None`` when this transport or deployment cannot
        fetch by id, so the caller falls back to a full read.

        The kind is the smallest one holding at least a sample's worth, because
        its catalog is the cheapest to list. Decryption uses the same key for
        every kind, so which kind proves it does not matter. Nothing is cached
        and nothing is reported as damaged: this is a probe, and the next real
        read of the kind does both.
        """
        client = self._client(config)
        if not all(callable(getattr(client, name, None)) for name in ("sync_catalog", "sync_blobs")):
            return None
        filled = sorted((count, kind) for kind, count in counts.items() if count)
        if not filled:
            return "", 0, 0, []
        _, kind = next(
            ((count, kind) for count, kind in filled if count >= self.ROUNDTRIP_SAMPLE),
            filled[-1],
        )
        token = config.server_token or ""
        catalog = await client.sync_catalog(token, metric_type=kind)
        if catalog is None:
            return None
        blob_ids = [str(row["blob_id"]) for row in catalog if row.get("blob_id")][: self.ROUNDTRIP_SAMPLE]
        rows = await client.sync_blobs(token, blob_ids=blob_ids, metric_type=kind) if blob_ids else []
        decrypted = 0
        errors: list[str] = []
        for row in rows:
            try:
                self._decrypt_row(row, config)
            except (KeyError, TypeError, VaultbeatCryptoError, ValueError) as error:
                errors.append(f"{_safe_row_id(row.get('id', '<unknown>'))}: decrypt_failed ({type(error).__name__})")
            else:
                decrypted += 1
        return kind, decrypted, len(rows), errors

    async def _capability_report(
        self,
        config: LocalServerConfig | None = None,
        *,
        counts: dict[str, int | None] | None = None,
        failure: Exception | None = None,
    ) -> dict[str, Any]:
        """Which metric kinds actually have data for this account, and which
        tools are consequently dead weight.

        Deliberately does NOT claim to know the app's version: an empty kind
        means either "the app is older than this feature" or "the user has never
        recorded it / not granted that HealthKit permission". Both are reported
        the same way, because the actionable advice is identical — update the app
        and check the permission — and pretending to distinguish them would be
        guessing.

        `counts` are the doctor's per-kind digests when it already has them;
        `failure` is why it could not get them, so the same requests are not
        made twice. Without either, this asks for the digests itself.
        """
        present: list[str] = []
        absent: list[str] = []
        try:
            if failure is not None:
                raise failure
            if counts is None and config is not None and not self._demo:
                counts = await self._kind_counts(config)
            if counts is None:
                # Demo mode, or a transport with no digests: count what decrypts.
                # Demo records come from this same call, so the demo report is
                # derived rather than written by hand.
                records, _ = await self.sync_decrypted_records()
                tallied: dict[str, int | None] = {}
                for record in records:
                    kind = record.metric_type or "sleep"
                    tallied[kind] = (tallied.get(kind) or 0) + 1
                counts = tallied
        except VaultbeatTrialExpiredError as error:
            # Distinguished from the generic failure below because it is not one:
            # nothing is broken and nothing is missing. Reporting it as "could not
            # read cloud data" invites the reader to hunt for a fault, and the
            # conclusion a user draws from a health app that suddenly cannot see
            # anything is that their records are gone.
            return {"available": False, "reason": str(error)}
        except Exception:  # noqa: BLE001 — diagnostics must not raise
            return {"available": False, "reason": "could not read cloud data"}

        unchecked: list[str] = []
        for kind in sorted(KNOWN_METRIC_TYPES):
            if kind in counts and counts[kind] is None:
                unchecked.append(kind)
            else:
                (present if counts.get(kind) else absent).append(kind)

        gated = {k: v for k, v in self.KIND_MIN_APP_RELEASE.items() if k in absent}
        report: dict[str, Any] = {
            "available": True,
            "kinds_with_data": present,
            "kinds_without_data": absent,
            # A kind with one record and a kind with a year of them were reported
            # identically — both just a name in kinds_with_data — so "this server
            # is still filling in" and "this server has everything" looked the
            # same in the one report meant to tell them apart. That matters most
            # right after pairing, when a partial history is the normal state.
            #
            # ⚠️ Counts only, deliberately no date range: DecryptedRecord carries
            # `created_at`, which is an UPLOAD-batch timestamp, not the day the
            # record is about — a backfill uploads years of history in minutes
            # (Invariant 38). Deriving a coverage window from it would print a
            # precise-looking range that is simply false. A real range needs each
            # kind's payload parsed for its own business date; until then a count
            # is the honest signal.
            #
            # 🔑 Since 2026-10-02 these are the per-kind digest counts: records
            # SEALED FOR THIS SERVER, which includes any this server cannot
            # decrypt (a damaged one is reported by the read that reaches it).
            # Counting by decrypting was 51 MB on a cold machine (GitHub #9).
            "record_counts": {k: counts[k] for k in present},
            # 🗑 `owner_prefixes` lived here (2026-09-06 → 2026-10-02): every owner
            # present in the data, so an agent could pick one for the `owner`
            # argument. 0.9.0 removed that argument (reads default to the paired
            # account, `partner=true` selects the other), and the list cost a
            # download of every record to build — the bulk of a 100 s cold
            # doctor. The one job left to it, warning a LEGACY binding that two
            # people's rows are indistinguishable, is now the `owner_identity`
            # check's own wording. Whether a partner shares a kind is answered by
            # reading that kind with `partner=true`.
            "possibly_needs_newer_app": gated,
            # 🔴 Cause (1) is FIRST because it is the one a brand-new user actually
            # hits, and the only one they can act on in seconds. Until 2026-08-11 this
            # note listed the other three only, so someone who had just finished
            # binding — the exact moment this report is most often read — was sent to
            # check their iOS version and their Health permissions, both dead ends,
            # while the real answer was "your history hasn't been sealed for this
            # server yet". Measured that day: a freshly bound server held 4 of 18
            # kinds and stayed there until the owner tapped Re-sync by hand.
            "note": (
                "Empty kinds listed under possibly_needs_newer_app have four possible "
                "causes, in the order worth checking: (1) this MCP server was bound "
                "recently and your history has not finished sealing for it — each "
                "server gets its own encrypted copy, so a new one starts empty and "
                f"fills in; open the app and tap {RESYNC}, then re-run this check "
                "in a few minutes. "
                "(2) the app is older than the stated date for that kind. (3) Apple "
                "Health access for the kind was never granted — a read denial is "
                "invisible to the app, so it looks identical to having no data; "
                f"recover via {HEALTH_ACCESS}. (4) it "
                "genuinely has not been recorded yet."
            )
            if gated
            # ⚠️ Until 2026-10-02 everything that was not `gated` read "Every kind
            # has data" — including an account with no water or no workouts at
            # all, whose empty kinds are simply not ones an app date explains.
            else (
                "Kinds under kinds_without_data have nothing sealed for this server. "
                "Three possible causes, in the order worth checking: (1) this MCP "
                "server was bound recently and your history has not finished sealing "
                f"for it; open the app and tap {RESYNC}, then re-run this check in a "
                "few minutes. (2) Apple Health access for the kind was never granted "
                f"— recover via {HEALTH_ACCESS}. (3) it genuinely has not been "
                "recorded yet."
            )
            if absent
            else "Every kind this server knows about has data.",
        }
        if unchecked:
            # Not folded into either list: a kind whose count request failed is
            # neither known to be empty nor known to hold data (Invariant 57
            # (absence-has-more-than-one-cause)).
            report["kinds_not_checked"] = unchecked
            report["not_checked_note"] = (
                "The cloud did not answer for the kinds under kinds_not_checked this "
                "time, so they are in neither list. Re-run the check to count them."
            )
        return report

    def status(self) -> dict[str, Any]:
        # Demo facts ride ON TOP of the real ones; `initialized` and `bound` keep
        # reporting the actual config. Making them True would be the one lie that
        # matters: an agent that believes it is bound will try to write, and will
        # read every subsequent synthetic number as its owner's real health
        # record.
        #
        # It is NOT true that a demo install is "usually neither" — this comment
        # said so until 2026-08-20, and the assumption was load-bearing enough to
        # be wrong twice. Demo mode is orthogonal to binding: it reads no config,
        # so it happily runs on a fully bound machine, where `status` correctly
        # reports bound: true while every read tool is serving synthetic records.
        # Both statements are individually true and together they mislead, so the
        # gap is closed by a field that answers the question neither of them does
        # — `data_source` — rather than by making one of them lie.
        from vaultbeat_mcp_local.demo import DEMO_ENV

        try:
            config = self.store.load()
        except ConfigError as error:
            # `status` is the FIRST thing a stuck user runs, and until 2026-08-20
            # this raised — through `_watermark_demo`, which only wraps returns,
            # so in demo mode the exception escaped unstamped and surfaced as a
            # bare ToolError. Worse, what it threw was the private-key recovery
            # essay, whose central warning is "DO NOT DELETE config.json or every
            # record already encrypted becomes permanently unreadable" — a
            # sentence with no referent in demo mode, which has no key and no
            # encrypted records. Catch it and report it as a field.
            return {
                **self._demo_block(bound=False),
                # The file EXISTS (load() only raises after exists()), so this is
                # true — and it is the answer that matters, because a consumer
                # reading `initialized: False` would recommend `bind`, and
                # minting a fresh identity is the single most destructive thing
                # to do to an install whose key is merely unreadable.
                "initialized": True,
                "bound": False,
                "data_source": "synthetic" if self._demo else "account",
                "config_error": (
                    f"{type(error).__name__} at {self.store.path} — not used in demo "
                    f"mode, which reads no config and decrypts nothing. Run without "
                    f"{DEMO_ENV} for the full diagnosis."
                    if self._demo
                    else str(error)
                ),
                "next_step": (
                    f"Nothing to do for demo mode. To diagnose the real install, unset "
                    f"{DEMO_ENV} and run `vaultbeat-apple-health doctor`."
                    if self._demo
                    else (
                        "Run `vaultbeat-apple-health doctor` for the full private-key report. "
                        "DO NOT delete the config file and DO NOT run `bind` — the key "
                        "is unreadable, not absent, and re-binding mints a new identity "
                        "that cannot decrypt anything already stored."
                    )
                ),
            }

        demo_block = self._demo_block(bound=bool(config and config.is_bound))

        if not config:
            # `next_step` added 2026-08-11. The website tells first-time users to run
            # `status` to "verify the server", and the honest answer at that point was
            # two bare falses — which reads like a failure to someone who has just
            # installed the thing, and says nothing about what to do next. It is
            # ADD-ONLY: agents keying off `bound` / `initialized` are unaffected.
            return {
                **demo_block,
                "initialized": False,
                "bound": False,
                "data_source": "synthetic" if self._demo else "account",
                "next_step": (
                    "Not paired yet — this is the expected state before first use. "
                    "Run: uvx vaultbeat-apple-health@latest bind "
                    "then scan the QR code in the iOS app under "
                    f"{CONNECT_SERVER}. "
                    "Requires the Vaultbeat iOS app; connecting is open on every plan."
                ),
            }

        access = self.access_snapshot(config)

        return {
            **demo_block,
            "initialized": True,
            "bound": config.is_bound,
            # `bound` answers "is a binding stored", which in demo mode is a
            # true answer to the wrong question — the binding is real and is
            # being BYPASSED. Nothing here previously answered "where did these
            # numbers come from", so an agent had to infer it, and the available
            # signals pointed the wrong way. Emitted in both modes on purpose: a
            # missing field and "not demo" must not look the same.
            "data_source": "synthetic" if self._demo else "account",
            **(
                {}
                if config.is_bound
                else {
                    "next_step": (
                        "Keys exist but no phone has authorized this server yet. "
                        "Re-run `bind` and scan the QR code in the iOS app."
                    )
                }
            ),
            "server_name": config.server_name,
            "server_id": _bound_uuid(config.server_id),
            "api_base_url": config.api_base_url,
            "poll_id": config.poll_id,
            "public_key_base64": config.public_key_base64,
            "bound_at": config.bound_at,
            "last_sync_at": config.last_sync_at,
            # Pairing-time trial snapshot (vb-016). ADD-ONLY and absent when no
            # deadline was recorded — absence is the claim-free rendering of
            # "grandfathered / paid / unknown", same as the iOS Settings row.
            **({"access": access} if access else {}),
            "owner_identity_bound": bool(config.owner_user_id and config.owner_public_key_base64),
            # 🔴 The prefix `skill.md` sends every agent HERE to fetch ("Get the
            # prefixes from `vaultbeat_status`, and pass one"). Until 2026-09-07
            # this dict carried only the boolean above, so that documented step
            # had no source and the agent's two exits were both wrong: omit
            # `owner` and pool two bodies into one average (the blend
            # `_attach_owner_guard` exists to catch), or copy the literal
            # `"a1a1"` out of a tool description and match nothing.
            #
            # ⚠️ NOT the same field as `_attach_owner_guard`'s
            # `owner_user_id_prefixes` (plural), and they must not be merged:
            # that one answers "who is mixed into THIS result" and only appears
            # once the mixing already happened, on results with >1 owner. This
            # one answers "which one am I", is available before any read, and is
            # the only source of that answer on a SINGLE-owner account — where
            # the plural field never fires and an agent otherwise cannot confirm
            # its filter is the user rather than the partner.
            #
            # Eight chars because the `owner=` filter is a `startswith` and
            # every other prefix surface is `[:8]` (demo.py's DEMO_OWNER_PREFIX
            # carries the same note). Zero extra I/O: `config` is already loaded
            # on the line above. Absent rather than null when unbound — absence
            # is the claim-free rendering, same as `access`.
            **({"owner_user_id_prefix": uid[:8]} if (uid := _bound_uuid(config.owner_user_id)) else {}),
            "owner_device_bound": bool(config.owner_device_id),
            "config_path": str(self.store.path),
        }

    def _demo_block(self, *, bound: bool) -> dict[str, Any]:
        """The demo facts spliced into `status()`, or `{}` when demo mode is off.

        Takes `bound` because the honest wording depends on it: on an unbound
        machine demo mode is simply what is running, while on a BOUND one it is
        overriding a real, working binding — and `bound: true` sitting beside a
        note that says binding "still requires a real paired account" reads as
        confirmation that the binding is in use. It is not.
        """

        from vaultbeat_mcp_local.demo import demo_status

        return demo_status(bound=bound) if self._demo else {}

    def _client(self, config: LocalServerConfig) -> CloudClientProtocol:
        return self._cloud_client or VaultbeatCloudClient(config.api_base_url)

    @staticmethod
    def _decrypt_row(row: dict[str, Any], config: LocalServerConfig) -> DecryptedRecord:
        blob = row.get("encrypted_sleep_blobs")
        if isinstance(blob, list):
            blob = blob[0] if blob else None
        if not isinstance(blob, dict):
            raise ValueError("missing encrypted_sleep_blobs payload")

        plaintext = decrypt_blob_payload(
            ciphertext_base64=str(blob["ciphertext"]),
            encrypted_data_key_base64=str(row["encrypted_data_key"]),
            private_key_base64=config.private_key_base64,
        )
        # The row's plaintext columns are the server's words, and every one of
        # them is copied into results and error lines an agent reads: each is
        # kept only in the shape it claims to have (Anti-pattern 23, review R5).
        kind = blob.get("metric_type")
        return DecryptedRecord(
            envelope_id=_safe_row_id(row["id"]),
            blob_id=_safe_row_id(row["blob_id"]),
            metric_type=kind if kind in KNOWN_METRIC_TYPES else None,
            created_at=_safe_shaped(blob.get("created_at"), _INSTANT),
            payload=decode_json_payload(plaintext),
            owner_user_id=_safe_shaped(blob.get("owner_user_id"), _UUID),
        )
