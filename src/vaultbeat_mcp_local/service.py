from __future__ import annotations

import asyncio
import json
import logging
import secrets
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Protocol

from vaultbeat_mcp_local.cache import LocalRecordCache
from vaultbeat_mcp_local.client import (
    PollBindingResult,
    VaultbeatCloudClient,
    VaultbeatUnsupportedMetricError,
)
from vaultbeat_mcp_local.crypto import (
    RecipientKey,
    VaultbeatCryptoError,
    decode_json_payload,
    decrypt_blob_payload,
    encrypt_blob_payload,
)
from vaultbeat_mcp_local.store import ConfigStore, LocalServerConfig, now_iso


_LOG = logging.getLogger("vaultbeat_mcp_local.service")

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
    }
)

# Note target kinds (mirrors iOS VaultbeatNoteTargetKind). Unknown kinds are
# accepted as-is so a newer app adding a kind doesn't brick older decoders.
NOTE_TARGET_KINDS = frozenset({"sleep", "menstrual"})

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

# Menstrual flow enum (mirrors the iOS HKCategoryValueVaginalBloodFlow mapping).
MENSTRUAL_FLOW_VALUES = frozenset({"unspecified", "light", "medium", "heavy", "none"})

class CloudClientProtocol(Protocol):
    async def poll_binding(self, poll_id: str) -> PollBindingResult: ...

    async def sync(
        self, server_token: str, *, metric_type: str | None = None
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
        """Inverse of `to_dict` — used to rehydrate cache entries."""

        return cls(
            envelope_id=str(raw.get("envelope_id", "")),
            blob_id=str(raw.get("blob_id", "")),
            metric_type=(str(raw["metric_type"]) if raw.get("metric_type") is not None else None),
            created_at=(str(raw["created_at"]) if raw.get("created_at") is not None else None),
            payload=raw.get("payload"),
            owner_user_id=(
                str(raw["owner_user_id"]) if raw.get("owner_user_id") is not None else None
            ),
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
    only in presentation layers. bodyFatPercent and bmi are reserved fields the iOS
    payload (VaultbeatBodySharedCloudPayload) currently always sends as null.
    """

    day_id: str
    day_start_date: str
    weight_kg: float
    body_fat_percent: float | None
    bmi: float | None
    owner_user_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "day_id": self.day_id,
            "day_start_date": self.day_start_date,
            **_local_date_fields(self.day_start_date),
            "weight_kg": self.weight_kg,
            "body_fat_percent": self.body_fat_percent,
            "bmi": self.bmi,
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "note_id": self.note_id,
            "target_kind": self.target_kind,
            "target_date": self.target_date,
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
            "date": self.date,
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
    {dayID, dayStartDate, weightKg, bodyFatPercent, bmi} — weightKg required (kg),
    bodyFatPercent/bmi nullable reserved fields (currently always null from iOS).
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


def summarize_strength(entries: list[StrengthRecord], *, limit_days: int | None = None) -> dict[str, Any]:
    """Recent strength sessions, newest day first, with per-session volume.

    Dedup by entry_id (newest updated_at wins — edits upsert the same blob id).
    Pass `limit_days` to keep only the most recent N sessions after dedup.
    """

    by_id: dict[str, StrengthRecord] = {}
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
        "session_count": len(ordered),
        "sessions": [entry.to_dict() for entry in ordered],
    }


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
    local = parsed.astimezone(datetime.now(timezone.utc).astimezone().tzinfo)
    fields: dict[str, Any] = {"local_date": local.date().isoformat()}
    if with_time:
        fields["local_time"] = local.strftime("%Y-%m-%dT%H:%M")
    return fields


_ERRORS_NOTE = (
    "Each entry in `errors` is ONE blob that failed to decrypt (`decrypt_failed`: "
    "usually a historical-backfill blob sealed with a stale envelope key — cosmetic) "
    "or to parse (`parse_failed`: payload written by an older/newer schema). "
    "A failed blob is skipped; every other record in this result is complete, so a "
    "handful of errors among hundreds of records is NOT a data-integrity problem."
)


def _attach_errors(summary: dict[str, Any], errors: list[str]) -> dict[str, Any]:
    """Standard error reporting: the raw list plus, when non-empty, a note that
    explains what an error means so callers stop treating two stale-key blobs
    as a broken dataset (2026-07-23 client feedback)."""

    summary["errors"] = errors
    if errors:
        summary["errors_note"] = _ERRORS_NOTE
    return summary


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
        + "). Aggregate numbers blend both people and are meaningless — "
        're-query with owner="<prefix>" to select one person.'
    )
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
        value = value.astimezone(datetime.now(timezone.utc).astimezone().tzinfo)
    return value.date()


def _local_midnight_iso(day: date) -> str:
    """UTC ISO8601 instant for local midnight of `day` — the inverse of
    ``_local_calendar_day``, used when the agent writes a brand-new entry for a
    given local calendar day. Same "server runs in the phone's timezone"
    assumption as that function; see its docstring.
    """

    local_tz = datetime.now(timezone.utc).astimezone().tzinfo
    local_midnight = datetime(day.year, day.month, day.day, tzinfo=local_tz)
    return local_midnight.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


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
        payload["last_cycle_start_date"] = starts[-1].isoformat() if starts else None
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
    payload["last_cycle_start_date"] = last_start.isoformat()
    payload["predicted_next_period_start_date"] = predicted.isoformat()

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


def summarize_symptoms(days: list[SymptomDay]) -> dict[str, Any]:
    """Recent symptom days grouped by data owner, plus per-owner type counts.

    Both partners can track symptoms, so days are grouped by `owner_user_id`
    (None → "unknown", e.g. blobs fetched before the edge function returned
    ownership). Within an owner, days dedup by day_id (newest day_start_date
    wins — matching the one-blob-per-day upsert) and sort newest-first.
    `symptom_counts` counts logged days per symptom type, skipping explicit
    "notPresent" entries so "logged as absent" doesn't inflate the tally.
    """

    owners: dict[str, dict[str, SymptomDay]] = {}
    for day in days:
        owner_key = day.owner_user_id or "unknown"
        by_id = owners.setdefault(owner_key, {})
        existing = by_id.get(day.day_id)
        if existing is None or day.day_start_date >= existing.day_start_date:
            by_id[day.day_id] = day

    owner_summaries: list[dict[str, Any]] = []
    for owner_key in sorted(owners):
        ordered = sorted(owners[owner_key].values(), key=lambda d: d.day_start_date, reverse=True)
        type_counts: dict[str, int] = {}
        for day in ordered:
            for sample in day.samples:
                if sample.severity == "notPresent":
                    continue
                type_counts[sample.symptom_type] = type_counts.get(sample.symptom_type, 0) + 1
        owner_summaries.append(
            {
                "owner_user_id": None if owner_key == "unknown" else owner_key,
                "day_count": len(ordered),
                "symptom_counts": dict(sorted(type_counts.items(), key=lambda kv: -kv[1])),
                "days": [day.to_dict() for day in ordered],
            }
        )

    return {
        "sensitive": True,
        "owners": owner_summaries,
        "owner_count": len(owner_summaries),
        "total_day_count": sum(o["day_count"] for o in owner_summaries),
    }


def _select_primary_sessions(sessions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pick one primary session per local_date using iOS mergeSameDaySessions priority.

    Priority (highest wins):
      1. non-inBedOnly beats inBedOnly
      2. hasStageDetail beats non-hasStageDetail (Watch beats iPhone)
      3. longer totalSleepMinutes beats shorter
      4. earlier bedtime wins (tie-break)
    """

    by_date: dict[str, list[dict[str, Any]]] = {}
    for s in sessions:
        d = s.get("local_date", "")
        if d:
            by_date.setdefault(d, []).append(s)

    daily: list[dict[str, Any]] = []
    for day_key in sorted(by_date.keys(), reverse=True):
        candidates = by_date[day_key]
        best = max(candidates, key=lambda s: (
            not s.get("is_in_bed_only", False),
            s.get("has_stage_detail", False),
            s.get("total_sleep_minutes", 0),
            -(datetime.fromisoformat(s["bedtime"]).timestamp()
              if s.get("bedtime") else 0),
        ))
        h, m = divmod(best["total_sleep_minutes"], 60)
        daily.append({
            "date": day_key,
            "total_sleep_minutes": best["total_sleep_minutes"],
            "duration_label": f"{h}h{m:02d}m",
            "bedtime": best.get("bedtime"),
            "wake_time": best.get("wake_time"),
            "has_stage_detail": best.get("has_stage_detail"),
            "stage_minutes": best.get("stage_minutes"),
        })

    return daily


class VaultbeatLocalService:
    def __init__(
        self,
        store: ConfigStore,
        cloud_client: CloudClientProtocol | None = None,
        cache: LocalRecordCache | None = None,
    ):
        self.store = store
        self._cloud_client = cloud_client
        self._cache = cache

    @property
    def cache(self) -> LocalRecordCache:
        if self._cache is None:
            self._cache = LocalRecordCache(self.store.path.parent / "cache")
        return self._cache

    def start_binding(
        self,
        *,
        server_name: str = "Local AI Server",
        api_base_url: str = "https://wjpnyxglgtmtgjuuhwru.supabase.co/functions/v1",
    ) -> BindingSession:
        config = self.store.ensure_initialized(server_name=server_name, api_base_url=api_base_url)
        poll_id = secrets.token_urlsafe(24)
        config = self.store.update(
            server_name=server_name.strip() or config.server_name,
            api_base_url=api_base_url.rstrip("/") or config.api_base_url,
            poll_id=poll_id,
            server_id=None,
            server_token=None,
            bound_at=None,
            last_sync_at=None,
        )
        # A (re)bind may land on a different server identity; cached plaintext
        # from the previous binding must not answer for the new one.
        self.cache.clear()
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
            raise RuntimeError("No active binding session; run `vaultbeat-mcp-local bind` first")

        result = await self._client(config).poll_binding(config.poll_id)
        if result.status == "bound":
            if not result.server_id or not result.server_token:
                raise RuntimeError("Cloud returned bound without server credentials")
            self.store.update(
                server_id=result.server_id,
                server_token=result.server_token,
                owner_user_id=result.owner_user_id,
                owner_public_key_base64=result.owner_public_key_base64,
                owner_device_id=result.owner_device_id,
                poll_id=None,
                bound_at=now_iso(),
            )
        return result

    async def poll_until_bound(self, *, timeout_sec: int = 300, interval_sec: float = 7.0) -> PollBindingResult:
        deadline = asyncio.get_running_loop().time() + timeout_sec
        while True:
            result = await self.poll_once()
            if result.status == "bound":
                return result
            if asyncio.get_running_loop().time() >= deadline:
                return result
            await asyncio.sleep(interval_sec)

    async def sync_decrypted_records(
        self,
        *,
        limit: int | None = None,
        metric_type: str | None = None,
        fresh: bool = False,
    ) -> tuple[list[DecryptedRecord], list[str]]:
        """Fetch + decrypt this server's records, cache-first.

        `metric_type` narrows the fetch server-side (older mcp-sync deployments
        ignore the parameter, so the local filter below stays authoritative —
        the parameter is an optimization, never a correctness dependency).
        Within the cache TTL a repeat query answers from local plaintext with
        ZERO network; `fresh=True` forces a cloud round trip. The cache always
        stores the FULL result set for its key — `limit` only trims the copy
        returned to the caller.
        """

        if metric_type is not None and metric_type not in KNOWN_METRIC_TYPES:
            # Fail fast locally: an unknown value would (a) 400 on the new edge,
            # and (b) poison a cache key — e.g. "all" maps to the same file as
            # the unfiltered set. Membership check beats both.
            raise ValueError(
                f"unknown metric_type {metric_type!r}; expected one of "
                f"{', '.join(sorted(KNOWN_METRIC_TYPES))}"
            )

        config = self.store.require_bound()
        server_token = config.server_token
        server_id = config.server_id or ""
        if not server_token:
            raise RuntimeError("Local MCP server is not bound; run `vaultbeat-mcp-local bind` first")

        if not fresh:
            cached = self.cache.load(server_id=server_id, metric_type=metric_type)
            if cached is not None:
                cached_records, cached_errors = cached
                records = [DecryptedRecord.from_dict(row) for row in cached_records]
                if limit is not None:
                    records = records[:limit]
                return records, cached_errors

        try:
            envelope_rows = await self._client(config).sync(server_token, metric_type=metric_type)
        except VaultbeatUnsupportedMetricError as error:
            # Version skew: this MCP server knows a kind the deployed edge
            # function does not. Degrade to "this one kind is unavailable"
            # rather than raising — every other tool call still works, and the
            # agent gets a message it can relay instead of an opaque failure.
            # (2026-07-22: the opposite behaviour took ALL default HRV reads
            # down for two days.)
            return [], [
                f"unsupported_metric:{error.metric_type} — the Vaultbeat cloud "
                "has not been updated to serve this data type yet; every other "
                "data type is unaffected"
            ]
        records = []
        errors: list[str] = []

        for row in envelope_rows:
            try:
                records.append(self._decrypt_row(row, config))
            except (KeyError, TypeError, VaultbeatCryptoError, ValueError) as error:
                envelope_id = str(row.get("id", "<unknown>"))
                # Stage-tagged so a consumer can tell "sealed with a stale key /
                # corrupt ciphertext" apart from the parse_failed entries the
                # per-metric decoders append (2026-07-23 client feedback: a bare
                # exception name gave no clue whether the error mattered).
                errors.append(f"{envelope_id}: decrypt_failed ({type(error).__name__})")

        if metric_type is not None:
            # Defensive filter: also correct against pre-metric_type edge deploys.
            records = [r for r in records if (r.metric_type or METRIC_SLEEP) == metric_type]

        self.cache.save(
            [record.to_dict() for record in records],
            server_id=server_id,
            metric_type=metric_type,
            errors=errors,
        )
        self.store.update(last_sync_at=now_iso())
        if limit is not None:
            records = records[:limit]
        return records, errors

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

    async def sleep_records(
        self, *, limit: int | None = None, fresh: bool = False,
        owner: str | None = None,
    ) -> dict[str, Any]:
        """Return recent sleep sessions with per-day primary selection matching iOS app.

        Each session carries `local_date` (Asia/Shanghai), `has_stage_detail`, and
        `is_in_bed_only` flags. The top-level `daily_summary` picks one primary
        session per local date using the same priority as the iOS app's
        `mergeSameDaySessions`: non-inBedOnly > hasStageDetail > longest duration
        > earliest bedtime.

        `limit` means "how many nights to return", not "how many blobs to fetch".
        F8 per-source assembly creates 2-3 blobs per night (Watch stages + iPhone
        inBed + possibly OtterLife); truncating blobs before per-day selection
        drops the stage-detailed blob and returns inBed-only data. So we fetch ALL
        blobs, run per-day selection, then truncate nights.

        *owner*: if given, only include records whose ``owner_user_id`` starts
        with this prefix.
        """

        records, errors = await self._records_for_metric(METRIC_SLEEP, limit=None, fresh=fresh)
        if owner:
            records = [r for r in records if r.owner_user_id and r.owner_user_id.startswith(owner)]
        sessions: list[dict[str, Any]] = []
        tz_local = datetime.now(timezone.utc).astimezone().tzinfo

        for record in records:
            try:
                payload = record.payload
                session = payload.get("session", payload)
                samples = session.get("samples", [])
                stage_minutes: dict[str, int] = {}
                for sample in samples:
                    stage = sample.get("stage", "unknown")
                    start = sample.get("startDate", "")
                    end = sample.get("endDate", "")
                    if start and end:
                        try:
                            t0 = datetime.fromisoformat(start.replace("Z", "+00:00"))
                            t1 = datetime.fromisoformat(end.replace("Z", "+00:00"))
                            mins = max(int((t1 - t0).total_seconds() / 60), 0)
                        except (ValueError, TypeError):
                            mins = 0
                    else:
                        mins = 0
                    stage_minutes[stage] = stage_minutes.get(stage, 0) + mins

                actual_sleep_stages = {
                    "asleepCore", "asleepDeep", "asleepREM", "asleepUnspecified"
                }
                total_sleep_min = sum(
                    v for k, v in stage_minutes.items() if k in actual_sleep_stages
                )
                distinct_actual = {
                    k for k in stage_minutes if k in actual_sleep_stages and stage_minutes[k] > 0
                }
                has_stage_detail = len(distinct_actual) >= 2
                is_in_bed_only = total_sleep_min == 0

                # Convert sessionDate UTC to local date
                sd_raw = session.get("sessionDate", "")
                try:
                    sd_utc = datetime.fromisoformat(sd_raw.replace("Z", "+00:00"))
                    local_date = sd_utc.astimezone(tz_local).strftime("%Y-%m-%d")
                except (ValueError, TypeError):
                    local_date = sd_raw[:10] if sd_raw else ""

                # Convert bedtime/wakeTime to local ISO for display
                bedtime_raw = session.get("bedtime", "")
                wake_raw = session.get("wakeTime", "")
                try:
                    bedtime_local = datetime.fromisoformat(
                        bedtime_raw.replace("Z", "+00:00")
                    ).astimezone(tz_local).strftime("%Y-%m-%dT%H:%M")
                except (ValueError, TypeError, AttributeError):
                    bedtime_local = bedtime_raw
                try:
                    wake_local = datetime.fromisoformat(
                        wake_raw.replace("Z", "+00:00")
                    ).astimezone(tz_local).strftime("%Y-%m-%dT%H:%M")
                except (ValueError, TypeError, AttributeError):
                    wake_local = wake_raw

                sessions.append({
                    "envelope_id": record.envelope_id,
                    "blob_id": record.blob_id,
                    "local_date": local_date,
                    "session_date_utc": sd_raw,
                    "bedtime": bedtime_local,
                    "wake_time": wake_local,
                    "provenance": session.get("provenance", "healthkitSleep"),
                    "total_sleep_minutes": total_sleep_min,
                    "has_stage_detail": has_stage_detail,
                    "is_in_bed_only": is_in_bed_only,
                    "stage_minutes": stage_minutes,
                    "sample_count": len(samples),
                    "heart_rate_samples": len(payload.get("heartRateSamples", [])),
                    "respiratory_rate_samples": len(payload.get("respiratoryRateSamples", [])),
                })
            except (KeyError, TypeError, VaultbeatCryptoError, ValueError) as error:
                errors.append(f"{record.envelope_id}: parse_failed ({type(error).__name__}: {error})")

        daily_summary = _select_primary_sessions(sessions)

        if limit is not None:
            daily_summary = daily_summary[:limit]
            kept_dates = {d["date"] for d in daily_summary}
            sessions = [s for s in sessions if s.get("local_date") in kept_dates]

        summary = {
            "daily_summary": daily_summary,
            "sessions": sessions,
            "count": len(sessions),
        }
        _attach_errors(summary, errors)
        return _attach_owner_guard(summary, records, owner)

    async def sleep_detail_records(
        self, *, limit: int | None = None, fresh: bool = False,
        owner: str | None = None,
    ) -> dict[str, Any]:
        """Return per-night time-aligned HR + RR + sleep stage data.

        Each vital-sign sample is tagged with the sleep stage active at that
        moment.  Output is one object per night (primary session only), sorted
        newest-first, each containing a chronological ``timeline`` array.

        *owner*: if given, only include records whose ``owner_user_id`` starts
        with this prefix (e.g. ``"dce9b9cf"`` or ``"f8350dfc"``).
        """

        records, errors = await self._records_for_metric(METRIC_SLEEP, limit=None, fresh=fresh)
        if owner:
            records = [r for r in records if r.owner_user_id and r.owner_user_id.startswith(owner)]
        tz_local = datetime.now(timezone.utc).astimezone().tzinfo

        def _to_local_iso(raw: str) -> str:
            try:
                return datetime.fromisoformat(
                    raw.replace("Z", "+00:00")
                ).astimezone(tz_local).strftime("%Y-%m-%dT%H:%M:%S")
            except (ValueError, TypeError, AttributeError):
                return raw

        def _to_local_short(raw: str) -> str:
            try:
                return datetime.fromisoformat(
                    raw.replace("Z", "+00:00")
                ).astimezone(tz_local).strftime("%Y-%m-%dT%H:%M")
            except (ValueError, TypeError, AttributeError):
                return raw

        def _stage_at(ts_utc: str, stage_intervals: list[tuple[float, float, str]]) -> str:
            try:
                t = datetime.fromisoformat(ts_utc.replace("Z", "+00:00")).timestamp()
            except (ValueError, TypeError):
                return "unknown"
            for start_ts, end_ts, stage in stage_intervals:
                if start_ts <= t <= end_ts:
                    return stage
            return "between_stages"

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

                sd_raw = session.get("sessionDate", "")
                try:
                    sd_utc = datetime.fromisoformat(sd_raw.replace("Z", "+00:00"))
                    local_date = sd_utc.astimezone(tz_local).strftime("%Y-%m-%d")
                except (ValueError, TypeError):
                    local_date = sd_raw[:10] if sd_raw else ""

                total_sleep_min = 0
                for s in samples:
                    stage = s.get("stage", "")
                    if stage in actual_sleep_stages:
                        try:
                            t0 = datetime.fromisoformat(s["startDate"].replace("Z", "+00:00"))
                            t1 = datetime.fromisoformat(s["endDate"].replace("Z", "+00:00"))
                            total_sleep_min += max(int((t1 - t0).total_seconds() / 60), 0)
                        except (ValueError, TypeError, KeyError):
                            pass

                stage_intervals: list[tuple[float, float, str]] = []
                stage_minutes: dict[str, int] = {}
                for s in samples:
                    try:
                        t0 = datetime.fromisoformat(s["startDate"].replace("Z", "+00:00"))
                        t1 = datetime.fromisoformat(s["endDate"].replace("Z", "+00:00"))
                        stg = s.get("stage", "unknown")
                        stage_intervals.append((t0.timestamp(), t1.timestamp(), stg))
                        mins = max(int((t1 - t0).total_seconds() / 60), 0)
                        stage_minutes[stg] = stage_minutes.get(stg, 0) + mins
                    except (ValueError, TypeError, KeyError):
                        pass
                stage_intervals.sort(key=lambda x: x[0])

                stage_intervals_out: list[dict[str, str]] = []
                for si_start, si_end, si_stage in stage_intervals:
                    stage_intervals_out.append({
                        "stage": si_stage,
                        "start": datetime.fromtimestamp(si_start, tz=tz_local).strftime("%Y-%m-%dT%H:%M:%S"),
                        "end": datetime.fromtimestamp(si_end, tz=tz_local).strftime("%Y-%m-%dT%H:%M:%S"),
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
                        "time": _to_local_iso(ts_raw),
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

                all_nights.append({
                    "envelope_id": record.envelope_id,
                    "local_date": local_date,
                    "bedtime": _to_local_short(session.get("bedtime", "")),
                    "wake_time": _to_local_short(session.get("wakeTime", "")),
                    "total_sleep_minutes": total_sleep_min,
                    "has_stage_detail": has_stage_detail,
                    "is_in_bed_only": is_in_bed_only,
                    "stage_minutes": stage_minutes,
                    "stage_intervals": stage_intervals_out,
                    "stage_vitals": stage_vitals,
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
            result_nights.append(best)

        if limit is not None:
            result_nights = result_nights[:limit]

        summary = {
            "nights": result_nights,
            "count": len(result_nights),
        }
        _attach_errors(summary, errors)
        return _attach_owner_guard(summary, records, owner)

    async def water_intake_summary(self, *, limit: int | None = None, owner: str | None = None, fresh: bool = False) -> dict[str, Any]:
        """Return recent daily water intake plus the computed average over the window."""

        records, errors = await self._records_for_metric(METRIC_WATER, limit=None, fresh=fresh)
        if owner:
            records = [r for r in records if r.owner_user_id and r.owner_user_id.startswith(owner)]
        if limit is not None:
            records = records[:limit]
        days: list[WaterDay] = []
        for record in records:
            try:
                days.append(parse_water_day(record.payload, owner_user_id=record.owner_user_id))
            except (KeyError, TypeError, VaultbeatCryptoError, ValueError) as error:
                errors.append(f"{record.envelope_id}: parse_failed ({type(error).__name__}: {error})")
        summary = summarize_water_intake(days)
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
            records = [r for r in records if r.owner_user_id and r.owner_user_id.startswith(owner)]
        if limit is not None:
            records = records[:limit]
        days: list[BodyDay] = []
        for record in records:
            try:
                days.append(parse_body_day(record.payload, owner_user_id=record.owner_user_id))
            except (KeyError, TypeError, VaultbeatCryptoError, ValueError) as error:
                errors.append(f"{record.envelope_id}: parse_failed ({type(error).__name__}: {error})")
        summary = summarize_weight_trend(days, goal_kg=goal_kg)
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
            records = [r for r in records if r.owner_user_id and r.owner_user_id.startswith(owner)]
        if limit is not None:
            records = records[:limit]
        if not records:
            _LOG.info("no menstrual envelopes present (likely not opted in on iOS)")
        else:
            _LOG.info("decoding %d menstrual envelope(s); sensitive, kept local", len(records))
        days: list[MenstrualDay] = []
        menstrual_owners: set[str] = set()
        for record in records:
            try:
                days.append(parse_menstrual_day(record.payload, owner_user_id=record.owner_user_id))
                if record.owner_user_id:
                    menstrual_owners.add(record.owner_user_id)
            except (KeyError, TypeError, VaultbeatCryptoError, ValueError) as error:
                errors.append(f"{record.envelope_id}: parse_failed ({type(error).__name__}: {error})")
        wrist_readings = await self._wrist_readings_for_owner(menstrual_owners, errors, fresh=fresh)
        summary = summarize_menstrual_cycle(days, wrist_readings=wrist_readings)
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
            records = [r for r in records if r.owner_user_id and r.owner_user_id.startswith(owner)]
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
        if limit is not None:
            days = days[:limit]
        summary = {"days": [d.to_dict() for d in days], "count": len(days)}
        _attach_errors(summary, errors)
        return _attach_owner_guard(summary, records, owner)

    async def resting_hr_records(self, *, limit: int | None = None, owner: str | None = None, fresh: bool = False) -> dict[str, Any]:
        """Return recent resting heart rate samples."""

        records, errors = await self._records_for_metric(METRIC_RESTING_HR, limit=None, fresh=fresh)
        if owner:
            records = [r for r in records if r.owner_user_id and r.owner_user_id.startswith(owner)]
        hr_records: list[RestingHrRecord] = []
        for record in records:
            try:
                hr_records.append(parse_resting_hr_record(record.payload, owner_user_id=record.owner_user_id))
            except (KeyError, TypeError, VaultbeatCryptoError, ValueError) as error:
                errors.append(f"{record.envelope_id}: parse_failed ({type(error).__name__}: {error})")
        # Business-time sort before the cut (see activity_summary's comment).
        hr_records.sort(key=lambda r: r.date, reverse=True)
        if limit is not None:
            hr_records = hr_records[:limit]
        bpms = [r.bpm for r in hr_records]
        average_bpm = sum(bpms) / len(bpms) if bpms else None
        summary = {
            "records": [r.to_dict() for r in hr_records],
            "count": len(hr_records),
            "average_bpm": round(average_bpm, 1) if average_bpm is not None else None,
        }
        _attach_errors(summary, errors)
        return _attach_owner_guard(summary, records, owner)

    async def workout_records(self, *, limit: int | None = None, owner: str | None = None, fresh: bool = False) -> dict[str, Any]:
        """Return recent workout sessions."""

        records, errors = await self._records_for_metric(METRIC_WORKOUT, limit=None, fresh=fresh)
        if owner:
            records = [r for r in records if r.owner_user_id and r.owner_user_id.startswith(owner)]
        workouts: list[WorkoutRecord] = []
        for record in records:
            try:
                workouts.append(parse_workout_record(record.payload, owner_user_id=record.owner_user_id))
            except (KeyError, TypeError, VaultbeatCryptoError, ValueError) as error:
                errors.append(f"{record.envelope_id}: parse_failed ({type(error).__name__}: {error})")
        # Business-time sort before the cut (see activity_summary's comment).
        workouts.sort(key=lambda w: w.start_date, reverse=True)
        if limit is not None:
            workouts = workouts[:limit]
        total_duration = sum(w.duration_seconds for w in workouts)
        summary = {
            "workouts": [w.to_dict() for w in workouts],
            "count": len(workouts),
            "total_duration_hours": round(total_duration / 3600, 2),
        }
        _attach_errors(summary, errors)
        return _attach_owner_guard(summary, records, owner)

    async def mindfulness_summary(self, *, limit: int | None = None, owner: str | None = None, fresh: bool = False) -> dict[str, Any]:
        """Return recent daily mindfulness data (session count and total minutes)."""

        records, errors = await self._records_for_metric(METRIC_MINDFULNESS, limit=None, fresh=fresh)
        if owner:
            records = [r for r in records if r.owner_user_id and r.owner_user_id.startswith(owner)]
        days: list[MindfulnessDay] = []
        for record in records:
            try:
                days.append(parse_mindfulness_day(record.payload, owner_user_id=record.owner_user_id))
            except (KeyError, TypeError, VaultbeatCryptoError, ValueError) as error:
                errors.append(f"{record.envelope_id}: parse_failed ({type(error).__name__}: {error})")
        # Business-time sort before the cut (see activity_summary's comment).
        days.sort(key=lambda d: d.day_start_date, reverse=True)
        if limit is not None:
            days = days[:limit]
        total_minutes = sum(d.total_minutes for d in days)
        summary = {
            "days": [d.to_dict() for d in days],
            "count": len(days),
            "total_minutes": round(total_minutes, 1),
        }
        _attach_errors(summary, errors)
        return _attach_owner_guard(summary, records, owner)

    async def hrv_records(self, *, limit: int | None = None, owner: str | None = None, fresh: bool = False) -> dict[str, Any]:
        """Return recent HRV (SDNN) samples."""

        records, errors = await self._records_for_metric(METRIC_HRV, limit=None, fresh=fresh)
        if owner:
            records = [r for r in records if r.owner_user_id and r.owner_user_id.startswith(owner)]
        hrv_list: list[HRVRecord] = []
        for record in records:
            try:
                hrv_list.append(parse_hrv_record(record.payload, owner_user_id=record.owner_user_id))
            except (KeyError, TypeError, VaultbeatCryptoError, ValueError) as error:
                errors.append(f"{record.envelope_id}: parse_failed ({type(error).__name__}: {error})")
        # Business-time sort before the cut (see activity_summary's comment).
        hrv_list.sort(key=lambda r: r.date, reverse=True)
        if limit is not None:
            hrv_list = hrv_list[:limit]
        sdnns = [r.sdnn_ms for r in hrv_list]
        average_sdnn = sum(sdnns) / len(sdnns) if sdnns else None
        summary = {
            "records": [r.to_dict() for r in hrv_list],
            "count": len(hrv_list),
            "average_sdnn_ms": round(average_sdnn, 1) if average_sdnn is not None else None,
        }
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
            records = [r for r in records if r.owner_user_id and r.owner_user_id.startswith(owner)]
        buckets: list[HRVHourlyBucket] = []
        for record in records:
            try:
                buckets.append(parse_hrv_hourly_record(record.payload, owner_user_id=record.owner_user_id))
            except (KeyError, TypeError, VaultbeatCryptoError, ValueError) as error:
                errors.append(f"{record.envelope_id}: parse_failed ({type(error).__name__}: {error})")
        # Business-time sort before the cut (see activity_summary's comment).
        buckets.sort(key=lambda b: b.date, reverse=True)
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
        _attach_errors(summary, errors)
        return _attach_owner_guard(summary, records, owner)

    async def wrist_temp_records(self, *, limit: int | None = None, owner: str | None = None, fresh: bool = False) -> dict[str, Any]:
        """Return recent sleeping wrist temperature samples."""

        records, errors = await self._records_for_metric(METRIC_WRIST_TEMP, limit=None, fresh=fresh)
        if owner:
            records = [r for r in records if r.owner_user_id and r.owner_user_id.startswith(owner)]
        temp_list: list[WristTempRecord] = []
        for record in records:
            try:
                temp_list.append(parse_wrist_temp_record(record.payload, owner_user_id=record.owner_user_id))
            except (KeyError, TypeError, VaultbeatCryptoError, ValueError) as error:
                errors.append(f"{record.envelope_id}: parse_failed ({type(error).__name__}: {error})")
        # Business-time sort before the cut (see activity_summary's comment).
        temp_list.sort(key=lambda r: r.date, reverse=True)
        if limit is not None:
            temp_list = temp_list[:limit]
        deltas = [r.temperature_delta_celsius for r in temp_list]
        average_delta = sum(deltas) / len(deltas) if deltas else None
        summary = {
            "records": [r.to_dict() for r in temp_list],
            "count": len(temp_list),
            "average_delta_celsius": round(average_delta, 2) if average_delta is not None else None,
        }
        _attach_errors(summary, errors)
        return _attach_owner_guard(summary, records, owner)

    async def basal_energy_records(self, *, limit: int | None = None, owner: str | None = None, fresh: bool = False) -> dict[str, Any]:
        """Return recent basal-energy-burned samples (Watch BMR estimate, kcal).

        Watch typically emits hourly samples, so a `limit` of ~168 covers a week.
        Groups by local calendar day for a daily-BMR view (usually 1500-2000
        kcal for an active young adult). Combine with `get_activity`'s
        `active_energy_kcal` for a proper TDEE (see `total_energy_burned`).
        """

        records, errors = await self._records_for_metric(METRIC_BASAL_ENERGY, limit=None, fresh=fresh)
        if owner:
            records = [r for r in records if r.owner_user_id and r.owner_user_id.startswith(owner)]
        if limit is not None:
            records = records[:limit]

        parsed: list[BasalEnergyRecord] = []
        for record in records:
            try:
                parsed.append(parse_basal_energy_record(record.payload, owner_user_id=record.owner_user_id))
            except (KeyError, TypeError, VaultbeatCryptoError, ValueError) as error:
                errors.append(f"{record.envelope_id}: parse_failed ({type(error).__name__}: {error})")

        # Group by LOCAL calendar day. The old `r.date[:10]` cut took the UTC
        # date prefix, which for a UTC+8 user mis-filed every sample between
        # local 00:00-08:00 (= previous UTC day) into the PREVIOUS day —
        # ~600-700 kcal of sleeping basal per night, self-cancelling on middle
        # days but visibly wrong on the edges (2026-07-24 11:40 "today" showed
        # 191.9 kcal when local 00:00-11:40 alone should hold ~900).
        by_day: dict[str, float] = {}
        for r in parsed:
            try:
                day_key = _local_calendar_day(_parse_iso8601(r.date)).isoformat()
            except ValueError:
                day_key = r.date[:10]  # unparseable date: degrade to old cut
            by_day[day_key] = by_day.get(day_key, 0.0) + r.kcal
        daily: list[dict[str, Any]] = sorted(
            [{"day": d, "basal_kcal": round(kcal, 1)} for d, kcal in by_day.items()],
            key=lambda x: str(x["day"]),
            reverse=True,
        )
        latest_day_kcal: float | None = float(daily[0]["basal_kcal"]) if daily else None
        avg_daily: float | None = (
            round(sum(float(d["basal_kcal"]) for d in daily) / len(daily), 1) if daily else None
        )

        summary = {
            "sample_count": len(parsed),
            "day_count": len(daily),
            "latest_day_basal_kcal": latest_day_kcal,
            "average_daily_basal_kcal": avg_daily,
            "daily": daily[:30],  # newest 30 days
        }
        _attach_errors(summary, errors)
        return _attach_owner_guard(summary, records, owner)

    async def total_energy_burned(
        self,
        *,
        days: int = 7,
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
        """

        # Pull both underlying streams in parallel — this is a compute
        # aggregation, no new envelope fetches once both caches are warm.
        basal_task = asyncio.create_task(self.basal_energy_records(owner=owner, fresh=fresh))
        activity_task = asyncio.create_task(self.activity_summary(owner=owner, fresh=fresh))
        basal = await basal_task
        activity = await activity_task

        # Build lookups: day -> kcal
        basal_by_day = {d["day"]: d["basal_kcal"] for d in basal.get("daily", [])}

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

        all_days = sorted(set(basal_by_day) | set(active_by_day), reverse=True)[:days]

        # Today is a PARTIAL day (its basal/active are still accumulating) —
        # 2026-07-24 at 11:40 it showed 213 kcal and dragged a ~2617 average
        # down to 2136, a distortion the size of an entire diet deficit. It
        # stays in `days` (callers may want the live number) but is flagged
        # and excluded from the average.
        today_key = _local_calendar_day(datetime.now(timezone.utc)).isoformat()

        out = []
        for day in all_days:
            b = basal_by_day.get(day)
            a = active_by_day.get(day, 0.0)
            total = (b or 0.0) + a
            out.append({
                "day": day,
                "basal_kcal": b,
                "active_kcal": round(a, 1),
                "total_kcal": round(total, 1),
                "basal_missing": b is None,
                "partial": day == today_key,
            })

        totals_with_basal: list[float] = [
            float(d["total_kcal"] or 0.0)
            for d in out
            if not d["basal_missing"] and not d["partial"]
        ]
        avg_tdee: float | None = (
            round(sum(totals_with_basal) / len(totals_with_basal), 1)
            if totals_with_basal
            else None
        )

        return {
            "days_returned": len(out),
            "average_tdee_kcal": avg_tdee,
            "average_note": "average excludes today (partial, still accumulating) and basal-missing days",
            "days": out,
            "basal_errors": basal.get("errors", []),
            "activity_errors": activity.get("errors", []),
        }

    async def vo2max_records(self, *, limit: int | None = None, owner: str | None = None, fresh: bool = False) -> dict[str, Any]:
        """Return recent VO2Max samples with a peak / trough summary.

        Motivated by the 2026-11 萨武神山 备训 trend tracking (target 45+ from
        current ~39.3, health.md 训练方案段). Watch computes VO2Max periodically
        during outdoor brisk walk/run bouts — sparse samples over weeks/months,
        so newest first and no artificial gap-filling.
        """

        records, errors = await self._records_for_metric(METRIC_VO2MAX, limit=None, fresh=fresh)
        if owner:
            records = [r for r in records if r.owner_user_id and r.owner_user_id.startswith(owner)]
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
        _attach_errors(summary, errors)
        return _attach_owner_guard(summary, records, owner)

    async def symptom_summary(self, *, limit: int | None = None, fresh: bool = False) -> dict[str, Any]:
        """Return recent symptom days grouped by data owner.

        Symptom blobs only arrive when someone opted in on iOS (own AI or the
        partner-AI ladder). Sensitive — decoded locally, never re-exported.
        """

        records, errors = await self._records_for_metric(METRIC_SYMPTOM, limit=limit, fresh=fresh)
        if not records:
            _LOG.info("no symptom envelopes present (likely not opted in on iOS)")
        else:
            _LOG.info("decoding %d symptom envelope(s); sensitive, kept local", len(records))
        days: list[SymptomDay] = []
        for record in records:
            try:
                days.append(parse_symptom_day(record.payload, owner_user_id=record.owner_user_id))
            except (KeyError, TypeError, VaultbeatCryptoError, ValueError) as error:
                errors.append(f"{record.envelope_id}: parse_failed ({type(error).__name__}: {error})")
        summary = summarize_symptoms(days)
        return _attach_errors(summary, errors)

    async def notes_summary(
        self, *, limit: int | None = None, target_kind: str | None = None, fresh: bool = False
    ) -> dict[str, Any]:
        """Return recent free-text notes grouped by target kind (sleep/menstrual).

        Notes are written manually in Vaultbeat by either partner; each carries its
        writer (owner_user_id). Sensitive free text — decoded locally, never
        re-exported.
        """

        records, errors = await self._records_for_metric(METRIC_NOTE, limit=limit, fresh=fresh)
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
        summary = summarize_notes(notes, target_kind=target_kind)
        return _attach_errors(summary, errors)

    async def strength_summary(
        self, *, limit: int | None = None, limit_days: int | None = None, fresh: bool = False
    ) -> dict[str, Any]:
        """Return recent strength-training sessions with exercise-level detail.

        Logged manually in Vaultbeat (HealthKit's workout type has no per-set
        data). Owner's own sessions only — strength has no partner fan-out.
        """

        records, errors = await self._records_for_metric(METRIC_STRENGTH, limit=limit, fresh=fresh)
        if not records:
            _LOG.info("no strength envelopes present (nothing logged yet)")
        entries: list[StrengthRecord] = []
        for record in records:
            try:
                entries.append(parse_strength(record.payload, owner_user_id=record.owner_user_id))
            except (KeyError, TypeError, VaultbeatCryptoError, ValueError) as error:
                errors.append(f"{record.envelope_id}: parse_failed ({type(error).__name__}: {error})")
        summary = summarize_strength(entries, limit_days=limit_days)
        return _attach_errors(summary, errors)

    async def log_strength_entry(
        self,
        *,
        date: str,
        exercises: list[dict[str, Any]],
        note: str | None = None,
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

        Requires a bind that carried owner identity through the handshake
        (owner_user_id / owner_public_key_base64 / owner_device_id) — a bind
        from before this feature shipped predates that and must re-bind.
        """

        from datetime import date as _date_type

        requested_day = _date_type.fromisoformat(date)
        normalized_exercises = _normalize_strength_exercises(exercises)
        cleaned_note = note.strip() if isinstance(note, str) and note.strip() else None

        config = self.store.require_bound()
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
                "Run `vaultbeat-mcp-local bind` again to re-pair."
            )

        existing_summary = await self.strength_summary(fresh=True)
        existing_entry_id: str | None = None
        existing_created_at: str | None = None
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
                break

        now_iso_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        entry_id = existing_entry_id or ("strength-" + secrets.token_hex(16))
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

        server_response = await self._client(config).write_strength_blob(
            server_token, blob=blob, envelopes=envelope_rows
        )

        refreshed = await self.strength_summary(fresh=True)
        session = next(
            (s for s in refreshed.get("sessions", []) if s.get("entry_id") == entry_id), None
        )
        return {
            "entry_id": entry_id,
            "date": requested_day.isoformat(),
            "updated_existing_day": existing_entry_id is not None,
            "server_response": server_response,
            "session": session,
        }

    async def food_summary(
        self, *, limit: int | None = None, limit_days: int | None = None, fresh: bool = False
    ) -> dict[str, Any]:
        """Return recent daily food-intake logs with per-day meals + items.

        Logged manually in Vaultbeat (no automatic HealthKit source — HealthKit's
        dietary types are point samples that don't survive as "what a meal actually
        was"). Owner's own days only — food has no partner fan-out in v1.
        """

        records, errors = await self._records_for_metric(METRIC_FOOD, limit=limit, fresh=fresh)
        if not records:
            _LOG.info("no food envelopes present (nothing logged yet)")
        entries: list[FoodRecord] = []
        for record in records:
            try:
                entries.append(parse_food(record.payload, owner_user_id=record.owner_user_id))
            except (KeyError, TypeError, VaultbeatCryptoError, ValueError) as error:
                errors.append(f"{record.envelope_id}: parse_failed ({type(error).__name__}: {error})")
        summary = summarize_food(entries, limit_days=limit_days)
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
          ``_merge_food_meals``). ``note=None`` keeps the existing day note.

        Requires a bind that carried owner identity through the handshake — a
        bind from before this feature shipped predates that and must re-bind.
        """

        from datetime import date as _date_type

        requested_day = _date_type.fromisoformat(date)
        normalized_meals = _normalize_food_meals(meals)
        cleaned_note = note.strip() if isinstance(note, str) and note.strip() else None

        config = self.store.require_bound()
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
                "Run `vaultbeat-mcp-local bind` again to re-pair."
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

        if merge and existing_day is not None:
            existing_meals = existing_day.get("meals")
            normalized_meals = _merge_food_meals(
                existing_meals if isinstance(existing_meals, list) else [], normalized_meals
            )
            if cleaned_note is None:
                existing_note = existing_day.get("note")
                if isinstance(existing_note, str) and existing_note.strip():
                    cleaned_note = existing_note

        now_iso_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        entry_id = existing_entry_id or ("food-" + secrets.token_hex(16))
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

        server_response = await self._client(config).write_food_blob(
            server_token, blob=blob, envelopes=envelope_rows
        )

        refreshed = await self.food_summary(fresh=True)
        day = next(
            (d for d in refreshed.get("days", []) if d.get("entry_id") == entry_id), None
        )
        return {
            "entry_id": entry_id,
            "date": requested_day.isoformat(),
            "updated_existing_day": existing_entry_id is not None,
            "merge_mode": merge,
            "server_response": server_response,
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
        (visible to `get_weight_trend`). It does NOT write to Apple Health
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

        config = self.store.require_bound()
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
                "Run `vaultbeat-mcp-local bind` again to re-pair."
            )

        if weight_kg <= 0 or weight_kg > 500:
            raise ValueError(f"weight_kg out of realistic range: {weight_kg}")

        # Match iOS: dayID = "body-{dayStart.epoch}", dayStart = local midnight
        local_tz = datetime.now(timezone.utc).astimezone().tzinfo
        day_start_local = datetime(
            requested_day.year, requested_day.month, requested_day.day, tzinfo=local_tz
        )
        day_start_epoch = int(day_start_local.timestamp())
        day_id = f"body-{day_start_epoch}"

        plaintext_obj = {
            "dayID": day_id,
            "dayStartDate": day_start_local.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
            "weightKg": float(weight_kg),
            "bodyFatPercent": None,
            "bmi": None,
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
            "id": day_id,
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

        server_response = await self._client(config).write_body_blob(
            server_token, blob=blob, envelopes=envelope_rows
        )

        # Verify by re-reading (cache-bypass)
        refreshed = await self.weight_trend_summary(fresh=True, limit=30)
        latest = refreshed.get("days", [{}])[0] if refreshed.get("days") else {}
        return {
            "day_id": day_id,
            "date": requested_day.isoformat(),
            "weight_kg": weight_kg,
            "server_response": server_response,
            "latest_after_write": latest,
        }

    async def log_note(
        self,
        *,
        text: str,
        kind: str = "general",
        date: str | None = None,
    ) -> dict[str, Any]:
        """Encrypt and upsert one agent-authored note on the owner's behalf.

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
        agent-authored note per (kind, local day): logging the same kind+day
        again reuses the noteID and overwrites in place — to APPEND to an
        existing note, `get_notes` first and resend the combined text.
        Sealed for owner + this MCP server (own AI only, no partner fan-out).
        """

        from datetime import date as _date_type

        allowed_kinds = {"mood", "general"}
        if kind not in allowed_kinds:
            raise ValueError(
                f"unsupported note kind {kind!r}; expected one of {sorted(allowed_kinds)} "
                "(sleep/menstrual notes are iOS-authored)"
            )
        cleaned_text = text.strip() if isinstance(text, str) else ""
        if not cleaned_text:
            raise ValueError("note text must be a non-empty string")
        requested_day = _date_type.fromisoformat(date) if date else _date_type.today()

        config = self.store.require_bound()
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
                "Run `vaultbeat-mcp-local bind` again to re-pair."
            )

        # Upsert key: an existing agent-authored note for the same (kind, day).
        existing_note_id: str | None = None
        existing_created_at: str | None = None
        existing_summary = await self.notes_summary(target_kind=kind, fresh=True)
        for kind_group in existing_summary.get("kinds", []):
            for note_row in kind_group.get("notes", []):
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
                    break
            if existing_note_id:
                break

        now_iso_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        note_id = existing_note_id or ("note-" + secrets.token_hex(16))
        plaintext_obj = {
            "noteID": note_id,
            "targetKind": kind,
            "targetDate": _local_midnight_iso(requested_day),
            "text": cleaned_text,
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

        server_response = await self._client(config).write_note_blob(
            server_token, blob=blob, envelopes=envelope_rows
        )

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
            "date": requested_day.isoformat(),
            "updated_existing_note": existing_note_id is not None,
            "server_response": server_response,
            "note": written,
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

        config = self.store.load()
        if not config:
            add(
                "config", False, f"no config at {self.store.path}",
                hint="Run `vaultbeat-mcp bind` to initialize and pair with the iOS app.",
            )
            return {"ok": False, "checks": checks}
        add("config", True, str(self.store.path))

        add(
            "identity_key", bool(config.private_key_base64),
            "private key present" if config.private_key_base64 else "private key missing",
            hint="The Keychain entry is gone or unreadable. Re-run `vaultbeat-mcp bind` to mint a fresh identity.",
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
            hint="Run `vaultbeat-mcp bind`, then scan the QR with Vaultbeat on iOS "
            "(Settings → Data & AI → MCP Server). Codes expire after 10 minutes — "
            "if the phone scanned but this side stayed pending, re-run bind for a fresh code.",
        )
        # Informational only: legacy bindings predate the owner-identity
        # handshake and still decrypt fine — absence must not fail the doctor.
        add(
            "owner_identity", True,
            "owner identity received"
            if config.owner_user_id and config.owner_public_key_base64
            else "owner identity missing (legacy binding — reads unaffected)",
        )

        if config.is_bound and reachable:
            try:
                records, errors = await self._records_for_metric(
                    METRIC_SLEEP, limit=1, fresh=True
                )
                if errors and not records:
                    add(
                        "data_roundtrip", False,
                        f"fetch ok but decrypt failed ({errors[0]})",
                        hint="The stored key can no longer decrypt your data — "
                        "delete this server in the iOS app and bind again.",
                    )
                else:
                    add("data_roundtrip", True, f"decrypted {len(records)} sleep record(s)")
            except Exception as error:  # noqa: BLE001 — diagnostic surface, report everything
                add(
                    "data_roundtrip", False, f"{type(error).__name__}: {error}",
                    hint="The server token may have been revoked — check the server "
                    "still exists in the iOS app (Settings → Data & AI), or bind again.",
                )

        return {"ok": all(check["ok"] for check in checks), "checks": checks}

    def status(self) -> dict[str, Any]:
        config = self.store.load()
        if not config:
            return {"initialized": False, "bound": False}

        return {
            "initialized": True,
            "bound": config.is_bound,
            "server_name": config.server_name,
            "server_id": config.server_id,
            "api_base_url": config.api_base_url,
            "poll_id": config.poll_id,
            "public_key_base64": config.public_key_base64,
            "bound_at": config.bound_at,
            "last_sync_at": config.last_sync_at,
            "owner_identity_bound": bool(config.owner_user_id and config.owner_public_key_base64),
            "owner_device_bound": bool(config.owner_device_id),
            "config_path": str(self.store.path),
        }

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
        owner_user_id = blob.get("owner_user_id")
        return DecryptedRecord(
            envelope_id=str(row["id"]),
            blob_id=str(row["blob_id"]),
            metric_type=blob.get("metric_type"),
            created_at=blob.get("created_at"),
            payload=decode_json_payload(plaintext),
            owner_user_id=str(owner_user_id) if owner_user_id else None,
        )
