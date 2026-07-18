from __future__ import annotations

import asyncio
import json
import logging
import secrets
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Protocol

from vaultbeat_mcp_local.cache import LocalRecordCache
from vaultbeat_mcp_local.client import PollBindingResult, VaultbeatCloudClient
from vaultbeat_mcp_local.crypto import (
    VaultbeatCryptoError,
    decode_json_payload,
    decrypt_blob_payload,
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
METRIC_WRIST_TEMP = "wrist_temp"
METRIC_SYMPTOM = "symptom"
METRIC_NOTE = "note"

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
        METRIC_WRIST_TEMP,
        METRIC_SYMPTOM,
        METRIC_NOTE,
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
        return {"record_id": self.record_id, "date": self.date, "bpm": self.bpm, "owner_user_id": self.owner_user_id}


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
        return {"record_id": self.record_id, "date": self.date, "sdnn_ms": self.sdnn_ms, "owner_user_id": self.owner_user_id}


@dataclass(frozen=True)
class WristTempRecord:
    """One sleeping wrist temperature sample decoded from a metric_type="wrist_temp" blob."""

    record_id: str
    date: str
    temperature_delta_celsius: float
    owner_user_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "date": self.date,
            "temperature_delta_celsius": self.temperature_delta_celsius,
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
            "text": self.text,
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

        envelope_rows = await self._client(config).sync(server_token, metric_type=metric_type)
        records = []
        errors: list[str] = []

        for row in envelope_rows:
            try:
                records.append(self._decrypt_row(row, config))
            except (KeyError, TypeError, VaultbeatCryptoError, ValueError) as error:
                envelope_id = str(row.get("id", "<unknown>"))
                errors.append(f"{envelope_id}: {type(error).__name__}")

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
                errors.append(f"{record.envelope_id}: {type(error).__name__}")

        daily_summary = _select_primary_sessions(sessions)

        if limit is not None:
            daily_summary = daily_summary[:limit]
            kept_dates = {d["date"] for d in daily_summary}
            sessions = [s for s in sessions if s.get("local_date") in kept_dates]

        return {
            "daily_summary": daily_summary,
            "sessions": sessions,
            "count": len(sessions),
            "errors": errors,
        }

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
                errors.append(f"{record.envelope_id}: {type(error).__name__}")

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

        return {
            "nights": result_nights,
            "count": len(result_nights),
            "errors": errors,
        }

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
                errors.append(f"{record.envelope_id}: {type(error).__name__}")
        summary = summarize_water_intake(days)
        summary["errors"] = errors
        return summary

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
                errors.append(f"{record.envelope_id}: {type(error).__name__}")
        summary = summarize_weight_trend(days, goal_kg=goal_kg)
        summary["errors"] = errors
        return summary

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
                errors.append(f"{record.envelope_id}: {type(error).__name__}")
        wrist_readings = await self._wrist_readings_for_owner(menstrual_owners, errors, fresh=fresh)
        summary = summarize_menstrual_cycle(days, wrist_readings=wrist_readings)
        summary["errors"] = errors
        return summary

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
                errors.append(f"{record.envelope_id}: {type(error).__name__}")
        return readings or None

    async def activity_summary(self, *, limit: int | None = None, owner: str | None = None, fresh: bool = False) -> dict[str, Any]:
        """Return recent daily activity ring data (steps, energy, exercise, stand, distance)."""

        records, errors = await self._records_for_metric(METRIC_ACTIVITY, limit=None, fresh=fresh)
        if owner:
            records = [r for r in records if r.owner_user_id and r.owner_user_id.startswith(owner)]
        if limit is not None:
            records = records[:limit]
        days: list[ActivityDay] = []
        for record in records:
            try:
                days.append(parse_activity_day(record.payload, owner_user_id=record.owner_user_id))
            except (KeyError, TypeError, VaultbeatCryptoError, ValueError) as error:
                errors.append(f"{record.envelope_id}: {type(error).__name__}")
        return {"days": [d.to_dict() for d in days], "count": len(days), "errors": errors}

    async def resting_hr_records(self, *, limit: int | None = None, owner: str | None = None, fresh: bool = False) -> dict[str, Any]:
        """Return recent resting heart rate samples."""

        records, errors = await self._records_for_metric(METRIC_RESTING_HR, limit=None, fresh=fresh)
        if owner:
            records = [r for r in records if r.owner_user_id and r.owner_user_id.startswith(owner)]
        if limit is not None:
            records = records[:limit]
        hr_records: list[RestingHrRecord] = []
        for record in records:
            try:
                hr_records.append(parse_resting_hr_record(record.payload, owner_user_id=record.owner_user_id))
            except (KeyError, TypeError, VaultbeatCryptoError, ValueError) as error:
                errors.append(f"{record.envelope_id}: {type(error).__name__}")
        bpms = [r.bpm for r in hr_records]
        average_bpm = sum(bpms) / len(bpms) if bpms else None
        return {
            "records": [r.to_dict() for r in hr_records],
            "count": len(hr_records),
            "average_bpm": round(average_bpm, 1) if average_bpm is not None else None,
            "errors": errors,
        }

    async def workout_records(self, *, limit: int | None = None, owner: str | None = None, fresh: bool = False) -> dict[str, Any]:
        """Return recent workout sessions."""

        records, errors = await self._records_for_metric(METRIC_WORKOUT, limit=None, fresh=fresh)
        if owner:
            records = [r for r in records if r.owner_user_id and r.owner_user_id.startswith(owner)]
        if limit is not None:
            records = records[:limit]
        workouts: list[WorkoutRecord] = []
        for record in records:
            try:
                workouts.append(parse_workout_record(record.payload, owner_user_id=record.owner_user_id))
            except (KeyError, TypeError, VaultbeatCryptoError, ValueError) as error:
                errors.append(f"{record.envelope_id}: {type(error).__name__}")
        total_duration = sum(w.duration_seconds for w in workouts)
        return {
            "workouts": [w.to_dict() for w in workouts],
            "count": len(workouts),
            "total_duration_hours": round(total_duration / 3600, 2),
            "errors": errors,
        }

    async def mindfulness_summary(self, *, limit: int | None = None, owner: str | None = None, fresh: bool = False) -> dict[str, Any]:
        """Return recent daily mindfulness data (session count and total minutes)."""

        records, errors = await self._records_for_metric(METRIC_MINDFULNESS, limit=None, fresh=fresh)
        if owner:
            records = [r for r in records if r.owner_user_id and r.owner_user_id.startswith(owner)]
        if limit is not None:
            records = records[:limit]
        days: list[MindfulnessDay] = []
        for record in records:
            try:
                days.append(parse_mindfulness_day(record.payload, owner_user_id=record.owner_user_id))
            except (KeyError, TypeError, VaultbeatCryptoError, ValueError) as error:
                errors.append(f"{record.envelope_id}: {type(error).__name__}")
        total_minutes = sum(d.total_minutes for d in days)
        return {
            "days": [d.to_dict() for d in days],
            "count": len(days),
            "total_minutes": round(total_minutes, 1),
            "errors": errors,
        }

    async def hrv_records(self, *, limit: int | None = None, owner: str | None = None, fresh: bool = False) -> dict[str, Any]:
        """Return recent HRV (SDNN) samples."""

        records, errors = await self._records_for_metric(METRIC_HRV, limit=None, fresh=fresh)
        if owner:
            records = [r for r in records if r.owner_user_id and r.owner_user_id.startswith(owner)]
        if limit is not None:
            records = records[:limit]
        hrv_list: list[HRVRecord] = []
        for record in records:
            try:
                hrv_list.append(parse_hrv_record(record.payload, owner_user_id=record.owner_user_id))
            except (KeyError, TypeError, VaultbeatCryptoError, ValueError) as error:
                errors.append(f"{record.envelope_id}: {type(error).__name__}")
        sdnns = [r.sdnn_ms for r in hrv_list]
        average_sdnn = sum(sdnns) / len(sdnns) if sdnns else None
        return {
            "records": [r.to_dict() for r in hrv_list],
            "count": len(hrv_list),
            "average_sdnn_ms": round(average_sdnn, 1) if average_sdnn is not None else None,
            "errors": errors,
        }

    async def wrist_temp_records(self, *, limit: int | None = None, owner: str | None = None, fresh: bool = False) -> dict[str, Any]:
        """Return recent sleeping wrist temperature samples."""

        records, errors = await self._records_for_metric(METRIC_WRIST_TEMP, limit=None, fresh=fresh)
        if owner:
            records = [r for r in records if r.owner_user_id and r.owner_user_id.startswith(owner)]
        if limit is not None:
            records = records[:limit]
        temp_list: list[WristTempRecord] = []
        for record in records:
            try:
                temp_list.append(parse_wrist_temp_record(record.payload, owner_user_id=record.owner_user_id))
            except (KeyError, TypeError, VaultbeatCryptoError, ValueError) as error:
                errors.append(f"{record.envelope_id}: {type(error).__name__}")
        deltas = [r.temperature_delta_celsius for r in temp_list]
        average_delta = sum(deltas) / len(deltas) if deltas else None
        return {
            "records": [r.to_dict() for r in temp_list],
            "count": len(temp_list),
            "average_delta_celsius": round(average_delta, 2) if average_delta is not None else None,
            "errors": errors,
        }

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
                errors.append(f"{record.envelope_id}: {type(error).__name__}")
        summary = summarize_symptoms(days)
        summary["errors"] = errors
        return summary

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
                errors.append(f"{record.envelope_id}: {type(error).__name__}")
        summary = summarize_notes(notes, target_kind=target_kind)
        summary["errors"] = errors
        return summary

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
