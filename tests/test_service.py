from __future__ import annotations

import base64
import asyncio
import json
import stat
from datetime import date, datetime
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import x25519
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from vaultbeat_mcp_local.client import PollBindingResult
from vaultbeat_mcp_local.crypto import ENVELOPE_INFO
from vaultbeat_mcp_local.crypto import VaultbeatCryptoError
from vaultbeat_mcp_local.service import (
    BodyDay,
    detect_ovulation_from_wrist_temp,
    MenstrualDay,
    MenstrualSample,
    VaultbeatLocalService,
    WaterDay,
    parse_body_day,
    parse_menstrual_day,
    parse_water_day,
    summarize_menstrual_cycle,
    summarize_water_intake,
    summarize_weight_trend,
)
from vaultbeat_mcp_local.store import ConfigStore


class FakeCloudClient:
    def __init__(self) -> None:
        self.poll_id: str | None = None
        self.synced_with_token: str | None = None
        self.envelopes: list[dict[str, Any]] = []
        # Every sync call's metric_type, in order — lets cache tests count
        # round trips and assert the server-side narrowing parameter.
        self.sync_calls: list[str | None] = []
        # When True the fake ignores metric_type (an old edge deployment),
        # proving the service's defensive local filter stays authoritative.
        self.ignores_metric_type = False
        # Owner identity the bind handshake would carry.
        self.owner_user_id: str | None = None
        self.owner_public_key_base64: str | None = None
        self.owner_device_id: str | None = None

    async def poll_binding(self, poll_id: str) -> PollBindingResult:
        self.poll_id = poll_id
        return PollBindingResult(
            status="bound",
            server_id="server-1",
            server_token="token-1",
            owner_user_id=self.owner_user_id,
            owner_public_key_base64=self.owner_public_key_base64,
            owner_device_id=self.owner_device_id,
        )

    async def sync(
        self, server_token: str, *, metric_type: str | None = None
    ) -> list[dict[str, Any]]:
        self.synced_with_token = server_token
        self.sync_calls.append(metric_type)
        if metric_type is None or self.ignores_metric_type:
            return self.envelopes

        def _matches(row: dict[str, Any]) -> bool:
            blob = row.get("encrypted_sleep_blobs") or {}
            effective = blob.get("metric_type") or "sleep"
            return effective == metric_type

        return [row for row in self.envelopes if _matches(row)]


def _make_envelope(
    public_key_base64: str,
    plaintext: bytes,
    *,
    metric_type: str = "sleep",
    envelope_id: str = "env-1",
    blob_id: str = "blob-1",
    owner_user_id: str | None = None,
) -> dict[str, Any]:
    recipient_public = x25519.X25519PublicKey.from_public_bytes(base64.b64decode(public_key_base64))
    sender_private = x25519.X25519PrivateKey.generate()
    shared_secret = sender_private.exchange(recipient_public)
    wrapping_key = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=b"",
        info=ENVELOPE_INFO,
    ).derive(shared_secret)
    dek = b"\x07" * 32
    wrapped_nonce = b"\x03" * 12
    wrapped_dek = wrapped_nonce + AESGCM(wrapping_key).encrypt(wrapped_nonce, dek, None)
    sender_public_raw = sender_private.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    encrypted_data_key = base64.b64encode(
        json.dumps(
            {
                "senderPublicKeyBase64": base64.b64encode(sender_public_raw).decode(),
                "wrappedSymmetricKeyBase64": base64.b64encode(wrapped_dek).decode(),
            }
        ).encode()
    ).decode()

    ciphertext_nonce = b"\x04" * 12
    ciphertext = ciphertext_nonce + AESGCM(dek).encrypt(ciphertext_nonce, plaintext, None)
    blob: dict[str, Any] = {
        "id": blob_id,
        "ciphertext": base64.b64encode(ciphertext).decode(),
        "metric_type": metric_type,
        "created_at": "2026-04-27T00:00:00Z",
    }
    if owner_user_id is not None:
        blob["owner_user_id"] = owner_user_id
    return {
        "id": envelope_id,
        "blob_id": blob_id,
        "encrypted_data_key": encrypted_data_key,
        "encrypted_sleep_blobs": blob,
    }


def test_binding_session_persists_credentials_and_secure_file(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    cloud = FakeCloudClient()
    service = VaultbeatLocalService(ConfigStore(config_path), cloud)

    session = service.start_binding(server_name="Mac Studio", api_base_url="https://api.test")
    result = asyncio.run(service.poll_once())
    saved = ConfigStore(config_path).load()

    assert result.status == "bound"
    assert cloud.poll_id == session.poll_id
    assert saved is not None
    assert saved.server_id == "server-1"
    assert saved.server_token == "token-1"
    assert saved.poll_id is None
    assert stat.S_IMODE(config_path.stat().st_mode) == 0o600


def test_sync_decrypts_cloud_envelopes(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    cloud = FakeCloudClient()
    service = VaultbeatLocalService(ConfigStore(config_path), cloud)

    service.start_binding(server_name="Mac Studio", api_base_url="https://api.test")
    asyncio.run(service.poll_once())
    config = ConfigStore(config_path).require_bound()
    cloud.envelopes = [_make_envelope(config.public_key_base64, b'{"stage":"asleep"}')]

    records, errors = asyncio.run(service.sync_decrypted_records(limit=10))

    assert errors == []
    assert cloud.synced_with_token == "token-1"
    assert records[0].payload == {"stage": "asleep"}
    assert records[0].blob_id == "blob-1"


def _water_payload(day_id: str, day_start: str, container: float, refill_count: int) -> bytes:
    refills = [
        {"timestamp": day_start, "containerVolumeLiters": container} for _ in range(refill_count)
    ]
    return json.dumps(
        {
            "dayID": day_id,
            "dayStartDate": day_start,
            "containerVolumeLiters": container,
            "refillEvents": refills,
        }
    ).encode()


def _body_payload(day_id: str, day_start: str, weight_kg: float) -> bytes:
    return json.dumps(
        {
            "dayID": day_id,
            "dayStartDate": day_start,
            "weightKg": weight_kg,
            "bodyFatPercent": None,
            "bmi": None,
        }
    ).encode()


def _menstrual_payload(day_id: str, day_start: str, flow: str) -> bytes:
    return json.dumps(
        {
            "dayID": day_id,
            "dayStartDate": day_start,
            "samples": [{"startDate": day_start, "endDate": day_start, "flow": flow}],
        }
    ).encode()


# --- pure decode / aggregate (fabricated dicts, no network) -----------------------------


def test_parse_water_day_counts_refills_and_volume() -> None:
    day = parse_water_day(
        {
            "dayID": "water-1",
            "dayStartDate": "2026-06-05T00:00:00Z",
            "containerVolumeLiters": 6.0,
            "refillEvents": [
                {"timestamp": "2026-06-05T08:30:00Z", "containerVolumeLiters": 6.0},
                {"timestamp": "2026-06-05T18:30:00Z", "containerVolumeLiters": 6.0},
            ],
        }
    )
    assert day.refill_count == 2
    assert day.container_volume_liters == 6.0
    assert day.intake_liters == 12.0


def test_summarize_water_intake_averages_over_window() -> None:
    days = [
        WaterDay("water-1", "2026-06-04T00:00:00Z", 2.0, refill_count=3),  # 6.0 L
        WaterDay("water-2", "2026-06-05T00:00:00Z", 1.0, refill_count=2),  # 2.0 L
    ]
    summary = summarize_water_intake(days)
    assert summary["day_count"] == 2
    assert summary["average_daily_intake_liters"] == 4.0  # (6 + 2) / 2
    assert summary["days"][0]["day_id"] == "water-2"  # newest first


def test_summarize_water_intake_empty_reports_none() -> None:
    summary = summarize_water_intake([])
    assert summary == {"days": [], "average_daily_intake_liters": None, "day_count": 0}


def test_summarize_water_intake_dedups_by_day_id() -> None:
    days = [
        WaterDay("water-1", "2026-06-05T00:00:00Z", 2.0, refill_count=1),  # stale
        WaterDay("water-1", "2026-06-05T12:00:00Z", 2.0, refill_count=4),  # newer wins
    ]
    summary = summarize_water_intake(days)
    assert summary["day_count"] == 1
    assert summary["average_daily_intake_liters"] == 8.0  # 4 refills * 2.0 L


def test_parse_body_day_decodes_weight_and_reserved_nulls() -> None:
    day = parse_body_day(
        {
            "dayID": "body-1",
            "dayStartDate": "2026-06-05T00:00:00Z",
            "weightKg": 82.5,
            "bodyFatPercent": None,
            "bmi": None,
        }
    )
    assert day.weight_kg == 82.5
    assert day.body_fat_percent is None
    assert day.bmi is None


def test_parse_body_day_requires_numeric_weight() -> None:
    try:
        parse_body_day({"dayID": "body-1", "dayStartDate": "2026-06-05T00:00:00Z", "weightKg": "82"})
    except VaultbeatCryptoError:
        pass
    else:
        raise AssertionError("expected VaultbeatCryptoError for non-numeric weightKg")


def test_summarize_weight_trend_empty_reports_none() -> None:
    summary = summarize_weight_trend([])
    assert summary["day_count"] == 0
    assert summary["latest_kg"] is None
    assert summary["weekly_rate_kg_per_week"] is None
    assert summary["delta_to_goal_kg"] is None


def test_summarize_weight_trend_stats_and_goal_delta() -> None:
    days = [
        BodyDay("d1", "2026-06-01T00:00:00Z", 84.0, None, None),
        BodyDay("d2", "2026-06-08T00:00:00Z", 83.0, None, None),
        BodyDay("d3", "2026-06-15T00:00:00Z", 82.0, None, None),
    ]
    summary = summarize_weight_trend(days, goal_kg=72.5)
    assert summary["day_count"] == 3
    assert summary["days"][0]["day_id"] == "d3"  # newest first
    assert summary["latest_kg"] == 82.0
    assert summary["average_kg"] == 83.0
    assert summary["min_kg"] == 82.0
    assert summary["max_kg"] == 84.0
    # latest - goal; positive = still above the goal.
    assert summary["delta_to_goal_kg"] == 82.0 - 72.5
    # Exactly -1 kg per 7 days -> OLS slope is -1.0 kg/week (iOS WeightRangeAggregate parity).
    assert abs(summary["weekly_rate_kg_per_week"] - (-1.0)) < 1e-9


def test_summarize_weight_trend_single_day_has_no_rate() -> None:
    summary = summarize_weight_trend([BodyDay("d1", "2026-06-01T00:00:00Z", 82.5, None, None)])
    assert summary["latest_kg"] == 82.5
    assert summary["weekly_rate_kg_per_week"] is None
    assert summary["delta_to_goal_kg"] is None  # no goal supplied -> not assumed


def test_summarize_weight_trend_dedups_by_day_id() -> None:
    days = [
        BodyDay("d1", "2026-06-01T00:00:00Z", 84.0, None, None),  # stale
        BodyDay("d1", "2026-06-01T12:00:00Z", 83.0, None, None),  # newer wins
    ]
    summary = summarize_weight_trend(days)
    assert summary["day_count"] == 1
    assert summary["latest_kg"] == 83.0


def test_parse_menstrual_day_validates_flow() -> None:
    day = parse_menstrual_day(
        {
            "dayID": "menstrual-1",
            "dayStartDate": "2026-06-05T00:00:00Z",
            "samples": [
                {
                    "startDate": "2026-06-05T00:00:00Z",
                    "endDate": "2026-06-05T23:59:59Z",
                    "flow": "medium",
                }
            ],
        }
    )
    assert day.samples[0].flow == "medium"


def test_summarize_menstrual_cycle_predicts_from_average_gap() -> None:
    days = [
        MenstrualDay("d1", "2026-04-01T00:00:00Z", [MenstrualSample("", "", "medium")]),
        MenstrualDay("d2", "2026-04-29T00:00:00Z", [MenstrualSample("", "", "heavy")]),
        MenstrualDay("d3", "2026-05-29T00:00:00Z", [MenstrualSample("", "", "medium")]),
    ]
    summary = summarize_menstrual_cycle(days)
    assert summary["sensitive"] is True
    # gaps: 28 and 30 days -> median 29
    assert summary["average_cycle_length_days"] == 29
    # 2 gaps < 3 -> no spread estimate
    assert summary["cycle_length_variability_days"] is None
    assert summary["last_cycle_start_date"].startswith("2026-05-29")
    # 2026-05-29 + 29 days = 2026-06-27
    assert summary["predicted_next_period_start_date"].startswith("2026-06-27")


def _menstrual_days_from_gaps(start: str, gaps: list[int]) -> list[MenstrualDay]:
    from datetime import timedelta

    cursor = datetime.fromisoformat(start)
    dates = [cursor]
    for gap in gaps:
        cursor = cursor + timedelta(days=gap)
        dates.append(cursor)
    return [
        MenstrualDay(f"d{i}", d.strftime("%Y-%m-%dT00:00:00Z"), [MenstrualSample("", "", "medium")])
        for i, d in enumerate(dates)
    ]


def test_summarize_menstrual_cycle_median_resists_missed_month() -> None:
    # One missed logging month (56-day gap) in a 28-day rhythm: the median
    # stays 28 (a mean would say 34) and the MAD shrugs off the outlier.
    # Mirrors Swift's testMedianResistsOneMissedLoggingMonth.
    days = _menstrual_days_from_gaps("2026-01-01", [28, 28, 56, 28, 28])
    summary = summarize_menstrual_cycle(days)
    assert summary["average_cycle_length_days"] == 28
    assert summary["cycle_length_variability_days"] == 0
    assert summary["predicted_next_period_start_date"].startswith("2026-07-16")


def test_summarize_menstrual_cycle_window_ignores_ancient_rhythm() -> None:
    # 10 old 20-day gaps then 7 recent 30-day gaps: the 12-gap window sees
    # [20×5, 30×7] → median 30 (whole-history median would be 20).
    # Mirrors Swift's testStatisticsWindowIgnoresAncientRhythm.
    days = _menstrual_days_from_gaps("2025-01-01", [20] * 10 + [30] * 7)
    summary = summarize_menstrual_cycle(days)
    assert summary["average_cycle_length_days"] == 30


def test_summarize_menstrual_cycle_variability_is_mad() -> None:
    # gaps 26,28,31 → median 28, deviations [2,0,3] → MAD 2.
    # Mirrors Swift's testVariabilityReportsMedianAbsoluteDeviation.
    days = _menstrual_days_from_gaps("2026-05-01", [26, 28, 31])
    summary = summarize_menstrual_cycle(days)
    assert summary["average_cycle_length_days"] == 28
    assert summary["cycle_length_variability_days"] == 2


def _wrist_readings(spec: list[tuple[str, float]]) -> list[tuple[datetime, float]]:
    return [(datetime.fromisoformat(day), delta) for day, delta in spec]


def test_detect_ovulation_textbook_shift() -> None:
    # Flat follicular phase then a sustained +0.25°C plateau from Jun 15 →
    # ovulation Jun 14. Mirrors Swift's testDetectsTextbookBiphasicShift.
    readings = _wrist_readings(
        [(f"2026-06-{d:02d}", (d % 3) * 0.02 - 0.02) for d in range(2, 15)]
        + [("2026-06-15", 0.25), ("2026-06-16", 0.28), ("2026-06-17", 0.30)]
    )
    ovulation = detect_ovulation_from_wrist_temp(readings, datetime(2026, 6, 1))
    assert ovulation == date(2026, 6, 14)


def test_detect_ovulation_single_hot_night_does_not_trigger() -> None:
    readings = _wrist_readings(
        [(f"2026-06-{d:02d}", 0.40 if d == 10 else 0.0) for d in range(2, 17)]
    )
    assert detect_ovulation_from_wrist_temp(readings, datetime(2026, 6, 1)) is None


def test_detect_ovulation_missing_night_tolerance_is_bounded() -> None:
    # One Watch-less night inside the plateau is fine; a 2-night hole breaks it.
    # Mirrors Swift's testMissingNightToleranceIsBounded.
    base = [(f"2026-06-{d:02d}", 0.0) for d in range(2, 15)]
    one_hole = _wrist_readings(base + [("2026-06-15", 0.25), ("2026-06-17", 0.26), ("2026-06-18", 0.27)])
    assert detect_ovulation_from_wrist_temp(one_hole, datetime(2026, 6, 1)) == date(2026, 6, 14)

    two_holes = _wrist_readings(base + [("2026-06-15", 0.25), ("2026-06-18", 0.26), ("2026-06-19", 0.27)])
    assert detect_ovulation_from_wrist_temp(two_holes, datetime(2026, 6, 1)) is None


def test_summarize_menstrual_cycle_fuses_detected_ovulation() -> None:
    # A detected biphasic shift re-anchors the prediction to ovulation + 14 —
    # mirrors Swift's VaultbeatMenstrualCycleSummary.calibrated so the app and
    # the AI keep agreeing on the date.
    days = _menstrual_days_from_gaps("2026-04-01", [28, 28])  # last start 2026-05-27
    readings = _wrist_readings(
        [(f"2026-05-{d:02d}", 0.0) for d in range(28, 32)]
        + [(f"2026-06-{d:02d}", 0.0) for d in range(1, 10)]
        + [("2026-06-10", 0.25), ("2026-06-11", 0.28), ("2026-06-12", 0.30)]
    )
    summary = summarize_menstrual_cycle(days, wrist_readings=readings)
    assert summary["detected_ovulation_date"] == "2026-06-09"
    assert summary["prediction_calibrated_by_ovulation"] is True
    # 2026-06-09 + 14 = 2026-06-23 (statistics alone would say 05-27 + 28 = 06-24).
    assert summary["predicted_next_period_start_date"].startswith("2026-06-23")


def test_summarize_menstrual_cycle_without_readings_stays_statistical() -> None:
    days = _menstrual_days_from_gaps("2026-04-01", [28, 28])
    summary = summarize_menstrual_cycle(days)
    assert summary["prediction_calibrated_by_ovulation"] is False
    assert summary["detected_ovulation_date"] is None


def test_detect_ovulation_excludes_previous_cycle() -> None:
    # The previous cycle's warm luteal tail must not participate.
    readings = _wrist_readings(
        [(f"2026-05-{d:02d}", 0.30) for d in range(25, 32)]
        + [(f"2026-06-{d:02d}", 0.0) for d in range(2, 9)]
    )
    assert detect_ovulation_from_wrist_temp(readings, datetime(2026, 6, 1)) is None


def test_summarize_menstrual_cycle_groups_contiguous_bleeding_into_one_cycle() -> None:
    # Three consecutive bleeding days are ONE cycle start, not three.
    days = [
        MenstrualDay("d1", "2026-05-01T00:00:00Z", [MenstrualSample("", "", "heavy")]),
        MenstrualDay("d2", "2026-05-02T00:00:00Z", [MenstrualSample("", "", "medium")]),
        MenstrualDay("d3", "2026-05-03T00:00:00Z", [MenstrualSample("", "", "light")]),
    ]
    summary = summarize_menstrual_cycle(days)
    # Only one cycle start -> not enough to predict.
    assert summary["predicted_next_period_start_date"] is None
    assert "Insufficient history" in summary["prediction_note"]
    assert summary["last_cycle_start_date"].startswith("2026-05-01")


def test_summarize_menstrual_cycle_insufficient_history() -> None:
    days = [MenstrualDay("d1", "2026-05-01T00:00:00Z", [MenstrualSample("", "", "medium")])]
    summary = summarize_menstrual_cycle(days)
    assert summary["average_cycle_length_days"] is None
    assert summary["predicted_next_period_start_date"] is None
    assert "Insufficient history" in summary["prediction_note"]


def test_summarize_menstrual_cycle_ignores_non_bleeding_flows() -> None:
    # "none" (explicit no-flow) days never count as a cycle start.
    days = [MenstrualDay("d1", "2026-05-10T00:00:00Z", [MenstrualSample("", "", "none")])]
    summary = summarize_menstrual_cycle(days)
    assert summary["last_cycle_start_date"] is None
    assert summary["predicted_next_period_start_date"] is None


def test_summarize_menstrual_cycle_counts_unspecified_as_bleeding() -> None:
    # HealthKit "unspecified" = flow occurred without an amount (Apple Health's
    # quick period log) — it must count as bleeding, mirroring Swift `isBleeding`.
    days = [
        MenstrualDay("d1", "2026-04-12T00:00:00Z", [MenstrualSample("", "", "unspecified")]),
        MenstrualDay("d2", "2026-05-10T00:00:00Z", [MenstrualSample("", "", "unspecified")]),
    ]
    summary = summarize_menstrual_cycle(days)
    assert summary["last_cycle_start_date"].startswith("2026-05-10")
    assert summary["average_cycle_length_days"] == 28
    assert summary["predicted_next_period_start_date"].startswith("2026-06-07")


# --- end-to-end through the service (decrypt -> route by metric_type -> aggregate) -------


def _bound_service(tmp_path: Path) -> tuple[VaultbeatLocalService, FakeCloudClient, str]:
    config_path = tmp_path / "config.json"
    cloud = FakeCloudClient()
    service = VaultbeatLocalService(ConfigStore(config_path), cloud)
    service.start_binding(server_name="Mac Studio", api_base_url="https://api.test")
    asyncio.run(service.poll_once())
    public_key = ConfigStore(config_path).require_bound().public_key_base64
    return service, cloud, public_key


def test_water_intake_summary_routes_by_metric_type(tmp_path: Path) -> None:
    service, cloud, public_key = _bound_service(tmp_path)
    cloud.envelopes = [
        _make_envelope(
            public_key,
            b'{"stage":"asleep"}',
            metric_type="sleep",
            envelope_id="env-sleep",
            blob_id="blob-sleep",
        ),
        _make_envelope(
            public_key,
            _water_payload("water-1", "2026-06-05T00:00:00Z", 6.0, refill_count=2),
            metric_type="water",
            envelope_id="env-water",
            blob_id="blob-water",
        ),
    ]

    summary = asyncio.run(service.water_intake_summary(limit=10))

    assert summary["errors"] == []
    assert summary["day_count"] == 1  # the sleep blob was filtered out
    assert summary["average_daily_intake_liters"] == 12.0  # 2 refills * 6.0 L


def test_weight_trend_summary_routes_by_metric_type(tmp_path: Path) -> None:
    service, cloud, public_key = _bound_service(tmp_path)
    cloud.envelopes = [
        _make_envelope(
            public_key,
            b'{"stage":"asleep"}',
            metric_type="sleep",
            envelope_id="env-sleep",
            blob_id="blob-sleep",
        ),
        _make_envelope(
            public_key,
            _body_payload("body-1", "2026-06-05T00:00:00Z", 82.5),
            metric_type="body",
            envelope_id="env-body",
            blob_id="blob-body",
        ),
    ]

    summary = asyncio.run(service.weight_trend_summary(limit=10, goal_kg=72.5))

    assert summary["errors"] == []
    assert summary["day_count"] == 1  # the sleep blob was filtered out
    assert summary["latest_kg"] == 82.5
    assert summary["delta_to_goal_kg"] == 10.0


def test_menstrual_cycle_summary_routes_by_metric_type(tmp_path: Path) -> None:
    service, cloud, public_key = _bound_service(tmp_path)
    cloud.envelopes = [
        _make_envelope(
            public_key,
            _menstrual_payload("m-1", "2026-04-01T00:00:00Z", "medium"),
            metric_type="menstrual",
            envelope_id="env-m1",
            blob_id="blob-m1",
        ),
        _make_envelope(
            public_key,
            _menstrual_payload("m-2", "2026-04-29T00:00:00Z", "heavy"),
            metric_type="menstrual",
            envelope_id="env-m2",
            blob_id="blob-m2",
        ),
    ]

    summary = asyncio.run(service.menstrual_cycle_summary(limit=60))

    assert summary["errors"] == []
    assert summary["sensitive"] is True
    assert summary["day_count"] == 2
    assert summary["average_cycle_length_days"] == 28.0  # one 28-day gap
    assert summary["predicted_next_period_start_date"].startswith("2026-05-27")


def test_menstrual_cycle_summary_absent_when_not_opted_in(tmp_path: Path) -> None:
    service, cloud, public_key = _bound_service(tmp_path)
    cloud.envelopes = [
        _make_envelope(public_key, b'{"stage":"asleep"}', metric_type="sleep"),
    ]

    summary = asyncio.run(service.menstrual_cycle_summary(limit=60))

    assert summary["errors"] == []
    assert summary["day_count"] == 0
    assert summary["predicted_next_period_start_date"] is None


# ── symptom / note kinds (added 2026-07-04) ──────────────────────────────────


def _symptom_payload(day_id: str, day_start: str, samples: list[dict[str, str]]) -> bytes:
    return json.dumps(
        {"dayID": day_id, "dayStartDate": day_start, "samples": samples}
    ).encode()


def _note_payload(
    note_id: str,
    kind: str,
    target_date: str,
    text: str,
    *,
    updated_at: str | None = "2026-07-04T08:00:00Z",
) -> bytes:
    payload: dict[str, Any] = {
        "noteID": note_id,
        "targetKind": kind,
        "targetDate": target_date,
        "text": text,
        "createdAt": "2026-07-04T04:00:00Z",
    }
    if updated_at is not None:
        payload["updatedAt"] = updated_at
    return json.dumps(payload).encode()


def test_symptom_summary_groups_by_owner_and_skips_not_present(tmp_path: Path) -> None:
    service, cloud, public_key = _bound_service(tmp_path)
    cloud.envelopes = [
        _make_envelope(
            public_key,
            _symptom_payload(
                "symptom-aaaa1111-100",
                "2026-07-04T00:00:00Z",
                [
                    {"symptomType": "abdominalCramps", "severity": "moderate",
                     "startDate": "2026-07-04T03:00:00Z", "endDate": "2026-07-04T05:00:00Z"},
                    {"symptomType": "headache", "severity": "notPresent",
                     "startDate": "2026-07-04T03:00:00Z", "endDate": "2026-07-04T03:00:00Z"},
                ],
            ),
            metric_type="symptom",
            envelope_id="env-s1",
            blob_id="symptom-aaaa1111-100",
            owner_user_id="f8350dfc-0000-0000-0000-000000000000",
        ),
        _make_envelope(
            public_key,
            _symptom_payload(
                "symptom-bbbb2222-100",
                "2026-07-04T00:00:00Z",
                [{"symptomType": "coughing", "severity": "mild",
                  "startDate": "2026-07-04T01:00:00Z", "endDate": "2026-07-04T01:00:00Z"}],
            ),
            metric_type="symptom",
            envelope_id="env-s2",
            blob_id="symptom-bbbb2222-100",
            owner_user_id="dce9b9cf-0000-0000-0000-000000000000",
        ),
    ]

    summary = asyncio.run(service.symptom_summary(limit=50))

    assert summary["errors"] == []
    assert summary["sensitive"] is True
    assert summary["owner_count"] == 2
    by_owner = {o["owner_user_id"]: o for o in summary["owners"]}
    her = by_owner["f8350dfc-0000-0000-0000-000000000000"]
    # notPresent entries must not inflate the per-type tally.
    assert her["symptom_counts"] == {"abdominalCramps": 1}
    him = by_owner["dce9b9cf-0000-0000-0000-000000000000"]
    assert him["symptom_counts"] == {"coughing": 1}


def test_symptom_summary_collects_unknown_severity_into_errors(tmp_path: Path) -> None:
    service, cloud, public_key = _bound_service(tmp_path)
    cloud.envelopes = [
        _make_envelope(
            public_key,
            _symptom_payload(
                "symptom-cccc3333-100",
                "2026-07-04T00:00:00Z",
                [{"symptomType": "headache", "severity": "catastrophic",
                  "startDate": "2026-07-04T03:00:00Z", "endDate": "2026-07-04T03:00:00Z"}],
            ),
            metric_type="symptom",
            envelope_id="env-bad",
            blob_id="symptom-cccc3333-100",
        ),
    ]

    summary = asyncio.run(service.symptom_summary(limit=50))

    assert summary["owner_count"] == 0
    assert len(summary["errors"]) == 1
    assert "env-bad" in summary["errors"][0]


def test_notes_summary_dedups_edits_and_filters_kind(tmp_path: Path) -> None:
    service, cloud, public_key = _bound_service(tmp_path)
    note_id = "note-aabbccdd00112233aabbccdd00112233"
    cloud.envelopes = [
        _make_envelope(
            public_key,
            _note_payload(note_id, "menstrual", "2026-07-04T00:00:00Z", "原文",
                          updated_at="2026-07-04T05:00:00Z"),
            metric_type="note",
            envelope_id="env-n1",
            blob_id=note_id,
            owner_user_id="f8350dfc-0000-0000-0000-000000000000",
        ),
        _make_envelope(
            public_key,
            _note_payload(note_id, "menstrual", "2026-07-04T00:00:00Z", "编辑后",
                          updated_at="2026-07-04T09:00:00Z"),
            metric_type="note",
            envelope_id="env-n1-edit",
            blob_id=note_id,
            owner_user_id="f8350dfc-0000-0000-0000-000000000000",
        ),
        _make_envelope(
            public_key,
            _note_payload("note-ffee00112233445566778899aabbccdd", "sleep",
                          "2026-07-03T00:00:00Z", "昨晚舍友很吵"),
            metric_type="note",
            envelope_id="env-n2",
            blob_id="note-ffee00112233445566778899aabbccdd",
            owner_user_id="dce9b9cf-0000-0000-0000-000000000000",
        ),
    ]

    all_notes = asyncio.run(service.notes_summary(limit=50))
    assert all_notes["errors"] == []
    assert all_notes["total_note_count"] == 2  # edit deduped by note_id
    kinds = {k["target_kind"]: k for k in all_notes["kinds"]}
    assert kinds["menstrual"]["notes"][0]["text"] == "编辑后"
    assert kinds["menstrual"]["notes"][0]["owner_user_id"].startswith("f8350dfc")
    assert kinds["sleep"]["notes"][0]["owner_user_id"].startswith("dce9b9cf")

    only_sleep = asyncio.run(service.notes_summary(limit=50, target_kind="sleep"))
    assert only_sleep["total_note_count"] == 1
    assert only_sleep["kinds"][0]["target_kind"] == "sleep"


def test_tampered_envelope_lands_in_errors_without_killing_sync(tmp_path: Path) -> None:
    """InvalidTag regression (2026-07-04 audit P0): one mis-keyed/corrupted envelope
    must land in errors[] — not abort the whole sync across all metric types."""

    service, cloud, public_key = _bound_service(tmp_path)
    good = _make_envelope(
        public_key,
        _note_payload("note-11223344556677889900112233445566", "sleep",
                      "2026-07-03T00:00:00Z", "好的备注"),
        metric_type="note",
        envelope_id="env-good",
        blob_id="note-11223344556677889900112233445566",
    )
    tampered = _make_envelope(
        public_key,
        _note_payload("note-99887766554433221100998877665544", "sleep",
                      "2026-07-02T00:00:00Z", "会被篡改"),
        metric_type="note",
        envelope_id="env-tampered",
        blob_id="note-99887766554433221100998877665544",
    )
    raw = bytearray(base64.b64decode(tampered["encrypted_sleep_blobs"]["ciphertext"]))
    raw[-1] ^= 0x01  # flip a tag byte → AES-GCM auth failure
    tampered["encrypted_sleep_blobs"]["ciphertext"] = base64.b64encode(bytes(raw)).decode()
    cloud.envelopes = [tampered, good]

    summary = asyncio.run(service.notes_summary(limit=50))

    assert summary["total_note_count"] == 1
    assert summary["kinds"][0]["notes"][0]["text"] == "好的备注"
    assert len(summary["errors"]) == 1
    assert "env-tampered" in summary["errors"][0]
