from __future__ import annotations

import base64
import asyncio
import json
import stat
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import x25519
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from vaultbeat_mcp_local.client import PollBindingResult, VaultbeatUnsupportedMetricError
from vaultbeat_mcp_local.crypto import ENVELOPE_INFO
from vaultbeat_mcp_local.crypto import VaultbeatCryptoError
from vaultbeat_mcp_local.crypto import generate_x25519_keypair
from vaultbeat_mcp_local.service import (
    BodyDay,
    _local_calendar_day,
    _parse_iso8601,
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
        # Identity the bind handshake resolves to. Mutable so a test can model
        # re-binding onto a DIFFERENT server row (the only case that may throw
        # away cached plaintext) versus back onto the same one.
        self.bound_server_id = "server-1"
        self.bound_server_token = "token-1"
        # Owner identity the bind handshake would carry.
        self.owner_user_id: str | None = None
        self.owner_public_key_base64: str | None = None
        self.owner_device_id: str | None = None
        # Every write_strength_blob call, in order — lets tests assert what
        # was actually sent (ownership, recipients, ciphertext presence)
        # without a real network round trip.
        self.write_calls: list[dict[str, Any]] = []
        # Metric types this fake "edge deployment" does not know about, i.e. an
        # edge older than this client. Requesting one raises the same error the
        # real edge's 400 invalid_metric_type produces.
        self.unsupported_metric_types: set[str] = set()
        # Every report_decrypt_failures call, in order — {items, server_token}.
        self.report_calls: list[dict[str, Any]] = []
        # When set, report_decrypt_failures raises this instead of recording —
        # used to prove a reporting failure never breaks the read itself.
        self.report_raises: Exception | None = None

    async def poll_binding(self, poll_id: str) -> PollBindingResult:
        self.poll_id = poll_id
        return PollBindingResult(
            status="bound",
            server_id=self.bound_server_id,
            server_token=self.bound_server_token,
            owner_user_id=self.owner_user_id,
            owner_public_key_base64=self.owner_public_key_base64,
            owner_device_id=self.owner_device_id,
        )

    async def sync(
        self, server_token: str, *, metric_type: str | None = None
    ) -> list[dict[str, Any]]:
        self.synced_with_token = server_token
        self.sync_calls.append(metric_type)
        if metric_type is not None and metric_type in self.unsupported_metric_types:
            raise VaultbeatUnsupportedMetricError(metric_type, ["sleep", "water"])
        if metric_type is None or self.ignores_metric_type:
            return self.envelopes

        def _matches(row: dict[str, Any]) -> bool:
            blob = row.get("encrypted_sleep_blobs") or {}
            effective = blob.get("metric_type") or "sleep"
            return effective == metric_type

        return [row for row in self.envelopes if _matches(row)]

    async def write_strength_blob(
        self, server_token: str, *, blob: dict[str, Any], envelopes: list[dict[str, Any]]
    ) -> dict[str, Any]:
        return await self._record_write(server_token, blob=blob, envelopes=envelopes)

    async def write_food_blob(
        self, server_token: str, *, blob: dict[str, Any], envelopes: list[dict[str, Any]]
    ) -> dict[str, Any]:
        # Same fake shape as write_strength_blob — the edge functions differ in
        # the metric-scope validation, but from the client's perspective both
        # sides just POST a sealed blob + envelopes and get back an upsert count.
        return await self._record_write(server_token, blob=blob, envelopes=envelopes)

    async def write_body_blob(
        self, server_token: str, *, blob: dict[str, Any], envelopes: list[dict[str, Any]]
    ) -> dict[str, Any]:
        return await self._record_write(server_token, blob=blob, envelopes=envelopes)

    async def write_note_blob(
        self, server_token: str, *, blob: dict[str, Any], envelopes: list[dict[str, Any]]
    ) -> dict[str, Any]:
        return await self._record_write(server_token, blob=blob, envelopes=envelopes)

    async def report_decrypt_failures(self, server_token: str, *, items: list[dict[str, str]]) -> None:
        if self.report_raises is not None:
            raise self.report_raises
        self.report_calls.append({"server_token": server_token, "items": items})

    async def _record_write(
        self, server_token: str, *, blob: dict[str, Any], envelopes: list[dict[str, Any]]
    ) -> dict[str, Any]:
        self.write_calls.append(
            {"server_token": server_token, "blob": blob, "envelopes": envelopes}
        )
        mcp_envelope = next(
            (e for e in envelopes if e.get("recipient_kind") == "mcp_server"), None
        )
        if mcp_envelope is not None:
            # Mirror what mcp-sync's read side actually returns: only the
            # mcp_server-kind envelope round-trips back — this server can
            # never decrypt the owner_user envelope (sealed to the PHONE's
            # identity key, not this machine's).
            self.envelopes = [
                row for row in self.envelopes if row.get("blob_id") != blob["id"]
            ] + [
                {
                    "id": f"env-write-{blob['id']}",
                    "blob_id": blob["id"],
                    "encrypted_data_key": mcp_envelope["encrypted_data_key"],
                    "encrypted_sleep_blobs": {
                        "id": blob["id"],
                        "ciphertext": blob["ciphertext"],
                        "metric_type": blob["metric_type"],
                        "created_at": "2026-07-19T12:00:00Z",
                        "owner_user_id": blob["owner_user_id"],
                    },
                }
            ]
        return {"upserted_blobs": 1, "upserted_envelopes": len(envelopes), "blob_id": blob["id"]}


def _make_envelope(
    public_key_base64: str,
    plaintext: bytes,
    *,
    metric_type: str = "sleep",
    envelope_id: str = "env-1",
    blob_id: str = "blob-1",
    owner_user_id: str | None = None,
    dek: bytes = b"\x07" * 32,
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


def test_start_binding_does_not_destroy_a_working_binding(tmp_path: Path) -> None:
    """Invariant 54 (a): a binding session nobody completes must leave the
    existing credentials intact.

    The old code nulled server_id/server_token/bound_at right here, before any
    scan — so running `bind` to *diagnose* a problem unbound the user, silently,
    while the cloud kept the row. That is the single reachable path to fino's
    all-NULL config of 2026-08-10, and via the `vaultbeat_start_binding` tool it
    means any agent that "just re-checks its binding" unbinds its owner.
    """

    config_path = tmp_path / "config.json"
    service = VaultbeatLocalService(ConfigStore(config_path), FakeCloudClient())
    service.start_binding(server_name="Mac Studio", api_base_url="https://api.test")
    asyncio.run(service.poll_once())

    service.start_binding(server_name="Mac Studio", api_base_url="https://api.test")

    pending = ConfigStore(config_path).load()
    assert pending is not None
    assert pending.poll_id is not None, "a session should be open"
    # Still fully usable while waiting for a scan that may never come.
    assert pending.server_id == "server-1"
    assert pending.server_token == "token-1"
    assert pending.bound_at is not None
    assert pending.is_bound
    ConfigStore(config_path).require_bound()


def test_completing_a_binding_swaps_credentials_at_that_moment(tmp_path: Path) -> None:
    """The other direction: deferring the wipe must not defer the SWAP. When a
    scan does land on a new identity, the new credentials take over."""

    config_path = tmp_path / "config.json"
    cloud = FakeCloudClient()
    service = VaultbeatLocalService(ConfigStore(config_path), cloud)
    service.start_binding(server_name="Mac Studio", api_base_url="https://api.test")
    asyncio.run(service.poll_once())
    ConfigStore(config_path).update(last_sync_at="2026-08-10T00:00:00Z")

    cloud.bound_server_id = "server-2"
    cloud.bound_server_token = "token-2"
    service.start_binding(server_name="Mac Studio", api_base_url="https://api.test")
    asyncio.run(service.poll_once())

    saved = ConfigStore(config_path).load()
    assert saved is not None
    assert saved.server_id == "server-2"
    assert saved.server_token == "token-2"
    assert saved.poll_id is None
    # A different identity's sync history says nothing about this one.
    assert saved.last_sync_at is None


def test_rebinding_to_the_same_identity_keeps_its_sync_history(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    service = VaultbeatLocalService(ConfigStore(config_path), FakeCloudClient())
    service.start_binding(server_name="Mac Studio", api_base_url="https://api.test")
    asyncio.run(service.poll_once())
    ConfigStore(config_path).update(last_sync_at="2026-08-10T00:00:00Z")

    service.start_binding(server_name="Mac Studio", api_base_url="https://api.test")
    asyncio.run(service.poll_once())

    saved = ConfigStore(config_path).load()
    assert saved is not None
    assert saved.last_sync_at == "2026-08-10T00:00:00Z"


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


# ---------------------------------------------------------------------------
# The MCP-side half of the blob-integrity GC (AGENTS.md Invariant 34):
# a proven DEK mismatch is reported to the cloud; a merely-unreadable ("not
# for me") envelope never is.
# ---------------------------------------------------------------------------


def test_sync_reports_dek_mismatch_but_not_a_healthy_record(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    cloud = FakeCloudClient()
    service = VaultbeatLocalService(ConfigStore(config_path), cloud)
    service.start_binding(server_name="Mac Studio", api_base_url="https://api.test")
    asyncio.run(service.poll_once())
    config = ConfigStore(config_path).require_bound()

    healthy = _make_envelope(
        config.public_key_base64, b'{"bpm":64}', metric_type="resting_hr", envelope_id="env-ok", blob_id="resting_hr-1"
    )
    # Same construction as the production bug: an envelope that unwraps to a
    # VALID dek (\x09*32), spliced onto ciphertext sealed under a DIFFERENT dek
    # (\x07*32, the default). Proven damage, not a permissions problem.
    wrong_dek_row = _make_envelope(
        config.public_key_base64, b'{"bpm":99}', metric_type="resting_hr", envelope_id="env-bad", blob_id="resting_hr-2"
    )
    mismatched_envelope = _make_envelope(
        config.public_key_base64, b"unused", metric_type="resting_hr", envelope_id="env-mismatch", dek=b"\x09" * 32
    )
    wrong_dek_row["encrypted_data_key"] = mismatched_envelope["encrypted_data_key"]
    cloud.envelopes = [healthy, wrong_dek_row]

    records, errors = asyncio.run(service.sync_decrypted_records(limit=10, fresh=True))

    assert len(records) == 1, "the healthy record must still decrypt and land in results"
    assert records[0].blob_id == "resting_hr-1"
    assert len(errors) == 1
    assert "decrypt_failed" in errors[0]

    assert len(cloud.report_calls) == 1, "exactly one report call, for the mismatched blob only"
    reported_items = cloud.report_calls[0]["items"]
    assert reported_items == [{"blob_id": "resting_hr-2", "metric_type": "resting_hr"}]
    assert cloud.report_calls[0]["server_token"] == "token-1"


def test_sync_never_reports_an_envelope_sealed_for_a_different_recipient(tmp_path: Path) -> None:
    """Sealing for a DIFFERENT public key reproduces a rotated-key / not-for-me
    envelope. This must NEVER be reported — Invariant 34's whole point is that
    only PROVEN damage triggers a repair signal."""
    config_path = tmp_path / "config.json"
    cloud = FakeCloudClient()
    service = VaultbeatLocalService(ConfigStore(config_path), cloud)
    service.start_binding(server_name="Mac Studio", api_base_url="https://api.test")
    asyncio.run(service.poll_once())

    stranger_private, stranger_public = generate_x25519_keypair()
    cloud.envelopes = [_make_envelope(stranger_public, b'{"bpm":64}', metric_type="resting_hr", blob_id="resting_hr-3")]

    records, errors = asyncio.run(service.sync_decrypted_records(limit=10, fresh=True))

    assert records == []
    assert len(errors) == 1
    assert cloud.report_calls == [], "not-for-me is not damage — never report it"


def test_sync_survives_a_reporting_failure(tmp_path: Path) -> None:
    """The report call is best-effort: if the network/edge fails, the READ must
    still return its (correctly classified) results and errors unchanged."""
    config_path = tmp_path / "config.json"
    cloud = FakeCloudClient()
    service = VaultbeatLocalService(ConfigStore(config_path), cloud)
    service.start_binding(server_name="Mac Studio", api_base_url="https://api.test")
    asyncio.run(service.poll_once())
    config = ConfigStore(config_path).require_bound()

    wrong_dek_row = _make_envelope(config.public_key_base64, b'{"bpm":99}', metric_type="resting_hr", blob_id="resting_hr-4")
    mismatched_envelope = _make_envelope(config.public_key_base64, b"unused", metric_type="resting_hr", dek=b"\x09" * 32)
    wrong_dek_row["encrypted_data_key"] = mismatched_envelope["encrypted_data_key"]
    cloud.envelopes = [wrong_dek_row]
    cloud.report_raises = RuntimeError("edge unreachable")

    records, errors = asyncio.run(service.sync_decrypted_records(limit=10, fresh=True))

    assert records == []
    assert len(errors) == 1
    assert "decrypt_failed" in errors[0]


def test_sync_reports_nothing_when_everything_is_healthy(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    cloud = FakeCloudClient()
    service = VaultbeatLocalService(ConfigStore(config_path), cloud)
    service.start_binding(server_name="Mac Studio", api_base_url="https://api.test")
    asyncio.run(service.poll_once())
    config = ConfigStore(config_path).require_bound()
    cloud.envelopes = [_make_envelope(config.public_key_base64, b'{"stage":"asleep"}')]

    asyncio.run(service.sync_decrypted_records(limit=10, fresh=True))

    assert cloud.report_calls == [], "the steady state must cost zero report calls"


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


def test_parse_body_day_decodes_weight_when_composition_is_absent() -> None:
    """A pre-2026-08-05 blob has no composition at all — not even the keys."""

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
    # add-only: the field the payload never carried decodes as None, not a crash
    assert day.lean_body_mass_kg is None


def test_parse_body_day_decodes_full_scale_composition() -> None:
    """What a smart scale actually produces once iOS reads all four types.

    body_fat_percent is 0–100 here because iOS converts HealthKit's native 0–1
    fraction once, at the read edge — this asserts the decoder does NOT convert
    a second time.
    """

    day = parse_body_day(
        {
            "dayID": "body-1",
            "dayStartDate": "2026-08-05T00:00:00Z",
            "weightKg": 83.1,
            "bodyFatPercent": 22.4,
            "bmi": 24.9,
            "leanBodyMassKg": 64.5,
        }
    )
    assert day.weight_kg == 83.1
    assert day.body_fat_percent == 22.4
    assert day.bmi == 24.9
    assert day.lean_body_mass_kg == 64.5
    assert day.to_dict()["lean_body_mass_kg"] == 64.5


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


def test_doctor_reports_missing_config_with_bind_hint(tmp_path: Path) -> None:
    service = VaultbeatLocalService(ConfigStore(tmp_path / "config.json"), FakeCloudClient())

    report = asyncio.run(service.doctor())

    assert report["ok"] is False
    config_check = report["checks"][0]
    assert config_check["name"] == "config"
    assert config_check["ok"] is False
    assert "bind" in config_check["hint"]


def test_doctor_passes_on_bound_server_with_reachable_cloud(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    cloud = FakeCloudClient()
    service = VaultbeatLocalService(ConfigStore(config_path), cloud)
    service.start_binding(server_name="Mac Studio", api_base_url="https://api.test")
    asyncio.run(service.poll_once())
    service._probe_cloud = lambda url: (True, "cloud answered HTTP 401")  # type: ignore[method-assign]

    report = asyncio.run(service.doctor())

    assert report["ok"] is True
    names = {check["name"]: check for check in report["checks"]}
    assert names["bound"]["ok"] is True
    assert names["cloud_reachable"]["ok"] is True
    assert names["data_roundtrip"]["ok"] is True


def test_doctor_flags_decrypt_failure_with_rebind_hint(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    cloud = FakeCloudClient()
    service = VaultbeatLocalService(ConfigStore(config_path), cloud)
    service.start_binding(server_name="Mac Studio", api_base_url="https://api.test")
    asyncio.run(service.poll_once())
    service._probe_cloud = lambda url: (True, "cloud answered HTTP 401")  # type: ignore[method-assign]
    # An envelope wrapped for a DIFFERENT recipient key — decrypt must fail.
    foreign_key = base64.b64encode(
        x25519.X25519PrivateKey.generate()
        .public_key()
        .public_bytes(
            encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw
        )
    ).decode()
    cloud.envelopes = [_make_envelope(foreign_key, b'{"stage":"asleep"}')]

    report = asyncio.run(service.doctor())

    assert report["ok"] is False
    roundtrip = next(check for check in report["checks"] if check["name"] == "data_roundtrip")
    assert roundtrip["ok"] is False
    hint = roundtrip["hint"]
    # Re-binding must be the FIRST instruction, and deletion must not read like
    # a step. This hint used to open with "delete this server in the iOS app and
    # bind again" — the one action that makes the situation unrecoverable, since
    # deleting the row fires a BEFORE DELETE trigger that takes every envelope
    # addressed to it (20k+ on a two-month-old binding) and only ≤365 days of
    # sleep can ever be re-sealed. Since binding became an upsert, re-binding
    # alone lands on the same row and keeps them.
    assert "bind" in hint
    assert hint.index("bind") < hint.index("Deleting"), (
        "re-binding must come before deletion, not after it"
    )
    # Deletion must be named as costing history — but NOT as irreversible, which
    # is what a first draft of this said. A re-bind afterwards does re-seal most
    # of it (agent-written kinds in full, from the local JSON store); what it
    # cannot reach is history outside each kind's window. Overstating that is its
    # own failure: a user told the loss is total has no reason to re-bind at all.
    assert "costs history" in hint
    assert "older than those windows" in hint


def test_doctor_unreachable_cloud_skips_data_roundtrip(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    cloud = FakeCloudClient()
    service = VaultbeatLocalService(ConfigStore(config_path), cloud)
    service.start_binding(server_name="Mac Studio", api_base_url="https://api.test")
    asyncio.run(service.poll_once())
    service._probe_cloud = lambda url: (False, "ConnectError: refused")  # type: ignore[method-assign]

    report = asyncio.run(service.doctor())

    assert report["ok"] is False
    names = [check["name"] for check in report["checks"]]
    assert "data_roundtrip" not in names


# ── strength kind (added 2026-07-19) ─────────────────────────────────────────


def _strength_payload(
    entry_id: str,
    date: str,
    exercises: list[dict[str, Any]],
    *,
    note: str | None = None,
    updated_at: str | None = "2026-07-19T14:00:00Z",
) -> bytes:
    payload: dict[str, Any] = {
        "entryID": entry_id,
        "date": date,
        "exercises": exercises,
        "createdAt": "2026-07-19T12:00:00Z",
    }
    if note is not None:
        payload["note"] = note
    if updated_at is not None:
        payload["updatedAt"] = updated_at
    return json.dumps(payload).encode()


def test_strength_summary_orders_newest_first_and_computes_volume(tmp_path: Path) -> None:
    service, cloud, public_key = _bound_service(tmp_path)
    cloud.envelopes = [
        _make_envelope(
            public_key,
            _strength_payload(
                "strength-aaaa000011112222aaaa000011112222",
                "2026-07-18T00:00:00Z",
                [{"name": "卧推", "sets": [{"weightKg": 30.0, "reps": 8}]}],
            ),
            metric_type="strength",
            envelope_id="env-st1",
            blob_id="strength-aaaa000011112222aaaa000011112222",
            owner_user_id="dce9b9cf-0000-0000-0000-000000000000",
        ),
        _make_envelope(
            public_key,
            _strength_payload(
                "strength-bbbb000011112222bbbb000011112222",
                "2026-07-19T00:00:00Z",
                [
                    {"name": "三头下压", "sets": [{"weightKg": 18.0, "reps": 12}, {"weightKg": 18.0, "reps": 10}]},
                    {"name": "龙门架夹胸", "sets": [{"weightKg": 31.5, "reps": 10}]},
                ],
                note="腰有点紧",
            ),
            metric_type="strength",
            envelope_id="env-st2",
            blob_id="strength-bbbb000011112222bbbb000011112222",
            owner_user_id="dce9b9cf-0000-0000-0000-000000000000",
        ),
    ]

    summary = asyncio.run(service.strength_summary(limit=50))

    assert summary["errors"] == []
    assert summary["session_count"] == 2
    newest = summary["sessions"][0]
    assert newest["date"].startswith("2026-07-19")
    assert newest["note"] == "腰有点紧"
    # 18×12 + 18×10 + 31.5×10 = 216 + 180 + 315 = 711
    assert newest["total_volume_kg"] == 711.0
    assert summary["sessions"][1]["total_volume_kg"] == 240.0


def test_strength_summary_dedups_edits_newest_wins(tmp_path: Path) -> None:
    service, cloud, public_key = _bound_service(tmp_path)
    entry_id = "strength-cccc000011112222cccc000011112222"
    cloud.envelopes = [
        _make_envelope(
            public_key,
            _strength_payload(
                entry_id,
                "2026-07-19T00:00:00Z",
                [{"name": "卧推", "sets": [{"weightKg": 27.5, "reps": 12}]}],
                updated_at="2026-07-19T13:00:00Z",
            ),
            metric_type="strength",
            envelope_id="env-st-orig",
            blob_id=entry_id,
        ),
        _make_envelope(
            public_key,
            _strength_payload(
                entry_id,
                "2026-07-19T00:00:00Z",
                [
                    {"name": "卧推", "sets": [{"weightKg": 27.5, "reps": 12}]},
                    {"name": "上斜推", "sets": [{"weightKg": 7.5, "reps": 9}]},
                ],
                updated_at="2026-07-19T15:00:00Z",
            ),
            metric_type="strength",
            envelope_id="env-st-edit",
            blob_id=entry_id,
        ),
    ]

    summary = asyncio.run(service.strength_summary(limit=50))

    assert summary["session_count"] == 1
    assert len(summary["sessions"][0]["exercises"]) == 2


def test_strength_summary_collects_malformed_into_errors(tmp_path: Path) -> None:
    service, cloud, public_key = _bound_service(tmp_path)
    cloud.envelopes = [
        _make_envelope(
            public_key,
            json.dumps({"entryID": "strength-bad", "date": "2026-07-19T00:00:00Z", "exercises": []}).encode(),
            metric_type="strength",
            envelope_id="env-st-bad",
            blob_id="strength-bad",
        ),
    ]

    summary = asyncio.run(service.strength_summary(limit=50))

    assert summary["session_count"] == 0
    assert len(summary["errors"]) == 1
    assert "env-st-bad" in summary["errors"][0]


# ── log_strength_entry (agent write path, added 2026-07-20) ─────────────────


def _bound_service_with_owner_identity(tmp_path: Path) -> tuple[VaultbeatLocalService, FakeCloudClient, str]:
    """A bind that carries owner_user_id/owner_public_key_base64/owner_device_id
    through the handshake — the shape `log_strength_entry` requires. Distinct
    from `_bound_service` (used by every pre-existing test) so those stay
    unaffected by this feature.
    """

    config_path = tmp_path / "config.json"
    cloud = FakeCloudClient()
    owner_private = x25519.X25519PrivateKey.generate()
    owner_public_b64 = base64.b64encode(
        owner_private.public_key().public_bytes(
            encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw
        )
    ).decode()
    cloud.owner_user_id = "dce9b9cf-0000-0000-0000-000000000000"
    cloud.owner_public_key_base64 = owner_public_b64
    cloud.owner_device_id = "device-owner-1"

    service = VaultbeatLocalService(ConfigStore(config_path), cloud)
    service.start_binding(server_name="Mac Studio", api_base_url="https://api.test")
    asyncio.run(service.poll_once())
    public_key = ConfigStore(config_path).require_bound().public_key_base64
    return service, cloud, public_key


def test_log_strength_entry_creates_fresh_entry_and_round_trips(tmp_path: Path) -> None:
    service, cloud, _public_key = _bound_service_with_owner_identity(tmp_path)

    result = asyncio.run(
        service.log_strength_entry(
            date="2026-07-19",
            exercises=[
                {"name": "坐姿推胸机", "sets": [{"weightKg": 27.5, "reps": 10}]},
                {"name": "自由杠铃卧推", "sets": [{"weightKg": 30, "reps": 8}]},
            ],
            note="胸日",
        )
    )

    assert result["updated_existing_day"] is False
    assert result["date"] == "2026-07-19"
    assert result["entry_id"].startswith("strength-")
    assert len(cloud.write_calls) == 1

    written_blob = cloud.write_calls[0]["blob"]
    assert written_blob["owner_user_id"] == cloud.owner_user_id
    assert written_blob["source_device_id"] == cloud.owner_device_id
    assert written_blob["metric_type"] == "strength"

    written_envelopes = cloud.write_calls[0]["envelopes"]
    kinds = {e["recipient_kind"] for e in written_envelopes}
    assert kinds == {"owner_user", "mcp_server"}

    # The tool's own return value already reflects the round trip. The wire
    # `date` is a UTC instant for LOCAL midnight — not a "2026-07-19..." prefix
    # once the machine's UTC offset is non-zero — so decode it back through
    # the same local-day helper the service itself uses, rather than assuming UTC.
    session = result["session"]
    assert session is not None
    assert _local_calendar_day(_parse_iso8601(session["date"])) == date(2026, 7, 19)
    assert session["note"] == "胸日"
    assert len(session["exercises"]) == 2

    # And a fresh get_strength_log call (the actual MCP tool an agent uses to
    # read back) sees it too.
    summary = asyncio.run(service.strength_summary(limit=50, fresh=True))
    assert summary["session_count"] == 1
    assert summary["sessions"][0]["entry_id"] == result["entry_id"]


def test_log_strength_entry_editing_same_day_reuses_entry_id(tmp_path: Path) -> None:
    service, cloud, _public_key = _bound_service_with_owner_identity(tmp_path)

    first = asyncio.run(
        service.log_strength_entry(
            date="2026-07-19",
            exercises=[{"name": "卧推", "sets": [{"weightKg": 30, "reps": 8}]}],
        )
    )
    second = asyncio.run(
        service.log_strength_entry(
            date="2026-07-19",
            exercises=[
                {"name": "卧推", "sets": [{"weightKg": 30, "reps": 8}]},
                {"name": "上斜推", "sets": [{"weightKg": 7.5, "reps": 9}]},
            ],
        )
    )

    assert second["updated_existing_day"] is True
    assert second["entry_id"] == first["entry_id"]

    summary = asyncio.run(service.strength_summary(limit=50, fresh=True))
    assert summary["session_count"] == 1  # same day = same blob id, not a duplicate
    assert len(summary["sessions"][0]["exercises"]) == 2


def test_log_strength_entry_different_days_create_separate_entries(tmp_path: Path) -> None:
    service, cloud, _public_key = _bound_service_with_owner_identity(tmp_path)

    day1 = asyncio.run(
        service.log_strength_entry(
            date="2026-07-18", exercises=[{"name": "深蹲", "sets": [{"weightKg": 60, "reps": 5}]}]
        )
    )
    day2 = asyncio.run(
        service.log_strength_entry(
            date="2026-07-19", exercises=[{"name": "卧推", "sets": [{"weightKg": 30, "reps": 8}]}]
        )
    )

    assert day1["entry_id"] != day2["entry_id"]
    summary = asyncio.run(service.strength_summary(limit=50, fresh=True))
    assert summary["session_count"] == 2


def test_log_strength_entry_rejects_bind_missing_owner_identity(tmp_path: Path) -> None:
    # _bound_service (not _with_owner_identity) never sets owner_user_id/etc —
    # the pre-2026-07-20 bind shape this must reject with a clear message.
    service, _cloud, _public_key = _bound_service(tmp_path)

    try:
        asyncio.run(service.log_strength_entry(date="2026-07-19", exercises=[{"name": "卧推", "sets": [{"weightKg": 30, "reps": 8}]}]))
        raised = False
    except RuntimeError as error:
        raised = True
        assert "re-pair" in str(error) or "re-bind" in str(error) or "bind" in str(error)
    assert raised


def test_log_strength_entry_rejects_empty_exercises(tmp_path: Path) -> None:
    service, _cloud, _public_key = _bound_service_with_owner_identity(tmp_path)

    try:
        asyncio.run(service.log_strength_entry(date="2026-07-19", exercises=[]))
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_log_strength_entry_accepts_snake_case_weight_key(tmp_path: Path) -> None:
    service, _cloud, _public_key = _bound_service_with_owner_identity(tmp_path)

    result = asyncio.run(
        service.log_strength_entry(
            date="2026-07-19",
            exercises=[{"name": "深蹲", "sets": [{"weight_kg": 60, "reps": 5}]}],
        )
    )

    session = result["session"]
    assert session["exercises"][0]["sets"][0]["weightKg"] == 60.0


# ── food kind (added 2026-07-20) ─────────────────────────────────────────────


def _food_payload(
    entry_id: str,
    date: str,
    meals: list[dict[str, Any]],
    *,
    note: str | None = None,
    updated_at: str | None = "2026-07-20T20:00:00Z",
) -> bytes:
    payload: dict[str, Any] = {
        "entryID": entry_id,
        "date": date,
        "meals": meals,
        "createdAt": "2026-07-20T18:00:00Z",
    }
    if note is not None:
        payload["note"] = note
    if updated_at is not None:
        payload["updatedAt"] = updated_at
    return json.dumps(payload).encode()


def test_food_summary_orders_newest_first_and_counts_items(tmp_path: Path) -> None:
    service, cloud, public_key = _bound_service(tmp_path)
    cloud.envelopes = [
        _make_envelope(
            public_key,
            _food_payload(
                "food-aaaa000011112222aaaa000011112222",
                "2026-07-19T00:00:00Z",
                [{"name": "dinner", "items": [{"food": "秦镇老碗红烧牛腩盖饭"}]}],
            ),
            metric_type="food",
            envelope_id="env-fd1",
            blob_id="food-aaaa000011112222aaaa000011112222",
            owner_user_id="dce9b9cf-0000-0000-0000-000000000000",
        ),
        _make_envelope(
            public_key,
            _food_payload(
                "food-bbbb000011112222bbbb000011112222",
                "2026-07-20T00:00:00Z",
                [
                    {
                        "name": "dinner",
                        "items": [
                            {"food": "青椒肉丝", "portion": "1 份"},
                            {"food": "番茄炒蛋", "portion": "1 份"},
                        ],
                    }
                ],
                note="宿舍自炊，救回中午空腹后的低血糖",
            ),
            metric_type="food",
            envelope_id="env-fd2",
            blob_id="food-bbbb000011112222bbbb000011112222",
            owner_user_id="dce9b9cf-0000-0000-0000-000000000000",
        ),
    ]

    summary = asyncio.run(service.food_summary(limit=50))

    assert summary["errors"] == []
    assert summary["day_count"] == 2
    newest = summary["days"][0]
    assert newest["date"].startswith("2026-07-20")
    assert newest["note"].startswith("宿舍自炊")
    assert newest["total_item_count"] == 2
    assert summary["days"][1]["total_item_count"] == 1


def test_food_summary_dedups_edits_newest_wins(tmp_path: Path) -> None:
    service, cloud, public_key = _bound_service(tmp_path)
    entry_id = "food-cccc000011112222cccc000011112222"
    cloud.envelopes = [
        _make_envelope(
            public_key,
            _food_payload(
                entry_id,
                "2026-07-20T00:00:00Z",
                [{"name": "lunch", "items": [{"food": "香蕉"}]}],
                updated_at="2026-07-20T12:00:00Z",
            ),
            metric_type="food",
            envelope_id="env-fd-orig",
            blob_id=entry_id,
        ),
        _make_envelope(
            public_key,
            _food_payload(
                entry_id,
                "2026-07-20T00:00:00Z",
                [
                    {"name": "lunch", "items": [{"food": "香蕉"}]},
                    {"name": "dinner", "items": [{"food": "青椒肉丝"}, {"food": "番茄炒蛋"}]},
                ],
                updated_at="2026-07-20T20:00:00Z",
            ),
            metric_type="food",
            envelope_id="env-fd-edit",
            blob_id=entry_id,
        ),
    ]

    summary = asyncio.run(service.food_summary(limit=50))

    assert summary["day_count"] == 1
    assert len(summary["days"][0]["meals"]) == 2
    assert summary["days"][0]["total_item_count"] == 3


def test_food_summary_collects_malformed_into_errors(tmp_path: Path) -> None:
    service, cloud, public_key = _bound_service(tmp_path)
    cloud.envelopes = [
        _make_envelope(
            public_key,
            json.dumps({"entryID": "food-bad", "date": "2026-07-20T00:00:00Z", "meals": []}).encode(),
            metric_type="food",
            envelope_id="env-fd-bad",
            blob_id="food-bad",
        ),
    ]

    summary = asyncio.run(service.food_summary(limit=50))

    assert summary["day_count"] == 0
    assert len(summary["errors"]) == 1
    assert "env-fd-bad" in summary["errors"][0]


# ── log_food_entry (agent write path) ────────────────────────────────────────


def test_log_food_entry_creates_fresh_entry_and_round_trips(tmp_path: Path) -> None:
    service, cloud, _public_key = _bound_service_with_owner_identity(tmp_path)

    result = asyncio.run(
        service.log_food_entry(
            date="2026-07-20",
            meals=[
                {
                    "name": "dinner",
                    "timeOfDay": "19:00",
                    "items": [
                        {"food": "青椒肉丝", "portion": "1 份"},
                        {"food": "番茄炒蛋", "portion": "1 份"},
                    ],
                }
            ],
            note="宿舍自炊",
        )
    )

    assert result["updated_existing_day"] is False
    assert result["date"] == "2026-07-20"
    assert result["entry_id"].startswith("food-")
    assert len(cloud.write_calls) == 1

    written_blob = cloud.write_calls[0]["blob"]
    assert written_blob["owner_user_id"] == cloud.owner_user_id
    assert written_blob["source_device_id"] == cloud.owner_device_id
    assert written_blob["metric_type"] == "food"

    written_envelopes = cloud.write_calls[0]["envelopes"]
    kinds = {e["recipient_kind"] for e in written_envelopes}
    assert kinds == {"owner_user", "mcp_server"}

    day = result["day"]
    assert day is not None
    assert _local_calendar_day(_parse_iso8601(day["date"])) == date(2026, 7, 20)
    assert day["note"] == "宿舍自炊"
    assert len(day["meals"]) == 1
    assert day["meals"][0]["timeOfDay"] == "19:00"
    assert day["total_item_count"] == 2

    summary = asyncio.run(service.food_summary(limit=50, fresh=True))
    assert summary["day_count"] == 1
    assert summary["days"][0]["entry_id"] == result["entry_id"]


def test_log_food_entry_editing_same_day_reuses_entry_id(tmp_path: Path) -> None:
    service, cloud, _public_key = _bound_service_with_owner_identity(tmp_path)

    first = asyncio.run(
        service.log_food_entry(
            date="2026-07-20",
            meals=[{"name": "lunch", "items": [{"food": "香蕉"}]}],
        )
    )
    second = asyncio.run(
        service.log_food_entry(
            date="2026-07-20",
            meals=[
                {"name": "lunch", "items": [{"food": "香蕉"}]},
                {"name": "dinner", "items": [{"food": "青椒肉丝"}, {"food": "番茄炒蛋"}]},
            ],
        )
    )

    assert second["updated_existing_day"] is True
    assert second["entry_id"] == first["entry_id"]

    summary = asyncio.run(service.food_summary(limit=50, fresh=True))
    assert summary["day_count"] == 1
    assert len(summary["days"][0]["meals"]) == 2


def test_log_food_entry_rejects_bind_missing_owner_identity(tmp_path: Path) -> None:
    service, _cloud, _public_key = _bound_service(tmp_path)

    try:
        asyncio.run(
            service.log_food_entry(
                date="2026-07-20",
                meals=[{"items": [{"food": "香蕉"}]}],
            )
        )
        raised = False
    except RuntimeError as error:
        raised = True
        assert "re-pair" in str(error) or "re-bind" in str(error) or "bind" in str(error)
    assert raised


def test_log_food_entry_rejects_empty_meals(tmp_path: Path) -> None:
    service, _cloud, _public_key = _bound_service_with_owner_identity(tmp_path)

    try:
        asyncio.run(service.log_food_entry(date="2026-07-20", meals=[]))
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_log_food_entry_rejects_meal_with_no_items(tmp_path: Path) -> None:
    service, _cloud, _public_key = _bound_service_with_owner_identity(tmp_path)

    try:
        asyncio.run(service.log_food_entry(date="2026-07-20", meals=[{"name": "lunch", "items": []}]))
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_log_food_entry_accepts_minimum_shape_just_food_name(tmp_path: Path) -> None:
    """The lowest-friction path: an item with only `food` (no portion, no meal
    name/time) still records — otherwise a rushed "log 香蕉" would be blocked
    and the friction argument for AI-side nutrition estimation collapses."""

    service, _cloud, _public_key = _bound_service_with_owner_identity(tmp_path)

    result = asyncio.run(
        service.log_food_entry(
            date="2026-07-20",
            meals=[{"items": [{"food": "香蕉"}]}],
        )
    )

    day = result["day"]
    assert day is not None
    assert day["meals"][0]["items"][0]["food"] == "香蕉"
    # No portion field survives when none was provided (kept clean, not "": null).
    assert "portion" not in day["meals"][0]["items"][0]


# ── vo2max kind (added 2026-07-20) ───────────────────────────────────────────


def _vo2max_payload(
    sample_id: str,
    sample_start_date: str,
    vo2_max_ml_kg_min: float,
) -> bytes:
    payload: dict[str, Any] = {
        "sampleID": sample_id,
        "sampleStartDate": sample_start_date,
        "vo2MaxMlKgMin": vo2_max_ml_kg_min,
    }
    return json.dumps(payload).encode()


def test_vo2max_records_computes_latest_peak_trough_average(tmp_path: Path) -> None:
    service, cloud, public_key = _bound_service(tmp_path)
    cloud.envelopes = [
        # ordering across envelope IDs is DELIBERATELY not date-ordered;
        # the service must sort newest-first by created_at, not by input order.
        _make_envelope(
            public_key,
            _vo2max_payload("vo2max-1728000000", "2025-10-04T14:20:00Z", 42.0),
            metric_type="vo2max",
            envelope_id="env-vo-a",
            blob_id="vo2max-1728000000",
        ),
        _make_envelope(
            public_key,
            _vo2max_payload("vo2max-1735000000", "2025-12-24T09:15:00Z", 35.6),
            metric_type="vo2max",
            envelope_id="env-vo-b",
            blob_id="vo2max-1735000000",
        ),
        _make_envelope(
            public_key,
            _vo2max_payload("vo2max-1729800000", "2025-10-25T16:00:00Z", 45.7),
            metric_type="vo2max",
            envelope_id="env-vo-c",
            blob_id="vo2max-1729800000",
        ),
    ]

    summary = asyncio.run(service.vo2max_records(limit=50))

    assert summary["errors"] == []
    assert summary["count"] == 3
    # peak / trough / average are order-agnostic; latest is order-dependent
    # (created_at desc). Every envelope in this fixture uses the same synthetic
    # created_at ("2026-04-27T00:00:00Z"), so newest-first is decided by blob id;
    # peak/trough/average are the deterministic assertions to lean on.
    assert summary["peak_ml_kg_min"] == 45.7
    assert summary["trough_ml_kg_min"] == 35.6
    assert summary["average_ml_kg_min"] == round((42.0 + 35.6 + 45.7) / 3, 1)


def test_vo2max_records_owner_prefix_filter_selects_one_person(tmp_path: Path) -> None:
    service, cloud, public_key = _bound_service(tmp_path)
    cloud.envelopes = [
        _make_envelope(
            public_key,
            _vo2max_payload("vo2max-1", "2026-04-19T10:00:00Z", 39.3),
            metric_type="vo2max",
            envelope_id="env-vo-owner",
            blob_id="vo2max-1",
            owner_user_id="dce9b9cf-5daf-470e-9606-c5076cce55ae",
        ),
        _make_envelope(
            public_key,
            _vo2max_payload("vo2max-2", "2026-04-20T10:00:00Z", 38.1),
            metric_type="vo2max",
            envelope_id="env-vo-partner",
            blob_id="vo2max-2",
            owner_user_id="f8350dfc-1111-2222-3333-444444444444",
        ),
    ]

    only_owner = asyncio.run(service.vo2max_records(limit=50, owner="dce9"))
    only_partner = asyncio.run(service.vo2max_records(limit=50, owner="f835"))

    assert only_owner["count"] == 1
    assert only_partner["count"] == 1
    assert only_owner["records"][0]["vo2_max_ml_kg_min"] == 39.3
    assert only_partner["records"][0]["vo2_max_ml_kg_min"] == 38.1


def test_vo2max_records_collects_malformed_into_errors(tmp_path: Path) -> None:
    service, cloud, public_key = _bound_service(tmp_path)
    cloud.envelopes = [
        _make_envelope(
            public_key,
            json.dumps({"sampleID": "vo2max-bad", "sampleStartDate": "2026-07-20T10:00:00Z"}).encode(),
            metric_type="vo2max",
            envelope_id="env-vo-bad",
            blob_id="vo2max-bad",
        ),
    ]

    summary = asyncio.run(service.vo2max_records(limit=50))

    assert summary["count"] == 0
    assert len(summary["errors"]) == 1
    assert "env-vo-bad" in summary["errors"][0]


# ── 2026-07-24 MCP feedback sprint ───────────────────────────────────────────
# Covers: owner mix-guard, local_date fields, stage-tagged errors + errors_note,
# log_food_entry merge mode, structured nutrition fields, log_note.


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


def test_weight_trend_unfiltered_mixed_owners_gets_warning(tmp_path: Path) -> None:
    service, cloud, public_key = _bound_service(tmp_path)
    cloud.envelopes = [
        _make_envelope(
            public_key,
            _body_payload("body-1", "2026-07-21T16:00:00Z", 82.75),
            metric_type="body",
            envelope_id="env-body-owner",
            blob_id="body-1",
            owner_user_id="dce9b9cf-5daf-470e-9606-c5076cce55ae",
        ),
        _make_envelope(
            public_key,
            _body_payload("body-2", "2026-07-20T16:00:00Z", 40.2),
            metric_type="body",
            envelope_id="env-body-partner",
            blob_id="body-2",
            owner_user_id="f8350dfc-1111-2222-3333-444444444444",
        ),
    ]

    mixed = asyncio.run(service.weight_trend_summary(limit=50))
    assert mixed["mixed_owners"] is True
    assert mixed["owner_user_id_prefixes"] == ["dce9b9cf", "f8350dfc"]
    assert "warning" in mixed

    filtered = asyncio.run(service.weight_trend_summary(limit=50, owner="dce9", fresh=True))
    assert "mixed_owners" not in filtered
    assert filtered["latest_kg"] == 82.75


def test_weight_day_carries_local_date(tmp_path: Path) -> None:
    service, cloud, public_key = _bound_service(tmp_path)
    day_start = "2026-07-21T16:00:00Z"
    cloud.envelopes = [
        _make_envelope(
            public_key,
            _body_payload("body-1", day_start, 82.55),
            metric_type="body",
            envelope_id="env-body-1",
            blob_id="body-1",
            owner_user_id="dce9b9cf-5daf-470e-9606-c5076cce55ae",
        ),
    ]

    summary = asyncio.run(service.weight_trend_summary(limit=50, owner="dce9"))
    day = summary["days"][0]
    # local_date must match the same local-day bucketing the service uses
    # elsewhere (never assume the test machine's UTC offset).
    assert day["local_date"] == _local_calendar_day(_parse_iso8601(day_start)).isoformat()


def test_parse_failure_is_stage_tagged_and_carries_errors_note(tmp_path: Path) -> None:
    service, cloud, public_key = _bound_service(tmp_path)
    cloud.envelopes = [
        _make_envelope(
            public_key,
            json.dumps({"dayID": "body-bad", "dayStartDate": "2026-07-20T16:00:00Z"}).encode(),
            metric_type="body",
            envelope_id="env-body-bad",
            blob_id="body-bad",
        ),
    ]

    summary = asyncio.run(service.weight_trend_summary(limit=50))
    assert len(summary["errors"]) == 1
    assert "parse_failed" in summary["errors"][0]
    assert "errors_note" in summary


def test_log_food_entry_merge_appends_without_dropping(tmp_path: Path) -> None:
    service, cloud, _public_key = _bound_service_with_owner_identity(tmp_path)

    first = asyncio.run(
        service.log_food_entry(
            date="2026-07-20",
            meals=[
                {"name": "lunch", "items": [{"food": "鸡胸肉", "portion": "150g"}]},
                {"name": "snack", "items": [{"food": "香蕉", "portion": "1 根"}]},
            ],
            note="训练日",
        )
    )
    second = asyncio.run(
        service.log_food_entry(
            date="2026-07-20",
            meals=[
                {"name": "lunch", "items": [{"food": "米饭", "portion": "半碗"}]},
                {"name": "dinner", "items": [{"food": "牛腱", "portion": "100g"}]},
            ],
            merge=True,
        )
    )

    assert second["entry_id"] == first["entry_id"]
    assert second["updated_existing_day"] is True
    assert second["merge_mode"] is True

    day = second["day"]
    meals_by_name = {m.get("name"): m for m in day["meals"]}
    # Same-name meal got its items appended; other meals survived untouched.
    assert [i["food"] for i in meals_by_name["lunch"]["items"]] == ["鸡胸肉", "米饭"]
    assert [i["food"] for i in meals_by_name["snack"]["items"]] == ["香蕉"]
    assert [i["food"] for i in meals_by_name["dinner"]["items"]] == ["牛腱"]
    # note=None in merge mode keeps the existing day note.
    assert day["note"] == "训练日"


def test_log_food_entry_replace_mode_still_replaces(tmp_path: Path) -> None:
    service, cloud, _public_key = _bound_service_with_owner_identity(tmp_path)

    asyncio.run(
        service.log_food_entry(
            date="2026-07-20",
            meals=[{"name": "lunch", "items": [{"food": "鸡胸肉"}]}],
        )
    )
    replaced = asyncio.run(
        service.log_food_entry(
            date="2026-07-20",
            meals=[{"name": "dinner", "items": [{"food": "牛腱"}]}],
        )
    )

    day = replaced["day"]
    assert [m.get("name") for m in day["meals"]] == ["dinner"]


def test_log_food_entry_structured_nutrition_persists(tmp_path: Path) -> None:
    service, cloud, _public_key = _bound_service_with_owner_identity(tmp_path)

    result = asyncio.run(
        service.log_food_entry(
            date="2026-07-21",
            meals=[
                {
                    "name": "breakfast",
                    "items": [
                        # camelCase wire key and snake_case alias both accepted.
                        {"food": "香蕉", "portion": "1 根", "kcal": 105, "proteinGrams": 1.3},
                        {"food": "全安素", "portion": "3 勺", "kcal": 150, "protein_g": 6.8},
                    ],
                }
            ],
        )
    )

    items = result["day"]["meals"][0]["items"]
    assert items[0]["kcal"] == 105.0
    assert items[0]["proteinGrams"] == 1.3
    assert items[1]["kcal"] == 150.0
    assert items[1]["proteinGrams"] == 6.8

    # And the read-back tool sees the same numbers (no silent stripping).
    summary = asyncio.run(service.food_summary(limit=50, fresh=True))
    read_items = summary["days"][0]["meals"][0]["items"]
    assert read_items[0]["kcal"] == 105.0
    assert read_items[1]["proteinGrams"] == 6.8


def test_log_food_entry_rejects_bad_nutrition_values(tmp_path: Path) -> None:
    service, cloud, _public_key = _bound_service_with_owner_identity(tmp_path)

    try:
        asyncio.run(
            service.log_food_entry(
                date="2026-07-21",
                meals=[{"items": [{"food": "香蕉", "kcal": -5}]}],
            )
        )
        raise AssertionError("negative kcal should raise")
    except ValueError as error:
        assert "negative" in str(error)

    try:
        asyncio.run(
            service.log_food_entry(
                date="2026-07-21",
                meals=[{"items": [{"food": "香蕉", "protein_g": "很多"}]}],
            )
        )
        raise AssertionError("non-numeric protein should raise")
    except ValueError as error:
        assert "non-numeric" in str(error)


def _body_day_id_for(year: int, month: int, day: int) -> str:
    """The dayID log_weight_entry will compute for that LOCAL calendar day."""

    local_tz = datetime.now().astimezone().tzinfo
    return f"body-{int(datetime(year, month, day, tzinfo=local_tz).timestamp())}"


def test_log_weight_entry_preserves_scale_composition(tmp_path: Path, monkeypatch: Any) -> None:
    """Invariant 41: a bare weight log must not wipe that day's scale reading.

    Before iOS read body-composition samples (2026-08-05) these fields were
    always null, so this whole-day overwrite was free. The moment a scale starts
    populating them, `log_weight_entry` becomes a silent eraser unless it carries
    them over — an agent has no way to supply fat/BMI/lean mass itself.
    """

    service, cloud, _public_key = _bound_service_with_owner_identity(tmp_path)
    day_id = _body_day_id_for(2026, 8, 5)

    # Only the FIRST read (the carry-over lookup) is faked; the verification read
    # afterwards runs for real, so the assertion below decrypts what was actually
    # sealed. Asserting the returned `preserved_composition` instead does NOT
    # work — it is a value this method computes, so it stays correct even if the
    # plaintext drops it (verified: breaking the spread left that test green).
    real_summary = service.weight_trend_summary
    reads = {"n": 0}

    async def _summary(**kwargs: Any) -> dict[str, Any]:
        reads["n"] += 1
        if reads["n"] == 1:
            return {
                "days": [
                    {
                        "day_id": day_id,
                        "weight_kg": 83.1,
                        "body_fat_percent": 22.4,
                        "bmi": 24.9,
                        "lean_body_mass_kg": 64.5,
                    }
                ]
            }
        return await real_summary(**kwargs)

    monkeypatch.setattr(service, "weight_trend_summary", _summary)

    asyncio.run(service.log_weight_entry(weight_kg=82.6, date="2026-08-05"))
    assert cloud.write_calls[-1]["blob"]["metric_type"] == "body"

    # Round-trip: read the blob back through real decryption.
    written = asyncio.run(real_summary(fresh=True))
    same_day = [d for d in written["days"] if d["day_id"] == day_id]
    assert same_day, "the weight blob just written should read back"
    assert same_day[0]["weight_kg"] == 82.6, "the new weight must land"
    assert same_day[0]["body_fat_percent"] == 22.4, "scale body fat was erased by a bare weight log"
    assert same_day[0]["bmi"] == 24.9
    assert same_day[0]["lean_body_mass_kg"] == 64.5


def test_log_weight_entry_writes_null_composition_when_the_day_is_new(tmp_path: Path) -> None:
    """The other half: nothing to carry over must not invent values."""

    service, cloud, _public_key = _bound_service_with_owner_identity(tmp_path)

    result = asyncio.run(service.log_weight_entry(weight_kg=82.6, date="2026-08-05"))

    assert result["preserved_composition"] == {
        "bodyFatPercent": None,
        "bmi": None,
        "leanBodyMassKg": None,
    }
    assert cloud.write_calls[-1]["blob"]["metric_type"] == "body"


def test_log_note_creates_and_round_trips(tmp_path: Path) -> None:
    service, cloud, _public_key = _bound_service_with_owner_identity(tmp_path)

    result = asyncio.run(
        service.log_note(text="今天有点低落，Matt 的合同还没回音", kind="mood", date="2026-07-24")
    )

    assert result["note_id"].startswith("note-")
    assert result["updated_existing_note"] is False
    assert result["kind"] == "mood"

    written_blob = cloud.write_calls[-1]["blob"]
    assert written_blob["metric_type"] == "note"
    assert written_blob["owner_user_id"] == cloud.owner_user_id
    kinds = {e["recipient_kind"] for e in cloud.write_calls[-1]["envelopes"]}
    assert kinds == {"owner_user", "mcp_server"}

    note = result["note"]
    assert note is not None
    assert note["text"].startswith("今天有点低落")
    assert note["target_kind"] == "mood"
    assert _local_calendar_day(_parse_iso8601(note["target_date"])) == date(2026, 7, 24)


def test_log_note_same_kind_day_upserts_in_place(tmp_path: Path) -> None:
    service, cloud, _public_key = _bound_service_with_owner_identity(tmp_path)

    first = asyncio.run(service.log_note(text="v1", kind="general", date="2026-07-24"))
    second = asyncio.run(service.log_note(text="v2 修订", kind="general", date="2026-07-24"))

    assert second["note_id"] == first["note_id"]
    assert second["updated_existing_note"] is True
    assert second["note"]["text"] == "v2 修订"

    # A different kind on the same day is a separate note.
    other = asyncio.run(service.log_note(text="mood note", kind="mood", date="2026-07-24"))
    assert other["note_id"] != first["note_id"]


def test_log_note_rejects_ios_kinds_and_empty_text(tmp_path: Path) -> None:
    service, cloud, _public_key = _bound_service_with_owner_identity(tmp_path)

    for bad_kind in ("sleep", "menstrual", "diary"):
        try:
            asyncio.run(service.log_note(text="x", kind=bad_kind))
            raise AssertionError(f"kind {bad_kind!r} should raise")
        except ValueError as error:
            assert "unsupported note kind" in str(error)

    try:
        asyncio.run(service.log_note(text="   ", kind="mood"))
        raise AssertionError("blank text should raise")
    except ValueError as error:
        assert "non-empty" in str(error)


# ── 2026-07-24 full-tool sweep fixes ─────────────────────────────────────────
# Covers: business-time sort before the limit cut (backfill shuffles upload
# order), wrist-temp honest field name, TDEE partial-today exclusion.


def _mindfulness_payload(day_id: str, day_start: str, minutes: float) -> bytes:
    return json.dumps(
        {"dayID": day_id, "dayStartDate": day_start, "sessionCount": 1, "totalMinutes": minutes}
    ).encode()


def test_mindfulness_sorted_by_business_day_not_upload_order(tmp_path: Path) -> None:
    service, cloud, public_key = _bound_service(tmp_path)
    # Upload order (== created_at order in the fake) is deliberately shuffled
    # vs the payload's own days — a history backfill does exactly this.
    cloud.envelopes = [
        _make_envelope(
            public_key,
            _mindfulness_payload("m-2", "2026-04-24T16:00:00Z", 1.4),
            metric_type="mindfulness",
            envelope_id="env-m2",
            blob_id="m-2",
            owner_user_id="dce9b9cf-0000-0000-0000-000000000000",
        ),
        _make_envelope(
            public_key,
            _mindfulness_payload("m-1", "2026-03-02T16:00:00Z", 8.5),
            metric_type="mindfulness",
            envelope_id="env-m1",
            blob_id="m-1",
            owner_user_id="dce9b9cf-0000-0000-0000-000000000000",
        ),
        _make_envelope(
            public_key,
            _mindfulness_payload("m-3", "2026-07-20T16:00:00Z", 3.0),
            metric_type="mindfulness",
            envelope_id="env-m3",
            blob_id="m-3",
            owner_user_id="dce9b9cf-0000-0000-0000-000000000000",
        ),
    ]

    summary = asyncio.run(service.mindfulness_summary(limit=2, owner="dce9"))

    returned_days = [d["day_start_date"] for d in summary["days"]]
    # Newest two by BUSINESS day — the July record must survive the cut even
    # though it was uploaded last, and order must be newest-first.
    assert returned_days == ["2026-07-20T16:00:00Z", "2026-04-24T16:00:00Z"]


def test_wrist_temp_carries_honest_absolute_field(tmp_path: Path) -> None:
    service, cloud, public_key = _bound_service(tmp_path)
    cloud.envelopes = [
        _make_envelope(
            public_key,
            json.dumps(
                {
                    "dayID": "wrist-1",
                    "dayStartDate": "2026-07-23T16:55:00Z",
                    "temperatureDeltaCelsius": 35.99,
                }
            ).encode(),
            metric_type="wrist_temp",
            envelope_id="env-wrist-1",
            blob_id="wrist-1",
            owner_user_id="dce9b9cf-0000-0000-0000-000000000000",
        ),
    ]

    summary = asyncio.run(service.wrist_temp_records(limit=5, owner="dce9"))
    record = summary["records"][0]
    assert record["wrist_temperature_celsius"] == 35.99
    # Legacy misnomer stays for back-compat, same value.
    assert record["temperature_delta_celsius"] == 35.99


def test_total_energy_burned_excludes_partial_today_from_average(tmp_path: Path) -> None:
    from datetime import date as _date_type, timedelta as _td

    from vaultbeat_mcp_local.service import _local_midnight_iso

    service, cloud, public_key = _bound_service(tmp_path)
    today = _date_type.today()

    def _basal_payload(record_id: str, start_iso: str, kcal: float) -> bytes:
        return json.dumps(
            {"sampleID": record_id, "sampleStartDate": start_iso, "basalEnergyKcal": kcal}
        ).encode()

    def _activity_payload(day_id: str, day_start: str) -> bytes:
        return json.dumps(
            {
                "dayID": day_id,
                "dayStartDate": day_start,
                "stepCount": 1000,
                "activeEnergyKcal": 100.0,
                "exerciseMinutes": 10,
                "standMinutes": 5,
                "distanceMeters": 800.0,
            }
        ).encode()

    envelopes = []
    for i, (day, kcal) in enumerate(
        [(today, 200.0), (today - _td(days=1), 2000.0), (today - _td(days=2), 2100.0)]
    ):
        midnight = _local_midnight_iso(day)
        envelopes.append(
            _make_envelope(
                public_key,
                _basal_payload(f"basal-{i}", midnight, kcal),
                metric_type="basal_energy",
                envelope_id=f"env-basal-{i}",
                blob_id=f"basal-{i}",
                owner_user_id="dce9b9cf-0000-0000-0000-000000000000",
            )
        )
        envelopes.append(
            _make_envelope(
                public_key,
                _activity_payload(f"act-{i}", midnight),
                metric_type="activity",
                envelope_id=f"env-act-{i}",
                blob_id=f"act-{i}",
                owner_user_id="dce9b9cf-0000-0000-0000-000000000000",
            )
        )
    cloud.envelopes = envelopes

    result = asyncio.run(service.total_energy_burned(days=7, owner="dce9"))

    by_day = {d["day"]: d for d in result["days"]}
    assert by_day[today.isoformat()]["partial"] is True
    # Average = the two COMPLETE days only: (2000+100 + 2100+100) / 2 = 2150.
    assert result["average_tdee_kcal"] == 2150.0


def test_unsupported_metric_degrades_instead_of_raising(tmp_path: Path) -> None:
    """An edge older than this client must cost one kind, not the whole read.

    Pins the fix for the 2026-07-22 incident shape: `hrv_hourly` shipped without
    being added to the edge allowlist, and because it was also `get_hrv`'s new
    default granularity, every default HRV read raised for two days instead of
    reporting that one kind as unavailable.
    """
    config_path = tmp_path / "config.json"
    cloud = FakeCloudClient()
    service = VaultbeatLocalService(ConfigStore(config_path), cloud)

    service.start_binding(server_name="Mac Studio", api_base_url="https://api.test")
    asyncio.run(service.poll_once())
    cloud.unsupported_metric_types = {"hrv_hourly"}

    records, errors = asyncio.run(
        service.sync_decrypted_records(metric_type="hrv_hourly", fresh=True)
    )

    # Degraded, not raised.
    assert records == []
    assert len(errors) == 1
    assert errors[0].startswith("unsupported_metric:hrv_hourly")
    # The message must be actionable by an agent relaying it to a human.
    assert "cloud" in errors[0].lower()

    # Crucially: a DIFFERENT kind still works on the very same client.
    config = ConfigStore(config_path).require_bound()
    cloud.envelopes = [_make_envelope(config.public_key_base64, b'{"stage":"asleep"}')]
    records, errors = asyncio.run(
        service.sync_decrypted_records(metric_type="sleep", fresh=True)
    )
    assert errors == []
    assert records[0].payload == {"stage": "asleep"}


def test_doctor_reports_capability_gap_for_an_older_app(tmp_path: Path) -> None:
    """An account whose app predates a feature must be told, not left guessing.

    The first paying customer (2026-07-25) runs App Store 1.2.0, which has no
    executor for strength / food / vo2max / basal_energy / hrv_hourly. Those
    tools return empty with no error — identical, from the outside, to a broken
    install. doctor names them and the iOS date each requires.
    """
    config_path = tmp_path / "config.json"
    cloud = FakeCloudClient()
    service = VaultbeatLocalService(ConfigStore(config_path), cloud)
    service.start_binding(server_name="Mac Studio", api_base_url="https://api.test")
    asyncio.run(service.poll_once())
    config = ConfigStore(config_path).require_bound()

    # An old app writes sleep and activity, and nothing added after 2026-07-19.
    cloud.envelopes = [
        _make_envelope(config.public_key_base64, b'{"stage":"asleep"}'),
    ]

    report = asyncio.run(service.doctor())
    caps = report["capabilities"]

    assert caps["available"] is True
    gated = caps["possibly_needs_newer_app"]
    assert set(gated) == {"strength", "food", "vo2max", "basal_energy", "hrv_hourly"}
    assert gated["basal_energy"] == "2026-07-21"
    # It must not claim to know the app version — only that a newer one is needed.
    # (The previous form of this line asserted `"possibly" in "possibly_needs_newer_app"`,
    # a substring check against a literal — always true, testing nothing.)
    assert "possibly_needs_newer_app" in caps

    # The note must not present "your app is old" as THE explanation: an empty
    # kind has three indistinguishable causes from the server's side, and the
    # permission one is the least obvious because a HealthKit read denial is
    # invisible to the app — it reports success and simply delivers nothing.
    # Naming the recovery path matters: the paying customer who prompted this
    # had 7 kinds empty from authorization, not app age.
    note = caps["note"]
    assert "build from the stated date or later" in note
    assert "Apple Health access" in note
    assert "not been recorded yet" in note


def test_doctor_capability_report_is_quiet_when_everything_has_data(tmp_path: Path) -> None:
    """No false alarm for an up-to-date account (the owner's own case)."""
    config_path = tmp_path / "config.json"
    cloud = FakeCloudClient()
    service = VaultbeatLocalService(ConfigStore(config_path), cloud)
    service.start_binding(server_name="Mac Studio", api_base_url="https://api.test")
    asyncio.run(service.poll_once())
    config = ConfigStore(config_path).require_bound()

    from vaultbeat_mcp_local.service import KNOWN_METRIC_TYPES

    cloud.envelopes = [
        _make_envelope(config.public_key_base64, b'{"v":1}', metric_type=kind, blob_id=f"blob-{kind}", envelope_id=f"env-{kind}")
        for kind in sorted(KNOWN_METRIC_TYPES)
    ]

    caps = asyncio.run(service.doctor())["capabilities"]
    assert caps["possibly_needs_newer_app"] == {}
    assert caps["kinds_without_data"] == []


# ── limit must cut on BUSINESS time, not upload order (2026-07-27) ────────────
#
# `_records_for_metric` sorts by created_at = UPLOAD BATCH time (one batch holds
# up to 50 blobs, all sharing a timestamp). Cutting there and parsing afterwards
# silently drops the newest days whenever a history backfill uploads old data
# late. Eight kinds were fixed 2026-07-24; water/body/menstrual/basal still cut
# early until now. Reproduced live: get_weight_trend(limit=10, owner="dce9")
# hid the real 2026-07-21 weigh-in (83.0 kg) while keeping an older 07-10 one.


def _basal_payload(sample_id: str, sample_start: str, kcal: float) -> bytes:
    return json.dumps(
        {
            "sampleID": sample_id,
            "sampleStartDate": sample_start,
            "basalEnergyKcal": kcal,
        }
    ).encode()


# Upload order (== the fake's created_at order) deliberately disagrees with the
# payloads' own days, exactly like a backfill batch.
_UPLOAD_ORDER_DAYS = [
    "2026-04-24T16:00:00Z",
    "2026-03-02T16:00:00Z",
    "2026-07-20T16:00:00Z",
]
_NEWEST_TWO_DAYS = ["2026-07-20T16:00:00Z", "2026-04-24T16:00:00Z"]
_TEST_OWNER = "dce9b9cf-0000-0000-0000-000000000000"

# strength / food / note carry a plain local day ("YYYY-MM-DD"), not a timestamp.
# Same three days in the same deliberately-wrong upload order.
_UPLOAD_ORDER_DATES = ["2026-04-24", "2026-03-02", "2026-07-20"]
_NEWEST_TWO_DATES = ["2026-07-20", "2026-04-24"]


@pytest.mark.parametrize(
    ("metric_type", "build_payload", "call", "read_days", "expected"),
    [
        (
            "water",
            lambda i, day: _water_payload(f"w-{i}", day, 0.5, 4),
            lambda svc: svc.water_intake_summary(limit=2, owner="dce9"),
            lambda summary: [d["day_start_date"] for d in summary["days"]],
            _NEWEST_TWO_DAYS,
        ),
        (
            "body",
            lambda i, day: _body_payload(f"b-{i}", day, 80.0 + i),
            lambda svc: svc.weight_trend_summary(limit=2, owner="dce9"),
            lambda summary: [d["day_start_date"] for d in summary["days"]],
            _NEWEST_TWO_DAYS,
        ),
        (
            "menstrual",
            lambda i, day: _menstrual_payload(f"mc-{i}", day, "medium"),
            lambda svc: svc.menstrual_cycle_summary(limit=2, owner="dce9"),
            lambda summary: [d["day_start_date"] for d in summary["days"]],
            _NEWEST_TWO_DAYS,
        ),
        # ── The four hand-logged kinds (added 2026-07-29, third regression) ──
        # These are the ones the owner writes daily via log_*, so "log today's
        # session, then ask what I trained recently, and not see it" is the
        # concrete failure. Their default limits (120/120/120/90) meant EVERY
        # call cut on upload order, not just calls that passed limit explicitly.
        (
            "symptom",
            lambda i, day: _symptom_payload(
                f"sy-{i}",
                day,
                [
                    {
                        "symptomType": "headache",
                        "severity": "mild",
                        "startDate": day,
                        "endDate": day,
                    }
                ],
            ),
            lambda svc: svc.symptom_summary(limit=2),
            lambda summary: [d["day_start_date"] for d in summary["owners"][0]["days"]],
            _NEWEST_TWO_DAYS,
        ),
        (
            "note",
            lambda i, day: _note_payload(f"n-{i}", "general", day[:10], f"note {i}"),
            lambda svc: svc.notes_summary(limit=2),
            lambda summary: [n["target_date"] for n in summary["kinds"][0]["notes"]],
            _NEWEST_TWO_DATES,
        ),
        (
            "strength",
            lambda i, day: _strength_payload(
                f"st-{i}", day[:10], [{"name": "squat", "sets": [{"weightKg": 60, "reps": 5}]}]
            ),
            lambda svc: svc.strength_summary(limit=2),
            lambda summary: [s["date"] for s in summary["sessions"]],
            _NEWEST_TWO_DATES,
        ),
        (
            "food",
            lambda i, day: _food_payload(
                f"fd-{i}", day[:10], [{"items": [{"food": "rice"}]}]
            ),
            lambda svc: svc.food_summary(limit=2),
            lambda summary: [d["date"] for d in summary["days"]],
            _NEWEST_TWO_DATES,
        ),
    ],
)
def test_limit_cuts_on_business_day_not_upload_order(
    tmp_path: Path,
    metric_type: str,
    build_payload: Any,
    call: Any,
    read_days: Any,
    expected: list[str],
) -> None:
    service, cloud, public_key = _bound_service(tmp_path)
    cloud.envelopes = [
        _make_envelope(
            public_key,
            build_payload(i, day),
            metric_type=metric_type,
            envelope_id=f"env-{metric_type}-{i}",
            blob_id=f"blob-{metric_type}-{i}",
            owner_user_id=_TEST_OWNER,
        )
        for i, day in enumerate(_UPLOAD_ORDER_DAYS)
    ]

    summary = asyncio.run(call(service))

    # The July record must survive the cut even though it was uploaded LAST,
    # and the result must be newest-first.
    assert read_days(summary) == expected


def test_basal_energy_limit_cuts_on_business_day_not_upload_order(tmp_path: Path) -> None:
    """Same bug, asserted on kcal because the daily `day` key is timezone-local.

    The three sample days are months apart, so no timezone can reorder them.
    """
    service, cloud, public_key = _bound_service(tmp_path)
    kcal_by_day = {
        "2026-04-24T16:00:00Z": 200.0,
        "2026-03-02T16:00:00Z": 100.0,
        "2026-07-20T16:00:00Z": 300.0,
    }
    cloud.envelopes = [
        _make_envelope(
            public_key,
            _basal_payload(f"be-{i}", day, kcal_by_day[day]),
            metric_type="basal_energy",
            envelope_id=f"env-be-{i}",
            blob_id=f"blob-be-{i}",
            owner_user_id=_TEST_OWNER,
        )
        for i, day in enumerate(_UPLOAD_ORDER_DAYS)
    ]

    summary = asyncio.run(service.basal_energy_records(limit=2, owner="dce9"))

    assert summary["sample_count"] == 2
    # July (uploaded last) survives and leads; March (oldest) is the one cut.
    assert [d["basal_kcal"] for d in summary["daily"]] == [300.0, 200.0]


# ── sleep: "0h00m" was a lie on Watch-less nights (2026-07-27) ────────────────


def _sleep_payload(
    session_date: str,
    bedtime: str,
    wake_time: str,
    samples: list[dict[str, str]],
) -> bytes:
    return json.dumps(
        {
            "session": {
                "sessionDate": session_date,
                "bedtime": bedtime,
                "wakeTime": wake_time,
                "provenance": "healthkitSleep",
                "samples": samples,
            },
            "heartRateSamples": [],
            "respiratoryRateSamples": [],
        }
    ).encode()


def _sleep_envelopes(public_key: str) -> list[dict[str, Any]]:
    """Two nights: the newer one in-bed-only (Watch off), the older one staged."""
    in_bed_only = _sleep_payload(
        "2026-07-25T16:00:00Z",
        "2026-07-25T15:00:00Z",
        "2026-07-25T22:00:00Z",
        [
            {
                "stage": "inBed",
                "startDate": "2026-07-25T15:00:00Z",
                "endDate": "2026-07-25T22:00:00Z",
            }
        ],
    )
    staged = _sleep_payload(
        "2026-07-20T16:00:00Z",
        "2026-07-20T15:00:00Z",
        "2026-07-20T22:00:00Z",
        [
            {
                "stage": "asleepCore",
                "startDate": "2026-07-20T15:00:00Z",
                "endDate": "2026-07-20T19:00:00Z",
            },
            {
                "stage": "asleepDeep",
                "startDate": "2026-07-20T19:00:00Z",
                "endDate": "2026-07-20T22:00:00Z",
            },
        ],
    )
    return [
        _make_envelope(
            public_key,
            in_bed_only,
            metric_type="sleep",
            envelope_id="env-sleep-inbed",
            blob_id="blob-sleep-inbed",
            owner_user_id=_TEST_OWNER,
        ),
        _make_envelope(
            public_key,
            staged,
            metric_type="sleep",
            envelope_id="env-sleep-staged",
            blob_id="blob-sleep-staged",
            owner_user_id=_TEST_OWNER,
        ),
    ]


def test_in_bed_only_night_is_labelled_no_sleep_data_not_zero(tmp_path: Path) -> None:
    service, cloud, public_key = _bound_service(tmp_path)
    cloud.envelopes = _sleep_envelopes(public_key)

    summary = asyncio.run(service.sleep_records(limit=5, owner="dce9"))
    newest, older = summary["daily_summary"][0], summary["daily_summary"][1]

    # The honest zero stays — sleep genuinely was not measured that night —
    # but nothing in the output may read as "slept 0 hours".
    assert newest["total_sleep_minutes"] == 0
    assert newest["is_in_bed_only"] is True
    assert newest["duration_label"] == "no sleep data"
    assert newest["in_bed_minutes"] == 420

    # A normal staged night is untouched.
    assert older["is_in_bed_only"] is False
    assert older["duration_label"] == "7h00m"
    assert older["in_bed_minutes"] == 0


def test_sleep_detail_in_bed_only_night_carries_the_same_labels(tmp_path: Path) -> None:
    service, cloud, public_key = _bound_service(tmp_path)
    cloud.envelopes = _sleep_envelopes(public_key)

    summary = asyncio.run(service.sleep_detail_records(limit=5, owner="dce9"))
    newest, older = summary["nights"][0], summary["nights"][1]

    assert newest["is_in_bed_only"] is True
    assert newest["duration_label"] == "no sleep data"
    assert newest["in_bed_minutes"] == 420
    assert newest["total_sleep_minutes"] == 0

    assert older["duration_label"] == "7h00m"
    assert older["total_sleep_minutes"] == 420


def _short_samples_payload() -> bytes:
    """Three 90-second core-sleep samples = 4.5 min of real sleep."""
    windows = [("15:00:00", "15:01:30"), ("15:02:00", "15:03:30"), ("15:04:00", "15:05:30")]
    return _sleep_payload(
        "2026-07-25T16:00:00Z",
        "2026-07-25T15:00:00Z",
        "2026-07-25T15:06:00Z",
        [
            {
                "stage": "asleepCore",
                "startDate": f"2026-07-25T{start}Z",
                "endDate": f"2026-07-25T{end}Z",
            }
            for start, end in windows
        ],
    )


def test_stage_minutes_truncate_once_per_stage_not_once_per_sample(tmp_path: Path) -> None:
    """Per-sample int(seconds/60) threw away up to 59s per sample.

    Three 90-second samples are 4.5 min of sleep; the old code reported 3
    (1+1+1). Across 315 real nights the loss averaged 6.8 min, worst case 17.
    """
    service, cloud, public_key = _bound_service(tmp_path)
    cloud.envelopes = [
        _make_envelope(
            public_key,
            _short_samples_payload(),
            metric_type="sleep",
            envelope_id="env-sleep-short",
            blob_id="blob-sleep-short",
            owner_user_id=_TEST_OWNER,
        )
    ]

    summary = asyncio.run(service.sleep_records(limit=5, owner="dce9"))
    night = summary["daily_summary"][0]
    assert night["total_sleep_minutes"] == 4
    assert night["stage_minutes"]["asleepCore"] == 4

    detail = asyncio.run(service.sleep_detail_records(limit=5, owner="dce9"))
    assert detail["nights"][0]["total_sleep_minutes"] == 4


# ── get_sleep_detail's timeline is opt-in ────────────────────────────────────
# 2026-07-28: the per-sample `timeline` array is ~80% of this payload (~13k
# characters/night), so the old limit=5 default returned ~66k characters and
# overflowed a 25k-token MCP client at limit=4. The owner's own skill passes
# --limit 1 and never hit it; a third-party agent reading only the tool
# description did. Derived stage_* fields must keep working without it.


def _sleep_payload_with_vitals() -> bytes:
    """One staged night that actually carries HR + RR samples."""
    return json.dumps(
        {
            "session": {
                "sessionDate": "2026-07-20T16:00:00Z",
                "bedtime": "2026-07-20T15:00:00Z",
                "wakeTime": "2026-07-20T22:00:00Z",
                "provenance": "healthkitSleep",
                "samples": [
                    {
                        "stage": "asleepCore",
                        "startDate": "2026-07-20T15:00:00Z",
                        "endDate": "2026-07-20T18:00:00Z",
                    },
                    {
                        "stage": "asleepDeep",
                        "startDate": "2026-07-20T18:00:00Z",
                        "endDate": "2026-07-20T22:00:00Z",
                    },
                ],
            },
            "heartRateSamples": [
                {"startDate": "2026-07-20T16:00:00Z", "value": 58},
                {"startDate": "2026-07-20T19:00:00Z", "value": 51},
            ],
            "respiratoryRateSamples": [
                {"startDate": "2026-07-20T16:30:00Z", "value": 14.5},
            ],
        }
    ).encode()


def _service_with_vitals_night(tmp_path: Path) -> Any:
    service, cloud, public_key = _bound_service(tmp_path)
    cloud.envelopes = [
        _make_envelope(
            public_key,
            _sleep_payload_with_vitals(),
            metric_type="sleep",
            envelope_id="env-sleep-vitals",
            blob_id="blob-sleep-vitals",
            owner_user_id=_TEST_OWNER,
        )
    ]
    return service


def test_sleep_detail_omits_timeline_by_default(tmp_path: Path) -> None:
    service = _service_with_vitals_night(tmp_path)

    summary = asyncio.run(service.sleep_detail_records(limit=5, owner="dce9"))
    night = summary["nights"][0]

    assert summary["timeline_included"] is False
    assert "timeline" not in night
    # The derived fields are the whole point of dropping it — they must survive.
    assert night["stage_intervals"]
    assert night["stage_minutes"]["asleepCore"] == 180
    assert night["stage_vitals"]["asleepCore"]["hr_mean"] == 58
    assert night["stage_vitals"]["asleepDeep"]["hr_mean"] == 51
    # Sample counts still tell the caller vitals exist and can be fetched.
    assert night["hr_samples"] == 2
    assert night["rr_samples"] == 1


def test_sleep_detail_include_timeline_returns_the_samples(tmp_path: Path) -> None:
    service = _service_with_vitals_night(tmp_path)

    summary = asyncio.run(
        service.sleep_detail_records(limit=5, owner="dce9", include_timeline=True)
    )
    night = summary["nights"][0]

    assert summary["timeline_included"] is True
    assert [point["hr"] for point in night["timeline"]] == [58, 58, 51]
    assert [point["stage"] for point in night["timeline"]] == [
        "asleepCore",
        "asleepCore",
        "asleepDeep",
    ]
    # Same aggregates either way — the flag controls payload size, not content.
    assert night["stage_vitals"]["asleepCore"]["hr_mean"] == 58


def test_sleep_detail_reports_malformed_samples_instead_of_swallowing(tmp_path: Path) -> None:
    """The two `except: pass` loops dropped bad samples with no trace.

    Every other decode path in this module collects failures into `errors`.
    """
    service, cloud, public_key = _bound_service(tmp_path)
    payload = _sleep_payload(
        "2026-07-25T16:00:00Z",
        "2026-07-25T15:00:00Z",
        "2026-07-25T22:00:00Z",
        [
            {
                "stage": "asleepCore",
                "startDate": "2026-07-25T15:00:00Z",
                "endDate": "2026-07-25T19:00:00Z",
            },
            {"stage": "asleepDeep", "startDate": "2026-07-25T19:00:00Z"},  # no endDate
        ],
    )
    cloud.envelopes = [
        _make_envelope(
            public_key,
            payload,
            metric_type="sleep",
            envelope_id="env-sleep-bad",
            blob_id="blob-sleep-bad",
            owner_user_id=_TEST_OWNER,
        )
    ]

    summary = asyncio.run(service.sleep_detail_records(limit=5, owner="dce9"))

    assert len(summary["errors"]) == 1
    assert "1 sleep sample(s) skipped" in summary["errors"][0]
    assert "env-sleep-bad" in summary["errors"][0]
    assert summary["errors_note"]
    # The good sample still decodes — one bad sample must not lose the night.
    assert summary["nights"][0]["total_sleep_minutes"] == 240


# ── Whole-day writers must not delete what the caller never mentioned ─────────
# 2026-07-27 lost the note '腿日 + 腹肌' off a real session: log_strength_entry
# rewrote the day without it, because omitting `note` meant "erase" rather than
# "leave alone". strength additionally had no merge mode at all, so a second
# call in the same session silently dropped the exercises of the first.


def test_log_strength_entry_merge_appends_without_dropping(tmp_path: Path) -> None:
    service, _cloud, _public_key = _bound_service_with_owner_identity(tmp_path)

    first = asyncio.run(
        service.log_strength_entry(
            date="2026-07-20",
            exercises=[
                {"name": "腿弯举", "sets": [{"weightKg": 22.5, "reps": 15}]},
                {"name": "腿伸展", "sets": [{"weightKg": 30.0, "reps": 12}]},
            ],
            note="腿日 + 腹肌",
        )
    )
    second = asyncio.run(
        service.log_strength_entry(
            date="2026-07-20",
            exercises=[
                {"name": "腿弯举", "sets": [{"weightKg": 25.0, "reps": 12}]},
                {"name": "小腿提踵", "sets": [{"weightKg": 40.0, "reps": 20}]},
            ],
            merge=True,
        )
    )

    assert second["entry_id"] == first["entry_id"]
    assert second["merge_mode"] is True
    assert second["replaced_exercises"] == []

    by_name = {e["name"]: e for e in second["session"]["exercises"]}
    # Same-name exercise got the new set appended, not swapped.
    assert [s["weightKg"] for s in by_name["腿弯举"]["sets"]] == [22.5, 25.0]
    # An exercise the second call never mentioned survives untouched.
    assert [s["weightKg"] for s in by_name["腿伸展"]["sets"]] == [30.0]
    assert [s["weightKg"] for s in by_name["小腿提踵"]["sets"]] == [40.0]


def test_log_strength_entry_omitting_note_keeps_it(tmp_path: Path) -> None:
    """The exact 2026-07-27 loss: a follow-up write with no `note` erased it."""
    service, _cloud, _public_key = _bound_service_with_owner_identity(tmp_path)

    asyncio.run(
        service.log_strength_entry(
            date="2026-07-20",
            exercises=[{"name": "卧推", "sets": [{"weightKg": 40.0, "reps": 8}]}],
            note="腿日 + 腹肌",
        )
    )
    second = asyncio.run(
        service.log_strength_entry(
            date="2026-07-20",
            exercises=[{"name": "卧推", "sets": [{"weightKg": 42.5, "reps": 8}]}],
        )
    )

    assert second["session"]["note"] == "腿日 + 腹肌"


def test_log_strength_entry_empty_note_clears_it(tmp_path: Path) -> None:
    """`None` means leave alone, so there must still be a way to erase."""
    service, _cloud, _public_key = _bound_service_with_owner_identity(tmp_path)

    asyncio.run(
        service.log_strength_entry(
            date="2026-07-20",
            exercises=[{"name": "卧推", "sets": [{"weightKg": 40.0, "reps": 8}]}],
            note="写错了",
        )
    )
    second = asyncio.run(
        service.log_strength_entry(
            date="2026-07-20",
            exercises=[{"name": "卧推", "sets": [{"weightKg": 40.0, "reps": 8}]}],
            note="",
        )
    )

    assert second["session"]["note"] is None


def test_log_strength_entry_replace_mode_names_what_it_deleted(tmp_path: Path) -> None:
    """Replace still replaces — but the caller is told, instead of guessing."""
    service, _cloud, _public_key = _bound_service_with_owner_identity(tmp_path)

    asyncio.run(
        service.log_strength_entry(
            date="2026-07-20",
            exercises=[
                {"name": "腿弯举", "sets": [{"weightKg": 22.5, "reps": 15}]},
                {"name": "腿伸展", "sets": [{"weightKg": 30.0, "reps": 12}]},
            ],
        )
    )
    second = asyncio.run(
        service.log_strength_entry(
            date="2026-07-20",
            exercises=[{"name": "小腿提踵", "sets": [{"weightKg": 40.0, "reps": 20}]}],
        )
    )

    assert second["merge_mode"] is False
    assert sorted(second["replaced_exercises"]) == ["腿伸展", "腿弯举"]
    assert [e["name"] for e in second["session"]["exercises"]] == ["小腿提踵"]


def test_log_strength_entry_fresh_day_reports_nothing_replaced(tmp_path: Path) -> None:
    service, _cloud, _public_key = _bound_service_with_owner_identity(tmp_path)

    result = asyncio.run(
        service.log_strength_entry(
            date="2026-07-20",
            exercises=[{"name": "卧推", "sets": [{"weightKg": 40.0, "reps": 8}]}],
        )
    )

    assert result["replaced_exercises"] == []
    assert result["updated_existing_day"] is False


def test_log_food_entry_omitting_note_keeps_it_in_replace_mode(tmp_path: Path) -> None:
    """Note preservation used to be merge-only; replace-mode calls erased it."""
    service, _cloud, _public_key = _bound_service_with_owner_identity(tmp_path)

    asyncio.run(
        service.log_food_entry(
            date="2026-07-20",
            meals=[{"name": "lunch", "items": [{"food": "鸡胸肉"}]}],
            note="减脂日",
        )
    )
    second = asyncio.run(
        service.log_food_entry(
            date="2026-07-20",
            meals=[{"name": "dinner", "items": [{"food": "牛腱"}]}],
        )
    )

    assert second["merge_mode"] is False
    assert second["day"]["note"] == "减脂日"
    assert second["replaced_meals"] == ["lunch"]


# ── log_note was the last whole-day writer with no merge ──────────────────────
# 2026-07-28: food and strength got merge + replaced_* receipts on 07-28, but
# log_note still silently overwrote its (kind, day) note and pushed
# read-modify-write onto the caller via its docstring. Notes carry symptoms,
# which arrive in installments across a day, so second-write-of-the-day is the
# normal case — and an agent that had just learned `merge=True` elsewhere would
# reasonably assume it existed here too.


def test_log_note_merge_appends_without_dropping(tmp_path: Path) -> None:
    service, _cloud, _public_key = _bound_service_with_owner_identity(tmp_path)

    first = asyncio.run(
        service.log_note(text="中午恶心", kind="general", date="2026-07-24")
    )
    second = asyncio.run(
        service.log_note(text="晚上头晕", kind="general", date="2026-07-24", merge=True)
    )

    assert second["note_id"] == first["note_id"]
    assert second["merge_mode"] is True
    assert second["replaced_text"] is None
    # Morning symptom survives; evening one is appended on its own line.
    assert second["note"]["text"] == "中午恶心\n晚上头晕"


def test_log_note_replace_mode_reports_what_it_deleted(tmp_path: Path) -> None:
    """Replace still replaces — but the caller is told, instead of guessing."""
    service, _cloud, _public_key = _bound_service_with_owner_identity(tmp_path)

    asyncio.run(service.log_note(text="中午恶心", kind="general", date="2026-07-24"))
    second = asyncio.run(service.log_note(text="晚上头晕", kind="general", date="2026-07-24"))

    assert second["merge_mode"] is False
    assert second["replaced_text"] == "中午恶心"
    assert second["note"]["text"] == "晚上头晕"


def test_log_note_fresh_day_reports_nothing_replaced(tmp_path: Path) -> None:
    service, _cloud, _public_key = _bound_service_with_owner_identity(tmp_path)

    result = asyncio.run(service.log_note(text="第一条", kind="mood", date="2026-07-24"))

    assert result["replaced_text"] is None
    assert result["updated_existing_note"] is False

    # merge=True on a day with no note yet is a plain create, not a leading
    # newline glued onto empty text.
    other = asyncio.run(
        service.log_note(text="第一条", kind="general", date="2026-07-24", merge=True)
    )
    assert other["note"]["text"] == "第一条"
    assert other["replaced_text"] is None


def test_log_note_merge_is_scoped_to_same_kind_and_day(tmp_path: Path) -> None:
    """Merging must not vacuum up a different kind's or day's note."""
    service, _cloud, _public_key = _bound_service_with_owner_identity(tmp_path)

    asyncio.run(service.log_note(text="mood 的", kind="mood", date="2026-07-24"))
    asyncio.run(service.log_note(text="前一天的", kind="general", date="2026-07-23"))
    merged = asyncio.run(
        service.log_note(text="今天的", kind="general", date="2026-07-24", merge=True)
    )

    assert merged["note"]["text"] == "今天的"
    assert merged["replaced_text"] is None


# ---------------------------------------------------------------------------
# poll_until_bound — the link blips, and a blip must not cost the user their QR
# ---------------------------------------------------------------------------


class FlakyPollClient(FakeCloudClient):
    """A cloud client whose polls fail on a scripted schedule.

    `failures` is consumed one entry per poll: an exception is raised, None
    means answer normally. `bind_after` polls (successful or not) the binding
    completes.
    """

    def __init__(self, failures: list[BaseException | None], *, bind_after: int = 10_000) -> None:
        super().__init__()
        self.failures = list(failures)
        self.bind_after = bind_after
        self.attempts = 0

    async def poll_binding(self, poll_id: str) -> PollBindingResult:
        self.poll_id = poll_id
        self.attempts += 1
        if self.failures:
            failure = self.failures.pop(0)
            if failure is not None:
                raise failure
        if self.attempts >= self.bind_after:
            return PollBindingResult(
                status="bound",
                server_id=self.bound_server_id,
                server_token=self.bound_server_token,
            )
        return PollBindingResult(status="pending")


def test_poll_until_bound_survives_a_transient_failure(tmp_path: Path) -> None:
    """A single blip used to end the whole bind.

    The exception went straight up, the user re-ran `bind`, and that minted a
    NEW pollID — invalidating the QR they had just scanned. On a loop that runs
    5-10 minutes at one request every 7s over a link measured at ~97.3%, at
    least one blip is close to certain, so the common path was failing for a
    reason unrelated to binding.
    """

    import httpx

    cloud = FlakyPollClient([httpx.ReadTimeout("blip"), None], bind_after=2)
    service = VaultbeatLocalService(ConfigStore(tmp_path / "config.json"), cloud)
    service.start_binding(api_base_url="https://api.test")

    result = asyncio.run(service.poll_until_bound(timeout_sec=60, interval_sec=0))

    assert result.status == "bound"
    assert cloud.attempts == 2


def test_poll_until_bound_gives_up_on_a_sustained_outage(tmp_path: Path) -> None:
    """Tolerating blips must not become waiting out the full timeout on a dead
    uplink — the user would watch a spinner for five minutes to be told nothing."""

    import httpx

    limit = VaultbeatLocalService.POLL_CONSECUTIVE_FAILURE_LIMIT
    cloud = FlakyPollClient([httpx.ConnectError("down")] * (limit + 3))
    service = VaultbeatLocalService(ConfigStore(tmp_path / "config.json"), cloud)
    service.start_binding(api_base_url="https://api.test")

    with pytest.raises(httpx.ConnectError):
        asyncio.run(service.poll_until_bound(timeout_sec=600, interval_sec=0))

    assert cloud.attempts == limit


def test_poll_until_bound_counts_failures_consecutively_not_cumulatively(tmp_path: Path) -> None:
    """Blips spread across a long poll are the NORMAL case at 2-3% loss; only an
    unbroken run of them means the link is actually gone."""

    import httpx

    limit = VaultbeatLocalService.POLL_CONSECUTIVE_FAILURE_LIMIT
    schedule: list[BaseException | None] = []
    for _ in range(limit + 2):
        schedule.extend([httpx.ReadTimeout("blip"), None])
    cloud = FlakyPollClient(schedule, bind_after=len(schedule))
    service = VaultbeatLocalService(ConfigStore(tmp_path / "config.json"), cloud)
    service.start_binding(api_base_url="https://api.test")

    result = asyncio.run(service.poll_until_bound(timeout_sec=600, interval_sec=0))

    assert result.status == "bound"


def test_poll_until_bound_never_reports_pending_when_every_poll_failed(tmp_path: Path) -> None:
    """"Still waiting for a scan" and "this machine never reached the server"
    are opposite diagnoses, and the second must not be reported as the first."""

    import httpx

    cloud = FlakyPollClient([httpx.ReadTimeout("blip")])
    service = VaultbeatLocalService(ConfigStore(tmp_path / "config.json"), cloud)
    service.start_binding(api_base_url="https://api.test")

    with pytest.raises(httpx.ReadTimeout):
        asyncio.run(service.poll_until_bound(timeout_sec=0, interval_sec=0))
