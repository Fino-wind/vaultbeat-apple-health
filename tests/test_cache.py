"""LocalRecordCache + service cache-wiring tests.

The cache is the AGENTS.md "MCP Local Server Performance" fix: within the TTL a
repeat query must answer from local plaintext with ZERO cloud round trips, and
`fresh=True` must force one. Reuses `FakeCloudClient` / `_make_envelope` from
test_service so the envelopes are real E2EE ciphertext, not stubs.
"""

from __future__ import annotations

import asyncio
import json
import stat
from pathlib import Path

from vaultbeat_mcp_local.cache import LocalRecordCache
from vaultbeat_mcp_local.service import VaultbeatLocalService
from vaultbeat_mcp_local.store import ConfigStore

from test_service import FakeCloudClient, _make_envelope, _water_payload


def _bound(tmp_path: Path, **cache_kwargs) -> tuple[VaultbeatLocalService, FakeCloudClient, str]:
    config_path = tmp_path / "config.json"
    cloud = FakeCloudClient()
    cache = LocalRecordCache(tmp_path / "cache", **cache_kwargs)
    service = VaultbeatLocalService(ConfigStore(config_path), cloud, cache=cache)
    service.start_binding(server_name="Mac Studio", api_base_url="https://api.test")
    asyncio.run(service.poll_once())
    public_key = ConfigStore(config_path).require_bound().public_key_base64
    return service, cloud, public_key


def test_second_query_within_ttl_is_zero_network(tmp_path: Path) -> None:
    service, cloud, public_key = _bound(tmp_path, ttl_seconds=3600)
    cloud.envelopes = [_make_envelope(public_key, b'{"stage":"asleep"}')]

    first, errors_first = asyncio.run(service.sync_decrypted_records())
    second, errors_second = asyncio.run(service.sync_decrypted_records())

    assert errors_first == errors_second == []
    assert len(cloud.sync_calls) == 1  # the second answer came from disk
    assert [r.to_dict() for r in second] == [r.to_dict() for r in first]


def test_fresh_forces_cloud_round_trip(tmp_path: Path) -> None:
    service, cloud, public_key = _bound(tmp_path, ttl_seconds=3600)
    cloud.envelopes = [_make_envelope(public_key, b'{"stage":"asleep"}')]

    asyncio.run(service.sync_decrypted_records())
    cloud.envelopes = [
        _make_envelope(public_key, b'{"stage":"awake"}', envelope_id="env-2", blob_id="blob-2")
    ]
    records, _ = asyncio.run(service.sync_decrypted_records(fresh=True))

    assert len(cloud.sync_calls) == 2
    assert records[0].payload == {"stage": "awake"}


def test_metric_keys_are_isolated_and_narrow_server_side(tmp_path: Path) -> None:
    service, cloud, public_key = _bound(tmp_path, ttl_seconds=3600)
    cloud.envelopes = [
        _make_envelope(public_key, b'{"stage":"asleep"}', envelope_id="env-s", blob_id="blob-s"),
        _make_envelope(
            public_key,
            _water_payload("water-1", "2026-06-05T00:00:00Z", 6.0, refill_count=2),
            metric_type="water",
            envelope_id="env-w",
            blob_id="blob-w",
        ),
    ]

    water = asyncio.run(service.water_intake_summary())
    sleep = asyncio.run(service.sleep_records())
    water_again = asyncio.run(service.water_intake_summary())

    assert water["day_count"] == 1
    assert sleep["count"] == 1
    assert water_again["day_count"] == 1
    # water(miss) + wrist? no — water & sleep each miss once, the repeat hits.
    assert cloud.sync_calls == ["water", "sleep"]


def test_old_edge_ignoring_metric_type_still_filters_locally(tmp_path: Path) -> None:
    service, cloud, public_key = _bound(tmp_path, ttl_seconds=3600)
    cloud.ignores_metric_type = True
    cloud.envelopes = [
        _make_envelope(public_key, b'{"stage":"asleep"}', envelope_id="env-s", blob_id="blob-s"),
        _make_envelope(
            public_key,
            _water_payload("water-1", "2026-06-05T00:00:00Z", 6.0, refill_count=2),
            metric_type="water",
            envelope_id="env-w",
            blob_id="blob-w",
        ),
    ]

    summary = asyncio.run(service.water_intake_summary())

    assert summary["errors"] == []
    assert summary["day_count"] == 1  # sleep row dropped by the defensive filter


def test_limit_never_truncates_the_cached_set(tmp_path: Path) -> None:
    service, cloud, public_key = _bound(tmp_path, ttl_seconds=3600)
    cloud.envelopes = [
        _make_envelope(public_key, b'{"n":1}', envelope_id="env-1", blob_id="blob-1"),
        _make_envelope(public_key, b'{"n":2}', envelope_id="env-2", blob_id="blob-2"),
    ]

    limited, _ = asyncio.run(service.sync_decrypted_records(limit=1))
    full_from_cache, _ = asyncio.run(service.sync_decrypted_records())

    assert len(limited) == 1
    assert len(full_from_cache) == 2
    assert len(cloud.sync_calls) == 1


def test_expired_ttl_refetches(tmp_path: Path) -> None:
    service, cloud, public_key = _bound(tmp_path, ttl_seconds=3600)
    cloud.envelopes = [_make_envelope(public_key, b'{"stage":"asleep"}')]
    asyncio.run(service.sync_decrypted_records())

    cache_file = tmp_path / "cache" / "records-all.json"
    raw = json.loads(cache_file.read_text())
    raw["fetched_at"] = raw["fetched_at"] - 7200
    cache_file.write_text(json.dumps(raw))

    asyncio.run(service.sync_decrypted_records())
    assert len(cloud.sync_calls) == 2


def test_other_servers_cache_is_a_miss(tmp_path: Path) -> None:
    service, cloud, public_key = _bound(tmp_path, ttl_seconds=3600)
    cloud.envelopes = [_make_envelope(public_key, b'{"stage":"asleep"}')]
    asyncio.run(service.sync_decrypted_records())

    cache_file = tmp_path / "cache" / "records-all.json"
    raw = json.loads(cache_file.read_text())
    raw["server_id"] = "someone-else"
    cache_file.write_text(json.dumps(raw))

    asyncio.run(service.sync_decrypted_records())
    assert len(cloud.sync_calls) == 2


def test_ttl_zero_disables_cache_entirely(tmp_path: Path) -> None:
    service, cloud, public_key = _bound(tmp_path, ttl_seconds=0)
    cloud.envelopes = [_make_envelope(public_key, b'{"stage":"asleep"}')]

    asyncio.run(service.sync_decrypted_records())
    asyncio.run(service.sync_decrypted_records())

    assert len(cloud.sync_calls) == 2
    assert not (tmp_path / "cache").exists()  # disabled cache writes nothing


def test_rebinding_clears_cached_plaintext(tmp_path: Path) -> None:
    service, cloud, public_key = _bound(tmp_path, ttl_seconds=3600)
    cloud.envelopes = [_make_envelope(public_key, b'{"stage":"asleep"}')]
    asyncio.run(service.sync_decrypted_records())
    assert list((tmp_path / "cache").glob("records-*.json"))

    service.start_binding(server_name="Mac Studio", api_base_url="https://api.test")

    assert not list((tmp_path / "cache").glob("records-*.json"))


def test_cache_files_are_owner_only(tmp_path: Path) -> None:
    service, cloud, public_key = _bound(tmp_path, ttl_seconds=3600)
    cloud.envelopes = [_make_envelope(public_key, b'{"stage":"asleep"}')]
    asyncio.run(service.sync_decrypted_records())

    cache_file = tmp_path / "cache" / "records-all.json"
    assert stat.S_IMODE(cache_file.stat().st_mode) == 0o600
    assert stat.S_IMODE(cache_file.parent.stat().st_mode) == 0o700


def test_cache_hit_replays_decrypt_errors(tmp_path: Path) -> None:
    """A poisoned envelope's error (and the CLI's exit-3 signal) must not
    vanish on cache warmth — the cache stores and replays the fetch's errors."""

    service, cloud, public_key = _bound(tmp_path, ttl_seconds=3600)
    good = _make_envelope(public_key, b'{"stage":"asleep"}', envelope_id="env-good", blob_id="b1")
    poisoned = _make_envelope(
        public_key, b'{"stage":"asleep"}', envelope_id="env-bad", blob_id="b2"
    )
    poisoned["encrypted_data_key"] = "AAAA"  # undecryptable envelope
    cloud.envelopes = [good, poisoned]

    first_records, first_errors = asyncio.run(service.sync_decrypted_records())
    warm_records, warm_errors = asyncio.run(service.sync_decrypted_records())

    assert len(cloud.sync_calls) == 1
    assert [r.envelope_id for r in warm_records] == [r.envelope_id for r in first_records]
    assert first_errors and warm_errors == first_errors


def test_unknown_metric_type_is_rejected_before_any_io(tmp_path: Path) -> None:
    """Free-text metric_type must never reach the cache path (file name) or the
    edge (400): '../..' traversal and the 'all'-key collision both die here."""

    service, cloud, public_key = _bound(tmp_path, ttl_seconds=3600)
    cloud.envelopes = [_make_envelope(public_key, b'{"stage":"asleep"}')]

    for bad in ("all", "../../../../tmp/pwn", "resting-hr"):
        try:
            asyncio.run(service.sync_decrypted_records(metric_type=bad))
        except ValueError as error:
            assert "unknown metric_type" in str(error)
        else:  # pragma: no cover - fail loudly if the guard disappears
            raise AssertionError(f"metric_type {bad!r} was accepted")

    assert cloud.sync_calls == []  # nothing reached the network
    assert not list((tmp_path / "cache").glob("records-*"))  # nothing reached disk


def test_menstrual_summary_reuses_cache_across_internal_queries(tmp_path: Path) -> None:
    """The menstrual summary triggers menstrual + wrist_temp fetches; a repeat
    within the TTL must not touch the network at all (the old double full-sync
    per call was the single worst latency source)."""

    service, cloud, public_key = _bound(tmp_path, ttl_seconds=3600)
    cloud.envelopes = [
        _make_envelope(
            public_key,
            json.dumps(
                {
                    "dayID": "m-1",
                    "dayStartDate": "2026-06-01T00:00:00Z",
                    "samples": [
                        {
                            "startDate": "2026-06-01T00:00:00Z",
                            "endDate": "2026-06-01T00:00:00Z",
                            "flow": "medium",
                        }
                    ],
                }
            ).encode(),
            metric_type="menstrual",
            envelope_id="env-m",
            blob_id="blob-m",
            owner_user_id="f8350dfc-0000-0000-0000-000000000000",
        ),
    ]

    asyncio.run(service.menstrual_cycle_summary())
    first_round = list(cloud.sync_calls)
    asyncio.run(service.menstrual_cycle_summary())

    assert first_round == ["menstrual", "wrist_temp"]
    assert cloud.sync_calls == first_round  # second call fully cache-served
