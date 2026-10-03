"""LocalRecordCache + service cache-wiring tests.

The cache is the "MCP Local Server Performance" fix (docs/sync-egress-redesign.md): within the TTL a
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
    sleep = asyncio.run(service.sleep_detail_records())
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


def test_landing_on_a_different_identity_clears_cached_plaintext(tmp_path: Path) -> None:
    """Cached plaintext belongs to one server identity. Binding onto a
    DIFFERENT one leaves it unreadable-by-anyone health data on disk."""

    service, cloud, public_key = _bound(tmp_path, ttl_seconds=3600)
    cloud.envelopes = [_make_envelope(public_key, b'{"stage":"asleep"}')]
    asyncio.run(service.sync_decrypted_records())
    assert list((tmp_path / "cache").glob("records-*.json"))

    cloud.bound_server_id = "server-2"
    service.start_binding(server_name="Mac Studio", api_base_url="https://api.test")
    asyncio.run(service.poll_once())

    assert not list((tmp_path / "cache").glob("records-*.json"))


def test_starting_a_binding_nobody_completes_keeps_the_cache(tmp_path: Path) -> None:
    """The mirror of the test above, and the one that matters more: the cache
    must survive a session that never gets scanned. Clearing it at
    start_binding time was half of Invariant 54 (a) — the other half being the
    credentials themselves (see test_service)."""

    service, cloud, public_key = _bound(tmp_path, ttl_seconds=3600)
    cloud.envelopes = [_make_envelope(public_key, b'{"stage":"asleep"}')]
    asyncio.run(service.sync_decrypted_records())
    assert list((tmp_path / "cache").glob("records-*.json"))

    service.start_binding(server_name="Mac Studio", api_base_url="https://api.test")

    assert list((tmp_path / "cache").glob("records-*.json"))


def test_rebinding_to_the_same_identity_keeps_the_cache(tmp_path: Path) -> None:
    """Once the server-side upsert lands, re-binding resolves to the SAME row —
    same private key, same records. Dropping the cache there would make the
    common recovery path needlessly expensive."""

    service, cloud, public_key = _bound(tmp_path, ttl_seconds=3600)
    cloud.envelopes = [_make_envelope(public_key, b'{"stage":"asleep"}')]
    asyncio.run(service.sync_decrypted_records())
    assert list((tmp_path / "cache").glob("records-*.json"))

    service.start_binding(server_name="Mac Studio", api_base_url="https://api.test")
    asyncio.run(service.poll_once())

    assert list((tmp_path / "cache").glob("records-*.json"))


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
            owner_user_id="b2b2b2b2-0000-0000-0000-000000000000",
        ),
    ]

    asyncio.run(service.menstrual_cycle_summary())
    first_round = list(cloud.sync_calls)
    asyncio.run(service.menstrual_cycle_summary())

    assert first_round == ["menstrual", "wrist_temp"]
    assert cloud.sync_calls == first_round  # second call fully cache-served


def test_clear_also_removes_interrupted_write_temps(tmp_path: Path) -> None:
    """A temp holds decrypted plaintext too; rebinding must not leave it behind."""
    from vaultbeat_mcp_local.cache import LocalRecordCache

    cache = LocalRecordCache(tmp_path)
    (tmp_path / "records-sleep.json").write_text("{}")
    (tmp_path / "records-activity.json.tmp-3741-717ba21d").write_text("{}")
    cache.clear()
    assert list(tmp_path.iterdir()) == []


def test_abandoned_temps_are_swept_on_start_but_fresh_ones_are_not(tmp_path: Path) -> None:
    import os
    import time

    from vaultbeat_mcp_local.cache import LocalRecordCache

    old = tmp_path / "records-sleep.json.tmp-1-aaaaaaaa"
    fresh = tmp_path / "records-hrv.json.tmp-2-bbbbbbbb"
    kept = tmp_path / "records-sleep.json"
    for path in (old, fresh, kept):
        path.write_text("{}")
    an_hour_ago = time.time() - 3600
    os.utime(old, (an_hour_ago, an_hour_ago))

    LocalRecordCache(tmp_path)

    assert not old.exists(), "a temp older than any live write is abandoned"
    assert fresh.exists(), "a concurrent writer's in-flight temp must survive"
    assert kept.exists()


def test_the_newest_starting_fetch_wins_the_cache_across_processes(tmp_path: Path) -> None:
    """Review V5: two MCP processes share one pairing (two Claude Code sessions).
    A plain read that BEGAN before the other process's write and finished after
    it must not replace the cache that write's re-read left — while the writer's
    re-read must still beat an older fetch that merely finished later."""
    cache = LocalRecordCache(tmp_path, ttl_seconds=3600)
    path = tmp_path / "records-strength.json"

    cache.save([{"v": "written"}], server_id="s", metric_type="strength", started_at=2000.0)
    cache.save([{"v": "stale"}], server_id="s", metric_type="strength", started_at=1000.0)
    assert json.loads(path.read_text())["records"] == [{"v": "written"}], "a fetch begun earlier lost"

    cache.save([{"v": "newer"}], server_id="s", metric_type="strength", started_at=3000.0)
    assert json.loads(path.read_text())["records"] == [{"v": "newer"}]


def test_a_committed_write_expires_its_kind_but_keeps_the_rows(tmp_path: Path) -> None:
    """Review V6: a write whose re-read failed left the pre-write snapshot
    answering plain reads for the whole TTL. The kind now expires the moment
    the write returns — the TTL read misses — while the digest path keeps its
    rows, so the next read still fetches only what changed."""
    service, cloud, public_key = _bound(tmp_path, ttl_seconds=3600)
    cloud.envelopes = [_make_envelope(
        public_key, _water_payload("water-1784505600", "2026-07-20T00:00:00Z", 0.5, 2), metric_type="water",
    )]
    asyncio.run(service.sync_decrypted_records(metric_type="water"))
    server_id = service.store.require_bound().server_id or ""
    assert service.cache.load(server_id=server_id, metric_type="water") is not None

    async def accepted() -> dict[str, int]:
        return {"upserted_blobs": 1}

    generation_before = service._sync_generation.get("water", 0)
    service._inflight_syncs["water"] = "a download begun before the write"  # type: ignore[assignment]
    assert asyncio.run(service._committed("water", accepted())) == {"upserted_blobs": 1}
    assert "water" not in service._inflight_syncs, "no plain read may join a pre-write download"

    assert service.cache.load(server_id=server_id, metric_type="water") is None
    persisted = service.cache.load_persisted(server_id=server_id, metric_type="water")
    assert persisted is not None and len(persisted[0]) == 1
    assert service._sync_generation.get("water", 0) == generation_before + 1, \
        "a fetch begun before the write may no longer save"
