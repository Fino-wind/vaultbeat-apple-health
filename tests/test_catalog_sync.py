"""Catalog-mode sync: fetch only what changed.

Why this file exists — the failure it guards is SILENT DATA LOSS, not a slow
read. A full fetch is self-correcting: whatever the server has, the client gets.
The moment the client starts deciding what it "already has", a wrong decision
means a record that exists in the cloud never reaches the agent, with no error
anywhere. `PM-2026-07-25-strength-food-invisible` is the same shape and it hid
for days.

Context: reading one kind measured 12,108,143 bytes against production on
2026-07-29; the digest that decides whether that fetch is needed measured 119.
Three users had burned 21.877 GB of a 5 GB monthly allowance re-downloading
history that had not changed.

Note that every OTHER test module drives `FakeCloudClient`, which does NOT speak
catalog mode — so those 208 tests all exercise the legacy full-fetch path and
collectively assert the backward-compatibility requirement. This file is the
only place the new path runs.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

# `from test_service import`, NOT `from tests.test_service import`.
#
# tests/ has no __init__.py, so it is not a package; pytest makes sibling
# modules importable by inserting THIS directory into sys.path. The dotted form
# additionally needs the project root on sys.path, which is true under
# `python -m pytest` (it adds the cwd) and false under `uv run pytest` — the
# command CI actually uses. So the dotted import passed locally 218/218 and
# failed every CI run from the first push, for four commits, until the owner
# noticed the GitHub email. Verify test changes with `uv run --frozen pytest`.
from test_service import (  # type: ignore[import-not-found]
    FakeCloudClient,
    _bound_service,
    _make_envelope,
)


class CatalogCloudClient(FakeCloudClient):
    """A fake edge deployment that DOES speak catalog mode.

    Row versions live in `xmins` (blob_id → xmin), mirroring Postgres's per-row
    transaction id. Bump one to simulate an in-place edit; drop a row to
    simulate a delete.
    """

    def __init__(self) -> None:
        super().__init__()
        self.xmins: dict[str, str] = {}
        self.digest_calls: list[str | None] = []
        self.catalog_calls: list[str | None] = []
        self.blob_fetches: list[list[str]] = []
        # Set True to emulate an edge deployed BEFORE catalog mode: it ignores
        # `fields` and answers with the whole payload.
        self.pretend_legacy_edge = False

    MAX_BLOB_IDS_PER_REQUEST = 500

    def _visible(self, metric_type: str | None) -> list[dict[str, Any]]:
        if metric_type is None:
            return list(self.envelopes)

        def _matches(row: dict[str, Any]) -> bool:
            blob = row.get("encrypted_sleep_blobs") or {}
            return (blob.get("metric_type") or "sleep") == metric_type

        return [row for row in self.envelopes if _matches(row)]

    def _xmin_for(self, row: dict[str, Any]) -> str:
        return self.xmins.get(str(row.get("blob_id", "")), "1")

    async def sync_digest(
        self, server_token: str, *, metric_type: str | None = None
    ) -> tuple[dict[str, Any] | None, list[dict[str, Any]] | None]:
        self.digest_calls.append(metric_type)
        if self.pretend_legacy_edge:
            return None, self._visible(metric_type)
        rows = self._visible(metric_type)
        values = [int(self._xmin_for(r)) for r in rows]
        return (
            {
                "count": len(rows),
                "max_xmin": str(max(values) if values else 0),
                "sum_xmin": str(sum(values)),
            },
            None,
        )

    async def sync_catalog(
        self, server_token: str, *, metric_type: str | None = None
    ) -> list[dict[str, Any]] | None:
        self.catalog_calls.append(metric_type)
        return [
            {"blob_id": str(r.get("blob_id", "")), "xmin": self._xmin_for(r)}
            for r in self._visible(metric_type)
        ]

    async def sync_blobs(
        self, server_token: str, *, blob_ids: list[str], metric_type: str | None = None
    ) -> list[dict[str, Any]]:
        self.blob_fetches.append(list(blob_ids))
        wanted = set(blob_ids)
        return [r for r in self._visible(metric_type) if str(r.get("blob_id", "")) in wanted]


def _sleep_payload(day: str) -> bytes:
    return json.dumps(
        {
            "sessionID": f"s-{day}",
            "sessionDate": f"{day}T00:00:00Z",
            "bedtime": f"{day}T23:00:00Z",
            "wakeTime": f"{day}T07:00:00Z",
            "totalSleepMinutes": 480,
            "stages": [],
        }
    ).encode()


def _catalog_service(tmp_path: Path) -> tuple[Any, CatalogCloudClient, str]:
    """A bound service wired to the catalog-speaking fake."""
    service, _, public_key = _bound_service(tmp_path)
    cloud = CatalogCloudClient()
    # Swap the transport after binding: the bind handshake needs the plain fake,
    # and `_client()` reads this attribute on every call, so the catalog-aware
    # one takes over from here.
    service._cloud_client = cloud  # type: ignore[attr-defined]
    return service, cloud, public_key


def _seed(cloud: CatalogCloudClient, public_key: str, days: list[str]) -> None:
    cloud.envelopes = [
        _make_envelope(
            public_key,
            _sleep_payload(day),
            metric_type="sleep",
            envelope_id=f"env-{day}",
            blob_id=f"blob-{day}",
        )
        for day in days
    ]
    cloud.xmins = {f"blob-{day}": str(100 + i) for i, day in enumerate(days)}


def test_unchanged_library_transfers_no_blobs_at_all(tmp_path: Path) -> None:
    """The whole point: a second read of unchanged data fetches zero ciphertext.

    `fresh=True` deliberately, to prove --fresh no longer means "re-download
    everything" — the digest re-verifies against the server, which is strictly
    stronger than a TTL, so honouring it costs 119 bytes instead of 12 MB.
    """
    service, cloud, public_key = _catalog_service(tmp_path)
    _seed(cloud, public_key, ["2026-07-20", "2026-07-21", "2026-07-22"])

    first, _ = asyncio.run(service.sync_decrypted_records(metric_type="sleep", fresh=True))
    assert len(first) == 3

    cloud.sync_calls.clear()
    cloud.blob_fetches.clear()

    second, _ = asyncio.run(service.sync_decrypted_records(metric_type="sleep", fresh=True))

    assert [r.blob_id for r in second] == [r.blob_id for r in first]
    assert cloud.blob_fetches == [], "unchanged data must not be re-fetched"
    assert cloud.sync_calls == [], "must not fall back to a full sync"


def test_only_the_changed_row_is_refetched(tmp_path: Path) -> None:
    service, cloud, public_key = _catalog_service(tmp_path)
    _seed(cloud, public_key, ["2026-07-20", "2026-07-21", "2026-07-22"])
    asyncio.run(service.sync_decrypted_records(metric_type="sleep", fresh=True))

    cloud.blob_fetches.clear()
    cloud.sync_calls.clear()
    # An in-place edit: same blob id, new row version.
    cloud.xmins["blob-2026-07-21"] = "999"

    records, _ = asyncio.run(service.sync_decrypted_records(metric_type="sleep", fresh=True))

    assert cloud.blob_fetches == [["blob-2026-07-21"]]
    assert cloud.sync_calls == []
    assert len(records) == 3, "the two unchanged rows must survive the merge"
    assert sorted(r.blob_id for r in records) == [
        "blob-2026-07-20",
        "blob-2026-07-21",
        "blob-2026-07-22",
    ]


def test_row_deleted_server_side_disappears_locally(tmp_path: Path) -> None:
    """A row absent from the catalog is gone — the client must not keep serving
    it from local plaintext forever."""
    service, cloud, public_key = _catalog_service(tmp_path)
    _seed(cloud, public_key, ["2026-07-20", "2026-07-21", "2026-07-22"])
    asyncio.run(service.sync_decrypted_records(metric_type="sleep", fresh=True))

    cloud.envelopes = [
        r for r in cloud.envelopes if str(r.get("blob_id")) != "blob-2026-07-21"
    ]
    cloud.xmins.pop("blob-2026-07-21")

    records, _ = asyncio.run(service.sync_decrypted_records(metric_type="sleep", fresh=True))

    assert sorted(r.blob_id for r in records) == ["blob-2026-07-20", "blob-2026-07-22"]


def test_new_row_is_added_without_refetching_the_rest(tmp_path: Path) -> None:
    service, cloud, public_key = _catalog_service(tmp_path)
    _seed(cloud, public_key, ["2026-07-20", "2026-07-21"])
    asyncio.run(service.sync_decrypted_records(metric_type="sleep", fresh=True))

    cloud.blob_fetches.clear()
    cloud.envelopes.append(
        _make_envelope(
            public_key,
            _sleep_payload("2026-07-23"),
            metric_type="sleep",
            envelope_id="env-2026-07-23",
            blob_id="blob-2026-07-23",
        )
    )
    cloud.xmins["blob-2026-07-23"] = "500"

    records, _ = asyncio.run(service.sync_decrypted_records(metric_type="sleep", fresh=True))

    assert cloud.blob_fetches == [["blob-2026-07-23"]]
    assert len(records) == 3


def test_catalog_path_returns_exactly_what_a_full_fetch_would(tmp_path: Path) -> None:
    """THE equivalence check the design doc calls for.

    Two services over identical data: one that can only full-fetch, one driven
    through catalog + diff + merge. Every decrypted field must match. If the
    diff logic ever drops or mangles a row, this is what catches it — the
    per-behaviour tests above would still pass while the DATA silently differed.
    """
    days = ["2026-07-18", "2026-07-19", "2026-07-20", "2026-07-21"]

    legacy_service, legacy_cloud, legacy_key = _bound_service(tmp_path / "legacy")
    legacy_cloud.envelopes = [
        _make_envelope(
            legacy_key,
            _sleep_payload(day),
            metric_type="sleep",
            envelope_id=f"env-{day}",
            blob_id=f"blob-{day}",
        )
        for day in days
    ]
    expected, expected_errors = asyncio.run(
        legacy_service.sync_decrypted_records(metric_type="sleep", fresh=True)
    )

    service, cloud, public_key = _catalog_service(tmp_path / "catalog")
    _seed(cloud, public_key, days)
    # Prime, then force the diff path by editing one row and adding another —
    # so the final result is assembled from BOTH reused and refetched rows.
    asyncio.run(service.sync_decrypted_records(metric_type="sleep", fresh=True))
    cloud.xmins["blob-2026-07-19"] = "777"
    actual, actual_errors = asyncio.run(
        service.sync_decrypted_records(metric_type="sleep", fresh=True)
    )

    assert actual_errors == expected_errors
    by_blob_expected = {r.blob_id: r.payload for r in expected}
    by_blob_actual = {r.blob_id: r.payload for r in actual}
    assert by_blob_actual == by_blob_expected


def test_edge_without_catalog_mode_still_works(tmp_path: Path) -> None:
    """An older deployment ignores `fields` and returns the full payload. That
    response IS the data — the client must use it rather than pay twice."""
    service, cloud, public_key = _catalog_service(tmp_path)
    _seed(cloud, public_key, ["2026-07-20", "2026-07-21"])
    asyncio.run(service.sync_decrypted_records(metric_type="sleep", fresh=True))

    cloud.pretend_legacy_edge = True
    cloud.sync_calls.clear()
    cloud.blob_fetches.clear()

    records, _ = asyncio.run(service.sync_decrypted_records(metric_type="sleep", fresh=True))

    assert len(records) == 2
    assert cloud.blob_fetches == []
    assert cloud.sync_calls == [], "the legacy response was already the full data"


def test_local_digest_matches_the_servers_arithmetic(tmp_path: Path) -> None:
    """The client recomputes the digest from a catalog response after a full
    fetch. If that arithmetic drifts from mcp-sync's, every later comparison
    fails and the client silently degrades to full fetches forever — expensive,
    never wrong, and invisible without this assertion."""
    from vaultbeat_mcp_local.service import VaultbeatLocalService

    service, cloud, public_key = _catalog_service(tmp_path)
    _seed(cloud, public_key, ["2026-07-20", "2026-07-21", "2026-07-22"])

    server_digest, _ = asyncio.run(cloud.sync_digest("t", metric_type="sleep"))
    catalog = asyncio.run(cloud.sync_catalog("t", metric_type="sleep"))
    assert catalog is not None
    local = VaultbeatLocalService._digest_from_catalog(
        {str(r["blob_id"]): str(r["xmin"]) for r in catalog}
    )

    assert local == server_digest


# ── doctor: client version freshness ─────────────────────────────────────────
#
# Why this lives here rather than in a server-side test: the upgrade prompt is
# generated ENTIRELY on the client. PyPI hands over one version string; every
# word the user's agent reads is hardcoded in service.py. The rejected design
# was a server-supplied `notice` string — that would write arbitrary text into
# the agent's context, and this server exposes write tools while the agent
# usually has filesystem/shell MCPs attached too. These tests pin the safe
# shape: a comparison of two version numbers, nothing more.


def _doctor_version_check(service: object) -> dict:
    report = asyncio.run(service.doctor())  # type: ignore[attr-defined]
    return next(c for c in report["checks"] if c["name"] == "client_version")


def test_doctor_flags_an_outdated_client(tmp_path: Path, monkeypatch) -> None:
    service, _, _ = _catalog_service(tmp_path)
    monkeypatch.setenv("VAULTBEAT_MCP_FAKE_LATEST", "999.0.0")

    check = _doctor_version_check(service)

    assert check["ok"] is False
    assert "999.0.0 available" in check["detail"]
    # The remedy must be in the hint — a user told "you are behind" with no
    # command to run learns nothing actionable.
    assert "uvx --refresh vaultbeat-apple-health" in check["hint"]


def test_doctor_passes_when_client_is_current(tmp_path: Path, monkeypatch) -> None:
    from vaultbeat_mcp_local import __version__ as installed

    service, _, _ = _catalog_service(tmp_path)
    monkeypatch.setenv("VAULTBEAT_MCP_FAKE_LATEST", installed)

    check = _doctor_version_check(service)

    assert check["ok"] is True
    assert "latest" in check["detail"]


def test_doctor_does_not_fail_when_pypi_is_unreachable(tmp_path: Path, monkeypatch) -> None:
    """An offline machine still has a working install. A diagnostic that cries
    wolf about the network is one people learn to ignore."""
    service, _, _ = _catalog_service(tmp_path)
    monkeypatch.setenv("VAULTBEAT_MCP_FAKE_LATEST", "")

    check = _doctor_version_check(service)

    assert check["ok"] is True
    assert "could not reach PyPI" in check["detail"]
