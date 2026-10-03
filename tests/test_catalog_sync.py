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

import pytest

from vaultbeat_mcp_local.client import VaultbeatCloudError

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

    `fresh=True` deliberately, to prove it no longer means "re-download
    everything" — the digest re-verifies against the server, which is strictly
    stronger than a TTL, so honouring it costs 119 bytes instead of 12 MB.
    (Named for the `--fresh` CLI flag, which 0.7.4 removed along with the data
    subcommands; the argument itself is on every read tool.)
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


def test_doctor_pypi_probe_names_the_distribution_in_pyproject() -> None:
    """`_PYPI_URL` must ask PyPI about the package this code actually ships as.

    The 2026-08-31 rename left it pointing at the frozen `vaultbeat-mcp`
    distribution (last release 0.6.1), so `_client_version_status` compared
    every install against a package that will never publish again and answered
    "installed (latest)" forever — the one command whose job is to say "you are
    behind" was structurally unable to (Invariant 67 ③/④, fixed in 0.6.3). Both
    MCP release guards read the name out of pyproject and stayed green through
    it; this ties the probe to that same source so the next rename cannot.
    """
    import tomllib

    from vaultbeat_mcp_local.service import VaultbeatLocalService

    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with pyproject.open("rb") as handle:
        distribution = tomllib.load(handle)["project"]["name"]

    assert VaultbeatLocalService._PYPI_URL == f"https://pypi.org/pypi/{distribution}/json"


# ── doctor on a cold machine: count by digest, decrypt a sample ──────────────
#
# GitHub #9 (2026-10-02): the doctor read sleep with `fresh=True` and then
# decrypted every kind just to count them — 51 MB and ~100 s on a cold machine,
# past the 60 s many MCP clients allow one call, on the command every guide
# names as the first stop. These pin the cost (what was fetched), not only the
# verdict, because the old code produced the right verdict too.


def _doctor_ready(tmp_path: Path) -> tuple[Any, CatalogCloudClient, str]:
    service, cloud, public_key = _catalog_service(tmp_path)
    service._probe_cloud = lambda url: (True, "cloud answered HTTP 401")
    return service, cloud, public_key


def _library(public_key: str, sizes: dict[str, int], *, foreign: dict[str, int] | None = None) -> list[dict[str, Any]]:
    """`sizes` records per kind; the first `foreign[kind]` of a kind are sealed
    to somebody else's key, so they reach this server and fail to decrypt."""
    import base64

    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import x25519

    stranger = base64.b64encode(
        x25519.X25519PrivateKey.generate().public_key().public_bytes(
            encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw
        )
    ).decode()
    rows = []
    for kind, size in sizes.items():
        for index in range(size):
            key = stranger if index < (foreign or {}).get(kind, 0) else public_key
            rows.append(
                _make_envelope(
                    key,
                    b'{"v":1}',
                    metric_type=kind,
                    envelope_id=f"env-{kind}-{index}",
                    blob_id=f"blob-{kind}-{index}",
                )
            )
    return rows


def _check(report: dict[str, Any], name: str) -> dict[str, Any]:
    return next(check for check in report["checks"] if check["name"] == name)


def test_cold_doctor_fetches_a_sample_and_never_the_library(tmp_path: Path) -> None:
    from vaultbeat_mcp_local.service import KNOWN_METRIC_TYPES

    service, cloud, public_key = _doctor_ready(tmp_path)
    cloud.envelopes = _library(public_key, {"sleep": 50, "basal_energy": 80, "water": 5, "profile": 1})

    report = asyncio.run(service.doctor())

    # The whole point: no full read of anything, one digest per kind, and the
    # only ciphertext fetched is the sample.
    assert cloud.sync_calls == [], "a full read of a kind is the 51 MB this replaced"
    assert sorted(cloud.digest_calls) == sorted(KNOWN_METRIC_TYPES), (
        "one digest per kind, asked once — the capability report must reuse them"
    )
    assert sum(len(batch) for batch in cloud.blob_fetches) <= service.ROUNDTRIP_SAMPLE
    # The smallest kind holding a full sample, not the 1-record profile (one
    # damaged record would fail it alone) and not a big kind (dear to list).
    assert cloud.catalog_calls == ["water"]

    roundtrip = _check(report, "data_roundtrip")
    assert roundtrip["ok"] is True
    assert roundtrip["detail"] == "decrypted 3 of 3 sampled water record(s)"

    caps = report["capabilities"]
    assert caps["record_counts"] == {"basal_energy": 80, "profile": 1, "sleep": 50, "water": 5}
    assert "kinds_not_checked" not in caps
    assert "owner_prefixes" not in caps, "it needed every record downloaded; 0.9.0 has no `owner`"


def test_doctor_round_trip_passes_while_any_sampled_record_decrypts(tmp_path: Path) -> None:
    """One damaged record must not read as a broken install — the full read
    this replaced failed only when NOTHING decrypted, and so must the sample."""
    service, cloud, public_key = _doctor_ready(tmp_path)
    cloud.envelopes = _library(public_key, {"water": 3}, foreign={"water": 2})

    roundtrip = _check(asyncio.run(service.doctor()), "data_roundtrip")

    assert roundtrip["ok"] is True
    assert roundtrip["detail"] == "decrypted 1 of 3 sampled water record(s)"


def test_doctor_round_trip_fails_with_the_rebind_hint_when_nothing_decrypts(tmp_path: Path) -> None:
    service, cloud, public_key = _doctor_ready(tmp_path)
    cloud.envelopes = _library(public_key, {"water": 4}, foreign={"water": 4})

    roundtrip = _check(asyncio.run(service.doctor()), "data_roundtrip")

    assert roundtrip["ok"] is False
    assert "decrypt failed" in roundtrip["detail"]
    assert roundtrip["hint"].index("bind") < roundtrip["hint"].index("Deleting")


def test_doctor_with_nothing_sealed_says_decryption_was_not_tested(tmp_path: Path) -> None:
    """A brand-new binding holds nothing yet. Passing is fair — the cloud took
    the token — but the detail must not claim a decryption that never ran."""
    service, cloud, _public_key = _doctor_ready(tmp_path)

    report = asyncio.run(service.doctor())
    roundtrip = _check(report, "data_roundtrip")

    assert roundtrip["ok"] is True
    assert "has not been tested" in roundtrip["detail"]
    assert cloud.blob_fetches == [] and cloud.catalog_calls == []
    caps = report["capabilities"]
    assert caps["kinds_with_data"] == []
    assert "Every kind" not in caps["note"]


def test_doctor_lists_a_kind_it_could_not_count_in_neither_list(tmp_path: Path) -> None:
    """Invariant 57 (absence-has-more-than-one-cause): a failed count is not an
    empty kind. Reporting it as empty would send the reader to re-sync or to
    Health permissions for a kind that may be full."""
    from vaultbeat_mcp_local.client import VaultbeatCloudError

    service, cloud, public_key = _doctor_ready(tmp_path)
    cloud.envelopes = _library(public_key, {"sleep": 5, "water": 5})
    real_digest = cloud.sync_digest

    async def flaky(server_token: str, *, metric_type: str | None = None) -> Any:
        if metric_type == "water":
            raise VaultbeatCloudError("Cloud request failed: HTTP 502")
        return await real_digest(server_token, metric_type=metric_type)

    cloud.sync_digest = flaky  # type: ignore[method-assign]

    caps = asyncio.run(service.doctor())["capabilities"]

    assert caps["kinds_not_checked"] == ["water"]
    assert "water" not in caps["kinds_with_data"]
    assert "water" not in caps["kinds_without_data"]
    assert caps["kinds_with_data"] == ["sleep"]


def test_doctor_reports_an_expired_trial_once_and_does_not_ask_again(tmp_path: Path) -> None:
    from vaultbeat_mcp_local.client import VaultbeatTrialExpiredError
    from vaultbeat_mcp_local.service import KNOWN_METRIC_TYPES

    service, cloud, _public_key = _doctor_ready(tmp_path)
    calls: list[str | None] = []

    async def refused(server_token: str, *, metric_type: str | None = None) -> Any:
        calls.append(metric_type)
        raise VaultbeatTrialExpiredError("2026-09-01T00:00:00Z")

    cloud.sync_digest = refused  # type: ignore[method-assign]

    report = asyncio.run(service.doctor())

    roundtrip = _check(report, "data_roundtrip")
    assert roundtrip["ok"] is False
    assert "Re-running `bind` will not help" in roundtrip["hint"]
    assert report["capabilities"]["available"] is False
    assert report["capabilities"]["reason"] == roundtrip["detail"]
    # One probe, then nothing: an account-level refusal is not asked seventeen
    # more times, and the capability report does not retry what just failed.
    assert len(calls) == 1 and calls[0] in KNOWN_METRIC_TYPES


def test_doctor_with_a_dead_token_asks_once_not_eighteen_times(tmp_path: Path) -> None:
    """Review R4 (2026-10-03): eighteen concurrent 401s from one doctor run.

    `mcp-sync` locks an IP out for 15 minutes after 20 unauthorised requests,
    so a doctor fanning out every kind at once with a dead token put the agent's
    next read over the line. One probe goes first; the rest only if it worked,
    and never more than `_CONCURRENT_DIGESTS` at a time.
    """
    from vaultbeat_mcp_local.client import VaultbeatCloudError
    from vaultbeat_mcp_local.service import _CONCURRENT_DIGESTS, KNOWN_METRIC_TYPES

    service, cloud, public_key = _doctor_ready(tmp_path)
    calls: list[str | None] = []

    async def unauthorised(server_token: str, *, metric_type: str | None = None) -> Any:
        calls.append(metric_type)
        raise VaultbeatCloudError("Cloud request failed: HTTP 401")

    cloud.sync_digest = unauthorised  # type: ignore[method-assign]
    report = asyncio.run(service.doctor())
    assert _check(report, "data_roundtrip")["ok"] is False
    assert len(calls) == 1

    # Healthy token: every kind is asked, a few at a time.
    service, cloud, public_key = _doctor_ready(tmp_path / "ok")
    cloud.envelopes = _library(public_key, {"sleep": 5})
    real_digest = cloud.sync_digest
    in_flight = 0
    peak = 0

    async def counted(server_token: str, *, metric_type: str | None = None) -> Any:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        try:
            await asyncio.sleep(0.001)
            return await real_digest(server_token, metric_type=metric_type)
        finally:
            in_flight -= 1

    cloud.sync_digest = counted  # type: ignore[method-assign]
    caps = asyncio.run(service.doctor())["capabilities"]
    assert caps["kinds_with_data"] == ["sleep"]
    assert len(caps["kinds_with_data"]) + len(caps["kinds_without_data"]) == len(KNOWN_METRIC_TYPES)
    assert 1 < peak <= _CONCURRENT_DIGESTS


@pytest.mark.parametrize(
    ("error", "says_rebind"),
    [
        (lambda: VaultbeatCloudError("Cloud request failed: HTTP 503", status_code=503), False),
        (lambda: VaultbeatCloudError(
            "Cloud request failed: HTTP 401 error=invalid_token", status_code=401, code="invalid_token"), True),
        (lambda: ConnectionError("connection reset"), False),
        # A gateway 401 with no `error` code (an Edge Function redeployed with
        # verify_jwt reset) is not this machine's token: no re-bind.
        (lambda: VaultbeatCloudError("Cloud request failed: HTTP 401", status_code=401), False),
    ],
)
def test_doctor_prescribes_a_rebind_only_for_a_rejected_token(
    tmp_path: Path, error: Any, says_rebind: bool,
) -> None:
    """Review V7 (2026-10-03): with one probe going first (R4), a single 503 or
    dropped connection was told "the server token is no longer accepted —
    re-run bind". Only a rejected credential is a re-bind."""
    service, cloud, _public_key = _doctor_ready(tmp_path)

    async def failing(server_token: str, *, metric_type: str | None = None) -> Any:
        raise error()

    cloud.sync_digest = failing  # type: ignore[method-assign]
    roundtrip = _check(asyncio.run(service.doctor()), "data_roundtrip")
    assert roundtrip["ok"] is False
    assert ("no longer accepted" in roundtrip["hint"]) is says_rebind
    assert ("Do not re-pair" in roundtrip["hint"]) is (not says_rebind)


def test_doctor_does_not_call_a_rate_limit_unrelated_to_the_pairing(tmp_path: Path) -> None:
    """Review V7 follow-up: mcp-sync limits an address only after repeated
    rejected tokens, so its 429 must not be told "nothing points at the pairing"."""
    service, cloud, _public_key = _doctor_ready(tmp_path)

    async def limited(server_token: str, *, metric_type: str | None = None) -> Any:
        raise VaultbeatCloudError("Cloud request failed: HTTP 429 error=rate_limited", status_code=429, code="rate_limited")

    cloud.sync_digest = limited  # type: ignore[method-assign]
    roundtrip = _check(asyncio.run(service.doctor()), "data_roundtrip")
    assert "Wait 15 minutes" in roundtrip["hint"]
    assert "nothing in the error points at this machine's pairing" not in roundtrip["hint"]
