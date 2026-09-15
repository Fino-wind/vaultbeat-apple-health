"""`sync_blobs` packs requests by URL length, not by id count.

Why this file exists: `MAX_BLOB_IDS_PER_REQUEST = 500` was the only ceiling,
and for `basal_energy` on a two-year account it produced URLs the Supabase
edge runtime silently refuses to send onward — every read of that kind
returned `envelope_query_failed` after ~11 s from 2026-09-13 until this
changed. The count was never the variable: measured the same day, 475 ids
passed for one kind and failed for another, because their ids differ in
length. See `MAX_BLOB_IDS_URL_BUDGET` for the full measurement table.

These assert the RULE (stay under the budget, emit everything, honour both
ceilings), never a specific batch count — a budget tweak must not turn the
suite red for a reason that is not a defect.
"""

from __future__ import annotations

import asyncio
from typing import Any
from urllib.parse import quote

import httpx

from vaultbeat_mcp_local.client import VaultbeatCloudClient

BUDGET = VaultbeatCloudClient.MAX_BLOB_IDS_URL_BUDGET
COUNT_CAP = VaultbeatCloudClient.MAX_BLOB_IDS_PER_REQUEST

# The lowest request URL measured FAILING in production, 2026-09-15: 14,854
# characters (basal, 490 ids). 14,565 passed. This is an observed property of
# the deployed edge runtime, not a setting of ours — so assertions about safety
# anchor HERE and never on `BUDGET`.
#
# 🔴 An assertion written as `<= BUDGET` proves nothing about safety: widen the
# budget and the assertion widens with it. Exactly that happened while writing
# this file — raising BUDGET to 10^9 (i.e. restoring the bug) left one of these
# tests green. Anything claiming "this URL is short enough" compares against
# MEASURED_FAILURE_FLOOR; `BUDGET` may only appear in consistency checks.
MEASURED_FAILURE_FLOOR = 14_854


def encoded_len(ids: list[str]) -> int:
    """Exactly what lands in the query string for these ids."""
    return len(quote(",".join(ids), safe=""))


def chunks(ids: list[str]) -> list[list[str]]:
    return list(VaultbeatCloudClient._chunk_blob_ids(ids))


def test_the_configured_budget_stays_under_what_was_measured_to_fail() -> None:
    """Guards the constant itself — the one thing no chunking test can catch."""
    assert BUDGET < MEASURED_FAILURE_FLOOR


def test_every_chunk_fits_the_url_budget_with_long_ids() -> None:
    # 33 chars is the longest blob_id shape in production today
    # (`basal_energy-1788746400-udce9b9cf`).
    ids = [f"basal_energy-17887{i:05d}-udce9b9cf" for i in range(2000)]
    for chunk in chunks(ids):
        # Measured floor, not BUDGET — see the comment on MEASURED_FAILURE_FLOOR.
        assert encoded_len(chunk) < MEASURED_FAILURE_FLOOR, (
            f"chunk of {len(chunk)} ids would reach a URL length measured to fail"
        )
        assert encoded_len(chunk) <= BUDGET, f"chunk of {len(chunk)} ids exceeds the budget"


def test_long_ids_are_cut_by_length_before_they_reach_the_count_cap() -> None:
    """The regression itself: 500 long ids used to go out as ONE request."""
    ids = [f"basal_energy-17887{i:05d}-udce9b9cf" for i in range(COUNT_CAP)]
    assert encoded_len(ids) > BUDGET, "fixture no longer reproduces the oversized request"
    produced = chunks(ids)
    assert len(produced) > 1, "the whole point is that this no longer goes out as one request"
    assert max(len(c) for c in produced) < COUNT_CAP


def test_short_ids_are_still_bounded_by_the_count_cap() -> None:
    """Both ceilings are load-bearing: the server 400s above 500 ids."""
    ids = [f"w-{i}" for i in range(1500)]
    produced = chunks(ids)
    assert encoded_len(ids[:COUNT_CAP]) < BUDGET, "fixture must be length-cheap to test the cap"
    assert all(len(c) <= COUNT_CAP for c in produced)
    assert max(len(c) for c in produced) == COUNT_CAP


def test_nothing_is_dropped_or_reordered() -> None:
    ids = [f"sleep-{i}-{'x' * (i % 40)}" for i in range(3000)]
    flat = [b for chunk in chunks(ids) for b in chunk]
    assert flat == ids


def test_an_id_longer_than_the_whole_budget_still_goes_out() -> None:
    """A 500 on one request beats a row that silently never gets fetched."""
    monster = "m" * (BUDGET + 500)
    produced = chunks(["a", monster, "b"])
    assert [b for chunk in produced for b in chunk] == ["a", monster, "b"]
    assert [monster] in produced


def test_empty_input_produces_no_requests() -> None:
    assert chunks([]) == []


def test_sync_blobs_sends_every_id_across_budgeted_requests() -> None:
    """End to end through the real request path, not just the splitter."""
    ids = [f"basal_energy-17887{i:05d}-udce9b9cf" for i in range(1200)]
    seen_urls: list[str] = []
    seen_ids: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_urls.append(str(request.url))
        blob_ids = request.url.params["blob_ids"]
        seen_ids.extend(blob_ids.split(","))
        return httpx.Response(
            200, json={"envelopes": [{"blob_id": b} for b in blob_ids.split(",")]}
        )

    client = VaultbeatCloudClient(
        "https://example.test/functions/v1", transport=httpx.MockTransport(handler)
    )
    rows: list[dict[str, Any]] = asyncio.run(
        client.sync_blobs("tok", blob_ids=ids, metric_type="basal_energy")
    )

    assert seen_ids == ids, "ids must arrive intact, in order, exactly once"
    assert [r["blob_id"] for r in rows] == ids
    assert len(seen_urls) > 1
    # The measured failure threshold was ~14.5 KB of request URL; every URL we
    # emit must stay clear of it with the whole query string counted, not just
    # the ids.
    assert max(len(u) for u in seen_urls) < 14_000
