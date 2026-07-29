from __future__ import annotations

import httpx

from vaultbeat_mcp_local.client import VaultbeatCloudClient


# Why these exist: the read budget went 20s -> 90s on 2026-07-27 because Supabase
# edge cold starts measure 20-63s and the heaviest call (`sync`) is the first one
# the cold-backup script makes — it failed three consecutive runs before anyone
# noticed. Widening a timeout is exactly the kind of change that silently rots
# (someone "tidies" the constant, or drops the connect split), so the split is
# pinned here rather than left to the comment.


def test_read_budget_outlasts_a_cold_edge_start() -> None:
    """90s read: measured cold starts reach 63s, the old 20s ceiling did not."""
    timeout = VaultbeatCloudClient("https://example.test")._timeout()
    assert isinstance(timeout, httpx.Timeout)
    assert timeout.read == 90.0
    assert timeout.write == 90.0
    assert timeout.pool == 90.0


def test_connect_stays_short_so_a_broken_tunnel_fails_fast() -> None:
    """A dead proxy/DNS blackhole must surface in ~10s, not after the read budget.

    This is the half that makes the widening safe: without it, every network
    outage would look like a 90-second hang.
    """
    timeout = VaultbeatCloudClient("https://example.test")._timeout()
    assert timeout.connect == 10.0
    assert timeout.connect < timeout.read


def test_an_explicit_tight_timeout_is_never_widened() -> None:
    """Callers asking for a tight budget get it — connect is clamped, not raised.

    Guards the `min()`: a naive `connect=CONNECT_TIMEOUT_SECONDS` would turn a
    deliberate `timeout=5` into a 10s connect, i.e. silently ignore the caller.
    """
    timeout = VaultbeatCloudClient("https://example.test", timeout=5.0)._timeout()
    assert timeout.connect == 5.0
    assert timeout.read == 5.0


def test_an_explicit_generous_timeout_keeps_the_short_connect() -> None:
    """Raising the read budget must not drag connect up with it."""
    timeout = VaultbeatCloudClient("https://example.test", timeout=300.0)._timeout()
    assert timeout.read == 300.0
    assert timeout.connect == 10.0


def test_base_url_trailing_slash_is_normalised() -> None:
    """Every endpoint is built as f"{api_base_url}/name" — a kept trailing slash
    would produce `//name`. Cheap to assert, annoying to debug."""
    assert (
        VaultbeatCloudClient("https://example.test/functions/v1/").api_base_url
        == "https://example.test/functions/v1"
    )
