from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import httpx


class VaultbeatCloudError(RuntimeError):
    pass


class VaultbeatUnsupportedMetricError(VaultbeatCloudError):
    """The server does not recognise a metric type this client knows about.

    Happens when the MCP server is newer than the deployed `mcp-sync` edge
    function — the client asks for a kind the server's allowlist predates. It is
    a version-skew condition, not a failure: every OTHER kind still works, so
    callers should degrade to "this one kind is unavailable" instead of failing
    the whole read. Kept a subclass of VaultbeatCloudError so existing
    `except VaultbeatCloudError` handlers keep working unchanged.

    Real incident this models: 2026-07-22 shipped `hrv_hourly` without adding it
    to the edge allowlist, and because it was also `get_hrv`'s new DEFAULT
    granularity, every default HRV read 400'd for two days rather than
    returning "hourly unavailable, here's raw".
    """

    def __init__(self, metric_type: str, allowed: list[str] | None = None) -> None:
        self.metric_type = metric_type
        self.allowed = allowed or []
        super().__init__(
            f"Server does not support metric_type={metric_type!r}. "
            "The Vaultbeat cloud function is older than this MCP server — "
            "other data types are unaffected."
        )


@dataclass(frozen=True)
class PollBindingResult:
    status: str
    server_id: str | None = None
    server_token: str | None = None
    # Owner identity returned by mcp-poll-binding (bind handshake); all may be
    # null for legacy binds or a user with no identity/device row.
    owner_user_id: str | None = None
    owner_public_key_base64: str | None = None
    owner_device_id: str | None = None
    request_id: str | None = None


class VaultbeatCloudClient:
    # Read budget. Supabase edge functions cold-start at 20-63s measured
    # (2026-07-27), and the old 20s ceiling meant the heaviest call — `sync`,
    # which the cold-backup script runs first — timed out nearly every time the
    # edge had gone cold. It failed three runs in a row that day before anyone
    # looked, because `httpx.ReadTimeout` stringifies to "" and the CLI's
    # catch-all printed a bare `error:` with no type name. Warming the edge with
    # a throwaway request (the backup script does this) helps but does not fix
    # the ceiling; this does.
    DEFAULT_TIMEOUT_SECONDS = 90.0

    # Connect stays short on purpose. A dead tunnel / DNS blackhole must surface
    # in seconds, not after the full read budget — this box's uplink goes through
    # a proxy that does flap. Separating the two is the whole point: tolerate a
    # slow server, fail fast on a broken path.
    CONNECT_TIMEOUT_SECONDS = 10.0

    def __init__(self, api_base_url: str, *, timeout: float = DEFAULT_TIMEOUT_SECONDS):
        self.api_base_url = api_base_url.rstrip("/")
        self.timeout = timeout

    def _timeout(self) -> httpx.Timeout:
        """httpx timeout with a short connect and a long read.

        Built per call rather than in `__init__` so httpx stays lazily imported
        (the cache-hit CLI path must not pay its import cost). A caller passing
        an explicit `timeout=` still wins — connect is clamped to it so a
        deliberately tight budget is never silently widened.
        """
        import httpx

        return httpx.Timeout(
            self.timeout, connect=min(self.CONNECT_TIMEOUT_SECONDS, self.timeout)
        )

    async def poll_binding(self, poll_id: str) -> PollBindingResult:
        # httpx is imported lazily so the CLI's cache-hit path (no network)
        # never pays its import cost.
        import httpx

        async with httpx.AsyncClient(timeout=self._timeout()) as client:
            response = await client.post(
                f"{self.api_base_url}/mcp-poll-binding",
                params={"pollID": poll_id},
            )
        payload = self._decode_response(response)
        status = str(payload.get("status", "pending"))
        return PollBindingResult(
            status=status,
            server_id=payload.get("serverID"),
            server_token=payload.get("serverToken"),
            owner_user_id=payload.get("ownerUserID"),
            owner_public_key_base64=payload.get("ownerPublicKeyBase64"),
            owner_device_id=payload.get("ownerDeviceID"),
            request_id=payload.get("request_id"),
        )

    async def sync(
        self, server_token: str, *, metric_type: str | None = None
    ) -> list[dict[str, Any]]:
        """Fetch this server's envelopes; `metric_type` narrows server-side.

        Older mcp-sync deployments ignore the parameter and return everything —
        the service layer keeps its own defensive filter, so passing it is
        always safe (an optimization, never a correctness dependency).
        """

        import httpx

        params: dict[str, str] = {}
        if metric_type:
            params["metric_type"] = metric_type
        async with httpx.AsyncClient(timeout=self._timeout()) as client:
            response = await client.get(
                f"{self.api_base_url}/mcp-sync",
                headers={"Authorization": f"Bearer {server_token}"},
                params=params,
            )
        payload = self._decode_response(response)
        envelopes = payload.get("envelopes", [])
        if not isinstance(envelopes, list):
            raise VaultbeatCloudError("Cloud response has invalid envelopes shape")
        return [row for row in envelopes if isinstance(row, dict)]

    async def write_strength_blob(
        self,
        server_token: str,
        *,
        blob: dict[str, Any],
        envelopes: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Upsert one already-encrypted strength blob + its envelopes.

        The server never sees plaintext — `blob`/`envelopes` are pre-sealed by
        the caller (``encrypt_blob_payload``). Mirrors ``mcp-sync``'s bearer-token
        auth; the edge function does its own strength-only + ownership validation
        server-side, this client is a thin transport.
        """

        import httpx

        async with httpx.AsyncClient(timeout=self._timeout()) as client:
            response = await client.post(
                f"{self.api_base_url}/mcp-write-strength",
                headers={"Authorization": f"Bearer {server_token}"},
                json={"blob": blob, "envelopes": envelopes},
            )
        return self._decode_response(response)

    async def write_food_blob(
        self,
        server_token: str,
        *,
        blob: dict[str, Any],
        envelopes: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Same shape as `write_strength_blob`, different endpoint. The metric-
        specific split (one edge function per writable kind) keeps each function's
        allow-listed metric_type + recipient scope tight — a food write can never
        widen its blast radius into strength/sleep/etc.
        """

        import httpx

        async with httpx.AsyncClient(timeout=self._timeout()) as client:
            response = await client.post(
                f"{self.api_base_url}/mcp-write-food",
                headers={"Authorization": f"Bearer {server_token}"},
                json={"blob": blob, "envelopes": envelopes},
            )
        return self._decode_response(response)

    async def write_body_blob(
        self,
        server_token: str,
        *,
        blob: dict[str, Any],
        envelopes: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Agent-write path for metric_type="body" (weight). Same shape as
        strength/food; edge fn narrows to body-only allow-list."""

        import httpx

        async with httpx.AsyncClient(timeout=self._timeout()) as client:
            response = await client.post(
                f"{self.api_base_url}/mcp-write-body",
                headers={"Authorization": f"Bearer {server_token}"},
                json={"blob": blob, "envelopes": envelopes},
            )
        return self._decode_response(response)

    async def write_note_blob(
        self,
        server_token: str,
        *,
        blob: dict[str, Any],
        envelopes: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Agent-write path for metric_type="note" (mood/general annotations).
        Same shape as strength/food/body; edge fn narrows to note-only."""

        import httpx

        async with httpx.AsyncClient(timeout=self._timeout()) as client:
            response = await client.post(
                f"{self.api_base_url}/mcp-write-note",
                headers={"Authorization": f"Bearer {server_token}"},
                json={"blob": blob, "envelopes": envelopes},
            )
        return self._decode_response(response)

    async def report_decrypt_failures(
        self,
        server_token: str,
        *,
        items: list[dict[str, str]],
    ) -> None:
        """Tells the cloud "these blob ids are undecryptable from MY (this MCP
        server's) key" — the half of the blob-integrity GC that only this
        server can supply.

        Why this exists: iOS's `VaultbeatBlobIntegrityRepairCoordinator` can
        only verify the OWNER's own envelope (it has no other private key). A
        blob whose owner envelope is healthy but whose mcp_server envelope is
        stale — exactly what Invariant 33's duplicate-blobID bug left behind —
        is invisible to that check and never gets repaired. This call is the
        other eye: whenever THIS server's own decrypt fails with a proven
        DEK-mismatch (never on a mere "not for me"), it reports the blob id so
        iOS can forget its upload fingerprint and re-upload it, regardless of
        what the owner-side check concluded. See AGENTS.md Invariant 34.

        Best-effort by design (see the caller in service.py): a report that
        never arrives just means this blob waits for the next call that
        reaches it, not a broken read.
        """

        if not items:
            return

        import httpx

        async with httpx.AsyncClient(timeout=self._timeout()) as client:
            await client.post(
                f"{self.api_base_url}/mcp-report-decrypt-failures",
                headers={"Authorization": f"Bearer {server_token}"},
                json={"items": items},
            )

    @staticmethod
    def _decode_response(response: "httpx.Response") -> dict[str, Any]:
        try:
            payload = response.json()
        except ValueError as error:
            raise VaultbeatCloudError(f"Cloud returned non-JSON response: HTTP {response.status_code}") from error

        if response.status_code >= 400:
            error_code = payload.get("error") if isinstance(payload, dict) else None
            request_id = payload.get("request_id") if isinstance(payload, dict) else None

            # Version skew gets its own type so callers can degrade one kind
            # instead of failing the whole read. The edge echoes back both the
            # rejected value and its allowlist, so carry them for a precise
            # message.
            if error_code == "invalid_metric_type" and isinstance(payload, dict):
                allowed = payload.get("allowed")
                raise VaultbeatUnsupportedMetricError(
                    str(payload.get("metric_type", "<unknown>")),
                    [str(x) for x in allowed] if isinstance(allowed, list) else None,
                )

            detail = f"Cloud request failed: HTTP {response.status_code}"
            if error_code:
                detail += f" error={error_code}"
            if request_id:
                detail += f" request_id={request_id}"
            raise VaultbeatCloudError(detail)

        if not isinstance(payload, dict):
            raise VaultbeatCloudError("Cloud returned invalid JSON response")
        return payload
