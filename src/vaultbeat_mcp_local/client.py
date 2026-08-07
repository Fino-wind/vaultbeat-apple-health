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

    # Transient-failure retry budget. This box's uplink goes through a proxy
    # that intermittently answers large responses with a bare 503 (logged as a
    # Known Bug since 2026-07-29: `curl` on the same endpoint succeeds while
    # httpx's proxied request dies with ProxyError) — catalog mode shrank the
    # responses that trip it, but nothing retried. 2 retries × short backoff
    # covers a flap without turning a real outage into a 3× wait.
    MAX_TRANSIENT_RETRIES = 2
    RETRY_BACKOFF_SECONDS = 0.5

    # Platform-layer statuses worth one more try. These come from the gateway
    # in FRONT of the edge function (cold start, brief unavailability), not
    # from our code — our functions answer errors as JSON with an `error`
    # field. Every endpoint this client talks to is idempotent (reads, or
    # upserts keyed by blob id), so retrying a POST is safe.
    RETRYABLE_STATUS_CODES = frozenset({502, 503, 504})

    def __init__(
        self,
        api_base_url: str,
        *,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        transport: Any | None = None,
    ):
        self.api_base_url = api_base_url.rstrip("/")
        self.timeout = timeout
        # Test seam only: lets the retry tests drive httpx.MockTransport
        # through the real request path. Production callers never pass it.
        self._transport = transport

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

    async def _request(
        self,
        method: str,
        path: str,
        *,
        headers: dict[str, str] | None = None,
        params: dict[str, str] | None = None,
        json: Any | None = None,
    ) -> "httpx.Response":
        """One HTTP call with transient-failure retries.

        Retries (up to MAX_TRANSIENT_RETRIES, linear backoff):
        - `httpx.ProxyError` / `ConnectError` / `ReadError` /
          `RemoteProtocolError` — the connection-layer flaps a proxied uplink
          produces. NOT timeouts: `ReadTimeout` means the 90s read budget was
          genuinely spent, and paying it again on an edge cold-start would turn
          one slow call into three.
        - HTTP 502/503/504 — gateway-layer, before our code ran.

        Anything else (4xx, our own JSON errors, timeouts) surfaces exactly as
        before — `_decode_response` stays the single interpreter.
        """

        import asyncio

        import httpx

        transient_exceptions = (
            httpx.ProxyError,
            httpx.ConnectError,
            httpx.ReadError,
            httpx.RemoteProtocolError,
        )

        last_error: Exception | None = None
        async with httpx.AsyncClient(
            timeout=self._timeout(), transport=self._transport
        ) as client:
            for attempt in range(1 + self.MAX_TRANSIENT_RETRIES):
                if attempt > 0:
                    await asyncio.sleep(self.RETRY_BACKOFF_SECONDS * attempt)
                try:
                    response = await client.request(
                        method,
                        f"{self.api_base_url}{path}",
                        headers=headers,
                        params=params,
                        json=json,
                    )
                except transient_exceptions as error:
                    last_error = error
                    continue
                if (
                    response.status_code in self.RETRYABLE_STATUS_CODES
                    and attempt < self.MAX_TRANSIENT_RETRIES
                ):
                    last_error = None
                    continue
                return response
        raise VaultbeatCloudError(
            f"Cloud request failed after {1 + self.MAX_TRANSIENT_RETRIES} attempts: "
            f"{type(last_error).__name__ if last_error else 'transient error'}: {last_error}"
        ) from last_error

    async def poll_binding(self, poll_id: str) -> PollBindingResult:
        response = await self._request(
            "POST", "/mcp-poll-binding", params={"pollID": poll_id}
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

        params: dict[str, str] = {}
        if metric_type:
            params["metric_type"] = metric_type
        response = await self._request(
            "GET",
            "/mcp-sync",
            headers={"Authorization": f"Bearer {server_token}"},
            params=params,
        )
        payload = self._decode_response(response)
        envelopes = payload.get("envelopes", [])
        if not isinstance(envelopes, list):
            raise VaultbeatCloudError("Cloud response has invalid envelopes shape")
        return [row for row in envelopes if isinstance(row, dict)]

    async def sync_digest(
        self, server_token: str, *, metric_type: str | None = None
    ) -> tuple[dict[str, Any] | None, list[dict[str, Any]] | None]:
        """Ask what the server holds, in ~119 bytes.

        Returns ``(digest, legacy_envelopes)``:

        * ``(digest, None)`` — a catalog-aware deployment answered.
        * ``(None, envelopes)`` — an OLDER deployment ignored ``fields`` and sent
          the full payload instead. That response is not wasted: it IS the data,
          so the caller uses it directly rather than paying for a second fetch.
          This is the same "the parameter is an optimization, never a dependency"
          contract ``metric_type`` has, and it is what lets a client upgrade
          before the edge function does.

        Never raises on an unknown shape — an unrecognised body degrades to the
        legacy path rather than failing a read.
        """

        params: dict[str, str] = {"fields": "digest"}
        if metric_type:
            params["metric_type"] = metric_type
        response = await self._request(
            "GET",
            "/mcp-sync",
            headers={"Authorization": f"Bearer {server_token}"},
            params=params,
        )
        payload = self._decode_response(response)
        digest = payload.get("digest")
        if isinstance(digest, dict):
            return digest, None
        envelopes = payload.get("envelopes")
        if isinstance(envelopes, list):
            return None, [row for row in envelopes if isinstance(row, dict)]
        return None, None

    async def sync_catalog(
        self, server_token: str, *, metric_type: str | None = None
    ) -> list[dict[str, Any]] | None:
        """The `{blob_id, xmin}` list the client diffs against.

        ``None`` means this deployment has no catalog mode — caller falls back to
        a full fetch.
        """

        params: dict[str, str] = {"fields": "meta"}
        if metric_type:
            params["metric_type"] = metric_type
        response = await self._request(
            "GET",
            "/mcp-sync",
            headers={"Authorization": f"Bearer {server_token}"},
            params=params,
        )
        payload = self._decode_response(response)
        catalog = payload.get("catalog")
        if not isinstance(catalog, list):
            return None
        return [row for row in catalog if isinstance(row, dict)]

    # Kept in step with mcp-sync's MAX_BLOB_IDS. Exceeding it is a 400, so the
    # caller chunks; this constant is what it chunks by.
    MAX_BLOB_IDS_PER_REQUEST = 500

    async def sync_blobs(
        self, server_token: str, *, blob_ids: list[str], metric_type: str | None = None
    ) -> list[dict[str, Any]]:
        """Fetch ONLY these blobs, in the same envelope shape as a full sync.

        Chunked at MAX_BLOB_IDS_PER_REQUEST so the caller never has to think
        about URL limits.
        """

        if not blob_ids:
            return []
        collected: list[dict[str, Any]] = []
        for start in range(0, len(blob_ids), self.MAX_BLOB_IDS_PER_REQUEST):
            chunk = blob_ids[start : start + self.MAX_BLOB_IDS_PER_REQUEST]
            params: dict[str, str] = {"blob_ids": ",".join(chunk)}
            if metric_type:
                params["metric_type"] = metric_type
            response = await self._request(
                "GET",
                "/mcp-sync",
                headers={"Authorization": f"Bearer {server_token}"},
                params=params,
            )
            payload = self._decode_response(response)
            envelopes = payload.get("envelopes", [])
            if not isinstance(envelopes, list):
                raise VaultbeatCloudError("Cloud response has invalid envelopes shape")
            collected.extend(row for row in envelopes if isinstance(row, dict))
        return collected

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

        response = await self._request(
            "POST",
            "/mcp-write-strength",
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

        response = await self._request(
            "POST",
            "/mcp-write-food",
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

        response = await self._request(
            "POST",
            "/mcp-write-body",
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

        response = await self._request(
            "POST",
            "/mcp-write-note",
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

        await self._request(
            "POST",
            "/mcp-report-decrypt-failures",
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
