from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import httpx


class VaultbeatCloudError(RuntimeError):
    pass


class VaultbeatTrialExpiredError(VaultbeatCloudError):
    """The 3-day Pro trial has ended; this server can no longer read or write.

    Its own type rather than a generic HTTP 403 for one reason: this is the only
    failure in the whole client that is not a malfunction. Nothing is broken,
    nothing is lost, and the fix is a purchase rather than a repair — so it must
    never surface as `Cloud request failed: HTTP 403`, which sends a user to
    check their network, re-run `bind`, or file a bug about data loss.

    ⚠️ The sentence is built HERE from two machine values (an error code and a
    timestamp), never echoed from the server. A server-chosen sentence reaching
    an agent's context is a prompt-injection channel by construction — the tool
    result and a real user turn arrive under the same role, so the model cannot
    tell them apart (Anti-pattern 23). The edge function deliberately returns no
    prose at all; if a `detail` string ever appears in that response, it is a
    regression, not a convenience.

    ⚠️ It says the data is intact. That is the first thing a user assumes has
    gone wrong when their AI abruptly cannot see their health records, and the
    assumption leads somewhere expensive (reinstall, re-pair, panic).
    """

    def __init__(self, trial_ended_at: str | None = None) -> None:
        self.trial_ended_at = trial_ended_at
        when = f" on {trial_ended_at[:10]}" if trial_ended_at else ""
        # vb-005 (2026-08-26): the old sentence said "open the app and
        # subscribe" without a location or a price — and the app's own
        # settings row was not tappable, so the user stood at the door with
        # money in hand and no till. Name the exact path and the prices here
        # (client-side literals, never echoed from the server — Anti-pattern
        # 23 still holds), so the highest-intent moment this product ever
        # produces carries everything needed to act on it.
        super().__init__(
            f"Your 3-day Vaultbeat Pro trial ended{when}. "
            "To restore access: open the Vaultbeat app on your iPhone → "
            "Settings → Membership, and pick a Pro plan ($9.99/month, "
            "$79.99/year, or $149.99 once, forever). Nothing was deleted — "
            "your health data is still encrypted in your account and "
            "reappears the moment Pro is active."
        )


class VaultbeatRecordNotAgentWritableError(VaultbeatCloudError):
    """This day is sealed for a recipient this MCP server cannot re-seal for.

    Happens to users with a bound partner: iOS seals `.allVisible` kinds (body,
    water, sleep, menstrual) for the PARTNER's MCP server too, and a write
    endpoint only ever holds the public key of the server that authenticated it
    — so rewriting the row here would leave the partner's AI permanently unable
    to read it. The server refuses (409) rather than damage it, which is
    correct; what was missing until 2026-08-07 is that the refusal reached the
    agent as a bare `HTTP 409 error=…` with no action attached.

    The wording is generated HERE, never echoed from the server: a server-chosen
    sentence reaching an agent's context is a prompt-injection channel by
    construction (Anti-pattern 23). The server sends facts (an error code, a
    list of recipient kinds); the client turns them into instructions.
    """

    def __init__(self, uncoverable_kinds: list[str] | None = None) -> None:
        self.uncoverable_kinds = uncoverable_kinds or []
        who = {
            "partner_user": "your partner",
            "mcp_server": "another AI server (likely your partner's)",
        }
        named = [who.get(k, k) for k in self.uncoverable_kinds]
        audience = f" ({', '.join(named)})" if named else ""
        super().__init__(
            "This day is already shared with a recipient this server cannot "
            f"re-encrypt for{audience}, so it cannot be rewritten from here — "
            "doing so would leave them unable to read it. "
            "Edit this day in the Vaultbeat iOS app instead; the app can seal "
            "for every recipient. Days that only you can see (strength, food, "
            "notes) are unaffected."
        )


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
    # ISO8601 end of the free Pro trial, present only when this bind STARTED one
    # (null for a grandfathered user, an already-running trial, or an older edge
    # deployment that does not send the field). Purely informational — the
    # entitlement decision is always the server's.
    trial_ends_at: str | None = None
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

    # Platform-layer statuses worth one more try. Mostly the gateway in FRONT
    # of the edge function (cold start, brief unavailability) — though not
    # exclusively: `mcpWrite.ts` itself answers 503 when its auth backend is
    # unreachable, so "503 means our code never ran" is an assumption, not a
    # guarantee. Retrying is still right for those: they are transient by
    # construction and the write is an idempotent upsert keyed by blob id.
    RETRYABLE_STATUS_CODES = frozenset({502, 503, 504})

    # 🔴 Endpoints that MUST NOT be retried, whatever the failure looks like.
    #
    # `mcp-poll-binding` is consume-on-read: it deletes the pending row with a
    # DELETE ... RETURNING and hands back the ONLY plaintext copy of the
    # serverToken (the database keeps a PBKDF2 hash). If the response is lost
    # in transit — precisely the ProxyError/ReadError class this retry exists
    # for — a replay finds the row gone and gets `status: "expired"`. That
    # turns a binding that ACTUALLY SUCCEEDED into a permanent failure: the
    # token is unrecoverable, and the server row created before the poll is
    # left orphaned. The retry is the difference between "re-scan the QR" and
    # "the token no longer exists anywhere".
    #
    # Found 2026-08-07 by an adversarial audit of the commit that introduced
    # retries — whose own comment claimed every endpoint here was idempotent.
    # Verify that claim against each endpoint's semantics, not its HTTP verb.
    NON_IDEMPOTENT_PATHS = frozenset({"/mcp-poll-binding"})

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

        # A consume-on-read endpoint gets exactly one attempt: replaying it
        # after a lost response destroys state the first attempt consumed.
        max_attempts = 1 if path in self.NON_IDEMPOTENT_PATHS else 1 + self.MAX_TRANSIENT_RETRIES

        last_error: Exception | None = None
        async with httpx.AsyncClient(
            timeout=self._timeout(), transport=self._transport
        ) as client:
            for attempt in range(max_attempts):
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
                    and attempt < max_attempts - 1
                ):
                    # Not cleared: if the NEXT attempt dies with a transport
                    # error, the raise below should still be able to name the
                    # 5xx that preceded it rather than reporting only the last
                    # symptom of a mixed sequence.
                    last_error = VaultbeatCloudError(f"HTTP {response.status_code}")
                    continue
                return response
        raise VaultbeatCloudError(
            f"Cloud request failed after {max_attempts} attempt(s): "
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
            trial_ends_at=payload.get("trialEndsAt"),
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
        what the owner-side check concluded. See CLAUDE.md Invariant 34.

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
            # Same shape as below: a code the client can act on gets its own
            # type and its own LOCALLY-generated wording. `uncoverable_recipient_kinds`
            # is an enum list, not prose — safe to read.
            if error_code == "envelope_recipients_not_coverable" and isinstance(payload, dict):
                kinds = payload.get("uncoverable_recipient_kinds")
                raise VaultbeatRecordNotAgentWritableError(
                    [str(k) for k in kinds] if isinstance(kinds, list) else None
                )

            # Checked before the generic branches: an ended trial is the one
            # 4xx here that is not a malfunction, and the generic wording
            # ("Cloud request failed") would send the user to debug a network.
            if error_code == "trial_expired":
                ended = payload.get("trial_ended_at") if isinstance(payload, dict) else None
                raise VaultbeatTrialExpiredError(str(ended) if ended else None)

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
