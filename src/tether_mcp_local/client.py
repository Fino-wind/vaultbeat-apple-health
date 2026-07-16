from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import httpx


class TetherCloudError(RuntimeError):
    pass


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


class TetherCloudClient:
    def __init__(self, api_base_url: str, *, timeout: float = 20.0):
        self.api_base_url = api_base_url.rstrip("/")
        self.timeout = timeout

    async def poll_binding(self, poll_id: str) -> PollBindingResult:
        # httpx is imported lazily so the CLI's cache-hit path (no network)
        # never pays its import cost.
        import httpx

        async with httpx.AsyncClient(timeout=self.timeout) as client:
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
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(
                f"{self.api_base_url}/mcp-sync",
                headers={"Authorization": f"Bearer {server_token}"},
                params=params,
            )
        payload = self._decode_response(response)
        envelopes = payload.get("envelopes", [])
        if not isinstance(envelopes, list):
            raise TetherCloudError("Cloud response has invalid envelopes shape")
        return [row for row in envelopes if isinstance(row, dict)]

    @staticmethod
    def _decode_response(response: "httpx.Response") -> dict[str, Any]:
        try:
            payload = response.json()
        except ValueError as error:
            raise TetherCloudError(f"Cloud returned non-JSON response: HTTP {response.status_code}") from error

        if response.status_code >= 400:
            error_code = payload.get("error") if isinstance(payload, dict) else None
            request_id = payload.get("request_id") if isinstance(payload, dict) else None
            detail = f"Cloud request failed: HTTP {response.status_code}"
            if error_code:
                detail += f" error={error_code}"
            if request_id:
                detail += f" request_id={request_id}"
            raise TetherCloudError(detail)

        if not isinstance(payload, dict):
            raise TetherCloudError("Cloud returned invalid JSON response")
        return payload
