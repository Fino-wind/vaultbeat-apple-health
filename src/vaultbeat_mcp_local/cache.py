from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from vaultbeat_mcp_local.store import write_secret_file

_LOG = logging.getLogger("vaultbeat_mcp_local.cache")

# Cache freshness window. Health data lands in the cloud at the phone's
# background-sync cadence (minutes-to-hours), so an agent asking twice within a
# few minutes should not pay a second cloud round trip. Override with
# VAULTBEAT_MCP_CACHE_TTL (seconds); 0 disables the cache entirely.
DEFAULT_TTL_SECONDS = 600.0
TTL_ENV = "VAULTBEAT_MCP_CACHE_TTL"
# Pre-rename env var, honored as a fallback so existing setups keep working.
_LEGACY_TTL_ENV = "TETHER_MCP_CACHE_TTL"


def _ttl_from_env(default: float) -> float:
    raw = os.getenv(TTL_ENV, "").strip() or os.getenv(_LEGACY_TTL_ENV, "").strip()
    if not raw:
        return default
    try:
        return max(float(raw), 0.0)
    except ValueError:
        _LOG.warning("Ignoring non-numeric %s=%r", TTL_ENV, raw)
        return default


class LocalRecordCache:
    """Decrypted-record cache on the user's own machine.

    Stores DECRYPTED plaintext JSON — acceptable by design: the whole point of
    the local MCP server is that decryption happens in the user's trusted
    environment, and the files are owner-only (0600 via `write_secret_file`,
    0700 directory). E2EE protects the wire and the cloud, not the user's own
    disk from the user.

    One file per metric_type (plus one for the unfiltered "all" set), each
    stamped with the server_id it was fetched for, a fetch timestamp, and the
    fetch's decrypt-error list (a cache hit must report the same errors the
    fetch did): a bind to a different server or an expired TTL reads as a miss.
    """

    def __init__(self, directory: Path, *, ttl_seconds: float | None = None):
        self.directory = directory
        # An explicit ttl_seconds always wins; the env var only replaces the
        # built-in default — so `LocalRecordCache(dir, ttl_seconds=0)` really
        # disables the cache regardless of the caller's shell environment.
        self.ttl_seconds = (
            _ttl_from_env(DEFAULT_TTL_SECONDS) if ttl_seconds is None else max(ttl_seconds, 0.0)
        )

    @property
    def enabled(self) -> bool:
        return self.ttl_seconds > 0

    def _path(self, metric_type: str | None) -> Path:
        return self.directory / f"records-{metric_type or 'all'}.json"

    def load(
        self, *, server_id: str, metric_type: str | None
    ) -> tuple[list[dict[str, Any]], list[str]] | None:
        """Cached (records, errors), or None on miss/stale/mismatch/disabled.

        The early `return None`s below are ordinary cache-miss control flow,
        not swallowed failures — the caller answers a miss with a cloud fetch.
        """

        if not self.enabled:
            return None
        path = self._path(metric_type)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, json.JSONDecodeError) as error:
            _LOG.warning("Discarding unreadable cache %s: %s", path.name, type(error).__name__)
            return None

        if not isinstance(raw, dict) or raw.get("server_id") != server_id:
            return None
        fetched_at = raw.get("fetched_at")
        if not isinstance(fetched_at, (int, float)):
            return None
        if (time.time() - float(fetched_at)) > self.ttl_seconds:
            return None
        records = raw.get("records")
        if not isinstance(records, list):
            return None
        errors = raw.get("errors")
        if not isinstance(errors, list):
            errors = []
        return (
            [row for row in records if isinstance(row, dict)],
            [str(item) for item in errors],
        )

    def save(
        self,
        records: list[dict[str, Any]],
        *,
        server_id: str,
        metric_type: str | None,
        errors: list[str] | None = None,
    ) -> None:
        """Persist a FULL (never limit-truncated) result set; best-effort.

        `errors` carries the fetch's per-envelope decrypt failures so a cache
        hit reports the same problem set the underlying fetch did — without it,
        a poisoned envelope's error (and the CLI's exit code 3) would flap
        with cache warmth.
        """

        if not self.enabled:
            return
        payload = {
            "server_id": server_id,
            "metric_type": metric_type,
            "fetched_at": time.time(),
            "records": records,
            "errors": errors or [],
        }
        try:
            write_secret_file(
                self._path(metric_type),
                json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n",
                harden_parent=True,
            )
        except OSError as error:
            _LOG.warning("Cache write failed (%s); continuing uncached", type(error).__name__)

    def clear(self) -> None:
        """Drop every cached record file (used when (re)binding)."""

        try:
            entries = list(self.directory.glob("records-*.json"))
        except OSError as error:
            _LOG.warning(
                "Cache clear could not list %s (%s); stale plaintext may remain",
                self.directory,
                type(error).__name__,
            )
            return
        for path in entries:
            try:
                path.unlink()
            except OSError as error:
                _LOG.warning(
                    "Cache clear could not remove %s (%s); stale plaintext may remain",
                    path.name,
                    type(error).__name__,
                )
                continue
