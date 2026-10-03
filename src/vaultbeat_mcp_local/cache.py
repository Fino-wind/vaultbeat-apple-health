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
        self._sweep_abandoned_temps()

    #: A temp older than this outlived its writer. `write_secret_file` publishes
    #: with `os.replace` inside the same call, so a live write holds its temp for
    #: milliseconds; ten minutes only guards against a concurrent writer on a
    #: very slow disk, never against a real one.
    _ABANDONED_TEMP_AGE_SECONDS = 600.0

    def _sweep_abandoned_temps(self) -> None:
        """Delete `records-*.tmp-*` files a killed writer left behind.

        They hold decrypted health plaintext. `write_secret_file` unlinks its
        temp only on a Python exception, so a SIGKILL, a client restarting the
        server mid-write or a crash orphans it, and nothing else ever looked at
        them (audit 2026-09-06 `orphaned-plaintext-cache-tmp-files`; one was
        sitting on the owner's machine, 346 KB of activity, from 2026-08-26).
        """

        try:
            temps = list(self.directory.glob("records-*.tmp-*"))
        except OSError:
            return
        now = time.time()
        for path in temps:
            try:
                if now - path.stat().st_mtime > self._ABANDONED_TEMP_AGE_SECONDS:
                    path.unlink()
            except OSError:
                continue

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

    def load_persisted(
        self, *, server_id: str, metric_type: str | None
    ) -> tuple[list[dict[str, Any]], list[str], dict[str, Any] | None, dict[str, str]] | None:
        """Same file as `load`, but WITHOUT the TTL check.

        The two exist because they answer different questions:

        * `load`  — "may I skip the network entirely?" (TTL says the data is
                    recent enough that a round trip is not worth it)
        * `load_persisted` — "what do I already hold?" Freshness here is decided
                    by comparing the server's digest against the stored one, so
                    age is irrelevant: a year-old cache whose digest still
                    matches is exactly as correct as a one-second-old one.

        Returns `(records, errors, digest, blob_xmins)`. A file written before
        this feature has no digest and reads as `(records, errors, None, {})` —
        the caller treats a missing digest as "cannot verify", which costs one
        full fetch and then self-heals.

        Disabled (TTL=0) reads as a miss like everything else: that setting means
        "keep nothing on disk", so there is by definition nothing to reuse.
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
        records = raw.get("records")
        if not isinstance(records, list):
            return None
        errors = raw.get("errors")
        if not isinstance(errors, list):
            errors = []
        digest = raw.get("digest")
        if not isinstance(digest, dict):
            digest = None
        xmins_raw = raw.get("blob_xmins")
        blob_xmins: dict[str, str] = {}
        if isinstance(xmins_raw, dict):
            blob_xmins = {str(k): str(v) for k, v in xmins_raw.items()}
        return (
            [row for row in records if isinstance(row, dict)],
            [str(item) for item in errors],
            digest,
            blob_xmins,
        )

    def save(
        self,
        records: list[dict[str, Any]],
        *,
        server_id: str,
        metric_type: str | None,
        errors: list[str] | None = None,
        digest: dict[str, Any] | None = None,
        blob_xmins: dict[str, str] | None = None,
        started_at: float | None = None,
    ) -> None:
        """Persist a FULL (never limit-truncated) result set; best-effort.

        `started_at` is when the fetch that produced these rows BEGAN. The
        newest-starting fetch of a kind wins, across processes: the file's
        mtime is set to its fetch's start, and a save whose fetch began before
        that is dropped (review V5, 2026-10-03). The in-process generation
        check (R6) could not see a second MCP process sharing this pairing —
        two Claude Code sessions do exactly that — whose plain read, begun
        before the other's write, landed after it and hid the write for the
        whole TTL. Comparing starts, not save times, is what lets the writer's
        own re-read win over an older fetch that merely finished later.

        `errors` carries the fetch's per-envelope decrypt failures so a cache
        hit reports the same problem set the underlying fetch did — without it,
        a poisoned envelope's error (and the CLI's exit code 3) would flap
        with cache warmth.
        """

        # Still gated on `self.enabled`, deliberately. The catalog/digest path
        # would work better with rows kept on disk even at TTL=0, and that was
        # briefly implemented — but TTL=0 is how a user says "do not write my
        # decrypted health data to disk", and that is a PRIVACY choice, not a
        # bandwidth-tuning knob. Honour it; such a user pays for full fetches,
        # which is the trade they asked for.
        if not self.enabled:
            return
        path = self._path(metric_type)
        if started_at is not None:
            try:
                if path.stat().st_mtime > started_at:
                    _LOG.debug("Cache save skipped for %s: a newer fetch already wrote it", path.name)
                    return
            except FileNotFoundError:
                pass
            except OSError as error:
                _LOG.warning("Cache stat failed (%s); saving anyway", type(error).__name__)
        payload = {
            "server_id": server_id,
            "metric_type": metric_type,
            "fetched_at": time.time(),
            "records": records,
            "errors": errors or [],
            "digest": digest,
            "blob_xmins": blob_xmins or {},
        }
        try:
            write_secret_file(
                path,
                json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n",
                harden_parent=True,
            )
            if started_at is not None:
                os.utime(path, (started_at, started_at))
        except OSError as error:
            _LOG.warning("Cache write failed (%s); continuing uncached", type(error).__name__)

    def expire(self, metric_type: str | None) -> None:
        """Make a kind's cache stale without discarding it (review V6).

        Called the moment a write is committed. The TTL path (`load`) then
        misses and the next read goes to the cloud; the digest path
        (`load_persisted`) keeps its rows and versions, so that read still
        fetches only what changed. Before this, a write whose re-read failed
        left the pre-write snapshot to answer plain reads for the whole TTL.
        """
        if not self.enabled:
            return
        path = self._path(metric_type)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        except (OSError, json.JSONDecodeError) as error:
            _LOG.warning("Cache expire could not read %s (%s); removing it", path.name, type(error).__name__)
            try:
                path.unlink()
            except OSError:
                _LOG.warning("Cache expire could not remove %s", path.name)
            return
        if not isinstance(raw, dict):
            return
        raw["fetched_at"] = 0
        try:
            write_secret_file(path, json.dumps(raw, separators=(",", ":"), sort_keys=True) + "\n", harden_parent=True)
        except OSError as error:
            _LOG.warning("Cache expire failed (%s)", type(error).__name__)

    def clear(self) -> None:
        """Drop every cached record file (used when (re)binding).

        `records-*`, not `records-*.json`: the narrower glob could not see the
        `.tmp-<pid>-<hex>` files an interrupted write leaves, so re-binding to a
        NEW account left the previous account's decrypted health data on disk —
        precisely what the rebind call site clears the cache to prevent. The
        directory holds nothing else.
        """

        try:
            entries = list(self.directory.glob("records-*"))
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
