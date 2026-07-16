from __future__ import annotations

import json
import logging
import os
import secrets
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import keyring

from tether_mcp_local.crypto import generate_x25519_keypair, public_key_from_private


DEFAULT_API_BASE_URL = "https://wjpnyxglgtmtgjuuhwru.supabase.co/functions/v1"
CONFIG_ENV = "TETHER_MCP_CONFIG"

# Keychain service name — stable namespace that matches the macOS bundle-ID convention.
# Changing this string would orphan existing Keychain entries, so treat it as frozen.
_KEYCHAIN_SERVICE = "com.jiayuan.tether.mcp-local"

logger = logging.getLogger(__name__)


class ConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class LocalServerConfig:
    server_name: str
    api_base_url: str
    private_key_base64: str
    public_key_base64: str
    poll_id: str | None = None
    server_id: str | None = None
    server_token: str | None = None
    http_token: str | None = None
    # Owner identity carried through the bind handshake (bind handshake), so the
    # agent write path can seal an owner_user envelope (owner can read its own AI's writes;
    # addressed by owner_user_id, sealed to owner_public_key_base64) and stamp a valid
    # source_device_id. All nullable: a legacy/partner-less bind, or a user with no
    # identity/device row, simply binds without them.
    owner_user_id: str | None = None
    owner_public_key_base64: str | None = None
    owner_device_id: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    bound_at: str | None = None
    last_sync_at: str | None = None

    @property
    def is_bound(self) -> bool:
        return bool(self.server_id and self.server_token)

    def redacted(self) -> dict[str, Any]:
        payload = asdict(self)
        if payload.get("private_key_base64"):
            payload["private_key_base64"] = "<redacted>"
        if payload.get("server_token"):
            payload["server_token"] = "<redacted>"
        if payload.get("http_token"):
            payload["http_token"] = "<redacted>"
        return payload


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def write_secret_file(path: Path, text: str, *, harden_parent: bool = False) -> None:
    """Atomically write `text` to `path` as an owner-only (0600) file.

    Opens with O_CREAT|O_EXCL at mode 0600 so the bytes never momentarily exist
    at a looser umask mode (closes the write-then-chmod TOCTOU). The temp name
    is unique PER WRITER (pid + random) so concurrent processes writing the
    same target (a CLI run racing the MCP server on config.json or a cache
    file) cannot delete each other's in-flight temp — each os.replace publishes
    its own complete bytes, last writer wins. When `harden_parent` is set the
    immediate parent directory is forced to 0700 — only safe for directories we
    own (the config dir), never for a user-chosen output path whose directory
    mode we must not mutate.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    if harden_parent:
        os.chmod(path.parent, 0o700)
    temp_path = path.with_suffix(path.suffix + f".tmp-{os.getpid()}-{secrets.token_hex(4)}")
    fd = os.open(temp_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise
    os.replace(temp_path, path)
    os.chmod(path, 0o600)


def _keychain_username(config_path: Path) -> str:
    """Stable per-config Keychain username.

    Keyed on the resolved config file path so two different ``--config`` paths
    never collide in the same Keychain service namespace.  The path is fixed
    from the moment the config is first initialised, so the username is stable
    across reloads.
    """
    return f"private_key:{config_path.resolve()}"


def _keychain_store(config_path: Path, private_key_base64: str) -> None:
    """Write *private_key_base64* into the macOS login Keychain.

    Raises :class:`ConfigError` (never swallows) on any Keychain failure so
    callers always know whether the key was persisted.
    """
    username = _keychain_username(config_path)
    try:
        keyring.set_password(_KEYCHAIN_SERVICE, username, private_key_base64)
    except Exception as exc:
        logger.error(
            "Keychain write failed for service=%r username=%r: %s: %s",
            _KEYCHAIN_SERVICE,
            username,
            type(exc).__name__,
            exc,
        )
        raise ConfigError(
            f"Failed to store private key in Keychain ({type(exc).__name__}): {exc}"
        ) from exc


def _keychain_load(config_path: Path) -> str | None:
    """Read the private key from the Keychain; returns *None* if absent.

    Raises :class:`ConfigError` on unexpected Keychain errors so callers see
    the failure rather than silently treating it as a missing key.
    """
    username = _keychain_username(config_path)
    try:
        return keyring.get_password(_KEYCHAIN_SERVICE, username)
    except Exception as exc:
        logger.error(
            "Keychain read failed for service=%r username=%r: %s: %s",
            _KEYCHAIN_SERVICE,
            username,
            type(exc).__name__,
            exc,
        )
        raise ConfigError(
            f"Failed to read private key from Keychain ({type(exc).__name__}): {exc}"
        ) from exc


def default_config_path() -> Path:
    override = os.getenv(CONFIG_ENV, "").strip()
    if override:
        return Path(override).expanduser()
    return Path.home() / ".tether" / "mcp-local" / "config.json"


class ConfigStore:
    def __init__(self, path: Path | None = None):
        self.path = path or default_config_path()

    def load(self) -> LocalServerConfig | None:
        if not self.path.exists():
            return None

        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ConfigError(f"Invalid config JSON at {self.path}") from error

        # --- legacy migration -------------------------------------------------
        # Older configs stored the private key as plaintext JSON.  Detect that,
        # move it to the Keychain, then rewrite the JSON file without the key.
        legacy_key = str(raw.get("private_key_base64", "")).strip()
        if legacy_key:
            logger.info(
                "Migrating plaintext private_key_base64 from %s to Keychain", self.path
            )
            _keychain_store(self.path, legacy_key)
            del raw["private_key_base64"]
            # Preserve the existing updated_at; we only strip the key field.
            write_secret_file(
                self.path,
                json.dumps(raw, indent=2, sort_keys=True) + "\n",
                harden_parent=True,
            )
        # ----------------------------------------------------------------------

        private_key_base64 = _keychain_load(self.path) or ""
        public_key_base64 = str(raw.get("public_key_base64", "")).strip()
        if private_key_base64 and not public_key_base64:
            public_key_base64 = public_key_from_private(private_key_base64)

        if not private_key_base64 or not public_key_base64:
            raise ConfigError(f"Config at {self.path} is missing key material")

        return LocalServerConfig(
            server_name=str(raw.get("server_name", "Local AI Server")).strip() or "Local AI Server",
            api_base_url=str(raw.get("api_base_url", DEFAULT_API_BASE_URL)).rstrip("/"),
            private_key_base64=private_key_base64,
            public_key_base64=public_key_base64,
            poll_id=raw.get("poll_id"),
            server_id=raw.get("server_id"),
            server_token=raw.get("server_token"),
            http_token=raw.get("http_token"),
            owner_user_id=raw.get("owner_user_id"),
            owner_public_key_base64=raw.get("owner_public_key_base64"),
            owner_device_id=raw.get("owner_device_id"),
            created_at=raw.get("created_at"),
            updated_at=raw.get("updated_at"),
            bound_at=raw.get("bound_at"),
            last_sync_at=raw.get("last_sync_at"),
        )

    def save(self, config: LocalServerConfig) -> None:
        # Persist the private key to the Keychain first so it is never written
        # to disk, but only when it actually CHANGED: a Keychain write is a
        # securityd XPC round trip and an auth-prompt hazard after codesign
        # churn, and save() now sits on the query path (update(last_sync_at=…)
        # on every cloud fetch). Reads are cheap and prompt-free by comparison.
        if _keychain_load(self.path) != config.private_key_base64:
            _keychain_store(self.path, config.private_key_base64)

        payload = asdict(config)
        payload["updated_at"] = now_iso()
        # Strip the private key from the on-disk representation.
        payload.pop("private_key_base64", None)
        write_secret_file(
            self.path,
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            harden_parent=True,
        )

    def ensure_initialized(
        self,
        *,
        server_name: str = "Local AI Server",
        api_base_url: str = DEFAULT_API_BASE_URL,
    ) -> LocalServerConfig:
        existing = self.load()
        if existing:
            return existing

        private_key_base64, public_key_base64 = generate_x25519_keypair()
        now = now_iso()
        config = LocalServerConfig(
            server_name=server_name.strip() or "Local AI Server",
            api_base_url=api_base_url.rstrip("/"),
            private_key_base64=private_key_base64,
            public_key_base64=public_key_base64,
            created_at=now,
            updated_at=now,
        )
        self.save(config)
        return config

    def update(self, **changes: Any) -> LocalServerConfig:
        current = self.load()
        if not current:
            raise ConfigError("Local MCP server is not initialized")
        next_config = LocalServerConfig(**{**asdict(current), **changes})
        self.save(next_config)
        return next_config

    def ensure_http_token(self) -> str:
        current = self.load()
        if not current:
            raise ConfigError("Local MCP server is not initialized")
        if current.http_token:
            return current.http_token
        token = secrets.token_urlsafe(32)
        self.update(http_token=token)
        return token

    def require_bound(self) -> LocalServerConfig:
        current = self.load()
        if not current:
            raise ConfigError("Local MCP server is not initialized")
        if not current.is_bound:
            raise ConfigError("Local MCP server is not bound; run `tether-mcp-local bind` first")
        return current
