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

from vaultbeat_mcp_local.crypto import generate_x25519_keypair, public_key_from_private


DEFAULT_API_BASE_URL = "https://wjpnyxglgtmtgjuuhwru.supabase.co/functions/v1"
CONFIG_ENV = "VAULTBEAT_MCP_CONFIG"

# Slug-free id-only form — survives any future App Store rename (the slugged
# URL rotted once already when tether became vaultbeat).
APP_STORE_URL = "https://apps.apple.com/app/id6759241985"

# The tail of every "not paired" refusal, written for the SERVER-FIRST user:
# someone who found this package on the MCP Registry or PyPI, ran
# `claude mcp add`, and asked their agent a question — without ever hearing
# that an iPhone app exists. The old message ("call `vaultbeat_start_binding`
# then `vaultbeat_poll_binding`") sent that person's agent off to print a QR
# code they had nothing to scan with; the two-sided structure only surfaced
# in the timeout text five minutes later. So the FIRST sentence states the
# structure, the app link comes before any tool name, and the demo mode this
# package already ships is offered at the exact moment it is most useful.
# Every word here is client-side (Anti-pattern 23) — nothing is echoed from
# any server.
PAIRING_GUIDANCE = (
    "Pairing needs the free Vaultbeat iOS app on an iPhone — the app is where "
    "the health data comes from, and scanning a QR code with it is the only "
    f"way to authorize this server. Install it from {APP_STORE_URL}, then run "
    "`uvx vaultbeat-mcp@latest bind` in a real terminal and scan the QR it "
    "prints (in the app: Settings → Data & AI → Connect an AI server). The "
    "`vaultbeat_start_binding` / `vaultbeat_poll_binding` tools do the same, "
    "but a QR relayed through an agent's output often does not render — the "
    "terminal command is the reliable path. No iPhone or no app yet? Restart "
    "this MCP server with VAULTBEAT_DEMO=1 in its environment (CLI flag: "
    "--demo) to explore every read tool on synthetic data."
)
# Pre-rename env var, honored as a fallback so existing setups keep working.
_LEGACY_CONFIG_ENV = "TETHER_MCP_CONFIG"

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
    # ISO8601 trial deadline the cloud reported at the MOST RECENT successful
    # pairing — a bind-time SNAPSHOT, not a live entitlement. None means "no
    # deadline was reported then" (grandfathered / already paid / an edge older
    # than 2026-08-17), and a stored value never updates when the user later
    # subscribes or buys lifetime in the iOS app: the server enforces the real
    # rule on every request, this field only lets status/doctor say what was
    # known at pairing instead of nothing at all (vb-016).
    trial_ends_at: str | None = None
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


PRIVATE_KEY_ENV = "VAULTBEAT_PRIVATE_KEY"


def identity_file_path(config_path: Path) -> Path:
    """Where the private key lands when no system keyring is available.

    🔴 A SEPARATE file, never a field inside config.json, and that separation is
    a real security boundary rather than tidiness:

        server_token alone  → can download this user's ciphertext, cannot read it
        private key alone   → can decrypt, but has nothing to decrypt
        both together       → plaintext health data

    config.json already holds `server_token` and `http_token`, and it is the file
    people cat when something breaks, paste into a bug report, and copy when they
    migrate a machine. Putting the decryption key in it collapses two independent
    secrets into one, so a single careless `cat` becomes total exposure.
    """
    return config_path.parent / "identity.key"


def _keyring_unavailable(exc: BaseException) -> bool:
    """True only when the platform has NO usable keyring backend at all.

    🔴 The distinction is the whole safety of the fallback. "No backend exists"
    is a fact about the machine and warrants writing a file; anything else — a
    locked keychain, a denied prompt, a D-Bus hiccup — is a transient or
    permission failure, and silently downgrading THOSE to a plaintext file on a
    Mac would be a security incident dressed as fault tolerance.

    Matched by name rather than by import because the exception moved modules
    across keyring versions and a wrong import path would make every failure
    look downgradeable — the dangerous direction.
    """
    return type(exc).__name__ == "NoKeyringError"


def _keyring_env_hint() -> str:
    """What an AGENT should do about a missing keyring, in words it can act on.

    keyring's own message is "No recommended backend was available. Install a
    recommended 3rd party backend package; or, ..." — which sent Hermes off to
    search the web, try `keyring.backends.null`, and pip-install `keyrings.alt`,
    three separate times over a week (2026-08-10 / 08-12 / 08-17) without ever
    fixing it. The actual cause on that machine was two missing environment
    variables, which its message never mentions.

    So this names the most common cause FIRST and gives the exact command to
    recover it, because the reader is usually a program that cannot inspect its
    own launch environment.
    """
    return (
        "\n\nIF YOU ARE AN AI AGENT: this is usually NOT a missing package.\n"
        "The most common cause on Linux is that this process was started without\n"
        "the desktop session's environment, so it cannot reach the system keyring\n"
        "even though one is running. Check from an interactive shell:\n"
        "    env | grep -E 'DBUS_SESSION_BUS_ADDRESS|XDG_RUNTIME_DIR'\n"
        "and add both to this MCP server's launch environment.\n"
        "On a headless server with no keyring at all, do nothing: Vaultbeat\n"
        "stores the key in a 0600 identity.key file automatically when no\n"
        "backend exists.\n"
        f"Do NOT set PYTHON_KEYRING_BACKEND to the null backend to silence this.\n"
        "That backend accepts writes and stores nothing, which is worse than the\n"
        "error you are trying to remove.\n"
        f"({PRIVATE_KEY_ENV} also works, but only if you already hold the key from\n"
        "systemd-creds, a vault or a KMS — there is no command that prints it.)"
    )


def _key_lookup_diagnosis(config_path: Path) -> str:
    """Report what was actually found in each of the three key locations.

    Exists because "missing key material" named only config.json, and the
    reader's next move — delete it and start over — is the single most
    destructive action available here: the key usually lives OUTSIDE that file,
    so removing it mints a new identity and orphans every envelope already
    sealed to the old one (see `ensure_initialized`, and the 8721 envelopes it
    documents).

    Reporting the LAYER, never the value. And every probe is individually
    guarded: this runs on the error path, so a failure inside the diagnosis
    must never replace the error it is trying to explain.
    """
    lines: list[str] = []
    keyring_is_a_dead_end = False

    injected = ""
    try:
        injected = os.getenv(PRIVATE_KEY_ENV, "").strip()
    except Exception:  # pragma: no cover - os.getenv does not raise in practice
        pass
    lines.append(
        f"  {PRIVATE_KEY_ENV}: {'set' if injected else 'not set'}"
    )

    try:
        backend = type(keyring.get_keyring()).__name__
    except Exception as exc:
        backend = f"could not be determined ({type(exc).__name__})"
    try:
        stored = keyring.get_password(_KEYCHAIN_SERVICE, _keychain_username(config_path))
        if stored:
            lines.append(f"  system keyring [{backend}]: holds a key")
        else:
            lines.append(f"  system keyring [{backend}]: no entry for this config path")
            keyring_is_a_dead_end = True
    except Exception as exc:
        lines.append(f"  system keyring [{backend}]: unreachable ({type(exc).__name__})")
        keyring_is_a_dead_end = True

    try:
        identity_path = identity_file_path(config_path)
        lines.append(
            f"  {identity_path}: {'present' if identity_path.is_file() else 'not found'}"
        )
    except Exception as exc:  # pragma: no cover - path arithmetic does not raise
        lines.append(f"  identity.key: could not be checked ({type(exc).__name__})")

    report = "The private key was looked for in all three of its locations:\n" + "\n".join(lines)
    if keyring_is_a_dead_end:
        report += _keyring_env_hint()
    return report


def _keychain_store(config_path: Path, private_key_base64: str) -> None:
    """Persist the private key, preferring the OS keyring.

    Order: system keyring → 0600 file. `PRIVATE_KEY_ENV` is deliberately NOT
    written to — an externally-injected key belongs to whoever injected it, and
    copying it onto disk would defeat the reason they injected it.

    🔴 The write is VERIFIED BY READING IT BACK, and that is not belt-and-braces
    — it is the only thing standing between a headless install and silent key
    loss. `keyring.backends.null.Keyring` implements every operation as a bare
    `pass`: `set_password` returns None and raises nothing, so a
    "succeeded → return" check hands back success while storing the key
    NOWHERE. The file fallback below never runs, `save()` then strips the key
    from config.json as designed, and the freshly minted private key ceases to
    exist — measured 2026-08-17 on a clean config: bind reports success and the
    directory contains only config.json, with no identity.key.

    That backend is not exotic. It is what `PYTHON_KEYRING_BACKEND=...null...`
    selects, which is exactly what someone sets to silence a keyring error on a
    server — the machine most likely to have no keyring is the one most likely
    to reach this branch.

    So the predicate is "can I read back what I just wrote", not "did the call
    raise". A write that reports success is only an echo of the request; the
    read is the evidence.
    """
    username = _keychain_username(config_path)
    try:
        keyring.set_password(_KEYCHAIN_SERVICE, username, private_key_base64)
        if keyring.get_password(_KEYCHAIN_SERVICE, username) == private_key_base64:
            return
        logger.warning(
            "The keyring backend accepted the private key but did not store it "
            "(read-back returned nothing, or a different value). Falling back to a "
            "0600 file — this is what the null backend does."
        )
    except Exception as exc:
        if not _keyring_unavailable(exc):
            logger.error(
                "Keychain write failed for service=%r username=%r: %s: %s",
                _KEYCHAIN_SERVICE, username, type(exc).__name__, exc,
            )
            raise ConfigError(
                f"Failed to store private key in Keychain ({type(exc).__name__}): {exc}"
                f"{_keyring_env_hint()}"
            ) from exc

    path = identity_file_path(config_path)
    logger.warning(
        "No system keyring available; storing the private key at %s (mode 0600). "
        "Anyone who can read that file can decrypt this account's health data.",
        path,
    )
    write_secret_file(path, private_key_base64 + "\n", harden_parent=True)


def _keychain_load(config_path: Path) -> str | None:
    """Read the private key: environment → system keyring → 0600 file.

    The env var wins so an operator can hand the key in from systemd-creds, a
    vault, or a KMS without us knowing or caring where it came from — the one
    escape hatch for people who will not accept a key on disk.
    """
    injected = os.getenv(PRIVATE_KEY_ENV, "").strip()
    if injected:
        return injected

    username = _keychain_username(config_path)
    try:
        stored = keyring.get_password(_KEYCHAIN_SERVICE, username)
        if stored:
            return stored
    except Exception as exc:
        if not _keyring_unavailable(exc):
            logger.error(
                "Keychain read failed for service=%r username=%r: %s: %s",
                _KEYCHAIN_SERVICE, username, type(exc).__name__, exc,
            )
            raise ConfigError(
                f"Failed to read private key from Keychain ({type(exc).__name__}): {exc}"
                f"{_keyring_env_hint()}"
            ) from exc

    path = identity_file_path(config_path)
    if path.is_file():
        return path.read_text(encoding="utf-8").strip() or None
    return None


def default_config_path() -> Path:
    override = os.getenv(CONFIG_ENV, "").strip() or os.getenv(_LEGACY_CONFIG_ENV, "").strip()
    if override:
        return Path(override).expanduser()
    # Frozen at the pre-rename path: the Keychain username embeds the resolved
    # config path (see _keychain_username), so moving this directory orphans
    # both the bound config AND its private-key Keychain entry for every
    # existing installation. Treat like _KEYCHAIN_SERVICE — never rename.
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
        if not private_key_base64:
            raise ConfigError(
                f"Config at {self.path} exists, but its private key could not be read.\n"
                f"{_key_lookup_diagnosis(self.path)}\n\n"
                f"DO NOT DELETE {self.path} to start over. The private key is not "
                "stored in it, so deleting it does not clear a bad key — it mints a "
                "NEW identity, and every record already encrypted for the old one "
                "becomes permanently unreadable."
            )

        # Always derive, then compare — never trust the stored value on its own.
        # These can disagree, and when they do everything downstream is quietly
        # wrong: the QR payload sent at bind time comes from the stored public
        # key while decryption uses this private key, so pairing "succeeds" and
        # then nothing can ever be decrypted. The way in is mundane — lose
        # config.json while the keyring is unreachable, let the file fallback
        # mint a new identity, then restore keyring access: now the keyring key
        # (older, higher priority) is paired with the newer stored public key.
        try:
            derived_public_key = public_key_from_private(private_key_base64)
        except Exception as error:
            raise ConfigError(
                f"The private key for {self.path} is present but unusable "
                f"({type(error).__name__}). It may be truncated or corrupted.\n"
                f"{_key_lookup_diagnosis(self.path)}"
            ) from error

        stored_public_key = str(raw.get("public_key_base64", "")).strip()
        if stored_public_key and stored_public_key != derived_public_key:
            raise ConfigError(
                f"The public key recorded in {self.path} does not match the private "
                "key currently in use — they belong to two different identities, so "
                "nothing sealed for either one can be decrypted reliably.\n"
                f"{_key_lookup_diagnosis(self.path)}\n\n"
                "This usually means a second identity was created while the first "
                "key was unreachable. Whichever identity the cloud holds envelopes "
                "for is the one worth keeping; re-pair this machine once you know "
                "which private key that is."
            )
        public_key_base64 = derived_public_key

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
            trial_ends_at=raw.get("trial_ends_at"),
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

        # config.json is gone. Before minting anything, look for THIS config
        # path's private key in the Keychain — because `save()` below would
        # overwrite it, and that write is the single most destructive thing in
        # this codebase.
        #
        # 🔴 What it costs when it goes wrong, measured: on 2026-08-10 this
        # function ran on fino with the Keychain still holding the key behind
        # `a3c145ff9279`. It minted `2be19c3d72d0`, saved, and the old private
        # key ceased to exist anywhere on earth. The 8721 envelopes sealed to
        # the old one are unreadable by anyone, permanently — and her phone kept
        # adding to them for a day afterwards. Invariant 54 (a) is about
        # destroying a TOKEN (a re-bind gets it back); this destroys the KEY.
        #
        # A missing config with a surviving Keychain entry is not an
        # uninitialised install — it is an install that lost one file, and the
        # thing that actually defines its identity is still here. So recover it.
        # The alternative that was considered and rejected: warn and refuse.
        # Refusing leaves a user stuck behind an error they cannot action, while
        # recovery is what they would have asked for in every case — the private
        # key IS the identity, and re-deriving the public key from it is exact.
        #
        # ⚠️ This is deliberately keyed on the config PATH (`_keychain_username`
        # embeds it), so a `--config` pointing somewhere new still gets a fresh
        # identity. Only the path that owned the key can reclaim it.
        recovered_private_key = _keychain_load(self.path)
        if recovered_private_key:
            logger.warning(
                "Config %s is missing but its private key is still in the Keychain — "
                "recovering the existing identity instead of generating a new one. "
                "Minting a new keypair here would permanently orphan every envelope "
                "already sealed to the old public key.",
                self.path,
            )
            private_key_base64 = recovered_private_key
            public_key_base64 = public_key_from_private(recovered_private_key)
        else:
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
        # This raise is the FIRST error a server-first user (MCP Registry /
        # PyPI → `claude mcp add` → first question) ever sees, so both branches
        # carry the full two-sided-structure guidance — see PAIRING_GUIDANCE.
        current = self.load()
        if not current:
            raise ConfigError(
                f"This Vaultbeat MCP server is not paired with anyone's data yet. {PAIRING_GUIDANCE}"
            )
        if not current.is_bound:
            raise ConfigError(
                f"This Vaultbeat MCP server has keys but no iPhone has authorized it yet. {PAIRING_GUIDANCE}"
            )
        return current
