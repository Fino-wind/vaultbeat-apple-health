from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from vaultbeat_mcp_local.store import (
    ConfigError,
    ConfigStore,
    LocalServerConfig,
    _KEYCHAIN_SERVICE,
    _keychain_username,
    write_secret_file,
)


# ---------------------------------------------------------------------------
# write_secret_file helpers — no Keychain involvement
# ---------------------------------------------------------------------------


def test_write_secret_file_is_owner_only(tmp_path: Path) -> None:
    target = tmp_path / "secret.txt"

    write_secret_file(target, "hello")

    assert target.read_text(encoding="utf-8") == "hello"
    assert stat.S_IMODE(target.stat().st_mode) == 0o600


def test_write_secret_file_hardens_parent_only_when_asked(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "secret.txt"

    write_secret_file(target, "x", harden_parent=True)

    assert stat.S_IMODE(target.parent.stat().st_mode) == 0o700


def test_write_secret_file_overwrites_existing(tmp_path: Path) -> None:
    # O_EXCL guards the temp file, not the target, so re-writing must still succeed.
    target = tmp_path / "secret.txt"

    write_secret_file(target, "first")
    write_secret_file(target, "second")

    assert target.read_text(encoding="utf-8") == "second"
    assert stat.S_IMODE(target.stat().st_mode) == 0o600


# ---------------------------------------------------------------------------
# ConfigStore — all tests that touch save/load must use fake_keychain
# ---------------------------------------------------------------------------


def test_ensure_http_token_generates_then_is_idempotent(
    tmp_path: Path, fake_keychain: dict[tuple[str, str], str]
) -> None:
    store = ConfigStore(tmp_path / "config.json")
    store.ensure_initialized()

    first = store.ensure_http_token()

    assert first
    loaded = store.load()
    assert loaded is not None
    assert loaded.http_token == first
    assert store.ensure_http_token() == first  # idempotent: no second token minted


def test_ensure_http_token_requires_initialized_config(
    tmp_path: Path, fake_keychain: dict[tuple[str, str], str]
) -> None:
    store = ConfigStore(tmp_path / "config.json")

    with pytest.raises(ConfigError):
        store.ensure_http_token()


def test_redacted_masks_http_token() -> None:
    config = LocalServerConfig(
        server_name="Mac Studio",
        api_base_url="https://api.test",
        private_key_base64="priv",
        public_key_base64="pub",
        http_token="super-secret-http-token",
    )

    redacted = config.redacted()

    assert redacted["http_token"] == "<redacted>"
    assert "super-secret-http-token" not in str(redacted)


# ---------------------------------------------------------------------------
# Keychain-specific behaviour
# ---------------------------------------------------------------------------


def test_fresh_save_stores_key_in_keychain_not_json(
    tmp_path: Path, fake_keychain: dict[tuple[str, str], str]
) -> None:
    """A brand-new init must never write the private key to the JSON file."""
    config_path = tmp_path / "config.json"
    store = ConfigStore(config_path)

    config = store.ensure_initialized()

    # Key must be in the fake Keychain.
    username = _keychain_username(config_path)
    assert (_KEYCHAIN_SERVICE, username) in fake_keychain
    assert fake_keychain[(_KEYCHAIN_SERVICE, username)] == config.private_key_base64

    # Key must NOT appear in the on-disk JSON.
    on_disk = json.loads(config_path.read_text(encoding="utf-8"))
    assert "private_key_base64" not in on_disk


def test_legacy_json_key_is_migrated_to_keychain(
    tmp_path: Path, fake_keychain: dict[tuple[str, str], str]
) -> None:
    """Loading a legacy config that still has plaintext private_key_base64
    must migrate it to the Keychain and strip the field from the JSON file."""
    config_path = tmp_path / "config.json"

    # Write a legacy-style config with the key in the JSON.
    legacy_payload = {
        "api_base_url": "https://wjpnyxglgtmtgjuuhwru.supabase.co/functions/v1",
        "server_name": "Old Mac",
        "private_key_base64": "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=",
        "public_key_base64": "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=",
    }
    write_secret_file(config_path, json.dumps(legacy_payload) + "\n")

    store = ConfigStore(config_path)
    loaded = store.load()

    assert loaded is not None
    # The key is now in the Keychain.
    username = _keychain_username(config_path)
    assert fake_keychain.get((_KEYCHAIN_SERVICE, username)) == legacy_payload["private_key_base64"]

    # The JSON file no longer contains the plaintext key.
    on_disk = json.loads(config_path.read_text(encoding="utf-8"))
    assert "private_key_base64" not in on_disk

    # The loaded config carries the correct key.
    assert loaded.private_key_base64 == legacy_payload["private_key_base64"]


def test_round_trip_load_returns_same_key(
    tmp_path: Path, fake_keychain: dict[tuple[str, str], str]
) -> None:
    """save() → load() must return an identical private key."""
    config_path = tmp_path / "config.json"
    store = ConfigStore(config_path)

    original = store.ensure_initialized()
    loaded = store.load()

    assert loaded is not None
    assert loaded.private_key_base64 == original.private_key_base64


def test_redacted_masks_private_key(
    tmp_path: Path, fake_keychain: dict[tuple[str, str], str]
) -> None:
    """The plaintext private key must not appear in redacted() output."""
    config_path = tmp_path / "config.json"
    store = ConfigStore(config_path)
    config = store.ensure_initialized()

    redacted = config.redacted()

    assert redacted["private_key_base64"] == "<redacted>"
    assert config.private_key_base64 not in str(redacted)
