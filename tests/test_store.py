from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from vaultbeat_mcp_local.crypto import generate_x25519_keypair
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

    # A REAL keypair, not two copies of the same placeholder. load() derives the
    # public key from the private one and refuses a pair that disagrees, because
    # a mismatch means two identities got crossed and nothing will decrypt — so a
    # fixture whose "public key" equals its private key no longer represents any
    # config that could exist.
    legacy_private, legacy_public = generate_x25519_keypair()
    legacy_payload = {
        "api_base_url": "https://wjpnyxglgtmtgjuuhwru.supabase.co/functions/v1",
        "server_name": "Old Mac",
        "private_key_base64": legacy_private,
        "public_key_base64": legacy_public,
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


# ---------------------------------------------------------------------------
# ensure_initialized — the private key is the identity, so losing it is the
# one unrecoverable failure in this codebase (P0-g)
# ---------------------------------------------------------------------------


def test_ensure_initialized_recovers_the_identity_when_only_the_config_is_missing(
    tmp_path: Path, fake_keychain: dict[tuple[str, str], str]
) -> None:
    """Losing config.json must NOT mint a new keypair while the old private key
    is still in the Keychain.

    This happened on fino on 2026-08-10: a new key was minted, `save()`
    overwrote the Keychain entry, and the 8721 envelopes sealed to the previous
    public key became unreadable by anyone, permanently — while the phone kept
    adding to them. Invariant 54 (a) covers destroying a TOKEN, which a re-bind
    recovers; this destroys the KEY, which nothing recovers.
    """

    config_path = tmp_path / "config.json"
    store = ConfigStore(config_path)
    original = store.ensure_initialized(server_name="fino")

    config_path.unlink()
    assert _keychain_username(config_path) is not None
    assert (_KEYCHAIN_SERVICE, _keychain_username(config_path)) in fake_keychain

    recovered = ConfigStore(config_path).ensure_initialized(server_name="fino")

    assert recovered.private_key_base64 == original.private_key_base64
    assert recovered.public_key_base64 == original.public_key_base64, (
        "a new public key here orphans every envelope already sealed to the old one"
    )
    assert (
        fake_keychain[(_KEYCHAIN_SERVICE, _keychain_username(config_path))]
        == original.private_key_base64
    )


def test_ensure_initialized_still_mints_when_the_key_is_gone_too(
    tmp_path: Path, fake_keychain: dict[tuple[str, str], str]
) -> None:
    """The opposite direction: recovery must not become "never mint a new key".

    A genuinely fresh install — no config, no Keychain entry — still has to get
    an identity, otherwise the fix for the case above would brick first-run.
    """

    config_path = tmp_path / "config.json"
    original = ConfigStore(config_path).ensure_initialized()

    config_path.unlink()
    fake_keychain.clear()

    fresh = ConfigStore(config_path).ensure_initialized()

    assert fresh.public_key_base64 != original.public_key_base64
    assert fresh.private_key_base64


def test_ensure_initialized_gives_a_different_config_path_its_own_identity(
    tmp_path: Path, fake_keychain: dict[tuple[str, str], str]
) -> None:
    """Recovery is keyed on the config PATH, which is what makes a second
    binding profile (e.g. the status probe) possible at all — it must not
    inherit the default install's key just because one exists."""

    first = ConfigStore(tmp_path / "a" / "config.json").ensure_initialized()
    second = ConfigStore(tmp_path / "b" / "config.json").ensure_initialized()

    assert first.public_key_base64 != second.public_key_base64


# ---------------------------------------------------------------------------
# The public/private consistency check — added 2026-08-18 after independent
# verification found it had ZERO coverage in the direction that matters:
# deleting the whole comparison from store.py left all 266 tests green.
#
# The reverse direction was already pinned by accident (inverting the operator
# reddens ~117 tests, because "a normal config must load" is asserted
# everywhere), so what was missing is proof that it FIRES. Without that, the
# next person who finds the block redundant deletes it and CI says nothing —
# the same shape as Invariant 58.
# ---------------------------------------------------------------------------


def test_a_keyring_key_that_disagrees_with_the_stored_public_key_is_refused(
    tmp_path: Path, fake_keychain: dict[tuple[str, str], str]
) -> None:
    """Two identities crossed must fail loudly rather than half-work.

    Reachable in practice: lose config.json while the keyring is unreachable,
    let the file fallback mint a new identity, then restore keyring access —
    now the keyring key (older, higher priority) is paired with the newer
    stored public key. Before this check the mismatch was silent, and the
    damage was invisible: bind sends the STORED public key while decryption
    uses the private one, so pairing succeeds and nothing ever decrypts.
    """
    config_path = tmp_path / "config.json"
    store = ConfigStore(config_path)
    store.ensure_initialized()

    # Swap the keyring's key for a different, perfectly valid one.
    other_private, _ = generate_x25519_keypair()
    username = _keychain_username(config_path)
    fake_keychain[(_KEYCHAIN_SERVICE, username)] = other_private

    with pytest.raises(ConfigError) as caught:
        store.load()
    assert "does not match" in str(caught.value)


def test_an_unusable_private_key_is_reported_as_such_not_as_a_mismatch(
    tmp_path: Path, fake_keychain: dict[tuple[str, str], str]
) -> None:
    """A corrupted key and two crossed identities need different answers.

    Both surface at the same line; only the second is about identity. Telling
    someone their keys belong to different identities when the real problem is
    a truncated file sends them looking for a second install that never existed.
    """
    config_path = tmp_path / "config.json"
    store = ConfigStore(config_path)
    store.ensure_initialized()

    username = _keychain_username(config_path)
    fake_keychain[(_KEYCHAIN_SERVICE, username)] = "not-valid-base64!!!"

    with pytest.raises(ConfigError) as caught:
        store.load()
    assert "present but unusable" in str(caught.value)


def test_a_config_with_no_reachable_key_names_all_three_locations(
    tmp_path: Path, fake_keychain: dict[tuple[str, str], str]
) -> None:
    """The error must name every place looked, and must not suggest deleting.

    'missing key material' used to name config.json and nothing else, so the
    reader's next move was to delete it — the one irreversible action here,
    since the key lives outside that file and deleting mints a new identity.
    """
    config_path = tmp_path / "config.json"
    store = ConfigStore(config_path)
    store.ensure_initialized()

    fake_keychain.clear()  # key gone from the keyring, and no identity.key exists

    with pytest.raises(ConfigError) as caught:
        store.load()
    message = str(caught.value)
    assert "VAULTBEAT_PRIVATE_KEY" in message
    assert "system keyring" in message
    assert "identity.key" in message
    assert "DO NOT DELETE" in message
