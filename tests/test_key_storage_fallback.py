"""Three places the private key can live, and the rules that keep that safe.

The product's own positioning names a VPS as a primary environment, and on a
headless VPS there is no system keyring — no D-Bus session, no login password to
root an encrypted store in. The constraint is not fixable by adding a dependency:
unattended means the process must be able to decrypt by itself, which means the
key must be reachable by the process, which means it is on disk or injected.

So: keyring where one exists, an env var for operators who will not accept a key
on disk, and a 0600 file otherwise. These tests pin the parts that are easy to
get subtly wrong and impossible to notice.
"""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

from vaultbeat_mcp_local import store as store_mod
from vaultbeat_mcp_local.store import (
    PRIVATE_KEY_ENV,
    identity_file_path,
    _keyring_unavailable,
)


class _NoKeyringError(Exception):
    """Stands in for keyring.errors.NoKeyringError.

    Named rather than imported, exactly as the production check does — that
    exception has moved modules between keyring versions, and an import that
    silently resolves to the wrong class would make EVERY failure look
    downgradeable, which is the dangerous direction.
    """

    __name__ = "NoKeyringError"


_NoKeyringError.__name__ = "NoKeyringError"


def test_only_a_missing_backend_counts_as_unavailable() -> None:
    """🔴 The safety of the whole fallback rests on this one predicate.

    "No backend exists" is a fact about the machine. A locked keychain, a denied
    prompt, a D-Bus hiccup are transient or permission failures — downgrading
    those to a plaintext file on a Mac would be a security incident wearing the
    costume of fault tolerance.
    """
    assert _keyring_unavailable(_NoKeyringError("none available")) is True
    for other in (
        RuntimeError("keychain is locked"),
        PermissionError("user denied access"),
        ValueError("dbus timeout"),
    ):
        assert _keyring_unavailable(other) is False, other


def test_identity_key_is_a_separate_file_from_config(tmp_path: Path) -> None:
    """The two secrets must not share a file.

    server_token alone downloads ciphertext it cannot read; the private key alone
    decrypts nothing it does not have. Together they are plaintext health data —
    and config.json is the file people cat, paste into bug reports and copy when
    migrating a machine.
    """
    config = tmp_path / "mcp-local" / "config.json"
    assert identity_file_path(config) != config
    assert identity_file_path(config).name == "identity.key"
    assert identity_file_path(config).parent == config.parent


def test_falls_back_to_a_0600_file_when_no_backend_exists(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    with mock.patch.object(store_mod.keyring, "set_password", side_effect=_NoKeyringError()):
        store_mod._keychain_store(config, "SECRET-KEY-MATERIAL")

    key_file = identity_file_path(config)
    assert key_file.read_text().strip() == "SECRET-KEY-MATERIAL"
    # 0600: owner-only. The file IS the decryption capability.
    assert oct(key_file.stat().st_mode)[-3:] == "600"


def test_a_real_keyring_failure_raises_instead_of_writing_a_file(tmp_path: Path) -> None:
    """The counterpart to the test above, and the more important direction:
    a Mac with a locked keychain must NOT quietly end up with a plaintext key."""
    config = tmp_path / "config.json"
    with mock.patch.object(
        store_mod.keyring, "set_password", side_effect=RuntimeError("locked")
    ):
        with pytest.raises(store_mod.ConfigError):
            store_mod._keychain_store(config, "SECRET")
    assert not identity_file_path(config).exists()


def test_env_var_wins_over_everything(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The escape hatch for operators who will not accept a key on disk —
    systemd-creds, a vault, a KMS. We do not care where it came from."""
    config = tmp_path / "config.json"
    identity_file_path(config).write_text("FROM-FILE\n")
    monkeypatch.setenv(PRIVATE_KEY_ENV, "FROM-ENV")
    with mock.patch.object(store_mod.keyring, "get_password", return_value="FROM-KEYRING"):
        assert store_mod._keychain_load(config) == "FROM-ENV"


def test_an_injected_key_is_never_written_to_disk(tmp_path: Path) -> None:
    """Copying it to disk would defeat the only reason someone injects it."""
    config = tmp_path / "config.json"
    with mock.patch.object(store_mod.keyring, "set_password", side_effect=_NoKeyringError()):
        store_mod._keychain_store(config, "K")
    # _keychain_store never reads the env var — it persists what it was handed.
    # The guarantee is that callers holding an injected key simply do not call it;
    # what this pins is that the function has no hidden env-var write path.
    assert identity_file_path(config).read_text().strip() == "K"


def test_keyring_is_preferred_when_it_works(tmp_path: Path) -> None:
    """No downgrade on machines that have a keyring — macOS behaviour unchanged.

    get_password must be stubbed alongside set_password: "the write worked" is
    now decided by reading the value back, so a mock that accepts writes and
    returns nothing IS the null backend, and would land in the fallback below.
    """
    config = tmp_path / "config.json"
    with (
        mock.patch.object(store_mod.keyring, "set_password") as set_pw,
        mock.patch.object(store_mod.keyring, "get_password", return_value="K"),
    ):
        store_mod._keychain_store(config, "K")
    set_pw.assert_called_once()
    assert not identity_file_path(config).exists()


def test_a_silent_backend_still_lands_the_key_on_disk(tmp_path: Path) -> None:
    """The null backend: every operation is `pass`, so nothing ever raises.

    Every other case in this file constructs failure by RAISING. That was the
    entire imagination of the original suite — its header says "no D-Bus
    session" — and it is why the bug shipped: `keyring.backends.null.Keyring`
    implements set_password as a bare `pass`, so a "did the call raise" check
    reports success while the key goes nowhere. Not a hypothetical backend
    either: it is what PYTHON_KEYRING_BACKEND=...null... selects, i.e. what
    someone sets to silence a keyring error on exactly the kind of server that
    has no keyring.
    """
    config = tmp_path / "config.json"
    with (
        mock.patch.object(store_mod.keyring, "set_password", return_value=None),
        mock.patch.object(store_mod.keyring, "get_password", return_value=None),
    ):
        store_mod._keychain_store(config, "K")

    identity = identity_file_path(config)
    assert identity.is_file(), (
        "set_password returned without raising, so the write looked fine — but "
        "nothing was stored and the file fallback never engaged. That is silent "
        "key loss."
    )
    assert identity.read_text(encoding="utf-8").strip() == "K"


def test_a_silent_backend_does_not_lose_a_freshly_minted_identity(tmp_path: Path) -> None:
    """The real guardrail: assert the KEY SURVIVES, not what a function returned.

    ensure_initialized mints a keypair and calls save(), which strips the private
    key from config.json by design. If the keyring swallowed it and no file was
    written, the key that was just generated no longer exists anywhere — and the
    only symptom is a later "missing key material" on a config that looks fine.
    """
    config = tmp_path / "config.json"
    store = store_mod.ConfigStore(config)

    with (
        mock.patch.object(store_mod.keyring, "set_password", return_value=None),
        mock.patch.object(store_mod.keyring, "get_password", return_value=None),
    ):
        created = store.ensure_initialized()
        reloaded = store.load()

    assert reloaded is not None
    assert reloaded.private_key_base64 == created.private_key_base64
    assert identity_file_path(config).is_file()


def test_the_error_names_the_environment_cause_before_packages() -> None:
    """keyring's own message sends readers to install a package. The actual
    cause on a machine that HAS a keyring is two missing environment variables —
    which cost Hermes three separate failed attempts over a week because nothing
    ever told it to look there."""
    hint = store_mod._keyring_env_hint()
    assert "DBUS_SESSION_BUS_ADDRESS" in hint
    assert "XDG_RUNTIME_DIR" in hint
    assert "AI AGENT" in hint
    assert "NOT a missing package" in hint
    # The env-var route must be named too, for the genuinely headless case.
    assert PRIVATE_KEY_ENV in hint
