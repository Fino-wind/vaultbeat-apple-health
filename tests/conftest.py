"""Shared pytest fixtures for the vaultbeat-mcp-local test suite."""

from __future__ import annotations

import pytest

import vaultbeat_mcp_local.store as store_module


@pytest.fixture(autouse=True)
def fake_keychain(monkeypatch: pytest.MonkeyPatch) -> dict[tuple[str, str], str]:
    """Replace the real macOS Keychain with an in-memory dict.

    Monkeypatches ``keyring.set_password``, ``keyring.get_password``, and
    ``keyring.delete_password`` (imported into ``vaultbeat_mcp_local.store``) so
    tests never pop a Keychain authorisation dialog and can run headless in CI.

    Returns the backing dict so tests can inspect what was stored.
    The dict key is ``(service, username)``; the value is the stored secret.
    """
    _store: dict[tuple[str, str], str] = {}

    def _set(service: str, username: str, password: str) -> None:
        _store[(service, username)] = password

    def _get(service: str, username: str) -> str | None:
        return _store.get((service, username))

    def _delete(service: str, username: str) -> None:
        _store.pop((service, username), None)

    monkeypatch.setattr(store_module.keyring, "set_password", _set)
    monkeypatch.setattr(store_module.keyring, "get_password", _get)
    monkeypatch.setattr(store_module.keyring, "delete_password", _delete)

    return _store
