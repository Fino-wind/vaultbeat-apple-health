"""Shared pytest fixtures for the vaultbeat-mcp-local test suite."""

from __future__ import annotations

import os

import pytest

import vaultbeat_mcp_local.store as store_module
from vaultbeat_mcp_local.demo import DEMO_ENV


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


@pytest.fixture(autouse=True)
def no_pypi_lookup(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep `doctor` off the network.

    `doctor` compares the installed version against PyPI so a user running an
    old client is told to upgrade (the alternative — letting the SERVER send a
    notice string — was rejected as a prompt-injection channel; see the comment
    on `_client_version_status`).

    Left alone, every doctor test would make a real HTTPS call: slow, broken
    offline, and its verdict would flip the day a release is published. Empty
    string selects the "could not reach PyPI" branch, which is deliberately a
    PASS. Tests that care about the comparison set this variable themselves.
    """

    monkeypatch.setenv("VAULTBEAT_MCP_FAKE_LATEST", "")


@pytest.fixture(autouse=True)
def demo_env_is_off_unless_a_test_asks_for_it() -> None:
    """Fail loudly if `VAULTBEAT_DEMO` is set at the START of any test.

    Demo mode swaps EVERY number this server returns for a synthetic one, at a
    choke point above the binding check, the cache and the network. A test that
    runs with it on is not testing the code it names — it is testing `demo.py`,
    and it passes for the wrong reason with nothing anywhere saying so.

    Two ways that happens, and the message names both because they need different
    fixes:

    · a test set it via `os.environ` instead of `monkeypatch.setenv`, so nothing
      undid it and every later test in the process inherited it. This is why the
      check is an assertion rather than a silent `delenv` — clearing it here
      would repair the symptom and hide the leak permanently.
    · the developer has it exported in their shell, in which case the whole suite
      is meaningless and ~300 confusing green results are strictly worse than one
      loud red one.

    Tests that WANT demo mode set it inside their own body (see
    `tests/test_demo.py`); autouse fixtures are set up before the test runs, so
    this sees the state they inherited, not the state they chose.
    """

    leaked = os.environ.get(DEMO_ENV)
    assert leaked is None, (
        f"{DEMO_ENV}={leaked!r} was already set when this test started. Demo mode "
        f"replaces every value the server returns, so this test would pass or fail "
        f"for reasons unrelated to what it asserts. Either an earlier test set it "
        f"without monkeypatch (use `monkeypatch.setenv` so it is undone), or it is "
        f"exported in your shell (`unset {DEMO_ENV}` and re-run)."
    )
