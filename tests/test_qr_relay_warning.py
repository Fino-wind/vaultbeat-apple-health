"""The QR code reaches the agent; it does not necessarily reach the human.

Observed 2026-08-17: `bind` run from Hermes' TUI produced a blank gap where the
QR code should be, and the agent — reading a tool result that plainly contained
one — told the user "QR 码已经弹出来了！" and asked them to scan it. The user was
pointed confidently at nothing.

That failure is not fixable from the agent's side. It receives bytes, has no
view of its own rendering layer, and there is no signal in the protocol that
says "your output survived". So the CLI has to say it, and these tests pin what
it says — the wording IS the mechanism here, not decoration around it.
"""

from __future__ import annotations

import io
import sys
from unittest import mock

from vaultbeat_mcp_local.cli import _QR_RELAY_WARNING, _tty_hint


def test_the_warning_addresses_the_agent_not_the_human() -> None:
    """Aimed at a machine reader, because the human cannot act on it.

    A user staring at a blank gap already knows something is wrong; the one who
    needs telling is the agent about to claim otherwise.
    """
    assert "AI AGENTS" in _QR_RELAY_WARNING


def test_it_forbids_the_specific_false_claim_that_was_observed() -> None:
    """Not "may not render correctly" — that is read as a caveat and ignored.

    The instruction has to name the thing not to say, because the failure was
    an agent asserting the code appeared while reading one.
    """
    lowered = _QR_RELAY_WARNING.lower()
    assert "do not tell the user the qr code appeared" in lowered
    assert "cannot detect this from your side" in lowered


def test_it_names_the_action_that_always_works() -> None:
    """A warning with no exit turns into "so what do I do" one turn later."""
    assert "uvx vaultbeat-mcp@latest bind" in _QR_RELAY_WARNING
    assert "real terminal" in _QR_RELAY_WARNING


def test_it_pre_absolves_the_user() -> None:
    """The user's own reading of a blank gap is "I did something wrong", and
    that sends them to re-run, re-install, or give up. Say whose fault it is."""
    lowered = _QR_RELAY_WARNING.lower()
    assert "not their mistake" in lowered
    assert "not a bug in vaultbeat" in lowered


def test_non_tty_adds_a_structural_hint() -> None:
    """`isatty()` false is PROOF nobody is watching this stream — a stronger
    signal than the warning, which has to cover the pty-exists-but-renderer-eats-it
    case. Both are kept because neither subsumes the other."""
    fake = io.StringIO()  # a StringIO is never a tty
    with mock.patch.object(sys, "stdout", fake):
        hint = _tty_hint()
    assert "not a terminal" in hint
    assert "not reaching a human" in hint


def test_tty_stays_quiet() -> None:
    """A human at a real terminal sees the QR fine — the extra line would be
    noise, and noise is how the warning above stops being read."""
    fake = mock.Mock()
    fake.isatty.return_value = True
    with mock.patch.object(sys, "stdout", fake):
        assert _tty_hint() == ""
