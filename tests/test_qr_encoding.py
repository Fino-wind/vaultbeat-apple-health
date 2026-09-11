"""A QR code drawn into a GBK pipe took the whole pairing down with it.

Observed 2026-09-11, reported by a user pairing from Codex on Windows: the agent
said "Windows 终端的 GBK 编码把二维码字符弄坏了" and went off to generate a PNG by
hand. It was right about the cause and wrong about the severity — this was not a
cosmetic problem.

`qrcode.print_ascii` is hard-coded to draw with cp437 255 / 223 / 220 / 219, and
in GBK:

    U+00A0 (white)  does not exist
    U+2580 ▀        does not exist
    U+2584 ▄        exists, two bytes — full width
    U+2588 █        exists, two bytes — full width

So the white modules cannot be encoded at all while the black ones take two columns.
With `errors='strict'` — which is what Python uses on Windows the moment stdout
is a PIPE rather than a console, i.e. on every agent-invoked command — the write
raises `UnicodeEncodeError` partway through the drawing. `handle_bind` then never
reaches `poll_until_bound`, so the pairing the server has already opened sits
there with nobody claiming it: the user can scan a code they somehow obtained and
nothing will happen, because the local half died before it started listening.

These tests pin the two properties that matter, in the order they matter:

  1. `bind` survives an encoding that cannot draw a QR code.
  2. When it cannot draw one, it says so in terms an agent can act on.
"""

from __future__ import annotations

import io
import sys
from unittest import mock

import pytest

from vaultbeat_mcp_local.cli import (
    _QR_ENCODING_NOTE,
    _QR_ESCAPE_ROUTES,
    _QR_GLYPHS,
    _print_qr,
    _stdout_encoding_can_draw_a_qr,
)

PAYLOAD = '{"pollID":"abc","publicKeyBase64":"k","serverName":"n"}'


class _GbkStdout(io.TextIOWrapper):
    """stdout as Python builds it on a simplified-Chinese Windows behind a pipe.

    The two details that matter are both defaults rather than choices:
    `locale.getpreferredencoding()` is cp936 there, and a redirected stream gets
    `errors='strict'`. A test that relaxed either one would pass while the real
    machine crashes.
    """

    def __init__(self) -> None:
        super().__init__(io.BytesIO(), encoding="gbk", errors="strict", newline="")

    def isatty(self) -> bool:  # pragma: no cover - trivial
        return False

    def decoded(self) -> str:
        self.flush()
        return self.buffer.getvalue().decode("gbk", errors="replace")


def test_the_glyph_probe_names_characters_gbk_actually_rejects() -> None:
    """Guards the probe itself, which is one silent edit away from useless.

    The first glyph is NBSP. A plain space looks identical in every editor and
    encodes fine in GBK, so "tidying" the literal would leave a probe that always
    returns True on exactly the machines it exists to catch — and nothing else in
    the suite would notice.
    """
    assert "\u00a0" in _QR_GLYPHS, "the white module must stay NBSP, not a space"
    with pytest.raises(UnicodeEncodeError):
        _QR_GLYPHS.encode("gbk")
    # ...and the probe is not simply always-false: UTF-8 must still pass.
    _QR_GLYPHS.encode("utf-8")


def test_bind_does_not_die_when_stdout_cannot_encode_the_code() -> None:
    """The whole point. A drawing problem must not end the pairing.

    Before the fix this raised UnicodeEncodeError out of `_print_qr`, killing
    `bind` before it ever polled.
    """
    stdout = _GbkStdout()
    with mock.patch.object(sys, "stdout", stdout):
        _print_qr(PAYLOAD)  # must not raise

    assert "emitted as UTF-8 bytes" in stdout.decoded()


def test_it_tells_the_agent_not_to_re_run_bind() -> None:
    """The obvious recovery is the one that destroys the session.

    Re-running `bind` mints a new pollID and invalidates whatever the user has
    already scanned, so an agent that "retries" turns a rendering glitch into a
    pairing that can never complete.
    """
    stdout = _GbkStdout()
    with mock.patch.object(sys, "stdout", stdout):
        _print_qr(PAYLOAD)

    out = stdout.decoded().lower()
    assert "do not re-run `bind`" in out
    assert "still live and still waiting" in out


def test_it_offers_a_route_that_works_without_a_working_terminal() -> None:
    """The user in the report was behind an agent, not a terminal.

    Telling them to `chcp 65001` is useless if nobody is at a console — so the
    first suggestion has to be one the AGENT can carry out on its own, which is
    exactly what the Codex session improvised before this text existed.

    🔑 And it must name "import from Photos". iOS has decoded a saved image
    since `VaultbeatQRImageDecoder` shipped (no camera, no Photos permission),
    but an agent cannot know that — left unsaid, it hands over a PNG and the
    user tries to photograph one screen with another.
    """
    lowered = _QR_ESCAPE_ROUTES.lower()
    assert "render the payload printed above as a qr image yourself" in lowered
    assert "import from photos" in lowered


def test_a_utf8_pipe_gets_a_code_and_no_warning() -> None:
    """The common case must stay silent, or the warning becomes wallpaper."""
    stdout = io.TextIOWrapper(io.BytesIO(), encoding="utf-8", errors="strict", newline="")
    with mock.patch.object(sys, "stdout", stdout):
        _print_qr(PAYLOAD)
    stdout.flush()
    out = stdout.buffer.getvalue().decode("utf-8")

    assert "█" in out, "a real QR code should have been drawn"
    assert "emitted as UTF-8 bytes" not in out
    assert _QR_ESCAPE_ROUTES.splitlines()[0] not in out


def test_the_probe_reads_the_live_stdout_not_a_cached_value() -> None:
    """It has to answer for the stream in front of it, per call."""
    with mock.patch.object(sys, "stdout", _GbkStdout()):
        assert _stdout_encoding_can_draw_a_qr() is False
    with mock.patch.object(
        sys, "stdout", io.TextIOWrapper(io.BytesIO(), encoding="utf-8")
    ):
        assert _stdout_encoding_can_draw_a_qr() is True


def test_the_note_states_what_was_done_never_that_the_code_is_unreadable() -> None:
    """The first draft of this text said "probably unreadable". It was wrong.

    Because the bytes go out as UTF-8 whatever stdout claims, a UTF-8-decoding
    agent — nearly all of them — receives an INTACT code and would be reading a
    flat contradiction of what is in front of it. That is the same mistake as
    the lock-screen moon in `CLAUDE.md § Product Positioning`, one layer down:
    a guess rendered with the confidence of an observation.

    The note may report what the process DID (emitted UTF-8) and what it cannot
    know (how the reader decodes). It may not deliver a verdict on the drawing.
    """
    lowered = _QR_ENCODING_NOTE.lower()
    assert "emitted as utf-8 bytes" in lowered
    for verdict in ("probably unreadable", "is unreadable", "is destroyed", "do not use"):
        assert verdict not in lowered, f"states a verdict it cannot support: {verdict!r}"
    # It must still give the reader a way to TELL, rather than leaving it hanging.
    assert "if you are reading utf-8" in lowered
    assert "mojibake" in lowered


def test_everything_printed_around_the_qr_code_is_pure_ascii() -> None:
    """One stream must not carry two encodings, and this one already did.

    The QR drawing now goes out as UTF-8 bytes regardless of what stdout claims
    (that is what stops `bind` crashing). Every `print()` beside it still goes
    through stdout's own encoder — so a single em dash in this copy put GBK
    bytes and UTF-8 bytes in the same stream, and a reader decoding either way
    got half of it wrong: the agent saw a perfect QR code followed by "NOTE ??
    this QR code", the cmd.exe user saw the reverse.

    ⚠️ `_QR_RELAY_WARNING` had that em dash from the day it was written, long
    before any of this. It never showed up because nothing else in the stream
    disagreed with it yet.

    An em dash is worth nothing here. ASCII is the only encoding every reader of
    this stream agrees on, so the copy stays inside it.
    """
    from vaultbeat_mcp_local import cli

    for name in (
        "_QR_RELAY_WARNING",
        "_QR_ESCAPE_ROUTES",
        "_QR_ENCODING_NOTE",
        "_QR_NOT_DRAWN",
    ):
        text = getattr(cli, name)
        offenders = sorted({f"U+{ord(c):04X} {c!r}" for c in text if ord(c) > 127})
        assert not offenders, f"{name} must stay ASCII; found {offenders}"


def test_bind_still_polls_when_drawing_blows_up_in_an_unforeseen_way() -> None:
    """The guarantee is about the pairing, not about the picture.

    `_print_qr` names the encoding failures it knows how to degrade. This pins
    the outer promise: whatever else goes wrong while drawing — broken pipe,
    closed stdout, a future `qrcode` raising somewhere new — `handle_bind` still
    reaches `poll_until_bound`. The session is already open on the server by
    this point; failing to draw it is cosmetic, failing to claim it is not.
    """
    from unittest.mock import MagicMock

    import vaultbeat_mcp_local.cli as cli

    args = MagicMock(no_qr=False, timeout=1, interval=1, server_name="n", api_base_url=None)
    service = MagicMock()
    service.start_binding.return_value = MagicMock(qr_payload_json='{"pollID":"x"}')
    polled = MagicMock(status="bound", server_name="n", trial_ends_at=None)

    async def _poll(**_kwargs: object) -> object:
        return polled

    service.poll_until_bound = _poll

    with mock.patch.object(cli, "_service", return_value=service), mock.patch.object(
        cli, "_print_qr", side_effect=BrokenPipeError("nobody home")
    ):
        cli.handle_bind(args)  # must not raise

    assert service.start_binding.called, "the session must still have been opened"
