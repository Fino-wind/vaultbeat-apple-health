"""A QR code drawn into a GBK pipe took the whole pairing down with it.

Seen 2026-09-11 in a screenshot of a Codex session pairing on Windows: the agent
said "Windows 终端的 GBK 编码把二维码字符弄坏了" and went off to generate a PNG by
hand. It was right about the cause and wrong about the severity — this was not a
cosmetic problem.

⚠️ Who was at that keyboard is not recorded, and the first version of this file
said "reported by a user" — which was an inference, not something the screenshot
showed. Kept as a note because the file is about exactly that distinction.

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


def test_every_string_bind_prints_is_pure_ascii() -> None:
    """One stream must not carry two encodings, and this one already did.

    The QR drawing goes out as UTF-8 bytes regardless of what stdout claims (that
    is what stops `bind` crashing). Every `print()` beside it still goes through
    stdout's own encoder, so a single em dash puts GBK bytes and UTF-8 bytes in
    the same stream and a reader decoding either way gets half of it wrong: the
    agent sees a perfect QR code followed by "NOTE ?? this QR code", the cmd.exe
    user sees the reverse.

    🔴 The first version of this test checked FOUR MODULE CONSTANTS by name, and
    was called `test_everything_printed_around_the_qr_code_is_pure_ascii` — a
    name that promised the whole bind path while a hand-written tuple covered a
    twelfth of it. It passed while `_tty_hint()` still returned an em dash, and
    that one is the worst possible place for it: `_tty_hint` returns text ONLY
    when `isatty()` is false — i.e. on every agent-invoked run, the exact case
    this whole file exists for.

    So the literals are now collected from the SOURCE by AST. A tuple of names
    can only ever guard what someone remembered to add to it; walking the tree
    guards what the code actually prints.
    """
    import ast
    import pathlib

    from vaultbeat_mcp_local import cli

    source = pathlib.Path(cli.__file__).read_text()
    tree = ast.parse(source)

    # Every string literal that `handle_bind` or `_tty_hint` can put on stdout:
    # print() arguments, returned strings, and the module constants they name.
    printed: list[tuple[str, str]] = []

    def literals_under(node: ast.AST, where: str) -> None:
        for sub in ast.walk(node):
            if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                printed.append((where, sub.value))
            elif isinstance(sub, ast.Name) and sub.id.startswith("_QR_"):
                value = getattr(cli, sub.id, None)
                if isinstance(value, str):
                    printed.append((f"{where} -> {sub.id}", value))

    # ⚠️ Only what actually reaches stdout: `print()` arguments and returned
    # strings. Walking the whole function body would also collect DOCSTRINGS —
    # the first draft did, and failed on `_tty_hint`'s own docstring, which no
    # user ever sees. A test that flags harmless text teaches people to ignore it.
    for node in ast.walk(tree):
        if not (isinstance(node, ast.FunctionDef) and node.name in {"handle_bind", "_tty_hint"}):
            continue
        for sub in ast.walk(node):
            emits = None
            if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name) and sub.func.id == "print":
                emits = sub.args
            elif isinstance(sub, ast.Return) and sub.value is not None:
                emits = [sub.value]
            if emits is None:
                continue
            for arg in emits:
                literals_under(arg, node.name)

    assert printed, "AST walk found nothing — the function names must have changed"

    offenders = [
        (where, f"U+{ord(c):04X} {c!r}", text[:60])
        for where, text in printed
        for c in text
        if ord(c) > 127
    ]
    assert not offenders, (
        "bind prints non-ASCII; one stream would carry two encodings:\n"
        + "\n".join(f"  {w}: {c} in {t!r}" for w, c, t in offenders)
    )
