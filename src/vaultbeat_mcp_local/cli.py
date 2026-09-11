from __future__ import annotations

import argparse
import asyncio
import io
import json
import os
import sys
from pathlib import Path
from typing import Any

from vaultbeat_mcp_local import __version__
from vaultbeat_mcp_local.demo import DEMO_BANNER, DEMO_ENV, demo_enabled
from vaultbeat_mcp_local.demo_watermark import watermark_demo_result
from vaultbeat_mcp_local.service import (
    KNOWN_METRIC_TYPES,
    READABLE_NOTE_KINDS,
    VaultbeatLocalService,
)
from vaultbeat_mcp_local.store import DEFAULT_API_BASE_URL, ConfigStore, write_secret_file


def _store(args: argparse.Namespace) -> ConfigStore:
    return ConfigStore(Path(args.config).expanduser() if args.config else None)


def _service(args: argparse.Namespace) -> VaultbeatLocalService:
    return VaultbeatLocalService(_store(args))


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _health_json(payload: Any) -> str:
    """Serialise a health payload, stamping it when this is a demo run.

    Paired with `_emit_decrypted`: that function decides WHERE the bytes go,
    this one decides what they say. Kept separate so the `--output` file and
    stdout cannot drift apart — before 2026-08-27 they were two `json.dumps`
    calls in two functions, and adding the stamp to one of them would have been
    the whole bug again one level down.

    🔴 `sort_keys` is OFF for a stamped payload and ON otherwise, and that
    asymmetry is the point rather than an oversight. `watermark_demo_result`
    puts `demo_warning` first so the banner is literally the first thing in the
    text; sorting moves it behind `count` and `daily_summary` — behind an array
    long enough that a truncated paste loses it entirely. Sorting buys stability
    of key order across future code edits, which is worth something for a real
    export used as a diff baseline and worth nothing set against the one
    guarantee demo output exists to make. Determinism is unaffected either way:
    the generator is seeded and the dicts are built in a fixed order, so two
    runs still emit identical bytes (`test_demo.py` asserts that directly).
    """

    if not demo_enabled():
        return json.dumps(payload, indent=2, sort_keys=True)
    return json.dumps(watermark_demo_result(payload), indent=2, sort_keys=False)


def _warn_demo_on_stderr() -> None:
    """Put the banner beside a JSON payload that is going to stdout.

    stderr, not stdout, because stdout here IS a JSON document: a banner line
    printed into it breaks `vaultbeat-apple-health --demo sleep | jq` and every other
    consumer that parses this output. (`doctor` can print its `[DEMO]` line to
    stdout precisely because its human rendering is not JSON.)

    ⚠️ This is the SECOND marker and never the only one. stderr is exactly what
    disappears under `> out.json`, which is the case Invariant 61
    (demo-is-a-boundary-not-a-flag) cares about most — so the in-band stamp from
    `_health_json` is the load-bearing one, and this exists for the human who
    has the terminal in front of them and will not read a key in the middle of
    200 lines.
    """

    if demo_enabled():
        print(DEMO_BANNER, file=sys.stderr)


def _wrote_message(output_path: Path, label: str) -> str:
    """Describe the file just written — differently, when nothing was decrypted.

    The real-run sentence ("holds unencrypted health data — treat it as
    sensitive") is a false statement about a demo file: nothing was fetched,
    nothing was decrypted, and no real person is in it. Telling an operator to
    handle synthetic output as sensitive is the mirror of the bug this whole
    change is about — a claim rendered with the confidence of a fact when the
    code knows better.
    """

    if demo_enabled():
        return (
            f"Wrote SYNTHETIC {label} to {output_path} (file mode 0600). "
            "This is demo output: nothing was decrypted and it belongs to no real "
            "person. The payload says so in its own first key, so it stays labelled "
            "if this file is pasted somewhere else."
        )
    return (
        f"Wrote DECRYPTED {label} to {output_path} (file mode 0600). "
        "This file holds unencrypted health data — treat it as sensitive."
    )


_QR_RELAY_WARNING = """\
NOTE FOR AI AGENTS RELAYING THIS OUTPUT: read before you answer the user.

You are about to receive a QR code drawn with block characters. Receiving it
does NOT mean the user can see it. Many agent front-ends (Hermes' TUI, chat
clients, log viewers, anything that re-renders tool output) drop those
characters, leaving the user a blank gap while you read a perfectly good QR
code. You cannot detect this from your side.

So do not tell the user the QR code appeared, and do not ask them to scan
"the code above". Tell them to run this in a real terminal themselves:

    uvx vaultbeat-apple-health@latest bind

If they say they see nothing, that is the expected outcome: it is your
rendering layer, not their mistake, and not a bug in Vaultbeat."""


def _tty_hint() -> str:
    """Extra line when nothing is attached to a terminal at all.

    `isatty()` false means the output is being piped or captured — so there is
    certainly no human watching this stream, and drawing a QR code into it is
    decoration for a log file. It is a STRUCTURAL signal, unlike the warning
    above, which has to cover the case where a pty exists but a renderer eats
    the output anyway (Hermes). Neither check subsumes the other, which is why
    both are here.

    ⚠️ THREE now, not two — `_stdout_encoding_can_draw_a_qr` joined them in 0.7.2
    and answers a third question: whether stdout's own encoder can represent the
    glyphs at all. That one used to be fatal rather than cosmetic (a GBK pipe
    raised mid-drawing and `bind` never reached its polling loop), so it is the
    one of the three that had teeth. Still none subsumes another.
    """
    if sys.stdout.isatty():
        return ""
    return (
        "\n(stdout is not a terminal: nobody is watching this stream directly, "
        "so the QR code below is almost certainly not reaching a human.)"
    )


# The four characters `qrcode.print_ascii` is hard-coded to draw with, as cp437
# code points 255 / 223 / 220 / 219: NBSP, ▀, ▄, █.
#
# 🔴 TWO of them do not exist in GBK at all (NBSP and ▀), and the other two
# encode to TWO bytes there — i.e. they are full-width, so a "black" module
# occupies two columns while a replaced "white" one occupies one, and every row
# of the code ends up a different width. On a simplified-Chinese Windows the
# damage is not cosmetic; see `_QR_ENCODING_NOTE`.
# ⚠️ Escapes, not literals. The first one is NBSP — indistinguishable from a
# plain space in every editor, and a plain space encodes fine in GBK, so
# "tidying" it would silently turn this probe into one that always passes on
# exactly the machines it exists to catch.
_QR_GLYPHS = "\u00a0\u2580\u2584\u2588"

_QR_ESCAPE_ROUTES = """\
THE PAIRING IS STILL LIVE AND STILL WAITING. Do not re-run `bind`: that mints a
new pollID and invalidates anything the user has already scanned.

Routes that work from here:

  1. Render the payload printed above as a QR image yourself (it is plain JSON;
     any QR library, or `qrencode -o pair.png '<payload>'`), send that file to
     the user's phone, and tell them to open the scanner in the app and use
     "import from Photos" instead of the camera. The app decodes a saved image,
     so nothing has to be pointed at a screen. This command keeps polling while
     they do.
  2. Tell the user to run `chcp 65001` and try `bind` again in a fresh terminal.
  3. Re-run with `--no-qr` to get the payload as text only."""

# 🔴 Says what was DONE and what is UNKNOWN, and never that the code is
# unreadable — because most of the time it is not.
#
# The bytes go out as UTF-8 regardless of what stdout claims (see
# `_emit_qr_drawing`), so an agent that decodes UTF-8 —
# which is nearly all of them — receives an intact QR code. Telling that reader
# "the code above is probably unreadable" is a false statement about something
# it can see with its own eyes, and the credibility it spends is the same
# credibility `_QR_RELAY_WARNING` needs a few lines earlier.
#
# What IS known: this stream declares an encoding that cannot represent the
# glyphs. That is evidence about the reader, not a verdict on the drawing, and
# it is stated as such.
_QR_ENCODING_NOTE = """\
NOTE: this QR code was emitted as UTF-8 bytes, bypassing this stream's declared
encoding ({encoding}), which cannot represent the characters a QR code is drawn
with.

If you are reading UTF-8, the code above is intact: use it.
If you are seeing mojibake or rows of uneven length, the encodings disagree and
the code's geometry is gone: do not ask the user to scan it.

""" + _QR_ESCAPE_ROUTES

# The harder failure: nothing renderable left the process at all.
_QR_NOT_DRAWN = """\
NOTE FOR AI AGENTS: no QR code was drawn. This stream's encoding ({encoding})
cannot represent the characters one is drawn with, and the byte-level fallback
was unavailable too. There is nothing above to scan.

""" + _QR_ESCAPE_ROUTES


def _stdout_encoding_name() -> str:
    return getattr(sys.stdout, "encoding", None) or "unknown"


def _stdout_encoding_can_draw_a_qr() -> bool:
    """Can the characters survive the encoding stdout says it is using?

    ⚠️ This asks a NARROWER question than "will the user see a QR code", and
    deliberately so — it is the one question with a mechanical answer. Whether a
    renderer downstream eats the glyphs anyway is `_QR_RELAY_WARNING`'s problem,
    and whether anyone is watching at all is `_tty_hint`'s. Three checks, none
    subsuming another.
    """
    encoding = getattr(sys.stdout, "encoding", None)
    if not encoding:
        return False
    try:
        _QR_GLYPHS.encode(encoding)
    except (UnicodeEncodeError, LookupError):
        return False
    return True


def _emit_qr_drawing(drawing: str) -> bool:
    """Get the code onto stdout, preferring the byte layer. False = nothing came out.

    🔑 The byte path is the half that FIXES the agent case rather than
    apologising for it. On Windows, Python only bypasses the locale encoding
    when stdout is attached to a real console; the moment it is a pipe - which
    is every agent-invoked command - it falls back to `cp936` with
    `errors='strict'`, and printing the drawing raises `UnicodeEncodeError`
    before `poll_until_bound` is ever reached. Writing UTF-8 bytes cannot raise
    that, and agent front-ends decode UTF-8, so the code arrives intact.

    ⚠️ The two paths are EXCLUSIVE, which is the point of the early return. A
    `buffer.write` that dies partway has already put half a code on screen;
    retrying through the text layer would print a second one below it. And there
    is nothing to gain by trying - the text layer is strictly weaker (it is the
    one that raises on GBK), so the only thing it could add is a duplicate.
    The text path exists solely for a stdout with no `buffer` at all, which
    means a replaced or captured stream, which means nothing was written yet.
    """
    buffer = getattr(sys.stdout, "buffer", None)
    if buffer is not None:
        try:
            sys.stdout.flush()
            buffer.write(drawing.encode("utf-8"))
            buffer.flush()
        except Exception:
            return False
        return True

    try:
        sys.stdout.write(drawing)
        sys.stdout.flush()
    except Exception:
        return False
    return True


def _print_qr(payload: str) -> None:
    try:
        import qrcode  # type: ignore[import-untyped]
    except ModuleNotFoundError:
        # Unreachable on a normal install since 0.3.10 — qrcode is a hard
        # dependency now. It stays as a fallback for the one case that can still
        # produce it: an environment assembled by hand, or a very old install
        # still on the era when this was the `[qr]` extra.
        #
        # ⚠️ Before that change this branch WAS the common path, and it was a
        # dead end at the worst possible moment: the user has a payload on
        # screen and a phone in hand, and is told to install something and run
        # the command again — which also mints a new pollID, invalidating
        # anything already scanned. Print the payload-only fallback rather than
        # sending them away empty-handed.
        print("Could not render a QR code (the `qrcode` package is missing).")
        print("Scan the payload above with the Vaultbeat app, or reinstall with:")
        print("  uvx --refresh vaultbeat-apple-health bind")
        return

    qr = qrcode.QRCode(border=2)
    qr.add_data(payload)
    qr.make(fit=True)

    # Render to a string first. `qr.print_ascii()` writing straight to
    # `sys.stdout` is what made this a CRASH rather than a cosmetic problem: on
    # a pipe under a GBK locale the write raises mid-drawing, `handle_bind`
    # never reaches `poll_until_bound`, and the pairing the server has already
    # opened is left with nobody claiming it. Buffering moves every encoding
    # decision to code that can catch it.
    drawing = io.StringIO()
    qr.print_ascii(out=drawing, invert=True)
    art = drawing.getvalue()

    if not _emit_qr_drawing(art):
        # Nothing renderable got out. The payload is already on screen above this
        # call, so the pairing is still completable: say how, and let the caller
        # carry on polling.
        print(_QR_NOT_DRAWN.format(encoding=_stdout_encoding_name()))
        return

    # Bytes left the process, which is not the same as a QR code reaching a
    # human: a downstream reader that decodes as GBK sees mangled rows. The
    # declared encoding is the only evidence available here about what that
    # reader expects, so use it — and say so only when it says there is a
    # problem, to keep this silent on the UTF-8 machines that are the norm.
    if not _stdout_encoding_can_draw_a_qr():
        print()
        print(_QR_ENCODING_NOTE.format(encoding=_stdout_encoding_name()))


def handle_init(args: argparse.Namespace) -> int:
    config = _store(args).ensure_initialized(
        server_name=args.server_name,
        api_base_url=args.api_base_url,
    )
    _print_json(config.redacted())
    return 0


def handle_status(args: argparse.Namespace) -> int:
    _print_json(_service(args).status())
    return 0


def handle_doctor(args: argparse.Namespace) -> int:
    report = asyncio.run(_service(args).doctor())
    if getattr(args, "json", False):
        _print_json(report)
    else:
        # First line, before any [OK], so it cannot be scrolled past: every
        # number below is synthetic and the checks below cover a demo, not an
        # install.
        if report.get("demo_mode"):
            print(f"[DEMO] {report.get('demo_warning', '')}")
            print()
        for check in report["checks"]:
            marker = "[OK]  " if check["ok"] else "[FAIL]"
            print(f"{marker} {check['name']}: {check['detail']}")
            if not check["ok"] and check.get("hint"):
                print(f"       → {check['hint']}")

        # Capability gap: tools that exist here but have nothing to read. Without
        # this, an empty result is indistinguishable from a broken one — the
        # question "why does get_total_energy_burned return nothing" has no
        # answer anywhere else.
        caps = report.get("capabilities") or {}
        if caps.get("available"):
            # 🔴 The field is `possibly_needs_newer_app` and this rendering used
            # to drop the "possibly": it printed "needs an iOS build from <date>
            # or later" as a flat assertion. Its own producer's docstring says
            # the opposite — "deliberately does NOT claim to know the app's
            # version" — and the note printed one line below ranks the version
            # SECOND of four causes, behind "this server was bound recently".
            #
            # The five gated kinds are strength / food / vo2max / basal_energy /
            # hrv_hourly: the first two are manual-entry only, vo2max is sparse
            # by design (Apple computes it during outdoor walk/run/hike), and the
            # last two need an Apple Watch. So a brand-new user on the CURRENT
            # build, with no Watch and no lifts logged, was told five times that
            # their app was out of date — at the exact step `skill.md` sends
            # every new user to as the verification of the whole setup. They go
            # to the App Store, find no update, and conclude the pairing failed.
            gated = caps.get("possibly_needs_newer_app") or {}
            if gated:
                print()
                print("[WARN] Some tools have no data yet on this account:")
                print("       Most often this server was bound recently and the history")
                print("       is still sealing for it — open the app, tap Settings →")
                print("       Data & AI → 'Re-sync all health data to AI', then re-run.")
                for kind, since in sorted(gated.items()):
                    print(f"       {kind} — one possible cause is an iOS build older than {since}")
                print(f"       → {caps['note']}")
            else:
                print(f"[OK]   capabilities: all {len(caps.get('kinds_with_data', []))} data types present")

            # Printed unconditionally when the account has more than one owner:
            # this is the ONLY place a prefix can be discovered, and passing one
            # is what keeps a "weight trend" from being two people's weights
            # averaged together with nothing in the payload saying so.
            owners = caps.get("owner_prefixes") or []
            if len(owners) > 1:
                print()
                print(f"[INFO] This account has {len(owners)} data owners: {', '.join(owners)}")
                print("       Pass one as `owner` on every read, or results blend both people.")

        # Rows this run deliberately skipped, each with its reason. In demo mode
        # the checks that ARE run all pass, so without this the CLI ends on "All
        # checks passed." while config, binding, cloud reachability and the data
        # round trip were never attempted — the exact green-tick-as-evidence
        # problem the demo branch of `doctor()` exists to avoid. The JSON carried
        # it from the start; the human-facing rendering dropped it, which is the
        # half a person actually reads.
        skipped = report.get("not_checked") or {}
        if skipped:
            print()
            print("[SKIP] Not attempted in this run:")
            for name, reason in sorted(skipped.items()):
                print(f"       {name} — {reason}")
            if report.get("next_step"):
                print(f"       → {report['next_step']}")

        # What was NOT checked. Printed last and unconditionally — a passing run
        # is precisely when this matters, because "All checks passed" otherwise
        # reads as "the whole setup is fine" and sends the reader hunting on the
        # side of the pipe nobody just tested.
        scope = report.get("scope") or {}
        if scope:
            print()
            # Printed before the scope caveat because it is the more actionable of
            # the two: it tells the operator where their decryption key physically
            # is, which on a headless box is a plaintext file they may not know
            # exists.
            if scope.get("private_key_location"):
                print(f"[KEY]  Private key: {scope['private_key_location']}")
                print()
            print(f"[NOTE] Not checked: {scope['does_not_cover']}")
            received = [
                name for name, present in (scope.get("env_overrides_received") or {}).items()
                if present
            ]
            if received:
                print(f"       This process did receive: {', '.join(received)}")
            else:
                print("       This process received no Vaultbeat environment overrides.")

        print("All checks passed." if report["ok"] else "Some checks failed — follow the hints above.")
    return 0 if report["ok"] else 1


#: The two sentences `bind` prints on success once a trial clock has started.
#:
#: Module constants rather than inline strings so a test can assert them without
#: driving a whole network bind — this exact pair was FALSE from 2026-09-03 to
#: 2026-09-06 and nothing anywhere went red, because the sentence was in one
#: file and the change that falsified it was in another. See the comment at the
#: call site for why they are two facts and not one.
TRIAL_UNLOCK_LINE = "The AI interface is unlocked until {date}."

UPLOAD_WINDOW_LINE = (
    "Separately: on Free and during the trial your last 7 days are what reaches the "
    "cloud, so that is the window your AI can read. Asking for a month returns a week "
    "— that is the plan boundary, not a sync that has not finished. Pro uploads the "
    "whole history."
)


def handle_bind(args: argparse.Namespace) -> int:
    service = _service(args)
    session = service.start_binding(
        server_name=args.server_name,
        api_base_url=args.api_base_url,
    )
    # 🔴 Say this BEFORE the QR, and say it to the agent rather than to the human.
    #
    # An agent relaying this command CANNOT know whether the person can see the
    # QR code: it receives bytes, not a screen. Hermes' TUI drops the block
    # characters the code is drawn with, so the user gets a blank gap — and the
    # agent, reading a tool result that plainly contains a QR code, told them
    # "QR 码已经弹出来了！" (observed 2026-08-17). That is worse than a plain
    # failure: the user is being pointed confidently at nothing, and concludes
    # they did something wrong.
    #
    # The fix has to come from here, because the agent has no way to check. It
    # cannot see its own rendering layer, and there is no signal in the protocol
    # that says "your output survived". So the CLI states the uncertainty and
    # names the one action that always works.
    #
    # ⚠️ Written for a machine reader on purpose — imperative, and about what NOT
    # to claim. "Some terminals may not display this correctly" is the polite
    # phrasing and it does not work: an agent reads it as a caveat and proceeds
    # to assert the code appeared anyway.
    print(_QR_RELAY_WARNING + _tty_hint())
    print()
    print("Scan this payload with Vaultbeat on iOS:")
    print(session.qr_payload_json)
    if not args.no_qr:
        # 🔴 Drawing the code is BEST-EFFORT; claiming the pairing is not.
        #
        # `_print_qr` handles the encoding failures it can name and degrades with
        # something useful to say. This catch is for the ones it cannot — a
        # broken pipe, a closed stdout, a future qrcode release that raises
        # somewhere new. Whatever it is, the session on the server is already
        # open and the payload is already printed above, so the only unrecoverable
        # outcome is not reaching `poll_until_bound` — which is exactly what
        # happened on GBK Windows until 0.7.2.
        #
        # `Exception`, not `BaseException`: Ctrl-C must still stop the command.
        try:
            _print_qr(session.qr_payload_json)
        except Exception as exc:  # pragma: no cover - defence in depth
            print(f"(could not draw a QR code: {exc!r} - scan the payload above)")

    result = asyncio.run(
        service.poll_until_bound(
            timeout_sec=args.timeout,
            interval_sec=args.interval,
        )
    )
    if result.status != "bound":
        # Until 0.2.8 this said only "still pending, re-run poll", which tells a
        # first-time reader nothing. `uvx vaultbeat-apple-health` installs cleanly for
        # anyone, so the people who reach this line are usually the ones who
        # never read the README — and the reason no scan arrived is almost
        # always that the phone side is not ready.
        #
        # ⚠️ This block said "Pro-only — Free and Plus accounts cannot complete a
        # pairing" until 2026-08-17. That was true and is now false: connecting
        # is open to every tier.
        #
        # 🔴 It must not mention Pro, subscriptions or trials AT ALL, and that is
        # a product decision, not an omission (owner, 2026-08-17: 宣传语不要说需要
        # pro。直接吸引过来。app 里面自然能体验到). The first replacement here still
        # said "no subscription needed — 3 days of Pro free", which reads as
        # helpful and is not: it puts a price and a deadline in front of someone
        # who has not yet seen the product work. This text is ACQUISITION copy —
        # its only job is to get the app installed and the QR scanned. The app
        # shows what the plan is, once there is something to have an opinion
        # about. Same rule as `server.json`'s description and the website: what a
        # stranger reads first should be the reason to come, not the terms.
        #
        # The App Store link deliberately carries the numeric id and NO slug.
        # The slug follows the app's display name, so the ones written across
        # the site, the public README and research-facts.json all still say
        # `tether-ai-health-sync` and will break on the rename; id-only never
        # does.
        print("Binding is still pending: nothing scanned this within the timeout.")
        print()
        # Step 2 names a screen the reader does not land on first. Opening
        # "Connect an AI server" for the first time starts a GUIDE — the scanner
        # is its last step — and this block's own reasoning below establishes
        # that the likeliest reader here has never opened the app at all (they
        # were installing it while bind timed out). So this is precisely the
        # reader for whom "scan the QR from that screen" describes a screen that
        # is not in front of them, and the failure is silent: they see a walk-
        # through, conclude they are in the wrong place, and go looking.
        #
        # Same sentence, same fix, third copy. The site's two (`support/page.tsx`
        # :92 rich text and :104 plain) were corrected in edf8222; this one was
        # not found by that audit because it lives in another language in another
        # part of the repo. `VaultbeatMCPGuideSheet.swift:442` stays as it is on
        # purpose — its reader is INSIDE the guide already, so for them the
        # sentence is true. Whether a shared sentence is wrong depends on where
        # its reader is standing, which is also why grepping for the string is
        # not enough to decide.
        print("What the phone side needs:")
        print("  1. Vaultbeat for iOS: free to install, free to connect.")
        print("     https://apps.apple.com/app/id6759241985")
        print("  2. In the app: Settings -> Data & AI -> Connect an AI server")
        print("     First time through, the app walks you through setup and the")
        print("     scanner comes last, follow it to the end.")
        print("  3. Scan the QR above from that screen.")
        print()
        # 🔴 The two cases have DIFFERENT answers, and the previous single
        # paragraph gave the wrong one to whichever reader was in the other case.
        #
        # It said the resume window is "~5 more minutes", derived from `bind`
        # giving up at 300s while the pending row lives 600s. That arithmetic
        # assumes the row is created when bind starts. It is not: the row is
        # written by `mcp_bind_local_server`, which the PHONE calls with its own
        # JWT when the code is scanned (20260811090000, `now() + interval '10
        # minutes'`). So the clock starts at the scan, not at bind — a user who
        # has not scanned yet has no clock running at all, and one who scanned a
        # moment ago has the full ten minutes rather than five.
        #
        # Which matters because the likeliest reader here has NOT scanned: they
        # were installing the app while bind timed out. The old text opened with
        # "Already scanned it?" and told them their window was half gone.
        #
        # The command is `vaultbeat-apple-health` (the distribution name since
        # the 0.6.2 rename). `vaultbeat-mcp` / `vaultbeat-mcp-local` survive as
        # console-script aliases for pre-rename installs, but the documented
        # install path everywhere (site, skill.md, README) is
        # `uvx vaultbeat-apple-health` — and uvx resolves entry points by PACKAGE
        # name, so `uvx vaultbeat-mcp-local poll` fails outright with "not found
        # in the package registry" and `uvx vaultbeat-mcp poll` installs the
        # frozen old distribution (Invariant 67). Printing either to the one
        # reader who by definition has not read the README was the worst
        # possible place to get this wrong.
        print("Not scanned yet? The QR above is still good, nothing expires until")
        print("someone scans it. Scan it FIRST, then pick the pairing back up with:")
        print("  uvx vaultbeat-apple-health poll")
        print()
        print("Already scanned? You have 10 minutes from the moment you scanned to")
        print("run that same command. After that, run `uvx vaultbeat-apple-health bind` for")
        print("a fresh code.")
        print()
        # Ordering matters and the failure is counter-intuitive: the pairing row
        # is created BY THE SCAN, so polling before scanning finds no row and the
        # endpoint correctly answers `expired` — whose own advice is to re-run
        # `bind`. Following that mints a new pollID and invalidates the QR still
        # on screen, turning "one scan away" into "cannot complete". Cheap to
        # warn about, expensive to discover.
        print("(Running poll BEFORE scanning reports `expired`, that is just")
        print(" 'no scan has arrived', not a dead code. Don't re-run bind on it;")
        print(" a new bind replaces the QR you are looking at.)")
        return 2

    print(f"Bound local MCP server: {result.server_id}")
    # vb-003 (2026-08-26): binding succeeds, but the AGENT still cannot reach
    # this server until it is registered with the MCP client — and this success
    # message used to say "Go ask your AI" while omitting that step entirely.
    # A user who did as told found Claude listing zero vaultbeat tools and
    # concluded the product was broken. The registration command belongs HERE,
    # at the exact moment the user is looking at a terminal that just said
    # everything worked.
    print()
    print("One last step: register this server with your MCP client.")
    print("For Claude Code, run:")
    print()
    print("  claude mcp add vaultbeat-health -- uvx vaultbeat-apple-health@latest serve --transport stdio")
    print()
    print("(Other clients: see https://vaultbeat.app/mcp for setup lines.)")
    # ⚠️ Informational, NOT a sales line. The acquisition copy above deliberately
    # says nothing about plans; this is the other side of that rule — once the
    # thing is working, the user is entitled to know what they have. So it states
    # a date and stops. No "subscribe to keep it", no price: at this moment the
    # user has had zero seconds of the product, and asking for money before they
    # have felt it is exactly the ordering the owner rejected.
    #
    # Absent for a grandfathered user, a trial already running, or an edge
    # deployment older than 2026-08-17 — saying nothing is correct in all three,
    # so this stays conditional rather than guessing a date.
    # 🔴 Two separate facts, and merging them is what made this line false.
    #
    # It read "Full access to your health data is on until <date>" until
    # 2026-09-06. The trial unlocks the AI INTERFACE; it does not lift the
    # UPLOAD WINDOW, which Invariant 72 (free-tier-uploads-seven-days, shipped
    # 2026-09-03) clamps to 7 days for `.trial` exactly as for `.free`. The two
    # were one sentence, so adding the clamp made the sentence a falsehood with
    # nothing anywhere going red.
    #
    # 🔑 And it was false EVERY time it printed, not occasionally: a fresh trial
    # clock is suppressed for grandfathered users, for a trial already running,
    # and for paid Pro (20260819063000_no_trial_clock_for_paid.sql) — i.e. for
    # precisely the accounts that are NOT clamped. `trial_ends_at` is non-nil if
    # and only if the reader is on the 7-day window.
    #
    # Why the second line is not a sales line despite naming Pro: without the
    # reason, a user who asks for a month and receives a week has a working
    # product that looks broken, and the obvious next move is to re-sync or
    # re-pair — neither of which can ever help. Stating the boundary is what
    # makes the short answer legible. Still no price, still no "subscribe now".
    if result.trial_ends_at:
        print()
        print(TRIAL_UNLOCK_LINE.format(date=result.trial_ends_at[:10]))
        print(UPLOAD_WINDOW_LINE)
        print("Once registered above, go ask your AI something: that is what this was for.")
    return 0


def handle_poll(args: argparse.Namespace) -> int:
    result = asyncio.run(_service(args).poll_once())
    _print_json(
        {
            "status": result.status,
            "server_id": result.server_id,
            "request_id": result.request_id,
        }
    )
    return 0 if result.status == "bound" else 2


def handle_sync(args: argparse.Namespace) -> int:
    svc = _service(args)
    metric_type = getattr(args, "metric_type", None)
    if metric_type:
        records, errors = asyncio.run(svc._records_for_metric(metric_type, limit=args.limit, fresh=args.fresh))
    else:
        records, errors = asyncio.run(svc.sync_decrypted_records(limit=args.limit, fresh=args.fresh))
    payload = {
        "records": [record.to_dict() for record in records],
        "errors": errors,
    }
    # Routed through the same funnel as every other data subcommand rather than
    # writing the file itself: this branch used to be a second `json.dumps` and a
    # second "DECRYPTED" sentence, i.e. one of the two places the demo stamp would
    # have had to be added twice or be wrong once.
    _emit_decrypted(payload, args, label=f"{len(records)} raw records")
    return 0 if not errors else 3


def _emit_decrypted(payload: dict[str, Any], args: argparse.Namespace, *, label: str) -> None:
    """THE place health data leaves this CLI — stdout or a 0600 file.

    Every data subcommand ends here, `sync` included since 2026-08-27. That
    matters beyond tidiness: the demo watermark is a judgement, and a judgement
    with two exits gets remembered at one of them (Invariant 58
    (one-funnel-per-event)). This function had a near-twin inside `handle_sync`
    doing the same two things slightly differently — exactly the shape that lets
    a fix look complete while covering half the surface, which is how the
    watermark itself came to cover the MCP exit only.

    The file branch deliberately gets the SAME bytes as the stdout branch, out of
    one `_health_json` call. `--output` is the case that matters most in demo
    mode (a file outlives the session that could explain it), so it must not be
    the branch that is easier to forget.

    ⚠️ `label` now carries its own noun ("… summary", "43 raw records"); the
    sentence around it no longer appends one, because `sync` writes records
    rather than a summary and the old wording called them a summary anyway.
    """

    text = _health_json(payload)
    if args.output:
        output_path = Path(args.output).expanduser()
        write_secret_file(output_path, text + "\n")
        print(_wrote_message(output_path, label))
        return
    print(text)
    _warn_demo_on_stderr()


def handle_sleep(args: argparse.Namespace) -> int:
    result = asyncio.run(_service(args).sleep_records(
        limit=args.limit, fresh=args.fresh,
        owner=getattr(args, "owner", None),
    ))
    _emit_decrypted(result, args, label="sleep sessions summary")
    return 0 if not result.get("errors") else 3


def handle_sleep_detail(args: argparse.Namespace) -> int:
    result = asyncio.run(_service(args).sleep_detail_records(
        limit=args.limit, fresh=args.fresh,
        owner=getattr(args, "owner", None),
        # CLI keeps the timeline (2026-07-28): it defaults to off for MCP callers
        # because the array is ~13k characters/night and overflows an LLM context,
        # but this output goes to a file or a terminal where size is not a budget.
        # This subcommand's own help text promises the timeline.
        include_timeline=True,
    ))
    _emit_decrypted(result, args, label="sleep detail summary (HR+RR+stage timeline)")
    return 0 if not result.get("errors") else 3


def handle_water(args: argparse.Namespace) -> int:
    summary = asyncio.run(_service(args).water_intake_summary(limit=args.limit, fresh=args.fresh, owner=getattr(args, "owner", None)))
    _emit_decrypted(summary, args, label="water intake summary")
    return 0 if not summary.get("errors") else 3


def handle_weight(args: argparse.Namespace) -> int:
    summary = asyncio.run(
        _service(args).weight_trend_summary(limit=args.limit, goal_kg=args.goal_kg, fresh=args.fresh, owner=getattr(args, "owner", None))
    )
    _emit_decrypted(summary, args, label="weight trend summary")
    return 0 if not summary.get("errors") else 3


def handle_menstrual(args: argparse.Namespace) -> int:
    # stderr, not stdout: stdout is the JSON document. Three of the fifteen data
    # subcommands printed this notice to stdout, which put a prose line ahead of
    # the JSON and broke `vaultbeat-apple-health menstrual | jq` outright — for
    # exactly the three kinds whose sensitivity the notice is about. The other
    # twelve were always clean, so the shape of the output depended on which
    # kind you asked for.
    #
    # The notice itself stays, and stays visible: a terminal shows stderr, so a
    # human reads it exactly as before. Only redirection changes, which is the
    # whole point — `> file` and `| jq` now receive the document alone.
    # `DEMO_BANNER` above already had it right; these three predate that.
    print(
        "Note: menstrual data is sensitive; it is decrypted locally and never re-exported. "
        "It only appears if the user explicitly opted in on iOS.",
        file=sys.stderr,
    )
    summary = asyncio.run(_service(args).menstrual_cycle_summary(limit=args.limit, fresh=args.fresh, owner=getattr(args, "owner", None)))
    _emit_decrypted(summary, args, label="menstrual cycle summary")
    return 0 if not summary.get("errors") else 3


def handle_activity(args: argparse.Namespace) -> int:
    summary = asyncio.run(_service(args).activity_summary(limit=args.limit, fresh=args.fresh, owner=getattr(args, "owner", None)))
    _emit_decrypted(summary, args, label="activity rings summary")
    return 0 if not summary.get("errors") else 3


def handle_resting_hr(args: argparse.Namespace) -> int:
    result = asyncio.run(_service(args).resting_hr_records(limit=args.limit, fresh=args.fresh, owner=getattr(args, "owner", None)))
    _emit_decrypted(result, args, label="resting heart rate summary")
    return 0 if not result.get("errors") else 3


def handle_workouts(args: argparse.Namespace) -> int:
    result = asyncio.run(_service(args).workout_records(limit=args.limit, fresh=args.fresh, owner=getattr(args, "owner", None)))
    _emit_decrypted(result, args, label="workouts summary")
    return 0 if not result.get("errors") else 3


def handle_mindfulness(args: argparse.Namespace) -> int:
    result = asyncio.run(_service(args).mindfulness_summary(limit=args.limit, fresh=args.fresh, owner=getattr(args, "owner", None)))
    _emit_decrypted(result, args, label="mindfulness summary")
    return 0 if not result.get("errors") else 3


def handle_hrv(args: argparse.Namespace) -> int:
    granularity = getattr(args, "granularity", "hourly")
    service_obj = _service(args)
    if granularity == "raw":
        result = asyncio.run(
            service_obj.hrv_records(
                limit=args.limit, fresh=args.fresh, owner=getattr(args, "owner", None)
            )
        )
        label = "HRV summary (SDNN raw samples)"
    else:
        result = asyncio.run(
            service_obj.hrv_hourly_records(
                limit=args.limit, fresh=args.fresh, owner=getattr(args, "owner", None)
            )
        )
        label = "HRV summary (SDNN hourly averages)"
    _emit_decrypted(result, args, label=label)
    return 0 if not result.get("errors") else 3


def handle_wrist_temp(args: argparse.Namespace) -> int:
    result = asyncio.run(_service(args).wrist_temp_records(limit=args.limit, fresh=args.fresh, owner=getattr(args, "owner", None)))
    _emit_decrypted(result, args, label="wrist temperature summary")
    return 0 if not result.get("errors") else 3


def handle_notes(args: argparse.Namespace) -> int:
    print(
        "Note: notes are sensitive free text; they are decrypted locally and never re-exported.",
        file=sys.stderr,
    )
    summary = asyncio.run(
        _service(args).notes_summary(limit=args.limit, target_kind=args.kind, fresh=args.fresh)
    )
    _emit_decrypted(summary, args, label="notes summary")
    return 0 if not summary.get("errors") else 3


def handle_strength(args: argparse.Namespace) -> int:
    summary = asyncio.run(
        _service(args).strength_summary(limit=args.limit, limit_days=args.days, fresh=args.fresh)
    )
    _emit_decrypted(summary, args, label="strength sessions summary")
    return 0 if not summary.get("errors") else 3


def handle_symptoms(args: argparse.Namespace) -> int:
    print(
        "Note: symptom data is sensitive; it is decrypted locally and never re-exported. "
        "It only appears if a user explicitly opted in on iOS.",
        file=sys.stderr,
    )
    summary = asyncio.run(_service(args).symptom_summary(limit=args.limit, fresh=args.fresh))
    _emit_decrypted(summary, args, label="symptoms summary")
    return 0 if not summary.get("errors") else 3


def _resolve_http_token(store: ConfigStore) -> str | None:
    # Pre-rename TETHER_ env var honored as a fallback for existing setups.
    env_token = os.getenv("VAULTBEAT_MCP_HTTP_TOKEN", "").strip() or os.getenv("TETHER_MCP_HTTP_TOKEN", "").strip()
    if env_token:
        return env_token
    config = store.load()
    return config.http_token if config else None


def _print_http_token(token: str, args: argparse.Namespace) -> None:
    url = f"http://{args.host}:{args.port}{args.path}"
    snippet = {
        "servers": {
            "vaultbeat-local": {
                "type": "http",
                "url": url,
                "headers": {"Authorization": f"Bearer {token}"},
            }
        }
    }
    print("HTTP bearer token (store it securely; shown only once):")
    print(f"  {token}")
    print()
    print("Send it from your MCP client as:")
    print(f"  Authorization: Bearer {token}")
    print()
    print("Example mcp.json (VS Code / Cursor style):")
    print(json.dumps(snippet, indent=2))


def handle_serve(args: argparse.Namespace) -> int:
    # Imported lazily: the MCP SDK import chain is ~1.4s, which every OTHER
    # subcommand (the CLI data path) must not pay.
    from vaultbeat_mcp_local.mcp_server import run_mcp_server

    store = _store(args)
    is_http = args.transport in ("http", "streamable-http")

    if args.generate_token:
        _print_http_token(store.ensure_http_token(), args)
        return 0

    if args.show_token:
        config = store.load()
        stored = config.http_token if config else None
        if not stored:
            print("No HTTP token set. Run `vaultbeat-apple-health serve --generate-token` first.")
            return 1
        print(stored)
        return 0

    token: str | None = None
    if is_http and not args.no_token:
        token = _resolve_http_token(store)
        if token:
            print(
                f"HTTP bearer auth enabled (token {token[:6]}..., len {len(token)}). "
                "Run `serve --show-token` to reveal it."
            )

    run_mcp_server(
        store,
        transport=args.transport,
        host=args.host,
        port=args.port,
        path=args.path,
        json_response=not args.sse_response,
        stateless_http=not args.stateful_http,
        token=token,
        allow_remote=args.allow_remote,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    # `--help` is where a stuck user checks what this thing is called. Reporting
    # the deprecated alias there sent them toward a name that `uvx` — the only
    # install path the docs give — cannot resolve.
    parser = argparse.ArgumentParser(prog="vaultbeat-apple-health")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--config", help="Path to config JSON. Defaults to ~/.tether/mcp-local/config.json.")
    # 🔴 A FLAG AND AN ENV VAR, NEVER A CONFIG FIELD. Both die with the process.
    # A persisted `"demo": true` would survive a reboot while being invisible in
    # every screen anyone looks at — and weeks later "how did I sleep last week?"
    # would get a confident, plausible, entirely fabricated answer about a real
    # person's health. That failure has no loud edge; it just quietly stops being
    # true. The cost of this choice is that demo mode must be re-stated on every
    # invocation, which is the point.
    parser.add_argument(
        "--demo",
        action="store_true",
        help=(
            "Serve SYNTHETIC health records instead of reading a real account. No "
            "binding, no network, no decryption; write commands refuse. For trying "
            "the tools out or reproducing a bug without handing over a key. Applies "
            "to this invocation only — equivalent to setting VAULTBEAT_DEMO=1."
        ),
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Generate a local keypair and config file.")
    init_parser.add_argument("--server-name", default="Local AI Server")
    init_parser.add_argument("--api-base-url", default=DEFAULT_API_BASE_URL)
    init_parser.set_defaults(func=handle_init)

    bind_parser = subparsers.add_parser("bind", help="Show a QR payload and wait for iOS authorization.")
    # Every option carries help text because `bind` is the first command a new user
    # runs, and until 2026-08-11 `bind --help` printed the flag names and nothing
    # else — not even the units on --timeout. --server-name matters most: it defaults
    # to a generic label, so a user with several machines ends up with a list of
    # identically-named servers and no way to tell which is which.
    bind_parser.add_argument(
        "--server-name",
        default="Local AI Server",
        help=(
            "Label shown in the iOS app's authorized-server list. Give each machine "
            "its own name (e.g. 'work laptop') — otherwise every server you bind "
            "carries the same default and you cannot tell them apart later, "
            "including when deciding which one is safe to remove. "
            "(default: %(default)s)"
        ),
    )
    bind_parser.add_argument(
        "--api-base-url",
        default=DEFAULT_API_BASE_URL,
        help="Vaultbeat cloud endpoint. Change only for self-hosting. (default: %(default)s)",
    )
    bind_parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        # 🔴 Read "SECONDS to keep the QR code valid ... or subscribe first"
        # until 2026-09-07. Both halves were wrong, and wrong in the same
        # direction: they made pairing sound narrower and more conditional than
        # it is, at the first command anyone runs.
        #
        # · This flag does not govern the QR's lifetime. It is how long THIS
        #   COMMAND waits. The pairing row on the cloud lives 600s
        #   (`mcp-bind-local` sets expires_at = now + 600000), so at the default
        #   the QR OUTLIVES the wait by five minutes — raising it up to that
        #   ceiling costs nothing. Saying the flag keeps the QR valid implies
        #   the opposite: that a timeout kills the code you are looking at.
        # · Pairing is open on EVERY plan. `vaultbeat_poll_binding`'s own
        #   docstring states it ("Connecting is open on every plan, so there is
        #   no tier to check"), so this line had the package contradicting
        #   itself — and the half a new user reads first is the one that puts a
        #   price in front of the one step that has none.
        help=(
            "SECONDS this command waits for the phone scan before giving up. "
            "Pairing is free on every plan; raise this if you still need to "
            "install the iOS app. The QR stays scannable for 10 minutes either "
            "way. (default: %(default)s)"
        ),
    )
    # 7.0s keeps polling under the cloud's 10/min IP rate limit (60/7 ≈ 8.6/min);
    # the old 3.0s default (20/min) self-tripped a 429 on first bind.
    bind_parser.add_argument(
        "--interval",
        type=float,
        default=7.0,
        help=(
            "SECONDS between checks for the scan. Lowering this is not faster — "
            "below ~6s it trips the server's rate limit and the bind fails. "
            "(default: %(default)s)"
        ),
    )
    bind_parser.add_argument(
        "--no-qr",
        action="store_true",
        help=(
            "Print the pairing payload as text only, without the QR block. For "
            "terminals that mangle block characters — scan it from another device "
            "or paste it manually."
        ),
    )
    bind_parser.set_defaults(func=handle_bind)

    poll_parser = subparsers.add_parser("poll", help="Poll once for a pending iOS authorization.")
    poll_parser.set_defaults(func=handle_poll)

    sync_parser = subparsers.add_parser("sync", help="Fetch and decrypt all encrypted health payloads.")
    sync_parser.add_argument("--limit", type=int)
    sync_parser.add_argument(
        "--fresh",
        action="store_true",
        help="Bypass the local record cache and force a cloud fetch.",
    )
    sync_parser.add_argument(
        "--metric-type",
        dest="metric_type",
        choices=sorted(KNOWN_METRIC_TYPES),
        help="Filter to a single metric type.",
    )
    sync_parser.add_argument("--output")
    sync_parser.set_defaults(func=handle_sync)

    sleep_parser = subparsers.add_parser(
        "sleep", help="Decrypt recent sleep sessions with stage breakdown."
    )
    sleep_parser.add_argument("--limit", type=int)
    sleep_parser.add_argument(
        "--owner",
        help="Only include records from this owner. Run `doctor` to see the real "
        "prefixes on this account — a made-up one matches nothing and returns "
        "an empty result rather than an error.",
    )
    sleep_parser.add_argument(
        "--fresh",
        action="store_true",
        help="Bypass the local record cache and force a cloud fetch.",
    )
    sleep_parser.add_argument("--output")
    sleep_parser.set_defaults(func=handle_sleep)

    sleep_detail_parser = subparsers.add_parser(
        "sleep-detail",
        help="Per-night timeline: each HR/RR sample tagged with the concurrent sleep stage.",
    )
    sleep_detail_parser.add_argument("--limit", type=int, help="Keep at most N most recent nights.")
    sleep_detail_parser.add_argument(
        "--owner",
        help="Only include records from this owner. Run `doctor` to see the real "
        "prefixes on this account — a made-up one matches nothing and returns "
        "an empty result rather than an error.",
    )
    sleep_detail_parser.add_argument(
        "--fresh",
        action="store_true",
        help="Bypass the local record cache and force a cloud fetch.",
    )
    sleep_detail_parser.add_argument("--output")
    sleep_detail_parser.set_defaults(func=handle_sleep_detail)

    water_parser = subparsers.add_parser(
        "water", help="Decrypt recent water intake and compute the daily average."
    )
    water_parser.add_argument("--limit", type=int)
    water_parser.add_argument(
        "--owner",
        help="Only include records from this owner. Run `doctor` to see the real "
        "prefixes on this account — a made-up one matches nothing and returns "
        "an empty result rather than an error.",
    )
    water_parser.add_argument(
        "--fresh",
        action="store_true",
        help="Bypass the local record cache and force a cloud fetch.",
    )
    water_parser.add_argument("--output")
    water_parser.set_defaults(func=handle_water)

    weight_parser = subparsers.add_parser(
        "weight",
        help="Decrypt recent body weight (kg) and compute the trend (latest/avg/min/max/weekly rate).",
    )
    weight_parser.add_argument("--limit", type=int, help="Keep at most N most recent body days.")
    weight_parser.add_argument(
        "--owner",
        help="Only include records from this owner. Run `doctor` to see the real "
        "prefixes on this account — a made-up one matches nothing and returns "
        "an empty result rather than an error.",
    )
    weight_parser.add_argument(
        "--fresh",
        action="store_true",
        help="Bypass the local record cache and force a cloud fetch.",
    )
    weight_parser.add_argument(
        "--goal-kg",
        dest="goal_kg",
        type=float,
        help="Goal weight in kg to compute delta_to_goal_kg (negative = below goal); omit to skip.",
    )
    weight_parser.add_argument("--output")
    weight_parser.set_defaults(func=handle_weight)

    menstrual_parser = subparsers.add_parser(
        "menstrual",
        help="Decrypt recent menstrual cycle (sensitive) and predict the next period.",
    )
    menstrual_parser.add_argument("--limit", type=int)
    menstrual_parser.add_argument(
        "--owner",
        help="Only include records from this owner. Run `doctor` to see the real "
        "prefixes on this account — a made-up one matches nothing and returns "
        "an empty result rather than an error.",
    )
    menstrual_parser.add_argument(
        "--fresh",
        action="store_true",
        help="Bypass the local record cache and force a cloud fetch.",
    )
    menstrual_parser.add_argument("--output")
    menstrual_parser.set_defaults(func=handle_menstrual)

    activity_parser = subparsers.add_parser(
        "activity", help="Decrypt recent daily activity rings (steps, energy, exercise, stand, distance)."
    )
    activity_parser.add_argument("--limit", type=int)
    activity_parser.add_argument(
        "--owner",
        help="Only include records from this owner. Run `doctor` to see the real "
        "prefixes on this account — a made-up one matches nothing and returns "
        "an empty result rather than an error.",
    )
    activity_parser.add_argument(
        "--fresh",
        action="store_true",
        help="Bypass the local record cache and force a cloud fetch.",
    )
    activity_parser.add_argument("--output")
    activity_parser.set_defaults(func=handle_activity)

    resting_hr_parser = subparsers.add_parser(
        "resting-hr", help="Decrypt recent resting heart rate samples."
    )
    resting_hr_parser.add_argument("--limit", type=int)
    resting_hr_parser.add_argument(
        "--owner",
        help="Only include records from this owner. Run `doctor` to see the real "
        "prefixes on this account — a made-up one matches nothing and returns "
        "an empty result rather than an error.",
    )
    resting_hr_parser.add_argument(
        "--fresh",
        action="store_true",
        help="Bypass the local record cache and force a cloud fetch.",
    )
    resting_hr_parser.add_argument("--output")
    resting_hr_parser.set_defaults(func=handle_resting_hr)

    workouts_parser = subparsers.add_parser(
        "workouts", help="Decrypt recent workout sessions."
    )
    workouts_parser.add_argument("--limit", type=int)
    workouts_parser.add_argument(
        "--owner",
        help="Only include records from this owner. Run `doctor` to see the real "
        "prefixes on this account — a made-up one matches nothing and returns "
        "an empty result rather than an error.",
    )
    workouts_parser.add_argument(
        "--fresh",
        action="store_true",
        help="Bypass the local record cache and force a cloud fetch.",
    )
    workouts_parser.add_argument("--output")
    workouts_parser.set_defaults(func=handle_workouts)

    mindfulness_parser = subparsers.add_parser(
        "mindfulness", help="Decrypt recent daily mindfulness summaries."
    )
    mindfulness_parser.add_argument("--limit", type=int)
    mindfulness_parser.add_argument(
        "--owner",
        help="Only include records from this owner. Run `doctor` to see the real "
        "prefixes on this account — a made-up one matches nothing and returns "
        "an empty result rather than an error.",
    )
    mindfulness_parser.add_argument(
        "--fresh",
        action="store_true",
        help="Bypass the local record cache and force a cloud fetch.",
    )
    mindfulness_parser.add_argument("--output")
    mindfulness_parser.set_defaults(func=handle_mindfulness)

    hrv_parser = subparsers.add_parser(
        "hrv", help="Decrypt recent HRV (SDNN) — hourly aggregated by default, raw with --granularity raw."
    )
    hrv_parser.add_argument("--limit", type=int)
    hrv_parser.add_argument(
        "--owner",
        help="Only include records from this owner. Run `doctor` to see the real "
        "prefixes on this account — a made-up one matches nothing and returns "
        "an empty result rather than an error.",
    )
    hrv_parser.add_argument(
        "--fresh",
        action="store_true",
        help="Bypass the local record cache and force a cloud fetch.",
    )
    hrv_parser.add_argument(
        "--granularity",
        choices=("hourly", "raw"),
        default="hourly",
        help="hourly = one bucket per UTC hour (30d window, ≤720 records, saves context). "
             "raw = one record per SDNN sample (3d window, spike-precision).",
    )
    hrv_parser.add_argument("--output")
    hrv_parser.set_defaults(func=handle_hrv)

    wrist_temp_parser = subparsers.add_parser(
        "wrist-temp", help="Decrypt recent sleeping wrist temperature samples."
    )
    wrist_temp_parser.add_argument("--limit", type=int)
    wrist_temp_parser.add_argument(
        "--owner",
        help="Only include records from this owner. Run `doctor` to see the real "
        "prefixes on this account — a made-up one matches nothing and returns "
        "an empty result rather than an error.",
    )
    wrist_temp_parser.add_argument(
        "--fresh",
        action="store_true",
        help="Bypass the local record cache and force a cloud fetch.",
    )
    wrist_temp_parser.add_argument("--output")
    wrist_temp_parser.set_defaults(func=handle_wrist_temp)

    symptoms_parser = subparsers.add_parser(
        "symptoms",
        help="Decrypt recent HealthKit symptom days (sensitive), grouped by data owner.",
    )
    symptoms_parser.add_argument("--limit", type=int)
    symptoms_parser.add_argument(
        "--fresh",
        action="store_true",
        help="Bypass the local record cache and force a cloud fetch.",
    )
    symptoms_parser.add_argument("--output")
    symptoms_parser.set_defaults(func=handle_symptoms)

    notes_parser = subparsers.add_parser(
        "notes",
        help=(
            "Decrypt recent free-text notes (iOS sleep/menstrual annotations and "
            "agent-authored mood/general notes, sensitive)."
        ),
    )
    notes_parser.add_argument("--limit", type=int)
    notes_parser.add_argument(
        "--fresh",
        action="store_true",
        help="Bypass the local record cache and force a cloud fetch.",
    )
    notes_parser.add_argument(
        # 2026-07-27: was choices=["sleep", "menstrual"] — the iOS-authored pair
        # only, so the mood/general notes this server writes itself were
        # unfilterable. Read side takes the union of both sets.
        "--kind", choices=sorted(READABLE_NOTE_KINDS), help="Keep only one target kind."
    )
    notes_parser.add_argument("--output")
    notes_parser.set_defaults(func=handle_notes)

    strength_parser = subparsers.add_parser(
        "strength",
        help="Decrypt recent strength-training sessions (exercise-level sets × reps).",
    )
    strength_parser.add_argument("--limit", type=int)
    strength_parser.add_argument(
        "--fresh",
        action="store_true",
        help="Bypass the local record cache and force a cloud fetch.",
    )
    strength_parser.add_argument(
        "--days", type=int, help="Keep only the most recent N sessions."
    )
    strength_parser.add_argument("--output")
    strength_parser.set_defaults(func=handle_strength)

    status_parser = subparsers.add_parser("status", help="Show local binding state.")
    status_parser.set_defaults(func=handle_status)

    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Self-diagnose the install and binding (config, key, cloud reachability, data round-trip).",
    )
    doctor_parser.add_argument("--json", action="store_true", help="Machine-readable output.")
    doctor_parser.set_defaults(func=handle_doctor)

    serve_parser = subparsers.add_parser("serve", help="Run the MCP server over stdio or streamable HTTP.")
    serve_parser.add_argument(
        "--transport",
        choices=["stdio", "http", "streamable-http"],
        default="stdio",
        help="MCP transport. `http` is an alias for `streamable-http`.",
    )
    serve_parser.add_argument("--host", default="127.0.0.1", help="HTTP bind host when using HTTP transport.")
    serve_parser.add_argument("--port", type=int, default=8000, help="HTTP bind port when using HTTP transport.")
    serve_parser.add_argument("--path", default="/mcp", help="HTTP endpoint path when using HTTP transport.")
    serve_parser.add_argument(
        "--sse-response",
        action="store_true",
        help="Use SSE-style HTTP responses instead of JSON responses.",
    )
    serve_parser.add_argument(
        "--stateful-http",
        action="store_true",
        help="Disable stateless HTTP mode for clients that require sessions.",
    )
    serve_parser.add_argument(
        "--allow-remote",
        action="store_true",
        help="Permit binding a non-loopback host (requires a token); confirms intentional network exposure.",
    )
    serve_parser.add_argument(
        "--generate-token",
        action="store_true",
        help="Generate and persist an HTTP bearer token, print it once with client config, then exit.",
    )
    serve_parser.add_argument(
        "--show-token",
        action="store_true",
        help="Print the stored HTTP bearer token and exit.",
    )
    serve_parser.add_argument(
        "--no-token",
        action="store_true",
        help="Loopback only: serve HTTP without bearer auth.",
    )
    serve_parser.set_defaults(func=handle_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Set here, once, rather than inside `_service()` / `_store()`: `serve` never
    # goes through either — it hands a ConfigStore to `run_mcp_server`, which asks
    # `demo_enabled()` itself — so a flag applied in the service factory would be
    # silently ignored by the one subcommand a reviewer is most likely to run.
    #
    # Only ever set, never unset: `--demo` turns demo mode ON, and its absence
    # means "whatever the environment already said", so a wrapper that exports
    # VAULTBEAT_DEMO keeps working.
    if getattr(args, "demo", False):
        os.environ[DEMO_ENV] = "1"

    try:
        return int(args.func(args))
    except Exception as error:
        # Always name the exception type. Several failures this CLI actually hits
        # stringify to nothing — `httpx.ReadTimeout` is the worst of them, and it
        # is also the most common (Supabase edge cold starts). Printing only
        # `str(error)` turned three consecutive cold-backup failures on
        # 2026-07-27 into a bare `error:` with zero diagnostic content; finding
        # the cause meant bypassing this handler and calling `args.func(args)` by
        # hand to see the traceback.
        detail = str(error)
        kind = type(error).__name__
        parser.exit(1, f"error: {kind}: {detail}\n" if detail else f"error: {kind}\n")


if __name__ == "__main__":
    raise SystemExit(main())
