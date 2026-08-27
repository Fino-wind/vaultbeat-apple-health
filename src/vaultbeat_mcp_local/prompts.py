"""The prompt library served over MCP `prompts/list` and `prompts/get`.

WHY THIS EXISTS
---------------
`prompts/*` is the only protocol channel through which an agent can ASK what a
server is good for. Without it, the usage of 29 tools exists solely in their
docstrings and every user's agent invents its own analysis routine — which means
the quality of the answer is a property of whichever agent happened to connect,
and we have no purchase at all on the two mistakes that do real damage with
health data: reporting an association as a cause, and reading a gap as a zero.

WHAT IS ACTUALLY WORTH COPYING FROM THE COMPETITOR
--------------------------------------------------
Not the count. `health-export-mcp` ships 22 prompts; the transferable idea is
that its safety layer is a pair of CONCATENATED CONSTANTS (`STYLE`, `ABSENCE`)
rather than a paragraph retyped into each entry. That is Invariant 58
(one-funnel-per-event) applied to prose: the discipline needs one place to be
edited, or half the library ends up describing an older policy than the other
half. This file keeps that shape and writes its own two constants, because ours
have to say different things (see `ABSENCE` — an empty result here has four
causes, not one).

The set is deliberately small. Every entry below answers "whose problem, and
which specific way does an agent get this wrong without it"; a prompt that
cannot answer that is a tool list in a trench coat, and an agent that already
has `tools/list` does not need one.

HOUSE RULES FOR ADDING ONE
--------------------------
· Name the tools it should call, so the agent does not have to guess between
  `get_sleep_detail` and `get_sleep` (there is no `get_sleep`).
· Append `STYLE`, and append `ABSENCE` too whenever the prompt reads a kind that
  can legitimately be empty — which is most of them.
· Never encode a threshold, band or grade. "Good HRV" is not ours to define, and
  a number invented in a prompt is indistinguishable, downstream, from one that
  came out of the data.
· Placeholders are `{{name}}`, and every argument carries its own `fallback`
  for when the caller omits it — never a dangling `{{...}}` left in the text for
  the model to read as literal content. Write the fallback and the surrounding
  sentence together; they have to make one grammatical sentence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "ABSENCE",
    "PROMPTS",
    "STYLE",
    "PromptArg",
    "VaultbeatPrompt",
    "register_prompts",
    "render_prompt",
]


# ── The two constants the whole library shares ──────────────────────────────

#: Appended to every prompt. The rules an answer about someone's body has to
#: obey regardless of which kind it is reading.
#:
#: The last clause is the one that is ours rather than industry-standard, and it
#: is lifted deliberately from the Product Positioning rule at the top of
#: `AGENTS.md`: the app may state what was OBSERVED and may not state what was
#: inferred from the clock. That rule was written after a lock screen rendered a
#: guess ("partner asleep", derived from the hour) in the same typeface as a
#: fact. An agent holding these tools is one more surface where a guess can be
#: dressed as a reading, and it is the surface with the most fluent voice.
STYLE = (
    " Style rules: say what the data covers before you conclude anything — which "
    "days are present, which are missing, and how recent the newest one is. "
    "Describe, do not prescribe: no diagnosis, no treatment advice, no invented "
    "scores, grades or verdicts. State an association as an association, never as "
    "a cause. Quote the actual numbers, units and dates you used. And when the "
    "honest answer is that you do not know, give the last thing that IS known "
    "with its timestamp and stop there, rather than filling the gap with "
    "something that reads like a measurement."
)

#: Appended wherever a kind can legitimately come back empty, which is nearly
#: everywhere.
#:
#: 🔴 Ours is longer than the competitor's one-liner on purpose: their absence
#: has a single cause (not exported), ours has four, and they call for opposite
#: actions. `service.py`'s `_capability_report` documents the cost of getting
#: this wrong — its note used to omit the cause a brand-new user actually hits
#: (a freshly bound server starts with an empty history because every server
#: gets its own encrypted copy) and sent people to check their iOS version and
#: Health permissions, both dead ends. That is Invariant 57
#: (absence-has-more-than-one-cause), and an agent guessing between the four is
#: the same failure with a friendlier voice.
ABSENCE = (
    " Absence rule: an empty result never means the value was zero. It means one "
    "of four different things — the data was never recorded, Apple Health "
    "permission was not granted for that type, the iOS app is older than the "
    "feature, or this server was bound recently and its own encrypted copy of "
    "the history has not finished syncing yet. Those need opposite fixes, so do "
    "not pick one. Call `vaultbeat_doctor`, which is the tool that tells them "
    "apart, and report what it says."
)

#: Prefixed when the server is running on synthetic data. Prompts are answers
#: too — Invariant 61 (demo-is-a-boundary-not-a-flag) asks that every surface
#: say so, and `prompts/list` is a surface a client can read before it calls a
#: single tool.
DEMO_PREFIX = (
    "[SYNTHETIC DEMO DATA] This server is in demo mode: every number these tools "
    "return is generated locally and belongs to no real person. Follow the "
    "instruction below as written, but say plainly in your answer that it "
    "describes a demo. "
)


@dataclass(frozen=True)
class PromptArg:
    """One optional argument, and what the text should say without it.

    🔴 The fallback is PER ARGUMENT rather than one generic sentence, because a
    generic one has to fit every grammatical slot and does not. "…looked out of
    date(unspecified context — pick a sensible one)" was the first draft's real
    output: correct by the rule, unreadable in place. Arguments split into two
    kinds and they need opposite treatment — a window length has a sensible
    default the model should pick and announce, while an optional aside should
    simply not be there when it was not given.
    """

    name: str
    description: str
    fallback: str
    """Substituted verbatim when the caller omits the argument. May be empty."""


@dataclass(frozen=True)
class VaultbeatPrompt:
    """One entry. Plain data — no functions, no environment reads.

    Kept serialisable so the same table can be rendered somewhere that is not a
    running MCP server (a docs page, the website's tool list) without importing
    the SDK.
    """

    name: str
    title: str
    description: str
    template: str
    arguments: tuple[PromptArg, ...] = field(default=())
    """All optional, per the MCP spec — see `render_prompt`."""


# ── The library ─────────────────────────────────────────────────────────────
#
# Eight. Each line of the comment above each entry is the "whose problem" test.

PROMPTS: tuple[VaultbeatPrompt, ...] = (
    # Anyone, every morning. Without it an agent reaches for whichever three
    # kinds it thought of first and calls the result a readiness score.
    VaultbeatPrompt(
        name="daily_brief",
        title="Daily brief",
        description="A short readout of the most recent day, set against the fortnight behind it.",
        template=(
            "Give me a brief on my most recent day of Vaultbeat data. Call "
            "`vaultbeat_status` first and name the newest date you can see; if it is "
            "older than yesterday, say so plainly and keep every claim inside that "
            "date. Then read `get_sleep_detail`, `get_hrv`, `get_resting_hr` and "
            "`get_activity` for the latest day, and set each against the 14 days "
            "before it so a number has something to be compared with. Finish with the "
            "single largest change and the numbers behind it. Under 150 words."
            + STYLE
            + ABSENCE
        ),
    ),
    # Anyone who wears the watch to bed. The specific trap is ours: this schema
    # distinguishes "in bed, never measured" from "slept zero minutes"
    # (Invariant 39 (in-bed-is-not-zero)) and an agent that averages the first
    # into the second reports nights that did not happen.
    VaultbeatPrompt(
        name="sleep_review",
        title="Sleep review",
        description="A descriptive read of recent nights: duration, stages, and the nights that were not measured.",
        template=(
            "Describe my recent sleep. Use `get_sleep_detail` for the {{nights}} "
            "nights. Report duration and the stage breakdown where stages exist. "
            "🔴 Treat `is_in_bed_only` as its own category: those are nights the watch "
            "recorded time in bed without ever measuring sleep, and folding them in as "
            "zero-minute nights invents bad nights that did not happen — read "
            "`duration_label` rather than reasoning from the raw minutes. Say how many "
            "nights of the window actually carried measurements before you describe "
            "any trend."
            + STYLE
            + ABSENCE
        ),
        arguments=(PromptArg("nights", "How many nights to read. Defaults to 14.", "last 14"),),
    ),
    # Anyone eating to a target. The trap is arithmetic and silent: basal energy
    # arrives one blob per UTC hour, so a day missing hours produces a TDEE that
    # is simply too low, and nothing about the number looks wrong.
    VaultbeatPrompt(
        name="energy_balance",
        title="Energy balance",
        description="Calories in against calories out, with the incomplete days excluded rather than averaged in.",
        template=(
            "Work out my energy balance over the {{days}} days. Read "
            "`get_total_energy_burned` and `get_food_log` for the same window. "
            "🔴 Before averaging anything, check the coverage fields the burn data "
            "carries: basal energy is stored one record per hour, so a day with only "
            "part of its hours present reports a total that is too low for a reason "
            "that has nothing to do with me. The tool already excludes short days from "
            "its own average and names them — use its `average_tdee_kcal` and its "
            "stated day count rather than re-averaging the daily rows yourself, and "
            "tell me which days were dropped. Food logging is manual, so a day with no "
            "entries is a day I did not log, not a day I did not eat: never subtract "
            "from an unlogged day."
            + STYLE
            + ABSENCE
        ),
        arguments=(PromptArg("days", "Window length in days. Defaults to 14.", "last 14"),),
    ),
    # Anyone lifting. Strength data here is hand-logged sets rather than watch
    # exports, so the useful comparison is volume over weeks, and the recovery
    # signals sit in different tools entirely.
    VaultbeatPrompt(
        name="training_block_review",
        title="Training block review",
        description="Recent lifting and cardio volume, next to the recovery signals from the same weeks.",
        template=(
            "Review my recent training. Use `get_strength_log` for the {{days}} "
            "days and report per-session volume (sets, reps, load) and how the working "
            "weights moved per movement. Add `get_workouts` for the cardio in the same "
            "window, and `get_vo2max` if it has anything — it is sparse by nature, "
            "computed only during outdoor brisk bouts, so days apart is normal and not "
            "a gap to explain. Then put `get_resting_hr` and `get_hrv` beside it for "
            "the same dates. Those two are context, not a verdict: say what moved "
            "together and leave it there, because a week where both moved is still one "
            "person and one week."
            + STYLE
            + ABSENCE
        ),
        arguments=(PromptArg("days", "Window length in days. Defaults to 28.", "last 28"),),
    ),
    # For the person tracking a cycle. The trap is comparing to the adjacent
    # week, which mixes phases and manufactures a trend out of the rhythm.
    VaultbeatPrompt(
        name="cycle_aware_read",
        title="Cycle-aware read",
        description="Read a metric against the same phase of earlier cycles instead of against last week.",
        template=(
            "Give me a cycle-aware read of {{metric}}. Start with "
            "`get_menstrual_cycle` to get the recorded cycle starts and the observed "
            "cycle lengths. Then compare the current phase with the SAME phase of "
            "earlier cycles, using date ranges you build from those starts — not with "
            "the adjacent weeks, which mix phases together and manufacture a trend out "
            "of an ordinary rhythm. `get_wrist_temp` and `get_symptoms` belong in this "
            "picture where they have data. Say how many prior cycles you actually had "
            "to compare against; with fewer than two, say a like-phase comparison is "
            "not possible yet and stop rather than substituting a weekly one. Phase "
            "labels are coarse and derived from logged periods; they predict nothing."
            + STYLE
            + ABSENCE
        ),
        arguments=(PromptArg("metric", "Which metric to read through the cycle, e.g. sleep, HRV, resting heart rate.",
                             "whichever metric has the most complete history — say which you picked"),),
    ),
    # Both people in a paired account — the thing no competitor has. Two traps:
    # results can silently mix owners, and the moment a partner's numbers get an
    # opinion attached, sharing stops being safe to leave on.
    VaultbeatPrompt(
        name="partner_check_in",
        title="Partner check-in",
        description="Read both people's shared data side by side, without turning either into a judgement.",
        template=(
            "Show me how both of us are doing over the {{days}} days. These tools "
            "take an `owner` prefix; call each one twice, once per person, rather than "
            "reading an unfiltered result — an unfiltered call returns both people's "
            "rows together and carries a `mixed_owners` warning for exactly that "
            "reason, and an average across two bodies is a number that was true of "
            "neither. Use whichever of `get_sleep_detail`, `get_hrv`, `get_resting_hr`, "
            "`get_activity` and `get_weight_trend` have data for both. Report the two "
            "columns plainly. 🔴 Sharing is opt-in per data type and is a kindness, not "
            "a monitoring feature: describe what my partner's numbers show and stop "
            "there. No assessment of them, no advice for them, nothing they would be "
            "unhappy to read over my shoulder. If a type is present for one of us and "
            "not the other, that is a sharing toggle, never a finding."
            + STYLE
            + ABSENCE
        ),
        arguments=(PromptArg("days", "Window length in days. Defaults to 14.", "last 14"),),
    ),
    # Anyone who talks to their agent while at the gym or the table. The trap is
    # that a helpful model fills in a set it was not told about; here that writes
    # a fabricated number into someone's permanent health record.
    VaultbeatPrompt(
        name="log_from_conversation",
        title="Log what I just told you",
        description="Turn something I said in passing into a Vaultbeat entry, without inventing the parts I left out.",
        template=(
            "Log this into Vaultbeat: {{entry}}. Pick the right tool — "
            "`log_strength_entry` for sets and reps, `log_food_entry` for a meal, "
            "`log_weight_entry` for a weigh-in, `log_note` for anything subjective "
            "such as symptoms, mood or how a session felt. Use the `_append` variants "
            "when adding to something already logged today, so an afternoon meal does "
            "not overwrite the morning one. 🔴 Write only what I actually said. If a "
            "weight, a rep count, a portion or a time is missing, ask me — do not "
            "reach for a plausible value and do not carry one over from a previous "
            "session. This goes into a permanent record I will later read as fact, and "
            "a number I never said is worse than a gap. Read the entry back to me "
            "after writing it."
            + STYLE
        ),
        arguments=(PromptArg("entry", "What to log, in plain words — e.g. 'squat 5x5 at 80kg' or 'ate two eggs and oats'.",
                             "(nothing was given — ask me what to log and write nothing until I answer)"),),
    ),
    # Anyone who just installed this and sees nothing. Documented as the most
    # expensive wrong turn in the product: the four causes need opposite fixes,
    # and the natural guess (permissions) is usually the wrong one on day one.
    VaultbeatPrompt(
        name="why_is_this_empty",
        title="Why is this empty?",
        description="Work out which of the four causes is behind an empty or stale result, instead of guessing.",
        template=(
            "A Vaultbeat read came back empty or looked out of date. {{context}}Work "
            "out which cause it is before suggesting anything. Call `vaultbeat_doctor` "
            "and read all of it, including the checks it says it did NOT run and the "
            "scope note about what it structurally cannot see — this process is a "
            "subprocess of my AI client, so it cannot inspect the command line, the "
            "environment or the config file the client actually used, and a green "
            "check is not evidence about that half. Then say which of these it is: "
            "nothing was ever recorded, Apple Health permission is missing for that "
            "type, the iOS app predates the feature, or this server was bound recently "
            "and its own encrypted copy of the history is still filling in. Name one, "
            "give the single next action for that one, and say what would prove you "
            "right. Do not hand me a checklist of all four."
            + STYLE
        ),
        arguments=(PromptArg("context", "Optional — which tool was empty, and what you expected to see.", ""),),
    ),
)


# ── Rendering ───────────────────────────────────────────────────────────────


def render_prompt(prompt: VaultbeatPrompt, arguments: dict[str, Any] | None = None, *, demo: bool = False) -> str:
    """Fill `{{placeholders}}` and return the final instruction text.

    🔴 An argument the caller left out becomes an instruction to choose, never a
    surviving `{{days}}`. A dangling placeholder is not a neutral failure: the
    model reads it as literal content and either quotes it back or invents a
    binding for it, and both look like the prompt worked. Every argument here is
    optional by design (the MCP `prompts/get` spec lets a client omit them), so
    this path is the normal one, not the error one.
    """

    text = prompt.template
    supplied = arguments or {}
    for argument in prompt.arguments:
        value = supplied.get(argument.name)
        # A blank string is a client sending nothing, not a value of nothing.
        rendered = str(value).strip() if value is not None and str(value).strip() else argument.fallback
        text = text.replace("{{" + argument.name + "}}", rendered)
    return (DEMO_PREFIX + text) if demo else text


def register_prompts(mcp: Any, *, demo: bool = False) -> int:
    """Install every prompt on a FastMCP instance. Returns how many.

    ⚠️ `demo` is PASSED IN rather than read from the environment here, matching
    how `run_mcp_server` freezes it once at startup and hands it down. The
    reason is recorded there in full: the server used to re-read the environment
    per call, so flipping it mid-session left half the surfaces saying "real"
    while the data had already become synthetic. Every surface that can say
    "demo" has to learn it from the same frozen value or they can disagree.

    Built with `Prompt(...)` directly rather than the `@mcp.prompt()` decorator
    because the decorator derives a prompt's arguments from a Python function
    signature, and these are table data — the alternative is synthesising eight
    functions with eight signatures to describe strings we already have.
    """

    from mcp.server.fastmcp.prompts.base import Prompt, PromptArgument

    for entry in PROMPTS:
        def _render(_entry: VaultbeatPrompt = entry, **arguments: Any) -> str:
            return render_prompt(_entry, arguments, demo=demo)

        mcp.add_prompt(
            Prompt(
                name=entry.name,
                title=entry.title,
                description=entry.description,
                arguments=[
                    PromptArgument(name=a.name, description=a.description, required=False)
                    for a in entry.arguments
                ],
                fn=_render,
                # Explicit despite defaulting to None in the model: the SDK
                # declares it without a default, so mypy requires it. It is the
                # name of the parameter FastMCP would inject a `Context` into,
                # and these renderers take none.
                context_kwarg=None,
            )
        )
    return len(PROMPTS)
