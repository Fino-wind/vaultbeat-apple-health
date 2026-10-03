"""Arithmetic over already-decrypted daily series: trend, period compare, correlate.

WHY THIS EXISTS
---------------
Without it an agent that wants "is my resting heart rate drifting up?" has to
pull the raw rows and do the regression itself, once per question, and it does
that badly — a Pearson coefficient computed token-by-token is the single most
reliably wrong number an LLM produces. Three tools move that arithmetic to a
place where it is deterministic, testable, and identical between two sessions
asking the same thing.

WHAT IT DELIBERATELY DOES NOT DO
--------------------------------
It does not interpret. No "strong correlation", no "improving", no "healthy
range", no verdict, no grade. That is the `Vaultbeat does not render` line from
`CLAUDE.md` applied one layer down: **honesty is ours, conclusions are the
user's and their agent's.** A module that shipped the word "strong" beside an
r of 0.6 would be deciding, on behalf of every user, that 0.6 is strong — for
their body, for that pair of metrics, over that many days. It is not ours to
decide, and an adjective travels much further than the number it came from.

So every function here returns numbers plus the facts needed to judge them
(`n_days`, `n_pairs`, `span_days`), never an adjective.

THREE INVARIANTS THIS LAYER MUST NOT BREAK
------------------------------------------
1. **Never invent a day.** Missing days are not interpolated, not carried
   forward, and not read as zero — that is Invariant 57 (absence has more than
   one cause) in arithmetic form. `correlate` uses only days present on BOTH
   sides, and says how many that was.
2. **Never collapse without saying so.** When a day holds several samples they
   are averaged, and `rows_per_day` reports it, because "30 rows" over 4 days is
   a different answer than over 30 and the shape of the output cannot tell them
   apart on its own (Invariant 62).
3. **Report refusal as refusal.** Too few points returns `null` plus a `reason`,
   never a number computed from two points that will read as a finding.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Any

__all__ = [
    "SERIES",
    "SeriesSpec",
    "compare_periods",
    "correlate",
    "daily_series",
    "excluded_days",
    "series_catalog",
    "trend",
]

#: The fewest aligned points that produce a coefficient at all.
#:
#: Three, not two: any two points are perfectly collinear, so a two-point
#: Pearson is ±1.0 by construction and carries no information whatsoever while
#: looking exactly like a finding. Refusing is the only honest output there.
#: This is a floor on ARITHMETIC VALIDITY, not a judgement about sufficiency —
#: three points is still very little, which is why `n_pairs` always ships beside
#: the coefficient rather than being replaced by a "reliable/unreliable" flag we
#: would have had to invent a threshold for.
MIN_PAIRS_FOR_CORRELATION = 3

#: Below this a slope is arithmetic noise dressed as a direction.
MIN_POINTS_FOR_TREND = 3


@dataclass(frozen=True)
class SeriesSpec:
    """How to get one number per day out of one read tool's response.

    `array` + `field` are a path into the summary that read tool already
    returns, so a series can never disagree with what `get_<kind>` prints — the
    alternative (a second query path) is the shape that lets two surfaces of the
    same product report different numbers.
    """

    name: str
    """What an agent passes. Named for the QUANTITY, not for the tool."""

    method: str
    """The `VaultbeatLocalService` coroutine that fetches it."""

    array: str
    """Which list in that response holds the per-day rows."""

    field: str
    """The numeric key inside a row."""

    unit: str

    direction_note: str = ""
    """Optional: what a rising line means, when that is not obvious.

    Only ever a UNIT fact ("more is longer sleep"), never a health claim
    ("higher is better") — the second is a verdict and does not belong here.
    """

    exclude_when: tuple[str, ...] = ()
    """Row flags that mean "this day's value is not a real day's value".

    A row carrying any of these set truthy is dropped BEFORE bucketing and
    reported by `excluded_days`. It exists for the kinds whose short days are
    short because the DATA is short (Invariant 62 (coverage-before-average)):
    basal energy on a day the Watch spent on the charger reads as a low
    metabolism, and averaging it in drags every number built on it down — in
    one direction only, so the error never cancels. The read method already
    computes the flag; this is how the arithmetic layer honours it instead of
    re-deriving a threshold of its own.
    """

    cumulative: bool = False
    """True when a day's value is a total ACCUMULATED over that day.

    Steps and energy accrue: a day's number is the sum of everything that
    happened, and a half-finished day reads as a low day rather than a missing
    one. Resting heart rate and VO2max are measurements of a state instead —
    today's value does not grow as the day goes on, so a fresh one is as
    complete as it will ever be.

    The distinction is load-bearing for the reader, not decorative: on a
    cumulative series the newest value is routinely a partial day and must not
    be compared against completed ones, which is exactly the mistake an agent
    makes when it reports "steps are down today" at 9am.
    """


#: Every quantity that is genuinely one-number-per-day.
#:
#: Sleep is present as per-night numbers read from `sleep_nights` (duration,
#: timing, stages, awakenings, vitals); its night-by-night table is its own tool.
#: Kinds with an irreducibly richer shape (strength sets, food
#: entries, notes, symptoms, cycle, workouts) are deliberately ABSENT: flattening
#: a workout into "duration" would answer a question nobody asked while hiding
#: the ones they did. Their tools stay the way to read them.
SERIES: tuple[SeriesSpec, ...] = (
    # 🔴 A night the Watch was not worn carries `total_sleep_minutes: 0` and
    # `is_in_bed_only: true` — sleep was never MEASURED, not zero (Invariant 39
    # (in-bed-is-not-zero)). The sleep tools have said so since 2026-07-27;
    # this spec did not, so from 0.7.0 every average / trend / correlation over
    # sleep counted those nights as zero sleep (2026-09-24: 18 of 400 nights,
    # mean 416 → 435 min once excluded). Same shape as the wrist-temp note:
    # the read tool knew, the series did not.
    SeriesSpec("sleep_minutes", "sleep_nights", "nights", "asleep_minutes", "minutes",
               "higher means more time asleep that night; nights the Watch was not worn are "
               "excluded and listed, never read as zero; so are days whose only sleep was in the "
               "daytime (`daytime_main_sleep`: a nap, not a night) and main sleeps under 3 h with "
               "no stage breakdown (`short_unstaged`: a nap or a partly recorded night). Each "
               "excluded day is listed with its times and minutes, and stays in `get_sleep_nights`",
               exclude_when=("is_in_bed_only", "motion_inferred", "daytime_main_sleep", "short_unstaged")),
    # ── Sleep structure and timing (2026-09-24) ─────────────────────────────
    # Every one reads `sleep_nights`, the same rows `get_sleep_nights` prints,
    # so `get_metric` fetches them in ONE decrypt and an average can always be
    # checked against the nights it came from.
    #
    # Clock series are MINUTES FROM THE MIDNIGHT THAT OPENS THE WAKE DAY,
    # negative before it: a clock time cannot be averaged as a clock time
    # (23:50 and 00:10 average to noon). Stage series are excluded, not
    # zeroed, on nights without stage detail (Apple did not stage them) — the same trap
    # `sleep_minutes` fell into with unworn nights.
    SeriesSpec("bedtime_minutes", "sleep_nights", "nights", "bedtime_minutes", "minutes from midnight",
               "when sleep began, in minutes from the midnight opening the wake day: -30 = 23:30, "
               "90 = 01:30. Averages across midnight correctly this way; convert back to a clock "
               "time before telling the user. Days whose only main sleep began after 08:00 and "
               "ended the same day (a daytime or evening nap), and main sleeps under 3 h with no "
               "stage breakdown, are excluded and listed",
               exclude_when=("is_in_bed_only", "motion_inferred", "daytime_main_sleep", "short_unstaged")),
    SeriesSpec("wake_minutes", "sleep_nights", "nights", "wake_minutes", "minutes from midnight",
               "when the night's main sleep ended, minutes from that day's midnight (450 = 07:30)",
               exclude_when=("is_in_bed_only", "motion_inferred", "daytime_main_sleep", "short_unstaged")),
    SeriesSpec("sleep_midpoint_minutes", "sleep_nights", "nights", "midpoint_minutes", "minutes from midnight",
               "halfway between bedtime and wake, minutes from midnight — the usual measure of "
               "sleep timing; compare weekdays with weekends for social jet lag",
               exclude_when=("is_in_bed_only", "motion_inferred", "daytime_main_sleep", "short_unstaged")),
    SeriesSpec("deep_sleep_minutes", "sleep_nights", "nights", "deep_minutes", "minutes",
               "Watch-staged nights only", exclude_when=("is_in_bed_only", "motion_inferred", "no_stage_detail")),
    SeriesSpec("rem_sleep_minutes", "sleep_nights", "nights", "rem_minutes", "minutes",
               "Watch-staged nights only", exclude_when=("is_in_bed_only", "motion_inferred", "no_stage_detail")),
    SeriesSpec("core_sleep_minutes", "sleep_nights", "nights", "core_minutes", "minutes",
               "Watch-staged nights only", exclude_when=("is_in_bed_only", "motion_inferred", "no_stage_detail")),
    SeriesSpec("awake_in_sleep_minutes", "sleep_nights", "nights", "awake_minutes", "minutes",
               "time scored awake inside the night's sleep; Watch-staged nights only",
               exclude_when=("is_in_bed_only", "motion_inferred", "no_stage_detail")),
    SeriesSpec("deep_sleep_percent", "sleep_nights", "nights", "deep_percent", "%",
               "deep as a share of time asleep that night (0-100); average this rather than "
               "dividing two averaged series", exclude_when=("is_in_bed_only", "motion_inferred", "no_stage_detail")),
    SeriesSpec("rem_sleep_percent", "sleep_nights", "nights", "rem_percent", "%",
               "REM as a share of time asleep that night (0-100)",
               exclude_when=("is_in_bed_only", "motion_inferred", "no_stage_detail")),
    SeriesSpec("awakenings", "sleep_nights", "nights", "awakenings", "count",
               "awake intervals between the first and last asleep sample; waking for the day "
               "is not counted. Watch-staged nights only",
               exclude_when=("is_in_bed_only", "motion_inferred", "no_stage_detail")),
    SeriesSpec("longest_sleep_bout_minutes", "sleep_nights", "nights", "longest_sleep_bout_minutes",
               "minutes", "the longest run of sleep with no awake interval in it",
               exclude_when=("is_in_bed_only", "motion_inferred", "no_stage_detail")),
    SeriesSpec("sleep_24h_minutes", "sleep_nights", "nights", "total_sleep_24h_minutes", "minutes",
               "ALL measured sleep that day — the main sleep plus naps and the other half of a "
               "broken night. `sleep_minutes` is the main sleep alone, over the same days: a day "
               "with no night sleep (only a daytime nap or a short fragment) is excluded from both "
               "and listed",
               # The same days as `sleep_minutes`, so the day's total can never
               # average BELOW its main sleep: with the short fragments in here and
               # out of there, a year read 393 against 403 minutes (2026-10-03).
               exclude_when=("is_in_bed_only", "motion_inferred", "daytime_main_sleep", "short_unstaged")),
    SeriesSpec("sleep_segments", "sleep_nights", "nights", "sleep_segments", "count",
               "separate sleeps that day; 1 = one unbroken main sleep, 2+ = naps or a broken night",
               exclude_when=("is_in_bed_only", "motion_inferred",)),
    SeriesSpec("sleep_hr_mean", "sleep_nights", "nights", "sleep_hr_mean", "bpm",
               "mean heart rate across the asleep stages of the night",
               exclude_when=("is_in_bed_only", "motion_inferred",)),
    SeriesSpec("sleep_rr_mean", "sleep_nights", "nights", "sleep_rr_mean", "breaths/min",
               "mean respiratory rate across the asleep stages of the night",
               exclude_when=("is_in_bed_only", "motion_inferred",)),
    # `in_bed_minutes` was a series here until 2026-09-24 and is deliberately
    # gone: it is the total of `inBed` samples, which a Watch night does not
    # write, so it read 0 on 381 of 400 real nights and averaged to "18 minutes
    # in bed" for September. It is not one number per night; it is the
    # complement of sleep_minutes. The field stays on the sleep tools' rows,
    # where it is read beside `is_in_bed_only` and means what it says.
    SeriesSpec("resting_hr", "resting_hr_records", "records", "bpm", "bpm"),
    SeriesSpec("hrv_sdnn", "hrv_records", "records", "sdnn_ms", "ms",
               "a day's value is the mean of that day's samples; for the samples themselves "
               "use get_intraday"),
    # 🔴 Was `wrist_temp_delta` reading `temperature_delta_celsius`, with a note
    # saying the value is "a delta from the wearer's own baseline, so it is
    # signed". Both halves were false: that field carries the ABSOLUTE reading
    # (byte-identical to `wrist_temperature_celsius` on every row checked,
    # 2026-09-23), which the old `get_wrist_temp` docstring already admitted as
    # a wire-contract misnomer. A cycle analysis reading 35.7 as a deviation
    # would treat body temperature as a signal. Renamed rather than re-noted,
    # because the NAME was the lie and the tool-name merge breaks callers anyway.
    SeriesSpec("wrist_temp", "wrist_temp_records", "records", "wrist_temperature_celsius", "°C",
               "absolute skin temperature measured during sleep (~35-37 °C), NOT a deviation "
               "from baseline; derive a deviation yourself against the person's own recent mean"),
    SeriesSpec("vo2max", "vo2max_records", "records", "vo2_max_ml_kg_min", "mL/kg/min",
               "the Watch estimates this occasionally, so days with data can be weeks apart"),
    SeriesSpec("weight_kg", "weight_trend_summary", "days", "weight_kg", "kg"),
    # Body composition rides on the same body blob as weight, but only a smart
    # scale writes it — most days carry weight alone, so these series are
    # sparser than weight_kg by nature, not by loss. They were reachable only
    # through the per-kind weight tool until 0.9.0 folded that tool into
    # get_metric; dropping them there would have taken away the one reason
    # someone buys a body-fat scale.
    SeriesSpec("body_fat_percent", "weight_trend_summary", "days", "body_fat_percent", "%",
               "0-100; only days a smart scale measured it"),
    SeriesSpec("bmi", "weight_trend_summary", "days", "bmi", "kg/m²",
               "only days a smart scale or Apple Health recorded it"),
    SeriesSpec("lean_body_mass_kg", "weight_trend_summary", "days", "lean_body_mass_kg", "kg",
               "only days a smart scale measured it"),
    SeriesSpec("water_liters", "water_intake_summary", "days", "intake_liters", "L", cumulative=True),
    SeriesSpec("water_refills", "water_intake_summary", "days", "refill_count", "refills",
               "water_liters = refills x the day's container size", cumulative=True),
    SeriesSpec("water_container_liters", "water_intake_summary", "days", "container_volume_liters", "L",
               "the container size set for that day, not an amount drunk"),
    SeriesSpec("steps", "activity_summary", "days", "step_count", "steps", cumulative=True),
    SeriesSpec("active_energy", "activity_summary", "days", "active_energy_kcal", "kcal", cumulative=True),
    SeriesSpec("exercise_minutes", "activity_summary", "days", "exercise_minutes", "minutes", cumulative=True),
    SeriesSpec("stand_minutes", "activity_summary", "days", "stand_minutes", "minutes", cumulative=True),
    SeriesSpec("distance_meters", "activity_summary", "days", "distance_meters", "m", cumulative=True),
    SeriesSpec("basal_energy", "basal_energy_records", "daily", "basal_kcal", "kcal",
               "days the Watch covered fewer than the threshold of hourly buckets are excluded "
               "and listed, never averaged in",
               exclude_when=("incomplete",), cumulative=True),
    SeriesSpec("total_energy", "total_energy_records", "days", "total_kcal", "kcal",
               "basal + active per day (TDEE); days with no or short basal coverage and today "
               "are excluded and listed",
               exclude_when=("partial", "basal_missing", "basal_incomplete"), cumulative=True),
    SeriesSpec("mindfulness_minutes", "mindfulness_summary", "days", "total_minutes", "minutes", cumulative=True),
    SeriesSpec("mindfulness_sessions", "mindfulness_summary", "days", "session_count", "sessions", cumulative=True),
)

_BY_NAME = {spec.name: spec for spec in SERIES}


def series_catalog() -> list[dict[str, Any]]:
    """The self-describing list an agent should read before guessing a name."""

    return [
        {"series": s.name, "unit": s.unit, "cumulative": s.cumulative,
         **({"note": s.direction_note} if s.direction_note else {})}
        for s in SERIES
    ]


def lookup(name: str) -> SeriesSpec | None:
    return _BY_NAME.get(name)


def _day_of(row: Any) -> str | None:
    """The row's local calendar day.

    Deliberately a narrow copy of `service._coverage_day_of`'s key list rather
    than an import: this module is pure arithmetic with no service import, and
    the reverse dependency (service imports analysis) is the one that keeps the
    layering acyclic. The keys are a payload fact, not a policy — if a fourth
    ever appears, both places want it.
    """

    if not isinstance(row, dict):
        return None
    for key in ("local_date", "day", "date"):
        value = row.get(key)
        if isinstance(value, str) and len(value) >= 10 and value[4] == "-" and value[7] == "-":
            return value[:10]
    return None


def daily_series(summary: dict[str, Any], spec: SeriesSpec) -> tuple[dict[str, float], int]:
    """Collapse one read response into `{day: value}` plus the row count consumed.

    Several rows on one day are AVERAGED, and the caller is expected to surface
    the row count so a reader can tell 30 samples over 4 days from 30 over 30.
    Averaging (rather than last-wins) is the choice that cannot silently drop a
    measurement — but it is also why the count has to travel with the result.
    """

    rows = summary.get(spec.array)
    if not isinstance(rows, list):
        return {}, 0

    buckets: dict[str, list[float]] = {}
    consumed = 0
    for row in rows:
        day = _day_of(row)
        if day is None or not isinstance(row, dict):
            continue
        if any(row.get(flag) for flag in spec.exclude_when):
            continue
        value = row.get(spec.field)
        # bool is an int subclass; a True here would silently become 1.0.
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if math.isnan(value) or math.isinf(value):
            continue
        buckets.setdefault(day, []).append(float(value))
        consumed += 1

    return {day: sum(vs) / len(vs) for day, vs in buckets.items()}, consumed


#: What an excluded night carries besides its day and reason. `other_sleep_minutes`
#: only when there is some: a split night's second half is sleep the day did have.
_NIGHT_EXCLUSION_FACTS = ("bedtime", "wake_time", "asleep_minutes", "other_sleep_minutes")


def excluded_days(summary: dict[str, Any], spec: SeriesSpec) -> list[dict[str, Any]]:
    """The days `daily_series` dropped on purpose, each with the flag that dropped it.

    Named rather than counted: the caller is an LLM that cannot see the days it
    did not receive, and "averaged over 11 days" is indistinguishable from a
    wrong answer unless it can see which days went where and why.
    """

    if not spec.exclude_when:
        return []
    rows = summary.get(spec.array)
    if not isinstance(rows, list):
        return []
    out: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        day = _day_of(row)
        reason = next((flag for flag in spec.exclude_when if row.get(flag)), None)
        if day is not None and reason is not None:
            entry: dict[str, Any] = {"day": day, "reason": reason}
            # A night left out is still a sleep somebody had: name it, so an
            # agent can say "plus a 15:09-18:36 nap" without a second read
            # (owner, 2026-10-03: naps stay out of nightly sleep, but people and
            # the AI must still see them). Facts of the ROW, so they are the
            # same on every sleep series and survive being hoisted.
            if spec.array == "nights":
                for key in _NIGHT_EXCLUSION_FACTS:
                    if row.get(key):
                        entry[key] = row[key]
            out.append(entry)
    return out



def _linear_fit(xs: list[float], ys: list[float]) -> tuple[float, float] | None:
    """Ordinary least squares, or None when x has no spread."""

    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    denom = sum((x - mean_x) ** 2 for x in xs)
    if denom == 0:
        return None
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denom
    return slope, mean_y - slope * mean_x


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def trend(points: dict[str, float], spec: SeriesSpec) -> dict[str, Any]:
    """Least-squares slope over calendar position, plus the endpoints it fits.

    The x axis is DAYS SINCE THE FIRST DAY, not the row index — otherwise a
    series with a two-month gap in the middle is fitted as though its points
    were evenly spaced, and the slope silently becomes per-observation instead
    of per-day. `slope_per_day` is only comparable across series because of that
    choice, so it must not be "simplified" back to an index.
    """

    days = sorted(points)
    values = [points[d] for d in days]
    # NO `n_days` / `first_day` / `span_days` here: the caller attaches the
    # project's standard `coverage` block, which already carries all four under
    # the names `STYLE` tells every agent to quote. A second vocabulary for the
    # same facts is a mirror — it would rot into disagreeing with the block
    # printed beside it, and an agent following STYLE would read the wrong one.
    result: dict[str, Any] = {
        "series": spec.name,
        "unit": spec.unit,
        "first_value": values[0] if values else None,
        "last_value": values[-1] if values else None,
        "mean": sum(values) / len(values) if values else None,
        "median": _median(values) if values else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }

    if len(days) < MIN_POINTS_FOR_TREND:
        result["slope_per_day"] = None
        result["change"] = None
        result["change_pct"] = None
        result["reason"] = (
            f"{len(days)} day(s) of data; a slope needs at least "
            f"{MIN_POINTS_FOR_TREND}. This is not a finding of 'no trend' — it is "
            "too little data to fit one."
        )
        return result

    try:
        origin = date.fromisoformat(days[0])
        xs = [float((date.fromisoformat(d) - origin).days) for d in days]
    except ValueError:
        xs = [float(i) for i in range(len(days))]

    fit = _linear_fit(xs, values)
    result["slope_per_day"] = fit[0] if fit else None
    # 🔴 `change` is the FITTED line's rise across the window, not last day minus
    # first day. It used to be the latter, and one short night on the first day
    # of a window reported "+54 min, +18.7%" beside a slope of -2.8 min/day
    # (2026-09-24, real data) — a sentence with the direction reversed. Two
    # single days are the noisiest numbers in the series; the fit is what the
    # whole window says. The raw difference survives, named for what it is.
    if fit:
        start = fit[1] + fit[0] * xs[0]
        end = fit[1] + fit[0] * xs[-1]
        result["change"] = end - start
        # A percentage of a baseline near zero is arithmetic, not information
        # (a fitted start of 0.17 gives "+5400%"). Below 5% of the window's
        # own mean the baseline is too close to zero to divide by.
        mean = result["mean"] or 0.0
        result["change_pct"] = (
            (end - start) / abs(start) * 100.0 if abs(start) >= 0.05 * abs(mean) and start else None
        )
    else:
        result["change"] = None
        result["change_pct"] = None
    result["endpoint_difference"] = values[-1] - values[0]
    result["change_note"] = (
        "`change` / `change_pct` are the fitted line's rise over the window. "
        "`endpoint_difference` is last day minus first day — two single days, "
        "so one unusual night at either end can give it the opposite sign."
    )
    return result


def compare_periods(
    recent: dict[str, float], previous: dict[str, float], spec: SeriesSpec
) -> dict[str, Any]:
    """Two windows side by side. Differences only — no verdict about which is better."""

    def block(points: dict[str, float]) -> dict[str, Any]:
        days = sorted(points)
        values = [points[d] for d in days]
        return {
            "n_days": len(days),
            "first_day": days[0] if days else None,
            "last_day": days[-1] if days else None,
            "mean": sum(values) / len(values) if values else None,
            "median": _median(values) if values else None,
            "min": min(values) if values else None,
            "max": max(values) if values else None,
        }

    a, b = block(recent), block(previous)
    result: dict[str, Any] = {
        "series": spec.name,
        "unit": spec.unit,
        "recent": a,
        "previous": b,
        "mean_change": None,
        "mean_change_pct": None,
        "median_change": None,
    }

    if a["mean"] is None or b["mean"] is None:
        result["reason"] = (
            "One of the two windows has no data. An empty window is not a value of "
            "zero — call `vaultbeat_doctor` if you need to know why it is empty."
        )
        return result

    result["mean_change"] = a["mean"] - b["mean"]
    result["mean_change_pct"] = (
        (a["mean"] - b["mean"]) / abs(b["mean"]) * 100.0 if b["mean"] else None
    )
    result["median_change"] = a["median"] - b["median"]
    return result


#: Shipped inside every correlation result rather than left to the prompt layer.
#:
#: `STYLE` already says "state an association as an association" — but STYLE
#: rides on prompts and on the handshake, and a result object gets quoted long
#: after either has scrolled out of an agent's window. This is the one output
#: where that distinction does measurable damage, so it carries its own warning.
CORRELATION_CAVEAT = (
    "Pearson r over paired days of two observational series. It measures linear "
    "co-movement only: it cannot show that either metric caused the other, it "
    "does not detect non-linear relationships, and anything that moved both "
    "(illness, travel, a changed routine, a new device) produces the same number "
    "as a direct link would. Quote r together with n_pairs — a large r over few "
    "days is ordinary chance. Do not translate r into a word."
)


def correlate(
    a_points: dict[str, float], b_points: dict[str, float], a: SeriesSpec, b: SeriesSpec
) -> dict[str, Any]:
    """Pearson r over days present in BOTH series.

    Days missing on either side are dropped, never filled — see this module's
    invariant 1. `n_pairs` is therefore usually smaller than either input, and
    the response says so explicitly rather than leaving an agent to infer that
    a 30-day request produced an 11-day answer.
    """

    shared = sorted(set(a_points) & set(b_points))
    xs = [a_points[d] for d in shared]
    ys = [b_points[d] for d in shared]

    # `n_pairs` stays even though the attached `coverage` block counts the same
    # days: coverage answers "how much data is this" for every tool in the
    # product, while n_pairs answers the question unique to a correlation —
    # how many days had BOTH. `n_days_a`/`n_days_b` are what make the gap
    # legible (30 and 30 producing 11 pairs is the interesting case, and neither
    # the coverage block nor r can show it).
    result: dict[str, Any] = {
        "series_a": a.name,
        "series_b": b.name,
        "unit_a": a.unit,
        "unit_b": b.unit,
        "n_pairs": len(shared),
        "n_days_a": len(a_points),
        "n_days_b": len(b_points),
        "pearson_r": None,
        "caveat": CORRELATION_CAVEAT,
    }

    if len(shared) < MIN_PAIRS_FOR_CORRELATION:
        result["reason"] = (
            f"Only {len(shared)} day(s) have BOTH metrics recorded "
            f"(a: {len(a_points)}, b: {len(b_points)}); a coefficient needs at least "
            f"{MIN_PAIRS_FOR_CORRELATION}. Two points are perfectly collinear, so a "
            "number here would be an artefact rather than a measurement. This is not "
            "evidence that the two are unrelated."
        )
        return result

    mean_x, mean_y = sum(xs) / len(xs), sum(ys) / len(ys)
    dx = [x - mean_x for x in xs]
    dy = [y - mean_y for y in ys]
    denom = math.sqrt(sum(v * v for v in dx)) * math.sqrt(sum(v * v for v in dy))
    if denom == 0:
        result["reason"] = (
            "One of the two series does not vary at all over the shared days, so "
            "correlation is undefined (not zero)."
        )
        return result

    result["pearson_r"] = sum(x * y for x, y in zip(dx, dy)) / denom
    return result
