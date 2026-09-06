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


#: Every quantity that is genuinely one-number-per-day.
#:
#: Kinds with an irreducibly richer shape (sleep stages, strength sets, food
#: entries, notes, symptoms, cycle, workouts) are deliberately ABSENT: flattening
#: a workout into "duration" would answer a question nobody asked while hiding
#: the ones they did. Their tools stay the way to read them.
SERIES: tuple[SeriesSpec, ...] = (
    SeriesSpec("sleep_minutes", "sleep_records", "daily_summary", "total_sleep_minutes", "minutes",
               "higher means more time asleep that night"),
    SeriesSpec("in_bed_minutes", "sleep_records", "daily_summary", "in_bed_minutes", "minutes"),
    SeriesSpec("resting_hr", "resting_hr_records", "records", "bpm", "bpm"),
    SeriesSpec("hrv_sdnn", "hrv_records", "records", "sdnn_ms", "ms"),
    SeriesSpec("wrist_temp_delta", "wrist_temp_records", "records", "temperature_delta_celsius", "°C",
               "a delta from the wearer's own baseline, so it is signed"),
    SeriesSpec("vo2max", "vo2max_records", "records", "vo2_max_ml_kg_min", "mL/kg/min"),
    SeriesSpec("weight_kg", "weight_trend_summary", "days", "weight_kg", "kg"),
    SeriesSpec("water_liters", "water_intake_summary", "days", "intake_liters", "L"),
    SeriesSpec("steps", "activity_summary", "days", "step_count", "steps"),
    SeriesSpec("active_energy", "activity_summary", "days", "active_energy_kcal", "kcal"),
    SeriesSpec("exercise_minutes", "activity_summary", "days", "exercise_minutes", "minutes"),
    SeriesSpec("stand_minutes", "activity_summary", "days", "stand_minutes", "minutes"),
    SeriesSpec("distance_meters", "activity_summary", "days", "distance_meters", "m"),
    SeriesSpec("basal_energy", "basal_energy_records", "daily", "basal_kcal", "kcal"),
    SeriesSpec("mindfulness_minutes", "mindfulness_summary", "days", "total_minutes", "minutes"),
)

_BY_NAME = {spec.name: spec for spec in SERIES}


def series_catalog() -> list[dict[str, str]]:
    """The self-describing list an agent should read before guessing a name."""

    return [
        {"series": s.name, "unit": s.unit, "read_with": s.method.replace("_records", "").replace("_summary", ""),
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
        value = row.get(spec.field)
        # bool is an int subclass; a True here would silently become 1.0.
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if math.isnan(value) or math.isinf(value):
            continue
        buckets.setdefault(day, []).append(float(value))
        consumed += 1

    return {day: sum(vs) / len(vs) for day, vs in buckets.items()}, consumed


def _span_days(days: list[str]) -> int | None:
    if not days:
        return None
    try:
        return (date.fromisoformat(days[-1]) - date.fromisoformat(days[0])).days + 1
    except ValueError:
        return None


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
    result["change"] = values[-1] - values[0]
    result["change_pct"] = (
        ((values[-1] - values[0]) / abs(values[0]) * 100.0) if values[0] else None
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
