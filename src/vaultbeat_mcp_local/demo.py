"""Deterministic synthetic dataset behind demo mode — every tool, zero iPhone.

WHY THIS EXISTS
---------------
Vaultbeat is E2EE: the cloud holds ciphertext and only a bound private key can
open it. That is the product's whole claim, and it means a reviewer without an
iPhone and a paired account structurally CANNOT see a single non-empty tool
result. The Anthropic Plugin Directory asks for a "fully populated test
account"; we cannot hand one over without handing over a key. Demo mode is the
answer: one command, no binding, every read tool returning its real shape.

It doubles as acquisition. An agent that can run `get_sleep_detail` and see the
actual response shape before anyone installs anything is a much shorter sales
pitch than a README table.

WHERE IT PLUGS IN
-----------------
`VaultbeatLocalService.sync_decrypted_records` — AFTER the `KNOWN_METRIC_TYPES`
membership check, BEFORE `self.store.require_bound()`::

    if demo_enabled():
        return demo_sync_result(metric_type=metric_type, limit=limit)

That single point is the real choke point: every data-returning tool reaches it
through `_records_for_metric`, and `vaultbeat_doctor`'s capability report calls
`sync_decrypted_records` directly, so one branch covers every kind without a
second implementation anywhere. (Deliberately not writing a tool COUNT here —
`grep -c '_records_for_metric(' service.py` counts call sites, not tools, and
the two numbers differ because a couple of methods read twice.)

Three properties of that location are load-bearing, not incidental:

1. It is ABOVE `cache.load` / `cache.save`, so demo mode is structurally
   incapable of writing synthetic records into the on-disk plaintext cache.
   A demo run cannot contaminate a real run, ever — not by a bug, by
   construction.
2. It is ABOVE `require_bound()`, so nothing here needs a config file, a
   server token, or a private key. `status` / `doctor` keep telling the truth
   ("not bound"), which is exactly what they should say.
3. It injects PLAINTEXT `DecryptedRecord` objects, deliberately skipping
   encryption. Sealing fake blobs would need a fake config → a fake binding →
   a fake `server_id` in the cache, i.e. we would have to make the server LIE
   about being bound in order to demo it. Not worth it, and the lie would
   outlive the demo.

WHAT IT COVERS, AND WHAT IT DOES NOT
------------------------------------
Covered: every read tool, across every kind in `KNOWN_METRIC_TYPES` (17 at the
time of writing — `missing_kinds()` is what keeps that true, not this
sentence), plus `vaultbeat_doctor`'s capability report, which derives from data
and therefore reports the demo dataset's kinds rather than an empty account.

NOT covered, on purpose: `log_strength_entry` / `log_food_entry` /
`log_weight_entry` / `log_note` and the two binding tools. Those seal a blob
with a real key and POST it to a real edge function; there is no honest way to
fake them, and a write tool that pretends to succeed is worse than one that
says it needs an account.

THE THREE RULES THIS FILE OBEYS
-------------------------------
1. **The kind list is DERIVED, never retyped.** `KNOWN_METRIC_TYPES` is
   imported from `service`, and `scripts/ci/check_metric_type_contract.py`
   cross-checks that frozenset against ten registration sites — but the guard
   cannot see this file. A hand-typed dict of kinds here would be an
   eleventh copy AND the one nothing watches (Invariant 18). So `_BUILDERS`
   is checked against the imported set: a new kind with no builder shows up in
   `missing_kinds()` and returns an empty list for that kind alone, loudly,
   instead of silently shipping a demo that is quietly missing a feature.

2. **Determinism.** A fixed seed per kind and a FIXED anchor day — no
   `datetime.now()`, no `random` module-level state, no `hash()` (salted per
   process). Two runs on any machine emit byte-identical payloads, which is
   what makes demo output assertable in tests and comparable in bug reports.
   Per-kind seeding also means `demo_records("sleep")` is exactly the sleep
   slice of `demo_records(None)`; a shared generator would make a kind's data
   depend on which other kinds were built first.

3. **The data is BENIGN.** No migraine episodes, no crashed HRV, no night
   shifts — the storyline devices a synthetic-health-data generator reaches for
   by default. This product asks people to hand their health record to an AI;
   a fake "your HRV collapsed" in the first thing they ever see does real harm
   to a real person reading it. There is enough structure to exercise every
   aggregation (a gentle weight trend so the OLS slope is non-zero, a regular
   cycle so prediction and ovulation calibration have something to find,
   weekday/weekend activity variation, progressive overload in the gym) and
   nothing that reads as pathology.

TIMESTAMP CONVENTIONS
---------------------
Day-bucket fields (`dayStartDate`, `targetDate`, strength/food `date`) are
anchored at **12:00 UTC**, not midnight. Every reader converts them to the
local calendar day (`_local_date_fields`, `_local_calendar_day`), and noon UTC
is the only anchor whose local date is the intended date on every machine from
UTC-12 to UTC+11 — a midnight anchor shifts the whole dataset a day for every
reader west of Greenwich. Real blobs carry local-midnight-in-UTC; the demo
trades that fidelity for the same label everywhere, which matters more when
the reader is a stranger checking whether the tool works.

Intra-day instants (sleep bedtime/wake, hourly buckets, workout start/end) are
real UTC instants and render in the reader's own timezone, exactly like real
data does. Durations, counts and kcal are timezone-independent, so every
aggregate number is identical everywhere.

WATERMARKING
------------
Every id carries a `demo-` prefix and both owner ids start with `demo000`.
A screenshot, a pasted tool result, or a bug report built on demo data
identifies itself in the payload, so it can never be mistaken for a real
export. Do not "tidy" those prefixes into something realistic.

⚠️ IMPORT DIRECTION: this module imports from `service`; `service` must import
this one *inside* the function (a module-level import there would be a cycle).
"""

from __future__ import annotations

import logging
import os
import random
import zlib
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable

from vaultbeat_mcp_local.service import DecryptedRecord, KNOWN_METRIC_TYPES

_LOG = logging.getLogger("vaultbeat_mcp_local.demo")


# ── Public knobs ─────────────────────────────────────────────────────────────

#: Env var that turns demo mode on. Named alongside `VAULTBEAT_MCP_CONFIG` /
#: `VAULTBEAT_PRIVATE_KEY` in `store.py`. Add it to `_scope_report`'s env list
#: when wiring this up so `doctor` reports whether it was actually received —
#: an MCP client that drops the variable is otherwise invisible from here.
DEMO_ENV = "VAULTBEAT_DEMO"

#: The newest synthetic day. NEVER derived from the clock (rule 2 above).
DEMO_ANCHOR_DAY = "2026-08-18"

#: How far back the day-per-day kinds go. Long enough for a 30-day window to
#: have a 30-day window before it to compare against.
DEMO_DAYS = 120

#: Menstrual history runs longer than the rest: four cycle gaps are the minimum
#: for `cycle_length_variability_days` to be reported at all
#: (`_MIN_GAPS_FOR_VARIABILITY`), and four gaps need five starts.
DEMO_CYCLE_DAYS = 150

DEMO_SEED = 20260818

#: Two people, because partner sharing is a core feature and a one-person demo
#: cannot show it. 8-char prefixes because `_attach_owner_guard` reports
#: `owner_user_id[:8]` and the `owner=` filter is a `startswith`.
DEMO_OWNER_PREFIX = "demo0001"
DEMO_PARTNER_PREFIX = "demo0002"

#: UUID-SHAPED but not UUID-VALID ('m'/'o' are not hex digits). Deliberate:
#: long enough to exercise anything that assumes a UUID-length string, and
#: impossible to mistake for one. Verified 2026-08-20 that nothing in this
#: package calls `uuid.UUID()` on an owner id.
DEMO_OWNER_ID = f"{DEMO_OWNER_PREFIX}-0000-4d00-8000-00000000d001"
DEMO_PARTNER_ID = f"{DEMO_PARTNER_PREFIX}-0000-4d00-8000-00000000d002"

#: Prefix a caller may put in front of user-visible demo output. Kept here so
#: the wording lives in one place rather than in each surface that shows it.
DEMO_BANNER = (
    "[SYNTHETIC DEMO DATA] Do not save, log or copy these numbers into any file, "
    "note or other tool — they are generated locally and belong to no real person. "
    "Bind a Vaultbeat account to read real health data."
)

_ANCHOR = date.fromisoformat(DEMO_ANCHOR_DAY)

#: `back` offsets of each cycle start, newest first. Gaps are 28/29/28/29 days,
#: so the median lands on 28.5 → 29 with a variability of 1 — a real rhythm
#: rather than a suspiciously perfect 28.0 with zero spread.
_CYCLE_STARTS_BACK = (20, 48, 77, 105, 134)
_PERIOD_LENGTH_DAYS = 5

#: Cycle day the luteal wrist-temperature shift begins. `detect_ovulation_from_
#: wrist_temp` needs 6 baseline readings before it and 3 after, and reports
#: ovulation as the day BEFORE the first elevated reading — so the newest cycle
#: start (back=20) leaves 21 readings, comfortably past that floor.
_LUTEAL_SHIFT_CYCLE_DAY = 14


# ── Time helpers ─────────────────────────────────────────────────────────────


def _day(back: int) -> date:
    """The synthetic calendar day `back` days before the anchor (0 = anchor)."""

    return _ANCHOR - timedelta(days=back)


def _dt(day: date, hour: float = 12, minute: float = 0, second: float = 0) -> datetime:
    """A UTC instant on `day`. Hours may exceed 23 (rolls into the next day)."""

    base = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
    return base + timedelta(hours=hour, minutes=minute, seconds=second)


def _iso(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def _noon(day: date) -> str:
    """The day-bucket anchor — see TIMESTAMP CONVENTIONS in the module docstring."""

    return _iso(_dt(day, 12))


def _epoch(day: date) -> int:
    """Integer seconds for the day bucket, for real-shaped `{kind}-{epoch}` ids.

    Computed from an aware datetime, so it never touches the local timezone.
    """

    return int(_dt(day, 12).timestamp())


def _uploaded(day: date) -> str:
    """Synthetic `created_at`.

    An hour after the day bucket, so sorting by upload time agrees with
    business time. Real backfills do NOT have that property (Invariant 38 exists
    because of it) — but a demo whose newest-first ordering looks scrambled
    reads as a bug, and every reader here already cuts on business time.
    """

    return _iso(_dt(day, 13, 5))


def _rng(kind: str) -> random.Random:
    """A generator seeded from the kind name — stable across processes.

    `hash()` is salted per interpreter and would break determinism between
    runs; `zlib.crc32` over the kind bytes is fixed forever.
    """

    return random.Random(DEMO_SEED ^ zlib.crc32(kind.encode("utf-8")))


def _record(
    kind: str,
    suffix: str,
    *,
    owner: str,
    payload: Any,
    created_at: str,
) -> DecryptedRecord:
    """One synthetic record in the exact shape the decrypt path produces."""

    return DecryptedRecord(
        envelope_id=f"demo-env-{kind}-{suffix}",
        blob_id=f"demo-{kind}-{suffix}",
        metric_type=kind,
        created_at=created_at,
        payload=payload,
        owner_user_id=owner,
    )


def _cycle_day(back: int) -> int:
    """Days since the most recent cycle start on or before `_day(back)`.

    "Most recent start on or before" = the smallest `back` offset in
    `_CYCLE_STARTS_BACK` that is still >= this one (a larger offset is an
    older day). The oldest start covers the whole window, so this is total.
    """

    for start_back in sorted(_CYCLE_STARTS_BACK):
        if start_back >= back:
            return start_back - back
    return 0


def _week_wave(back: int) -> float:
    """A weekly rhythm in [-1, 1], phase-locked to the calendar (not to `back`)."""

    return {0: 0.4, 1: 0.2, 2: 0.0, 3: 0.2, 4: -0.3, 5: -1.0, 6: -0.6}[_day(back).weekday()]


# ── Per-kind builders ────────────────────────────────────────────────────────
#
# Each takes its own seeded generator and returns records newest-day-first is
# NOT required (the readers sort), but every builder walks `back` ascending so
# the output is stable and readable.


def _build_sleep(rng: random.Random) -> list[DecryptedRecord]:
    """`{session: {sessionDate, bedtime, wakeTime, provenance, samples}, heartRateSamples, respiratoryRateSamples}`.

    Stages are laid out as real cycles (core → deep → REM, deep front-loaded)
    inside an `inBed` sample that spans the whole night, so `stage_minutes`,
    `has_stage_detail`, `in_bed_minutes` and the per-stage vitals in
    `get_sleep_detail` all have honest inputs.

    One partner night is deliberately in-bed-only (Watch left on the charger).
    That is the Invariant 39 case — the reader must render it as "no sleep
    data", never "0h00m" — and it is worth showing precisely because it looks
    like breakage until you know what it means.
    """

    out: list[DecryptedRecord] = []
    people = (
        (DEMO_OWNER_ID, "o", 455),
        (DEMO_PARTNER_ID, "p", 430),
    )
    for owner, tag, base_minutes in people:
        for back in range(DEMO_DAYS):
            wake_day = _day(back)
            bed_day = wake_day - timedelta(days=1)
            suffix = f"{wake_day.isoformat()}-{tag}"

            bedtime = _dt(bed_day, 22, rng.randint(5, 55))
            watch_off = owner == DEMO_PARTNER_ID and back == 6

            if watch_off:
                wake = bedtime + timedelta(minutes=rng.randint(440, 480))
                samples = [
                    {
                        "stage": "inBed",
                        "startDate": _iso(bedtime),
                        "endDate": _iso(wake),
                    }
                ]
                hr_samples: list[dict[str, Any]] = []
                rr_samples: list[dict[str, Any]] = []
            else:
                target = int(
                    base_minutes + _week_wave(back) * 22 + rng.uniform(-28, 28)
                )
                latency = rng.randint(8, 19)
                cycles = 4
                gaps = [rng.randint(2, 7) for _ in range(cycles - 1)]

                samples = []
                cursor = bedtime
                samples.append(
                    {
                        "stage": "awake",
                        "startDate": _iso(cursor),
                        "endDate": _iso(cursor + timedelta(minutes=latency)),
                    }
                )
                cursor += timedelta(minutes=latency)

                per_cycle = target / cycles
                for index in range(cycles):
                    # Deep sleep front-loads, REM back-loads — the shape a Watch
                    # actually records, and what makes `stage_vitals` interesting.
                    deep_share = (0.30, 0.24, 0.13, 0.07)[index]
                    core = int(per_cycle * rng.uniform(0.48, 0.55))
                    deep = int(per_cycle * deep_share * rng.uniform(0.85, 1.15))
                    rem = max(int(per_cycle) - core - deep, 6)
                    for stage, minutes in (
                        ("asleepCore", core),
                        ("asleepDeep", deep),
                        ("asleepREM", rem),
                    ):
                        samples.append(
                            {
                                "stage": stage,
                                "startDate": _iso(cursor),
                                "endDate": _iso(cursor + timedelta(minutes=minutes)),
                            }
                        )
                        cursor += timedelta(minutes=minutes)
                    if index < cycles - 1:
                        samples.append(
                            {
                                "stage": "awake",
                                "startDate": _iso(cursor),
                                "endDate": _iso(cursor + timedelta(minutes=gaps[index])),
                            }
                        )
                        cursor += timedelta(minutes=gaps[index])

                wake = cursor
                samples.insert(
                    0,
                    {
                        "stage": "inBed",
                        "startDate": _iso(bedtime),
                        "endDate": _iso(wake),
                    },
                )

                hr_samples = []
                point = bedtime + timedelta(minutes=25)
                while point < wake:
                    hr_samples.append(
                        {"startDate": _iso(point), "value": round(rng.uniform(50.0, 61.0), 1)}
                    )
                    point += timedelta(minutes=rng.randint(70, 100))
                rr_samples = []
                point = bedtime + timedelta(minutes=55)
                while point < wake:
                    rr_samples.append(
                        {"startDate": _iso(point), "value": round(rng.uniform(13.2, 16.1), 1)}
                    )
                    point += timedelta(minutes=rng.randint(130, 190))

            payload = {
                "session": {
                    # Noon UTC, so `local_date` is this calendar day everywhere.
                    "sessionDate": _noon(wake_day),
                    "bedtime": _iso(bedtime),
                    "wakeTime": _iso(wake),
                    "provenance": "healthkitSleep",
                    "samples": samples,
                },
                "heartRateSamples": hr_samples,
                "respiratoryRateSamples": rr_samples,
            }
            out.append(
                _record(
                    "sleep",
                    suffix,
                    owner=owner,
                    payload=payload,
                    created_at=_uploaded(wake_day),
                )
            )
    return out


def _build_water(rng: random.Random) -> list[DecryptedRecord]:
    """`{dayID, dayStartDate, containerVolumeLiters, refillEvents: [{timestamp, containerVolumeLiters}]}`.

    Intake = refill count × that day's container volume, so the two people
    carry different container sizes and the aggregate differs per person.

    ⚠️ `dayID` is `water-{epoch}` — IDENTICAL for both partners on the same
    day, exactly as iOS mints it. `summarize_water_intake` dedups by `dayID`
    with no owner in the key, so an unfiltered query collapses the two people
    into one series and fires the `mixed_owners` warning telling you to
    re-query with `owner=`. That is real behaviour and the demo should show it,
    not paper over it with a fake per-owner id.
    """

    out: list[DecryptedRecord] = []
    people = ((DEMO_OWNER_ID, "o", 0.75, (6, 10)), (DEMO_PARTNER_ID, "p", 0.5, (5, 9)))
    for owner, tag, volume, (low, high) in people:
        for back in range(DEMO_DAYS):
            day = _day(back)
            refills = rng.randint(low, high)
            events = []
            for index in range(refills):
                hour = 8 + index * (13 / max(refills, 1))
                events.append(
                    {
                        "timestamp": _iso(_dt(day, hour, rng.randint(0, 59))),
                        "containerVolumeLiters": volume,
                    }
                )
            payload = {
                "dayID": f"water-{_epoch(day)}",
                "dayStartDate": _noon(day),
                "containerVolumeLiters": volume,
                "refillEvents": events,
            }
            out.append(
                _record(
                    "water",
                    f"{day.isoformat()}-{tag}",
                    owner=owner,
                    payload=payload,
                    created_at=_uploaded(day),
                )
            )
    return out


def _build_body(rng: random.Random) -> list[DecryptedRecord]:
    """`{dayID, dayStartDate, weightKg, bodyFatPercent?, bmi?, leanBodyMassKg?}`.

    The owner trends down ~0.15 kg/week — enough for the OLS slope in
    `summarize_weight_trend` to be clearly non-zero, gentle enough to read as
    healthy, and (this is the part that took a second pass) the exact rate a
    ~170 kcal/day deficit predicts at the 7,700 kcal/kg rule of thumb. The food
    log, the basal and activity kinds and this series are four independent
    tools describing one body; when they disagree the reader has no way to tell
    which one is lying, and "the demo's own numbers contradict each other" is a
    worse first impression than any single wrong value.

    The partner is flat. Roughly one day in seven has no weigh-in, because
    nobody steps on a scale 120 mornings in a row.

    Only the owner carries the three composition fields (smart scale); the
    partner omits them entirely, which is the normal "bathroom scale" shape and
    exercises the nullable path added 2026-08-05.

    Same shared-`dayID` note as water: `body-{epoch}` collides across partners
    by design.
    """

    out: list[DecryptedRecord] = []
    for back in range(DEMO_DAYS):
        day = _day(back)
        # Owner: 70.6 kg at the anchor, 2.6 kg heavier 120 days ago.
        # 0.0221 kg/day == 0.155 kg/week — pinned to the calorie deficit, see
        # the docstring. Changing it means changing the food catalog too.
        if rng.random() > 0.14:
            weight = round(70.6 + back * 0.0221 + rng.uniform(-0.35, 0.35), 1)
            body_fat = round(16.4 + back * 0.013 + rng.uniform(-0.2, 0.2), 1)
            payload: dict[str, Any] = {
                "dayID": f"body-{_epoch(day)}",
                "dayStartDate": _noon(day),
                "weightKg": weight,
                "bodyFatPercent": body_fat,
                "bmi": round(weight / (1.78 * 1.78), 1),
                "leanBodyMassKg": round(weight * (1 - body_fat / 100), 1),
            }
            out.append(
                _record(
                    "body",
                    f"{day.isoformat()}-o",
                    owner=DEMO_OWNER_ID,
                    payload=payload,
                    created_at=_uploaded(day),
                )
            )
        if rng.random() > 0.35:
            out.append(
                _record(
                    "body",
                    f"{day.isoformat()}-p",
                    owner=DEMO_PARTNER_ID,
                    payload={
                        "dayID": f"body-{_epoch(day)}",
                        "dayStartDate": _noon(day),
                        "weightKg": round(54.8 + rng.uniform(-0.3, 0.3), 1),
                    },
                    created_at=_uploaded(day),
                )
            )
    return out


def _build_menstrual(rng: random.Random) -> list[DecryptedRecord]:
    """`{dayID, dayStartDate, samples: [{startDate, endDate, flow}]}`.

    Owner only. Two reasons, and the second one is the interesting one:

    * iOS writes a menstrual blob only for days that HAVE flow samples, so the
      dataset is ~5 days per cycle rather than one per day.
    * `_wrist_readings_for_owner` only calibrates ovulation when there is
      EXACTLY ONE menstrual owner and that owner also owns the wrist-temp
      blobs. Wrist temp is `.ownDevicesOnly`, so putting the cycle on the
      partner would make that path unreachable — and it is the most
      sophisticated inference the product does. The app is gender-neutral by
      design (whoever tracks, benefits), so the same person logging squats and
      a cycle is the intended shape, not an oddity.

    Flow stays light/medium. `heavy` is a perfectly ordinary HealthKit value,
    but nothing in a first-contact dataset needs to be at the top of a scale.
    """

    out: list[DecryptedRecord] = []
    flows = ("medium", "medium", "light", "light", "unspecified")
    for start_back in _CYCLE_STARTS_BACK:
        for offset in range(_PERIOD_LENGTH_DAYS):
            back = start_back - offset
            if back < 0 or back >= DEMO_CYCLE_DAYS:
                continue
            day = _day(back)
            payload = {
                "dayID": f"menstrual-{_epoch(day)}",
                "dayStartDate": _noon(day),
                "samples": [
                    {
                        "startDate": _iso(_dt(day, 8, rng.randint(0, 59))),
                        "endDate": _iso(_dt(day, 20, rng.randint(0, 59))),
                        "flow": flows[offset],
                    }
                ],
            }
            out.append(
                _record(
                    "menstrual",
                    day.isoformat(),
                    owner=DEMO_OWNER_ID,
                    payload=payload,
                    created_at=_uploaded(day),
                )
            )
    return out


def _build_activity(rng: random.Random) -> list[DecryptedRecord]:
    """`{dayID, dayStartDate, stepCount, activeEnergyKcal, exerciseMinutes, standMinutes, distanceMeters}`.

    Owner only (`.ownDevicesOnly` — no partner envelope is ever sealed).
    `standMinutes` really is minutes (`appleStandTime`), not stand hours.
    `activeEnergyKcal` is the half of TDEE that `get_total_energy_burned` joins
    against basal, so the two kinds have to cover the same days.
    """

    out: list[DecryptedRecord] = []
    for back in range(DEMO_DAYS):
        day = _day(back)
        wave = _week_wave(back)
        steps = int(9600 + wave * 1900 + rng.uniform(-1700, 1700))
        payload = {
            "dayID": f"activity-{_epoch(day)}",
            "dayStartDate": _noon(day),
            "stepCount": steps,
            "activeEnergyKcal": round(430 + wave * 120 + rng.uniform(-90, 150), 1),
            "exerciseMinutes": int(max(12, 38 + wave * 14 + rng.uniform(-14, 20))),
            "standMinutes": int(240 + wave * 40 + rng.uniform(-45, 60)),
            "distanceMeters": round(steps * rng.uniform(0.68, 0.76), 1),
        }
        out.append(
            _record(
                "activity",
                day.isoformat(),
                owner=DEMO_OWNER_ID,
                payload=payload,
                created_at=_uploaded(day),
            )
        )
    return out


def _build_resting_hr(rng: random.Random) -> list[DecryptedRecord]:
    """`{dayID, dayStartDate, restingHeartRateBPM}` — owner only.

    Drifts from ~57 down to ~53 over the window: the ordinary effect of four
    months of consistent training, and the direction that reads as reassuring
    rather than alarming.
    """

    out: list[DecryptedRecord] = []
    for back in range(DEMO_DAYS):
        day = _day(back)
        payload = {
            "dayID": f"resting_hr-{_epoch(day)}",
            "dayStartDate": _noon(day),
            "restingHeartRateBPM": round(53.0 + back * 0.032 + rng.uniform(-1.4, 1.4), 1),
        }
        out.append(
            _record(
                "resting_hr",
                day.isoformat(),
                owner=DEMO_OWNER_ID,
                payload=payload,
                created_at=_uploaded(day),
            )
        )
    return out


#: Weekday → (activityType, minutes range). Mon/Wed/Fri lift, Sunday runs
#: outdoors (the only session Apple will compute a VO₂ max from), Saturday
#: walks. Weekday numbers are `date.weekday()`: Monday is 0.
_WORKOUT_PLAN: dict[int, tuple[str, int, int]] = {
    0: ("TraditionalStrengthTraining", 55, 72),
    2: ("TraditionalStrengthTraining", 55, 72),
    4: ("TraditionalStrengthTraining", 50, 68),
    5: ("Walking", 35, 55),
    6: ("Running", 28, 48),
}


def _build_workout(rng: random.Random) -> list[DecryptedRecord]:
    """`{workoutID, activityType, startDate, endDate, durationSeconds, activeKcal, distanceMeters}`.

    Owner only. Five sessions a week on a fixed weekday plan, so the schedule
    is legible instead of looking like noise — the point of a demo is that the
    reader can tell at a glance that the data means something.
    """

    out: list[DecryptedRecord] = []
    for back in range(DEMO_DAYS):
        day = _day(back)
        plan = _WORKOUT_PLAN.get(day.weekday())
        if plan is None:
            continue
        activity_type, low, high = plan
        minutes = rng.randint(low, high)
        start = _dt(day, 17 if activity_type == "TraditionalStrengthTraining" else 8, rng.randint(0, 40))
        end = start + timedelta(minutes=minutes)
        if activity_type == "Running":
            distance: float | None = round(minutes * rng.uniform(165.0, 190.0), 1)
            kcal = round(minutes * rng.uniform(10.5, 12.5), 1)
        elif activity_type == "Walking":
            distance = round(minutes * rng.uniform(78.0, 92.0), 1)
            kcal = round(minutes * rng.uniform(4.0, 5.2), 1)
        else:
            distance = None
            kcal = round(minutes * rng.uniform(5.4, 7.0), 1)
        payload: dict[str, Any] = {
            "workoutID": f"workout-{_epoch(day)}",
            "activityType": activity_type,
            "startDate": _iso(start),
            "endDate": _iso(end),
            "durationSeconds": float(minutes * 60),
            "activeKcal": kcal,
        }
        if distance is not None:
            payload["distanceMeters"] = distance
        out.append(
            _record(
                "workout",
                day.isoformat(),
                owner=DEMO_OWNER_ID,
                payload=payload,
                created_at=_uploaded(day),
            )
        )
    return out


def _build_mindfulness(rng: random.Random) -> list[DecryptedRecord]:
    """`{dayID, dayStartDate, sessionCount, totalMinutes}` — owner only, ~4 days a week."""

    out: list[DecryptedRecord] = []
    for back in range(DEMO_DAYS):
        if rng.random() > 0.58:
            continue
        day = _day(back)
        sessions = rng.randint(1, 2)
        payload = {
            "dayID": f"mindfulness-{_epoch(day)}",
            "dayStartDate": _noon(day),
            "sessionCount": sessions,
            "totalMinutes": round(sessions * rng.uniform(7.5, 14.0), 1),
        }
        out.append(
            _record(
                "mindfulness",
                day.isoformat(),
                owner=DEMO_OWNER_ID,
                payload=payload,
                created_at=_uploaded(day),
            )
        )
    return out


def _build_hrv(rng: random.Random) -> list[DecryptedRecord]:
    """`{dayID, dayStartDate, sdnnMilliseconds}` — RAW per-sample, owner only.

    Only the last 14 days, because the raw kind's rolling window is 3 days and
    everything older arrives via history backfill (Invariant 25's dual track:
    raw is for spike precision over a short window, `hrv_hourly` is the
    aggregate the MCP tool defaults to). A raw series stretching back four
    months would misrepresent how the product actually behaves.

    Note the field names: even a per-sample kind uses `dayID`/`dayStartDate`.
    """

    out: list[DecryptedRecord] = []
    for back in range(14):
        day = _day(back)
        for index in range(6):
            moment = _dt(day, 1 + index * 3.5, rng.randint(0, 59))
            payload = {
                "dayID": f"hrv-{int(moment.timestamp())}",
                "dayStartDate": _iso(moment),
                "sdnnMilliseconds": round(rng.uniform(46.0, 79.0), 1),
            }
            out.append(
                _record(
                    "hrv",
                    f"{day.isoformat()}-{index}",
                    owner=DEMO_OWNER_ID,
                    payload=payload,
                    created_at=_uploaded(day),
                )
            )
    return out


def _build_hrv_hourly(rng: random.Random) -> list[DecryptedRecord]:
    """`{hourID, hourStartDate, avgSdnnMilliseconds, sampleCount}` — owner only, 30 days.

    Nine buckets a day, not twenty-four: a Watch measures HRV mostly while you
    sleep plus the odd daytime reading, and a full 24×30 grid would be a
    fiction that happens to be bigger.

    🔴 `hourStartDate` MUST be hour-aligned UTC (Invariant 26): the nightly
    dedup RPC buckets by `unix_seconds / 3600`, so an off-the-hour anchor
    misfiles rows. `_dt(day, h)` with an integer hour guarantees it.
    """

    out: list[DecryptedRecord] = []
    night_hours = (0, 1, 2, 3, 4, 5)
    day_hours = (9, 13, 20)
    for back in range(30):
        day = _day(back)
        for hour in night_hours + day_hours:
            asleep = hour in night_hours
            payload = {
                "hourID": f"hrv_hourly-{int(_dt(day, hour).timestamp())}",
                "hourStartDate": _iso(_dt(day, hour)),
                "avgSdnnMilliseconds": round(
                    rng.uniform(58.0, 78.0) if asleep else rng.uniform(38.0, 56.0), 1
                ),
                "sampleCount": rng.randint(3, 9) if asleep else rng.randint(1, 3),
            }
            out.append(
                _record(
                    "hrv_hourly",
                    f"{day.isoformat()}-{hour:02d}",
                    owner=DEMO_OWNER_ID,
                    payload=payload,
                    created_at=_uploaded(day),
                )
            )
    return out


def _build_wrist_temp(rng: random.Random) -> list[DecryptedRecord]:
    """`{dayID, dayStartDate, temperatureDeltaCelsius}` — owner only, one reading a night.

    Shaped as a real biphasic curve: slightly below baseline in the follicular
    phase, ~0.35 °C higher from `_LUTEAL_SHIFT_CYCLE_DAY` onward. That is what
    makes `detect_ovulation_from_wrist_temp` fire, which in turn re-anchors the
    next-period prediction to `ovulation + 14` in `get_menstrual_cycle` — the
    one place two kinds combine into something neither could say alone, and
    therefore the single best thing this demo can show.

    Noise is kept at ±0.03 °C on purpose. The detector's threshold is +0.15 °C
    over a 6-reading median; wider noise could trip a false shift early in the
    follicular phase and hand the demo an ovulation date that means nothing.
    """

    out: list[DecryptedRecord] = []
    for back in range(DEMO_DAYS):
        day = _day(back)
        luteal = _cycle_day(back) >= _LUTEAL_SHIFT_CYCLE_DAY
        delta = (0.35 if luteal else -0.08) + rng.uniform(-0.03, 0.03)
        payload = {
            "dayID": f"wrist_temp-{_epoch(day)}",
            "dayStartDate": _noon(day),
            "temperatureDeltaCelsius": round(delta, 2),
        }
        out.append(
            _record(
                "wrist_temp",
                day.isoformat(),
                owner=DEMO_OWNER_ID,
                payload=payload,
                created_at=_uploaded(day),
            )
        )
    return out


def _build_vo2max(rng: random.Random) -> list[DecryptedRecord]:
    """`{sampleID, sampleStartDate, vo2MaxMlKgMin}` — owner only, Sundays only.

    Sparse by nature, not by laziness: Apple only computes VO₂ max during an
    outdoor walk/run/hike of ~20 minutes or more, so ~17 samples across four
    months is the correct density. Trends 42.4 → 45.2 with the training block.
    """

    out: list[DecryptedRecord] = []
    for back in range(DEMO_DAYS):
        day = _day(back)
        if day.weekday() != 6:
            continue
        moment = _dt(day, 9, rng.randint(0, 45))
        payload = {
            "sampleID": f"vo2max-{int(moment.timestamp())}",
            "sampleStartDate": _iso(moment),
            "vo2MaxMlKgMin": round(45.2 - back * 0.023 + rng.uniform(-0.35, 0.35), 1),
        }
        out.append(
            _record(
                "vo2max",
                day.isoformat(),
                owner=DEMO_OWNER_ID,
                payload=payload,
                created_at=_uploaded(day),
            )
        )
    return out


def _build_basal_energy(rng: random.Random) -> list[DecryptedRecord]:
    """`{sampleID, sampleStartDate, basalEnergyKcal}` — owner only, hourly, 30 days.

    Hourly buckets (Invariant 24: the raw stream is ~177 samples/day/person and
    has no structural ceiling, so iOS aggregates before upload). 24 × 30 = 720
    records summing to ~1750 kcal/day, which `get_total_energy_burned` adds to
    `activity.active_energy_kcal` for a real TDEE.

    🔴 Hour-aligned UTC, same reason as `hrv_hourly`.
    """

    out: list[DecryptedRecord] = []
    for back in range(30):
        day = _day(back)
        for hour in range(24):
            # BMR dips overnight and peaks mid-afternoon.
            asleep = hour < 6
            payload = {
                "sampleID": f"basal_energy-{int(_dt(day, hour).timestamp())}",
                "sampleStartDate": _iso(_dt(day, hour)),
                "basalEnergyKcal": round(
                    (rng.uniform(64.0, 70.0) if asleep else rng.uniform(72.0, 80.0)), 1
                ),
            }
            out.append(
                _record(
                    "basal_energy",
                    f"{day.isoformat()}-{hour:02d}",
                    owner=DEMO_OWNER_ID,
                    payload=payload,
                    created_at=_uploaded(day),
                )
            )
    return out


def _build_symptom(rng: random.Random) -> list[DecryptedRecord]:
    """`{dayID, dayStartDate, samples: [{symptomType, severity, startDate, endDate}]}`.

    Both people, because `summarize_symptoms` groups by owner and a one-person
    dataset cannot demonstrate that. The blob id is
    `symptom-{device8}-{epoch}` — device-discriminated, as Invariant 17
    requires for a kind both partners write, so the two series do NOT collide
    the way water and body deliberately do.

    Content is ordinary cycle-adjacent stuff at `mild`, with a single
    `moderate` on the first day of a period. `severity` values come from
    `SYMPTOM_SEVERITY_VALUES`; note that `moodChanges` is a presence type
    (`present`) and `appetiteChanges` has its own scale (`increased`).
    """

    out: list[DecryptedRecord] = []
    for back in range(DEMO_DAYS):
        day = _day(back)
        cycle_day = _cycle_day(back)
        samples: list[dict[str, str]] = []
        if cycle_day == 0:
            samples.append({"symptomType": "abdominalCramps", "severity": "moderate"})
            samples.append({"symptomType": "fatigue", "severity": "mild"})
        elif cycle_day in (1, 2):
            samples.append({"symptomType": "abdominalCramps", "severity": "mild"})
        elif 25 <= cycle_day <= 28:
            samples.append({"symptomType": "bloating", "severity": "mild"})
            samples.append({"symptomType": "appetiteChanges", "severity": "increased"})
        if samples:
            out.append(
                _record(
                    "symptom",
                    f"{day.isoformat()}-o",
                    owner=DEMO_OWNER_ID,
                    payload={
                        "dayID": f"symptom-demo0001-{_epoch(day)}",
                        "dayStartDate": _noon(day),
                        "samples": [
                            {
                                **sample,
                                "startDate": _iso(_dt(day, 8)),
                                "endDate": _iso(_dt(day, 22)),
                            }
                            for sample in samples
                        ],
                    },
                    created_at=_uploaded(day),
                )
            )
        # The partner logs occasionally, which is what makes the per-owner
        # grouping in `summarize_symptoms` visible rather than theoretical.
        if rng.random() > 0.9:
            symptom_type = rng.choice(("moodChanges", "fatigue", "lowerBackPain"))
            severity = "present" if symptom_type == "moodChanges" else "mild"
            out.append(
                _record(
                    "symptom",
                    f"{day.isoformat()}-p",
                    owner=DEMO_PARTNER_ID,
                    payload={
                        "dayID": f"symptom-demo0002-{_epoch(day)}",
                        "dayStartDate": _noon(day),
                        "samples": [
                            {
                                "symptomType": symptom_type,
                                "severity": severity,
                                "startDate": _iso(_dt(day, 9)),
                                "endDate": _iso(_dt(day, 21)),
                            }
                        ],
                    },
                    created_at=_uploaded(day),
                )
            )
    return out


#: (back, targetKind, owner, text). `sleep` and `menstrual` are the iOS-authored
#: kinds; `mood` and `general` are the two an agent may write via `log_note`
#: (AGENT_NOTE_KINDS). All four appear so `get_notes(target_kind=...)` has
#: something to filter in every case.
_NOTES: tuple[tuple[int, str, str, str], ...] = (
    (2, "sleep", DEMO_OWNER_ID, "Slept with the window open. Woke up before the alarm."),
    (3, "general", DEMO_OWNER_ID, "Started walking home from the station instead of taking the bus."),
    (5, "mood", DEMO_OWNER_ID, "Good week. Finished the chapter I had been stuck on."),
    (9, "sleep", DEMO_PARTNER_ID, "Left the watch on the charger, so no sleep stages for tonight."),
    (12, "menstrual", DEMO_OWNER_ID, "Day one. Kept the session light and skipped the last set."),
    (18, "general", DEMO_OWNER_ID, "Swapped the afternoon coffee for tea. Falling asleep faster."),
    (24, "mood", DEMO_OWNER_ID, "Quiet weekend at home. Wanted exactly that."),
    (31, "sleep", DEMO_OWNER_ID, "Late dinner, took a while to settle."),
    (40, "menstrual", DEMO_OWNER_ID, "Cramps eased by the afternoon without anything for it."),
    (47, "general", DEMO_OWNER_ID, "Added a fourth set to the squats. Felt fine the next day."),
    (58, "mood", DEMO_OWNER_ID, "Long call with family. Worth staying up for."),
    (73, "sleep", DEMO_OWNER_ID, "Hotel bed, unfamiliar pillow. Still got seven hours."),
)


def _build_note(rng: random.Random) -> list[DecryptedRecord]:
    """`{noteID, targetKind, targetDate, text, createdAt, updatedAt}`.

    Hand-written rather than generated: notes are the one kind whose value is
    entirely in the words, and randomly assembled sentences would read as
    filler in the exact place a reader is looking for meaning. The ids stay
    opaque and random-shaped, matching the real `note-{16B hex}` scheme (which
    is why `note` is collision-free without a device prefix, Invariant 17).
    """

    out: list[DecryptedRecord] = []
    for index, (back, target_kind, owner, text) in enumerate(_NOTES):
        day = _day(back)
        note_id = f"note-{rng.getrandbits(64):016x}"
        payload = {
            "noteID": note_id,
            "targetKind": target_kind,
            "targetDate": _noon(day),
            "text": text,
            "createdAt": _iso(_dt(day, 21, 30)),
            "updatedAt": _iso(_dt(day, 21, 30)),
        }
        out.append(
            _record(
                "note",
                f"{index:02d}",
                owner=owner,
                payload=payload,
                created_at=_uploaded(day),
            )
        )
    return out


#: Three-day rotation. (exercise, sets, reps, starting weight kg at the OLDEST
#: session). Pull-Up carries 0.0 kg — bodyweight — which is legal and keeps
#: `total_volume_kg` honest instead of inventing a load nobody lifted.
_STRENGTH_ROTATION: tuple[tuple[str, tuple[tuple[str, int, int, float], ...]], ...] = (
    (
        "Lower A",
        (
            ("Back Squat", 5, 5, 80.0),
            ("Romanian Deadlift", 4, 8, 82.5),
            ("Leg Press", 3, 12, 140.0),
        ),
    ),
    (
        "Upper A",
        (
            ("Bench Press", 5, 5, 60.0),
            ("Barbell Row", 4, 8, 52.5),
            ("Pull-Up", 4, 8, 0.0),
        ),
    ),
    (
        "Upper B",
        (
            ("Overhead Press", 5, 5, 40.0),
            ("Incline Dumbbell Press", 4, 10, 22.0),
            ("Lat Pulldown", 4, 12, 50.0),
        ),
    ),
)


def _build_strength(rng: random.Random) -> list[DecryptedRecord]:
    """`{entryID, date, exercises: [{name, sets: [{weightKg, reps}]}], note?, createdAt, updatedAt}`.

    Owner only (`.ownDevicesOnly`, no partner fan-out). Mon/Wed/Fri, rotating
    through three sessions, with the load stepping up 2.5 kg every three weeks
    so `total_volume_kg` climbs across the window instead of oscillating around
    a mean. Progressive overload is the entire reason anyone keeps a strength
    log; a demo where the numbers wander says nothing.
    """

    out: list[DecryptedRecord] = []
    session_index = 0
    for back in reversed(range(DEMO_DAYS)):  # oldest first, so progression rises
        day = _day(back)
        if day.weekday() not in (0, 2, 4):
            continue
        label, plan = _STRENGTH_ROTATION[session_index % len(_STRENGTH_ROTATION)]
        step = 2.5 * (session_index // 9)  # ~3 weeks at 3 sessions/week
        exercises = []
        for name, set_count, reps, base_weight in plan:
            weight = base_weight + (step if base_weight > 0 else 0.0)
            sets = []
            for set_number in range(set_count):
                # The last set usually drops a rep or two — real logs do.
                actual_reps = reps if set_number < set_count - 1 else max(reps - rng.randint(0, 2), 1)
                sets.append({"weightKg": round(weight, 1), "reps": actual_reps})
            exercises.append({"name": name, "sets": sets})
        payload: dict[str, Any] = {
            "entryID": f"strength-{rng.getrandbits(64):016x}",
            "date": _noon(day),
            "exercises": exercises,
            "createdAt": _iso(_dt(day, 18, 40)),
            "updatedAt": _iso(_dt(day, 18, 40)),
        }
        if session_index % 7 == 0:
            payload["note"] = f"{label} — felt strong, kept the rest short."
        out.append(
            _record(
                "strength",
                day.isoformat(),
                owner=DEMO_OWNER_ID,
                payload=payload,
                created_at=_uploaded(day),
            )
        )
        session_index += 1
    return out


#: (food, portion, kcal, protein g, fat g, carb g). Structured nutrition on
#: every item, deliberately: the fields shipped 2026-07-24 and then went
#: unfilled for weeks while estimates lived in free-text `portion` strings that
#: no chart can read. A demo that leaves them null teaches the wrong habit.
#:
#: Each meal is ANCHOR items (every day) plus one rotating side, rather than a
#: free draw from a pool. A free draw was the first attempt and it landed the
#: daily total anywhere from 1,600 to 2,400 kcal depending on whether the
#: protein source happened to be picked — which broke the one thing the food
#: log has to do here, which is agree with `get_total_energy_burned`. Anchors
#: hold the total in a band; the sides supply the variety.
_BREAKFAST_ANCHOR = ("Greek yogurt", "200 g", 180, 20.0, 5.0, 12.0)
_BREAKFAST_SIDES: tuple[tuple[str, str, int, float, float, float], ...] = (
    ("Oatmeal with banana", "1 bowl", 380, 12.0, 7.0, 66.0),
    ("Peanut butter toast", "2 slices", 375, 12.0, 14.0, 48.0),
    ("Two eggs on toast", "2 eggs + 1 slice", 345, 21.0, 13.0, 33.0),
)
_LUNCH_ANCHORS: tuple[tuple[str, str, int, float, float, float], ...] = (
    ("Chicken breast", "150 g", 250, 47.0, 5.5, 0.0),
    ("Brown rice", "1 bowl", 340, 7.0, 2.5, 72.0),
)
_LUNCH_SIDES: tuple[tuple[str, str, int, float, float, float], ...] = (
    ("Stir-fried greens", "1 plate", 120, 4.0, 7.0, 9.0),
    ("Miso soup and salad", "1 set", 130, 5.0, 6.0, 11.0),
    ("Roasted broccoli", "1 plate", 110, 5.0, 6.0, 8.0),
)
_DINNER_MAINS: tuple[tuple[str, str, int, float, float, float], ...] = (
    ("Salmon fillet", "150 g", 310, 34.0, 19.0, 0.0),
    ("Tofu and mushrooms", "1 plate", 300, 26.0, 18.0, 12.0),
    ("Lean beef stir-fry", "150 g", 320, 33.0, 16.0, 6.0),
)
_DINNER_ANCHOR = ("Sweet potato", "1 medium", 180, 3.5, 0.3, 41.0)
_DINNER_SIDES: tuple[tuple[str, str, int, float, float, float], ...] = (
    ("Mixed salad", "1 bowl", 90, 3.0, 5.0, 8.0),
    ("Steamed greens", "1 plate", 85, 4.0, 4.0, 7.0),
)
_SNACKS: tuple[tuple[str, str, int, float, float, float], ...] = (
    ("Whey shake", "1 scoop", 130, 25.0, 1.5, 3.0),
    ("Almonds and an apple", "30 g + 1", 270, 6.5, 15.3, 31.0),
    ("Cottage cheese", "150 g", 160, 20.0, 5.0, 6.0),
)


def _food_item(entry: tuple[str, str, int, float, float, float]) -> dict[str, Any]:
    food, portion, kcal, protein, fat, carb = entry
    return {
        "food": food,
        "portion": portion,
        "kcal": kcal,
        "proteinGrams": protein,
        "fatGrams": fat,
        "carbGrams": carb,
    }


def _build_food(rng: random.Random) -> list[DecryptedRecord]:
    """`{entryID, date, meals: [{name, timeOfDay, items: [...]}], note?, createdAt, updatedAt}`.

    Owner only. Three meals plus a snack, landing around 2,000 kcal and ~145 g
    protein. Those two numbers are not decoration: basal (~1,750 kcal from
    `basal_energy`) plus active (~430 from `activity`) puts TDEE near 2,180, so
    a ~2,020 intake is a ~170 kcal/day deficit — which is exactly the
    ~0.15 kg/week the weight series loses. Four kinds have to agree here, and a demo
    where intake contradicts burn invites the reader to conclude one of the
    tools is broken rather than that the person ate a lot on Tuesday.
    """

    out: list[DecryptedRecord] = []
    for back in range(DEMO_DAYS):
        day = _day(back)
        meals: list[dict[str, Any]] = [
            {
                "name": "Breakfast",
                "timeOfDay": f"07:{rng.randint(10, 55):02d}",
                "items": [
                    _food_item(_BREAKFAST_ANCHOR),
                    _food_item(rng.choice(_BREAKFAST_SIDES)),
                ],
            },
            {
                "name": "Lunch",
                "timeOfDay": f"12:{rng.randint(0, 45):02d}",
                "items": [_food_item(e) for e in _LUNCH_ANCHORS]
                + [_food_item(rng.choice(_LUNCH_SIDES))],
            },
            {
                "name": "Dinner",
                "timeOfDay": f"19:{rng.randint(0, 50):02d}",
                "items": [
                    _food_item(rng.choice(_DINNER_MAINS)),
                    _food_item(_DINNER_ANCHOR),
                    _food_item(rng.choice(_DINNER_SIDES)),
                ],
            },
            {
                "name": "Snack",
                "timeOfDay": f"16:{rng.randint(0, 55):02d}",
                "items": [_food_item(rng.choice(_SNACKS))],
            },
        ]
        payload: dict[str, Any] = {
            "entryID": f"food-{rng.getrandbits(64):016x}",
            "date": _noon(day),
            "meals": meals,
            "createdAt": _iso(_dt(day, 20, 15)),
            "updatedAt": _iso(_dt(day, 20, 15)),
        }
        out.append(
            _record(
                "food",
                day.isoformat(),
                owner=DEMO_OWNER_ID,
                payload=payload,
                created_at=_uploaded(day),
            )
        )
    return out


# ── Registry + coverage ──────────────────────────────────────────────────────

#: Keys are checked against the IMPORTED `KNOWN_METRIC_TYPES`, never against a
#: retyped list — see rule 1 in the module docstring.
_BUILDERS: dict[str, Callable[[random.Random], list[DecryptedRecord]]] = {
    "sleep": _build_sleep,
    "water": _build_water,
    "body": _build_body,
    "menstrual": _build_menstrual,
    "activity": _build_activity,
    "resting_hr": _build_resting_hr,
    "workout": _build_workout,
    "mindfulness": _build_mindfulness,
    "hrv": _build_hrv,
    "hrv_hourly": _build_hrv_hourly,
    "wrist_temp": _build_wrist_temp,
    "vo2max": _build_vo2max,
    "basal_energy": _build_basal_energy,
    "symptom": _build_symptom,
    "note": _build_note,
    "strength": _build_strength,
    "food": _build_food,
}


def missing_kinds() -> frozenset[str]:
    """Known metric kinds this file has no builder for.

    Asserted empty by the test suite. When a new kind lands, that assertion is
    what says "the demo does not cover this yet" out loud, instead of the demo
    quietly shipping with a hole in it — the same failure mode
    `check_metric_type_contract.py` exists to prevent for the other ten
    registration sites, which cannot reach this file.
    """

    return frozenset(KNOWN_METRIC_TYPES) - frozenset(_BUILDERS)


def stale_kinds() -> frozenset[str]:
    """Builders for kinds that are no longer in `KNOWN_METRIC_TYPES`.

    The other direction, and worth checking too: a retired kind whose builder
    is still here would fabricate records for something the read path has
    stopped accepting.
    """

    return frozenset(_BUILDERS) - frozenset(KNOWN_METRIC_TYPES)


_CACHE: dict[str, list[DecryptedRecord]] = {}


def reset_cache() -> None:
    """Drop the memoized dataset. For tests that assert regeneration."""

    _CACHE.clear()


def _records_for(kind: str) -> list[DecryptedRecord]:
    if kind in _CACHE:
        return _CACHE[kind]
    builder = _BUILDERS.get(kind)
    if builder is None:
        # Empty rather than raising: an unbuilt new kind should cost that ONE
        # tool its output, not take down the other sixteen. `missing_kinds()`
        # is where it becomes an assertion.
        _LOG.warning(
            "demo mode has no dataset for metric_type=%r; that tool will return empty", kind
        )
        _CACHE[kind] = []
        return _CACHE[kind]
    records = builder(_rng(kind))
    _CACHE[kind] = records
    return records


# ── Public API ───────────────────────────────────────────────────────────────


def demo_enabled() -> bool:
    """True when `VAULTBEAT_DEMO` is set to something truthy.

    Deliberately strict about what counts as on: "0", "false" and "no" mean
    off, so a wrapper that sets the variable to a falsey string cannot
    accidentally serve synthetic health data to somebody expecting their own.
    """

    raw = os.getenv(DEMO_ENV, "").strip().lower()
    return raw not in ("", "0", "false", "no", "off")


def demo_records(metric_type: str | None = None) -> list[DecryptedRecord]:
    """Synthetic records for one kind, or for every kind when `metric_type` is None.

    Mirrors what `sync_decrypted_records` returns AFTER decryption: the same
    frozen `DecryptedRecord` objects, with `payload` already decoded from JSON.

    The kind list comes from `KNOWN_METRIC_TYPES`, iterated in sorted order so
    the concatenation is stable; per-kind seeding means the slice for one kind
    is identical whether you ask for it alone or as part of the whole set.

    The returned list is fresh, but the records inside it are memoized and
    shared across calls (they are frozen dataclasses and every reader in this
    package treats `payload` as read-only). Do not mutate a payload in place.
    """

    if metric_type is not None:
        return list(_records_for(metric_type))
    out: list[DecryptedRecord] = []
    for kind in sorted(KNOWN_METRIC_TYPES):
        out.extend(_records_for(kind))
    return out


def demo_sync_result(
    *,
    metric_type: str | None = None,
    limit: int | None = None,
) -> tuple[list[DecryptedRecord], list[str]]:
    """Drop-in replacement for `sync_decrypted_records`' return value.

    Same `(records, errors)` contract, so the injection at the top of that
    method is one line and every caller downstream is untouched. `errors` is
    always empty: a decrypt error means a real blob failed to open, and
    inventing one here would put a fake data-integrity problem in front of
    somebody evaluating whether to trust this product with their health record.

    `limit` is applied the same way the real method applies it — the full set
    is built, then the copy handed back is trimmed.
    """

    records = demo_records(metric_type)
    if limit is not None:
        records = records[:limit]
    return records, []


def demo_status(*, bound: bool = False) -> dict[str, Any]:
    """Facts a caller may splice into `status()` / `doctor()` while demo mode is on.

    Machine values plus one banner string that this module owns — the sentence
    is built here, not received from anywhere, so it is not a channel anything
    external can write into (Anti-pattern 23).

    `bound` changes only the wording, and it has to. Demo mode reads no config,
    so it runs perfectly well on a machine that IS paired — and there the old
    note ("binding still requires a real paired account") sat next to
    `bound: true` and read as confirmation that the stored binding was in use.
    It was not: the binding is intact, connected, and completely bypassed. The
    default is False so a caller that does not know stays on the wording that
    claims least.
    """

    return {
        # Banner first — see `_watermark_demo`. `demo_status` is spread into
        # `status()`, whose result is the first thing a first-time user calls.
        # NOTE the key: `demo_warning`, matching every other demo-bearing
        # surface. It was `demo_banner` until 2026-08-20 — a split that had one
        # concrete cost beyond tidiness: `cli.py` renders `demo_warning`, so
        # `status` was the ONE command that printed no banner at all.
        "demo_warning": DEMO_BANNER,
        "demo_mode": True,
        "demo_env": DEMO_ENV,
        "demo_anchor_day": DEMO_ANCHOR_DAY,
        "demo_owner_prefixes": [DEMO_OWNER_PREFIX, DEMO_PARTNER_PREFIX],
        "demo_kinds": sorted(_BUILDERS),
        "demo_record_count": sum(len(_records_for(k)) for k in sorted(_BUILDERS)),
        "demo_write_tools_available": False,
        "demo_note": (
            (
                "Read tools serve synthetic records generated on this machine; nothing "
                "is fetched or decrypted. THIS MACHINE IS PAIRED AND ITS BINDING IS "
                "BEING BYPASSED — the real account is untouched and no real record has "
                "been read, but every number below is synthetic. The log_* write tools "
                "stay disabled while demo mode is on."
                if bound
                else "Read tools serve synthetic records generated on this machine; "
                "nothing is fetched or decrypted. The log_* write tools and binding "
                "still require a real paired account."
            )
        ),
    }
