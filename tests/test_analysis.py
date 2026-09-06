"""Arithmetic, and the three refusals that matter more than the arithmetic.

Half of this file asserts that a number is right; the other half asserts that
NO number is produced where one would be an artefact. The second half is the
reason the module exists — an agent asked for a correlation will otherwise
compute one itself, from two points, and report it.
"""

from __future__ import annotations

import asyncio
import inspect
import math
from pathlib import Path
from typing import Any

import pytest

from vaultbeat_mcp_local.analysis import (
    MIN_PAIRS_FOR_CORRELATION,
    SERIES,
    SeriesSpec,
    compare_periods,
    correlate,
    daily_series,
    lookup,
    series_catalog,
    trend,
)
from vaultbeat_mcp_local.service import VaultbeatLocalService
from vaultbeat_mcp_local.store import ConfigStore

SPEC = SeriesSpec("probe", "resting_hr_records", "records", "bpm", "bpm")
OTHER = SeriesSpec("probe_b", "hrv_records", "records", "sdnn_ms", "ms")


def _days(*pairs: tuple[str, float]) -> dict[str, float]:
    return dict(pairs)


# ── The registry is wired to real code ──────────────────────────────────────


def test_every_series_names_a_service_method_that_exists() -> None:
    """A typo in `SERIES` would surface as "no data" — the least debuggable
    failure this product has, because it is indistinguishable from the four
    legitimate causes of an empty result (Invariant 57)."""

    for spec in SERIES:
        method = getattr(VaultbeatLocalService, spec.method, None)
        assert method is not None, f"{spec.name}: no service method {spec.method!r}"
        assert inspect.iscoroutinefunction(method), f"{spec.name}: {spec.method} is not async"


def test_series_names_are_unique_and_catalogued() -> None:
    names = [s.name for s in SERIES]
    assert len(names) == len(set(names))
    assert {entry["series"] for entry in series_catalog()} == set(names)
    for entry in series_catalog():
        assert entry["unit"], entry


def test_every_series_carries_a_unit() -> None:
    """A bare number with no unit is the input to a wrong sentence."""

    for spec in SERIES:
        assert spec.unit.strip(), spec.name


def test_lookup_is_exact_not_fuzzy() -> None:
    assert lookup("resting_hr") is not None
    assert lookup("Resting_HR") is None, "a near-miss must fail loudly, not silently pick one"
    assert lookup("") is None


# ── Bucketing ───────────────────────────────────────────────────────────────


def test_several_samples_in_one_day_are_averaged_and_counted() -> None:
    summary = {
        "records": [
            {"local_date": "2026-08-01", "bpm": 50},
            {"local_date": "2026-08-01", "bpm": 60},
            {"local_date": "2026-08-02", "bpm": 55},
        ]
    }
    points, consumed = daily_series(summary, SPEC)
    assert points == {"2026-08-01": 55.0, "2026-08-02": 55.0}
    assert consumed == 3, "the row count has to survive so an agent can tell 3 days from 3 samples"


def test_a_bool_is_not_a_measurement() -> None:
    """`bool` is an `int` subclass; `True` would silently bucket as 1.0."""

    points, _ = daily_series({"records": [{"local_date": "2026-08-01", "bpm": True}]}, SPEC)
    assert points == {}


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_values_are_dropped(bad: float) -> None:
    points, _ = daily_series({"records": [{"local_date": "2026-08-01", "bpm": bad}]}, SPEC)
    assert points == {}


def test_rows_without_a_day_are_skipped_not_guessed() -> None:
    summary = {"records": [{"bpm": 50}, {"local_date": "2026-08-01", "bpm": 60}]}
    points, consumed = daily_series(summary, SPEC)
    assert points == {"2026-08-01": 60.0}
    assert consumed == 1


def test_a_missing_array_is_empty_not_an_error() -> None:
    assert daily_series({}, SPEC) == ({}, 0)
    assert daily_series({"records": "not a list"}, SPEC) == ({}, 0)


# ── Trend ───────────────────────────────────────────────────────────────────


def test_slope_is_per_day_not_per_observation() -> None:
    """THE load-bearing test of this module.

    Three points spanning 10 days rise by 10 in total, so the slope is 1.0/day.
    Fitting against the row INDEX instead gives 5.0 — a five-fold overstatement
    that looks entirely plausible and would be reported as fact. Any refactor
    that "simplifies" the x axis back to enumerate() fails here.
    """

    result = trend(
        _days(("2026-08-01", 10.0), ("2026-08-06", 15.0), ("2026-08-11", 20.0)), SPEC
    )
    assert result["slope_per_day"] == pytest.approx(1.0)
    assert result["change"] == pytest.approx(10.0)


def test_trend_reports_endpoints_and_spread() -> None:
    result = trend(_days(("2026-08-01", 10.0), ("2026-08-02", 30.0), ("2026-08-03", 20.0)), SPEC)
    assert result["first_value"] == 10.0
    assert result["last_value"] == 20.0
    assert result["mean"] == pytest.approx(20.0)
    assert result["median"] == pytest.approx(20.0)
    assert result["min"] == 10.0
    assert result["max"] == 30.0
    assert result["change_pct"] == pytest.approx(100.0)


def test_two_points_refuse_a_slope_and_say_why() -> None:
    result = trend(_days(("2026-08-01", 10.0), ("2026-08-02", 20.0)), SPEC)
    assert result["slope_per_day"] is None
    assert "not a finding of 'no trend'" in result["reason"]


def test_an_empty_series_produces_no_numbers_at_all() -> None:
    result = trend({}, SPEC)
    assert result["mean"] is None and result["slope_per_day"] is None
    assert result["first_value"] is None


def test_change_pct_is_none_rather_than_infinite_when_the_baseline_is_zero() -> None:
    result = trend(_days(("2026-08-01", 0.0), ("2026-08-02", 5.0), ("2026-08-03", 9.0)), SPEC)
    assert result["change_pct"] is None
    assert result["change"] == pytest.approx(9.0)


def test_a_flat_series_has_a_zero_slope_not_a_refusal() -> None:
    """Zero is a measurement here; None would say "could not compute"."""

    result = trend(_days(("2026-08-01", 7.0), ("2026-08-02", 7.0), ("2026-08-03", 7.0)), SPEC)
    assert result["slope_per_day"] == pytest.approx(0.0)


# ── Compare ─────────────────────────────────────────────────────────────────


def test_compare_reports_both_windows_and_their_difference() -> None:
    recent = _days(("2026-08-10", 60.0), ("2026-08-11", 64.0))
    previous = _days(("2026-08-01", 50.0), ("2026-08-02", 50.0))
    result = compare_periods(recent, previous, SPEC)
    assert result["recent"]["mean"] == pytest.approx(62.0)
    assert result["previous"]["mean"] == pytest.approx(50.0)
    assert result["mean_change"] == pytest.approx(12.0)
    assert result["mean_change_pct"] == pytest.approx(24.0)


def test_compare_names_the_days_each_window_actually_covers() -> None:
    """Without these an agent cannot tell "the week before" from "six weeks before"."""

    result = compare_periods(
        _days(("2026-08-10", 1.0)), _days(("2026-06-01", 1.0)), SPEC
    )
    assert result["recent"]["first_day"] == "2026-08-10"
    assert result["previous"]["last_day"] == "2026-06-01"


def test_an_empty_window_refuses_rather_than_comparing_against_zero() -> None:
    result = compare_periods(_days(("2026-08-10", 60.0)), {}, SPEC)
    assert result["mean_change"] is None
    assert "not a value of zero" in result["reason"]


def test_compare_says_nothing_about_which_window_is_better() -> None:
    """The line from `CLAUDE.md`: we give numbers, the agent and the user decide."""

    result = compare_periods(
        _days(("2026-08-10", 90.0)), _days(("2026-08-01", 50.0)), SPEC
    )
    blob = " ".join(str(v) for v in result.values()).lower()
    for verdict in ("better", "worse", "improv", "declin", "healthy", "poor", "good"):
        assert verdict not in blob, f"the response editorialises: {verdict!r}"


# ── Correlate ───────────────────────────────────────────────────────────────


def test_perfect_positive_and_negative_relationships() -> None:
    a = _days(("2026-08-01", 1.0), ("2026-08-02", 2.0), ("2026-08-03", 3.0))
    up = _days(("2026-08-01", 10.0), ("2026-08-02", 20.0), ("2026-08-03", 30.0))
    down = _days(("2026-08-01", 30.0), ("2026-08-02", 20.0), ("2026-08-03", 10.0))
    assert correlate(a, up, SPEC, OTHER)["pearson_r"] == pytest.approx(1.0)
    assert correlate(a, down, SPEC, OTHER)["pearson_r"] == pytest.approx(-1.0)


def test_pearson_matches_a_value_computed_from_the_definition() -> None:
    """Closed form, so a refactor cannot quietly change the formula.

    x = 1,2,3,4,5 (mean 3) · y = 2,4,5,4,5 (mean 4)
      Σdx·dy = 6 · Σdx² = 10 · Σdy² = 6  →  r = 6/√60 ≈ 0.774596669

    🔑 The first draft of this test asserted 0.8, from a coefficient computed in
    my head while writing it. It was wrong by 3%, it looked entirely reasonable,
    and only the implementation disagreeing with it surfaced that. That is this
    whole module's argument in one line: an LLM asked for a correlation WILL
    produce one, and it will be plausible. The expected value below is written
    as the arithmetic rather than as a decimal so the next reader checks the
    definition instead of trusting a number someone typed.
    """

    a = {f"2026-08-0{i}": float(i) for i in range(1, 6)}
    b = dict(zip(a, [2.0, 4.0, 5.0, 4.0, 5.0]))
    assert correlate(a, b, SPEC, OTHER)["pearson_r"] == pytest.approx(6 / math.sqrt(10 * 6))


def test_only_days_present_on_both_sides_are_paired() -> None:
    """Missing days are dropped, never filled — the module's invariant 1."""

    a = _days(("2026-08-01", 1.0), ("2026-08-02", 2.0), ("2026-08-03", 3.0), ("2026-08-04", 4.0))
    b = _days(("2026-08-02", 20.0), ("2026-08-03", 30.0), ("2026-08-09", 90.0))
    result = correlate(a, b, SPEC, OTHER)
    assert result["n_pairs"] == 2
    assert result["n_days_a"] == 4 and result["n_days_b"] == 3
    assert result["pearson_r"] is None, "2 pairs is below the floor"


@pytest.mark.parametrize("n", [0, 1, 2])
def test_too_few_pairs_refuses_with_the_counts_that_explain_it(n: int) -> None:
    a = {f"2026-08-0{i}": float(i) for i in range(1, n + 1)}
    b = {f"2026-08-0{i}": float(i * 2) for i in range(1, n + 1)}
    result = correlate(a, b, SPEC, OTHER)
    assert result["pearson_r"] is None
    assert str(MIN_PAIRS_FOR_CORRELATION) in result["reason"]
    assert "not evidence that the two are unrelated" in result["reason"]


def test_a_constant_series_is_undefined_not_zero() -> None:
    """r = 0 would read as "no relationship"; the truth is "unanswerable"."""

    a = _days(("2026-08-01", 5.0), ("2026-08-02", 5.0), ("2026-08-03", 5.0))
    b = _days(("2026-08-01", 1.0), ("2026-08-02", 2.0), ("2026-08-03", 3.0))
    result = correlate(a, b, SPEC, OTHER)
    assert result["pearson_r"] is None
    assert "undefined (not zero)" in result["reason"]


def test_every_correlation_carries_the_causation_caveat() -> None:
    a = _days(("2026-08-01", 1.0), ("2026-08-02", 2.0), ("2026-08-03", 3.0))
    for other in (a, {}):
        result = correlate(a, other, SPEC, OTHER)
        assert "cannot show that either metric caused the other" in result["caveat"]


def test_the_result_never_grades_the_coefficient() -> None:
    a = _days(("2026-08-01", 1.0), ("2026-08-02", 2.0), ("2026-08-03", 3.0))
    b = _days(("2026-08-01", 2.0), ("2026-08-02", 4.0), ("2026-08-03", 6.0))
    result = correlate(a, b, SPEC, OTHER)
    assert result["pearson_r"] == pytest.approx(1.0)
    blob = " ".join(str(v) for v in result.values() if v is not None).lower()
    # "strong" appears nowhere; the caveat says not to translate r into a word,
    # and shipping the word ourselves would be the same mistake with authority.
    assert "strong correlation" not in blob
    assert "weak correlation" not in blob


# ── Wired into the real service ─────────────────────────────────────────────


def _demo_service(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> VaultbeatLocalService:
    import vaultbeat_mcp_local.demo as demo_module

    monkeypatch.setenv(demo_module.DEMO_ENV, "1")
    demo_module.reset_cache()
    return VaultbeatLocalService(ConfigStore(tmp_path / "config.json"), demo=True)


def test_every_series_returns_days_through_the_real_service(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Not just that the method exists — that the array and field names find data.

    A wrong `field` passes `test_every_series_names_a_service_method_that_exists`
    and then returns an empty series forever.
    """

    service = _demo_service(tmp_path, monkeypatch)
    for spec in SERIES:
        result = asyncio.run(service.metric_trend(series=spec.name, days=7))
        assert result.get("coverage"), f"{spec.name}: no coverage block"
        assert result["coverage"]["days_covered"] > 0, (
            f"{spec.name}: resolved to zero days — check `array`/`field` against "
            f"what {spec.method} actually returns"
        )


def test_an_unknown_series_answers_with_the_available_ones(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An agent that guessed wrong needs the list, not a traceback."""

    service = _demo_service(tmp_path, monkeypatch)
    result = asyncio.run(service.metric_trend(series="blood_pressure", days=7))
    assert result["error"] == "unknown_series"
    assert {entry["series"] for entry in result["available_series"]} == {s.name for s in SERIES}


def test_correlation_coverage_describes_the_shared_days_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Invariant 62: coverage must describe the set the number came from.

    A union here would advertise 30 days behind an r built on 11.
    """

    service = _demo_service(tmp_path, monkeypatch)
    result = asyncio.run(
        service.metric_correlate(series_a="sleep_minutes", series_b="vo2max", days=60)
    )
    assert result["coverage"]["days_covered"] == result["n_pairs"]
    assert result["n_pairs"] <= min(result["n_days_a"], result["n_days_b"])


def test_compare_windows_do_not_overlap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = _demo_service(tmp_path, monkeypatch)
    result = asyncio.run(service.metric_compare_periods(series="resting_hr", days=7))
    recent, previous = result["recent"], result["previous"]
    assert previous["last_day"] < recent["first_day"], (
        "the two windows share a day, so the same reading is on both sides of the change"
    )


def test_a_sparse_series_says_so_through_span_days(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`days` counts days WITH DATA, so span is the only thing that reveals sparsity."""

    service = _demo_service(tmp_path, monkeypatch)
    result = asyncio.run(service.metric_trend(series="vo2max", days=30))
    coverage = result["coverage"]
    assert coverage["span_days"] is not None
    assert coverage["span_days"] >= coverage["days_covered"]


def test_analysis_results_never_carry_a_parallel_day_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One vocabulary for "how much data is this", and it is `coverage`.

    `STYLE` tells every agent to quote `coverage.days_covered`; a second field
    saying the same thing under another name is a mirror that will eventually
    disagree with the block printed beside it.
    """

    service = _demo_service(tmp_path, monkeypatch)
    for call in (
        service.metric_trend(series="resting_hr", days=7),
        service.metric_correlate(series_a="resting_hr", series_b="steps", days=7),
    ):
        result = asyncio.run(call)
        assert "n_days" not in result
        assert "span_days" not in result
        assert "first_day" not in result


def test_analysis_tools_do_not_reach_the_network_in_demo_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same guarantee every other read tool gives; asserted because these three
    fan out to a read method twice and a regression would double the traffic."""

    service = _demo_service(tmp_path, monkeypatch)
    result: dict[str, Any] = asyncio.run(
        service.metric_correlate(series_a="steps", series_b="active_energy", days=14)
    )
    assert math.isfinite(result["pearson_r"] or 0.0)
    assert not (tmp_path / "config.json").exists()
