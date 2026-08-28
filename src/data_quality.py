"""Pure-Python helpers for turning a raw metric (a count, a rate, a min/max) into a
structured pass/fail/warn result. Used by governance/04_data_quality_checks.py, which
runs the SQL from docs/DESIGN.md §8 and hands the resulting scalars to these functions --
kept dependency-free (no Spark) so the pass/fail logic itself is what's unit tested here,
the same split as src/features.py.
"""
from dataclasses import dataclass

FAIL = "FAIL"
WARN = "WARN"
PASS = "PASS"


@dataclass(frozen=True)
class CheckResult:
    check_name: str
    category: str
    status: str
    actual_value: float
    expected: str
    message: str


def _result(check_name: str, category: str, actual_value: float, passed: bool, expected: str, fail_message: str, warn_only: bool) -> CheckResult:
    if passed:
        status, message = PASS, f"OK -- {actual_value} satisfies {expected}"
    else:
        status, message = (WARN if warn_only else FAIL), fail_message
    return CheckResult(check_name, category, status, float(actual_value), expected, message)


def zero_count_check(check_name: str, category: str, count: int, description: str) -> CheckResult:
    """A count that must be exactly zero -- key nulls, surviving duplicates, orphaned
    foreign keys, out-of-domain rows. Any non-zero count is a hard FAIL."""
    return _result(
        check_name, category, count, passed=(count == 0), expected="count == 0",
        fail_message=f"{description}: found {count} offending row(s), expected 0",
        warn_only=False,
    )


def range_check(check_name: str, category: str, value: float, low: float, high: float, description: str) -> CheckResult:
    """A value that must fall within [low, high] -- e.g. a probability in [0, 1].
    Hard FAIL: values outside a declared domain indicate corrupted data, not drift."""
    passed = low <= value <= high
    return _result(
        check_name, category, value, passed=passed, expected=f"{low} <= value <= {high}",
        fail_message=f"{description}: {value} outside expected range [{low}, {high}]",
        warn_only=False,
    )


def freshness_check(check_name: str, category: str, age_minutes: float, max_age_minutes: float, description: str) -> CheckResult:
    """Data age since the last ingest -- a hard FAIL past the threshold, since stale data
    silently serving predictions is exactly the kind of failure a customer needs to see caught."""
    passed = age_minutes <= max_age_minutes
    return _result(
        check_name, category, age_minutes, passed=passed, expected=f"age <= {max_age_minutes} min",
        fail_message=f"{description}: last ingest was {age_minutes:.1f} min ago, expected within {max_age_minutes} min",
        warn_only=False,
    )


def drift_check(check_name: str, category: str, current_rate: float, baseline_rate: float, max_relative_deviation: float, description: str) -> CheckResult:
    """Warn (never fail) when a rate strays too far from a historical baseline -- e.g. the
    live flagged-fraud rate vs. the labeled baseline. Drift is a signal to investigate, not
    proof the pipeline is broken, so it must never block the job the way the checks above do."""
    if baseline_rate == 0:
        deviation = float("inf") if current_rate else 0.0
    else:
        deviation = abs(current_rate - baseline_rate) / baseline_rate
    passed = deviation <= max_relative_deviation
    return _result(
        check_name, category, current_rate, passed=passed,
        expected=f"within {max_relative_deviation:.0%} of baseline {baseline_rate}",
        fail_message=f"{description}: {current_rate} deviates {deviation:.1%} from baseline {baseline_rate} (max {max_relative_deviation:.0%})",
        warn_only=True,
    )


def summarize(results: list[CheckResult]) -> dict:
    counts = {PASS: 0, WARN: 0, FAIL: 0}
    for r in results:
        counts[r.status] += 1
    return counts
