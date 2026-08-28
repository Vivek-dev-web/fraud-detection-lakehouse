import pytest

from src.data_quality import (
    FAIL,
    PASS,
    WARN,
    drift_check,
    freshness_check,
    range_check,
    summarize,
    zero_count_check,
)


def test_zero_count_check_passes_on_zero():
    result = zero_count_check("nulls", "integrity", 0, "key nulls")
    assert result.status == PASS


def test_zero_count_check_fails_on_nonzero():
    result = zero_count_check("nulls", "integrity", 3, "key nulls")
    assert result.status == FAIL
    assert "3" in result.message


def test_range_check_passes_within_bounds():
    assert range_check("prob_domain", "domain", 0.5, 0.0, 1.0, "fraud_probability").status == PASS


def test_range_check_passes_on_boundary():
    # inclusive bounds -- exactly 0 or 1 is a valid probability, not a violation
    assert range_check("prob_domain", "domain", 1.0, 0.0, 1.0, "fraud_probability").status == PASS


def test_range_check_fails_outside_bounds():
    result = range_check("prob_domain", "domain", 1.5, 0.0, 1.0, "fraud_probability")
    assert result.status == FAIL


def test_freshness_check_passes_within_threshold():
    assert freshness_check("ingest_freshness", "freshness", 30.0, 60.0, "bronze_transactions").status == PASS


def test_freshness_check_fails_when_stale():
    result = freshness_check("ingest_freshness", "freshness", 90.0, 60.0, "bronze_transactions")
    assert result.status == FAIL
    assert "90.0" in result.message


def test_drift_check_passes_within_deviation():
    assert drift_check("flag_rate_drift", "drift", 0.04, 0.04, 0.5, "flagged rate").status == PASS


def test_drift_check_warns_not_fails_when_deviating():
    # drift must degrade to WARN, never FAIL -- it should never be able to block the job
    result = drift_check("flag_rate_drift", "drift", 0.20, 0.04, 0.5, "flagged rate")
    assert result.status == WARN


def test_drift_check_zero_baseline_zero_current_passes():
    assert drift_check("flag_rate_drift", "drift", 0.0, 0.0, 0.5, "flagged rate").status == PASS


def test_drift_check_zero_baseline_nonzero_current_warns():
    result = drift_check("flag_rate_drift", "drift", 0.05, 0.0, 0.5, "flagged rate")
    assert result.status == WARN


def test_summarize_counts_by_status():
    results = [
        zero_count_check("a", "integrity", 0, "x"),
        zero_count_check("b", "integrity", 1, "y"),
        drift_check("c", "drift", 0.2, 0.04, 0.5, "z"),
    ]
    assert summarize(results) == {PASS: 1, WARN: 1, FAIL: 1}
