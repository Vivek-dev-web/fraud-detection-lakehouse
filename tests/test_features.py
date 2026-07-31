from datetime import datetime, timedelta

import pytest

from src.features import (
    amount_zscore,
    is_foreign_transaction,
    is_high_risk_country,
    risk_score,
    velocity_count,
)


def test_is_foreign_transaction_true_when_countries_differ():
    assert is_foreign_transaction("GB", "US") is True


def test_is_foreign_transaction_false_when_countries_match():
    assert is_foreign_transaction("US", "US") is False


def test_is_high_risk_country_known_list():
    assert is_high_risk_country("NG") is True
    assert is_high_risk_country("RU") is True


def test_is_high_risk_country_unknown_is_false():
    assert is_high_risk_country("US") is False


def test_amount_zscore_typical():
    assert amount_zscore(150.0, 100.0, 25.0) == pytest.approx(2.0)


def test_amount_zscore_below_mean_is_negative():
    assert amount_zscore(50.0, 100.0, 25.0) == pytest.approx(-2.0)


def test_amount_zscore_zero_stddev_returns_neutral_zero():
    # a brand-new account with one historical transaction has stddev 0/None;
    # dividing would raise, so this must return a neutral score instead of crashing
    assert amount_zscore(500.0, 500.0, 0.0) == 0.0


def test_velocity_count_counts_within_window():
    now = datetime(2026, 6, 1, 12, 0, 0)
    history = [now - timedelta(seconds=60), now - timedelta(seconds=300), now - timedelta(seconds=900)]
    # two of the three prior events are within the 600s window, plus the current one
    assert velocity_count(history, now, window_seconds=600) == 3


def test_velocity_count_empty_history_is_one():
    now = datetime(2026, 6, 1, 12, 0, 0)
    assert velocity_count([], now) == 1


def test_velocity_count_excludes_events_outside_window():
    now = datetime(2026, 6, 1, 12, 0, 0)
    history = [now - timedelta(seconds=601)]
    assert velocity_count(history, now, window_seconds=600) == 1


def test_risk_score_bounded_between_zero_and_one():
    score = risk_score(is_foreign=True, is_high_risk=True, velocity=50, zscore=10.0)
    assert 0.0 <= score <= 1.0
    assert score == 1.0  # all signals maxed out should saturate at 1.0


def test_risk_score_zero_for_benign_transaction():
    assert risk_score(is_foreign=False, is_high_risk=False, velocity=1, zscore=0.0) == pytest.approx(0.025)


def test_risk_score_negative_zscore_does_not_reduce_below_baseline():
    baseline = risk_score(is_foreign=False, is_high_risk=False, velocity=1, zscore=0.0)
    negative = risk_score(is_foreign=False, is_high_risk=False, velocity=1, zscore=-5.0)
    assert negative == pytest.approx(baseline)
