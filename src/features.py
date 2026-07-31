"""Pure-Python reference implementations of the fraud feature engineering logic in
transforms/gold.py. Kept dependency-free (no Spark) so they're fast to unit test;
the DLT pipeline reimplements the same rules as vectorized DataFrame operations for
scale, but the intent -- and the edge cases -- should match what's tested here.
"""
from datetime import datetime, timedelta

HIGH_RISK_COUNTRIES = {"NG", "RU"}


def is_foreign_transaction(txn_country: str, home_country: str) -> bool:
    return txn_country != home_country


def is_high_risk_country(txn_country: str) -> bool:
    return txn_country in HIGH_RISK_COUNTRIES


def amount_zscore(amount: float, account_avg: float, account_stddev: float) -> float:
    """Standard score of `amount` against the account's historical mean/stddev.
    A zero (or missing) stddev would divide by zero for a brand-new account with
    a single transaction -- treat that as a neutral z-score of 0 rather than raising.
    """
    if not account_stddev:
        return 0.0
    return (amount - account_avg) / account_stddev


def velocity_count(event_times: list[datetime], current_time: datetime, window_seconds: int = 600) -> int:
    """Count of prior events (inclusive of current_time) within window_seconds before it."""
    window_start = current_time - timedelta(seconds=window_seconds)
    return sum(1 for t in event_times if window_start <= t <= current_time) + 1


def risk_score(is_foreign: bool, is_high_risk: bool, velocity: int, zscore: float) -> float:
    """Simple bounded composite risk score in [0, 1], combining the four signals above.
    Not the ML model -- a cheap rule-based score usable before a model exists (e.g. to
    sanity-check a newly registered model's output isn't wildly off from the heuristic).
    """
    score = 0.0
    score += 0.25 if is_foreign else 0.0
    score += 0.35 if is_high_risk else 0.0
    score += min(velocity / 10.0, 1.0) * 0.25
    score += min(max(zscore, 0.0) / 5.0, 1.0) * 0.15
    return min(score, 1.0)
