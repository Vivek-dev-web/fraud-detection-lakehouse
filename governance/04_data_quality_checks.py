# Databricks notebook source
# MAGIC %md
# MAGIC ### Automated data quality checks
# MAGIC Runs the checks documented in `docs/DESIGN.md` §8 (row-count reconciliation, null/
# MAGIC duplicate checks, referential integrity, domain/range checks, freshness, and
# MAGIC label/prediction drift) as a real job task instead of a manual runbook, and writes
# MAGIC every result to `gold_dq_results` so pass/fail history is queryable and dashboardable.
# MAGIC
# MAGIC Row-integrity and domain checks (nulls, duplicates, orphaned keys, out-of-range
# MAGIC values) are hard gates -- any FAIL raises and stops the job, same as a Lakeflow
# MAGIC `expect_or_drop` would. Drift (the live flagged rate vs. the historical baseline)
# MAGIC only ever WARNs -- see `drift_check` below for why it must never fail the job.
# MAGIC
# MAGIC The pass/fail/warn logic is reimplemented locally rather than imported from
# MAGIC `src/data_quality.py` -- same split as `transforms/gold.py` vs. `src/features.py`:
# MAGIC `src/` is the unit-tested pure-Python spec, notebooks reimplement it for execution,
# MAGIC since a bundle-deployed job task's `sys.path` doesn't reliably include the bundle
# MAGIC root (verified live: a first attempt at `from src.data_quality import ...` raised
# MAGIC `ModuleNotFoundError: No module named 'src'` when run as a job task).

# COMMAND ----------
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

# COMMAND ----------
dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "fraud_detection")
dbutils.widgets.text("max_freshness_minutes", "1440")
dbutils.widgets.text("max_drift_deviation", "0.75")
dbutils.widgets.text("enforce_quality_gate", "true")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
max_freshness_minutes = float(dbutils.widgets.get("max_freshness_minutes"))
max_drift_deviation = float(dbutils.widgets.get("max_drift_deviation"))
enforce_quality_gate = dbutils.widgets.get("enforce_quality_gate").lower() == "true"

spark.sql(f"USE CATALOG {catalog}")
spark.sql(f"USE SCHEMA {schema}")


# COMMAND ----------
# MAGIC %md #### Check evaluation helpers -- mirrors `src/data_quality.py`, see note above

# COMMAND ----------
@dataclass(frozen=True)
class CheckResult:
    check_name: str
    category: str
    status: str
    actual_value: float
    expected: str
    message: str


def _result(check_name, category, actual_value, passed, expected, fail_message, warn_only):
    if passed:
        status, message = "PASS", f"OK -- {actual_value} satisfies {expected}"
    else:
        status, message = ("WARN" if warn_only else "FAIL"), fail_message
    return CheckResult(check_name, category, status, float(actual_value), expected, message)


def zero_count_check(check_name, category, count, description):
    return _result(
        check_name, category, count, passed=(count == 0), expected="count == 0",
        fail_message=f"{description}: found {count} offending row(s), expected 0", warn_only=False,
    )


def freshness_check(check_name, category, age_minutes, max_age_minutes, description):
    passed = age_minutes <= max_age_minutes
    return _result(
        check_name, category, age_minutes, passed=passed, expected=f"age <= {max_age_minutes} min",
        fail_message=f"{description}: last ingest was {age_minutes:.1f} min ago, expected within {max_age_minutes} min",
        warn_only=False,
    )


def drift_check(check_name, category, current_rate, baseline_rate, max_relative_deviation, description):
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


def summarize(results):
    counts = {"PASS": 0, "WARN": 0, "FAIL": 0}
    for r in results:
        counts[r.status] += 1
    return counts


def scalar(sql: str):
    return spark.sql(sql).collect()[0][0]


results: list[CheckResult] = []

# COMMAND ----------
# MAGIC %md #### Row-count reconciliation (§8.1) -- silver should never exceed bronze

# COMMAND ----------
bronze_count = scalar("SELECT COUNT(*) FROM bronze_transactions")
silver_count = scalar("SELECT COUNT(*) FROM silver_transactions")
gold_count = scalar("SELECT COUNT(*) FROM gold_fraud_features")

results.append(
    zero_count_check(
        "silver_not_exceeding_bronze",
        "row_count",
        max(silver_count - bronze_count, 0),
        f"silver_transactions ({silver_count}) should never exceed bronze_transactions ({bronze_count})",
    )
)
results.append(
    zero_count_check(
        "gold_features_nonzero",
        "row_count",
        0 if gold_count > 0 else 1,
        f"gold_fraud_features has {gold_count} rows",
    )
)

# COMMAND ----------
# MAGIC %md #### Null & duplicate checks (§8.2)

# COMMAND ----------
key_nulls = scalar(
    "SELECT COUNT(*) FROM silver_transactions "
    "WHERE transaction_id IS NULL OR account_id IS NULL OR merchant_id IS NULL"
)
results.append(zero_count_check("silver_key_nulls", "integrity", key_nulls, "silver_transactions key columns"))

surviving_dupes = scalar(
    "SELECT COUNT(*) FROM (SELECT transaction_id FROM silver_transactions "
    "GROUP BY transaction_id HAVING COUNT(*) > 1)"
)
results.append(zero_count_check("silver_duplicate_transaction_ids", "integrity", surviving_dupes, "silver_transactions dedup"))

# COMMAND ----------
# MAGIC %md #### Referential integrity (§8.3)

# COMMAND ----------
orphaned_accounts = scalar(
    "SELECT COUNT(*) FROM bronze_transactions t "
    "LEFT ANTI JOIN silver_accounts a ON t.account_id = a.account_id"
)
results.append(zero_count_check("no_orphaned_accounts", "integrity", orphaned_accounts, "bronze_transactions -> silver_accounts"))

orphaned_merchants = scalar(
    "SELECT COUNT(*) FROM bronze_transactions t "
    "LEFT ANTI JOIN silver_merchants m ON t.merchant_id = m.merchant_id"
)
results.append(zero_count_check("no_orphaned_merchants", "integrity", orphaned_merchants, "bronze_transactions -> silver_merchants"))

# COMMAND ----------
# MAGIC %md #### Domain & range checks (§8.4)

# COMMAND ----------
bad_amount_or_currency = scalar(
    "SELECT COUNT(*) FROM silver_transactions WHERE amount <= 0 OR currency <> 'USD'"
)
results.append(zero_count_check("silver_amount_currency_domain", "domain", bad_amount_or_currency, "silver_transactions amount/currency"))

bad_probability = scalar(
    "SELECT COUNT(*) FROM gold_fraud_predictions WHERE fraud_probability NOT BETWEEN 0 AND 1"
)
results.append(zero_count_check("prediction_probability_domain", "domain", bad_probability, "gold_fraud_predictions.fraud_probability"))

bad_calendar_fields = scalar(
    "SELECT COUNT(*) FROM gold_fraud_features "
    "WHERE hour_of_day NOT BETWEEN 0 AND 23 OR day_of_week NOT BETWEEN 1 AND 7"
)
results.append(zero_count_check("feature_calendar_domain", "domain", bad_calendar_fields, "gold_fraud_features hour_of_day/day_of_week"))

# COMMAND ----------
# MAGIC %md #### Freshness (§8.6)
# MAGIC Measured against `gold_fraud_predictions.scored_at`, not `bronze_transactions._ingest_ts` --
# MAGIC the sample data generator uses a fixed seed and writes to the same filenames every run
# MAGIC (§4.1), so Auto Loader never reprocesses already-seen files and `_ingest_ts` freezes at
# MAGIC the first-ever run. `scored_at` reflects `batch_inference`'s own execution time on every
# MAGIC run (it always overwrites), which is what "is this pipeline actually running" should mean.

# COMMAND ----------
age_minutes = scalar("SELECT TIMESTAMPDIFF(MINUTE, MAX(scored_at), current_timestamp()) FROM gold_fraud_predictions")
results.append(
    freshness_check("predictions_freshness", "freshness", age_minutes, max_freshness_minutes, "gold_fraud_predictions.scored_at")
)

# COMMAND ----------
# MAGIC %md #### Label & prediction sanity -- drift only, never a hard fail (§8.7)

# COMMAND ----------
baseline_rate = scalar("SELECT AVG(CAST(is_fraud AS DOUBLE)) FROM gold_fraud_features WHERE is_fraud IS NOT NULL") or 0.0
live_flagged_rate = scalar("SELECT AVG(CAST(predicted_fraud AS DOUBLE)) FROM gold_fraud_predictions") or 0.0
results.append(
    drift_check(
        "flagged_rate_vs_baseline",
        "drift",
        live_flagged_rate,
        baseline_rate,
        max_drift_deviation,
        "live predicted_fraud rate vs. historical labeled baseline",
    )
)

# COMMAND ----------
# MAGIC %md #### Persist results + summary

# COMMAND ----------
run_id = str(uuid.uuid4())
checked_at = datetime.now(timezone.utc)

rows = [
    (run_id, checked_at, r.check_name, r.category, r.status, r.actual_value, r.expected, r.message)
    for r in results
]
results_df = spark.createDataFrame(
    rows, schema="run_id string, checked_at timestamp, check_name string, category string, status string, "
    "actual_value double, expected string, message string"
)
results_df.write.format("delta").mode("append").saveAsTable("gold_dq_results")

counts = summarize(results)
print(f"Data quality run {run_id}: {counts['PASS']} passed, {counts['WARN']} warned, {counts['FAIL']} failed.")
for r in results:
    marker = {"PASS": "[PASS]", "WARN": "[WARN]", "FAIL": "[FAIL]"}[r.status]
    print(f"  {marker} {r.check_name} ({r.category}): {r.message}")

if enforce_quality_gate and counts["FAIL"] > 0:
    raise AssertionError(
        f"Data quality gate failed: {counts['FAIL']} check(s) failed. See gold_dq_results (run_id={run_id})."
    )
