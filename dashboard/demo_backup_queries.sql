-- Backup queries for live demos, in case the Lakeview dashboard widgets are unavailable
-- (see docs/DESIGN.md §4.9 build note -- a client-side health-check loop on this workspace
-- can prevent Lakeview from ever firing its query, independent of the data or dashboard
-- definition, both confirmed correct). Run these directly in the SQL Editor against the
-- Serverless Starter Warehouse -- each one is the exact query behind one dashboard widget.

-- ============================================================
-- Fraud Overview page
-- ============================================================

-- Widget: Flagged Transactions (counter)
SELECT COUNT_IF(`predicted_fraud`) AS flagged_transactions
FROM workspace.fraud_detection.gold_fraud_predictions;

-- Widget: Fraud by Category (bar chart)
SELECT `category`, SUM(`fraud_count`) AS fraud_count
FROM workspace.fraud_detection.gold_daily_fraud_summary
GROUP BY `category`
ORDER BY fraud_count DESC;

-- Widget: Fraud Rate Trend (line chart)
SELECT `txn_date`, AVG(`fraud_rate`) AS fraud_rate
FROM workspace.fraud_detection.gold_daily_fraud_summary
GROUP BY `txn_date`
ORDER BY txn_date;

-- ============================================================
-- Data Quality page
-- ============================================================

-- Widget: Open Data Quality Issues (counter)
SELECT COUNT_IF(`status` <> 'PASS') AS issues_found
FROM workspace.fraud_detection.gold_dq_results
WHERE run_id = (
    SELECT run_id FROM workspace.fraud_detection.gold_dq_results ORDER BY checked_at DESC LIMIT 1
);

-- Widget: Data Quality Results (table)
SELECT check_name, category, status, actual_value, message
FROM workspace.fraud_detection.gold_dq_results
WHERE run_id = (
    SELECT run_id FROM workspace.fraud_detection.gold_dq_results ORDER BY checked_at DESC LIMIT 1
)
ORDER BY status DESC, category;

-- ============================================================
-- Supporting talking points (not on the dashboard, good context if asked)
-- ============================================================

-- Model status: confirms the champion alias resolves to a READY model version
-- databricks model-versions list workspace.fraud_detection.fraud_classifier --profile <name>

-- Full check history across every job run, not just the latest -- shows the gate has been
-- exercised repeatedly, not just passed once
SELECT run_id, checked_at, COUNT_IF(status = 'PASS') AS passed,
       COUNT_IF(status = 'WARN') AS warned, COUNT_IF(status = 'FAIL') AS failed
FROM workspace.fraud_detection.gold_dq_results
GROUP BY run_id, checked_at
ORDER BY checked_at DESC;
