# Fraud Detection Lakehouse (Databricks)

An end-to-end fraud-detection project on Databricks covering a wider slice of the
platform than a plain ETL pipeline: a **declarative Lakeflow (Delta Live Tables)
pipeline**, **MLflow model training + Unity Catalog Model Registry**, **batch
inference**, **Unity Catalog governance** (column masks, row filters, tags), a
**Lakeview dashboard + SQL alert**, and **CI/CD** via GitHub Actions.

## Architecture

```
                 ┌─────────────────────── Lakeflow Declarative Pipeline ───────────────────────┐
raw txn/account/ │  BRONZE              SILVER                   GOLD                          │
merchant files   │  Auto Loader   ->    expectations,       ->   feature engineering +          │
(UC volume)       │  ingestion          dedupe, enrich            business aggregates            │
                 └────────────────────────────────────────────────────────────────────────────┘
                                                                        │
                                                                        v
                                                      ┌─────────────────────────────────┐
                                                      │  ML: train (MLflow) -> register  │
                                                      │  in UC Model Registry (champion) │
                                                      │  -> batch inference               │
                                                      └─────────────────────────────────┘
                                                                        │
                                                                        v
                                        governance (masks/row filters/tags) -> dashboard + alert
```

| Domain | Financial transactions -- credit-card fraud detection |
|---|---|
| Bronze | `bronze_accounts`, `bronze_merchants` (batch dims), `bronze_transactions` (Auto Loader, labeled historical + unlabeled streaming batches unioned) |
| Silver | `silver_accounts`, `silver_merchants` (deduped), `silver_transactions` (Lakeflow **expectations**: valid ids, positive amount, valid currency; enriched with account/merchant dims) |
| Gold | `gold_fraud_features` (feature engineering: velocity, z-score, foreign-txn flag), `gold_daily_fraud_summary`, `gold_account_risk_scores` (business aggregates) |
| ML | `gold_fraud_predictions` (batch-scored via a Unity Catalog-registered MLflow model) |

## What this project demonstrates

- **Lakeflow Declarative Pipelines** (`transforms/`): `@dlt.table`, `@dlt.expect_or_drop` /
  `@dlt.expect` data-quality expectations, streaming (`dlt.read_stream`, Auto Loader) vs.
  full-refresh materialized (`dlt.read`) tables in the same pipeline.
- **MLflow + Unity Catalog Model Registry** (`ml/`): a run tracked with params/metrics/model
  artifact, registered as `<catalog>.<schema>.fraud_classifier`, promoted via a `champion`
  alias; batch inference loads the model with `mlflow.pyfunc.spark_udf`.
- **Unity Catalog governance** (`governance/`): column masking functions for PII
  (`customer_name`, `email`, `phone`), a row filter restricting `HIGH` risk-segment rows,
  and classification tags -- all gated on `is_account_group_member('admins')`.
- **Lakeview dashboard + SQL alert** (`dashboard/deploy_dashboard.py`): 3-widget dashboard
  (flagged-transaction counter, fraud-by-category bar chart, fraud-rate trend line) plus an
  alert that fires when the latest day's fraud rate exceeds 5%.
- **CI/CD** (`.github/workflows/ci.yml`): pytest unit tests + `databricks bundle validate`
  on every push/PR.
- **Unit-testable business logic** (`src/features.py`, `tests/`): the feature-engineering
  rules (z-score, velocity, risk score) exist as plain-Python functions with edge-case tests
  (zero-stddev, empty history, score saturation), independent of Spark.

## Deploy

Requires the [Databricks CLI](https://docs.databricks.com/dev-tools/cli/index.html), an auth
profile, and a schema + `raw_landing` volume already created:

```powershell
databricks schemas create fraud_detection workspace --profile <name>
databricks volumes create workspace fraud_detection raw_landing MANAGED --profile <name>

databricks bundle validate --profile <name>
databricks bundle deploy --profile <name>
databricks bundle run fraud_detection_job --profile <name>
```

This runs 5 tasks in order: `generate_sample_data` -> `run_pipeline` (triggers the Lakeflow
pipeline: bronze -> silver -> gold) -> `train_model` -> `batch_inference` -> `apply_governance`.

Then deploy the dashboard + alert (not a bundle resource -- Lakeview/Alerts APIs are handled
via a standalone script using the Databricks SDK):

```powershell
pip install -r requirements.txt
python dashboard/deploy_dashboard.py --profile <name> --catalog workspace --schema fraud_detection --warehouse-id <sql-warehouse-id>
```

Find a warehouse id with `databricks warehouses list --profile <name>`.

## Local development

```powershell
pip install -r requirements.txt
pytest tests/ -v
```

## Notes / caveats

- **Single-admin workspaces won't visibly see masking/row-filter effects** -- the governance
  functions check `is_account_group_member('admins')`, so the workspace owner always sees
  unmasked data. Add a non-admin user/group to see the restriction apply.
- **Column masks/row filters only take on `STREAMING_TABLE`s and plain managed tables, not
  `MATERIALIZED_VIEW`s** -- verified on a real deployment: `silver_transactions` (streaming
  table) and `gold_fraud_predictions` (plain managed table) both took the mask/row filter
  fine; `silver_accounts` and `gold_account_risk_scores` (materialized views) silently
  didn't. `governance/03_apply_governance.py` already wraps every statement in try/except and
  logs `[OK]`/`[SKIPPED]`, so it won't fail the job -- but PII in `silver_accounts` isn't
  actually masked. If that matters, either change `silver_accounts`/`silver_merchants` from
  `dlt.table` to a streaming table, or apply the mask to a plain managed table built on top.
- **Lakeflow full refreshes can reset masks/row filters** applied outside the pipeline
  definition -- rerun `governance/03_apply_governance.py` after a full pipeline refresh.
- The Lakeview dashboard JSON schema is not yet a stable public contract; if
  `deploy_dashboard.py` errors on widget/dataset shape, check the
  [Lakeview API docs](https://docs.databricks.com/api/workspace/lakeview) for the current
  schema.
- `fraud_threshold` (batch inference flagging cutoff) defaults to `0.5`; override via the job
  task's `base_parameters` in `databricks.yml` or the Databricks UI.
