# Fraud Detection Lakehouse (Databricks)

An end-to-end fraud-detection project on Databricks covering a wider slice of the
platform than a plain ETL pipeline: a **declarative Lakeflow (Delta Live Tables)
pipeline**, **MLflow model training + Unity Catalog Model Registry**, **batch
inference**, **automated data quality checks** with a real pass/fail gate,
**Unity Catalog governance** (column masks, row filters, tags), a **Lakeview
dashboard + SQL alert**, and **CI/CD** via GitHub Actions.

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
                                        data quality gate -> governance (masks/row filters/tags) -> dashboard + alert
```

| Domain | Financial transactions -- credit-card fraud detection |
|---|---|
| Bronze | `bronze_accounts`, `bronze_merchants` (batch dims), `bronze_transactions` (Auto Loader, labeled historical + unlabeled streaming batches unioned) |
| Silver | `silver_accounts`, `silver_merchants` (deduped), `silver_transactions` (Lakeflow **expectations**: valid ids, positive amount, valid currency; enriched with account/merchant dims) |
| Gold | `gold_fraud_features` (feature engineering: velocity, z-score, foreign-txn flag), `gold_daily_fraud_summary`, `gold_account_risk_scores` (business aggregates) |
| ML | `gold_fraud_predictions` (batch-scored via a Unity Catalog-registered MLflow model) |
| DQ | `gold_dq_results` (row-count, integrity, domain, freshness, and drift checks -- one row per check per job run) |

## What this project demonstrates

- **Lakeflow Declarative Pipelines** (`transforms/`): `@dlt.table`, `@dlt.expect_or_drop` /
  `@dlt.expect` data-quality expectations, streaming (`dlt.read_stream`, Auto Loader) vs.
  full-refresh materialized (`dlt.read`) tables in the same pipeline.
- **MLflow + Unity Catalog Model Registry** (`ml/`): a run tracked with params/metrics/model
  artifact, registered as `<catalog>.<schema>.fraud_classifier`, promoted via a `champion`
  alias; batch inference loads the model with `mlflow.pyfunc.spark_udf`.
- **Automated data quality checks with a real gate** (`governance/04_data_quality_checks.py`):
  row-count reconciliation, null/duplicate/orphan checks, domain/range checks, and freshness
  are hard gates that fail the job on violation; live flagged-rate vs. historical baseline
  drift is a WARN-only signal that can never fail the job. Every check's result is written to
  `gold_dq_results` and shown on the dashboard's Data Quality page.
- **Unity Catalog governance** (`governance/`): column masking functions for PII
  (`customer_name`, `email`, `phone`), a row filter restricting `HIGH` risk-segment rows,
  classification tags -- all gated on `is_account_group_member('admins')` -- and an access
  boundary granting `SELECT` on the five gold tables to `account users` while bronze/silver
  stay ungranted (Unity Catalog is secure-by-default, so no explicit revoke is needed).
- **Lakeview dashboard + SQL alert** (`dashboard/deploy_dashboard.py`): a Fraud Overview page
  (flagged-transaction counter, fraud-by-category bar chart, fraud-rate trend line) and a Data
  Quality page (open-issues counter, per-check results table), plus an alert that fires when
  the latest day's fraud rate exceeds 5%.
- **CI/CD** (`.github/workflows/ci.yml`, `azure-pipelines.yml`): pytest unit tests +
  `databricks bundle validate` on every push/PR, on both GitHub Actions and Azure Pipelines.
- **Unit-testable business logic** (`src/features.py`, `src/data_quality.py`, `tests/`): the
  feature-engineering rules (z-score, velocity, risk score) and the data-quality pass/fail/warn
  logic exist as plain-Python functions with edge-case tests, independent of Spark.

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

This runs 6 tasks in order: `generate_sample_data` -> `run_pipeline` (triggers the Lakeflow
pipeline: bronze -> silver -> gold) -> `train_model` -> `batch_inference` ->
`data_quality_checks` -> `apply_governance`. If a hard-gate check fails, the job stops before
`apply_governance` runs -- set the `enforce_quality_gate` task parameter to `false` to log
without blocking.

Then deploy the dashboard + alert (not a bundle resource -- Lakeview/Alerts APIs are handled
via a standalone script using the Databricks SDK):

```powershell
pip install -r requirements.txt
python dashboard/deploy_dashboard.py --profile <name> --catalog workspace --schema fraud_detection --warehouse-id <sql-warehouse-id>
```

Find a warehouse id with `databricks warehouses list --profile <name>`.

If the published dashboard ever shows "No data" / "Visualization has no fields selected" on
every widget at once, that's not a data or permissions problem -- confirmed live: the
underlying tables have rows, the exact widget queries return correct results when run
directly, and the SQL warehouse is healthy with `num_active_sessions: 0` -- meaning Lakeview
never even opens a session to run the query. It's a client-side health-check retry loop
(`/api/2.0/popproxy/health/...` requests repeatedly self-cancelling with `net::ERR_ABORTED`)
that prevents the query from ever firing, independent of the dashboard definition. Try an
incognito window or a different browser first. `dashboard/demo_backup_queries.sql` has the
exact query behind every widget, verified working, for running live in the SQL Editor as a
fallback.

## Local development

```powershell
pip install -r requirements.txt
pytest tests/ -v
```

## CI/CD setup

`.github/workflows/ci.yml` runs automatically on GitHub -- no setup beyond the
`DATABRICKS_HOST`/`DATABRICKS_TOKEN` repo secrets already being present.

`azure-pipelines.yml` needs a one-time manual hookup before it runs anything, since the code
lives on GitHub, not in Azure Repos:

1. In the [Azure DevOps project](https://dev.azure.com/vivekjpr/VivekTiwari_Project1), go to
   **Pipelines -> New pipeline -> GitHub**, authorize access, and select this repo.
2. Choose **Existing Azure Pipelines YAML file** and point it at `/azure-pipelines.yml`.
3. Before the first run, add two pipeline variables (**Edit -> Variables**):
   `DATABRICKS_HOST` (plain) and `DATABRICKS_TOKEN` (check **Keep this value secret**).
4. For the `Deploy` stage's approval gate: **Pipelines -> Environments -> New environment**,
   name it `databricks-dev`, skip the resource picker. Then on that environment's page,
   **... -> Approvals and checks -> Approvals**, and add yourself as approver. Without this
   step the Deploy stage will fail waiting on an environment that doesn't exist yet.
5. Repeat step 4 for the `DeployProd` stage, but name the environment `databricks-prod` --
   it needs its own independent approval gate so approving a dev deploy never implicitly
   approves prod.

The `prod` bundle target writes to `workspace.fraud_detection_prod` in the same Free Edition
workspace as `dev` (there's no second Databricks workspace here) -- its schema and
`raw_landing`/`checkpoints` volumes need to exist before the first `DeployProd` run:

```bash
databricks schemas create fraud_detection_prod workspace --profile <name>
databricks volumes create workspace fraud_detection_prod raw_landing MANAGED --profile <name>
databricks volumes create workspace fraud_detection_prod checkpoints MANAGED --profile <name>
```

Free tier covers all of this comfortably: Boards and the first 5 users are free on the Basic
plan, and Azure Pipelines gives 1,800 free minutes/month on Microsoft-hosted agents -- a full
`bundle deploy` + `bundle run` takes roughly 10-15 minutes, so even running both Deploy and
DeployProd on every merge stays well inside the free tier for realistic usage.

## Notes / caveats

- **Single-admin workspaces won't visibly see masking/row-filter effects** -- the governance
  functions check `is_account_group_member('admins')`, so the workspace owner always sees
  unmasked data. Add a non-admin user/group to see the restriction apply.
- **Gold-layer access grants target `account users`, not a custom group** -- Free Edition has
  no account-console access to create one. Verified: granting to a freshly created
  workspace-local group fails with `PRINCIPAL_DOES_NOT_EXIST`, since UC grants require
  account-level identities. Swap `account users` for a real customer-facing account group on
  a paid tier.
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
