# Databricks notebook source
# MAGIC %md
# MAGIC ### Unity Catalog governance
# MAGIC Applies column masks (PII redaction), row filters (high-risk account restriction),
# MAGIC and classification tags on top of the tables the Lakeflow pipeline produces. All
# MAGIC checks key off `is_account_group_member('admins')` -- on a single-admin workspace
# MAGIC that means the mechanism is correctly wired even though the current user won't
# MAGIC visibly see anything masked; add a non-admin user/group to see it take effect.
# MAGIC
# MAGIC Note: a Lakeflow **full refresh** of a table can reset masks/row filters applied
# MAGIC outside the pipeline definition -- rerun this notebook after a full refresh.

# COMMAND ----------
dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "fraud_detection")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

FQ = f"{catalog}.{schema}"


def run_sql(sql, description):
    try:
        spark.sql(sql)
        print(f"[OK]      {description}")
    except Exception as e:  # noqa: BLE001 -- best-effort; some features may be unsupported on this workspace/DBR
        print(f"[SKIPPED] {description} -- {e}")


# COMMAND ----------
# MAGIC %md #### Masking + row-filter functions

# COMMAND ----------
run_sql(
    f"""
    CREATE OR REPLACE FUNCTION {FQ}.mask_pii(value STRING)
    RETURNS STRING
    RETURN CASE WHEN is_account_group_member('admins') THEN value ELSE '***MASKED***' END
    """,
    "create mask_pii function",
)

run_sql(
    f"""
    CREATE OR REPLACE FUNCTION {FQ}.row_filter_high_risk(risk_segment STRING)
    RETURNS BOOLEAN
    RETURN is_account_group_member('admins') OR risk_segment != 'HIGH'
    """,
    "create row_filter_high_risk function",
)

# COMMAND ----------
# MAGIC %md #### Column masks on PII fields

# COMMAND ----------
for table, columns in [
    ("silver_accounts", ["customer_name", "email", "phone"]),
    ("silver_transactions", ["account_name", "email", "phone"]),
]:
    for col in columns:
        run_sql(
            f"ALTER TABLE {FQ}.{table} ALTER COLUMN {col} SET MASK {FQ}.mask_pii",
            f"mask {table}.{col}",
        )

# COMMAND ----------
# MAGIC %md #### Row filters restricting HIGH risk_segment rows to admins

# COMMAND ----------
for table in ["silver_transactions", "gold_fraud_predictions", "gold_account_risk_scores"]:
    run_sql(
        f"ALTER TABLE {FQ}.{table} SET ROW FILTER {FQ}.row_filter_high_risk ON (risk_segment)",
        f"row filter on {table}",
    )

# COMMAND ----------
# MAGIC %md #### Classification tags

# COMMAND ----------
run_sql(f"ALTER TABLE {FQ}.silver_accounts SET TAGS ('data_classification' = 'pii')", "tag silver_accounts")
run_sql(f"ALTER TABLE {FQ}.silver_transactions SET TAGS ('data_classification' = 'pii')", "tag silver_transactions")

for table, col in [
    ("silver_accounts", "customer_name"),
    ("silver_accounts", "email"),
    ("silver_accounts", "phone"),
    ("silver_transactions", "email"),
    ("silver_transactions", "phone"),
]:
    run_sql(f"ALTER TABLE {FQ}.{table} ALTER COLUMN {col} SET TAGS ('pii' = 'true')", f"tag {table}.{col}")

print("Governance setup complete.")
