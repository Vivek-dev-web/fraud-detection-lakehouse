# Databricks notebook source
# MAGIC %md
# MAGIC ### Silver (Lakeflow Declarative Pipeline)
# MAGIC Cleans and validates Bronze transactions with declarative **expectations** (bad rows
# MAGIC are dropped and counted in the pipeline's data-quality metrics, not silently lost),
# MAGIC dedupes, and enriches with account/merchant dimensions.

# COMMAND ----------
import dlt
from pyspark.sql import functions as F
from pyspark.sql.window import Window


@dlt.table(comment="Deduped account dimension.")
def silver_accounts():
    return (
        dlt.read("bronze_accounts")
        .withColumn("_rn", F.row_number().over(Window.partitionBy("account_id").orderBy(F.col("_ingest_ts").desc())))
        .filter("_rn = 1")
        .drop("_rn")
    )


@dlt.table(comment="Deduped merchant dimension.")
def silver_merchants():
    return (
        dlt.read("bronze_merchants")
        .withColumn("_rn", F.row_number().over(Window.partitionBy("merchant_id").orderBy(F.col("_ingest_ts").desc())))
        .filter("_rn = 1")
        .drop("_rn")
    )


@dlt.table(comment="Cleaned, deduped, enriched transactions.")
@dlt.expect_or_drop("valid_transaction_id", "transaction_id IS NOT NULL")
@dlt.expect_or_drop("valid_account_id", "account_id IS NOT NULL")
@dlt.expect_or_drop("valid_merchant_id", "merchant_id IS NOT NULL")
@dlt.expect_or_drop("positive_amount", "amount > 0")
@dlt.expect_or_drop("valid_currency", "currency = 'USD'")
@dlt.expect("known_account", "account_name IS NOT NULL")
@dlt.expect("known_merchant", "merchant_name IS NOT NULL")
def silver_transactions():
    accounts = dlt.read("silver_accounts").select(
        "account_id",
        F.col("customer_name").alias("account_name"),
        "email",
        "phone",
        "home_country",
        "risk_segment",
    )
    merchants = dlt.read("silver_merchants").select(
        "merchant_id", F.col("merchant_name").alias("merchant_name"), "category", F.col("country").alias("merchant_country")
    )

    txns = dlt.read_stream("bronze_transactions").withColumn("transaction_ts", F.to_timestamp("transaction_ts"))

    return txns.join(F.broadcast(accounts), "account_id", "left").join(
        F.broadcast(merchants), "merchant_id", "left"
    )
