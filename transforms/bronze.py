# Databricks notebook source
# MAGIC %md
# MAGIC ### Bronze (Lakeflow Declarative Pipeline)
# MAGIC Raw ingestion, minimal transformation. Dimensions are small reference dumps
# MAGIC (batch, full-refresh materialized). Transactions (labeled historical + unlabeled
# MAGIC streaming batches share one schema) are ingested incrementally via Auto Loader.

# COMMAND ----------
import dlt
from pyspark.sql.functions import col, current_timestamp
from pyspark.sql.types import BooleanType, DoubleType, StringType, StructField, StructType

RAW_ROOT = spark.conf.get("raw_root")

TRANSACTION_SCHEMA = StructType(
    [
        StructField("transaction_id", StringType()),
        StructField("account_id", StringType()),
        StructField("merchant_id", StringType()),
        StructField("amount", DoubleType()),
        StructField("currency", StringType()),
        StructField("transaction_ts", StringType()),
        StructField("is_card_present", BooleanType()),
        StructField("device_id", StringType()),
        StructField("txn_country", StringType()),
        StructField("is_fraud", BooleanType()),
    ]
)


@dlt.table(comment="Raw account dimension, batch-refreshed each pipeline run.")
def bronze_accounts():
    return spark.read.json(f"{RAW_ROOT}/accounts").withColumn("_ingest_ts", current_timestamp())


@dlt.table(comment="Raw merchant dimension, batch-refreshed each pipeline run.")
def bronze_merchants():
    return spark.read.json(f"{RAW_ROOT}/merchants").withColumn("_ingest_ts", current_timestamp())


@dlt.table(comment="Raw transactions (labeled historical + unlabeled stream), ingested incrementally via Auto Loader.")
def bronze_transactions():
    # _metadata is a hidden per-source column that doesn't survive a union, so it must be
    # extracted into a regular column on each stream individually before combining them.
    labeled = (
        spark.readStream.format("cloudFiles")
        .option("cloudFiles.format", "json")
        .schema(TRANSACTION_SCHEMA)
        .load(f"{RAW_ROOT}/transactions_labeled")
        .withColumn("_source_file", col("_metadata.file_path"))
    )
    stream = (
        spark.readStream.format("cloudFiles")
        .option("cloudFiles.format", "json")
        .schema(TRANSACTION_SCHEMA)
        .load(f"{RAW_ROOT}/transactions_stream")
        .withColumn("_source_file", col("_metadata.file_path"))
    )
    return labeled.unionByName(stream).withColumn("_ingest_ts", current_timestamp())
