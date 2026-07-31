# Databricks notebook source
# MAGIC %md
# MAGIC ### Gold (Lakeflow Declarative Pipeline)
# MAGIC Feature engineering for the fraud model (`gold_fraud_features`) plus business-facing
# MAGIC aggregates (`gold_daily_fraud_summary`, `gold_account_risk_scores`) for the dashboard.
# MAGIC These are full-refresh materialized views (`dlt.read`, not `dlt.read_stream`) since the
# MAGIC windowed/account-level aggregations need to see the whole table each run.

# COMMAND ----------
import dlt
from pyspark.sql import functions as F
from pyspark.sql.window import Window

HIGH_RISK_COUNTRIES = ["NG", "RU"]


@dlt.table(comment="Feature-engineered transactions: the model's training/inference input.")
def gold_fraud_features():
    txns = dlt.read("silver_transactions")

    velocity_window = (
        Window.partitionBy("account_id").orderBy(F.col("transaction_ts").cast("long")).rangeBetween(-600, 0)
    )
    account_window = Window.partitionBy("account_id")

    return (
        txns.withColumn("is_foreign_txn", (F.col("txn_country") != F.col("home_country")).cast("int"))
        .withColumn("is_high_risk_country", F.col("txn_country").isin(HIGH_RISK_COUNTRIES).cast("int"))
        .withColumn("hour_of_day", F.hour("transaction_ts"))
        .withColumn("day_of_week", F.dayofweek("transaction_ts"))
        .withColumn("txn_count_last_10min", F.count("transaction_id").over(velocity_window))
        .withColumn("account_avg_amount", F.avg("amount").over(account_window))
        .withColumn("account_stddev_amount", F.coalesce(F.stddev("amount").over(account_window), F.lit(1.0)))
        .withColumn(
            "amount_zscore",
            (F.col("amount") - F.col("account_avg_amount")) / F.when(F.col("account_stddev_amount") == 0, 1.0).otherwise(F.col("account_stddev_amount")),
        )
        .select(
            "transaction_id",
            "account_id",
            "merchant_id",
            "amount",
            "transaction_ts",
            "is_card_present",
            "txn_country",
            "home_country",
            "risk_segment",
            "category",
            "is_foreign_txn",
            "is_high_risk_country",
            "hour_of_day",
            "day_of_week",
            "txn_count_last_10min",
            "amount_zscore",
            "is_fraud",
        )
    )


@dlt.table(comment="Daily fraud rate by merchant category and country, computed from labeled history.")
def gold_daily_fraud_summary():
    return (
        dlt.read("gold_fraud_features")
        .filter(F.col("is_fraud").isNotNull())
        .withColumn("txn_date", F.to_date("transaction_ts"))
        .groupBy("txn_date", "category", "txn_country")
        .agg(
            F.count("transaction_id").alias("txn_count"),
            F.sum(F.col("is_fraud").cast("int")).alias("fraud_count"),
            F.round(F.sum("amount"), 2).alias("total_amount"),
            F.round(F.sum(F.when(F.col("is_fraud"), F.col("amount")).otherwise(0.0)), 2).alias("fraud_amount"),
        )
        .withColumn("fraud_rate", F.round(F.col("fraud_count") / F.col("txn_count"), 4))
    )


@dlt.table(comment="Per-account risk rollup from labeled history.")
def gold_account_risk_scores():
    features = dlt.read("gold_fraud_features").filter(F.col("is_fraud").isNotNull())
    accounts = dlt.read("silver_accounts")

    agg = features.groupBy("account_id").agg(
        F.count("transaction_id").alias("txn_count"),
        F.sum(F.col("is_fraud").cast("int")).alias("fraud_count"),
        F.round(F.avg("amount"), 2).alias("avg_amount"),
        F.round(F.avg("is_foreign_txn"), 3).alias("foreign_txn_rate"),
        F.round(F.max("amount"), 2).alias("max_amount"),
    )

    return agg.withColumn("fraud_rate", F.round(F.col("fraud_count") / F.col("txn_count"), 4)).join(
        accounts.select("account_id", "risk_segment", "home_country"), "account_id", "left"
    )
