# Databricks notebook source
# MAGIC %md
# MAGIC ### Batch inference
# MAGIC Scores the unlabeled (live/streaming) rows in `gold_fraud_features` using the
# MAGIC `champion`-aliased model from the Unity Catalog Model Registry, writing results to
# MAGIC `gold_fraud_predictions`. Runs as a plain batch job task (not part of the Lakeflow
# MAGIC pipeline) so it can depend on the training task having registered a model first.

# COMMAND ----------
# MAGIC %pip install scikit-learn==1.5.1 mlflow==2.19.0 -q
# MAGIC dbutils.library.restartPython()

# COMMAND ----------
import os

os.environ["MLFLOW_USE_DATABRICKS_SDK_MODEL_ARTIFACTS_REPO_FOR_UC"] = "true"

# COMMAND ----------
dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "fraud_detection")
dbutils.widgets.text("fraud_threshold", "0.5")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
fraud_threshold = float(dbutils.widgets.get("fraud_threshold"))

spark.sql(f"USE CATALOG {catalog}")
spark.sql(f"USE SCHEMA {schema}")

MODEL_NAME = f"{catalog}.{schema}.fraud_classifier"

FEATURE_COLS = [
    "amount",
    "is_card_present",
    "is_foreign_txn",
    "is_high_risk_country",
    "hour_of_day",
    "day_of_week",
    "txn_count_last_10min",
    "amount_zscore",
]

# COMMAND ----------
# mlflow.pyfunc.spark_udf() delegates to Databricks' UDF-sandbox environment detection,
# which chokes on this workspace's serverless runtime version string
# ("18.x-aarch64-photon-scala2" isn't valid PEP 440 for `packaging.version.Version`).
# Loading the sklearn model directly and scoring with a plain pandas_udf sidesteps that
# sandboxing path entirely -- fine for a model this small.
import mlflow
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import ArrayType, DoubleType

mlflow.set_registry_uri("databricks-uc")

model_uri = f"models:/{MODEL_NAME}@champion"
sk_model = mlflow.sklearn.load_model(model_uri)
# no spark.sparkContext.broadcast() -- SparkContext isn't reachable on serverless compute.
# The model is small, so closing over it directly (cloudpickled with the UDF) is fine.


@pandas_udf(ArrayType(DoubleType()))
def predict_proba_udf(*cols: pd.Series) -> pd.Series:
    # sklearn's _validate_data enforces feature *names* match training, not just position --
    # pd.concat on bare positional Series gives generic "_0.._N" names, so relabel explicitly.
    X = pd.concat(cols, axis=1)
    X.columns = FEATURE_COLS
    proba = sk_model.predict_proba(X)
    return pd.Series(list(proba))


# COMMAND ----------
to_score = (
    spark.table("gold_fraud_features")
    .filter(F.col("is_fraud").isNull())
    .withColumn("is_card_present_int", F.col("is_card_present").cast("int"))
)

feature_cols_for_model = [c if c != "is_card_present" else "is_card_present_int" for c in FEATURE_COLS]

scored = to_score.withColumn("_proba", predict_proba_udf(*feature_cols_for_model)).withColumn(
    "fraud_probability", F.round(F.col("_proba")[1], 4)
)

predictions = scored.withColumn(
    "predicted_fraud", F.col("fraud_probability") >= F.lit(fraud_threshold)
).withColumn("scored_at", F.current_timestamp()).select(
    "transaction_id",
    "account_id",
    "merchant_id",
    "amount",
    "transaction_ts",
    "txn_country",
    "home_country",
    "risk_segment",
    "category",
    "is_foreign_txn",
    "is_high_risk_country",
    "txn_count_last_10min",
    "amount_zscore",
    "fraud_probability",
    "predicted_fraud",
    "scored_at",
)

predictions.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    "gold_fraud_predictions"
)

flagged = predictions.filter("predicted_fraud").count()
total = predictions.count()
print(f"Scored {total} transactions, flagged {flagged} as fraud ({flagged / total:.3%}) at threshold {fraud_threshold}.")
display(predictions.orderBy(F.col("fraud_probability").desc()).limit(20))
