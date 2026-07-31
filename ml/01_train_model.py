# Databricks notebook source
# MAGIC %md
# MAGIC ### Train fraud classifier
# MAGIC Trains a gradient-boosted classifier on labeled historical transactions from
# MAGIC `gold_fraud_features` (produced by the Lakeflow pipeline), tracks the run with
# MAGIC MLflow, and registers the model in the Unity Catalog Model Registry with a
# MAGIC `champion` alias for the batch inference notebook to pick up.

# COMMAND ----------
# MAGIC %pip install scikit-learn==1.5.1 mlflow==2.19.0 -q
# MAGIC dbutils.library.restartPython()

# COMMAND ----------
import os

# required for UC model-artifact credential vending on this workspace/mlflow version --
# see https://github.com/mlflow/mlflow/issues (PERMISSION_DENIED generating temporary
# model-version credentials without this flag)
os.environ["MLFLOW_USE_DATABRICKS_SDK_MODEL_ARTIFACTS_REPO_FOR_UC"] = "true"

# COMMAND ----------
dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "fraud_detection")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

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
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from mlflow.models.signature import infer_signature
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------
labeled_df = spark.table("gold_fraud_features").filter("is_fraud IS NOT NULL")
pdf = labeled_df.select(*FEATURE_COLS, "is_fraud").toPandas()
pdf["is_card_present"] = pdf["is_card_present"].astype(int)
pdf["is_fraud"] = pdf["is_fraud"].astype(int)

X = pdf[FEATURE_COLS]
y = pdf["is_fraud"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training rows: {len(X_train)}, test rows: {len(X_test)}, fraud rate: {y.mean():.3f}")

# COMMAND ----------
with mlflow.start_run(run_name="fraud_classifier_gbt") as run:
    params = {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05, "random_state": 42}
    mlflow.log_params(params)

    model = GradientBoostingClassifier(**params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "average_precision": average_precision_score(y_test, y_proba),
    }
    mlflow.log_metrics(metrics)
    print(metrics)

    signature = infer_signature(X_train, model.predict_proba(X_train))
    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        signature=signature,
        input_example=X_train.head(5),
        registered_model_name=MODEL_NAME,
    )

    run_id = run.info.run_id

# COMMAND ----------
client = MlflowClient()
latest_version = max(int(v.version) for v in client.search_model_versions(f"name='{MODEL_NAME}'"))
client.set_registered_model_alias(MODEL_NAME, "champion", latest_version)

print(f"Registered {MODEL_NAME} version {latest_version} and set alias 'champion' (run_id={run_id}).")
