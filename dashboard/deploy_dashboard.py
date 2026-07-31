"""Creates a Lakeview dashboard + a SQL alert on top of the Gold fraud tables.

Run locally (not on a Databricks cluster) against a configured CLI profile:

    python dashboard/deploy_dashboard.py --profile medallion \
        --catalog workspace --schema fraud_detection --warehouse-id <id>

Uses the Databricks SDK's low-level `api_client.do()` so this isn't pinned to a
specific typed-model SDK version -- the Lakeview/Alerts APIs are still evolving.
"""
import argparse
import json

from databricks.sdk import WorkspaceClient


def build_dashboard_json(catalog: str, schema: str) -> dict:
    fq = f"{catalog}.{schema}"
    return {
        "datasets": [
            {
                "name": "ds_daily_summary",
                "displayName": "Daily Fraud Summary",
                "query": f"SELECT * FROM {fq}.gold_daily_fraud_summary",
            },
            {
                "name": "ds_predictions",
                "displayName": "Fraud Predictions",
                "query": f"SELECT * FROM {fq}.gold_fraud_predictions",
            },
        ],
        "pages": [
            {
                "name": "main",
                "displayName": "Fraud Overview",
                "layout": [
                    {
                        "widget": {
                            "name": "widget_flagged_count",
                            "queries": [
                                {
                                    "name": "q1",
                                    "query": {
                                        "datasetName": "ds_predictions",
                                        "fields": [
                                            {"name": "flagged_count", "expression": "COUNT_IF(`predicted_fraud`)"}
                                        ],
                                        "disaggregated": False,
                                    },
                                }
                            ],
                            "spec": {
                                "version": 2,
                                "widgetType": "counter",
                                "encodings": {
                                    "value": {"fieldName": "flagged_count", "displayName": "Flagged Transactions"}
                                },
                            },
                        },
                        "position": {"x": 0, "y": 0, "width": 2, "height": 4},
                    },
                    {
                        "widget": {
                            "name": "widget_fraud_by_category",
                            "queries": [
                                {
                                    "name": "q2",
                                    "query": {
                                        "datasetName": "ds_daily_summary",
                                        "fields": [
                                            {"name": "category", "expression": "`category`"},
                                            {"name": "fraud_count", "expression": "SUM(`fraud_count`)"},
                                        ],
                                        "disaggregated": False,
                                    },
                                }
                            ],
                            "spec": {
                                "version": 2,
                                "widgetType": "bar",
                                "encodings": {
                                    "x": {"fieldName": "category", "displayName": "Category", "scale": {"type": "categorical"}},
                                    "y": {"fieldName": "fraud_count", "displayName": "Fraud Count", "scale": {"type": "quantitative"}},
                                },
                            },
                        },
                        "position": {"x": 2, "y": 0, "width": 4, "height": 4},
                    },
                    {
                        "widget": {
                            "name": "widget_fraud_rate_trend",
                            "queries": [
                                {
                                    "name": "q3",
                                    "query": {
                                        "datasetName": "ds_daily_summary",
                                        "fields": [
                                            {"name": "txn_date", "expression": "`txn_date`"},
                                            {"name": "fraud_rate", "expression": "AVG(`fraud_rate`)"},
                                        ],
                                        "disaggregated": False,
                                    },
                                }
                            ],
                            "spec": {
                                "version": 2,
                                "widgetType": "line",
                                "encodings": {
                                    "x": {"fieldName": "txn_date", "displayName": "Date", "scale": {"type": "temporal"}},
                                    "y": {"fieldName": "fraud_rate", "displayName": "Fraud Rate", "scale": {"type": "quantitative"}},
                                },
                            },
                        },
                        "position": {"x": 0, "y": 4, "width": 6, "height": 4},
                    },
                ],
            }
        ],
    }


def deploy_dashboard(w: WorkspaceClient, catalog: str, schema: str, warehouse_id: str) -> str:
    dashboard_json = build_dashboard_json(catalog, schema)
    display_name = "Fraud Detection Overview"

    existing = [d for d in w.lakeview.list() if d.display_name == display_name]
    body = {
        "display_name": display_name,
        "serialized_dashboard": json.dumps(dashboard_json),
        "warehouse_id": warehouse_id,
    }

    if existing:
        dashboard_id = existing[0].dashboard_id
        w.api_client.do("PATCH", f"/api/2.0/lakeview/dashboards/{dashboard_id}", body=body)
        print(f"Updated existing dashboard {dashboard_id}")
    else:
        resp = w.api_client.do("POST", "/api/2.0/lakeview/dashboards", body=body)
        dashboard_id = resp["dashboard_id"]
        print(f"Created dashboard {dashboard_id}")

    w.api_client.do(
        "POST",
        f"/api/2.0/lakeview/dashboards/{dashboard_id}/published",
        body={"embed_credentials": False, "warehouse_id": warehouse_id},
    )
    print(f"Published dashboard: {w.config.host}/dashboardsv3/{dashboard_id}/published")
    return dashboard_id


def deploy_alert(w: WorkspaceClient, catalog: str, schema: str, warehouse_id: str) -> None:
    """Uses the legacy typed Alerts/Queries SDK API (name, options, query_id) -- this SDK
    version doesn't expose the newer query_text-based Alerts v2 API, and guessing its raw
    REST body shape ("Invalid request: Missing alert definition") wasn't worth chasing.
    """
    from databricks.sdk.service import sql as sdk_sql

    fq = f"{catalog}.{schema}"
    query_name = "Fraud Detection - Latest Daily Fraud Rate"
    alert_name = "High daily fraud rate"
    query_text = (
        f"SELECT SUM(fraud_count) / SUM(txn_count) AS fraud_rate "
        f"FROM {fq}.gold_daily_fraud_summary "
        f"WHERE txn_date = (SELECT MAX(txn_date) FROM {fq}.gold_daily_fraud_summary)"
    )

    data_source_id = next(
        (ds.id for ds in w.data_sources.list() if ds.warehouse_id == warehouse_id), None
    )
    if not data_source_id:
        print(f"[SKIPPED] alert creation -- no data source found for warehouse_id={warehouse_id}")
        return

    existing_query = next((q for q in w.queries.list() if q.name == query_name), None)
    if existing_query:
        query = w.queries.update(existing_query.id, query=query_text)
    else:
        query = w.queries.create(
            data_source_id=data_source_id,
            name=query_name,
            query=query_text,
        )
    print(f"Query ready: {query.id}")

    existing_alert = next((a for a in w.alerts.list() if a.name == alert_name), None)
    options = sdk_sql.AlertOptions(column="fraud_rate", op=">", value=0.05)
    if existing_alert:
        w.alerts.update(
            id=existing_alert.id,
            name=alert_name,
            options=options,
            query_id=query.id,
            update_mask="name,options,query_id",
        )
        print(f"Updated existing alert {existing_alert.id}")
    else:
        alert = w.alerts.create(name=alert_name, options=options, query_id=query.id)
        print(f"Created alert {alert.id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--catalog", default="workspace")
    parser.add_argument("--schema", default="fraud_detection")
    parser.add_argument("--warehouse-id", required=True)
    args = parser.parse_args()

    client = WorkspaceClient(profile=args.profile)
    deploy_dashboard(client, args.catalog, args.schema, args.warehouse_id)
    deploy_alert(client, args.catalog, args.schema, args.warehouse_id)
