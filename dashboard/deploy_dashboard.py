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
            {
                "name": "ds_dq_latest",
                "displayName": "Latest Data Quality Run",
                "query": (
                    f"SELECT * FROM {fq}.gold_dq_results "
                    f"WHERE run_id = (SELECT run_id FROM {fq}.gold_dq_results ORDER BY checked_at DESC LIMIT 1)"
                ),
            },
        ],
        "pages": [
            {
                "name": "main_v2",
                "displayName": "Fraud Overview",
                "layout": [
                    {
                        "widget": {
                            "name": "widget_flagged_count_v2",
                            "queries": [
                                {
                                    "name": "q1v2",
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
                            "name": "widget_fraud_by_category_v2",
                            "queries": [
                                {
                                    "name": "q2v2",
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
                            "name": "widget_fraud_rate_trend_v2",
                            "queries": [
                                {
                                    "name": "q3v2",
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
                "pageType": "PAGE_TYPE_CANVAS",
            },
            {
                "name": "data_quality",
                "displayName": "Data Quality",
                "layout": [
                    {
                        "widget": {
                            "name": "widget_dq_issues_count",
                            "queries": [
                                {
                                    "name": "q4",
                                    "query": {
                                        "datasetName": "ds_dq_latest",
                                        "fields": [
                                            {"name": "issues_found", "expression": "COUNT_IF(`status` <> 'PASS')"}
                                        ],
                                        "disaggregated": False,
                                    },
                                }
                            ],
                            "spec": {
                                "version": 2,
                                "widgetType": "counter",
                                "encodings": {
                                    "value": {"fieldName": "issues_found", "displayName": "Open Data Quality Issues"}
                                },
                            },
                        },
                        "position": {"x": 0, "y": 0, "width": 2, "height": 4},
                    },
                    {
                        "widget": {
                            "name": "widget_dq_results_table",
                            "queries": [
                                {
                                    "name": "q5",
                                    "query": {
                                        "datasetName": "ds_dq_latest",
                                        "fields": [
                                            {"name": "check_name", "expression": "`check_name`"},
                                            {"name": "category", "expression": "`category`"},
                                            {"name": "status", "expression": "`status`"},
                                            {"name": "actual_value", "expression": "`actual_value`"},
                                            {"name": "message", "expression": "`message`"},
                                        ],
                                        "disaggregated": True,
                                    },
                                }
                            ],
                            "spec": {
                                "version": 1,
                                "widgetType": "table",
                                "encodings": {
                                    "columns": [
                                        {"fieldName": "check_name", "displayName": "Check", "type": "string"},
                                        {"fieldName": "category", "displayName": "Category", "type": "string"},
                                        {"fieldName": "status", "displayName": "Status", "type": "string"},
                                        {"fieldName": "actual_value", "displayName": "Value", "type": "float"},
                                        {"fieldName": "message", "displayName": "Details", "type": "string"},
                                    ]
                                },
                            },
                        },
                        "position": {"x": 2, "y": 0, "width": 4, "height": 6},
                    },
                ],
                "pageType": "PAGE_TYPE_CANVAS",
            },
        ],
    }


def deploy_dashboard(w: WorkspaceClient, catalog: str, schema: str, warehouse_id: str, display_name: str) -> str:
    dashboard_json = build_dashboard_json(catalog, schema)

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

    # embed_credentials=True runs queries as the publisher, not the viewer -- with False,
    # every viewer's own live session has to execute the query against the warehouse, which
    # silently failed for a session opened via an auto-login link (queries that worked fine
    # via the API returned real data; the published page showed "No data" / "no fields
    # selected" instead of an error). Fine to embed on a single-admin workspace; revisit if
    # this is ever shared with a viewer who shouldn't see unmasked/unfiltered rows.
    w.api_client.do(
        "POST",
        f"/api/2.0/lakeview/dashboards/{dashboard_id}/published",
        body={"embed_credentials": True, "warehouse_id": warehouse_id},
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
            alert_id=existing_alert.id,
            name=alert_name,
            options=options,
            query_id=query.id,
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
    parser.add_argument(
        "--display-name",
        default="Fraud Detection Overview",
        help="Dashboard display name -- lookup key for update-in-place. Pass a new name to "
        "create a fresh dashboard object instead of patching an existing one.",
    )
    args = parser.parse_args()

    client = WorkspaceClient(profile=args.profile)
    deploy_dashboard(client, args.catalog, args.schema, args.warehouse_id, args.display_name)
    deploy_alert(client, args.catalog, args.schema, args.warehouse_id)
