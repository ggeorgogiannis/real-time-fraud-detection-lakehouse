from datetime import timedelta

import pendulum
from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import DAG

with DAG(
    dag_id="fraud_lakehouse_daily",
    description="Run the daily lakehouse pipeline and refresh analytical views.",
    schedule="0 2 * * *",
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    default_args={
        "owner": "fraud-lakehouse",
        "retries": 2,
        "retry_delay": timedelta(minutes=5),
    },
    tags=["fraud-detection", "batch", "lakehouse"],
) as dag:
    run_batch_pipeline = BashOperator(
        task_id="run_batch_pipeline",
        bash_command="""
set -euo pipefail

fraud-lakehouse run \
    --raw-dir /opt/airflow/data/raw \
    --output-dir /opt/airflow/data \
    --ingested-at-utc "{{ (data_interval_end | default(dag_run.run_after, true)).isoformat() }}" \
    --log-level INFO
""",
        execution_timeout=timedelta(hours=2),
        do_xcom_push=False,
    )

    build_analytics_database = BashOperator(
        task_id="build_analytics_database",
        bash_command="""
set -euo pipefail

fraud-lakehouse build-analytics \
    --silver-dir data/silver \
    --gold-dir data/gold \
    --database-path data/analytics/fraud_lakehouse.duckdb \
    --log-level INFO
""",
        cwd="/opt/airflow",
        execution_timeout=timedelta(minutes=30),
        do_xcom_push=False,
    )

    run_batch_pipeline >> build_analytics_database
