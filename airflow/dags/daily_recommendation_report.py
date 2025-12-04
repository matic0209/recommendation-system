"""Airflow DAG to generate the daily recommendation monitoring report."""
from __future__ import annotations

import pendulum
from airflow import DAG
from airflow.operators.bash import BashOperator

with DAG(
    dag_id="daily_recommendation_report",
    schedule_interval="0 20 * * *",  # 每天北京时间 04:00 => 20:00 UTC 前一日
    start_date=pendulum.datetime(2025, 10, 1, tz="UTC"),
    catchup=True,
    max_active_runs=1,
    tags=["recommendation", "monitoring"],
) as dag:
    generate_report = BashOperator(
        task_id="generate_daily_report",
        bash_command=(
            "cd /opt/recommend && "
            "MATOMO_DB_HOST={{ var.value.get('MATOMO_DB_HOST', 'matomo') }} "
            "python -m pipeline.daily_report --date {{ ds }}"
        ),
    )

    generate_report
