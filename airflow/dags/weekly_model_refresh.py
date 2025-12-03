"""
Airflow DAG: Weekly retraining and deployment using Matomo request-id data.
"""
from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.exceptions import AirflowSkipException
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

from config.settings import DATA_DIR

PROCESSED_DIR = DATA_DIR / "processed"

DEFAULT_ARGS = {
    "owner": "recsys",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=15),
}


def _check_minimum_data(**_) -> None:
    """Ensure there is enough tracking data before running expensive refresh."""
    import pandas as pd

    exposures_path = PROCESSED_DIR / "recommend_exposures.parquet"
    clicks_path = PROCESSED_DIR / "recommend_clicks.parquet"
    labels_path = PROCESSED_DIR / "ranking_labels_by_dataset.parquet"

    exposures = pd.read_parquet(exposures_path) if exposures_path.exists() else pd.DataFrame()
    clicks = pd.read_parquet(clicks_path) if clicks_path.exists() else pd.DataFrame()
    labels = pd.read_parquet(labels_path) if labels_path.exists() else pd.DataFrame()

    if exposures.empty:
        raise AirflowSkipException("Exposure log is empty; skipping weekly refresh.")

    if labels.empty or labels.get("label", pd.Series(dtype=int)).sum() < 5:
        raise AirflowSkipException("Insufficient positive datasets; postponing ranking training.")

    if clicks.empty:
        # warning but continue
        print("Warning: No clicks detected yet; ranking training will fall back to interactions.")


with DAG(
    dag_id="weekly_model_refresh",
    description="Weekly refresh of recommendation models using Matomo request-id data",
    default_args=DEFAULT_ARGS,
    schedule_interval="0 3 * * 1",  # Every Monday 03:00 UTC
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["recommendation", "weekly", "matomo"],
) as dag:

    aggregate_matomo_events = BashOperator(
        task_id="aggregate_matomo_events",
        bash_command="python -m pipeline.aggregate_matomo_events",
    )

    build_training_labels = BashOperator(
        task_id="build_training_labels",
        bash_command="python -m pipeline.build_training_labels",
    )

    verify_training_processed = BashOperator(
        task_id="verify_training_files",
        bash_command="python -m pipeline.verify_processed_files --group training",
    )

    verify_readiness = PythonOperator(
        task_id="verify_training_readiness",
        python_callable=_check_minimum_data,
    )

    build_features = BashOperator(
        task_id="build_features",
        bash_command="python -m pipeline.build_features",
    )

    train_models = BashOperator(
        task_id="train_models",
        bash_command="python -m pipeline.train_models",
    )

    recall_engine = BashOperator(
        task_id="recall_engine",
        bash_command="python -m pipeline.recall_engine_v2",
    )

    reload_api_models = BashOperator(
        task_id="reload_api_models",
        bash_command=(
            "curl -sS -X POST http://recommendation-api:8000/models/reload "
            "-H 'Content-Type: application/json' "
            "-d '{\"mode\": \"primary\"}'"
        ),
    )

    evaluate_tracking = BashOperator(
        task_id="evaluate_tracking",
        bash_command="python -m pipeline.evaluate_v2",
    )

    optimize_experiments = BashOperator(
        task_id="optimize_experiments",
        bash_command="python -m pipeline.variant_optimizer --experiment recommendation_detail --endpoint recommend_detail",
    )

    verify_matomo_processed = BashOperator(
        task_id="verify_matomo_files",
        bash_command="python -m pipeline.verify_processed_files --group matomo",
    )

    aggregate_matomo_events >> verify_matomo_processed >> build_training_labels >> verify_training_processed >> verify_readiness >> build_features >> train_models >> recall_engine >> reload_api_models >> evaluate_tracking >> optimize_experiments
