"""
增量数据更新 DAG
定时从 JSON 数据源读取增量数据，更新特征和模型。
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

DEFAULT_ARGS = {
    "owner": "recommend-system",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="incremental_data_update",
    description="增量更新推荐系统数据和模型",
    default_args=DEFAULT_ARGS,
    schedule_interval="0 * * * *",  # 每小时第 0 分钟执行
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["recommend", "incremental", "data-update"],
) as dag:

    extract_incremental = BashOperator(
        task_id="extract_incremental_data",
        bash_command="python -m pipeline.extract_load",
    )

    clean_data = BashOperator(
        task_id="clean_data",
        bash_command="python -m pipeline.data_cleaner",
    )

    build_features = BashOperator(
        task_id="build_features",
        bash_command="python -m pipeline.build_features_v2",
    )

    train_models = BashOperator(
        task_id="train_models",
        bash_command="python -m pipeline.train_models",
    )

    update_recall = BashOperator(
        task_id="update_recall_engine",
        bash_command="python -m pipeline.recall_engine_v2",
    )

    reload_api = BashOperator(
        task_id="reload_api_models",
        bash_command="curl -X POST http://recommendation-api:8000/models/reload",
    )

    clear_cache = BashOperator(
        task_id="clear_redis_cache",
        bash_command="redis-cli -h redis -p 6379 FLUSHDB",
    )

    extract_incremental >> clean_data >> build_features >> train_models >> update_recall >> [
        reload_api,
        clear_cache,
    ]
