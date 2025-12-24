"""
增量数据更新 DAG
定时从 JSON 数据源读取增量数据，更新特征和模型。
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator


def flush_redis_cache() -> None:
    """Flush Redis feature DB so API reload picks up fresh models/features."""
    import os
    import logging

    import redis

    logger = logging.getLogger(__name__)
    redis_host = os.getenv("REDIS_HOST", "redis")
    redis_port = int(os.getenv("REDIS_PORT", "6379"))
    redis_db = int(os.getenv("REDIS_FEATURE_DB", "1"))
    redis_password = os.getenv("REDIS_PASSWORD")

    try:
        client = redis.Redis(host=redis_host, port=redis_port, db=redis_db, password=redis_password)
        client.flushdb()
        logger.info("Flushed Redis cache (host=%s, port=%d, db=%d)", redis_host, redis_port, redis_db)
    except redis.RedisError as exc:  # pragma: no cover - operational safeguard
        logger.exception("Failed to flush Redis cache")
        raise RuntimeError("Failed to flush Redis cache") from exc

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
    schedule_interval="0 */4 * * *",  # 每 4 小时执行一次
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["recommend", "incremental", "data-update"],
) as dag:

    extract_incremental = BashOperator(
        task_id="extract_incremental_data",
        bash_command="python -m pipeline.extract_load",
    )

    build_features = BashOperator(
        task_id="build_features",
        bash_command="python -m pipeline.build_features",
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
        bash_command=(
            "curl -sS -X POST http://recommendation-api:8000/models/reload "
            "-H 'Content-Type: application/json' "
            "-d '{\"mode\": \"primary\"}'"
        ),
    )

    clear_cache = PythonOperator(
        task_id="clear_redis_cache",
        python_callable=flush_redis_cache,
    )

    extract_incremental >> build_features >> train_models >> update_recall >> [
        reload_api,
        clear_cache,
    ]
