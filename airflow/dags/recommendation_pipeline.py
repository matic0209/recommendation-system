"""
Airflow DAG：推荐系统端到端流水线。

该 DAG 包含以下步骤：
1. 抽取业务/Matomo 数据
2. 构建与清洗特征（含 Redis 同步）
3. 运行数据质量检查并导出 Prometheus 指标
4. 训练召回/排序模型并记录 MLflow
5. 构建多路召回索引
6. 基于曝光日志与 Matomo 数据生成评估报告

执行环境依赖 docker-compose 中的 recommendation-api 镜像所安装的依赖。
"""

from datetime import datetime, timedelta
import os
import sys

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

# 初始化 Sentry（如果配置了）
sys.path.insert(0, '/opt/recommend')
try:
    from app.sentry_config import init_sentry
    sentry_initialized = init_sentry(
        service_name="airflow-scheduler",
        enable_tracing=True,
        traces_sample_rate=0.5,  # Airflow 任务较少，使用更高采样率
    )
    if sentry_initialized:
        print("Sentry initialized for Airflow DAG")
except ImportError:
    print("Sentry integration not available")
    sentry_initialized = False

def task_failure_callback(context):
    """Airflow 任务失败时的回调，发送错误到 Sentry"""
    if not sentry_initialized:
        return

    try:
        import sentry_sdk
        from app.sentry_config import capture_exception_with_context

        task_instance = context.get('task_instance')
        exception = context.get('exception')

        # 设置任务上下文
        with sentry_sdk.configure_scope() as scope:
            scope.set_tag("dag_id", context.get('dag').dag_id)
            scope.set_tag("task_id", task_instance.task_id)
            scope.set_tag("execution_date", str(context.get('execution_date')))
            scope.set_context("airflow", {
                "dag_id": context.get('dag').dag_id,
                "task_id": task_instance.task_id,
                "execution_date": str(context.get('execution_date')),
                "try_number": task_instance.try_number,
            })

        if exception:
            capture_exception_with_context(
                exception,
                level="error",
                fingerprint=["airflow", context.get('dag').dag_id, task_instance.task_id],
                dag_id=context.get('dag').dag_id,
                task_id=task_instance.task_id,
                execution_date=str(context.get('execution_date')),
            )
        else:
            sentry_sdk.capture_message(
                f"Airflow task failed: {task_instance.task_id}",
                level="error",
            )
    except Exception as e:
        print(f"Failed to send error to Sentry: {e}")


DEFAULT_ARGS = {
    "owner": "recsys",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
    "on_failure_callback": task_failure_callback,
}

with DAG(
    dag_id="recommendation_pipeline",
    description="推荐系统端到端数据与模型流水线",
    default_args=DEFAULT_ARGS,
    schedule_interval="0 18 * * *",  # 每天北京时间 2 点 => 18:00 UTC 前一日
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["recommendation", "mlops"],
) as dag:
    HEAVY_PIPELINE_POOL = os.getenv("RECSYS_HEAVY_POOL", "recsys_heavy")

    extract_load = BashOperator(
        task_id="extract_load",
        bash_command=(
            "python -m pipeline.extract_load "
            "{% if dag_run and dag_run.conf.get('full_refresh') %}--full-refresh{% endif %}"
        ),
        env=None,
        pool=HEAVY_PIPELINE_POOL,
    )

    build_features = BashOperator(
        task_id="build_features",
        bash_command="python -m pipeline.build_features",
        pool=HEAVY_PIPELINE_POOL,
    )

    data_quality = BashOperator(
        task_id="data_quality",
        bash_command="python -m pipeline.data_quality_v2",
        pool=HEAVY_PIPELINE_POOL,
    )

    aggregate_matomo_events = BashOperator(
        task_id="aggregate_matomo_events",
        bash_command="python -m pipeline.aggregate_matomo_events",
        pool=HEAVY_PIPELINE_POOL,
    )

    build_training_labels = BashOperator(
        task_id="build_training_labels",
        bash_command="python -m pipeline.build_training_labels",
        pool=HEAVY_PIPELINE_POOL,
    )

    train_models = BashOperator(
        task_id="train_models",
        bash_command="python -m pipeline.train_models",
        pool=HEAVY_PIPELINE_POOL,
    )

    recall_engine = BashOperator(
        task_id="recall_engine",
        bash_command="python -m pipeline.recall_engine_v2",
        pool=HEAVY_PIPELINE_POOL,
    )

    evaluate = BashOperator(
        task_id="evaluate",
        bash_command="python -m pipeline.evaluate_v2",
        pool=HEAVY_PIPELINE_POOL,
    )

    train_channel_weights = BashOperator(
        task_id="train_channel_weights",
        bash_command="python -m pipeline.train_channel_weights",
        pool=HEAVY_PIPELINE_POOL,
    )

    reconcile_metrics = BashOperator(
        task_id="reconcile_metrics",
        bash_command="python -m scripts.reconcile_business_metrics --start {{ ds }} --end {{ ds }}",
        pool=HEAVY_PIPELINE_POOL,
    )

    # Note: image_embeddings task removed - visual features are optional and require sentence-transformers
    # Statistical image features are still included in build_features
    (
        extract_load
        >> build_features
        >> data_quality
        >> aggregate_matomo_events
        >> build_training_labels
        >> train_models
        >> recall_engine
        >> evaluate
        >> train_channel_weights
        >> reconcile_metrics
    )
