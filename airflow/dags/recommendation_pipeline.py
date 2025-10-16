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

from airflow import DAG
from airflow.operators.bash import BashOperator

DEFAULT_ARGS = {
    "owner": "recsys",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}

with DAG(
    dag_id="recommendation_pipeline",
    description="推荐系统端到端数据与模型流水线",
    default_args=DEFAULT_ARGS,
    schedule_interval="0 2 * * *",  # 每天凌晨 2 点执行
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["recommendation", "mlops"],
) as dag:

    extract_load = BashOperator(
        task_id="extract_load",
        bash_command="python -m pipeline.extract_load",
        env=None,
    )

    build_features = BashOperator(
        task_id="build_features",
        bash_command="python -m pipeline.build_features",
    )

    data_quality = BashOperator(
        task_id="data_quality",
        bash_command="python -m pipeline.data_quality_v2",
    )

    train_models = BashOperator(
        task_id="train_models",
        bash_command="python -m pipeline.train_models",
    )

    recall_engine = BashOperator(
        task_id="recall_engine",
        bash_command="python -m pipeline.recall_engine_v2",
    )

    evaluate = BashOperator(
        task_id="evaluate",
        bash_command="python -m pipeline.evaluate",
    )

    reconcile_metrics = BashOperator(
        task_id="reconcile_metrics",
        bash_command="python -m scripts.reconcile_business_metrics --start {{ ds }} --end {{ ds }}",
    )

    # Note: image_embeddings task removed - visual features are optional and require sentence-transformers
    # Statistical image features are still included in build_features
    extract_load >> build_features >> data_quality >> train_models >> recall_engine >> evaluate >> reconcile_metrics
