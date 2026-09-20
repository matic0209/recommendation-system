"""
Airflow DAG：推荐系统端到端流水线。

该 DAG 包含以下步骤：
1. 抽取业务/Matomo 数据
2. 构建与清洗特征（含 Redis 同步）
3. 标签增强（零样本分类，与质量检查并行）
4. 运行数据质量检查并导出 Prometheus 指标
5. 训练召回/排序模型并记录 MLflow
6. 构建多路召回索引（使用增强后的标签）
7. 基于曝光日志与 Matomo 数据生成评估报告
8. 检测模型变化并按需重启服务（含健康检查和回滚）

DAG 结构（enhance_tags 与主流程并行）：
                              ┌─→ enhance_tags ───────────────────────────────────────────┐
                              │                                                            ↓
extract_load → build_features ─┤                                                      recall_engine → evaluate → ... → restart_if_changed
                              │                                                            ↑
                              └─→ data_quality → aggregate → labels → train_models ───────┘

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

    # 标签增强任务：使用零样本分类为新数据集添加标准化类别标签
    # --missing-only: 只处理没有标签的新数据集（增量模式）
    # 与主流程并行运行，在 recall_engine 之前汇合
    enhance_tags = BashOperator(
        task_id="enhance_tags",
        bash_command="python -m pipeline.enhance_tags --missing-only",
        pool=HEAVY_PIPELINE_POOL,
        # 即使失败也不阻塞后续任务（标签增强是可选增强，不是核心功能）
        trigger_rule="all_done",
    )

    recall_engine = BashOperator(
        task_id="recall_engine",
        bash_command="python -m pipeline.recall_engine_v2",
        pool=HEAVY_PIPELINE_POOL,
        # 等待 train_models 和 enhance_tags 都完成（无论成功或失败）
        trigger_rule="none_failed_min_one_success",
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

    # 服务重启任务：检测模型是否变化，变化则重启服务并验证健康状态
    # 只在模型文件hash变化时才会实际重启，否则跳过
    #
    # ⚠️ 坑（2026-09-16 实测定位）：bash_command 若以 ".sh" 结尾，Airflow 会把它当作
    # 「模板文件路径」交给 Jinja loader 去加载（BashOperator.template_ext = ('.sh', '.bash')），
    # 每次解析 DAG 都会报：
    #     TemplateNotFound: bash /opt/recommend/scripts/restart_service.sh
    # 导致该任务永远无法正常执行 —— 于是索引/权重即使更新了，推荐服务也不会重新加载。
    # 修复：命令以 `; exit $?` 收尾（既不触发模板文件加载，又完整保留脚本的退出码，
    #       脚本失败时任务照样会失败，不会被静默吞掉）。
    restart_if_changed = BashOperator(
        task_id="restart_if_changed",
        bash_command="bash /opt/recommend/scripts/restart_service.sh; exit $?",
        pool=HEAVY_PIPELINE_POOL,
    )

    # Note: image_embeddings task removed - visual features are optional and require sentence-transformers
    # Statistical image features are still included in build_features
    #
    # DAG 结构：enhance_tags 与主流程并行运行
    #                               ┌─→ enhance_tags ───────────────────────────────────────────┐
    #                               │                                                            ↓
    # extract_load → build_features ─┤                                                      recall_engine → ...
    #                               │                                                            ↑
    #                               └─→ data_quality → aggregate → labels → train_models ───────┘

    # 主流程入口
    extract_load >> build_features

    # 分支1：标签增强（与主流程并行）
    build_features >> enhance_tags >> recall_engine

    # 分支2：主流程（数据质量检查 → 训练）
    (
        build_features
        >> data_quality
        >> aggregate_matomo_events
        >> build_training_labels
        >> train_models
        >> recall_engine
    )

    # 后续流程
    (
        recall_engine
        >> evaluate
        >> train_channel_weights
        >> reconcile_metrics
        >> restart_if_changed
    )
