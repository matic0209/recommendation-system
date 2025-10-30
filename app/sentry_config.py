"""
Sentry 集成配置模块

提供统一的 Sentry 初始化和配置，支持 FastAPI、Airflow 和离线流水线的监控。
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

import sentry_sdk
from sentry_sdk.integrations.logging import LoggingIntegration
from sentry_sdk.integrations.redis import RedisIntegration
from sentry_sdk.integrations.threading import ThreadingIntegration

LOGGER = logging.getLogger(__name__)


def init_sentry(
    *,
    service_name: str,
    enable_tracing: bool = True,
    traces_sample_rate: float = 0.1,
    profiles_sample_rate: float = 0.1,
    environment: Optional[str] = None,
) -> bool:
    """
    初始化 Sentry SDK

    Args:
        service_name: 服务名称（用于标识不同组件）
        enable_tracing: 是否启用性能追踪
        traces_sample_rate: 性能追踪采样率（0.0-1.0）
        profiles_sample_rate: 性能分析采样率（0.0-1.0）
        environment: 环境名称（production/staging/development）

    Returns:
        是否成功初始化
    """
    dsn = os.getenv("SENTRY_DSN")

    if not dsn:
        LOGGER.warning("SENTRY_DSN not set, Sentry monitoring disabled")
        return False

    # 从环境变量获取配置
    if environment is None:
        environment = os.getenv("SENTRY_ENVIRONMENT", "production")

    # 根据环境调整采样率
    if environment == "production":
        traces_sample_rate = float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", str(traces_sample_rate)))
        profiles_sample_rate = float(os.getenv("SENTRY_PROFILES_SAMPLE_RATE", str(profiles_sample_rate)))
    else:
        # 开发/测试环境使用更高采样率
        traces_sample_rate = float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", "1.0"))
        profiles_sample_rate = float(os.getenv("SENTRY_PROFILES_SAMPLE_RATE", "1.0"))

    # 配置日志集成
    logging_integration = LoggingIntegration(
        level=logging.INFO,  # Capture info and above as breadcrumbs
        event_level=logging.ERROR,  # Send errors as events
    )

    # 基础集成列表
    integrations = [
        logging_integration,
        RedisIntegration(),
        ThreadingIntegration(propagate_hub=True),
    ]

    # FastAPI 集成（仅在 API 服务中）
    if service_name in ["recommendation-api", "notification-gateway"]:
        try:
            from sentry_sdk.integrations.fastapi import FastApiIntegration
            from sentry_sdk.integrations.starlette import StarletteIntegration
            integrations.extend([
                FastApiIntegration(transaction_style="endpoint"),
                StarletteIntegration(transaction_style="endpoint"),
            ])
            LOGGER.info("Sentry FastAPI integration enabled")
        except ImportError:
            LOGGER.warning("FastAPI integration not available")

    # Airflow 集成（仅在 Airflow 组件中）
    if service_name.startswith("airflow-"):
        try:
            from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration
            integrations.append(SqlalchemyIntegration())
            LOGGER.info("Sentry SQLAlchemy integration enabled")
        except ImportError:
            LOGGER.warning("SQLAlchemy integration not available")

    try:
        sentry_sdk.init(
            dsn=dsn,
            integrations=integrations,
            traces_sample_rate=traces_sample_rate if enable_tracing else 0.0,
            profiles_sample_rate=profiles_sample_rate if enable_tracing else 0.0,
            environment=environment,
            release=os.getenv("SENTRY_RELEASE", "unknown"),
            send_default_pii=True,  # 包含用户 IP、请求头等信息
            before_send=before_send_filter,
            before_breadcrumb=before_breadcrumb_filter,
            # 设置标签
            _experiments={
                "profiles_sample_rate": profiles_sample_rate,
            },
        )

        # 设置全局标签
        sentry_sdk.set_tag("service", service_name)

        LOGGER.info(
            "Sentry initialized (service=%s, env=%s, traces_rate=%.2f, profiles_rate=%.2f)",
            service_name,
            environment,
            traces_sample_rate,
            profiles_sample_rate,
        )
        return True

    except Exception as exc:
        LOGGER.error("Failed to initialize Sentry: %s", exc)
        return False


def before_send_filter(event: Dict[str, Any], hint: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    事件过滤器：在发送到 Sentry 前处理事件

    可以用于：
    1. 过滤掉某些不需要上报的异常
    2. 清理敏感信息
    3. 添加额外上下文
    """
    # 过滤掉某些预期的异常
    if "exc_info" in hint:
        exc_type, exc_value, tb = hint["exc_info"]

        # 过滤掉健康检查相关的错误
        if "health" in str(exc_value).lower():
            return None

        # 过滤掉 HTTPException (这些是预期的业务异常)
        if exc_type.__name__ == "HTTPException":
            # 只上报 5xx 错误
            if hasattr(exc_value, "status_code") and exc_value.status_code < 500:
                return None

    # 清理敏感信息
    if "request" in event:
        request = event["request"]
        if "headers" in request:
            # 移除敏感请求头
            sensitive_headers = ["authorization", "cookie", "x-api-key"]
            for header in sensitive_headers:
                request["headers"].pop(header, None)

    return event


def before_breadcrumb_filter(crumb: Dict[str, Any], hint: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    面包屑过滤器：过滤掉噪音日志
    """
    # 过滤掉健康检查相关的日志
    if crumb.get("category") == "httplib" and crumb.get("data", {}).get("url", "").endswith("/health"):
        return None

    # 过滤掉 Prometheus 指标请求
    if crumb.get("category") == "httplib" and crumb.get("data", {}).get("url", "").endswith("/metrics"):
        return None

    return crumb


def set_user_context(user_id: Optional[int]) -> None:
    """设置用户上下文"""
    if user_id:
        sentry_sdk.set_user({"id": str(user_id)})


def set_request_context(
    request_id: str,
    endpoint: Optional[str] = None,
    dataset_id: Optional[int] = None,
    **extra_context,
) -> None:
    """
    设置请求上下文

    Args:
        request_id: 请求 ID
        endpoint: API 端点名称
        dataset_id: 数据集 ID
        **extra_context: 其他自定义上下文
    """
    with sentry_sdk.configure_scope() as scope:
        scope.set_tag("request_id", request_id)
        if endpoint:
            scope.set_tag("endpoint", endpoint)
        if dataset_id:
            scope.set_context("dataset", {"dataset_id": dataset_id})

        # 添加额外上下文
        if extra_context:
            scope.set_context("custom", extra_context)


def set_recommendation_context(
    *,
    algorithm_version: Optional[str] = None,
    variant: Optional[str] = None,
    experiment_variant: Optional[str] = None,
    degrade_reason: Optional[str] = None,
    channel_weights: Optional[Dict[str, float]] = None,
) -> None:
    """
    设置推荐系统特定的上下文

    Args:
        algorithm_version: 算法版本（MLflow run_id）
        variant: 模型变体（primary/shadow/fallback）
        experiment_variant: 实验变体
        degrade_reason: 降级原因
        channel_weights: 召回通道权重
    """
    with sentry_sdk.configure_scope() as scope:
        context = {}

        if algorithm_version:
            scope.set_tag("algorithm_version", algorithm_version)
            context["algorithm_version"] = algorithm_version

        if variant:
            scope.set_tag("model_variant", variant)
            context["variant"] = variant

        if experiment_variant:
            scope.set_tag("experiment_variant", experiment_variant)
            context["experiment_variant"] = experiment_variant

        if degrade_reason:
            scope.set_tag("degrade_reason", degrade_reason)
            context["degrade_reason"] = degrade_reason

        if channel_weights:
            context["channel_weights"] = channel_weights

        if context:
            scope.set_context("recommendation", context)


def capture_exception_with_context(
    exception: Exception,
    *,
    level: str = "error",
    fingerprint: Optional[list] = None,
    **extra_context,
) -> None:
    """
    捕获异常并添加额外上下文

    Args:
        exception: 异常对象
        level: 严重级别（fatal/error/warning/info/debug）
        fingerprint: 用于分组的指纹
        **extra_context: 额外上下文
    """
    with sentry_sdk.push_scope() as scope:
        scope.level = level

        if fingerprint:
            scope.fingerprint = fingerprint

        if extra_context:
            for key, value in extra_context.items():
                scope.set_extra(key, value)

        sentry_sdk.capture_exception(exception)


def capture_message_with_context(
    message: str,
    *,
    level: str = "info",
    **extra_context,
) -> None:
    """
    捕获消息并添加额外上下文

    Args:
        message: 消息内容
        level: 严重级别
        **extra_context: 额外上下文
    """
    with sentry_sdk.push_scope() as scope:
        scope.level = level

        if extra_context:
            for key, value in extra_context.items():
                scope.set_extra(key, value)

        sentry_sdk.capture_message(message)


def add_breadcrumb(
    message: str,
    category: str,
    level: str = "info",
    **data,
) -> None:
    """
    添加面包屑（追踪事件序列）

    Args:
        message: 消息内容
        category: 分类（api/cache/model/data 等）
        level: 级别
        **data: 附加数据
    """
    sentry_sdk.add_breadcrumb(
        category=category,
        message=message,
        level=level,
        data=data,
    )


def start_transaction(name: str, op: str) -> Any:
    """
    开始一个性能追踪事务

    Args:
        name: 事务名称
        op: 操作类型（http.server/db.query/cache.get 等）

    Returns:
        Transaction 对象，使用 with 语句管理
    """
    return sentry_sdk.start_transaction(name=name, op=op)


def start_span(op: str, description: str) -> Any:
    """
    在当前事务中开始一个 span

    Args:
        op: 操作类型
        description: 描述

    Returns:
        Span 对象，使用 with 语句管理
    """
    return sentry_sdk.start_span(op=op, description=description)
