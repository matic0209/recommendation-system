"""Sentry utilities for offline pipeline monitoring."""
from __future__ import annotations

import functools
import logging
import time
from typing import Any, Callable, Optional

LOGGER = logging.getLogger(__name__)


def get_sentry_funcs():
    """获取 Sentry 函数（延迟导入）"""
    try:
        from app.sentry_config import (
            capture_exception_with_context,
            capture_message_with_context,
            add_breadcrumb,
        )
        return capture_exception_with_context, capture_message_with_context, add_breadcrumb
    except ImportError:
        return None, None, None


def monitor_pipeline_step(step_name: str, critical: bool = False):
    """
    装饰器：监控离线流水线步骤

    Args:
        step_name: 步骤名称
        critical: 是否为关键步骤（关键步骤失败会发送错误级别告警）

    Usage:
        @monitor_pipeline_step("extract_load", critical=True)
        def extract_and_load_data():
            # 数据抽取和加载逻辑
            pass
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            capture_exc, capture_msg, add_bc = get_sentry_funcs()

            # 记录开始
            if add_bc:
                add_bc(
                    message=f"Pipeline step started: {step_name}",
                    category="pipeline",
                    level="info",
                    step_name=step_name,
                    critical=critical,
                )

            start_time = time.time()

            try:
                LOGGER.info("Starting pipeline step: %s", step_name)
                result = func(*args, **kwargs)
                duration = time.time() - start_time

                LOGGER.info(
                    "Pipeline step completed: %s (duration: %.2fs)",
                    step_name,
                    duration,
                )

                # 记录成功
                if add_bc:
                    add_bc(
                        message=f"Pipeline step completed: {step_name}",
                        category="pipeline",
                        level="info",
                        step_name=step_name,
                        duration_seconds=duration,
                    )

                return result

            except Exception as exc:
                duration = time.time() - start_time
                error_level = "error" if critical else "warning"

                LOGGER.error(
                    "Pipeline step failed: %s (duration: %.2fs, error: %s)",
                    step_name,
                    duration,
                    exc,
                )

                # 发送到 Sentry
                if capture_exc:
                    capture_exc(
                        exc,
                        level=error_level,
                        fingerprint=["pipeline", step_name, type(exc).__name__],
                        step_name=step_name,
                        critical=critical,
                        duration_seconds=duration,
                    )

                # 重新抛出异常
                raise

        return wrapper

    return decorator


def track_data_quality_issue(
    check_name: str,
    severity: str,
    details: dict,
    metric_value: Optional[float] = None,
    threshold: Optional[float] = None,
) -> None:
    """
    追踪数据质量问题

    Args:
        check_name: 检查名称（如 "missing_values_check"）
        severity: 严重程度（critical/warning/info）
        details: 问题详情
        metric_value: 指标值
        threshold: 阈值

    Example:
        track_data_quality_issue(
            check_name="missing_price_check",
            severity="warning",
            details={"missing_count": 150, "total_count": 10000},
            metric_value=0.015,
            threshold=0.01,
        )
    """
    capture_exc, capture_msg, add_bc = get_sentry_funcs()

    if not capture_msg:
        return

    # 构造消息
    message = f"Data quality issue: {check_name}"
    if metric_value is not None and threshold is not None:
        message += f" (value: {metric_value:.4f}, threshold: {threshold:.4f})"

    # 发送到 Sentry
    level_map = {
        "critical": "error",
        "warning": "warning",
        "info": "info",
    }
    sentry_level = level_map.get(severity, "warning")

    capture_msg(
        message,
        level=sentry_level,
        check_name=check_name,
        severity=severity,
        metric_value=metric_value,
        threshold=threshold,
        **details,
    )

    LOGGER.warning(
        "Data quality issue: %s (severity=%s, details=%s)",
        check_name,
        severity,
        details,
    )


def track_model_training_issue(
    model_name: str,
    issue_type: str,
    details: dict,
    metrics: Optional[dict] = None,
) -> None:
    """
    追踪模型训练问题

    Args:
        model_name: 模型名称（如 "lightgbm_ranker"）
        issue_type: 问题类型（如 "low_performance", "overfitting", "training_failed"）
        details: 问题详情
        metrics: 模型指标

    Example:
        track_model_training_issue(
            model_name="lightgbm_ranker",
            issue_type="low_performance",
            details={"reason": "NDCG@10 below threshold"},
            metrics={"ndcg_at_10": 0.45, "threshold": 0.50},
        )
    """
    capture_exc, capture_msg, add_bc = get_sentry_funcs()

    if not capture_msg:
        return

    message = f"Model training issue: {model_name} - {issue_type}"

    # 确定严重级别
    severity = "error" if issue_type == "training_failed" else "warning"

    capture_msg(
        message,
        level=severity,
        model_name=model_name,
        issue_type=issue_type,
        metrics=metrics,
        **details,
    )

    LOGGER.warning(
        "Model training issue: %s (type=%s, details=%s, metrics=%s)",
        model_name,
        issue_type,
        details,
        metrics,
    )


def track_feature_store_sync_issue(
    operation: str,
    error: Exception,
    affected_count: int,
    total_count: int,
) -> None:
    """
    追踪特征存储同步问题

    Args:
        operation: 操作类型（如 "redis_sync", "sqlite_write"）
        error: 异常对象
        affected_count: 受影响的记录数
        total_count: 总记录数

    Example:
        track_feature_store_sync_issue(
            operation="redis_sync",
            error=redis_error,
            affected_count=1500,
            total_count=10000,
        )
    """
    capture_exc, capture_msg, add_bc = get_sentry_funcs()

    if not capture_exc:
        return

    failure_rate = affected_count / total_count if total_count > 0 else 0

    capture_exc(
        error,
        level="error",
        fingerprint=["feature_store", operation, type(error).__name__],
        operation=operation,
        affected_count=affected_count,
        total_count=total_count,
        failure_rate=failure_rate,
    )

    LOGGER.error(
        "Feature store sync issue: %s (affected=%d/%d, rate=%.2f%%)",
        operation,
        affected_count,
        total_count,
        failure_rate * 100,
    )


def track_database_issue(
    operation: str,
    database: str,
    error: Exception,
    query_info: Optional[dict] = None,
) -> None:
    """
    追踪数据库问题

    Args:
        operation: 操作类型（如 "extract", "load", "query"）
        database: 数据库名称（如 "business_db", "matomo_db"）
        error: 异常对象
        query_info: 查询信息（可选）

    Example:
        track_database_issue(
            operation="extract",
            database="business_db",
            error=connection_error,
            query_info={"table": "datasets", "limit": 10000},
        )
    """
    capture_exc, capture_msg, add_bc = get_sentry_funcs()

    if not capture_exc:
        return

    capture_exc(
        error,
        level="error",
        fingerprint=["database", database, operation, type(error).__name__],
        operation=operation,
        database=database,
        query_info=query_info or {},
    )

    LOGGER.error(
        "Database issue: %s on %s (operation=%s, info=%s)",
        type(error).__name__,
        database,
        operation,
        query_info,
    )


__all__ = [
    "monitor_pipeline_step",
    "track_data_quality_issue",
    "track_model_training_issue",
    "track_feature_store_sync_issue",
    "track_database_issue",
]
