"""
P0-02: 数据库连接池监控模块

功能：
1. 监控连接池状态（大小、已用、可用）
2. 检测连接泄漏
3. 记录连接池性能指标
4. 生成连接池健康报告

使用方法：
    from pipeline.connection_pool_monitor import log_pool_status, get_pool_metrics

    # 记录连接池状态
    log_pool_status(engine)

    # 获取指标
    metrics = get_pool_metrics(engine)
"""

import logging
import time
from typing import Dict, Optional
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.pool import Pool

LOGGER = logging.getLogger(__name__)


def get_pool_metrics(engine: Engine) -> Dict[str, int]:
    """
    获取连接池指标

    Returns:
        Dict with keys:
        - pool_size: 连接池大小
        - checked_in: 空闲连接数
        - checked_out: 使用中的连接数
        - overflow: 溢出连接数
        - total_connections: 总连接数
    """
    pool = engine.pool

    try:
        metrics = {
            "pool_size": pool.size(),
            "checked_in": pool.checkedin(),
            "checked_out": pool.checkedout(),
            "overflow": pool.overflow(),
            "total_connections": pool.size() + pool.overflow(),
        }
    except Exception as exc:
        LOGGER.warning(f"Failed to get pool metrics: {exc}")
        metrics = {
            "pool_size": 0,
            "checked_in": 0,
            "checked_out": 0,
            "overflow": 0,
            "total_connections": 0,
        }

    return metrics


def log_pool_status(engine: Engine, label: str = "default") -> None:
    """
    记录连接池状态到日志

    Args:
        engine: SQLAlchemy engine
        label: 标签，用于区分不同的数据库连接
    """
    metrics = get_pool_metrics(engine)

    LOGGER.info(
        "[%s] Connection Pool Status: "
        "size=%d, checked_in=%d, checked_out=%d, overflow=%d, total=%d",
        label,
        metrics["pool_size"],
        metrics["checked_in"],
        metrics["checked_out"],
        metrics["overflow"],
        metrics["total_connections"],
    )


def check_connection_leak(
    engine: Engine,
    threshold: int = 5,
    label: str = "default"
) -> bool:
    """
    检查是否存在连接泄漏

    Args:
        engine: SQLAlchemy engine
        threshold: checked_out 连接数的阈值
        label: 标签

    Returns:
        True if potential leak detected, False otherwise
    """
    metrics = get_pool_metrics(engine)

    if metrics["checked_out"] > threshold:
        LOGGER.warning(
            "[%s] Potential connection leak detected: "
            "%d connections checked out (threshold: %d)",
            label,
            metrics["checked_out"],
            threshold
        )
        return True

    return False


def get_pool_utilization(engine: Engine) -> float:
    """
    计算连接池利用率

    Returns:
        Utilization ratio (0.0 to 1.0+)
    """
    metrics = get_pool_metrics(engine)
    pool_size = metrics["pool_size"]

    if pool_size == 0:
        return 0.0

    return metrics["checked_out"] / pool_size


class PoolMonitor:
    """连接池监控器"""

    def __init__(self, engine: Engine, label: str = "default"):
        self.engine = engine
        self.label = label
        self.start_time = time.time()
        self.metrics_history = []

    def record_snapshot(self) -> Dict:
        """记录当前连接池快照"""
        metrics = get_pool_metrics(self.engine)
        metrics["timestamp"] = time.time()
        metrics["elapsed_seconds"] = time.time() - self.start_time
        metrics["utilization"] = get_pool_utilization(self.engine)

        self.metrics_history.append(metrics)
        return metrics

    def get_statistics(self) -> Dict:
        """获取统计信息"""
        if not self.metrics_history:
            return {}

        import statistics

        checked_out_values = [m["checked_out"] for m in self.metrics_history]
        utilization_values = [m["utilization"] for m in self.metrics_history]

        return {
            "total_snapshots": len(self.metrics_history),
            "avg_checked_out": statistics.mean(checked_out_values),
            "max_checked_out": max(checked_out_values),
            "avg_utilization": statistics.mean(utilization_values),
            "max_utilization": max(utilization_values),
        }

    def print_report(self) -> None:
        """打印监控报告"""
        stats = self.get_statistics()

        print(f"\n{'=' * 70}")
        print(f"Connection Pool Report [{self.label}]")
        print(f"{'=' * 70}")

        if not stats:
            print("No data collected")
            return

        print(f"Total Snapshots: {stats['total_snapshots']}")
        print(f"Average Connections Used: {stats['avg_checked_out']:.1f}")
        print(f"Max Connections Used: {stats['max_checked_out']}")
        print(f"Average Utilization: {stats['avg_utilization']:.1%}")
        print(f"Max Utilization: {stats['max_utilization']:.1%}")

        # 判断健康状态
        if stats['max_utilization'] > 0.9:
            print("\n⚠️  WARNING: Pool utilization exceeded 90%")
            print("   Consider increasing pool_size or max_overflow")
        elif stats['max_utilization'] > 0.7:
            print("\n ℹ️  INFO: Pool utilization above 70%")
            print("   Monitor for potential bottlenecks")
        else:
            print("\n✓ Pool utilization is healthy")

        print(f"{'=' * 70}\n")


def monitor_engine_with_context(engine: Engine, label: str = "default"):
    """
    使用 context manager 监控引擎

    Usage:
        with monitor_engine_with_context(engine, "business") as monitor:
            # Your database operations
            pass
        # Automatically prints report on exit
    """
    class EngineMonitorContext:
        def __init__(self, engine, label):
            self.monitor = PoolMonitor(engine, label)

        def __enter__(self):
            self.monitor.record_snapshot()  # Initial snapshot
            return self.monitor

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.monitor.record_snapshot()  # Final snapshot
            self.monitor.print_report()

    return EngineMonitorContext(engine, label)


if __name__ == "__main__":
    # Example usage
    from config.settings import load_database_configs

    logging.basicConfig(level=logging.INFO)

    configs = load_database_configs()

    # Monitor business database
    business_config = configs["business"]
    business_engine = create_engine(
        business_config.sqlalchemy_url(),
        **business_config.get_engine_kwargs()
    )

    with monitor_engine_with_context(business_engine, "business") as monitor:
        # Simulate some queries
        with business_engine.connect() as conn:
            monitor.record_snapshot()
            result = conn.execute("SELECT 1")
            monitor.record_snapshot()

    business_engine.dispose()
