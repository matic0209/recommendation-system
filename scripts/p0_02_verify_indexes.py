#!/usr/bin/env python3
"""
P0-01: 索引验证和性能测试脚本

功能：
1. 验证索引是否正确创建
2. 测试查询性能提升
3. 生成索引使用报告
4. 监控索引效率

使用方法：
    python scripts/p0_02_verify_indexes.py --test-queries
    python scripts/p0_02_verify_indexes.py --monitor
"""

import argparse
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from sqlalchemy import create_engine, text

from config.settings import load_database_configs, DATA_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
LOGGER = logging.getLogger(__name__)

# 期望的索引列表
EXPECTED_INDEXES = {
    "business": {
        "user": ["idx_update_time", "idx_id_update_time"],
        "dataset": ["idx_update_time", "idx_create_time", "idx_status_update_time"],
        "task": ["idx_create_time", "idx_update_time", "idx_user_create_time", "idx_dataset_create_time"],
        "api_order": ["idx_create_time", "idx_update_time"],
        "dataset_image": ["idx_update_time", "idx_create_time"],
    },
    "matomo": {
        "matomo_log_visit": ["idx_visit_last_action_time", "idx_server_time", "idx_visitor_time"],
        "matomo_log_link_visit_action": ["idx_server_time", "idx_visit_time"],
        "matomo_log_conversion": ["idx_server_time", "idx_visit_conversion_time"],
    }
}


def get_existing_indexes(engine, table_name: str) -> List[Dict]:
    """获取表的现有索引"""
    query = text("""
        SELECT
            INDEX_NAME,
            GROUP_CONCAT(COLUMN_NAME ORDER BY SEQ_IN_INDEX) AS columns,
            INDEX_TYPE,
            NON_UNIQUE,
            CARDINALITY
        FROM information_schema.STATISTICS
        WHERE TABLE_SCHEMA = DATABASE()
          AND TABLE_NAME = :table_name
        GROUP BY INDEX_NAME, INDEX_TYPE, NON_UNIQUE, CARDINALITY
        ORDER BY INDEX_NAME
    """)

    with engine.connect() as conn:
        result = conn.execute(query, {"table_name": table_name})
        return [dict(row._mapping) for row in result]


def verify_indexes(configs: Dict) -> Dict[str, Dict]:
    """验证所有索引是否正确创建"""
    LOGGER.info("Starting index verification...")
    results = {}

    for source, expected_tables in EXPECTED_INDEXES.items():
        config = configs[source]
        engine = create_engine(config.sqlalchemy_url(), **config.get_engine_kwargs())
        results[source] = {}

        LOGGER.info(f"Checking {source} database...")

        for table, expected_indexes in expected_tables.items():
            existing = get_existing_indexes(engine, table)
            existing_names = {idx['INDEX_NAME'] for idx in existing}

            # 检查每个期望的索引
            table_result = {
                "expected": expected_indexes,
                "found": [],
                "missing": [],
                "details": existing
            }

            for idx_name in expected_indexes:
                if idx_name in existing_names:
                    table_result["found"].append(idx_name)
                    LOGGER.info(f"  ✓ {table}.{idx_name} - OK")
                else:
                    table_result["missing"].append(idx_name)
                    LOGGER.warning(f"  ✗ {table}.{idx_name} - MISSING")

            results[source][table] = table_result

        engine.dispose()

    return results


def test_query_performance(configs: Dict) -> Dict[str, List[Dict]]:
    """测试查询性能（执行 EXPLAIN）"""
    LOGGER.info("Testing query performance...")

    cutoff_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d %H:%M:%S")
    cutoff_timestamp = int(time.time()) - 7 * 86400

    # 测试查询列表
    test_queries = {
        "business": [
            ("user", f"SELECT * FROM user WHERE update_time > '{cutoff_date}' ORDER BY update_time LIMIT 1000"),
            ("dataset", f"SELECT * FROM dataset WHERE update_time > '{cutoff_date}' ORDER BY update_time LIMIT 1000"),
            ("task", f"SELECT * FROM task WHERE create_time > '{cutoff_date}' ORDER BY create_time LIMIT 1000"),
            ("api_order", f"SELECT * FROM api_order WHERE create_time > '{cutoff_date}' ORDER BY create_time LIMIT 1000"),
        ],
        "matomo": [
            ("matomo_log_visit", f"SELECT * FROM matomo_log_visit WHERE visit_last_action_time > {cutoff_timestamp} ORDER BY visit_last_action_time LIMIT 1000"),
            ("matomo_log_link_visit_action", f"SELECT * FROM matomo_log_link_visit_action WHERE server_time > {cutoff_timestamp} ORDER BY server_time LIMIT 1000"),
        ]
    }

    results = {}

    for source, queries in test_queries.items():
        config = configs[source]
        engine = create_engine(config.sqlalchemy_url(), **config.get_engine_kwargs())
        results[source] = []

        LOGGER.info(f"Testing {source} queries...")

        for table, query in queries:
            explain_query = f"EXPLAIN {query}"

            with engine.connect() as conn:
                start = time.time()
                explain_result = conn.execute(text(explain_query))
                duration = time.time() - start

                explain_rows = [dict(row._mapping) for row in explain_result]

                # 检查是否使用了索引
                uses_index = any(
                    row.get('key') and row.get('key') != 'NULL' and 'idx_' in str(row.get('key'))
                    for row in explain_rows
                )

                result = {
                    "table": table,
                    "query": query[:100] + "...",
                    "duration_ms": round(duration * 1000, 2),
                    "uses_index": uses_index,
                    "explain": explain_rows
                }

                results[source].append(result)

                status = "✓ USING INDEX" if uses_index else "✗ NOT USING INDEX"
                LOGGER.info(f"  {table}: {status} ({result['duration_ms']}ms)")

        engine.dispose()

    return results


def benchmark_extraction(configs: Dict) -> Dict[str, float]:
    """基准测试：实际执行抽取查询并测量时间"""
    LOGGER.info("Running extraction benchmark...")

    cutoff_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")

    benchmarks = {}

    # 业务库测试
    config = configs["business"]
    engine = create_engine(config.sqlalchemy_url(), **config.get_engine_kwargs())

    tables = ["user", "dataset", "task"]
    for table in tables:
        query = f"SELECT * FROM {table} WHERE update_time > '{cutoff_date}' OR create_time > '{cutoff_date}' LIMIT 10000"

        start = time.time()
        with engine.connect() as conn:
            result = conn.execute(text(query))
            df = pd.DataFrame(result.fetchall())
        duration = time.time() - start

        benchmarks[f"business.{table}"] = {
            "rows": len(df),
            "duration_seconds": round(duration, 3)
        }
        LOGGER.info(f"  {table}: {len(df)} rows in {duration:.3f}s")

    engine.dispose()

    # Matomo 库测试
    config = configs["matomo"]
    engine = create_engine(config.sqlalchemy_url(), **config.get_engine_kwargs())

    cutoff_ts = int(time.time()) - 86400
    query = f"SELECT * FROM matomo_log_visit WHERE visit_last_action_time > {cutoff_ts} LIMIT 10000"

    start = time.time()
    with engine.connect() as conn:
        result = conn.execute(text(query))
        df = pd.DataFrame(result.fetchall())
    duration = time.time() - start

    benchmarks["matomo.matomo_log_visit"] = {
        "rows": len(df),
        "duration_seconds": round(duration, 3)
    }
    LOGGER.info(f"  matomo_log_visit: {len(df)} rows in {duration:.3f}s")

    engine.dispose()

    return benchmarks


def monitor_index_usage(configs: Dict, duration_hours: int = 24) -> Dict:
    """监控索引使用情况（需要 performance_schema 启用）"""
    LOGGER.info(f"Monitoring index usage over {duration_hours} hours...")

    # 注意：这需要 MySQL 的 performance_schema 启用
    query = text("""
        SELECT
            TABLE_NAME,
            INDEX_NAME,
            COUNT_STAR as total_accesses,
            SUM_TIMER_WAIT/1000000000000 as total_time_sec,
            COUNT_READ as read_count,
            COUNT_WRITE as write_count,
            COUNT_FETCH as fetch_count,
            SUM_TIMER_FETCH/1000000000000 as fetch_time_sec
        FROM performance_schema.table_io_waits_summary_by_index_usage
        WHERE OBJECT_SCHEMA = DATABASE()
          AND INDEX_NAME LIKE 'idx_%'
        ORDER BY total_accesses DESC
    """)

    results = {}

    for source in ["business", "matomo"]:
        config = configs[source]
        engine = create_engine(config.sqlalchemy_url(), **config.get_engine_kwargs())

        try:
            with engine.connect() as conn:
                result = conn.execute(query)
                usage_data = [dict(row._mapping) for row in result]
                results[source] = usage_data

                LOGGER.info(f"Found {len(usage_data)} index usage records for {source}")
        except Exception as exc:
            LOGGER.warning(f"Could not query performance_schema for {source}: {exc}")
            LOGGER.warning("Make sure performance_schema is enabled in MySQL")
            results[source] = []

        engine.dispose()

    return results


def generate_report(
    verification: Dict,
    performance: Dict,
    benchmarks: Dict,
    usage: Dict
) -> None:
    """生成完整的验证报告"""
    report_dir = DATA_DIR / "evaluation"
    report_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = report_dir / f"index_verification_{timestamp}.json"

    report = {
        "generated_at": datetime.now().isoformat(),
        "verification": verification,
        "performance_tests": performance,
        "extraction_benchmarks": benchmarks,
        "index_usage": usage,
        "summary": {
            "total_expected": sum(
                len(tables) for tables in EXPECTED_INDEXES.values()
            ),
            "total_found": sum(
                len(table_result["found"])
                for source_result in verification.values()
                for table_result in source_result.values()
            ),
            "total_missing": sum(
                len(table_result["missing"])
                for source_result in verification.values()
                for table_result in source_result.values()
            ),
        }
    }

    # 保存 JSON 报告
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    LOGGER.info(f"Report saved to: {report_path}")

    # 打印摘要
    print("\n" + "=" * 80)
    print("INDEX VERIFICATION SUMMARY")
    print("=" * 80)

    summary = report["summary"]
    print(f"Total Expected Indexes: {summary['total_expected']}")
    print(f"Total Found: {summary['total_found']}")
    print(f"Total Missing: {summary['total_missing']}")

    if summary['total_missing'] == 0:
        print("\n✓ All indexes are properly created!")
    else:
        print(f"\n✗ {summary['total_missing']} indexes are missing. Please check the report.")

    print("\n" + "=" * 80)
    print("PERFORMANCE TEST RESULTS")
    print("=" * 80)

    for source, tests in performance.items():
        print(f"\n{source.upper()}:")
        for test in tests:
            status = "✓" if test["uses_index"] else "✗"
            print(f"  {status} {test['table']}: {test['duration_ms']}ms")

    print("\n" + "=" * 80)
    print("EXTRACTION BENCHMARK")
    print("=" * 80)

    for table, bench in benchmarks.items():
        print(f"  {table}: {bench['rows']} rows in {bench['duration_seconds']}s")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Verify and test database indexes")
    parser.add_argument(
        "--test-queries",
        action="store_true",
        help="Run query performance tests"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run extraction benchmark"
    )
    parser.add_argument(
        "--monitor",
        action="store_true",
        help="Monitor index usage (requires performance_schema)"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run all tests"
    )

    args = parser.parse_args()

    # 默认运行验证
    if not any([args.test_queries, args.benchmark, args.monitor, args.full]):
        args.full = True

    configs = load_database_configs()

    # 1. 验证索引
    verification = verify_indexes(configs)

    # 2. 性能测试（可选）
    performance = {}
    if args.test_queries or args.full:
        performance = test_query_performance(configs)

    # 3. 基准测试（可选）
    benchmarks = {}
    if args.benchmark or args.full:
        benchmarks = benchmark_extraction(configs)

    # 4. 监控索引使用（可选）
    usage = {}
    if args.monitor or args.full:
        usage = monitor_index_usage(configs)

    # 5. 生成报告
    generate_report(verification, performance, benchmarks, usage)

    LOGGER.info("Index verification completed!")


if __name__ == "__main__":
    main()
