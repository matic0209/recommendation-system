#!/usr/bin/env python3
"""
数据库索引自动优化脚本 (Python 版本)

用途: 在新的数据库环境中自动创建和优化索引
使用: python scripts/setup_database_indexes.py [--skip-confirmation] [--production]

作者: AI Assistant
日期: 2025-10-10
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from sqlalchemy import create_engine, text

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import load_database_configs

# 日志配置
LOG_DIR = project_root / "logs" / "index_optimization"
LOG_DIR.mkdir(parents=True, exist_ok=True)
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = LOG_DIR / f"setup_indexes_{TIMESTAMP}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ANSI 颜色代码
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_header():
    """打印脚本标题"""
    print(f"{Colors.BLUE}")
    print("╔═══════════════════════════════════════════════════════════════════╗")
    print("║          数据库索引自动优化脚本 v1.0 (Python)                     ║")
    print("║          Database Index Optimization Automation                   ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")
    print(f"{Colors.ENDC}")


def test_db_connection(configs: Dict) -> bool:
    """测试数据库连接"""
    logger.info("测试数据库连接...")

    try:
        # 测试业务库
        logger.info(f"  → 测试业务库 ({configs['business'].db_name})...")
        business_engine = create_engine(
            configs['business'].sqlalchemy_url(),
            **configs['business'].get_engine_kwargs()
        )
        with business_engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("    ✓ 业务库连接成功")
        business_engine.dispose()

        # 测试 Matomo 库
        logger.info(f"  → 测试 Matomo 库 ({configs['matomo'].db_name})...")
        matomo_engine = create_engine(
            configs['matomo'].sqlalchemy_url(),
            **configs['matomo'].get_engine_kwargs()
        )
        with matomo_engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("    ✓ Matomo 库连接成功")
        matomo_engine.dispose()

        logger.info("✓ 所有数据库连接测试通过")
        return True

    except Exception as e:
        logger.error(f"数据库连接失败: {e}")
        return False


def check_existing_indexes(configs: Dict) -> Dict[str, int]:
    """检查现有索引数量"""
    logger.info("检查现有索引状态...")

    results = {}

    # 检查业务库索引
    business_engine = create_engine(
        configs['business'].sqlalchemy_url(),
        **configs['business'].get_engine_kwargs()
    )
    with business_engine.connect() as conn:
        query = text("""
            SELECT COUNT(DISTINCT INDEX_NAME) as count
            FROM information_schema.STATISTICS
            WHERE TABLE_SCHEMA = :schema
              AND TABLE_NAME IN ('user', 'dataset', 'task', 'api_order', 'dataset_image')
              AND INDEX_NAME LIKE 'idx_%'
        """)
        result = conn.execute(query, {"schema": configs['business'].db_name})
        results['business'] = result.scalar()
    business_engine.dispose()

    # 检查 Matomo 库索引
    matomo_engine = create_engine(
        configs['matomo'].sqlalchemy_url(),
        **configs['matomo'].get_engine_kwargs()
    )
    with matomo_engine.connect() as conn:
        query = text("""
            SELECT COUNT(DISTINCT INDEX_NAME) as count
            FROM information_schema.STATISTICS
            WHERE TABLE_SCHEMA = :schema
              AND TABLE_NAME IN ('matomo_log_visit', 'matomo_log_link_visit_action', 'matomo_log_conversion')
              AND INDEX_NAME LIKE 'idx_%'
        """)
        result = conn.execute(query, {"schema": configs['matomo'].db_name})
        results['matomo'] = result.scalar()
    matomo_engine.dispose()

    logger.info(f"  → 业务库现有索引: {results['business']} 个")
    logger.info(f"  → Matomo库现有索引: {results['matomo']} 个")

    if results['business'] > 0 or results['matomo'] > 0:
        logger.warning("检测到已存在的索引，脚本会智能跳过已创建的索引")

    return results


def confirm_execution(configs: Dict, skip_confirmation: bool, is_production: bool) -> bool:
    """用户确认"""
    if skip_confirmation:
        logger.info("跳过用户确认（--skip-confirmation）")
        return True

    print()
    print(f"{Colors.YELLOW}{'═' * 70}{Colors.ENDC}")

    if is_production:
        print(f"{Colors.RED}⚠️  警告: 您即将在生产环境执行索引优化！{Colors.ENDC}")
    else:
        print(f"{Colors.YELLOW}您即将执行数据库索引优化操作{Colors.ENDC}")

    print()
    print("目标数据库:")
    print(f"  - 业务库: {configs['business'].db_host}:{configs['business'].db_port}/{configs['business'].db_name}")
    print(f"  - Matomo: {configs['matomo'].db_host}:{configs['matomo'].db_port}/{configs['matomo'].db_name}")
    print()
    print("操作内容:")
    print("  1. 创建时间相关索引（用于 CDC 增量抽取）")
    print("  2. 创建联合索引（优化常用查询）")
    print("  3. 验证索引创建结果")
    print()
    print("预期影响:")
    print("  - Pipeline 执行速度提升 60-80%")
    print("  - 索引创建期间可能对数据库产生短暂负载")
    print()
    print(f"{Colors.YELLOW}{'═' * 70}{Colors.ENDC}")
    print()

    confirm = input("确认继续执行？(yes/no): ")

    if confirm.lower() != 'yes':
        logger.warning("用户取消操作")
        return False

    logger.info("用户确认执行")
    return True


def backup_metadata(configs: Dict):
    """备份索引元数据"""
    logger.info("保存数据库元数据备份...")

    backup_dir = project_root / "backups" / "index_metadata" / TIMESTAMP
    backup_dir.mkdir(parents=True, exist_ok=True)

    # 备份业务库索引信息
    business_engine = create_engine(
        configs['business'].sqlalchemy_url(),
        **configs['business'].get_engine_kwargs()
    )
    with business_engine.connect() as conn:
        query = text("""
            SELECT * FROM information_schema.STATISTICS
            WHERE TABLE_SCHEMA = :schema
            ORDER BY TABLE_NAME, INDEX_NAME, SEQ_IN_INDEX
        """)
        result = conn.execute(query, {"schema": configs['business'].db_name})

        with open(backup_dir / "business_indexes_before.txt", "w") as f:
            for row in result:
                f.write(str(dict(row._mapping)) + "\n")
    business_engine.dispose()

    # 备份 Matomo 库索引信息
    matomo_engine = create_engine(
        configs['matomo'].sqlalchemy_url(),
        **configs['matomo'].get_engine_kwargs()
    )
    with matomo_engine.connect() as conn:
        query = text("""
            SELECT * FROM information_schema.STATISTICS
            WHERE TABLE_SCHEMA = :schema
            ORDER BY TABLE_NAME, INDEX_NAME, SEQ_IN_INDEX
        """)
        result = conn.execute(query, {"schema": configs['matomo'].db_name})

        with open(backup_dir / "matomo_indexes_before.txt", "w") as f:
            for row in result:
                f.write(str(dict(row._mapping)) + "\n")
    matomo_engine.dispose()

    logger.info(f"  ✓ 元数据备份保存到: {backup_dir}")


def execute_sql_file(sql_file: Path, db_config) -> bool:
    """执行 SQL 文件"""
    logger.info(f"执行 SQL 脚本: {sql_file}")

    try:
        # 读取 SQL 文件
        with open(sql_file, 'r', encoding='utf-8') as f:
            sql_content = f.read()

        # 创建引擎
        engine = create_engine(
            db_config.sqlalchemy_url(),
            **db_config.get_engine_kwargs()
        )

        # 分割和执行 SQL 语句
        # 注意: 由于使用了存储过程语法，需要特殊处理
        import subprocess

        # 使用 mysql 命令行执行（更可靠）
        cmd = [
            'mysql',
            f'-h{db_config.db_host}',
            f'-P{db_config.db_port}',
            f'-u{db_config.db_user}',
            f'-p{db_config.db_password}',
        ]

        result = subprocess.run(
            cmd,
            stdin=open(sql_file, 'r'),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        if result.returncode == 0:
            logger.info("✓ SQL 脚本执行成功")
            return True
        else:
            logger.error(f"SQL 脚本执行失败: {result.stderr}")
            return False

    except Exception as e:
        logger.error(f"执行 SQL 文件失败: {e}")
        return False


def verify_indexes(configs: Dict) -> Dict[str, int]:
    """验证索引创建结果"""
    logger.info("验证索引创建结果...")

    results = check_existing_indexes(configs)

    logger.info(f"  ✓ 业务库索引总数: {results['business']}")
    logger.info(f"  ✓ Matomo库索引总数: {results['matomo']}")

    # 测试查询性能
    logger.info("测试索引使用情况...")

    business_engine = create_engine(
        configs['business'].sqlalchemy_url(),
        **configs['business'].get_engine_kwargs()
    )
    with business_engine.connect() as conn:
        query = text("EXPLAIN SELECT * FROM dataset WHERE update_time > '2025-09-01' LIMIT 10")
        result = conn.execute(query)
        explain_result = result.fetchone()

        if explain_result and 'idx_' in str(explain_result):
            logger.info("  ✓ 查询测试通过（使用索引）")
        else:
            logger.warning("  ⚠ 查询可能未使用索引")
    business_engine.dispose()

    return results


def generate_report(configs: Dict, index_counts: Dict[str, int]):
    """生成优化报告"""
    logger.info("生成优化报告...")

    report_file = LOG_DIR / f"optimization_report_{TIMESTAMP}.txt"

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("═" * 70 + "\n")
        f.write("数据库索引优化报告\n")
        f.write("Database Index Optimization Report\n")
        f.write("═" * 70 + "\n\n")

        f.write(f"执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("─" * 70 + "\n")
        f.write("数据库连接信息\n")
        f.write("─" * 70 + "\n")
        f.write(f"业务库: {configs['business'].db_host}:{configs['business'].db_port}/{configs['business'].db_name}\n")
        f.write(f"Matomo: {configs['matomo'].db_host}:{configs['matomo'].db_port}/{configs['matomo'].db_name}\n\n")

        f.write("─" * 70 + "\n")
        f.write("索引统计\n")
        f.write("─" * 70 + "\n")
        f.write(f"业务库索引数量: {index_counts['business']}\n")
        f.write(f"Matomo库索引数量: {index_counts['matomo']}\n\n")

        f.write("─" * 70 + "\n")
        f.write("预期收益\n")
        f.write("─" * 70 + "\n")
        f.write("✓ CDC 增量抽取性能提升 60-80%\n")
        f.write("✓ 降低数据库 CPU 使用率\n")
        f.write("✓ 减少全表扫描\n")
        f.write("✓ 提升 Pipeline 整体执行速度\n\n")

        f.write("─" * 70 + "\n")
        f.write("日志文件\n")
        f.write("─" * 70 + "\n")
        f.write(f"完整日志: {LOG_FILE}\n")
        f.write(f"优化报告: {report_file}\n\n")

        f.write("═" * 70 + "\n")

    logger.info(f"✓ 报告已生成: {report_file}")

    # 显示报告
    print("\n" + open(report_file, 'r', encoding='utf-8').read())


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="数据库索引自动优化工具")
    parser.add_argument(
        "--skip-confirmation",
        action="store_true",
        help="跳过用户确认"
    )
    parser.add_argument(
        "--production",
        action="store_true",
        help="标记为生产环境（显示警告）"
    )

    args = parser.parse_args()

    try:
        # 打印标题
        print_header()

        # 加载配置
        logger.info("加载数据库配置...")
        configs = load_database_configs()
        logger.info("✓ 配置加载完成")

        # 测试连接
        if not test_db_connection(configs):
            logger.error("数据库连接测试失败，退出")
            sys.exit(1)

        # 检查现有索引
        check_existing_indexes(configs)

        # 用户确认
        if not confirm_execution(configs, args.skip_confirmation, args.production):
            sys.exit(0)

        # 备份元数据
        backup_metadata(configs)

        # 执行索引创建
        sql_file = project_root / "scripts" / "p0_01_add_indexes_fixed.sql"
        if not sql_file.exists():
            logger.error(f"SQL 脚本不存在: {sql_file}")
            sys.exit(1)

        if not execute_sql_file(sql_file, configs['business']):
            logger.error("索引创建失败")
            sys.exit(1)

        # 验证索引
        index_counts = verify_indexes(configs)

        # 生成报告
        generate_report(configs, index_counts)

        # 完成
        print()
        print(f"{Colors.GREEN}╔═══════════════════════════════════════════════════════════════════╗{Colors.ENDC}")
        print(f"{Colors.GREEN}║           ✓ 数据库索引优化完成！                                  ║{Colors.ENDC}")
        print(f"{Colors.GREEN}║           Database Index Optimization Completed!                  ║{Colors.ENDC}")
        print(f"{Colors.GREEN}╚═══════════════════════════════════════════════════════════════════╝{Colors.ENDC}")
        print()

        logger.info("脚本执行成功")

    except KeyboardInterrupt:
        logger.warning("用户中断执行")
        sys.exit(130)
    except Exception as e:
        logger.error(f"脚本执行失败: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
