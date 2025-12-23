#!/usr/bin/env python3
"""
验证数据源配置脚本

该脚本用于验证 .env.prod 中的数据源配置是否正确且生效：
1. 检查环境变量是否正确设置
2. 验证数据源配置是否符合预期
3. 检查路径是否存在并可访问
4. 尝试实际加载数据验证配置是否真正生效
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import (
    DATA_SOURCE,
    DATA_JSON_DIR,
    BUSINESS_SOURCE_MODE,
    MATOMO_SOURCE_MODE,
    SOURCE_DATA_MODES,
    DATASET_IMAGE_ROOT,
    load_database_configs,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
LOGGER = logging.getLogger(__name__)


def print_section(title: str) -> None:
    """打印分隔线和标题"""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def check_env_variables() -> bool:
    """检查环境变量是否正确设置"""
    print_section("1. 环境变量检查")

    env_vars = {
        "DATA_SOURCE": os.getenv("DATA_SOURCE"),
        "BUSINESS_DATA_SOURCE": os.getenv("BUSINESS_DATA_SOURCE"),
        "MATOMO_DATA_SOURCE": os.getenv("MATOMO_DATA_SOURCE"),
        "DATA_JSON_DIR": os.getenv("DATA_JSON_DIR"),
        "DATASET_IMAGE_ROOT": os.getenv("DATASET_IMAGE_ROOT"),
    }

    all_ok = True
    for key, value in env_vars.items():
        status = "✓" if value else "✗"
        print(f"  {status} {key:25} = {value or '(未设置)'}")
        if not value and key in ["BUSINESS_DATA_SOURCE", "MATOMO_DATA_SOURCE", "DATA_JSON_DIR"]:
            all_ok = False

    return all_ok


def check_config_values() -> bool:
    """检查配置值是否符合预期"""
    print_section("2. 配置值验证")

    expected_config = {
        "DATA_SOURCE": ("json", DATA_SOURCE),
        "BUSINESS_DATA_SOURCE": ("json", BUSINESS_SOURCE_MODE),
        "MATOMO_DATA_SOURCE": ("database", MATOMO_SOURCE_MODE),
    }

    all_ok = True
    for key, (expected, actual) in expected_config.items():
        matches = expected == actual
        status = "✓" if matches else "✗"
        print(f"  {status} {key:25} = {actual:10} (期望: {expected})")
        if not matches:
            all_ok = False

    print(f"\n  SOURCE_DATA_MODES:")
    for source, mode in SOURCE_DATA_MODES.items():
        print(f"    - {source:10} : {mode}")

    return all_ok


def check_paths() -> bool:
    """检查路径是否存在并可访问"""
    print_section("3. 路径检查")

    paths_to_check = {
        "DATA_JSON_DIR": DATA_JSON_DIR,
        "DATASET_IMAGE_ROOT": DATASET_IMAGE_ROOT,
    }

    all_ok = True
    for name, path in paths_to_check.items():
        exists = path.exists()
        is_dir = path.is_dir() if exists else False
        readable = os.access(path, os.R_OK) if exists else False

        status = "✓" if (exists and is_dir and readable) else "✗"
        print(f"  {status} {name:20}")
        print(f"      路径: {path}")
        print(f"      存在: {exists}, 是目录: {is_dir}, 可读: {readable}")

        if not (exists and is_dir and readable):
            all_ok = False

    return all_ok


def check_json_files() -> bool:
    """检查 JSON 数据文件是否存在"""
    print_section("4. JSON 数据文件检查")

    required_files = [
        "user.json",
        "dataset.json",
        "task.json",
        "api_order.json",
        "dataset_image.json",
    ]

    all_ok = True
    for filename in required_files:
        file_path = DATA_JSON_DIR / filename
        exists = file_path.exists()
        readable = os.access(file_path, os.R_OK) if exists else False

        status = "✓" if (exists and readable) else "✗"
        print(f"  {status} {filename:20} - {file_path}")

        if exists and readable:
            size = file_path.stat().st_size
            print(f"      大小: {size:,} bytes ({size / 1024 / 1024:.2f} MB)")

        if not (exists and readable):
            all_ok = False

    return all_ok


def test_json_data_loading() -> bool:
    """测试从 JSON 加载数据"""
    print_section("5. JSON 数据加载测试")

    test_files = ["user.json", "dataset.json"]
    all_ok = True

    for filename in test_files:
        file_path = DATA_JSON_DIR / filename
        if not file_path.exists():
            print(f"  ✗ {filename} - 文件不存在")
            all_ok = False
            continue

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            record_count = len(data)
            print(f"  ✓ {filename}")
            print(f"      记录数: {record_count:,}")

            if record_count > 0:
                first_record = data[0]
                print(f"      字段数: {len(first_record)}")
                print(f"      字段名: {', '.join(list(first_record.keys())[:5])}...")

        except json.JSONDecodeError as e:
            print(f"  ✗ {filename} - JSON 解析错误: {e}")
            all_ok = False
        except Exception as e:
            print(f"  ✗ {filename} - 加载失败: {e}")
            all_ok = False

    return all_ok


def check_database_config() -> bool:
    """检查数据库配置（用于 Matomo 数据源）"""
    print_section("6. 数据库配置检查 (Matomo)")

    try:
        db_configs = load_database_configs()
        matomo_config = db_configs.get("matomo")

        if not matomo_config:
            print("  ✗ Matomo 数据库配置未找到")
            return False

        print(f"  ✓ Matomo 数据库配置:")
        print(f"      Host: {matomo_config.host}")
        print(f"      Port: {matomo_config.port}")
        print(f"      Database: {matomo_config.name}")
        print(f"      User: {matomo_config.user}")
        print(f"      Password: {'*' * len(matomo_config.password) if matomo_config.password else '(未设置)'}")

        return True

    except Exception as e:
        print(f"  ✗ 数据库配置加载失败: {e}")
        return False


def test_database_connection() -> bool:
    """测试数据库连接（仅测试连接，不执行查询）"""
    print_section("7. 数据库连接测试 (Matomo)")

    if MATOMO_SOURCE_MODE != "database":
        print("  ⊘ Matomo 数据源不是 database，跳过连接测试")
        return True

    try:
        from sqlalchemy import create_engine, text

        db_configs = load_database_configs()
        matomo_config = db_configs.get("matomo")

        if not matomo_config:
            print("  ✗ Matomo 数据库配置未找到")
            return False

        engine = create_engine(
            matomo_config.sqlalchemy_url(),
            pool_pre_ping=True,
            connect_args={"connect_timeout": 5}
        )

        # 测试连接
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            result.fetchone()

        print(f"  ✓ Matomo 数据库连接成功")
        print(f"      URL: {matomo_config.host}:{matomo_config.port}/{matomo_config.name}")

        engine.dispose()
        return True

    except ImportError:
        print("  ⊘ sqlalchemy 未安装，跳过数据库连接测试")
        return True
    except Exception as e:
        print(f"  ✗ Matomo 数据库连接失败: {e}")
        return False


def generate_summary(results: dict) -> None:
    """生成验证摘要"""
    print_section("验证摘要")

    total = len(results)
    passed = sum(1 for v in results.values() if v)
    failed = total - passed

    print(f"\n  总计: {total} 项检查")
    print(f"  通过: {passed} 项 ✓")
    print(f"  失败: {failed} 项 ✗")
    print()

    for name, status in results.items():
        symbol = "✓" if status else "✗"
        print(f"  {symbol} {name}")

    print()
    if failed == 0:
        print("  🎉 所有检查通过！数据源配置正确且生效。")
    else:
        print(f"  ⚠️  有 {failed} 项检查未通过，请检查配置。")
    print()


def main():
    """主函数"""
    print("\n" + "="*70)
    print("  数据源配置验证脚本")
    print("="*70)
    print(f"\n  当前工作目录: {os.getcwd()}")
    print(f"  脚本位置: {Path(__file__).resolve()}")

    results = {
        "环境变量设置": check_env_variables(),
        "配置值验证": check_config_values(),
        "路径检查": check_paths(),
        "JSON文件检查": check_json_files(),
        "JSON数据加载": test_json_data_loading(),
        "数据库配置": check_database_config(),
        "数据库连接": test_database_connection(),
    }

    generate_summary(results)

    # 返回退出码
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
