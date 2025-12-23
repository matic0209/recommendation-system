#!/bin/bash
# 快速验证数据源配置的脚本（在 Docker 容器内运行）

set -e

echo "======================================================================"
echo "  数据源配置验证（Docker容器内）"
echo "======================================================================"
echo ""

# 检测 docker-compose 版本
if command -v docker-compose &> /dev/null; then
    DOCKER_COMPOSE="docker-compose"
elif docker compose version &> /dev/null; then
    DOCKER_COMPOSE="docker compose"
else
    echo "错误: 未找到 docker-compose 或 docker compose 命令"
    exit 1
fi

# 检查 recommendation-api 服务是否在运行
if ! $DOCKER_COMPOSE ps recommendation-api | grep -q "Up"; then
    echo "⚠️  警告: recommendation-api 服务未运行"
    exit 1
fi

echo "使用服务: recommendation-api"
echo ""

# 在容器内运行验证脚本
$DOCKER_COMPOSE exec -T recommendation-api python3 <<'PYTHON_SCRIPT'
import os
import sys
import json
from pathlib import Path

def print_section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

def check_env_and_config():
    """检查环境变量和配置值"""
    print_section("1. 环境变量和配置检查")

    # 导入配置
    sys.path.insert(0, '/app')
    from config.settings import (
        DATA_SOURCE,
        DATA_JSON_DIR,
        BUSINESS_SOURCE_MODE,
        MATOMO_SOURCE_MODE,
        SOURCE_DATA_MODES,
        DATASET_IMAGE_ROOT,
    )

    # 环境变量
    env_vars = {
        "DATA_SOURCE": os.getenv("DATA_SOURCE"),
        "BUSINESS_DATA_SOURCE": os.getenv("BUSINESS_DATA_SOURCE"),
        "MATOMO_DATA_SOURCE": os.getenv("MATOMO_DATA_SOURCE"),
        "DATA_JSON_DIR": os.getenv("DATA_JSON_DIR"),
    }

    print("\n  环境变量:")
    for key, value in env_vars.items():
        status = "✓" if value else "✗"
        print(f"    {status} {key:25} = {value or '(未设置)'}")

    # 配置值
    print("\n  配置值 (从 config.settings):")
    print(f"    • DATA_SOURCE            = {DATA_SOURCE}")
    print(f"    • BUSINESS_SOURCE_MODE   = {BUSINESS_SOURCE_MODE}")
    print(f"    • MATOMO_SOURCE_MODE     = {MATOMO_SOURCE_MODE}")
    print(f"    • DATA_JSON_DIR          = {DATA_JSON_DIR}")
    print(f"    • DATASET_IMAGE_ROOT     = {DATASET_IMAGE_ROOT}")

    print("\n  SOURCE_DATA_MODES 字典:")
    for source, mode in SOURCE_DATA_MODES.items():
        print(f"    • {source:10} -> {mode}")

    # 验证配置是否符合预期
    print("\n  预期配置验证:")
    checks = {
        "BUSINESS_SOURCE_MODE = 'json'": BUSINESS_SOURCE_MODE == "json",
        "MATOMO_SOURCE_MODE = 'database'": MATOMO_SOURCE_MODE == "database",
    }

    all_passed = True
    for check_name, passed in checks.items():
        status = "✓" if passed else "✗"
        print(f"    {status} {check_name}")
        if not passed:
            all_passed = False

    return all_passed

def check_paths():
    """检查路径"""
    print_section("2. 路径检查")

    sys.path.insert(0, '/app')
    from config.settings import DATA_JSON_DIR, DATASET_IMAGE_ROOT

    paths_to_check = {
        "DATA_JSON_DIR": Path(str(DATA_JSON_DIR)),
        "DATASET_IMAGE_ROOT": Path(str(DATASET_IMAGE_ROOT)),
    }

    all_ok = True
    for name, path in paths_to_check.items():
        exists = path.exists()
        is_dir = path.is_dir() if exists else False
        readable = os.access(path, os.R_OK) if exists else False

        status = "✓" if (exists and is_dir and readable) else "✗"
        print(f"  {status} {name}")
        print(f"      路径: {path}")
        print(f"      存在: {exists}, 是目录: {is_dir}, 可读: {readable}")

        if not (exists and is_dir and readable):
            all_ok = False

    return all_ok

def check_json_files():
    """检查 JSON 文件"""
    print_section("3. JSON 文件检查")

    sys.path.insert(0, '/app')
    from config.settings import DATA_JSON_DIR

    json_dir = Path(str(DATA_JSON_DIR))
    required_files = [
        "user.json",
        "dataset.json",
        "task.json",
        "api_order.json",
        "dataset_image.json",
    ]

    all_ok = True
    for filename in required_files:
        file_path = json_dir / filename
        exists = file_path.exists()

        if exists:
            size = file_path.stat().st_size
            readable = os.access(file_path, os.R_OK)
            status = "✓" if readable else "✗"
            print(f"  {status} {filename:20} ({size:,} bytes)")

            # 尝试读取并显示记录数
            if readable:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    print(f"      记录数: {len(data):,}")
                except Exception as e:
                    print(f"      ⚠️  读取失败: {e}")
                    all_ok = False
        else:
            print(f"  ✗ {filename:20} (文件不存在)")
            all_ok = False

    return all_ok

def check_database_config():
    """检查数据库配置"""
    print_section("4. 数据库配置检查 (Matomo)")

    sys.path.insert(0, '/app')
    from config.settings import load_database_configs, MATOMO_SOURCE_MODE

    if MATOMO_SOURCE_MODE != "database":
        print(f"  ⊘ MATOMO_SOURCE_MODE = '{MATOMO_SOURCE_MODE}', 跳过数据库检查")
        return True

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

        return True
    except Exception as e:
        print(f"  ✗ 数据库配置加载失败: {e}")
        return False

def main():
    print("\n  容器信息:")
    print(f"    • 工作目录: {os.getcwd()}")
    print(f"    • Python路径: {sys.executable}")

    results = {
        "环境变量和配置": check_env_and_config(),
        "路径检查": check_paths(),
        "JSON文件检查": check_json_files(),
        "数据库配置": check_database_config(),
    }

    # 摘要
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
        sys.exit(0)
    else:
        print(f"  ⚠️  有 {failed} 项检查未通过，请检查配置。")
        sys.exit(1)

if __name__ == "__main__":
    main()
PYTHON_SCRIPT

exit $?
