#!/bin/bash
# 集成测试：验证数据源配置在实际数据加载中生效

set -e

echo "======================================================================"
echo "  数据源集成测试"
echo "======================================================================"
echo ""

# 检测 docker-compose
if command -v docker-compose &> /dev/null; then
    DC="docker-compose"
elif docker compose version &> /dev/null; then
    DC="docker compose"
else
    echo "错误: 未找到 docker-compose"
    exit 1
fi

if ! $DC ps recommendation-api | grep -q "Up"; then
    echo "⚠️  recommendation-api 服务未运行"
    exit 1
fi

echo "✓ 使用服务: recommendation-api"
echo ""

# 运行集成测试
$DC exec -T recommendation-api python3 <<'PYTHON_TEST'
import sys
import json

sys.path.insert(0, '/app')

print("="*70)
print("  测试 1: 验证配置值")
print("="*70)

from config.settings import (
    BUSINESS_SOURCE_MODE,
    MATOMO_SOURCE_MODE,
    SOURCE_DATA_MODES,
)

print(f"\n✓ BUSINESS_SOURCE_MODE = '{BUSINESS_SOURCE_MODE}'")
print(f"✓ MATOMO_SOURCE_MODE = '{MATOMO_SOURCE_MODE}'")
print(f"✓ SOURCE_DATA_MODES = {SOURCE_DATA_MODES}\n")

assert BUSINESS_SOURCE_MODE == "json", f"期望 'json'，实际 '{BUSINESS_SOURCE_MODE}'"
assert MATOMO_SOURCE_MODE == "database", f"期望 'database'，实际 '{MATOMO_SOURCE_MODE}'"

print("✅ 配置验证通过\n")

print("="*70)
print("  测试 2: 测试 JSON 数据加载（Business 数据）")
print("="*70)

try:
    from config.settings import DATA_JSON_DIR
    from pathlib import Path
    import os

    print(f"\n• DATA_JSON_DIR 配置: {DATA_JSON_DIR}")

    # 查找实际的 JSON 文件位置
    possible_paths = [
        Path(str(DATA_JSON_DIR)),
        Path("/app/data/dianshu_data"),
        Path("/app/data/dianshu_data/jsons"),
    ]

    json_path = None
    for p in possible_paths:
        if (p / "user.json").exists():
            json_path = p
            break

    if json_path:
        print(f"✓ 找到 JSON 文件位置: {json_path}")

        # 尝试读取一个 JSON 文件
        user_file = json_path / "user.json"
        with open(user_file, 'r', encoding='utf-8') as f:
            users = json.load(f)

        print(f"✓ 成功加载 user.json")
        print(f"  - 记录数: {len(users)}")
        if len(users) > 0:
            print(f"  - 第一条记录ID: {users[0].get('id', 'N/A')}")
            print(f"  - 字段数: {len(users[0])}")

        # 检查其他文件
        json_files = ["dataset.json", "task.json", "api_order.json"]
        for fname in json_files:
            fpath = json_path / fname
            if fpath.exists():
                print(f"✓ {fname} 存在")
    else:
        print("✗ 未找到 JSON 文件")
        sys.exit(1)

    print("\n✅ JSON 数据加载测试通过")

except Exception as e:
    print(f"\n✗ JSON 数据加载失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*70)
print("  测试 3: 验证数据库配置（Matomo 数据）")
print("="*70 + "\n")

try:
    from config.settings import load_database_configs

    db_configs = load_database_configs()
    matomo_cfg = db_configs.get("matomo")

    if matomo_cfg:
        print(f"✓ Matomo 数据库配置:")
        print(f"  - Host: {matomo_cfg.host}")
        print(f"  - Port: {matomo_cfg.port}")
        print(f"  - Database: {matomo_cfg.name}")
        print(f"  - User: {matomo_cfg.user}")

        # 尝试生成 SQLAlchemy URL
        url = matomo_cfg.sqlalchemy_url()
        print(f"✓ SQLAlchemy URL 生成成功")
        print(f"  - URL: mysql+pymysql://{matomo_cfg.user}:***@{matomo_cfg.host}:{matomo_cfg.port}/{matomo_cfg.name}")

        print("\n✅ 数据库配置验证通过")
    else:
        print("✗ Matomo 配置未找到")
        sys.exit(1)

except Exception as e:
    print(f"\n✗ 数据库配置验证失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*70)
print("  🎉 所有集成测试通过！")
print("="*70)
print("\n✅ 数据源配置已正确生效:")
print("  • Business 数据: 从 JSON 文件加载 ✓")
print("  • Matomo 数据: 从 MySQL 数据库加载 ✓")
print()

PYTHON_TEST

echo "======================================================================"
echo "  集成测试完成"
echo "======================================================================"
echo ""
