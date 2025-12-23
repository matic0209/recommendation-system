#!/bin/bash
# 清除 Business 数据源状态，强制下次运行全量加载 JSON 数据

set -e

echo "======================================================================"
echo "  清除 Business 数据源状态 - 强制全量加载"
echo "======================================================================"
echo ""

STATE_FILE="data/_metadata/extract_state.json"

if [ ! -f "$STATE_FILE" ]; then
    echo "✓ 状态文件不存在，首次运行将自动全量加载"
    exit 0
fi

echo "1️⃣  当前状态（清除前）："
echo ""
cat "$STATE_FILE" | python3 -m json.tool 2>/dev/null || cat "$STATE_FILE"

echo ""
echo "2️⃣  备份当前状态..."
echo ""
BACKUP_FILE="${STATE_FILE}.backup_$(date +%Y%m%d_%H%M%S)"
cp "$STATE_FILE" "$BACKUP_FILE"
echo "✓ 已备份到: $BACKUP_FILE"

echo ""
echo "3️⃣  清除 business 数据源的 watermark..."
echo ""

# 使用 Python 清除 business 部分
python3 <<'PYTHON_SCRIPT'
import json
from pathlib import Path

state_file = Path("data/_metadata/extract_state.json")
data = json.loads(state_file.read_text())

# 清除 business 的状态
if "business" in data:
    print("  删除的表状态:")
    for table, state in data["business"].items():
        print(f"    • {table}: watermark={state.get('watermark')}")
    del data["business"]
    print("\n✓ 已清除 business 数据源状态")
else:
    print("  ⚠️  business 数据源状态不存在")

# 保存
state_file.write_text(json.dumps(data, indent=2))
print(f"\n✓ 已保存到: {state_file}")
PYTHON_SCRIPT

echo ""
echo "4️⃣  清除后的状态："
echo ""
cat "$STATE_FILE" | python3 -m json.tool 2>/dev/null || cat "$STATE_FILE"

echo ""
echo "======================================================================"
echo "  ✅ 完成"
echo "======================================================================"
echo ""
echo "下次运行 recommendation pipeline 时："
echo "  • Business 数据将从 JSON 全量文件加载（user.json, dataset.json 等）"
echo "  • Matomo 数据继续增量加载（保持现有 watermark）"
echo ""
echo "如需恢复状态，运行："
echo "  cp $BACKUP_FILE $STATE_FILE"
echo ""
