#!/bin/bash
# 诊断数据加载状态，检查是否需要全量加载

set -e

echo "======================================================================"
echo "  数据加载状态诊断"
echo "======================================================================"
echo ""

STATE_FILE="data/_metadata/extract_state.json"

echo "1️⃣  检查配置..."
echo ""
echo "数据源配置:"
grep -E "^(DATA_SOURCE|BUSINESS_DATA_SOURCE|MATOMO_DATA_SOURCE)=" .env.prod 2>/dev/null || echo "  ⚠️  未找到 .env.prod"

echo ""
echo "2️⃣  检查状态文件..."
echo ""
if [ -f "$STATE_FILE" ]; then
    echo "✓ 状态文件存在: $STATE_FILE"
    echo ""
    echo "当前 watermarks:"
    cat "$STATE_FILE" | python3 -c "
import sys, json
data = json.load(sys.stdin)
for source, tables in data.items():
    print(f'\n  {source}:')
    for table, state in tables.items():
        watermark = state.get('watermark', 'N/A')
        print(f'    • {table:25} = {watermark}')
"
else
    echo "⚠️  状态文件不存在"
    echo "   首次运行将自动全量加载"
fi

echo ""
echo "3️⃣  检查现有 Parquet 数据..."
echo ""
for dir in data/business data/matomo; do
    if [ -d "$dir" ]; then
        echo "$dir:"
        ls -lh "$dir"/*.parquet 2>/dev/null | awk '{print "  • " $9 ": " $5 " (" $6 " " $7 ")"}' || echo "  (无数据)"
    fi
done

echo ""
echo "4️⃣  检查 JSON 源文件..."
echo ""
JSON_DIR=$(grep "^DATA_JSON_DIR=" .env.prod 2>/dev/null | cut -d= -f2 || echo "data/dianshu_data")
echo "JSON 目录: $JSON_DIR"

if [ -d "$JSON_DIR" ]; then
    echo ""
    echo "全量文件:"
    for file in user.json dataset.json task.json api_order.json dataset_image.json; do
        if [ -f "$JSON_DIR/$file" ]; then
            size=$(ls -lh "$JSON_DIR/$file" | awk '{print $5}')
            modified=$(ls -l "$JSON_DIR/$file" | awk '{print $6, $7, $8}')
            echo "  ✓ $file ($size, $modified)"
        elif [ -f "$JSON_DIR/jsons/$file" ]; then
            size=$(ls -lh "$JSON_DIR/jsons/$file" | awk '{print $5}')
            modified=$(ls -l "$JSON_DIR/jsons/$file" | awk '{print $6, $7, $8}')
            echo "  ✓ jsons/$file ($size, $modified)"
        else
            echo "  ✗ $file (不存在)"
        fi
    done

    echo ""
    echo "增量文件数量:"
    for prefix in user dataset task api_order dataset_image; do
        count=$(find "$JSON_DIR" -name "${prefix}_*.json" 2>/dev/null | wc -l)
        echo "  • ${prefix}_*.json: $count 个"
    done
else
    echo "✗ JSON 目录不存在: $JSON_DIR"
fi

echo ""
echo "======================================================================"
echo "  诊断结果"
echo "======================================================================"
echo ""

# 检查是否需要清除状态
if [ -f "$STATE_FILE" ]; then
    python3 <<'PYTHON_SCRIPT'
import json
from pathlib import Path
from datetime import datetime

state_file = Path("data/_metadata/extract_state.json")
data = json.loads(state_file.read_text())

need_reset = False

if "business" in data:
    print("⚠️  检测到 business 数据源有历史 watermark")
    print("")
    print("如果你刚切换到 JSON 数据源，建议:")
    print("  1. 运行以下命令清除状态，强制全量加载:")
    print("     bash scripts/reset_business_state.sh")
    print("")
    print("  2. 或在 Airflow DAG 中运行 extract_load 任务时使用:")
    print("     --full-refresh 参数")
    print("")
    need_reset = True
else:
    print("✓ business 数据源无历史状态，首次运行将全量加载")

if not need_reset:
    print("")
    print("✓ 无需额外操作，可以直接运行 pipeline")
PYTHON_SCRIPT
else
    echo "✓ 无历史状态，首次运行将自动全量加载"
fi

echo ""
