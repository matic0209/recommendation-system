#!/bin/bash
# 生产环境紧急修复脚本 - 禁用多进程避免 -7 崩溃

set -e

echo "======================================================================"
echo "  生产环境紧急修复：禁用 SBERT 多进程"
echo "======================================================================"
echo ""

# 检测 docker-compose
if command -v docker-compose &> /dev/null; then
    DC="docker-compose"
elif docker compose version &> /dev/null; then
    DC="docker compose"
else
    echo "✗ 错误: 未找到 docker-compose"
    exit 1
fi

echo "1️⃣  确认当前配置..."
echo ""
grep "TEXT_EMBED" .env.prod || echo "未找到 TEXT_EMBED 配置"

echo ""
echo "2️⃣  重启 airflow-scheduler 应用新配置..."
echo ""
$DC restart airflow-scheduler

echo ""
echo "等待服务重启..."
sleep 5

echo ""
echo "3️⃣  验证新配置已生效..."
echo ""
$DC exec airflow-scheduler env | grep "TEXT_EMBED" || echo "⚠️  容器中未找到 TEXT_EMBED 环境变量"

echo ""
echo "======================================================================"
echo "  ✅ 修复完成"
echo "======================================================================"
echo ""
echo "配置已应用："
echo "  TEXT_EMBED_WORKERS=1  (禁用多进程)"
echo ""
echo "下一步："
echo "  1. 在 Airflow UI 重新运行失败的 train_models 任务"
echo "  2. 监控日志确认不再出现 -7 错误"
echo ""
