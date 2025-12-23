#!/bin/bash
# 在 Docker 容器内验证数据源配置的脚本

set -e

echo "======================================================================"
echo "  在 Docker 容器内验证数据源配置"
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

echo "使用命令: $DOCKER_COMPOSE"
echo ""

# 检查 recommendation-api 服务是否在运行
if ! $DOCKER_COMPOSE ps recommendation-api | grep -q "Up"; then
    echo "⚠️  警告: recommendation-api 服务未运行，尝试启动..."
    $DOCKER_COMPOSE up -d recommendation-api
    echo "等待服务启动..."
    sleep 10
fi

echo "======================================================================"
echo "  执行验证脚本"
echo "======================================================================"
echo ""

# 在 recommendation-api 容器内运行验证脚本
$DOCKER_COMPOSE exec -T recommendation-api python3 /opt/recommend/scripts/verify_data_source.py

exit_code=$?

echo ""
echo "======================================================================"
echo "  验证完成"
echo "======================================================================"
echo ""

if [ $exit_code -eq 0 ]; then
    echo "✓ 所有检查通过！"
else
    echo "✗ 部分检查未通过，退出码: $exit_code"
fi

exit $exit_code
