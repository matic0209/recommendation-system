#!/bin/bash
# 生产环境部署脚本
# 使用 .env.prod 配置文件

set -e

echo "=========================================="
echo "生产环境部署"
echo "=========================================="

# 检查 .env.prod 文件
if [ ! -f ".env.prod" ]; then
    echo "❌ 错误: .env.prod 文件不存在"
    echo "请创建 .env.prod 文件，配置生产环境参数"
    exit 1
fi

echo "✓ 使用配置文件: .env.prod"

# 停止现有容器
echo ""
echo ">>> 停止现有容器..."
docker-compose -f docker-compose.yml -f docker-compose.prod.yml down

# 拉取最新代码（如果需要）
if [ "$1" == "--pull" ]; then
    echo ""
    echo ">>> 拉取最新代码..."
    git pull origin master
fi

# 构建镜像
echo ""
echo ">>> 构建镜像..."
docker-compose -f docker-compose.yml -f docker-compose.prod.yml build

# 启动服务
echo ""
echo ">>> 启动服务..."
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# 等待服务启动
echo ""
echo ">>> 等待服务启动..."
sleep 10

# 检查服务状态
echo ""
echo ">>> 检查服务状态..."
docker-compose -f docker-compose.yml -f docker-compose.prod.yml ps

echo ""
echo "=========================================="
echo "✅ 部署完成！"
echo "=========================================="
echo ""
echo "服务访问地址（根据 .env.prod 配置）："
echo "  - 推荐 API: http://localhost:\${RECOMMEND_API_HOST_PORT}/docs"
echo "  - Airflow: http://localhost:\${AIRFLOW_WEB_HOST_PORT}"
echo "  - MLflow: http://localhost:\${MLFLOW_HOST_PORT}"
echo ""
echo "查看日志："
echo "  docker-compose -f docker-compose.yml -f docker-compose.prod.yml logs -f [service_name]"
echo ""
