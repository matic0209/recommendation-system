#!/bin/bash
# 生产环境部署脚本
# 使用 .env.prod 配置文件

set -e  # 遇到错误立即退出

# 定义颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "生产环境部署"
echo "=========================================="

# 检查 .env.prod 文件
if [ ! -f ".env.prod" ]; then
    echo -e "${RED}❌ 错误: .env.prod 文件不存在${NC}"
    echo "请创建 .env.prod 文件，配置生产环境参数"
    echo "参考模板: .env.example"
    exit 1
fi

echo -e "${GREEN}✓ 使用配置文件: .env.prod${NC}"

# 加载 .env.prod 配置（用于显示端口信息）
export $(grep -v '^#' .env.prod | xargs)

# 显示关键配置
echo ""
echo "关键配置："
echo "  - 推荐 API 端口: ${RECOMMEND_API_HOST_PORT:-8090}"
echo "  - Airflow 端口: ${AIRFLOW_WEB_HOST_PORT:-8080}"
echo "  - MLflow 端口: ${MLFLOW_HOST_PORT:-5000}"
echo "  - HuggingFace 镜像: ${HF_ENDPOINT:-https://huggingface.co}"
echo ""

# 确认是否继续
read -p "确认继续部署？(y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "部署已取消"
    exit 0
fi

# 拉取最新代码（如果需要）
if [ "$1" == "--pull" ]; then
    echo ""
    echo ">>> 拉取最新代码..."
    git pull origin master || {
        echo -e "${RED}❌ Git pull 失败${NC}"
        exit 1
    }
fi

# 停止现有容器
echo ""
echo ">>> 停止现有容器..."
docker-compose -f docker-compose.yml -f docker-compose.prod.yml down || {
    echo -e "${YELLOW}⚠ 停止容器时出现警告，继续执行...${NC}"
}

# 构建镜像
echo ""
echo ">>> 构建镜像..."
docker-compose -f docker-compose.yml -f docker-compose.prod.yml build || {
    echo -e "${RED}❌ 镜像构建失败${NC}"
    exit 1
}

# 启动服务
echo ""
echo ">>> 启动服务..."
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d || {
    echo -e "${RED}❌ 服务启动失败${NC}"
    exit 1
}

# 等待服务启动
echo ""
echo ">>> 等待服务启动..."
for i in {1..30}; do
    echo -n "."
    sleep 1
done
echo ""

# 检查服务状态
echo ""
echo ">>> 检查服务状态..."
docker-compose -f docker-compose.yml -f docker-compose.prod.yml ps

# 检查关键服务健康状态
echo ""
echo ">>> 检查服务健康状态..."

# 检查 Redis
if docker-compose -f docker-compose.yml -f docker-compose.prod.yml exec -T redis redis-cli ping > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Redis: 运行正常${NC}"
else
    echo -e "${RED}✗ Redis: 未就绪${NC}"
fi

# 检查 MLflow
if curl -sf "http://localhost:${MLFLOW_HOST_PORT:-5000}/health" > /dev/null 2>&1; then
    echo -e "${GREEN}✓ MLflow: 运行正常${NC}"
else
    echo -e "${YELLOW}⚠ MLflow: 未就绪（可能仍在启动中）${NC}"
fi

# 检查推荐 API
if curl -sf "http://localhost:${RECOMMEND_API_HOST_PORT:-8090}/health" > /dev/null 2>&1; then
    echo -e "${GREEN}✓ 推荐 API: 运行正常${NC}"
else
    echo -e "${YELLOW}⚠ 推荐 API: 未就绪（可能仍在启动中）${NC}"
fi

# 检查 Airflow Webserver
if curl -sf "http://localhost:${AIRFLOW_WEB_HOST_PORT:-8080}/health" > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Airflow Webserver: 运行正常${NC}"
else
    echo -e "${YELLOW}⚠ Airflow Webserver: 未就绪（可能仍在启动中）${NC}"
fi

echo ""
echo "=========================================="
echo -e "${GREEN}✅ 部署完成！${NC}"
echo "=========================================="
echo ""
echo "服务访问地址："
echo "  - 推荐 API: http://localhost:${RECOMMEND_API_HOST_PORT:-8090}/docs"
echo "  - Airflow: http://localhost:${AIRFLOW_WEB_HOST_PORT:-8080}"
echo "  - MLflow: http://localhost:${MLFLOW_HOST_PORT:-5000}"
echo "  - Grafana: http://localhost:${GRAFANA_HOST_PORT:-3000}"
echo "  - Prometheus: http://localhost:${PROMETHEUS_HOST_PORT:-9090}"
echo ""
echo "常用命令："
echo "  # 查看所有服务状态"
echo "  docker-compose -f docker-compose.yml -f docker-compose.prod.yml ps"
echo ""
echo "  # 查看特定服务日志"
echo "  docker-compose -f docker-compose.yml -f docker-compose.prod.yml logs -f airflow-scheduler"
echo ""
echo "  # 重启特定服务"
echo "  docker-compose -f docker-compose.yml -f docker-compose.prod.yml restart recommendation-api"
echo ""
echo "  # 停止所有服务"
echo "  docker-compose -f docker-compose.yml -f docker-compose.prod.yml down"
echo ""
echo "注意："
echo "  - 某些服务可能需要额外时间完全启动"
echo "  - 如果服务未就绪，请等待1-2分钟后再访问"
echo "  - 查看日志排查问题：docker-compose -f docker-compose.yml -f docker-compose.prod.yml logs"
echo ""
