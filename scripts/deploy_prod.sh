#!/bin/bash
# 生产环境部署脚本
# 使用 .env.prod 配置文件

set -e  # 遇到错误立即退出

# 定义颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检测 Docker Compose 版本
if command -v docker-compose &> /dev/null; then
    DOCKER_COMPOSE="docker-compose"
    DC_VERSION="V1"
elif docker compose version &> /dev/null; then
    DOCKER_COMPOSE="docker compose"
    DC_VERSION="V2"
else
    echo -e "${RED}❌ 错误: 未找到 docker-compose 或 docker compose${NC}"
    echo "请先安装 Docker Compose"
    exit 1
fi

echo "=========================================="
echo "生产环境部署"
echo "=========================================="
echo -e "${GREEN}✓ 使用 Docker Compose ${DC_VERSION}: ${DOCKER_COMPOSE}${NC}"

# 检查 .env.prod 文件
if [ ! -f ".env.prod" ]; then
    echo -e "${RED}❌ 错误: .env.prod 文件不存在${NC}"
    echo "请创建 .env.prod 文件，配置生产环境参数"
    echo "参考模板: .env.example"
    exit 1
fi

echo -e "${GREEN}✓ 使用配置文件: .env.prod${NC}"

# 加载 .env.prod 配置（用于显示端口信息）
# 过滤掉注释行和行内注释，只保留 KEY=VALUE 格式
set -a
source <(grep -v '^#' .env.prod | sed 's/#.*$//' | sed 's/[[:space:]]*$//' | grep -v '^$')
set +a

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
$DOCKER_COMPOSE -f docker-compose.yml -f docker-compose.prod.yml down || {
    echo -e "${YELLOW}⚠ 停止容器时出现警告，继续执行...${NC}"
}

# 构建镜像
echo ""
echo ">>> 构建镜像..."
$DOCKER_COMPOSE -f docker-compose.yml -f docker-compose.prod.yml build || {
    echo -e "${RED}❌ 镜像构建失败${NC}"
    exit 1
}

# 启动服务
echo ""
echo ">>> 启动服务..."
$DOCKER_COMPOSE -f docker-compose.yml -f docker-compose.prod.yml up -d || {
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
$DOCKER_COMPOSE -f docker-compose.yml -f docker-compose.prod.yml ps

# 检查关键服务健康状态
echo ""
echo ">>> 检查服务健康状态..."

# 检查 Redis
if $DOCKER_COMPOSE -f docker-compose.yml -f docker-compose.prod.yml exec -T redis redis-cli ping > /dev/null 2>&1; then
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

# 部署后验证和初始化
echo ""
echo "=========================================="
echo ">>> 部署后验证..."
echo "=========================================="

# 1. 检查数据文件是否存在
echo ""
echo "1. 检查数据文件..."
if [ -f "data/processed/dataset_features.parquet" ]; then
    echo -e "${GREEN}✓ 数据集特征文件存在${NC}"
else
    echo -e "${YELLOW}⚠ 数据集特征文件不存在，可能需要运行训练流程${NC}"
fi

if [ -f "models/ranking_model.txt" ]; then
    echo -e "${GREEN}✓ 排序模型文件存在${NC}"
else
    echo -e "${YELLOW}⚠ 排序模型文件不存在，可能需要运行训练流程${NC}"
fi

# 2. 测试推荐 API 功能
echo ""
echo "2. 测试推荐 API 功能..."
API_HEALTH=$(curl -sf "http://localhost:${RECOMMEND_API_HOST_PORT:-8090}/health" 2>/dev/null)
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ API 健康检查通过${NC}"

    # 测试推荐接口（如果 API 已就绪）
    TEST_RESPONSE=$(curl -sf "http://localhost:${RECOMMEND_API_HOST_PORT:-8090}/recommend/detail/1?user_id=123&limit=5" 2>/dev/null)
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ 推荐接口测试通过${NC}"
    else
        echo -e "${YELLOW}⚠ 推荐接口测试失败（可能数据未加载）${NC}"
    fi
else
    echo -e "${YELLOW}⚠ API 尚未就绪，跳过功能测试${NC}"
fi

# 3. 检查 Airflow DAG 状态
echo ""
echo "3. 检查 Airflow DAG 状态..."
if curl -sf "http://localhost:${AIRFLOW_WEB_HOST_PORT:-8080}/health" > /dev/null 2>&1; then
    # 等待 Airflow 完全启动
    sleep 5

    # 列出可用的 DAG（需要等 Airflow 完全就绪）
    DAG_COUNT=$($DOCKER_COMPOSE -f docker-compose.yml -f docker-compose.prod.yml exec -T airflow-webserver airflow dags list 2>/dev/null | grep -v "dag_id" | grep -v "^$" | wc -l)
    if [ "$DAG_COUNT" -gt 0 ]; then
        echo -e "${GREEN}✓ 检测到 $DAG_COUNT 个 Airflow DAG${NC}"
        echo "  可用 DAG："
        $DOCKER_COMPOSE -f docker-compose.yml -f docker-compose.prod.yml exec -T airflow-webserver airflow dags list 2>/dev/null | grep -v "dag_id" | head -5 | sed 's/^/    /'
    else
        echo -e "${YELLOW}⚠ 未检测到 Airflow DAG（可能仍在加载）${NC}"
    fi
else
    echo -e "${YELLOW}⚠ Airflow 未就绪，跳过 DAG 检查${NC}"
fi

# 4. 询问是否触发训练流程
echo ""
echo "4. 初始化数据和模型..."
if [ ! -f "models/ranking_model.txt" ] || [ ! -f "data/processed/dataset_features.parquet" ]; then
    echo -e "${YELLOW}检测到模型或数据文件缺失${NC}"
    read -p "是否立即触发训练流程？(y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo ""
        echo ">>> 触发训练流程..."

        # 检查 Airflow 中是否有训练 DAG
        TRAIN_DAG=$($DOCKER_COMPOSE -f docker-compose.yml -f docker-compose.prod.yml exec -T airflow-webserver airflow dags list 2>/dev/null | grep -i "train" | head -1 | awk '{print $1}')

        if [ -n "$TRAIN_DAG" ]; then
            echo "找到训练 DAG: $TRAIN_DAG"
            echo "触发 DAG 运行..."
            $DOCKER_COMPOSE -f docker-compose.yml -f docker-compose.prod.yml exec -T airflow-webserver airflow dags trigger "$TRAIN_DAG"
            echo -e "${GREEN}✓ 训练流程已触发${NC}"
            echo "  查看进度: http://localhost:${AIRFLOW_WEB_HOST_PORT:-8080}"
        else
            echo -e "${YELLOW}未找到训练 DAG，手动运行训练：${NC}"
            echo "  $DOCKER_COMPOSE -f docker-compose.yml -f docker-compose.prod.yml exec airflow-scheduler python3 -m pipeline.train_models"
        fi
    else
        echo "跳过训练流程"
    fi
else
    echo -e "${GREEN}✓ 模型和数据文件已存在${NC}"
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
echo "  $DOCKER_COMPOSE -f docker-compose.yml -f docker-compose.prod.yml ps"
echo ""
echo "  # 查看特定服务日志"
echo "  $DOCKER_COMPOSE -f docker-compose.yml -f docker-compose.prod.yml logs -f airflow-scheduler"
echo ""
echo "  # 重启特定服务"
echo "  $DOCKER_COMPOSE -f docker-compose.yml -f docker-compose.prod.yml restart recommendation-api"
echo ""
echo "  # 停止所有服务"
echo "  $DOCKER_COMPOSE -f docker-compose.yml -f docker-compose.prod.yml down"
echo ""
echo "注意："
echo "  - 某些服务可能需要额外时间完全启动"
echo "  - 如果服务未就绪，请等待1-2分钟后再访问"
echo "  - 查看日志排查问题：$DOCKER_COMPOSE -f docker-compose.yml -f docker-compose.prod.yml logs"
echo ""
