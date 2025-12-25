#!/bin/bash
# 应用生产环境内存优化修复
# 使用方法: bash scripts/apply_production_memory_fix.sh

set -e

echo "=============================================="
echo "生产环境内存优化部署脚本"
echo "=============================================="
echo ""

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 1. 检查当前环境
echo -e "${YELLOW}[1/5]${NC} 检查当前环境..."

if [ ! -f "docker-compose.yml" ]; then
    echo -e "${RED}错误: 未找到 docker-compose.yml${NC}"
    echo "请在项目根目录运行此脚本"
    exit 1
fi

if ! docker ps | grep -q airflow; then
    echo -e "${RED}警告: Airflow 容器未运行${NC}"
    echo "继续部署配置，稍后需要手动启动服务"
fi

echo -e "${GREEN}✓ 环境检查完成${NC}"
echo ""

# 2. 备份当前环境文件
echo -e "${YELLOW}[2/5]${NC} 备份当前配置..."

ENV_FILE=".env"
if [ ! -f "$ENV_FILE" ]; then
    ENV_FILE=".env.prod"
fi

if [ -f "$ENV_FILE" ]; then
    BACKUP_FILE="${ENV_FILE}.backup.$(date +%Y%m%d_%H%M%S)"
    cp "$ENV_FILE" "$BACKUP_FILE"
    echo -e "${GREEN}✓ 已备份到: $BACKUP_FILE${NC}"
else
    echo -e "${YELLOW}⚠ 未找到 .env 文件，将创建新文件${NC}"
    ENV_FILE=".env"
fi

echo ""

# 3. 添加或更新环境变量
echo -e "${YELLOW}[3/5]${NC} 配置内存优化参数..."

# 检查是否已有配置
if grep -q "SIMILARITY_MICRO_BATCH_SIZE" "$ENV_FILE" 2>/dev/null; then
    echo "检测到已有配置，更新现有值..."
    sed -i 's/^SIMILARITY_MICRO_BATCH_SIZE=.*/SIMILARITY_MICRO_BATCH_SIZE=50/' "$ENV_FILE"
    sed -i 's/^SIMILARITY_TOP_K=.*/SIMILARITY_TOP_K=200/' "$ENV_FILE"
else
    echo "添加新的优化配置..."
    cat >> "$ENV_FILE" << 'EOF'

# ==============================================
# 内存优化配置 (2025-12-25)
# ==============================================
# 相似度计算优化 - 逐行流处理，避免 200GB 内存峰值
SIMILARITY_MICRO_BATCH_SIZE=50
SIMILARITY_TOP_K=200

# 排序数据准备优化 - 避免 merge 内存爆炸
MAX_RANKING_SAMPLES=0

# 如果仍然内存不足（如 150GB 限制），可以：
# 1. 降低相似度计算参数：
#    SIMILARITY_MICRO_BATCH_SIZE=20
#    SIMILARITY_TOP_K=100
#
# 2. 限制排序样本数（推荐先尝试这个）：
#    MAX_RANKING_SAMPLES=5000000    # 500万样本
#    MAX_RANKING_SAMPLES=2000000    # 200万样本（更激进）
EOF
fi

echo -e "${GREEN}✓ 配置已更新${NC}"
cat "$ENV_FILE" | grep -A 5 "SIMILARITY_MICRO_BATCH_SIZE" || true
echo ""

# 4. 显示内存限制建议
echo -e "${YELLOW}[4/5]${NC} 检查 Docker 内存限制..."

if grep -q "deploy:" docker-compose.yml; then
    echo -e "${GREEN}✓ 已配置 Docker 资源限制${NC}"
else
    echo -e "${YELLOW}⚠ 建议在 docker-compose.yml 中添加内存限制${NC}"
    echo ""
    echo "示例配置 (添加到 airflow-scheduler 服务):"
    echo "---"
    cat << 'EOF'
  airflow-scheduler:
    ...
    deploy:
      resources:
        limits:
          memory: 32G  # 根据实际服务器资源调整
        reservations:
          memory: 16G
EOF
    echo "---"
fi

echo ""

# 5. 重启服务
echo -e "${YELLOW}[5/5]${NC} 重启 Airflow 服务..."

read -p "是否立即重启 Airflow 服务? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "重启中..."

    if docker ps | grep -q airflow-scheduler; then
        docker-compose restart airflow-scheduler airflow-webserver
        echo -e "${GREEN}✓ 服务已重启${NC}"
    else
        docker-compose up -d airflow-scheduler airflow-webserver
        echo -e "${GREEN}✓ 服务已启动${NC}"
    fi

    echo ""
    echo "等待服务启动..."
    sleep 5

    # 验证服务状态
    if docker ps | grep -q airflow-scheduler; then
        echo -e "${GREEN}✓ Airflow 调度器运行正常${NC}"
    else
        echo -e "${RED}✗ Airflow 调度器启动失败，请检查日志${NC}"
    fi
else
    echo -e "${YELLOW}跳过重启，请稍后手动执行:${NC}"
    echo "  docker-compose restart airflow-scheduler airflow-webserver"
fi

echo ""
echo "=============================================="
echo -e "${GREEN}部署完成！${NC}"
echo "=============================================="
echo ""
echo "下一步操作:"
echo ""
echo "1. 查看实时日志（验证优化已生效）:"
echo "   docker-compose logs -f airflow-scheduler | grep -E 'micro-batch|Progress'"
echo ""
echo "2. 监控内存占用:"
echo "   watch -n 5 'docker stats --no-stream | grep airflow-scheduler'"
echo ""
echo "3. 预期日志输出:"
echo "   INFO Computing similarity in micro-batches (total=X, micro_batch_size=50, top_k=200)"
echo "   INFO Progress: X/Y items (Z.Z%)"
echo ""
echo "4. 查看详细文档:"
echo "   cat docs/PRODUCTION_MEMORY_TUNING.md"
echo ""
echo "=============================================="
