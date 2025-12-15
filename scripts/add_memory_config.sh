#!/bin/bash
# 自动添加内存优化配置到 .env 文件

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ENV_FILE="${PROJECT_DIR}/.env"

echo "========================================="
echo "内存优化配置自动添加脚本"
echo "========================================="
echo ""

# 检查 .env 文件是否存在
if [ ! -f "$ENV_FILE" ]; then
    echo "错误: .env 文件不存在于 $ENV_FILE"
    exit 1
fi

echo "找到 .env 文件: $ENV_FILE"
echo ""

# 检查是否已经有内存优化配置
if grep -q "SIMILARITY_BATCH_SIZE" "$ENV_FILE"; then
    echo "⚠️  .env 文件中已经存在内存优化配置"
    echo ""
    echo "当前配置:"
    grep -A 10 "Memory Optimization" "$ENV_FILE" || grep -A 5 "SIMILARITY_BATCH_SIZE" "$ENV_FILE"
    echo ""
    read -p "是否要覆盖现有配置? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "已取消操作"
        exit 0
    fi

    # 删除旧配置
    echo "正在删除旧配置..."
    sed -i.bak '/# Memory Optimization/,/MALLOC_TRIM_THRESHOLD_=/d' "$ENV_FILE"
    sed -i.bak '/SIMILARITY_BATCH_SIZE/d; /SIMILARITY_TOP_K/d; /USE_FAISS_RECALL/d; /RANKING_CVR_WEIGHT/d; /PYTHONHASHSEED/d; /MALLOC_TRIM_THRESHOLD_/d' "$ENV_FILE"
fi

# 备份 .env 文件
BACKUP_FILE="${ENV_FILE}.backup_$(date +%Y%m%d_%H%M%S)"
cp "$ENV_FILE" "$BACKUP_FILE"
echo "✓ 已备份 .env 到: $BACKUP_FILE"
echo ""

# 添加内存优化配置
echo "正在添加内存优化配置..."
cat >> "$ENV_FILE" << 'EOF'

# ============================================
# 内存优化配置 (Memory Optimization)
# ============================================
# 相似度计算批次大小（越小越省内存，但速度越慢）
# 推荐值：1000（默认），内存不足时降到 500
SIMILARITY_BATCH_SIZE=1000

# 每个数据集保留的 top-K 相似项数量
# 推荐值：200（默认），召回需求低时可降到 100
SIMILARITY_TOP_K=200

# 是否启用 Faiss 向量召回（可能占用较多内存）
# 推荐值：1（启用），内存受限时设为 0
USE_FAISS_RECALL=1

# 排序模型 CVR 权重
RANKING_CVR_WEIGHT=0.5

# Python 内存管理优化
PYTHONHASHSEED=0
MALLOC_TRIM_THRESHOLD_=100000
EOF

echo "✓ 已添加内存优化配置"
echo ""

# 显示添加的配置
echo "========================================="
echo "新添加的配置:"
echo "========================================="
tail -n 18 "$ENV_FILE"
echo ""

# 验证配置
echo "========================================="
echo "验证配置"
echo "========================================="

SIMILARITY_BATCH_SIZE=$(grep "^SIMILARITY_BATCH_SIZE=" "$ENV_FILE" | cut -d'=' -f2)
SIMILARITY_TOP_K=$(grep "^SIMILARITY_TOP_K=" "$ENV_FILE" | cut -d'=' -f2)
USE_FAISS_RECALL=$(grep "^USE_FAISS_RECALL=" "$ENV_FILE" | cut -d'=' -f2)

echo "✓ SIMILARITY_BATCH_SIZE = $SIMILARITY_BATCH_SIZE"
echo "✓ SIMILARITY_TOP_K = $SIMILARITY_TOP_K"
echo "✓ USE_FAISS_RECALL = $USE_FAISS_RECALL"
echo ""

# 询问是否重启服务
echo "========================================="
echo "下一步"
echo "========================================="
echo ""
echo "配置已添加，但需要重启 Docker 服务才能生效。"
echo ""
read -p "是否现在重启 Docker 服务? (y/N): " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "正在重启 Docker 服务..."
    cd "$PROJECT_DIR"

    echo "1. 停止服务..."
    docker-compose down

    echo ""
    echo "2. 启动服务..."
    docker-compose up -d

    echo ""
    echo "3. 等待服务启动..."
    sleep 10

    echo ""
    echo "4. 检查服务状态..."
    docker-compose ps

    echo ""
    echo "========================================="
    echo "✓ 服务重启完成"
    echo "========================================="
    echo ""
    echo "查看优化日志:"
    echo "  docker-compose logs -f airflow-scheduler | grep -i memory"
    echo ""
    echo "查看容器内存使用:"
    echo "  docker stats --no-stream"
    echo ""
else
    echo ""
    echo "========================================="
    echo "配置已添加，但服务未重启"
    echo "========================================="
    echo ""
    echo "手动重启服务命令:"
    echo "  cd $PROJECT_DIR"
    echo "  docker-compose down"
    echo "  docker-compose up -d"
    echo ""
fi

echo "完成！"
