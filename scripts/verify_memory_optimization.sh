#!/bin/bash
# 验证生产环境内存优化是否生效
# 使用方法: bash scripts/verify_memory_optimization.sh

set -e

echo "=============================================="
echo "内存优化验证脚本"
echo "=============================================="
echo ""

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 1. 检查环境变量配置
echo -e "${BLUE}[检查 1/4]${NC} 环境变量配置"
echo "---"

ENV_FILE=".env"
[ ! -f "$ENV_FILE" ] && ENV_FILE=".env.prod"

if [ -f "$ENV_FILE" ]; then
    MICRO_BATCH=$(grep "^SIMILARITY_MICRO_BATCH_SIZE=" "$ENV_FILE" | cut -d'=' -f2)
    TOP_K=$(grep "^SIMILARITY_TOP_K=" "$ENV_FILE" | cut -d'=' -f2)

    if [ -n "$MICRO_BATCH" ] && [ -n "$TOP_K" ]; then
        echo -e "${GREEN}✓ 配置存在${NC}"
        echo "  SIMILARITY_MICRO_BATCH_SIZE=$MICRO_BATCH"
        echo "  SIMILARITY_TOP_K=$TOP_K"

        if [ "$MICRO_BATCH" -le 100 ]; then
            echo -e "${GREEN}✓ 微批大小合理 (<= 100)${NC}"
        else
            echo -e "${YELLOW}⚠ 微批大小偏大 (> 100)，可能仍有内存压力${NC}"
        fi
    else
        echo -e "${RED}✗ 未找到优化配置${NC}"
        echo "  请运行: bash scripts/apply_production_memory_fix.sh"
    fi
else
    echo -e "${RED}✗ 未找到环境文件${NC}"
fi

echo ""

# 2. 检查代码版本
echo -e "${BLUE}[检查 2/4]${NC} 代码优化版本"
echo "---"

if grep -q "MEMORY-OPTIMIZED: Process row-by-row" pipeline/memory_optimizer.py; then
    echo -e "${GREEN}✓ 代码已更新为逐行流处理版本${NC}"

    # 检查关键特征
    if grep -q "micro_batch_size = int(os.getenv" pipeline/memory_optimizer.py; then
        echo -e "${GREEN}✓ 支持环境变量配置${NC}"
    fi

    if grep -q "for idx in range(batch_start, batch_end):" pipeline/memory_optimizer.py; then
        echo -e "${GREEN}✓ 实现了逐行处理${NC}"
    fi
else
    echo -e "${RED}✗ 代码未更新${NC}"
    echo "  请拉取最新代码: git pull origin master"
fi

echo ""

# 3. 检查 Airflow 服务状态
echo -e "${BLUE}[检查 3/4]${NC} Airflow 服务状态"
echo "---"

if docker ps | grep -q airflow-scheduler; then
    echo -e "${GREEN}✓ Airflow 调度器正在运行${NC}"

    # 检查容器内存限制
    MEM_LIMIT=$(docker inspect airflow-scheduler --format='{{.HostConfig.Memory}}' 2>/dev/null || echo "0")
    if [ "$MEM_LIMIT" = "0" ]; then
        echo -e "${YELLOW}⚠ 未设置容器内存限制${NC}"
        echo "  建议在 docker-compose.yml 中添加 memory limit"
    else
        MEM_GB=$((MEM_LIMIT / 1024 / 1024 / 1024))
        echo -e "${GREEN}✓ 容器内存限制: ${MEM_GB}GB${NC}"
    fi

    # 显示当前内存使用
    echo ""
    echo "当前内存使用:"
    docker stats --no-stream airflow-scheduler | awk 'NR==1 || NR==2 {print "  " $0}'
else
    echo -e "${RED}✗ Airflow 调度器未运行${NC}"
    echo "  请启动服务: docker-compose up -d"
fi

echo ""

# 4. 检查最近的运行日志
echo -e "${BLUE}[检查 4/4]${NC} 最近的运行日志"
echo "---"

if docker ps | grep -q airflow-scheduler; then
    # 检查是否有新版本的日志输出
    if docker logs airflow-scheduler 2>&1 | grep -q "micro-batch"; then
        echo -e "${GREEN}✓ 发现优化版本的日志输出${NC}"
        echo ""
        echo "最近的相似度计算日志:"
        docker logs airflow-scheduler 2>&1 | grep -E "micro-batch|Progress:" | tail -5 | sed 's/^/  /'
    else
        echo -e "${YELLOW}⚠ 未发现优化版本的日志${NC}"
        echo "  可能原因:"
        echo "  1. 模型训练任务尚未运行"
        echo "  2. 服务重启后尚未触发训练"
        echo "  3. 日志已轮换"
        echo ""
        echo "  手动触发训练任务:"
        echo "    docker exec airflow-scheduler airflow dags trigger incremental_data_update"
    fi

    # 检查是否有 OOM 错误
    if docker logs airflow-scheduler 2>&1 | tail -100 | grep -qi "oom\|out of memory\|killed"; then
        echo ""
        echo -e "${RED}✗ 发现内存不足错误${NC}"
        echo "  最近的错误日志:"
        docker logs airflow-scheduler 2>&1 | grep -i "oom\|out of memory\|killed" | tail -3 | sed 's/^/  /'
    else
        echo -e "${GREEN}✓ 最近 100 条日志无 OOM 错误${NC}"
    fi
else
    echo -e "${YELLOW}⚠ 无法检查日志（服务未运行）${NC}"
fi

echo ""
echo "=============================================="
echo -e "${BLUE}验证总结${NC}"
echo "=============================================="
echo ""

# 综合评分
SCORE=0
MAX_SCORE=4

# 评分逻辑
if [ -n "$MICRO_BATCH" ] && [ -n "$TOP_K" ]; then
    SCORE=$((SCORE + 1))
fi

if grep -q "MEMORY-OPTIMIZED: Process row-by-row" pipeline/memory_optimizer.py; then
    SCORE=$((SCORE + 1))
fi

if docker ps | grep -q airflow-scheduler; then
    SCORE=$((SCORE + 1))
fi

if docker logs airflow-scheduler 2>&1 | grep -q "micro-batch"; then
    SCORE=$((SCORE + 1))
fi

# 显示结果
echo "优化完成度: $SCORE / $MAX_SCORE"
echo ""

if [ $SCORE -eq $MAX_SCORE ]; then
    echo -e "${GREEN}✓ 优化完全部署！${NC}"
    echo ""
    echo "建议:"
    echo "  1. 持续监控内存使用: watch -n 5 'docker stats --no-stream'"
    echo "  2. 关注任务执行时间是否在可接受范围内"
    echo "  3. 查看详细文档: docs/PRODUCTION_MEMORY_TUNING.md"
elif [ $SCORE -ge 2 ]; then
    echo -e "${YELLOW}⚠ 优化部分完成${NC}"
    echo ""
    echo "剩余步骤:"
    [ -z "$MICRO_BATCH" ] && echo "  - 配置环境变量: bash scripts/apply_production_memory_fix.sh"
    ! grep -q "MEMORY-OPTIMIZED: Process row-by-row" pipeline/memory_optimizer.py && echo "  - 更新代码: git pull origin master"
    ! docker ps | grep -q airflow-scheduler && echo "  - 启动服务: docker-compose up -d"
else
    echo -e "${RED}✗ 优化尚未部署${NC}"
    echo ""
    echo "快速部署:"
    echo "  bash scripts/apply_production_memory_fix.sh"
fi

echo ""
echo "=============================================="
