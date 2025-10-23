#!/bin/bash
# 推荐系统迁移打包脚本
# 使用方法：bash scripts/package_for_migration.sh [full|minimal|models]

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查是否在项目根目录
if [ ! -f "docker-compose.yml" ]; then
    print_error "请在项目根目录运行此脚本"
    exit 1
fi

# 获取打包模式
MODE=${1:-full}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="/tmp/recommend_migration"

mkdir -p "$OUTPUT_DIR"

print_info "开始打包推荐系统（模式: $MODE）"
print_info "时间戳: $TIMESTAMP"

case $MODE in
    full)
        print_info "完整打包模式 - 包含代码、数据、模型"
        PACKAGE_NAME="recommend_full_${TIMESTAMP}.tar.gz"

        tar -czf "$OUTPUT_DIR/$PACKAGE_NAME" \
            --exclude='venv' \
            --exclude='.git' \
            --exclude='logs/*.log' \
            --exclude='mlruns' \
            --exclude='__pycache__' \
            --exclude='*.pyc' \
            --exclude='.pytest_cache' \
            --exclude='notification_gateway/webhook.log' \
            -C .. $(basename $PWD)
        ;;

    minimal)
        print_info "最小打包模式 - 仅代码和 JSON 数据"
        PACKAGE_NAME="recommend_minimal_${TIMESTAMP}.tar.gz"

        tar -czf "$OUTPUT_DIR/$PACKAGE_NAME" \
            --exclude='venv' \
            --exclude='.git' \
            --exclude='logs' \
            --exclude='mlruns' \
            --exclude='models' \
            --exclude='data/business' \
            --exclude='data/cleaned' \
            --exclude='data/processed' \
            --exclude='data/evaluation' \
            --exclude='__pycache__' \
            --exclude='*.pyc' \
            --exclude='.pytest_cache' \
            -C .. $(basename $PWD)
        ;;

    models)
        print_info "仅模型打包模式 - 模型 + 配置"
        PACKAGE_NAME="recommend_models_${TIMESTAMP}.tar.gz"

        if [ ! -d "models" ]; then
            print_error "models 目录不存在，请先运行 Pipeline"
            exit 1
        fi

        tar -czf "$OUTPUT_DIR/$PACKAGE_NAME" \
            models/ \
            .env \
            .env.example
        ;;

    *)
        print_error "未知模式: $MODE"
        echo "使用方法: $0 [full|minimal|models]"
        echo "  full    - 完整打包（代码+数据+模型）"
        echo "  minimal - 最小打包（代码+JSON数据）"
        echo "  models  - 仅打包模型和配置"
        exit 1
        ;;
esac

# 获取文件大小
PACKAGE_PATH="$OUTPUT_DIR/$PACKAGE_NAME"
FILE_SIZE=$(du -h "$PACKAGE_PATH" | cut -f1)

print_info "打包完成！"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  打包文件: $PACKAGE_NAME"
echo "  文件路径: $PACKAGE_PATH"
echo "  文件大小: $FILE_SIZE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 生成 MD5 校验和
print_info "生成 MD5 校验和..."
md5sum "$PACKAGE_PATH" > "$PACKAGE_PATH.md5"
print_info "MD5: $(cat $PACKAGE_PATH.md5)"
echo ""

# 打印后续步骤
print_info "后续步骤："
echo "1. 传输到新服务器："
echo "   scp $PACKAGE_PATH user@new-server:/tmp/"
echo ""
echo "2. 验证文件完整性："
echo "   md5sum -c $PACKAGE_NAME.md5"
echo ""
echo "3. 在新服务器上解压："

if [ "$MODE" = "full" ] || [ "$MODE" = "minimal" ]; then
    echo "   sudo tar -xzf /tmp/$PACKAGE_NAME -C /opt/"
    echo "   sudo chown -R \$USER:\$USER /opt/recommend"
    echo "   cd /opt/recommend"
elif [ "$MODE" = "models" ]; then
    echo "   cd /opt/recommend"
    echo "   tar -xzf /tmp/$PACKAGE_NAME"
fi

echo ""

if [ "$MODE" = "minimal" ]; then
    print_warn "注意：最小打包模式需要在新服务器上运行 Pipeline"
    echo "   export PYTHONPATH=/opt/recommend:\$PYTHONPATH"
    echo "   bash scripts/run_pipeline.sh"
    echo ""
fi

print_info "详细迁移指南请参考: docs/MIGRATION_GUIDE.md"

# 列出打包内容预览
print_info "打包内容预览（前 20 行）："
tar -tzf "$PACKAGE_PATH" | head -20
echo "..."
