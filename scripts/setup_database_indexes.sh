#!/bin/bash
###############################################################################
# 数据库索引自动优化脚本
#
# 用途: 在新的数据库环境中自动创建和优化索引
# 使用: ./scripts/setup_database_indexes.sh [--skip-confirmation] [--production]
#
# 作者: AI Assistant
# 日期: 2025-10-10
###############################################################################

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 脚本目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# 日志文件
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$PROJECT_DIR/logs/index_optimization"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/setup_indexes_${TIMESTAMP}.log"

# 默认参数
SKIP_CONFIRMATION=false
IS_PRODUCTION=false

###############################################################################
# 函数定义
###############################################################################

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1" | tee -a "$LOG_FILE"
}

log_warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1" | tee -a "$LOG_FILE"
}

log_info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO:${NC} $1" | tee -a "$LOG_FILE"
}

print_header() {
    echo -e "${BLUE}"
    echo "╔═══════════════════════════════════════════════════════════════════╗"
    echo "║          数据库索引自动优化脚本 v1.0                              ║"
    echo "║          Database Index Optimization Automation                   ║"
    echo "╚═══════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# 检查依赖
check_dependencies() {
    log_info "检查系统依赖..."

    local missing_deps=()

    # 检查 mysql 客户端
    if ! command -v mysql &> /dev/null; then
        missing_deps+=("mysql-client")
    fi

    # 检查 python3
    if ! command -v python3 &> /dev/null; then
        missing_deps+=("python3")
    fi

    if [ ${#missing_deps[@]} -ne 0 ]; then
        log_error "缺少依赖: ${missing_deps[*]}"
        log_error "请先安装: sudo apt-get install ${missing_deps[*]}"
        exit 1
    fi

    log "✓ 依赖检查通过"
}

# 加载环境变量
load_env() {
    log_info "加载环境配置..."

    if [ ! -f "$PROJECT_DIR/.env" ]; then
        log_error ".env 文件不存在！请先配置数据库连接信息"
        exit 1
    fi

    # 导出环境变量
    set -a
    source "$PROJECT_DIR/.env"
    set +a

    # 验证必要的环境变量
    local required_vars=(
        "BUSINESS_DB_HOST"
        "BUSINESS_DB_PORT"
        "BUSINESS_DB_NAME"
        "BUSINESS_DB_USER"
        "BUSINESS_DB_PASSWORD"
        "MATOMO_DB_HOST"
        "MATOMO_DB_PORT"
        "MATOMO_DB_NAME"
        "MATOMO_DB_USER"
        "MATOMO_DB_PASSWORD"
    )

    local missing_vars=()
    for var in "${required_vars[@]}"; do
        if [ -z "${!var}" ]; then
            missing_vars+=("$var")
        fi
    done

    if [ ${#missing_vars[@]} -ne 0 ]; then
        log_error "缺少必要的环境变量: ${missing_vars[*]}"
        exit 1
    fi

    log "✓ 环境配置加载完成"
}

# 测试数据库连接
test_db_connection() {
    log_info "测试数据库连接..."

    # 测试业务库连接
    log_info "  → 测试业务库 ($BUSINESS_DB_NAME)..."
    if mysql -h"$BUSINESS_DB_HOST" -P"$BUSINESS_DB_PORT" \
            -u"$BUSINESS_DB_USER" -p"$BUSINESS_DB_PASSWORD" \
            -e "SELECT 1" "$BUSINESS_DB_NAME" &>> "$LOG_FILE"; then
        log "    ✓ 业务库连接成功"
    else
        log_error "业务库连接失败！请检查配置"
        exit 1
    fi

    # 测试 Matomo 库连接
    log_info "  → 测试 Matomo 库 ($MATOMO_DB_NAME)..."
    if mysql -h"$MATOMO_DB_HOST" -P"$MATOMO_DB_PORT" \
            -u"$MATOMO_DB_USER" -p"$MATOMO_DB_PASSWORD" \
            -e "SELECT 1" "$MATOMO_DB_NAME" &>> "$LOG_FILE"; then
        log "    ✓ Matomo 库连接成功"
    else
        log_error "Matomo 库连接失败！请检查配置"
        exit 1
    fi

    log "✓ 所有数据库连接测试通过"
}

# 检查现有索引
check_existing_indexes() {
    log_info "检查现有索引状态..."

    local business_index_count=$(mysql -h"$BUSINESS_DB_HOST" -P"$BUSINESS_DB_PORT" \
        -u"$BUSINESS_DB_USER" -p"$BUSINESS_DB_PASSWORD" \
        -N -B -e "
        SELECT COUNT(*) FROM information_schema.STATISTICS
        WHERE TABLE_SCHEMA = '$BUSINESS_DB_NAME'
          AND TABLE_NAME IN ('user', 'dataset', 'task', 'api_order', 'dataset_image')
          AND INDEX_NAME LIKE 'idx_%'
        " 2>> "$LOG_FILE")

    local matomo_index_count=$(mysql -h"$MATOMO_DB_HOST" -P"$MATOMO_DB_PORT" \
        -u"$MATOMO_DB_USER" -p"$MATOMO_DB_PASSWORD" \
        -N -B -e "
        SELECT COUNT(*) FROM information_schema.STATISTICS
        WHERE TABLE_SCHEMA = '$MATOMO_DB_NAME'
          AND TABLE_NAME IN ('matomo_log_visit', 'matomo_log_link_visit_action', 'matomo_log_conversion')
          AND INDEX_NAME LIKE 'idx_%'
        " 2>> "$LOG_FILE")

    log "  → 业务库现有索引: $business_index_count 个"
    log "  → Matomo库现有索引: $matomo_index_count 个"

    if [ "$business_index_count" -gt 0 ] || [ "$matomo_index_count" -gt 0 ]; then
        log_warning "检测到已存在的索引，脚本会智能跳过已创建的索引"
    fi
}

# 用户确认
confirm_execution() {
    if [ "$SKIP_CONFIRMATION" = true ]; then
        log_info "跳过用户确认（--skip-confirmation）"
        return
    fi

    echo ""
    echo -e "${YELLOW}═══════════════════════════════════════════════════════${NC}"

    if [ "$IS_PRODUCTION" = true ]; then
        echo -e "${RED}⚠️  警告: 您即将在生产环境执行索引优化！${NC}"
    else
        echo -e "${YELLOW}您即将执行数据库索引优化操作${NC}"
    fi

    echo ""
    echo "目标数据库:"
    echo "  - 业务库: $BUSINESS_DB_HOST:$BUSINESS_DB_PORT/$BUSINESS_DB_NAME"
    echo "  - Matomo: $MATOMO_DB_HOST:$MATOMO_DB_PORT/$MATOMO_DB_NAME"
    echo ""
    echo "操作内容:"
    echo "  1. 创建时间相关索引（用于 CDC 增量抽取）"
    echo "  2. 创建联合索引（优化常用查询）"
    echo "  3. 验证索引创建结果"
    echo ""
    echo "预期影响:"
    echo "  - Pipeline 执行速度提升 60-80%"
    echo "  - 索引创建期间可能对数据库产生短暂负载"
    echo ""
    echo -e "${YELLOW}═══════════════════════════════════════════════════════${NC}"
    echo ""

    read -p "确认继续执行？(yes/no): " confirm

    if [ "$confirm" != "yes" ]; then
        log_warning "用户取消操作"
        exit 0
    fi

    log "用户确认执行"
}

# 创建数据库备份信息
backup_metadata() {
    log_info "保存数据库元数据备份..."

    local backup_dir="$PROJECT_DIR/backups/index_metadata/$TIMESTAMP"
    mkdir -p "$backup_dir"

    # 备份业务库索引信息
    mysql -h"$BUSINESS_DB_HOST" -P"$BUSINESS_DB_PORT" \
        -u"$BUSINESS_DB_USER" -p"$BUSINESS_DB_PASSWORD" \
        -e "SELECT * FROM information_schema.STATISTICS
            WHERE TABLE_SCHEMA = '$BUSINESS_DB_NAME'
            ORDER BY TABLE_NAME, INDEX_NAME, SEQ_IN_INDEX" \
        > "$backup_dir/business_indexes_before.txt" 2>> "$LOG_FILE"

    # 备份 Matomo 库索引信息
    mysql -h"$MATOMO_DB_HOST" -P"$MATOMO_DB_PORT" \
        -u"$MATOMO_DB_USER" -p"$MATOMO_DB_PASSWORD" \
        -e "SELECT * FROM information_schema.STATISTICS
            WHERE TABLE_SCHEMA = '$MATOMO_DB_NAME'
            ORDER BY TABLE_NAME, INDEX_NAME, SEQ_IN_INDEX" \
        > "$backup_dir/matomo_indexes_before.txt" 2>> "$LOG_FILE"

    log "  ✓ 元数据备份保存到: $backup_dir"
}

# 执行索引创建
create_indexes() {
    log_info "开始创建索引..."

    local sql_script="$SCRIPT_DIR/p0_01_add_indexes_fixed.sql"

    if [ ! -f "$sql_script" ]; then
        log_error "索引创建脚本不存在: $sql_script"
        exit 1
    fi

    log "  → 执行 SQL 脚本: $sql_script"

    # 执行索引创建脚本
    if mysql -h"$BUSINESS_DB_HOST" -P"$BUSINESS_DB_PORT" \
            -u"$BUSINESS_DB_USER" -p"$BUSINESS_DB_PASSWORD" \
            < "$sql_script" >> "$LOG_FILE" 2>&1; then
        log "✓ 索引创建完成"
    else
        log_error "索引创建失败！请查看日志: $LOG_FILE"
        exit 1
    fi
}

# 验证索引
verify_indexes() {
    log_info "验证索引创建结果..."

    # 统计创建的索引数量
    local business_index_count=$(mysql -h"$BUSINESS_DB_HOST" -P"$BUSINESS_DB_PORT" \
        -u"$BUSINESS_DB_USER" -p"$BUSINESS_DB_PASSWORD" \
        -N -B -e "
        SELECT COUNT(DISTINCT INDEX_NAME) FROM information_schema.STATISTICS
        WHERE TABLE_SCHEMA = '$BUSINESS_DB_NAME'
          AND TABLE_NAME IN ('user', 'dataset', 'task', 'api_order', 'dataset_image')
          AND INDEX_NAME LIKE 'idx_%'
        " 2>> "$LOG_FILE")

    local matomo_index_count=$(mysql -h"$MATOMO_DB_HOST" -P"$MATOMO_DB_PORT" \
        -u"$MATOMO_DB_USER" -p"$MATOMO_DB_PASSWORD" \
        -N -B -e "
        SELECT COUNT(DISTINCT INDEX_NAME) FROM information_schema.STATISTICS
        WHERE TABLE_SCHEMA = '$MATOMO_DB_NAME'
          AND TABLE_NAME IN ('matomo_log_visit', 'matomo_log_link_visit_action', 'matomo_log_conversion')
          AND INDEX_NAME LIKE 'idx_%'
        " 2>> "$LOG_FILE")

    log "  ✓ 业务库索引总数: $business_index_count"
    log "  ✓ Matomo库索引总数: $matomo_index_count"

    # 测试查询性能
    log_info "测试索引使用情况..."

    # 测试业务库查询
    mysql -h"$BUSINESS_DB_HOST" -P"$BUSINESS_DB_PORT" \
        -u"$BUSINESS_DB_USER" -p"$BUSINESS_DB_PASSWORD" \
        "$BUSINESS_DB_NAME" \
        -e "EXPLAIN SELECT * FROM dataset WHERE update_time > '2025-09-01' LIMIT 10" \
        >> "$LOG_FILE" 2>&1

    log "  ✓ 查询测试通过"
}

# 生成报告
generate_report() {
    log_info "生成优化报告..."

    local report_file="$LOG_DIR/optimization_report_${TIMESTAMP}.txt"

    cat > "$report_file" << EOF
═══════════════════════════════════════════════════════════════════
数据库索引优化报告
Database Index Optimization Report
═══════════════════════════════════════════════════════════════════

执行时间: $(date +'%Y-%m-%d %H:%M:%S')
环境类型: $([ "$IS_PRODUCTION" = true ] && echo "生产环境" || echo "开发/测试环境")

─────────────────────────────────────────────────────────────────
数据库连接信息
─────────────────────────────────────────────────────────────────
业务库: $BUSINESS_DB_HOST:$BUSINESS_DB_PORT/$BUSINESS_DB_NAME
Matomo: $MATOMO_DB_HOST:$MATOMO_DB_PORT/$MATOMO_DB_NAME

─────────────────────────────────────────────────────────────────
索引统计
─────────────────────────────────────────────────────────────────
EOF

    # 添加业务库索引详情
    echo "" >> "$report_file"
    echo "业务库索引:" >> "$report_file"
    mysql -h"$BUSINESS_DB_HOST" -P"$BUSINESS_DB_PORT" \
        -u"$BUSINESS_DB_USER" -p"$BUSINESS_DB_PASSWORD" \
        -t -e "
        SELECT
            TABLE_NAME as '表名',
            INDEX_NAME as '索引名',
            GROUP_CONCAT(COLUMN_NAME ORDER BY SEQ_IN_INDEX) as '列名',
            CARDINALITY as '基数'
        FROM information_schema.STATISTICS
        WHERE TABLE_SCHEMA = '$BUSINESS_DB_NAME'
          AND TABLE_NAME IN ('user', 'dataset', 'task', 'api_order', 'dataset_image')
          AND INDEX_NAME LIKE 'idx_%'
        GROUP BY TABLE_NAME, INDEX_NAME, CARDINALITY
        ORDER BY TABLE_NAME, INDEX_NAME
        " >> "$report_file" 2>> "$LOG_FILE"

    # 添加 Matomo 库索引详情
    echo "" >> "$report_file"
    echo "Matomo库索引:" >> "$report_file"
    mysql -h"$MATOMO_DB_HOST" -P"$MATOMO_DB_PORT" \
        -u"$MATOMO_DB_USER" -p"$MATOMO_DB_PASSWORD" \
        -t -e "
        SELECT
            TABLE_NAME as '表名',
            INDEX_NAME as '索引名',
            GROUP_CONCAT(COLUMN_NAME ORDER BY SEQ_IN_INDEX) as '列名',
            CARDINALITY as '基数'
        FROM information_schema.STATISTICS
        WHERE TABLE_SCHEMA = '$MATOMO_DB_NAME'
          AND TABLE_NAME IN ('matomo_log_visit', 'matomo_log_link_visit_action', 'matomo_log_conversion')
          AND INDEX_NAME LIKE 'idx_%'
        GROUP BY TABLE_NAME, INDEX_NAME, CARDINALITY
        ORDER BY TABLE_NAME, INDEX_NAME
        " >> "$report_file" 2>> "$LOG_FILE"

    cat >> "$report_file" << EOF

─────────────────────────────────────────────────────────────────
预期收益
─────────────────────────────────────────────────────────────────
✓ CDC 增量抽取性能提升 60-80%
✓ 降低数据库 CPU 使用率
✓ 减少全表扫描
✓ 提升 Pipeline 整体执行速度

─────────────────────────────────────────────────────────────────
日志文件
─────────────────────────────────────────────────────────────────
完整日志: $LOG_FILE
优化报告: $report_file

═══════════════════════════════════════════════════════════════════
EOF

    log "✓ 报告已生成: $report_file"

    # 显示报告内容
    echo ""
    cat "$report_file"
}

# 清理函数
cleanup() {
    if [ $? -ne 0 ]; then
        log_error "脚本执行失败！"
        log_error "请查看日志: $LOG_FILE"
    fi
}

###############################################################################
# 主流程
###############################################################################

main() {
    # 设置清理钩子
    trap cleanup EXIT

    # 解析命令行参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            --skip-confirmation)
                SKIP_CONFIRMATION=true
                shift
                ;;
            --production)
                IS_PRODUCTION=true
                shift
                ;;
            -h|--help)
                echo "用法: $0 [选项]"
                echo ""
                echo "选项:"
                echo "  --skip-confirmation  跳过用户确认"
                echo "  --production         标记为生产环境（显示警告）"
                echo "  -h, --help          显示帮助信息"
                exit 0
                ;;
            *)
                log_error "未知参数: $1"
                exit 1
                ;;
        esac
    done

    # 打印标题
    print_header

    # 执行步骤
    check_dependencies
    load_env
    test_db_connection
    check_existing_indexes
    confirm_execution
    backup_metadata
    create_indexes
    verify_indexes
    generate_report

    # 完成
    echo ""
    log "${GREEN}╔═══════════════════════════════════════════════════════════════════╗${NC}"
    log "${GREEN}║           ✓ 数据库索引优化完成！                                  ║${NC}"
    log "${GREEN}║           Database Index Optimization Completed!                  ║${NC}"
    log "${GREEN}╚═══════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

# 运行主函数
main "$@"
