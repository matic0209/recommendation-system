#!/bin/bash
# restart_service.sh - Restart recommendation-api service if model changed
#
# 由 Airflow 流水线在训练结束后调用（task: restart_if_changed）。
# 作用：模型/索引发生变化时重启推荐服务，让它重新加载数据快照（否则服务会一直
#       用启动时的旧快照，这正是 2026-03~09 “看了又看”退化的原因之一）。
#
# 功能：
# - 通过模型/索引文件哈希判断是否真的需要重启
# - 重启前备份、重启后健康检查、失败自动回滚
#
# ===== 2026-09-16 修复记录（实测发现的三处硬伤）=====
# 1) MODEL_FILE 原为 models/ranker.lgb —— 该文件从来不存在，
#    导致哈希只覆盖了 item_to_tags_enhanced.json，模型换版也可能不触发重启。
#    现改为 models/rank_model.pkl，并把类别索引 category_index.json 一并纳入哈希。
# 2) 冒烟测试原调用 POST /recommend —— 该接口不存在，健康检查必然失败，
#    会走"回滚"分支并最终以退出码 1 结束（任务失败）。现改为校验 /health 里的
#    models_loaded 字段（推荐服务真实的健康信号）。
# 3) 重启依赖 docker CLI，但 airflow 容器里通常没装 docker 命令 —— 即使
#    .sh 模板问题修好，这一步也会失败。现增加"通过 docker.sock 直接调用
#    Docker API"的兜底（纯标准库，无额外依赖；compose 已把 socket 挂进调度器）。
#
# 注意：本脚本的 bash_command 调用方必须以 "; exit $?" 结尾（见 DAG 注释），
#       否则 Airflow 会把以 .sh 结尾的命令当模板文件解析而报 TemplateNotFound。

set -e

# Configuration
MODELS_DIR="/opt/recommend/models"
BACKUP_DIR="/opt/recommend/models/backup"
HASH_FILE="${MODELS_DIR}/.model_hash"
MODEL_FILE="${MODELS_DIR}/rank_model.pkl"                # 排序模型（原 ranker.lgb 不存在）
TAGS_FILE="${MODELS_DIR}/item_to_tags_enhanced.json"     # 标签索引（服务启动时加载）
CATEGORY_FILE="${MODELS_DIR}/category_index.json"        # 类别索引（决定能否"认识"新数据集）
# ★ 服务在【启动时】加载的全部产物：任何一项变化都必须重启服务才能生效。
#   清单来源：逐个核对 app/main.py 的加载点（早期版本只监控前 3 个，
#   导致 channel_weights.json / item_sim_*.pkl / top_items.json 等更新后服务仍用旧值）。
WATCH_FILES=(
    "$MODEL_FILE"                                  # 排序模型
    "$TAGS_FILE"                                   # 标签倒排（增强版）
    "$CATEGORY_FILE"                               # 12+1 类行业索引
    "${MODELS_DIR}/channel_weights.json"           # 渠道权重（CTR 训练每日重算）
    "${MODELS_DIR}/item_sim_content.pkl"           # 内容相似（标签/描述）
    "${MODELS_DIR}/item_sim_behavior.pkl"          # 行为相似（协同）
    "${MODELS_DIR}/user_similarity.pkl"            # 用户相似
    "${MODELS_DIR}/top_items.json"                 # 全局热门榜（兜底榜数据源）
    "${MODELS_DIR}/tag_to_items_enhanced.json"     # 标签→items 倒排（增强版）
    "${MODELS_DIR}/price_bucket_index.json"        # 价格分桶索引
)
API_URL="http://recommendation-api:8000"
CONTAINER_NAME="recommendation-api"
DOCKER_SOCK="/var/run/docker.sock"
MAX_RETRIES=5
RETRY_INTERVAL=10
STARTUP_WAIT=30

# Logging
log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] $1"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] $1" >&2
}

log_warn() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [WARN] $1"
}

# Calculate hash of model/index files
calculate_model_hash() {
    local hash=""
    for f in "${WATCH_FILES[@]}"; do
        if [ -f "$f" ]; then
            hash="${hash}$(md5sum "$f" | cut -d' ' -f1)"
        fi
    done
    echo "$hash"
}

# Check if model has changed
# 返回 0 = 有变化（需要重启）；1 = 无变化（跳过）
check_model_changed() {
    local old_hash=""
    local new_hash=""

    if [ -f "$HASH_FILE" ]; then
        old_hash=$(cat "$HASH_FILE")
    fi

    new_hash=$(calculate_model_hash)

    if [ -z "$new_hash" ]; then
        log_error "No model/index files found; watched: ${WATCH_FILES[*]}"
        return 1
    fi

    if [ "$old_hash" = "$new_hash" ]; then
        log_info "Model unchanged (hash: ${new_hash:0:16}...)"
        return 1
    else
        log_info "Model changed!"
        log_info "  Old hash: ${old_hash:0:16}..."
        log_info "  New hash: ${new_hash:0:16}..."
        return 0
    fi
}

# Backup current model
backup_model() {
    log_info "Backing up current model..."
    mkdir -p "$BACKUP_DIR"

    for f in "${WATCH_FILES[@]}"; do
        if [ -f "$f" ]; then
            cp "$f" "${BACKUP_DIR}/$(basename "$f").bak"
        fi
    done
    if [ -f "$HASH_FILE" ]; then
        cp "$HASH_FILE" "${BACKUP_DIR}/.model_hash.bak"
    fi

    log_info "Backup completed"
}

# Restore from backup
restore_backup() {
    log_warn "Restoring from backup..."

    for f in "${WATCH_FILES[@]}"; do
        if [ -f "${BACKUP_DIR}/$(basename "$f").bak" ]; then
            cp "${BACKUP_DIR}/$(basename "$f").bak" "$f"
        fi
    done
    if [ -f "${BACKUP_DIR}/.model_hash.bak" ]; then
        cp "${BACKUP_DIR}/.model_hash.bak" "$HASH_FILE"
    fi

    log_warn "Backup restored"
}

# Health check：/health 返回 {"status":"healthy",...,"models_loaded":true}
health_check() {
    log_info "Running health check..."

    for i in $(seq 1 $MAX_RETRIES); do
        log_info "Health check attempt $i/$MAX_RETRIES..."

        HEALTH_BODY=$(curl -sf "${API_URL}/health" 2>&1 || echo "FAILED")
        if echo "$HEALTH_BODY" | grep -qE '"models_loaded"[[:space:]]*:[[:space:]]*true'; then
            log_info "Health check passed (models_loaded=true)"
            return 0
        fi
        log_warn "Health endpoint not ready: $HEALTH_BODY"

        if [ $i -lt $MAX_RETRIES ]; then
            log_info "Retrying in ${RETRY_INTERVAL}s..."
            sleep $RETRY_INTERVAL
        fi
    done

    log_error "Health check failed after $MAX_RETRIES attempts!"
    return 1
}

# Restart service
restart_service() {
    log_info "Restarting ${CONTAINER_NAME} service..."

    if command -v docker &> /dev/null; then
        docker restart "$CONTAINER_NAME"
    elif [ -S "$DOCKER_SOCK" ]; then
        # 容器内通常没有 docker CLI，改为通过挂载进来的 docker.sock 直接调用 Docker API
        log_info "docker CLI not found; using Docker API over $DOCKER_SOCK"
        python3 - <<'PY' || { log_error "Restart via Docker socket failed"; return 1; }
import http.client
import socket
import sys

class UnixHTTPConnection(http.client.HTTPConnection):
    def __init__(self, socket_path):
        super().__init__("localhost")
        self._socket_path = socket_path

    def connect(self):
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(60)
        sock.connect(self._socket_path)
        self.sock = sock

try:
    conn = UnixHTTPConnection("/var/run/docker.sock")
    conn.request("POST", "/containers/recommendation-api/restart?t=10")
    resp = conn.getresponse()
    print("Docker API resp: %s %s" % (resp.status, resp.reason))
    sys.exit(0 if resp.status in (204, 304) else 1)
except Exception as exc:  # noqa: BLE001
    print("Docker API error: %s" % exc)
    sys.exit(1)
PY
    else
        log_error "Neither docker CLI nor $DOCKER_SOCK is available; cannot restart service"
        return 1
    fi

    log_info "Waiting ${STARTUP_WAIT}s for service to start..."
    sleep $STARTUP_WAIT
}

# Save new hash
save_hash() {
    local new_hash
    new_hash=$(calculate_model_hash)
    echo "$new_hash" > "$HASH_FILE"
    log_info "Saved new model hash"
}

# Main
main() {
    log_info "============================================================"
    log_info "Service Restart Check"
    log_info "============================================================"

    # Check if model changed
    if ! check_model_changed; then
        log_info "No restart needed"
        log_info "============================================================"
        exit 0
    fi

    # Model changed, proceed with restart
    log_info "Proceeding with service restart..."

    # Backup current model
    backup_model

    # Restart service
    if ! restart_service; then
        log_error "Failed to restart service"
        exit 1
    fi

    # Health check
    if health_check; then
        # Success - save new hash
        save_hash
        log_info "============================================================"
        log_info "Service restarted successfully!"
        log_info "============================================================"
        exit 0
    else
        # Failed - rollback
        log_error "Health check failed, rolling back..."
        restore_backup

        # Restart again with old model
        restart_service

        if health_check; then
            log_warn "Rollback successful, service running with previous model"
        else
            log_error "CRITICAL: Rollback failed! Manual intervention required!"
        fi

        log_info "============================================================"
        exit 1
    fi
}

main "$@"
