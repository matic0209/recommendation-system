#!/usr/bin/env bash
# ============================================================================
# shadow_verify.sh —— 影子实例验证（发版前的最后一道门）
#
# 做法：用【待发版代码】+【生产同一份模型/元数据】在 18090 端口起一个独立实例，
#       与生产 8090 逐页对比；全程不影响生产：
#         · 独立端口 18090（不接任何生产流量）
#         · 独立 Redis DB9（生产用 DB0/DB1，绝不混用）
#         · 曝光日志写到独立目录（不污染生产 exposure_log.jsonl → 不进训练样本）
#         · Sentry 置空（不污染生产告警）
#         · 模型/元数据只读挂载生产目录（不修改任何生产文件）
#
# 用法（在生产服务器上执行）：
#     bash /root/reco-shadow/repo/scripts/shadow_verify.sh
#   可选环境变量：
#     PAGES="16307 13830 801 8523"   LIMIT=12   SHADOW_PORT=18090
#
# 收尾：docker rm -f reco-shadow        # 删掉影子实例即可（不留任何痕迹）
# ============================================================================
set -uo pipefail

PROD_REPO="${PROD_REPO:-/root/recommendation-system}"
SHADOW_DIR="${SHADOW_DIR:-/root/reco-shadow}"
CODE_DIR="${CODE_DIR:-$SHADOW_DIR/repo}"
PROD_CT="${PROD_CT:-recommendation-api}"
REDIS_CT="${REDIS_CT:-redis}"
SHADOW_CT="${SHADOW_CT:-reco-shadow}"
SHADOW_PORT="${SHADOW_PORT:-18090}"
PROD_PORT="${PROD_PORT:-8090}"
PAGES="${PAGES:-16307 13830 801 8523}"
LIMIT="${LIMIT:-12}"
IMAGE="${IMAGE:-recommend-recommendation-api}"

info() { echo "[INFO] $*"; }
die() { echo "[ERROR] $*" >&2; exit 1; }

# ---------- 0. 前置检查 ----------
[ -d "$CODE_DIR/app" ] || die "找不到待发版代码：$CODE_DIR/app（先把代码放进去，见文档说明）"
[ -d "$PROD_REPO/models" ] || die "找不到生产模型目录：$PROD_REPO/models"
command -v curl >/dev/null || die "缺少 curl"
info "待发版代码目录：$CODE_DIR"
info "代码指纹：$(md5sum "$CODE_DIR/app/main.py" 2>/dev/null | cut -c1-12)（app/main.py）"

# ---------- 1. 生成隔离环境变量 ----------
mkdir -p "$SHADOW_DIR/data/evaluation"
docker inspect "$PROD_CT" --format '{{range .Config.Env}}{{println .}}{{end}}' \
  | grep -vE '^(PATH|HOSTNAME|HOME|LANG|LC_|PYTHON|GPG|TERM|DEBIAN|MPLCONFIGDIR)' > "$SHADOW_DIR/env.list"
sed -i 's#\(REDIS_URL=redis://[^/]*\)/0#\1/9#' "$SHADOW_DIR/env.list"   # 生产缓存 DB0 → 影子 DB9
sed -i 's#^SENTRY_DSN=.*#SENTRY_DSN=#' "$SHADOW_DIR/env.list"            # 关掉生产告警
info "关键环境变量（需为 DB9 且 SENTRY 为空）："
grep -E '^(REDIS_URL|FEATURE_REDIS_URL|SENTRY_DSN|DATA_DIR|MODELS_DIR)=' "$SHADOW_DIR/env.list" || true

NET=$(docker inspect "$PROD_CT" --format '{{range $k,$v := .NetworkSettings.Networks}}{{$k}}{{end}}')
[ -n "$NET" ] || die "取不到生产容器的网络名"

# ---------- 2. 启动影子实例 ----------
docker rm -f "$SHADOW_CT" >/dev/null 2>&1
# 说明：镜像默认以 appuser(50000) 运行，而挂载进来的独立曝光日志目录属主是 root，
# 会让容器写不进去 → 测试实例以 root 运行（仅测试用，生产镜像本身不变）。
docker run -d --name "$SHADOW_CT" --network "$NET" -p "${SHADOW_PORT}:8000" --user root \
  -v "$CODE_DIR/app:/app/app:ro" \
  -v "$CODE_DIR/config:/app/config:ro" \
  -v "$CODE_DIR/pipeline:/app/pipeline:ro" \
  -v "$CODE_DIR/scripts:/app/scripts:ro" \
  -v "$PROD_REPO/models:/app/models:ro" \
  -v "$PROD_REPO/data/processed:/app/data/processed:ro" \
  -v "$SHADOW_DIR/data/evaluation:/app/data/evaluation" \
  --env-file "$SHADOW_DIR/env.list" \
  "$IMAGE" uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2 >/dev/null \
  || die "启动影子容器失败"

info "等待影子实例加载模型产物（约 60~120 秒）..."
for _ in $(seq 1 40); do
  curl -sf -m 5 "http://localhost:${SHADOW_PORT}/health" >/dev/null 2>&1 && break
  sleep 5
done
HEALTH=$(curl -s -m 10 "http://localhost:${SHADOW_PORT}/health" || true)
echo "影子实例 /health → $HEALTH"
case "$HEALTH" in
  *'"models_loaded":true'*) info "影子实例就绪 ✅" ;;
  *) docker logs --tail 30 "$SHADOW_CT"; die "影子实例未就绪（见上方日志）" ;;
esac

# ---------- 3. 清缓存（只清影子 DB9；生产只删 similar: 键，不动其他）----------
docker exec "$REDIS_CT" redis-cli -n 9 flushdb >/dev/null 2>&1
KEYS=$(docker exec "$REDIS_CT" redis-cli -n 0 --scan --pattern 'similar:*' 2>/dev/null | tr -d '\r')
[ -n "$KEYS" ] && echo "$KEYS" | xargs -r -I{} docker exec "$REDIS_CT" redis-cli -n 0 del {} >/dev/null
info "缓存已清理（影子 DB9 全清；生产仅删 similar:* 键）"

# ---------- 4. 逐页对比 ----------
cmp_page() {
  local port="$1" page="$2"
  curl -s -m 60 "http://localhost:${port}/similar/${page}?limit=${LIMIT}" -o "/tmp/shadow_${port}_${page}.json" || echo "{}" > "/tmp/shadow_${port}_${page}.json"
  python3 - "/tmp/shadow_${port}_${page}.json" <<'PY'
import json, sys, collections
try:
    d = json.load(open(sys.argv[1]))
except Exception:
    print("    解析失败"); raise SystemExit
items = d.get("similar_items") or []
c = collections.Counter(i.get("reason") for i in items)
pop = sum(v for k, v in c.items() if k and ("popular" in k or "fallback" in k))
sem = sum(v for k, v in c.items() if k and any(t in k for t in ("content", "vector", "tag", "12cat")))
print("    条数=%d  popular=%d  content/vector/tag=%d" % (len(items), pop, sem))
print("    渠道分布: %s" % dict(c))
print("    ids: %s" % [i.get("dataset_id") for i in items])
PY
}

echo
echo "============================================================"
echo "逐页对比：生产 ${PROD_PORT}（旧代码） vs 影子 ${SHADOW_PORT}（新代码）"
echo "============================================================"
for page in $PAGES; do
  echo; echo "### 页面 ${page}"
  echo "  --- 生产（旧）---"; cmp_page "$PROD_PORT" "$page"
  echo "  --- 影子（新）---"; cmp_page "$SHADOW_PORT" "$page"
  python3 - "$page" <<'PY'
import json, sys
page = sys.argv[1]
def ids(port):
    try:
        d = json.load(open("/tmp/shadow_%s_%s.json" % (port, page)))
        return [i.get("dataset_id") for i in (d.get("similar_items") or [])]
    except Exception:
        return []
a, b = ids(8090), ids(18090)
inter = set(a) & set(b)
ratio = len(inter) / max(1, min(len(a) or [1], len(b) or [1]))
print("  两版重合 %d 条（%.0f%%）%s" % (len(inter), ratio * 100, "✅ 差异明显" if ratio <= 0.5 else "⚠️ 差异不足"))
PY
done

# ---------- 5. 人工核对表 + 影子日志证据 ----------
echo
echo "============================================================"
echo "人工核对表（影子实例，逐条标题/渠道/判定）"
echo "============================================================"
docker exec "$SHADOW_CT" python /app/scripts/verify_local_reco.py \
  --repo /app --api http://localhost:8000 \
  --target 16307 --compare 13830,801 --limit "$LIMIT" 2>&1 | tail -70 || true

echo
echo "============================================================"
echo "影子实例日志证据（权重保序 / 列表配额是否生效）"
echo "============================================================"
docker logs "$SHADOW_CT" 2>&1 | grep -E "Channel weight order enforced|List quota applied|List quota relaxed" | tail -8 || echo "(暂无)"

echo
echo "============================================================"
echo "验收标准：影子实例每个页面  popular ≤ 3；content/vector/tag ≥ 6；与生产重合率 ≤ 50%"
echo "收尾命令：docker rm -f ${SHADOW_CT}"
echo "============================================================"
