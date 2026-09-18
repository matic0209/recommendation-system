#!/usr/bin/env bash
# -*- coding: utf-8 -*-
# ============================================================================
# 维护窗口发版脚本 (deploy_window.sh)
# ----------------------------------------------------------------------------
# 设计目标：把耗时的、有风险的动作全部提前完成（拉代码 / 构建镜像 / 影子验收 /
# 金丝雀验收 / 名单就位 / 备份锚点），窗口内只做"换容器 + 验收"，约 3~5 分钟。
#
# 用法（在服务器仓库根目录）：
#     bash scripts/deploy_window.sh --check      # 只做前置检查，不动生产（窗口前先跑）
#     bash scripts/deploy_window.sh             # 正式发版（会重建 recommendation-api 容器）
#     bash scripts/deploy_window.sh --with-scheduler   # 同时重启 airflow-scheduler
#
# 退出码：0 = 发版成功且验收通过；1 = 失败（请立刻执行 scripts/rollback.sh）
# ============================================================================
set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE_MAIN="recommend-recommendation-api"
API_URL="http://localhost:8090"
EXPECT_APP_MD5="${EXPECT_APP_MD5:-049149c80033f5f3cbab1c39c3c83457}"
EXPECT_LIST_COUNT="${EXPECT_LIST_COUNT:-11325}"
PAGES="${PAGES:-16307,16306,16305,16304,16346,13830,14297,15098,801,8523,20,22}"

CHECK_ONLY=0
WITH_SCHEDULER=0
TARGET_IMAGE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --check) CHECK_ONLY=1; shift ;;
    --with-scheduler) WITH_SCHEDULER=1; shift ;;
    --tag) TARGET_IMAGE="${2:-}"; shift 2 ;;
    *) echo "未知参数: $1"; echo "用法: $0 [--check] [--with-scheduler] [--tag <镜像:标签>]"; exit 1 ;;
  esac
done

ok()   { echo "  ✅ $*"; }
warn() { echo "  ⚠️  $*"; }
bad()  { echo "  ❌ $*"; FAILED=1; }
step() { echo; echo "── $* ────────────────────────────────────────"; }
FAILED=0

echo "════════════════════════════════════════════════════════════════"
echo " 推荐服务发版（维护窗口模式）  仓库=$REPO"
echo "════════════════════════════════════════════════════════════════"
cd "$REPO" || { echo "仓库目录不存在: $REPO"; exit 1; }

# ---------------------------------------------------------------------------
step "1/6 前置检查（任何一项 ❌ 都不要继续）"
# ---------------------------------------------------------------------------
# 1.1 代码指纹
APP_MD5=$(md5sum app/main.py | awk '{print $1}')
[ "$APP_MD5" = "$EXPECT_APP_MD5" ] && ok "app/main.py 指纹正确（${APP_MD5}）" \
  || bad "app/main.py 指纹不符（期望 ${EXPECT_APP_MD5}，实际 ${APP_MD5}）→ 代码没拉全/被改过"

# 1.2 发版镜像（--tag 指定优先，否则用 :latest）
DEPLOY_IMAGE="${TARGET_IMAGE:-${IMAGE_MAIN}:latest}"
if ! docker image inspect "$DEPLOY_IMAGE" >/dev/null 2>&1; then
  bad "找不到发版镜像 $DEPLOY_IMAGE → 窗口前先构建并打标签（docker compose build + docker tag ...:candidate-<日期>）"
else
  ok "发版镜像: ${DEPLOY_IMAGE}（$(docker image inspect "$DEPLOY_IMAGE" --format '{{.Id}}' | cut -c8-19)，构建于 $(docker image inspect "$DEPLOY_IMAGE" --format '{{.Created}}' | cut -c1-19)）"
  # ★ 直接读【镜像里】的 app/main.py 指纹：确保镜像里的代码就是我们要发的那份
  IMG_APP_MD5=$(docker run --rm --entrypoint md5sum "$DEPLOY_IMAGE" /app/app/main.py 2>/dev/null | awk '{print $1}')
  if [ -z "$IMG_APP_MD5" ]; then
    warn "无法读取镜像内的 app/main.py 指纹（跳过该校验）"
  elif [ "$IMG_APP_MD5" = "$EXPECT_APP_MD5" ]; then
    ok "镜像内 app/main.py 指纹正确（${IMG_APP_MD5}）"
  else
    bad "镜像内 app/main.py 指纹不符（镜像=${IMG_APP_MD5}，期望=${EXPECT_APP_MD5}）→ 镜像不是用当前代码构建的"
  fi
fi
CAND=$(docker images --format '{{.Repository}}:{{.Tag}} {{.ID}}' | grep "${IMAGE_MAIN}:candidate" | sort | tail -1 | awk '{print $2}')
[ -n "$CAND" ] && ok "候选镜像标签在位（${CAND}）" || warn "没有 candidate 标签（不影响发版，只影响对照）"

# 1.3 回滚镜像备份（没有就自动补一个）
BACKUP_TAG=$(docker images --format '{{.Repository}}:{{.Tag}}' | grep "${IMAGE_MAIN}:backup" | sort | tail -1)
if [ -z "$BACKUP_TAG" ]; then
  NEW_TAG="${IMAGE_MAIN}:backup-$(date +%Y%m%d-%H%M)"
  docker tag "$IMAGE_MAIN:latest" "$NEW_TAG" 2>/dev/null && warn "本次未发现备份标签 → 已用当前 latest 打了一个($NEW_TAG)，真正的回滚点请确认它指向发版前的镜像" \
    || bad "无法创建回滚镜像标签"
else
  ok "回滚镜像备份在位（${BACKUP_TAG}）"
fi

# 1.4 下架名单
if [ -f models/excluded_dataset_ids.json ]; then
  CNT=$(python3 -c "import json;print(len(json.load(open('models/excluded_dataset_ids.json'))['ids']))" 2>/dev/null || echo "?")
  [ "$CNT" = "$EXPECT_LIST_COUNT" ] && ok "下架名单在位（$CNT 个 id）" \
    || warn "名单 id 数为 ${CNT}（期望 ${EXPECT_LIST_COUNT}）→ 确认是否用了最新名单"
else
  bad "缺少 models/excluded_dataset_ids.json → 下架过滤不会生效"
fi

# 1.5 compose 文件可解析
if docker compose config >/dev/null 2>&1; then ok "docker compose 配置可解析"; else bad "docker compose 配置解析失败"; fi

# 1.6 当前服务状态
if curl -sf -m 5 "$API_URL/health" >/dev/null 2>&1; then ok "当前服务健康（$API_URL/health）"; else warn "当前服务 /health 未通过（发版前记录一下，便于对比）"; fi

if [ "$FAILED" = "1" ]; then
  echo; echo "❌ 前置检查未通过 → 先修掉上面的问题（窗口前处理好），不要发版"; exit 1
fi
echo; echo "✅ 前置检查全部通过"

if [ "$CHECK_ONLY" = "1" ]; then
  echo "（--check 模式，未做任何变更）"; exit 0
fi

# ---------------------------------------------------------------------------
step "2/6 记录发版前状态（用于回滚判断）"
# ---------------------------------------------------------------------------
B="/root/release_backup_$(date +%Y%m%d-%H%M)"
mkdir -p "$B"
git rev-parse HEAD            | tee "$B/pre_deploy_head.txt"
md5sum app/main.py            | tee "$B/pre_deploy_md5.txt"
docker images --format '{{.Repository}}:{{.Tag}} {{.ID}}' | grep "$IMAGE_MAIN" | tee "$B/pre_images.txt"
echo "$BACKUP_TAG"            | tee "$B/rollback_image_tag.txt"
ok "已写入 $B"

# ---------------------------------------------------------------------------
step "3/6 切换推荐服务（重建容器，服务中断约 60~90 秒）"
# ---------------------------------------------------------------------------
# 预置候选镜像模式：先把 :latest 指到发版镜像，compose 便会用它重建
if [ -n "$TARGET_IMAGE" ] && [ "$TARGET_IMAGE" != "${IMAGE_MAIN}:latest" ]; then
  docker tag "$TARGET_IMAGE" "${IMAGE_MAIN}:latest" \
    && ok "已把 ${IMAGE_MAIN}:latest 指向 ${TARGET_IMAGE}" \
    || { bad "打标签失败"; exit 1; }
fi
docker compose up -d --no-deps --force-recreate recommendation-api || { bad "容器重建失败 → 执行 bash scripts/rollback.sh"; exit 1; }
ok "容器已重建（仅 recommendation-api，未动 redis/mlflow/postgres）"

echo "  等待服务就绪（最多 200 秒）..."
READY=0
for _ in $(seq 1 40); do
  if curl -sf -m 5 "$API_URL/health" >/dev/null 2>&1; then READY=1; break; fi
  sleep 5
done
[ "$READY" = "1" ] && ok "服务已就绪：$(curl -s -m 5 "$API_URL/health")" || { bad "/health 未就绪 → 执行 bash scripts/rollback.sh"; exit 1; }

# ★ 清掉旧响应缓存：否则验收可能读到"切换前算好并缓存"的旧结果（曾因此误判）
for pat in 'similar:*' 'recommend:*'; do
  docker exec redis redis-cli -n 0 --scan --pattern "$pat" 2>/dev/null | tr -d '\r' \
    | xargs -r -I{} docker exec redis redis-cli -n 0 del {} >/dev/null 2>&1
done
ok "已清理缓存（similar:* / recommend:*）——避免验收读到旧结果"

# ---------------------------------------------------------------------------
step "4/6 可选：重启 airflow-scheduler（让 DAG/脚本改动生效）"
# ---------------------------------------------------------------------------
if [ "$WITH_SCHEDULER" = "1" ]; then
  docker restart airflow-scheduler >/dev/null && ok "airflow-scheduler 已重启"
  sleep 8
  docker logs --tail 30 airflow-scheduler 2>&1 | grep -i TemplateNotFound && bad "scheduler 日志仍有模板错误" || ok "无模板错误"
else
  warn "未重启 scheduler（脚本/DAG 的改动不会生效）；需要就加 --with-scheduler"
fi

# ---------------------------------------------------------------------------
step "5/6 生产验收（同一个脚本，退出码必须为 0）"
# ---------------------------------------------------------------------------
PAGES="$PAGES" bash scripts/deploy_check.sh
RC=$?
[ "$RC" = "0" ] && ok "验收通过" || bad "验收未通过（退出码 ${RC}）"

# ---------------------------------------------------------------------------
step "6/6 结论"
# ---------------------------------------------------------------------------
if [ "$FAILED" = "1" ]; then
  echo "❌ 发版未通过验收 → 立刻执行：bash scripts/rollback.sh"
  echo "   （旧版 /similar 本来就是坏的，回滚后不要用 deploy_check.sh 判断，只看 /health 与页面可用）"
  exit 1
fi
echo "✅ 发版成功。下一步建议："
echo "   1) 页面自检：https://dianshudata.com/dataDetail/16307 →“看了又看”不应出现“该数据已下架”"
echo "   2) 明天观察 02:00 的流水线是否全绿（restart_if_changed 应成功）"
echo "   3) 收尾：docker rm -f reco-shadow reco-canary"
exit 0
