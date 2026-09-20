#!/usr/bin/env bash
# -*- coding: utf-8 -*-
# ============================================================================
# 回滚脚本 (rollback.sh)
# ----------------------------------------------------------------------------
# 用法：
#     bash scripts/rollback.sh --list                      # 看有哪些可回滚的镜像
#     bash scripts/rollback.sh                             # 回滚到最近一次 backup 标签
#     bash scripts/rollback.sh --tag <镜像:标签>            # 回滚到指定镜像
#
# 说明：
#   · 只重建 recommendation-api 容器（--no-deps，不动 redis/mlflow/postgres）
#   · 回滚后 **不要** 用 deploy_check.sh 判断成败：旧版 /similar 本来就是坏的，
#     判定标准是：服务能起来 + /health 正常 + 接口有返回。
#   · 名单文件属于数据，不随镜像回滚；如需摘除：mv models/excluded_dataset_ids.json /tmp/
#     然后 docker restart recommendation-api（30 秒）
# ============================================================================
set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE_MAIN="recommend-recommendation-api"
API_URL="http://localhost:8090"

TARGET_TAG=""
LIST_ONLY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --list) LIST_ONLY=1; shift ;;
    --tag) TARGET_TAG="${2:-}"; shift 2 ;;
    *) echo "未知参数: $1"; exit 1 ;;
  esac
done

cd "$REPO" || { echo "仓库目录不存在: $REPO"; exit 1; }

echo "════════════════════════════════════════════════════════════════"
echo " 可用回滚镜像"
echo "════════════════════════════════════════════════════════════════"
docker images --format '{{.Repository}}:{{.Tag}}  {{.ID}}  构建于 {{.CreatedSince}}' | grep "$IMAGE_MAIN" | sort
echo
docker inspect recommendation-api --format '  当前容器运行镜像: {{.Image}} 启动于 {{.State.StartedAt}}' 2>/dev/null || true
echo

if [ "$LIST_ONLY" = "1" ]; then exit 0; fi

# 决定回滚目标
if [ -z "$TARGET_TAG" ]; then
  RECORDED=$(ls -t /root/release_backup_*/rollback_image_tag.txt 2>/dev/null | head -1)
  if [ -n "$RECORDED" ]; then TARGET_TAG=$(cat "$RECORDED"); fi
fi
if [ -z "$TARGET_TAG" ]; then
  TARGET_TAG=$(docker images --format '{{.Repository}}:{{.Tag}}' | grep "${IMAGE_MAIN}:backup" | sort | tail -1)
fi
if [ -z "$TARGET_TAG" ]; then
  echo "❌ 找不到任何备份镜像标签（:backup-*），无法自动回滚。"
  echo "   可用 --tag 手动指定，或从 /root/reco-snapshot-*/ 里取回代码后重新构建。"
  exit 1
fi

echo "即将回滚到: $TARGET_TAG"
echo "（如有疑问先 Ctrl-C；确认后继续）"
sleep 5

# 执行回滚
docker tag "$TARGET_TAG" "$IMAGE_MAIN:latest" || { echo "❌ 打标签失败"; exit 1; }
docker compose up -d --no-deps --force-recreate recommendation-api || { echo "❌ 容器重建失败（需人工介入）"; exit 1; }

echo "等待服务就绪（最多 200 秒）..."
for _ in $(seq 1 40); do
  if curl -sf -m 5 "$API_URL/health" >/dev/null 2>&1; then break; fi
  sleep 5
done

echo
echo "── 回滚后检查（注意：不要用 deploy_check.sh） ──────────────────"
HEALTH=$(curl -s -m 10 "$API_URL/health" 2>/dev/null || echo "FAILED")
echo "  /health  => $HEALTH"
case "$HEALTH" in
  *models_loaded*true*|*healthy*) echo "  ✅ 服务已就绪" ;;
  *) echo "  ⚠️ /health 异常 → 看日志：docker logs --tail 50 recommendation-api" ;;
esac
echo -n "  /similar/16307 是否有返回: "
curl -s -m 30 "http://localhost:8090/similar/16307?limit=12" | head -c 120; echo
echo
echo "  当前容器镜像: $(docker inspect recommendation-api --format '{{.Image}}')"
echo "  代码版本(可选回退): git reset --hard \$(cat /root/release_backup_*/pre_deploy_head.txt)"
echo "  摘除下架名单(可选): mv models/excluded_dataset_ids.json /tmp/ && docker restart recommendation-api"
exit 0
