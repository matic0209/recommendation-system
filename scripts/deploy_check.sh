#!/usr/bin/env bash
# -*- coding: utf-8 -*-
# ============================================================================
# 发版验收自检 (deploy_check.sh)
# ----------------------------------------------------------------------------
# 同一个脚本，影子先跑一遍、生产发版后再跑一遍 → 结果一致才算发布成功。
#
# 用法（在服务器仓库根目录）：
#   bash scripts/deploy_check.sh                    # 检查生产（容器 recommendation-api）
#   bash scripts/deploy_check.sh reco-shadow        # 检查影子实例（发版前必做）
#
# 检查项：
#   ① 服务健康 /health（models_loaded）
#   ② 下架名单是否生效（启动日志里的 Excluded N datasets）
#   ③ 5 个页面的 /similar：条数、热门来源条数(≤3)、是否出现被排除的 id
#   ④ 反复调用 /similar（默认 12 次）→ 被排除 id 出现次数必须为 0（覆盖探索通道）
#   ⑤ 响应耗时（P95）
#
# 退出码：0 = 全部通过；1 = 有失败项（可用于发版流程判断）
# ============================================================================
set -uo pipefail

CONTAINER="${1:-recommendation-api}"
LIMIT="${LIMIT:-12}"
ROUNDS="${ROUNDS:-12}"
PAGES="${PAGES:-16307,13830,801,8523,15098}"

echo "════════════════════════════════════════════════════════════════"
echo " 发版验收自检   容器=$CONTAINER  页面=$PAGES  轮次=$ROUNDS"
echo "════════════════════════════════════════════════════════════════"

# 容器内 python 直接访问 localhost:8000，无需宿主机依赖
docker exec -i \
  -e LIMIT="$LIMIT" -e ROUNDS="$ROUNDS" -e PAGES="$PAGES" \
  "$CONTAINER" python - <<'PY'
import glob
import json
import os
import statistics
import time
import urllib.request

BASE = "http://localhost:" + os.environ.get("PORT", "8000")
LIMIT = int(os.environ.get("LIMIT", "12"))
ROUNDS = int(os.environ.get("ROUNDS", "12"))
PAGES = [int(x) for x in os.environ.get("PAGES", "16307").split(",") if x.strip()]

failures = []
notes = []


def get(path, timeout=60):
    t0 = time.time()
    with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
        body = json.loads(r.read().decode("utf-8"))
    return body, (time.time() - t0) * 1000.0


# ---------- 读取下架名单（容器内路径） ----------
excluded = set()
_cands = [os.environ["EXCLUDE_FILE"]] if os.environ.get("EXCLUDE_FILE") else []
_cands += ["/app/models/excluded_dataset_ids.json", "models/excluded_dataset_ids.json"]
for cand in _cands:
    if os.path.exists(cand):
        try:
            payload = json.loads(open(cand, encoding="utf-8").read())
            if isinstance(payload, dict):
                payload = payload.get("ids") or []
            excluded = {int(x) for x in payload}
            notes.append(f"名单文件: {cand}（{len(excluded)} 个 id）")
        except Exception as exc:
            failures.append(f"名单文件解析失败 {cand}: {exc}")
        break
if not excluded:
    failures.append("未找到/未加载下架名单文件（models/excluded_dataset_ids.json）")

# ---------- ① 健康检查 ----------
print("\n① /health")
try:
    health, ms = get("/health")
    print("   ", json.dumps(health, ensure_ascii=False)[:220])
    bad = str(health.get("status", "")).lower() in ("error", "unhealthy", "fail", "failed")
    if bad:
        failures.append(f"/health 返回异常: {health}")
    elif not (health.get("models_loaded") or health.get("status")):
        notes.append("/health 缺少 models_loaded/status 字段（人工确认一下）")
    else:
        notes.append(f"/health 正常（{ms:.0f}ms）")
except Exception as exc:
    failures.append(f"/health 请求失败: {exc}")

# ---------- ② 启动日志里的 Excluded 行 ----------
print("\n② 启动日志（服务日志写在 /app/logs，不在 docker logs 里）")
try:
    logs = sorted(glob.glob(os.environ.get("LOG_GLOB", "/app/logs/*.log")), key=os.path.getmtime)
    hit = None
    for path in reversed(logs[-3:] or []):
        for line in open(path, encoding="utf-8", errors="ignore"):
            if "Excluded" in line and "datasets" in line:
                hit = (os.path.basename(path), line.strip()[:200])
    if hit:
        print(f"    ✅ {hit[0]}: {hit[1]}")
        notes.append("启动日志出现 Excluded 行")
    elif not logs:
        # 影子实例没有挂载 /app/logs，日志只进 stdout —— 这种情况降级为提示，避免误判为失败
        print("    ℹ 未找到日志文件（该实例可能把日志只写 stdout）→ 请人工确认：")
        print("      docker logs <容器> 2>&1 | grep -i Excluded")
        notes.append("未找到日志文件（非阻塞，需人工用 docker logs 确认）")
    else:
        print("    ⚠ 有日志文件但未找到 Excluded 行")
        failures.append("启动日志缺少 Excluded 行（名单可能未加载）")
except Exception as exc:
    notes.append(f"日志读取异常（不阻塞）: {exc}")

# ---------- ③ 各页面 /similar ----------
print("\n③ /similar 逐页检查（热门来源必须 ≤3，且不得出现被排除 id）")
latencies = []
for page in PAGES:
    try:
        body, ms = get(f"/similar/{page}?limit={LIMIT}")
        latencies.append(ms)
        items = body.get("similar_items") or []
        reasons = {}
        ids = []
        for it in items:
            rid = int(it.get("dataset_id", -1))
            ids.append(rid)
            r = it.get("reason") or ""
            reasons[r] = reasons.get(r, 0) + 1
        popular = sum(v for k, v in reasons.items() if "popular" in k or "fallback" in k)
        semantic = sum(v for k, v in reasons.items() if any(t in k for t in ("content", "vector", "tag")))
        leaked = sorted(set(ids) & excluded)
        ok = len(items) > 0 and popular <= 3 and not leaked
        print("    %-7d 条数=%-3d 热门=%-3d 内容/语义=%-3d 泄漏=%-2d %s  (%.0fms)"
              % (page, len(items), popular, semantic, len(leaked), "✅" if ok else "❌", ms))
        if len(items) == 0:
            failures.append(f"/similar/{page} 返回 0 条")
        if popular > 3:
            failures.append(f"/similar/{page} 热门来源 {popular} 条 > 3")
        if leaked:
            failures.append(f"/similar/{page} 出现被排除 id: {leaked[:10]}")
    except Exception as exc:
        failures.append(f"/similar/{page} 请求失败: {exc}")

# ---------- ④ 反复调用（覆盖探索通道随机性） ----------
print(f"\n④ 连续 {ROUNDS} 轮 /similar/{{首个页面}} → 被排除 id 出现次数必须为 0")
probe_page = PAGES[0] if PAGES else 16307
total_leak = 0
for i in range(ROUNDS):
    try:
        body, ms = get(f"/similar/{probe_page}?limit={LIMIT}")
        latencies.append(ms)
        ids = [int(it.get("dataset_id", -1)) for it in (body.get("similar_items") or [])]
        total_leak += len(set(ids) & excluded)
    except Exception as exc:
        failures.append(f"第 {i+1} 轮请求失败: {exc}")
print(f"    累计泄漏 {total_leak} 条")
if total_leak:
    failures.append(f"探索/召回通道仍返回被排除数据（{total_leak} 条）")

# ---------- ⑤ 耗时 ----------
print("\n⑤ 响应耗时")
if latencies:
    latencies.sort()
    p95 = latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))]
    print("    样本=%d  中位=%.0fms  P95=%.0fms  最大=%.0fms"
          % (len(latencies), statistics.median(latencies), p95, max(latencies)))
    if p95 > 3000:
        failures.append(f"P95 耗时 {p95:.0f}ms 过高")

# ---------- 汇总 ----------
print("\n" + "═" * 64)
if notes:
    for n in notes:
        print("  · " + n)
if failures:
    print("\n  ❌ 未通过项：")
    for f in failures:
        print("     - " + f)
    print("\n  结论：不通过，先不要发版 / 回滚")
    raise SystemExit(1)
print("\n  ✅ 全部检查通过")
print("  结论：该实例状态正常，可以进入下一步")
PY

code=$?
docker_code=$code
if [ $docker_code -ne 0 ]; then
  echo
  echo "（退出码 ${docker_code}）"
fi
exit $docker_code
