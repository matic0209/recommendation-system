#!/usr/bin/env bash
# -*- coding: utf-8 -*-
# ============================================================================
# 数据集状态体检 (check_dataset_status.sh)
# ----------------------------------------------------------------------------
# 一次查清一个数据集"为什么会被推荐 / 到底下架没有"：
#   ① 业务库最新状态（扫 /dianshu 增量导出，判定 is_delete / publish_status / status）
#   ② 是否还在推荐系统候选池（特征表）
#   ③ 是否在热销榜（未登录/兜底时展示的那批）
#   ④ 是否真的被推荐过（扫曝光日志，给出页面/位置/召回渠道/时间）
#
# 用法（在服务器上，仓库根目录下）：
#   bash scripts/check_dataset_status.sh 8658 15769
#
# 可选环境变量：
#   CONTAINER=airflow-scheduler     容器名（该容器挂了 /dianshu 与整个仓库）
#   JSON_DIR=/dianshu/backup/data/dianshu_data/jsons
#   REPO=/opt/recommend             容器内的仓库路径
#   LOG_TAIL_MB=60                  曝光日志只扫末尾多少 MB（日志可能很大）
# ============================================================================
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "用法: bash $0 <数据集id> [id...]     例如: bash $0 8658 15769"
  exit 1
fi

CONTAINER="${CONTAINER:-airflow-scheduler}"
JSON_DIR="${JSON_DIR:-/dianshu/backup/data/dianshu_data/jsons}"
REPO="${REPO:-/opt/recommend}"
LOG_TAIL_MB="${LOG_TAIL_MB:-60}"

echo "容器=$CONTAINER  增量目录=$JSON_DIR  仓库=$REPO  曝光日志扫描=$LOG_TAIL_MB MB"
echo

# 注意：必须加 -i，否则 heredoc 传不进去（stdin 不会被挂载）
docker exec -i \
  -e IDS="$*" -e JSON_DIR="$JSON_DIR" -e REPO="$REPO" -e LOG_TAIL_MB="$LOG_TAIL_MB" \
  "$CONTAINER" python - <<'PY'
import glob
import json
import os
from datetime import datetime

ids = [int(x) for x in os.environ["IDS"].split()]
json_dir = os.environ["JSON_DIR"]
repo = os.environ["REPO"]
tail_mb = int(os.environ.get("LOG_TAIL_MB", "60"))
want = set(ids)
STATUS_KEYS = ("is_delete", "publish_status", "status")

print("=" * 78)
print("① 业务库最新状态（扫增量导出 *.json，每 id 取最新一条）")
print("=" * 78)
files = sorted(set(glob.glob(os.path.join(json_dir, "dataset*.json"))), key=os.path.getmtime)
latest, seen_in = {}, {i: [] for i in ids}
scanned = 0
for path in files:
    try:
        payload = json.loads(open(path, encoding="utf-8").read())
    except Exception:
        continue
    scanned += 1
    for row in (payload if isinstance(payload, list) else [payload]):
        if not isinstance(row, dict) or not all(k in row for k in STATUS_KEYS):
            continue  # 跳过 dataset_image 等子表行
        did = row.get("id") or row.get("dataset_id")
        if did is None:
            continue
        did = int(did)
        latest[did] = row
        if did in want:
            seen_in[did].append((os.path.getmtime(path), os.path.basename(path)))

print("扫描增量文件 %d 个，覆盖 %d 个数据集" % (scanned, len(latest)))
if files:
    print("增量文件时间跨度: 最早 %s ~ 最新 %s" % (os.path.basename(files[0]), os.path.basename(files[-1])))
    print("  ↑ 早于此时间发生的状态变更【无法从增量里查到】——这正是状态盲区的成因")
for i in ids:
    rec = latest.get(i)
    if rec is None:
        hist = seen_in[i]
        if hist:
            print("  %-7d ⚠ 出现过 %d 次但都无状态字段（异常）" % (i, len(hist)))
        else:
            print("  %-7d ❓ 增量里【从未出现】→ 状态盲区（老数据从未变更）" % i)
        continue
    bad = []
    if str(rec.get("is_delete")) == "1":
        bad.append("is_delete=1(已删除)")
    if str(rec.get("publish_status")) != "2":
        bad.append("publish_status=%s(非已发布)" % rec.get("publish_status"))
    if str(rec.get("status")) != "2":
        bad.append("status=%s(非正常)" % rec.get("status"))
    hist = seen_in[i]
    print("  %-7d is_delete=%-3s publish_status=%-3s status=%-3s update_time=%-20s %s"
          % (i, rec.get("is_delete"), rec.get("publish_status"), rec.get("status"),
             str(rec.get("update_time"))[:19], "← 判定：【%s】" % "；".join(bad) if bad else "← 判定：健康（在售）"))
    if hist:
        print("           最后一次出现在增量文件: %s" % hist[-1][1])
    if rec.get("dataset_name"):
        print("           名称: %s" % str(rec.get("dataset_name"))[:60])

print()
print("=" * 78)
print("② 是否还在推荐系统候选池（%s/data/processed/dataset_features.parquet）" % repo)
print("=" * 78)
try:
    import pandas as pd
    df = pd.read_parquet(os.path.join(repo, "data/processed/dataset_features.parquet"),
                         columns=["dataset_id", "dataset_name", "tag"])
    pool = {int(a): (b, c) for a, b, c in zip(df.dataset_id, df.dataset_name, df.tag)}
    print("候选池总数: %d" % len(pool))
    for i in ids:
        if i in pool:
            name, tag = pool[i]
            print("  %-7d ✅ 在池子里 → %s" % (i, str(name)[:55]))
            print("           标签: %s" % (str(tag)[:70] or "（无）"))
        else:
            print("  %-7d ❌ 不在候选池（推荐系统不可能召回它）" % i)
except Exception as exc:  # noqa: BLE001
    print("  读取特征表失败: %s" % exc)

print()
print("=" * 78)
print("③ 是否在热销榜（models/top_items.json，未登录/兜底时展示的那批）")
print("=" * 78)
try:
    top = json.load(open(os.path.join(repo, "models/top_items.json"), encoding="utf-8"))
    rows = top if isinstance(top, list) else (top.get("items") or [])
    tids = []
    for x in rows:
        try:
            tids.append(int(x.get("dataset_id") or x.get("id")) if isinstance(x, dict) else int(x))
        except Exception:
            pass
    print("热销榜条数: %d" % len(tids))
    for i in ids:
        pos = tids.index(i) + 1 if i in tids else 0
        print("  %-7d %s" % (i, "⚠ 在热销榜第 %d 位（未登录用户会看到）" % pos if pos else "不在热销榜"))
except Exception as exc:  # noqa: BLE001
    print("  读取热销榜失败: %s" % exc)

print()
print("=" * 78)
print("④ 是否真的被推荐过（扫描曝光日志末尾 %d MB）" % tail_mb)
print("=" * 78)
log_path = os.path.join(repo, "data/evaluation/exposure_log.jsonl")
try:
    size = os.path.getsize(log_path)
    with open(log_path, "rb") as fh:
        if size > tail_mb * 1024 * 1024:
            fh.seek(size - tail_mb * 1024 * 1024)
            fh.readline()  # 丢弃半行
        hits = {i: [] for i in ids}
        total = 0
        for raw in fh:
            total += 1
            try:
                rec = json.loads(raw)
            except Exception:
                continue
            for item in (rec.get("items") or []):
                did = item.get("dataset_id") or item.get("item_id") or item.get("id")
                try:
                    did = int(did)
                except Exception:
                    continue
                if did in hits:
                    hits[did].append((rec.get("timestamp"), rec.get("page_id"), rec.get("user_id"),
                                      item.get("position"), item.get("reason")))
        print("日志大小 %.1f MB，末尾扫描 %d 条推荐记录" % (size / 1048576.0, total))
        for i in ids:
            h = hits[i]
            if not h:
                print("  %-7d 末尾记录里【没有被推荐过】" % i)
                continue
            print("  %-7d ⚠ 被推荐过 %d 次，最近 3 次：" % (i, len(h)))
            for ts, page, user, pos, reason in h[-3:]:
                print("           ts=%s 页面=%s 用户=%s 位置=%s 渠道=%s"
                      % (str(ts)[:19], page, user, pos, reason))
except FileNotFoundError:
    print("  找不到曝光日志: %s" % log_path)
except Exception as exc:  # noqa: BLE001
    print("  扫描失败: %s" % exc)

print()
print("体检完成 (%s)" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
PY
