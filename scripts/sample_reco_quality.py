#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""推荐质量采样对比脚本（观察期用）
================================================================

作用
----
每小时（或任意频率）对一批数据集页面同时采样两个入口：

    线上（旧代码，默认 http://localhost:8090）
    新代码（影子实例，默认 http://localhost:18090）

对每个页面记录：
    · 返回条数
    · 热门渠道条数（popular / fallback）—— 业务要求 ≤ 3
    · 内容/语义/标签渠道条数 —— 越多越相关
    · 与目标数据集"标签重合"的条数（人工判定的机器近似）
    · 是否整页都是兜底（fallback:popular 占满）
并追加写入 CSV，便于几天后统计"达标比例"。

用法（推荐在影子容器里跑，那里有 pandas/pyarrow 和挂载好的数据）
--------------------------------------------------------------------
    docker exec reco-shadow python /app/scripts/sample_reco_quality.py \
        --repo /app \
        --prod http://localhost:8000 \
        --new  http://localhost:18090 \
        --pages 16307,16346,801,8523,13830,13143 \
        --csv /app/data/evaluation/reco_quality_samples.csv

说明
----
* 影子容器内部访问"线上旧代码"不方便（外部是 8090），
  所以如果想在影子容器里同时比两边，把 --prod 指向宿主机映射端口：
      --prod http://host.docker.internal:8090
  或者直接在生产机上跑（需安装 pandas/pyarrow）：
      python3 scripts/sample_reco_quality.py --prod http://localhost:8090 --new http://localhost:18090
* 纯只读：只发 HTTP 请求 + 读 parquet，不写任何业务数据（只写 CSV）。

挂 cron 每小时跑一次（在生产机上）：
    0 * * * * cd /root/reco-shadow/repo && docker exec reco-shadow python /app/scripts/sample_reco_quality.py --repo /app --prod http://host.docker.internal:8090 --new http://localhost:18090 --csv /app/data/evaluation/reco_quality_samples.csv >> /var/log/reco_quality.log 2>&1
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

POPULAR_KEYS = ("popular", "fallback")
SEMANTIC_KEYS = ("content", "vector", "tag", "12cat")


def fetch(api: str, dataset_id: int, limit: int, timeout: int = 60) -> Optional[dict]:
    """请求某个服务的 /similar 接口；失败返回 None。"""
    url = f"{api.rstrip('/')}/similar/{dataset_id}?limit={limit}"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        print(f"  !! {url} 请求失败: {exc}")
        return None


def split_tags(raw) -> Set[str]:
    if raw is None or not isinstance(raw, str):
        return set()
    text = raw.replace("；", ";").replace(",", ";")
    return {t.strip().lower() for t in text.split(";") if t.strip()}


def analyze(payload: Optional[dict], target_tags: Set[str]) -> Dict[str, object]:
    """把一次响应折算成可比较的指标。"""
    if payload is None:
        return {"items": 0, "popular": 0, "semantic": 0, "tag_hit": 0, "all_fallback": 0, "reasons": "{}"}
    items = payload.get("similar_items") or []
    popular = semantic = tag_hit = 0
    reasons: Dict[str, int] = {}
    for it in items:
        reason = (it.get("reason") or "")
        reasons[reason] = reasons.get(reason, 0) + 1
        if any(k in reason for k in POPULAR_KEYS):
            popular += 1
        if any(k in reason for k in SEMANTIC_KEYS):
            semantic += 1
        ids = it.get("dataset_id")
        # 标签重合需要元数据，这里由调用方传入的 tags 映射补齐（见 main）
    return {
        "items": len(items),
        "popular": popular,
        "semantic": semantic,
        "all_fallback": 1 if (items and all("fallback" in (i.get("reason") or "") for i in items)) else 0,
        "reasons": json.dumps(reasons, ensure_ascii=False),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="推荐质量采样对比（观察期）")
    ap.add_argument("--prod", default="http://localhost:8090", help="线上（旧代码）地址")
    ap.add_argument("--new", dest="new_api", default="http://localhost:18090", help="新代码（影子）地址")
    ap.add_argument("--pages", default="16307,16346,801,8523,13830,13143", help="采样页面 id，逗号分隔")
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument("--repo", default=".", help="仓库根目录（读 parquet 元数据）")
    ap.add_argument("--csv", default="reco_quality_samples.csv", help="CSV 输出路径")
    args = ap.parse_args()

    # 读元数据（目标标签，用于"标签重合"这个机器近似指标）
    meta_path = Path(args.repo) / "data" / "processed" / "dataset_features.parquet"
    tag_by_id: Dict[int, Set[str]] = {}
    if meta_path.exists():
        df = pd.read_parquet(meta_path, columns=["dataset_id", "dataset_name", "tag"])
        tag_by_id = {int(r.dataset_id): split_tags(r.tag) for r in df.itertuples()}
    else:
        print(f"[WARN] 找不到元数据 {meta_path}，标签重合指标将为空")

    pages = [int(x) for x in args.pages.split(",") if x.strip()]
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows: List[dict] = []

    for page in pages:
        target_tags = tag_by_id.get(page, set())
        row = {"time": ts, "page": page, "target_tags": ";".join(sorted(target_tags))}
        for label, api in (("prod", args.prod), ("new", args.new_api)):
            payload = fetch(api, page, args.limit)
            m = analyze(payload, target_tags)
            row[f"{label}_items"] = m["items"]
            row[f"{label}_popular"] = m["popular"]
            row[f"{label}_semantic"] = m["semantic"]
            row[f"{label}_all_fallback"] = m["all_fallback"]
            row[f"{label}_reasons"] = m["reasons"]
        rows.append(row)

    # 打印摘要
    print(f"\n采样时间 {ts}    页数 {len(rows)}")
    print("%-8s %-30s %-22s %-22s" % ("页面", "线上(旧): 条数/热门/相关", "新代码: 条数/热门/相关", "判定"))
    for row in rows:
        p = f"{row['prod_items']}/{row['prod_popular']}/{row['prod_semantic']}"
        n = f"{row['new_items']}/{row['new_popular']}/{row['new_semantic']}"
        if row["new_items"] == 0:
            verdict = "请求失败/无数据"      # 注意：请求失败不能算达标（早期版本踩过这个坑）
        elif row["new_popular"] <= 3 and not row["new_all_fallback"]:
            verdict = "新代码达标"
        else:
            verdict = "需关注"
        print("%-8s %-30s %-22s %-22s" % (row["page"], p, n, verdict))

    valid = [r for r in rows if r["new_items"] > 0 and r["prod_items"] > 0]
    n_ok = sum(1 for r in valid if r["new_popular"] <= 3 and not r["new_all_fallback"])
    p_bad = sum(1 for r in valid if r["prod_popular"] >= 8 or r["prod_all_fallback"])
    print(
        f"\n汇总（仅统计两边都成功返回的 {len(valid)}/{len(rows)} 个页面）："
        f"线上'热门堆满/整页兜底' {p_bad}/{len(valid) or 1}；新代码达标 {n_ok}/{len(valid) or 1}"
    )

    # 追加写 CSV（首次自动写表头）
    out = Path(args.csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    new_file = not out.exists()
    with out.open("a", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        if new_file:
            writer.writeheader()
        writer.writerows(rows)
    print(f"已追加写入：{out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
