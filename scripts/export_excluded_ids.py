#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""导出"已下架/已删除"数据集名单（供推荐系统过滤）

背景
----
业务库 MySQL 的状态字段（`is_delete` / `publish_status` / `status`）**从来没有**
同步到推荐系统（parquet 与 models 文件里都没有状态列），所以推荐链路此前根本
不知道哪些数据已下架/已删除。

但业务库每 30 分钟会把**变更过的记录**导出成 JSON 到：
    /dianshu/backup/data/dianshu_data/jsons/dataset_<日期>_<时间>.json
这些文件里**带状态字段**，因此可以离线算出"不可售名单"，不需要直连业务库。

本脚本做的事
------------
1. 按文件时间顺序扫描 `dataset_*.json`，对每个 id 保留**最新**一条状态记录；
2. 按规则筛出不可售的 id（默认：is_delete==1 或 publish_status!=2）；
3. 输出 `models/excluded_dataset_ids.json`（服务启动时读取），
   并可选用 `--only-in-recommender` 只保留"仍在推荐候选池里"的 id；
4. 打印可核对的统计信息（各规则命中数、盲区数量）。

配套说明
--------
* 服务侧支持两个来源：环境变量 `RECO_EXCLUDED_DATASET_IDS` 与
  `models/excluded_dataset_ids.json`（格式 `[1,2]` 或 `{"ids": [1,2]}`）。
* **盲区**：只在"从未变更过"的老数据集上出现（它们不会出现在增量文件里）。
  脚本会打印盲区 id 的数量；若需要覆盖盲区，请从业务库导出**全量**名单，
  用 `--merge` 合并进来。
* 生效方式：写完文件后**重启推荐服务**（约 30 秒）；
  做"秒级生效"的动态名单需要 Redis 版本（见 docs）。

用法
----
# 在 airflow 容器里跑（该容器挂了 /dianshu 与整个仓库）
docker exec airflow-scheduler python /opt/recommend/scripts/export_excluded_ids.py \
    --json-dir /dianshu/backup/data/dianshu_data/jsons \
    --features /opt/recommend/data/processed/dataset_features.parquet \
    --out /opt/recommend/models/excluded_dataset_ids.json \
    --only-in-recommender

# 先看不写（推荐先跑一次确认数字）
... --dry-run
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Set, Tuple

STATUS_KEYS = ("is_delete", "publish_status", "status")


def load_latest_states(json_dir: str, patterns: Tuple[str, ...]) -> Tuple[Dict[int, dict], int]:
    """扫描增量文件，返回 {dataset_id: 最新状态记录} 与扫描文件数。"""
    files = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(json_dir, pat)))
    files = sorted(set(files), key=os.path.getmtime)

    latest: Dict[int, dict] = {}
    for path in files:
        try:
            payload = json.loads(Path(path).read_text())
        except Exception:
            continue
        rows = payload if isinstance(payload, list) else [payload]
        for row in rows:
            if not isinstance(row, dict):
                continue
            # 只认真正的 dataset 行：必须同时带三个状态字段
            # （目录里还混着 dataset_image 等子表，它们只有 id/dataset_id）
            if not all(k in row for k in STATUS_KEYS):
                continue
            did = row.get("id") or row.get("dataset_id")
            if did is None:
                continue
            try:
                latest[int(did)] = row
            except (TypeError, ValueError):
                continue
    return latest, len(files)


def read_id_file(path: str) -> Set[int]:
    """读取外部名单（支持 JSON 数组 / {"ids": [...]} / CSV（取第一列）/ 每行一个 id 的 TXT）。

    注意：CSV 只取**第一列**，避免把 `8658,1,0,孔雀东南飞` 里的 1、0 误当成 id。
    因此从业务库导出时，建议直接 `SELECT id FROM dataset WHERE ...` 只导一列。
    """
    ids: Set[int] = set()
    text = Path(path).read_text(encoding="utf-8").strip()
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            payload = payload.get("ids") or []
        for item in payload or []:
            ids.add(int(item))
        return ids
    except Exception:
        pass
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if "," in line or "\t" in line:              # CSV / TSV → 只取第一列
            field = line.replace("\t", ",").split(",")[0].strip().strip('"\'')
            if field.isdigit():
                ids.add(int(field))
            continue                                  # 表头或异常行直接跳过
        for token in line.replace(";", " ").split():  # TXT：允许一行多个 id
            token = token.strip().strip('"\'')
            if token.isdigit():
                ids.add(int(token))
    return ids


def load_recommender_ids(features_path: str) -> Set[int]:
    """推荐系统当前候选池的 id 集合（从特征表读）。"""
    import pandas as pd  # 延迟导入：只有需要时才依赖

    df = pd.read_parquet(features_path, columns=["dataset_id"])
    return {int(x) for x in df.dataset_id}


def main() -> int:
    ap = argparse.ArgumentParser(description="导出已下架/已删除数据集名单")
    ap.add_argument("--json-dir", default="/dianshu/backup/data/dianshu_data/jsons",
                    help="业务库增量导出目录")
    ap.add_argument("--pattern", action="append", default=None,
                    help="增量文件名匹配（可多次指定，默认 dataset_*.json）")
    ap.add_argument("--features", default="/opt/recommend/data/processed/dataset_features.parquet",
                    help="特征表路径（用于 --only-in-recommender 与盲区统计）")
    ap.add_argument("--out", default="/opt/recommend/models/excluded_dataset_ids.json",
                    help="输出的名单文件路径")
    ap.add_argument("--only-in-recommender", action="store_true",
                    help="只保留仍在推荐候选池里的 id（名单更小更聚焦）")
    ap.add_argument("--include-unpublished", action="store_true", default=True,
                    help="把 publish_status != 2 也算作不可售（默认开启）")
    ap.add_argument("--merge", action="append", default=None,
                    help="合并外部全量名单（业务库导出），可多次指定")
    ap.add_argument("--dry-run", action="store_true", help="只打印，不写文件")
    args = ap.parse_args()

    patterns = tuple(args.pattern) if args.pattern else ("dataset_*.json",)
    latest, n_files = load_latest_states(args.json_dir, patterns)
    print(f"扫描增量文件 {n_files} 个，得到带状态的 dataset 记录 {len(latest)} 个 id")

    deleted = {i for i, r in latest.items() if str(r.get("is_delete")) == "1"}
    unpublished = {i for i, r in latest.items() if str(r.get("publish_status")) != "2"}
    bad = set(deleted)
    if args.include_unpublished:
        bad |= unpublished
    print(f"  is_delete=1（已删除）        : {len(deleted)} 个")
    print(f"  publish_status!=2（未发布/下架）: {len(unpublished)} 个")
    print(f"  规则合计需排除                : {len(bad)} 个")

    for extra in args.merge or []:
        extra_ids = read_id_file(extra)
        print(f"  合并外部名单 {extra}: {len(extra_ids)} 个 id")
        bad |= extra_ids

    rec_ids: Set[int] = set()
    if args.features and Path(args.features).exists():
        try:
            rec_ids = load_recommender_ids(args.features)
        except Exception as exc:  # noqa: BLE001
            print(f"  [WARN] 读取特征表失败：{exc}")

    if rec_ids:
        in_pool = sorted(bad & rec_ids)
        blind = sorted(rec_ids - set(latest))
        print()
        print(f"推荐系统候选池: {len(rec_ids)} 个")
        print(f"  名单中仍在候选池里的  : {len(in_pool)} 个  ← 这些就是当前会被推荐的坏数据")
        print(f"  状态盲区(增量无记录)  : {len(blind)} 个  ← 需业务库全量名单覆盖")
        if in_pool:
            print(f"  将被排除的 id: {in_pool}")
        if args.only_in_recommender:
            bad = set(in_pool)

    print(f"\n最终写入名单: {len(bad)} 个")

    if args.dry_run:
        print("[dry-run] 未写文件。内容预览：")
        print(json.dumps(sorted(bad), ensure_ascii=False)[:2000])
        return 0

    payload = {
        "_comment": "不可售/已下架数据集黑名单（服务启动时读取并从内存索引剔除）",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "rule": "is_delete==1" + (" | publish_status!=2" if args.include_unpublished else ""),
        "source": args.json_dir,
        "stats": {
            "scanned_files": n_files,
            "ids_with_status": len(latest),
            "deleted": len(deleted),
            "unpublished": len(unpublished),
        },
        "ids": sorted(bad),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写入: {out}")
    print("生效方式：docker restart recommendation-api（约 30 秒）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
