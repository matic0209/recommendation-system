#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从平台接口同步"不可推荐"数据集名单（权威版）

为什么用接口而不是业务库
------------------------
业务库 `dataset` 表里**没有 datasetStatus 这一列**（只有 status=审核态、
publish_status=发布态、is_delete=删除标记），而页面/接口判定"已下架"用的是
后端计算出的 `datasetStatus`：

    8658  → datasetStatus=2  → 页面显示"该数据已下架"
    16307 → datasetStatus=1  → 正常在售
    15769 → 接口返回"数据集不存在" → 页面显示"该数据已删除"

所以状态字段必须从接口取（已实测**匿名可调用**，无需登录）。

本脚本做的事
------------
1. 读候选 id（默认取特征表 dataset_features.parquet 里的全部 id）；
2. 并发调用 `POST https://api.dianshudata.com/dataset/datasetDetail`；
3. 分类：
      datasetStatus == 1                → 在售（保留）
      datasetStatus != 1                → 已下架（排除）
      接口无数据 / "数据集不存在"        → 已删除（排除）
      网络失败（重试后仍失败）           → unknown（默认**不排除**，避免误杀）
4. 写出 `models/excluded_dataset_ids.json`（服务启动时读取）+ 明细 CSV 报告。

实测吞吐：串行约 6 req/s；默认 8 并发 → 15000 个 id 约 5-6 分钟。

用法
----
# 先小批量试跑（只看不写）
python3 scripts/sync_excluded_from_api.py --features data/processed/dataset_features.parquet \
        --limit 200 --dry-run

# 全量生成（建议在容器里跑，那里有 pandas + 网络）
docker exec -it airflow-scheduler python /opt/recommend/scripts/sync_excluded_from_api.py \
        --features /opt/recommend/data/processed/dataset_features.parquet \
        --out /opt/recommend/models/excluded_dataset_ids.json \
        --csv /opt/recommend/data/evaluation/dataset_status_report.csv

# 只核验指定的 id（用于抽查、验证名单）
python3 scripts/sync_excluded_from_api.py --ids-file /tmp/check_ids.txt --dry-run

生成后生效：docker restart recommendation-api（约 30 秒）
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

API_URL = "https://api.dianshudata.com/dataset/datasetDetail"


def fetch_status(dataset_id: int, timeout: int, retries: int) -> Tuple[int, str, str, Optional[str]]:
    """查询单个数据集状态。

    返回 (id, verdict, detail, name)：
      verdict ∈ {onsale, delisted, missing, unknown}
    """
    body = json.dumps({"datasetId": dataset_id}).encode("utf-8")
    last_err = ""
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(
                API_URL, data=body,
                headers={"Content-Type": "application/json", "User-Agent": "reco-status-sync/1.0"},
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            data = payload.get("data")
            if not data:
                desc = str(payload.get("resultDesc") or "")
                code = payload.get("resultCode")
                # 只有明确的"不存在/已删除"才算 missing（确定性结论）；
                # 其它无数据响应（限流、500、参数错）算 unknown，避免误杀
                if "不存在" in desc or "已删除" in desc:
                    return dataset_id, "missing", desc, None
                return dataset_id, "unknown", f"无数据: {desc} (resultCode={code})", None
            status = data.get("datasetStatus")
            name = data.get("datasetName")
            try:
                status_int = int(status)
            except (TypeError, ValueError):
                return dataset_id, "unknown", f"datasetStatus 异常: {status!r}", name
            if status_int == 1:
                return dataset_id, "onsale", "datasetStatus=1", name
            return dataset_id, "delisted", f"datasetStatus={status_int}", name
        except Exception as exc:  # noqa: BLE001
            last_err = f"{type(exc).__name__}: {exc}"
            if attempt < retries:
                time.sleep(0.4 * (attempt + 1) + random.random() * 0.3)
    return dataset_id, "unknown", last_err[:80], None


def load_ids(args) -> List[int]:
    if args.ids_file:
        ids = []
        for line in Path(args.ids_file).read_text().splitlines():
            token = line.strip().split(",")[0].strip()
            if token.isdigit():
                ids.append(int(token))
        return ids
    import pandas as pd  # 延迟导入
    df = pd.read_parquet(args.features, columns=["dataset_id"])
    ids = sorted({int(x) for x in df.dataset_id})
    return ids


def main() -> int:
    ap = argparse.ArgumentParser(description="从平台接口同步不可推荐数据集名单")
    ap.add_argument("--features", default="data/processed/dataset_features.parquet",
                    help="特征表（取全部候选 id）")
    ap.add_argument("--ids-file", default=None, help="只核验这个文件里的 id（每行一个）")
    ap.add_argument("--out", default="models/excluded_dataset_ids.json", help="名单输出路径")
    ap.add_argument("--csv", default=None, help="明细报告 CSV 路径（可选，便于人工核对）")
    ap.add_argument("--workers", type=int, default=8, help="并发数（默认 8，请勿过大以免影响线上）")
    ap.add_argument("--timeout", type=int, default=15, help="单请求超时秒数")
    ap.add_argument("--retries", type=int, default=2, help="失败重试次数")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 个（试跑用）")
    ap.add_argument("--exclude-unknown", action="store_true",
                    help="把查询失败的也列入排除名单（默认不列，避免误杀）")
    ap.add_argument("--dry-run", action="store_true", help="只打印，不写文件")
    args = ap.parse_args()

    started = time.time()
    ids = load_ids(args)
    if args.limit:
        ids = ids[: args.limit]
    print("待核验数据集: %d 个（并发 %d）" % (len(ids), args.workers))

    results: Dict[int, Tuple[str, str, Optional[str]]] = {}
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(fetch_status, i, args.timeout, args.retries) for i in ids]
        for fut in as_completed(futures):
            did, verdict, detail, name = fut.result()
            results[did] = (verdict, detail, name)
            done += 1
            if done % 500 == 0 or done == len(ids):
                elapsed = time.time() - started
                print("  进度 %d/%d  (%.0f 秒，%.1f req/s)" % (done, len(ids), elapsed, done / max(elapsed, 0.001)))

    onsale = [i for i, v in results.items() if v[0] == "onsale"]
    delisted = sorted(i for i, v in results.items() if v[0] == "delisted")
    missing = sorted(i for i, v in results.items() if v[0] == "missing")
    unknown = sorted(i for i, v in results.items() if v[0] == "unknown")

    print()
    print("=" * 70)
    print("核验结果（耗时 %.0f 秒）" % (time.time() - started))
    print("  在售   datasetStatus=1 : %d" % len(onsale))
    print("  已下架 datasetStatus!=1: %d" % len(delisted))
    print("  已删除 接口无数据      : %d" % len(missing))
    print("  查询失败(不排除)       : %d" % len(unknown))
    if unknown and len(unknown) / max(len(ids), 1) > 0.05:
        print("  ⚠ 失败比例 >5%%，建议降低并发(--workers 4)或加 --retries 后重跑")

    excluded = sorted(set(delisted) | set(missing))
    if args.exclude_unknown:
        excluded = sorted(set(excluded) | set(unknown))
    print("  → 最终排除名单: %d 个" % len(excluded))
    if delisted[:20]:
        print("  已下架示例: %s" % delisted[:20])
    if missing[:20]:
        print("  已删除示例: %s" % missing[:20])

    if args.csv:
        out_csv = Path(args.csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="", encoding="utf-8-sig") as fh:
            w = csv.writer(fh)
            w.writerow(["dataset_id", "verdict", "detail", "dataset_name"])
            for did in sorted(results):
                verdict, detail, name = results[did]
                w.writerow([did, verdict, detail, name or ""])
        print("  明细已写入: %s" % out_csv)

    if args.dry_run:
        print("\n[dry-run] 未写名单文件。预览前 50 个: %s" % excluded[:50])
        return 0

    payload = {
        "_comment": "不可推荐数据集黑名单（服务启动时读取并从内存索引彻底剔除）。由 scripts/sync_excluded_from_api.py 从平台接口生成。",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "rule": "平台接口 datasetStatus != 1（已下架）或接口返回“数据集不存在”（已删除）",
        "source": API_URL,
        "stats": {
            "checked": len(ids), "onsale": len(onsale), "delisted": len(delisted),
            "missing": len(missing), "unknown": len(unknown),
        },
        "ids": excluded,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n已写入名单: %s" % out)
    print("生效方式: docker restart recommendation-api（约 30 秒）；启动日志会打印 Excluded N datasets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
