#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""典枢推荐系统 · 推荐结果人工核对工具
================================================================

用途
----
把“目标数据集”和它返回的 N 条推荐**并排打印出来**，逐条给出：
    标题 / 价格 / 召回渠道(reason) / 标签 / 相关判定
并汇总成一张“验收表”，最后可选生成一个 HTML 看板（浏览器打开即可肉眼核对）。

为什么需要它
------------
只看“渠道分布”数字容易被误导：
  - `fallback:popular` 占多数 = **发生了降级兜底**，不是正常结果；
  - `popular+rank` = 靠全局热销榜凑数，和当前页面无关；
  - 只有 `content/vector/tag` 才是“因为内容像”而推荐。
所以最终判据是：**人眼看得懂的相关性 + 渠道来源 + 热门占比** 三者一起看。

判定口径（可在 judge() 里按业务需要调整）
-----------------------------------------
  相关       = 候选与目标在“标签”或“12 类行业类别”上有交集
  不相关     = 无交集，且来源是 popular / fallback（典型“热销凑数”）
  待人工判断 = 无交集，但来自 content/vector/tag/price 等（可能是语义相近的技术类数据）

验收标准（业务规则）
--------------------
  1) 12 条里 popular（含 fallback）来源 ≤ 3 条              —— 硬性
  2) content / vector / tag 来源应占多数                    —— 硬性
  3) 与目标相关（人眼可判）的条目应尽量多
  4) 两个不相关的页面，返回结果不应大面积重复
  5) 不应出现已下架数据集（配合 RECO_EXCLUDED_DATASET_IDS / 黑名单）

用法示例
--------
  # 基本：核对 16307 这一个页面
  python3 scripts/verify_local_reco.py --target 16307

  # 一次看多个页面 + 生成 HTML 看板
  python3 scripts/verify_local_reco.py --target 16307 --compare 13830,801 \
      --api http://localhost:18090 --limit 12 --html docs/验证看板.html

  # 对照线上（注意：线上是旧规则旧数据，只作基线）
  python3 scripts/verify_local_reco.py --target 16307 --api http://localhost:18090

注意
----
* 本地服务地址默认 http://localhost:18090（本地容器）；
* 需要 pandas + pyarrow 读 parquet 元数据（项目 venv 或容器内 python 均可）；
* /similar 结果在服务端有 300 秒缓存，改完代码后请先清缓存：
      docker exec reco-redis redis-cli -n 0 flushdb
"""
from __future__ import annotations

import argparse
import html as html_mod
import json
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

# 视为“热销兜底”的渠道前缀
POPULAR_CHANNELS = {"popular", "fallback"}

# 标签/类别交集为空时，这些渠道仍可能是语义相近（标为待人工判断）
SEMANTIC_CHANNELS = {"content", "vector", "tag", "12cat", "category", "usercf", "behavior", "price"}


# ---------------------------------------------------------------- 数据读取
def fetch_similar(api: str, dataset_id: int, limit: int) -> dict:
    """调用推荐接口，返回原始 JSON。"""
    url = f"{api.rstrip('/')}/similar/{dataset_id}?limit={limit}"
    with urllib.request.urlopen(url, timeout=90) as resp:
        return json.loads(resp.read().decode("utf-8"))


def load_metadata(repo: Path) -> pd.DataFrame:
    """读取数据集元数据（标题/标签/价格/公司），用于人工核对。"""
    parquet = repo / "data" / "processed" / "dataset_features.parquet"
    cols = ["dataset_id", "dataset_name", "tag", "price", "create_company_name"]
    df = pd.read_parquet(parquet, columns=cols)
    return df.set_index("dataset_id")


def load_categories(repo: Path) -> Dict[str, List[str]]:
    """读取“数据集 → 12 类行业类别”映射（JSON 的 key 是字符串）。"""
    path = repo / "models" / "item_to_categories.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- 判定逻辑
def split_tags(raw) -> Set[str]:
    """把 'A;B;C' 形式的标签串切成小写集合，忽略空项。"""
    if raw is None or not isinstance(raw, str):
        return set()
    text = raw.replace("；", ";").replace(",", ";")
    return {t.strip().lower() for t in text.split(";") if t.strip()}


def judge(target_tags: Set[str], target_cats: Set[str],
          cand_tags: Set[str], cand_cats: Set[str], reason: str) -> Tuple[str, str]:
    """给出“相关 / 不相关 / 待人工判断”结论及依据。

    返回 (结论, 依据)
    """
    tag_hit = target_tags & cand_tags
    cat_hit = target_cats & cand_cats
    if tag_hit:
        return "相关", "标签命中: " + ",".join(sorted(tag_hit)[:3])
    if cat_hit:
        return "相关", "同类目: " + ",".join(sorted(cat_hit)[:3])

    channel = (reason or "").split("+")[0]
    if channel in POPULAR_CHANNELS:
        return "不相关", "热门凑数（与目标无共同标签/类别）"
    if channel in SEMANTIC_CHANNELS:
        return "待人工判断", f"来自 {channel} 渠道但无标签/类别交集"
    return "待人工判断", channel or "无渠道"


def reason_popular(reason: str) -> bool:
    """该条推荐是否来自热门/兜底渠道。"""
    return any(part in POPULAR_CHANNELS for part in (reason or "").split("+"))


# ---------------------------------------------------------------- 单页核对
def check_one(meta: pd.DataFrame, cats: Dict[str, List[str]],
              dataset_id: int, payload: dict) -> dict:
    """核对单个页面，返回结构化结果。"""
    items = payload.get("similar_items") or []
    target_row = meta.loc[dataset_id] if dataset_id in meta.index else None
    target_tags = split_tags(target_row["tag"]) if target_row is not None else set()
    target_cats = {c.lower() for c in (cats.get(str(dataset_id)) or [])}

    rows = []
    for idx, item in enumerate(items, start=1):
        did = item.get("dataset_id")
        reason = item.get("reason") or ""
        cand_tags: Set[str] = set()
        cand_cats: Set[str] = set()
        name, price, company = "(无元数据)", None, ""
        if did in meta.index:
            row = meta.loc[did]
            name = str(row["dataset_name"])
            price = float(row["price"]) if pd.notna(row["price"]) else None
            company = str(row["create_company_name"] or "")
            cand_tags = split_tags(row["tag"])
        cand_cats = {c.lower() for c in (cats.get(str(did)) or [])}

        verdict, why = judge(target_tags, target_cats, cand_tags, cand_cats, reason)
        rows.append({
            "no": idx, "dataset_id": did, "name": name, "price": price,
            "company": company, "reason": reason, "verdict": verdict, "why": why,
            "tags": ";".join(sorted(cand_tags)) or "(无标签)",
        })

    reason_counter = Counter(r["reason"] for r in rows)
    popular_count = sum(1 for r in rows if reason_popular(r["reason"]))
    related = sum(1 for r in rows if r["verdict"] == "相关")
    return {
        "target_id": dataset_id,
        "target_name": str(target_row["dataset_name"]) if target_row is not None else "(无元数据)",
        "target_tags": ";".join(sorted(target_tags)) or "(无标签)",
        "target_cats": ";".join(sorted(target_cats)) or "(无类别)",
        "limit": len(rows),
        "rows": rows,
        "reason_counter": dict(reason_counter),
        "popular_count": popular_count,
        "related_count": related,
        "max_popular_allowed": 3,
    }


# ---------------------------------------------------------------- 控制台输出
def print_report(result: dict) -> None:
    """把核对结果打印成人类可读的表格。"""
    print("=" * 100)
    print(f"目标数据集 {result['target_id']}：{result['target_name']}")
    print(f"  目标标签：{result['target_tags']}")
    print(f"  目标类别：{result['target_cats']}")
    print("-" * 100)
    print(f"{'#':<3}{'ID':<8}{'价格':<9}{'判定':<12}{'召回渠道':<26}标题")
    for r in result["rows"]:
        price = f"￥{r['price']:.2f}" if r["price"] is not None else "?"
        print(f"{r['no']:<3}{r['dataset_id']:<8}{price:<9}{r['verdict']:<12}{r['reason']:<26}{r['name'][:38]}")
    print("-" * 100)
    print(f"  渠道分布：{result['reason_counter']}")
    print(f"  热门来源：{result['popular_count']} 条（验收要求 ≤ {result['max_popular_allowed']}）"
          f" → {'✅ 达标' if result['popular_count'] <= result['max_popular_allowed'] else '❌ 超标'}")
    print(f"  人眼可判为相关：{result['related_count']} 条")
    print()


def print_overlap(checks: List[dict]) -> None:
    """检查多个页面之间的结果重复率（不相关页面不应大面积重复）。"""
    if len(checks) < 2:
        return
    print("=" * 100)
    print("跨页面重复检查（两个不相关的页面不应返回同一批结果）")
    id_sets = {c["target_id"]: {r["dataset_id"] for r in c["rows"]} for c in checks}
    ids = list(id_sets)
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = ids[i], ids[j]
            inter = id_sets[a] & id_sets[b]
            ratio = len(inter) / max(1, min(len(id_sets[a]), len(id_sets[b])))
            flag = "✅" if ratio <= 0.5 else "⚠️ 重复偏高"
            print(f"  {a} vs {b}: 重合 {len(inter)} 条（{ratio:.0%}）{flag} {sorted(inter)}")
    print()


# ---------------------------------------------------------------- HTML 看板
def build_html(checks: List[dict], out_path: Path, api: str) -> None:
    """生成一个可读的 HTML 看板（用主题变量，跟随 Hermes 明暗主题）。"""
    parts: List[str] = []
    for c in checks:
        rows_html = []
        for r in c["rows"]:
            color = {"相关": "var(--accent)", "不相关": "#c0392b", "待人工判断": "#b7791f"}.get(r["verdict"], "inherit")
            price = f"￥{r['price']:.2f}" if r["price"] is not None else "?"
            rows_html.append(
                f"<tr><td>{r['no']}</td><td>{r['dataset_id']}</td>"
                f"<td style='max-width:340px'>{html_mod.escape(r['name'])}</td>"
                f"<td>{price}</td>"
                f"<td style='font-family:ui-monospace,monospace;font-size:12px'>{html_mod.escape(r['reason'])}</td>"
                f"<td style='color:{color};font-weight:600'>{r['verdict']}</td>"
                f"<td style='font-size:12px'>{html_mod.escape(r['why'])}</td>"
                f"<td style='font-size:11px;color:var(--muted-foreground)'>{html_mod.escape(r['tags'])[:80]}</td></tr>"
            )
        ok = c["popular_count"] <= c["max_popular_allowed"]
        parts.append(f"""
<section>
  <h2>目标 {c['target_id']}　<span class="sub">{html_mod.escape(c['target_name'][:70])}</span></h2>
  <p class="meta">目标标签：{html_mod.escape(c['target_tags'])}　｜　目标类别：{html_mod.escape(c['target_cats'])}</p>
  <p class="badges">
    <span class="badge {'ok' if ok else 'bad'}">热门来源 {c['popular_count']} / 上限 {c['max_popular_allowed']}　{'达标' if ok else '超标'}</span>
    <span class="badge">人眼可判相关 {c['related_count']} 条</span>
    <span class="badge">共 {c['limit']} 条</span>
  </p>
  <table>
    <thead><tr><th>#</th><th>数据集ID</th><th>标题</th><th>价格</th><th>召回渠道</th><th>判定</th><th>依据</th><th>标签</th></tr></thead>
    <tbody>{''.join(rows_html)}</tbody>
  </table>
</section>""")

    overlap_html = ""
    if len(checks) >= 2:
        id_sets = {c["target_id"]: {r["dataset_id"] for r in c["rows"]} for c in checks}
        ids = list(id_sets)
        lines = []
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = ids[i], ids[j]
                inter = id_sets[a] & id_sets[b]
                ratio = len(inter) / max(1, min(len(id_sets[a]), len(id_sets[b])))
                lines.append(f"<li>{a} vs {b}：重合 {len(inter)} 条（{ratio:.0%}）"
                             f"{'✅ 正常' if ratio <= 0.5 else '⚠️ 重复偏高'}　{sorted(inter)}</li>")
        overlap_html = ("<section><h2>跨页面重复检查</h2><p class='meta'>"
                        "两个不相关的页面不应返回同一批结果</p><ul>" + "".join(lines) + "</ul></section>")

    doc = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>推荐结果核对看板</title>
<style>
  body {{ color: var(--foreground); font-family: inherit; margin: 0; padding: 4px 0; }}
  h1 {{ font-size: 18px; margin: 0 0 4px; }}
  h2 {{ font-size: 15px; margin: 18px 0 6px; }}
  .sub {{ font-size: 12px; color: var(--muted-foreground); font-weight: 400; }}
  .meta {{ font-size: 12px; color: var(--muted-foreground); margin: 2px 0 6px; }}
  .badges {{ margin: 6px 0; }}
  .badge {{ display: inline-block; font-size: 12px; padding: 2px 8px; margin-right: 6px;
            border: 1px solid var(--border); border-radius: 10px; color: var(--muted-foreground); }}
  .badge.ok {{ color: var(--accent); border-color: var(--accent); }}
  .badge.bad {{ color: #c0392b; border-color: #c0392b; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th, td {{ text-align: left; padding: 4px 6px; border-bottom: 1px solid var(--border); vertical-align: top; }}
  th {{ color: var(--muted-foreground); font-weight: 600; font-size: 12px; }}
  ul {{ font-size: 12px; color: var(--muted-foreground); }}
</style></head><body>
<h1>推荐结果核对看板</h1>
<p class="meta">数据来自 <code>{html_mod.escape(api)}</code>　｜　判定口径：标签或 12 类类别有交集＝相关；无交集且来自 popular/fallback＝不相关</p>
{''.join(parts)}
{overlap_html}
</body></html>"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(doc, encoding="utf-8")
    print(f"HTML 看板已生成：{out_path}")


# ---------------------------------------------------------------- 主流程
def main() -> int:
    parser = argparse.ArgumentParser(description="推荐结果人工核对工具")
    parser.add_argument("--target", type=int, required=True, help="目标数据集 ID")
    parser.add_argument("--compare", default="", help="额外核对的数据集 ID，逗号分隔")
    parser.add_argument("--api", default="http://localhost:18090", help="推荐服务地址")
    parser.add_argument("--limit", type=int, default=12, help="每个页面取多少条")
    parser.add_argument("--repo", default=".", help="仓库根目录（用于读 parquet / models）")
    parser.add_argument("--html", default="", help="生成 HTML 看板的输出路径")
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    meta = load_metadata(repo)
    cats = load_categories(repo)

    targets = [args.target] + [int(x) for x in args.compare.split(",") if x.strip()]
    checks = []
    for ds in targets:
        try:
            payload = fetch_similar(args.api, ds, args.limit)
        except Exception as exc:  # 网络/接口异常直接提示，避免误判为“推荐结果为 0”
            print(f"❌ 请求 {ds} 失败：{exc}")
            continue
        result = check_one(meta, cats, ds, payload)
        checks.append(result)
        print_report(result)

    print_overlap(checks)
    if args.html and checks:
        build_html(checks, Path(args.html), args.api)
    return 0 if checks else 1


if __name__ == "__main__":
    raise SystemExit(main())
