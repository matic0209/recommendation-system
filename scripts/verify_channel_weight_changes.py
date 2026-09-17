"""权重 / 列表配额 / 相关度特征的运行时自检脚本（不发请求，纯本地逻辑校验）。

用法（需要装好项目依赖的解释器，且在仓库根目录执行）：
    cd <repo> && python scripts/verify_channel_weight_changes.py

覆盖点：
  1) app.main 可正常 import
  2) 渠道权重次序 vector=content > behavior > popular，且能抵抗 CTR 训练结果的
     反向覆盖（限幅 + 保序）
  3) _compute_relevance_features：5 个相关度特征取值与训练口径一致
  4) 无标签（NaN）不会被误判成"标签相同"
  5) 列表配额：popular 占比上限、非目标行业单一类别上限、目标自身行业豁免
  6) 不可售/已下架数据集剔除（黑名单 + 多级索引清理）
  7) _augment_weight：默认值 + 实验覆盖 + 非法值回退
  8) _env_float：非法环境变量回退默认值
  9) 动态权重：按比例转移后总量守恒、popular 不再被放大
 10) MMR λ 与探索率的默认值/环境变量覆盖
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.getcwd())

import pandas as pd  # noqa: E402

import app.main as m  # noqa: E402

FAILED = []


def check(name: str, cond: bool, extra: str = "") -> None:
    status = "PASS" if cond else "FAIL"
    if not cond:
        FAILED.append(name)
    print(f"[{status}] {name}" + (f"  ({extra})" if extra else ""))


def section(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


section("1) 模块 import")
check("app.main 可导入", True, f"文件 {m.__file__}")
check("新增常量存在", hasattr(m, "DEFAULT_AUGMENT_WEIGHTS") and hasattr(m, "TOP_CATEGORY_TOKENS"))
print("   DEFAULT_CHANNEL_WEIGHTS =", m.DEFAULT_CHANNEL_WEIGHTS)
print("   DEFAULT_AUGMENT_WEIGHTS =", m.DEFAULT_AUGMENT_WEIGHTS)
print("   EXPLORATION_EPSILON =", m.EXPLORATION_EPSILON,
      "| MAX_POPULAR_SHARE =", m.MAX_POPULAR_SHARE,
      "| MAX_OTHER_CATEGORY_SHARE =", m.MAX_OTHER_CATEGORY_SHARE)

section("2) 渠道权重保序：vector=content > behavior > popular")
base = m._enforce_channel_weight_order(dict(m.DEFAULT_CHANNEL_WEIGHTS))
check("默认表本身满足次序",
      min(base["vector"], base["content"]) > base["behavior"] > base["popular"],
      str({k: round(v, 4) for k, v in base.items()}))
# 模拟 models/channel_weights.json 给出与业务规则相反的值
trained = {"behavior": 9.0, "content": 5.0, "vector": 0.3, "popular": 9.0}
fixed = m._enforce_channel_weight_order({**dict(m.DEFAULT_CHANNEL_WEIGHTS), **trained})
print("   反向覆盖输入 =", trained)
print("   校正结果     =", {k: round(v, 4) for k, v in fixed.items()})
check("校正后仍满足 vector=content > behavior > popular",
      min(fixed["vector"], fixed["content"]) > fixed["behavior"] > fixed["popular"])
check("限幅生效（每个渠道都在默认值 ±40% 内）",
      all(
          abs(fixed[c] - m.DEFAULT_CHANNEL_WEIGHTS[c]) <= m.DEFAULT_CHANNEL_WEIGHTS[c] * 0.4 + 1e-9
          for c in ("vector", "content", "behavior", "popular")
      ))
check("popular 不会被抬到接近 content",
      fixed["popular"] < min(fixed["vector"], fixed["content"]) * 0.2)

section("3) _compute_relevance_features")
raw = pd.DataFrame(
    {
        "dataset_id": [1, 2, 3, 4, 5],
        "tag": [
            "金融财经;数据交易",      # 1 = 目标
            "金融财经;数据集",        # 2 同类别 + 1 个共同标签
            "医疗健康;医院",          # 3 不同类别
            None,                     # 4 无标签（NaN）
            "金融财经;数据交易",       # 5 与目标完全相同
        ],
    }
).set_index("dataset_id")
categories = {1: ["金融财经"], 2: ["金融财经", "互联网科技"], 3: ["医疗健康"], 5: ["金融财经"]}

rel = m._compute_relevance_features([1, 2, 3, 4, 5], raw, 1, categories)
print(rel.to_string())
check("列齐全", list(rel.columns) == [
    "tag_overlap_count", "tag_jaccard_similarity", "same_top_category",
    "category_overlap", "category_match",
], str(list(rel.columns)))
check("目标自身 tag_overlap=2", rel.loc[1, "tag_overlap_count"] == 2.0)
check("目标自身 jaccard=1", rel.loc[1, "tag_jaccard_similarity"] == 1.0)
check("#2 标签交集=1", rel.loc[2, "tag_overlap_count"] == 1.0)
check("#2 jaccard=1/3", abs(rel.loc[2, "tag_jaccard_similarity"] - 1 / 3) < 1e-9)
check("#2 同类别=1", rel.loc[2, "same_top_category"] == 1.0)
check("#2 category_overlap=1", rel.loc[2, "category_overlap"] == 1.0)
check("#2 category_match=1", rel.loc[2, "category_match"] == 1.0)
check("#3 不同类别 same_top_category=0", rel.loc[3, "same_top_category"] == 0.0)
check("#3 category_match=0", rel.loc[3, "category_match"] == 0.0)
check("#4 NaN 标签不被误判为相同",
      rel.loc[4, "tag_overlap_count"] == 0.0 and rel.loc[4, "tag_jaccard_similarity"] == 0.0)
check("#5 完全相同 jaccard=1", rel.loc[5, "tag_jaccard_similarity"] == 1.0)

rel_none = m._compute_relevance_features([1, 2], raw, None, categories)
check("target=None 时全为 0", rel_none.values.sum() == 0.0 and rel_none.shape == (2, 5))
rel_empty = m._compute_relevance_features([], raw, 1, categories)
check("空候选返回空表", rel_empty.empty and list(rel_empty.columns) == list(rel.columns))
rel_missing = m._compute_relevance_features([999], raw, 1, categories)
check("候选不在 raw 中 → 0", rel_missing.values.sum() == 0.0)
rel_strkeys = m._compute_relevance_features([1, 2], raw, 1, {"1": ["金融财经"], "2": ["金融财经"]})
check("item_to_categories 用 str 键也能命中", rel_strkeys.loc[2, "category_match"] == 1.0)
rel_nocat = m._compute_relevance_features([1, 2], raw, 1, None)
check("无类别索引时 category 列为 0", rel_nocat[["category_overlap", "category_match"]].values.sum() == 0.0)

section("4) 最终列表配额（复现线上『鼠标指针刷屏』场景）")
# 线上实况：目标数据集是"具身智能"类，候选池里混着 10 条热销鼠标指针（popular 渠道）
# + 若干正常候选（vector/content 渠道）。这里用 12 条互不相同的正常候选作对照。
mouse_ids = list(range(20001, 20011))          # 10 条：同一行业类别（数字产品）
other_ids = list(range(30001, 30013))          # 12 条：各自独立类别，覆盖足够多位置
ranked = mouse_ids + other_ids
item_to_categories = {i: ["数字产品"] for i in mouse_ids}
item_to_categories.update({i: f"类别{i}" for i in other_ids})

# 场景 A：目标数据集不是"数字产品"，鼠标指针来自 popular 渠道（线上最典型的情况）
reasons_popular = {**{i: "popular" for i in mouse_ids}, **{i: "vector" for i in other_ids}}
sel_a = m._apply_list_quotas(ranked, reasons_popular, limit=12,
                             target_dataset_id=99999, item_to_categories=item_to_categories)
mouse_a = sum(1 for i in sel_a if i in mouse_ids)
print("   场景A（鼠标指针来自 popular）结果:", sel_a)
check("A: 12 条里 popular 来源最多 3 条", mouse_a <= 3, f"实际 {mouse_a}")
check("A: 返回条数仍为 12", len(sel_a) == 12, f"实际 {len(sel_a)}")
check("A: 未被配额挡掉的正常候选被优先补进来",
      sum(1 for i in sel_a if i in other_ids) >= 9)

# 场景 B：鼠标指针不来自 popular，而是 vector 渠道 → 检验"非目标行业单一类别上限"
reasons_vector = {i: "vector" for i in ranked}
sel_b = m._apply_list_quotas(ranked, reasons_vector, limit=12,
                             target_dataset_id=99999, item_to_categories=item_to_categories)
mouse_b = sum(1 for i in sel_b if i in mouse_ids)
print("   场景B（全部来自 vector，目标非数字产品）结果:", sel_b)
check("B: 非目标行业的单一类别最多 3 条", mouse_b <= 3, f"实际 {mouse_b}")
check("B: 返回条数仍为 12", len(sel_b) == 12, f"实际 {len(sel_b)}")

# 场景 C：目标数据集本身就是"数字产品"（鼠标指针详情页）→ 自身行业应豁免
item_to_categories[99999] = ["数字产品"]
sel_c = m._apply_list_quotas(ranked, reasons_vector, limit=12,
                             target_dataset_id=99999, item_to_categories=item_to_categories)
mouse_c = sum(1 for i in sel_c if i in mouse_ids)
print("   场景C（目标即数字产品）结果:", sel_c)
check("C: 目标自身行业豁免类别配额（同类可占多数）", mouse_c > 3, f"实际 {mouse_c}")
check("C: 返回条数仍为 12", len(sel_c) == 12, f"实际 {len(sel_c)}")

# 场景 D：几乎全是超配额项时，放宽补齐以保证条数不缩水（有日志）
sel_d = m._apply_list_quotas(ranked, reasons_popular, limit=12,
                             target_dataset_id=99999,
                             item_to_categories={i: ["数字产品"] for i in mouse_ids})
print("   场景D（候选几乎全被配额挡住）结果条数:", len(sel_d))
check("D: 极端情况下仍返回 12 条（放宽兜底）", len(sel_d) == 12, f"实际 {len(sel_d)}")

check("E: 配额按 limit 缩放（limit=12 → 3；limit=20 → 5）",
      m._apply_list_quotas(list(range(1, 41)),
                           {i: "popular" for i in range(1, 41)},
                           limit=20,
                           item_to_categories={i: f"C{i}" for i in range(1, 41)})[:5]
      == list(range(1, 6)) and len(
          m._apply_list_quotas([i for i in range(1, 41) if i % 2],
                               {i: "popular" for i in range(1, 41) if i % 2},
                               limit=20,
                               item_to_categories={i: f"C{i}" for i in range(1, 41)})) >= 5)

section("5) 不可售 / 已下架数据集剔除")
bundle = SimpleNamespace(
    behavior={1: {2: 0.9, 999: 0.5}, 999: {2: 0.8}},
    content={1: {999: 0.7, 3: 0.4}},
    vector={1: [{"dataset_id": 999, "score": 20.0}, {"dataset_id": 3, "score": 18.0}]},
    popular=[999, 3, 2],
)
recall = {
    "tag_to_items": {"金融财经": {999, 3}},
    "category_to_items": {"金融财经": {999, 3}},
    "price_bucket_index": {"0": {999, 3}},
    "item_to_tags": {999: ["x"], 3: ["y"]},
    "item_to_categories": {999: ["金融财经"], 3: ["医疗健康"]},
    "user_similarity": {7: [(999, 0.9), (3, 0.8)]},
}
metadata = {999: {"title": "已下架"}, 3: {"title": "正常"}}
tags = {999: ["x"], 3: ["y"]}
pruned = m._prune_unavailable_datasets(
    bundle=bundle, recall_indices=recall, metadata=metadata, dataset_tags=tags, excluded={999}
)
print(f"   清理引用数 = {pruned}")
check("元数据/标签已剔除", 999 not in metadata and 999 not in tags)
check("behavior 键与邻居都已剔除",
      999 not in bundle.behavior and all(999 not in v for v in bundle.behavior.values()))
check("content 邻居已剔除", 999 not in bundle.content[1])
check("vector 邻居已剔除", all(e["dataset_id"] != 999 for e in bundle.vector[1]))
check("popular 榜单已剔除", 999 not in bundle.popular)
check("倒排索引已剔除",
      999 not in recall["tag_to_items"]["金融财经"] and 999 not in recall["price_bucket_index"]["0"])
check("类别索引已剔除", 999 not in recall["item_to_categories"] and 999 not in recall["item_to_tags"])
check("相似用户列表已剔除", all(entry[0] != 999 for entry in recall["user_similarity"][7]))

scores, reasons_d = {999: 1.0, 3: 0.5}, {999: "popular", 3: "tag"}
dropped = m._drop_excluded_from_scores(scores, reasons_d, {999})
check("请求内兜底剔除生效", dropped == 1 and 999 not in scores and 999 not in reasons_d)

os.environ["TMP_EXCLUDED"] = "111, 222,abc"
os.environ["RECO_EXCLUDED_DATASET_IDS"] = os.environ["TMP_EXCLUDED"]
loaded = m._load_excluded_dataset_ids()
check("黑名单解析（含非法项容错）", loaded == {111, 222}, str(sorted(loaded)))
del os.environ["RECO_EXCLUDED_DATASET_IDS"], os.environ["TMP_EXCLUDED"]

section("6) _augment_weight 覆盖机制")
check("默认 12cat=0.6", m._augment_weight("12cat") == 0.6)
check("默认 tag=0.4", m._augment_weight("tag") == 0.4)
check("实验覆盖 augment_12cat=0.9", m._augment_weight("12cat", {"augment_12cat": 0.9}) == 0.9)
check("非法覆盖值回退默认", m._augment_weight("12cat", {"augment_12cat": "abc"}) == 0.6)
check("未提供覆盖时回退默认", m._augment_weight("usercf", {"content": 0.9}) == 0.6)

section("7) _env_float 安全性")
check("缺省时返回默认值", m._env_float("NO_SUCH_ENV_XYZ", 0.42) == 0.42)
os.environ["TMP_ENV_FLOAT"] = "0.7"
check("正常读取", m._env_float("TMP_ENV_FLOAT", 0.5) == 0.7)
os.environ["TMP_ENV_FLOAT"] = "abc"
check("非法值回退默认", m._env_float("TMP_ENV_FLOAT", 0.5) == 0.5)
os.environ["TMP_ENV_FLOAT"] = "9"
check("超上限被夹住", m._env_float("TMP_ENV_FLOAT", 0.5, maximum=0.9) == 0.9)
del os.environ["TMP_ENV_FLOAT"]

section("8) 动态渠道权重：按比例转移")
bundle_dyn = SimpleNamespace(behavior={}, content={}, vector={})


class _State:
    dataset_tags: dict = {}
    user_history: dict = {}


adj = m._compute_dynamic_channel_weights(
    m.DEFAULT_CHANNEL_WEIGHTS, dataset_id=1, user_id=None, bundle=bundle_dyn, state=_State()
)
print("   base =", m.DEFAULT_CHANNEL_WEIGHTS)
print("   adj  =", {k: round(v, 4) for k, v in adj.items()})
check("无用户历史时 total 守恒",
      abs(sum(adj.values()) - sum(m.DEFAULT_CHANNEL_WEIGHTS.values())) < 1e-9)
check("popular 不被放大（<0.15）", adj["popular"] < 0.15, f"popular={adj['popular']:.4f}")

section("9) MMR λ 与探索率")
check("similar 默认 λ=0.4", m._compute_mmr_lambda(endpoint="similar", request_context=None) == 0.4)
check("detail 默认 λ=0.5", m._compute_mmr_lambda(endpoint="recommend_detail", request_context=None) == 0.5)
check("search 场景 λ=0.6",
      m._compute_mmr_lambda(endpoint="recommend_detail", request_context={"source": "search"}) == 0.6)
check("landing 场景 λ=0.3",
      m._compute_mmr_lambda(endpoint="recommend_detail", request_context={"source": "landing"}) == 0.3)
check("移动端 λ 再降 0.1",
      abs(m._compute_mmr_lambda(endpoint="recommend_detail", request_context={"device_type": "mobile"}) - 0.4) < 1e-9)
check("默认探索率 0.10", m.EXPLORATION_EPSILON == 0.10)

out = m._apply_exploration(list(range(100)), set(range(1000)), epsilon=0.10)
check("探索保留 90% 高分段", out[:90] == list(range(90)), f"len={len(out)}")
check("探索总量不变", len(out) == 100)

section("结果：" + ("全部通过" if not FAILED else f"{len(FAILED)} 项失败 -> {FAILED}"))
sys.exit(1 if FAILED else 0)
