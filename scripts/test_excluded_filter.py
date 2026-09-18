#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""下架过滤相关函数的单元测试（自包含，不需要启动服务）

背景：`app/main.py` 里的三个函数决定"哪些下架数据会被剔除"：
    _load_excluded_dataset_ids()   —— 读取名单（环境变量 + 文件）
    _prune_unavailable_datasets()   —— 启动时从内存索引彻底剔除
    _drop_excluded_from_scores()    —— 请求期兜底过滤

这些函数**没有启动服务也可能出错**（例如索引结构变化、名单格式异常），
历史上正是因为缺少函数级测试，才让一个 O(键数×名单规模) 的性能写法混了进去。
本脚本用 ast 从 app/main.py 提取函数源码后在本进程内执行，逐个断言覆盖。

用法：python3 scripts/test_excluded_filter.py         # 退出码 0 = 全部通过
"""
from __future__ import annotations

import ast
import os
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch

MAIN = Path(__file__).resolve().parent.parent / "app" / "main.py"
WANTED = ("_load_excluded_dataset_ids", "_prune_unavailable_datasets", "_drop_excluded_from_scores")


class _FakeLogger:
    def __getattr__(self, _name):
        return lambda *a, **k: None


def _extract_functions() -> Dict[str, Any]:
    """从 app/main.py 里按名字取出函数源码并执行，返回函数字典。

    不 import app.main（那会拉起 fastapi/redis 等整套依赖），只取需要的函数，
    这样测试又快又隔离。
    """
    source = MAIN.read_text(encoding="utf-8")
    lines = source.splitlines()
    tree = ast.parse(source)
    funcs: Dict[str, Any] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in WANTED:
            segment = "\n".join(lines[node.lineno - 1: node.end_lineno])
            funcs[node.name] = textwrap.dedent(segment)
    missing = [name for name in WANTED if name not in funcs]
    if missing:
        raise SystemExit(f"❌ 在 app/main.py 中找不到函数: {missing}（函数被重命名或删除？）")

    namespace: Dict[str, Any] = {
        "os": os,
        "json": __import__("json"),
        "LOGGER": _FakeLogger(),
        "MODELS_DIR": Path(tempfile.mkdtemp()),
        "Set": set, "Dict": dict, "Optional": None, "Any": Any,
    }
    # typing 名字在函数签名里用到
    import typing
    for name in ("Set", "Dict", "Optional", "Any", "List"):
        namespace[name] = getattr(typing, name)
    for name, src in funcs.items():
        exec(compile(src, f"<extracted:{name}>", "exec"), namespace)  # noqa: S102
    return {name: namespace[name] for name in funcs}


F = _extract_functions()
failures = []


def check(label: str, cond: bool, extra: str = "") -> None:
    print(("  ✅ " if cond else "  ❌ ") + label + (f"  {extra}" if extra and not cond else ""))
    if not cond:
        failures.append(label)


print("=" * 74)
print("① _load_excluded_dataset_ids：名单来源与异常容错")
print("=" * 74)
MODELS_DIR = Path(tempfile.mkdtemp())
F["_load_excluded_dataset_ids"].__globals__["MODELS_DIR"] = MODELS_DIR
tmp_json = MODELS_DIR / "excluded_dataset_ids.json"

tmp_json.write_text("[1, 2, 3]", encoding="utf-8")
check("列表格式 [1,2,3]", F["_load_excluded_dataset_ids"]() == {1, 2, 3})

tmp_json.write_text('{"ids": [7, 8], "_comment": "x", "stats": {}}', encoding="utf-8")
check("字典格式（含额外键）", F["_load_excluded_dataset_ids"]() == {7, 8})

tmp_json.write_text('{"ids": ["9", 10, "bad", null]}', encoding="utf-8")
check("混合类型/脏数据不崩且跳过非法项", F["_load_excluded_dataset_ids"]() == {9, 10})

tmp_json.write_text("这不是合法 JSON", encoding="utf-8")
check("文件损坏时不抛异常（返回空集）", F["_load_excluded_dataset_ids"]() == set())

tmp_json.unlink()
with patch.dict(os.environ, {"RECO_EXCLUDED_DATASET_IDS": "11, 12;13  bad"}, clear=False):
    check("环境变量逗号/分号/空格分隔", F["_load_excluded_dataset_ids"]() == {11, 12, 13})
with patch.dict(os.environ, {"RECO_EXCLUDED_DATASET_IDS": ""}, clear=False):
    tmp_json.write_text("[21]", encoding="utf-8")
    check("两种来源同时生效", F["_load_excluded_dataset_ids"]() == {21})

print()
print("=" * 74)
print("② _prune_unavailable_datasets：剔除是否覆盖所有索引结构")
print("=" * 74)


class Bundle:
    def __init__(self):
        self.behavior = {1: {2: 0.9, 3: 0.5}, 2: {1: 0.9}, 3: {1: 0.5}}
        self.content = {1: {2: 0.8, 9: 0.4}, 9: {1: 0.4, 2: 0.3}}
        self.vector = {
            1: [{"dataset_id": 2, "score": 0.7}, {"dataset_id": 9, "score": 0.6},
                {"dataset_id": "坏数据"}],        # 脏数据放在【未被排除】的源上，才能真正走到容错分支
            9: [{"dataset_id": 1, "score": 0.6}],  # 该源整体会被剔除
        }
        self.popular = [1, 2, 9, 3]


metadata = {1: {}, 2: {}, 3: {}, 9: {}}
dataset_tags = {1: ["a"], 2: ["b"], 9: ["c"]}
recall = {
    "tag_to_items": {"a": {1, 2, 9}, "b": [1, 2]},
    "category_to_items": {"cat": {1, 9}},
    "category_index": {"某公司": {1, 2, 9}},          # 企业索引（/similar 的 category 通道用它取候选）
    "price_bucket_index": {"0": [1, 2, 9]},
    "item_to_tags": {1: ["a"], 9: ["c"]},
    "item_to_categories": {"1": ["cat"], 9: ["cat"]},
    "user_similarity": {100: [[1, 0.9], [2, 0.8]], 200: [["坏数据"], [3, 0.5]]},
}
bundle = Bundle()
excluded = {9}

removed = F["_prune_unavailable_datasets"](
    bundle=bundle, recall_indices=recall, metadata=metadata,
    dataset_tags=dataset_tags, excluded=excluded,
)

check("元数据剔除", 9 not in metadata, str(metadata.keys()))
check("标签映射剔除", 9 not in dataset_tags)
check("behavior 作为源被剔除", 9 not in bundle.behavior)
check("behavior 邻居值被剔除", all(9 not in v for v in bundle.behavior.values()))
check("content 作为源被剔除", 9 not in bundle.content)
check("content 邻居值被剔除", all(9 not in v for v in bundle.content.values()))
check("vector 作为源被剔除", 9 not in bundle.vector)
check("vector 条目被剔除", all(e.get("dataset_id") != 9 for ents in bundle.vector.values() for e in ents))
check("vector 脏数据条目保留（不崩溃）",
      any(e.get("dataset_id") == "坏数据" for ents in bundle.vector.values() for e in ents))
check("热门榜剔除", 9 not in bundle.popular)
check("热门榜保序保量（无 9）", bundle.popular == [1, 2, 3], str(bundle.popular))
check("tag_to_items(set) 剔除", all(9 not in v for v in recall["tag_to_items"].values() if isinstance(v, set)))
check("tag_to_items(list) 剔除", recall["tag_to_items"]["b"] == [1, 2])
check("category_to_items 剔除", 9 not in recall["category_to_items"]["cat"])
check("category_index（企业索引）剔除", 9 not in recall["category_index"]["某公司"],
      str(recall["category_index"]))
check("price_bucket_index 剔除", 9 not in recall["price_bucket_index"]["0"])
check("item_to_tags 剔除", 9 not in recall["item_to_tags"])
check("item_to_categories 剔除", 9 not in recall["item_to_categories"])
check("user_similarity 剔除（数字键）", [e for e in recall["user_similarity"][100] if e == [9, 0.9]] == [])
check("user_similarity 脏结构保留", recall["user_similarity"][200] == [["坏数据"], [3, 0.5]])
check("非排除项完好保留", metadata.get(1) is not None and metadata.get(2) is not None)
check("返回值 removed > 0", removed > 0, f"removed={removed}")

empty_calls = F["_prune_unavailable_datasets"](
    bundle=Bundle(), recall_indices={}, metadata={1: {}}, dataset_tags={}, excluded=set())
check("空名单时直接返回 0（不影响既有行为）", empty_calls == 0)

print()
print("=" * 74)
print("③ _drop_excluded_from_scores：请求期兜底")
print("=" * 74)
scores = {1: 0.9, 9: 0.8, 2: 0.7}
reasons = {1: "content+rank", 9: "popular+rank", 2: "vector+rank"}
dropped = F["_drop_excluded_from_scores"](scores, reasons, {9})
check("被排除项从分数表移除", 9 not in scores and dropped == 1)
check("原因表同步移除", 9 not in reasons)
check("其它项不受影响", scores == {1: 0.9, 2: 0.7} and reasons[1] == "content+rank")
check("空名单不改变任何东西",
      F["_drop_excluded_from_scores"]({1: 0.5}, {1: "x"}, set()) == 0)

print()
print("=" * 74)
if failures:
    print(f"❌ 未通过 {len(failures)} 项：")
    for f in failures:
        print("   - " + f)
    raise SystemExit(1)
print("✅ 全部断言通过（下架过滤逻辑覆盖：名单解析 / 全索引剔除 / 脏数据容错 / 请求期兜底）")
sys.exit(0)
