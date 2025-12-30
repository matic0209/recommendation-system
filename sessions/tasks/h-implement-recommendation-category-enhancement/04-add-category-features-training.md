# 子任务04: 添加类别相关性特征（训练阶段）

## 目标
在`pipeline/train_models.py`中添加3个类别相关性特征的计算逻辑，用于LightGBM Ranker训练。

## 成功标准
- [ ] 在`_prepare_ranking_dataset()`中成功添加3个新特征
- [ ] 添加4个辅助函数（tag解析、overlap计算、jaccard、same_category）
- [ ] 特征计算逻辑正确（与Context Manifest一致）
- [ ] 代码通过语法检查
- [ ] 使用测试数据验证特征值合理

## 新增特征定义

1. **tag_overlap_count**: target与candidate共享的tag数量（float）
2. **tag_jaccard_similarity**: Jaccard相似度 = |A∩B| / |A∪B|（float, [0,1]）
3. **same_top_category**: 是否有相同的大类标签（binary, 0/1）

## 实施步骤

### 1. 添加辅助函数

在`pipeline/train_models.py`文件开头（约Line 30-50）添加：

```python
# ===== Category Relevance Helper Functions =====

def _parse_tags(tag_str: str) -> List[str]:
    """
    解析tag字符串为list

    Args:
        tag_str: 分号分隔的tag字符串，如"金融;数据;科技"

    Returns:
        标签列表，已转小写并去除空格
    """
    if not tag_str or pd.isna(tag_str):
        return []
    return [t.strip().lower() for t in str(tag_str).split(";") if t.strip()]


def _count_tag_overlap(tags1_str: str, tags2_str: str) -> float:
    """
    计算两个tag字符串的重叠数量

    Args:
        tags1_str: target item的tags
        tags2_str: candidate item的tags

    Returns:
        重叠tag的数量
    """
    tags1 = set(_parse_tags(tags1_str))
    tags2 = set(_parse_tags(tags2_str))
    return float(len(tags1 & tags2))


def _jaccard_similarity(tags1: List[str], tags2: List[str]) -> float:
    """
    计算Jaccard相似度

    Args:
        tags1: 第一个tag列表
        tags2: 第二个tag列表

    Returns:
        Jaccard相似度 [0, 1]
    """
    if not tags1 or not tags2:
        return 0.0
    set1 = set(t.lower().strip() for t in tags1 if t)
    set2 = set(t.lower().strip() for t in tags2 if t)
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return float(intersection) / float(union) if union > 0 else 0.0


def _has_same_top_category(tags1_str: str, tags2_str: str) -> int:
    """
    检查两个item是否有相同的top category（标准大类）

    Args:
        tags1_str: target item的tags
        tags2_str: candidate item的tags

    Returns:
        1 if有相同大类, else 0
    """
    # 定义标准大类列表
    TOP_CATEGORIES = {
        "金融", "医疗健康", "政府政务", "交通运输",
        "教育培训", "能源环保", "农业", "科技",
        "商业零售", "文化娱乐", "社会民生"
    }

    tags1 = set(_parse_tags(tags1_str))
    tags2 = set(_parse_tags(tags2_str))

    # 找到两个item的top categories
    cat1 = tags1 & TOP_CATEGORIES
    cat2 = tags2 & TOP_CATEGORIES

    # 是否有交集
    return 1 if len(cat1 & cat2) > 0 else 0
```

### 2. 修改`_prepare_ranking_dataset()`函数

在`pipeline/train_models.py::_prepare_ranking_dataset()`（约Line 928-1114）中修改：

**位置1**: 在Line 944-946处，确保tag字段已处理：
```python
# 已有代码（确认存在）
base["tag"] = base.get("tag", "").fillna("").astype(str)
```

**位置2**: 在Line 999后（content_richness特征后）添加类别特征：

```python
    # === Phase 2 (continued): Add Category Relevance Features ===
    LOGGER.info("Computing category relevance features...")

    # 我们需要target_dataset_id作为参考
    # 训练时，使用每个request中position=1的item作为"pseudo target"
    if "request_id" in merged.columns and "position" in merged.columns:
        # 找到每个request的第一个item（假设为target）
        target_items = merged[merged["position"] == 1][["request_id", "dataset_id", "tag"]]
        target_items = target_items.rename(columns={
            "dataset_id": "target_dataset_id",
            "tag": "target_tag"
        })

        # Merge回merged DataFrame
        merged = merged.merge(target_items, on="request_id", how="left")

        # 计算类别相关性特征
        merged["tag_overlap_count"] = merged.apply(
            lambda row: _count_tag_overlap(
                row.get("target_tag", ""),
                row.get("tag", "")
            ),
            axis=1
        )

        merged["tag_jaccard_similarity"] = merged.apply(
            lambda row: _jaccard_similarity(
                _parse_tags(row.get("target_tag", "")),
                _parse_tags(row.get("tag", ""))
            ),
            axis=1
        )

        merged["same_top_category"] = merged.apply(
            lambda row: float(_has_same_top_category(
                row.get("target_tag", ""),
                row.get("tag", "")
            )),
            axis=1
        )

        # 清理临时列
        merged = merged.drop(columns=["target_dataset_id", "target_tag"], errors="ignore")

        LOGGER.info(f"Added category features: tag_overlap_count, tag_jaccard_similarity, same_top_category")
    else:
        # 如果没有request_id或position，填充默认值
        LOGGER.warning("Missing request_id/position columns, filling category features with 0")
        merged["tag_overlap_count"] = 0.0
        merged["tag_jaccard_similarity"] = 0.0
        merged["same_top_category"] = 0.0
```

**位置3**: 在Line 1000-1005处，更新特征列表：

```python
    # 更新base_columns列表
    base_columns = [
        "price_log", "description_length", "tag_count", "weight_log",
        "interaction_count", "popularity_rank", "popularity_percentile",
        "price_bucket", "days_since_last_interaction", "interaction_density",
        "has_description", "has_tags", "content_richness",
        # 新增类别相关性特征
        "tag_overlap_count", "tag_jaccard_similarity", "same_top_category"
    ]
```

## 验证步骤

### 1. 语法检查
```bash
python3 -m py_compile pipeline/train_models.py
```

### 2. 使用小数据集测试
```python
# 创建测试脚本 scripts/test_category_features.py
import pandas as pd
from pipeline.train_models import (
    _parse_tags,
    _count_tag_overlap,
    _jaccard_similarity,
    _has_same_top_category
)

# 测试数据
test_cases = [
    {
        "tags1": "金融;数据;科技",
        "tags2": "金融;医疗健康",
        "expected_overlap": 1.0,
        "expected_jaccard": 0.25,  # 1 / 4
        "expected_same_cat": 1
    },
    {
        "tags1": "农业;环保",
        "tags2": "科技;商业零售",
        "expected_overlap": 0.0,
        "expected_jaccard": 0.0,
        "expected_same_cat": 0
    },
]

print("测试类别特征函数...")
for i, case in enumerate(test_cases):
    overlap = _count_tag_overlap(case["tags1"], case["tags2"])
    jaccard = _jaccard_similarity(
        _parse_tags(case["tags1"]),
        _parse_tags(case["tags2"])
    )
    same_cat = _has_same_top_category(case["tags1"], case["tags2"])

    assert overlap == case["expected_overlap"], f"Case {i}: overlap不匹配"
    assert abs(jaccard - case["expected_jaccard"]) < 0.01, f"Case {i}: jaccard不匹配"
    assert same_cat == case["expected_same_cat"], f"Case {i}: same_cat不匹配"

    print(f"✅ Case {i} 通过")

print("\n所有测试通过！")
```

### 3. 运行测试
```bash
python3 scripts/test_category_features.py
```

## 注意事项

1. **使用增强后的tags**: 确保数据来自`dataset_features_enhanced.parquet`
2. **特征缺失处理**: 如果无法计算target context，填充0.0
3. **性能优化**: apply()可能较慢，可考虑向量化（但优先保证正确性）
4. **日志输出**: 添加LOGGER.info记录特征计算进度

## 下一步
完成后进入子任务05：添加类别特征（推理阶段）
