# 子任务05: 添加类别相关性特征（推理阶段）

## 目标
在`app/main.py`中修改5个位置，支持推理时动态计算类别相关性特征（基于target_dataset_id）。

## 成功标准
- [ ] 复制辅助函数到app/main.py
- [ ] `_build_static_ranking_features()`支持target_dataset_id参数
- [ ] `_compute_ranking_features()`传递target_dataset_id
- [ ] `_apply_ranking()`接收target_dataset_id
- [ ] `recommend_for_detail()`传入正确的target
- [ ] 代码通过语法检查

## 实施步骤

### 1. 添加辅助函数到app/main.py

在文件开头（约Line 100-200，导入语句后）添加：

```python
# ===== Category Relevance Helper Functions =====
# (复制自pipeline/train_models.py)

def _parse_tags_for_category(tag_str: str) -> List[str]:
    """解析tag字符串为list (推理版本)"""
    if not tag_str or pd.isna(tag_str):
        return []
    return [t.strip().lower() for t in str(tag_str).split(";") if t.strip()]


def _count_tag_overlap_inference(tags1_str: str, tags2_str: str) -> float:
    """计算tag重叠数量 (推理版本)"""
    tags1 = set(_parse_tags_for_category(tags1_str))
    tags2 = set(_parse_tags_for_category(tags2_str))
    return float(len(tags1 & tags2))


def _jaccard_similarity_inference(tags1_str: str, tags2_str: str) -> float:
    """
    计算Jaccard相似度 (推理版本)

    Note: 与训练时的_jaccard_similarity保持一致
    """
    tags1 = _parse_tags_for_category(tags1_str)
    tags2 = _parse_tags_for_category(tags2_str)

    if not tags1 or not tags2:
        return 0.0

    set1 = set(tags1)
    set2 = set(tags2)
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return float(intersection) / float(union) if union > 0 else 0.0


def _has_same_top_category_inference(tags1_str: str, tags2_str: str) -> float:
    """检查是否有相同的大类 (推理版本，返回float)"""
    TOP_CATEGORIES = {
        "金融", "医疗健康", "政府政务", "交通运输",
        "教育培训", "能源环保", "农业", "科技",
        "商业零售", "文化娱乐", "社会民生"
    }

    tags1 = set(_parse_tags_for_category(tags1_str))
    tags2 = set(_parse_tags_for_category(tags2_str))

    cat1 = tags1 & TOP_CATEGORIES
    cat2 = tags2 & TOP_CATEGORIES

    return 1.0 if len(cat1 & cat2) > 0 else 0.0
```

### 2. 修改`_build_static_ranking_features()`

在`app/main.py`（约Line 1951-2064）修改函数签名和逻辑：

**修改函数签名（Line 1951）**:
```python
def _build_static_ranking_features(
    dataset_ids: List[int],
    raw_features: pd.DataFrame,
    dataset_stats: pd.DataFrame,
    slot_metrics_aggregated: pd.DataFrame = None,
    feature_overrides: Optional[Dict[str, Any]] = None,
    stats_overrides: Optional[Dict[str, Any]] = None,
    target_dataset_id: Optional[int] = None,  # 新增参数
) -> pd.DataFrame:
```

**在函数末尾（Line 2018后）添加类别特征计算**:
```python
    # === Category Relevance Features (需要target context) ===
    if target_dataset_id is not None:
        # 获取target item的tags
        target_tags = raw_features[raw_features["dataset_id"] == target_dataset_id]["tag"].values
        target_tags_str = target_tags[0] if len(target_tags) > 0 else ""

        # 计算类别相关性特征
        features["tag_overlap_count"] = selected["tag"].apply(
            lambda x: _count_tag_overlap_inference(target_tags_str, str(x))
        )
        features["tag_jaccard_similarity"] = selected["tag"].apply(
            lambda x: _jaccard_similarity_inference(target_tags_str, str(x))
        )
        features["same_top_category"] = selected["tag"].apply(
            lambda x: _has_same_top_category_inference(target_tags_str, str(x))
        )
    else:
        # 无target context时填充默认值（与训练时一致）
        features["tag_overlap_count"] = 0.0
        features["tag_jaccard_similarity"] = 0.0
        features["same_top_category"] = 0.0

    return features
```

### 3. 修改`_compute_ranking_features()`

在`app/main.py`（约Line 2067-2200）修改函数签名和调用：

**修改函数签名（Line 2067）**:
```python
def _compute_ranking_features(
    dataset_ids: List[int],
    raw_features: pd.DataFrame,
    dataset_stats: pd.DataFrame,
    slot_metrics_aggregated: pd.DataFrame,
    endpoint: str,
    variant: str,
    request_context: Dict[str, str],
    target_dataset_id: Optional[int] = None,  # 新增参数
) -> pd.DataFrame:
```

**修改调用`_build_static_ranking_features`（Line 2099处）**:
```python
        filled = _build_static_ranking_features(
            missing_ids,
            indexed_raw,
            indexed_stats,
            indexed_slot,
            target_dataset_id=target_dataset_id  # 传递target
        )
```

### 4. 修改`_apply_ranking()`

在`app/main.py`（约Line 2311-2400）修改函数签名和调用：

**修改函数签名（Line 2311）**:
```python
async def _apply_ranking(
    scores: Dict[int, float],
    reasons: Dict[int, str],
    rank_model,
    raw_features: pd.DataFrame,
    dataset_stats: pd.DataFrame,
    slot_metrics_aggregated: pd.DataFrame,
    endpoint: str,
    variant: str,
    request_context: Dict[str, str],
    target_dataset_id: Optional[int] = None,  # 新增参数
) -> Dict[int, float]:
```

**修改调用`_compute_ranking_features`（Line 2350处）**:
```python
        features = _compute_ranking_features(
            dataset_ids=list(scores.keys()),
            raw_features=raw_features,
            dataset_stats=dataset_stats,
            slot_metrics_aggregated=slot_metrics_aggregated,
            endpoint=endpoint,
            variant=variant,
            request_context=request_context,
            target_dataset_id=target_dataset_id  # 传递target
        )
```

### 5. 修改`recommend_for_detail()`调用ranking

在`app/main.py`（约Line 2900-3300，`recommend_for_detail`函数中）：

找到调用`_apply_ranking`的位置（约Line 3100），修改为：

```python
            scores = await _apply_ranking(
                scores=scores,
                reasons=reasons,
                rank_model=state.rank_model,
                raw_features=state.raw_features,
                dataset_stats=state.dataset_stats,
                slot_metrics_aggregated=state.slot_metrics_aggregated,
                endpoint=endpoint,
                variant=variant,
                request_context=request_context,
                target_dataset_id=dataset_id  # 传入target (就是请求的dataset_id)
            )
```

## 验证步骤

### 1. 语法检查
```bash
python3 -m py_compile app/main.py
```

### 2. 单元测试
创建`scripts/test_inference_category_features.py`:

```python
import pandas as pd
from app.main import (
    _parse_tags_for_category,
    _count_tag_overlap_inference,
    _jaccard_similarity_inference,
    _has_same_top_category_inference
)

# 测试辅助函数
tags1 = "金融;数据;科技"
tags2 = "金融;医疗健康"

overlap = _count_tag_overlap_inference(tags1, tags2)
assert overlap == 1.0, f"overlap错误: {overlap}"

jaccard = _jaccard_similarity_inference(tags1, tags2)
assert abs(jaccard - 0.25) < 0.01, f"jaccard错误: {jaccard}"

same_cat = _has_same_top_category_inference(tags1, tags2)
assert same_cat == 1.0, f"same_cat错误: {same_cat}"

print("✅ 推理阶段类别特征函数测试通过！")
```

### 3. 集成测试（模拟推荐请求）
```bash
# 启动本地服务（使用当前代码）
# 发送测试请求
curl "http://localhost:8000/api/v1/recommend/detail?dataset_id=123&user_id=456&limit=10"

# 检查响应正常
# 检查日志中是否有类别特征相关的warning/error
```

## 注意事项

1. **函数命名**: 推理版本的辅助函数加`_inference`后缀，避免与训练版本冲突
2. **target_dataset_id来源**: 在`recommend_for_detail`中，target就是请求的`dataset_id`
3. **特征一致性**: 确保推理时计算的特征与训练时完全一致（名称、逻辑、数据类型）
4. **默认值**: 如果无法获取target，填充0.0（与训练时保持一致）

## 潜在问题

### 问题1: target_dataset_id不在raw_features中
**解决**: 添加检查逻辑
```python
if target_dataset_id is not None and target_dataset_id in raw_features["dataset_id"].values:
    # 正常计算
else:
    # 填充默认值
```

### 问题2: 推理时特征顺序与训练不一致
**解决**: 使用DataFrame保证列顺序，LightGBM会自动对齐

## 下一步
完成后进入子任务06：重新训练模型
