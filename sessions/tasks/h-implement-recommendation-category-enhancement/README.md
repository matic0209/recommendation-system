---
name: h-implement-recommendation-category-enhancement
branch: feature/recommendation-category-enhancement
status: pending
created: 2025-12-30
---

# 推荐系统类别增强：Tags自动化 + 类别相关性特征

## Problem/Goal

当前推荐系统存在**类别跨度过大**的问题：推荐结果中包含金融、医疗、农业等差异很大的类别，用户期望看到更聚焦的推荐。

核心问题：
1. **用户标签质量差**：tags由用户自己打的，不准确且缺失严重
2. **缺乏类别约束**：推荐流程中没有大类过滤机制
3. **排序模型未利用类别信息**：LightGBM Ranker缺少类别相关性特征

解决方案：
1. **Tags自动增强**：使用零样本分类模型（方案C-1）从description提取标准化大类标签
2. **特征工程**：为Ranker添加类别相关性特征（Jaccard相似度、同大类标识等）
3. **模型重训练**：让排序模型学习类别相关性权重

## Success Criteria
- [ ] 完成全量item的tags自动增强，覆盖率>95%
- [ ] 生成增强后的索引文件（enhanced_tags.json、tag_to_items_enhanced.json）
- [ ] 在build_features_v2.py中成功添加4+个类别相关性特征
- [ ] 重新训练的LightGBM模型中类别特征重要性排名进入Top 10
- [ ] 推荐结果中同大类item比例从当前<50%提升到>80%
- [ ] A/B测试显示点击率或转化率有提升（或至少不下降）

## Context Manifest

### How Tags Currently Work in the Recommendation System

#### Tags在系统中的完整生命周期

**1. 数据源与存储 (Data Source & Storage)**

Tags来自用户手动输入,存储在数据库dataset表的`tag`字段中。当数据流经ETL pipeline时:

- **原始数据位置**: `/home/ubuntu/recommend/data/cleaned/dataset_features.parquet`
  - 字段结构: `dataset_id`, `dataset_name`, `description`, `tag`, `price`, `create_company_name`, `cover_id`, `create_time`
  - Tag格式: 分号分隔的字符串 (例如: `"数字经济;数据要素;数据市场发展"`)

- **处理后数据**: `/home/ubuntu/recommend/data/processed/dataset_features_v2.parquet`
  - 通过`pipeline/build_features_v2.py`的`FeatureEngineV2`类处理
  - 增加了75+特征,包括`tag_count`, `has_tags`等tag相关派生特征

**2. Tags在训练阶段的处理流程**

**训练入口**: `pipeline/train_models.py::main()` (Line 1644-1900)

```python
# 数据加载 (Line 1654-1658)
interactions = _load_frame(PROCESSED_DIR / "interactions.parquet")
dataset_features = _load_frame(PROCESSED_DIR / "dataset_features.parquet")
dataset_stats = _load_frame(PROCESSED_DIR / "dataset_stats.parquet")
```

**Tag索引构建** (由`pipeline/recall_engine_v2.py`执行):

a. **Tag倒排索引** (Line 102-139):
```python
def train_tag_inverted_index(dataset_features) -> (tag_to_items, item_to_tags):
    tag_to_items = defaultdict(set)  # 标签 -> 包含该标签的dataset_id集合
    item_to_tags = {}                 # dataset_id -> 标签列表

    for row in dataset_features:
        tags_str = row.get("tag", "")
        # 关键处理逻辑: 分号分隔,转小写,去空格
        tags = [t.strip().lower() for t in str(tags_str).split(";") if t.strip()]

        item_to_tags[dataset_id] = tags
        for tag in tags:
            tag_to_items[tag].add(dataset_id)
```

**输出文件**:
- `/home/ubuntu/recommend/models/tag_to_items.json`: `{"金融": [20, 35, ...], "医疗健康": [...]}`
- `/home/ubuntu/recommend/models/item_to_tags.json`: `{"20": ["数字经济", "数据要素"], ...}`

b. **Category索引** (Line 141-170): 基于`create_company_name`构建
c. **Price Bucket索引** (Line 172-212): 基于价格区间构建

**3. Tags在特征工程中的应用**

**在`pipeline/build_features_v2.py`中**:

a. **Dataset-level特征** (Line 218-377):
```python
def build_dataset_features_v2():
    # Line 290-298: Tag统计特征
    features["tag_count"] = features["tag"].apply(
        lambda x: len([t for t in str(x).split(";") if t.strip()]) if pd.notna(x) else 0
    )
    features["has_tags"] = (features["tag_count"] > 0).astype(int)
```

b. **User-level特征** (Line 37-135):
```python
def _compute_user_tag_preferences(interactions, dataset_features):
    # Line 166-202: 提取用户top3偏好标签
    # 输出: tag_diversity, top_tag_1, top_tag_1_count, top_tag_2, top_tag_3
```

**4. Tags在Ranking Model训练中的角色**

**特征准备** (`pipeline/train_models.py::_prepare_ranking_dataset`, Line 928-1114):

```python
def _prepare_ranking_dataset(dataset_features, dataset_stats, labels, slot_metrics):
    # Line 944-946: Tag处理
    base["tag"] = base.get("tag", "").fillna("").astype(str)
    base["tag_count"] = base["tag"].apply(
        lambda text: float(len([t for t in text.split(';') if t.strip()]))
    )

    # Line 996-998: 内容丰富度特征
    merged["has_tags"] = (merged["tag_count"] > 0).astype(float)
    merged["content_richness"] = merged["description_length"] * merged["tag_count"]

    # Line 1000-1005: 最终特征列表
    base_columns = [
        "price_log", "description_length", "tag_count", "weight_log",
        "interaction_count", "popularity_rank", "popularity_percentile",
        "price_bucket", "days_since_last_interaction", "interaction_density",
        "has_description", "has_tags", "content_richness"
    ]
```

**关键发现**: 当前ranking model**仅使用tag_count统计特征**,没有利用tag内容语义!

**LightGBM Ranker训练** (Line 1302-1443):
```python
def _train_lightgbm_ranker(data, target, group_sizes, feature_info):
    # Line 1357-1366: 核心参数
    base_params = {
        "objective": "lambdarank",  # Learning-to-Rank目标
        "metric": "ndcg",            # NDCG评估指标
        "n_estimators": 300,
        "learning_rate": 0.05,
        "num_leaves": 63,
    }

    # Line 1384-1394: 训练配置
    fit_kwargs = {
        "X": X_train,
        "y": y_train,
        "group": group_train,  # 关键: 按request_id分组
        "categorical_feature": feature_info["categorical"],
        "eval_at": [5, 10],    # 评估top-5和top-10
    }
```

**模型输出** (Line 1816-1824):
- `/home/ubuntu/recommend/models/rank_model.pkl`: Pipeline对象
- 包含StandardScaler + LGBMRanker
- Feature importance保存在model metadata中

**5. Tags在推荐服务中的使用**

**服务启动时加载** (`app/main.py::_load_recall_assets`, Line 713-743):

```python
def _load_recall_assets(base_dir):
    recall_assets = {}

    # Line 723-736: 加载tag索引
    for name in ["tag_to_items", "item_to_tags", "category_index", "price_bucket_index"]:
        data = _load_json_file(base_dir / f"{name}.json")
        if data:
            # tag_to_items: 转为set加速查找
            if name in {"tag_to_items", "category_index", "price_bucket_index"}:
                normalized = {key: {int(v) for v in value} for key, value in data.items()}
                recall_assets[name] = normalized
            else:
                recall_assets[name] = data  # item_to_tags保持list
```

**内存数据结构**:
```python
state.recall_indices = {
    "tag_to_items": {"金融": {20, 35, ...}, ...},      # dict[str, set[int]]
    "item_to_tags": {20: ["数字经济", "数据要素"], ...}, # dict[int, list[str]]
    "category_index": {...},
    "price_bucket_index": {...},
}

state.dataset_tags = {20: ["数字经济", "数据要素"], ...}  # dict[int, list[str]]
# 从dataset metadata解析,用于实时计算
```

**Tags在召回阶段的使用** (`app/main.py::_augment_with_multi_channel`, Line 1474-1570):

```python
def _augment_with_multi_channel(state, target_id, scores, reasons, limit, user_id):
    # Tag召回 (Line 1498-1520)
    item_to_tags = recall.get("item_to_tags", {})
    tag_to_items = recall.get("tag_to_items", {})

    # 获取target item的tags
    target_tags = item_to_tags.get(str(target_id)) or state.dataset_tags.get(target_id, [])

    if target_tags and tag_to_items:
        candidate_scores = {}
        target_set = set(tag.lower() for tag in target_tags if tag)  # 转小写

        # 遍历target的每个tag,找到所有包含该tag的候选item
        for tag in target_set:
            for candidate in tag_to_items.get(tag, set()):
                if candidate == target_id:
                    continue
                # 计算Jaccard相似度: 标签交集大小
                candidate_tags = set(tag.lower() for tag in state.dataset_tags.get(candidate, []))
                overlap = len(target_set & candidate_tags)
                if overlap:
                    candidate_scores[candidate] += overlap

        # 归一化到[0,1]并加权
        normalized_scores = _normalize_channel_scores(candidate_scores)
        for dataset_id, norm_score in sorted(...):
            scores[dataset_id] += norm_score * 0.4  # Tag召回权重0.4
            reasons[dataset_id] = "tag"
```

**Tags在个性化调整中的使用** (`app/main.py::_apply_personalization`, Line 1855-1950):

```python
def _apply_personalization(user_id, scores, reasons, state, behavior):
    # Line 1899-1934: 基于用户tag偏好调整分数
    user_tag_preferences = _get_user_tag_preferences(state, user_id)
    # user_tag_preferences: {"金融": 0.6, "医疗": 0.3, ...}  # 加权归一化

    if user_tag_preferences:
        for dataset_id in scores.keys():
            item_tags = state.dataset_tags.get(dataset_id, [])
            if item_tags:
                # 计算用户偏好与item tags的匹配度
                tag_match_score = sum(
                    user_tag_preferences.get(tag.lower(), 0.0) for tag in item_tags
                )
                # Boost分数
                if tag_match_score > 0:
                    scores[dataset_id] *= (1.0 + 0.3 * tag_match_score)
                    reasons[dataset_id] += "+tag_pref"
```

**Tags在Ranking阶段的使用** (`app/main.py::_build_static_ranking_features`, Line 1951-2064):

```python
def _build_static_ranking_features(dataset_ids, raw_features, dataset_stats):
    # Line 1970-1979: Tag特征
    selected["tag"] = selected.get("tag", "").fillna("").astype(str)
    selected["tag_count"] = selected.get("tag", "").apply(
        lambda text: float(len([t for t in str(text).split(';') if t.strip()]))
    )

    # Line 2016-2018: 派生特征
    features["has_tags"] = (features["tag_count"] > 0).astype(float)
    features["content_richness"] = features["description_length"] * features["tag_count"]

    # 这些特征输入到LightGBM Ranker进行打分
```

**Ranker预测** (`app/main.py::_apply_ranking`, Line 2311-2400):
```python
def _apply_ranking(scores, reasons, rank_model, raw_features, ...):
    # Line 2350-2377: 使用LightGBM Ranker打分
    features = _compute_ranking_features(...)  # 包含tag_count, has_tags等
    probabilities = _predict_rank_scores(rank_model, features)

    # 将ranker分数叠加到召回分数
    for dataset_id, prob in probabilities.items():
        scores[dataset_id] += prob  # Ranker输出范围(-∞, +∞)
        reasons[dataset_id] += "+rank"
```

### Current Tag Quality Issues (用户确认的问题)

**问题1: Tag缺失严重**
- 用户手动打标,覆盖率低
- 许多高质量dataset没有tags → tag召回失效

**问题2: Tag不准确**
- 用户自由输入,粒度不一致
- 例如同时存在"金融"和"互联网金融"
- 没有标准化的分类体系

**问题3: 缺乏类别相关性特征**
- Ranker只使用`tag_count`统计特征
- 没有利用target item与candidate item之间的tag overlap
- 无法区分"同类别高相关"vs"跨类别低相关"

### How the Zero-Shot Classification Model Will Be Used

**方案C-1**: 使用Erlangshen-Roberta-110M-NLI进行零样本分类

**模型特点**:
- 轻量级: 110M参数,CPU可运行
- 中文优化: 针对中文NLI任务fine-tuned
- 零样本: 无需训练数据,直接分类
- HuggingFace模型: `IDEA-CCNL/Erlangshen-Roberta-110M-NLI`

**输入输出**:
```python
# 输入
text = dataset["description"]  # item描述文本
labels = ["金融", "医疗健康", "政府政务", "交通运输", ...]  # 11个候选大类

# 输出
scores = model(text, labels)  # [0.85, 0.12, 0.03, ...]
top_categories = ["金融", "科技"]  # 取top-2或top-3
```

**处理流程** (新增脚本: `pipeline/enhance_tags.py`):

```python
def enhance_tags_with_zero_shot():
    # 1. 加载原始数据
    dataset_features = pd.read_parquet("data/cleaned/dataset_features.parquet")

    # 2. 加载零样本分类模型
    from transformers import pipeline
    classifier = pipeline(
        "zero-shot-classification",
        model="IDEA-CCNL/Erlangshen-Roberta-110M-NLI",
        device=-1  # CPU
    )

    # 3. 批量预测
    CATEGORIES = ["金融", "医疗健康", "政府政务", ...]
    enhanced_tags = []

    for idx, row in dataset_features.iterrows():
        description = row["description"]
        original_tags = row["tag"]

        # 零样本分类
        result = classifier(description, CATEGORIES, multi_label=True)

        # 取top-2或top-3作为增强tags
        new_tags = [label for label, score in zip(result["labels"], result["scores"])
                    if score > 0.5][:3]

        # 合并原始tags和增强tags
        merged_tags = list(set(original_tags.split(";") + new_tags))
        enhanced_tags.append(";".join(merged_tags))

    # 4. 保存增强后的数据
    dataset_features["tag_enhanced"] = enhanced_tags
    dataset_features.to_parquet("data/processed/dataset_features_enhanced.parquet")

    # 5. 重新构建tag索引
    # (调用recall_engine_v2.train_tag_inverted_index)
```

**性能预估**:
- 处理速度: ~50-100 items/second (CPU)
- 10,000 items: 20-30分钟
- 可使用batch processing加速

### Where to Add Category Relevance Features for Ranking Model

**目标**: 在LightGBM Ranker中添加类别相关性特征,让模型学习"同大类优先推荐"

**修改位置1**: `pipeline/build_features_v2.py::build_cross_features()` (Line 379-447)

这个函数构建user-item交叉特征,但**当前仅用于演示**,未在实际训练中调用。我们需要在ranking dataset准备阶段添加类别特征。

**修改位置2** (主要): `pipeline/train_models.py::_prepare_ranking_dataset()` (Line 928-1114)

当前特征列表 (Line 1000-1005):
```python
base_columns = [
    "price_log", "description_length", "tag_count", "weight_log", "interaction_count",
    "popularity_rank", "popularity_percentile", "price_bucket",
    "days_since_last_interaction", "interaction_density",
    "has_description", "has_tags", "content_richness"
]
```

**新增特征** (在Line 999后添加):
```python
# === Category Relevance Features ===
# 需要访问request context中的target_dataset_id
# 但_prepare_ranking_dataset是离线训练,没有target context

# 解决方案: 使用training samples中的position=1的item作为"pseudo target"
if not samples.empty and "request_id" in samples.columns:
    # 每个request的第一个item视为target
    target_items = samples[samples["position"] == 1][["request_id", "dataset_id"]]
    target_items = target_items.rename(columns={"dataset_id": "target_dataset_id"})

    merged = merged.merge(target_items, on="request_id", how="left")

    # 获取target和candidate的tags
    target_tags_map = {row["target_dataset_id"]: dataset_features[dataset_features["dataset_id"] == row["target_dataset_id"]]["tag"].values[0]
                       for _, row in target_items.iterrows()}

    # 计算类别相关性特征
    merged["tag_overlap_count"] = merged.apply(
        lambda row: _count_tag_overlap(
            target_tags_map.get(row["target_dataset_id"], ""),
            row["tag"]
        ), axis=1
    )

    merged["tag_jaccard_similarity"] = merged.apply(
        lambda row: _jaccard_similarity(
            _parse_tags(target_tags_map.get(row["target_dataset_id"], "")),
            _parse_tags(row["tag"])
        ), axis=1
    )

    merged["same_top_category"] = merged.apply(
        lambda row: _has_same_top_category(
            target_tags_map.get(row["target_dataset_id"], ""),
            row["tag"]
        ), axis=1
    ).astype(float)

    # 添加到特征列表
    base_columns.extend([
        "tag_overlap_count",
        "tag_jaccard_similarity",
        "same_top_category"
    ])
```

**辅助函数** (在文件开头添加):
```python
def _parse_tags(tag_str: str) -> List[str]:
    """解析tag字符串为list"""
    if not tag_str:
        return []
    return [t.strip().lower() for t in str(tag_str).split(";") if t.strip()]

def _count_tag_overlap(tags1_str: str, tags2_str: str) -> float:
    """计算tag重叠数量"""
    tags1 = set(_parse_tags(tags1_str))
    tags2 = set(_parse_tags(tags2_str))
    return float(len(tags1 & tags2))

def _jaccard_similarity(tags1: List[str], tags2: List[str]) -> float:
    """计算Jaccard相似度 (已存在于app/main.py,需复制)"""
    if not tags1 or not tags2:
        return 0.0
    set1 = set(t.lower() for t in tags1)
    set2 = set(t.lower() for t in tags2)
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return float(intersection) / float(union) if union > 0 else 0.0

def _has_same_top_category(tags1_str: str, tags2_str: str) -> int:
    """检查是否有相同的top category (候选大类中的任一个)"""
    tags1 = set(_parse_tags(tags1_str))
    tags2 = set(_parse_tags(tags2_str))

    # 定义大类列表
    TOP_CATEGORIES = {"金融", "医疗健康", "政府政务", "交通运输",
                      "教育培训", "能源环保", "农业", "科技",
                      "商业零售", "文化娱乐", "社会民生"}

    # 找到两个item的top categories
    cat1 = tags1 & TOP_CATEGORIES
    cat2 = tags2 & TOP_CATEGORIES

    # 是否有交集
    return 1 if len(cat1 & cat2) > 0 else 0
```

**修改位置3**: `app/main.py::_build_static_ranking_features()` (Line 1951-2064)

推理时需要动态计算类别相关性特征(基于target item):

```python
def _build_static_ranking_features(
    dataset_ids, raw_features, dataset_stats, slot_metrics_aggregated,
    feature_overrides=None, stats_overrides=None,
    target_dataset_id=None,  # 新增参数
):
    # ... 现有特征构建 ...

    # Line 2018后添加:
    # === Category Relevance Features (需要target context) ===
    if target_dataset_id is not None:
        target_tags = raw_features[raw_features["dataset_id"] == target_dataset_id]["tag"].values
        target_tags_str = target_tags[0] if len(target_tags) > 0 else ""

        features["tag_overlap_count"] = selected["tag"].apply(
            lambda x: _count_tag_overlap(target_tags_str, x)
        )
        features["tag_jaccard_similarity"] = selected["tag"].apply(
            lambda x: _jaccard_similarity(_parse_tags(target_tags_str), _parse_tags(x))
        )
        features["same_top_category"] = selected["tag"].apply(
            lambda x: float(_has_same_top_category(target_tags_str, x))
        )
    else:
        # 无target context时填充默认值
        features["tag_overlap_count"] = 0.0
        features["tag_jaccard_similarity"] = 0.0
        features["same_top_category"] = 0.0

    return features
```

**修改位置4**: `app/main.py::_compute_ranking_features()` (Line 2067-2200)

调用`_build_static_ranking_features`时传入target_dataset_id:

```python
def _compute_ranking_features(
    dataset_ids, raw_features, dataset_stats, ...,
    target_dataset_id=None,  # 新增参数
):
    # Line 2099处修改
    filled = _build_static_ranking_features(
        missing_ids, indexed_raw, indexed_stats, indexed_slot,
        target_dataset_id=target_dataset_id  # 传入target
    )
```

**修改位置5**: `app/main.py::_apply_ranking()` (Line 2311-2400)

推荐API调用ranking时需要传入target:

```python
def _apply_ranking(scores, reasons, rank_model, ..., target_dataset_id=None):
    features = _compute_ranking_features(
        dataset_ids=list(scores.keys()),
        raw_features=raw_features,
        ...,
        target_dataset_id=target_dataset_id  # 新增
    )
```

### Retraining Workflow After Enhancement

**完整流程**:

```bash
# Step 1: 增强tags (新脚本)
python -m pipeline.enhance_tags
# 输出: data/processed/dataset_features_enhanced.parquet
#       models/tag_to_items_enhanced.json
#       models/item_to_tags_enhanced.json

# Step 2: 重新构建特征 (使用增强后的tags)
python -m pipeline.build_features_v2
# 输出: data/processed/dataset_features_v2.parquet (包含增强tags)
#       data/processed/user_features_v2.parquet

# Step 3: 重新训练模型 (包含新的类别相关性特征)
python -m pipeline.train_models
# 输出: models/rank_model.pkl (包含类别特征)
#       models/behavior_sim.pkl
#       models/content_sim.pkl
#       models/top_items.json

# Step 4: 重启推荐服务加载新模型
docker-compose restart recommendation-api
```

### Data Files & Locations Reference

**原始数据**:
- `/home/ubuntu/recommend/data/cleaned/dataset_features.parquet` - 清洗后的dataset特征
- `/home/ubuntu/recommend/data/cleaned/interactions.parquet` - 用户交互记录
- `/home/ubuntu/recommend/data/cleaned/user_profile.parquet` - 用户画像

**处理后特征**:
- `/home/ubuntu/recommend/data/processed/dataset_features_v2.parquet` - 75+特征
- `/home/ubuntu/recommend/data/processed/user_features_v2.parquet` - 30+用户特征
- `/home/ubuntu/recommend/data/processed/dataset_stats.parquet` - 统计特征
- `/home/ubuntu/recommend/data/processed/ranking_training_samples.parquet` - Ranker训练样本
- `/home/ubuntu/recommend/data/processed/ranking_labels_by_dataset.parquet` - 标注数据

**模型文件**:
- `/home/ubuntu/recommend/models/rank_model.pkl` - LightGBM Ranker (Pipeline对象)
- `/home/ubuntu/recommend/models/tag_to_items.json` - Tag倒排索引
- `/home/ubuntu/recommend/models/item_to_tags.json` - Item->Tags映射
- `/home/ubuntu/recommend/models/category_index.json` - 公司分类索引
- `/home/ubuntu/recommend/models/price_bucket_index.json` - 价格分桶索引
- `/home/ubuntu/recommend/models/top_items.json` - Popular召回榜单
- `/home/ubuntu/recommend/models/item_recall_vector.json` - Vector召回索引
- `/home/ubuntu/recommend/models/model_registry.json` - 模型版本注册表

**Feature Store**:
- `/home/ubuntu/recommend/data/processed/feature_store.db` - SQLite特征库
  - Tables: `dataset_features_v2`, `user_features_v2`, `interactions_v2`, `dataset_stats_v2`

### Key Code Patterns & Conventions

**Tag解析标准**:
```python
# 所有tag处理统一使用此pattern
tags = [t.strip().lower() for t in str(tags_str).split(";") if t.strip()]
```

**特征命名规范**:
- Dataset特征: `price_log`, `tag_count`, `has_tags`, `content_richness`
- User特征: `user_` prefix (例如: `user_purchase_count`, `user_top_tag`)
- Cross特征: 反映user-item交互 (例如: `price_match_score`, `tag_overlap_count`)

**内存优化Pattern**:
```python
# 使用optimize_dataframe_memory减少内存占用
from pipeline.memory_optimizer import optimize_dataframe_memory
df = optimize_dataframe_memory(df)
```

**Logging Pattern**:
```python
LOGGER.info("Built tag index: %d unique tags, %d items", len(tag_to_items), len(item_to_tags))
```

### Critical Gotchas & Edge Cases

**问题1: Training vs Inference特征不一致**
- Training时使用request中position=1作为pseudo target计算类别特征
- Inference时使用实际的target_dataset_id
- 解决: 确保特征计算逻辑完全一致

**问题2: Tag索引需要同步更新**
- 增强tags后,必须重新运行`recall_engine_v2.train_tag_inverted_index()`
- 否则tag召回会使用旧索引

**问题3: Description质量差异**
- 部分dataset的description是HTML富文本,需要清理
- 零样本分类前需要去除HTML标签: `from bs4 import BeautifulSoup`

**问题4: 特征缺失处理**
- 增强tags后,原始tag为空的item仍需保留
- 类别相关性特征在无target时填充0.0

**问题5: HuggingFace镜像配置**
- 使用`HF_ENDPOINT=https://hf-mirror.com`环境变量
- 已在`pipeline/train_models.py` Line 30-37配置

### Success Metrics & Validation

**验证点1: Tags覆盖率**
```python
# 增强前后对比
coverage_before = (dataset_features["tag"].notna() & (dataset_features["tag"] != "")).mean()
coverage_after = (dataset_features["tag_enhanced"].notna() & (dataset_features["tag_enhanced"] != "")).mean()
# 目标: >95%
```

**验证点2: 特征重要性**
```python
# 训练后检查feature importance
rank_model = pickle.load(open("models/rank_model.pkl", "rb"))
importances = rank_model["feature_importances"]
# 目标: tag_overlap_count, tag_jaccard_similarity进入Top 10
```

**验证点3: 同大类推荐比例**
```python
# 推荐结果中同大类比例
def evaluate_category_coherence(recommendations, target_item):
    target_cats = set(get_top_categories(target_item))
    same_cat_count = sum(
        1 for item in recommendations
        if len(set(get_top_categories(item)) & target_cats) > 0
    )
    return same_cat_count / len(recommendations)
# 目标: 从<50%提升到>80%
```

**验证点4: Ranking模型性能**
```python
# NDCG@10指标
# 期望: 类别特征添加后NDCG@10不下降(或有提升)
```

## User Notes

### 技术选型
- **方案C-1**: CPU + 轻量级模型（Erlangshen-Roberta-110M-NLI）
- **镜像站**: 使用 hf-mirror.com 加速模型下载
- **处理方式**: 离线批量增强（预计20-30分钟处理1万个item）

### 候选大类定义
```python
CATEGORIES = [
    "金融", "医疗健康", "政府政务", "交通运输",
    "教育培训", "能源环保", "农业", "科技",
    "商业零售", "文化娱乐", "社会民生"
]
```

### Description质量
用户确认description质量较高，适合用于NLP提取

## Work Log
- [2025-12-30] 任务创建，完成技术方案讨论和选型
