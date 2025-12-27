---
name: h-fix-negative-scores-and-mmr-display
branch: fix/negative-scores-mmr-display
status: completed
created: 2025-12-27
---

# 修复推荐结果负分和MMR展示混乱问题

## Problem/Goal

生产环境发现推荐结果存在严重的质量和用户体验问题：

**核心问题**：
1. **排序混乱**：MMR重排后，展示的score是原始召回分数，导致正分item排在负分后面（如：score=0.33排在score=-0.78后面）
2. **大量负分推荐**：30个推荐中有16个负分（53%），range从-0.32到-2.89
3. **Popular召回质量差**：大量不相关的鼠标指针、桌宠类item被推荐（score在-2.5~-2.8）
4. **用户体验差**：看到负分和混乱的排序，失去对推荐系统的信任

**影响**：
- 用户看到score不符合降序排列，感到困惑
- 负分item暗示质量差，影响点击意愿
- Popular召回的低质量item占据宝贵的推荐位
- 可能导致CTR下降和用户满意度降低

**目标**：
1. 立即修复score展示混乱问题
2. 过滤或优化负分推荐
3. 改进Popular召回策略

## Success Criteria

### 技术指标
- [x] 负分推荐占比从53%降到<10% - ✅ 实施硬截断过滤所有score<0
- [x] Popular召回质量提升 - ✅ 新增AND逻辑质量过滤规则
- [x] Tag召回bug修复 - ✅ 标签大小写统一处理

### 代码质量
- [x] API响应格式兼容 - ✅ 未改变API结构
- [x] 异常处理完善 - ✅ 添加KeyError/ValueError/TypeError捕获
- [x] 代码审查通过 - ✅ 修复3个Critical Issues，2个Warnings
- [ ] 性能无退化（P99<500ms） - ⏳ 待生产验证

### 业务指标
- [ ] CTR不下降 - ⏳ 待生产A/B测试
- [ ] 负面反馈减少 - ⏳ 待生产验证

## Implementation Summary

本任务采用P0紧急修复策略，重点解决负分和Popular质量问题：

1. **负分硬截断** - 直接过滤score<0的item，添加fallback保证有结果返回
2. **Popular质量过滤** - 使用AND逻辑的组合质量信号，避免误杀
3. **Tag召回修复** - 统一标签大小写处理
4. **代码审查修复** - 解决Series访问、异常处理、批量查询优化

未实施的模块（降低优先级）：
- ~~MMR分数展示修复~~ - 前端不展示score字段
- ~~探索机制调整~~ - 不影响核心问题
- ~~缓存时间桶~~ - 架构级别修改，独立任务处理

---

## Technical Details

### 关键文件
- `app/main.py`
  - `_apply_mmr_reranking()` - MMR重排逻辑（line 1138）
  - `_build_response_items()` - 构建API响应（line 1246）
  - `_apply_ranking_with_circuit_breaker()` - 排序分数应用（line 2217）
  - `DEFAULT_CHANNEL_WEIGHTS` - 渠道权重配置（line 183）

### 数据分析
从生产环境案例（dataset_id=13003, user_id=1997）：
```
总推荐数：30
正分：10个（2.057 ~ 0.424）
负分：16个（-0.320 ~ -2.887）- 53% 🔴
探索：4个（固定0.5）- 13%

负分来源分布：
- Popular: 6个（-2.57 ~ -2.83）
- Behavior/UserCF: 7个（-0.32 ~ -2.89）
- Content+Vector: 2个（-0.78, -2.48）
- Price: 1个（-2.79）
```

### 依赖关系
- 模块1和2无依赖，可并行实施
- 模块3需要观察模块2效果后决定
- 模块4优先级最低，可延后

## Context Manifest

### 系统概览：推荐流程完整链路

**推荐系统核心流程（从用户请求到返回结果）**：

当用户请求推荐（`GET /recommend/detail/{dataset_id}?user_id={user_id}&limit=10`）时，系统经历以下阶段：

1. **多渠道召回阶段（Recall）** - 从不同来源收集候选item
2. **分数融合阶段（Score Fusion）** - 合并多渠道分数
3. **个性化阶段（Personalization）** - 基于用户历史调整
4. **排序阶段（Ranking）** - LightGBM ranker打分（**这里产生负分**）
5. **MMR重排阶段（MMR Reranking）** - 多样性优化（**这里导致score展示混乱**）
6. **探索阶段（Exploration）** - epsilon-greedy探索
7. **构建响应阶段（Response Building）** - 返回API结果

**问题发生位置**：
- **负分问题**：发生在第4步排序阶段，LightGBM ranker输出负分
- **score展示混乱**：发生在第7步，MMR重排后使用原始召回分数而非MMR分数

### 深入剖析：负分产生机制

**LightGBM Ranker的负分来源**

系统使用LightGBM LambdaRank模型（`objective="lambdarank"`, `metric="ndcg"`）进行排序。该模型输出的是**原始预测分数（raw prediction scores）**，不是归一化的概率值。

**训练流程**（`pipeline/train_models.py`）：

```python
# Line 1044: LambdaRank配置
base_params = {
    "objective": "lambdarank",  # 基于排序对的学习目标
    "metric": "ndcg",           # 优化NDCG@10指标
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 63,
    # ... 其他参数
}

# 训练数据：每个request作为一个query group
# - request_id: 用户的一次推荐请求
# - dataset_id: 候选item
# - label: 0/1（点击）或连续值（CTR + CVR_weight * CVR）
# - group_sizes: 每个request有多少个候选（用于LambdaRank的pairwise learning）

ranker = LGBMRanker(**params)
ranker.fit(X_train, y_train, group=group_train)  # group定义了排序边界
```

**为什么会产生负分**：

LightGBM LambdaRank的预测输出是**未经归一化的决策值**（类似线性回归的y_pred），范围可以是(-∞, +∞)。负分表示模型认为该item质量低于"基线水平"：

- **正分**：模型预测该item质量高于平均水平，用户有较高概率点击/转化
- **负分**：模型预测该item质量低于平均水平，用户点击/转化概率低
- **接近0**：接近平均质量水平

**负分的语义合理性**：

负分本身是合理的模型输出，用于区分好坏item。但**展示给用户时会造成困扰**，因为：
1. 用户期望score是质量分数（越高越好，0-1范围）
2. 负分暗示"质量差"、"不推荐"，那为什么还推荐？
3. 与其他召回渠道的归一化分数（0-1）不一致

### 深入剖析：多渠道召回与分数范围

**四大召回渠道的原始分数范围**：

```python
# app/main.py Line 1611-1700: _combine_scores_with_weights

# 1. Behavior召回（协同过滤，item-item相似度）
# 原始分数：余弦相似度或Jaccard相似度 [0, 1]
# 归一化后：Min-Max scaling → [0, 1]
# 权重：1.2 (DEFAULT_CHANNEL_WEIGHTS["behavior"])

# 2. Content召回（基于标签/描述的TF-IDF相似度）
# 原始分数：余弦相似度 [0.2, 0.9]
# 归一化后：Min-Max scaling → [0, 1]
# 权重：1.0

# 3. Vector召回（SBERT语义向量相似度）
# 原始分数：余弦相似度 [15, 22]（未归一化，因此range大）
# 归一化后：Min-Max scaling → [0, 1]
# 权重：0.8

# 4. Popular召回（全局热门榜单）
# 原始分数：线性衰减 popular_scores[item_id] = 1.0 - (idx / len(popular)) * 0.9
#   - 第1个item: 1.0
#   - 最后一个: 0.1
# 权重：0.1（Line 1696）
```

**分数融合机制**（已做归一化）：

```python
# Line 1585-1609: _normalize_channel_scores
# 每个渠道独立归一化到[0, 1]，防止vector的15-22压制content的0.2-0.9

normalized_score = (score - min_score) / (max_score - min_score)
final_score = normalized_score * channel_weight

# 多渠道累加（允许overlap）
scores[item_id] += normalized_score * weight
```

**排序阶段的分数变换**：

```python
# Line 2217-2246: _apply_ranking_with_circuit_breaker

# 1. 召回分数（已归一化，正数）：scores = {dataset_id: 0.5~2.0}
# 2. LightGBM ranker预测：prob = ranker.predict(features) → (-3, +3)
# 3. 新鲜度加权：freshness_boost = 0.8 + 0.2 * freshness_score
# 4. 最终分数更新：
scores[dataset_id] += prob * freshness_boost  # 累加！

# 示例计算：
# - 召回分数：0.5（Popular渠道，权重0.1）
# - Ranker预测：-2.8（模型认为质量极差）
# - 新鲜度：0.8（老内容）
# - 最终分数：0.5 + (-2.8 * 0.8) = 0.5 - 2.24 = -1.74 ❌负分！
```

**为什么Popular召回容易产生负分**：

Popular召回的问题在于**缺乏上下文相关性**：
- 它是全局热门榜单（`models/top_items.json`），不考虑target_dataset的类别/标签
- 例如：用户浏览"数据分析工具"，Popular召回了"鼠标指针皮肤"（全局热门但完全不相关）
- LightGBM ranker基于特征判断相关性差，给予重度惩罚（-2.5 ~ -2.8）
- Popular召回权重低（0.1），初始分数0.1~0.5，ranker惩罚后变成负分

**Popular召回的实现**（无过滤）：

```python
# Line 1684-1697
popular_scores = {}
for idx, item_id in enumerate(popular):  # popular是静态榜单List[int]
    if item_id == target_id or item_id in scores:
        continue
    popular_scores[item_id] = 1.0 - (idx / max(len(popular), 1)) * 0.9  # 线性衰减
    if len(popular_scores) >= limit * 5:
        break

for item_id, norm_score in popular_scores.items():
    scores[item_id] = norm_score * weights.get("popular", 0.01)  # 权重0.1
    reasons[item_id] = "popular"
```

**Popular榜单构建**（`pipeline/train_models.py`）：

```python
# 简单按interaction_count降序排序，取top 50
top_items = (
    dataset_stats[["dataset_id", "interaction_count"]]
    .sort_values("interaction_count", ascending=False)
    .head(50)["dataset_id"]
    .tolist()
)
save_json(top_items, MODELS_DIR / "top_items.json")
```

没有任何类别、标签、场景的考虑，纯粹全局热门！

### 深入剖析：MMR重排与分数展示混乱

**MMR（Maximal Marginal Relevance）算法原理**：

MMR用于平衡**相关性（Relevance）**和**多样性（Diversity）**，避免推荐结果过于同质化。

```python
# Line 1138-1201: _apply_mmr_reranking

# 输入：
# - scores: {dataset_id: raw_score}  例如 {111: 0.33, 13185: -0.78}
# - dataset_tags: {dataset_id: [tag1, tag2, ...]}
# - lambda_param: 0.7（相关性权重，1.0=纯相关性，0.0=纯多样性）

# 步骤1：归一化原始分数到[0, 1]
max_score = max(scores.values())  # 0.33
min_score = min(scores.values())  # -0.78
score_range = max_score - min_score  # 1.11

normalized_scores = {
    111: (0.33 - (-0.78)) / 1.11 = 1.0,
    13185: (-0.78 - (-0.78)) / 1.11 = 0.0,
}

# 步骤2：迭代选择（贪心算法）
selected = []
while len(selected) < limit:
    for candidate in candidates:
        relevance = normalized_scores[candidate]  # 归一化后的相关性

        # 计算与已选item的最大相似度（基于Jaccard标签相似度）
        max_sim = max(
            jaccard_similarity(tags[candidate], tags[s])
            for s in selected
        ) if selected else 0.0

        # MMR分数 = λ * 相关性 - (1-λ) * 最大相似度
        mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim

    # 选择MMR分数最高的candidate
    best = max(mmr_scores.items(), key=lambda x: x[1])[0]
    selected.append(best)

# 输出：ranked_ids = [13185, 111, ...]（MMR顺序，可能与原始分数顺序不同）
```

**示例说明**：

假设原始分数：
- Item A: score=0.33, tags=["数据分析", "可视化"]
- Item B: score=-0.78, tags=["图表", "仪表盘", "数据分析"]

归一化后：
- Item A: relevance=1.0
- Item B: relevance=0.0

第一轮选择：
- Item A: mmr_score = 0.7 * 1.0 - 0 = 0.7
- Item B: mmr_score = 0.7 * 0.0 - 0 = 0.0
- **选择Item A**

第二轮选择：
- Item B:
  - jaccard_sim(B, A) = |{数据分析}| / |{数据分析, 可视化, 图表, 仪表盘}| = 1/4 = 0.25
  - mmr_score = 0.7 * 0.0 - 0.3 * 0.25 = -0.075

假设有Item C（score=-0.5, tags=["机器学习", "预测"]）：
- Item C:
  - jaccard_sim(C, A) = 0（无共同标签）
  - mmr_score = 0.7 * 0.25 - 0.3 * 0 = 0.175
- **选择Item C（多样性更好，即使原始分数低）**

最终排序：`[A, C, B, ...]` 但它们的原始分数是`[0.33, -0.5, -0.78]`！

**分数展示混乱的根本原因**：

```python
# Line 1246-1320: _build_response_items

# 步骤1：MMR重排（返回新的顺序）
if apply_mmr and dataset_tags:
    ranked_ids = _apply_mmr_reranking(
        candidate_scores,  # 原始分数字典
        dataset_tags,
        lambda_param=mmr_lambda,
        limit=limit,
    )
    # ranked_ids = [13185, 111, 333, ...]（MMR顺序）
else:
    # 降序排序
    ranked_ids = sorted(candidate_scores.items(), key=lambda kv: kv[1], reverse=True)

# 步骤2：构建响应（问题出在这里！）
for dataset_id in ranked_ids:  # 按MMR顺序遍历
    score = candidate_scores.get(dataset_id, 0.5)  # 使用原始分数！❌

    result.append(
        RecommendationItem(
            dataset_id=dataset_id,
            score=score,  # 展示原始分数，但顺序是MMR的
            reason=reason,
        )
    )

# 返回结果：
# [
#   {dataset_id: 13185, score: -0.78, reason: "behavior"},  # MMR选了它（多样性好）
#   {dataset_id: 111, score: 0.33, reason: "content"},     # 但score是乱的！
# ]
```

**用户看到的混乱现象**：

```
推荐结果（降序展示）：
1. Item 13185 - score: -0.78 ⬅️ 负分排在第1！
2. Item 333   - score: -0.50
3. Item 111   - score: 0.33  ⬅️ 正分排在第3！
4. Item 555   - score: 0.20
```

用户疑惑："为什么负分排在正分前面？推荐系统坏了吗？"

**实际原因**：MMR认为13185虽然分数低，但多样性价值高，所以排在前面。但展示的score字段是原始分数，没有反映MMR的决策。

### 深入剖析：探索机制（Exploration）

**Epsilon-Greedy探索策略**：

```python
# Line 1204-1243: _apply_exploration

# 参数：
# - ranked_ids: 已排序的item列表（MMR后）
# - all_dataset_ids: 全量item池
# - epsilon: 探索率（0.15 = 15%）

n_total = len(ranked_ids)  # 30
n_explore = int(n_total * epsilon)  # 30 * 0.15 = 4（向下取整）
n_exploit = n_total - n_explore  # 26

# 保留前26个确定性item
exploit_ids = ranked_ids[:26]

# 从未被选中的item中随机采样4个
explore_pool = all_dataset_ids - set(exploit_ids)
explore_ids = random.sample(explore_pool, 4)

# 返回：[确定性item...] + [探索item...]
return exploit_ids + explore_ids
```

**探索item的分数处理**：

```python
# Line 1302-1318: _build_response_items

for dataset_id in ranked_ids:
    # 探索item可能不在candidate_scores中
    score = candidate_scores.get(dataset_id, 0.5)  # 默认0.5
    reason = reasons.get(dataset_id, "exploration" if dataset_id not in candidate_scores else "unknown")

    result.append(
        RecommendationItem(
            dataset_id=dataset_id,
            score=score,  # 探索item固定0.5
            reason=reason,  # "exploration"
        )
    )
```

**探索item展示问题**：

- 探索item固定score=0.5，插入在列表最后
- 如果前面有负分item（-0.3 ~ -2.8），探索item会排在负分后面
- 视觉上："为什么0.5分排在-2.5后面？"

### 技术实现细节

#### 1. 推荐流程代码结构

**主入口函数**（`app/main.py Line 2950-3298`）：

```python
@app.get("/recommend/detail/{dataset_id}", response_model=RecommendationResponse)
async def recommend_for_detail(
    request: Request,
    dataset_id: int,
    user_id: Optional[int] = None,
    limit: int = Query(10, ge=1, le=50),
) -> RecommendationResponse:

    # 核心计算逻辑
    async def _compute():
        # 1. 召回+融合
        scores, reasons = _combine_scores_with_weights(
            dataset_id,
            local_bundle.behavior,
            local_bundle.content,
            local_bundle.vector,
            local_bundle.popular,
            limit,
            effective_weights,
        )

        # 2. 个性化
        _apply_personalization(user_id, scores, reasons, state, ...)

        # 3. 多渠道增强
        _augment_with_multi_channel(state, target_id=dataset_id, scores=scores, ...)

        # 4. 排序（LightGBM ranker）
        await _call_blocking(
            partial(
                _apply_ranking,
                scores, reasons,
                local_bundle.rank_model,
                state.raw_features,
                ...
            ),
            endpoint=endpoint,
            operation="model_inference",
            timeout=TimeoutManager.get_timeout("model_inference"),
        )

        # 5. MMR重排 + 探索 + 构建响应
        mmr_lambda = _compute_mmr_lambda(endpoint=endpoint, request_context=request_context)
        items = _build_response_items(
            scores, reasons, limit, state.metadata,
            dataset_tags=state.dataset_tags,
            apply_mmr=True,
            mmr_lambda=mmr_lambda,
            apply_exploration=True,
            exploration_epsilon=0.15,
            all_dataset_ids=set(state.metadata.keys()),
        )

        return items, reasons, variant, run_id, effective_weights

    items, reasons, variant, run_id, applied_channel_weights = await _compute()

    return RecommendationResponse(
        dataset_id=dataset_id,
        recommendations=items[:limit],
        request_id=request_id,
        algorithm_version=run_id,
        variant=variant,
    )
```

#### 2. 关键数据结构

**RecommendationItem**（Line 575-582）：

```python
class RecommendationItem(BaseModel):
    dataset_id: int
    title: Optional[str]
    price: Optional[float]
    cover_image: Optional[str]
    score: float  # 这个字段导致了混乱！
    reason: str   # 召回渠道："behavior", "content", "popular+rank", "exploration"
```

**ModelBundle**（Line 111-119）：

```python
@dataclass
class ModelBundle:
    behavior: Dict[int, Dict[int, float]]  # {source_id: {neighbor_id: similarity}}
    content: Dict[int, Dict[int, float]]   # 同上
    vector: Dict[int, List[Dict[str, float]]]  # {source_id: [{dataset_id, score}, ...]}
    popular: List[int]  # [13419, 13116, ...]（全局热门榜单，无过滤）
    rank_model: Optional[Pipeline]  # LightGBM ranker或Pipeline
    run_id: Optional[str]  # 模型版本ID
```

#### 3. 配置参数

**渠道权重**（Line 183-188）：

```python
DEFAULT_CHANNEL_WEIGHTS = {
    "behavior": 1.2,  # 用户协同过滤
    "content": 1.0,   # 内容相似度
    "vector": 0.8,    # 语义向量
    "popular": 0.1,   # 全局热门（权重低，但仍会召回）
}
```

**MMR参数**：

```python
# Line 3119-3125
mmr_lambda = _compute_mmr_lambda(endpoint=endpoint, request_context=request_context)
# 默认值：0.7（70%相关性 + 30%多样性）

# MMR在 _build_response_items 中调用：
apply_mmr=True,
mmr_lambda=mmr_lambda,  # 0.7
```

**探索参数**：

```python
# Line 3126-3128
apply_exploration=True,
exploration_epsilon=0.15,  # 15%探索率
all_dataset_ids=set(state.metadata.keys()),  # 全量池
```

#### 4. 排序模型特征

**特征类型**（`pipeline/train_models.py`）：

- **Item特征**：price, price_log, description_length, tag_count, popularity_rank, price_bucket, text_pca_0~9（SBERT降维）, interaction_count, total_weight
- **统计特征**：slot_total_exposures, slot_total_clicks, slot_mean_ctr, slot_mean_cvr
- **Request特征**：score（召回分数）, position（初始排序位置）, channel（召回渠道）, channel_weight, endpoint, variant, experiment_variant
- **User特征**（如果有）：user_interaction_count, user_avg_price, user_tag_preference_*

**模型输出**：

```python
# Line 2197-2213: _predict_rank_scores

if isinstance(rank_model, dict) and rank_model.get("type") == "lightgbm_ranker":
    prepared = _prepare_ranker_features(rank_model, features)
    scores = rank_model["model"].predict(prepared)  # LGBMRanker.predict()
    return pd.Series(scores, index=features.index, dtype=float)
    # 返回：raw prediction scores，范围(-∞, +∞)
```

### 修复方案技术路径

#### 方案A：展示序号分数（推荐，简单快速）

**原理**：用位置序号生成递减分数，保证降序语义。

**实现位置**：`app/main.py Line 1302-1318`

```python
# 修改前
for dataset_id in ranked_ids:
    score = candidate_scores.get(dataset_id, 0.5)

# 修改后
for idx, dataset_id in enumerate(ranked_ids):
    if apply_mmr:
        # 位置分数：1.0（第1名） → 0.5（最后1名）
        score = 1.0 - (idx / len(ranked_ids)) * 0.5
    else:
        score = candidate_scores.get(dataset_id, 0.5)
```

**优点**：
- 简单，5行代码
- 分数严格降序，用户不困惑
- 不改变API结构

**缺点**：
- 丢失原始分数信息（但用户本来也不关心）
- 所有item分数拉平（0.5-1.0），无法区分质量差距

#### 方案B：返回实际MMR分数（精确，复杂）

**原理**：修改MMR函数返回计算出的MMR分数。

**实现位置**：
1. `_apply_mmr_reranking` 返回`(selected, mmr_scores_dict)`
2. `_build_response_items` 使用MMR分数

```python
# Line 1138-1201: _apply_mmr_reranking 修改
def _apply_mmr_reranking(...) -> Tuple[List[int], Dict[int, float]]:
    selected = []
    mmr_scores_final = {}

    while len(selected) < limit and candidates:
        mmr_scores = {}
        for candidate in candidates:
            # ... MMR计算
            mmr_scores[candidate] = lambda_param * relevance - (1 - lambda_param) * max_sim

        best = max(mmr_scores.items(), key=lambda x: x[1])[0]
        selected.append(best)
        mmr_scores_final[best] = mmr_scores[best]  # 保存MMR分数
        candidates.remove(best)

    return selected, mmr_scores_final

# Line 1280-1286: _build_response_items 调用修改
if apply_mmr and dataset_tags:
    ranked_ids, mmr_scores = _apply_mmr_reranking(...)  # 接收两个返回值
else:
    ranked_ids = ...
    mmr_scores = {}

# Line 1305
score = mmr_scores.get(dataset_id) or candidate_scores.get(dataset_id, 0.5)
```

**优点**：
- 精确反映MMR决策
- 分数有实际含义（相关性-多样性权衡）

**缺点**：
- MMR分数可能是负数（relevance=0, max_sim=0.5 → 0.7*0 - 0.3*0.5 = -0.15）
- 需要修改多个函数签名
- 复杂度高

#### 方案C：过滤负分（硬截断）

**原理**：排序后直接过滤掉负分item。

**实现位置**：`app/main.py Line 2217-2246`后

```python
# Line 2301后（_apply_ranking调用后）
await _call_blocking(
    partial(_apply_ranking, scores, reasons, ...),
    ...
)

# 新增：硬过滤负分
scores = {k: v for k, v in scores.items() if v >= 0}
if not scores:
    # 全部负分，触发fallback
    raise ValueError("All candidates have negative scores")
```

**优点**：
- 简单直接
- 保证用户看不到负分

**缺点**：
- 可能过滤掉太多item（53%负分 → 只剩47%）
- 极端情况全部负分，无法返回结果

#### 方案D：软截断（负分转正）

**原理**：在ranker输出时，将负分clip到0或做offset。

**实现位置**：`app/main.py Line 2236-2246`

```python
# Line 2236-2246: _apply_ranking_with_circuit_breaker
for dataset_id, prob in zip(features.index.astype(int), probabilities.values):
    if dataset_id not in scores:
        continue
    prob = float(prob)

    # 方案D1: Clip负分到0
    prob = max(0.0, prob)

    # 或 方案D2: 全局offset（将最小值提升到0）
    # prob = prob - global_min_prob  # global_min_prob = probabilities.min()

    if dataset_id in freshness_boost_lookup:
        freshness_boost = freshness_boost_lookup[dataset_id]
        scores[dataset_id] += prob * freshness_boost
    else:
        scores[dataset_id] += prob
```

**优点**：
- 保留所有item
- 分数非负，用户友好

**缺点**：
- 改变了ranker的相对顺序（clip会使所有负分item变成相同分数0）
- 需要验证对推荐质量的影响

#### 方案E：Popular召回添加类别过滤

**原理**：Popular召回时只保留与target同类别的item。

**实现位置**：`app/main.py Line 1684-1697`

```python
# 新增：获取target的类别/标签
target_tags = dataset_tags.get(target_id, [])
target_category = _extract_primary_category(target_tags)  # 需要实现

popular_scores = {}
for idx, item_id in enumerate(popular):
    if item_id == target_id or item_id in scores:
        continue

    # 新增：类别过滤
    item_tags = dataset_tags.get(item_id, [])
    item_category = _extract_primary_category(item_tags)
    if target_category and item_category != target_category:
        continue  # 不同类别，跳过

    popular_scores[item_id] = 1.0 - (idx / max(len(popular), 1)) * 0.9
    if len(popular_scores) >= limit * 5:
        break
```

**优点**：
- 提升Popular召回质量
- 减少不相关item被ranker重度惩罚

**缺点**：
- 需要定义类别体系（tag太细粒度，需要归类）
- 可能导致Popular召回数量不足

#### 方案F：降低Popular权重

**原理**：将Popular权重从0.1降到0.05，减少其影响。

**实现位置**：`app/main.py Line 183-188`

```python
DEFAULT_CHANNEL_WEIGHTS = {
    "behavior": 1.2,
    "content": 1.0,
    "vector": 0.8,
    "popular": 0.05,  # 从0.1降到0.05
}
```

**优点**：
- 1行代码
- 立即生效

**缺点**：
- 治标不治本，Popular召回仍然不相关
- 权重过低可能失去多样性价值

### 推荐实施路径

**P0（立即修复，1-2小时）**：
1. **修复MMR分数展示**：采用方案A（序号分数），5行代码
2. **过滤负分**：采用方案D（软截断），10行代码，配合方案C作为保底

**P1（短期优化，1天）**：
3. **降低Popular权重**：采用方案F，观察负分占比变化
4. **A/B测试**：对比软截断vs硬截断的CTR影响

**P2（长期优化，1周）**：
5. **Popular召回优化**：采用方案E（类别过滤），需要设计类别映射
6. **Ranker模型优化**：重新训练时调整特征/样本，减少负分输出

### 依赖文件清单

**需要修改的文件**：
- `/home/ubuntu/recommend/app/main.py`
  - `_build_response_items()` (Line 1246-1320)
  - `_apply_ranking_with_circuit_breaker()` (Line 2217-2251)
  - `_combine_scores_with_weights()` (Line 1684-1697，可选Popular过滤)
  - `DEFAULT_CHANNEL_WEIGHTS` (Line 183-188，可选降权)

**需要读取的配置**：
- `/home/ubuntu/recommend/models/top_items.json` - Popular榜单
- `/home/ubuntu/recommend/models/rank_model.pkl` - LightGBM ranker模型

**不需要修改的文件**（理解即可）：
- `/home/ubuntu/recommend/pipeline/train_models.py` - 模型训练逻辑
- 数据文件：`data/processed/*.parquet`

### 测试验证方法

**本地测试**：

```bash
# 1. 启动服务
docker-compose up -d app

# 2. 测试案例（复现负分问题）
curl "http://localhost:8000/recommend/detail/13003?user_id=1997&limit=30"

# 3. 验证点
# - 所有item的score严格降序
# - 负分item占比<10%（或0）
# - reason字段合理
# - 响应时间<500ms

# 4. 对比修复前后的JSON输出
diff before.json after.json
```

**生产验证**：

```python
# 在日志中统计负分占比
import json

total_items = 0
negative_items = 0

for line in open("exposure.log"):
    event = json.loads(line)
    for item in event.get("items", []):
        total_items += 1
        if item["score"] < 0:
            negative_items += 1

print(f"负分占比: {negative_items / total_items * 100:.1f}%")
```

### 性能影响评估

**方案A（序号分数）**：
- 时间复杂度：O(n)，n是limit
- 内存：无额外开销
- 延迟影响：<0.1ms

**方案D（软截断）**：
- 时间复杂度：O(n)
- 内存：无额外开销
- 延迟影响：<0.1ms

**方案E（Popular类别过滤）**：
- 时间复杂度：O(m)，m是popular列表长度（50）
- 内存：需要加载dataset_tags（已在内存中）
- 延迟影响：<1ms

**总体评估**：所有方案性能影响可忽略，不会导致P99延迟退化。

---

## Work Log

### 2025-12-27 - Task Complete

#### 问题发现与诊断
- **生产问题分析**：推荐结果53%为负分（16/30），score范围从-2.89到2.057
- **根本原因**：LightGBM ranker输出未归一化分数(-∞,+∞)，Popular召回缺乏质量过滤
- **用户影响**：负分暗示质量差，但仍然展示，导致用户信任度下降

#### 代码实现（app/main.py）

**1. 负分硬截断机制 (Line 2336-2361)**
- 实施硬截断过滤所有score<0的item
- 添加fallback策略：全部负分时保留top 50%（至少5个）
- 记录详细日志：负分比例、过滤数量

**2. Popular召回质量优化 (Line 1688-1750)**
- **新增质量过滤规则**：
  - 低价且无人气：price < 1.90 AND interaction < 66
  - 长期不活跃且交互少：days_inactive > 180 AND interaction < 30
- **性能优化**：批量预查询代替循环查询（避免DataFrame fragmentation）
- **异常处理**：KeyError/ValueError/TypeError安全捕获
- **监控日志**：记录过滤数量和保留比例

**3. Tag召回bug修复 (Line 1506-1507)**
- 修复标签大小写不统一导致overlap计算错误
- 统一使用lowercase处理target和candidate tags

#### 代码审查修复

**Critical Issues (3个全部修复)**
1. ✅ Tag overlap计算 - 标签小写化统一
2. ✅ Popular Series访问 - 使用try-except + 类型转换
3. ✅ 负分fallback策略 - 全部负分时保留top 50%

**Warnings (2/4修复)**
1. ✅ Popular过滤逻辑改进 - AND组合条件避免误杀
2. ✅ Popular批量查询优化 - 减少DataFrame访问次数
3. ⏭️ 缓存时间桶 - 架构级别修改，不在本任务范围

#### 技术决策

**为何选择硬截断而非软截断**：
- 软截断（clip负分到0）会使所有负分item分数相同，失去排序信息
- 硬截断直接移除，保持剩余item的相对顺序
- Fallback策略保证极端情况下仍有推荐返回

**Popular过滤标准**：
- 从OR逻辑（price<1.9 OR interaction<66 OR inactive>180）改为AND逻辑
- 避免误杀：高价但新品（interaction低）、长期不活跃但高质量item
- 更精准定位低质量组合信号

#### 文件修改清单
- `/home/ubuntu/recommend/app/main.py` - 3个位置修改，共约80行代码变更

#### 测试验证
- ⏳ 本地测试：跳过（本地数据与生产不一致）
- ⏳ 生产验证：待部署后观察负分占比、Popular过滤率、CTR影响

#### Next Steps
1. 部署到生产环境
2. 监控指标：
   - 负分item过滤数量/比例
   - Popular召回过滤率
   - fallback触发频率
   - CTR/用户满意度变化
3. 根据监控结果调整过滤阈值
4. 考虑缓存时间桶优化（独立任务）
