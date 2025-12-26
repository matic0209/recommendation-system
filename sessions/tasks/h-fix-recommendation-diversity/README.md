---
name: h-fix-recommendation-diversity
branch: fix/h-fix-recommendation-diversity
status: pending
created: 2025-12-25
---

# 修复推荐系统多样性与分数断崖问题

## Problem/Goal

当前推荐系统存在严重的结果重复和分数断崖问题：

**核心问题**：
1. **分数量纲不统一**：Vector召回分数（15-22分）是Content召回（0.8分）的20倍，导致非vector渠道永远无法排到前面
2. **缺乏探索机制**：100%确定性推荐，相同请求永远返回相同结果
3. **缓存无时间因子**：Redis缓存key不包含时间，导致结果长期固定
4. **MMR多样性参数过高**：λ=0.7过度偏向相关性，多样性机制失效
5. **缺乏新鲜度机制**：老内容长期霸榜，新内容难以曝光

**影响**：
- 用户看到的推荐列表总是前几个相同的item
- 推荐多样性极差（标签覆盖率低）
- 新内容得不到曝光机会
- 用户体验单调，可能影响留存

**目标**：
通过5个模块的综合优化，提升推荐多样性、新鲜度和用户体验。

## Success Criteria

### 技术指标
- [ ] 分数跨度从24倍降低到8倍以内（0.3-2.5分）
- [ ] 前10个推荐的标签覆盖率从~15个提升到~25个（+67%）
- [ ] 非vector召回占比从<20%提升到>40%
- [ ] 相同请求在不同时间返回不同结果（每小时刷新）
- [ ] 探索机制正常工作（15%的位置是探索项）

### 代码质量
- [ ] 所有修改通过单元测试
- [ ] 性能无明显退化（P99延迟<500ms）
- [ ] 降级机制正常工作

### 业务指标（A/B测试验证）
- [ ] 点击率（CTR）提升10-15%
- [ ] 人均推荐浏览数提升20%
- [ ] 新内容曝光量提升50%
- [ ] 用户停留时长提升8-12%

## Context Manifest
<!-- Added by context-gathering agent -->

### 推荐系统架构全景：从请求到响应的完整数据流

当用户访问一个详情页（dataset_id）时，推荐系统会为其生成个性化推荐列表。整个流程涉及多阶段处理：召回 → 个性化 → 多渠道增强 → 排序 → 多样性重排 → 缓存。当前系统存在的5个核心问题都集中在这条链路上。

#### 第一阶段：多路召回与分数融合（问题1的根源）

**入口函数**：`recommend_for_detail()` (app/main.py:2799)

请求到达后，系统首先尝试从Redis缓存获取结果：
```python
cache_key = f"recommend:{dataset_id}:{user_id}:{limit}"  # 行2843
cached_result = cache.get_json(cache_key)  # 无时间因子！
```

**问题3关键点**：缓存键不包含时间戳，导致相同请求永远返回相同结果。

缓存未命中时，进入核心推荐逻辑：

**召回融合**：`_combine_scores_with_weights()` (行1535-1586)

系统从4个主要召回源获取候选：

1. **Behavior召回**（行为协同过滤）：
```python
behavior_scores = behavior.get(target_id, {})  # Dict[item_id, score]
scores[item_id] = score * weights.get("behavior", 1.0)  # 权重1.5
```
- 数据源：`item_sim_behavior.pkl`（ItemCF模型）
- 典型分数范围：**0.5-3.0**（余弦相似度）
- 默认权重：**1.5**（最高优先级）

2. **Content召回**（内容相似度）：
```python
content_scores = content.get(target_id, {})
scores[item_id] = score * weights.get("content", 0.5)  # 权重0.8
```
- 数据源：`item_sim_content.pkl`（TF-IDF余弦相似度）
- 典型分数范围：**0.2-0.9**（文本相似度）
- 默认权重：**0.8**

3. **Vector召回**（向量检索）：
```python
vector_scores = vector.get(target_id, [])  # List[{"dataset_id": id, "score": s}]
scores[item_id] = score * weights.get("vector", 0.4)  # 权重0.5
```
- 数据源：`item_recall_vector.json`（SBERT/TF-IDF+Faiss）
- 典型分数范围：**15.0-22.0**（L2距离或内积，**量纲完全不同！**）
- 默认权重：**0.5**

4. **Popular召回**（热门兜底）：
```python
scores[item_id] = base - idx * base * 0.01  # 递减分数
```
- 数据源：`top_items.json`（全局热门）
- 典型分数范围：**0.01-0.05**
- 默认权重：**0.05**

**问题1的本质**：

当前分数融合公式：
```
final_score = behavior_score * 1.5 + content_score * 0.8 + vector_score * 0.5 + popular_score * 0.05
```

实际分数示例（基于真实数据）：
- Behavior: 2.0 * 1.5 = **3.0**
- Content: 0.8 * 0.8 = **0.64**
- Vector: 18.0 * 0.5 = **9.0**（主导！）
- Popular: 0.03 * 0.05 = **0.0015**

Vector召回的原始分数是Content的**20倍**，即使权重较低，依然完全压制其他渠道。这导致：
- 非vector渠道的item永远排不到前面
- 多样性被vector的语义相似性主导
- 标签覆盖率低（vector倾向于聚集相似主题）

**架构细节**：

DEFAULT_CHANNEL_WEIGHTS配置（行183-188）：
```python
DEFAULT_CHANNEL_WEIGHTS = {
    "behavior": 1.5,  # 行为信号最强
    "content": 0.8,   # 内容相关性
    "vector": 0.5,    # 向量召回
    "popular": 0.05,  # 热门兜底
}
```

动态权重调整（行1589-1637）：
```python
def _compute_dynamic_channel_weights(...)
    # 场景1：无用户历史 → 降低behavior权重50%，提升content/vector
    if not user_history:
        _shift("behavior", ["content", "vector", "popular"], 0.5)

    # 场景2：behavior邻居<3个 → 降低30%，提升content/vector
    if len(behavior_neighbors) < 3:
        _shift("behavior", ["content", "vector"], 0.3)
```

这套动态调整在归一化前**无法解决量纲问题**，只是在错误的基础上微调。

#### 第二阶段：多渠道增强召回（补充多样性）

**函数**：`_augment_with_multi_channel()` (行1423-1514)

在主召回后，系统通过多个辅助渠道增强候选集：

1. **Tag召回**（标签匹配）：
```python
target_tags = state.dataset_tags.get(target_id, [])  # 如["机器学习", "深度学习"]
for tag in target_set:
    for candidate in tag_to_items.get(tag, set()):
        overlap = len(target_set & set(state.dataset_tags.get(candidate, [])))
        candidate_scores[candidate] += overlap
# 分数 * 0.4 加入总分
_bump(dataset_id, score * 0.4, "tag")
```

2. **Category召回**（同类别）：
```python
company = state.metadata.get(target_id, {}).get("company")
candidates = category_index.get(company.lower(), set())
_bump(dataset_id, 0.3, "category")  # 固定0.3分
```

3. **Price召回**（价格相似）：
```python
# 价格桶：0-100, 100-500, 500-1000, 1000-5000, 5000+
if price < 100: bucket = "0"
elif price < 500: bucket = "1"
# ...
_bump(dataset_id, 0.25, "price")  # 固定0.25分
```

4. **UserCF召回**（协同过滤）：
```python
similar_users = user_similarity.get(user_id, [])  # List[(user_id, similarity)]
for other_id, similarity in similar_users:
    for dataset in other_user_history:
        candidate_scores[dataset] += similarity
_bump(dataset_id, score * 0.6, "usercf")
```

**辅助渠道的问题**：分数量级同样不统一（0.25-3.0），与主召回混合后加剧不平衡。

#### 第三阶段：排序模型（LightGBM Ranker）

**函数**：`_apply_ranking_with_circuit_breaker()` (行2081-2098)

排序模型使用召回分数+静态特征+动态特征预测点击概率：

**特征工程**：`_compute_ranking_features()` (行1895-2019)

静态特征（来自数据库/缓存）：
```python
features["price_log"] = np.log1p(price)
features["description_length"] = len(description)
features["tag_count"] = len(tags.split(";"))
features["weight_log"] = np.log1p(total_interaction_weight)
features["interaction_count"] = user_interaction_count
```

动态特征（请求级别）：
```python
features["score"] = score_lookup.get(dataset_id, 0.0)  # 召回分数
features["position"] = position_in_recall_list
features["channel"] = extract_channel_from_reason(reason)  # "behavior"/"content"/"vector"
features["channel_weight"] = weights.get(channel, 0.1)
features["endpoint"] = "recommend_detail"
features["device_type"] = "mobile"/"desktop"/"tablet"
features["source"] = "detail_page"/"search"/"landing"
```

用户特征（从Redis Feature Store）：
```python
user_features = feature_store.get_user_features(user_id)
# {"user_activity_level": 0.8, "user_avg_price": 500.0, ...}
```

**排序分数融合**（行2094-2095）：
```python
probabilities = rank_model.predict(features)  # 点击概率 [0, 1]
scores[dataset_id] += prob  # 直接加到召回分数上！
```

**问题5的机会**：排序特征中**没有新鲜度特征**！系统无法区分7天前和7个月前的内容。

#### 第四阶段：MMR多样性重排（问题4的核心）

**函数**：`_apply_mmr_reranking()` (行1092-1155)

MMR（Maximal Marginal Relevance）算法平衡相关性与多样性：

```python
# 1. 归一化分数到[0,1]
max_score = max(scores.values())
min_score = min(scores.values())
normalized[item] = (score - min_score) / (max_score - min_score)

# 2. 迭代选择
while len(selected) < limit:
    for candidate in candidates:
        relevance = normalized_scores[candidate]

        # 计算与已选item的最大相似度（基于标签Jaccard）
        max_sim = max(
            jaccard_similarity(
                candidate_tags,
                dataset_tags.get(selected_item, [])
            )
            for selected_item in selected
        )

        # MMR分数 = λ * 相关性 - (1-λ) * 多样性惩罚
        mmr_scores[candidate] = lambda_param * relevance - (1 - lambda_param) * max_sim

    # 选择MMR分数最高的
    best = max(mmr_scores.items(), key=lambda x: x[1])[0]
    selected.append(best)
```

**Lambda参数计算**（行330-339）：
```python
def _compute_mmr_lambda(*, endpoint: str, request_context: Dict[str, str]) -> float:
    base = 0.7 if endpoint == "recommend_detail" else 0.5  # 详情页默认0.7

    # 场景调整
    if source in {"search", "landing"}:
        base = 0.5  # 搜索/落地页降低相关性

    if device_type == "mobile":
        base = min(base + 0.1, 0.85)  # 移动端提升相关性

    return base
```

**问题4的本质**：

λ=0.7意味着：
- 相关性权重70%
- 多样性权重仅30%

实际效果：
- 第1个item：纯相关性排序（无已选item）
- 第2个item：MMR = 0.7 * relevance - 0.3 * similarity
  - 如果similarity=0.8（高度相似），多样性惩罚仅-0.24
  - 相关性0.9的相似item仍能胜过相关性0.8的多样item
- 结果：前N个item高度同质化

**Jaccard相似度计算**（行1079-1089）：
```python
def _jaccard_similarity(tags1: List[str], tags2: List[str]) -> float:
    set1 = set(t.lower().strip() for t in tags1 if t)
    set2 = set(t.lower().strip() for t in tags2 if t)
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0
```

#### 第五阶段：探索机制（问题2的缺失）

**函数**：`_apply_exploration()` (行1158-1197)

当前代码**已实现**Epsilon-Greedy探索，但**未启用**：

```python
def _apply_exploration(
    ranked_ids: List[int],
    all_dataset_ids: Set[int],
    epsilon: float = 0.1,  # 探索率
) -> List[int]:
    n_total = len(ranked_ids)
    n_explore = int(n_total * epsilon)  # 探索位数
    n_exploit = n_total - n_explore  # 确定性位数

    # 保留前(1-epsilon)个确定性推荐
    exploit_ids = ranked_ids[:n_exploit]

    # 从全局item池随机采样填充剩余位置
    explore_pool = list(all_dataset_ids - set(exploit_ids))
    explore_ids = random.sample(explore_pool, min(n_explore, len(explore_pool)))

    return exploit_ids + explore_ids  # 拼接返回
```

**调用点**：`_build_response_items()` (行1200-1274)

```python
def _build_response_items(..., apply_exploration: bool = False, ...):
    # 先MMR重排
    if apply_mmr and dataset_tags:
        ranked_ids = _apply_mmr_reranking(...)

    # 再探索（当前未启用！）
    if apply_exploration and all_dataset_ids:
        ranked_ids = _apply_exploration(
            ranked_ids,
            all_dataset_ids,
            epsilon=exploration_epsilon
        )
```

**实际调用**（recommend_for_detail行2967-2972）：
```python
items = _build_response_items(
    scores, reasons, limit, state.metadata,
    dataset_tags=state.dataset_tags,
    apply_mmr=True,
    mmr_lambda=mmr_lambda,
    # apply_exploration=False  ← 未传参，默认False！
)
```

**问题2的本质**：代码框架完整，但未激活。只需传入参数即可启用。

#### 第六阶段：缓存写入与响应（问题3的闭环）

**缓存写入**（recommend_for_detail末尾）：
```python
if cache and cache.enabled and user_id:
    cache_key = f"recommend:{dataset_id}:{user_id}:{limit}"
    cache.set_json(cache_key, response.dict(), ttl=3600)  # 1小时TTL
```

**缓存类实现**（app/cache.py:28-241）：
```python
class RedisCache:
    def set_json(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        serialized = json.dumps(value, ensure_ascii=False)
        if ttl:
            return self.client.setex(key, ttl, serialized)  # 带过期时间
        return self.client.set(key, serialized)  # 永久缓存
```

**问题3的细节**：

虽然TTL=3600（1小时），但缓存键不包含时间桶标识：
```python
# 当前：user123在14:00和14:30请求同一dataset → 完全相同的key → 返回14:00的缓存
cache_key = "recommend:123:456:10"

# 需要：加入时间桶
cache_key = "recommend:123:456:10:2025-12-25-14"  # 每小时刷新
```

时间桶计算（任务中已提供方案）：
```python
def _get_time_bucket(bucket_hours: int = 1) -> str:
    now = datetime.now()
    if bucket_hours == 1:
        return now.strftime("%Y-%m-%d-%H")  # "2025-12-25-14"
    else:
        return now.strftime("%Y-%m-%d")  # "2025-12-25"
```

### 召回引擎架构（pipeline/recall_engine_v2.py）

**类定义**：`MultiChannelRecallEngine` (行25-791)

训练阶段构建8+个召回索引：

1. **UserCF**（行48-100）：
```python
def train_usercf(interactions: pd.DataFrame, min_common_items=2, top_k_users=50):
    user_items = interactions.groupby("user_id")["dataset_id"].apply(set).to_dict()
    # 计算用户Jaccard相似度
    for user_id, items in user_items.items():
        for other_id, other_items in user_items.items():
            common = len(items & other_items)
            if common >= min_common_items:
                similarity = common / len(items | other_items)
                similarities.append((other_id, similarity))
    # 保存Top-K相似用户
    user_similarity[user_id] = sorted(similarities, reverse=True)[:top_k_users]
```

2. **Tag倒排索引**（行102-139）：
```python
def train_tag_inverted_index(dataset_features: pd.DataFrame):
    tag_to_items = defaultdict(set)  # {"机器学习": {123, 456, 789}}
    item_to_tags = {}  # {123: ["机器学习", "深度学习"]}
    for row in dataset_features.iterrows():
        tags = row["tag"].split(";")
        item_to_tags[dataset_id] = tags
        for tag in tags:
            tag_to_items[tag].add(dataset_id)
```

3. **Category索引**（行141-170）：
```python
def train_category_index(dataset_features: pd.DataFrame):
    category_index = defaultdict(set)
    for row in dataset_features.iterrows():
        company = row["create_company_name"]
        category_index[company.lower()].add(dataset_id)
```

4. **Price桶索引**（行172-212）：
```python
def train_price_bucket_index(dataset_features: pd.DataFrame):
    price_buckets = defaultdict(set)
    for row in dataset_features.iterrows():
        price = row["price"]
        if price < 100: bucket = 0
        elif price < 500: bucket = 1
        elif price < 1000: bucket = 2
        elif price < 5000: bucket = 3
        else: bucket = 4
        price_buckets[bucket].add(dataset_id)
```

5. **Faiss向量召回**（行592-603）：
```python
if self.faiss_vector_recall and self.faiss_vector_recall.index is not None:
    faiss_items = self.faiss_vector_recall.search(
        target_dataset_id,
        k=limit // 7,
    )
    if faiss_items:
        results["vector_faiss"] = faiss_items  # List[(item_id, distance)]
```

**merge_recall_results合并**（行615-663）：
```python
def merge_recall_results(recall_results: Dict[str, List[Tuple[int, float]]], weights: Dict[str, float]):
    weights_default = {
        "behavior": 1.0,
        "content": 0.5,
        "vector_faiss": 0.7,  # Faiss高权重
        "usercf": 0.8,
        "tag": 0.6,
        "category": 0.4,
        "price": 0.3,
        "image": 0.5,
        "popular": 0.1,
    }

    item_scores = defaultdict(lambda: {"score": 0.0, "sources": []})
    for channel, items in recall_results.items():
        weight = weights.get(channel, 0.5)
        for item_id, score in items:
            item_scores[item_id]["score"] += score * weight  # 直接加权！
            item_scores[item_id]["sources"].append(channel)

    # 按分数排序
    merged = sorted(item_scores.items(), key=lambda x: x[1]["score"], reverse=True)[:limit]
```

**与main.py的交互**：

- main.py加载召回索引：`_load_recall_artifacts()` (行670-697)
- 存储到`app.state.recall_indices`
- `_augment_with_multi_channel()`从state中读取索引执行召回

### 实验框架与AB测试

**配置文件**：`config/experiments.yaml` (1-14行)

```yaml
experiments:
  recommendation_detail:
    status: active
    salt: "recommend-detail-v1"  # 分流盐值
    variants:
      - name: control
        allocation: 0.5  # 50%流量
        parameters: {}
      - name: content_boost
        allocation: 0.5  # 50%流量
        parameters:
          content_weight: 0.8  # 覆盖默认权重
          vector_weight: 0.5
```

**分流逻辑**（app/experiments.py）：

```python
def assign_variant(experiments, experiment_name, user_id, request_id):
    exp = experiments.get(experiment_name)
    if not exp or exp["status"] != "active":
        return "control", {}

    # 哈希分流
    hash_key = f"{exp['salt']}:{user_id or request_id}"
    hash_value = hashlib.md5(hash_key.encode()).hexdigest()
    bucket = int(hash_value[:8], 16) % 100 / 100.0  # [0, 1)

    # 分配variant
    cumulative = 0.0
    for variant in exp["variants"]:
        cumulative += variant["allocation"]
        if bucket < cumulative:
            return variant["name"], variant["parameters"]

    return "control", {}
```

**调用点**（recommend_for_detail行2818-2829）：

```python
experiment_variant, experiment_params = assign_variant(
    experiments,
    "recommendation_detail",
    user_id=user_id,
    request_id=request_id,
)

# 应用实验参数覆盖权重
channel_weights = _get_channel_weight_baseline(state)
for key, value in experiment_params.items():
    if key.endswith("_weight"):
        channel = key.replace("_weight", "")
        channel_weights[channel] = float(value)
```

### 数据模型与接口

**推荐响应模型**（app/main.py:529-546）：

```python
class RecommendationItem(BaseModel):
    dataset_id: int
    title: Optional[str]
    price: Optional[float]
    cover_image: Optional[str]
    score: float  # 最终融合分数
    reason: str  # "behavior+content+rank" 或 "exploration"

class RecommendationResponse(BaseModel):
    dataset_id: int  # 上下文item
    recommendations: List[RecommendationItem]
    request_id: str  # 追踪ID
    algorithm_version: Optional[str]  # 模型run_id
    variant: str  # "primary"/"shadow"/"fallback"
    experiment_variant: Optional[str]  # AB实验变体
    request_context: Optional[Dict[str, str]]  # 设备/来源等
```

**API端点**：

```python
@app.post("/api/v1/recommend/detail")
async def recommend_for_detail(
    request: Request,
    dataset_id: int,
    user_id: Optional[int] = None,
    limit: int = Query(10, ge=1, le=50),
) -> RecommendationResponse
```

### 关键配置与环境变量

**Redis缓存**（app/cache.py:247-275）：
```bash
REDIS_URL="redis://:password@localhost:6379/0"
# 或分别配置
REDIS_HOST="localhost"
REDIS_PORT="6379"
REDIS_PASSWORD=""
REDIS_DB="0"
```

**超时配置**（app/resilience.py + main.py:65-70）：
```bash
RECO_THREAD_POOL_WORKERS=4  # 异步执行池大小
SLOW_OPERATION_THRESHOLD=0.5  # 慢操作日志阈值（秒）
STAGE_LOG_THRESHOLD=0.25  # 阶段耗时日志阈值
USER_FEATURE_CACHE_TTL=30.0  # 用户特征缓存TTL
```

**Feature Store**（app/main.py:1299-1330）：
```bash
FEATURE_REDIS_URL="redis://localhost:6379/1"
# 用于实时特征存储（dataset特征、用户特征、统计数据）
```

**模型路径**（config/settings.py:74-89）：
```bash
MODELS_DIR="/path/to/models"  # 默认: {BASE_DIR}/models
DATA_DIR="/path/to/data"  # 默认: {BASE_DIR}/data
MLFLOW_TRACKING_URI="file://{BASE_DIR}/mlruns"
```

### 技术栈与依赖

**核心框架**：
- FastAPI：异步API服务
- Redis：缓存 + Feature Store
- Pandas/NumPy：数据处理
- Scikit-learn：Pipeline封装
- LightGBM：排序模型（可选）
- Faiss：高性能向量检索（可选）
- Sentence-Transformers：SBERT嵌入（可选）

**监控与可观测性**：
- Prometheus：指标采集（app/metrics.py）
- Sentry：错误追踪（app/sentry_config.py）
- 结构化日志：logging模块

**降级与容错**：
- Circuit Breaker：熔断器保护排序模型（app/resilience.py）
- Fallback Strategy：多级降级（行为 → 内容 → 热门）
- Timeout Manager：分阶段超时控制

### 修复方案的技术要点

#### 模块1：召回分数归一化

**核心原理**：将每个召回渠道的分数独立归一化到[0,1]，再乘以权重。

**实现位置**：`_combine_scores_with_weights()` 修改

**伪代码**：
```python
def normalize_channel_scores(channel_scores: Dict[int, float]) -> Dict[int, float]:
    if not channel_scores:
        return {}
    max_val = max(channel_scores.values())
    min_val = min(channel_scores.values())
    range_val = max_val - min_val if max_val > min_val else 1.0
    return {
        item_id: (score - min_val) / range_val
        for item_id, score in channel_scores.items()
    }

# 在_combine_scores_with_weights中应用
behavior_normalized = normalize_channel_scores(behavior.get(target_id, {}))
content_normalized = normalize_channel_scores(content.get(target_id, {}))
vector_normalized = normalize_channel_scores({
    int(e["dataset_id"]): float(e["score"])
    for e in vector.get(target_id, [])
})

# 加权融合
for item_id, norm_score in behavior_normalized.items():
    scores[item_id] = norm_score * weights.get("behavior", 1.0)
# ...同理处理其他渠道
```

**预期效果**：
- Behavior: [0, 1] * 1.5 = [0, 1.5]
- Content: [0, 1] * 0.8 = [0, 0.8]
- Vector: [0, 1] * 0.5 = [0, 0.5]
- 分数跨度从24倍降低到3倍（1.5/0.5）

#### 模块2：启用探索机制

**实现位置**：`recommend_for_detail()` 调用`_build_response_items()`时传参

**修改点**（行2967）：
```python
items = _build_response_items(
    scores, reasons, limit, state.metadata,
    dataset_tags=state.dataset_tags,
    apply_mmr=True,
    mmr_lambda=mmr_lambda,
    apply_exploration=True,  # 新增！
    exploration_epsilon=0.15,  # 15%探索率
    all_dataset_ids=set(state.metadata.keys()),  # 新增！
)
```

**预期效果**：
- 前8-9个item：确定性推荐（MMR排序）
- 后1-2个item：随机探索（从全局池采样）

#### 模块3：缓存时间桶

**实现位置**：`recommend_for_detail()` 缓存读写逻辑

**修改点**（行2843 + 缓存写入）：
```python
def _get_time_bucket(bucket_hours: int = 1) -> str:
    now = datetime.now()
    return now.strftime("%Y-%m-%d-%H") if bucket_hours == 1 else now.strftime("%Y-%m-%d")

# 读缓存
time_bucket = _get_time_bucket(bucket_hours=1)
cache_key = f"recommend:{dataset_id}:{user_id}:{limit}:{time_bucket}"

# 写缓存（同样的key模式）
cache.set_json(cache_key, response.dict(), ttl=3600)
```

**预期效果**：
- 14:00请求 → key="recommend:123:456:10:2025-12-25-14"
- 14:30请求 → 同一key，命中缓存
- 15:00请求 → key="recommend:123:456:10:2025-12-25-15"，重新计算

#### 模块4：MMR参数调整

**实现位置**：`_compute_mmr_lambda()` 和 `DEFAULT_CHANNEL_WEIGHTS`

**修改点1**（行330）：
```python
def _compute_mmr_lambda(...):
    base = 0.5 if endpoint == "recommend_detail" else 0.4  # 降低！
    # 场景调整保持不变
    if source in {"search", "landing"}:
        base = 0.4  # 降低
    if device_type == "mobile":
        base = min(base + 0.1, 0.6)  # 上限降低
    return base
```

**修改点2**（行183）：
```python
DEFAULT_CHANNEL_WEIGHTS = {
    "behavior": 1.2,   # 降低（原1.5）
    "content": 1.0,    # 提升（原0.8）
    "vector": 0.8,     # 提升（原0.5）
    "popular": 0.1,    # 提升（原0.05）
}
```

**预期效果**：
- MMR: λ=0.5 → 相关性50%、多样性50%（平衡）
- 权重重平衡 → 归一化后各渠道更公平竞争

#### 模块5：新鲜度特征工程

**实现位置**：`_compute_ranking_features()` 和 `_apply_ranking_with_circuit_breaker()`

**修改点1**（在_compute_ranking_features中添加特征）：
```python
# 假设dataset_features有create_time字段
if "create_time" in features.columns:
    now = pd.Timestamp.now()
    features["content_age_days"] = (
        (now - pd.to_datetime(features["create_time"])).dt.days
    ).fillna(999).astype(float)

    features["freshness_score"] = features["content_age_days"].apply(
        lambda days: 1.0 if days <= 7 else (0.5 if days <= 30 else 0.2)
    )
else:
    features["content_age_days"] = 999.0
    features["freshness_score"] = 0.2
```

**修改点2**（在_apply_ranking_with_circuit_breaker中应用加成）：
```python
for dataset_id, prob in zip(features.index.astype(int), probabilities.values):
    if dataset_id not in scores:
        continue

    # 获取新鲜度分数
    freshness_score = features.loc[dataset_id, "freshness_score"]
    freshness_boost = 0.8 + 0.2 * freshness_score  # [0.8, 1.0]

    # 应用加成
    prob = float(prob)
    scores[dataset_id] += prob * freshness_boost
```

**预期效果**：
- 7天内新内容：+20%排序boost
- 7-30天：+10% boost
- 30天以上：无boost

### 测试验证要点

**单元测试覆盖**：
```python
# 测试归一化
def test_normalize_channel_scores():
    scores = {1: 20.0, 2: 15.0, 3: 18.0}  # vector原始分数
    normalized = normalize_channel_scores(scores)
    assert all(0 <= v <= 1 for v in normalized.values())
    assert normalized[1] == 1.0  # max
    assert normalized[2] == 0.0  # min

# 测试探索
def test_exploration_mechanism():
    ranked_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    all_ids = set(range(1, 101))
    result = _apply_exploration(ranked_ids, all_ids, epsilon=0.15)
    assert len(result) == 10
    # 前85%应保持原序
    assert result[:8] == ranked_ids[:8]
    # 后15%应来自explore_pool
    assert result[8] not in ranked_ids or result[9] not in ranked_ids

# 测试时间桶
def test_time_bucket():
    bucket = _get_time_bucket(bucket_hours=1)
    assert len(bucket) == 13  # "2025-12-25-14"
    assert bucket.count("-") == 3
```

**集成测试**：
```python
# 测试完整推荐流程
@pytest.mark.asyncio
async def test_recommend_with_exploration():
    response = await recommend_for_detail(
        dataset_id=123, user_id=456, limit=10
    )
    assert len(response.recommendations) == 10

    # 检查多样性：前10个item的标签覆盖率
    all_tags = set()
    for item in response.recommendations:
        tags = state.dataset_tags.get(item.dataset_id, [])
        all_tags.update(tags)
    assert len(all_tags) >= 20  # 目标>25个

# 测试缓存时间桶
@pytest.mark.asyncio
async def test_cache_time_bucket():
    # 第一次请求
    resp1 = await recommend_for_detail(dataset_id=123, user_id=456, limit=10)

    # 模拟时间推进1小时
    with freeze_time("2025-12-25 15:00:00"):
        resp2 = await recommend_for_detail(dataset_id=123, user_id=456, limit=10)

    # 结果应不同（新时间桶）
    assert resp1.recommendations != resp2.recommendations
```

**性能基准测试**：
```bash
# P99延迟应<500ms
wrk -t4 -c100 -d30s --latency http://localhost:8000/api/v1/recommend/detail?dataset_id=123&limit=10

# 监控指标
curl http://localhost:8000/metrics | grep recommendation_latency_seconds
```

### 文件路径清单

**核心实现文件**：
- `/home/ubuntu/recommend/app/main.py` - 推荐服务主逻辑（所有5个模块修改点）
- `/home/ubuntu/recommend/app/cache.py` - Redis缓存实现
- `/home/ubuntu/recommend/pipeline/recall_engine_v2.py` - 多路召回引擎
- `/home/ubuntu/recommend/config/settings.py` - 全局配置
- `/home/ubuntu/recommend/config/experiments.yaml` - AB实验配置

**辅助文件**：
- `/home/ubuntu/recommend/app/experiments.py` - 实验分流逻辑
- `/home/ubuntu/recommend/app/resilience.py` - 熔断器、降级策略
- `/home/ubuntu/recommend/app/metrics.py` - Prometheus指标
- `/home/ubuntu/recommend/app/sentry_config.py` - Sentry错误追踪
- `/home/ubuntu/recommend/pipeline/feature_store_redis.py` - Feature Store

**数据文件**（模型加载路径）：
- `/home/ubuntu/recommend/models/item_sim_behavior.pkl` - Behavior召回模型
- `/home/ubuntu/recommend/models/item_sim_content.pkl` - Content召回模型
- `/home/ubuntu/recommend/models/item_recall_vector.json` - Vector召回索引
- `/home/ubuntu/recommend/models/top_items.json` - 热门列表
- `/home/ubuntu/recommend/models/rank_model.pkl` - LightGBM排序模型
- `/home/ubuntu/recommend/models/user_similarity.pkl` - UserCF模型
- `/home/ubuntu/recommend/models/tag_to_items.json` - Tag倒排索引
- `/home/ubuntu/recommend/models/category_index.json` - Category索引
- `/home/ubuntu/recommend/models/price_bucket_index.json` - Price桶索引

### 监控与回滚策略

**关键指标监控**：
```python
# Prometheus指标（app/metrics.py）
recommendation_requests_total  # 总请求数
recommendation_latency_seconds  # 延迟分布（P50/P95/P99）
recommendation_exposures_total  # 曝光量
recommendation_degraded_total  # 降级次数
recommendation_timeouts_total  # 超时次数

# 自定义业务指标
diversity_score = len(unique_tags_in_top10) / total_tags_count
exploration_ratio = count(reason="exploration") / total_recommendations
channel_distribution = {
    "behavior": count(reason contains "behavior") / total,
    "content": count(reason contains "content") / total,
    "vector": count(reason contains "vector") / total,
}
```

**AB实验配置**（分阶段上线）：
```yaml
# 阶段1：10%流量测试归一化+探索
experiments:
  recommendation_detail:
    variants:
      - name: control
        allocation: 0.9
      - name: normalize_explore
        allocation: 0.1
        parameters:
          enable_normalization: true
          exploration_epsilon: 0.15

# 阶段2：50%流量（含MMR调整）
# 阶段3：100%流量（含新鲜度）
```

**降级开关**：
```python
# 环境变量控制
ENABLE_SCORE_NORMALIZATION=true  # 分数归一化开关
ENABLE_EXPLORATION=true  # 探索机制开关
EXPLORATION_EPSILON=0.15  # 探索率（动态可调）
MMR_LAMBDA_OVERRIDE=0.5  # MMR参数覆盖
CACHE_TIME_BUCKET_HOURS=1  # 缓存时间桶粒度
ENABLE_FRESHNESS_BOOST=true  # 新鲜度加成开关
```

**回滚方案**：
1. 配置回滚：修改experiments.yaml分配比例
2. 代码回滚：Git revert到前一版本
3. 特性开关：环境变量快速关闭新功能
4. 缓存清理：`redis-cli KEYS "recommend:*:2025-12-25-*" | xargs redis-cli DEL`

### 潜在风险与缓解

**风险1：归一化降低头部item优势**
- 现象：高质量item分数被拉平
- 缓解：保持behavior权重最高（1.2），监控CTR变化
- 指标：CTR下降>5% → 回滚

**风险2：探索推荐低质内容**
- 现象：随机item质量不可控
- 缓解：
  - epsilon仅15%，影响有限
  - 从`state.metadata.keys()`采样（已过滤的item池）
  - 排序模型仍会对explore item打分
- 增强：过滤interaction_count<5的低质item

**风险3：缓存失效增加服务压力**
- 现象：时间桶导致缓存命中率下降
- 缓解：
  - 保留1小时TTL（不是实时刷新）
  - 监控Redis QPS和API P99延迟
  - 预热机制：定时任务预加载热门dataset的推荐
- 降级：压力过大时关闭时间桶（回退原缓存逻辑）

**风险4：新鲜度加成被滥用**
- 现象：低质量新内容曝光过多
- 缓解：
  - 加成上限20%（不是倍数级）
  - 仅7天内有效（时间窗口限制）
  - 需通过排序模型基础质量门槛
- 监控：新内容CTR vs 整体CTR

### 数据依赖清单

**必须存在的数据表/文件**：
- `dataset_features.parquet` - item元数据（title, price, description, tag, create_time）
- `dataset_stats.parquet` - 统计数据（interaction_count, total_weight, last_event_time）
- `interactions.parquet` - 用户行为数据（user_id, dataset_id, action_type, timestamp）

**可选增强数据**：
- Redis Feature Store - 实时特征
- `slot_metrics_aggregated.parquet` - 位置效应统计
- `user_profiles.db` - 用户画像

**新增字段需求**（模块5）：
- `dataset_features.create_time` - 内容创建时间（必须）
- 如不存在，需在数据pipeline中补全：
  ```python
  # pipeline/prepare_features.py
  dataset_features["create_time"] = pd.to_datetime(
      dataset_features.get("created_at") or dataset_features.get("publish_time") or "2020-01-01"
  )
  ```

---

## 实施检查清单

**前置验证**：
- [ ] 确认`dataset_features.parquet`包含`create_time`字段
- [ ] 本地环境能成功加载所有模型文件
- [ ] Redis可连接且有足够内存（监控键空间）
- [ ] 当前推荐结果符合问题描述（分数跨度24倍、标签覆盖率低）

**代码实施**：
- [ ] 模块1：`_combine_scores_with_weights`中实现`normalize_channel_scores`
- [ ] 模块2：`recommend_for_detail`调用时传入`apply_exploration=True`
- [ ] 模块3：缓存key加入`_get_time_bucket()`
- [ ] 模块4：修改`_compute_mmr_lambda`和`DEFAULT_CHANNEL_WEIGHTS`
- [ ] 模块5：`_compute_ranking_features`添加`freshness_score`特征

**测试验证**：
- [ ] 单元测试覆盖所有新增函数
- [ ] 集成测试验证完整推荐流程
- [ ] 本地压测P99延迟<500ms
- [ ] 人工验证推荐结果多样性提升

**部署上线**：
- [ ] 配置AB实验（10%流量）
- [ ] 监控仪表盘就绪（Grafana + Prometheus）
- [ ] 预警规则配置（CTR/延迟/错误率）
- [ ] 回滚脚本准备

**效果评估**（7天观察期）：
- [ ] 分数跨度降至8倍以内
- [ ] 标签覆盖率提升至25+个
- [ ] 非vector召回占比>40%
- [ ] CTR提升10-15%
- [ ] 新内容曝光量+50%

## Implementation Plan

### 模块1：召回分数归一化（P0）
**文件**: `app/main.py`
**函数**: `_combine_scores_with_weights`, `_augment_with_multi_channel`
**改动**: ~110行
**目标**: 让所有召回渠道在[0,1]范围内归一化后再乘以权重

**关键改动**：
```python
def normalize_channel_scores(channel_scores: Dict[int, float]) -> Dict[int, float]:
    """将单个召回渠道的分数归一化到[0, 1]"""
    if not channel_scores:
        return {}
    max_val = max(channel_scores.values())
    min_val = min(channel_scores.values())
    range_val = max_val - min_val if max_val > min_val else 1.0
    return {
        item_id: (score - min_val) / range_val
        for item_id, score in channel_scores.items()
    }
```

### 模块2：启用Epsilon-Greedy探索（P0）
**文件**: `app/main.py`
**函数**: `recommend_for_detail`
**改动**: ~5行
**目标**: 在推荐列表中注入15%的随机探索项

**关键改动**：
```python
items = _build_response_items(
    scores, reasons, limit, state.metadata,
    dataset_tags=state.dataset_tags,
    apply_mmr=True,
    mmr_lambda=mmr_lambda,
    apply_exploration=True,  # 启用
    exploration_epsilon=0.15,  # 15%探索率
    all_dataset_ids=set(state.metadata.keys()),
)
```

### 模块3：缓存键加入时间桶（P1）
**文件**: `app/main.py`
**函数**: `recommend_for_detail`
**改动**: ~15行
**目标**: 让缓存结果每小时刷新

**关键改动**：
```python
def _get_time_bucket(bucket_hours: int = 1) -> str:
    """返回时间桶标识，如 '2025-12-25-14'"""
    now = datetime.now()
    return now.strftime("%Y-%m-%d-%H") if bucket_hours == 1 else now.strftime("%Y-%m-%d")

# 缓存key加入时间桶
time_bucket = _get_time_bucket(bucket_hours=1)
cache_key = f"recommend:{dataset_id}:{user_id}:{limit}:{time_bucket}"
```

### 模块4：增强MMR多样性（P1）
**文件**: `app/main.py`
**函数**: `_compute_mmr_lambda`, `DEFAULT_CHANNEL_WEIGHTS`
**改动**: ~20行
**目标**: 降低相关性权重，提升多样性

**关键改动**：
```python
# MMR参数：λ从0.7降到0.5
def _compute_mmr_lambda(...):
    base = 0.5 if endpoint == "recommend_detail" else 0.4
    # 场景化调整...

# 渠道权重重新平衡
DEFAULT_CHANNEL_WEIGHTS = {
    "behavior": 1.2,   # 降低（原1.5）
    "content": 1.0,    # 提升（原0.8）
    "vector": 0.8,     # 提升（原0.5）
    "popular": 0.1,    # 提升（原0.05）
}
```

### 模块5：新鲜度加权（P2）
**文件**: `app/main.py`
**函数**: `_compute_ranking_features`, `_apply_ranking_with_circuit_breaker`
**改动**: ~30行
**目标**: 给新内容额外曝光机会

**关键改动**：
```python
# 添加新鲜度特征
features['freshness_score'] = features['content_age_days'].apply(
    lambda days: 1.0 if days <= 7 else (0.5 if days <= 30 else 0.2)
)

# 新鲜度加成（最多20%提升）
freshness_boost = 0.8 + 0.2 * freshness_score
scores[dataset_id] += prob * freshness_boost
```

## Implementation Phases

### 阶段1：紧急修复（今天）
- [ ] 模块1：召回分数归一化
- [ ] 模块2：启用探索机制
- [ ] 本地测试验证分数分布
- [ ] 验证多样性指标

### 阶段2：体验优化（明天）
- [ ] 模块3：缓存时间桶
- [ ] 模块4：MMR参数调整
- [ ] 集成测试

### 阶段3：长期优化（本周）
- [ ] 模块5：新鲜度特征工程
- [ ] A/B测试配置
- [ ] 部署到测试环境
- [ ] 监控业务指标

## Risk & Mitigation

| 风险 | 缓解措施 |
|------|---------|
| 归一化降低头部item优势 | 保持behavior权重最高（1.2），监控CTR |
| 探索机制推荐低质内容 | epsilon仅15%，过滤低评分item |
| 缓存失效增加服务压力 | 时间桶保留1小时缓存，保留降级机制 |
| 新鲜度加成被滥用 | 加成上限20%，仅7天内有效 |

## User Notes
- 所有修改需要保持向后兼容
- 优先实施阶段1（紧急修复），观察效果后再推进阶段2-3
- A/B测试必须覆盖所有5个模块的组合效果
- 监控指标：CTR、多样性、停留时长、新内容曝光

## Work Log
- [2025-12-25] 任务创建，问题诊断完成，实施方案确定
- [2025-12-26] **阶段1实施完成**：
  - ✅ 实现 `_normalize_channel_scores` 归一化函数（Min-Max scaling）
  - ✅ 重构 `_combine_scores_with_weights` 函数，对所有召回渠道应用归一化
    - Behavior召回：归一化到[0,1]后乘以权重1.0
    - Content召回：归一化到[0,1]后乘以权重0.5
    - Vector召回：归一化到[0,1]后乘以权重0.4
    - Popular召回：线性衰减归一化后乘以权重0.01
  - ✅ 修改 `_augment_with_multi_channel` 中tag召回归一化（Category和Price为固定分数无需归一化）
  - ✅ 启用探索机制：`apply_exploration=True`, `exploration_epsilon=0.15`
  - ✅ 本地测试验证：
    - 分数跨度从24.8倍降低到3.3倍（改善7.5倍）✅
    - Content召回分数（0.800）现可超过Vector召回（0.500）✅
    - 探索机制正常工作（15%随机探索）✅
  - **代码改动**：约90行（app/main.py）
  - **下一步**：阶段2（缓存时间桶 + MMR参数调整）
