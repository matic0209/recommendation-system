# 推荐系统算法说明文档

> 最后更新: 2026-02
> 版本: v2.1 (权重体系调整 + 相关度特征修复 + 环境变量补齐)

## 目录

1. [系统概述](#系统概述)
2. [推荐流程](#推荐流程)
3. [召回渠道](#召回渠道)
4. [排序模型](#排序模型)
5. [质量控制](#质量控制)
6. [多样性优化](#多样性优化)
7. [探索机制](#探索机制)
8. [配置参数](#配置参数)
9. [监控与调优](#监控与调优)
10. [常见问题](#常见问题)

---

## 系统概述

本推荐系统是一个多阶段流水线架构，用于数据交易平台的数据集推荐。系统采用"召回-排序-重排"的经典架构，支持多种召回策略和个性化排序。

### 核心特点

- **多渠道召回**: 5种召回策略并行执行
- **LightGBM排序**: 基于LambdaRank的学习排序模型
- **类别召回(12cat)**: 基于13个行业类别的精准召回
- **MMR多样性**: 平衡相关性与多样性
- **探索机制**: epsilon-greedy策略发现新内容

### 系统架构图

```
┌─────────────────────────────────────────────────────────────────────┐
│                         推荐请求 (dataset_id)                        │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          多渠道召回 (Recall)                         │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│  │ Behavior │ │ Content  │ │  Vector  │ │ Category │ │ Popular  │  │
│  │  (1.2)   │ │  (1.0)   │ │  (0.8)   │ │  (1.0)   │ │  (0.1)   │  │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         分数融合 (Score Fusion)                      │
│                    加权求和 + Min-Max归一化                          │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       LightGBM排序 (Ranking)                         │
│                    LambdaRank + 13类别特征                           │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        质量过滤 (Quality Filter)                     │
│                    百分位过滤(P30, 砍掉最差 30%)                      │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        MMR重排 (Reranking)                           │
│           λ=0.5(detail)/0.4(similar)（相关性 vs 多样性）              │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        探索 (Exploration)                            │
│                    ε=0.10 (10%随机探索，仅 detail)                    │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          推荐结果 (Top-K)                            │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 推荐流程

### 阶段1: 多渠道召回

从多个维度召回候选数据集，每个渠道独立计算相关性分数。

### 阶段2: 分数融合

将各渠道分数进行归一化和加权融合：
```python
final_score = Σ(channel_weight × normalized_score)
```

### 阶段3: LightGBM排序

使用机器学习模型对候选集进行精排，输入特征包括：
- 召回分数特征
- 价格特征
- 交互统计特征
- 13类别one-hot特征

### 阶段4: 质量过滤

- 移除排序分数为负的低质量item
- 应用百分位过滤(P30)剔除尾部item

### 阶段5: MMR重排

使用Maximal Marginal Relevance算法优化结果多样性。

### 阶段6: 探索

以一定概率插入随机item，发现潜在优质内容。

---

## 召回渠道

### 1. Behavior召回 (行为协同)

基于用户行为的协同过滤推荐。

| 属性 | 值 |
|------|-----|
| 算法 | User-CF + Item-CF |
| 分数范围 | [0, 1] |
| 归一化 | Min-Max |
| 默认权重 | 1.2 |
| 数据源 | 用户购买/浏览历史 |

### 2. Content召回 (内容相似)

基于数据集描述和标签的文本相似度。

| 属性 | 值 |
|------|-----|
| 算法 | TF-IDF + 余弦相似度 |
| 分数范围 | [0.2, 0.9] |
| 归一化 | Min-Max |
| 默认权重 | 0.9 |
| 数据源 | 数据集描述、标签 |

### 3. Vector召回 (语义向量)

基于SBERT语义向量的深度语义匹配。

| 属性 | 值 |
|------|-----|
| 算法 | SBERT embedding + 余弦相似度 |
| 分数范围 | [15, 22] (原始) |
| 归一化 | Min-Max |
| 默认权重 | 0.6 |
| 向量维度 | 768 |

### 4. Category召回 (12cat类别召回) ⭐

基于行业类别的精准匹配召回，是提升推荐相关性的核心渠道。

| 属性 | 值 |
|------|-----|
| 算法 | 类别匹配 + 类别重叠度评分 |
| 分数范围 | [0, 1] |
| 归一化 | 类别数归一化 |
| 加分系数 | 0.6（`DEFAULT_AUGMENT_WEIGHTS["12cat"]`，旧版为硬编码 0.5） |
| 类别数量 | 13个 |

> ⚠️ **类别召回不参与"归一化×权重"的融合计算**，它是在归一化之后直接加一个绝对分
> （`app/main.py` 的 `_augment_with_multi_channel`），因此它的 0.6 与其他渠道的
> 权重（behavior/content/vector/popular）**不是同一量纲，不能直接比大小**。
> 可通过实验参数 `augment_12cat_weight` 覆盖。

### 5. 辅助召回加分系数（绝对分量纲）

| 渠道 | 系数 | 来源 |
|------|------|------|
| tag | 0.4 | `DEFAULT_AUGMENT_WEIGHTS` |
| 12cat | 0.6 | 同上 |
| category（同公司，遗留） | 0.3 | 同上 |
| price（价格分桶） | 0.2 | 同上 |
| usercf（相似用户） | 0.6 | 同上 |

#### 支持的类别 (13类)

| 类别 | 说明 | 数据集数量 | 召回次数(月) |
|------|------|-----------|-------------|
| 文化娱乐 | 影视、游戏、音乐等 | 4,625 | 179,393 |
| 金融财经 | 证券、银行、保险 | 1,994 | 261,741 |
| 农业农村 | 农业数据、乡村发展 | 2,439 | 27,144 |
| 社会民生 | 民生服务、公共事务 | 1,835 | 147,872 |
| 医疗健康 | 医疗、健康、医药 | 1,657 | 136,892 |
| 政府政务 | 政务数据、政策法规 | 1,429 | 24,979 |
| 互联网科技 | IT、AI、软件 | 827 | 241,106 |
| 教育科研 | 教育、学术研究 | 684 | 82,657 |
| 商业零售 | 电商、零售、消费 | 615 | 45,529 |
| 交通物流 | 交通、物流、运输 | 450 | 81,053 |
| 能源环保 | 能源、环保、气候 | 328 | 41,918 |
| **数字产品** | 鼠标皮肤、壁纸、工具软件 | 184 | 28,661 |
| 工业制造 | 制造业、工业数据 | 113 | 10,450 |

#### 类别分类方法

使用零样本分类模型对数据集描述进行自动分类：

```python
# 模型: IDEA-CCNL/Erlangshen-Roberta-330M-NLI
# 脚本: pipeline/enhance_tags.py

classifier = pipeline("zero-shot-classification", model=MODEL_NAME)
result = classifier(text, candidate_labels=TOP_CATEGORIES, multi_label=True)

# 阈值: 0.5 (DEFAULT_THRESHOLD，置信度 >= 0.5 才分配该类别)
# 旧文档写的 0.3 是历史值；想提高覆盖率可用 python -m pipeline.enhance_tags --threshold 0.3
```

#### 类别召回逻辑

```python
def category_recall(target_id):
    # 1. 获取目标数据集的类别
    target_cats = item_to_categories.get(target_id, [])

    # 2. 遍历每个类别，获取同类别候选
    candidates = {}
    for cat in target_cats:
        for candidate_id in category_to_items.get(cat, []):
            if candidate_id == target_id:
                continue
            # 3. 计算类别重叠度分数
            candidate_cats = item_to_categories.get(candidate_id, [])
            overlap = len(set(target_cats) & set(candidate_cats))
            score = overlap / len(target_cats)
            candidates[candidate_id] = max(candidates.get(candidate_id, 0), score)

    return candidates
```

### 5. Popular召回 (热门推荐)

全局热门榜单，作为冷启动和补充。

| 属性 | 值 |
|------|-----|
| 算法 | 交互次数排序 + 线性衰减 |
| 分数范围 | [0.1, 1.0] |
| 归一化 | 已归一化 |
| 默认权重 | 0.1 |
| 质量过滤 | 双层过滤机制 |

#### 质量过滤规则

**训练阶段过滤**:
```python
price >= 0.5 AND
interaction_count >= 10 AND
days_since_last_purchase <= 730
```

**运行时过滤**:
```python
NOT (price < 1.90 AND interaction_count < 66) AND
NOT (days_inactive > 180 AND interaction_count < 30)
```

---

## 排序模型

### LightGBM LambdaRank

使用LightGBM的学习排序功能进行精排。

#### 模型配置

```python
params = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
}
```

#### 输入特征

| 特征组 | 特征 | 说明 |
|--------|------|------|
| 召回特征 | behavior_score | 行为召回分数 |
| | content_score | 内容召回分数 |
| | vector_score | 向量召回分数 |
| | category_score | 类别召回分数 |
| | popular_score | 热门召回分数 |
| 价格特征 | price | 数据集价格 |
| | price_bucket | 价格分桶 |
| 统计特征 | interaction_count | 交互次数 |
| | days_since_purchase | 最近购买天数 |
| 类别特征 | cat_政府政务 ~ cat_数字产品 | 13个类别one-hot |

#### 输出说明

- 输出范围: (-∞, +∞)
- 正分: 质量高于平均水平
- 负分: 质量低于平均水平

---

## 质量控制

### 负分硬截断（历史机制，已下线）

> 2025-12-27 引入的"直接移除 score < 0 的候选"已不再存在于 `app/main.py`，
> 现行为**百分位过滤**，下文为其真实实现。

负分只代表"LightGBM 原始预测分低于本次候选的平均水平"，并不是绝对质量判断，
因此现在用相对分位的统一口径处理。

### 百分位过滤 (P30)

移除分数处于底部 30% 的候选（默认阈值 `FILTER_PERCENTILE = 30`）。

```python
if total_items >= 10:                      # 候选 < 10 个时不启用过滤
    threshold = np.percentile(score_values, 30)
    low_score_items = [i for i, s in scores.items() if s < threshold]
    if len(low_score_items) > total_items * 0.5:   # 安全兜底
        keep_ids = 分数最高的 50%（至少 5 个）
        low_score_items = [i for i in low_score_items if i not in keep_ids]
```

- 阈值可调位置：`app/main.py` 的 `FILTER_PERCENTILE` / `MIN_CANDIDATES_FOR_FILTER`
- 日志关键字：`Percentile filter (p30, threshold=...)`


---

## 多样性优化

### MMR (Maximal Marginal Relevance)

平衡相关性与多样性的重排算法。

#### 公式

```
MMR = λ × Relevance(item) - (1-λ) × max(Similarity(item, selected))
```

#### 参数

| 参数 | 值 | 说明 |
|------|-----|------|
| λ (lambda) | detail 0.5 / similar 0.4 | 相关性权重（越大越偏相关性） |
| 1-λ | 0.5 / 0.6 | 多样性权重 |
| 相似度计算 | Jaccard | 基于标签的相似度 |
| 场景修正 | search→0.6、landing/home→0.3、移动端 −0.1 | 见 `_compute_mmr_lambda` |
| 环境变量 | `MMR_LAMBDA` / `MMR_LAMBDA_SIMILAR` | 调整基准值（0.1~0.9） |

---

## 探索机制

### Epsilon-Greedy

以一定概率把结果尾部的条目替换为随机数据集，平衡利用与探索。

| 参数 | 值 | 说明 |
|------|-----|------|
| ε (epsilon) | 0.10 | 探索概率（原硬编码 0.15） |
| 1-ε | 0.90 | 利用概率 |
| 环境变量 | `EXPLORATION_EPSILON` | 可调范围 0~0.5 |
| 生效范围 | 仅 `/recommend/detail` | `/similar` 不做探索 |


---

## 配置参数

### 渠道权重

```python
# app/main.py
DEFAULT_CHANNEL_WEIGHTS = {
    "vector": 1.2,    # 语义向量 —— 一级（业务规则：vector = content > behavior > popular）
    "content": 1.2,   # 内容/标签相似 —— 一级
    "behavior": 0.9,  # 行为协同 —— 二级
    "popular": 0.05,  # 全局热门 —— 三级（仅兜底，且最终列表最多占 25%）
    "12cat": 0.8,     # 仅作为排序特征 channel_weight 的取值，不参与融合
}

# 辅助渠道加分系数（绝对分量纲，与上面的融合权重不是同一套刻度）
DEFAULT_AUGMENT_WEIGHTS = {
    "tag": 0.4, "12cat": 0.6, "category": 0.3, "price": 0.2, "usercf": 0.6,
}
```

> ⚠️ **上面的默认值可能不生效**：服务启动时若存在 `models/channel_weights.json`
> （由 `pipeline/train_channel_weights.py` 每天按真实 CTR/CVR 重算），其中的
> `behavior/content/vector/popular` 会覆盖代码默认值——但**覆盖结果会被
> `_enforce_channel_weight_order()` 限幅（±40%）并强制保序**，
> `vector = content > behavior > popular` 这个业务次序不会被反转。
> 调权重前请先查看该文件；详细说明见 [推荐权重与推荐策略调整说明](./推荐权重调整说明.md)。

### 最终列表配额（2026-02 新增）

分数排完序之后还有一层"业务配额"，避免某类商品/某个渠道刷屏：

| 规则 | 参数 | 默认值 | 效果（limit=12） |
|------|------|--------|-----------------|
| popular 来源占比上限 | `MAX_POPULAR_SHARE` | 0.25 | 最多 3 条来自全局热门 |
| 非目标行业单一类别上限 | `MAX_OTHER_CATEGORY_SHARE` | 0.25 | 最多 3 条同属某个"与当前数据集无关"的行业类别 |
| 目标自身行业 | — | 豁免 | 鼠标指针详情页可以正常返回多条鼠标指针 |

实现位置：`app/main.py` 的 `_apply_list_quotas()`（在 MMR 之后、探索之前），
候选不足时会放宽补齐以保证返回条数不缩水，并记录 `List quota applied` 日志。

### 在售状态过滤

推荐链路增加了"不可售/已下架数据集"的剔除能力（`_prune_unavailable_datasets` /
`RECO_EXCLUDED_DATASET_IDS` / `models/excluded_dataset_ids.json`）：
被剔除的数据集不会出现在召回、探索或兜底列表里。实测线上曾出现
`datasetStatus=2`（已下架）的数据被推荐，此机制用于兜底；根治需要让特征表
带上在售标记。

### 动态权重调整

`_compute_dynamic_channel_weights` 会在数据稀疏时把 `behavior`/`content`/`vector`
的权重转移给其他渠道，**转移量按各目标渠道当前权重的占比分配**（旧版为等额均分，
会把 popular 从 0.02 抬到 0.22）。触发条件：无用户历史、邻居渠道样本 < 3、
无向量、无标签。

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| POPULAR_MIN_PRICE | 0.5 | 热门召回最低价格（**训练阶段**，每日流水线） |
| POPULAR_MIN_INTERACTION | 10 | 热门召回最低交互数（训练阶段） |
| POPULAR_MAX_INACTIVE_DAYS | 730 | 热门召回最大不活跃天数（训练阶段） |
| POPULAR_ENABLE_FILTER | true | 是否启用训练阶段质量过滤 |
| MMR_LAMBDA | 0.5 | `/recommend/detail` 的 MMR 相关性基准 λ（0.1~0.9） |
| MMR_LAMBDA_SIMILAR | 0.4 | `/similar` 的 MMR 相关性基准 λ（0.1~0.9） |
| EXPLORATION_EPSILON | 0.10 | 探索率（0~0.5），仅 `/recommend/detail` 生效 |

> 这三个 MMR/探索相关的环境变量是本次改动新补齐的（此前文档写了但代码未读取，
> λ/ε 是硬编码）。写错值只会回退默认值并打告警，不会导致请求异常。
> 热门召回的**运行时**质量门槛（`price < 1.90 且 交互 < 66`、`不活跃 > 180 天 且 交互 < 30`）
> 目前是代码内常量，见 `app/main.py` 的 `_combine_scores_with_weights`。

### 实验参数（config/experiments.yaml，热加载）

| 参数 | 作用 |
|------|------|
| `behavior_weight` / `content_weight` / `vector_weight` / `popular_weight` | 覆盖融合层渠道权重 |
| `augment_tag_weight` / `augment_12cat_weight` / `augment_category_weight` / `augment_price_weight` / `augment_usercf_weight` | 覆盖辅助渠道加分系数 |

### 索引文件

| 文件 | 说明 | 当前数量 |
|------|------|---------|
| models/item_to_categories.json | 数据集→类别映射 | 8,342 items |
| models/category_to_items.json | 类别→数据集映射 | 13 categories |
| models/top_items.json | 热门榜单 | - |
| models/lightgbm_ranker.txt | 排序模型 | - |
| models/channel_weights.json | 每日 CTR 训练出的渠道权重（会覆盖代码默认值） | - |


---

## 监控与调优

### 日志监控

#### 12cat类别召回日志

```bash
# 查看类别召回统计
docker logs recommendation-api 2>&1 | grep "12cat:"

# 示例输出:
# 12cat: target_id=123, has_item_to_cats=8342, has_cat_to_items=13, target_cats=['金融财经']
# 12cat cat=金融财经 has 1994 candidates
```

#### 关键指标

| 指标 | 计算方式 | 健康阈值 |
|------|---------|---------|
| 类别覆盖率 | 有类别请求数 / 总请求数 | > 50% |
| 召回成功率 | 成功请求数 / 总请求数 | > 99% |
| 负分比例 | 负分item数 / 总候选数 | < 30% |

### 生产运行数据 (2026-01)

| 指标 | 数值 |
|------|------|
| 总请求数 | 1,465,483 |
| 成功率 | 99.999% |
| 类别覆盖率 | 54.4% |
| 12cat召回请求 | 1,450,446 |

---

## 常见问题

### Q: 为什么有些数据集没有类别？

**原因**:
1. 数据集描述为空或仅包含图片
2. 描述文本过短 (< 10字符)
3. 分类置信度低于阈值

**解决**:
```bash
# 降低阈值重新分类
python -m pipeline.enhance_tags --threshold 0.3

# 只处理特定日期后的数据
python -m pipeline.enhance_tags --since 2025-10-01 --threshold 0.3
```

### Q: 如何添加新类别？

1. 修改 `pipeline/enhance_tags.py` 中的 `TOP_CATEGORIES`
2. 运行分类脚本
3. 重启服务: `docker stop recommendation-api && docker start recommendation-api`

### Q: 推荐结果相关性差？

**排查步骤**:
1. 检查目标数据集是否有类别
2. 查看召回日志中的 `target_cats`
3. 确认各渠道权重配置

---

## 更新历史

| 日期 | 版本 | 更新内容 |
|------|------|---------|
| 2026-02 | v2.1 | 权重体系调整：content 0.9 / vector 0.6 / popular 0.05；辅助渠道系数改为 `DEFAULT_AUGMENT_WEIGHTS` 可配（12cat 0.5→0.6）；动态权重改为按比例转移（修复 popular 被放大 11 倍）；**修复排序模型 5 个相关度特征在推理端恒为 0 的训练/推理不一致问题**；补齐 `MMR_LAMBDA` / `MMR_LAMBDA_SIMILAR` / `EXPLORATION_EPSILON` 环境变量（探索率 0.15→0.10）；曝光日志记录实际生效权重。详见 [推荐权重调整说明](./推荐权重调整说明.md) |
| 2026-01-28 | v2.0 | 添加"数字产品"类别; 修复HTML实体解码; 添加--since参数 |
| 2025-12-28 | v1.3 | Popular召回双层质量过滤 |
| 2025-12-27 | v1.2 | 负分硬截断机制（后续已被百分位过滤取代）; Tag召回大小写修复 |
| 2025-12-20 | v1.1 | 12cat类别召回上线 |
| 2025-12-01 | v1.0 | 初始版本 |

---

## 参考文件

| 文件 | 说明 |
|------|------|
| `app/main.py` | 推荐API主逻辑 |
| `pipeline/enhance_tags.py` | 标签增强/分类脚本（类别体系来源，默认阈值 0.5） |
| `pipeline/train_models.py` | 模型训练脚本 |
| `pipeline/train_channel_weights.py` | 每日按 CTR/CVR 重算 `models/channel_weights.json` |
| `scripts/verify_channel_weight_changes.py` | 权重/相关度特征的运行时自检脚本 |
| `docs/推荐权重调整说明.md` | 权重调参指南（生效层级、改哪里、如何验证回滚） |
| `CLAUDE.md` | 项目开发指南 |
