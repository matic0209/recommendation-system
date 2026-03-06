# 推荐系统算法说明文档

> 最后更新: 2026-01-28
> 版本: v2.0 (含12cat类别召回)

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
│                    负分截断 + 百分位过滤(P30)                         │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        MMR重排 (Reranking)                           │
│                    λ=0.7 (70%相关性 + 30%多样性)                      │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        探索 (Exploration)                            │
│                    ε=0.15 (15%随机探索)                              │
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
| 默认权重 | 1.0 |
| 数据源 | 数据集描述、标签 |

### 3. Vector召回 (语义向量)

基于SBERT语义向量的深度语义匹配。

| 属性 | 值 |
|------|-----|
| 算法 | SBERT embedding + 余弦相似度 |
| 分数范围 | [15, 22] (原始) |
| 归一化 | Min-Max |
| 默认权重 | 0.8 |
| 向量维度 | 768 |

### 4. Category召回 (12cat类别召回) ⭐

基于行业类别的精准匹配召回，是提升推荐相关性的核心渠道。

| 属性 | 值 |
|------|-----|
| 算法 | 类别匹配 + 类别重叠度评分 |
| 分数范围 | [0, 1] |
| 归一化 | 类别数归一化 |
| 默认权重 | 1.0 |
| 类别数量 | 13个 |

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

# 阈值: 0.3 (置信度 >= 0.3 则分配该类别)
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

### 负分硬截断

移除排序分数为负的低质量item。

```python
# 过滤负分item
positive_items = [item for item in candidates if item.score >= 0]

# Fallback: 如果全部为负分，保留分数最高的50%
if len(positive_items) == 0:
    sorted_items = sorted(candidates, key=lambda x: x.score, reverse=True)
    positive_items = sorted_items[:max(5, len(sorted_items) // 2)]
```

### 百分位过滤 (P30)

移除分数处于底部30%的item。

```python
threshold = np.percentile(scores, 30)
filtered_items = [item for item in candidates if item.score >= threshold]
```

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
| λ (lambda) | 0.7 | 相关性权重 |
| 1-λ | 0.3 | 多样性权重 |
| 相似度计算 | Jaccard | 基于标签的相似度 |

---

## 探索机制

### Epsilon-Greedy

以一定概率插入随机item，平衡利用与探索。

| 参数 | 值 | 说明 |
|------|-----|------|
| ε (epsilon) | 0.15 | 探索概率 |
| 1-ε | 0.85 | 利用概率 |

---

## 配置参数

### 渠道权重

```python
DEFAULT_CHANNEL_WEIGHTS = {
    "behavior": 1.2,   # 行为协同
    "content": 1.0,    # 内容相似
    "vector": 0.8,     # 语义向量
    "category": 1.0,   # 类别召回
    "popular": 0.1,    # 热门推荐
}
```

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| POPULAR_MIN_PRICE | 0.5 | 热门召回最低价格 |
| POPULAR_MIN_INTERACTION | 10 | 热门召回最低交互数 |
| POPULAR_MAX_INACTIVE_DAYS | 730 | 热门召回最大不活跃天数 |
| MMR_LAMBDA | 0.7 | MMR相关性权重 |
| EXPLORATION_EPSILON | 0.15 | 探索概率 |

### 索引文件

| 文件 | 说明 | 当前数量 |
|------|------|---------|
| models/item_to_categories.json | 数据集→类别映射 | 8,342 items |
| models/category_to_items.json | 类别→数据集映射 | 13 categories |
| models/top_items.json | 热门榜单 | - |
| models/lightgbm_ranker.txt | 排序模型 | - |

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
| 2026-01-28 | v2.0 | 添加"数字产品"类别; 修复HTML实体解码; 添加--since参数 |
| 2025-12-28 | v1.3 | Popular召回双层质量过滤 |
| 2025-12-27 | v1.2 | 负分硬截断机制; Tag召回大小写修复 |
| 2025-12-20 | v1.1 | 12cat类别召回上线 |
| 2025-12-01 | v1.0 | 初始版本 |

---

## 参考文件

| 文件 | 说明 |
|------|------|
| `app/main.py` | 推荐API主逻辑 |
| `pipeline/enhance_tags.py` | 标签增强/分类脚本 |
| `pipeline/train_models.py` | 模型训练脚本 |
| `CLAUDE.md` | 项目开发指南 |
