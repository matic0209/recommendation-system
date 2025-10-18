# 推荐算法与推荐理由说明（中文）

本文介绍当前推荐系统的算法结构、离线训练流程以及线上 `reason` 字段的含义，便于研发与运营人员理解推荐策略与结果解释。

---

## 1. 推荐算法整体架构

推荐分为两大阶段：**召回** 与 **排序**。

### 1.1 召回层（Recall）

召回阶段负责从海量候选中筛出可能感兴趣的候选，当前包含：

1. **行为召回（Behavior）**  
   - 根据用户历史交互（浏览 / 订购等），使用共现计数构建 `item_sim_behavior.pkl`；  
   - 对每个数据集生成相似数据集集合，权重为共现次数归一化；  
   - 推荐理由对应 `behavior`、`behavior+category`。

2. **内容召回（Content）**  
   - 文本部分：拼接标题、tag，构建 TF-IDF 向量，计算余弦相似度；  
   - 图像部分（可选）：使用 CLIP 模型生成图像 embedding，与文本相似度加权融合；  
   - 产出 `item_sim_content.pkl`、`item_recall_vector.json` 等，推荐理由对应 `content`、`content+category`、`content+tag` 等。

3. **热门/类目召回（Popular/Category）**  
   - 基于全局统计生成 `top_items.json`，用于冷启动或热门推荐；  
   - 可叠加类目信息，推荐理由为 `category`、`category+price`、`tag+category`。


### 1.2 排序层（Ranking）

排序模型用于对召回结果进一步打分排序：

1. 构建排序特征（`dataset_features_v2.parquet` + `dataset_stats_v2.parquet`）：  
   - 文本长度、tag 数量、类别信息；  
   - 交互统计（曝光次数、权重）；  
   - 价格特征（`price_log` 等）；  
   - 图像特征（`has_cover`、`has_images`、`image_richness_score` 等）。

2. 标签：  
   - 根据历史行为（是否存在交互）构建二分类标签；  
   - 单调递增的曝光统计也参与排序指标。

3. 模型：  
   - 默认使用 LightGBM；特征通过 `StandardScaler` 后进入 `LGBMClassifier`；  
   - 模型保存在 `models/rank_model.pkl`，评估指标写入 MLflow。


### 1.3 影子模型（Shadow Rollout）

在线服务支持影子模型机制，可在运行时加载 `shadow` 模型进行灰度评估，控制 rollout 百分比。如果启用影子模型，推荐结果中也会带上版本信息，便于 A/B 对比。

---

## 2. 离线训练流程

![流程示意](placeholder) <!-- 如需图示可后续补图 -->

1. **数据抽取**：`pipeline.extract_load` 按水位抽取 Business / Matomo 数据；
2. **特征工程**：`pipeline.build_features` 生成基本特征、清洗流程、图像嵌入；
3. **模型训练**：`pipeline.train_models` 训练行为召回、内容召回、排序模型，产出模型文件并记录至 MLflow；
4. **评估与指标**：  
   - `pipeline.evaluate` 根据曝光日志与 Matomo 行为计算 CTR/CVR 等指标；  
   - `pipeline.reconcile_metrics` 与业务指标对账；
5. **模型上线**：训练完成后模型文件放置 `models/`，在线服务可直接读取或通过 `/reload` 接口刷新。

---

## 3. 在线推荐接口与 `reason` 字段

### 3.1 主要接口

| 接口 | 示例 | 说明 |
|------|------|------|
| `GET /health` | `/health` | 服务健康检查，返回模型加载状态 |
| `GET /similar/{dataset_id}` | `/similar/123?top_n=10` | 基于召回的相似推荐 |
| `GET /recommend/detail/{dataset_id}` | `/recommend/detail/123?user_id=7&top_n=5` | 详情页推荐，整合召回 + 排序结果 |

### 3.2 `reason` 字段说明

`reason` 用于解释推荐结果依据，便于前端展示或排查链路。当前可能出现的值及含义如下：

| reason | 说明 | 典型场景 |
|--------|------|----------|
| `behavior` | 用户行为召回命中 | 与用户历史互动数据集相似 |
| `behavior+category` | 行为 + 类目共同作用 | 同一类目下同类型内容互推 |
| `content` | 文本 / 内容相似度召回 | 利用标题、描述、tag 相似推荐 |
| `content+category` | 内容 + 类目 | 内容相似且属于同类 |
| `content+tag` | 内容 + 标签信息 | tag 具备强约束时使用 |
| `category` | 纯类目推荐 | 类目热门或同类别替代推荐 |
| `category+price` | 类目 + 价格区间 | 同类且价格接近 |
| `tag+category` | 标签 + 类目 | 需要双重约束场景 |

> `reason` 的取值随着召回/排序模块的扩展会调整，可在 `app/main.py` 中查看具体逻辑。

### 3.3 推荐兜底策略

当所有召回策略均未命中时，系统会回退到热门列表（热门 reason）。如果排序模型出现异常（如缺少特征），会记录 warning 并降级；可在 `/health` 输出或日志中查看。

---

## 4. 评估指标

评估阶段结合曝光日志与 Matomo 行为生成以下指标：

| 指标 | 说明 | 来源 |
|------|------|------|
| `exposures_total` | 总曝光次数 | `exposure_log.jsonl` |
| `exposures_unique_users` | 曝光用户数 | 同上 |
| `exposures_ctr` | 点击率 | 曝光与 Matomo 行为匹配 |
| `exposures_cvr` | 转化率 | 曝光与 Matomo 转化匹配 |
| `ranking_spearman` | 排序分数与行为权重的相关性 | 排序模型评估 |
| `ranking_hit_rate_top20` | 排序结果命中热门数据集的比例 | 排序模型评估 |
| `vector_recall_hit_rate_at_50` | 向量召回命中率 | 向量召回 vs 实际共现 |

指标写入 `data/evaluation/summary.json`、`exposure_metrics.json`，并可在 Grafana dashboard 中可视化。

---

## 5. 扩展建议

1. **特征扩展**  
   - 用户画像（职业、偏好标签）；  
   - 数据集的时效性、质量评分；  
   - 实时行为特征。

2. **算法迭代**  
   - 引入基于向量数据库的召回，实现更低延迟的向量检索；  
   - 排序模型切换为二阶段（如列表排序 + rerank）；  
   - 利用曝光/点击数据训练精细化模型（可在 `pipeline.train_models` 中增加新 pipeline）。

3. **在线 AB 测试**  
   - 使用影子模型机制调整 `shadow_rollout`；  
   - `app/experiments.py` 可扩展策略分流。

4. **推荐理由展示**  
   - 根据 `reason` 在前端展示对应文案，比如“因为你浏览过 XX”、“与当前内容同类”等，提升解释性。

---

如需进一步了解代码细节，可阅读：
- `pipeline/train_models.py`：召回/排序模型训练流程；
- `pipeline/build_features.py`：特征生成、清洗及图像嵌入；
- `app/main.py`：在线接口与 `reason` 逻辑；
- `docs/README.md`、`docs/RUNBOOK.md`：项目概览与运维说明。

欢迎在模型迭代后更新文档，保持与代码一致。祝使用顺利！ 
