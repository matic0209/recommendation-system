---
name: m-implement-popular-quality-filter
branch: feature/popular-quality-filter
status: in-progress
created: 2025-12-28
---

# 为Popular召回榜单实现质量过滤机制

## Problem/Goal

当前Popular召回策略（`pipeline/train_models.py`中的`build_popular_items()`函数）直接按交互权重排序取top 50，没有任何质量控制。这导致榜单中可能包含：
- 低价商品（< 0.5元）
- 无互动或低互动商品（< 10次）
- 长期不活跃商品（超过2年未购买）

本任务目标是在训练阶段添加质量过滤机制，确保Popular榜单仅包含高质量item，并通过Sentry监控榜单质量。

## Success Criteria
- [ ] `build_popular_items()`函数成功添加质量过滤逻辑（宽松过滤：价格≥0.5 且 互动≥10 且 730天内活跃）
- [ ] 过滤阈值通过环境变量可配置（POPULAR_MIN_PRICE、POPULAR_MIN_INTERACTION、POPULAR_MAX_INACTIVE_DAYS）
- [ ] 添加Sentry告警：当过滤比例>70%、平均价格<1.0元、或数量不足时自动告警
- [ ] 详细日志记录过滤统计（各维度过滤数量、质量指标）
- [ ] 向后兼容：特征缺失时自动降级，支持快速禁用过滤（POPULAR_ENABLE_FILTER=false）
- [ ] 运行训练流程验证功能正常，使用analyze_popular_quality.py验证过滤效果

## Context Manifest

### 核心业务流程：Popular召回榜单的生成与使用

**当前工作原理（无质量控制）：**

在训练阶段（`pipeline/train_models.py`的`main()`函数，第1331行），系统通过以下流程生成Popular召回榜单：

1. **数据加载**（第1340-1375行）：
   - 加载`data/processed/interactions.parquet`：包含用户-商品交互记录（columns: user_id, dataset_id, weight, last_event_time）
   - 加载`data/processed/dataset_features.parquet`：包含商品的全部特征，包括质量字段（price, interaction_count, days_since_last_purchase等36个特征）
   - 加载`data/processed/dataset_stats.parquet`：包含统计指标（interaction_count, total_weight, last_event_time）
   - 注意：dataset_features已经通过`build_features_v2.py`将stats合并进去，所以interaction_count等字段直接在features中可用

2. **Popular榜单构建**（第1399-1400行）：
   ```python
   LOGGER.info("Building popular items list...")
   popular_items = build_popular_items(interactions)
   ```

3. **`build_popular_items()`函数当前实现**（第319-323行）：
   ```python
   def build_popular_items(interactions: pd.DataFrame, top_k: int = 50) -> List[int]:
       if interactions.empty:
           return []
       counts = interactions.groupby("dataset_id")["weight"].sum().sort_values(ascending=False)
       return counts.head(top_k).index.astype(int).tolist()
   ```

   **问题**：仅按交互权重排序，完全没有质量控制！这导致低价（0.1元）、低互动（0-2次）、长期不活跃（575天）的商品也能进入榜单。

4. **榜单保存**（第1477-1483行）：
   ```python
   popular_path = MODELS_DIR / "top_items.json"
   save_json(popular_items, popular_path)  # 保存为JSON列表：[13737, 13638, ...]
   ```

5. **在线服务使用**（`app/main.py`第1695-1753行）：
   - 加载`models/top_items.json`作为Popular召回源
   - 当前已有**运行时过滤**（第1716-1734行），但训练阶段完全没有过滤
   - 运行时过滤规则：低价低互动（price < 1.90 AND interaction < 66）或长期不活跃低互动（days_inactive > 180 AND interaction < 30）
   - 运行时过滤的问题：如果训练时榜单全是低质量item，运行时过滤后可能只剩几个item，导致召回不足

**为什么需要训练阶段过滤：**

分析当前数据（通过`analyze_popular_quality.py`）发现：
- Popular榜单中包含价格低至0.1元的商品
- 部分商品互动量为0-2次（远低于平均值）
- 部分商品超过500天未有购买记录
- 如果50个Popular item中有30个被运行时过滤掉，召回质量和多样性都会受损

### 实现方案：在训练阶段添加质量过滤

**关键修改点：**

1. **`build_popular_items()`函数重写**（第319-323行）：
   - 需要传入`dataset_features`参数获取质量字段
   - 在排序后、取top_k前，应用质量过滤
   - 过滤后如果数量不足，继续扩展直到满足top_k或候选池耗尽

2. **调用点修改**（第1399-1400行）：
   - 传入`dataset_features`参数
   - 传入配置参数（从环境变量读取）

3. **质量过滤逻辑设计**：
   - 过滤条件（宽松）：price >= 0.5 AND interaction_count >= 10 AND days_since_last_purchase <= 730
   - 环境变量配置：
     - `POPULAR_MIN_PRICE`（默认0.5）
     - `POPULAR_MIN_INTERACTION`（默认10）
     - `POPULAR_MAX_INACTIVE_DAYS`（默认730）
     - `POPULAR_ENABLE_FILTER`（默认true，可设为false快速禁用）

4. **向后兼容降级策略**：
   - 如果`dataset_features`为空或缺失质量字段，自动降级到原逻辑（仅按weight排序）
   - 如果过滤后数量严重不足（<10），发送Sentry告警但仍返回过滤后的结果
   - 通过环境变量`POPULAR_ENABLE_FILTER=false`可快速禁用过滤

5. **Sentry告警机制**（使用`track_data_quality_issue`）：
   - 告警场景1：过滤比例>70%（说明Popular池质量太差）
   - 告警场景2：过滤后平均价格<1.0元（说明过滤阈值可能太宽松）
   - 告警场景3：过滤后数量<10（说明过滤太严格或数据质量问题）
   - 调用示例（参考`pipeline/sentry_utils.py`第110-169行）：
     ```python
     from pipeline.sentry_utils import track_data_quality_issue

     track_data_quality_issue(
         check_name="popular_quality_filter",
         severity="warning",  # or "critical"
         details={
             "total_candidates": total_count,
             "filtered_count": filtered_count,
             "kept_count": kept_count,
             "avg_price": avg_price,
         },
         metric_value=filter_ratio,
         threshold=0.7,
     )
     ```

6. **详细日志记录**（参考现有日志模式）：
   - 使用`LOGGER.info()`记录过滤统计
   - 记录内容：总候选数、过滤数、保留数、各维度过滤明细、质量指标（平均价格、平均互动、平均活跃度）
   - 参考现有日志模式（如第1388-1400行）

### 技术实现细节

#### 数据结构和字段映射

**`dataset_features` DataFrame结构**（来自`data/processed/dataset_features.parquet`）：
```python
# 关键质量字段（已验证存在）：
- dataset_id: int - 商品ID
- price: float - 价格（元）
- interaction_count: float - 互动次数（来自stats合并）
- days_since_last_purchase: float - 距上次购买天数
- total_weight: float - 总交互权重
- popularity_score: float - 人气分数（log1p(interaction_count)）
- freshness_score: float - 新鲜度分数（1/(days_since_last_purchase + 1)）

# 其他可用字段（36个特征总计）：
- description_length, tag_count, word_count
- popularity_rank, popularity_percentile
- image_count, has_images, has_cover
- 等等...
```

**`interactions` DataFrame结构**：
```python
- user_id: int
- dataset_id: int
- weight: float - 交互权重（范围0.095-3.687，均值0.78）
- last_event_time: datetime
```

**当前Popular榜单格式**（`models/top_items.json`）：
```json
[13737, 13638, 13830, ...]  // 简单的int列表，长度50
```

#### 函数签名设计

```python
def build_popular_items(
    interactions: pd.DataFrame,
    top_k: int = 50,
    dataset_features: Optional[pd.DataFrame] = None,
    min_price: float = 0.5,
    min_interaction: int = 10,
    max_inactive_days: int = 730,
    enable_filter: bool = True,
) -> List[int]:
    """
    Build popular items list with quality filtering.

    Args:
        interactions: User-item interactions with weight column
        top_k: Target number of items to return
        dataset_features: Dataset features including quality fields
        min_price: Minimum price threshold (yuan)
        min_interaction: Minimum interaction count
        max_inactive_days: Maximum days since last purchase
        enable_filter: Enable quality filtering (disable for backward compatibility)

    Returns:
        List of dataset_ids (integers)
    """
```

#### 环境变量读取模式

参考现有模式（第85-87行、第1046-1048行）：
```python
# 在函数内或调用点读取环境变量
min_price = float(os.getenv("POPULAR_MIN_PRICE", "0.5"))
min_interaction = int(os.getenv("POPULAR_MIN_INTERACTION", "10"))
max_inactive_days = int(os.getenv("POPULAR_MAX_INACTIVE_DAYS", "730"))
enable_filter = os.getenv("POPULAR_ENABLE_FILTER", "true").lower() == "true"
```

#### 过滤逻辑伪代码

```python
# 1. 按weight排序获取候选池（扩大到top_k*3以应对过滤）
candidates = interactions.groupby("dataset_id")["weight"].sum().sort_values(ascending=False)
candidate_ids = candidates.head(top_k * 3).index.tolist()

# 2. 如果features可用且过滤启用，应用质量过滤
if enable_filter and dataset_features is not None:
    # 获取候选item的质量特征
    quality_features = dataset_features[dataset_features['dataset_id'].isin(candidate_ids)]

    # 应用过滤条件
    mask = (
        (quality_features['price'] >= min_price) &
        (quality_features['interaction_count'] >= min_interaction) &
        (quality_features['days_since_last_purchase'] <= max_inactive_days)
    )

    filtered_ids = quality_features[mask]['dataset_id'].tolist()

    # 按原始weight顺序保留
    result = [id for id in candidate_ids if id in filtered_ids][:top_k]

    # 记录统计和告警
    filter_ratio = 1 - len(filtered_ids) / len(candidate_ids)
    if filter_ratio > 0.7:
        track_data_quality_issue(...)
else:
    # 降级：直接返回top_k
    result = candidate_ids[:top_k]

return result
```

#### Sentry告警集成

导入（文件顶部已有，第77行）：
```python
from pipeline.sentry_utils import init_pipeline_sentry, monitor_pipeline_step
```

新增导入：
```python
from pipeline.sentry_utils import track_data_quality_issue
```

调用位置：在`build_popular_items()`函数内，过滤完成后
```python
# 计算质量指标
avg_price = quality_features[mask]['price'].mean() if len(filtered_ids) > 0 else 0
filter_ratio = (len(candidate_ids) - len(filtered_ids)) / len(candidate_ids)

# 告警条件1：过滤比例>70%
if filter_ratio > 0.7:
    track_data_quality_issue(
        check_name="popular_filter_ratio_high",
        severity="warning",
        details={
            "total_candidates": len(candidate_ids),
            "filtered_count": len(candidate_ids) - len(filtered_ids),
            "kept_count": len(filtered_ids),
            "filter_ratio": filter_ratio,
        },
        metric_value=filter_ratio,
        threshold=0.7,
    )

# 告警条件2：平均价格<1.0
if avg_price < 1.0 and len(filtered_ids) > 0:
    track_data_quality_issue(
        check_name="popular_avg_price_low",
        severity="warning",
        details={
            "avg_price": avg_price,
            "item_count": len(filtered_ids),
        },
        metric_value=avg_price,
        threshold=1.0,
    )

# 告警条件3：数量不足
if len(filtered_ids) < 10:
    track_data_quality_issue(
        check_name="popular_count_insufficient",
        severity="critical" if len(filtered_ids) < 5 else "warning",
        details={
            "kept_count": len(filtered_ids),
            "target_count": top_k,
            "filters": {
                "min_price": min_price,
                "min_interaction": min_interaction,
                "max_inactive_days": max_inactive_days,
            },
        },
        metric_value=len(filtered_ids),
        threshold=10,
    )
```

#### 日志记录模式

参考现有日志（第1388-1400行）：
```python
LOGGER.info("Building popular items list with quality filter...")
LOGGER.info(
    "Popular filter config: min_price=%.2f, min_interaction=%d, max_inactive_days=%d, enabled=%s",
    min_price, min_interaction, max_inactive_days, enable_filter
)

# 过滤后
LOGGER.info(
    "Popular filtering results: total=%d, filtered_out=%d, kept=%d (%.1f%%), "
    "avg_price=%.2f, avg_interaction=%.1f, avg_inactive_days=%.1f",
    total_count,
    filtered_count,
    kept_count,
    (kept_count / total_count * 100) if total_count > 0 else 0,
    avg_price,
    avg_interaction,
    avg_inactive_days,
)

# 详细维度统计
LOGGER.info(
    "Filter breakdown: price_filtered=%d, interaction_filtered=%d, inactive_filtered=%d",
    price_filtered_count,
    interaction_filtered_count,
    inactive_filtered_count,
)
```

### 文件位置和修改清单

#### 1. 核心修改文件

**`/home/ubuntu/recommend/pipeline/train_models.py`**

- **修改点1**：`build_popular_items()`函数（第319-323行）
  - 当前：5行简单实现
  - 修改后：约80-100行（包含过滤逻辑、统计计算、告警、日志）
  - 新增导入：`track_data_quality_issue`

- **修改点2**：调用点（第1399-1400行）
  - 当前：`popular_items = build_popular_items(interactions)`
  - 修改后：
    ```python
    popular_items = build_popular_items(
        interactions,
        dataset_features=dataset_features,
        min_price=float(os.getenv("POPULAR_MIN_PRICE", "0.5")),
        min_interaction=int(os.getenv("POPULAR_MIN_INTERACTION", "10")),
        max_inactive_days=int(os.getenv("POPULAR_MAX_INACTIVE_DAYS", "730")),
        enable_filter=os.getenv("POPULAR_ENABLE_FILTER", "true").lower() == "true",
    )
    ```

#### 2. 配置文件

**`/home/ubuntu/recommend/.env`**（可选，用于本地测试）

新增环境变量（在第140行后添加）：
```bash
# Popular召回质量过滤配置
POPULAR_MIN_PRICE=0.5
POPULAR_MIN_INTERACTION=10
POPULAR_MAX_INACTIVE_DAYS=730
POPULAR_ENABLE_FILTER=true
```

#### 3. 验证工具

**`/home/ubuntu/recommend/analyze_popular_quality.py`**（已存在）

- 用途：分析Popular榜单质量
- 运行：`python analyze_popular_quality.py`
- 输出：质量统计、低质量item识别、过滤建议

### 数据流和依赖关系

```
训练流程（pipeline/train_models.py::main）:
  1. 加载数据（第1340-1375行）
     ├─ interactions.parquet (user-item交互)
     ├─ dataset_features.parquet (商品特征，包含质量字段)
     └─ dataset_stats.parquet (统计指标，已合并到features)

  2. 训练模型（第1388-1400行）
     ├─ 行为相似度模型
     ├─ 内容相似度模型
     ├─ 向量召回模型
     └─ Popular召回榜单 ← 【本任务修改点】

  3. 保存模型（第1475-1484行）
     └─ models/top_items.json ← 【输出文件】

在线服务（app/main.py）:
  1. 启动时加载模型（第1695行）
     └─ 读取models/top_items.json

  2. 请求时使用Popular召回（第1695-1753行）
     ├─ 批量查询features（第1699-1709行）
     ├─ 运行时过滤（第1716-1734行）【已存在，作为二次保障】
     └─ 计算分数和融合（第1737-1753行）
```

### 错误处理和边界情况

**必须处理的边界情况：**

1. **dataset_features为None或空**：
   - 降级到原逻辑，仅按weight排序
   - 记录WARNING日志

2. **质量字段缺失**（price/interaction_count/days_since_last_purchase不存在）：
   - 尝试从可用字段过滤，缺失字段跳过该维度
   - 如果全部缺失，降级到原逻辑
   - 记录WARNING日志

3. **过滤后数量为0**：
   - 不返回空列表，而是放宽条件重试（如只过滤price）
   - 如果仍为空，降级到原逻辑
   - 发送CRITICAL级别Sentry告警

4. **数据类型异常**（price不是数字等）：
   - try-except包裹过滤逻辑
   - 异常时降级到原逻辑
   - 记录ERROR日志和Sentry异常

5. **interactions为空**：
   - 直接返回空列表（原逻辑已处理）

**错误处理模式：**

```python
try:
    # 质量过滤逻辑
    if dataset_features is None or dataset_features.empty:
        LOGGER.warning("dataset_features not available, falling back to weight-only ranking")
        return candidate_ids[:top_k]

    # 检查必需字段
    required_cols = ['dataset_id', 'price', 'interaction_count', 'days_since_last_purchase']
    missing_cols = [col for col in required_cols if col not in dataset_features.columns]
    if missing_cols:
        LOGGER.warning("Missing quality columns: %s, falling back to weight-only ranking", missing_cols)
        return candidate_ids[:top_k]

    # 执行过滤...

except Exception as e:
    LOGGER.error("Popular quality filter failed: %s, falling back to weight-only ranking", e)
    # 可选：发送Sentry异常
    return candidate_ids[:top_k]
```

### 测试验证计划

1. **单元测试点**（可选，不强制）：
   - 测试过滤逻辑正确性
   - 测试降级场景
   - 测试边界情况

2. **集成测试**（必需）：
   - 运行完整训练流程：`python -m pipeline.train_models`
   - 验证`models/top_items.json`生成成功
   - 检查日志输出是否包含过滤统计
   - 验证Sentry告警是否按预期发送（如果触发条件）

3. **质量验证**（必需）：
   - 运行`python analyze_popular_quality.py`
   - 对比过滤前后的质量指标：
     - 平均价格提升
     - 平均互动量提升
     - 平均活跃度提升
   - 确保过滤后仍有足够数量（至少40个，理想50个）

4. **在线服务验证**（可选）：
   - 重启API服务加载新模型
   - 调用推荐接口观察Popular召回质量
   - 检查运行时过滤统计（应该显著减少）

### 相关代码模式和约定

**项目代码规范**（参考`CLAUDE.md`）：
- Python 3.9+兼容
- 类型提示：函数参数和返回值都需要
- 日志：使用`LOGGER`（已在文件顶部定义，第14行）
- 错误处理：使用具体异常类型，避免bare except
- 文档字符串：所有函数都需要docstring

**现有模式参考**：
- 环境变量读取：参考第85-87行、第1046-1048行
- DataFrame操作：参考第319-323行、第614-680行
- 日志记录：参考第1388-1400行
- 错误处理：参考第490-499行
- Sentry集成：参考`pipeline/sentry_utils.py`第110-169行

**命名约定**：
- 函数：snake_case（如`build_popular_items`）
- 变量：snake_case（如`filter_ratio`）
- 常量：UPPERCASE（如`POPULAR_MIN_PRICE`）
- 类型提示：使用`Optional[X]`表示可选参数

### 性能考虑

- **数据量**：当前interactions约7285条，dataset_features约1000+条，过滤操作很轻量
- **内存**：候选池扩大到top_k*3（150条），内存开销可忽略
- **计算复杂度**：O(n log n)排序 + O(n)过滤，总体很快（<1秒）
- **优化点**：批量查询features（使用`.isin()`而非循环），参考app/main.py第1702-1707行的模式

## User Notes
<!-- Any specific notes or requirements from the developer -->

## Work Log
- [2025-12-28] 任务启动：创建feature分支，开始实施Popular质量过滤
