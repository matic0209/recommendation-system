# 内存优化指南

## 问题描述

之前的 DAG 运行时会占满内存导致服务器关机，主要原因包括：

1. **相似度矩阵计算**：在 `train_models.py` 中计算 N×N 的完整相似度矩阵，对于大量数据集会占用数百 MB 甚至 GB 内存
2. **DataFrame 内存效率低**：未优化数据类型，使用默认的 int64/float64
3. **缺少显式内存释放**：大对象使用后未及时删除和触发垃圾回收
4. **并发执行**：DAG 某些步骤可能并发导致内存峰值

## 优化措施

### 1. 分批计算相似度矩阵（最重要）

**文件**: `pipeline/memory_optimizer.py`

新增 `compute_similarity_in_batches()` 函数，将相似度计算改为分批处理：

- **之前**: 一次性计算 N×N 矩阵（例如 10,000 个数据集 = 400MB）
- **现在**: 每次处理 1,000 个数据集，只保留 top-K 相似项
- **内存节省**: ~80-90%

```python
# 使用示例
similarity_dict = compute_similarity_in_batches(
    text_matrix,
    batch_size=1000,  # 可通过环境变量 SIMILARITY_BATCH_SIZE 调整
    top_k=200,        # 可通过环境变量 SIMILARITY_TOP_K 调整
)
```

### 2. DataFrame 内存优化

**文件**: `pipeline/memory_optimizer.py`

新增 `optimize_dataframe_memory()` 函数，自动将数据类型降级：

- **int64** → **int8/int16/int32**（根据实际范围）
- **float64** → **float32**
- **内存节省**: 通常 50-70%

```python
# 使用示例
df = optimize_dataframe_memory(df)
# 输出: Memory optimization: 100.00 MB -> 35.23 MB (64.8% reduction)
```

### 3. 显式内存释放

在关键步骤后添加：

```python
# 删除不再需要的大对象
del large_dataframe, similarity_matrix

# 触发垃圾回收
reduce_memory_usage()
```

### 4. 多模态相似度优化

**文件**: `pipeline/memory_optimizer.py`

新增 `build_sparse_similarity_matrix()` 函数，支持文本+图像的多模态相似度计算：

- 分批处理文本和图像相似度
- 只在批次级别合并，避免创建完整矩阵
- 只保留 top-K 结果

### 5. 修改后的文件

已优化的文件：

1. **pipeline/memory_optimizer.py** (新文件)
   - `optimize_dataframe_memory()`: DataFrame 内存优化
   - `compute_similarity_in_batches()`: 分批相似度计算
   - `build_sparse_similarity_matrix()`: 多模态相似度
   - `reduce_memory_usage()`: 显式垃圾回收

2. **pipeline/train_models.py**
   - 修改 `train_content_similarity()` 使用批处理
   - 在 `main()` 中添加 DataFrame 优化
   - 在模型训练步骤间添加内存清理

3. **pipeline/build_features.py**
   - 在 `main()` 中添加 DataFrame 优化
   - 在特征构建步骤间添加内存清理

### 6. 行为召回邻居裁剪

**文件**: `pipeline/train_models.py`

- 新增 `BEHAVIOR_SIMILARITY_TOP_K` 环境变量限制行为相似度召回的邻居数量
- 默认值与 `SIMILARITY_TOP_K` 一致（200），可按需降低到 50/100
- 只保留高频共现的 top-K 邻居，字典体积大幅下降，常规生产数据能再节省 50%+ 的内存

## 配置参数

通过环境变量控制内存优化行为：

```bash
# 加载配置
source .env.memory

# 或在 docker-compose.yml 中添加:
environment:
  - SIMILARITY_BATCH_SIZE=1000
  - SIMILARITY_TOP_K=200
  - USE_FAISS_RECALL=1
```

### 参数说明

| 参数 | 默认值 | 说明 | 调整建议 |
|------|--------|------|----------|
| `SIMILARITY_BATCH_SIZE` | 1000 | 相似度计算批次大小 | 内存不足时减少到 500 或 250 |
| `SIMILARITY_TOP_K` | 200 | 保留 top-K 相似项 | 召回需求低时可减少到 100 |
| `BEHAVIOR_SIMILARITY_TOP_K` | 继承 `SIMILARITY_TOP_K` | 用户行为召回每个数据集保留的相似邻居数 | 生产内存紧张时可调到 50~100 |
| `USE_FAISS_RECALL` | 1 | 是否使用 Faiss 向量召回 | 内存受限时可设为 0 |

## 效果预估

### 内存使用对比（10,000 个数据集示例）

| 组件 | 优化前 | 优化后 | 节省 |
|------|--------|--------|------|
| 相似度矩阵 | ~800 MB | ~80 MB | 90% |
| DataFrame (interactions) | ~50 MB | ~17 MB | 66% |
| DataFrame (features) | ~150 MB | ~52 MB | 65% |
| **总计** | **~1 GB** | **~150 MB** | **85%** |

### 运行时间影响

- **相似度计算**: 增加约 10-15%（分批处理开销）
- **DataFrame 优化**: 几乎无影响（<1%）
- **整体 DAG**: 增加约 5-10%

**性能/内存权衡**: 运行时间略微增加，但避免 OOM，总体更可靠。

## 监控和调优

### 查看内存使用

在 DAG 运行时监控内存：

```bash
# 实时监控
watch -n 1 free -h

# 查看 Python 进程内存
ps aux | grep python | awk '{print $6/1024 "MB", $11}'
```

### 日志检查

优化后的代码会输出内存使用日志：

```
INFO Memory optimization: 100.00 MB -> 35.23 MB (64.8% reduction)
INFO Computing similarity in batches (total=5000, batch_size=1000, top_k=200)
INFO Progress: 10/10 batches (100.0%)
INFO Memory usage: 450.2 MB -> 180.5 MB (freed 269.7 MB)
```

### 调优步骤

如果仍然遇到内存问题：

1. **减少批次大小**:
   ```bash
   export SIMILARITY_BATCH_SIZE=500
   ```

2. **减少 top-K**:
   ```bash
   export SIMILARITY_TOP_K=100
   ```

3. **禁用 Faiss**:
   ```bash
   export USE_FAISS_RECALL=0
   ```

4. **限制 Airflow 并发**:
   在 `airflow.cfg` 中设置:
   ```ini
   [core]
   parallelism = 2
   dag_concurrency = 1
   max_active_runs_per_dag = 1
   ```

## 其他建议

### 1. 增加 Swap 空间（临时措施）

如果物理内存仍不足：

```bash
# 创建 4GB swap
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# 永久生效
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

### 2. 错峰运行 DAG

修改 DAG schedule，避免并发：

```python
# recommendation_pipeline.py
schedule_interval="0 18 * * *"  # 每天 18:00 UTC

# weekly_model_refresh.py
schedule_interval="0 12 * * 6"  # 周六 12:00 UTC (避开日常任务)

# incremental_data_update.py
schedule_interval="0 */4 * * *"  # 每4小时 (错开整点)
```

### 3. 使用增量处理

对于增量更新 DAG，只处理新数据：

```python
# 只加载最近 N 天的数据
recent_data = df[df['create_time'] > (datetime.now() - timedelta(days=7))]
```

## 验证优化效果

运行测试验证内存优化：

```bash
# 测试 train_models
python -m pipeline.train_models

# 测试 build_features
python -m pipeline.build_features

# 监控内存峰值
/usr/bin/time -v python -m pipeline.train_models 2>&1 | grep "Maximum resident"
```

## 回滚方案

如果优化导致问题，可以临时回退：

```bash
# 备份优化后的文件
cp pipeline/train_models.py pipeline/train_models.py.optimized
cp pipeline/build_features.py pipeline/build_features.py.optimized

# 从 git 恢复原版本
git checkout HEAD~1 -- pipeline/train_models.py pipeline/build_features.py

# 或使用原始相似度计算（在代码中设置标志）
export USE_LEGACY_SIMILARITY=1
```

## 总结

通过以上优化措施，内存使用应该减少 **80-90%**，避免服务器因 OOM 关机。关键优化点：

1. ✅ **分批相似度计算** - 最重要的优化
2. ✅ **DataFrame 类型优化** - 简单但有效
3. ✅ **显式内存释放** - 避免内存累积
4. ✅ **可配置参数** - 灵活调整

如有问题，请查看日志中的内存使用信息，根据实际情况调整参数。
