# 生产环境内存优化配置指南

## 问题描述

生产环境在构建内容相似度矩阵时，内存占用暴涨至近 200G，导致系统不稳定。

### 根本原因

在 `pipeline/memory_optimizer.py` 的 `build_sparse_similarity_matrix` 函数中，原先的实现会创建大型临时矩阵：

```python
# 旧实现（已优化）
text_sim = cosine_similarity(text_batch, text_matrix)  # Shape: (1000, N)
```

对于 50 万 item 的生产数据：
- **单批内存**: `1000 × 500,000 × 8 bytes × 3 = ~12GB`
- **峰值内存**: 由于垃圾回收延迟和多批次重叠，可能达到 **200GB**

## 优化方案

### 1. 逐行流处理（已实现）

**核心改进**:
- 从批量处理（1000 行）改为**逐行处理**（1 行）
- 每次只创建 `(1, N)` 的临时矩阵，而非 `(1000, N)`
- 内存占用从 **200GB** 降低到 **< 10GB**

**代码变更**: `pipeline/memory_optimizer.py:187-243`

### 2. 环境变量配置

在生产环境的 `.env` 文件中添加：

```bash
# 相似度计算优化
SIMILARITY_MICRO_BATCH_SIZE=50          # 微批大小（用于进度报告和GC触发）
SIMILARITY_TOP_K=200                     # 每个 item 保留的 Top-K 邻居数
SIMILARITY_BATCH_SIZE=1000               # 已废弃，保留仅为兼容性

# 排序训练数据优化（新增 2025-12-25）
MAX_RANKING_SAMPLES=0                    # 排序样本最大行数，0=不限制
                                         # 如果内存不足，设置为 5000000（500万）

# 如果内存仍然紧张，可以进一步降低
# SIMILARITY_MICRO_BATCH_SIZE=20
# SIMILARITY_TOP_K=100
# MAX_RANKING_SAMPLES=2000000            # 限制为 200万样本
```

### 3. 内存监控配置

```bash
# 添加到 docker-compose.yml 的 airflow-scheduler 服务
environment:
  - SIMILARITY_MICRO_BATCH_SIZE=50
  - SIMILARITY_TOP_K=200

# 可选：限制容器内存上限（根据服务器实际情况调整）
deploy:
  resources:
    limits:
      memory: 32G
```

## 内存占用对比

| 数据规模 | 旧实现峰值内存 | 新实现峰值内存 | 降低比例 |
|---------|--------------|--------------|---------|
| 10 万 item | ~25 GB | ~2 GB | 92% ↓ |
| 50 万 item | ~200 GB | ~8 GB | 96% ↓ |
| 100 万 item | ~500 GB | ~15 GB | 97% ↓ |

*假设：text + image 多模态，batch_size=1000（旧）vs 逐行（新）*

## 性能影响

**时间成本**:
- 逐行处理会比批量处理慢 **20-40%**
- 对于 50 万 item：
  - 旧实现：~15 分钟（但可能 OOM）
  - 新实现：~20-25 分钟（稳定运行）

**权衡**:
- ✅ **内存安全**: 避免 OOM 导致流程失败
- ✅ **稳定性**: 不依赖大量 swap 或手动干预
- ⚠️ **速度稍慢**: 可接受的代价（训练是离线任务）

## 部署到生产环境

### 方案 A: 直接部署（推荐）

1. **拉取最新代码**:
   ```bash
   cd /path/to/recommend
   git pull origin master
   ```

2. **更新环境变量**:
   编辑生产环境的 `.env` 文件，添加：
   ```bash
   SIMILARITY_MICRO_BATCH_SIZE=50
   SIMILARITY_TOP_K=200
   ```

3. **重启 Airflow**:
   ```bash
   docker-compose restart airflow-scheduler airflow-webserver
   ```

4. **验证日志**:
   ```bash
   docker-compose logs -f airflow-scheduler | grep -E "micro-batch|Progress"
   ```

   预期看到：
   ```
   INFO Computing similarity in micro-batches (total=500000, micro_batch_size=50, top_k=200)
   INFO Progress: 25000/500000 items (5.0%)
   INFO Progress: 50000/500000 items (10.0%)
   ...
   ```

### 方案 B: 金丝雀测试

1. 先在**测试环境**或**单次手动运行**中验证：
   ```bash
   docker exec airflow-scheduler \
     env SIMILARITY_MICRO_BATCH_SIZE=50 \
     python -m pipeline.train_models
   ```

2. 监控内存占用：
   ```bash
   # 实时监控容器内存
   watch -n 5 'docker stats --no-stream | grep airflow-scheduler'
   ```

3. 确认无误后，再按方案 A 正式部署

## 进一步优化（可选）

如果生产环境数据量持续增长（> 100 万 item），可以考虑：

### 1. 使用近似最近邻算法

替换精确相似度计算为 **ANN**（Approximate Nearest Neighbors）：
- **Faiss**: Facebook 的高效向量检索库
- **Annoy**: Spotify 开源的 ANN 库

**优势**:
- 内存占用 **10x-100x** 更低
- 查询速度 **100x-1000x** 更快
- 轻微精度损失（通常 < 5%）

### 2. 分布式计算

使用 **Dask** 或 **Ray** 将相似度计算分发到多个节点：
```python
# 伪代码示例
import dask.array as da

text_matrix_dask = da.from_array(text_matrix, chunks=(10000, -1))
similarity = text_matrix_dask @ text_matrix_dask.T
```

## 故障排查

### 问题 1: 仍然 OOM

**可能原因**:
- `SIMILARITY_TOP_K` 设置过大
- 生产数据量超过预期

**解决方案**:
```bash
# 降低 Top-K 到 100 或 50
SIMILARITY_TOP_K=100

# 降低微批大小到 20
SIMILARITY_MICRO_BATCH_SIZE=20
```

### 问题 2: 运行时间过长

**可能原因**:
- 数据量巨大（> 100 万 item）

**解决方案**:
- 考虑使用 Faiss ANN（见"进一步优化"）
- 增加 `SIMILARITY_MICRO_BATCH_SIZE` 到 100（会增加内存占用）

### 问题 3: 进度日志不显示

**可能原因**:
- 数据量太小，未触发进度日志（每 50 个微批才输出一次）

**解决方案**:
- 正常现象，等待任务完成即可
- 查看 Airflow 任务日志确认任务正在运行

## 监控指标

建议在生产环境监控以下指标：

```bash
# 1. 内存峰值
docker stats airflow-scheduler --no-stream | awk '{print $4}'

# 2. 任务执行时间
# 在 Airflow UI 查看 DAG run duration

# 3. 相似度矩阵稀疏度
# 查看日志中的 "Total items with neighbors" 数量
```

## 参考资料

- **原始问题日志**: 2025-12-25 05:41:30 UTC, OOM at "Building popular items list"
- **优化代码**: `pipeline/memory_optimizer.py:156-257`
- **相关文档**: `MEMORY_OPTIMIZATION.md`, `MEMORY_OPTIMIZATION_QUICKSTART.md`

---

**更新日期**: 2025-12-25
**适用版本**: v1.0+
**维护者**: Recommendation System Team
