# 生产环境内存优化部署指南

## 📋 概述

本指南说明如何在生产环境中部署内存优化配置。

## 🔧 需要修改的文件

### 1. `.env` 文件（或 `.env.prod`）

在你的 `.env` 文件末尾添加以下内存优化配置：

```bash
# ============================================
# 内存优化配置 (Memory Optimization)
# ============================================
# 相似度计算批次大小（越小越省内存，但速度越慢）
# 推荐值：1000（默认），内存不足时降到 500
SIMILARITY_BATCH_SIZE=1000

# 每个数据集保留的 top-K 相似项数量
# 推荐值：200（默认），召回需求低时可降到 100
SIMILARITY_TOP_K=200

# 是否启用 Faiss 向量召回（可能占用较多内存）
# 推荐值：1（启用），内存受限时设为 0
USE_FAISS_RECALL=1

# 排序模型 CVR 权重
RANKING_CVR_WEIGHT=0.5

# Python 内存管理优化
PYTHONHASHSEED=0
MALLOC_TRIM_THRESHOLD_=100000
```

### 2. `docker-compose.yml` 文件（可选但推荐）

在 Airflow 相关服务中添加内存限制，防止单个服务占用过多内存。

找到以下服务配置，在每个服务下添加 `deploy` 部分：

#### 2.1 修改 `airflow-scheduler` 服务

在 `airflow-scheduler` 服务配置中添加（大约在第 208-247 行）：

```yaml
  airflow-scheduler:
    build:
      context: .
      dockerfile: Dockerfile.airflow
    image: recommend-airflow:latest
    container_name: airflow-scheduler
    restart: unless-stopped
    # 添加内存限制 ⬇️
    deploy:
      resources:
        limits:
          memory: 8G        # 最大使用 8GB 内存（根据服务器总内存调整）
        reservations:
          memory: 2G        # 预留 2GB 内存
    # ... 其他配置保持不变
```

#### 2.2 修改 `airflow-webserver` 服务

在 `airflow-webserver` 服务配置中添加（大约在第 163-206 行）：

```yaml
  airflow-webserver:
    build:
      context: .
      dockerfile: Dockerfile.airflow
    image: recommend-airflow:latest
    container_name: airflow-webserver
    restart: unless-stopped
    # 添加内存限制 ⬇️
    deploy:
      resources:
        limits:
          memory: 4G        # 最大使用 4GB 内存
        reservations:
          memory: 1G        # 预留 1GB 内存
    # ... 其他配置保持不变
```

#### 2.3 修改 `recommendation-api` 服务

在 `recommendation-api` 服务配置中添加（大约在第 37-81 行）：

```yaml
  recommendation-api:
    build:
      context: .
      dockerfile: Dockerfile
    image: recommend-recommendation-api
    container_name: recommendation-api
    # 添加内存限制 ⬇️
    deploy:
      resources:
        limits:
          memory: 6G        # 最大使用 6GB 内存
        reservations:
          memory: 2G        # 预留 2GB 内存
    # ... 其他配置保持不变
```

## 📊 内存分配建议（基于 62GB 总内存）

| 服务 | 内存限制 | 说明 |
|------|---------|------|
| airflow-scheduler | 8GB | DAG 执行主要在这里，需要较多内存 |
| airflow-webserver | 4GB | Web UI，内存需求较少 |
| recommendation-api | 6GB | 推荐服务，模型加载和推理 |
| redis | 2GB | 已在 docker-compose.yml 配置 |
| mlflow | 2GB | 实验跟踪 |
| postgres-airflow | 2GB | Airflow 元数据库 |
| 其他服务 | 2GB | Prometheus, Grafana 等 |
| **系统保留** | 36GB | 留给操作系统和缓存 |

**注意**: 以上是建议值，可根据实际运行情况调整。

## 🚀 部署步骤

### 方案 A: 使用测试环境配置（推荐）

```bash
# 1. 备份当前 .env
cp .env .env.backup_$(date +%Y%m%d_%H%M%S)

# 2. 在 .env 末尾添加内存优化配置
cat >> .env << 'EOF'

# ============================================
# 内存优化配置 (Memory Optimization)
# ============================================
SIMILARITY_BATCH_SIZE=1000
SIMILARITY_TOP_K=200
USE_FAISS_RECALL=1
RANKING_CVR_WEIGHT=0.5
PYTHONHASHSEED=0
MALLOC_TRIM_THRESHOLD_=100000
EOF

# 3. 验证配置已添加
tail -15 .env

# 4. 重启服务
docker-compose down
docker-compose up -d

# 5. 查看日志确认优化生效
docker-compose logs -f airflow-scheduler | grep -i "memory"
```

### 方案 B: 使用生产环境配置

```bash
# 1. 备份生产环境配置
cp .env.prod .env.prod.backup_$(date +%Y%m%d_%H%M%S)

# 2. 在 .env.prod 末尾添加内存优化配置
cat >> .env.prod << 'EOF'

# ============================================
# 内存优化配置 (Memory Optimization)
# ============================================
SIMILARITY_BATCH_SIZE=1000
SIMILARITY_TOP_K=200
USE_FAISS_RECALL=1
RANKING_CVR_WEIGHT=0.5
PYTHONHASHSEED=0
MALLOC_TRIM_THRESHOLD_=100000
EOF

# 3. 切换到生产环境配置
cp .env .env.testing_backup
cp .env.prod .env

# 4. 重启服务
docker-compose down
docker-compose up -d

# 5. 查看日志
docker-compose logs -f airflow-scheduler | grep -i "memory"
```

## ✅ 验证优化效果

### 1. 查看内存使用情况

```bash
# 实时监控系统内存
watch -n 2 free -h

# 查看 Docker 容器内存使用
docker stats --no-stream

# 查看 Airflow Scheduler 容器内存
docker stats airflow-scheduler --no-stream
```

### 2. 查看优化日志

```bash
# 查看 train_models 的内存优化日志
docker-compose logs airflow-scheduler 2>&1 | grep -A 5 "MEMORY-OPTIMIZED"

# 查看内存释放日志
docker-compose logs airflow-scheduler 2>&1 | grep "Memory usage:"

# 查看相似度批处理日志
docker-compose logs airflow-scheduler 2>&1 | grep "batches"
```

预期日志输出：
```
INFO MEMORY-OPTIMIZED MODEL TRAINING
INFO Optimizing DataFrame memory usage...
INFO Memory optimization: 100.00 MB -> 35.23 MB (64.8% reduction)
INFO Computing similarity in batches (total=5000, batch_size=1000, top_k=200)
INFO Memory usage: 450.2 MB -> 180.5 MB (freed 269.7 MB)
```

### 3. 测试 DAG 运行

```bash
# 触发一次完整的 DAG 运行
docker exec -it airflow-scheduler airflow dags trigger recommendation_pipeline

# 监控执行
docker exec -it airflow-scheduler airflow dags list-runs -d recommendation_pipeline

# 查看任务日志
docker-compose logs -f airflow-scheduler
```

## 📈 监控和调优

### 情况 1: 内存仍然不足

如果仍然出现 OOM，逐步调整参数：

```bash
# 步骤 1: 减少批次大小
SIMILARITY_BATCH_SIZE=500

# 步骤 2: 减少 top-K
SIMILARITY_TOP_K=100

# 步骤 3: 禁用 Faiss
USE_FAISS_RECALL=0
```

修改后重启：
```bash
docker-compose restart airflow-scheduler
```

### 情况 2: 运行速度太慢

如果优化后速度明显变慢：

```bash
# 增加批次大小（需要更多内存）
SIMILARITY_BATCH_SIZE=2000

# 或者考虑升级硬件
```

### 情况 3: DAG 任务失败

查看具体错误：
```bash
# 查看失败任务日志
docker exec -it airflow-scheduler airflow tasks test recommendation_pipeline train_models 2024-01-01

# 检查内存使用峰值
docker stats airflow-scheduler
```

## 🔄 回滚方案

如果优化导致问题，快速回滚：

```bash
# 1. 恢复备份的 .env
cp .env.backup_XXXXXX .env

# 2. 重启服务
docker-compose down
docker-compose up -d
```

## 📝 生产环境检查清单

部署前检查：

- [ ] 已备份 `.env` 文件
- [ ] 已在 `.env` 中添加内存优化参数
- [ ] 已在 `docker-compose.yml` 中配置内存限制（可选）
- [ ] 已确认服务器总内存足够（建议至少 32GB）
- [ ] 已确认 swap 空间充足（建议至少 8GB）
- [ ] 已通知团队即将重启服务

部署后检查：

- [ ] 所有服务正常启动（`docker-compose ps`）
- [ ] Airflow Web UI 可访问
- [ ] 推荐 API 健康检查通过
- [ ] 查看日志确认优化生效
- [ ] 触发一次测试 DAG 运行
- [ ] 监控内存使用情况
- [ ] 确认没有 OOM 错误

## 🆘 常见问题

### Q1: 修改 .env 后服务没有生效？

**A**: 需要重启容器使环境变量生效：
```bash
docker-compose down
docker-compose up -d
```

### Q2: docker-compose.yml 中的内存限制不生效？

**A**: 确保使用 `docker-compose` 而不是 `docker compose`（新版本），或者添加 `--compatibility` 标志：
```bash
docker-compose --compatibility up -d
```

### Q3: 如何查看某个 pipeline 步骤的内存使用？

**A**: 使用 `/usr/bin/time` 命令：
```bash
docker exec -it airflow-scheduler /usr/bin/time -v python -m pipeline.train_models
```

### Q4: 如何临时增加 swap？

**A**:
```bash
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

## 📞 支持

如有问题，检查以下资源：

1. **完整文档**: `MEMORY_OPTIMIZATION.md`
2. **快速指南**: `MEMORY_OPTIMIZATION_QUICKSTART.md`
3. **测试脚本**: `scripts/test_memory_optimization.py`
4. **日志位置**: `./airflow/logs/` 和 `docker-compose logs`

---

**部署时间估计**: 5-10 分钟（包括服务重启）
**预期效果**: 内存使用减少 80-90%，避免 OOM 导致的服务器关机
