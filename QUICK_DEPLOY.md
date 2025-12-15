# ⚡ 快速部署内存优化 - 5分钟搞定

## 🎯 目标
解决 DAG 运行时内存占满导致服务器关机的问题。

## 🚀 最快部署方式（推荐）

### 自动化脚本（一键搞定）

```bash
# 运行自动配置脚本
bash scripts/add_memory_config.sh
```

脚本会自动：
1. ✅ 备份你的 `.env` 文件
2. ✅ 添加内存优化配置
3. ✅ 询问是否重启服务
4. ✅ 显示配置结果

**就这么简单！**

---

## 📝 手动部署方式（如果自动脚本失败）

### 步骤 1: 备份配置

```bash
cp .env .env.backup_$(date +%Y%m%d_%H%M%S)
```

### 步骤 2: 添加内存优化配置

在 `.env` 文件**末尾**添加以下内容：

```bash
# ============================================
# 内存优化配置 (Memory Optimization)
# ============================================
SIMILARITY_BATCH_SIZE=1000
SIMILARITY_TOP_K=200
USE_FAISS_RECALL=1
RANKING_CVR_WEIGHT=0.5
PYTHONHASHSEED=0
MALLOC_TRIM_THRESHOLD_=100000
```

可以使用以下命令快速添加：

```bash
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
```

### 步骤 3: 重启服务

```bash
docker-compose down
docker-compose up -d
```

### 步骤 4: 验证

```bash
# 查看优化日志
docker-compose logs airflow-scheduler | grep -i "memory"

# 查看内存使用
docker stats --no-stream
```

---

## ✅ 预期效果

| 指标 | 优化前 | 优化后 | 改善 |
|------|--------|--------|------|
| 内存使用 | ~1GB | ~150MB | **↓ 85%** |
| 相似度矩阵 | 800MB | 80MB | **↓ 90%** |
| OOM 风险 | 高 | 低 | ✅ |

---

## 🔍 验证优化生效

查看日志中是否出现以下关键信息：

```bash
docker-compose logs airflow-scheduler 2>&1 | grep -A 2 "MEMORY-OPTIMIZED"
```

**预期输出**:
```
INFO MEMORY-OPTIMIZED MODEL TRAINING
INFO Optimizing DataFrame memory usage...
INFO Memory optimization: 100.00 MB -> 35.23 MB (64.8% reduction)
```

---

## 🎛️ 进阶调优（可选）

如果**内存仍然不足**，修改 `.env` 中的参数：

```bash
# 降低批次大小（更省内存）
SIMILARITY_BATCH_SIZE=500

# 减少 top-K（更省内存）
SIMILARITY_TOP_K=100

# 禁用 Faiss（大幅省内存）
USE_FAISS_RECALL=0
```

修改后重启：
```bash
docker-compose restart airflow-scheduler
```

---

## 📊 监控命令

```bash
# 实时监控系统内存
watch -n 2 free -h

# 查看容器内存使用
docker stats

# 查看 Airflow 日志
docker-compose logs -f airflow-scheduler
```

---

## 🔄 回滚（如果有问题）

```bash
# 恢复备份
cp .env.backup_XXXXXX .env

# 重启服务
docker-compose down
docker-compose up -d
```

---

## 📚 详细文档

- **完整部署指南**: `PRODUCTION_DEPLOYMENT.md`
- **技术文档**: `MEMORY_OPTIMIZATION.md`
- **测试脚本**: `scripts/test_memory_optimization.py`

---

## ⏱️ 时间估算

- **自动化部署**: 2-3 分钟
- **手动部署**: 5-10 分钟
- **验证测试**: 5-10 分钟

**总计**: 10-20 分钟完成整个部署和验证

---

## 🆘 遇到问题？

1. 检查日志: `docker-compose logs airflow-scheduler`
2. 查看内存: `docker stats`
3. 参考完整文档: `PRODUCTION_DEPLOYMENT.md`
4. 测试优化: `python scripts/test_memory_optimization.py`

---

**就这么简单！现在可以安全运行 DAG 了 🎉**
