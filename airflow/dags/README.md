# Airflow DAGs 说明

本目录包含推荐系统的 Airflow 数据流水线任务。

## 📋 DAG 列表

### 1. incremental_data_update

**用途：** 自动处理增量数据并更新模型

**调度：** 每小时执行一次（每小时的第 0 分钟）

**任务流程：**

```
extract_incremental_data
    ↓
clean_data
    ↓
build_features
    ↓
train_models
    ↓
update_recall_engine
    ↓
reload_api_models → clear_redis_cache
```

**任务说明：**

| 任务 ID | 说明 | 执行时间 |
|---------|------|---------|
| extract_incremental_data | 从 JSON 读取增量数据 | 1-5 分钟 |
| clean_data | 数据清洗和验证 | 2-10 分钟 |
| build_features | 特征工程 | 5-15 分钟 |
| train_models | 增量模型训练 | 5-20 分钟 |
| update_recall_engine | 更新召回索引 | 2-5 分钟 |
| reload_api_models | 通知 API 重载模型 | < 1 分钟 |
| clear_redis_cache | 清理缓存 | < 1 分钟 |

**预计总时间：** 15-50 分钟（取决于增量数据量）

---

## 🔄 增量数据处理机制

### JSON 文件命名约定

系统会自动识别并处理以下文件：

**全量文件：**
```
user.json
dataset.json
task.json
api_order.json
dataset_image.json
```

**增量文件：**
```
user_YYYYMMDD_HHMMSS.json
dataset_YYYYMMDD_HHMMSS.json
task_YYYYMMDD_HHMMSS.json
api_order_YYYYMMDD_HHMMSS.json
dataset_image_YYYYMMDD_HHMMSS.json
```

**示例：**
```
data/dianshu_data/
├── user.json                    # 基础全量
├── user_20251016_140000.json    # 2025-10-16 14:00 增量
├── user_20251016_150000.json    # 2025-10-16 15:00 增量
├── user_20251016_160000.json    # 2025-10-16 16:00 增量
└── ...
```

### 处理逻辑

1. **首次加载**：读取全量文件
2. **增量合并**：按时间顺序合并所有增量文件
3. **去重更新**：基于 ID 去重，保留最新记录
4. **特征更新**：只计算增量数据的特征
5. **模型更新**：增量训练或重新训练（根据数据量）

---

## 🚀 使用方法

### 启用 DAG

默认已启用，可以在 Airflow Web UI 中查看：

1. 访问 http://localhost:8080
2. 登录（admin/admin）
3. 找到 `incremental_data_update` DAG
4. 确认状态为 "On"

### 手动触发

**方式 1：Web UI**
1. 访问 http://localhost:8080
2. 点击 DAG `incremental_data_update`
3. 点击右上角 "Trigger DAG" 按钮

**方式 2：命令行**
```bash
# 触发 DAG
docker compose exec airflow-scheduler \
  airflow dags trigger incremental_data_update

# 查看最近运行
docker compose exec airflow-scheduler \
  airflow dags list-runs -d incremental_data_update

# 查看任务状态
docker compose exec airflow-scheduler \
  airflow tasks list incremental_data_update
```

### 修改调度频率

编辑 `incremental_data_update.py`：

```python
dag = DAG(
    'incremental_data_update',
    schedule_interval='0 * * * *',  # 每小时
    # 其他调度示例：
    # '0 */2 * * *'  # 每 2 小时
    # '0 0 * * *'    # 每天凌晨
    # '0 0 * * 0'    # 每周日凌晨
    ...
)
```

重启 Airflow 调度器：
```bash
docker compose restart airflow-scheduler
```

---

## 📊 监控和调试

### 查看 DAG 运行状态

**Web UI：**
- 访问 http://localhost:8080/dags/incremental_data_update/grid
- 绿色 = 成功，红色 = 失败，黄色 = 运行中

**命令行：**
```bash
# 查看最近 5 次运行
docker compose exec airflow-scheduler \
  airflow dags list-runs -d incremental_data_update --limit 5

# 查看特定任务的日志
docker compose exec airflow-scheduler \
  airflow tasks logs incremental_data_update extract_incremental_data <execution_date>
```

### 常见问题排查

**Q1: DAG 没有按时执行**

**检查调度器状态：**
```bash
docker compose logs airflow-scheduler | grep incremental_data_update
```

**确认 DAG 已启用：**
- 在 Web UI 中检查 DAG 状态开关

**Q2: 任务执行失败**

**查看失败原因：**
1. 在 Web UI 点击失败的任务
2. 查看 "Log" 标签
3. 或使用命令行：
```bash
docker compose exec airflow-scheduler \
  airflow tasks logs incremental_data_update <task_id> <execution_date>
```

**常见错误：**
- `FileNotFoundError`: 增量文件不存在 → 检查文件命名
- `ModuleNotFoundError`: Python 路径问题 → 检查 PYTHONPATH
- `Memory Error`: 内存不足 → 增加 Docker 内存限制

**Q3: 增量数据未生效**

**手动测试 Pipeline：**
```bash
docker compose exec recommendation-api bash -c \
  "cd /opt/recommend && \
   export PYTHONPATH=/opt/recommend && \
   python3 -m pipeline.extract_load --incremental"
```

---

## 🔧 开发指南

### 创建新的 DAG

1. 在此目录创建新的 `.py` 文件
2. 定义 DAG 和任务
3. Airflow 会自动识别新的 DAG

**示例：**
```python
from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'recommend-system',
    'start_date': datetime(2025, 10, 16),
    'retries': 1,
}

dag = DAG(
    'my_custom_dag',
    default_args=default_args,
    schedule_interval='@daily',
    catchup=False,
)

task1 = BashOperator(
    task_id='my_task',
    bash_command='echo "Hello from DAG"',
    dag=dag,
)
```

### 测试 DAG

```bash
# 验证 DAG 语法
docker compose exec airflow-scheduler \
  airflow dags list

# 测试单个任务
docker compose exec airflow-scheduler \
  airflow tasks test incremental_data_update extract_incremental_data 2025-10-16
```

---

## 📝 最佳实践

### 1. 增量文件管理

- **定期清理**：增量文件处理后可以归档或删除
- **备份**：建议保留最近 7 天的增量文件
- **监控**：设置文件数量告警，避免堆积过多

### 2. 性能优化

- **错峰执行**：避免高峰期运行 DAG
- **资源限制**：为 Airflow 任务设置合理的资源限制
- **并行执行**：对于独立任务，可以配置并行执行

### 3. 错误处理

- **重试机制**：关键任务配置 `retries` 参数
- **告警通知**：配置 Alertmanager 接收任务失败通知
- **降级策略**：如果增量更新失败，系统仍可使用旧模型

---

## 🔗 相关文档

- [Docker 部署指南](../../docs/DOCKER_DEPLOYMENT.md)
- [Pipeline 说明](../../docs/PIPELINE_OVERVIEW.md)
- [Airflow 官方文档](https://airflow.apache.org/docs/)

---

**维护者：** 推荐系统团队
**最后更新：** 2025-10-16
