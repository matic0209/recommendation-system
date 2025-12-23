# 数据加载机制说明

## 概述

推荐系统支持两种数据源模式：
- **JSON 模式**: 从 JSON 文件加载数据（适合离线数据）
- **Database 模式**: 从 MySQL 数据库加载数据（适合实时数据）

可以为不同数据源配置不同模式：
- `BUSINESS_DATA_SOURCE`: Business 数据（user, dataset, task, api_order, dataset_image）
- `MATOMO_DATA_SOURCE`: Matomo 分析数据

## 数据加载逻辑

### 全量加载 vs 增量加载

系统使用 **watermark 机制**追踪数据加载进度：

#### 1. 全量加载（Bootstrap）

**触发条件：**
- 首次运行（无历史 watermark）
- 手动指定 `--full-refresh` 参数
- 清除了状态文件

**行为：**
- 只加载全量文件：`user.json`, `dataset.json`, `task.json`, `api_order.json`, `dataset_image.json`
- 不处理增量文件
- 完成后记录 watermark

**示例：**
```bash
# Airflow DAG 中
python -m pipeline.extract_load --full-refresh

# 或手动运行
PYTHONPATH=. python pipeline/extract_load.py --full-refresh
```

#### 2. 增量加载（Incremental）

**触发条件：**
- 已有历史 watermark
- 正常调度运行

**行为：**
- 跳过全量文件
- 只处理比 watermark 新的增量文件：
  - `user_20251016_140000.json`
  - `dataset_20251016_150000.json`
  - 等
- 更新 watermark 到最新时间

## 状态文件

### 位置
```
data/_metadata/extract_state.json
```

### 格式
```json
{
  "business": {
    "user": {
      "watermark": "2025-09-19T14:49:33",
      "column": "create_time"
    },
    "dataset": {
      "watermark": "2025-09-19T14:32:20",
      "column": "update_time"
    }
  },
  "matomo": {
    "matomo_log_visit": {
      "watermark": "2025-10-31T06:52:49",
      "column": "visit_last_action_time"
    }
  }
}
```

### Watermark 含义
- 记录上次加载的最新数据时间戳
- 下次增量加载只处理更新的数据
- 不同表独立追踪

## 常见场景

### 场景 1: 首次部署

**问题：**
- 系统首次部署，没有历史数据

**解决：**
```bash
# 1. 确保 JSON 全量文件存在
ls data/dianshu_data/jsons/*.json

# 2. 直接运行 pipeline，自动全量加载
# 无需额外操作
```

### 场景 2: 从 Database 切换到 JSON

**问题：**
- 之前使用 database 模式，现在切换到 JSON
- 状态文件中有旧的 watermark
- 如果不处理，下次运行会增量加载，**导致数据缺失**

**解决方案 A - 清除状态（推荐）：**
```bash
# 1. 诊断当前状态
bash scripts/diagnose_data_loading.sh

# 2. 清除 business 数据源状态
bash scripts/reset_business_state.sh

# 3. 运行 pipeline，自动全量加载 JSON
```

**解决方案 B - 使用 full-refresh：**
```bash
# 在 Airflow DAG 的 extract_load 任务中
# 临时修改命令添加 --full-refresh
python -m pipeline.extract_load --full-refresh
```

### 场景 3: 数据损坏或需要重新加载

**问题：**
- Parquet 数据文件损坏
- 需要从 JSON 重新生成

**解决：**
```bash
# 方法 1: 清除状态并重新加载
bash scripts/reset_business_state.sh

# 方法 2: 直接删除 parquet 文件和状态
rm -rf data/business/*.parquet
rm -f data/_metadata/extract_state.json
```

### 场景 4: 只想更新某个表

**问题：**
- 只有 user 表需要重新加载
- 其他表保持不变

**解决：**
```python
# 手动编辑 data/_metadata/extract_state.json
# 删除 user 的 watermark 条目
{
  "business": {
    "dataset": {...},  # 保留
    "task": {...},     # 保留
    # "user": {...}    # 删除这行，user 将全量加载
  }
}
```

## JSON 文件命名规范

### 全量文件
```
user.json
dataset.json
task.json
api_order.json
dataset_image.json
```

### 增量文件
```
{table}_{timestamp}.json

示例:
user_20251016_140000.json
dataset_20251023_093000.json
```

**时间戳格式:** `YYYYMMDD_HHMMSS`

### 文件选择逻辑

**全量模式：**
- 只读取 `{table}.json`

**增量模式：**
- 列出所有 `{table}_*.json` 文件
- 从文件名提取时间戳
- 选择时间戳 > watermark 的文件
- 按时间顺序处理

## 诊断工具

### 1. 诊断数据加载状态
```bash
bash scripts/diagnose_data_loading.sh
```

**输出：**
- 当前数据源配置
- 现有 watermarks
- Parquet 文件列表
- JSON 文件列表
- 是否需要清除状态

### 2. 清除 Business 状态
```bash
bash scripts/reset_business_state.sh
```

**功能：**
- 备份当前状态
- 清除 business 数据源的 watermark
- 保留 matomo 数据源的 watermark

### 3. 验证数据源配置
```bash
bash scripts/verify_data_source_quick.sh
```

**检查：**
- 环境变量设置
- 配置值生效情况
- 路径和文件存在性

## Airflow DAG 集成

### 当前 DAG 配置

在 `airflow/dags/recommendation_pipeline.py` 中：

```python
extract_load = BashOperator(
    task_id='extract_load',
    bash_command='python -m pipeline.extract_load',
    # 如需全量刷新，改为：
    # bash_command='python -m pipeline.extract_load --full-refresh',
)
```

### 推荐做法

**生产环境：**
1. 正常运行使用增量加载（不加参数）
2. 数据源切换时，先清除状态：
   ```bash
   docker compose exec airflow-scheduler \
     bash /opt/recommend/scripts/reset_business_state.sh
   ```
3. 然后正常触发 DAG

**测试环境：**
- 可以始终使用 `--full-refresh` 确保数据完整性

## 数据一致性保证

### 增量文件的生成

**重要：** 增量文件应该由数据导出工具定期生成，包含：
- 自上次导出以来新增的记录
- 自上次导出以来更新的记录

### 避免数据丢失

**关键点：**
1. ✅ 确保全量文件是最新的完整数据快照
2. ✅ 增量文件按时间顺序覆盖所有更新
3. ✅ 切换数据源时清除状态或使用 full-refresh
4. ✅ 定期验证 parquet 数据的记录数

### 数据验证

```bash
# 检查 parquet 记录数
docker compose exec recommendation-api python3 <<'EOF'
import pyarrow.parquet as pq
for table in ['user', 'dataset', 'task', 'api_order']:
    try:
        path = f'/app/data/business/{table}.parquet'
        table_data = pq.read_table(path)
        print(f'{table}: {len(table_data)} 条记录')
    except:
        print(f'{table}: 文件不存在')
EOF
```

## 常见问题

### Q: 为什么切换到 JSON 后数据不完整？

**A:** 因为状态文件中有旧的 watermark，导致执行增量加载而不是全量加载。
**解决：** 运行 `bash scripts/reset_business_state.sh`

### Q: 全量加载和增量加载可以混用吗？

**A:** 可以。不同表独立追踪 watermark，可以：
- user 表增量加载
- dataset 表全量加载（删除其 watermark）

### Q: 如何确认是全量还是增量加载？

**A:** 查看日志：
```
# 全量/首次
Loading table 'user' (mode=bootstrap, files=1, watermark=None)

# 增量
Loading table 'user' (mode=incremental, files=3, watermark=2025-09-19T14:49:33)
```

### Q: watermark 时间戳从哪来？

**A:** 从数据记录中提取：
- user: `create_time` 字段
- dataset: `update_time` 字段
- task: `update_time` 字段
- 等

使用记录中最新的时间戳作为 watermark。

## 最佳实践

1. **定期备份状态文件**
   ```bash
   cp data/_metadata/extract_state.json \
      data/_metadata/extract_state.json.backup
   ```

2. **监控数据量变化**
   - 记录每次运行的 row_count
   - 异常波动时检查

3. **数据源切换流程**
   - 备份现有数据
   - 清除相关状态
   - 运行测试验证
   - 切换生产配置

4. **使用诊断工具**
   - 切换前运行 `diagnose_data_loading.sh`
   - 确认状态和文件一致

## 参考

- `pipeline/extract_load.py` - 数据加载实现
- `config/settings.py` - 配置管理
- `scripts/reset_business_state.sh` - 状态重置工具
- `scripts/diagnose_data_loading.sh` - 诊断工具
