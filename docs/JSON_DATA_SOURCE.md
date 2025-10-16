# JSON数据源配置指南

本文档说明如何使用JSON文件作为数据源，替代MySQL数据库。

---

## 变更概述

### 问题背景
原有系统从MySQL数据库读取业务数据，但实际部署环境中：
1. 业务数据已经导出为JSON文件（全量 + 增量）
2. JSON文件每小时更新一次，存放在指定目录
3. 不再需要直接连接MySQL数据库

### 解决方案
修改`pipeline/extract_load.py`，支持从JSON文件读取数据：
- 通过`.env`配置数据源类型（`DATA_SOURCE=json`）
- 配置JSON文件目录（`DATA_JSON_DIR`）
- 自动检测全量和增量文件
- 支持增量更新逻辑

---

## 配置方法

### 1. 环境变量配置

编辑`.env`文件，添加以下配置：

```ini
# 数据源类型：json 或 database
DATA_SOURCE=json

# JSON文件存放目录（绝对路径）
DATA_JSON_DIR=/home/ubuntu/recommend/data/dianshu_data
```

**说明：**
- `DATA_SOURCE=json`：使用JSON文件作为数据源
- `DATA_SOURCE=database`：使用MySQL数据库（默认原有方式）
- `DATA_JSON_DIR`：JSON文件存放的目录，需要是绝对路径

### 2. JSON文件命名规范

系统要求JSON文件按以下规则命名：

#### 全量文件
```
{table_name}.json
```
**示例：**
- `user.json` - 用户表全量数据
- `dataset.json` - 数据集表全量数据
- `task.json` - 任务订单表全量数据
- `api_order.json` - API订单表全量数据
- `dataset_image.json` - 数据集图片表全量数据

#### 增量文件
```
{table_name}_YYYYMMDD_HHMMSS.json
```
**示例：**
- `user_20251016_140000.json` - 2025-10-16 14:00:00 导出的用户增量数据
- `dataset_20251016_150000.json` - 2025-10-16 15:00:00 导出的数据集增量数据

**时间戳格式：**
- `YYYYMMDD`：年月日，如 `20251016` 表示 2025年10月16日
- `HHMMSS`：时分秒，如 `140000` 表示 14:00:00

### 3. JSON文件格式

所有JSON文件必须是**数组格式**，每个元素是一条记录：

```json
[
  {
    "id": 1,
    "user_name": "张三",
    "create_time": "2025-09-19T14:49:33",
    "update_time": "2025-10-16T12:00:00"
  },
  {
    "id": 2,
    "user_name": "李四",
    "create_time": "2025-09-20T10:30:00",
    "update_time": "2025-10-16T13:00:00"
  }
]
```

**必须包含时间字段（用于增量更新）：**
- `update_time`（首选）
- `modify_time`
- `create_time`

系统会自动检测这些字段，用于判断增量数据。

---

## 使用方法

### 全量导入

首次导入或需要重新加载所有数据时使用：

```bash
# 设置Python路径
export PYTHONPATH=/home/ubuntu/recommend:$PYTHONPATH

# 全量导入
python3 -m pipeline.extract_load --full-refresh
```

**执行逻辑：**
1. 读取所有表的全量文件（`{table}.json`）
2. 清空现有的parquet文件
3. 将JSON数据转换为parquet格式
4. 保存到`data/business/`目录

### 增量导入

定期更新数据时使用（推荐每小时执行）：

```bash
# 增量导入（首次运行等同于全量）
python3 -m pipeline.extract_load
```

**执行逻辑：**
1. 读取上次导入的watermark（最后一条记录的时间戳）
2. 查找所有时间戳晚于watermark的增量文件
3. 按时间顺序处理增量文件
4. 追加到现有的parquet文件
5. 更新watermark

**首次运行：**
- 如果没有watermark，自动加载全量文件（`{table}.json`）
- 设置mode为`bootstrap`

### 干运行测试

查看将要执行的操作，但不实际执行：

```bash
python3 -m pipeline.extract_load --dry-run
```

**输出示例：**
```
2025-10-16 14:46:40 INFO Data source mode: json
2025-10-16 14:46:40 INFO Using JSON data source from: /home/ubuntu/recommend/data/dianshu_data
2025-10-16 14:46:40 INFO Processing source 'business' from JSON files
2025-10-16 14:46:40 INFO [dry-run] would load table 'user' from JSON dir /home/ubuntu/recommend/data/dianshu_data
2025-10-16 14:46:40 INFO [dry-run] would load table 'dataset' from JSON dir /home/ubuntu/recommend/data/dianshu_data
```

---

## 工作流程

### 数据更新流程

```
1. 外部系统导出JSON文件
   ├─ 全量：每天导出 {table}.json
   └─ 增量：每小时导出 {table}_YYYYMMDD_HHMMSS.json

2. Cron任务定时执行pipeline
   ├─ 每小时执行：python3 -m pipeline.extract_load
   └─ 自动检测新的增量文件

3. Pipeline处理
   ├─ 读取JSON文件
   ├─ 转换为DataFrame
   ├─ 保存为Parquet格式
   └─ 更新watermark

4. 后续流程
   ├─ build_features_v2.py（特征工程）
   ├─ train_models.py（模型训练）
   └─ API服务读取特征
```

### 定时任务配置（推荐）

编辑crontab：
```bash
crontab -e
```

添加定时任务：
```cron
# 每小时5分执行增量导入
5 * * * * cd /home/ubuntu/recommend && PYTHONPATH=/home/ubuntu/recommend python3 -m pipeline.extract_load >> /var/log/extract_load.log 2>&1

# 每天凌晨2点执行全量导入
0 2 * * * cd /home/ubuntu/recommend && PYTHONPATH=/home/ubuntu/recommend python3 -m pipeline.extract_load --full-refresh >> /var/log/extract_load_full.log 2>&1
```

---

## 输出说明

### Parquet文件

数据将被转换为Parquet格式，保存在：
```
data/
└── business/
    ├── user.parquet              # 用户表数据（最新全量）
    ├── dataset.parquet           # 数据集表数据
    ├── task.parquet              # 任务订单表数据
    ├── api_order.parquet         # API订单表数据
    ├── dataset_image.parquet     # 数据集图片表数据
    └── user/                     # 分区数据（历史记录）
        └── load_time=20251016T140000Z.parquet
```

**文件说明：**
- `{table}.parquet`：最新的完整数据，用于特征工程和模型训练
- `{table}/load_time=*.parquet`：历史分区数据，用于审计和回溯

### 状态文件

系统会保存导入状态，用于增量更新：
```
data/
└── _metadata/
    └── extract_state.json
```

**状态文件内容：**
```json
{
  "business": {
    "user": {
      "watermark": "2025-10-16T14:00:00",
      "column": "update_time"
    },
    "dataset": {
      "watermark": "2025-10-16T14:30:00",
      "column": "update_time"
    }
  }
}
```

### 指标文件

每次执行会记录指标，用于监控：
```
data/
└── evaluation/
    └── extract_metrics.json
```

**指标内容：**
```json
[
  {
    "started_at": "2025-10-16T14:00:00Z",
    "data_source": "json",
    "dry_run": false,
    "full_refresh": false,
    "tables": [
      {
        "source": "business",
        "table": "user",
        "mode": "incremental",
        "rows": 125,
        "watermark": "2025-10-16T14:00:00",
        "incremental_column": "update_time"
      }
    ],
    "finished_at": "2025-10-16T14:02:30Z"
  }
]
```

---

## 常见问题

### Q1: 增量文件太多，如何清理？

**问题：**
随着时间推移，增量文件会越来越多，占用磁盘空间。

**解决方案：**
定期清理旧的增量文件（保留最近7天）：

```bash
# 创建清理脚本
cat > /home/ubuntu/recommend/scripts/cleanup_json_incremental.sh << 'EOF'
#!/bin/bash
JSON_DIR=/home/ubuntu/recommend/data/dianshu_data
DAYS_TO_KEEP=7

# 删除7天前的增量文件
find $JSON_DIR -name "*_[0-9]*_[0-9]*.json" -mtime +$DAYS_TO_KEEP -delete

echo "Cleaned up incremental JSON files older than $DAYS_TO_KEEP days"
EOF

chmod +x /home/ubuntu/recommend/scripts/cleanup_json_incremental.sh

# 添加到crontab（每天凌晨3点执行）
0 3 * * * /home/ubuntu/recommend/scripts/cleanup_json_incremental.sh >> /var/log/cleanup_json.log 2>&1
```

### Q2: 如何回滚到某个时间点？

**解决方案：**
1. 删除状态文件中的watermark
2. 手动指定要加载的增量文件

```bash
# 1. 备份当前状态
cp data/_metadata/extract_state.json data/_metadata/extract_state_backup.json

# 2. 删除watermark（重置为初始状态）
echo '{}' > data/_metadata/extract_state.json

# 3. 删除需要回滚的表的parquet文件
rm data/business/user.parquet
rm -rf data/business/user/

# 4. 重新导入特定时间点的全量文件
# 如果有备份，可以恢复特定时间点的全量文件到 JSON_DIR
```

### Q3: JSON文件格式错误怎么办？

**问题：**
JSON文件格式不正确，导致解析失败。

**排查步骤：**
```bash
# 1. 验证JSON格式
python3 << EOF
import json
with open('/home/ubuntu/recommend/data/dianshu_data/user.json') as f:
    data = json.load(f)
    print(f"Valid JSON, {len(data)} records")
EOF

# 2. 检查是否是数组格式
head -1 /home/ubuntu/recommend/data/dianshu_data/user.json
# 应该输出: [

# 3. 检查字段是否存在时间戳
python3 << EOF
import json
with open('/home/ubuntu/recommend/data/dianshu_data/user.json') as f:
    data = json.load(f)
    if data:
        print("Fields:", list(data[0].keys()))
        print("Has update_time:", 'update_time' in data[0])
EOF
```

### Q4: 文件权限问题

**问题：**
```
PermissionError: [Errno 13] Permission denied: 'data/business/user.parquet'
```

**原因：**
- 文件由Docker容器创建，属于其他用户
- 当前用户无权限修改

**解决方案：**
```bash
# 方法1：修改文件所有者（需要sudo权限）
sudo chown -R $USER:$USER /home/ubuntu/recommend/data/

# 方法2：使用Docker运行pipeline
docker-compose run --rm recommendation-api python3 -m pipeline.extract_load

# 方法3：修改输出目录权限
chmod -R 775 /home/ubuntu/recommend/data/
```

### Q5: 数据库模式 vs JSON模式如何切换？

**切换到JSON模式：**
```bash
# 编辑.env
sed -i 's/DATA_SOURCE=database/DATA_SOURCE=json/' .env

# 验证
grep DATA_SOURCE .env
```

**切换回数据库模式：**
```bash
# 编辑.env
sed -i 's/DATA_SOURCE=json/DATA_SOURCE=database/' .env

# 验证
grep DATA_SOURCE .env
```

---

## 性能优化建议

### 1. JSON文件大小控制

- 单个JSON文件建议不超过100MB
- 超过100MB时，考虑分表或分区

### 2. 增量文件频率

- 推荐每小时导出一次增量文件
- 避免过于频繁（< 15分钟），影响性能

### 3. Parquet压缩

Parquet文件自动使用Snappy压缩，节省磁盘空间：
- JSON文件：300MB
- Parquet文件：~100MB（压缩比约3:1）

### 4. 并行处理（未来优化）

当前版本是串行处理，未来可以考虑：
- 多进程并行加载多个表
- 使用Dask/Spark处理大文件

---

## 与原有数据库模式对比

| 特性 | 数据库模式 | JSON模式 |
|------|----------|---------|
| 数据源 | MySQL数据库 | JSON文件 |
| 网络依赖 | 需要数据库连接 | 无网络依赖 |
| 实时性 | 实时查询 | 定时同步（小时级） |
| 增量更新 | SQL WHERE查询 | 文件时间戳判断 |
| 性能 | 查询性能取决于索引 | 文件读取，性能稳定 |
| 维护成本 | 需要维护数据库连接 | 仅需管理文件权限 |
| 适用场景 | 需要实时数据 | 离线批处理 |

---

## 部署清单更新

在新机器部署时，如果使用JSON数据源：

### 1. 不需要配置数据库

可以跳过以下步骤：
- ❌ 安装MySQL客户端
- ❌ 配置数据库连接
- ❌ 创建数据库用户和权限
- ❌ 优化数据库索引

### 2. 需要配置JSON目录

```bash
# 1. 创建JSON数据目录
mkdir -p /home/ubuntu/recommend/data/dianshu_data

# 2. 配置.env
cat >> .env << EOF
DATA_SOURCE=json
DATA_JSON_DIR=/home/ubuntu/recommend/data/dianshu_data
EOF

# 3. 放置JSON文件
# 将全量JSON文件复制到该目录
cp /path/to/json/exports/*.json /home/ubuntu/recommend/data/dianshu_data/

# 4. 验证文件
ls -lh /home/ubuntu/recommend/data/dianshu_data/
```

### 3. 修改定时任务

原来的pipeline脚本无需修改，会自动根据`DATA_SOURCE`环境变量选择数据源。

---

## 监控建议

### 1. 文件监控

监控JSON文件的更新时间，确保数据源正常：

```bash
# 检查最新的增量文件
ls -lt /home/ubuntu/recommend/data/dianshu_data/*_*.json | head -5

# 检查文件大小（异常小可能是导出失败）
du -sh /home/ubuntu/recommend/data/dianshu_data/*.json
```

### 2. Pipeline监控

监控extract_load的执行情况：

```bash
# 查看最近一次执行的指标
cat data/evaluation/extract_metrics.json | tail -1 | python3 -m json.tool

# 查看watermark状态
cat data/_metadata/extract_state.json | python3 -m json.tool
```

### 3. 告警规则

建议设置以下告警：
- JSON文件超过2小时未更新
- Pipeline执行失败（exit code != 0）
- Parquet文件大小异常（突然变小）

---

## 总结

JSON数据源模式的优势：
✅ 无需数据库连接，简化部署
✅ 数据持久化为文件，便于备份和迁移
✅ 支持全量和增量更新，灵活高效
✅ 自动处理时间戳，支持断点续传

注意事项：
⚠️ JSON文件命名必须规范
⚠️ 需要定期清理旧的增量文件
⚠️ 文件权限需要正确设置
⚠️ 仅支持business源，不支持matomo源

如有问题，请查看日志：
- Pipeline日志：`/var/log/extract_load.log`
- 错误日志：执行命令的stderr输出

---

**文档版本：** v1.0.0
**更新时间：** 2025-10-16
**维护者：** 推荐系统团队
