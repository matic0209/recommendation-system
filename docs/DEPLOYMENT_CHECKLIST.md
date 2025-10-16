# 新机器部署清单

本文档提供在全新机器上部署推荐系统的完整步骤清单。

---

## 📋 部署概览

**系统架构：**
- 业务数据库（MySQL）：存储用户、数据集、订单等业务数据
- 行为数据库（MySQL，可选）：Matomo行为日志
- Redis：缓存和特征存储
- MLflow：模型管理和实验追踪
- FastAPI：推荐服务API
- Prometheus + Grafana：监控和可观测性
- Airflow：数据流水线调度

**核心依赖表（dianshu_backend数据库）：**
- ✅ `task` - 数据集订单表
- ✅ `api_order` - API订单表
- ✅ `dataset` - 数据集表
- ✅ `dataset_image` - 数据集图片表
- ✅ `user` - 用户表
- ❌ ~~`company`~~ - 已删除，不再使用
- ❌ ~~`dict`~~ - 已删除，不再使用

---

## 一、系统要求

### 1.1 基础环境

**操作系统：**
- Linux (推荐 Ubuntu 20.04+ / CentOS 8+)
- macOS 11+ (仅用于开发测试)

**硬件要求：**
- CPU: 4核+ (推荐8核)
- 内存: 8GB+ (推荐16GB)
- 磁盘: 20GB+ 可用空间
- 网络: 稳定的互联网连接

**必需软件：**
```bash
- Python 3.8+
- Docker 20.10+
- Docker Compose v2
- Git 2.20+
- MySQL 5.7+ / 8.0+ (业务数据库)
- curl / wget (用于测试)
```

### 1.2 安装基础工具

**Ubuntu/Debian:**
```bash
# 更新包管理器
sudo apt update

# 安装基础工具
sudo apt install -y \
    python3 \
    python3-pip \
    python3-venv \
    docker.io \
    docker-compose-v2 \
    git \
    curl \
    wget \
    mysql-client

# 启动Docker并设置开机自启
sudo systemctl start docker
sudo systemctl enable docker

# 添加当前用户到docker组（避免每次都要sudo）
sudo usermod -aG docker $USER

# 重新登录以使组权限生效
newgrp docker
```

**CentOS/RHEL:**
```bash
# 安装Docker
sudo yum install -y yum-utils
sudo yum-config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
sudo yum install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# 启动Docker
sudo systemctl start docker
sudo systemctl enable docker

# 安装Python和Git
sudo yum install -y python3 python3-pip git

# 添加用户到docker组
sudo usermod -aG docker $USER
```

**macOS:**
```bash
# 使用Homebrew安装
brew install python@3.8 docker docker-compose git

# 启动Docker Desktop
open /Applications/Docker.app
```

---

## 二、数据源准备

### 📌 重要说明

系统支持两种数据源模式：

1. **JSON文件模式**（推荐）：从JSON文件读取业务数据
2. **数据库模式**：从MySQL数据库实时查询

**选择建议：**
- 如果已有JSON数据导出，推荐使用JSON模式（无需数据库连接）
- 如果需要实时数据，使用数据库模式

### 2.1 数据源模式一：JSON文件（推荐）

**配置方式：**

编辑`.env`文件：
```ini
# 数据源配置
DATA_SOURCE=json
DATA_JSON_DIR=/path/to/json/data
```

**JSON文件要求：**

1. **文件命名规范：**
   - 全量文件：`{table_name}.json`（如 `user.json`, `dataset.json`）
   - 增量文件：`{table_name}_YYYYMMDD_HHMMSS.json`（如 `user_20251016_140000.json`）

2. **必需的表文件：**
   - `user.json` - 用户表
   - `dataset.json` - 数据集表
   - `task.json` - 任务订单表
   - `api_order.json` - API订单表
   - `dataset_image.json` - 数据集图片表

3. **JSON格式：**
   ```json
   [
     {
       "id": 1,
       "user_name": "张三",
       "update_time": "2025-10-16T14:00:00"
     }
   ]
   ```

**详细文档：** 参见 [`docs/JSON_DATA_SOURCE.md`](JSON_DATA_SOURCE.md)

### 2.2 数据源模式二：MySQL数据库

**仅在DATA_SOURCE=database时需要配置**

编辑`.env`文件：
```ini
# 数据源配置
DATA_SOURCE=database

# 数据库连接
BUSINESS_DB_HOST=127.0.0.1
BUSINESS_DB_PORT=3306
BUSINESS_DB_NAME=dianshu_backend
BUSINESS_DB_USER=root
BUSINESS_DB_PASSWORD=your_password
```

**数据库名称：** `dianshu_backend`

**必需表结构：**

| 表名 | 用途 | 关键字段 |
|-----|------|---------|
| `task` | 数据集订单 | create_user, dataset_id, price, pay_status, pay_time |
| `api_order` | API订单 | creator_id, api_id, price, pay_status, pay_time |
| `dataset` | 数据集信息 | id, dataset_name, price, tag, type_id |
| `dataset_image` | 数据集图片 | dataset_id, image_url, image_order |
| `user` | 用户信息 | id, user_name, company_name, province, city |

**创建数据库和用户：**
```sql
-- 创建数据库
CREATE DATABASE IF NOT EXISTS dianshu_backend
CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

-- 创建用户并授权（生产环境建议只读权限）
CREATE USER IF NOT EXISTS 'recommend_user'@'%' IDENTIFIED BY 'your_secure_password';
GRANT SELECT ON dianshu_backend.* TO 'recommend_user'@'%';
FLUSH PRIVILEGES;
```

### 2.2 行为数据库（可选）

**数据库名称：** `matomo`

如果没有Matomo行为数据，可以跳过，系统会自动降级为仅使用业务数据。

### 2.3 索引优化

**运行索引创建脚本（提升查询性能）：**
```bash
# 验证现有索引
python scripts/p0_02_verify_indexes.py

# 自动创建缺失索引（完整版）
python scripts/p0_02_verify_indexes.py --full

# 手动创建索引（可选）
mysql -h<host> -u<user> -p dianshu_backend < scripts/p0_01_add_indexes_fixed.sql
```

**关键索引：**
- `task`: (create_user, dataset_id, pay_status, update_time)
- `api_order`: (creator_id, api_id, pay_status, update_time)
- `dataset`: (id, is_delete, update_time, type_id)
- `user`: (id, is_valid, update_time)

---

## 三、部署步骤

### 3.1 克隆代码仓库

```bash
# 克隆到指定目录
git clone <repository-url> /opt/recommend
cd /opt/recommend

# 查看当前分支
git branch

# 切换到生产分支（如果需要）
git checkout main  # 或 master
```

### 3.2 配置环境变量

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑配置文件
vim .env  # 或使用 nano .env
```

**必填配置项：**

```ini
# ============ 业务数据库配置（必填） ============
BUSINESS_DB_HOST=127.0.0.1          # 数据库主机地址
BUSINESS_DB_PORT=3306               # 数据库端口
BUSINESS_DB_NAME=dianshu_backend    # 数据库名称
BUSINESS_DB_USER=recommend_user     # 数据库用户名
BUSINESS_DB_PASSWORD=your_password  # 数据库密码

# ============ 行为数据库配置（可选） ============
MATOMO_DB_HOST=127.0.0.1
MATOMO_DB_PORT=3306
MATOMO_DB_NAME=matomo
MATOMO_DB_USER=matomo_user
MATOMO_DB_PASSWORD=your_password

# ============ Redis配置（推荐） ============
REDIS_URL=redis://127.0.0.1:6379/0
FEATURE_REDIS_URL=redis://127.0.0.1:6379/1

# ============ MLflow配置（可选） ============
MLFLOW_TRACKING_URI=http://127.0.0.1:5000
MLFLOW_EXPERIMENT_NAME=dataset_recommendation

# ============ 数据库连接池优化 ============
DB_POOL_SIZE=10                     # 连接池大小
DB_MAX_OVERFLOW=20                  # 最大溢出连接数
DB_POOL_RECYCLE=3600                # 连接回收时间（秒）
DB_POOL_PRE_PING=true               # 连接前测试
DB_CONNECT_TIMEOUT=10               # 连接超时（秒）

# ============ 日志配置 ============
LOG_LEVEL=INFO                      # DEBUG, INFO, WARNING, ERROR

# ============ 特性开关 ============
USE_FAISS_RECALL=1                  # 是否启用Faiss向量召回
ENABLE_METRICS=true                 # 是否启用Prometheus指标
```

**数据库连接说明：**
- 如果数据库在本机：使用 `127.0.0.1`
- 如果数据库在Docker容器：使用 `host.docker.internal`（macOS/Windows）或容器IP
- 如果数据库在远程服务器：使用实际IP地址

### 3.3 安装Python依赖

```bash
# 创建虚拟环境（推荐）
python3 -m venv venv
source venv/bin/activate

# 升级pip
pip install --upgrade pip

# 安装核心依赖
pip install -r requirements.txt

# 可选：安装向量检索和ONNX支持
pip install faiss-cpu skl2onnx

# 验证安装
python -c "import fastapi, pandas, sklearn, redis; print('All dependencies installed successfully!')"
```

**依赖说明：**
- `fastapi + uvicorn`: API框架
- `pandas + numpy`: 数据处理
- `scikit-learn`: 机器学习
- `pymysql + sqlalchemy`: 数据库连接
- `redis + hiredis`: Redis客户端
- `mlflow`: 模型管理
- `prometheus-client`: 监控指标
- `lightgbm`: 排序模型
- `torch + torchvision`: 图像特征提取（可选）

### 3.4 启动基础服务（Docker）

```bash
# 启动所有服务
docker-compose up -d

# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f

# 查看特定服务日志
docker-compose logs -f redis
docker-compose logs -f mlflow
```

**启动的服务：**

| 服务 | 端口 | 用途 | 健康检查 |
|-----|------|------|---------|
| Redis | 6379 | 缓存和特征存储 | `redis-cli ping` |
| MLflow | 5000 | 模型管理 | `curl http://localhost:5000` |
| Prometheus | 9090 | 指标采集 | `curl http://localhost:9090/-/healthy` |
| Grafana | 3000 | 监控看板 | `curl http://localhost:3000/api/health` |
| Airflow Web | 8080 | 流水线UI | `curl http://localhost:8080/health` |
| Postgres | 5432 | Airflow元数据 | （内部使用） |

**验证服务：**
```bash
# 检查所有容器是否运行
docker-compose ps | grep Up

# 测试Redis连接
redis-cli ping
# 预期输出: PONG

# 测试MLflow
curl http://localhost:5000/api/2.0/mlflow/experiments/list
# 预期输出: JSON响应

# 访问监控面板
curl -s http://localhost:3000/api/health | grep ok
# 预期输出: {"commit":"...","database":"ok",...}
```

**常见问题：**
- 端口冲突：修改 `docker-compose.yml` 中的端口映射
- 权限问题：确保当前用户在docker组中
- 启动失败：查看日志 `docker-compose logs <service-name>`

### 3.5 数据处理与模型训练

#### 方式A：一键运行完整Pipeline（推荐）

```bash
# 1. 查看执行计划（不实际执行）
bash scripts/run_pipeline.sh --dry-run

# 2. 全量执行（抽取→清洗→特征→训练→评估）
bash scripts/run_pipeline.sh

# 3. 仅同步特征和模型（跳过数据抽取）
bash scripts/run_pipeline.sh --sync-only
```

**执行时间估计：**
- 数据抽取：5-15分钟（取决于数据量）
- 特征工程：10-30分钟
- 模型训练：5-20分钟
- 总计：约30-60分钟

#### 方式B：分步执行（适合调试）

```bash
# 设置Python路径
export PYTHONPATH=/opt/recommend:$PYTHONPATH

# 1. 数据抽取与加载
python3 -m pipeline.extract_load
# 产出: data/business/*.parquet, data/matomo/*.parquet

# 2. 数据清洗
python3 -m pipeline.clean_data
# 产出: data/cleaned/*.parquet

# 3. 图像特征提取（可选，需要dataset_image表有数据）
python3 -m pipeline.image_features
# 产出: data/processed/dataset_image_embeddings.parquet

# 4. 特征工程v2（增强版）
python3 -m pipeline.build_features_v2
# 产出: data/processed/*_features_v2.parquet

# 5. 模型训练
python3 -m pipeline.train_models
# 产出: models/*.pkl, models/*.json, models/rank_model.onnx

# 6. 召回引擎构建
python3 -m pipeline.recall_engine_v2
# 产出: models/item_recall_vector.json, models/tag_to_items.json 等

# 7. 数据质量评估
python3 -m pipeline.evaluate_quality_v2
# 产出: data/evaluation/data_quality_report_v2.json
```

#### 预期产出文件

**数据文件（`data/` 目录）：**
```
data/
├── business/                           # 业务库数据
│   ├── task.parquet
│   ├── api_order.parquet
│   ├── dataset.parquet
│   └── user.parquet
├── matomo/                             # 行为库数据（可选）
│   └── matomo_log_visit.parquet
├── cleaned/                            # 清洗后数据
│   ├── interactions_cleaned.parquet
│   ├── items_cleaned.parquet
│   └── users_cleaned.parquet
├── processed/                          # 特征数据
│   ├── interactions_features_v2.parquet
│   ├── items_features_v2.parquet
│   ├── users_features_v2.parquet
│   └── dataset_image_embeddings.parquet
└── evaluation/                         # 评估报告
    ├── data_quality_report_v2.json
    ├── data_quality_report.html
    └── data_quality_metrics.prom
```

**模型文件（`models/` 目录）：**
```
models/
├── item_sim_behavior.pkl          # 基于行为的物品相似度矩阵
├── item_sim_content.pkl           # 基于内容的物品相似度矩阵
├── user_similarity.pkl             # 用户相似度矩阵（UserCF）
├── rank_model.pkl                  # LightGBM排序模型（Pickle格式）
├── rank_model.onnx                 # LightGBM排序模型（ONNX格式）
├── item_recall_vector.json        # Faiss向量召回索引
├── tag_to_items.json               # 标签倒排索引
├── item_to_tags.json               # 物品标签映射
├── category_index.json             # 类目索引
├── price_bucket_index.json        # 价格区间索引
├── top_items.json                  # 全局热门榜单
├── model_registry.json             # 模型元信息和版本
└── ranking_scores_preview.json    # 排序分数预览（用于调试）
```

**检查产出：**
```bash
# 检查数据文件
ls -lh data/business/
ls -lh data/processed/

# 检查模型文件
ls -lh models/

# 查看模型元信息
cat models/model_registry.json | python3 -m json.tool
```

### 3.6 启动推荐API服务

#### 开发模式（支持热重载）

```bash
# 激活虚拟环境
source venv/bin/activate

# 启动API（单进程，支持代码热更新）
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# 查看日志
tail -f logs/app.log
```

#### 生产模式（多worker）

```bash
# 方式1：直接运行
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4

# 方式2：使用Docker
docker-compose up -d recommendation-api

# 方式3：使用Gunicorn（更好的进程管理）
gunicorn app.main:app \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000 \
    --timeout 120 \
    --access-logfile logs/access.log \
    --error-logfile logs/error.log
```

**Workers数量建议：**
- CPU密集型：`workers = (CPU核心数 * 2) + 1`
- IO密集型：`workers = CPU核心数 * 4`
- 一般建议：4-8个worker

**检查API进程：**
```bash
# 查看进程
ps aux | grep uvicorn

# 查看端口占用
netstat -tlnp | grep 8000

# 查看资源占用
top -p $(pgrep -f uvicorn)
```

---

## 四、验证部署

### 4.1 健康检查

```bash
# 基础健康检查
curl http://localhost:8000/health

# 预期输出
{
  "status": "healthy",
  "cache": "enabled",
  "models_loaded": true,
  "checks": {
    "redis": true,
    "models": true
  }
}
```

**健康检查失败排查：**
- `models_loaded: false`: 检查 `models/` 目录是否有模型文件
- `redis: false`: 检查Redis是否运行 `docker-compose ps redis`
- `status: unhealthy`: 查看详细日志 `curl http://localhost:8000/health?verbose=true`

### 4.2 API接口测试

#### 1. 个性化推荐（用户+物品）

```bash
# 基础请求
curl "http://localhost:8000/recommend/detail/1?user_id=100&top_n=5"

# 完整请求（带所有参数）
curl "http://localhost:8000/recommend/detail/123?user_id=456&top_n=10&skip_viewed=true&experiment_variant=control"
```

**响应示例：**
```json
{
  "dataset_id": 123,
  "recommendations": [
    {
      "dataset_id": 456,
      "title": "示例数据集",
      "price": 99.0,
      "cover_image": "https://example.com/image.jpg",
      "score": 0.95,
      "reason": "behavior+content+category+price+rank"
    }
  ],
  "metadata": {
    "user_id": 456,
    "algorithm": "hybrid",
    "model_version": "v1.0.0",
    "experiment_variant": "control"
  }
}
```

#### 2. 相似推荐（仅基于物品）

```bash
# 基础请求
curl "http://localhost:8000/similar/123?top_n=10"

# 带过滤条件
curl "http://localhost:8000/similar/123?top_n=10&min_score=0.5&max_price=100"
```

#### 3. 热门榜单

```bash
# 1小时热门
curl "http://localhost:8000/hot/trending?timeframe=1h&top_n=20"

# 24小时热门
curl "http://localhost:8000/hot/trending?timeframe=24h&top_n=50"

# 7天热门
curl "http://localhost:8000/hot/trending?timeframe=7d&top_n=100"
```

#### 4. Prometheus指标

```bash
# 获取所有指标
curl http://localhost:8000/metrics

# 查看特定指标
curl http://localhost:8000/metrics | grep recommendation_requests_total
curl http://localhost:8000/metrics | grep recommendation_latency_seconds
curl http://localhost:8000/metrics | grep cache_hit_rate
```

### 4.3 性能测试

#### 使用Apache Bench (ab)

```bash
# 安装ab（如果未安装）
sudo apt install apache2-utils  # Ubuntu/Debian
sudo yum install httpd-tools     # CentOS/RHEL

# 测试QPS（1000请求，10并发）
ab -n 1000 -c 10 "http://localhost:8000/similar/123?top_n=10"

# 查看统计信息
# - Requests per second (QPS)
# - Time per request (延迟)
# - Percentage of requests served within a certain time
```

#### 使用wrk

```bash
# 安装wrk
sudo apt install wrk  # Ubuntu/Debian

# 测试60秒（10连接，2线程）
wrk -t2 -c10 -d60s "http://localhost:8000/similar/123?top_n=10"

# 查看输出
# - Latency (50th, 75th, 90th, 99th percentile)
# - Requests/sec
# - Transfer/sec
```

#### 压测脚本示例

```bash
# 创建压测脚本
cat > load_test.sh << 'EOF'
#!/bin/bash
ENDPOINT="http://localhost:8000/similar/123?top_n=10"
DURATION=60
CONCURRENCY=20

echo "Starting load test..."
echo "Endpoint: $ENDPOINT"
echo "Duration: ${DURATION}s"
echo "Concurrency: $CONCURRENCY"

wrk -t4 -c$CONCURRENCY -d${DURATION}s $ENDPOINT

echo "Load test completed!"
EOF

# 运行压测
chmod +x load_test.sh
./load_test.sh
```

**性能基准（参考值）：**
- QPS: > 100 (单worker)
- P95延迟: < 100ms (有缓存)
- P99延迟: < 500ms
- 缓存命中率: > 80%

### 4.4 监控面板检查

#### Prometheus

```bash
# 访问Prometheus
open http://localhost:9090

# 或使用curl测试查询
curl -G http://localhost:9090/api/v1/query \
  --data-urlencode 'query=recommendation_requests_total'
```

**常用PromQL查询：**
```promql
# QPS
sum(rate(recommendation_requests_total[1m]))

# 成功率
sum(rate(recommendation_requests_total{status="success"}[5m]))
/
sum(rate(recommendation_requests_total[5m])) * 100

# P95延迟
histogram_quantile(0.95,
  sum(rate(recommendation_latency_seconds_bucket[5m])) by (le, endpoint)
)

# 缓存命中率
cache_hit_rate
```

#### Grafana

```bash
# 访问Grafana
open http://localhost:3000

# 默认登录
# 用户名: admin
# 密码: admin（首次登录需修改）
```

**导入仪表板：**
1. 登录Grafana
2. 点击 "+" → "Import"
3. 选择文件：`monitoring/grafana/dashboards/recommendation-overview.json`
4. 选择数据源：Prometheus
5. 点击 "Import"

#### MLflow

```bash
# 访问MLflow UI
open http://localhost:5000

# 查看实验
curl http://localhost:5000/api/2.0/mlflow/experiments/list
```

---

## 五、常见问题排查

### 5.1 数据库连接失败

**症状：**
```
pymysql.err.OperationalError: (2003, "Can't connect to MySQL server...")
```

**排查步骤：**

1. **检查数据库是否可访问**
```bash
# 测试连接
mysql -h<host> -P<port> -u<user> -p

# 检查端口是否开放
telnet <host> <port>
nc -zv <host> <port>
```

2. **检查.env配置**
```bash
cat .env | grep DB_

# 验证配置
python3 << EOF
import os
from dotenv import load_dotenv
load_dotenv()
print(f"Host: {os.getenv('BUSINESS_DB_HOST')}")
print(f"Port: {os.getenv('BUSINESS_DB_PORT')}")
print(f"Database: {os.getenv('BUSINESS_DB_NAME')}")
print(f"User: {os.getenv('BUSINESS_DB_USER')}")
EOF
```

3. **检查网络和防火墙**
```bash
# 检查防火墙规则
sudo iptables -L -n | grep 3306
sudo firewall-cmd --list-all | grep 3306  # CentOS/RHEL

# 检查MySQL绑定地址
mysql -e "SHOW VARIABLES LIKE 'bind_address';"
```

4. **检查用户权限**
```sql
-- 登录MySQL
mysql -uroot -p

-- 查看用户权限
SHOW GRANTS FOR 'recommend_user'@'%';

-- 重新授权（如果需要）
GRANT SELECT ON dianshu_backend.* TO 'recommend_user'@'%';
FLUSH PRIVILEGES;
```

### 5.2 模型文件缺失

**症状：**
```json
{
  "status": "unhealthy",
  "models_loaded": false
}
```

**解决方案：**

1. **检查模型目录**
```bash
ls -lh models/

# 应包含以下文件
# - item_sim_behavior.pkl
# - item_sim_content.pkl
# - rank_model.pkl
# - model_registry.json
```

2. **重新运行Pipeline**
```bash
# 完整流水线
bash scripts/run_pipeline.sh

# 或仅训练模型
python3 -m pipeline.train_models
```

3. **从备份恢复**
```bash
# 如果有模型备份
cp -r /backup/models/* ./models/

# 验证模型加载
python3 << EOF
import pickle
with open('models/item_sim_behavior.pkl', 'rb') as f:
    model = pickle.load(f)
    print(f"Model loaded successfully: {type(model)}")
EOF
```

4. **检查文件权限**
```bash
# 确保API进程有读权限
chmod -R 755 models/
chown -R $USER:$USER models/
```

### 5.3 Redis连接失败

**症状：**
```
redis.exceptions.ConnectionError: Error connecting to Redis
```

**排查步骤：**

1. **检查Redis服务**
```bash
# 查看Docker容器状态
docker-compose ps redis

# 查看Redis日志
docker-compose logs redis

# 重启Redis
docker-compose restart redis
```

2. **测试Redis连接**
```bash
# 使用redis-cli测试
redis-cli ping
# 预期输出: PONG

# 测试读写
redis-cli SET test_key "test_value"
redis-cli GET test_key
redis-cli DEL test_key
```

3. **检查.env配置**
```bash
cat .env | grep REDIS

# Python测试连接
python3 << EOF
import redis
import os
from dotenv import load_dotenv
load_dotenv()

r = redis.from_url(os.getenv('REDIS_URL'))
r.ping()
print("Redis connection successful!")
EOF
```

4. **检查Redis内存**
```bash
# 查看Redis内存使用
redis-cli INFO memory

# 清理缓存（如果内存不足）
redis-cli FLUSHDB
```

### 5.4 API响应慢/超时

**症状：**
- 请求超时（> 30s）
- P99延迟过高（> 1s）
- 缓存命中率低（< 50%）

**优化步骤：**

1. **检查缓存状态**
```bash
# 查看缓存命中率
curl http://localhost:8000/metrics | grep cache_hit

# 预热缓存
python3 << EOF
import requests
for dataset_id in range(1, 101):
    requests.get(f"http://localhost:8000/similar/{dataset_id}?top_n=10")
print("Cache warmed up!")
EOF
```

2. **检查数据库查询性能**
```bash
# 运行索引优化
python scripts/p0_02_verify_indexes.py --full

# 检查慢查询日志
mysql -e "SHOW VARIABLES LIKE 'slow_query_log%';"
mysql -e "SELECT * FROM mysql.slow_log LIMIT 10;"
```

3. **调整worker数量**
```bash
# 增加worker（如果CPU有余）
uvicorn app.main:app --workers 8

# 减少worker（如果内存不足）
uvicorn app.main:app --workers 2
```

4. **监控资源使用**
```bash
# CPU和内存
top -p $(pgrep -f uvicorn)

# 磁盘IO
iostat -x 1

# 网络
iftop -i eth0
```

### 5.5 内存不足

**症状：**
- 进程被OOM Killer杀死
- `MemoryError` 异常
- 系统响应缓慢

**解决方案：**

1. **限制Redis内存**
```yaml
# 在docker-compose.yml中已配置
redis:
  command: redis-server --maxmemory 2gb --maxmemory-policy allkeys-lru
```

2. **减少API worker数量**
```bash
# 从4个减少到2个
uvicorn app.main:app --workers 2
```

3. **优化模型加载**
```python
# 在app/main.py中，使用延迟加载或模型压缩
# 已实现懒加载机制，模型按需加载
```

4. **监控内存使用**
```bash
# 查看内存使用
free -h

# 查看进程内存
ps aux --sort=-%mem | head

# 释放缓存（临时解决）
sudo sync && sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
```

### 5.6 Docker相关问题

**容器无法启动：**
```bash
# 查看详细日志
docker-compose logs <service-name>

# 重新构建镜像
docker-compose build --no-cache <service-name>

# 清理旧容器和网络
docker-compose down -v
docker system prune -a
```

**端口冲突：**
```bash
# 查看端口占用
sudo netstat -tlnp | grep <port>

# 修改docker-compose.yml中的端口映射
# 例如：将 8000:8000 改为 8001:8000
```

**权限问题：**
```bash
# 添加用户到docker组
sudo usermod -aG docker $USER

# 重新登录
newgrp docker

# 测试
docker ps
```

---

## 六、生产环境优化建议

### 6.1 数据库优化

1. **创建只读副本**
```sql
-- 使用MySQL主从复制，推荐API读取从库
-- 配置从库连接
BUSINESS_DB_HOST=replica.example.com
```

2. **启用连接池**
```ini
# 已在.env中配置
DB_POOL_SIZE=20
DB_MAX_OVERFLOW=40
DB_POOL_RECYCLE=3600
DB_POOL_PRE_PING=true
```

3. **监控慢查询**
```sql
-- 启用慢查询日志
SET GLOBAL slow_query_log = 'ON';
SET GLOBAL long_query_time = 1;
SET GLOBAL slow_query_log_file = '/var/log/mysql/slow.log';
```

### 6.2 Redis优化

1. **持久化配置**
```bash
# AOF持久化（已在docker-compose.yml配置）
redis-server --appendonly yes --appendfsync everysec
```

2. **内存淘汰策略**
```bash
# LRU淘汰（已配置）
redis-server --maxmemory 4gb --maxmemory-policy allkeys-lru
```

3. **主从复制（可选）**
```yaml
# docker-compose.yml添加Redis从节点
redis-replica:
  image: redis:7-alpine
  command: redis-server --replicaof redis 6379
```

### 6.3 API服务优化

1. **使用Nginx反向代理**
```nginx
upstream recommend_api {
    server 127.0.0.1:8000;
    server 127.0.0.1:8001;
    server 127.0.0.1:8002;
    server 127.0.0.1:8003;
}

server {
    listen 80;
    server_name api.example.com;

    location / {
        proxy_pass http://recommend_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;

        # 超时配置
        proxy_connect_timeout 10s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;
    }
}
```

2. **启用HTTPS**
```bash
# 使用Let's Encrypt
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d api.example.com
```

3. **配置Systemd服务**
```ini
# /etc/systemd/system/recommend-api.service
[Unit]
Description=Recommendation API Service
After=network.target redis.service

[Service]
Type=notify
User=recommend
Group=recommend
WorkingDirectory=/opt/recommend
Environment="PATH=/opt/recommend/venv/bin"
ExecStart=/opt/recommend/venv/bin/uvicorn app.main:app \
    --host 0.0.0.0 --port 8000 --workers 4
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### 6.4 监控和告警

1. **配置AlertManager**
```yaml
# monitoring/alertmanager.yml
route:
  receiver: 'team-slack'
  group_by: ['alertname', 'severity']
  group_wait: 10s
  group_interval: 5m
  repeat_interval: 3h

receivers:
  - name: 'team-slack'
    slack_configs:
      - api_url: 'https://hooks.slack.com/services/YOUR/WEBHOOK/URL'
        channel: '#alerts'
        text: '{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'
```

2. **配置告警规则**
```yaml
# monitoring/alert_rules.yml
groups:
  - name: recommendation_api
    interval: 30s
    rules:
      - alert: HighErrorRate
        expr: |
          sum(rate(recommendation_requests_total{status="error"}[5m]))
          /
          sum(rate(recommendation_requests_total[5m])) > 0.05
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High error rate detected"

      - alert: HighLatency
        expr: |
          histogram_quantile(0.95,
            sum(rate(recommendation_latency_seconds_bucket[5m])) by (le)
          ) > 1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "P95 latency > 1s"
```

3. **日志聚合**
```bash
# 使用ELK或Loki收集日志
# 配置日志转发（示例：使用filebeat）
docker run -d \
  --name filebeat \
  -v /opt/recommend/logs:/logs:ro \
  -v ./filebeat.yml:/usr/share/filebeat/filebeat.yml:ro \
  docker.elastic.co/beats/filebeat:8.10.0
```

### 6.5 备份策略

1. **模型备份**
```bash
#!/bin/bash
# scripts/backup_models.sh
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backup/models"

mkdir -p $BACKUP_DIR
tar -czf $BACKUP_DIR/models_$DATE.tar.gz models/

# 保留最近30天的备份
find $BACKUP_DIR -name "models_*.tar.gz" -mtime +30 -delete

echo "Backup completed: $BACKUP_DIR/models_$DATE.tar.gz"
```

2. **数据备份**
```bash
#!/bin/bash
# scripts/backup_data.sh
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backup/data"

mkdir -p $BACKUP_DIR

# 备份Parquet文件
tar -czf $BACKUP_DIR/data_$DATE.tar.gz data/processed/

# 备份Redis（RDB快照）
docker exec redis redis-cli SAVE
docker cp redis:/data/dump.rdb $BACKUP_DIR/redis_$DATE.rdb

echo "Data backup completed"
```

3. **定时备份（Cron）**
```bash
# 编辑crontab
crontab -e

# 添加定时任务
# 每天凌晨2点备份模型
0 2 * * * /opt/recommend/scripts/backup_models.sh >> /var/log/backup.log 2>&1

# 每周日凌晨3点备份数据
0 3 * * 0 /opt/recommend/scripts/backup_data.sh >> /var/log/backup.log 2>&1
```

### 6.6 安全加固

1. **API认证**
```python
# 在app/main.py中添加API Key认证
from fastapi import Security, HTTPException
from fastapi.security.api_key import APIKeyHeader

API_KEY_HEADER = APIKeyHeader(name="X-API-Key")

async def verify_api_key(api_key: str = Security(API_KEY_HEADER)):
    if api_key != os.getenv("API_KEY"):
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key
```

2. **限流**
```python
# 使用slowapi
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.get("/similar/{dataset_id}")
@limiter.limit("100/minute")
async def get_similar(dataset_id: int):
    ...
```

3. **防火墙规则**
```bash
# 仅允许特定IP访问
sudo ufw allow from 10.0.0.0/8 to any port 8000
sudo ufw enable

# 或使用iptables
sudo iptables -A INPUT -p tcp --dport 8000 -s 10.0.0.0/8 -j ACCEPT
sudo iptables -A INPUT -p tcp --dport 8000 -j DROP
```

---

## 七、部署后检查清单

### 基础检查

- [ ] 数据库连接成功
- [ ] Redis连接成功
- [ ] 模型文件加载成功
- [ ] API健康检查返回 `healthy`
- [ ] 所有Docker容器运行正常

### 功能检查

- [ ] 个性化推荐接口正常
- [ ] 相似推荐接口正常
- [ ] 热门榜单接口正常
- [ ] Prometheus指标可访问
- [ ] 缓存命中率 > 50%

### 性能检查

- [ ] QPS > 100 (压测)
- [ ] P95延迟 < 200ms
- [ ] P99延迟 < 500ms
- [ ] CPU使用率 < 70%
- [ ] 内存使用率 < 80%

### 监控检查

- [ ] Prometheus采集正常
- [ ] Grafana仪表板显示正常
- [ ] 告警规则配置完成
- [ ] 日志输出正常

### 安全检查

- [ ] API认证已启用（生产环境）
- [ ] HTTPS已配置（生产环境）
- [ ] 防火墙规则已设置
- [ ] 敏感信息未暴露（.env不在代码库）

### 备份检查

- [ ] 模型备份脚本已配置
- [ ] 数据备份脚本已配置
- [ ] Cron定时任务已设置
- [ ] 备份恢复已测试

---

## 八、快速命令参考

### Docker管理

```bash
# 启动所有服务
docker-compose up -d

# 停止所有服务
docker-compose down

# 重启特定服务
docker-compose restart <service-name>

# 查看日志
docker-compose logs -f <service-name>

# 进入容器
docker-compose exec <service-name> bash

# 清理资源
docker system prune -a
```

### Pipeline管理

```bash
# 完整流水线
bash scripts/run_pipeline.sh

# 查看执行计划
bash scripts/run_pipeline.sh --dry-run

# 仅同步特征和模型
bash scripts/run_pipeline.sh --sync-only

# 单独步骤
python3 -m pipeline.extract_load
python3 -m pipeline.build_features_v2
python3 -m pipeline.train_models
```

### API管理

```bash
# 启动API（开发模式）
uvicorn app.main:app --reload

# 启动API（生产模式）
uvicorn app.main:app --workers 4

# 热更新模型
curl -X POST http://localhost:8000/models/reload

# 查看健康状态
curl http://localhost:8000/health
```

### 监控管理

```bash
# 查看指标
curl http://localhost:8000/metrics

# 查看Prometheus
open http://localhost:9090

# 查看Grafana
open http://localhost:3000

# 查看MLflow
open http://localhost:5000
```

---

## 九、相关文档

- [QUICKSTART.md](../QUICKSTART.md) - 快速开始指南
- [README.md](../README.md) - 项目概览
- [docs/ARCHITECTURE.md](ARCHITECTURE.md) - 架构设计
- [docs/API_REFERENCE.md](API_REFERENCE.md) - API接口文档
- [docs/OPERATIONS_SOP.md](OPERATIONS_SOP.md) - 运维手册
- [docs/PIPELINE_OVERVIEW.md](PIPELINE_OVERVIEW.md) - 流水线详解

---

## 十、支持与反馈

- 📖 文档问题：提交 Issue 到 GitHub
- 🐛 Bug报告：使用 GitHub Issues
- 💡 功能建议：通过 Pull Request 提交
- 📧 联系方式：[your-email@example.com]

---

**最后更新时间：** 2025-10-16
**版本：** v1.0.0
**维护者：** 推荐系统团队
