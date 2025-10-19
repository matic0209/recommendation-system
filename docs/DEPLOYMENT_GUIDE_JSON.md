# 推荐系统部署指南（JSON 数据源模式）

本文档适用于**使用 JSON 文件作为数据源**的快速部署。

## 📋 前提条件

### 系统要求
- **操作系统**：Linux (Ubuntu 20.04+ / CentOS 8+)
- **硬件**：4核+ CPU，8GB+ 内存，20GB+ 磁盘
- **软件**：
  - Docker 20.10+
  - Docker Compose v2
  - Python 3.8+
  - Git 2.20+

### 安装 Docker（如果未安装）

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install -y docker.io docker-compose-v2 python3 python3-pip git
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker $USER
newgrp docker
```

**CentOS/RHEL:**
```bash
sudo yum install -y yum-utils
sudo yum-config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
sudo yum install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin python3 git
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker $USER
```

---

## 🚀 快速部署步骤

### 1. 克隆代码

```bash
# 克隆到推荐目录
git clone <repository-url> /opt/recommend
cd /opt/recommend
```

### 2. 准备 JSON 数据文件

将 JSON 数据文件放到 `data/dianshu_data/` 目录：

```bash
# 创建数据目录
mkdir -p data/dianshu_data

# 将 JSON 文件复制到此目录
# 需要以下文件：
# - user.json
# - dataset.json
# - task.json
# - api_order.json
# - dataset_image.json
```

**JSON 文件格式示例：**
```json
[
  {
    "id": 1,
    "user_name": "张三",
    "update_time": "2025-10-16T14:00:00"
  }
]
```

### 3. 配置环境变量

```bash
# 复制配置模板
cp .env.example .env

# 编辑配置文件
vim .env
```

**必要的配置项：**

```ini
# ============ 数据源配置（JSON 模式）============
DATA_SOURCE=json
DATA_JSON_DIR=/opt/recommend/data/dianshu_data

# ============ Redis 配置 ============
REDIS_URL=redis://redis:6379/0
FEATURE_REDIS_URL=redis://redis:6379/1

# ============ MLflow 配置 ============
MLFLOW_TRACKING_URI=http://mlflow:5000
MLFLOW_EXPERIMENT_NAME=dataset_recommendation

# ============ 企业微信通知配置（可选）============
WEIXIN_CORP_ID=your_corp_id
WEIXIN_CORP_SECRET=your_corp_secret
WEIXIN_AGENT_ID=1000019
WEIXIN_DEFAULT_USER=YourName
```

**重要提示：**
- `DATA_JSON_DIR` 必须是**绝对路径**
- 如果使用 Docker，建议使用容器内的路径：`/app/data/dianshu_data`
- 企业微信配置是可选的，用于告警通知

### 4. 安装 Python 依赖

```bash
# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 安装依赖
pip install --upgrade pip
pip install -r requirements.txt

# 可选：向量检索支持
pip install faiss-cpu skl2onnx
```

### 5. 运行数据处理 Pipeline

```bash
# 设置 Python 路径
export PYTHONPATH=/opt/recommend:$PYTHONPATH

# 方式 A：一键运行完整 Pipeline（推荐）
bash scripts/run_pipeline.sh

# 方式 B：分步执行（调试用）
python3 -m pipeline.extract_load      # 数据抽取
python3 -m pipeline.clean_data        # 数据清洗
python3 -m pipeline.build_features_v2 # 特征工程
python3 -m pipeline.train_models      # 模型训练
python3 -m pipeline.recall_engine_v2  # 召回引擎
python3 -m pipeline.evaluate_quality_v2  # 质量评估
```

**预计执行时间：** 30-60 分钟（取决于数据量）

### 6. 启动 Docker 服务

```bash
# 启动所有服务
docker compose up -d

# 查看服务状态
docker compose ps

# 查看日志
docker compose logs -f
```

**启动的服务：**
| 服务名 | 端口 | 说明 |
|--------|------|------|
| redis | 6379 | 缓存和特征存储 |
| mlflow | 5000 | 模型管理 |
| recommendation-api | 8000 | 推荐 API |
| prometheus | 9090 | 监控指标采集 |
| grafana | 3000 | 监控看板 |
| alertmanager | 9093 | 告警管理 |
| notification-gateway | 9000 | 企业微信通知 |
| airflow-webserver | 8080 | 数据流水线 UI |
| airflow-scheduler | - | 调度器（后台） |
| postgres-airflow | 5432 | Airflow 元数据库 |

### 7. 验证部署

```bash
# 1. 健康检查
curl http://localhost:8000/health

# 预期输出：
# {
#   "status": "healthy",
#   "cache": "enabled",
#   "models_loaded": true
# }

# 2. 测试推荐接口
curl "http://localhost:8000/similar/123?top_n=10"

# 3. 测试企业微信通知（可选）
curl -X POST http://localhost:9000/test

# 4. 查看监控面板
# Prometheus: http://localhost:9090
# Grafana: http://localhost:3000 (admin/admin)
# MLflow: http://localhost:5000
# Airflow: http://localhost:8080 (admin/admin)
```

---

## 📁 目录结构

部署完成后的目录结构：

```
/opt/recommend/
├── data/
│   ├── dianshu_data/          # JSON 数据源（输入）
│   │   ├── user.json
│   │   ├── dataset.json
│   │   ├── task.json
│   │   ├── api_order.json
│   │   └── dataset_image.json
│   ├── business/              # 业务数据（Parquet）
│   ├── cleaned/               # 清洗后数据
│   ├── processed/             # 特征数据
│   └── evaluation/            # 评估报告
├── models/                    # 模型文件
│   ├── item_sim_behavior.pkl
│   ├── item_sim_content.pkl
│   ├── rank_model.pkl
│   ├── rank_model.onnx
│   └── model_registry.json
├── logs/                      # 日志文件
├── notification_gateway/      # 企业微信通知服务
├── monitoring/                # 监控配置
├── airflow/                   # Airflow DAGs
├── .env                       # 环境变量配置
└── docker-compose.yml         # Docker 服务编排
```

---

## 🔧 常见问题

### 1. JSON 数据文件找不到

**错误：** `FileNotFoundError: data/dianshu_data/user.json`

**解决：**
```bash
# 检查文件是否存在
ls -lh data/dianshu_data/

# 确保 .env 中的路径正确
cat .env | grep DATA_JSON_DIR

# 确保文件命名正确（必须是 user.json，不是 users.json）
```

### 2. 模型加载失败

**错误：** `models_loaded: false`

**解决：**
```bash
# 检查模型文件
ls -lh models/

# 重新训练模型
bash scripts/run_pipeline.sh

# 检查 models/model_registry.json
cat models/model_registry.json
```

### 3. Redis 连接失败

**错误：** `redis.exceptions.ConnectionError`

**解决：**
```bash
# 检查 Redis 容器
docker compose ps redis

# 查看 Redis 日志
docker compose logs redis

# 重启 Redis
docker compose restart redis

# 测试连接
redis-cli ping
```

### 4. Docker 端口冲突

**错误：** `Bind for 0.0.0.0:8000 failed: port is already allocated`

**解决：**
```bash
# 查看端口占用
sudo netstat -tlnp | grep 8000

# 方式 1：停止占用端口的服务
sudo kill <PID>

# 方式 2：修改 docker-compose.yml 改用其他端口
# 例如将 8000:8000 改为 8001:8000
```

### 5. 企业微信发送失败（错误码 60020）

**错误：** `not allow to access from your ip`

**解决：**
1. 登录企业微信管理后台：https://work.weixin.qq.com/
2. 进入"应用管理" -> 找到对应应用
3. 配置"企业可信 IP"，添加服务器公网 IP
4. 保存后重试

---

## 🔄 日常运维

### 更新数据

```bash
# 1. 将新的 JSON 文件放入 data/dianshu_data/
# 支持增量文件：dataset_20251016_140000.json

# 2. 重新运行 Pipeline
bash scripts/run_pipeline.sh

# 3. 重启 API（自动加载新模型）
docker compose restart recommendation-api
```

### 查看日志

```bash
# 查看 API 日志
docker compose logs -f recommendation-api

# 查看通知网关日志
docker compose logs -f notification-gateway

# 查看 Airflow 日志
docker compose logs -f airflow-scheduler
```

### 备份

```bash
# 备份模型
tar -czf models_backup_$(date +%Y%m%d).tar.gz models/

# 备份数据
tar -czf data_backup_$(date +%Y%m%d).tar.gz data/processed/

# 备份配置
cp .env .env.backup
```

### 停止服务

```bash
# 停止所有服务
docker compose down

# 停止并删除所有数据
docker compose down -v
```

---

## 📊 性能基准

| 指标 | 参考值 |
|------|--------|
| API QPS | > 100 (单 worker) |
| P95 延迟 | < 100ms (有缓存) |
| P99 延迟 | < 500ms |
| 缓存命中率 | > 80% |
| 内存占用 | < 4GB (含 Redis) |
| Pipeline 执行时间 | 30-60 分钟 |

---

## 🔗 相关文档

- [完整部署清单](DEPLOYMENT_CHECKLIST.md) - 详细的部署检查清单
- [JSON 数据源说明](JSON_DATA_SOURCE.md) - JSON 文件格式详解
- [API 接口文档](API_REFERENCE.md) - API 使用说明
- [运维手册](OPERATIONS_SOP.md) - 日常运维指南
- [企业微信通知配置](../notification_gateway/README.md) - Alertmanager 通知设置

---

## ✅ 部署检查清单

部署完成后，请确认以下项目：

- [ ] JSON 数据文件已准备（user.json, dataset.json 等）
- [ ] .env 配置文件已正确配置
- [ ] Python 依赖已安装
- [ ] Pipeline 成功执行，生成模型文件
- [ ] Docker 服务全部启动
- [ ] API 健康检查返回 healthy
- [ ] 推荐接口正常返回结果
- [ ] Redis 缓存正常工作
- [ ] Prometheus 指标可访问
- [ ] （可选）企业微信通知配置完成

---

**部署时间估计：** 1-2 小时（首次部署）

**维护者：** 推荐系统团队
**最后更新：** 2025-10-16
**版本：** v1.0
