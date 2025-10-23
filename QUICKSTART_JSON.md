# 快速开始 - Docker 纯容器化部署

## 🎯 新服务器 4 步快速部署

```bash
# 1. 克隆代码
git clone <repository-url> /opt/recommend && cd /opt/recommend

# 2. 准备 JSON 数据（放入 data/dianshu_data/ 目录）

# 3. 配置 .env（重点：DATA_JSON_DIR 使用宿主机绝对路径）
cp .env.example .env && vim .env

# 4. 初始训练 + 启动 Docker
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
bash scripts/run_pipeline.sh
docker compose up -d
```

**完成！** 系统会自动每小时更新增量数据。

---

## 📋 详细步骤

### 前提条件
- Docker + Docker Compose v2
- Python 3.8+
- JSON 数据文件（全量 + 增量）

### 1. 克隆代码

```bash
git clone <repository-url> /opt/recommend
cd /opt/recommend
```

### 2. 准备 JSON 数据

JSON 数据放在**宿主机**，Docker 会自动挂载：

```bash
# 确保数据目录存在
mkdir -p /opt/recommend/data/dianshu_data

# 复制 JSON 文件（全量）
# - user.json
# - dataset.json
# - task.json
# - api_order.json
# - dataset_image.json

# 增量文件（可选，系统会自动处理）
# - user_20251016_140000.json
# - dataset_20251016_150000.json
```

### 3. 配置 .env

```bash
cp .env.example .env
vim .env
```

**关键配置：**

```ini
# 数据源（宿主机绝对路径）
DATA_SOURCE=json
DATA_JSON_DIR=/opt/recommend/data/dianshu_data

# 企业微信（可选）
WEIXIN_CORP_ID=your_corp_id
WEIXIN_CORP_SECRET=your_secret
WEIXIN_DEFAULT_USER=YourName
```

### 4. 初始训练

首次部署需要在宿主机生成模型：

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=/opt/recommend:$PYTHONPATH
bash scripts/run_pipeline.sh
```

### 5. 启动 Docker

```bash
docker compose up -d
```

### 6. 验证

```bash
# 健康检查
curl http://localhost:8000/health

# 测试推荐
curl "http://localhost:8000/similar/123?top_n=10"
```

### 核心服务端口

| 服务 | 端口 | 说明 |
|------|------|------|
| 推荐 API | 8000 | 主要服务 |
| Grafana | 3000 | 监控看板 (admin/admin) |
| Prometheus | 9090 | 指标采集 |
| MLflow | 5000 | 模型管理 |
| Airflow | 8080 | 数据流水线 |
| 企业微信通知 | 9000 | 告警通知 |

### 测试接口

```bash
# 1. 健康检查
curl http://localhost:8000/health

# 2. 相似推荐（基于物品）
curl "http://localhost:8000/similar/123?top_n=10"

# 3. 个性化推荐（基于用户+物品）
curl "http://localhost:8000/recommend/detail/123?user_id=456&top_n=10"

# 4. 热门榜单
curl "http://localhost:8000/hot/trending?timeframe=24h&top_n=20"

# 5. Prometheus 指标
curl http://localhost:8000/metrics
```

---

## 🔄 增量数据自动更新

系统已配置 **Airflow DAG 每小时自动**处理增量数据：

**增量文件命名格式：**
```
user_20251016_140000.json      # 2025-10-16 14:00
dataset_20251016_150000.json   # 2025-10-16 15:00
```

**自动流程：**
1. Airflow 每小时执行 `incremental_data_update` DAG
2. 读取新的增量 JSON 文件
3. 合并到全量数据
4. 更新特征和模型
5. 重载 API 和清理缓存

**查看运行状态：**
- Airflow Web UI: http://localhost:8080 (admin/admin)

**手动触发更新：**
```bash
docker compose exec airflow-scheduler \
  airflow dags trigger incremental_data_update
```

---

## 📞 企业微信通知（可选）

1. 编辑 `.env`：
```ini
WEIXIN_CORP_ID=企业ID
WEIXIN_CORP_SECRET=应用Secret
WEIXIN_AGENT_ID=1000019
WEIXIN_DEFAULT_USER=接收人
```

2. 在企业微信后台配置服务器 IP 白名单

3. 测试：
```bash
curl -X POST http://localhost:9000/test
```

### 常见问题

**Q: Pipeline 执行失败？**
```bash
# 检查 JSON 文件是否存在
ls -lh data/dianshu_data/

# 检查 Python 路径
echo $PYTHONPATH
```

**Q: 模型加载失败？**
```bash
# 检查模型文件
ls -lh models/

# 重新训练
bash scripts/run_pipeline.sh
```

**Q: Docker 端口冲突？**
```bash
# 修改 docker-compose.yml 中的端口映射
# 例如：8000:8000 改为 8001:8000
```

### 数据更新流程

```bash
# 1. 放入新的 JSON 文件到 data/dianshu_data/

# 2. 重新运行 Pipeline
bash scripts/run_pipeline.sh

# 3. 重启 API
docker compose restart recommendation-api
```

### 完整文档

详细部署指南请参考：[docs/DEPLOYMENT_GUIDE_JSON.md](docs/DEPLOYMENT_GUIDE_JSON.md)

---

**预计部署时间：** 1-2 小时
**技术支持：** 推荐系统团队
