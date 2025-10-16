# 快速开始指南

## 概述

本指南帮助你快速启动升级后的推荐系统（Phase 1完成版本）。

## 前置要求

- Python 3.8+
- Docker & Docker Compose
- （可选）Kubernetes集群

## 方式1：Docker Compose（推荐用于测试）

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 启动所有服务

```bash
# 启动Redis、Prometheus、Grafana、MLflow
docker-compose up -d

# 查看服务状态
docker-compose ps
```

### 3. 启动推荐API

```bash
# 开发模式（热重载）
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# 生产模式（多worker）
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### 4. 验证服务

```bash
# 健康检查
curl http://localhost:8000/health

# API文档
open http://localhost:8000/docs

# Prometheus指标
curl http://localhost:8000/metrics

# 测试推荐接口
curl "http://localhost:8000/recommend/detail/1?user_id=100&limit=10"
```

### 5. 访问监控面板

- **API文档**: http://localhost:8000/docs
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (用户名: admin, 密码: admin)
- **MLflow**: http://localhost:5000

---

## 方式2：Kubernetes部署

### 1. 构建并推送镜像

```bash
# 构建镜像
docker build -t recommendation-api:v1.0 .

# 推送到镜像仓库（替换为你的仓库地址）
docker tag recommendation-api:v1.0 your-registry.com/recommendation-api:v1.0
docker push your-registry.com/recommendation-api:v1.0
```

### 2. 配置Secrets

```bash
# 编辑 k8s/secret.yaml，添加你的数据库凭证
# 生成base64编码
echo -n 'your-database-user' | base64
echo -n 'your-database-password' | base64
```

### 3. 部署到K8s

```bash
# 创建命名空间和配置
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/secret.yaml
kubectl apply -f k8s/configmap.yaml

# 部署Redis
kubectl apply -f k8s/redis-deployment.yaml

# 部署推荐服务
kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml

# （可选）配置Ingress
kubectl apply -f k8s/ingress.yaml
```

### 4. 验证部署

```bash
# 查看Pod状态
kubectl get pods -n recommendation

# 查看服务
kubectl get svc -n recommendation

# 查看日志
kubectl logs -f deployment/recommendation-api -n recommendation

# 端口转发（本地测试）
kubectl port-forward svc/recommendation-api 8000:80 -n recommendation
```

---

## 环境变量配置

创建 `.env` 文件：

```bash
# 数据库配置
DATABASE_URL=mysql+pymysql://user:password@localhost:3306/dianshu_backend
MATOMO_DB_URL=mysql+pymysql://user:password@localhost:3306/matomo

# Redis配置
REDIS_URL=redis://localhost:6379/0
# 或者分开配置
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# MLflow配置
MLFLOW_TRACKING_URI=http://localhost:5000

# 日志配置
LOG_LEVEL=INFO

# 监控配置（可选）
ENABLE_METRICS=true
```

---

## 测试API端点

### 1. 健康检查

```bash
curl http://localhost:8000/health
```

响应：
```json
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

### 2. 相似推荐

```bash
curl http://localhost:8000/similar/123?limit=10
```

### 3. 个性化推荐

```bash
curl "http://localhost:8000/recommend/detail/123?user_id=456&limit=10"
```

### 4. 热门榜单

```bash
# 1小时热门
curl "http://localhost:8000/hot/trending?limit=20&timeframe=1h"

# 24小时热门
curl "http://localhost:8000/hot/trending?limit=20&timeframe=24h"
```

### 5. Prometheus指标

```bash
curl http://localhost:8000/metrics
```

---

## 性能测试

### 使用Apache Bench

```bash
# 测试QPS（1000请求，10并发）
ab -n 1000 -c 10 http://localhost:8000/similar/123?limit=10
```

### 使用wrk

```bash
# 测试60秒（10个连接，2个线程）
wrk -t2 -c10 -d60s http://localhost:8000/similar/123?limit=10
```

---

## 监控与告警

### Prometheus查询示例

访问 http://localhost:9090/graph

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

### Grafana仪表板

1. 访问 http://localhost:3000
2. 登录（admin/admin）
3. 导入仪表板：`monitoring/grafana/dashboards/recommendation-overview.json`

---

## 故障排查

### 问题1: Redis连接失败

```bash
# 检查Redis是否运行
docker-compose ps redis

# 查看Redis日志
docker-compose logs redis

# 测试Redis连接
redis-cli ping
```

### 问题2: 模型文件缺失

```bash
# 检查models目录
ls -la models/

# 运行pipeline生成模型
python pipeline/train_models.py
```

### 问题3: 高延迟

```bash
# 查看Prometheus指标
curl http://localhost:8000/metrics | grep latency

# 检查缓存状态
curl http://localhost:8000/health
```

---

## 下一步

1. **数据准备**: 运行完整的数据pipeline
   ```bash
   bash scripts/run_pipeline.sh
   ```

2. **模型训练**: 训练新模型
   ```bash
   python pipeline/train_models.py
   ```

3. **性能优化**: 根据监控数据调整配置
   - 调整Redis缓存TTL
   - 调整K8s资源限制
   - 配置HPA阈值

4. **告警配置**: 配置实际的告警接收器
   - 编辑 `monitoring/alertmanager.yml`
   - 添加Slack/Email/企业微信配置

5. **开始Phase 2**: 数据质量与特征工程升级
   - 参考 `docs/PRODUCTION_UPGRADE_PLAN.md` Phase 2部分

---

## 常用命令

```bash
# Docker Compose
docker-compose up -d          # 启动所有服务
docker-compose down           # 停止所有服务
docker-compose logs -f        # 查看日志
docker-compose restart api    # 重启API服务

# Kubernetes
kubectl get pods -n recommendation                    # 查看Pod
kubectl logs -f pod-name -n recommendation           # 查看日志
kubectl describe pod pod-name -n recommendation      # 详细信息
kubectl delete pod pod-name -n recommendation        # 重启Pod

# 本地开发
uvicorn app.main:app --reload                        # 开发模式
pytest tests/                                         # 运行测试
```

---

## 支持

- 📖 完整文档: `docs/`
- 🐛 问题反馈: GitHub Issues
- 📊 监控: Grafana仪表板

**祝使用愉快！** 🚀
