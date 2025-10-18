# Docker 离线部署指南（中文）

本文详细说明如何在生产环境（含离线场景）部署推荐系统。假定线上环境已禁止外网访问，需要提前准备好镜像、模型及依赖。

---

## 1. 部署前准备

### 1.1 资源需求
| 组件 | 默认端口 | 说明 |
|------|----------|------|
| recommendation-api | 8000 | 在线推荐服务 |
| Airflow Web UI      | 8080 | DAG 管理界面 |
| MLflow              | 5000 | 模型训练记录 |
| Redis               | 6379 | 缓存 / 实验位 |
| Prometheus          | 9090 | 指标采集 |
| Alertmanager        | 9093 | 告警转发 |
| Grafana             | 3000 | 指标可视化 |

> 所有端口可在 `.env` 中通过 `*_HOST_PORT` 重写。

### 1.2 目录约定
| 宿主机路径 | 用途 | 权限建议 |
|------------|------|----------|
| `/opt/recommend/cache/sentence-transformers` | CLIP 模型缓存 | `chown -R 50000:50000` |
| `/opt/recommend/cache/huggingface`           | Hugging Face 缓存 | `chown -R 50000:50000` |
| `/var/lib/docker/volumes/recommend_mlflow-data/_data` | MLflow 存储 | `chown -R 50000:50000` |
| `<项目>/models` & `<项目>/data` | 模型 / 数据输出 | `chown -R 50000:50000` |

---

## 2. 在线机器准备镜像

1. 克隆仓库，安装 Docker & Compose；
2. 构建推荐 API 与 Airflow 镜像：
   ```bash
   docker compose build recommendation-api
   docker compose build airflow-webserver airflow-scheduler airflow-init
   ```
3. 保存镜像：
   ```bash
   docker save recommendation-recommendation-api:latest > recommendation-api.tar
   docker save recommend-airflow:latest > recommend-airflow.tar
   docker save grafana/grafana:latest prom/prometheus:latest prom/alertmanager:latest \
              redis:7-alpine ghcr.io/mlflow/mlflow:v2.16.0 postgres:15 > deps.tar
   ```

> 若使用内网镜像仓库，可改为 `docker tag` + 推送到内网仓库。

---

## 3. 下载并打包 CLIP 模型

1. 在线环境安装：
   ```bash
   pip install --upgrade huggingface_hub sentence-transformers
   huggingface-cli download sentence-transformers/clip-ViT-B-32 \
     --local-dir ./clip-ViT-B-32 --local-dir-use-symlinks False
   ```
2. 打包模型目录：
   ```bash
   tar czf clip-ViT-B-32.tar.gz clip-ViT-B-32
   ```

---

## 4. 传输到生产环境

- 将 `recommendation-api.tar`、`recommend-airflow.tar`、`deps.tar`、`clip-ViT-B-32.tar.gz` 通过内网或介质拷贝到生产机某目录（如 `/opt/packages`）。
- 如果有 JSON 数据或配置文件，也一并复制。

---

## 5. 生产机导入镜像与模型

```bash
docker load < recommendation-api.tar
docker load < recommend-airflow.tar
docker load < deps.tar
```

解压模型并设置权限：
```bash
sudo mkdir -p /opt/recommend/cache/sentence-transformers
sudo mkdir -p /opt/recommend/cache/huggingface/hub
sudo tar xzf clip-ViT-B-32.tar.gz -C /opt/recommend/cache/sentence-transformers/
sudo chown -R 50000:50000 /opt/recommend/cache
```

---

## 6. 配置环境变量与目录

### 6.1 `.env.prod` 示例
```ini
DATA_SOURCE=json
BUSINESS_DATA_SOURCE=json
MATOMO_DATA_SOURCE=database

DATA_JSON_DIR=/dianshu/backup/data/dianshu_data/jsons
CLIP_MODEL_PATH=/opt/recommend/cache/sentence-transformers/clip-ViT-B-32
SENTENCE_TRANSFORMERS_HOME=/opt/recommend/cache/sentence-transformers
HF_HOME=/opt/recommend/cache/huggingface
TRANSFORMERS_CACHE=/opt/recommend/cache/huggingface/hub
HF_HUB_OFFLINE=1

RECOMMEND_API_HOST_PORT=18080
AIRFLOW_WEB_HOST_PORT=18081
MLFLOW_HOST_PORT=15000
PROMETHEUS_HOST_PORT=19090
ALERTMANAGER_HOST_PORT=19093
GRAFANA_HOST_PORT=13000
NOTIFICATION_GATEWAY_HOST_PORT=19000
REDIS_HOST_PORT=16379
```

### 6.2 目录授权
```bash
sudo chown -R 50000:50000 models data
sudo chown -R 50000:50000 /var/lib/docker/volumes/recommend_mlflow-data/_data
sudo chown -R 50000:50000 /opt/recommend/cache
```

---

## 7. 启动 Docker 组件

```bash
docker compose --env-file .env.prod up -d
```

常用命令：
- 查看状态：`docker compose ps`
- 查看日志：`docker compose logs <service> -f`
- 停止服务：`docker compose down`

---

## 8. 首次全量训练

1. 进入宿主机（或 `airflow-webserver` 容器）执行：
   ```bash
   source venv/bin/activate
   export PYTHONPATH=$(pwd)
   bash scripts/run_pipeline.sh --full-refresh
   ```
   或在 Airflow 中触发 `recommendation_pipeline`，附加 `{"full_refresh": true}`。

2. 验证模型文件：`ls models/`，确认生成 `item_sim_behavior.pkl`、`rank_model.pkl` 等。

---

## 9. 验证与测试

```bash
./smoke_test.sh
# 或手动：
curl http://<host>:18080/health
curl "http://<host>:18080/similar/123?top_n=5"
curl "http://<host>:18080/recommend/detail/123?user_id=7&top_n=5"
```

若使用 Matomo + Grafana：
- Grafana：http://<host>:13000 (默认 admin / admin)  
- Prometheus：http://<host>:19090  
- MLflow：http://<host>:15000  
- Airflow：http://<host>:18081  
- Alertmanager：http://<host>:19093

---

## 10. 常见问题

| 问题 | 解决方案 |
|------|----------|
| CLIP 模型仍尝试联网 | 确认 `CLIP_MODEL_PATH`、`HF_HUB_OFFLINE=1` 已设；模型目录结构完整；容器挂载读写。 |
| 写模型报 `Permission denied` | `chown -R 50000:50000 models data /var/lib/docker/volumes/recommend_mlflow-data/_data` |
| JSON 目录找不到 | 在 Airflow 相关服务 `volumes` 中挂载 JSON 目录（只读）；`.env` 中路径与容器内一致。 |
| Matomo 时间比较报错 | 升级代码后会自动将 `server_time` 统一转 UTC；若仍有问题，检查 Matomo 原始数据是否为空。 |
| 告警收不到 | 检查 `monitoring/alertmanager.yml`、`notification_gateway` 配置及企业微信凭据。 |

---

## 11. 离线升级流程

1. 在在线机器重建镜像和模型 → 打包 → 拷贝。  
2. 生产机 `docker compose down` → `docker load` 新镜像 → `docker compose up -d`。  
3. 重跑 `build_features`、`train_models`（Airflow 或脚本）。  
4. 验证 API 和监控正常后，再次触发 `evaluate`、`reconcile_metrics`。

如需回滚，可使用旧镜像 tar 或保留的旧模型目录。

---

通过上述步骤，可在离线环境中稳定部署并维护推荐系统。建议把命令写成脚本或自动化流程，减少人工操作风险。欢迎结合运维手册持续完善。祝部署顺利！
