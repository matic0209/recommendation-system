# 推荐系统生产环境 Docker 部署指引

本文档说明如何在生产环境中使用 Docker Compose 部署推荐系统，并避免常见端口冲突。

---

## 1. 环境准备
- 操作系统：Linux (Ubuntu 20.04+/CentOS 8+)
- 依赖组件：Docker 20.10+、Docker Compose v2、Git
- 确保宿主机具备充足存储（数据与模型均通过 volume 挂载）

---

## 2. 准备配置文件

```bash
cd /opt/recommend
cp .env.example .env.prod
```

编辑 `.env.prod`，重点修改：

| 配置项 | 说明 |
| ------ | ---- |
| `DATA_JSON_DIR` | 宿主机 JSON 数据目录绝对路径，例如 `/data/dianshu_data` |
| `MLFLOW_TRACKING_URI` | 建议设为 `file://./mlruns`（默认值即可） |
| `RECOMMEND_API_HOST_PORT` 等端口变量 | 见下一节，务必分配不冲突的端口 |
| 企业微信变量 | 如需通知功能，填写正式的企业微信凭据 |

**推荐的端口示例（可按需调整）：**

```ini
RECOMMEND_API_HOST_PORT=18080
AIRFLOW_WEB_HOST_PORT=18081
MLFLOW_HOST_PORT=15000
PROMETHEUS_HOST_PORT=19090
ALERTMANAGER_HOST_PORT=19093
GRAFANA_HOST_PORT=13000
NOTIFICATION_GATEWAY_HOST_PORT=19000
REDIS_HOST_PORT=16379
```

如需进一步隔离，可结合防火墙或 Nginx/Traefik 仅对外暴露必要端口（通常只有推荐 API）。

---

## 3. 首次运行前的准备
1. 将全量/增量 JSON 数据放入 `DATA_JSON_DIR`
2. 在宿主机执行一次模型管线（需约 30~60 分钟）：

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   export PYTHONPATH=/opt/recommend:$PYTHONPATH
   bash scripts/run_pipeline.sh --full-refresh
   ```

3. 确认模型文件生成在 `models/` 目录

---

## 4. 启动生产环境

```bash
docker compose --env-file .env.prod up -d
```

验证：

```bash
docker compose --env-file .env.prod ps
curl http://<宿主机IP>:${RECOMMEND_API_HOST_PORT}/health
```

若使用反向代理，请确保将 `/health`、`/similar/*` 等接口正确转发到容器 `recommendation-api:8000`。

---

## 5. 日志与维护
- 查看 API 日志：`docker compose logs recommendation-api -f`
- 查看 Airflow Web：`http://<宿主机IP>:${AIRFLOW_WEB_HOST_PORT}`
- 查看监控：
  - Grafana：`http://<宿主机IP>:${GRAFANA_HOST_PORT}`（默认账号 `admin/admin`）
  - Prometheus：`http://<宿主机IP>:${PROMETHEUS_HOST_PORT}`
- 停止服务：`docker compose --env-file .env.prod down`

建议将 `.env.prod` 与敏感凭据保存在受控位置，并定期备份 `data/`、`models/`、`mlruns/` 目录。
