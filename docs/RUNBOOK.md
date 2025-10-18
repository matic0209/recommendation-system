# 运维手册（中文）

本手册面向日常运维人员，总结推荐系统在生产环境的常见操作、监控以及排错方式。

---

## 1. 关键服务与职责

| 服务 | 说明 | 日常关注点 |
|------|------|------------|
| `recommendation-api` | 在线推荐接口，FastAPI + Uvicorn | 健康检查、日志、模型加载状态 |
| `airflow-webserver / scheduler` | 管理 DAG、调度离线任务 | DAG 运行情况、任务失败告警 |
| `mlflow` | 模型训练记录与产物存储 | 训练 run 是否写入成功、artifact 权限 |
| `redis` | 缓存与实验位 | 连接数、内存占用 |
| `notification-gateway` | 企业微信告警中转 | `/health` 是否 200，企业微信凭据是否有效 |
| `prometheus / grafana / alertmanager` | 指标采集、可视化与告警 | 采集目标状态、告警通道是否可用 |
| `postgres-airflow` | Airflow 元数据库 | 磁盘容量、备份 |

---

## 2. 常用命令与入口

### 2.1 Docker Compose

```bash
docker compose ps                     # 查看整体状态
docker compose logs <service> -f      # 实时日志
docker compose up -d                  # 启动所有服务
docker compose down                   # 停止服务
```

### 2.2 Airflow CLI

```bash
docker compose exec airflow-webserver airflow dags list
docker compose exec airflow-webserver airflow dags list-runs -d recommendation_pipeline
docker compose exec airflow-webserver airflow tasks run recommendation_pipeline train_models 2025-10-18T00:00:00+00:00 --ignore-all-dependencies
docker compose exec airflow-webserver airflow tasks test recommendation_pipeline evaluate 2025-10-18T00:00:00+00:00
```

### 2.3 监控入口

- Grafana：`http://<host>:<GRAFANA_HOST_PORT>`（默认 3000，初始账号 `admin/admin`）  
- Prometheus：`http://<host>:<PROMETHEUS_HOST_PORT>`  
- MLflow：`http://<host>:<MLFLOW_HOST_PORT>`  
- Airflow：`http://<host>:<AIRFLOW_WEB_HOST_PORT>`  
- Alertmanager：`http://<host>:<ALERTMANAGER_HOST_PORT>`  
- Prometheus Targets：`http://<host>:<PROMETHEUS_HOST_PORT>/targets`

---

## 3. 日常巡检清单

1. **推荐 API**
   - `curl http://<host>:<port>/health` → `{"status":"healthy","models_loaded":true}`  
   - `./smoke_test.sh` 返回推荐结果。

2. **Airflow DAG**
   - Web UI 可视化 DAG 树，确认 `recommendation_pipeline` 成功率；  
   - 关注红色失败节点，进入日志查看原因；  
   - DAG run 失败可通过 `tasks run` 手动重试。

3. **监控与告警**
   - Grafana 仪表盘数据是否刷新；  
   - Prometheus Targets 是否全部 `UP`；  
   - Alertmanager 告警渠道是否正常（可发送测试告警）；  
   - `notification-gateway` `/health` 返回 200。

4. **模型 & 数据**
   - `models/` 目录下新模型时间戳是否更新；  
   - `data/evaluation/summary.json`、`exposure_metrics.json` 是否生成；  
   - `mlflow` Run 是否记录成功（尤其 artifacts）。

5. **资源使用**
   - 宿主机磁盘、CPU、内存；  
   - Redis、Postgres（Airflow）容量与备份计划。

---

## 4. 手动执行流程

### 4.1 全量重跑

```bash
docker compose exec airflow-webserver \
  airflow dags trigger recommendation_pipeline --conf '{"full_refresh": true}'
```

或在宿主机执行：
```bash
source venv/bin/activate
export PYTHONPATH=$(pwd)
bash scripts/run_pipeline.sh --full-refresh
```

### 4.2 单任务重跑

1. 查询可用的 `execution_date`：
   ```bash
   docker compose exec airflow-webserver airflow dags list-runs -d recommendation_pipeline
   ```
2. 指定任务重跑：
   ```bash
   docker compose exec airflow-webserver \
     airflow tasks run recommendation_pipeline train_models 2025-10-18T00:00:00+00:00 --ignore-all-dependencies
   ```
   若仅调试，可使用 `airflow tasks test`。

---

## 5. 常见故障与排查

| 症状 | 排查步骤 | 处理建议 |
|------|----------|----------|
| `CLIP` 模型再次访问 Hugging Face | 检查环境变量 `CLIP_MODEL_PATH`、`HF_HUB_OFFLINE=1`；确认 `/opt/recommend/cache/sentence-transformers/clip-ViT-B-32` 目录存在且可读写 | 将模型目录下所有文件复制完整并 `chown -R 50000:50000` |
| Airflow 报 `JSON data directory not found` | 查看 `.env` 中 `DATA_JSON_DIR`；确认容器 `volumes` 挂载了该路径；`docker compose exec airflow-webserver ls <目录>` | 修正 `.env` 并在 `docker-compose.yml` 中挂载 |
| `Permission denied` 写模型或数据 | 检查路径属主；执行 `sudo chown -R 50000:50000 <path>` | 对 `models/`、`data/`、`/var/lib/docker/volumes/recommend_mlflow-data/_data` 等目录授权 |
| `evaluate` 出现 `feature names missing` | 说明图片特征未生成，或 `_compute_ranking_features` 未补齐列 | 重新跑 `build_features`，或确认代码包含可选列补充 |
| `pandas Invalid comparison between tz-aware and tz-naive` | Matomo 行为数据含时区，已在代码统一转 UTC | 若仍报错，检查原始时间是否为空或格式混乱 |
| 企业微信告警未收到 | 查看 `notification-gateway` 日志 `/health`；确认企业微信凭据有效 | 更新 `.env` 中的 `WEIXIN_*`，确保服务能访问外网或内网代理 |
| MLflow artifacts `Permission denied` | `/mlflow` volume 权限不足 | `docker compose down` → `sudo chown -R 50000:50000 /var/lib/docker/volumes/recommend_mlflow-data/_data` → `docker compose up -d` |

---

## 6. 模型与缓存维护

- **离线模型更新**：将新的 `.pkl` / `.json` 文件写入 `models/` 后，调用 API 的 `/reload/primary`（如实现）或重启 `recommendation-api` 容器。  
- **CLIP 缓存**：存放于 `/opt/recommend/cache/sentence-transformers` 和 `/opt/recommend/cache/huggingface`，建议定期备份或打包。  
- **Redis 清理**：必要时进入容器执行 `redis-cli FLUSHALL`，但要注意实验位数据被清空。  
- **MLflow Run 清理**：可在 Web UI 删除 Run，也可以删除 `mlflow-data` 中的对应目录并同步数据库状态。

---

## 7. 监控与告警建议

1. **推荐 API 指标**  
   - QPS、响应时间、失败率（`/metrics` 暴露 Prometheus 指标）  
   - 模型版本、缓存命中率

2. **业务指标**  
   - 曝光数、点击数、CTR / CVR（评估阶段写入 `summary.json`）  
   - 不同算法版本的分布，可在 Grafana 设置字段切换

3. **资源监控**  
   - Docker 容器 CPU / 内存 / 磁盘 IO  
   - 数据目录与模型目录的磁盘占用

4. **告警通道**  
   - Alertmanager → `notification-gateway` → 企业微信  
   - 可配置心跳检测（如 API 失败率大于阈值、DAG 多次失败等）

---

## 8. 灾备与回滚

- **镜像回滚**：保留旧版 `docker save` tar 包或内网镜像标签，可随时 `docker load` / `docker compose up`。  
- **模型回滚**：`models/` 目录保留历史文件（可按时间戳备份），必要时手动替换。  
- **数据备份**：`data/`、`models/`、`mlflow-data` 建议定期备份到对象存储或 NAS。  
- **配置备份**：`.env.prod`、`monitoring/`、`docs/` 等文本文件纳入版本管理。

---

## 9. 日志定位

| 模块 | 日志位置 | 备注 |
|------|----------|------|
| 推荐 API | 宿主机 `logs/` 或容器 `docker compose logs recommendation-api` | 包含请求 ID、模型加载状态 |
| Airflow | `airflow/logs/` 或 Web UI 中查看 Task Log | Task 失败时首要线索 |
| Pipeline 脚本 | `data/_metadata/extract_metrics.json`、`extract_state.json` | 抽取水位、表行数等 |
| 评估/曝光 | `data/evaluation/*.json`、`exposure_log.jsonl` | 推荐效果 |
| 告警网关 | `docker compose logs notification-gateway` | 是否与企业微信通信失败 |

---

## 10. 维护建议

- 定期回顾 DAG 成功率，尤其是 `build_features`、`train_models`、`evaluate`。  
- 关注 `recommendation-api` WARN 级别日志（如缺少特征、模型熔断、CLIP 下载失败），及时排查。  
- 每季度审核 Docker Compose 版本依赖，统一处理 `version` 字段提醒。  
- 将部署/运维脚本纳入版本控制，避免操作口口相传。  
- 维护离线镜像与模型缓存的“白名单”及 MD5 校验，防止包错或缺失文件。  
- 对接企业微信告警时，如 IP 限制严格可使用跳板机或 API Gateway。

---

通过本 Runbook，可以快速定位常见问题、执行必要操作。建议结合《部署指南》《故障排查》一起阅读，并根据实际生产情况持续补充。欢迎在使用过程中将新的问题与经验回写至本文档。祝运维顺利！
