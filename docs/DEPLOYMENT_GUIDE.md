# 生产部署指南

本文档描述如何在生产环境部署推荐系统。部署默认使用随项目提供的 Docker Compose 栈，其中包含推荐 API、Airflow、Redis、MLflow、Prometheus/Alertmanager、Grafana 等组件。线上只需在 `.env` 中指定两个 MySQL 数据库（业务库、Matomo 行为库）的连接信息，其余服务将由 Compose 自动启动。如需复用公司已有中间件，可根据文档中的可选步骤覆盖默认设置。

---

## 1. 依赖清单

部署前请确认：

- **MySQL（业务库）**：包含 `dianshu_backend`（或等价库）数据，提供只读账号并允许创建索引。
- **MySQL（Matomo 行为库）**：提供 Matomo 行为数据，同样提供只读账号。
- **默认 Redis**：Compose 会启动内置 Redis 容器；如已有共享 Redis，可在 `.env` 中覆盖地址。
- **磁盘存储**：挂载 `data/`、`models/`、`logs/` 等目录，建议使用持久化卷或对象存储。
- **告警渠道（可选）**：若需推送至企业微信等，可按第 6 节配置 Alertmanager。

> 如需 MLflow、Airflow 等组件，亦可复用同一 Compose 文件，或单独部署至已有平台。

---

## 2. 配置 `.env`

拷贝项目根目录下的 `.env.example` 为 `.env`，至少需要设置以下变量指向生产数据库，其他项保持默认即可：

```bash
BUSINESS_DB_HOST=10.0.0.12          # 业务库 MySQL
BUSINESS_DB_PORT=3306
BUSINESS_DB_NAME=dianshu_backend
BUSINESS_DB_USER=recsys_readonly
BUSINESS_DB_PASSWORD=******         # 请确保具备 SELECT / 创建索引权限

MATOMO_DB_HOST=10.0.0.13
MATOMO_DB_PORT=3306
MATOMO_DB_NAME=matomo
MATOMO_DB_USER=recsys_readonly
MATOMO_DB_PASSWORD=******

# 若使用内置 Redis，可保留默认 redis://redis:6379/0；如需外部 Redis 再覆盖
REDIS_URL=redis://redis:6379/0
FEATURE_REDIS_URL=redis://redis:6379/1

MLFLOW_TRACKING_URI=http://mlflow.internal:5000   # 如无 MLflow 可留空或沿用 file:// 路径
LOG_LEVEL=INFO
```

如需 Airflow/Alertmanager 等服务，请同时补充对应账号（`AIRFLOW_USERNAME` / `AIRFLOW_PASSWORD` 等）。

---

## 3. 调整 Docker Compose

默认 `docker-compose.yml` 会启动以下服务：

| 服务 | 作用 | 监听端口 |
| --- | --- | --- |
| recommendation-api | FastAPI 推荐接口 | `8000` |
| redis | 特征/缓存存储 | `6379`（内部） |
| mlflow | 模型实验管理 | `5000` |
| airflow-* + postgres-airflow | 离线流水线调度 | `8080`（Web） |
| prometheus | 指标采集 | `9090` |
| alertmanager | 告警路由 | `9093` |
| grafana | 仪表盘展示 | `3000` |

如需使用公司已有 MySQL：

- 将 `.env` 中的 `BUSINESS_DB_HOST`、`MATOMO_DB_HOST` 等指向外部地址；
- 启动 Compose 时可跳过内置 `mysql-business`、`mysql-matomo` 服务，例如：  
  `docker compose up -d redis mlflow recommendation-api airflow-init postgres-airflow airflow-webserver airflow-scheduler prometheus alertmanager grafana`

若继续使用 Compose 自带 MySQL 仅用于本地演示，可保留默认配置。

---

## 4. 初始化步骤

### 4.1 安装索引与数据库准备

首次部署时，需要对业务库 / Matomo 库补充增量抽取所需索引。项目提供了自动化脚本：

```bash
# 以宿主机运行为例（需安装依赖）
pip install -r requirements.txt
python scripts/setup_database_indexes.py --apply
```

或通过容器执行：

```bash
docker compose run --rm recommendation-api python scripts/setup_database_indexes.py --apply
```

脚本会根据 `.env` 连接外部数据库，创建增量抽取使用的时间字段索引，并生成执行日志。

### 4.2 同步特征与模型产物

首次部署需跑一次离线流水线，以生成特征/模型：

```bash
# 全量执行抽取→特征→质量→训练→召回
docker compose run --rm recommendation-api ./scripts/run_pipeline.sh

# 如只需同步特征与模型（不重新抽取）
docker compose run --rm recommendation-api ./scripts/run_pipeline.sh --sync-only
```

流水线会将输出写入挂载的 `data/`、`models/` 目录，以及 Redis 特征库（若配置 `FEATURE_REDIS_URL`）。

### 4.3 启动在线服务

```bash
docker compose up -d recommendation-api
# 验证健康
curl http://<服务地址>:8000/health
```

如需多实例部署，请确保 `data/`、`models/`、`logs/` 目录使用共享存储或对象存储，并在负载均衡前配置健康检查。

### 4.4 调度与对账

- Airflow 已集成 `reconcile_metrics` 任务，会在每日流水线末尾写入 `data/evaluation/reconciliation_*.json`。
- 若未使用 Airflow，可改为 cron 定时：
  ```bash
  0 3 * * * docker compose run --rm recommendation-api \
      python -m scripts.reconcile_business_metrics --start $(date -d "yesterday" +\%F) --end $(date -d "yesterday" +\%F)
  ```

---

## 5. 服务使用与状态监控

| 组件 | 访问方式 | 常用检测 | 说明 |
| --- | --- | --- | --- |
| 推荐 API | `http://<主机>:8000` | `/health`、`/metrics` | 健康检查包含模型加载与 Redis 连接状态 |
| Airflow | `http://<主机>:8080` | DAG 状态、`reconcile_metrics` 任务 | 用于调度离线流水线与日级对账 |
| Redis | `docker compose exec redis redis-cli ping` | `redis-cli info` | 内置持久化至 `redis-data` 卷 |
| MLflow | `http://<主机>:5000` | UI 上查看实验/模型 | 可配置远程存储 |
| Prometheus | `http://<主机>:9090` | 查询 `data_quality_score` 等指标 | 采集 API 与离线快照指标 |
| Alertmanager | `http://<主机>:9093` | `/#/alerts` | 查看当前告警、静默策略 |
| Grafana | `http://<主机>:3000` | Dashboard `Recommendation Overview` | 默认账号 `admin`/`admin`（可在 `.env` 调整） |

监控指标：
- 在线服务：`recommendation_latency_seconds`、`recommendation_degraded_total`、`recommendation_timeouts_total`
- 数据质量：`data_quality_score{table=...}`、`data_schema_contract_status`
- 日级对账：`data/evaluation/reconciliation_*.json`（可扩展为 Prometheus 抓取）

日志路径挂载至宿主 `logs/`，Airflow 日志在 `airflow/logs/`。

---

## 6. Alertmanager 推送至企业微信

1. 在企业微信创建自建应用，获取 `CorpID`、`AgentId`、`Secret`。  
2. 将凭据写入部署环境变量或密钥文件（例如 `/opt/alertmanager/secrets.env`）。  
3. 修改 `monitoring/alertmanager.yml`，启用 `wechat_configs`：

```yaml
receivers:
  - name: "default"
    wechat_configs:
      - corp_id:    ${WECHAT_CORP_ID}
        agent_id:   ${WECHAT_AGENT_ID}
        api_secret: ${WECHAT_SECRET}
        to_party:   "运维部"
        message: |
          {{ template "wechat.default" . }}
        send_resolved: true

  - name: "data-quality"
    wechat_configs:
      - corp_id:    ${WECHAT_CORP_ID}
        agent_id:   ${WECHAT_AGENT_ID}
        api_secret: ${WECHAT_SECRET}
        to_party:   "数据平台组"
        message: |
          {{ template "wechat.data_quality" . }}
        send_resolved: true
```

Alertmanager 载入配置后可执行 `curl -X POST http://alertmanager:9093/-/reload` 重载，随即验证告警是否推送到企业微信。

---

## 7. 常见操作

- **刷新模型**：修改 `models/` 后调用 `POST /models/reload`，或执行 `docker compose exec recommendation-api python -m app.model_manager stage`.
- **故障排查**：查看 `logs/`、`data/evaluation/data_quality_report_v2.json`、`reconciliation_*.json`。
- **扩展监控**：Prometheus 快照位于 `data/evaluation/data_quality_metrics.prom`，Grafana Dashboard 在 `monitoring/grafana/`。
- **更新 Schema 契约**：编辑 `config/schema_contracts.yaml`，再运行 `python -m pipeline.data_quality_v2` 验证。

---

## 7. 最佳实践检查表

- [ ] `.env` 指向生产数据库和 Redis，账号具备只读 & 建索引权限。  
- [ ] 执行索引脚本，确认关键表（订单、Matomo 行为）存在时间字段索引。  
- [ ] 初次部署后跑完离线流水线并校验数据质量报告。  
- [ ] Alertmanager 告警成功推送至企业微信或其他渠道。  
- [ ] 对账脚本每日生成报表并回传到运维/数据团队。  
- [ ] Redis、模型目录、日志路径均有备份/监控。  
- [ ] Airflow 或其他调度系统部署成功并执行 `recommendation_pipeline` DAG。
- [ ] 若需代码提交即上线，参考 `docs/AUTO_RELEASE_PIPELINE.md` 配置自动化流程。

如部署中遇到特殊问题或需要支持多区域部署，请结合公司既有 SRE 流程定制调整。
