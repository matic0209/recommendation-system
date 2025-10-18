# 推荐系统项目说明

## 1. 项目概述

本仓库实现了面向数据集详情页的推荐服务，结合离线训练与在线服务两部分：

- **离线训练**  
  通过 Airflow 调度的 `recommendation_pipeline` DAG，完成数据抽取 → 特征工程 → 模型训练 → 指标评估 → 指标对账。  
  支持从业务 JSON 导出或数据库（Business / Matomo）读取增量，自动维护水位。

- **在线服务**  
  `recommendation-api` 基于 FastAPI，提供 `/health`、`/similar/{dataset_id}`、`/recommend/detail/{dataset_id}` 等接口。  
  模型文件通过 Docker 挂载在宿主机，可在离线环境更新。

- **监控与运维**  
  Prometheus + Grafana 监控关键指标；Alertmanager 经 `notification-gateway` 转发企业微信告警；MLflow 记录训练产物；Airflow 管理 DAG。

## 2. 仓库结构

```
├── app/                     # 在线服务代码
├── pipeline/                # 离线数据、特征、模型、评估脚本
├── data/                    # 数据目录（由任务生成）
├── models/                  # 模型文件（离线产出 → 在线引用）
├── docs/                    # 文档
├── monitoring/              # Prometheus / Grafana / Alertmanager 配置
├── docker-compose.yml       # 组件编排
├── Dockerfile*              # recommendation-api / Airflow 镜像构建
└── scripts/                 # 常用脚本
```

## 3. 数据与模型流程

1. **数据抽取 (`pipeline.extract_load`)**  
   - `DATA_SOURCE=json` 时读取 `DATA_JSON_DIR` 中的全量/增量 JSON；  
   - `DATA_SOURCE=database` 时直连 Business / Matomo 数据库；  
   - 水位记录在 `data/_metadata/extract_state.json`。

2. **特征 & 清洗 (`pipeline.build_features`)**  
   生成 `data/processed/*` 和 `_v2` 视图。图片特征使用本地缓存的 CLIP 模型（`SENTENCE_TRANSFORMERS_HOME`）。

3. **模型训练 (`pipeline.train_models`)**  
   输出召回模型、排序模型及 MLflow 记录。模型文件默认存放 `models/`。

4. **评估 (`pipeline.evaluate`)**  
   结合曝光日志（`data/evaluation/exposure_log.jsonl`）与 Matomo 行为，生成 `summary.json`、`exposure_metrics.json` 等评估文件。

5. **指标对账 (`pipeline.reconcile_metrics`)**  
   聚合推荐曝光、点击、转化与业务指标，用于监控闭环。

## 4. 运行组件

- **Airflow**：调度 DAG，`recommendation_pipeline` 为主流程。  
- **MLflow**：记录模型参数、指标、产物，挂载 `mlflow-data` 目录。  
- **Redis**：在线服务的缓存与实验位存储。  
- **Prometheus / Grafana / Alertmanager**：监控与告警。  
- **notification-gateway**：企业微信告警中转。  
- **Matomo**：推荐效果来源数据（点击 / 转化等）。

## 5. 快速上手

1. 创建 Python 虚拟环境并安装依赖：
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
2. 准备 `.env` 或 `.env.prod`：
   ```ini
   DATA_SOURCE=json
   DATA_JSON_DIR=/path/to/json
   BUSINESS_DATA_SOURCE=json
   MATOMO_DATA_SOURCE=database
   MATOMO_DB_HOST=…
   CLIP_MODEL_PATH=/opt/recommend/cache/sentence-transformers/clip-ViT-B-32
   ```
3. 首次全量训练：
   ```bash
   bash scripts/run_pipeline.sh --full-refresh
   ```
4. 启动 Docker 组件（详见《DOCKER 部署指南》）：
   ```bash
   docker compose up -d
   ```
5. 验证：
   ```bash
   ./smoke_test.sh
   ```

## 6. 日常运维要点

- **Airflow**  
  `docker compose exec airflow-webserver airflow dags list-runs -d recommendation_pipeline`  
  单独重跑任务：`airflow tasks run … <execution_date> --ignore-all-dependencies`

- **模型权限**  
  `models/`、`data/`、`mlflow-data`、CLIP 缓存目录需赋予 UID 50000 可写权限：
  ```bash
  sudo chown -R 50000:50000 models data /var/lib/docker/volumes/recommend_mlflow-data/_data /opt/recommend/cache
  ```

- **监控**  
  - Grafana：`http://<host>:3000`  
  - Prometheus：`http://<host>:9090`  
  - MLflow：`http://<host>:5000`  
  - Alertmanager：`http://<host>:9093`

- **日志路径**  
  - 推荐服务：`logs/`  
  - Airflow：`airflow/logs/`  
  - 曝光日志：`data/evaluation/exposure_log.jsonl`

- **常见问题**  
  - CLIP 模型：确保 `CLIP_MODEL_PATH` 指向宿主机缓存，设置 `HF_HUB_OFFLINE=1`；  
  - JSON 路径：Airflow 容器需挂载 `DATA_JSON_DIR`；  
  - 权限：`Permission denied` 多半是挂载目录未 `chown` 到 `appuser`。

## 7. API 快速参考

| 接口                            | 说明                     | 示例                                       |
|---------------------------------|--------------------------|--------------------------------------------|
| `GET /health`                   | 健康检查                 | 返回模型加载与缓存状态                     |
| `GET /similar/{dataset_id}`     | 相似推荐                 | `/similar/123?top_n=10`                    |
| `GET /recommend/detail/{id}`    | 详情页推荐               | `/recommend/detail/123?user_id=7&top_n=5`  |
| `GET /metrics`                  | Prometheus 指标          | 用于监控抓取                               |

更多细节参见后续文档：《DOCKER 部署指南》《运维手册》《常见问题与排错》。确保先阅读部署指南，再逐步进行上线与验证。  

如需英文文档，可根据本说明自行翻译或补充。欢迎在生产环境实践后将经验回写文档。祝使用顺利！
