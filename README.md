# 数据集推荐系统

本项目提供一套面向数据集商城的生产级推荐系统，覆盖数据抽取、特征构建、模型训练、实验治理及在线服务。所有文档均采用中文描述，方便研发/运维团队快速查阅。

---

## 1. 功能概览

- **离线流水线**：支持 CDC 增量抽取、数据清洗、增强特征（>70 个特征），并输出 Prometheus 质量快照。
- **多模态特征**：新增基于 CLIP 的图片向量抽取（CPU 友好），自动融合文本与视觉信号。
- **多模型召回**：行为协同过滤、内容向量、标签/行业/价格、UserCF、Faiss（可选）等多路召回。
- **LightGBM 排序**：基于增强特征训练 LightGBM 排序模型，自动导出 Pickle+ONNX。
- **在线 API**：FastAPI + 异步调用，内置线程池隔离、分阶段超时、熔断与多级降级（Redis → 预计算 → 热门）。
- **实验体系**：YAML 配置化实验，按用户 ID 稳定分桶，支持召回权重调参，曝光日志记录实验变体。
- **可观测性**：Prometheus 指标覆盖请求、降级、超时、特征质量；Grafana 大盘与 AlertManager 告警。
- **运维 SOP**：提供缓存/特征库故障演练、模型灰度、混沌测试指引。

---

## 2. 快速开始

### 2.1 环境准备

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

- 如需启用 Faiss/Sentence-BERT，请额外安装 `faiss-cpu` 与 `sentence-transformers`。
- 若需要导出 ONNX 排序模型，可安装 `skl2onnx`。

### 2.2 配置

```bash
cp .env.example .env
# 编辑数据库、Redis、特征库等连接信息
```

关键环境变量：

| 变量名 | 说明 |
| --- | --- |
| `BUSINESS_DB_*` / `MATOMO_DB_*` | 业务库与行为库 MySQL 配置 |
| `FEATURE_REDIS_URL` | Redis 特征库连接串（不填则回退 SQLite） |
| `REDIS_URL` | 在线缓存 Redis（推荐与特征库分库） |
| `EXPERIMENT_CONFIG_PATH` | 实验配置文件路径，默认 `config/experiments.yaml` |
| `USE_FAISS_RECALL` | 是否启用 Faiss 向量召回（默认 1，无法加载库时自动降级） |

### 2.3 一键执行流水线

```bash
# 查看即将执行的步骤
scripts/run_pipeline.sh --dry-run

# 全量执行（抽取→清洗→特征→质量→训练→召回索引→评估）
scripts/run_pipeline.sh

# 仅同步特征/模型（跳过抽取）
scripts/run_pipeline.sh --sync-only
```

流水线产出：

- 数据集市：`data/business/*.parquet`，行为库：`data/matomo/*.parquet`
- 清洗数据：`data/cleaned/*.parquet`，增强特征：`data/processed/*_v2.parquet`
- 质量报告：`data/evaluation/data_quality_report_v2.json` + `data_quality_report.html`
- Prometheus 快照：`data/evaluation/data_quality_metrics.prom`
- 模型与召回索引：`models/*.pkl / .json / rank_model.onnx`
- 图片向量缓存：`data/processed/dataset_image_embeddings.parquet`（如遇写权限限制会自动回退到 `cache/dataset_image_embeddings.parquet`）

Airflow 中的 `recommendation_pipeline` DAG 已增加 `image_embeddings → build_features → train_models` 链路，保证图片向量在特征构建前生成，并由 MLflow 记录多模态指标（使用的模态数、向量维度、融合权重等）。

---

## 3. 在线服务

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

主要改进点：

- `async` 端点 + 线程池隔离 + `TimeoutManager` 差异化超时
- 多渠道召回融合，支持实验权重、UserCF/标签/价格等附加信号
- Redis 缓存、特征库优先读取，失败自动降级 SQLite
- 曝光日志记录 `request_id`、模型版本、降级原因、实验变体
- Prometheus 指标：`recommendation_latency_seconds`、`recommendation_degraded_total`、`recommendation_timeouts_total` 等

可用接口：

| 路径 | 说明 |
| --- | --- |
| `GET /health` | 健康检查（模型加载 + Redis 状态） |
| `GET /metrics` | Prometheus 指标 |
| `GET /similar/{dataset_id}` | 多渠道相似推荐 |
| `GET /recommend/detail/{dataset_id}?user_id=...` | 详情页推荐（支持个性化 + 实验） |
| `GET /hot/trending?timeframe=1h` | 热度榜单 |
| `POST /models/reload` | 模型热更新 / 阴影模型 |

详细接口字段请参阅 `docs/API_REFERENCE.md`。

---

## 4. 文档索引

### 核心文档（必读）

- `QUICKSTART.md`：快速开始指南，Docker/K8s 部署流程
- `docs/ARCHITECTURE.md`：总体架构设计与组件关系
- `docs/DEVELOPER_GUIDE.md`：开发规范、目录结构与代码风格

### 功能文档

- `docs/PIPELINE_OVERVIEW.md`：离线流水线与特征描述
- `docs/FASTAPI_SERVICE.md`：在线服务实现细节与扩展点
- `docs/API_REFERENCE.md`：API 接口详细说明
- `docs/MODEL_CICD.md`：模型训练、版本管理与灰度策略
- `docs/HOT_RELOAD.md`：热更新流程与常见问题

### 运维文档

- `docs/OPERATIONS_SOP.md`：运维手册、故障演练与告警处置
- `docs/DEPLOYMENT_GUIDE.md`：生产环境部署指南
- `docs/DATABASE_INDEX_SETUP.md`：数据库索引优化指南

### 优化与评估

- `docs/PROJECT_OPTIMIZATION.md`：整体优化成果回顾
- `docs/OPTIMIZATION_TRACKER.md`：优化进度追踪
- `docs/P0_OPTIMIZATION_GUIDE.md`：P0 优化执行指南
- `docs/P0_EXECUTION_REPORT.md`：P0 优化执行报告
- `docs/PRODUCTION_OPTIMIZATION_TODO.md`：生产优化 TODO 清单
- `docs/MATOMO_EVALUATION.md`：行为日志对齐与效果评估
- `docs/AUTO_RELEASE_PIPELINE.md`：自动发布流水线

所有文档已统一为中文，并描述当前最终实现。

---

## 5. 常见运维操作

- 同步 Redis 特征库：`FEATURE_REDIS_URL` 配置后，运行 `scripts/run_pipeline.sh --sync-only`
- 灾备演练：按照 `docs/OPERATIONS_SOP.md` 执行缓存/特征库/模型故障演练
- 实验调整：修改 `config/experiments.yaml` 后调用 `/models/reload` 即可生效
- 指标/告警：Prometheus + Grafana + AlertManager 配置位于 `monitoring/`

---

## 6. 贡献指南

1. 新功能请先阅读 `docs/DEVELOPER_GUIDE.md` 中的代码规范与测试要求。
2. 所有变更需附带单元测试或说明原因，并在 MR/PR 中贴出关键指标。
3. 文档默认使用中文，如需英文版请在 `docs/i18n/` 下新增对应文件。

欢迎在 Issues 中提交问题或改进建议。祝开发顺利！
