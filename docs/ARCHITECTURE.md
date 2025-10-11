# 整体架构说明

本文档描述当前生产级推荐系统的整体架构，包括数据链路、模型体系、在线服务、实验治理与可观测性。所有内容基于最新实现，不再区分「v1/v2」或阶段性成果。

---

## 1. 总览

```
                ┌────────────────────────┐
                │       MySQL (业务库)    │
                └────────────┬───────────┘
                             │
                ┌────────────▼───────────┐
                │     Matomo 行为库       │
                └────────────┬───────────┘
                             │ CDC + 批量抽取
                             ▼
┌──────────────────────────────────────────────────────────────┐
│                    离线处理与特征层                          │
│  ┌──────────────┐  ┌───────────────┐  ┌────────────────────┐ │
│  │extract_load  │→│data_cleaner    │→│build_features(+v2)  │ │
│  └──────────────┘  └───────────────┘  └────────────────────┘ │
│             │                 │                 │            │
│             ▼                 ▼                 ▼            │
│      Parquet 数据       清洗结果/质量      SQLite + Redis 特征库│
│             │                 │                 │            │
│             └────→ data_quality_v2 → Prometheus 快照          │
│                                        │                      │
│                                        ▼                      │
│                          train_models + recall_engine         │
│                                        │                      │
│                                        ▼                      │
│                               MLflow + 模型仓库               │
└──────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────┐
│                         在线服务层                           │
│  FastAPI (async) + 线程池 + 超时/熔断 + 多级降级             │
│  ├─ Redis 缓存（推荐结果/热榜）                              │
│  ├─ Redis 特征库（可选） / SQLite 兜底                        │
│  ├─ 多路召回融合 (行为/内容/向量/标签/行业/价格/UserCF)      │
│  ├─ LightGBM 排序 + ONNX 导出                                │
│  ├─ 实验系统（YAML 配置 + 稳定分桶）                         │
│  └─ Telemetry & Prometheus 指标                              │
└──────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────┐
│                   可观测性 & 运维支持                        │
│  ├─ Prometheus / Grafana / AlertManager                      │
│  ├─ Matomo 行为对齐 + pipeline/evaluate                      │
│  ├─ docs/OPERATIONS_SOP.md（故障演练指南）                   │
│  └─ scripts/run_pipeline.sh --sync-only 快速修复            │
└──────────────────────────────────────────────────────────────┘
```

---

## 2. 数据与特征

### 2.1 数据抽取

- `pipeline/extract_load.py` 支持 CDC 增量提取业务库与 Matomo 行为表，维护 `data/_metadata/extract_state.json` 水位。
- 每次运行产出：
  - `data/business/<table>.parquet`、`data/matomo/<table>.parquet`
  - 分区备份 `data/<source>/<table>/load_time=*.parquet`
  - 运行指标 `data/evaluation/extract_metrics.json`

### 2.2 数据清洗

`pipeline/data_cleaner.py`（由 `build_features.py` 自动调用）完成：

- IQR/Z-Score 价格异常裁剪
- 主键去重、时间戳修正（过滤未来时间）
- 结果落地 `data/cleaned/*.parquet` 并输出统计 `cleaning_stats.json`

### 2.3 特征构建

`pipeline/build_features.py` 统一完成基础与增强特征：

- 生成交互、用户画像、数据集特征、统计表（基础版）
- 调用 `FeatureEngineV2` 构建增强特征：
  - 用户特征 >30 项（行为、时序、价格/标签偏好、活跃度等）
  - 物品特征 >40 项（文本统计、行业、价格桶、流行度、时效性等）
  - 用户-物品交叉特征用于排序训练
- 同时写入：
  - `data/processed/*.parquet` 与 `_v2.parquet`
  - SQLite `data/feature_store.db`（基础 + `_v2` 表）
  - Redis（可选）：用户/数据集 Hash、用户历史 ZSet、统计 Hash

### 2.4 数据质量

`pipeline/data_quality_v2.py` 输出三类结果：

- JSON：`data/evaluation/data_quality_report_v2.json`
- HTML：便于业务、PM 查看
- Prometheus 快照：`data/evaluation/data_quality_metrics.prom`

指标包含质量得分、空值、重复、Gini、时间完整性、异常列表，告警阈值可直接接入 AlertManager。

---

## 3. 模型与召回

### 3.1 召回

`pipeline/recall_engine_v2.py` 构建并落盘：

- 行为召回：`models/item_sim_behavior.pkl`
- 内容召回：`models/item_sim_content.pkl`
- 多路索引：
  - UserCF 相似用户 `models/user_similarity.pkl`
  - 标签倒排 `models/tag_to_items.json` / `item_to_tags.json`
  - 行业索引 `models/category_index.json`
  - 价格桶索引 `models/price_bucket_index.json`
  - Faiss 向量召回（可选）：模型与元信息 `models/faiss_recall.*`
- 热门榜：`models/top_items.json`

### 3.2 排序

`pipeline/train_models.py`：

- 读取增强特征表，训练 LightGBM（二分类），自动 fallback Logistic Regression。
- 导出 `models/rank_model.pkl`（Pipeline）与 `models/rank_model.onnx`（可选）。
- 记录评分快照 `ranking_scores_preview.json` 和指标（AUC、召回覆盖等）。
- 通过 MLflow (`mlruns/`) 记录参数、指标、artifact，并更新 `models/model_registry.json`。

---

## 4. 在线服务

### 4.1 启动流程

`app/main.py` 在 `startup` 中完成：

- 加载模型、召回索引、增强特征、用户画像、热榜。
- 尝试连接 Redis 特征库（失败自动降级）。
- 加载实验配置 `config/experiments.yaml`。
- 构建缓存、热度追踪、健康检查器。

### 4.2 请求处理

- 接口完全异步，阻塞操作通过 `ThreadPoolExecutor` 执行。
- `TimeoutManager` 提供 Redis/模型/总耗时三层超时，触发后写入 `recommendation_timeouts_total`。
- 多渠道召回融合 `DEFAULT_CHANNEL_WEIGHTS`，实验可覆盖权重。
- 个人化逻辑基于用户历史和标签偏好，自动去除已购项。
- 排序引擎启用熔断器，失败时保留召回分并触发 fallback。
- 降级策略：Redis → 预计算 → 热门，所有降级场景写入 `recommendation_degraded_total` 与曝光日志。
- 实验变体通过 `assign_variant` 计算，曝光日志 `context.experiment_variant` 记录变体。

### 4.3 指标与日志

- Prometheus：请求量、延时、队列长度、超时、降级、缓存命中、召回多路覆盖等。
- Telemetry：`data/evaluation/exposure_log.jsonl`，包含 request_id、用户、模型版本、实验、降级原因。

---

## 5. 部署与运维

- Docker/K8s 文件位于 `docker-compose.yml`、`k8s/`。
- Grafana/Prometheus/AlertManager 配置位于 `monitoring/`。
- 运维手册见 `docs/OPERATIONS_SOP.md`，涵盖：
  - 缓存/特征库故障处理
  - 模型灰度与回滚
  - Chaos 演练脚本建议
- 实验治理：更新 `config/experiments.yaml` 后调用 `/models/reload` 即可热加载。
- Redis 特征库重建：执行 `scripts/run_pipeline.sh --sync-only`。

---

## 6. 安全与拓展

- 鉴权与网关策略可在 Ingress/Nginx 层配置（限流、JWT、mTLS）；上线前务必结合公司安全组件。
- 允许通过环境变量调整线程池大小、超时阈值、召回权重。
- Faiss、Sentence-BERT、ONNX 为可选依赖，未安装时系统自动降级但保留主流程。

---

本架构文档会随着实现更新而同步维护，如有调整请在合并请求中同步修改。所有说明均以当前代码为准。***
