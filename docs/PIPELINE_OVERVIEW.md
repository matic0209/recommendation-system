# 数据处理与推荐流程概览

本文档总结当前推荐系统从数据到服务的完整流程，便于开发与运营人员理解各阶段职责。

## 1. 数据抽取（pipeline/extract_load.py）

1. 首次执行时全量导出业务库与 Matomo 库中的关键表，生成 `data/<source>/<table>.parquet`。
2. 后续运行会依据 `data/_metadata/extract_state.json` 中的水位（按时间列自动选择）实现 CDC 增量抽取，可通过 `--full-refresh` 强制重置。
3. 每次抽取同时在 `data/<source>/<table>/load_time=*.parquet` 目录下追加分区文件，并把吞吐量/耗时指标写入 `data/evaluation/extract_metrics.json` 作为性能基线。

> 配置通过 `.env` 或环境变量提供，脚本会自动读取。

## 2. 特征构建（pipeline/build_features.py）

构建阶段现在分为「基础产出」+「强化清洗/特征」两层：

1. **基础产出**
   - 汇总订单与 API 订单，计算 `log1p(price)` 权重 → `processed/interactions.parquet`
   - 提取数据集文本/价格信息 → `processed/dataset_features.parquet`
   - 生成用户基础画像 → `processed/user_profile.parquet`
   - 统计交互覆盖度 → `processed/dataset_stats.parquet`
   - 全量同步到 SQLite `feature_store.db`（表名：`interactions`、`dataset_features` 等）

2. **强化清洗（DataCleaner）**
   - 价格异常：支持 Z-Score/IQR 裁剪或移除，默认 IQR clip
   - 去重复：用户、交互、数据集均按主键去重
   - 时间清洗：修正未来时间戳、空值，输出 `data/cleaned/*.parquet`

3. **强化特征（增强版）**
   - 用户侧：行为统计、时序、标签/价格偏好等 30+ 特征 → `processed/user_features_v2.parquet`
   - 物品侧：文本 TF/IDF 稀疏统计、行业/价格衍生、转化率、流行度 → 40+ 特征
   - 交叉特征：用户-物品偏好匹配分、活跃度等（体现在 LightGBM 特征列）
   - 结果同时覆写基础视图，并以 `_v2` 表名写入 SQLite

4. **Redis 特征同步（可选）**
   - 若设置 `FEATURE_REDIS_URL`，脚本自动写入用户/物品 hash、用户历史 ZSET、数据集统计 hash
   - 用于线上 5ms 内特征读取，与 API 的 Redis fallback 配合

## 3. 模型训练（pipeline/train_models.py）

1. **行为相似模型**：基于强化交互表计算余弦共现 → `models/item_sim_behavior.pkl`
2. **内容相似模型**：TF-IDF + 余弦 → `models/item_sim_content.pkl`
3. **多路召回索引**（在 `pipeline/recall_engine_v2` 中训练）：
   - UserCF 相似用户表 `models/user_similarity.pkl`
   - 标签倒排 `models/tag_to_items.json`、行业索引、价格分桶索引
   - （可选）Faiss HNSW 向量检索（开启 `USE_FAISS_RECALL=1`）
4. **LightGBM 排序**：使用 50+ 特征训练 `LGBMClassifier`，输出 `models/rank_model.pkl` 与 `models/rank_model.onnx`
5. **热门榜单**：按时间衰减权重排序 → `models/top_items.json`
6. **模型管理**：触发 MLflow run，记录 recall/排序指标，更新 `models/model_registry.json`

## 4. 在线服务（app/main.py）

- 完整异步化：核心接口使用 `async` + 线程池隔离，同步 I/O（Redis/排名模型）被包装成带超时的 `_call_blocking`
- `TimeoutManager` 按操作（redis/model/total）施加上限，并在触发时写入 `recommendation_timeouts_total`
- 多级降级：Redis → 预计算 → 热门榜；所有降级在 `recommendation_degraded_total` 及曝光 `context.degrade_reason` 中可见
- 多渠道召回融合：
  - 基础 channel 权重（行为/内容/向量/热门）可通过实验配置调整
  - 标签/行业/价格/UserCF 召回作为附加候选注入
- 实验框架：加载 `config/experiments.yaml`，`assign_variant` 依据用户 ID 稳定分桶，曝光日志携带 `experiment_variant`
- 接口回包维持原结构（dataset_id/title/score/reason），但理由会附加 `+tag`、`+usercf` 等后缀

## 5. 推荐效果评估

- `pipeline/evaluate.py`：对比曝光日志与 Matomo 行为/转化，输出 CTR/CVR、GMV、Top-N 命中率等指标
- 数据质量报告 `pipeline/data_quality_v2.py`：生成 JSON + HTML 报告，并输出 Prometheus snapshot（`data_quality_score`）供告警
- 曝光日志（`data/evaluation/exposure_log.jsonl`）自动池化 request_id、模型版本、实验、降级信息

## 6. 调度与自动化

- 推荐使用仓库内的 Airflow Docker Compose（`airflow/docker-compose.yaml`）启动调度集群。
- DAG `recommendation_pipeline` 会依次执行抽取、特征、训练、数据质量与评估模块。
- 启动步骤详见 `airflow/README.md`，支持 CLI `docker compose run --rm airflow-cli airflow dags trigger recommendation_pipeline` 触发临时运行。

## 7. 常用命令

```bash
# 评估推荐效果
python -m pipeline.evaluate
```


## 8. 评估与监控

- `python -m pipeline.data_quality_v2`：输出 HTML/JSON 报告与 Prometheus 指标快照
- `python -m pipeline.evaluate`：生成 CTR/CVR/GMV 及多种覆盖率指标，默认在 `data/evaluation/summary.json`
- 监控栈（Prometheus + Grafana）默认抓取 `/metrics`，Alert 规则见 `monitoring/alerts/recommendation.yml`

```bash
# 一键跑完整流水线
scripts/run_pipeline.sh

# 单独重训模型（数据处理已完成的情况下）
python3 pipeline/train_models.py

# 启动接口服务
uvicorn app.main:app --reload
```

该流程确保 API 订单 (`api_order`) 与常规订单统一纳入交互数据，为后续推荐效果提供更全面的依据。
