# 开发者使用指南

本文档面向参与研发的工程师，说明代码结构、开发流程、调试要点及常见问题。请在提交代码或上线前阅读并遵循本指南。

---

## 1. 目录结构

```
app/                # FastAPI 在线服务
config/             # 全局配置与环境变量加载
pipeline/           # 数据抽取、清洗、特征、训练、评估脚本
models/             # 模型与索引产出（运行后生成）
data/               # 数据抽取与特征结果（运行后生成）
docs/               # 中文文档
monitoring/         # Prometheus / Grafana / AlertManager 配置
scripts/            # 一键脚本（如 run_pipeline.sh）
tests/              # 单元/集成测试
```

- 核心配置集中在 `config/settings.py`，默认读取 `.env`。
- 所有文档均为中文且描述最终实现，如需英文文档请另建目录。

---

## 2. 环境与依赖

1. 使用 Python 3.8+，推荐虚拟环境：
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
2. 可选依赖：
   - Faiss/Sentence-BERT：启用向量召回时安装 `faiss-cpu` 与 `sentence-transformers`
   - ONNX 导出：安装 `skl2onnx`
   - GPU 场景：将 LightGBM 切换到 GPU 版本后需更新 Pip 依赖
3. 数据库与 Redis 连接在 `.env` 中配置，模板见 `.env.example`。关键变量：
   - `BUSINESS_DB_*` / `MATOMO_DB_*`
   - `FEATURE_REDIS_URL`（特征库，可选）
   - `REDIS_URL`（缓存）
   - `MLFLOW_TRACKING_URI`、`MLFLOW_EXPERIMENT_NAME`
   - `EXPERIMENT_CONFIG_PATH`、`USE_FAISS_RECALL`

---

## 3. 开发流程

### 3.1 离线流水线

单步执行（推荐在调试阶段）：

```bash
python -m pipeline.extract_load --dry-run   # 查看待抽取表
python -m pipeline.extract_load             # CDC 抽取
python -m pipeline.build_features           # 清洗 + 增强特征 + Redis 同步
python -m pipeline.data_quality_v2          # 输出质量报告/Prometheus 指标
python -m pipeline.train_models             # 行为/内容/热门 + LightGBM 排序
python -m pipeline.recall_engine_v2         # 多路召回索引
python -m pipeline.evaluate                 # Matomo 对齐评估
```

一键脚本：

```bash
scripts/run_pipeline.sh         # 全量流程
scripts/run_pipeline.sh --sync-only  # 跳过抽取，仅同步特征与模型
```

产出位置说明：

| 类型 | 路径 |
| --- | --- |
| 原始数据 | `data/business/`、`data/matomo/` |
| 清洗数据 | `data/cleaned/` |
| 特征 | `data/processed/`（含 `_v2`） |
| 质量报告 | `data/evaluation/data_quality_report_v2.json/html` |
| Prometheus 快照 | `data/evaluation/data_quality_metrics.prom` |
| 模型 | `models/`（含 `rank_model.pkl/onnx`、召回索引） |
| MLflow | 默认 `mlruns/` |

### 3.2 测试与代码规范

- 本地测试：`pytest`
- 代码格式：遵循 PEP8 + Ruff（如项目后续引入）
- 长时间运行脚本请在命令后追加 `--dry-run` 或分步骤执行，避免误操作线上库
- 合并请求需包含：
  - 改动摘要
  - 运行 `pytest` / `scripts/run_pipeline.sh` 结果（如适用）
  - 相关文档更新

---

## 4. 在线服务调试

### 4.1 启动

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

常用接口：

| 接口 | 说明 |
| --- | --- |
| `GET /health` | 健康状态（缓存/模型） |
| `GET /metrics` | Prometheus 指标 |
| `GET /similar/{dataset_id}` | 相似数据集 |
| `GET /recommend/detail/{dataset_id}?user_id=...` | 详情页推荐 |
| `GET /hot/trending` | 热度榜 |
| `POST /models/reload` | 模型热更新/灰度 |

### 4.2 调试技巧

- 使用 `?limit=1` 等小值快速验证逻辑
- 查看日志：默认输出到 STDOUT，可在 `uvicorn` 启动时添加 `--log-config app/logging_config.py`
- 曝光日志：`data/evaluation/exposure_log.jsonl`，便于校验实验和降级
- Metrics：`recommendation_degraded_total`、`recommendation_timeouts_total`、`recommendation_thread_pool_queue_size` 等可帮助定位瓶颈

---

## 5. 实验与特征

### 5.1 实验系统

- 配置文件：`config/experiments.yaml`
- 结构示例：
  ```yaml
  experiments:
    recommendation_detail:
      status: active
      salt: recommend-detail-v1
      variants:
        - name: control
          allocation: 0.5
          parameters: {}
        - name: content_boost
          allocation: 0.5
          parameters:
            content_weight: 0.8
            vector_weight: 0.5
  ```
- 修改后通过 `/models/reload` 生效，无需重启
- 曝光日志 `context.experiment_variant` 存储变体，方便离线分析

### 5.2 特征扩展

- 新增用户/物品特征：修改 `pipeline/build_features_v2.py`
- 新增召回通道：扩展 `pipeline/recall_engine_v2.py` 并在 `app/main.py` 的 `_augment_with_multi_channel` 中融合
- 排序特征/模型：在 `_prepare_ranking_dataset` 和 `_train_ranking_model` 中调整

---

## 6. 故障处理

参见 `docs/OPERATIONS_SOP.md`，常见操作包括：

- `scripts/run_pipeline.sh --sync-only` 重新同步特征/模型
- 模型灰度→观察影子指标→上线
- Redis 故障：服务自动降级 SQLite/热门兜底，恢复后运行同步脚本
- 质量告警：查看 `data_quality_report_v2.*` 与 `Prometheus` 快照确认问题

---

## 7. 常见问题

| 问题 | 排查步骤 |
| --- | --- |
| 无推荐结果 | 检查 `data/processed/` 是否最新 → 查看日志中 `degrade_reason` → 确认热门兜底是否命中 |
| 延迟升高 | 观察 `recommendation_thread_pool_queue_size`、`recommendation_timeouts_total`，必要时调整线程池或特征查询策略 |
| 实验无效果 | 确认 `user_id` 是否传入、曝光日志中是否出现对应 `experiment_variant` |
| Redis 不可达 | 查看 `/health`，若 `cache=disabled` 则为正常降级；恢复后重新加载模型 |
| LightGBM 警告过多 | 检查特征是否存在常量列/极端分布，可在 `build_features_v2` 中补充归一化或裁剪 |

---

如需新增模块或对现有流程做大幅调整，请同步更新相关文档并在 MR 中说明设计方案。祝开发顺利！***
