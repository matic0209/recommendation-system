# 模型 CI/CD 流程

本文档说明从代码变更、离线训练、评估验证到上线与回滚的完整流程，覆盖工具链、环境变量与操作规范。

---

## 1. 分支与代码评审

1. 所有功能通过 Feature 分支开发，合并至 `master` 前必须创建 PR。
2. PR 需包含：
   - 功能说明与影响范围（数据、召回、排序、在线服务等）
   - 本地验证记录：`pytest`、`scripts/run_pipeline.sh --sync-only` 或核心步骤日志
   - 若涉及指标变化，请贴出前后对比（可来自 `data/evaluation/summary.json` 或曝光样本）
3. 评审重点：
   - 是否破坏数据抽取/特征/模型的幂等性
   - 是否兼容实验配置、降级逻辑
   - 指标与日志是否充分
   - 文档是否同步更新（本项要求此次迭代已满足）

---

## 2. CI（GitHub Actions）

工作流位于 `.github/workflows/tests.yml`，默认执行：

1. 安装基础依赖与 `pytest`
2. 运行 `pytest`
3. 执行 `python -m pipeline.build_features`（使用内置小样本）验证特征流程
4. 执行 `python -m pipeline.train_models` 验证模型训练、MLflow 记录与模型落盘
5. 执行 `python -m pipeline.evaluate` 确保评估脚本可运行

> Faiss/Sentence-BERT/ONNX 为可选依赖，CI 默认为关闭状态；如需覆盖相关路径，可在 workflow 中增加额外 job。

所有步骤通过后方可合并。

---

## 3. 线下训练与验证

### 3.1 准备数据

```bash
python -m pipeline.extract_load      # CDC 抽取
python -m pipeline.build_features    # 清洗 + 特征 + Redis 同步
python -m pipeline.data_quality_v2   # 质量报告 + Prometheus 快照
```

### 3.2 训练与召回索引

```bash
python -m pipeline.train_models      # 行为/内容 + LightGBM 排序
python -m pipeline.recall_engine_v2  # 多路召回索引
```

### 3.3 评估

```bash
python -m pipeline.evaluate
```

重点关注：

- `data/evaluation/data_quality_report_v2.json` 质量评分是否异常
- `data/evaluation/summary.json` 中的召回/排序指标（例如 `ranking_auc`、`recall_hit_rate`）
- `models/ranking_scores_preview.json` 的 Top 样本是否符合预期
- MLflow UI（`mlflow ui --backend-store-uri <URI> --port 5000`）查看运行记录

---

## 4. 发布流程

1. **打包产物**
   - `models/` 下的所有文件（含召回索引、排序模型、热门榜）
   - `models/model_registry.json`（当前 + 历史版本信息）
   - 如使用对象存储/制品仓库，请同步上传
2. **部署上线**
   - 更新服务镜像或环境变量
   - `scripts/run_pipeline.sh --sync-only`（如需同步 Redis 特征）
   - 调用 `/models/reload` 加载最新模型
   - 观察指标与曝光日志，确认实验、降级正常
3. **灰度策略**
   - 影子模型：`POST /models/reload` with `mode=shadow` & `rollout=0.05`
   - 通过曝光日志和 Grafana 面板观察影子流量效果
   - 指标稳定后再切换至 `mode=primary`

---

## 5. 回滚策略

1. 查阅 `models/model_registry.json` 中的 `history` 拿到上一个 `run_id`
2. 使用 MLflow 下载对应模型或直接恢复备份文件
3. 调用 `/models/reload`（`mode=primary`）即可回退
4. 若离线特征出现问题，可使用 `scripts/run_pipeline.sh --sync-only` 重新生成

---

## 6. 环境变量清单

| 变量 | 说明 |
| --- | --- |
| `BUSINESS_DB_*` / `MATOMO_DB_*` | 数据源连接 |
| `FEATURE_REDIS_URL` | Redis 特征库（可选） |
| `REDIS_URL` | 在线缓存 |
| `MLFLOW_TRACKING_URI` | MLflow 存储地址 |
| `MLFLOW_EXPERIMENT_NAME` | 实验名称，默认 `dataset_recommendation` |
| `EXPERIMENT_CONFIG_PATH` | 实验配置文件路径 |
| `USE_FAISS_RECALL` | 是否启用 Faiss（默认为 1，库缺失自动降级） |

生产环境需通过安全方式（CI Secrets、K8s Secret 等）注入上述变量。

---

## 7. 质量守门清单

- [ ] `pytest` 通过，核心逻辑覆盖率达标
- [ ] `scripts/run_pipeline.sh --sync-only` 运行成功，无阻塞或异常
- [ ] `data_quality_report_v2` 得分 ≥ 90（可根据业务调整）
- [ ] `ranking_auc` 与历史基线差异在可接受范围（±5%）
- [ ] 曝光日志确认降级/实验字段正常
- [ ] 文档与指标面板已同步更新

---

如流程或工具链有调整，请同步更新本文档并广播给团队，确保研发与运维人员对齐最新实践。***
