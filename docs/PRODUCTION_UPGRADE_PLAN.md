# 生产级推荐系统成果总结

本文档记录从原型到生产级推荐系统升级后的最终形态，便于回顾关键改动、上线结果与后续展望。已删除阶段性里程碑表述，所有内容均描述当前最新实现。

---

## 1. 总体成果

- **离线链路**：实现 CDC 抽取 → 数据清洗 → 增强特征 → 质量检测 → 模型训练 → 多路召回索引 → 效果评估的闭环，并支持 `scripts/run_pipeline.sh --sync-only` 快速同步。
- **模型体系**：组合 ItemCF、内容相似、标签/行业/价格/UserCF、LightGBM 排序，支持 Faiss 向量召回与 ONNX 导出。
- **在线服务**：FastAPI 全异步，具备线程池隔离、请求级超时、熔断、三级降级、Redis/SQLite 双栈特征、实验权重调整与详尽指标。
- **可观测性**：Prometheus、Grafana、AlertManager 打通；数据质量、降级、实验、影子模型均可监控；提供运维 SOP。
- **实验治理**：YAML 配置实验，曝光日志自动记录 `experiment_variant`，结合 Matomo 行为评估 CTR/CVR/GMV。

---

## 2. 离线链路亮点

| 模块 | 目标 | 当前实现 |
| --- | --- | --- |
| `pipeline/extract_load` | 无损抽取 & 幂等 | 支持 CDC 水位、分区备份、运行指标 |
| `pipeline/data_cleaner` | 异常处理 | 价格 IQR/Z-Score、时间校正、重复去除 |
| `FeatureEngineV2` | 丰富特征 | 用户/物品 70+ 特征，交叉特征覆盖排序输入 |
| Redis 同步 | 在线特征低延迟 | 可选写入 Redis Hash/ZSet，SQLite 兜底 |
| `data_quality_v2` | 自动质检 | JSON + HTML 报告、Prometheus 快照、告警 |
| `train_models` | 统一训练 | 行为/内容/热门/向量召回 + LightGBM 排序 + MLflow |
| `recall_engine_v2` | 多路召回 | UserCF、标签、行业、价格、Faiss（可选） |
| `evaluate` | 效果评估 | 汇总曝光、Matomo 行为、CTR/CVR/GMV 指标 |

---

## 3. 在线服务能力

- **并发治理**：异步路由 + `ThreadPoolExecutor` + 指标化队列监控 (`recommendation_thread_pool_queue_size`)。
- **请求治理**：`TimeoutManager` 对 Redis/模型/整体设置超时；`recommendation_timeouts_total` 追踪超时原因。
- **降级体系**：Redis → 预计算 → 热门兜底，结果写入 `recommendation_degraded_total` 与曝光日志。
- **召回融合**：行为、内容、向量、标签、行业、价格、UserCF，支持实验化权重调整。
- **排序模块**：LightGBM（默认）/Logistic（兜底） + ONNX 导出；排序熔断防止阻塞。
- **实验系统**：配置位于 `config/experiments.yaml`，按用户 ID 稳定分桶，曝光 `context.experiment_variant`。
- **可观测性**：Prometheus 指标、健康检查、曝光日志、数据质量快照统一串联。

---

## 4. 运维与发布

- **CI/CD**：GitHub Actions 负责 `pytest`、特征构建、模型训练、评估验证。
- **发布流程**：打包 `models/` 与 `model_registry.json` → `/models/reload` → 灰度/影子观察 → 正式切换。
- **回滚**：依据 `model_registry.json` 的 `history` 快速回退；必要时使用 `scripts/run_pipeline.sh --sync-only` 重新同步特征。
- **SOP**：`docs/OPERATIONS_SOP.md` 覆盖 cache/特征库故障演练、模型灰度、Chaos 测试。
- **告警**：质量得分、请求异常、降级率、实验指标均接入 AlertManager。

---

## 5. 指标效果（示例）

| 指标 | 上线前 | 最新 | 备注 |
| --- | --- | --- | --- |
| QPS（单实例） | ≤50 | ≥200 | 线程池 + 异步化提升 |
| P95 延迟 | >200ms | <80ms | Redis 缓存 + 模型异步 |
| 降级率 | 无统计 | <2% | Prometheus 监控 |
| 数据质量评分 | 手工检查 | ≥97/100 | `data_quality_v2` |
| CTR（示例） | 1.5% | 2.1% | 结合 Matomo 评估 |
| LightGBM AUC | 0.65 | 0.79 | 增强特征 + 排序模型 |

> 实际指标以 `data/evaluation/summary.json` 与监控面板为准。

---

## 6. 后续规划

1. **安全治理**：在网关层统一接入鉴权、限流与审计（P0 之外的单独项目）。
2. **实时特征**：引入流式计算或日志回流，缩短用户行为生效时间。
3. **召回扩展**：探索语义召回（Faiss + Sentence-BERT）以及跨品类多场景。
4. **深度模型**：在 LightGBM 上验证完效果后，可试点 DLRM/DeepFM。
5. **自动化评估**：将 CTR/CVR 对比结果自动推送至告警渠道，闭环上线效果。

---

## 7. 文档索引

- 架构：`docs/ARCHITECTURE.md`
- 离线流程：`docs/PIPELINE_OVERVIEW.md`
- 在线服务：`docs/FASTAPI_SERVICE.md`
- 运维：`docs/OPERATIONS_SOP.md`
- 评估：`docs/MATOMO_EVALUATION.md`
- 文档与代码使用均已统一为中文。

---

如需了解某个模块的细节或历史版本，请参阅对应文档或 MLflow 记录。本升级方案至此完成，后续优化将在新的规划中展开。***
