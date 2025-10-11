# Matomo 数据评估指引

本文档总结如何利用现有 Matomo MySQL（或导出的 Parquet）表来评估推荐算法上线后的表现，并给出基线计算与后续扩展方案。全流程以“只读 Matomo 数据”为原则，推荐曝光记录由推荐服务本地持久化（不回写 Matomo）。

## 1. 关键表结构

| 表 | 说明 | 与推荐评估的关系 |
| --- | --- | --- |
| `matomo_log_visit` | 访问会话级别信息（`idvisit`、`userid`、`visit_last_action_time` 等） | 用于把曝光/点击映射到用户、时间线 |
| `matomo_log_link_visit_action` | 每次页面浏览/事件的明细。`idaction_url`、`idaction_name` 指向 `matomo_log_action`，`server_time` 为事件时间 | 页面曝光、按钮点击、事件行为的主数据源 |
| `matomo_log_action` | 字符串字典：`name` 包含实际 URL/路径。当前数据中 `dataDetail/xxx`、`dataAPIDetail/xxx`、`payment/xxx` 等路径可解析出 `dataset_id` | 通过模式匹配把 Matomo 行为映射到数据集 |
| `matomo_log_conversion` | 订单/目标转化记录。`url`/`idaction_url` 同样可解析 `dataset_id`，并提供 `revenue` 等指标 | 用于评估推荐后的转化效果 |

> 现有 `pipeline/evaluate.py` 已示范如何利用上述表，正则匹配 `dataDetail/\d+` / `dataAPIDetail/\d+` 等模式。

## 2. 上线前离线基线

1. 执行 `python -m pipeline.extract_load` → `python -m pipeline.build_features` → `python -m pipeline.train_models`。
2. 运行 `python -m pipeline.evaluate` 获取 `data/evaluation/dataset_metrics.csv` 与 `summary.json`：
   - `total_views` / `total_conversions` / `total_revenue`
   - 排名 Spearman / Top20 命中率（使用当前模型产出的排序分数）
   - 向量召回命中率 `vector_recall_hit_rate_at_50`
3. 该报告提供“模型上线前”的历史表现基线。上线后可以对比新版本推荐的点击率、转化率是否优于历史基线。

## 3. 上线后数据收集

Matomo 默认不会记录“系统向用户展示了哪些推荐”。推荐服务需**本地持久化曝光日志**，推荐的字段如下：

| 字段 | 说明 |
| --- | --- |
| `request_id` | 推荐请求唯一 ID（UUID） |
| `user_id` / `idvisitor` | 与 Matomo `idvisit` / `userid` 对齐的标识（未登录用户可使用 `idvisitor`） |
| `page_id` | 页面上下文（如 `dataset_id`） |
| `rec_items` | 推荐结果列表（数组或 JSON） |
| `model_run_id` | 来自 `models/model_registry.json` 的 `run_id` |
| `reasons` | 可选，记录召回源（behavior/content/vector/rank） |
| `timestamp` | 推荐响应时间 |

建议保存到本地 SQLite（例如 `data/evaluation/recommender_exposure.db`）或 JSONL/Parquet。项目已在 `app.telemetry.record_exposure` 中默认写入 `data/evaluation/exposure_log.jsonl`。无需向 Matomo 写回。

## 4. 曝光与 Matomo 行为的关联

1. 通过 `request_id`（或 `user_id + page_id + timestamp`）在 Matomo 的 `matomo_log_link_visit_action` 中查找后续行为：
   - 浏览：`idaction_url`/`idaction_name` 对应推荐项的详情页。
   - 点击事件：假设前端点击推荐时会发送 Matomo event（`eventCategory=rec_click`、`eventAction=dataset_id`）。
   - 转化：`matomo_log_conversion` 中 `dataset_id` 与推荐项匹配。
2. 关联方法示例：
   - 同一 `user_id`（登录用户）或 `idvisit`（访客）下，`server_time` 在曝光时间后的若干分钟内的行为视为推荐带来的结果。
   - 若前端记录了 `request_id` 并随事件上报 Matomo，自定义维度 `dimensionX` 可直接 join。

## 5. 新版 `evaluate.py` 的扩展要点

要利用 Matomo 数据评估上线版本，可在现有脚本基础上新增：

```text
- 读取曝光日志（本地 Parquet/SQLite）
- 对每条曝光的 (user, dataset, request_id) 查 Matomo 行为
- 计算 CTR = 点击次数 / 曝光次数
- 计算 CVR = 转化次数 / 曝光次数
- 比较不同 `model_run_id`（即不同推荐算法版本）的指标
```

可把结果追加到 `data/evaluation/summary.json`，字段示例：

```json
{
  "model_run_id": "023d4dfe6dda...",
  "exposure_cnt": 1250,
  "click_cnt": 210,
  "ctr": 0.168,
  "conversion_cnt": 45,
  "cvr": 0.036,
  "avg_rank_score": 0.41
}
```

## 6. 基线 vs. 新版本对比

- **历史基线**：当前 `summary.json` 给出了历史行为下的转化、召回、排序指标，可作为新版本上线前的参考。
- **上线后**：
  1. 每日/每周运行 `python -m pipeline.evaluate --with-exposure`（可扩展为 CLI 参数）生成新版本指标。
  2. 将结果与最新一次训练 run（`models/model_registry.json` 中的 `run_id`）对齐，写入仪表盘或发送告警。
  3. 在 `docs/MODEL_CICD.md` 中约定：上线 24h 若 CTR/CVR 明显下降，则回滚到 `history` 中的上一版本。

## 7. Matomo 字段速查

| 字段 | 来源 | 说明 |
| --- | --- | --- |
| `matomo_log_action.name` | `matomo_log_action` | 包含 URL/事件名称：`dataDetail/13196` → 可解析 dataset_id |
| `matomo_log_link_visit_action.idvisit` | `matomo_log_link_visit_action` | 与 `matomo_log_visit.idvisit` 关联用户会话 |
| `matomo_log_link_visit_action.server_time` | - | 行为发生时间 |
| `matomo_log_conversion.revenue` | `matomo_log_conversion` | 转化金额，可用于 GMV 指标 |

## 8. 参考实现建议

1. **曝光持久化**：在 `app/main.py` 的 `_build_response_items` 返回前，写入 `data/evaluation/exposure_log.parquet`（可用 `pyarrow` 追加），或发送到 Kafka → Spark/Flink。内容包含 `model_run_id`、推荐列表等。
2. **评估脚本扩展**：在 `pipeline/evaluate.py` 中新增函数 `_load_exposure_log()` 和 `_merge_with_matomo()`，最终输出 CTR/CVR。
3. **仪表盘/告警**：
   - 利用 Matomo Web UI 或导出数据到 BI 工具（如 Superset/Grafana）。
   - 自定义阈值，例如 CTR 低于历史均值 20% 触发告警。

## 9. 注意事项

- 确保推荐 API 与前端上报使用统一的 `model_run_id`，可从 `models/model_registry.json` 读取。
- 未登录用户需使用 Matomo 的 `idvisitor` 或 Cookie ID 保持一致。
- 如需 A/B 测试，可在曝光日志中添加 `experiment_id` 字段，并在评估脚本中按实验组别输出指标。

通过上述流程，就算不直接写 Matomo，也能利用 Matomo 的原始行为数据 + 本地曝光日志建立完整的推荐效果评估链路。上线新算法后，只需保持日志对齐，即可持续追踪 CTR/CVR/GMV 等核心指标并与历史基线对比。

## 10. 示例数据

运行 `scripts/generate_mock_baseline.py` 会在 `data/matomo/` 与 `data/evaluation/` 下生成一份小规模的示例数据，再执行 `python3 -m pipeline.evaluate` 即可得到曝光/CTR/CVR 的基准报告。适合在本地自测流程或演示给非生产环境使用。
