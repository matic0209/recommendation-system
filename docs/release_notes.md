# Release Notes

## v1.1.0 (2025-11-01)

- **每日推荐日报**：`pipeline.daily_report` + Airflow DAG 自动生成，并包含曝光 → 点击 → 明细 → 转化漏斗与收入统计。
- **Report Viewer**：FastAPI 服务根据 JSON 动态渲染看板，支持历史选择、Top 数据集、Fallback/Variant 拆解。
- **实时指标**：新增 `recommendation_exposures_total`、`recommendation_fallback_ratio` 等 Prometheus 指标，跟踪 Fallback 占比与实验维度。
- **告警联动**：Alertmanager 新增曝光断流、Fallback 比例等规则，通过 Notification Gateway 推送企业微信并写入 Sentry。
- **Sentry 集成**：Prometheus 告警自动写入 Sentry，便于统一异常归档。

## v1.0.0 (2025-10)

- 推荐 API + 离线流水线初始上线。
- Matomo request_id 埋点、`pipeline.evaluate_v2` 精确 CTR/CVR。
