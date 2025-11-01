# Monitoring & Alerting

本系统采用 Prometheus + Alertmanager + Notification Gateway + Sentry 的监控链路，并输出 Prometheus 指标供 Grafana 展示。

## 1. Prometheus 指标

重点指标定义在 `app/metrics.py`：

| 指标 | 说明 |
| --- | --- |
| `recommendation_requests_total{endpoint, status}` | 推荐请求量与成功/失败次数 |
| `recommendation_latency_seconds` | 推荐接口时延直方图 |
| `recommendation_count{endpoint}` | 单次返回的推荐数 |
| `recommendation_exposures_total{endpoint,variant,experiment_variant,degrade_reason}` | 曝光明细（含 Fallback/实验信息） |
| `recommendation_fallback_ratio{endpoint}` | 运行时 fallback 比例 |
| `fallback_triggered_total{reason,level}` | 各级 fallback 命中次数 |
| `error_total{error_type,endpoint}` | 服务端错误计数 |

Scrape 配置见 `monitoring/prometheus.yml`，默认抓取 `recommendation-api:8000/metrics`、Airflow metrics 等。

## 2. Alertmanager 规则

`monitoring/alerts/recommendation.yml` 包含核心告警：

| 告警 | 条件 | 说明 |
| --- | --- | --- |
| `RecommendationHighErrorRate` | 错误率 > 5% | 接口稳定性异常 |
| `RecommendationLatencyHigh` | P95 > 200ms | 性能退化，需排查模型或下游依赖 |
| `DegradeRateSpike` | 降级速率 > 1/s | Redis/召回链路可能出问题 |
| `RecommendationFallbackRatioHigh` | fallback_ratio > 20% （持续 5m） | 模型或缓存命中异常 |
| `RecommendationExposureDrop` | 5 分钟曝光为 0 | 推荐或埋点链路断流 |
| `DataQualityScoreDrop/SchemaContractViolation` | 离线数据异常 | 依赖 `pipeline/data_quality` 指标 |

Alertmanager 路由见 `monitoring/alertmanager.yml`，所有告警发送至 Notification Gateway 的 webhook；数据质量相关单独转发。

## 3. Notification Gateway & Sentry

`notification_gateway/webhook.py` 支持两个入口：

- `/webhook/<receiver>`：接收 Alertmanager 告警，格式化后发企业微信；同时调用 `capture_message_with_context` 将告警写入 Sentry（包含 alertname、severity、endpoint、summary 等上下文）。
- `/webhook/sentry`：接收 Sentry 自身的 webhook，转发给企业微信。

部署：
```bash
docker compose up -d --build notification-gateway
```
默认监听 `:9000`，Alertmanager webhook URL 形如 `http://host.docker.internal:9000/webhook/recommend-default`。

## 4. Grafana Dashboard 建议

可按以下指标构建仪表：

- 请求量 / 错误率：`sum(rate(recommendation_requests_total{status!="success"}[5m]))`
- 时延：`histogram_quantile(0.95, sum(rate(recommendation_latency_seconds_bucket[5m])) by (le, endpoint))`
- 漏斗：`sum(rate(recommendation_exposures_total{variant="primary"}[5m]))`、`sum(rate(recommendation_exposures_total{degrade_reason="none"}[5m]))`
- Fallback 占比：`recommendation_fallback_ratio{endpoint="recommend_detail"}`
- 模型健康：`fallback_triggered_total` + `recommendation_degraded_total`

## 5. 运维流程

1. 修改 Prometheus 规则或 Alertmanager 路由后，需 `docker compose up -d --build prometheus alertmanager` 使配置生效。
2. Notification Gateway 支持热重启，如需更新 Sentry 逻辑或目标企业微信用户，重新部署容器即可。
3. 在 Sentry 中创建专门的 Dashboard 或 Issue 过滤器，关注标记为 `[Prometheus]` 的消息。
