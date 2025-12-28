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

## 5. 推荐质量监控（2025-12-27新增）

### 5.1 负分过滤监控

应用日志中自动记录负分过滤统计：

```
INFO - Negative score filter: removing 16/30 items (53.3%) with negative scores (sample: [13185, 13116, ...])
```

**关键字段**：
- 过滤数量: `removing X/Y items`
- 负分比例: `(XX.X%)`
- 样本ID: `sample: [...]`

**异常阈值**：
- 负分比例 > 30%: 需检查ranking模型质量
- 全部负分触发fallback: 查看`All X items have negative scores!`警告

### 5.2 Popular召回质量监控

**训练阶段监控**（`pipeline/train_models.py`，2025-12-28新增）:

训练日志中记录Popular榜单构建的质量过滤统计：

```
INFO - Quality filter applied: 38/150 items passed (74.7% filtered out)
INFO -   Filtered breakdown: low_price=45, low_interaction=32, inactive=18, missing_price=0, missing_interaction_count=0, missing_days_since_last_purchase=0
INFO - Popular list quality metrics: avg_price=3.45, avg_interaction=125.3, avg_inactive_days=89.2
```

**关键字段**：
- 过滤比例: `74.7% filtered out`
- 过滤明细: `low_price=X, low_interaction=Y, inactive=Z`
- 质量指标: `avg_price`, `avg_interaction`, `avg_inactive_days`

**Sentry告警**（自动触发）:
- 过滤比例 > 70%（warning）: Popular候选池质量太差
- 过滤后平均价格 < 1.0元（warning）: 过滤阈值可能太宽松
- 过滤后数量 < 5（critical）或 < 25（warning）: 过滤太严格或数据质量问题

**环境变量调优**（参考 `docs/ENVIRONMENT_CONFIG.md`）:
- 提高质量要求: `POPULAR_MIN_PRICE=1.0`, `POPULAR_MIN_INTERACTION=20`
- 放宽过滤条件: `POPULAR_MIN_PRICE=0.3`, `POPULAR_MIN_INTERACTION=5`
- 禁用过滤: `POPULAR_ENABLE_FILTER=false`

**运行时监控**（`app/main.py`，v1.1.1已有）:

应用日志中记录Popular召回过滤统计：

```
INFO - Popular recall quality filter: filtered 12 low-quality items, kept 38 items
```

**关键字段**：
- 过滤数量: `filtered X low-quality items`
- 保留数量: `kept Y items`

**异常阈值**：
- 过滤比例 > 50%: Popular榜单质量可能需要优化
- 保留数量 < 10: Popular召回贡献不足，考虑调整权重

### 5.3 推荐质量日志查询

**运行时日志**（API服务）:
```bash
# 查看负分过滤统计
docker logs recommendation-api 2>&1 | grep "Negative score filter"

# 查看Popular质量过滤统计（运行时）
docker logs recommendation-api 2>&1 | grep "Popular recall quality filter"

# 查看全部负分fallback触发
docker logs recommendation-api 2>&1 | grep "All.*items have negative scores"

# 统计负分占比趋势（需要日志聚合工具）
# 示例: 使用jq解析JSON格式日志
cat app.log | jq 'select(.message | contains("Negative score filter")) | .message'
```

**训练日志**（2025-12-28新增）:
```bash
# 查看Popular榜单质量过滤统计（训练阶段）
docker logs airflow-scheduler 2>&1 | grep "Quality filter applied"

# 查看过滤明细
docker logs airflow-scheduler 2>&1 | grep "Filtered breakdown"

# 查看质量指标
docker logs airflow-scheduler 2>&1 | grep "Popular list quality metrics"

# 查看Sentry告警触发
docker logs airflow-scheduler 2>&1 | grep "Popular quality alerts triggered"

# 完整训练流程日志
docker logs airflow-scheduler 2>&1 | grep -A 10 "Building popular items list"
```

## 6. 运维流程

1. 修改 Prometheus 规则或 Alertmanager 路由后，需 `docker compose up -d --build prometheus alertmanager` 使配置生效。
2. Notification Gateway 支持热重启，如需更新 Sentry 逻辑或目标企业微信用户，重新部署容器即可。
3. 在 Sentry 中创建专门的 Dashboard 或 Issue 过滤器，关注标记为 `[Prometheus]` 的消息。
4. **推荐质量巡检**（2025-12-27/28更新）：
   - 每日检查负分过滤比例，如持续>30%需review ranking模型
   - 每周检查Popular召回过滤率：
     - 训练阶段过滤率 > 70%触发Sentry告警，检查候选池质量
     - 运行时过滤率 > 50%需优化热门榜单构建逻辑
   - 关注fallback触发频率，全部负分情况应极少发生（<1%请求）
   - 监控Sentry中的`popular_items_quality_issues`告警，及时调整环境变量阈值
