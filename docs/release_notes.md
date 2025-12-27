# Release Notes

## v1.1.1 (2025-12-27)

**推荐质量优化**：

- **Popular召回质量过滤**：新增组合质量信号过滤低质量热门item，避免推荐不相关内容
  - 过滤规则: 低价且无人气 (price < 1.90 AND interaction < 66) 或 长期不活跃且交互少 (days_inactive > 180 AND interaction < 30)
  - 批量查询优化: 避免DataFrame fragmentation警告
  - 实现位置: `app/main.py` Line 1688-1750

- **负分硬截断机制**：自动过滤ranking后分数为负的item，提升推荐质量
  - 策略: 直接移除 score < 0 的item
  - Fallback保障: 全部负分时保留top 50%（至少5个）
  - 监控日志: 记录负分比例和过滤统计
  - 实现位置: `app/main.py` Line 2354-2378

- **Tag召回bug修复**：修复标签大小写不统一导致的overlap计算错误
  - 修复: 统一使用lowercase处理target和candidate tags
  - 影响: 提升tag召回准确性
  - 实现位置: `app/main.py` Line 1506-1507

**技术改进**：

- 性能优化: Popular召回使用批量DataFrame查询替代循环查询
- 异常处理: 添加KeyError/ValueError/TypeError安全捕获
- 代码质量: 修复3个Critical Issues和2个Warnings

## v1.1.0 (2025-11-01)

- **每日推荐日报**：`pipeline.daily_report` + Airflow DAG 自动生成，并包含曝光 → 点击 → 明细 → 转化漏斗与收入统计。
- **Report Viewer**：FastAPI 服务根据 JSON 动态渲染看板，支持历史选择、Top 数据集、Fallback/Variant 拆解。
- **实时指标**：新增 `recommendation_exposures_total`、`recommendation_fallback_ratio` 等 Prometheus 指标，跟踪 Fallback 占比与实验维度。
- **告警联动**：Alertmanager 新增曝光断流、Fallback 比例等规则，通过 Notification Gateway 推送企业微信并写入 Sentry。
- **Sentry 集成**：Prometheus 告警自动写入 Sentry，便于统一异常归档。

## v1.0.0 (2025-10)

- 推荐 API + 离线流水线初始上线。
- Matomo request_id 埋点、`pipeline.evaluate_v2` 精确 CTR/CVR。
