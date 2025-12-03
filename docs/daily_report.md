# Daily Recommendation Report

`pipeline.daily_report` 聚合推荐 API 与 Matomo 数据，生成每日漏斗报表。  
生成的 JSON 文件位于 `data/evaluation/daily_reports/recommendation_report_YYYY-MM-DD.json`，由 `report_viewer` 在线渲染。

## 1. 数据来源

| 数据 | 来源 | 说明 |
| --- | --- | --- |
| 曝光 | `data/evaluation/exposure_log.jsonl` | 推荐 API 每次返回时记录 request_id、variant、degrade_reason 等 |
| 点击 | `matomo_log_link_visit_action` | 通过 `custom_dimension_4` (request_id，兼容 1/2 作为回退) 精确匹配推荐点击 |
| 明细页浏览 | Matomo link action | 解析 `dataDetail/`、`dataAPIDetail/` 访问次数 |
| 转化 | `matomo_log_conversion` | 按 dataset_id 汇总目标转化和 Revenue |

## 2. 执行方式

### Airflow 定时任务
`airflow/dags/daily_recommendation_report.py` 每日 06:00 UTC 执行：

```bash
cd /opt/recommend
python -m pipeline.daily_report --date {{ ds }}
```

已经启用 `catchup=True`，首次部署 Scheduler 会自动回填历史日报。

### 手动执行
```bash
venv/bin/python -m pipeline.daily_report --date 2025-10-31
```
默认日期为前一天（UTC）。

## 3. JSON 结构

```
{
  "summary": {
    "date": "2025-10-31",
    "total_exposures": 123456,
    "total_clicks": 9876,
    "total_detail_views": 5432,
    "total_conversions": 321,
    "total_revenue": 12345.67,
    "overall_ctr": 0.08,
    "overall_detail_rate": 0.55,
    "overall_detail_to_conversion_rate": 0.059,
    ...
  },
  "breakdowns": {
    "by_version": [...],
    "by_endpoint": [...],
    "by_position": [...],
    "top_datasets": [...]
  },
  "operations": {
    "fallback_exposures": 123,
    "fallback_ratio": 0.01,
    "fallback_breakdown": [...],
    "variant_breakdown": [...],
    "experiment_breakdown": [...],
    "funnel_rates": {
      "ctr": 0.08,
      "click_to_detail": 0.55,
      "detail_to_conversion": 0.059
    }
  },
  "history_chart": [
    {"date": "...", "total_exposures": ..., "total_clicks": ..., ...},
    ...
  ]
}
```

## 4. Viewer 功能

运行：
```bash
docker compose up -d --build report-viewer
```
访问 `http://<host>:${REPORT_VIEWER_HOST_PORT}`，即可查看：

- 总曝光 / 点击 / 明细 / 转化量与转化率
- 曝光→点击→明细→转化漏斗
- Fallback、Variant、实验拆解
- Top 数据集表现及收入

右上角支持切换历史日报，“Download JSON” 可下载原始数据。

## 5. 常见问题

- **JSON 空或为 0**：检查曝光日志是否包含 `req_` 前缀、Matomo 是否同步了 request_id。
- **detail_views 过高**：确认 Matomo 上报的 URL 是否包含 `dataDetail`，避免其他页面污染。
- **转化缺失**：若 Matomo 未携带 request_id，则按 dataset 汇总，会导致 detail → 转化率偏高；需统一前端埋点。
