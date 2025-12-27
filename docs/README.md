# Recommendation Platform – Documentation

This document收敛了推荐系统的现状、核心组件以及如何部署/运维。

## 1. 系统概览

| 模块 | 说明 | 关键路径 |
| --- | --- | --- |
| 推荐 API | FastAPI 服务，提供 `/recommend/detail/{dataset_id}`、`/similar/{dataset_id}` 等实时推荐接口 | `app/main.py` |
| 离线流水线 | 数据抽取、特征构建、模型训练与评估脚本 | `pipeline/` |
| 监控链路 | Prometheus + Alertmanager + Notification Gateway + Sentry | `monitoring/`, `notification_gateway/` |
| 日报体系 | `pipeline.daily_report` 输出 JSON，`report_viewer` 动态渲染漏斗看板 | `pipeline/daily_report.py`, `report_viewer/` |

## 2. 快速部署

```bash
git clone https://github.com/matic0209/recommendation-system.git
cd recommendation-system
cp .env.example .env   # 按需调整 Redis/Sentry/端口
docker compose up -d --build
```

- 推荐 API 默认监听 `8090`（`RECOMMEND_API_HOST_PORT`）。
- 日报 viewer 默认映射端口 `8800`（`REPORT_VIEWER_HOST_PORT`）。
- Prometheus / Alertmanager / Grafana 端口参见 `.env`。

## 3. 数据 & 模型

1. **ETL**：`python -m pipeline.extract_load` 支持业务库（JSON/MySQL）与 Matomo 数据。
2. **特征构建**：`python -m pipeline.build_features_v2` 生成增强特征并同步到 SQLite/Redis。
3. **训练与评估**：
   ```bash
   python -m pipeline.train_models
   python -m pipeline.evaluate_v2
   ```
4. **模型热更新**：推荐 API 启动时自动加载最新模型，可通过 `/models/reload` 刷新。

## 4. 监控体系

| 采集项 | 描述 | Prometheus 指标 |
| --- | --- | --- |
| 请求量、时延 | `recommendation_requests_total`, `recommendation_latency_seconds` |
| 曝光 & 漏斗 | `recommendation_exposures_total`, `recommendation_fallback_ratio` |
| Fallback 统计 | `fallback_triggered_total`, `recommendation_degraded_total` |
| 错误告警 | `error_total` (FastAPI)、Alertmanager 规则 |

Alertmanager 告警（延迟、曝光断流、fallback 比例等）通过 Notification Gateway 推送企业微信并写入 Sentry。

## 5. 每日报告 & Viewer

Airflow DAG `daily_recommendation_report` 每日 06:00 UTC 运行：
```bash
python -m pipeline.daily_report --date {{ ds }}
```
输出 JSON 到 `data/evaluation/daily_reports/`。  
FastAPI viewer（`report_viewer/app.py`）根据 JSON 即时渲染漏斗看板，包含：

- 曝光 → 点击 → 明细页 → 下单漏斗及转化率
- Fallback/Variant/实验组拆解
- Top 数据集表现

详细交互见 `docs/daily_report.md`。

## 6. 重要脚本

| 脚本 | 用途 |
| --- | --- |
| `pipeline/daily_report.py` | 生成每日推荐日报 |
| `report_viewer/app.py` | 报表查看服务 |
| `notification_gateway/webhook.py` | 接收 Prometheus/Sentry Webhook，推送企业微信 & 写入 Sentry |
| `app/metrics.py` | Prometheus 指标定义 |

## 7. 开发常用命令

```bash
venv/bin/python -m pipeline.extract_load --full-refresh
venv/bin/python -m pipeline.train_models
venv/bin/python -m pipeline.daily_report --date 2025-10-31
uvicorn app.main:app --reload --port 8000
```

## 8. 推荐算法核心机制

### 8.1 多渠道召回策略

推荐系统使用四个召回渠道，每个渠道独立归一化后按权重融合：

| 渠道 | 说明 | 权重 | 质量控制 |
| --- | --- | --- | --- |
| Behavior | 用户协同过滤 + item-item相似度 | 1.2 | - |
| Content | 基于标签/描述的TF-IDF相似度 | 1.0 | - |
| Vector | SBERT语义向量相似度 | 0.8 | - |
| Popular | 全局热门榜单 | 0.1 | **质量过滤**（2025-12-27新增） |

**Popular召回质量过滤规则**：
- 过滤低价且无人气: `price < 1.90 AND interaction_count < 66`
- 过滤长期不活跃且交互少: `days_inactive > 180 AND interaction_count < 30`

### 8.2 排序与负分处理

**LightGBM Ranker特性**：
- 模型类型: LambdaRank (objective="lambdarank", metric="ndcg")
- 输出范围: (-∞, +∞) 原始预测分数（非归一化）
- 负分含义: 模型预测该item质量低于平均水平

**负分硬截断机制**（2025-12-27新增）：
- 自动过滤所有 `score < 0` 的item
- Fallback策略: 如全部负分，保留分数最高的50%（至少5个）
- 监控日志: 记录负分比例和过滤数量

### 8.3 Bug修复记录

**Tag召回大小写bug**（2025-12-27修复）：
- 问题: 标签大小写不统一导致overlap计算错误
- 修复: 统一使用lowercase处理target和candidate tags
- 影响: 提升tag召回准确性

参考实现: `app/main.py` Line 1506-1507, 1688-1750, 2354-2378

## 9. 版本说明

详见 `docs/release_notes.md`，当前版本重点：

- JSON 日报 + Viewer 漏斗展示
- Prometheus 曝光 / Fallback 实时指标
- Alertmanager → Notification Gateway → Sentry 联动
- **推荐质量优化**（2025-12-27）:
  - Popular召回质量过滤
  - 负分硬截断机制
  - Tag召回bug修复
