# 数据集推荐系统

面向数据集商城的推荐平台，覆盖离线流水线、在线 API、监控告警与效果日报。

## 功能概览

- **离线流水线**（`pipeline/`）：抽取业务库与 Matomo 行为数据、构建增强特征、训练排序模型并生成评估报告。
  - Popular榜单质量过滤: 训练阶段应用质量控制（2025-12-28新增）
- **在线服务**（`app/main.py`）：FastAPI + 多路召回 + LightGBM 排序，支持实验灰度、缓存降级与 Prometheus 指标。
  - 多渠道召回: Behavior / Content / Vector / Popular（双层质量过滤）
  - 负分硬截断: 自动过滤低质量推荐（2025-12-27新增）
- **监控链路**（`monitoring/`, `notification_gateway/`）：Prometheus/Alertmanager + 企业微信 + Sentry 自动联动。
- **每日漏斗报表**（`pipeline/daily_report.py`, `report_viewer/`）：生成曝光→点击→明细→转化漏斗并提供 Web 看板。

详细文档见 `docs/README.md`。

## 快速开始

```bash
git clone https://github.com/matic0209/recommendation-system.git
cd recommendation-system
cp .env.example .env    # 按需调整 Redis / MySQL / Sentry 等配置
docker compose up -d --build
```

启动后：

- 推荐 API => `http://localhost:8090`
- Prometheus => `http://localhost:9090`
- Alertmanager => `http://localhost:9093`
- Grafana => `http://localhost:3000`
- 日报 Viewer => `http://localhost:8800`

## 离线流水线

```bash
venv/bin/python -m pipeline.extract_load
venv/bin/python -m pipeline.build_features_v2
venv/bin/python -m pipeline.train_models
venv/bin/python -m pipeline.evaluate_v2
```

自动生成的日报可通过：
```bash
venv/bin/python -m pipeline.daily_report --date 2025-10-31
```

## 在线 API

- `GET /recommend/detail/{dataset_id}?user_id=...`
- `GET /similar/{dataset_id}`
- `POST /models/reload`
- `GET /metrics`

所有请求指标、fallback 比例与错误统计都会被导入 Prometheus。

## 监控与告警

- 新增 `recommendation_exposures_total`、`recommendation_fallback_ratio` 等指标。
- Alertmanager 规则在 fallback 超阈、曝光断流、时延过高时触发告警。
- Notification Gateway 会同时推送企业微信并将告警写入 Sentry。

更多细节参阅 `docs/monitoring.md`。

## 日报 Viewer

`docker compose up -d report-viewer`  
访问 `http://localhost:8800` 可查看每日漏斗、Fallback 拆解、Top 数据集和收入表现。  
详见 `docs/daily_report.md`。

## 当前版本

请参阅 `docs/release_notes.md` 了解最新功能与变更记录。
