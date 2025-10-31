# Daily Recommendation Report Viewer

Use the lightweight FastAPI service in `report_viewer/app.py` to browse the HTML/JSON reports produced by `pipeline.daily_report`.

## Prerequisites

- Reports located under `data/evaluation/daily_reports/` (Airflow DAG `daily_recommendation_report` will populate them daily)
- Dependencies already available in the repository (`fastapi`, `jinja2`)
- `REPORT_VIEWER_HOST_PORT` defined in `.env`（默认 `8800`，可按需修改）
- Airflow DAG `daily_recommendation_report` 现已启用 `catchup`，初次启动 Airflow Scheduler 会自动回填从 `2025-10-01` 至今的日报

## Local Run

```bash
uvicorn report_viewer.app:app --host 0.0.0.0 --port 8800
```

Then open <http://localhost:8800>.  
The viewer lists available HTML & JSON files; clicking “Open” renders the HTML report directly.

## Via Docker Compose

`docker-compose.yml` now包含 `report-viewer` 服务（默认映射 `8800` 端口）。只需：

```bash
docker compose up -d report-viewer
```

然后在宿主机访问 <http://localhost:8800> 即可浏览报告。

> 提示：端口取自 `.env` 的 `REPORT_VIEWER_HOST_PORT`，若修改该值需重新 `docker compose up -d report-viewer`。

## 手动在容器内启动（可选）

```bash
docker exec -it recommendation-api \
  /opt/venv/bin/uvicorn report_viewer.app:app --host 0.0.0.0 --port 8800
```

若用 `docker run` 单独启动，请映射端口 `-p 8800:8800`。

## Custom Report Directory

By default the viewer reads from `${DATA_DIR}/evaluation/daily_reports`.  
Override via environment variable:

```bash
DAILY_REPORT_DIR=/custom/path uvicorn report_viewer.app:app --host 0.0.0.0 --port 8800
```
