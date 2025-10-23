# Airflow Integration

This directory contains a self-contained Docker Compose stack that runs Apache Airflow and schedules the recommendation pipeline DAG in `airflow/dags/recommendation_pipeline.py`.

## Quick start

1. **Copy the environment template**
   ```bash
   cp airflow/.env.example airflow/.env
   ```
   Update the Fernet and webserver secrets with secure random values.

2. **Launch the Airflow services**
   ```bash
   cd airflow
   docker compose up airflow-init
   docker compose up -d
   ```
   The init step applies database migrations and creates the default admin user defined in `.env`.

3. **Access the UI**
   - Web server: http://localhost:8080
   - Default credentials: values of `AIRFLOW_USERNAME` / `AIRFLOW_PASSWORD`

4. **Enable and trigger the DAG**
   - 在 UI 中解除 `recommendation_pipeline` 的暂停状态，然后点击右上角 **Trigger DAG**。如需全量抽取，在弹窗的 *Config* 区域填入：
     ```json
     {
       "full_refresh": true
     }
     ```
   - CLI 方式：
     ```bash
     docker compose run --rm airflow-cli airflow dags trigger recommendation_pipeline \
       --conf '{"full_refresh": true}'
     ```

The stack mounts the entire repository into `/opt/recommend` so the DAG can execute the local modules (`pipeline.extract_load`, `pipeline.build_features`, etc.). The `docker-compose.yaml` also forwards `PYTHONPATH` and `RECOMMEND_PROJECT_DIR` so Airflow operators load code without extra packaging.

## Customising

- **Python interpreter**: Set `RECOMMEND_PYTHON_BIN` in `.env` if you prefer a different Python executable inside the containers.
- **Schedule**: Update `schedule_interval` in `airflow/dags/recommendation_pipeline.py` to change the run cadence (defaults to daily 02:00).
- **Extract behaviour**: Manual DAG triggers can include `{"full_refresh": true}` or `{"dry_run": true}` in the JSON configuration to force a complete reload or skip writes; otherwise the extract task runs incrementally using CDC watermarks stored in `data/_metadata/extract_state.json`.
- **Performance metrics**: Each extract task appends run telemetry to `data/evaluation/extract_metrics.json`, which can be surfaced in Airflow via custom sensors or external dashboards.
- **MLflow 集成**: DAG 运行时同样会读取根目录 `.env`，可在其中设置 `MLFLOW_TRACKING_URI`/`MLFLOW_EXPERIMENT_NAME` 让训练任务写入集中式模型库。
- **Additional dependencies**: Extend `AIRFLOW_PIP_ADDITIONAL_REQUIREMENTS` to install extra Python packages inside the Airflow image.

## Cleanup

Stop the stack and remove volumes when done:
```bash
cd airflow
docker compose down --volumes
```
