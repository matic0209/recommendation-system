# 自动化上线蓝图（单人维护）

本文档描述如何将推荐算法从「提交代码」到「生产服务重载」实现自动化，适用于仅由一名维护者负责的场景。流程分为四个阶段：代码验证 → 离线流水线 → 模型部署 → 监控回归。

---

## 1. 代码验证（CI）

现有 `.github/workflows/tests.yml` 已在每次提交到 `master/main` 时执行：

1. 安装依赖、运行 `pytest`。
2. 调用 `python -m pipeline.train_models` 和 `python -m pipeline.evaluate` 验证训练逻辑。

如需更严格的校验，可在 CI 中追加：

- `python -m pipeline.data_quality_v2`（快速数据质量检查）。
- `scripts/run_pipeline.sh --dry-run`（验证各阶段命令）。
- `flake8` / `black --check` 等代码规范工具。

---

## 2. 离线流水线触发

新增脚本 `scripts/automate_release.py` 用于自动化：

1. 调用 Airflow API 触发 `recommendation_pipeline` DAG。
2. 轮询 DAG 直到成功或失败。
3. DAG 成功后调用推荐 API `/models/reload` 热更新模型。

### 配置方式

1. 在部署环境或 CI 中设置以下环境变量（通常放在 GitHub Secrets）：

   | 变量 | 说明 |
   | --- | --- |
   | `AIRFLOW_BASE_URL` | Airflow Web API 地址，例 `https://airflow.example.com/api/v1` |
   | `AIRFLOW_USERNAME` / `AIRFLOW_PASSWORD` | Airflow API 认证信息 |
   | `AIRFLOW_DAG_ID` | 默认为 `recommendation_pipeline` |
   | `MODEL_RELOAD_URL` | 推荐 API `/models/reload` 完整地址 |
   | `MODEL_RELOAD_TOKEN` | 若 API 受鉴权，填 Bearer Token |

2. 在 GitHub Actions 新增 workflow（示例 `model-release.yml`）：

```yaml
name: Model Release

on:
  push:
    branches: [ main ]        # 合并主干时触发
  workflow_dispatch: {}       # 支持手动触发

jobs:
  release:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: Trigger pipeline & reload model
        env:
          AIRFLOW_BASE_URL: ${{ secrets.AIRFLOW_BASE_URL }}
          AIRFLOW_USERNAME: ${{ secrets.AIRFLOW_USERNAME }}
          AIRFLOW_PASSWORD: ${{ secrets.AIRFLOW_PASSWORD }}
          MODEL_RELOAD_URL: ${{ secrets.MODEL_RELOAD_URL }}
          MODEL_RELOAD_TOKEN: ${{ secrets.MODEL_RELOAD_TOKEN }}
        run: |
          python scripts/automate_release.py
```

> 若 GitHub 无法直接访问私有网络，可将 workflow 改为部署机上的 cron 或自托管 runner。脚本同样适用于手动运行：`python scripts/automate_release.py`.

---

## 3. 模型部署与灰度

推荐 API 的 `/models/reload` 会：

- 读取最新模型目录（`models/`）和 `model_registry.json`。
- 支持 Shadow 模型（配置 `shadow_rollout`）以小流量灰度。

建议流程：

1. Airflow DAG 在完成训练后更新 `models/model_registry.json`。
2. `scripts/automate_release.py` 成功后服务自动加载。
3. 若需灰度，可在流水线生成后调用 `/models/reload` 携带 `"mode": "shadow", "rollout": 0.1`。

如需发布 Docker 镜像，可在同一个 workflow 中增加：

```bash
docker build -t registry.example.com/recsys:${GITHUB_SHA} .
docker push registry.example.com/recsys:${GITHUB_SHA}
ssh deploy@server "docker pull registry.example.com/recsys:${GITHUB_SHA} && docker compose up -d recommendation-api"
```

---

## 4. 监控与回滚

自动化上线后需持续观察：

- Prometheus 指标：`recommendation_latency_seconds`、`recommendation_degraded_total`、`data_quality_score` 等。
- Airflow DAG 运行结果：成功/失败会被 `scripts/automate_release.py` 捕获并返回。
- 日级对账文件：`data/evaluation/reconciliation_*.json`。
- Alertmanager：企业微信告警确认事件闭环。

回滚建议：

1. 模型异常：使用 `models/model_registry.json` 找到上一版本 `run_id`，执行 `POST /models/reload` 指定 `run_id` 或在 `models/staging/` 替换。
2. 部署故障：通过 docker compose 回滚镜像或 `git revert` 配置后重新触发 workflow。
3. 告警：若数据质量告警频繁，考虑自动静默并阻断后续自动部署，可在 workflow 中加入质量阈值检测（如读取 `data_quality_metrics.prom`）。

---

## 5. 一键执行脚本（可选）

若在自有服务器运行，可编写简单 shell：

```bash
#!/bin/bash
set -euo pipefail

git pull origin main
docker compose build recommendation-api
docker compose up -d recommendation-api
python scripts/automate_release.py
```

结合 cron 或 systemd timer，即可实现场内自动化。

---

## 6. TODO（后续迭代）

- 将数据质量阈值嵌入 CI，失败时阻断自动部署。
- 在 workflow 中添加模型指标对比，自动决定是否发布。
- 把 `reconciliation_*.json` 推送到告警渠道，实现指标回流闭环。

通过上述方案，单人维护者只需合并代码，其余离线训练、模型上线、监控告警即可自动串联完成。
