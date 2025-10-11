# 模型热更新与灰度发布

本文档说明如何在不重启服务的情况下更新推荐模型，并规划小流量灰度。

## 1. 基本概念

- **Primary 模型**：当前在线使用的模型，文件位于 `models/`。
- **Shadow 模型**：即将上线或灰度中的模型，加载后只在部分请求中生效。
- **Run ID**：模型训练 run 的唯一标识，来自 `models/model_registry.json`。

## 2. 快速使用

1. **部署新主模型**
   ```bash
   python scripts/stage_model.py /path/to/artifacts --deploy
   curl -X POST http://localhost:8000/models/reload -H 'Content-Type: application/json' -d '{"mode":"primary"}'
   ```

2. **加载 Shadow 模型并设置灰度比例**
   ```bash
   python scripts/stage_model.py /path/to/new_run
   curl -X POST http://localhost:8000/models/reload      -H 'Content-Type: application/json'      -d '{"mode":"shadow", "source":"models/staging/new_run", "run_id":"new_run_id", "rollout":0.1}'
   ```

3. **修改灰度比例或回滚**
   ```bash
   curl -X POST http://localhost:8000/models/reload -H 'Content-Type: application/json' -d '{"mode":"primary", "rollout":0.0}'
   ```

## 3. API 说明

`POST /models/reload`

| 字段 | 必填 | 说明 |
| ---- | ---- | ---- |
| `mode` | 是 | `primary` 更新线上模型；`shadow` 加载灰度模型 |
| `source` | 否 | 模型文件目录（shadow 必填；primary 省略时使用 `models/` 现有文件） |
| `run_id` | 否 | 若 `source` 目录内没有 `model_registry.json`，需显式传 run_id |
| `rollout` | 否 | 灰度占比，范围 `[0,1]` |

返回示例：
```json
{
  "status": "ok",
  "mode": "shadow",
  "run_id": "new_run_id",
  "shadow_rollout": 0.1,
  "message": "Shadow model loaded"
}
```

## 4. 曝光与评估

- 推荐接口会写入 `data/evaluation/exposure_log.jsonl`，字段包含 `algorithm_version` 与 `context.variant`（primary/shadow）。
- `python -m pipeline.evaluate` 汇总 `exposure_metrics.json` 和 `summary.json`，可比较各版本 CTR/CVR/GMV，结合 `docs/MATOMO_EVALUATION.md` 做进一步分析。

## 5. 运维建议

- Shadow 模型只保存在内存，不会覆盖 `models/`。上线前先以较小 `rollout` 验证，指标稳定后再切换 primary。
- 回滚：在 `models/model_registry.json` 的 `history` 中找到上一版本 `run_id`，重新部署并调用 `mode=primary` 即可。
- 清理灰度：将 `rollout` 设置为 0，并删除 `models/staging/` 中不再使用的目录。
