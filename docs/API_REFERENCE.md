# 在线推荐服务 API 参考

本文档说明 `app/main.py` 暴露的 REST 接口，覆盖请求参数、响应结构、异常返回以及埋点信息。所有描述基于当前最终实现。

---

## 通用说明

- 基础 URL：`http://<host>:8000`
- 内容类型：`application/json`
- 认证：默认开放，可在网关层补充 JWT / AK/SK
- Swagger：`/docs`；ReDoc：`/redoc`
- 请求头建议携带 `X-Request-ID`；若未提供服务端会自动生成并回写，同时返回 `X-Response-Time`
- 所有时间字段使用 ISO8601 字符串，金额/得分为浮点数

返回体中 `reason` 字段包含召回来源（如 `behavior+tag`），`context.experiment_variant` 记录实验变体。

---

## 1. 健康检查

### `GET /health`

用于探活及观测缓存/模型加载状态。

**响应示例**
```json
{
  "status": "healthy",
  "cache": "enabled",
  "models_loaded": true,
  "checks": {
    "redis": true,
    "models": true
  }
}
```

- `status` 取值：`healthy` / `degraded` / `unhealthy`
- 当 Redis 不可用或模型未加载时，`checks` 会标记 `false`

---

## 2. 相似推荐

### `GET /similar/{dataset_id}`

返回与目标数据集最相似的列表（不区分登录/未登录）。

| 参数类型 | 参数名 | 是否必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| 路径 | `dataset_id` | 是 | - | 目标数据集 ID |
| 查询 | `limit` | 否 | 10 | 返回条数，范围 1~50 |

**响应示例**
```json
{
  "dataset_id": 13196,
  "similar_items": [
    {
      "dataset_id": 13004,
      "title": "示例数据集",
      "price": 99.0,
      "cover_image": "https://example.com/cover.jpg",
      "score": 0.672,
      "reason": "behavior+tag"
    }
  ]
}
```

- `reason` 可能包含：`behavior`、`content`、`vector`、`tag`、`category`、`price`、`popular` 等组合
- 若所有召回均为空且无法降级，将返回 `503 Service Unavailable`

---

## 3. 详情页推荐

### `GET /recommend/detail/{dataset_id}`

综合召回 + 排序，并对登录用户执行个性化处理与实验权重调整。

| 参数类型 | 参数名 | 是否必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| 路径 | `dataset_id` | 是 | - | 当前页面数据集 ID |
| 查询 | `user_id` | 否 | - | 登录用户 ID，用于个性化与实验分桶 |
| 查询 | `limit` | 否 | 10 | 返回条数，范围 1~50 |

**响应示例**
```json
{
  "dataset_id": 13196,
  "recommendations": [
    {
      "dataset_id": 13004,
      "title": "示例数据集",
      "price": 99.0,
      "cover_image": "https://example.com/cover.jpg",
      "score": 0.812,
      "reason": "behavior+usercf"
    }
  ]
}
```

**曝光埋点**
- 每次返回都会写入 `data/evaluation/exposure_log.jsonl`
- 字段包含：`request_id`、`user_id`、`page_id`、`algorithm_version`、`items`、`context`
- `context` 中的 `degrade_reason` 与 `experiment_variant` 便于离线评估

**特殊情况**
- 当所有候选为空且热门兜底失败：返回 `503`
- 当目标 `dataset_id` 不存在：返回 `404`

---

## 4. 热度榜

### `GET /hot/trending`

返回最近一段时间的热门数据集。

| 查询参数 | 默认值 | 说明 |
| --- | --- | --- |
| `timeframe` | `1h` | 取值 `1h` / `24h` |
| `limit` | `20` | 范围 1~100 |

返回列表仅包含 `dataset_id`、`title`、`price`、`cover_image`，来源优先热度统计，缺失时回退热门榜。

---

## 5. 模型热更新

### `POST /models/reload`

支持主模型替换、影子模型加载与灰度比例配置。

| 字段 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `mode` | string | `primary` | `primary` / `shadow` |
| `source` | string | `models/` | 可指定模型目录（例如 `models/staging/run_123`） |
| `rollout` | float | 0 | 影子模型灰度流量比例（0~1），仅在 `mode=shadow` 时生效 |
| `run_id` | string | 空 | 覆盖模型注册表中的 `run_id` |

`mode=shadow` 时服务会同时加载影子模型，并根据 `rollout` 在请求层随机分流。响应包含 `shadow_rollout` 与实际 `run_id`。

---

## 6. 错误码与响应头

| 状态码 | 场景 |
| --- | --- |
| `200` | 请求成功 |
| `404` | 目标数据集缺失或无候选 |
| `503` | 所有召回失败且无法降级（`context.degrade_reason` 会标明原因） |
| `500` | 其他服务器异常（请查看日志 `logs/`） |

响应头包含：

- `X-Request-ID`：请求唯一 ID（用于日志与曝光对齐）
- `X-Response-Time`：服务器处理耗时（秒）

---

## 7. 调试与排障

1. 使用 `/metrics` 监控延迟、降级、超时等指标，结合 Grafana 告警快速定位瓶颈。
2. 通过曝光日志 `context.degrade_reason` 判断是否触发 Redis/预计算/热门降级。
3. 若观察到 `reason` 仅剩 `popular`，说明召回数据不足，应检查离线流水线是否最新。
4. 实验参数调整后，建议验证：
   - `/recommend/detail` 是否返回新的 `reason`/得分
   - 曝光日志 `experiment_variant` 是否变化
5. 如需禁用 Faiss，可设置 `USE_FAISS_RECALL=0` 并调用 `/models/reload`。

如接口有新增字段或行为变化，请同步更新本文件，以保障前后端、公私域调用的对齐。***
