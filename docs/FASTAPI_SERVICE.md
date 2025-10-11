# FastAPI 服务实现说明

本文详细介绍在线服务层（`app/main.py`）的架构、关键流程以及扩展方式，适用于需要阅读、调试或扩展推荐 API 的开发人员。

---

## 1. 服务概述

- 框架：FastAPI + Uvicorn
- 模式：异步路由 + 线程池隔离
- 关键能力：
  - 启动时加载模型、召回索引、元数据、实验配置
  - Redis 缓存/特征库优先，SQLite 兜底
  - 多渠道召回融合 + LightGBM 排序 + 实验权重
  - 超时、熔断、三级降级、异常曝光
  - Prometheus 指标与曝光日志埋点

在线服务与离线流水线通过 `models/` 与 `data/processed/` 解耦，支持热更新。

---

## 2. 启动流程（`load_models`）

1. 初始化日志、线程池、缓存：
   - `get_cache()`：Redis 缓存（推荐结果、热门榜）
   - `get_hot_tracker()`：记录热点数据集中台
2. 加载模型包：
   - `item_sim_behavior.pkl`、`item_sim_content.pkl`
   - `top_items.json`、`item_recall_vector.json`
   - LightGBM 排序模型 `rank_model.pkl`（及 ONNX）
3. 加载特征与画像：
   - 数据集元信息/标签、统计表、用户画像、用户历史
   - 若配置 `FEATURE_REDIS_URL`，优先从 Redis 获取，失败降级 SQLite
4. 补充多路召回索引：
   - `user_similarity.pkl`、`tag_to_items.json`、`category_index.json`、`price_bucket_index.json` 等
5. 加载实验配置：
   - `config/experiments.yaml`，通过 `assign_variant` 决定请求的实验变体
6. 初始化 `FallbackStrategy`（Redis → 预计算 → 热门）与 `HealthChecker`。

通过将所有对象挂载在 `app.state` 上，避免全局变量同时保持线程安全。

---

## 3. 请求处理流程

### 3.1 中间件

`request_context_middleware`：

- 读取/生成 `X-Request-ID`
- 记录响应耗时，添加 `X-Response-Time`
- 方便日志、曝光与实验对齐

### 3.2 通用工具

- `_run_in_executor`：将阻塞操作提交至线程池，并更新队列长度指标 `recommendation_thread_pool_queue_size`
- `_call_blocking`：在协程内对阻塞函数设置超时，超时后写入 `recommendation_timeouts_total`
- `_serve_fallback`：统一处理三级降级，并打点 `recommendation_degraded_total`
- `_augment_with_multi_channel`：加载标签/行业/价格/UserCF 等召回结果叠加

### 3.3 接口逻辑

1. **`GET /similar/{dataset_id}`**
   - 优先读取缓存
   - 组合行为/内容/向量召回 + 多渠道补充
   - 排序失败或超时时降级热门榜
   - 无结果：返回 503，并在曝光日志标记 `degrade_reason`

2. **`GET /recommend/detail/{dataset_id}`**
   - 可选 `user_id`，用于：
     - 实验变体选择（调整渠道权重）
     - 个性化过滤（移除近期浏览/购买）
     - UserCF 与标签偏好加权
   - 执行 LightGBM 排序（带熔断器）
   - 融合请求、实验、降级信息写入曝光日志

3. **`GET /hot/trending`**
   - Redis 热度统计 → 热门榜兜底 → 元数据补全

4. **`POST /models/reload`**
   - 支持主模型替换与影子模型
   - `rollout` 控制流量比例（0~1）
   - 影子模型在请求阶段按比例随机分流并在曝光中标记 `variant`

5. **`GET /health`、`/metrics`**
   - Health：检查缓存与模型加载状态
   - Metrics：Prometheus 标准输出

---

## 4. 指标与日志

主要指标：

| 名称 | 标签 | 说明 |
| --- | --- | --- |
| `recommendation_requests_total` | `endpoint` / `status` | 请求量与成功率 |
| `recommendation_latency_seconds` | `endpoint` | 响应耗时直方图 |
| `recommendation_count` | `endpoint` | 返回条数 |
| `recommendation_degraded_total` | `endpoint` / `reason` | 降级次数 |
| `recommendation_timeouts_total` | `endpoint` / `operation` | 超时统计 |
| `recommendation_thread_pool_queue_size` | 无 | 线程池队列深度快照 |
| `cache_requests_total`、`cache_hit_rate` | - | 缓存命中情况 |
| `fallback_triggered_total` | `reason` / `level` | 兜底来源 |

曝光日志：`data/evaluation/exposure_log.jsonl`，字段包括：

- `request_id`、`user_id`、`page_id`
- `algorithm_version`（模型 run_id）
- `context.endpoint`、`context.variant`（primary/shadow）
- `context.degrade_reason`、`context.experiment_variant`

---

## 5. 扩展与调试

### 5.1 扩展建议

- **新增召回通道**：在 `pipeline/recall_engine_v2.py` 产出索引，再于 `_augment_with_multi_channel` 中注入得分。
- **实时特征**：可在 `FeatureEngineV2` 中新增字段，并扩展 Redis 同步逻辑。
- **鉴权限流**：建议在 Nginx/Ingress 或 FastAPI 中间件层引入（例如 JWT、请求配额）。
- **A/B 实验**：增加新的 Experiment 配置或实现多实验同时分桶。
- **监控**：可根据需要在 `app/metrics.py` 添加业务指标（如命中率、曝光量）。

### 5.2 调试技巧

1. 使用 `curl` 或 Swagger UI (`/docs`) 快速测试，注意传入 `X-Request-ID` 便于追踪。
2. 查看日志与曝光：`tail -f data/evaluation/exposure_log.jsonl`。
3. 监控降级：关注 `recommendation_degraded_total` 与 `reason` 标签。
4. 实验验证：修改 `config/experiments.yaml` → `/models/reload` → 调用接口，检查响应与曝光中的 `experiment_variant`。
5. 线程池与超时：如观察到队列堆积，可通过环境变量调整最大线程数或召回权重。

---

## 6. 常见问题

| 问题 | 处理建议 |
| --- | --- |
| `/health` 返回 `degraded` | 检查 Redis 或模型文件是否缺失，必要时运行 `scripts/run_pipeline.sh --sync-only` |
| 接口频繁返回热门兜底 | 查看召回索引是否最新、PIPELINE 是否成功运行、或是否触发了超时/熔断 |
| 排序模型报错或无法加载 | 确认 `rank_model.pkl` 有效，必要时删除后重新训练；ONNX 导出失败可忽略（有日志提醒） |
| 实验结果未生效 | 检查 `user_id` 是否传入、曝光日志 `experiment_variant` 是否变化、配置文件是否仍为 `status: active` |
| Redis 不可访问 | 服务会自动降级 SQLite/热门；恢复后运行 `--sync-only` 同步特征，并 `/models/reload` 刷新缓存 |

---

通过以上说明，可快速理解在线服务的关键逻辑并在出现问题时进行定位与扩展。若代码或指标发生变更，请同步更新本文档。***
