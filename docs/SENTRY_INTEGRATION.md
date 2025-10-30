# Sentry 集成指南

本文档介绍推荐系统的 Sentry 错误追踪和性能监控集成。

## 目录

- [快速开始](#快速开始)
- [配置说明](#配置说明)
- [监控范围](#监控范围)
- [使用示例](#使用示例)
- [最佳实践](#最佳实践)
- [故障排查](#故障排查)

---

## 快速开始

### 1. 安装依赖

```bash
pip install sentry-sdk[fastapi]==2.18.0
```

或者使用 requirements.txt:

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

在 `.env` 文件中添加 Sentry 配置：

```bash
# Sentry DSN（必需）
SENTRY_DSN=https://dc36186bcb57efbe0ff952f994c21be3@trace.dianshudata.com/11

# 环境标识（可选，默认 production）
SENTRY_ENVIRONMENT=production

# 性能追踪采样率（可选，默认 0.1）
SENTRY_TRACES_SAMPLE_RATE=0.1

# 性能分析采样率（可选，默认 0.1）
SENTRY_PROFILES_SAMPLE_RATE=0.1

# 版本标识（可选）
SENTRY_RELEASE=v1.0.0
```

### 3. 验证集成

运行测试脚本验证 Sentry 集成：

```bash
python scripts/test_sentry.py
```

预期输出：

```
============================================================
Sentry 集成测试
============================================================
测试 1: Sentry 基本初始化
✓ Sentry 初始化成功
...
总计: 6 通过, 0 失败
```

### 4. 启动服务

```bash
# FastAPI 服务
uvicorn app.main:app --host 0.0.0.0 --port 8000

# 查看日志确认 Sentry 已启用
# 应该看到: "Sentry monitoring enabled for recommendation-api"
```

---

## 配置说明

### 环境变量

| 变量名 | 必需 | 默认值 | 说明 |
|--------|------|--------|------|
| `SENTRY_DSN` | 是 | - | Sentry 项目 DSN，从 Sentry 控制台获取 |
| `SENTRY_ENVIRONMENT` | 否 | production | 环境标识（production/staging/development） |
| `SENTRY_TRACES_SAMPLE_RATE` | 否 | 0.1 | 性能追踪采样率（0.0-1.0） |
| `SENTRY_PROFILES_SAMPLE_RATE` | 否 | 0.1 | 性能分析采样率（0.0-1.0） |
| `SENTRY_RELEASE` | 否 | unknown | 版本标识，建议使用 git commit SHA |

### 采样率建议

**生产环境**:
- `SENTRY_TRACES_SAMPLE_RATE=0.1` (10%)
- `SENTRY_PROFILES_SAMPLE_RATE=0.1` (10%)

**开发/测试环境**:
- `SENTRY_TRACES_SAMPLE_RATE=1.0` (100%)
- `SENTRY_PROFILES_SAMPLE_RATE=1.0` (100%)

---

## 监控范围

### 1. FastAPI 在线服务

#### 监控内容
- ✅ API 端点错误和异常
- ✅ 推荐引擎失败（召回、排序、特征加载）
- ✅ 降级事件（熔断、超时、Redis 故障）
- ✅ 超时问题（Redis、模型推理、总请求）
- ✅ 性能追踪（请求延迟、模型推理时间）
- ✅ 用户行为追踪（request_id、user_id）

#### 监控的端点
- `GET /similar/{dataset_id}` - 相似推荐
- `GET /recommend/detail/{dataset_id}` - 详情页推荐
- `GET /health` - 健康检查
- `GET /hot/trending` - 热度榜单
- `POST /models/reload` - 模型热更新

#### 自动捕获
- HTTP 异常（5xx 错误）
- 未捕获的 Python 异常
- Redis 连接错误
- 模型推理失败
- 数据库查询错误

### 2. Airflow 离线流水线

#### 监控内容
- ✅ DAG 任务失败
- ✅ 数据质量异常
- ✅ 模型训练失败
- ✅ 数据库连接问题
- ✅ 特征同步失败

#### 监控的 DAG 任务
- `extract_load` - 数据抽取
- `build_features` - 特征构建
- `data_quality` - 数据质量检查
- `train_models` - 模型训练
- `recall_engine` - 召回引擎构建
- `evaluate` - 模型评估
- `reconcile_metrics` - 指标对账

#### 自动捕获
- 任务执行失败
- Python 异常
- 数据验证错误

### 3. 关键上下文信息

Sentry 会自动收集以下上下文：

**请求级别**:
- `request_id` - 请求唯一标识
- `endpoint` - API 端点
- `method` - HTTP 方法
- `url` - 完整请求 URL

**用户级别**:
- `user_id` - 用户 ID
- `ip_address` - 用户 IP（如果 send_default_pii=True）

**推荐级别**:
- `algorithm_version` - 算法版本（MLflow run_id）
- `model_variant` - 模型变体（primary/shadow/fallback）
- `experiment_variant` - 实验变体
- `degrade_reason` - 降级原因
- `channel_weights` - 召回通道权重

**Airflow 级别**:
- `dag_id` - DAG ID
- `task_id` - 任务 ID
- `execution_date` - 执行时间
- `try_number` - 重试次数

---

## 使用示例

### 1. 基本异常捕获

在代码中捕获异常：

```python
from app.sentry_config import capture_exception_with_context

try:
    result = risky_operation()
except Exception as e:
    capture_exception_with_context(
        e,
        level="error",
        fingerprint=["my_module", type(e).__name__],
        additional_context="额外的调试信息",
    )
    # 继续降级处理
```

### 2. 记录自定义消息

记录重要事件或警告：

```python
from app.sentry_config import capture_message_with_context

capture_message_with_context(
    "Redis 连接失败，切换到 SQLite 降级",
    level="warning",
    redis_url=redis_url,
    fallback_source="sqlite",
)
```

### 3. 添加面包屑

追踪事件序列：

```python
from app.sentry_config import add_breadcrumb

add_breadcrumb(
    message="开始加载召回模型",
    category="model",
    level="info",
    model_path="/path/to/model.pkl",
)
```

### 4. 性能追踪

追踪关键操作的性能：

```python
from app.sentry_config import start_transaction, start_span

# 开始一个事务
with start_transaction(name="recommendation_request", op="http.server"):
    # 追踪召回阶段
    with start_span(op="recall", description="multi-channel recall"):
        candidates = multi_channel_recall(dataset_id)

    # 追踪排序阶段
    with start_span(op="ranking", description="LightGBM ranking"):
        ranked = apply_ranking(candidates)

    return ranked
```

### 5. 设置用户和请求上下文

在处理请求时设置上下文：

```python
from app.sentry_config import set_user_context, set_request_context

# 设置用户上下文
set_user_context(user_id=123)

# 设置请求上下文
set_request_context(
    request_id="req_abc123",
    endpoint="/recommend/detail/456",
    dataset_id=456,
    limit=10,
)
```

---

## 最佳实践

### 1. 异常分组

使用 `fingerprint` 参数控制异常分组：

```python
capture_exception_with_context(
    exc,
    fingerprint=["endpoint_name", type(exc).__name__],
    # 这样可以按端点和异常类型分组
)
```

### 2. 敏感信息处理

Sentry 配置已自动过滤敏感请求头：
- `authorization`
- `cookie`
- `x-api-key`

如需额外过滤，在 `app/sentry_config.py:before_send_filter` 中添加。

### 3. 降级事件监控

系统降级时记录关键信息：

```python
from app.sentry_config import add_breadcrumb

# 记录降级原因
add_breadcrumb(
    message="触发降级策略",
    category="degradation",
    level="warning",
    data={
        "reason": "redis_timeout",
        "fallback_level": "precomputed",
    },
)
```

### 4. 采样策略

根据环境调整采样率：

**高流量端点**（如 `/similar`）:
- 采样率: 10-20%
- 避免发送过多事件

**低流量端点**（如 `/models/reload`）:
- 采样率: 100%
- 确保捕获所有问题

**Airflow 任务**:
- 采样率: 50-100%
- 任务数量少，可以全量监控

### 5. 告警规则

在 Sentry 中配置告警规则：

**推荐配置**:
1. **错误率告警**: 错误率 > 5% 时触发
2. **新问题告警**: 首次出现的错误立即通知
3. **回归告警**: 已解决的问题重新出现时通知
4. **性能告警**: P95 延迟 > 1s 时触发

---

## 故障排查

### 问题 1: Sentry 未初始化

**症状**:
```
Sentry monitoring disabled (SENTRY_DSN not configured)
```

**解决方案**:
1. 检查 `.env` 文件中是否设置了 `SENTRY_DSN`
2. 确认环境变量已加载：`echo $SENTRY_DSN`
3. 重启服务

### 问题 2: 事件未发送到 Sentry

**症状**: 代码执行正常，但 Sentry 控制台没有事件

**排查步骤**:

1. **检查采样率**:
   ```bash
   # 临时设置 100% 采样率测试
   export SENTRY_TRACES_SAMPLE_RATE=1.0
   ```

2. **检查网络连接**:
   ```bash
   curl -I https://trace.dianshudata.com
   ```

3. **查看日志**:
   ```bash
   # 启用 Sentry 调试日志
   export SENTRY_DEBUG=1
   uvicorn app.main:app
   ```

4. **手动刷新事件**:
   ```python
   import sentry_sdk
   sentry_sdk.flush(timeout=10)
   ```

### 问题 3: 过滤掉了重要异常

**症状**: 某些异常未在 Sentry 中显示

**排查**:

检查 `app/sentry_config.py:before_send_filter` 是否过滤了该异常：

```python
# 示例：HTTPException 4xx 被过滤
if exc_type.__name__ == "HTTPException":
    if hasattr(exc_value, "status_code") and exc_value.status_code < 500:
        return None  # 这里被过滤了
```

**解决**: 根据需要调整过滤逻辑。

### 问题 4: 性能追踪未显示

**症状**: Issue 正常，但 Performance 页面无数据

**原因**:
- `SENTRY_TRACES_SAMPLE_RATE=0` 或未设置
- 未使用 `start_transaction`

**解决**:
```bash
export SENTRY_TRACES_SAMPLE_RATE=0.5
```

### 问题 5: Docker 容器中 Sentry 失败

**症状**: 容器启动后 Sentry 不工作

**排查**:
1. 确认环境变量传递到容器：
   ```bash
   docker exec recommendation-api env | grep SENTRY
   ```

2. 检查 docker-compose.yml 是否配置了环境变量：
   ```yaml
   environment:
     SENTRY_DSN: ${SENTRY_DSN}
   ```

---

## 监控看板

### 推荐指标

在 Sentry 中关注以下指标：

**错误类型**:
- `TimeoutError` - 超时错误
- `KeyError` - 数据缺失
- `TypeError` - 类型错误
- `ConnectionError` - 连接失败

**标签统计**:
- `endpoint` - 按端点统计错误
- `model_variant` - 按模型变体统计
- `degrade_reason` - 降级原因分布
- `experiment_variant` - 实验变体影响

**性能指标**:
- P50, P95, P99 延迟
- 吞吐量（TPM - Transactions Per Minute）
- 错误率趋势

---

## 相关资源

- [Sentry Python SDK 文档](https://docs.sentry.io/platforms/python/)
- [Sentry FastAPI 集成](https://docs.sentry.io/platforms/python/guides/fastapi/)
- [Sentry 性能监控](https://docs.sentry.io/product/performance/)
- [Sentry 告警规则](https://docs.sentry.io/product/alerts/)

---

## 支持

如有问题，请：
1. 查看本文档的故障排查部分
2. 运行 `python scripts/test_sentry.py` 验证配置
3. 查看 Sentry 项目日志和事件详情
4. 联系团队获取支持
