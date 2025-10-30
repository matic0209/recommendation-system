# Sentry 快速启动指南

## ✅ 集成完成

Sentry 已成功集成到推荐系统中，包括：

- ✅ FastAPI 在线服务
- ✅ Airflow 离线流水线
- ✅ 自定义上下文和标签
- ✅ 性能追踪
- ✅ 错误分组和过滤

**测试结果**: 所有 6 项测试通过 ✓

---

## 🚀 立即开始使用

### 1. 设置环境变量

在 `.env` 文件中添加（已在 .env.example 中配置）:

```bash
SENTRY_DSN=https://dc36186bcb57efbe0ff952f994c21be3@trace.dianshudata.com/11
SENTRY_ENVIRONMENT=production
SENTRY_TRACES_SAMPLE_RATE=0.1
SENTRY_PROFILES_SAMPLE_RATE=0.1
```

### 2. 重新部署服务

```bash
# Docker Compose 方式
docker-compose down
docker-compose up -d recommendation-api airflow-scheduler airflow-webserver

# 或者本地开发
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 3. 查看监控数据

登录 Sentry 查看监控数据：
- URL: https://trace.dianshudata.com
- 项目: #11

---

## 📊 监控范围

### FastAPI 在线服务
- `/similar/{dataset_id}` - 相似推荐
- `/recommend/detail/{dataset_id}` - 详情页推荐
- 所有异常和降级事件
- 性能追踪（P50/P95/P99延迟）

### Airflow 离线流水线
- 所有 DAG 任务失败
- 数据质量检查失败
- 模型训练异常

### 关键上下文
- `request_id` - 请求追踪
- `user_id` - 用户追踪
- `algorithm_version` - 模型版本
- `degrade_reason` - 降级原因
- `experiment_variant` - 实验变体

---

## 🧪 验证集成

运行测试脚本验证：

```bash
SENTRY_DSN="你的DSN" python scripts/test_sentry.py
```

预期输出:
```
总计: 5 通过, 0 失败
```

---

## 📁 创建的文件

| 文件 | 说明 |
|------|------|
| `app/sentry_config.py` | Sentry 配置模块 |
| `scripts/test_sentry.py` | 验证测试脚本 |
| `docs/SENTRY_INTEGRATION.md` | 完整集成文档 |
| `SENTRY_QUICKSTART.md` | 本快速启动指南 |

---

## 📝 修改的文件

| 文件 | 改动 |
|------|------|
| `app/main.py` | 添加 Sentry 初始化和上下文追踪 |
| `airflow/dags/recommendation_pipeline.py` | 添加任务失败回调 |
| `requirements.txt` | 添加 sentry-sdk[fastapi] |
| `.env.example` | 添加 Sentry 配置示例 |

---

## 🎯 关键功能

### 1. 自动异常捕获
所有未捕获的异常会自动发送到 Sentry，包含完整堆栈跟踪和上下文。

### 2. 降级事件追踪
系统降级时（Redis 超时、模型失败等）会自动记录到 Sentry，便于分析降级频率和原因。

### 3. 性能监控
追踪关键操作的性能：
- 召回阶段耗时
- 排序阶段耗时
- Redis 读写耗时
- 整体请求延迟

### 4. 用户行为追踪
通过 `request_id` 和 `user_id` 追踪特定请求和用户的问题。

### 5. 实验追踪
记录每个请求的实验变体，便于 A/B 测试分析。

---

## 🔧 使用示例

### 手动捕获异常

```python
from app.sentry_config import capture_exception_with_context

try:
    risky_operation()
except Exception as e:
    capture_exception_with_context(
        e,
        level="error",
        dataset_id=123,
        additional_info="额外上下文",
    )
```

### 记录重要消息

```python
from app.sentry_config import capture_message_with_context

capture_message_with_context(
    "Redis 连接失败，切换到降级模式",
    level="warning",
    redis_url=redis_url,
)
```

### 性能追踪

```python
from app.sentry_config import start_transaction, start_span

with start_transaction(name="recommendation", op="http.server"):
    with start_span(op="recall", description="多路召回"):
        candidates = multi_channel_recall()

    with start_span(op="ranking", description="LightGBM 排序"):
        ranked = apply_ranking(candidates)
```

---

## ⚠️ 注意事项

### 1. 敏感信息
以下请求头会自动过滤，不会发送到 Sentry：
- `authorization`
- `cookie`
- `x-api-key`

### 2. 采样率
- 生产环境建议 10-20% 采样率
- 开发环境可以使用 100% 采样率
- 根据流量调整以控制成本

### 3. 事件分组
使用 `fingerprint` 参数控制异常分组：
```python
capture_exception_with_context(
    exc,
    fingerprint=["endpoint_name", type(exc).__name__],
)
```

---

## 📚 更多信息

详细文档请参阅：
- `docs/SENTRY_INTEGRATION.md` - 完整集成指南
- [Sentry Python SDK](https://docs.sentry.io/platforms/python/)
- [Sentry FastAPI 集成](https://docs.sentry.io/platforms/python/guides/fastapi/)

---

## 🎉 完成

Sentry 集成已完成！现在您可以：

1. ✅ 实时追踪错误和异常
2. ✅ 监控服务性能
3. ✅ 分析降级事件
4. ✅ 追踪用户问题
5. ✅ 评估实验效果

登录 https://trace.dianshudata.com 开始使用！
