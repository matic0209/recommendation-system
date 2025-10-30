# Sentry 监控点完整清单

本文档详细列出了推荐系统中所有 Sentry 监控点的位置、触发条件和使用方式。

## 目录

- [在线服务监控](#在线服务监控)
- [Redis 缓存监控](#redis-缓存监控)
- [模型管理监控](#模型管理监控)
- [降级策略监控](#降级策略监控)
- [离线流水线监控](#离线流水线监控)
- [通知网关监控](#通知网关监控)
- [监控点统计](#监控点统计)

---

## 在线服务监控

### 1. FastAPI 应用启动
**文件**: `app/main.py:1099-1108`

**监控点**:
- Sentry 初始化成功/失败

**触发条件**:
- 应用启动时

---

### 2. HTTP 请求中间件
**文件**: `app/main.py:166-181`

**监控点**:
- 所有 HTTP 请求的上下文设置
- Request ID、endpoint、method、URL 追踪

**触发条件**:
- 每个 HTTP 请求

**标签**:
- `request_id`: 请求唯一标识
- `endpoint`: API 端点
- `method`: HTTP 方法

---

### 3. 相似推荐端点 (`/similar`)
**文件**: `app/main.py:1259-1483`

**监控点**:
- 请求超时 (1356-1362)
- 异常捕获 (1380-1387)
- 推荐上下文设置 (1431-1437)

**触发条件**:
- 推荐请求超时（总超时 > 2s）
- 推荐计算异常
- 降级事件

**标签**:
- `dataset_id`: 数据集 ID
- `algorithm_version`: 算法版本
- `model_variant`: 模型变体 (primary/shadow/fallback)
- `degrade_reason`: 降级原因

**Fingerprint**: `["get_similar", 异常类型]`

---

### 4. 详情页推荐端点 (`/recommend/detail`)
**文件**: `app/main.py:1487-1740`

**监控点**:
- 用户上下文设置 (1525-1526)
- 请求超时 (1602-1608)
- 异常捕获 (1631-1640)
- 推荐上下文设置 (1687-1693)

**触发条件**:
- 同上

**额外标签**:
- `user_id`: 用户 ID
- `experiment_variant`: 实验变体

**Fingerprint**: `["recommend_for_detail", 异常类型]`

---

## Redis 缓存监控

### 1. Redis 连接失败
**文件**: `app/cache.py:60-76`

**监控点**:
- Redis 连接超时或连接错误
- 意外的 Redis 错误

**触发条件**:
- Redis 服务不可用
- 网络问题
- 认证失败

**标签**:
- `redis_host`: Redis 主机
- `redis_port`: Redis 端口
- `redis_db`: 数据库编号
- `error_type`: 错误类型

**Fingerprint**:
- `["redis", "connection_failed"]`
- `["redis", "unexpected_error"]`

**严重级别**:
- 连接失败: warning
- 意外错误: error

---

## 模型管理监控

### 1. 模型注册表解析失败
**文件**: `app/model_manager.py:36-48`

**监控点**:
- `model_registry.json` 文件解析失败

**触发条件**:
- JSON 格式错误
- 文件损坏

**标签**:
- `registry_path`: 注册表文件路径

**Fingerprint**: `["model", "registry_parse_failed"]`

**严重级别**: warning

---

### 2. 模型部署失败
**文件**: `app/model_manager.py:85-148`

**监控点**:
- 模型源目录不存在 (98-111)
- 模型文件复制失败 (137-148)
- 部署过程异常

**触发条件**:
- 源目录路径错误
- 磁盘空间不足
- 权限问题
- 文件系统错误

**标签**:
- `source_dir`: 源目录路径
- `target_dir`: 目标目录路径

**Fingerprint**:
- `["model", "source_not_found"]`
- `["model", "deployment_failed"]`

**严重级别**: error

**面包屑**:
- 部署开始
- 部署成功

---

## 降级策略监控

### 1. 预计算数据加载失败
**文件**: `app/resilience.py:75-86`

**监控点**:
- 预计算推荐文件加载失败

**触发条件**:
- pickle 文件损坏
- 文件不存在
- 反序列化失败

**标签**:
- `precomputed_dir`: 预计算目录路径

**Fingerprint**: `["fallback", "precomputed_load_failed"]`

**严重级别**: warning

---

### 2. 降级级别追踪
**文件**: `app/resilience.py:113-180`

**监控点**:
- Redis 降级失败 (137-145)
- 使用预计算降级 (Level 2) (151-159)
- 使用静态热门降级 (Level 3) (169-177)

**触发条件**:
- Redis 操作失败
- 预计算数据不存在
- 所有降级源均不可用

**面包屑类型**:
- `fallback` 类别
- Level 1 失败: warning
- Level 2 使用: info
- Level 3 使用: warning

**Level 3 标签**:
- `dataset_id`: 数据集 ID
- `user_id`: 用户 ID（如果有）
- `static_popular_count`: 静态热门列表大小

---

## 离线流水线监控

### 1. Airflow DAG 任务失败
**文件**: `airflow/dags/recommendation_pipeline.py:38-77`

**监控点**:
- 任何 DAG 任务执行失败

**触发条件**:
- 任务抛出异常
- 任务超时
- 资源不足

**标签**:
- `dag_id`: DAG 标识
- `task_id`: 任务标识
- `execution_date`: 执行日期
- `try_number`: 重试次数

**Fingerprint**: `["airflow", dag_id, task_id]`

**严重级别**: error

**上下文**:
- Airflow 特定上下文（DAG、任务、执行日期等）

---

### 2. 流水线步骤监控装饰器
**文件**: `pipeline/sentry_utils.py:25-94`

**使用方式**:
```python
from pipeline.sentry_utils import monitor_pipeline_step

@monitor_pipeline_step("extract_load", critical=True)
def extract_and_load_data():
    # 数据抽取逻辑
    pass
```

**监控点**:
- 步骤开始（面包屑）
- 步骤完成（面包屑）
- 步骤失败（异常）

**触发条件**:
- 步骤执行异常

**标签**:
- `step_name`: 步骤名称
- `critical`: 是否为关键步骤
- `duration_seconds`: 执行时长

**Fingerprint**: `["pipeline", step_name, 异常类型]`

**严重级别**:
- critical=True: error
- critical=False: warning

---

### 3. 数据质量监控
**文件**: `pipeline/sentry_utils.py:97-132`

**使用方式**:
```python
from pipeline.sentry_utils import track_data_quality_issue

track_data_quality_issue(
    check_name="missing_price_check",
    severity="warning",
    details={"missing_count": 150, "total_count": 10000},
    metric_value=0.015,
    threshold=0.01,
)
```

**监控点**:
- 数据质量检查失败
- 指标超出阈值

**触发条件**:
- 缺失值过多
- 异常值检测
- 数据分布异常

**标签**:
- `check_name`: 检查名称
- `severity`: 严重程度 (critical/warning/info)
- `metric_value`: 指标值
- `threshold`: 阈值
- 自定义详情字段

**严重级别**: 根据 severity 映射

---

### 4. 模型训练监控
**文件**: `pipeline/sentry_utils.py:135-175`

**使用方式**:
```python
from pipeline.sentry_utils import track_model_training_issue

track_model_training_issue(
    model_name="lightgbm_ranker",
    issue_type="low_performance",
    details={"reason": "NDCG@10 below threshold"},
    metrics={"ndcg_at_10": 0.45, "threshold": 0.50},
)
```

**监控点**:
- 模型性能低于预期
- 模型训练失败
- 过拟合检测

**触发条件**:
- 评估指标低于阈值
- 训练异常
- 验证集性能差

**标签**:
- `model_name`: 模型名称
- `issue_type`: 问题类型
- `metrics`: 模型指标
- 自定义详情字段

**严重级别**:
- training_failed: error
- 其他: warning

---

### 5. 特征存储同步监控
**文件**: `pipeline/sentry_utils.py:178-213`

**使用方式**:
```python
from pipeline.sentry_utils import track_feature_store_sync_issue

track_feature_store_sync_issue(
    operation="redis_sync",
    error=redis_error,
    affected_count=1500,
    total_count=10000,
)
```

**监控点**:
- Redis 特征同步失败
- SQLite 写入失败

**触发条件**:
- Redis 连接失败
- 批量操作部分失败
- 数据格式错误

**标签**:
- `operation`: 操作类型
- `affected_count`: 受影响记录数
- `total_count`: 总记录数
- `failure_rate`: 失败率

**Fingerprint**: `["feature_store", operation, 异常类型]`

**严重级别**: error

---

### 6. 数据库操作监控
**文件**: `pipeline/sentry_utils.py:216-250`

**使用方式**:
```python
from pipeline.sentry_utils import track_database_issue

track_database_issue(
    operation="extract",
    database="business_db",
    error=connection_error,
    query_info={"table": "datasets", "limit": 10000},
)
```

**监控点**:
- 数据库连接失败
- 查询超时
- 数据抽取失败

**触发条件**:
- 连接池耗尽
- 查询超时
- 网络问题

**标签**:
- `operation`: 操作类型
- `database`: 数据库名称
- `query_info`: 查询信息

**Fingerprint**: `["database", database, operation, 异常类型]`

**严重级别**: error

---

## 通知网关监控

### 1. 企业微信 Token 获取失败
**文件**: `notification_gateway/webhook.py:64-93`

**监控点**:
- Token API 调用失败
- Token 获取异常

**触发条件**:
- 企业微信 API 错误
- 网络超时
- 配置错误

**标签**:
- `error_code`: 错误代码
- `error_message`: 错误消息
- `corp_id`: 企业 ID（脱敏）

**Fingerprint**: `["weixin", "get_access_token_failed"]`

**严重级别**: error

---

### 2. 企业微信消息发送失败
**文件**: `notification_gateway/webhook.py:124-156`

**监控点**:
- 消息发送 API 失败
- 消息发送异常

**触发条件**:
- 企业微信 API 错误
- 用户不存在
- 权限不足

**标签**:
- `user_id`: 接收用户 ID
- `error_code`: 错误代码
- `error_message`: 错误消息
- `message_preview`: 消息预览（前 100 字符）

**Fingerprint**: `["weixin", "send_message_failed"]`

**严重级别**:
- API 失败: warning
- 异常: error

---

## 监控点统计

### 按组件统计

| 组件 | 监控点数量 | 关键监控点 |
|------|-----------|-----------|
| FastAPI 在线服务 | 6 | 请求异常、超时、降级 |
| Redis 缓存 | 2 | 连接失败、意外错误 |
| 模型管理 | 3 | 注册表解析、部署失败 |
| 降级策略 | 4 | 预计算加载、降级级别追踪 |
| Airflow DAG | 1 | 任务失败 |
| 离线流水线 | 6 | 步骤失败、数据质量、模型训练 |
| 通知网关 | 2 | Token 获取、消息发送 |
| **总计** | **24** | - |

### 按严重级别统计

| 严重级别 | 数量 | 占比 |
|---------|-----|------|
| Error | 15 | 62.5% |
| Warning | 8 | 33.3% |
| Info | 1 | 4.2% |

### 按触发频率统计

| 频率类别 | 监控点 | 说明 |
|---------|-------|------|
| 高频（每请求）| HTTP 中间件 | 每个 API 请求触发 |
| 中频（故障时）| Redis 连接、降级策略 | 服务故障时触发 |
| 低频（定时/手动）| 模型部署、Airflow 任务 | 定时或手动操作触发 |

---

## 监控点使用建议

### 1. 优先级分类

**P0（必须立即响应）**:
- Airflow DAG 任务失败
- 模型部署失败
- 数据库连接完全失败
- 所有降级级别失效（Level 3）

**P1（需要关注）**:
- Redis 连接失败
- 推荐请求超时
- 数据质量严重问题
- 模型训练失败

**P2（定期检查）**:
- 企业微信消息发送失败
- 模型注册表解析失败
- 预计算数据加载失败

### 2. 告警策略建议

**立即告警**:
```
- 任何 error 级别且 fingerprint 包含 ["airflow", *, *]
- 任何 fingerprint 包含 ["model", "deployment_failed"]
- 降级到 Level 3 的频率 > 10%
```

**每日汇总**:
```
- warning 级别的所有事件
- 降级事件统计
- 数据质量问题汇总
```

**每周回顾**:
```
- 所有 fingerprint 的趋势分析
- 高频问题的根因分析
- 优化建议
```

### 3. Fingerprint 使用规范

所有监控点都使用了 `fingerprint` 进行事件分组，格式为：

```
[组件, 操作/子组件, 异常类型（可选）]
```

**示例**:
- `["redis", "connection_failed"]`
- `["model", "registry_parse_failed"]`
- `["pipeline", "extract_load", "ValueError"]`
- `["weixin", "send_message_failed"]`

这样可以确保：
1. 相同类型的问题被正确分组
2. 不会产生过多的重复 Issue
3. 便于趋势分析和统计

---

## 快速参考

### 查找特定监控点

```bash
# 在代码中搜索 Sentry 监控点
grep -r "capture_exception_with_context" app/ pipeline/ airflow/ notification_gateway/
grep -r "capture_message_with_context" app/ pipeline/ airflow/ notification_gateway/
grep -r "add_breadcrumb" app/ pipeline/ airflow/ notification_gateway/
```

### 查看监控点上下文

每个监控点都包含丰富的上下文信息：
- 基本信息：request_id、user_id、dataset_id
- 环境信息：service、environment、release
- 错误信息：fingerprint、level、exception
- 自定义标签：根据具体监控点而定

---

## 相关文档

- [Sentry 集成指南](./SENTRY_INTEGRATION.md) - 完整集成文档
- [快速启动指南](../SENTRY_QUICKSTART.md) - 5 分钟上手
- [API 参考](./API_REFERENCE.md) - API 接口文档
- [运维手册](./OPERATIONS_SOP.md) - 故障处理流程

---

## 更新日志

- 2025-10-30: 初始版本，覆盖所有核心组件
- 添加了 24 个监控点，覆盖在线服务、离线流水线、通知网关
- 使用 fingerprint 进行事件分组，便于管理

---

## 支持

如有问题或建议，请：
1. 查看本文档的具体监控点说明
2. 参考 [Sentry 集成指南](./SENTRY_INTEGRATION.md) 的故障排查部分
3. 联系团队获取支持
