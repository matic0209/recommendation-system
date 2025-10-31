# 双告警体系策略

由于 Sentry 不支持自定义 webhook，我们采用**双告警体系**，各司其职。

## 架构设计

```
应用层错误告警（Sentry）
    ↓
Sentry 内置通知（邮件/钉钉/飞书等）
    ↓
开发团队


基础设施/业务指标告警（Prometheus + Alertmanager）
    ↓
notification-gateway
    ↓
企业微信
    ↓
运维团队
```

---

## 一、Sentry 告警配置（应用层错误）

### 适用场景
- 🔥 应用代码错误（Exception、Error）
- 🐛 未捕获的异常
- 📊 性能问题（慢查询、API 超时）
- 💾 数据库错误
- 🔗 第三方服务调用失败

### 推荐配置方式

#### 方案 1: Sentry Alerts（推荐）⭐

在 Sentry 项目中配置 Alert Rules：

1. **进入 Alerts 配置**
   - 项目 → Settings → Alerts
   - Create Alert Rule

2. **Critical 错误立即通知**
   ```
   规则名称: Critical Errors - Production
   触发条件:
     - When: An event is first seen
     - If: level = error OR level = fatal
     - Environment: production

   动作:
     - Send notification via Email to: team@company.com
     - 或: Send to Slack channel #alerts
     - 或: Send to 钉钉/飞书群
   ```

3. **高频错误汇总通知**
   ```
   规则名称: High Frequency Errors
   触发条件:
     - When: An issue's events exceed 100 in 1 hour
     - If: All events
     - Environment: production

   动作:
     - Send notification digest
   ```

4. **新错误立即通知**
   ```
   规则名称: New Issues - Production
   触发条件:
     - When: A new issue is created
     - If: has_tag(environment=production)

   动作:
     - Send notification immediately
   ```

#### 方案 2: Email 通知

**优点：** 简单、可靠、不需要额外配置

**配置步骤：**
1. Settings → Notifications
2. Email Alerts → 添加接收邮箱
3. 配置接收规则：
   - ✅ New issues
   - ✅ Regressions (已解决的问题再次出现)
   - ⬜ Workflow (状态变更，可选)

**建议：**
- 生产环境：立即通知
- 测试环境：每日汇总

#### 方案 3: 钉钉/飞书集成

如果你们用钉钉或飞书，可以使用官方集成：

**钉钉集成：**
1. 创建钉钉群机器人
2. 获取 Webhook URL
3. Sentry → Settings → Integrations → Custom Integration
4. 配置钉钉 Webhook（需要 Sentry 企业版）

**飞书集成：**
- 类似钉钉，需要企业版支持

---

## 二、Alertmanager 告警配置（基础设施）

### 适用场景
- 🖥️  服务器资源告警（CPU、内存、磁盘）
- 🔌 服务可用性（Redis、MySQL、API）
- 📊 业务指标告警（推荐成功率、延迟）
- 🔍 数据质量告警
- 📈 性能指标告警

### 当前配置（企业微信）✅

已完成配置，通过 notification-gateway 发送到企业微信。

### 优化建议

#### 1. 告警分级

在 `monitoring/alertmanager.yml` 中配置不同级别的路由：

```yaml
route:
  receiver: "default"
  group_by: ["alertname", "severity"]
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 12h

  routes:
    # Critical 告警：立即通知，不分组
    - receiver: "critical"
      matchers:
        - severity=critical
      group_wait: 0s
      group_interval: 5m
      repeat_interval: 1h
      continue: false

    # Warning 告警：15分钟汇总
    - receiver: "warning"
      matchers:
        - severity=warning
      group_wait: 15m
      group_interval: 15m
      repeat_interval: 3h
      continue: false

    # Info 告警：每日汇总
    - receiver: "info"
      matchers:
        - severity=info
      group_wait: 1h
      group_interval: 24h
      repeat_interval: 24h

receivers:
  - name: "critical"
    webhook_configs:
      - url: "http://host.docker.internal:9000/webhook/critical"
        send_resolved: true

  - name: "warning"
    webhook_configs:
      - url: "http://host.docker.internal:9000/webhook/warning"
        send_resolved: true

  - name: "info"
    webhook_configs:
      - url: "http://host.docker.internal:9000/webhook/info"
        send_resolved: true
```

#### 2. 告警静默规则

**工作时间外静默低优先级告警：**

```yaml
inhibit_rules:
  # Critical 告警会抑制同一服务的 Warning 告警
  - source_matchers:
      - severity=critical
    target_matchers:
      - severity=warning
    equal: ["alertname", "service"]

  # 服务宕机时抑制性能告警
  - source_matchers:
      - alertname=ServiceDown
    target_matchers:
      - alertname=~".*Latency|.*Performance"
    equal: ["service"]
```

#### 3. 告警模板优化

在 `notification_gateway/webhook.py` 中针对不同级别使用不同格式：

```python
def format_alert_by_severity(alerts, receiver_name):
    """根据告警级别格式化消息"""
    severity = alerts[0].get('labels', {}).get('severity', 'unknown')

    if severity == 'critical':
        # Critical: 详细信息，包含所有上下文
        return format_critical_alert(alerts, receiver_name)
    elif severity == 'warning':
        # Warning: 简洁信息，只包含关键字段
        return format_warning_alert(alerts, receiver_name)
    else:
        # Info: 汇总信息，多条告警合并
        return format_info_digest(alerts, receiver_name)
```

---

## 三、告警职责划分

| 告警类型 | 负责系统 | 通知渠道 | 通知对象 | 响应时间 |
|---------|---------|---------|---------|---------|
| **应用错误** | Sentry | 邮件/钉钉/飞书 | 开发团队 | 立即 |
| **API 5xx** | Sentry | 邮件/钉钉/飞书 | 开发+运维 | 立即 |
| **性能问题** | Sentry | 邮件（汇总） | 开发团队 | 每日 |
| **服务宕机** | Alertmanager | 企业微信 | 运维团队 | 立即 |
| **资源告警** | Alertmanager | 企业微信 | 运维团队 | 15分钟 |
| **业务指标** | Alertmanager | 企业微信 | 产品+开发 | 1小时 |
| **数据质量** | Alertmanager | 企业微信 | 数据团队 | 每日 |

---

## 四、消息格式对比

### Sentry 邮件通知（推荐格式）

**主题：**
```
[Production] ValueError in recommendation API
```

**内容：**
```
Error: ValueError: division by zero

Location: app/main.py in get_recommend at line 156
Environment: production
First Seen: 2025-10-31 12:30:45
Occurrences: 15
Users Affected: 8

View on Sentry: https://trace.dianshudata.com/issues/12345
```

### Alertmanager 企业微信（当前格式）

```
🔥 [FIRING] RedisConnectionFailed
━━━━━━━━━━━━━━━━
🚨 严重程度: CRITICAL
🖥️  实例: redis:6379
📦 任务: redis-exporter
⏰ 时间: 2025-10-31 12:35:00

📝 Redis 连接失败
Redis 服务不可用，推荐系统将降级运行
━━━━━━━━━━━━━━━━
```

---

## 五、推荐的配置方案

### 最小化配置（快速实施）✨

**Sentry：**
- ✅ 使用邮件通知
- ✅ 只通知 Critical 错误
- ✅ 测试环境邮件汇总

**Alertmanager：**
- ✅ 保持当前企业微信配置
- ✅ 按 severity 分级（critical/warning/info）
- ✅ 配置告警静默时间

### 进阶配置（优化体验）

**Sentry：**
- ⏳ 配置钉钉/飞书集成（如果有）
- ⏳ 配置 Alert Rules 实现智能告警
- ⏳ 设置错误率阈值告警

**Alertmanager：**
- ⏳ 实现告警去重和聚合
- ⏳ 配置值班轮换
- ⏳ 添加告警自动恢复通知

### 企业级配置（长期规划）

- 📅 引入统一告警平台（Grafana OnCall / PagerDuty）
- 📅 实现告警升级机制
- 📅 建立事件响应流程
- 📅 告警质量分析和优化

---

## 六、实施步骤

### 第一步：Sentry 邮件通知（5分钟）

```bash
# 在 Sentry 后台配置
1. Settings → Notifications → Email
2. 添加接收邮箱：your-team@company.com
3. 勾选通知类型：
   - ✅ New Issues
   - ✅ Regressions
   - ⬜ Deploys (可选)
```

### 第二步：Alertmanager 分级（15分钟）

```bash
# 更新配置文件
vim monitoring/alertmanager.yml

# 应用配置
docker compose restart alertmanager
```

### 第三步：测试验证（10分钟）

```bash
# 测试 Sentry
curl http://localhost:8090/test-sentry?error_type=exception

# 测试 Alertmanager
curl -X POST http://localhost:9093/api/v2/alerts -d @test_alert.json

# 检查通知是否收到
```

---

## 七、监控和优化

### 每周检查

- 📊 告警数量趋势
- 🔍 误报率（False Positive）
- ⏱️  平均响应时间
- ✅ 告警处理率

### 优化指标

| 指标 | 目标值 | 当前值 | 改进方向 |
|------|--------|--------|----------|
| 误报率 | < 10% | ? | 优化告警规则 |
| 平均响应时间 | < 15分钟 | ? | 优化通知渠道 |
| 告警覆盖率 | > 90% | ? | 添加监控指标 |

---

## 八、常见问题

### Q1: 告警太多怎么办？

**A:**
1. 提高告警阈值
2. 增加告警分组时间
3. 设置静默规则
4. 归档或忽略已知问题

### Q2: 如何避免告警风暴？

**A:**
1. 配置 `group_wait` 和 `group_interval`
2. 使用 `inhibit_rules` 抑制级联告警
3. 设置 `repeat_interval` 避免重复通知

### Q3: 错过重要告警怎么办？

**A:**
1. 使用多个通知渠道
2. 配置告警升级机制
3. 设置值班轮换
4. 定期检查 Sentry/Prometheus 后台

---

## 总结

**推荐配置：**

```
┌─────────────────────────────────────────┐
│            告警体系总览                  │
├─────────────────────────────────────────┤
│                                          │
│  应用层（Sentry）                        │
│  ├─ 错误告警 → 邮件 → 开发团队          │
│  └─ 性能告警 → 邮件汇总 → 开发团队      │
│                                          │
│  基础设施层（Alertmanager）             │
│  ├─ Critical → 企业微信 → 运维团队      │
│  ├─ Warning → 企业微信汇总 → 运维团队   │
│  └─ Info → 企业微信日报 → 产品团队      │
│                                          │
└─────────────────────────────────────────┘
```

**关键原则：**
1. ✅ 职责清晰：应用层 vs 基础设施层
2. ✅ 分级通知：Critical 立即，Warning 汇总，Info 日报
3. ✅ 避免过载：合理的阈值和分组策略
4. ✅ 持续优化：根据反馈调整规则

