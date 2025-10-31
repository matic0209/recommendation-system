# 告警系统整合方案

## 当前架构

```
应用错误/性能问题
    │
    ├─→ Sentry ──→ Sentry Dashboard
    │              (手动查看)
    │
基础设施/业务指标
    │
    └─→ Prometheus ──→ Alertmanager ──→ notification-gateway ──→ 企业微信
```

## 方案对比

### 方案 1: Sentry Webhook → notification-gateway（推荐）✨

**架构：**
```
应用错误              基础设施告警
    │                     │
    ↓                     ↓
Sentry              Alertmanager
    │                     │
    └─────→ notification-gateway ←─────┘
                  │
                  ↓
              企业微信
```

**优点：**
- ✅ 统一告警出口（notification-gateway）
- ✅ 统一消息格式
- ✅ 保持各系统独立性
- ✅ 实现简单，易于维护
- ✅ 可在 notification-gateway 做统一的告警去重、聚合、路由

**缺点：**
- ⚠️ 告警规则分散在两个系统
- ⚠️ 无法在 Alertmanager 统一管理 Sentry 告警

**适用场景：**
- 团队规模小到中等
- 告警量可控
- 需要快速实施

---

### 方案 2: Sentry → Alertmanager → notification-gateway

**架构：**
```
应用错误              基础设施告警
    │                     │
    ↓                     ↓
Sentry ──→ Alertmanager ←──── Prometheus
              │
              ↓
      notification-gateway
              │
              ↓
          企业微信
```

**优点：**
- ✅ Alertmanager 统一管理所有告警
- ✅ 统一的告警分组、抑制、静默规则
- ✅ 统一的告警路由策略

**缺点：**
- ❌ 需要 Sentry webhook → Alertmanager 格式转换
- ❌ Alertmanager 不是为应用错误设计的
- ❌ 丢失 Sentry 丰富的上下文信息（堆栈、breadcrumbs）

**适用场景：**
- 需要统一告警管理
- 告警量大，需要复杂的路由规则

---

### 方案 3: 使用专业告警平台（企业级）

**架构：**
```
Sentry ────┐
           ├──→ Grafana OnCall / PagerDuty / Opsgenie
Alertmanager ─┘              │
                             ↓
                         企业微信 / Slack / Email / 电话
```

**优点：**
- ✅ 专业的告警管理平台
- ✅ 强大的去重、聚合、升级功能
- ✅ 值班管理、事件响应流程
- ✅ 移动端 App 支持

**缺点：**
- ❌ 需要部署额外服务（或付费 SaaS）
- ❌ 学习成本
- ❌ 过度设计（对小团队）

**适用场景：**
- 大型团队
- 7x24 值班需求
- 复杂的事件响应流程

---

## 推荐方案：增强的统一网关

### 实施步骤

#### 1. 在 Sentry 配置 Webhook

在 Sentry 项目设置中：
- Settings → Integrations → WebHooks
- 添加 Webhook URL: `http://your-server:9000/webhook/sentry`
- 选择触发事件：`error.created`, `issue.created`

#### 2. 在 notification-gateway 添加 Sentry webhook 处理

增强 `notification_gateway/webhook.py`：
- 解析 Sentry webhook payload
- 格式化为统一的消息格式
- 添加告警级别映射
- 添加去重逻辑

#### 3. 告警分级策略

| 告警源 | 告警类型 | 级别 | 通知方式 | 示例 |
|--------|---------|------|---------|------|
| Sentry | Error (First seen) | 🚨 Critical | 立即通知 | 新的未知错误 |
| Sentry | Error (Recurring) | ⚠️ Warning | 汇总通知 (每小时) | 已知错误重复 |
| Sentry | Performance | ℹ️ Info | 每日汇总 | 慢查询 |
| Alertmanager | critical | 🚨 Critical | 立即通知 | Redis 宕机 |
| Alertmanager | warning | ⚠️ Warning | 汇总通知 (15分钟) | 磁盘使用率高 |
| Alertmanager | info | ℹ️ Info | 每日汇总 | 数据质量报告 |

#### 4. 告警去重和聚合

在 notification-gateway 实现：
- **指纹去重**：相同指纹的告警在时间窗口内只发送一次
- **时间聚合**：低优先级告警按时间窗口聚合（如每小时汇总）
- **告警抑制**：Critical 告警触发时抑制相关的 Warning 告警

#### 5. 统一消息格式

```
🚨 [应用错误] ValueError in recommend API
━━━━━━━━━━━━━━━━
🏷️  来源: Sentry
📍 环境: production
🔗 服务: recommendation-api
⏰ 时间: 2025-10-31 12:30:45

📝 错误: division by zero in calculate_score
📊 影响: 15 用户, 23 次发生
🔗 详情: https://sentry.io/issues/12345

━━━━━━━━━━━━━━━━
```

---

## 实施优先级

### Phase 1: 基础整合（本次实施）✨
- ✅ Sentry webhook → notification-gateway
- ✅ 统一消息格式
- ✅ 基本的告警路由

### Phase 2: 智能化（后续优化）
- ⏳ 告警去重和聚合
- ⏳ 告警级别自动升级
- ⏳ 告警静默规则

### Phase 3: 平台化（长期规划）
- 📅 告警历史查询
- 📅 告警趋势分析
- 📅 SLA 监控

---

## 对比总结

| 维度 | 方案1: 统一网关 | 方案2: Alertmanager统一 | 方案3: 专业平台 |
|------|----------------|----------------------|---------------|
| 实施难度 | ⭐ 简单 | ⭐⭐ 中等 | ⭐⭐⭐ 复杂 |
| 维护成本 | ⭐ 低 | ⭐⭐ 中等 | ⭐⭐⭐ 高 |
| 功能丰富度 | ⭐⭐ 基础 | ⭐⭐ 中等 | ⭐⭐⭐ 丰富 |
| 适用团队规模 | 小-中 | 中-大 | 大 |
| **推荐指数** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |

---

## 结论

**推荐使用方案 1（增强的统一网关）**，理由：
1. ✅ 快速实施，符合当前架构
2. ✅ 保持 Sentry 和 Alertmanager 的独立性和优势
3. ✅ notification-gateway 作为统一的告警出口，易于扩展
4. ✅ 后续可以逐步增强功能，不需要大规模重构

如果未来团队规模扩大、告警量增加，可以考虑迁移到方案 3（专业平台）。
