# Sentry Webhook 配置指南

## 目标

将 Sentry 错误告警通过 notification-gateway 发送到企业微信，实现统一的告警出口。

## 架构图

```
Sentry (应用错误)          Prometheus (基础设施)
       │                           │
       ↓                           ↓
   /webhook/sentry          Alertmanager
              │                   │
              └────→ notification-gateway ←────┘
                          │
                          ↓
                      企业微信
```

## 配置步骤

### 1. 在 Sentry 后台配置 Webhook

#### 方法 A：项目级别 Webhook（推荐）

1. 登录 Sentry: https://trace.dianshudata.com
2. 进入项目设置：
   - 选择项目（如 `recommendation-system`）
   - 点击 **Settings** → **Integrations**
   - 找到 **WebHooks**
3. 添加 WebHook：
   - 点击 **Add to Project** 或 **Configure**
   - **Callback URLs**: `http://your-server-ip:9000/webhook/sentry`
     - 生产环境示例：`http://123.456.789.0:9000/webhook/sentry`
     - 如果在内网：`http://10.0.0.1:9000/webhook/sentry`
   - 勾选需要通知的事件：
     - ✅ `error.created` - 新错误创建（推荐）
     - ✅ `issue.created` - 新问题创建（推荐）
     - ⬜ `issue.resolved` - 问题已解决（可选）
     - ⬜ `issue.ignored` - 问题被忽略（可选）
4. 点击 **Save Changes**

#### 方法 B：集成级别 Webhook

如果找不到项目集成，可以使用告警规则：

1. 进入项目设置 → **Alerts**
2. 点击 **Create Alert Rule**
3. 配置条件：
   - **When**: An event is first seen
   - **If**: All events
4. 配置动作：
   - **Then**: Send a notification via **WebHooks**
   - **Webhook URL**: `http://your-server-ip:9000/webhook/sentry`
5. 保存规则

---

### 2. 验证 Webhook 配置

#### 测试 1：手动触发测试事件

在 Sentry webhook 配置页面，通常有 **Send Test Event** 按钮：
1. 点击 **Send Test Event**
2. 检查企业微信是否收到测试消息
3. 检查 notification-gateway 日志：
   ```bash
   docker logs notification-gateway --tail 50 | grep Sentry
   ```

#### 测试 2：使用 API 触发真实错误

在生产环境触发测试错误：
```bash
# 触发测试异常
curl http://localhost:8090/test-sentry?error_type=exception
```

等待 5-10 秒，检查：
1. Sentry 后台是否有新的 Issue
2. 企业微信是否收到告警消息
3. notification-gateway 日志

---

### 3. 高级配置

#### 告警过滤（推荐）

为了避免告警风暴，建议在 Sentry 中配置过滤规则：

**在 Sentry Alert Rules 中配置：**

1. **Critical 错误立即通知：**
   - Condition: `level:error OR level:fatal`
   - Action: Send webhook

2. **Warning 错误每小时汇总：**
   - Condition: `level:warning`
   - Frequency: Hourly digest

3. **忽略已知错误：**
   - 在 Issue 页面点击 **Ignore**
   - 或设置 Inbound Filters 忽略特定错误

#### 告警分级策略

| 错误级别 | Sentry Level | 通知方式 | 示例 |
|---------|-------------|---------|------|
| 🚨 Critical | fatal/error (首次) | 立即通知 | 新的未知错误、数据库连接失败 |
| ⚠️ Warning | error (重复) | 每小时汇总 | 已知错误再次发生 |
| ℹ️ Info | warning | 每日汇总 | 慢查询、性能警告 |

---

### 4. 消息格式示例

配置完成后，收到的企业微信消息格式：

```
🆕 [Sentry 应用告警] ValueError in calculate_score
━━━━━━━━━━━━━━━━
🔥 级别: ERROR
🏷️  来源: Sentry
📍 环境: production
🖥️  服务: recommendation-api
📝 位置: app.main.get_recommend
⏰ 首次发现: 2025-10-31 12:30:45
📊 状态: 🔴 未解决
🔢 发生次数: 1
👥 影响用户: 15

🔗 详情: https://trace.dianshudata.com/issues/12345
━━━━━━━━━━━━━━━━
```

对比 Alertmanager 告警：

```
🔥 [FIRING] RedisConnectionFailed
━━━━━━━━━━━━━━━━
🚨 严重程度: CRITICAL
🏷️  来源: Prometheus
🖥️  实例: redis:6379
📦 任务: redis-exporter
⏰ 时间: 2025-10-31 12:35:00

📝 Redis 连接失败
Redis 服务不可用，推荐系统将降级运行
━━━━━━━━━━━━━━━━
```

---

### 5. 故障排查

#### 问题 1: Webhook 无法访问

**症状：** Sentry 显示 webhook 发送失败

**可能原因：**
- notification-gateway 服务未启动
- 端口 9000 未开放
- 防火墙阻止

**解决方法：**
```bash
# 检查服务状态
docker ps --filter "name=notification-gateway"

# 检查端口监听
netstat -tlnp | grep 9000

# 测试连接
curl http://localhost:9000/health
```

#### 问题 2: 收不到企业微信消息

**症状：** Webhook 调用成功，但没有收到企业微信消息

**可能原因：**
- 企业微信配置错误
- 用户 ID 不正确
- access_token 获取失败

**解决方法：**
```bash
# 查看 notification-gateway 日志
docker logs notification-gateway --tail 50

# 检查企业微信配置
docker exec notification-gateway env | grep WEIXIN

# 手动测试发送
curl -X POST http://localhost:9000/test
```

#### 问题 3: 告警过多

**症状：** 收到大量重复告警

**解决方法：**
1. 在 Sentry 中设置 Alert Rules 的频率限制
2. 使用 Sentry 的 **Rate Limits** 功能
3. 在 notification-gateway 中实现去重（未来功能）

---

### 6. 监控和维护

#### 日常检查

```bash
# 检查最近的 webhook 调用
docker logs notification-gateway --tail 100 | grep "Sentry"

# 检查发送成功率
docker logs notification-gateway --tail 1000 | grep "消息发送成功\|消息发送失败" | wc -l
```

#### 性能监控

notification-gateway 本身也会将自己的错误发送到 Sentry，形成自监控闭环。

---

## 告警策略建议

### Production 环境

- **Error/Fatal**: 立即通知
- **Warning**: 每小时汇总
- **Performance Issues**: 每日汇总

### Testing 环境

- **所有级别**: 立即通知（用于测试验证）
- 使用 `environment:testing` 标签区分

### Development 环境

- 不配置 webhook，仅在 Sentry 后台查看

---

## 下一步

配置完成后，建议：

1. ✅ 触发几个测试错误验证流程
2. ✅ 观察一周的告警量和质量
3. ✅ 根据实际情况调整告警规则
4. ⏳ 后续可以考虑添加告警去重和聚合功能

---

## 参考文档

- [Sentry Webhooks 官方文档](https://docs.sentry.io/product/integrations/integration-platform/webhooks/)
- [企业微信 API 文档](https://developer.work.weixin.qq.com/document/path/90236)
- 项目告警架构：`docs/ALERT_INTEGRATION_GUIDE.md`
