# Alertmanager 企业微信通知网关

这个服务接收 Alertmanager 的告警通知并转发到企业微信。

## 配置完成情况

✅ 已创建 notification-gateway webhook 服务
✅ 已配置 docker-compose.yml
✅ 已配置 .env 环境变量
✅ 服务健康检查正常

## 重要提示：需要配置企业微信 IP 白名单

当前测试发现错误：
```
errcode: 60020
errmsg: 'not allow to access from your ip'
from ip: 175.41.180.154
```

**这是因为企业微信应用需要配置可信 IP 地址。**

### 解决方法：

1. 登录企业微信管理后台：https://work.weixin.qq.com/
2. 进入 "应用管理" -> 找到你的应用（AgentID: 1000019）
3. 点击 "企业可信IP" 设置
4. 添加服务器的公网 IP：**175.41.180.154**
5. 保存配置

完成后，企业微信通知功能就可以正常工作了！

## 服务使用

### 1. 启动服务（Docker）

```bash
# 构建并启动 notification-gateway
docker compose up -d notification-gateway

# 查看日志
docker compose logs -f notification-gateway
```

### 2. 健康检查

```bash
curl http://localhost:9000/health
```

预期输出：
```json
{
  "status": "healthy",
  "config": {
    "corp_id": true,
    "corp_secret": true,
    "agent_id": 1000019,
    "default_user": "ZhangJinBo"
  }
}
```

### 3. 测试发送消息

```bash
curl -X POST http://localhost:9000/test
```

配置 IP 白名单后，应该会收到测试消息。

### 4. Alertmanager 配置

Alertmanager 已经配置好了，在 `monitoring/alertmanager.yml`：

```yaml
receivers:
  - name: "default"
    webhook_configs:
      - url: "http://notification-gateway:9000/webhook/recommend-default"
        send_resolved: true
  - name: "data-quality"
    webhook_configs:
      - url: "http://notification-gateway:9000/webhook/data-quality"
        send_resolved: true
```

## 环境变量

在 `.env` 文件中配置：

```ini
# 企业微信配置
WEIXIN_CORP_ID=wwdc92c6fc7b7d9115              # 企业 ID
WEIXIN_CORP_SECRET=nwL-_8kIsHp6assREMtbLmTTTq4Dw_WyUPYqNw9jGW8  # 应用 Secret
WEIXIN_AGENT_ID=1000019                         # 应用 AgentID
WEIXIN_DEFAULT_USER=ZhangJinBo                  # 默认接收人
```

## API 端点

- `GET /health` - 健康检查
- `GET /test` 或 `POST /test` - 发送测试消息
- `POST /webhook/<receiver_name>` - 接收 Alertmanager webhook

## 告警消息格式

系统会将 Alertmanager 的告警格式化为易读的企业微信消息：

```
🔥 [FIRING] DataQualityScoreDrop
━━━━━━━━━━━━━━━━
🚨 严重程度: CRITICAL
📊 表: users
⏰ 时间: 2025-10-16 15:30:00

📝 数据质量分数下降
详情: users 表的质量分数从 0.95 下降到 0.75
━━━━━━━━━━━━━━━━
```

## 故障排查

### 问题1：收不到消息

1. 检查服务是否运行：
```bash
docker compose ps notification-gateway
```

2. 查看服务日志：
```bash
docker compose logs notification-gateway
```

3. 确认 IP 白名单已配置（错误码 60020）

### 问题2：配置未生效

重启服务使配置生效：
```bash
docker compose restart notification-gateway
```

### 问题3：端口冲突

如果 9000 端口被占用，修改 docker-compose.yml：
```yaml
ports:
  - "9001:9000"  # 改用 9001 端口
```

## 下一步

1. **配置 IP 白名单**（必须）
2. 测试发送消息：`curl -X POST http://localhost:9000/test`
3. 触发一个测试告警验证完整流程
4. 根据需要调整消息格式和接收人

## 参考资料

- 企业微信API文档：https://developer.work.weixin.qq.com/document/
- Alertmanager配置：https://prometheus.io/docs/alerting/latest/configuration/
