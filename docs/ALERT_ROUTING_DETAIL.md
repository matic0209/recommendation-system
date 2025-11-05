# 告警推送详细清单

> 最后更新：2025-10-31

## 📊 当前配置概览

```
┌─────────────────────────────────────────────────────┐
│                告警推送总览                          │
├─────────────────────────────────────────────────────┤
│                                                      │
│  应用层错误 (Sentry)                                 │
│  ├─ 代码异常、API 错误、数据库错误、Redis 错误       │
│  └─ 推送到: Sentry Dashboard (需手动查看)           │
│                                                      │
│  基础设施告警 (Prometheus + Alertmanager)           │
│  ├─ 推荐 API 错误率/延迟、数据质量告警               │
│  └─ 推送到: 企业微信 (ZhangJinBo) ✅                │
│                                                      │
└─────────────────────────────────────────────────────┘
```

---

## 一、Sentry 告警（应用层错误）→ Sentry Dashboard

### 推送目标
- **当前状态**: ⚠️ 仅后台可见
- **推送地址**: https://trace.dianshudata.com
- **查看方式**: 手动登录 Issues 列表
- **建议配置**: 邮件通知（待配置）

### 告警清单

| 类别 | 告警内容 | 严重程度 | 示例 |
|------|---------|---------|------|
| **代码异常** | ValueError, TypeError, KeyError, ZeroDivisionError | ERROR | 参数错误、类型转换失败 |
| **数据库错误** | OperationalError, IntegrityError | CRITICAL | 连接失败、唯一键冲突 |
| **Redis 错误** | ConnectionError, TimeoutError | CRITICAL/WARNING | Redis 不可用、超时 |
| **API 错误** | /recommend/* 500 错误 | CRITICAL | 推荐算法失败 |
| **模型错误** | 模型加载失败、推理失败 | CRITICAL/ERROR | 模型文件损坏 |
| **Airflow 任务** | DAG 执行失败、数据同步失败 | CRITICAL/ERROR | ETL 任务异常 |
| **性能问题** | 慢查询、慢 API | WARNING | SQL 查询 >1s |

### 环境标签
- `production`: 生产环境
- `testing`: 测试环境

---

## 二、Alertmanager 告警 → 企业微信（ZhangJinBo）✅

### 推送目标
- **当前状态**: ✅ 已配置
- **推送渠道**: 企业微信
- **接收人**: ZhangJinBo
- **通知方式**: 实时推送

### 告警清单

#### 2.1 推荐服务性能告警

| 告警名称 | 触发条件 | 严重程度 | 企业微信消息格式 |
|---------|---------|---------|----------------|
| **RecommendationHighErrorRate** | 错误率 >5% 持续 2 分钟 | ⚠️ WARNING | 🔥 [FIRING] RecommendationHighErrorRate<br>⚠️ 严重程度: WARNING<br>📝 推荐 API 错误率超过 5% |
| **RecommendationLatencyHigh** | P95 延迟 >200ms 持续 5 分钟 | 🚨 CRITICAL | 🔥 [FIRING] RecommendationLatencyHigh<br>🚨 严重程度: CRITICAL<br>📝 推荐接口 P95 延迟超出 200ms |
| **DegradeRateSpike** | 降级次数 >1/s 持续 3 分钟 | ⚠️ WARNING | 🔥 [FIRING] DegradeRateSpike<br>⚠️ 严重程度: WARNING<br>📝 降级次数异常 |

#### 2.2 数据质量告警

| 告警名称 | 触发条件 | 严重程度 | 企业微信消息格式 |
|---------|---------|---------|----------------|
| **DataQualityScoreDrop** | 质量得分 <80 持续 10 分钟 | 🚨 CRITICAL | 🔥 [FIRING] DataQualityScoreDrop<br>🚨 严重程度: CRITICAL<br>📊 表: interactions<br>📝 数据质量得分低于阈值 |
| **SchemaContractViolation** | Schema 校验失败持续 5 分钟 | 🚨 CRITICAL | 🔥 [FIRING] SchemaContractViolation<br>🚨 严重程度: CRITICAL<br>📊 表: dataset_features<br>📝 Schema 合约校验失败 |

---

## 三、告警路由配置

根据 `monitoring/alertmanager.yml`:

```
默认接收器 (recommend-default)
├─ 接收所有未匹配的告警
└─ 推送到: 企业微信 (ZhangJinBo)

数据质量接收器 (data-quality)
├─ 匹配: DataQualityScoreDrop, SchemaContractViolation
└─ 推送到: 企业微信 (ZhangJinBo)

分组策略:
├─ 按 alertname, table, endpoint 分组
├─ 等待时间: 30秒
├─ 分组间隔: 5分钟
└─ 重复间隔: 3小时
```

---

## 四、快速参考表

### 告警推送汇总

| 告警源 | 告警类型 | 推送位置 | 接收人 | 状态 |
|--------|---------|---------|--------|------|
| **Sentry** | 代码异常 | Sentry Dashboard | 开发团队（手动查看） | ⚠️ 待配置邮件 |
| **Sentry** | API 错误 | Sentry Dashboard | 开发团队（手动查看） | ⚠️ 待配置邮件 |
| **Sentry** | 数据库错误 | Sentry Dashboard | 开发团队（手动查看） | ⚠️ 待配置邮件 |
| **Sentry** | Redis 错误 | Sentry Dashboard | 开发团队（手动查看） | ⚠️ 待配置邮件 |
| **Sentry** | 性能问题 | Sentry Performance | 开发团队（手动查看） | ⚠️ 待配置邮件 |
| **Alertmanager** | 推荐 API 错误率 | 企业微信 | ZhangJinBo | ✅ 已配置 |
| **Alertmanager** | 推荐 API 延迟 | 企业微信 | ZhangJinBo | ✅ 已配置 |
| **Alertmanager** | 降级告警 | 企业微信 | ZhangJinBo | ✅ 已配置 |
| **Alertmanager** | 数据质量 | 企业微信 | ZhangJinBo | ✅ 已配置 |
| **Alertmanager** | Schema 校验 | 企业微信 | ZhangJinBo | ✅ 已配置 |

---

## 五、配置优化建议

### 立即可做（推荐）⭐

**配置 Sentry 邮件通知：**
1. 登录 https://trace.dianshudata.com
2. Settings → Notifications → Email
3. 添加团队邮箱
4. 勾选通知类型：
   - ✅ New Issues
   - ✅ Regressions

### 短期优化（1周内）

**优化 Alertmanager 分级：**
1. Critical 告警：立即通知
2. Warning 告警：15分钟汇总
3. Info 告警：每日汇总

### 中期规划（1个月）

- 添加多个接收人
- 配置值班轮换
- 建立告警升级机制

---

## 六、相关文档

- 告警规则配置: `monitoring/alerts/recommendation.yml`
- Alertmanager 配置: `monitoring/alertmanager.yml`
- 双告警体系策略: `docs/DUAL_ALERT_STRATEGY.md`
- Sentry 集成指南: `docs/SENTRY_INTEGRATION.md`
