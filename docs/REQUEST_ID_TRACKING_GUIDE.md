# Request ID 双向追踪 - 完整部署指南

## 概述

本指南提供 Request ID 双向追踪的完整部署方案，确保能够准确计算推荐系统的CTR/CVR。

**改进前后对比**：

| 指标 | 改进前 | 改进后 |
|------|--------|--------|
| 点击来源识别 | 无法区分 | 精确识别推荐点击 |
| CTR准确性 | 虚高（包含所有访问） | 真实（只统计推荐点击） |
| 位置分析 | 不支持 | 支持（第1位vs第5位CTR） |
| A/B测试 | 不准确 | 精确对比不同版本 |
| request_id关联 | 无 | 曝光-点击精确匹配 |

---

## 架构图

```
┌─────────────┐
│   用户访问   │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────────────────────────┐
│ 后端服务调用推荐API                                       │
│ GET /recommend/detail/1?user_id=123                      │
│                                                           │
│ 返回:                                                     │
│ {                                                         │
│   "recommendations": [...],                               │
│   "request_id": "req_20251018_120530_abc123"  ← 关键！   │
│ }                                                         │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│ 前端渲染推荐列表                                          │
│                                                           │
│ <a href="/dataDetail/42?from=recommend&rid=req_xxx&pos=0"│
│    onclick="trackClick(...)">                             │
│   推荐数据集A                                             │
│ </a>                                                      │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│ 用户点击推荐                                             │
│                                                           │
│ 1. 发送Matomo事件                                         │
│    _paq.push(['setCustomDimension', 1, 'req_xxx']);      │
│    _paq.push(['trackEvent', 'Recommendation', 'Click'])  │
│                                                           │
│ 2. 跳转到详情页                                           │
│    window.location = "/dataDetail/42?from=recommend&..." │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│ Matomo记录                                                │
│                                                           │
│ log_link_visit_action:                                   │
│   - custom_dimension_1 = "req_20251018_120530_abc123"    │
│   - idaction_url = "/dataDetail/42?from=recommend&..."   │
│   - idaction_event_category = "Recommendation"           │
│   - server_time = 2025-10-18 12:05:30                    │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│ 推荐系统评估                                             │
│                                                           │
│ 1. 读取曝光日志:                                          │
│    request_id="req_xxx" → 曝光了dataset 42               │
│                                                           │
│ 2. 读取Matomo数据:                                        │
│    custom_dimension_1="req_xxx" → 点击了dataset 42       │
│                                                           │
│ 3. 通过request_id精确关联:                               │
│    CTR = 点击数(req_xxx) / 曝光数(req_xxx)               │
└─────────────────────────────────────────────────────────┘
```

---

## 部署步骤

### 步骤1: 配置Matomo自定义维度（必须先做）

**重要**：必须先在Matomo配置自定义维度，否则无法记录request_id！

#### 1.1 登录Matomo后台

访问你的Matomo管理后台（通常是 `http://matomo.yourdomain.com`）

#### 1.2 创建自定义维度

1. 进入 `Administration` (管理) → `Websites` (网站) → `Custom Dimensions` (自定义维度)
2. 点击 `Configure a new Custom Dimension` (配置新的自定义维度)
3. 配置如下：

| 字段 | 值 |
|------|------|
| Dimension Name | `recommendation_request_id` |
| Scope | `Visit` (访问级别) |
| Active | ✅ Yes |
| Description | Request ID for recommendation tracking |

4. 点击 `Create Custom Dimension`
5. **记住生成的维度ID**（通常是1，如果已有其他维度可能是2或3）

#### 1.3 验证配置

```sql
-- 在Matomo数据库执行
SELECT * FROM matomo_custom_dimensions WHERE idsite = 1;
```

应该能看到刚创建的维度。

---

### 步骤2: 更新推荐系统（你的部分）

#### 2.1 更新代码

```bash
cd /home/ubuntu/recommend

# 1. 确认main.py已修改（已完成）
git diff app/main.py  # 检查RecommendationResponse是否包含request_id

# 2. 重启API服务
docker-compose restart recommendation-api

# 或如果是直接运行
pkill -f "uvicorn app.main:app"
uvicorn app.main:app --host 0.0.0.0 --port 8000 &
```

#### 2.2 验证API返回

```bash
# 测试推荐API
curl "http://localhost:8000/recommend/detail/1?user_id=123&limit=5" | jq .

# 应该看到：
# {
#   "dataset_id": 1,
#   "recommendations": [...],
#   "request_id": "req_20251018_120530_abc123",  ← 必须有这个字段
#   "algorithm_version": "20251018T120530Z"
# }
```

如果没有 `request_id` 字段，检查代码是否正确部署。

---

### 步骤3: 前端团队改造

#### 3.1 提供文档

将 `docs/FRONTEND_INTEGRATION.md` 发送给前端团队，要点：

1. **API调用**：保存返回的 `request_id`
2. **链接生成**：包含 `?from=recommend&rid={request_id}&pos={position}`
3. **点击埋点**：发送Matomo事件时设置自定义维度

#### 3.2 前端验证清单

发送给前端团队的检查清单：

- [ ] 调用推荐API能获取到 `request_id`
- [ ] 推荐链接包含 `from=recommend` 参数
- [ ] 推荐链接包含 `rid={request_id}` 参数
- [ ] 点击时调用 `_paq.push(['setCustomDimension', 1, request_id])`
- [ ] 点击时发送 `_paq.push(['trackEvent', 'Recommendation', 'Click', ...])`
- [ ] 浏览器Network标签能看到Matomo请求
- [ ] Matomo请求包含 `dimension1=req_xxx` 参数

#### 3.3 示例前端代码（提供给前端）

```javascript
// React示例
function RecommendationList({ datasetId, userId }) {
  const [recommendations, setRecommendations] = useState([]);
  const [requestId, setRequestId] = useState('');

  useEffect(() => {
    fetch(`/api/recommend/detail/${datasetId}?user_id=${userId}`)
      .then(res => res.json())
      .then(data => {
        setRecommendations(data.recommendations);
        setRequestId(data.request_id);  // 保存
      });
  }, [datasetId, userId]);

  const handleClick = (item, index) => {
    if (window._paq) {
      window._paq.push(['setCustomDimension', 1, requestId]);
      window._paq.push([
        'trackEvent',
        'Recommendation',
        'Click',
        `dataset_${item.dataset_id}`,
        index
      ]);
    }
  };

  return (
    <div>
      {recommendations.map((item, index) => (
        <a
          key={item.dataset_id}
          href={`/dataDetail/${item.dataset_id}?from=recommend&rid=${requestId}&pos=${index}`}
          onClick={() => handleClick(item, index)}
        >
          {item.title}
        </a>
      ))}
    </div>
  );
}
```

---

### 步骤4: 后端团队改造（如果有独立后端）

#### 4.1 提供文档

将 `docs/BACKEND_INTEGRATION.md` 发送给后端团队，关键点：

- 调用推荐API时获取 `request_id`
- **透传** `request_id` 给前端（不要丢弃）
- 如果处理推荐结果，也要保留 `request_id`

#### 4.2 后端验证清单

- [ ] 调用推荐API成功
- [ ] 返回给前端的响应包含 `request_id`
- [ ] 设置了合理的超时时间（建议5秒）
- [ ] 添加了错误处理和降级策略
- [ ] 监控推荐API的调用延迟

---

### 步骤5: 端到端验证

#### 5.1 手动测试流程

```bash
# 1. 调用推荐API
curl "http://localhost:8000/recommend/detail/1?user_id=123" | jq .request_id
# 记录返回的request_id，例如：req_20251018_120530_abc123

# 2. 前端页面验证
# - 打开浏览器访问详情页
# - 查看推荐列表的链接，应该包含 ?from=recommend&rid=req_xxx
# - 右键检查元素，确认data-request-id属性

# 3. 点击推荐
# - 打开浏览器开发者工具 -> Network标签
# - 点击一个推荐
# - 查找Matomo请求（通常是piwik.php或matomo.php）
# - 检查请求参数是否包含 dimension1=req_xxx

# 4. 检查Matomo数据库（等待几分钟）
```

#### 5.2 数据库验证（重要！）

**在Matomo数据库**执行：

```sql
-- 查看最近的推荐点击
SELECT
    idlink_va,
    custom_dimension_1 as request_id,
    idaction_url,
    idaction_event_category,
    server_time
FROM matomo_log_link_visit_action
WHERE custom_dimension_1 LIKE 'req_%'
ORDER BY server_time DESC
LIMIT 10;
```

**期望结果**：
```
+----------+--------------------------------+---------------+------------------------+---------------------+
| idlink_va| request_id                     | idaction_url  | idaction_event_category| server_time         |
+----------+--------------------------------+---------------+------------------------+---------------------+
| 123456   | req_20251018_120530_abc123     | 12345         | 67890                  | 2025-10-18 12:05:30 |
+----------+--------------------------------+---------------+------------------------+---------------------+
```

如果 `custom_dimension_1` 列为空或NULL，说明：
- Matomo自定义维度未配置
- 前端没有正确调用 `setCustomDimension`
- Matomo版本不支持自定义维度

#### 5.3 推荐系统验证

```bash
cd /home/ubuntu/recommend

# 1. 抽取最新Matomo数据
python -m pipeline.extract_load --incremental

# 2. 运行新的评估脚本
python -m pipeline.evaluate_v2

# 3. 查看报告
cat data/evaluation/tracking_report_v2.json | jq .

# 应该看到：
# {
#   "status": "success",
#   "summary": {
#     "total_exposures": 100,
#     "total_clicks": 8,
#     "overall_ctr": 0.08,  ← 真实CTR，通常0.03-0.15
#     "unique_request_ids": 85
#   }
# }
```

---

## 对比旧版评估

运行对比测试，看改进效果：

```bash
# 1. 旧版评估（包含所有点击）
python -m pipeline.evaluate
cat data/evaluation/summary.json | jq '{ctr: .exposures_ctr, cvr: .exposures_cvr}'

# 输出可能是：
# {
#   "ctr": 0.45,  ← 虚高！
#   "cvr": 0.12
# }

# 2. 新版评估（只统计推荐点击）
python -m pipeline.evaluate_v2
cat data/evaluation/tracking_report_v2.json | jq '.summary | {ctr: .overall_ctr, cvr: .overall_cvr}'

# 输出应该是：
# {
#   "ctr": 0.08,  ← 真实值
#   "cvr": 0.02
# }

# CTR降低是正常的！说明之前被高估了。
```

---

## 常见问题排查

### Q1: Matomo数据库中没有custom_dimension_1列？

**原因**：Matomo自定义维度未配置

**解决**：
1. 确认Matomo版本 >= 3.0（支持自定义维度）
2. 重新按照步骤1配置自定义维度
3. 配置后需要等新数据进来，旧数据不会有这个字段

### Q2: custom_dimension_1列存在但都是NULL？

**原因**：前端没有正确调用setCustomDimension

**解决**：
```javascript
// 检查前端代码
console.log(window._paq);  // 确认Matomo已加载

// 手动测试
_paq.push(['setCustomDimension', 1, 'test_request_id']);
_paq.push(['trackEvent', 'Test', 'Test', 'test']);

// 等待1分钟后查询数据库，应该能看到test_request_id
```

### Q3: 评估报告显示 "no_data"？

**原因**：
- 曝光日志为空
- Matomo数据未抽取
- request_id格式不匹配

**解决**：
```bash
# 检查曝光日志
wc -l data/evaluation/exposure_log.jsonl
tail -n 3 data/evaluation/exposure_log.jsonl | jq .request_id

# 检查Matomo数据
ls -lh data/matomo/
python -c "
import pandas as pd
df = pd.read_parquet('data/matomo/matomo_log_link_visit_action.parquet')
print('Matomo数据行数:', len(df))
print('列名:', df.columns.tolist())
"
```

### Q4: CTR还是很高（>0.3）？

**可能原因**：
- 前端还没有完成改造，仍在统计所有点击
- URL参数过滤逻辑有问题

**检查**：
```bash
# 查看有多少点击带有request_id
python -c "
import pandas as pd
df = pd.read_parquet('data/matomo/matomo_log_link_visit_action.parquet')
print('总点击数:', len(df))
if 'custom_dimension_1' in df.columns:
    with_rid = df[df['custom_dimension_1'].notna()]
    print('带request_id的点击:', len(with_rid))
else:
    print('custom_dimension_1列不存在')
"
```

### Q5: 如何区分测试数据和真实数据？

**方法1**：使用特殊的user_id
```javascript
// 测试时使用user_id = 999999
fetch('/api/recommend/detail/1?user_id=999999')
```

**方法2**：过滤request_id
```python
# evaluate_v2.py中添加过滤
exposures = exposures[~exposures['user_id'].isin([999999])]
```

---

## 监控和告警

### 日常监控

```bash
# 每天检查
python -m pipeline.evaluate_v2

# 监控指标：
# 1. unique_request_ids数量（应该持续增长）
# 2. overall_ctr在合理范围（0.03-0.15）
# 3. 点击数 / 曝光数 的比例
```

### 设置告警

在Prometheus中配置告警规则：

```yaml
# monitoring/prometheus/alerts.yml
groups:
  - name: recommendation_tracking
    rules:
      - alert: LowTrackingCoverage
        expr: |
          (
            rate(recommendation_clicks_with_request_id[1h]) /
            rate(recommendation_total_clicks[1h])
          ) < 0.8
        for: 30m
        annotations:
          summary: "推荐点击追踪覆盖率低于80%"
          description: "可能是前端埋点失效"

      - alert: AbnormalCTR
        expr: recommendation_overall_ctr > 0.3
        for: 1h
        annotations:
          summary: "CTR异常偏高"
          description: "可能是追踪逻辑错误"
```

---

## 后续优化方向

### 1. 位置分析

基于position数据分析：
- 第1个推荐的CTR vs 第5个推荐
- 优化推荐排序策略

```bash
# 查看位置分布
cat data/evaluation/tracking_report_v2.json | jq '.by_position'
```

### 2. A/B测试

对比不同算法版本的CTR：

```bash
# 查看版本对比
cat data/evaluation/tracking_report_v2.json | jq '.by_version'
```

### 3. 用户分群

分析不同用户群体的CTR差异：
- 新用户 vs 老用户
- 不同年龄段
- 不同地域

### 4. 转化追踪

完善转化数据的request_id关联（需要关联session）

---

## 检查清单

### 部署前

- [ ] Matomo自定义维度已配置（维度ID确认）
- [ ] 推荐API已返回request_id（curl验证）
- [ ] 前端团队已收到接入文档
- [ ] 后端团队已收到调用文档
- [ ] evaluate_v2.py已部署

### 部署后第1天

- [ ] 手动测试完整流程（点击推荐）
- [ ] Matomo数据库能看到custom_dimension_1数据
- [ ] 运行evaluate_v2.py生成首个报告
- [ ] CTR在合理范围内（0.03-0.15）

### 第1周

- [ ] 每天运行evaluate_v2.py监控
- [ ] 对比新旧评估结果
- [ ] 追踪覆盖率 > 80%（带request_id的点击比例）
- [ ] 没有异常告警

### 长期

- [ ] 每周分析位置效果
- [ ] 每月分析A/B测试结果
- [ ] 基于真实CTR优化推荐算法

---

## 相关文档

- `docs/FRONTEND_INTEGRATION.md` - 前端接入文档
- `docs/BACKEND_INTEGRATION.md` - 后端调用文档
- `docs/TRACKING_VALIDATION_GUIDE.md` - 埋点验证指南
- `pipeline/evaluate_v2.py` - 新版评估脚本

---

## 联系方式

如遇问题，请联系：
- 推荐系统负责人：[你的名字]
- 邮箱：[你的邮箱]
- 文档更新日期：2025-10-18
