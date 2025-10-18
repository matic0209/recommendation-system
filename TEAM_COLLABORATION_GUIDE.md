# 推荐系统Request ID追踪 - 团队协作指南

## 项目背景

当前推荐系统的CTR/CVR计算不准确，原因是：
- **曝光**由推荐API记录（准确）
- **点击**从Matomo提取（不准确，包含所有访问来源）

导致CTR被严重**高估**，无法真实反映推荐效果。

**解决方案**：实施Request ID双向追踪，精确识别推荐点击。

---

## 改动概览

| 团队 | 改动内容 | 工作量 | 优先级 |
|------|----------|--------|--------|
| **推荐系统**（你） | API返回request_id + 评估脚本改造 | ✅ 已完成 | P0 |
| **前端团队** | 渲染推荐时保留request_id + Matomo埋点 | 2-3天 | P0 |
| **后端团队**（如有） | 透传request_id给前端 | 0.5-1天 | P0 |
| **Matomo管理员** | 配置自定义维度 | 0.5天 | P0 |

---

## 推荐系统已完成的改动

### 1. API接口变更

**影响范围**：所有调用推荐API的服务

**变更详情**：

#### 改动前
```json
GET /recommend/detail/1?user_id=123

Response:
{
  "dataset_id": 1,
  "recommendations": [...]
}
```

#### 改动后
```json
GET /recommend/detail/1?user_id=123

Response:
{
  "dataset_id": 1,
  "recommendations": [...],
  "request_id": "req_20251018_120530_abc123",  ← 新增
  "algorithm_version": "20251018T120530Z"      ← 新增
}
```

**兼容性**：
- ✅ 向后兼容，不影响现有调用
- ⚠️ 前端/后端需要保存并使用request_id

**受影响的接口**：
- `GET /recommend/detail/{dataset_id}` ← 主要接口
- `GET /similar/{dataset_id}` ← 相似推荐

### 2. 新增评估脚本

**文件**：`pipeline/evaluate_v2.py`

**功能**：
- 基于request_id精确匹配曝光和点击
- 计算真实的CTR/CVR
- 支持位置分析、版本对比

**使用**：
```bash
python -m pipeline.evaluate_v2
cat data/evaluation/tracking_report_v2.json | jq .
```

---

## 前端团队需要的改动

### 📄 文档

请前端团队阅读：`docs/FRONTEND_INTEGRATION.md`

### 核心改动点

#### 1. 保存API返回的request_id

```javascript
// 调用推荐API
const response = await fetch('/api/recommend/detail/1?user_id=123');
const data = await response.json();

// 保存request_id
const requestId = data.request_id;  // 关键！
const recommendations = data.recommendations;
```

#### 2. 渲染推荐时在链接中携带参数

```javascript
// 原来：
<a href="/dataDetail/42">推荐数据集</a>

// 改为：
<a href="/dataDetail/42?from=recommend&rid={request_id}&pos=0"
   data-request-id="{request_id}"
   data-position="0"
   onclick="trackClick(this)">
  推荐数据集
</a>
```

#### 3. 点击时发送Matomo事件

```javascript
function trackClick(element) {
  const requestId = element.dataset.requestId;
  const position = element.dataset.position;
  const datasetId = element.dataset.datasetId;

  // 关键！设置自定义维度
  _paq.push(['setCustomDimension', 1, requestId]);

  // 发送点击事件
  _paq.push([
    'trackEvent',
    'Recommendation',
    'Click',
    `dataset_${datasetId}`,
    parseInt(position)
  ]);
}
```

### 验证清单（请前端自测）

- [ ] 调用推荐API能获取到request_id
- [ ] 推荐链接包含 `?from=recommend&rid=xxx&pos=0`
- [ ] 点击时浏览器Network能看到Matomo请求
- [ ] Matomo请求包含 `dimension1=req_xxx` 参数

### 完整示例

我们提供了React、Vue3、原生JS的完整示例，详见 `docs/FRONTEND_INTEGRATION.md`。

---

## 后端团队需要的改动（如适用）

### 📄 文档

请后端团队阅读：`docs/BACKEND_INTEGRATION.md`

### 核心要求

如果你们的后端服务调用推荐API，**必须透传request_id给前端**：

#### 简单方式（推荐）

直接转发推荐API的完整响应：

```java
// Java示例
@GetMapping("/recommendations/{datasetId}")
public ResponseEntity<Map<String, Object>> getRecommendations(...) {
    // 调用推荐API
    Map<String, Object> response = restTemplate.getForObject(
        "http://recommendation-api:8000/recommend/detail/1?user_id=123",
        Map.class
    );

    // 直接返回（包含request_id）
    return ResponseEntity.ok(response);
}
```

#### 复杂方式

如果需要处理推荐数据，**务必保留request_id**：

```java
public CustomResponse getRecommendations(...) {
    Map<String, Object> apiResponse = callRecommendationAPI(...);

    // 提取数据
    List<Item> recommendations = (List) apiResponse.get("recommendations");
    String requestId = (String) apiResponse.get("request_id");  // 关键！

    // 处理数据...
    List<Item> processed = process(recommendations);

    // 返回时包含request_id
    return new CustomResponse(processed, requestId);  // 传给前端！
}
```

### 验证清单

- [ ] 调用推荐API成功
- [ ] 返回给前端的响应包含request_id字段
- [ ] 设置了超时（建议5秒）
- [ ] 添加了错误处理

---

## Matomo管理员需要的改动

### 配置自定义维度（必须先做！）

1. 登录Matomo管理后台
2. 进入 `Administration` → `Websites` → `Custom Dimensions`
3. 点击 `Configure a new Custom Dimension`
4. 配置：
   - Name: `recommendation_request_id`
   - Scope: `Visit`
   - Active: ✅ Yes
5. **记录生成的维度ID**（通常是1）
6. 通知前端团队：自定义维度ID = 1

### 验证

```sql
-- 在Matomo数据库执行
SELECT * FROM matomo_custom_dimensions WHERE idsite = 1;

-- 等前端埋点上线后，查看是否有数据
SELECT custom_dimension_1, COUNT(*)
FROM matomo_log_link_visit_action
WHERE custom_dimension_1 IS NOT NULL
GROUP BY custom_dimension_1
LIMIT 10;
```

---

## 上线顺序

### 阶段1: 准备阶段（第1天）

1. ✅ **推荐系统**：部署API改动（已完成）
2. **Matomo管理员**：配置自定义维度
3. **后端团队**：更新代码透传request_id
4. **前端团队**：开始开发

### 阶段2: 测试阶段（第2-3天）

1. **前端团队**：完成开发，自测验证清单
2. **推荐系统**：提供测试环境验证
3. **联调测试**：
   - 调用推荐API → 获取request_id
   - 前端渲染推荐 → 链接包含参数
   - 点击推荐 → Matomo收到事件
   - 推荐系统 → 数据库验证custom_dimension_1

### 阶段3: 灰度上线（第4天）

1. **前端团队**：先在10%流量灰度
2. **推荐系统**：运行evaluate_v2.py监控
3. **验证指标**：
   - unique_request_ids > 0
   - overall_ctr 在 0.03-0.15 范围
   - 追踪覆盖率 > 80%

### 阶段4: 全量上线（第5天）

1. **前端团队**：全量上线
2. **推荐系统**：每天运行evaluate_v2.py
3. **对比分析**：新旧CTR差异

---

## 验证方法

### 端到端测试

```bash
# 1. 推荐系统验证API
curl "http://localhost:8000/recommend/detail/1?user_id=123" | jq .request_id
# 应该返回：req_20251018_120530_abc123

# 2. 前端验证（请前端团队执行）
# - 打开推荐页面
# - 右键检查推荐链接，应包含 ?from=recommend&rid=req_xxx
# - 打开Network标签
# - 点击推荐
# - 查找Matomo请求，应包含 dimension1=req_xxx

# 3. Matomo数据库验证（请Matomo管理员执行）
SELECT custom_dimension_1, COUNT(*)
FROM matomo_log_link_visit_action
WHERE custom_dimension_1 LIKE 'req_%'
GROUP BY custom_dimension_1
LIMIT 10;

# 4. 推荐系统最终验证
cd /home/ubuntu/recommend
python -m pipeline.extract_load --incremental
python -m pipeline.evaluate_v2
cat data/evaluation/tracking_report_v2.json | jq '.summary'
```

---

## 常见问题

### Q1: 前端改造工作量多大？

**A**: 2-3天
- 保存request_id：0.5天
- 修改链接生成：1天
- 添加Matomo埋点：0.5天
- 测试验证：1天

### Q2: 是否必须同时上线？

**A**: 否，可以分阶段：
1. 先上推荐API改动（已完成，向后兼容）
2. 前端逐步改造（可按页面灰度）
3. 逐步提升追踪覆盖率

### Q3: 如何验证改造成功？

**A**: 三个指标：
1. Matomo数据库有custom_dimension_1数据
2. 推荐系统evaluate_v2.py能生成报告
3. CTR降到合理范围（0.03-0.15）

### Q4: 旧数据怎么办？

**A**: 旧数据无法追溯，只能：
- 继续用旧版evaluate.py分析旧数据
- 用新版evaluate_v2.py分析新数据
- 建立新的baseline

### Q5: 影响现有功能吗？

**A**: 不影响
- API新增字段，不影响现有调用
- 前端新增埋点，不影响现有流程
- 评估脚本独立，不影响现有报告

---

## 时间表

| 日期 | 团队 | 任务 | 负责人 |
|------|------|------|--------|
| Day 0 | 推荐系统 | ✅ API改动完成 | [你的名字] |
| Day 1 | Matomo | 配置自定义维度 | [Matomo管理员] |
| Day 1-2 | 后端 | 透传request_id | [后端负责人] |
| Day 1-3 | 前端 | 埋点开发 | [前端负责人] |
| Day 4 | 全员 | 联调测试 | - |
| Day 5 | 前端 | 灰度上线（10%） | [前端负责人] |
| Day 6 | 推荐系统 | 验证数据 | [你的名字] |
| Day 7 | 前端 | 全量上线 | [前端负责人] |

---

## 相关文档

### 给前端团队
- **主文档**：`docs/FRONTEND_INTEGRATION.md`
- 完整示例代码（React/Vue/JS）
- 验证清单

### 给后端团队
- **主文档**：`docs/BACKEND_INTEGRATION.md`
- 多语言示例（Java/Go/PHP/Python）
- 错误处理和降级策略

### 给Matomo管理员
- **配置指南**：`docs/REQUEST_ID_TRACKING_GUIDE.md` 步骤1

### 总体设计
- **完整方案**：`docs/REQUEST_ID_TRACKING_GUIDE.md`

---

## 联系方式

### 推荐系统团队
- 负责人：[你的名字]
- 邮箱：[你的邮箱]
- 技术支持：[工作时间]

### 问题反馈
- 前端问题：参考 `docs/FRONTEND_INTEGRATION.md` FAQ
- 后端问题：参考 `docs/BACKEND_INTEGRATION.md` FAQ
- 数据问题：联系推荐系统团队

---

## 附录：改动清单

### 推荐系统（已完成）

**新增文件**：
- `pipeline/evaluate_v2.py` - 新版评估脚本
- `docs/FRONTEND_INTEGRATION.md` - 前端接入文档
- `docs/BACKEND_INTEGRATION.md` - 后端调用文档
- `docs/REQUEST_ID_TRACKING_GUIDE.md` - 完整部署指南
- `TEAM_COLLABORATION_GUIDE.md` - 本文档

**修改文件**：
- `app/main.py` - RecommendationResponse/SimilarResponse增加request_id字段

### 前端团队（待开发）

**需修改组件**：
- 推荐列表组件（保存request_id）
- 链接生成逻辑（添加追踪参数）
- 点击事件处理（发送Matomo事件）

### 后端团队（待开发，如适用）

**需修改接口**：
- 调用推荐API的服务（透传request_id）

### Matomo（待配置）

**需配置**：
- 自定义维度1：recommendation_request_id

---

**最后更新**: 2025-10-18
**文档版本**: v1.0
