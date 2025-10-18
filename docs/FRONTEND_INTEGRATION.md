# 前端推荐埋点接入文档

## 背景

为了准确追踪推荐系统的效果（点击率CTR、转化率CVR），需要前端配合完成埋点。

**核心思路**：
- 后端推荐API返回 `request_id`（唯一标识这次推荐请求）
- 前端渲染推荐时，在链接中携带 `request_id` 和 `from=recommend` 参数
- 用户点击时，发送Matomo事件记录点击行为
- 后端通过关联 `request_id`，计算真实的CTR/CVR

---

## API变更说明

### 1. 推荐API返回格式

**接口**: `GET /recommend/detail/{dataset_id}?user_id={user_id}&limit=10`

**旧版返回**（不支持追踪）:
```json
{
  "dataset_id": 1,
  "recommendations": [
    {
      "dataset_id": 42,
      "title": "某某数据集",
      "score": 0.95,
      "reason": "collaborative_filtering",
      "price": 199.0,
      "cover_image": "https://..."
    },
    ...
  ]
}
```

**新版返回**（支持追踪）:
```json
{
  "dataset_id": 1,
  "recommendations": [...],
  "request_id": "req_20251018_120530_abc123",  // 新增：唯一请求ID
  "algorithm_version": "20251018T120530Z"      // 新增：算法版本（可选）
}
```

### 2. 相似推荐API

**接口**: `GET /similar/{dataset_id}?limit=10`

**新版返回**:
```json
{
  "dataset_id": 1,
  "similar_items": [...],
  "request_id": "req_20251018_120545_def456",
  "algorithm_version": "20251018T120530Z"
}
```

---

## 前端接入步骤

### 步骤1: 调用API并保存 request_id

```javascript
// 示例：调用推荐API
async function fetchRecommendations(datasetId, userId) {
  const response = await fetch(
    `/api/recommend/detail/${datasetId}?user_id=${userId}&limit=10`
  );
  const data = await response.json();

  return {
    recommendations: data.recommendations,
    requestId: data.request_id,          // 保存request_id
    algorithmVersion: data.algorithm_version
  };
}
```

### 步骤2: 渲染推荐列表时携带追踪参数

#### 方案A: 在URL中携带参数（必须）

```javascript
function renderRecommendationList(recommendations, requestId) {
  return recommendations.map((item, index) => {
    // 构造追踪链接
    const trackingUrl = buildTrackingUrl(item.dataset_id, requestId, index);

    return `
      <div class="recommendation-item">
        <a href="${trackingUrl}"
           data-request-id="${requestId}"
           data-position="${index}"
           data-dataset-id="${item.dataset_id}"
           onclick="trackClick(event, this)">
          <img src="${item.cover_image}" />
          <h4>${item.title}</h4>
          <p>相关度: ${(item.score * 100).toFixed(0)}%</p>
          <span class="price">¥${item.price}</span>
        </a>
      </div>
    `;
  });
}

// 构造带追踪参数的URL
function buildTrackingUrl(datasetId, requestId, position) {
  const params = new URLSearchParams({
    from: 'recommend',           // 标记：来自推荐
    rid: requestId,              // request_id，用于精确关联
    pos: position                // 推荐位置（0-based）
  });

  return `/dataDetail/${datasetId}?${params.toString()}`;
  // 结果: /dataDetail/42?from=recommend&rid=req_20251018_120530_abc123&pos=0
}
```

#### URL参数说明

| 参数 | 必需 | 说明 | 示例值 |
|------|------|------|--------|
| `from` | ✅ 是 | 标记来源为推荐 | `recommend` |
| `rid` | ✅ 是 | 推荐请求ID | `req_20251018_120530_abc123` |
| `pos` | 建议 | 推荐位置（0-based） | `0`, `1`, `2` |

### 步骤3: 发送Matomo点击事件（必须）

用户点击推荐时，**必须**发送Matomo事件，并通过**自定义维度**传递 `request_id`。

```javascript
function trackClick(event, element) {
  const datasetId = element.dataset.datasetId;
  const requestId = element.dataset.requestId;
  const position = element.dataset.position;

  // 确保Matomo已加载
  if (typeof _paq !== 'undefined') {
    // 1. 设置自定义维度1 = request_id（关键！）
    _paq.push(['setCustomDimension', customDimensionId=1, requestId]);

    // 2. 发送点击事件
    _paq.push([
      'trackEvent',
      'Recommendation',           // 事件类别
      'Click',                    // 事件动作
      `dataset_${datasetId}`,     // 事件名称
      parseInt(position)          // 事件值（推荐位置）
    ]);

    console.log('[Tracking] 推荐点击已记录', {
      datasetId,
      requestId,
      position
    });
  } else {
    console.warn('[Tracking] Matomo未加载，跳过埋点');
  }

  // 不阻止默认跳转
  return true;
}
```

#### ⚠️ Matomo自定义维度配置
已配置：纬度值为3

**重要**：需要在Matomo后台配置自定义维度。

1. 登录Matomo管理后台
2. 进入 `设置` > `网站` > `自定义维度`
3. 新增访问级别（Visit scope）自定义维度：
   - **维度ID**: 1
   - **名称**: `recommendation_request_id`
   - **范围**: Visit
   - **状态**: 激活

配置完成后，Matomo会在 `log_link_visit_action` 表中记录 `custom_dimension_1` 字段。

---

## 完整示例代码

### React 示例

```jsx
import React, { useState, useEffect } from 'react';

function RecommendationList({ datasetId, userId }) {
  const [recommendations, setRecommendations] = useState([]);
  const [requestId, setRequestId] = useState('');

  useEffect(() => {
    async function loadRecommendations() {
      const response = await fetch(
        `/api/recommend/detail/${datasetId}?user_id=${userId}&limit=10`
      );
      const data = await response.json();

      setRecommendations(data.recommendations);
      setRequestId(data.request_id);  // 保存request_id
    }

    loadRecommendations();
  }, [datasetId, userId]);

  const handleClick = (item, index) => {
    // 发送Matomo事件
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

  const getTrackingUrl = (datasetId, position) => {
    return `/dataDetail/${datasetId}?from=recommend&rid=${requestId}&pos=${position}`;
  };

  return (
    <div className="recommendations">
      <h3>为您推荐</h3>
      {recommendations.map((item, index) => (
        <div key={item.dataset_id} className="rec-item">
          <a
            href={getTrackingUrl(item.dataset_id, index)}
            onClick={() => handleClick(item, index)}
          >
            <img src={item.cover_image} alt={item.title} />
            <h4>{item.title}</h4>
            <p>相关度: {(item.score * 100).toFixed(0)}%</p>
            <span className="price">¥{item.price}</span>
          </a>
        </div>
      ))}
    </div>
  );
}

export default RecommendationList;
```

### Vue 3 示例

```vue
<template>
  <div class="recommendations">
    <h3>为您推荐</h3>
    <div
      v-for="(item, index) in recommendations"
      :key="item.dataset_id"
      class="rec-item"
    >
      <a
        :href="getTrackingUrl(item.dataset_id, index)"
        @click="handleClick(item, index)"
      >
        <img :src="item.cover_image" :alt="item.title" />
        <h4>{{ item.title }}</h4>
        <p>相关度: {{ (item.score * 100).toFixed(0) }}%</p>
        <span class="price">¥{{ item.price }}</span>
      </a>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue';

const props = defineProps(['datasetId', 'userId']);
const recommendations = ref([]);
const requestId = ref('');

onMounted(async () => {
  const response = await fetch(
    `/api/recommend/detail/${props.datasetId}?user_id=${props.userId}&limit=10`
  );
  const data = await response.json();

  recommendations.value = data.recommendations;
  requestId.value = data.request_id;
});

function getTrackingUrl(datasetId, position) {
  return `/dataDetail/${datasetId}?from=recommend&rid=${requestId.value}&pos=${position}`;
}

function handleClick(item, index) {
  if (window._paq) {
    window._paq.push(['setCustomDimension', 1, requestId.value]);
    window._paq.push([
      'trackEvent',
      'Recommendation',
      'Click',
      `dataset_${item.dataset_id}`,
      index
    ]);
  }
}
</script>
```

### 原生 JavaScript 示例

```javascript
// 加载推荐
async function loadRecommendations(datasetId, userId) {
  const response = await fetch(
    `/api/recommend/detail/${datasetId}?user_id=${userId}&limit=10`
  );
  const data = await response.json();

  renderRecommendations(data.recommendations, data.request_id);
}

// 渲染推荐列表
function renderRecommendations(recommendations, requestId) {
  const container = document.getElementById('recommendations');

  const html = recommendations.map((item, index) => {
    const url = `/dataDetail/${item.dataset_id}?from=recommend&rid=${requestId}&pos=${index}`;

    return `
      <div class="rec-item">
        <a href="${url}"
           data-request-id="${requestId}"
           data-dataset-id="${item.dataset_id}"
           data-position="${index}"
           onclick="trackRecommendationClick(this); return true;">
          <img src="${item.cover_image}" />
          <h4>${item.title}</h4>
          <p>相关度: ${(item.score * 100).toFixed(0)}%</p>
          <span class="price">¥${item.price}</span>
        </a>
      </div>
    `;
  }).join('');

  container.innerHTML = html;
}

// 点击追踪
function trackRecommendationClick(element) {
  const requestId = element.dataset.requestId;
  const datasetId = element.dataset.datasetId;
  const position = element.dataset.position;

  if (window._paq) {
    _paq.push(['setCustomDimension', 1, requestId]);
    _paq.push([
      'trackEvent',
      'Recommendation',
      'Click',
      `dataset_${datasetId}`,
      parseInt(position)
    ]);
  }
}

// 使用
loadRecommendations(1, 123);
```

---

## 前端验证清单

### 开发环境验证

- [ ] 调用推荐API能正常返回 `request_id` 字段
- [ ] 渲染的推荐链接包含 `?from=recommend&rid=xxx` 参数
- [ ] 点击推荐时，浏览器Network标签能看到Matomo请求
- [ ] Matomo请求中包含自定义维度（`dimension1=req_xxx`）

### 浏览器控制台验证

```javascript
// 1. 打开浏览器开发者工具 -> Console
// 2. 点击推荐后，检查是否有追踪日志
// 应该看到: [Tracking] 推荐点击已记录 { datasetId: 42, requestId: "req_...", position: 0 }

// 3. 检查Matomo对象
console.log(window._paq);  // 应该是一个数组

// 4. 手动测试发送事件
_paq.push(['setCustomDimension', 1, 'test_request_id']);
_paq.push(['trackEvent', 'Test', 'Click', 'test', 1]);
```

### 线上验证

1. **检查URL参数**
   - 点击推荐后，地址栏应显示 `/dataDetail/42?from=recommend&rid=req_20251018_120530_abc123&pos=0`

2. **检查Matomo后台**
   - 登录Matomo -> 访客 -> 自定义维度
   - 应该能看到 `recommendation_request_id` 的数据

3. **通知后端验证**
   - 后端同事会从Matomo数据库中验证是否记录了 `custom_dimension_1`

---

## 常见问题

### Q1: Matomo未加载怎么办？

**A**: 检查 `window._paq` 是否存在：

```javascript
function trackClick(element) {
  if (typeof window._paq === 'undefined') {
    console.warn('[Tracking] Matomo未加载，跳过埋点');
    // 可选：上报到监控系统
    reportError('matomo_not_loaded');
    return;
  }

  // 正常埋点逻辑...
}
```

### Q2: 单页应用（SPA）如何处理？

**A**: 确保每次路由切换时重新获取推荐和 `request_id`：

```javascript
// React Router 示例
useEffect(() => {
  loadRecommendations(datasetId, userId);
}, [datasetId, userId]);  // 依赖变化时重新加载
```

### Q3: 推荐为空时如何处理？

**A**: 如果API返回空列表，不需要埋点：

```javascript
if (data.recommendations.length === 0) {
  console.log('[Tracking] 无推荐结果，跳过埋点');
  return;
}
```

### Q4: 用户快速连续点击怎么办？

**A**: 可以加防抖或记录已点击：

```javascript
const clickedItems = new Set();

function handleClick(item, index) {
  const key = `${requestId}_${item.dataset_id}`;

  if (clickedItems.has(key)) {
    console.log('[Tracking] 重复点击，跳过埋点');
    return;
  }

  clickedItems.add(key);

  // 正常埋点逻辑...
}
```

---

## 联系方式

如有疑问，请联系后端团队：
- 负责人：[你的名字]
- 邮箱：[你的邮箱]
- 文档更新日期：2025-10-18

---

## 附录：完整追踪链路

```
1. 用户访问详情页
   ↓
2. 前端调用 GET /recommend/detail/1?user_id=123
   ↓
3. 后端返回:
   {
     "recommendations": [...],
     "request_id": "req_20251018_120530_abc123"
   }
   ↓
4. 前端渲染推荐列表，链接为:
   /dataDetail/42?from=recommend&rid=req_20251018_120530_abc123&pos=0
   ↓
5. 用户点击推荐
   ↓
6. 前端发送Matomo事件:
   _paq.push(['setCustomDimension', 1, 'req_20251018_120530_abc123']);
   _paq.push(['trackEvent', 'Recommendation', 'Click', 'dataset_42', 0]);
   ↓
7. 页面跳转到详情页（带追踪参数）
   ↓
8. Matomo记录到数据库:
   - log_link_visit_action.custom_dimension_1 = 'req_20251018_120530_abc123'
   - log_link_visit_action.idaction_url = '/dataDetail/42?from=recommend&...'
   - log_link_visit_action.idaction_event_category = 'Recommendation'
   ↓
9. 后端定期抽取Matomo数据
   ↓
10. 后端评估脚本关联:
    曝光(request_id=req_xxx) + 点击(custom_dimension_1=req_xxx)
    → 计算真实CTR
```

---

## 购买追踪（Conversion Tracking）

除了点击追踪，还需要追踪用户的购买行为来计算CVR（转化率）。

### 增加LocalStorage保存

在点击推荐时，除了发送Matomo事件，还要保存到LocalStorage：

```javascript
function handleRecommendClick(item, requestId, position) {
  // 1. 发送点击事件（已有）
  if (window._paq) {
    window._paq.push(['setCustomDimension', 1, requestId]);
    window._paq.push(['trackEvent', 'Recommendation', 'Click', `dataset_${item.dataset_id}`, position]);
  }

  // 2. 保存到LocalStorage（新增）
  try {
    const recommendData = {
      request_id: requestId,
      dataset_id: item.dataset_id,
      position: position,
      timestamp: new Date().toISOString()
    };
    localStorage.setItem('last_recommend_click', JSON.stringify(recommendData));

    // 设置7天过期
    const expiry = new Date().getTime() + (7 * 24 * 60 * 60 * 1000);
    localStorage.setItem('last_recommend_click_expiry', expiry.toString());
  } catch (e) {
    console.warn('[Tracking] Failed to save to localStorage', e);
  }
}
```

### 购买成功时发送归因事件

```javascript
// 在购买成功页面或购买回调中调用
function trackPurchaseSuccess(orderId, datasetId, revenue) {
  // 1. 尝试获取推荐归因数据
  let recommendData = null;
  try {
    const stored = localStorage.getItem('last_recommend_click');
    const expiry = localStorage.getItem('last_recommend_click_expiry');

    if (stored && expiry && new Date().getTime() < parseInt(expiry)) {
      recommendData = JSON.parse(stored);

      // 检查是否是同一个数据集
      if (recommendData.dataset_id !== datasetId) {
        recommendData = null;
      }
    }
  } catch (e) {
    console.warn('[Tracking] Failed to load recommend data', e);
  }

  // 2. 发送Matomo事件
  if (window._paq) {
    // 如果是推荐归因，设置自定义维度
    if (recommendData) {
      _paq.push(['setCustomDimension', 1, recommendData.request_id]);
      _paq.push(['setCustomDimension', 2, recommendData.position.toString()]);

      console.log('[Tracking] Purchase attributed to recommendation', recommendData);
    }

    // 发送购买事件
    _paq.push(['trackGoal', 1, revenue]); // Goal ID = 1 (需要在Matomo后台配置)
    _paq.push(['trackEcommerceOrder', orderId, revenue]);

    // 可选：发送自定义购买事件
    if (recommendData) {
      _paq.push([
        'trackEvent',
        'Recommendation',
        'Purchase',
        `dataset_${datasetId}`,
        Math.round(revenue)
      ]);
    }
  }

  // 3. 清除LocalStorage
  if (recommendData) {
    localStorage.removeItem('last_recommend_click');
    localStorage.removeItem('last_recommend_click_expiry');
  }
}
```

### Matomo配置要求

需要在Matomo后台额外配置：

1. **自定义维度2（position）**
   - Name: `recommendation_position`
   - Scope: `Action`
   - Active: Yes

2. **购买目标（Goal）**
   - Name: `Dataset Purchase`
   - Triggered: `Manually`
   - Allow revenue: Yes

### 验证购买追踪

```javascript
// 浏览器控制台测试
// 1. 检查LocalStorage
console.log(localStorage.getItem('last_recommend_click'));

// 2. 模拟购买
trackPurchaseSuccess('TEST_ORDER_123', 42, 199.00);

// 3. 查看Network标签，Matomo请求应包含：
// - dimension1=req_xxx
// - dimension2=0
// - idgoal=1
```

### 完整示例（React）

```jsx
// PurchaseSuccessPage.jsx
import { useEffect } from 'react';
import { useParams } from 'react-router-dom';

function PurchaseSuccessPage() {
  const { orderId } = useParams();
  const [orderDetails, setOrderDetails] = useState(null);

  useEffect(() => {
    // 获取订单详情
    fetch(`/api/orders/${orderId}`)
      .then(res => res.json())
      .then(order => {
        setOrderDetails(order);

        // 追踪购买
        trackPurchaseSuccess(order.id, order.dataset_id, order.amount);
      });
  }, [orderId]);

  return (
    <div>
      <h1>购买成功！</h1>
      {orderDetails && (
        <div>
          <p>订单号：{orderDetails.id}</p>
          <p>数据集：{orderDetails.dataset_name}</p>
          <p>金额：¥{orderDetails.amount}</p>
        </div>
      )}
    </div>
  );
}

function trackPurchaseSuccess(orderId, datasetId, revenue) {
  // 获取推荐归因数据
  let recommendData = null;
  try {
    const stored = localStorage.getItem('last_recommend_click');
    const expiry = localStorage.getItem('last_recommend_click_expiry');

    if (stored && expiry && new Date().getTime() < parseInt(expiry)) {
      recommendData = JSON.parse(stored);
      if (recommendData.dataset_id !== datasetId) {
        recommendData = null;
      }
    }
  } catch (e) {
    console.warn('Failed to load recommend data', e);
  }

  // 发送Matomo事件
  if (window._paq) {
    if (recommendData) {
      _paq.push(['setCustomDimension', 1, recommendData.request_id]);
      _paq.push(['setCustomDimension', 2, recommendData.position.toString()]);
    }

    _paq.push(['trackGoal', 1, revenue]);
    _paq.push(['trackEcommerceOrder', orderId, revenue]);

    if (recommendData) {
      _paq.push([
        'trackEvent',
        'Recommendation',
        'Purchase',
        `dataset_${datasetId}`,
        Math.round(revenue)
      ]);
    }
  }

  // 清除数据
  if (recommendData) {
    localStorage.removeItem('last_recommend_click');
    localStorage.removeItem('last_recommend_click_expiry');
  }
}
```

---

## 指标说明

### CTR（点击率）

```
CTR = 推荐点击数 / 推荐曝光数
```

正常范围：3% - 15%

### CVR（转化率）

```
CVR = 推荐购买数 / 推荐曝光数
```

正常范围：0.5% - 5%

### 示例

- 曝光：1000次推荐
- 点击：85次 → CTR = 8.5%
- 购买：12次 → CVR = 1.2%
- 购买转化率（点击→购买）：12/85 = 14.1%

