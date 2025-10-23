# 前端推荐点击埋点指南

## 问题背景

当前的CTR计算存在缺陷：
- **曝光**由后端API记录（准确）
- **点击**从Matomo读取（不准确，包含所有访问来源）

导致CTR被**高估**，因为无法区分用户是从推荐点击，还是通过搜索、收藏等其他方式访问。

---

## 解决方案：前端埋点标记推荐点击

### 方案1: URL参数法（推荐，改动最小）

#### 第一步：后端返回 request_id

修改推荐API返回，增加 `request_id` 字段：

```python
# app/main.py - 推荐接口返回
@app.get("/recommend/detail/{dataset_id}")
async def get_recommendations(...):
    # ... 现有逻辑 ...

    return {
        "dataset_id": dataset_id,
        "recommendations": items[:limit],
        "request_id": request_id,  # 新增：返回给前端
        "algorithm_version": run_id
    }
```

#### 第二步：前端在链接中加参数

```javascript
// 示例：React/Vue 组件
function RecommendationList({ datasetId, userId }) {
  const [recommendations, setRecommendations] = useState([]);
  const [requestId, setRequestId] = useState('');

  useEffect(() => {
    // 调用推荐API
    fetch(`/api/recommend/detail/${datasetId}?user_id=${userId}`)
      .then(res => res.json())
      .then(data => {
        setRecommendations(data.recommendations);
        setRequestId(data.request_id);  // 保存request_id
      });
  }, [datasetId, userId]);

  return (
    <div className="recommendations">
      <h3>为您推荐</h3>
      {recommendations.map((item, index) => (
        <a
          key={item.dataset_id}
          href={`/dataDetail/${item.dataset_id}?from=recommend&rid=${requestId}&pos=${index}`}
          className="recommendation-item"
        >
          <img src={item.cover_image} />
          <h4>{item.title}</h4>
          <p>评分: {item.score}</p>
        </a>
      ))}
    </div>
  );
}
```

**关键点：**
- `from=recommend` - 标记这是推荐点击
- `rid=${requestId}` - 关联到具体的推荐请求（可选，用于精确追踪）
- `pos=${index}` - 推荐位置（可选，用于分析位置bias）

#### 第三步：后端过滤推荐点击

修改 `pipeline/evaluate.py`，只统计带 `from=recommend` 的点击：

```python
def _parse_recommend_click(url: Optional[str]) -> Optional[Tuple[int, str]]:
    """
    解析推荐点击，返回 (dataset_id, request_id)
    只有URL中包含 from=recommend 才计入CTR
    """
    if not url or 'from=recommend' not in url:
        return None

    # 提取dataset_id
    for pattern in (DATA_DETAIL_PATTERN, API_DETAIL_PATTERN):
        match = pattern.search(url)
        if match:
            dataset_id = int(match.group(1))

            # 尝试提取request_id（可选）
            rid_match = re.search(r'rid=([^&]+)', url)
            request_id = rid_match.group(1) if rid_match else None

            return (dataset_id, request_id)

    return None
```

---

### 方案2: Matomo事件法（更精确，需要配置Matomo）

#### 前端发送点击事件

```javascript
function trackRecommendationClick(datasetId, requestId, position) {
  // 确保Matomo已加载
  if (typeof _paq !== 'undefined') {
    _paq.push([
      'trackEvent',
      'Recommendation',           // 事件类别
      'Click',                    // 事件动作
      `dataset_${datasetId}`,     // 事件名称
      position                    // 事件值（可选）
    ]);

    // 可选：设置自定义维度存储request_id
    _paq.push(['setCustomDimension', customDimensionId=1, requestId]);
  }
}

// 在点击时调用
<a
  href={`/dataDetail/${item.dataset_id}`}
  onClick={() => trackRecommendationClick(item.dataset_id, requestId, index)}
>
  {item.title}
</a>
```

#### 后端读取事件数据

从 `matomo_log_link_visit_action` 表中过滤：

```python
# 只统计event_category = 'Recommendation' 的记录
event_clicks = actions[
    (actions['idaction_event_category'] == 'Recommendation') &
    (actions['idaction_event_action'] == 'Click')
]
```

---

### 方案3: 双向追踪（生产环境推荐）

结合URL参数和Matomo事件，实现完整追踪链路：

```javascript
// 1. 渲染推荐列表时
function renderRecommendation(item, requestId, position) {
  return `
    <a
      href="/dataDetail/${item.dataset_id}?from=recommend&rid=${requestId}"
      data-request-id="${requestId}"
      data-position="${position}"
      data-dataset-id="${item.dataset_id}"
      onclick="trackClick(this); return true;"
    >
      ${item.title}
    </a>
  `;
}

// 2. 点击时记录事件
function trackClick(element) {
  const datasetId = element.dataset.datasetId;
  const requestId = element.dataset.requestId;
  const position = element.dataset.position;

  // 发送Matomo事件
  _paq.push(['trackEvent', 'Recommendation', 'Click', `dataset_${datasetId}`, position]);

  // 设置自定义维度（需要在Matomo后台配置）
  _paq.push(['setCustomDimension', 1, requestId]);

  // 可选：发送到自己的日志服务器
  fetch('/api/track/click', {
    method: 'POST',
    body: JSON.stringify({ datasetId, requestId, position, timestamp: new Date() })
  });
}
```

---

## 最小改动方案（快速上线）

如果前端改动困难，可以先用**Referer分析法**：

### 前端不改（或小改）

推荐列表页URL固定为：`/recommendations` 或 `/detail/{id}/recommendations`

### 后端分析Referer

```python
# evaluate.py
def _is_from_recommendation(row: pd.Series) -> bool:
    """检查是否来自推荐页面"""
    referer = row.get('referer_url', '')
    return '/recommendations' in referer or '/recommend' in referer

# 过滤点击
recommend_clicks = actions[actions.apply(_is_from_recommendation, axis=1)]
```

**缺点**：不够精确，但比不做任何区分要好。

---

## 验证埋点是否生效

### 1. 前端检查

打开浏览器开发者工具 -> Network标签：

```
点击推荐后，检查：
✓ URL包含 ?from=recommend 参数
✓ Matomo请求中包含事件数据（如果用事件法）
```

### 2. 后端验证

```bash
# 检查Matomo数据中是否有带参数的URL
python -c "
import pandas as pd
actions = pd.read_parquet('data/matomo/matomo_log_action.parquet')
recommend_urls = actions[actions['name'].str.contains('from=recommend', na=False)]
print(f'找到 {len(recommend_urls)} 条推荐点击URL')
print(recommend_urls['name'].head())
"
```

### 3. CTR对比

```bash
# 运行评估
python -m pipeline.evaluate

# 查看结果
cat data/evaluation/summary.json | jq '{ctr: .exposures_ctr, cvr: .exposures_cvr}'

# CTR应该降低到合理范围（通常0.03-0.10）
```

---

## 示例代码模板

### 完整前端示例（Vue3）

```vue
<template>
  <div class="recommendations">
    <h3>为您推荐</h3>
    <div
      v-for="(item, index) in items"
      :key="item.dataset_id"
      class="rec-item"
    >
      <a
        :href="getRecommendUrl(item.dataset_id, index)"
        @click="trackClick(item, index)"
      >
        <img :src="item.cover_image" />
        <h4>{{ item.title }}</h4>
        <span>相关度: {{ (item.score * 100).toFixed(0) }}%</span>
      </a>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue';

const props = defineProps(['datasetId', 'userId']);
const items = ref([]);
const requestId = ref('');

onMounted(async () => {
  const response = await fetch(
    `/api/recommend/detail/${props.datasetId}?user_id=${props.userId}`
  );
  const data = await response.json();
  items.value = data.recommendations;
  requestId.value = data.request_id;
});

function getRecommendUrl(datasetId, position) {
  return `/dataDetail/${datasetId}?from=recommend&rid=${requestId.value}&pos=${position}`;
}

function trackClick(item, position) {
  // 发送Matomo事件
  if (window._paq) {
    window._paq.push([
      'trackEvent',
      'Recommendation',
      'Click',
      `dataset_${item.dataset_id}`,
      position
    ]);
  }

  // 不阻止默认跳转
  return true;
}
</script>
```

---

## 改进后的评估效果

### 改进前（不准确）

```json
{
  "exposures_total": 10000,
  "exposures_ctr": 0.45,  // 虚高！包含所有访问
  "exposures_cvr": 0.12
}
```

### 改进后（准确）

```json
{
  "exposures_total": 10000,
  "recommend_clicks": 850,     // 只统计推荐点击
  "exposures_ctr": 0.085,      // 真实CTR
  "exposures_cvr": 0.021,
  "click_breakdown": {
    "from_recommend": 850,
    "from_search": 2300,
    "from_direct": 1350
  }
}
```

---

## 后续优化

一旦埋点完善后，可以做更深入的分析：

1. **位置bias分析**：第1个推荐的CTR vs 第5个推荐
2. **算法版本对比**：A/B测试不同模型的CTR
3. **用户分群**：新用户 vs 老用户的CTR差异
4. **时间趋势**：CTR随时间的变化

---

## 检查清单

上线前确认：
- [ ] 推荐API返回 `request_id`
- [ ] 前端链接包含 `from=recommend` 参数
- [ ] 点击时发送Matomo事件（可选）
- [ ] 后端过滤逻辑已更新
- [ ] 验证Matomo数据中能看到带参数的URL
- [ ] 重新计算CTR，数值在合理范围内

上线后监控：
- [ ] 每天检查推荐点击数量
- [ ] 对比改进前后的CTR变化
- [ ] 监控点击来源分布
