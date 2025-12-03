# 前端推荐埋点接入文档

## 背景

为了准确衡量推荐系统的点击率（CTR）与转化率（CVR），并让后续的评估脚本与训练管线能够利用这些数据，需要前端配合完成以下埋点动作：

- 推荐 API 返回 `request_id`；前端渲染推荐卡片时保存该 ID。
- 推荐卡片链接统一带上 `from=recommend`、`rid=request_id`、`pos=位置` 参数。
- 点击卡片时向 Matomo 发送事件：自定义维度记录 `dataset_id`、`request_id`，`trackEvent` 的 `value` 记录推荐位次（1-based）。
- 购买成功时回传同一批信息，确保 CTR/CVR 以及推荐训练特征都能闭环。

> **下游消费**：`pipeline.aggregate_matomo_events` 会读取 Matomo 中的 `custom_dimension_4`（request_id）、`custom_dimension_5`（`click_{position}_{datasetId}`）以及事件值 `position`，直接用于 CTR 分析、A/B 统计与模型特征，请务必保证字段完整且准确。

---

## API 变更说明

### 1. 推荐 API 返回格式

**接口**：`GET /recommend/detail/{dataset_id}?user_id={user_id}&limit=10`

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
    }
  ],
  "request_id": "req_20251018_120530_abc123",
  "algorithm_version": "20251018T120530Z"
}
```

- `request_id`：唯一标识本次推荐请求，贯穿点击、购买链路。
- `recommendations[].dataset_id`：用于拼接到 `custom_dimension_5`（`click_{pos}_{datasetId}`）。
- `algorithm_version`（可选）：方便灰度调试。

---

## 前端接入步骤

### 步骤 1：调用 API 并保存 `request_id`

```ts
type RecommendationResponse = {
  recommendations: RecommendationItem[];
  request_id: string;
};

export const fetchRecommendations = async (
  datasetId: number,
  userId: number
): Promise<RecommendationResponse> => {
  const res = await fetch(`/api/recommend/detail/${datasetId}?user_id=${userId}&limit=10`);
  const data = await res.json();
  return {
    recommendations: data.recommendations,
    request_id: data.request_id,
  };
};
```

### 步骤 2：渲染推荐列表并携带追踪参数

```ts
const buildTrackingUrl = (datasetId: number, requestId: string, position: number) => {
  const params = new URLSearchParams({
    from: 'recommend',
    rid: requestId,
    pos: String(position), // 1-based
  });
  return `/dataDetail/${datasetId}?${params.toString()}`;
};
```

```tsx
return recommendations.map((item, index) => {
  const position = index + 1;
  return (
    <a
      key={item.dataset_id}
      href={buildTrackingUrl(item.dataset_id, requestId, position)}
      data-request-id={requestId}
      data-dataset-id={item.dataset_id}
      data-position={position}
      onClick={(event) => trackClick(event, item, position)}
    >
      ...
    </a>
  );
});
```

| URL 参数 | 必填 | 说明 | 示例 |
| --- | --- | --- | --- |
| `from` | ✅ | 标记来源为推荐 | `recommend` |
| `rid` | ✅ | 推荐请求 ID | `req_20251018_120530_abc123` |
| `pos` | ✅ | 推荐位次，1 开始 | `1`, `2`, `3` |

### 步骤 3：发送 Matomo 点击事件（必须）

#### Matomo 自定义维度配置

由 Matomo 管理员在后台配置并告知维度 ID：

| 维度 ID | 作用 | Scope |
| --- | --- | --- |
| `dimension_dataset`（示例：1） | `dataset_id` | Visit |
| `dimension_request`（示例：2） | `request_id` | Visit |

> 如果使用既有维度，请以管理员下发的 ID 为准。

#### 点击埋点示例

```ts
const MATOMO_DIM_DATASET = Number(import.meta.env.VITE_MATOMO_DIM_DATASET);
const MATOMO_DIM_REQUEST = Number(import.meta.env.VITE_MATOMO_DIM_REQUEST);

export const trackClick = (event: MouseEvent, item: RecommendationItem, position: number) => {
  const requestId = (event.currentTarget as HTMLElement).dataset.requestId;
  if (!window._paq || !requestId) return;

  window._paq.push(['setCustomDimension', MATOMO_DIM_DATASET, String(item.dataset_id)]);
  window._paq.push(['setCustomDimension', MATOMO_DIM_REQUEST, requestId]);
  window._paq.push([
    'trackEvent',
    'Recommendation',
    'Click',
    `dataset_${item.dataset_id}`,
    position, // 1-based
  ]);

  // 可选：缓存最近一次点击用于购买归因
  cacheLastClick({ requestId, datasetId: item.dataset_id, position });
};
```

- **位置说明**：`position` 必须从 1 开始递增，Matomo `trackEvent` 的 value 字段会被训练/评估脚本读取，用于计算各位次 CTR。
- **发送顺序**：务必先 `setCustomDimension`，再 `trackEvent`，否则维度不会附着在事件上。
- **跳转前 flush**：如需确保事件送达，可使用 `navigator.sendBeacon` 或 `keepalive` fetch。

---

## Framework 示例

### React

（示例基于旧文档结构，已更新 position 与自定义维度）

```tsx
const RecommendationList = ({ datasetId, userId }: Props) => {
  const [recommendations, setRecommendations] = useState<RecommendationItem[]>([]);
  const [requestId, setRequestId] = useState('');

  useEffect(() => {
    fetchRecommendations(datasetId, userId).then(({ recommendations, request_id }) => {
      setRecommendations(recommendations);
      setRequestId(request_id);
    });
  }, [datasetId, userId]);

  const handleClick = (item: RecommendationItem, position: number) => {
    if (!window._paq || !requestId) return;
    window._paq.push(['setCustomDimension', MATOMO_DIM_DATASET, String(item.dataset_id)]);
    window._paq.push(['setCustomDimension', MATOMO_DIM_REQUEST, requestId]);
    window._paq.push(['trackEvent', 'Recommendation', 'Click', `dataset_${item.dataset_id}`, position]);
    cacheLastClick({ requestId, datasetId: item.dataset_id, position });
  };

  const trackingUrl = (datasetId: number, position: number) =>
    `/dataDetail/${datasetId}?from=recommend&rid=${requestId}&pos=${position}`;

  return (
    <div className="recommendations">
      {recommendations.map((item, index) => {
        const position = index + 1;
        return (
          <a
            key={item.dataset_id}
            href={trackingUrl(item.dataset_id, position)}
            data-request-id={requestId}
            onClick={() => handleClick(item, position)}
          >
            ...
          </a>
        );
      })}
    </div>
  );
};
```

### Vue 3

```vue
<a
  :href="buildUrl(item.dataset_id, index + 1)"
  @click="handleClick(item, index + 1)"
>
  ...
</a>

const handleClick = (item, position) => {
  if (!window._paq || !requestId.value) return;
  window._paq.push(['setCustomDimension', MATOMO_DIM_DATASET, String(item.dataset_id)]);
  window._paq.push(['setCustomDimension', MATOMO_DIM_REQUEST, requestId.value]);
  window._paq.push(['trackEvent', 'Recommendation', 'Click', `dataset_${item.dataset_id}`, position]);
  cacheLastClick({ requestId: requestId.value, datasetId: item.dataset_id, position });
};
```

---

## 前端验证清单

- [ ] 推荐 API 返回体包含 `request_id`。
- [ ] 推荐链接包含 `?from=recommend&rid=req_xxx&pos=1`（pos 为 1-based）。
- [ ] 点击后 Network 中的 `matomo.php` 请求包含：
  - `dimension{datasetDim}=dataset_id`
  - `dimension{requestDim}=req_xxx`
  - `e_c=Recommendation`、`e_a=Click`
  - `e_v=position`
- [ ] Matomo 实时日志能看到 Custom Dimensions；数据团队在 `matomo_log_link_visit_action` 中能查到 `custom_dimension_5`、`custom_dimension_4`。

> Console 调试：`localStorage.setItem('matomoDebug', '1')` 可在浏览器 Console 中看到 `_paq` 指令执行日志。

---

## 购买追踪（Conversion Tracking）

### 1. 保存最近一次推荐点击

```ts
const cacheLastClick = (payload: { requestId: string; datasetId: number; position: number }) => {
  const expiry = Date.now() + 7 * 24 * 60 * 60 * 1000;
  localStorage.setItem('last_recommend_click', JSON.stringify(payload));
  localStorage.setItem('last_recommend_click_expiry', String(expiry));
};
```

### 2. 购买成功时上报

```ts
const trackPurchaseSuccess = (order: { id: string; datasetId: number; amount: number }) => {
  if (!window._paq) return;

  const cached = loadLastClick(order.datasetId);
  if (cached) {
    window._paq.push(['setCustomDimension', MATOMO_DIM_DATASET, String(cached.datasetId)]);
    window._paq.push(['setCustomDimension', MATOMO_DIM_REQUEST, cached.requestId]);
  }

  window._paq.push(['trackEcommerceOrder', order.id, order.amount, order.amount, 0, 0, 0]);
  window._paq.push(['trackEvent', 'Recommendation', 'Purchase', `dataset_${order.datasetId}`, order.amount]);

  if (cached) clearLastClick();
};
```

> Matomo 自带电商 API，会自动把最近一次设置的自定义维度附加到订单转化中，数据团队即可用同一个 `request_id` 还原点击 → 购买链路。

### 3. Matomo 后台要求

1. 启用电商追踪（Administration → Ecommerce）。
2. 可选：配置 `Goal`（如 “Dataset Purchase”）用于快速监控。

### 4. 验证

- 浏览器 Network：订单成功后应有 `matomo.php` 请求，包含 `ec_id=orderId`、`revenue` 等参数。
- Matomo 后台 → 电子商务日志：可看到订单金额及 Custom Dimensions。
- 数据团队 SQL：

```sql
SELECT idorder, revenue, custom_dimension_1 AS request_id, custom_dimension_4 AS position
FROM matomo_log_conversion
ORDER BY server_time DESC
LIMIT 20;
```

---

## 常见问题

- **Matomo 未加载**：在 `trackClick/trackPurchase` 中判断 `window._paq` 是否存在，必要时上报监控。
- **位置缺失**：`pos` 参数与 `trackEvent value` 必须保持 1-based，一致传递；否则无法绘制位次 CTR 曲线。
- **重复点击/下单**：可使用 `Set` 或 `sessionStorage` 去重；购买成功后清除缓存，防止重复归因。
- **SPA 路由**：确保每次进入详情页都重新获取推荐与 `request_id`；或在路由参数中携带 `rid/pos`。
- **下游未见数据**：请与数据团队确认 Matomo 自定义维度 ID 是否匹配，以及事件是否落在正确的 `siteId`。

---

## 附录：链路示意

```
1. 推荐 API 返回 recommendations + request_id
2. 前端渲染卡片，链接包含 from/recommend + rid + pos
3. 用户点击 → Matomo 记录 Recommendation/Click 事件（附带 dataset/request/position）
4. 页面跳转到详情页，URL 携带 rid/pos
5. 用户购买 → 前端将缓存的 request_id/dataset_id/position 赋给 Matomo，并调用 trackEcommerceOrder
6. Matomo 数据导出 → pipeline.aggregate_matomo_events 读取 custom_dimension + position
7. 数据团队即可计算 CTR/CVR、训练特征、监控推荐效果
```

这份文档保持了旧版章节组织，但补充了 position 记录方式与下游使用方式；如需新增字段（例如 `experiment_id`），可在 Matomo 中再申请自定义维度或使用 `setCustomVariable`。
