# Matomo 前端集成指引

面向前端同学的电商追踪埋点指南，覆盖脚本加载、单页应用（SPA）/SSR 处理、电商事件（曝光、加购、下单）、自定义维度以及调试方法。目标是保证 Matomo 能按 `request_id`、`dataset_id`、`order_id` 等关键字段打通推荐链路和转化。

---

## 1. 基础依赖与准备

| 项 | 说明 |
| --- | --- |
| Matomo 服务器 | `https://matomo.example.com/`（替换为实际域名） |
| 站点 ID (`siteId`) | 运营侧在 Matomo 后台创建，以项目为单位配置，例如 `123` |
| 追踪 JS | 优先使用官方 `matomo.js`，可直接通过 `<script>` 引入，也可由 Tag Manager 托管 |
| 自定义维度 | `custom_dimension_4`: `request_id`（点击/浏览）；`custom_dimension_1`: 购买回退的 `request_id`；`custom_dimension_5`: `click_{position}_{datasetId}` 元数据 |

> **命名约定**  
> - 金额统一使用“元”为单位，与后端 `matomo_log_conversion` 对齐。  
> - `orderId`、`sku`、`datasetId` 等字符串需确保唯一，可复用业务已有 ID。

---

## 2. SDK 加载模板

在 HTML `<head>`（或框架入口模板）插入以下片段，确保尽早执行。若为 SSR，请在仅浏览器可执行的生命周期里插入。

```html
<script>
  window._paq = window._paq || [];
  window._paq.push(['setTrackerUrl', 'https://matomo.example.com/matomo.php']);
  window._paq.push(['setSiteId', '123']);
  window._paq.push(['enableLinkTracking']);
  window._paq.push(['enableHeartBeatTimer']); // 可选：长页面心跳
</script>
<script async defer src="https://matomo.example.com/matomo.js"></script>
```

**环境配置建议**

| 环境 | `trackerUrl` | `siteId` | 备注 |
| --- | --- | --- | --- |
| 本地 / Staging | 测试 Matomo 或生产地址 + 测试 Site | 单独 siteId，避免污染线上报表 | 可在 `.env` 或构建参数中注入 |
| 生产 | 正式 Matomo | 线上 siteId | 涉及 PII 时遵循隐私规范 |

---

## 3. 前端工具方法（示例）

建议封装统一的追踪助手，避免在各组件中直接操作 `window._paq`。以下 TypeScript 示例可在 React/Vue/Nuxt/Next 等项目中复用：

```ts
type MatomoEvent = Parameters<typeof window._paq.push>;

const enqueue = (...args: MatomoEvent) => {
  if (typeof window === 'undefined') return;
  const queue = (window._paq = window._paq || []);
  queue.push(args as any);
};

export const trackPageView = (url?: string, title?: string) => {
  if (url) enqueue(['setCustomUrl', url]);
  if (title) enqueue(['setDocumentTitle', title]);
  enqueue(['trackPageView']);
};

export const setUserContext = (userId?: string) => {
  if (userId) enqueue(['setUserId', userId]);
};

export const setCustomDimension = (id: number, value?: string) => {
  if (value) enqueue(['setCustomDimension', id, value]);
};
```

> SSR 注意：上述代码需在 `useEffect`、`onMounted` 等仅运行于浏览器的钩子内调用；或在编译阶段用 `process.client` / `typeof window !== 'undefined'` 守卫。

---

## 4. SPA 与路由切换

单页应用不会触发浏览器原生页面刷新，需要在路由变化时手动上报：

```ts
import { trackPageView } from './matomo';

router.afterEach((to) => {
  trackPageView(window.location.origin + to.fullPath, document.title);
});
```

若页面存在过滤条件切换（但 URL 不变），可在交互完成后调用 `trackPageView`，或使用 `trackEvent`：

```ts
enqueue(['trackEvent', 'Filter', 'apply', JSON.stringify(activeFilters), totalResults]);
```

HeartBeat 心跳可用于衡量停留时长，默认 15s，可通过 `enableHeartBeatTimer(10)` 调整。

---

## 5. 电商/购买流程埋点

Matomo 提供标准化的电商 API，用于记录商品曝光、加购、下单。

### 5.1 详情曝光

用户进入详情页时：

```ts
enqueue(['setCustomDimension', 4, requestId]);               // request_id（用于点击/详情事件）
enqueue(['setCustomDimension', 5, `click_${position}_${datasetId}`]); // 记录位次 + 数据集
enqueue(['setEcommerceView', sku, name, categoryPath, price]);
trackPageView(); // 保持页面浏览记录
```

`categoryPath` 可为数组或字符串（如 `"AI Dataset/Data Science"`）。

### 5.2 加入购物车 & 购物车更新

```ts
enqueue(['addEcommerceItem', sku, name, categoryPath, price, quantity]);
enqueue(['trackEcommerceCartUpdate', cartTotal]);
```

`cartTotal` 为当前购物车总金额。

### 5.3 下单/支付成功

在支付成功回调里统一上报，确保包含所有金额字段：

```ts
enqueue([
  'trackEcommerceOrder',
  orderId,              // 必填，唯一
  grandTotal,           // 订单总金额（含运费、税费）
  subTotal,             // 商品小计
  taxAmount,
  shippingFee,
  discountAmount
]);
```

> - 若只记录订阅类或一次性购买，`subTotal` 与 `grandTotal` 一致即可。  
> - 使用 `navigator.sendBeacon` 或 `keepalive: true` 的 `fetch`，避免用户关闭窗口导致事件丢失。

---

## 6. 自定义维度与属性对齐

| 维度 ID | 字段 | 设置时机 | 用途 |
| --- | --- | --- | --- |
| 1 | `dataset_id` | 详情曝光、加购、下单前 | 数据团队按数据集聚合转化 |
| 2 | `request_id` | 推荐曝光/点击跳转后立即设置 | 与推荐 API 日志打通，精确计算漏斗 |
| 3（示例） | `variant` | 实验/AB 标识 | 分析多版本效果 |

示例：

```ts
setCustomDimension(1, datasetId);
setCustomDimension(2, requestId);
setCustomDimension(3, variant);
```

如需同步用户标识或业务标签，可再使用：

```ts
enqueue(['setUserId', userId]);           // 站内登录 ID
enqueue(['setCustomVariable', 1, 'role', userRole, 'page']); // legacy 变量
```

---

## 7. 常见调试与验证流程

1. **浏览器 Console**：确认 `window._paq` 队列存在对应指令；必要时启用 `localStorage.setItem('matomoDebug', '1')` 查看日志。  
2. **Network 面板**：过滤 `matomo.php`，检查 querystring 中的 `idsite`、`ec_id`、`revenue`、`dimension1` 等参数。  
3. **Matomo 实时面板**：后台 “访客 → 实时日志” 可在 1 分钟内看到新事件。  
4. **后端校验**：数据团队会在 `matomo_log_link_visit_action` / `matomo_log_conversion` 中核对 `custom_dimension_4 = request_id`（点击）与 `custom_dimension_1 = request_id`（购买）、`idorder = OrderId`。  
5. **自动化测试（可选）**：在 E2E 测试里拦截 `matomo.php` 请求，断言 querystring，避免回归。

---

## 8. 故障排查清单

- **事件丢失**：检查脚本是否被 CSP 拦截；或页面跳转过快，需改用 `sendBeacon`/`keepalive`。  
- **金额不一致**：确认是否都以“元”为单位，且为数字类型；优惠券需传负数或在 `discountAmount` 字段体现。  
- **自定义维度缺失**：确保在 `trackEcommerceOrder` 之前调用 `setCustomDimension`，维度值会附加到同一访客会话。  
- **SPA 未记录 PV**：在路由 `afterEach` 里触发 `trackPageView`，并调用 `setCustomUrl`。  
- **多次触发下单**：下单接口需保证幂等，或在前端标记 `orderTracked`，避免重复 push。

---

## 9. 交付与对接

1. 前端根据本指引完成埋点，实现/调用统一的 `matomo.ts` 或 `tracking.ts`。  
2. 自测覆盖：详情曝光 + 加购 + 下单全链路。  
3. 将真实订单号、时间、`dataset_id` 汇报给数据团队，协助验证数据库是否写入。  
4. 上线前在 Staging 进行 1~2 天监控，确保 Matomo 报表与业务日志一致。

如需新增维度或事件（例如优惠券使用、推荐位曝光等），请在提测前与数据团队确认字段映射，避免后续重复改动。
