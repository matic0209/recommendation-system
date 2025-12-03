# Matomo 购买追踪完整指南

> 针对推荐系统的电商转化追踪详细实施方案

---

## 📋 目录

1. [购买流程概览](#购买流程概览)
2. [完整代码示例](#完整代码示例)
3. [各场景详细说明](#各场景详细说明)
4. [调试验证](#调试验证)
5. [常见问题](#常见问题)

---

## 购买流程概览

```
用户旅程                     前端追踪事件                      Matomo 数据
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. 推荐列表曝光
   ├─ 用户看到推荐数据集      setCustomDimension(2, requestId)
   └─                        trackEvent('Recommendation', 'impression')

2. 点击详情
   ├─ 跳转到详情页           setCustomDimension(1, datasetId)
   └─                        setEcommerceView()               → matomo_log_link_visit_action
                            trackPageView()

3. 加入购物车
   ├─ 用户点击加购按钮        addEcommerceItem()
   └─                        trackEcommerceCartUpdate()       → matomo_log_conversion (cartUpdate)

4. 提交订单
   ├─ 用户完成支付           trackEcommerceOrder()            → matomo_log_conversion (order)
   └─                        关键：orderId, grandTotal, 自定义维度
```

---

## 完整代码示例

### 1. 基础工具函数（matomo.ts）

```typescript
// matomo.ts - 统一的 Matomo 追踪工具

/**
 * Matomo 追踪队列
 */
declare global {
  interface Window {
    _paq: any[][];
  }
}

/**
 * 安全地向 Matomo 队列推送指令
 */
const enqueue = (...args: any[]) => {
  if (typeof window === 'undefined') return;
  window._paq = window._paq || [];
  window._paq.push(args);
};

/**
 * 设置用户 ID
 */
export const setUserId = (userId: string) => {
  enqueue(['setUserId', userId]);
};

/**
 * 设置自定义维度
 * @param id - 维度 ID (1: dataset_id, 2: request_id)
 * @param value - 维度值
 */
export const setCustomDimension = (id: number, value: string) => {
  enqueue(['setCustomDimension', id, value]);
};

/**
 * 追踪页面浏览
 */
export const trackPageView = (customUrl?: string, title?: string) => {
  if (customUrl) enqueue(['setCustomUrl', customUrl]);
  if (title) enqueue(['setDocumentTitle', title]);
  enqueue(['trackPageView']);
};

/**
 * 追踪自定义事件
 */
export const trackEvent = (
  category: string,
  action: string,
  name?: string,
  value?: number
) => {
  enqueue(['trackEvent', category, action, name, value]);
};

// ============================================
// 电商追踪相关
// ============================================

/**
 * 商品详情曝光
 * @param sku - 商品 SKU（数据集 ID）
 * @param name - 商品名称
 * @param category - 商品分类（可选）
 * @param price - 商品价格（元）
 */
export const trackProductView = (
  sku: string,
  name: string,
  category?: string | string[],
  price?: number
) => {
  enqueue(['setEcommerceView', sku, name, category, price]);
};

/**
 * 添加商品到购物车
 * @param sku - 商品 SKU
 * @param name - 商品名称
 * @param category - 商品分类
 * @param price - 单价（元）
 * @param quantity - 数量
 */
export const addToCart = (
  sku: string,
  name: string,
  category: string | string[],
  price: number,
  quantity: number = 1
) => {
  enqueue(['addEcommerceItem', sku, name, category, price, quantity]);
};

/**
 * 更新购物车（在 addToCart 之后调用）
 * @param cartTotal - 购物车总金额（元）
 */
export const updateCart = (cartTotal: number) => {
  enqueue(['trackEcommerceCartUpdate', cartTotal]);
};

/**
 * 追踪订单完成（核心方法）
 * @param orderId - 订单 ID（必须唯一）
 * @param grandTotal - 订单总金额（元，含运费/税费）
 * @param subTotal - 商品小计（元）
 * @param tax - 税费（元，可选）
 * @param shipping - 运费（元，可选）
 * @param discount - 优惠金额（元，可选）
 */
export const trackPurchase = (
  orderId: string,
  grandTotal: number,
  subTotal?: number,
  tax?: number,
  shipping?: number,
  discount?: number
) => {
  enqueue([
    'trackEcommerceOrder',
    orderId,
    grandTotal,
    subTotal || grandTotal,
    tax || 0,
    shipping || 0,
    discount || 0,
  ]);
};

/**
 * 清空购物车（移除所有商品）
 */
export const clearEcommerceCart = () => {
  enqueue(['clearEcommerceCart']);
};
```

---

### 2. 场景 1：详情页曝光

```typescript
// pages/dataset/[id].tsx

import { useEffect } from 'react';
import { useRouter } from 'next/router';
import {
  setCustomDimension,
  trackProductView,
  trackPageView,
} from '@/lib/matomo';

export default function DatasetDetailPage({ dataset, requestId }) {
  const router = useRouter();

  useEffect(() => {
    // 1. 设置自定义维度
    setCustomDimension(1, dataset.id.toString());        // dataset_id
    setCustomDimension(2, requestId);                    // request_id（从推荐 API 获取）

    // 2. 追踪商品详情曝光
    trackProductView(
      dataset.id.toString(),                             // sku
      dataset.title,                                     // name
      dataset.category || 'Dataset',                    // category
      dataset.price                                      // price（元）
    );

    // 3. 追踪页面浏览
    trackPageView();
  }, [dataset.id, requestId]);

  return (
    <div>
      <h1>{dataset.title}</h1>
      <p>价格: ¥{dataset.price}</p>
      {/* ... */}
    </div>
  );
}
```

---

### 3. 场景 2：加入购物车

```typescript
// components/AddToCartButton.tsx

import { addToCart, updateCart, trackEvent } from '@/lib/matomo';
import { useCart } from '@/hooks/useCart';

export function AddToCartButton({ dataset }) {
  const { cart, addItem } = useCart();

  const handleAddToCart = async () => {
    // 1. 业务逻辑：添加到购物车
    await addItem(dataset);

    // 2. Matomo 追踪：添加商品
    addToCart(
      dataset.id.toString(),                    // sku
      dataset.title,                            // name
      dataset.category || 'Dataset',           // category
      dataset.price,                            // price
      1                                         // quantity
    );

    // 3. Matomo 追踪：更新购物车总额
    const newTotal = cart.items.reduce((sum, item) => sum + item.price, 0) + dataset.price;
    updateCart(newTotal);

    // 4. 可选：追踪事件（用于分析）
    trackEvent(
      'Cart',                                   // category
      'add',                                    // action
      dataset.title,                            // name
      dataset.price                             // value
    );
  };

  return (
    <button onClick={handleAddToCart}>
      加入购物车
    </button>
  );
}
```

---

### 4. 场景 3：订单支付成功（最重要）⭐

```typescript
// pages/checkout/success.tsx

import { useEffect } from 'react';
import { useRouter } from 'next/router';
import {
  setCustomDimension,
  addToCart,
  trackPurchase,
  clearEcommerceCart,
} from '@/lib/matomo';

export default function OrderSuccessPage({ order }) {
  const router = useRouter();

  useEffect(() => {
    // 防止重复追踪
    const tracked = sessionStorage.getItem(`order_tracked_${order.id}`);
    if (tracked) return;

    // 1. 设置订单相关的自定义维度
    if (order.items.length > 0) {
      // 如果是单个商品订单，设置 dataset_id
      setCustomDimension(1, order.items[0].dataset_id.toString());
    }
    // 如果有 request_id（从订单数据获取），也设置
    if (order.request_id) {
      setCustomDimension(2, order.request_id);
    }

    // 2. 添加订单中的所有商品
    order.items.forEach((item) => {
      addToCart(
        item.dataset_id.toString(),            // sku
        item.title,                             // name
        item.category || 'Dataset',            // category
        item.price,                             // price
        item.quantity || 1                      // quantity
      );
    });

    // 3. 追踪订单（核心）
    trackPurchase(
      order.id.toString(),                     // orderId - 必须唯一！
      order.total_amount,                      // grandTotal（总金额，含运费等）
      order.subtotal,                          // subTotal（商品小计）
      order.tax || 0,                          // tax（税费）
      order.shipping_fee || 0,                 // shipping（运费）
      order.discount_amount || 0               // discount（优惠金额）
    );

    // 4. 清空电商购物车（Matomo 内部状态）
    clearEcommerceCart();

    // 5. 标记已追踪，避免重复
    sessionStorage.setItem(`order_tracked_${order.id}`, 'true');

    // 6. 可选：使用 sendBeacon 确保数据发送
    if (navigator.sendBeacon) {
      // Matomo 会自动使用 sendBeacon，这里只是提醒
      console.log('Order tracking sent via beacon');
    }
  }, [order.id]);

  return (
    <div>
      <h1>支付成功！</h1>
      <p>订单号：{order.id}</p>
      <p>总金额：¥{order.total_amount}</p>
    </div>
  );
}
```

---

### 5. 场景 4：购物车页面更新

```typescript
// pages/cart.tsx

import { useEffect } from 'react';
import { addToCart, updateCart, clearEcommerceCart } from '@/lib/matomo';
import { useCart } from '@/hooks/useCart';

export default function CartPage() {
  const { cart } = useCart();

  useEffect(() => {
    // 每次购物车变化时更新 Matomo
    if (cart.items.length === 0) {
      clearEcommerceCart();
      updateCart(0);
      return;
    }

    // 清空之前的状态
    clearEcommerceCart();

    // 重新添加所有商品
    cart.items.forEach((item) => {
      addToCart(
        item.dataset_id.toString(),
        item.title,
        item.category || 'Dataset',
        item.price,
        item.quantity || 1
      );
    });

    // 更新总额
    const total = cart.items.reduce((sum, item) =>
      sum + (item.price * (item.quantity || 1)), 0
    );
    updateCart(total);
  }, [cart.items]);

  return (
    <div>
      <h1>购物车</h1>
      {/* ... */}
    </div>
  );
}
```

---

## 各场景详细说明

### 💡 关键要点

#### 1. request_id 的传递

从推荐 API 获取的 `request_id` 需要在整个用户旅程中传递：

```typescript
// 推荐列表页
const { data } = await fetch('/api/recommend/detail/1?user_id=123&limit=10');
const requestId = data.request_id;

// 方式 1: URL 参数传递
router.push(`/dataset/${datasetId}?request_id=${requestId}`);

// 方式 2: LocalStorage 传递（跨页面）
localStorage.setItem('last_request_id', requestId);

// 方式 3: 订单数据中保存（推荐）
// 在创建订单时，将 request_id 保存到订单表
```

#### 2. 防止重复追踪

```typescript
// 使用 sessionStorage 标记
const trackOnce = (key: string, trackFn: () => void) => {
  const tracked = sessionStorage.getItem(key);
  if (tracked) {
    console.log(`Already tracked: ${key}`);
    return;
  }
  trackFn();
  sessionStorage.setItem(key, 'true');
};

// 使用
trackOnce(`purchase_${orderId}`, () => {
  trackPurchase(orderId, total);
});
```

#### 3. 金额单位统一

⚠️ **重要：所有金额必须使用"元"为单位**

```typescript
// ✅ 正确
trackPurchase('ORDER123', 99.00);  // 99 元

// ❌ 错误
trackPurchase('ORDER123', 9900);   // 不要用分
```

#### 4. 异步支付回调处理

```typescript
// 支付成功回调页面
useEffect(() => {
  const queryParams = new URLSearchParams(window.location.search);
  const orderId = queryParams.get('order_id');

  if (!orderId) return;

  // 从后端获取订单详情
  fetch(`/api/orders/${orderId}`)
    .then(res => res.json())
    .then(order => {
      if (order.status === 'paid') {
        trackPurchase(order.id, order.total_amount);
      }
    });
}, []);
```

---

## 调试验证

### 1. 浏览器控制台检查

```javascript
// 在浏览器控制台运行
console.log(window._paq);

// 应该看到类似：
// [
//   ['setCustomDimension', 1, '123'],
//   ['setCustomDimension', 2, 'req_abc...'],
//   ['addEcommerceItem', '123', 'Dataset Name', 'Category', 99.00, 1],
//   ['trackEcommerceOrder', 'ORDER123', 99.00, 99.00, 0, 0, 0]
// ]
```

### 2. Network 请求检查

打开 Chrome DevTools → Network 标签：

1. 过滤 `matomo.php`
2. 查看请求参数：
   ```
   idsite=123
   rec=1
   ec_id=ORDER123            ← 订单 ID
   revenue=99.00             ← 订单金额
   dimension1=123            ← dataset_id
   dimension2=req_abc...     ← request_id
   ```

### 3. Matomo 实时日志

登录 Matomo 后台：
1. 访客 → 实时日志
2. 应该在 1 分钟内看到新事件
3. 点击事件查看详情：
   - Custom Dimension 4: req_abc...（request_id）
   - Custom Dimension 5: click_1_123（位次 + dataset_id）
   - Ecommerce Order: ORDER123
   - Revenue: ¥99.00

### 4. 数据库验证

让数据团队查询：

```sql
-- 查询最近的订单转化
SELECT
  idorder,
  revenue,
  custom_dimension_1, -- 购买回退 request_id
  custom_dimension_4, -- 购买时写入的位次（如有）
  server_time
FROM matomo_log_conversion
WHERE idaction = 2  -- 订单转化
ORDER BY server_time DESC
LIMIT 10;
```

---

## 常见问题

### Q1: 支付成功后刷新页面，订单会重复追踪吗？

**A:** 使用 `sessionStorage` 或 `localStorage` 标记已追踪：

```typescript
const tracked = sessionStorage.getItem(`order_tracked_${orderId}`);
if (tracked) return;

trackPurchase(orderId, total);
sessionStorage.setItem(`order_tracked_${orderId}`, 'true');
```

---

### Q2: 单页应用（SPA）切换页面，如何重新追踪？

**A:** 在路由变化时手动触发：

```typescript
// Next.js
router.events.on('routeChangeComplete', (url) => {
  trackPageView(url);
});

// Vue Router
router.afterEach((to) => {
  trackPageView(to.fullPath);
});
```

---

### Q3: 如果订单包含多个商品，怎么追踪？

**A:** 在 `trackPurchase` 之前，循环添加所有商品：

```typescript
// 1. 添加所有商品
order.items.forEach(item => {
  addToCart(item.sku, item.name, item.category, item.price, item.quantity);
});

// 2. 追踪订单（总金额）
trackPurchase(order.id, order.total_amount, order.subtotal);
```

---

### Q4: request_id 如何跨页面传递？

**A:** 三种方式：

1. **URL 参数**（推荐用于详情页跳转）
   ```typescript
   router.push(`/dataset/123?request_id=${requestId}`);
   ```

2. **LocalStorage**（跨会话）
   ```typescript
   localStorage.setItem('last_request_id', requestId);
   ```

3. **订单数据保存**（最可靠）
   ```typescript
   // 创建订单时保存 request_id
   await createOrder({
     items: [...],
     request_id: requestId  // 保存到数据库
   });

   // 支付成功页面从订单数据获取
   setCustomDimension(2, order.request_id);
   ```

---

### Q5: 如何测试追踪是否成功？

**A:** 完整测试流程：

1. ✅ 浏览器控制台检查 `window._paq`
2. ✅ Network 检查 `matomo.php` 请求参数
3. ✅ Matomo 实时日志查看事件
4. ✅ 数据团队验证数据库记录
5. ✅ 等待 24 小时查看报表

---

### Q6: sendBeacon 是什么？为什么需要？

**A:** `navigator.sendBeacon` 确保在页面关闭时数据能发送：

```typescript
// Matomo 会自动使用，但可以手动确保
if (navigator.sendBeacon) {
  // 数据会在页面卸载时可靠发送
}

// 或者使用 fetch keepalive
fetch(url, {
  method: 'POST',
  body: data,
  keepalive: true  // 关键
});
```

---

## 完整示例：购买流程端到端

```typescript
// ========================================
// 1. 推荐列表页
// ========================================
function RecommendationList() {
  const handleClickDataset = async (datasetId: number) => {
    // 获取推荐 API 的 request_id
    const response = await fetch(`/api/recommend/detail/1?user_id=123`);
    const { request_id } = await response.json();

    // 跳转详情页，带上 request_id
    router.push(`/dataset/${datasetId}?request_id=${request_id}`);
  };
}

// ========================================
// 2. 详情页
// ========================================
function DatasetDetail({ dataset }) {
  useEffect(() => {
    const requestId = router.query.request_id as string;

    // 设置维度
    setCustomDimension(1, dataset.id.toString());
    setCustomDimension(2, requestId);

    // 商品曝光
    trackProductView(dataset.id.toString(), dataset.title, dataset.category, dataset.price);
    trackPageView();
  }, []);
}

// ========================================
// 3. 加购
// ========================================
function handleAddToCart() {
  addToCart(dataset.id.toString(), dataset.title, dataset.category, dataset.price, 1);
  updateCart(cartTotal);
}

// ========================================
// 4. 支付成功
// ========================================
function OrderSuccess({ order }) {
  useEffect(() => {
    // 添加商品
    order.items.forEach(item => {
      addToCart(item.sku, item.name, item.category, item.price, item.quantity);
    });

    // 追踪订单
    trackPurchase(order.id, order.total_amount);

    // 清空购物车
    clearEcommerceCart();
  }, []);
}
```

---

## 参考资料

- [Matomo 前端集成指引](./matomo_frontend_integration.md)
- [Matomo 电商追踪官方文档](https://developer.matomo.org/guides/tracking-javascript-guide#ecommerce)
- [自定义维度配置](../config/settings.py) - `MATOMO_REQUEST_DIMENSIONS` / `MATOMO_REQUEST_DIMENSION`
- [后端验证脚本](../scripts/verify_tracking.py)

---

## 联系方式

如需协助调试或新增追踪维度，请联系：
- 数据团队: data-team@company.com
- 后端团队: backend-team@company.com
