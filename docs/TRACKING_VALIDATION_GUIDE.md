# Request ID追踪功能 - 测试验证指南

## 核心问题

**挑战**：前后端尚未实施Request ID追踪，如何在发版前验证整个流程是正确的？

- ✅ API已返回request_id
- ✅ 曝光日志已记录request_id
- ❌ Matomo中没有带request_id的点击数据（前端未实施）
- ❓ 无法验证evaluate_v2.py的匹配逻辑

## 解决方案：三步验证法

### 方案1: 手动模拟数据验证（最快）

#### 步骤1: 获取真实request_id

```bash
# 调用API生成真实曝光
curl "http://localhost:8090/recommend/detail/1?user_id=123&limit=5" | jq .

# 记录返回值：
# request_id: "abc-123-def-456"
# recommendations[0].dataset_id: 5791
```

#### 步骤2: 手动插入Matomo测试数据

```sql
-- 连接Matomo数据库
USE matomo;

-- 插入带request_id的点击记录
INSERT INTO matomo_log_link_visit_action (
    idlink_va, idsite, idvisitor, idvisit,
    idaction_url, server_time, custom_dimension_3
) VALUES (
    999999,  -- 测试ID
    1,       -- 网站ID
    UNHEX(MD5('test_visitor')),
    999999,
    999999,
    NOW(),
    'abc-123-def-456'  -- 你刚才记录的request_id
);

-- 插入对应的action记录
INSERT INTO matomo_log_action (idaction, name, hash, type)
VALUES (999999, '/dataDetail/5791?from=recommend&rid=abc-123-def-456&pos=0',
        CRC32('/dataDetail/5791'), 1);
```

#### 步骤3: 运行evaluate_v2验证

```bash
python -m pipeline.evaluate_v2
cat data/evaluation/tracking_report_v2.json | jq '.summary'
```

**预期结果**:
- `total_clicks` > 0
- `unique_request_ids` > 0
- CTR计算正确

---

### 方案2: 使用自动化测试脚本

创建 `scripts/verify_tracking.py`:

```python
#!/usr/bin/env python3
"""快速验证Request ID追踪功能"""
import requests
import pymysql
import json
import time

def test_tracking():
    print("=== Request ID追踪验证 ===\n")

    # 1. 调用API
    print("[1] 调用推荐API...")
    resp = requests.get("http://localhost:8090/recommend/detail/1?user_id=test")
    data = resp.json()
    request_id = data['request_id']
    dataset_id = data['recommendations'][0]['dataset_id']
    print(f"✓ Request ID: {request_id}")
    print(f"✓ Dataset ID: {dataset_id}")

    # 2. 验证曝光日志
    print("\n[2] 验证曝光日志...")
    time.sleep(1)
    with open('data/logs/exposure_log.jsonl') as f:
        lines = f.readlines()
        found = any(request_id in line for line in lines[-10:])
        print("✓ 曝光已记录" if found else "✗ 曝光未找到")

    # 3. 插入Matomo测试数据
    print("\n[3] 插入Matomo测试数据...")
    conn = pymysql.connect(
        host='localhost', user='matomo_user',
        password='your_password', database='matomo'
    )
    with conn.cursor() as cur:
        # 插入点击
        cur.execute("""
            INSERT INTO matomo_log_link_visit_action
            (idlink_va, idsite, idvisitor, idvisit, idaction_url,
             server_time, custom_dimension_3)
            VALUES (999999, 1, UNHEX(MD5('test')), 999999, 999999,
                    NOW(), %s)
        """, (request_id,))

        # 插入action
        url = f"/dataDetail/{dataset_id}?from=recommend&rid={request_id}&pos=0"
        cur.execute("""
            INSERT INTO matomo_log_action (idaction, name, hash, type)
            VALUES (999999, %s, CRC32(%s), 1)
        """, (url, url))

        conn.commit()
    conn.close()
    print("✓ 测试数据已插入")

    # 4. 运行评估
    print("\n[4] 运行评估脚本...")
    import subprocess
    subprocess.run(['python', '-m', 'pipeline.evaluate_v2'])

    # 5. 显示结果
    print("\n[5] 验证结果:")
    with open('data/evaluation/tracking_report_v2.json') as f:
        report = json.load(f)
        summary = report['summary']
        print(f"  曝光: {summary['total_exposures']}")
        print(f"  点击: {summary['total_clicks']}")
        print(f"  CTR: {summary['overall_ctr']:.4f}")
        print(f"  Request IDs: {summary['unique_request_ids']}")

        if summary['total_clicks'] > 0:
            print("\n✓ 验证通过！追踪功能正常")
            return True
        else:
            print("\n✗ 验证失败：未匹配到点击")
            return False

if __name__ == '__main__':
    success = test_tracking()
    exit(0 if success else 1)
```

使用：
```bash
python scripts/verify_tracking.py
```

---

### 方案3: 前端本地测试（最真实）

前端开发时：

1. **实施追踪代码**（参考 FRONTEND_INTEGRATION.md）

2. **配置本地API**:
```javascript
const API_URL = 'http://localhost:8090';
```

3. **手动测试**:
   - 访问推荐页面
   - 查看链接: `<a href="/dataDetail/123?from=recommend&rid=xxx&pos=0">`
   - 点击推荐
   - 打开 DevTools > Network
   - 查找 Matomo请求，确认包含 `dimension1=req_xxx`

4. **验证数据库**:
```sql
SELECT custom_dimension_3, COUNT(*)
FROM matomo_log_link_visit_action
WHERE custom_dimension_3 LIKE 'req_%'
GROUP BY custom_dimension_3
LIMIT 10;
```

5. **运行评估**:
```bash
python -m pipeline.evaluate_v2
```

---

## 验证检查清单

### ✓ 准备阶段
- [ ] 推荐API返回 request_id
- [ ] Matomo有 custom_dimension_3 列
- [ ] evaluate_v2.py 脚本就绪

### ✓ 数据层验证
- [ ] 曝光日志包含 request_id
- [ ] Matomo测试数据包含 request_id
- [ ] Request ID格式一致

### ✓ 匹配逻辑验证
- [ ] evaluate_v2.py 成功加载曝光
- [ ] evaluate_v2.py 成功加载点击
- [ ] Request ID成功匹配
- [ ] CTR计算正确

### ✓ 端到端验证
- [ ] API → 曝光日志 ✓
- [ ] 前端 → Matomo ✓
- [ ] Matomo → 评估脚本 ✓
- [ ] CTR数值合理(0.03-0.15)

---

## 快速验证命令

```bash
# 1分钟验证脚本
#!/bin/bash

echo "=== 快速验证 ==="

# 测试API
REQUEST_ID=$(curl -s http://localhost:8090/recommend/detail/1?user_id=1 | jq -r .request_id)
echo "Request ID: $REQUEST_ID"

# 检查曝光
grep -q "$REQUEST_ID" data/logs/exposure_log.jsonl && echo "✓ 曝光OK" || echo "✗ 曝光失败"

# 运行评估
python -m pipeline.evaluate_v2 2>&1 | grep -E "(exposures|clicks|ctr)"

echo "完整报告: data/evaluation/tracking_report_v2.json"
```

---

## 常见问题

**Q: 没有Matomo数据库访问权限怎么办？**

A: 使用Matomo HTTP API插入数据：
```bash
curl "http://your-matomo/matomo.php" \
  -d "idsite=1" \
  -d "rec=1" \
  -d "url=http://localhost/dataDetail/123?from=recommend&rid=$REQUEST_ID&pos=0" \
  -d "dimension1=$REQUEST_ID"
```

**Q: evaluate_v2.py 显示 "No clicks found"?**

A: 检查：
1. Matomo custom_dimension_3 有数据吗？
2. URL包含 `from=recommend` 吗？
3. Request ID格式匹配吗？

**Q: CTR异常高（>0.5）？**

A: 可能原因：
- 测试数据污染
- URL过滤条件不正确
- 时间范围匹配问题

---

## 推荐验证流程

**第1步**（1小时）：手动模拟数据验证核心逻辑

**第2步**（1天）：前端本地环境验证集成

**第3步**（0.5天）：生产环境小流量灰度（10%）

**第4步**（监控1周）：追踪覆盖率 > 80% 后全量

---

**版本**: v1.0
**日期**: 2025-10-19
