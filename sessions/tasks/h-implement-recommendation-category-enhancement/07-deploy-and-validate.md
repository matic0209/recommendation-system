# 子任务07: 部署并验证效果

## 目标
重启推荐服务加载新模型和增强tags，验证推荐结果的类别聚焦度达到目标（同大类比例>80%）。

## 成功标准
- [ ] 服务成功重启并加载新模型
- [ ] 推荐API正常响应
- [ ] 同大类item比例从<50%提升到>80%
- [ ] 推荐结果多样性未明显下降
- [ ] 无性能退化（P99延迟<500ms）
- [ ] A/B测试配置就绪（如果需要）

## 实施步骤

### 1. 备份现有模型

```bash
# 创建备份目录
mkdir -p models/backup/$(date +%Y%m%d)

# 备份现有模型文件
cp models/rank_model.pkl models/backup/$(date +%Y%m%d)/
cp models/tag_to_items.json models/backup/$(date +%Y%m%d)/
cp models/item_to_tags.json models/backup/$(date +%Y%m%d)/

echo "模型已备份到 models/backup/$(date +%Y%m%d)/"
```

### 2. 更新模型文件（使用增强版本）

**选项A**: 直接替换（全量上线）
```bash
# 使用增强版本的tag索引
cp models/tag_to_items_enhanced.json models/tag_to_items.json
cp models/item_to_tags_enhanced.json models/item_to_tags.json

# rank_model.pkl已经是新训练的，包含类别特征
# 无需额外操作
```

**选项B**: 软链接（便于回滚）
```bash
# 创建软链接
ln -sf tag_to_items_enhanced.json models/tag_to_items.json
ln -sf item_to_tags_enhanced.json models/item_to_tags.json
```

### 3. 重启推荐服务

```bash
# 停止现有服务
docker-compose stop recommendation-api

# 或完全重启
docker-compose down recommendation-api

# 重新启动
docker-compose up -d recommendation-api

# 查看启动日志
docker-compose logs -f recommendation-api
```

### 4. 验证服务启动

```bash
# 等待服务启动（约10-30秒）
sleep 30

# 健康检查
curl http://localhost:8000/health

# 预期输出:
# {"status": "healthy", "model_loaded": true, ...}
```

### 5. 功能测试

创建`scripts/test_category_focus.py`:

```python
"""
测试推荐结果的类别聚焦度
"""
import requests
import json
from collections import Counter

API_URL = "http://localhost:8000/api/v1/recommend/detail"

# 标准大类列表
CATEGORIES = {
    "金融", "医疗健康", "政府政务", "交通运输",
    "教育培训", "能源环保", "农业", "科技",
    "商业零售", "文化娱乐", "社会民生"
}

def extract_categories(tags_list):
    """从tags列表中提取大类"""
    categories = []
    for tag in tags_list:
        if tag in CATEGORIES:
            categories.append(tag)
    return categories

def test_category_focus(dataset_id, user_id=123, limit=10):
    """测试单个推荐请求的类别聚焦度"""
    # 发送请求
    response = requests.get(API_URL, params={
        "dataset_id": dataset_id,
        "user_id": user_id,
        "limit": limit
    })

    if response.status_code != 200:
        print(f"❌ 请求失败: {response.status_code}")
        return None

    data = response.json()
    recommendations = data.get("recommendations", [])

    if not recommendations:
        print(f"⚠️  无推荐结果")
        return None

    # 加载item_to_tags获取tags
    with open("models/item_to_tags.json") as f:
        item_to_tags = json.load(f)

    # 分析类别分布
    target_categories = extract_categories(item_to_tags.get(str(dataset_id), []))
    target_main_cat = target_categories[0] if target_categories else "未知"

    category_counts = Counter()
    same_category_count = 0

    for item in recommendations:
        item_id = str(item["dataset_id"])
        item_tags = item_to_tags.get(item_id, [])
        item_categories = extract_categories(item_tags)

        if item_categories:
            category_counts[item_categories[0]] += 1
            if item_categories[0] == target_main_cat:
                same_category_count += 1

    same_category_ratio = same_category_count / len(recommendations) * 100

    print(f"\n{'='*60}")
    print(f"Dataset ID: {dataset_id}")
    print(f"Target主类别: {target_main_cat}")
    print(f"推荐结果数: {len(recommendations)}")
    print(f"同类别item数: {same_category_count}")
    print(f"同类别比例: {same_category_ratio:.1f}%")
    print(f"\n类别分布:")
    for cat, count in category_counts.most_common():
        print(f"  {cat}: {count} ({count/len(recommendations)*100:.1f}%)")

    return same_category_ratio

def main():
    """批量测试多个dataset"""
    # 随机选择10个dataset进行测试
    test_dataset_ids = [20, 123, 456, 789, 1000, 1234, 2000, 3000, 5000, 7000]

    ratios = []
    for dataset_id in test_dataset_ids:
        try:
            ratio = test_category_focus(dataset_id)
            if ratio is not None:
                ratios.append(ratio)
        except Exception as e:
            print(f"❌ Dataset {dataset_id} 测试失败: {e}")

    if ratios:
        avg_ratio = sum(ratios) / len(ratios)
        print(f"\n{'='*60}")
        print(f"平均同类别比例: {avg_ratio:.1f}%")
        print(f"测试样本数: {len(ratios)}")

        if avg_ratio > 80:
            print("✅ 达到目标！（>80%）")
        elif avg_ratio > 65:
            print("⚠️  接近目标（65-80%），可能需要调优")
        else:
            print("❌ 未达到目标（<65%），需要排查问题")

if __name__ == "__main__":
    main()
```

运行测试：
```bash
python3 scripts/test_category_focus.py
```

### 6. 性能测试

```bash
# 使用wrk进行压测
wrk -t4 -c100 -d30s --latency \
  "http://localhost:8000/api/v1/recommend/detail?dataset_id=123&user_id=456&limit=10"

# 检查P99延迟
# 预期: P99 < 500ms
```

### 7. 监控关键指标

创建`scripts/monitor_recommendation_quality.py`:

```python
"""
监控推荐质量指标
"""
import requests
import json
import time
from collections import defaultdict

API_URL = "http://localhost:8000/api/v1/recommend/detail"

def collect_metrics(num_requests=100):
    """收集推荐质量指标"""
    metrics = {
        "total_requests": 0,
        "successful_requests": 0,
        "same_category_ratios": [],
        "unique_tags_per_request": [],
        "latencies": [],
    }

    with open("models/item_to_tags.json") as f:
        item_to_tags = json.load(f)

    # 模拟多个请求
    test_dataset_ids = list(range(1, 1001, 10))  # 100个dataset

    for dataset_id in test_dataset_ids[:num_requests]:
        start_time = time.time()

        try:
            response = requests.get(API_URL, params={
                "dataset_id": dataset_id,
                "user_id": 123,
                "limit": 10
            }, timeout=2.0)

            latency = (time.time() - start_time) * 1000  # ms
            metrics["latencies"].append(latency)

            if response.status_code == 200:
                metrics["successful_requests"] += 1
                data = response.json()
                recs = data.get("recommendations", [])

                # 计算同类别比例
                target_tags = item_to_tags.get(str(dataset_id), [])
                target_cat = next((t for t in target_tags if t in CATEGORIES), None)

                if target_cat:
                    same_cat_count = sum(
                        1 for r in recs
                        if target_cat in item_to_tags.get(str(r["dataset_id"]), [])
                    )
                    ratio = same_cat_count / len(recs) * 100 if recs else 0
                    metrics["same_category_ratios"].append(ratio)

                # 计算标签多样性
                all_tags = set()
                for r in recs:
                    all_tags.update(item_to_tags.get(str(r["dataset_id"]), []))
                metrics["unique_tags_per_request"].append(len(all_tags))

        except Exception as e:
            print(f"Request {dataset_id} failed: {e}")

        metrics["total_requests"] += 1

    # 汇总统计
    if metrics["same_category_ratios"]:
        avg_same_cat = sum(metrics["same_category_ratios"]) / len(metrics["same_category_ratios"])
        avg_diversity = sum(metrics["unique_tags_per_request"]) / len(metrics["unique_tags_per_request"])
        p99_latency = sorted(metrics["latencies"])[int(len(metrics["latencies"]) * 0.99)]

        print(f"\n{'='*60}")
        print(f"推荐质量监控报告")
        print(f"{'='*60}")
        print(f"总请求数: {metrics['total_requests']}")
        print(f"成功请求数: {metrics['successful_requests']}")
        print(f"成功率: {metrics['successful_requests']/metrics['total_requests']*100:.1f}%")
        print(f"\n平均同类别比例: {avg_same_cat:.1f}%")
        print(f"平均标签多样性: {avg_diversity:.1f} 个唯一tag/请求")
        print(f"P99延迟: {p99_latency:.1f}ms")
        print(f"{'='*60}")

if __name__ == "__main__":
    CATEGORIES = {"金融", "医疗健康", "政府政务", "交通运输",
                  "教育培训", "能源环保", "农业", "科技",
                  "商业零售", "文化娱乐", "社会民生"}
    collect_metrics(num_requests=100)
```

### 8. A/B测试配置（可选）

如果需要分流测试，修改`config/experiments.yaml`:

```yaml
experiments:
  recommendation_detail:
    status: active
    salt: "recommend-category-v1"
    variants:
      - name: control
        allocation: 0.9  # 90%流量走原逻辑
        parameters: {}
      - name: category_enhanced
        allocation: 0.1  # 10%流量走新逻辑
        parameters:
          use_enhanced_tags: true
          enable_category_features: true
```

## 验证清单

- [ ] 服务启动无报错
- [ ] /health接口返回healthy
- [ ] 推荐API正常响应（10个测试请求全部成功）
- [ ] 平均同类别比例 > 80%
- [ ] 标签多样性 > 20个唯一tag/请求
- [ ] P99延迟 < 500ms
- [ ] 无明显异常日志

## 回滚方案

如果出现问题，快速回滚：

```bash
# 停止服务
docker-compose stop recommendation-api

# 恢复原模型
cp models/backup/$(date +%Y%m%d)/rank_model.pkl models/
cp models/backup/$(date +%Y%m%d)/tag_to_items.json models/
cp models/backup/$(date +%Y%m%d)/item_to_tags.json models/

# 重启服务
docker-compose up -d recommendation-api

echo "已回滚到原模型"
```

## 后续优化建议

1. **持续监控7天**
   - 类别聚焦度趋势
   - CTR/转化率变化
   - 用户反馈

2. **A/B测试分析**
   - 对比control vs category_enhanced
   - 统计显著性检验
   - 分用户群体分析

3. **模型迭代**
   - 根据线上表现调整特征权重
   - 优化大类定义（增加/合并类别）
   - 尝试更复杂的类别相关性特征

## 任务完成标志

✅ 所有子任务（01-07）完成
✅ 推荐结果同类别比例 > 80%
✅ 性能无退化
✅ 代码已提交到feature分支
✅ 准备合并到master

**恭喜！类别增强任务全部完成！** 🎉
