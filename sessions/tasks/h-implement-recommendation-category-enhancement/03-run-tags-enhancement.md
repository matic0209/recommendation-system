# 子任务03: 执行全量Tags增强

## 目标
在全量dataset_features数据上执行tags增强（预计10000个items，耗时20-30分钟）。

## 成功标准
- [ ] 全量数据处理完成（10000+ items）
- [ ] Tags覆盖率 > 95%
- [ ] 生成的3个文件完整无损
- [ ] 增强后tags质量抽查通过

## 实施步骤

### 1. 环境准备
```bash
# 确保在正确目录
cd /home/ubuntu/recommend

# 确认HF镜像配置
export HF_ENDPOINT=https://hf-mirror.com

# 确认数据文件存在
ls -lh data/cleaned/dataset_features.parquet
```

### 2. 执行增强（后台运行）
```bash
# 使用nohup后台运行，输出到日志
nohup python3 -m pipeline.enhance_tags > logs/enhance_tags.log 2>&1 &

# 记录进程ID
echo $! > logs/enhance_tags.pid
```

### 3. 监控进度
```bash
# 实时查看日志
tail -f logs/enhance_tags.log

# 查看进度（另一个终端）
watch -n 10 'tail -20 logs/enhance_tags.log | grep "增强Tags"'

# 检查checkpoint（每100个item保存一次）
ls -lh data/processed/enhance_tags_checkpoint.json
```

### 4. 预期输出
日志示例：
```
2025-12-30 10:00:00 - INFO - === Tags增强流程开始 ===
2025-12-30 10:00:01 - INFO - 加载数据: data/cleaned/dataset_features.parquet
2025-12-30 10:00:02 - INFO - 加载完成: 12948 行数据
2025-12-30 10:00:02 - INFO - 加载Erlangshen-Roberta-110M-NLI模型...
2025-12-30 10:00:15 - INFO - 模型加载成功！
2025-12-30 10:00:15 - INFO - 总计 12948 个items，已处理 0 个
2025-12-30 10:00:15 - INFO - 剩余 12948 个items待处理
增强Tags:   1%|▏         | 100/12948 [01:23<2:58:45,  1.20it/s]
增强Tags:   5%|▌         | 500/12948 [07:15<2:51:30,  1.21it/s]
...
增强Tags: 100%|██████████| 12948/12948 [25:47<00:00,  8.36it/s]
2025-12-30 10:26:02 - INFO - Tags增强完成，共处理 12948 个items
2025-12-30 10:26:03 - INFO - === 增强统计 ===
2025-12-30 10:26:03 - INFO - 总item数: 12948
2025-12-30 10:26:03 - INFO - 有tags的item数: 12450
2025-12-30 10:26:03 - INFO - Tags覆盖率: 96.15%
2025-12-30 10:26:03 - INFO - 平均tags数/item: 3.42
2025-12-30 10:26:03 - INFO - 唯一tag数: 1284
2025-12-30 10:26:03 - INFO - === Tags增强流程完成 ===
```

## 验证步骤

### 1. 检查输出文件
```bash
# 验证文件存在且大小合理
ls -lh data/processed/dataset_features_enhanced.parquet
ls -lh models/tag_to_items_enhanced.json
ls -lh models/item_to_tags_enhanced.json

# 预期大小
# dataset_features_enhanced.parquet: ~5-10MB
# tag_to_items_enhanced.json: ~500KB-2MB
# item_to_tags_enhanced.json: ~1-3MB
```

### 2. 验证数据完整性
```python
import pandas as pd
import json

# 读取增强后的数据
df = pd.read_parquet("data/processed/dataset_features_enhanced.parquet")

# 验证行数一致
assert len(df) == 12948, f"行数不匹配: {len(df)}"

# 验证enhanced_tags列存在
assert "enhanced_tags" in df.columns

# 验证覆盖率
coverage = (df["enhanced_tags"].str.len() > 0).sum() / len(df) * 100
print(f"Tags覆盖率: {coverage:.2f}%")
assert coverage > 95, f"覆盖率过低: {coverage:.2f}%"

# 验证索引文件
with open("models/tag_to_items_enhanced.json") as f:
    tag_to_items = json.load(f)
    print(f"唯一tag数: {len(tag_to_items)}")
    assert len(tag_to_items) > 100, "tag数量过少"

with open("models/item_to_tags_enhanced.json") as f:
    item_to_tags = json.load(f)
    print(f"Item数: {len(item_to_tags)}")
    assert len(item_to_tags) == len(df), "item数不匹配"

print("✅ 数据验证通过！")
```

### 3. 质量抽查
```python
# 抽查10个item的增强效果
import pandas as pd
import random

df = pd.read_parquet("data/processed/dataset_features_enhanced.parquet")

# 随机抽取10个有description的item
samples = df[df["description"].str.len() > 50].sample(10)

for idx, row in samples.iterrows():
    print(f"\n{'='*60}")
    print(f"Dataset ID: {row['dataset_id']}")
    print(f"Description: {row['description'][:200]}...")
    print(f"原始Tags: {row.get('tag', 'N/A')}")
    print(f"增强Tags: {row.get('enhanced_tags', 'N/A')}")
```

### 4. 大类分布统计
```python
import json
from collections import Counter

# 统计大类分布
with open("models/tag_to_items_enhanced.json") as f:
    tag_to_items = json.load(f)

# 定义大类
CATEGORIES = ["金融", "医疗健康", "政府政务", "交通运输", "教育培训",
              "能源环保", "农业", "科技", "商业零售", "文化娱乐", "社会民生"]

# 统计每个大类的item数
category_stats = {}
for cat in CATEGORIES:
    items = tag_to_items.get(cat, [])
    category_stats[cat] = len(items)

print("\n大类分布:")
for cat, count in sorted(category_stats.items(), key=lambda x: x[1], reverse=True):
    print(f"  {cat}: {count} items")
```

## 故障排查

### 问题1: 进程被杀（OOM）
**症状**: 日志中断，进程不存在
**解决**:
```bash
# 检查内存使用
free -h

# 限制batch size（在enhance_tags.py中）
# 或分批处理数据
```

### 问题2: 模型推理太慢
**症状**: 处理速度 < 0.5 it/s
**解决**: 这是正常的CPU推理速度，继续等待即可

### 问题3: checkpoint文件损坏
**症状**: 重启后报JSON解析错误
**解决**:
```bash
# 删除损坏的checkpoint，从头开始
rm data/processed/enhance_tags_checkpoint.json
```

## 下一步
完成后进入子任务04：添加类别特征（训练阶段）
