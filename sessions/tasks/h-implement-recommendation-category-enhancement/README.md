---
name: h-implement-recommendation-category-enhancement
branch: feature/recommendation-category-enhancement
status: pending
created: 2025-12-30
---

# 推荐系统类别增强：Tags自动化 + 类别相关性特征

## Problem/Goal

当前推荐系统存在**类别跨度过大**的问题：推荐结果中包含金融、医疗、农业等差异很大的类别，用户期望看到更聚焦的推荐。

核心问题：
1. **用户标签质量差**：tags由用户自己打的，不准确且缺失严重
2. **缺乏类别约束**：推荐流程中没有大类过滤机制
3. **排序模型未利用类别信息**：LightGBM Ranker缺少类别相关性特征

解决方案：
1. **Tags自动增强**：使用零样本分类模型（方案C-1）从description提取标准化大类标签
2. **特征工程**：为Ranker添加类别相关性特征（Jaccard相似度、同大类标识等）
3. **模型重训练**：让排序模型学习类别相关性权重

## Success Criteria
- [ ] 完成全量item的tags自动增强，覆盖率>95%
- [ ] 生成增强后的索引文件（enhanced_tags.json、tag_to_items_enhanced.json）
- [ ] 在build_features_v2.py中成功添加4+个类别相关性特征
- [ ] 重新训练的LightGBM模型中类别特征重要性排名进入Top 10
- [ ] 推荐结果中同大类item比例从当前<50%提升到>80%
- [ ] A/B测试显示点击率或转化率有提升（或至少不下降）

## Context Manifest
<!-- Added by context-gathering agent -->

## User Notes

### 技术选型
- **方案C-1**: CPU + 轻量级模型（Erlangshen-Roberta-110M-NLI）
- **镜像站**: 使用 hf-mirror.com 加速模型下载
- **处理方式**: 离线批量增强（预计20-30分钟处理1万个item）

### 候选大类定义
```python
CATEGORIES = [
    "金融", "医疗健康", "政府政务", "交通运输",
    "教育培训", "能源环保", "农业", "科技",
    "商业零售", "文化娱乐", "社会民生"
]
```

### Description质量
用户确认description质量较高，适合用于NLP提取

## Work Log
- [2025-12-30] 任务创建，完成技术方案讨论和选型
