---
index: recommendation-system
name: 推荐系统优化
description: 推荐算法改进、召回策略优化、排序模型调优、多样性增强、新鲜度机制等推荐系统相关任务
---

# 推荐系统优化

## Active Tasks

### High Priority
- `h-fix-recommendation-diversity/` - 修复推荐结果重复和分数断崖问题：召回归一化、探索机制、缓存优化、MMR多样性、新鲜度加权
- `h-fix-negative-scores-and-mmr-display/` - 修复推荐结果负分和MMR展示混乱问题：过滤负分、修复score展示、优化Popular召回
- `h-implement-recommendation-category-enhancement/` - 推荐类别增强：使用零样本分类模型自动增强tags、添加类别相关性特征、重训练Ranker模型

### Medium Priority
<!-- 中优先级任务将在此添加 -->

### Low Priority
<!-- 低优先级任务将在此添加 -->

### Investigate
<!-- 调研类任务将在此添加 -->

## Completed Tasks
- `m-implement-popular-quality-filter.md` - 为Popular召回榜单实现质量过滤机制：训练时过滤低质量item、Sentry监控告警、环境变量可配置 (2025-12-28)
