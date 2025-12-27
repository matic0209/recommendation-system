---
name: h-fix-negative-scores-and-mmr-display
branch: fix/negative-scores-mmr-display
status: pending
created: 2025-12-27
---

# 修复推荐结果负分和MMR展示混乱问题

## Problem/Goal

生产环境发现推荐结果存在严重的质量和用户体验问题：

**核心问题**：
1. **排序混乱**：MMR重排后，展示的score是原始召回分数，导致正分item排在负分后面（如：score=0.33排在score=-0.78后面）
2. **大量负分推荐**：30个推荐中有16个负分（53%），range从-0.32到-2.89
3. **Popular召回质量差**：大量不相关的鼠标指针、桌宠类item被推荐（score在-2.5~-2.8）
4. **用户体验差**：看到负分和混乱的排序，失去对推荐系统的信任

**影响**：
- 用户看到score不符合降序排列，感到困惑
- 负分item暗示质量差，影响点击意愿
- Popular召回的低质量item占据宝贵的推荐位
- 可能导致CTR下降和用户满意度降低

**目标**：
1. 立即修复score展示混乱问题
2. 过滤或优化负分推荐
3. 改进Popular召回策略

## Success Criteria

### 技术指标
- [ ] 返回的推荐结果score严格降序排列（exploration除外）
- [ ] 负分推荐占比从53%降到<10%（或完全过滤）
- [ ] Popular召回的item与当前场景相关度提升
- [ ] score字段语义清晰（要么是MMR分数，要么是排序位置分数）

### 代码质量
- [ ] 修改后API响应格式保持兼容
- [ ] 本地测试验证score展示正确
- [ ] 性能无退化（P99延迟<500ms）

### 业务指标（A/B测试验证）
- [ ] CTR不下降（理想情况提升5-10%）
- [ ] 用户对推荐结果的点击分布更均匀
- [ ] 负面反馈减少

## Implementation Plan

### 模块1：修复MMR分数展示混乱（P0）

**问题分析**：
```python
# 当前流程
1. 召回+排序 → candidate_scores = {id: score}  # 例如 {111: 0.33, 13185: -0.78}
2. MMR重排 → ranked_ids = [13185, 111, ...]    # MMR认为13185多样性好
3. 构建响应 → score = candidate_scores[id]     # 展示原始score，乱序！
```

**解决方案A：展示序号分数（推荐）**
```python
# 在 _build_response_items 中
if apply_mmr:
    # 使用位置序号生成递减分数
    score = 1.0 - (idx / len(ranked_ids)) * 0.5  # 1.0 → 0.5
```

**解决方案B：返回实际MMR分数**
```python
# 修改 _apply_mmr_reranking 同时返回MMR分数
return selected, mmr_scores_dict
# 在 _build_response_items 中使用MMR分数
```

**选择**：方案A更简单，方案B更精确。建议先实施A，观察效果。

---

### 模块2：过滤负分推荐（P0）

**问题分析**：
- LightGBM ranker输出负分，说明模型认为item质量差或与场景不匹配
- 负分item的reason分析：
  - Popular渠道：鼠标指针、桌宠（-2.5~-2.8）
  - Behavior/UserCF：协同过滤召回的低质量item（-0.3~-2.8）

**解决方案A：硬截断（推荐，快速见效）**
```python
# 在 _apply_ranking_with_circuit_breaker 后过滤
scores = {k: v for k, v in scores.items() if v >= 0}
```

**解决方案B：软截断（保留但降权）**
```python
# 在 _apply_ranking_with_circuit_breaker 中
scores[dataset_id] = max(0.0, scores[dataset_id] + prob * freshness_boost)
```

**选择**：先实施B（软截断），如果负分仍然出现，再启用A（硬截断）。

---

### 模块3：优化Popular召回策略（P1）

**问题分析**：
```json
负分Popular items:
- "三丽鸥小羊piano静态鼠标指针" score=-2.57
- "无限暖暖指针皮肤" score=-2.72
- "罗小黑键鼠桌宠" score=-2.74
```
这些item与用户浏览的dataset无关联，pure噪声。

**可能原因**：
1. Popular召回没有场景过滤（全局热门，不考虑上下文）
2. Popular权重过高（阶段2提升到0.1）
3. 这些item本身是低质量内容，被排序模型重度惩罚

**解决方案**：
```python
# 选项1：添加类别过滤
if popular_scores:
    # 只保留与target同类别的popular items
    filtered_popular = filter_by_category(popular_scores, target_category)

# 选项2：降低Popular权重
DEFAULT_CHANNEL_WEIGHTS["popular"] = 0.05  # 从0.1降回0.05

# 选项3：移除低质量item
# 在数据层面清理，标记鼠标指针、桌宠类为"utility"而非推荐候选
```

**选择**：短期降低权重（选项2），长期添加类别过滤（选项1）。

---

### 模块4：调整探索机制展示（P2）

**当前问题**：
- 探索item固定score=0.5
- 排在所有负分后面，视觉上也很奇怪

**优化方案**：
```python
# 方案1：探索item使用随机分数
score = random.uniform(0.3, 0.7)

# 方案2：探索item插入到中间位置
# 例如：前50%确定性，后50%混合探索
```

---

## Technical Details

### 关键文件
- `app/main.py`
  - `_apply_mmr_reranking()` - MMR重排逻辑（line 1138）
  - `_build_response_items()` - 构建API响应（line 1246）
  - `_apply_ranking_with_circuit_breaker()` - 排序分数应用（line 2217）
  - `DEFAULT_CHANNEL_WEIGHTS` - 渠道权重配置（line 183）

### 数据分析
从生产环境案例（dataset_id=13003, user_id=1997）：
```
总推荐数：30
正分：10个（2.057 ~ 0.424）
负分：16个（-0.320 ~ -2.887）- 53% 🔴
探索：4个（固定0.5）- 13%

负分来源分布：
- Popular: 6个（-2.57 ~ -2.83）
- Behavior/UserCF: 7个（-0.32 ~ -2.89）
- Content+Vector: 2个（-0.78, -2.48）
- Price: 1个（-2.79）
```

### 依赖关系
- 模块1和2无依赖，可并行实施
- 模块3需要观察模块2效果后决定
- 模块4优先级最低，可延后

## Work Log
- [2025-12-27] 任务创建，问题诊断完成
  - 分析生产环境推荐结果，发现53%负分问题
  - 识别MMR分数展示混乱导致用户困惑
  - 确定Popular召回质量问题
  - 制定P0/P1/P2优先级修复方案
