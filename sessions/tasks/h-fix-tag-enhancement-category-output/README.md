---
name: h-fix-tag-enhancement-category-output
branch: fix/tag-enhancement-category-output
status: pending
created: 2025-01-02
---

# 修复标签增强输出：分离12类别与原始标签

## Problem/Goal

330M零样本分类任务已成功为8137个数据集生成了12类标准分类（如：政府政务、金融财经、医疗健康等），但这些类别没有正确保存到推荐系统使用的索引文件中。

**当前问题**：
1. `item_to_tags_enhanced.json` 存储的是原始用户标签，不是12类别标准分类
2. `build_tag_indices` 函数使用 `tag_enhanced` 列（包含合并后的标签），而非纯净的 `new_categories` 列
3. 导致基于类别的推荐匹配率为0%

**根因分析**：
- `enhance_tags.py:362-363` 保存了两个列：
  - `tag_enhanced` = 原始标签 + 新类别（合并）
  - `new_categories` = 纯12类别列表
- `enhance_tags.py:417` 调用 `build_tag_indices(df)` 使用默认的 `tag_enhanced` 列
- `enhance_tags.py:244` 解析时将所有标签混在一起，导致12类别被原始标签稀释

## Success Criteria
- [ ] 创建新的12类别专用索引文件 `item_to_categories.json`
- [ ] 修改 `build_tag_indices` 或新建函数从 `new_categories` 列生成类别索引
- [ ] API推荐服务加载并使用新的类别索引
- [ ] 类别匹配率从0%提升到预期水平（>50%有类别的item应匹配）
- [ ] 重新运行训练流程验证类别特征生效

## Context Manifest

### 核心文件

| 文件 | 作用 | 关键行号 |
|------|------|----------|
| `pipeline/enhance_tags.py` | 标签增强主脚本 | `build_tag_indices`: 220-255, `save_outputs`: 406-428 |
| `app/main.py` | API服务 | `_load_recall_artifacts`: 716-751, Tag召回: 1506-1528 |
| `pipeline/train_models.py` | 模型训练 | 加载item_to_tags_enhanced: 1235-1248 |

### 数据文件

| 文件 | 说明 |
|------|------|
| `data/processed/dataset_features_enhanced.parquet` | 包含`new_categories`列的正确12类别数据 |
| `models/item_to_tags_enhanced.json` | 当前输出（包含混合标签，需修复） |
| `models/tag_to_items_enhanced.json` | 当前输出（包含混合标签，需修复） |

### 12类别定义
政府政务、金融财经、医疗健康、交通物流、教育科研、工业制造、商业零售、能源环保、文化娱乐、农业农村、互联网科技、社会民生

### 推荐修复方案
新增 `build_category_indices` 函数，从 `new_categories` 列构建纯净的类别索引：
- `models/item_to_categories.json` - item到12类别的映射
- `models/category_to_items.json` - 12类别到items的映射

## User Notes
<!-- Any specific notes or requirements from the developer -->

## Work Log
<!-- Updated as work progresses -->
- [2025-01-02] 创建任务，问题诊断完成
