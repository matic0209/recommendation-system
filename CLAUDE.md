
@sessions/CLAUDE.sessions.md

# Lamoom Python Project Guide

## Build/Test/Lint Commands
- Install deps: `poetry install`
- Run all tests: `poetry run pytest --cache-clear -vv tests`
- Run specific test: `poetry run pytest tests/path/to/test_file.py::test_function_name -v`
- Run with coverage: `make test`
- Format code: `make format` (runs black, isort, flake8, mypy)
- Individual formatting:
  - Black: `make make-black`
  - isort: `make make-isort`
  - Flake8: `make flake8`
  - Autopep8: `make autopep8`

## Code Style Guidelines
- Python 3.9+ compatible code
- Type hints required for all functions and methods
- Classes: PascalCase with descriptive names
- Functions/Variables: snake_case
- Constants: UPPERCASE_WITH_UNDERSCORES
- Imports organization with isort:
  1. Standard library imports
  2. Third-party imports
  3. Local application imports
- Error handling: Use specific exception types
- Logging: Use the logging module with appropriate levels
- Use dataclasses for structured data when applicable

## Project Conventions
- Use poetry for dependency management
- Add tests for all new functionality
- Maintain >80% test coverage (current min: 81%)
- Follow pre-commit hooks guidelines
- Document public APIs with docstrings

## Language
- 所有的回答和对话一定用中文！！

## Recommendation Service Architecture

### 推荐流程概览
推荐系统遵循多阶段流水线架构（参考文件: /home/ubuntu/recommend/app/main.py）：

1. **多渠道召回** (Recall) - 从不同来源收集候选item
2. **分数融合** (Score Fusion) - 归一化并合并多渠道分数
3. **个性化调整** (Personalization) - 基于用户历史行为
4. **排序** (Ranking) - LightGBM ranker模型打分
5. **质量过滤** (Quality Filtering) - 负分硬截断
6. **MMR重排** (MMR Reranking) - 多样性优化
7. **探索** (Exploration) - epsilon-greedy策略

### 召回渠道 (Recall Channels)

#### 1. Behavior召回
- 基于用户协同过滤 (User-CF) 和item-item相似度
- 原始分数范围: [0, 1] (余弦或Jaccard相似度)
- 归一化: Min-Max scaling
- 默认权重: 1.2

#### 2. Content召回
- 基于标签/描述的TF-IDF相似度
- 原始分数范围: [0.2, 0.9]
- 归一化: Min-Max scaling
- 默认权重: 1.0

#### 3. Vector召回
- SBERT语义向量相似度
- 原始分数范围: [15, 22] (未归一化)
- 归一化: Min-Max scaling
- 默认权重: 0.8

#### 4. Popular召回 (带质量过滤)
- 全局热门榜单 (models/top_items.json)
- 原始分数: 线性衰减 (第1名=1.0, 最后=0.1)
- 默认权重: 0.1
- **质量过滤规则** (2025-12-27新增):
  - 低价且无人气: price < 1.90 AND interaction_count < 66
  - 长期不活跃且交互少: days_inactive > 180 AND interaction_count < 30
- 实现位置: /home/ubuntu/recommend/app/main.py Line 1688-1750

### 排序与质量控制 (Ranking & Quality Control)

#### LightGBM Ranker
- 模型类型: LambdaRank (objective="lambdarank", metric="ndcg")
- 输出范围: (-∞, +∞) 原始预测分数 (非归一化)
- 负分含义: 模型预测该item质量低于平均水平

#### 负分硬截断机制 (2025-12-27新增)
- **目的**: 过滤排序后分数为负的低质量item
- **策略**: 直接移除score < 0的item
- **Fallback**: 如果全部为负分，保留分数最高的50%（至少5个）
- **日志**: 记录负分比例和过滤数量
- 实现位置: /home/ubuntu/recommend/app/main.py Line 2354-2378

#### Tag召回修复 (2025-12-27)
- **问题**: 标签大小写不统一导致overlap计算错误
- **修复**: 统一使用lowercase处理target和candidate tags
- 实现位置: /home/ubuntu/recommend/app/main.py Line 1506-1507

### 重要配置参数

#### 渠道权重 (DEFAULT_CHANNEL_WEIGHTS)
参考: /home/ubuntu/recommend/app/main.py Line 183-188
```python
{
    "behavior": 1.2,  # 用户协同过滤
    "content": 1.0,   # 内容相似度
    "vector": 0.8,    # 语义向量
    "popular": 0.1,   # 全局热门
}
```

#### MMR参数
- lambda_param: 0.7 (70%相关性 + 30%多样性)
- 基于Jaccard标签相似度计算多样性

#### 探索参数
- epsilon: 0.15 (15%探索率)
- 策略: epsilon-greedy随机采样

### 故障排查

#### 问题: 推荐结果中出现大量负分
- **原因**: LightGBM ranker输出未归一化分数，负分表示质量低于平均
- **解决**: 负分硬截断机制已自动过滤 (2025-12-27修复)
- **监控**: 查看日志中的"Negative score filter"记录

#### 问题: Popular召回质量差
- **原因**: 全局热门榜单缺乏上下文相关性
- **解决**: 质量过滤规则已添加 (2025-12-27修复)
- **监控**: 查看日志中的"Popular recall quality filter"记录

#### 问题: Score展示顺序混乱
- **原因**: MMR重排后展示原始分数（非MMR分数）
- **说明**: 这是预期行为，MMR优化多样性，展示顺序与原始分数可能不一致
- **影响**: 前端不展示score字段，用户不可见