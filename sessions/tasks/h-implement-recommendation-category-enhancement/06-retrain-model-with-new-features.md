# 子任务06: 重新训练模型

## 目标
使用增强后的tags数据和新增的类别特征重新训练LightGBM Ranker模型。

## 成功标准
- [ ] build_features_v2.py成功运行（使用enhanced tags）
- [ ] train_models.py成功训练（包含3个新特征）
- [ ] 新模型的NDCG@10不低于原模型
- [ ] 类别特征重要性排名进入Top 15
- [ ] 模型文件正常保存到models/目录

## 实施步骤

### 1. 确保使用增强后的tags数据

在`pipeline/build_features_v2.py`中，确认数据源指向增强版本：

检查Line 220附近（数据加载位置）：
```python
# 修改前（如果有）:
# dataset_features = pd.read_parquet(DATA_DIR / "cleaned" / "dataset_features.parquet")

# 修改后:
dataset_features = pd.read_parquet(DATA_DIR / "processed" / "dataset_features_enhanced.parquet")

# 确保使用enhanced_tags列
if "enhanced_tags" in dataset_features.columns:
    dataset_features["tag"] = dataset_features["enhanced_tags"]  # 覆盖原tag列
```

或者，更安全的做法是修改pipeline/train_models.py中的数据加载：

```python
# 在train_models.py的main()函数中（Line 1654-1658）
# 修改前:
# dataset_features = _load_frame(PROCESSED_DIR / "dataset_features.parquet")

# 修改后（优先加载enhanced版本）:
enhanced_path = PROCESSED_DIR / "dataset_features_enhanced.parquet"
if enhanced_path.exists():
    dataset_features = _load_frame(enhanced_path)
    # 使用enhanced_tags列
    if "enhanced_tags" in dataset_features.columns:
        dataset_features["tag"] = dataset_features["enhanced_tags"]
    LOGGER.info("Using enhanced tags from dataset_features_enhanced.parquet")
else:
    dataset_features = _load_frame(PROCESSED_DIR / "dataset_features.parquet")
    LOGGER.warning("Enhanced tags file not found, using original tags")
```

### 2. 运行特征构建

```bash
cd /home/ubuntu/recommend

# 运行build_features_v2（约5-10分钟）
python3 -m pipeline.build_features_v2

# 检查输出
ls -lh data/processed/dataset_features_v2.parquet
ls -lh data/processed/user_features_v2.parquet
```

预期日志：
```
[INFO] Loading dataset features...
[INFO] Using enhanced tags from dataset_features_enhanced.parquet
[INFO] Building dataset-level features...
[INFO] Building user-level features...
[INFO] Feature engineering complete: 75+ features generated
```

### 3. 运行模型训练

```bash
# 后台运行训练（约30-60分钟，取决于数据量）
nohup python3 -m pipeline.train_models > logs/train_models.log 2>&1 &

# 记录进程ID
echo $! > logs/train_models.pid
```

### 4. 监控训练进度

```bash
# 实时查看日志
tail -f logs/train_models.log

# 关键日志点:
# - "Computing category relevance features..." (约在10%进度)
# - "Training LightGBM Ranker..." (约在50%进度)
# - "Feature importance saved" (约在90%进度)
```

预期日志示例：
```
[INFO] Loading interactions data...
[INFO] Loading dataset features...
[INFO] Using enhanced tags from dataset_features_enhanced.parquet
[INFO] Preparing ranking dataset...
[INFO] Computing category relevance features...
[INFO] Added category features: tag_overlap_count, tag_jaccard_similarity, same_top_category
[INFO] Final feature count: 78 (75 original + 3 category)
[INFO] Training LightGBM Ranker...
[INFO] Training completed. Best iteration: 245
[INFO] Validation NDCG@10: 0.7623
[INFO] Feature importance (top 10):
  1. weight_log: 1245.3
  2. interaction_count: 1123.5
  3. tag_jaccard_similarity: 892.1  # 新特征！
  4. popularity_rank: 765.4
  5. same_top_category: 654.2  # 新特征！
  ...
[INFO] Model saved to models/rank_model.pkl
```

### 5. 验证模型输出

```bash
# 检查模型文件
ls -lh models/rank_model.pkl
ls -lh models/item_sim_behavior.pkl
ls -lh models/item_sim_content.pkl
ls -lh models/top_items.json

# 检查模型元数据
python3 -c "
import pickle
with open('models/rank_model.pkl', 'rb') as f:
    model = pickle.load(f)
    print('Model type:', type(model))
    if hasattr(model, 'feature_names_in_'):
        print('Features:', model.feature_names_in_)
"
```

### 6. 特征重要性分析

创建`scripts/analyze_feature_importance.py`:

```python
import pickle
import json
import pandas as pd

# 加载模型
with open('models/rank_model.pkl', 'rb') as f:
    pipeline = pickle.load(f)

# 提取LightGBM模型
if hasattr(pipeline, 'named_steps'):
    ranker = pipeline.named_steps['ranker']
else:
    ranker = pipeline

# 获取特征重要性
if hasattr(ranker, 'feature_importances_'):
    importance = ranker.feature_importances_
    feature_names = ranker.feature_name_

    # 创建DataFrame
    df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)

    print("\n=== Top 20 Feature Importance ===")
    print(df.head(20).to_string(index=False))

    # 检查类别特征排名
    category_features = ['tag_overlap_count', 'tag_jaccard_similarity', 'same_top_category']
    print("\n=== Category Features Ranking ===")
    for feat in category_features:
        if feat in df['feature'].values:
            rank = df[df['feature'] == feat].index[0] + 1
            imp = df[df['feature'] == feat]['importance'].values[0]
            print(f"  {feat}: Rank {rank}, Importance {imp:.2f}")
        else:
            print(f"  {feat}: NOT FOUND (ERROR!)")

    # 保存到JSON
    importance_dict = df.to_dict('records')
    with open('models/feature_importance.json', 'w') as f:
        json.dump(importance_dict, f, indent=2)
    print("\n特征重要性已保存到 models/feature_importance.json")
```

运行分析：
```bash
python3 scripts/analyze_feature_importance.py
```

## 成功验证清单

- [ ] 训练日志中包含"Added category features"
- [ ] 最终feature count = 78 (75 + 3)
- [ ] NDCG@10 >= 原模型NDCG（需要baseline对比）
- [ ] 至少1个类别特征进入Top 15
- [ ] models/rank_model.pkl文件存在且大小合理（约5-50MB）

## 对比分析

### 与原模型对比

```bash
# 备份原模型
cp models/rank_model.pkl models/rank_model_original.pkl

# 对比NDCG
# 查看原模型训练日志中的NDCG@10
# 查看新模型训练日志中的NDCG@10

# 预期：新模型NDCG@10应不降低，甚至提升1-3%
```

### 特征贡献分析

```python
# 计算类别特征的总贡献度
category_total = sum(importance for feat, importance in zip(feature_names, importances)
                     if feat in ['tag_overlap_count', 'tag_jaccard_similarity', 'same_top_category'])

total_importance = sum(importances)
category_contribution = category_total / total_importance * 100

print(f"类别特征总贡献度: {category_contribution:.2f}%")

# 预期：类别特征贡献度 > 5%
```

## 故障排查

### 问题1: 特征计算太慢（apply卡住）
**症状**: "Computing category relevance features"后长时间无输出
**解决**:
- 正常现象（apply()慢），等待即可
- 或临时减少训练数据量进行测试

### 问题2: KeyError: 'target_tag'
**症状**: 训练时报target_tag列不存在
**解决**: 检查_prepare_ranking_dataset中的merge逻辑，确保target_items正确生成

### 问题3: 新特征重要性为0
**症状**: 类别特征importance = 0.0
**解决**:
- 检查特征值是否全为0（可能target context未正确传递）
- 检查是否有足够的变化（variance > 0）

### 问题4: NDCG显著下降（>5%）
**症状**: 新模型NDCG比原模型低很多
**解决**:
- 检查特征计算逻辑是否有bug
- 检查enhanced_tags质量
- 可能需要调整LightGBM超参数

## 下一步
完成后进入子任务07：部署并验证效果
