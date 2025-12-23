#!/bin/bash
# 对比不同 PCA 维度的模型效果

echo "=========================================="
echo "PCA 维度对比实验"
echo "=========================================="

# 测试不同维度
dims=(5 10 15 20 30)

for dim in "${dims[@]}"; do
    echo ""
    echo ">>> 测试 PCA 维度: $dim"

    # 修改配置
    sed -i "s/^TEXT_PCA_COMPONENTS=.*/TEXT_PCA_COMPONENTS=$dim/" .env

    # 训练模型
    echo "训练模型..."
    PYTHONPATH=/home/ubuntu/recommend python3 -m pipeline.train_models

    # 记录结果
    echo "PCA_DIM=$dim 训练完成"

    # 提取验证集指标
    if [ -f "data/processed/ranking_model_metrics.json" ]; then
        cat data/processed/ranking_model_metrics.json
    fi
done

echo ""
echo "=========================================="
echo "对比完成！查看各维度的 AUC 和 LogLoss"
echo "=========================================="
