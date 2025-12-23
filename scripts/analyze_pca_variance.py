#!/usr/bin/env python3
"""
分析 SBERT embeddings 的 PCA 降维效果
帮助选择最优的 PCA 维度数量
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer

# 设置 HuggingFace 镜像
if os.getenv("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = os.getenv("HF_ENDPOINT")

def analyze_pca_components(max_components: int = 50):
    """
    分析不同 PCA 维度下的方差解释比例

    Args:
        max_components: 最大分析的主成分数量（默认50）
    """
    print("=" * 70)
    print("SBERT Embeddings PCA 降维分析")
    print("=" * 70)

    # 1. 加载数据
    print("\n📊 Step 1: 加载数据集特征...")
    dataset_features_path = "data/processed/dataset_features.parquet"
    if not os.path.exists(dataset_features_path):
        print(f"❌ 文件不存在: {dataset_features_path}")
        print("   请先运行训练流程生成 dataset_features.parquet")
        sys.exit(1)

    df = pd.read_parquet(dataset_features_path)
    print(f"   ✓ 加载了 {len(df)} 个数据集")

    # 2. 生成文本 embeddings
    print("\n🤖 Step 2: 生成 SBERT embeddings...")
    model_name = os.getenv("SBERT_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
    print(f"   模型: {model_name}")

    try:
        if not model_name.startswith("sentence-transformers/"):
            full_model_name = f"sentence-transformers/{model_name}"
        else:
            full_model_name = model_name
        model = SentenceTransformer(full_model_name, device='cpu')
    except Exception as e:
        print(f"   尝试不带前缀下载: {e}")
        model = SentenceTransformer(model_name, device='cpu')

    texts = (df["description"].fillna("") + " " + df["tag"].fillna("")).values
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
    print(f"   ✓ Embeddings shape: {embeddings.shape}")

    # 3. PCA 分析
    print(f"\n🔍 Step 3: PCA 降维分析 (max_components={max_components})...")
    pca = PCA(n_components=min(max_components, embeddings.shape[1]), random_state=42)
    pca.fit(embeddings)

    # 4. 计算累积解释方差
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

    # 5. 输出关键维度的方差解释比例
    print("\n" + "=" * 70)
    print("📈 方差解释比例分析")
    print("=" * 70)
    print(f"{'维度':<8} {'单个方差%':<12} {'累积方差%':<12} {'说明'}")
    print("-" * 70)

    key_dims = [1, 2, 3, 5, 10, 15, 20, 30, 40, 50]
    for n in key_dims:
        if n <= len(cumulative_variance_ratio):
            variance = explained_variance_ratio[n-1] * 100
            cumulative = cumulative_variance_ratio[n-1] * 100

            # 添加说明
            if cumulative >= 95:
                note = "✓ 优秀（>=95%）"
            elif cumulative >= 90:
                note = "✓ 良好（>=90%）"
            elif cumulative >= 80:
                note = "○ 可接受（>=80%）"
            else:
                note = "△ 信息损失较大"

            print(f"{n:<8} {variance:>10.2f}% {cumulative:>10.2f}% {note}")

    # 6. 推荐维度
    print("\n" + "=" * 70)
    print("💡 推荐配置")
    print("=" * 70)

    # 找到达到不同阈值的最小维度
    thresholds = [0.80, 0.85, 0.90, 0.95]
    for threshold in thresholds:
        n_components = np.argmax(cumulative_variance_ratio >= threshold) + 1
        if cumulative_variance_ratio[n_components-1] >= threshold:
            print(f"保留 {threshold*100:.0f}% 方差: TEXT_PCA_COMPONENTS={n_components}")

    # 7. 绘制方差曲线
    print("\n📊 生成可视化图表...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：单个主成分的方差解释比例
    components = range(1, len(explained_variance_ratio) + 1)
    ax1.bar(components, explained_variance_ratio * 100, alpha=0.7, color='steelblue')
    ax1.set_xlabel('主成分编号', fontsize=12)
    ax1.set_ylabel('方差解释比例 (%)', fontsize=12)
    ax1.set_title('各主成分的方差解释比例', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=1, color='r', linestyle='--', alpha=0.5, label='1% 阈值')
    ax1.legend()

    # 右图：累积方差解释比例
    ax2.plot(components, cumulative_variance_ratio * 100, 'o-', color='steelblue', linewidth=2)
    ax2.set_xlabel('主成分数量', fontsize=12)
    ax2.set_ylabel('累积方差解释比例 (%)', fontsize=12)
    ax2.set_title('累积方差解释比例曲线', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # 添加参考线
    for threshold in [80, 85, 90, 95]:
        ax2.axhline(y=threshold, color='r', linestyle='--', alpha=0.3)
        ax2.text(max_components * 0.7, threshold + 1, f'{threshold}%', fontsize=10, color='red')

    plt.tight_layout()
    output_path = "data/evaluation/pca_variance_analysis.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   ✓ 图表已保存: {output_path}")

    # 8. 实际降维示例
    print("\n" + "=" * 70)
    print("🎯 不同维度下的实际效果对比")
    print("=" * 70)

    test_dims = [5, 10, 15, 20, 30]
    for n in test_dims:
        if n <= len(cumulative_variance_ratio):
            pca_test = PCA(n_components=n, random_state=42)
            embeddings_reduced = pca_test.fit_transform(embeddings)

            # 计算重构误差
            embeddings_reconstructed = pca_test.inverse_transform(embeddings_reduced)
            mse = np.mean((embeddings - embeddings_reconstructed) ** 2)
            rmse = np.sqrt(mse)

            print(f"\n维度={n:2d} | 保留方差={cumulative_variance_ratio[n-1]*100:5.2f}% | "
                  f"RMSE={rmse:.4f} | 压缩率={384/n:.1f}x")

    print("\n" + "=" * 70)
    print("✅ 分析完成！")
    print("=" * 70)
    print("\n建议根据以上分析，在 .env 文件中设置：")
    print("TEXT_PCA_COMPONENTS=<选择的维度数>")
    print("\n权衡因素：")
    print("  • 更高维度 = 更多信息保留，但增加计算成本和过拟合风险")
    print("  • 更低维度 = 更快计算，但可能丢失重要信息")
    print("  • 推荐从 10-20 维开始，根据模型效果调整")
    print()

if __name__ == "__main__":
    analyze_pca_components(max_components=50)
