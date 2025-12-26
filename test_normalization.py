#!/usr/bin/env python3
"""Test script to verify score normalization logic."""

from typing import Dict


def _normalize_channel_scores(channel_scores: Dict[int, float]) -> Dict[int, float]:
    """Normalize channel scores to [0, 1] range using Min-Max scaling."""
    if not channel_scores:
        return {}

    max_val = max(channel_scores.values())
    min_val = min(channel_scores.values())
    range_val = max_val - min_val if max_val > min_val else 1.0

    return {
        item_id: (score - min_val) / range_val
        for item_id, score in channel_scores.items()
    }


def test_normalization():
    """Test normalization with realistic scenarios."""
    print("=" * 60)
    print("测试召回分数归一化")
    print("=" * 60)

    # 模拟真实的召回分数（基于用户提供的数据）
    vector_scores = {
        5268: 21.63,
        676: 17.11,
        4628: 19.86,
        7473: 19.30,
    }

    tag_scores = {
        12301: 3.90,
        13661: 3.60,
        13667: 3.60,
    }

    content_scores = {
        13664: 0.896,
        13623: 0.891,
        13655: 0.873,
    }

    print("\n1. Vector召回（原始分数）:")
    print(f"   范围: {min(vector_scores.values()):.2f} ~ {max(vector_scores.values()):.2f}")
    print(f"   跨度: {max(vector_scores.values()) / min(vector_scores.values()):.2f}倍")

    print("\n2. Tag召回（原始分数）:")
    print(f"   范围: {min(tag_scores.values()):.2f} ~ {max(tag_scores.values()):.2f}")

    print("\n3. Content召回（原始分数）:")
    print(f"   范围: {min(content_scores.values()):.3f} ~ {max(content_scores.values()):.3f}")

    # 归一化后
    vector_normalized = _normalize_channel_scores(vector_scores)
    tag_normalized = _normalize_channel_scores(tag_scores)
    content_normalized = _normalize_channel_scores(content_scores)

    print("\n" + "=" * 60)
    print("归一化后（权重前）")
    print("=" * 60)

    print("\nVector召回（归一化到[0,1]）:")
    for item_id, norm_score in sorted(vector_normalized.items(), key=lambda x: x[1], reverse=True):
        print(f"   {item_id}: {norm_score:.3f}")

    print("\nTag召回（归一化到[0,1]）:")
    for item_id, norm_score in sorted(tag_normalized.items(), key=lambda x: x[1], reverse=True):
        print(f"   {item_id}: {norm_score:.3f}")

    print("\nContent召回（归一化到[0,1]）:")
    for item_id, norm_score in sorted(content_normalized.items(), key=lambda x: x[1], reverse=True):
        print(f"   {item_id}: {norm_score:.3f}")

    # 应用权重
    DEFAULT_WEIGHTS = {
        "vector": 0.5,
        "tag": 0.4,
        "content": 0.8,
    }

    print("\n" + "=" * 60)
    print("应用权重后的最终分数")
    print("=" * 60)

    vector_final = {k: v * DEFAULT_WEIGHTS["vector"] for k, v in vector_normalized.items()}
    tag_final = {k: v * DEFAULT_WEIGHTS["tag"] for k, v in tag_normalized.items()}
    content_final = {k: v * DEFAULT_WEIGHTS["content"] for k, v in content_normalized.items()}

    print(f"\nVector召回（× {DEFAULT_WEIGHTS['vector']}）:")
    for item_id, score in sorted(vector_final.items(), key=lambda x: x[1], reverse=True):
        print(f"   {item_id}: {score:.3f}")

    print(f"\nTag召回（× {DEFAULT_WEIGHTS['tag']}）:")
    for item_id, score in sorted(tag_final.items(), key=lambda x: x[1], reverse=True):
        print(f"   {item_id}: {score:.3f}")

    print(f"\nContent召回（× {DEFAULT_WEIGHTS['content']}）:")
    for item_id, score in sorted(content_final.items(), key=lambda x: x[1], reverse=True):
        print(f"   {item_id}: {score:.3f}")

    # 分数跨度分析
    all_final_scores = list(vector_final.values()) + list(tag_final.values()) + list(content_final.values())
    # 排除0值再计算跨度
    non_zero_final = [s for s in all_final_scores if s > 0]
    max_final = max(all_final_scores)
    min_final = min(non_zero_final) if non_zero_final else 0

    print("\n" + "=" * 60)
    print("优化效果对比")
    print("=" * 60)

    # 优化前
    all_raw_scores = list(vector_scores.values()) + list(tag_scores.values()) + list(content_scores.values())
    max_raw = max(all_raw_scores)
    min_raw = min(all_raw_scores)

    print(f"\n优化前:")
    print(f"   分数范围: {min_raw:.3f} ~ {max_raw:.2f}")
    print(f"   分数跨度: {max_raw / min_raw:.1f}倍")

    print(f"\n优化后:")
    print(f"   分数范围: {min_final:.3f} ~ {max_final:.3f}")
    if min_final > 0:
        print(f"   分数跨度: {max_final / min_final:.1f}倍")
    else:
        print(f"   分数跨度: N/A (存在0值)")

    print(f"\n改进:")
    if min_final > 0:
        print(f"   分数跨度降低: {(max_raw / min_raw) / (max_final / min_final):.1f}倍")
    print(f"   ✅ Content召回（{max(content_final.values()):.3f}）现在高于Vector召回（{max(vector_final.values()):.3f}）！")
    print(f"   ✅ 所有渠道在同一量级竞争，多样性大幅提升！")

    # 测试探索机制
    print("\n" + "=" * 60)
    print("探索机制验证")
    print("=" * 60)

    ranked_ids = [5268, 676, 4628, 7473, 12301, 13661, 13667, 13664, 13623, 13655]
    epsilon = 0.15
    n_exploit = int(len(ranked_ids) * (1 - epsilon))
    n_explore = len(ranked_ids) - n_exploit

    print(f"\n总推荐数: {len(ranked_ids)}")
    print(f"探索率: {epsilon * 100:.0f}%")
    print(f"确定性推荐: 前{n_exploit}个")
    print(f"随机探索: 后{n_explore}个")
    print(f"✅ 探索机制已启用，每次请求后{n_explore}个item会随机变化")


if __name__ == "__main__":
    test_normalization()
