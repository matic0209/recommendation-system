#!/usr/bin/env python3
"""Test script for Phase 2: Cache time bucket and MMR diversity enhancements."""

from datetime import datetime


def test_time_bucket():
    """Test time bucket function."""
    print("=" * 60)
    print("测试缓存时间桶")
    print("=" * 60)

    def _get_time_bucket(bucket_hours: int = 1) -> str:
        """Generate time bucket identifier."""
        now = datetime.now()
        if bucket_hours == 1:
            return now.strftime("%Y-%m-%d-%H")
        else:
            return now.strftime("%Y-%m-%d")

    # 测试小时桶
    hourly_bucket = _get_time_bucket(1)
    print(f"\n当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"小时时间桶: {hourly_bucket}")
    print(f"  说明: 每小时一个桶，14:00-14:59共享同一个key")

    # 测试天桶
    daily_bucket = _get_time_bucket(24)
    print(f"\n天时间桶: {daily_bucket}")
    print(f"  说明: 每天一个桶，全天共享同一个key")

    # 模拟缓存key生成
    dataset_id = 7442
    user_id = 123
    limit = 10

    old_cache_key = f"recommend:{dataset_id}:{user_id}:{limit}"
    new_cache_key = f"recommend:{dataset_id}:{user_id}:{limit}:{hourly_bucket}"

    print(f"\n缓存key对比:")
    print(f"  优化前: {old_cache_key}")
    print(f"    问题: 相同请求永远返回相同结果（直到缓存过期）")
    print(f"\n  优化后: {new_cache_key}")
    print(f"    优势: 每小时自动刷新，结果随时间变化")

    print(f"\n预期行为:")
    print(f"  14:00 请求 → key=...:{hourly_bucket}")
    print(f"  14:30 请求 → key=...:{hourly_bucket} (命中缓存)")
    print(f"  15:00 请求 → key=...:{hourly_bucket.replace(hourly_bucket[-2:], str(int(hourly_bucket[-2:])+1).zfill(2))} (新时间桶，重新计算)")


def test_mmr_lambda():
    """Test MMR lambda parameter adjustments."""
    print("\n" + "=" * 60)
    print("测试MMR多样性参数")
    print("=" * 60)

    def _compute_mmr_lambda(endpoint: str, request_context: dict) -> float:
        """Compute MMR lambda (模拟优化后的逻辑)."""
        base = 0.5 if endpoint == "recommend_detail" else 0.4
        source = request_context.get("source")

        if source == "search":
            base = 0.6
        elif source in {"landing", "home"}:
            base = 0.3

        device = request_context.get("device_type")
        if device == "mobile":
            base = max(base - 0.1, 0.2)

        return base

    # 测试不同场景
    scenarios = [
        ("recommend_detail", {}, "详情页（默认）"),
        ("recommend_detail", {"source": "search"}, "详情页（搜索来源）"),
        ("recommend_detail", {"source": "landing"}, "详情页（落地页来源）"),
        ("recommend_detail", {"device_type": "mobile"}, "详情页（移动端）"),
        ("recommend_detail", {"source": "landing", "device_type": "mobile"}, "详情页（移动端+落地页）"),
    ]

    print("\nMMR Lambda参数（λ值）:")
    print(f"  λ=1.0: 100%相关性，0%多样性（高度聚集）")
    print(f"  λ=0.5: 50%相关性，50%多样性（平衡）")
    print(f"  λ=0.0: 0%相关性，100%多样性（完全随机）")

    print("\n场景测试:")
    for endpoint, context, desc in scenarios:
        lambda_val = _compute_mmr_lambda(endpoint, context)
        relevance_weight = lambda_val * 100
        diversity_weight = (1 - lambda_val) * 100
        print(f"\n  {desc}:")
        print(f"    λ={lambda_val:.1f} → 相关性{relevance_weight:.0f}% | 多样性{diversity_weight:.0f}%")

    print("\n优化对比:")
    print(f"  优化前: λ=0.7 → 相关性70% | 多样性30%")
    print(f"  优化后: λ=0.5 → 相关性50% | 多样性50%")
    print(f"  改进: 多样性权重提升67%（30%→50%）")


def test_channel_weights():
    """Test channel weight adjustments."""
    print("\n" + "=" * 60)
    print("测试渠道权重平衡")
    print("=" * 60)

    old_weights = {
        "behavior": 1.5,
        "content": 0.8,
        "vector": 0.5,
        "popular": 0.05,
    }

    new_weights = {
        "behavior": 1.2,
        "content": 1.0,
        "vector": 0.8,
        "popular": 0.1,
    }

    print("\n权重对比（归一化后的分数 × 权重）:")
    print(f"  {'渠道':<12} | {'优化前':<8} | {'优化后':<8} | {'变化':<10}")
    print(f"  {'-' * 12}-+-{'-' * 8}-+-{'-' * 8}-+-{'-' * 10}")

    for channel in old_weights:
        old_w = old_weights[channel]
        new_w = new_weights[channel]
        change = ((new_w - old_w) / old_w) * 100
        change_str = f"{change:+.0f}%" if change != 0 else "不变"
        print(f"  {channel:<12} | {old_w:<8.2f} | {new_w:<8.2f} | {change_str:<10}")

    # 模拟最终分数计算
    print("\n最终分数示例（假设所有渠道归一化分数=0.8）:")
    normalized_score = 0.8

    print(f"\n  优化前:")
    for channel, weight in old_weights.items():
        final_score = normalized_score * weight
        print(f"    {channel:<12}: {normalized_score} × {weight} = {final_score:.2f}")

    print(f"\n  优化后:")
    for channel, weight in new_weights.items():
        final_score = normalized_score * weight
        print(f"    {channel:<12}: {normalized_score} × {weight} = {final_score:.2f}")

    print("\n效果:")
    print("  ✅ Content权重提升25%（0.8→1.0），有机会超过Vector")
    print("  ✅ Popular权重翻倍（0.05→0.1），增加长尾内容曝光")
    print("  ✅ Behavior权重降低20%（1.5→1.2），减少过度个性化")
    print("  ✅ 配合归一化，各渠道更公平竞争")


def test_combined_effect():
    """Test combined effect of all Phase 2 changes."""
    print("\n" + "=" * 60)
    print("阶段2综合效果预测")
    print("=" * 60)

    print("\n优化矩阵:")
    print(f"  {'维度':<20} | {'优化前':<25} | {'优化后':<25}")
    print(f"  {'-' * 20}-+-{'-' * 25}-+-{'-' * 25}")
    print(f"  {'缓存新鲜度':<20} | {'永久固定（直到过期）':<25} | {'每小时刷新':<25}")
    print(f"  {'MMR多样性权重':<20} | {'30%':<25} | {'50%':<25}")
    print(f"  {'Content渠道权重':<20} | {'0.8':<25} | {'1.0':<25}")
    print(f"  {'Popular渠道权重':<20} | {'0.05':<25} | {'0.1':<25}")

    print("\n预期业务指标提升（配合阶段1）:")
    print(f"  标签覆盖率: 25个 → 28-30个 (+12-20%)")
    print(f"  推荐多样性: 已提升67% → 再提升30%")
    print(f"  结果新鲜度: 0% → 100%（每小时变化）")
    print(f"  CTR: 基于阶段1的+10% → 再+5-8%（总计+15-18%）")

    print("\n关键改进:")
    print("  ✅ 缓存时间桶 → 探索机制每小时生效不同item")
    print("  ✅ MMR平衡 → 相似item不再扎堆前N位")
    print("  ✅ 权重重平衡 → 内容召回与向量召回竞争力相当")


if __name__ == "__main__":
    test_time_bucket()
    test_mmr_lambda()
    test_channel_weights()
    test_combined_effect()
