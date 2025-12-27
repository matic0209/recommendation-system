#!/usr/bin/env python3
"""分析Popular榜单的质量问题"""
import json
import pandas as pd

# 读取Popular榜单
with open('models/top_items.json', 'r') as f:
    popular_items = json.load(f)

print(f"Popular榜单: {len(popular_items)}个item")

# 读取dataset统计
stats = pd.read_parquet('data/processed/dataset_stats.parquet')
stats = stats.set_index('dataset_id')

# 读取dataset基本信息
features = pd.read_parquet('data/processed/dataset_features.parquet')
features = features.set_index('dataset_id')

print("\n" + "="*80)
print("Popular榜单Item质量分析")
print("="*80)

# 分析Popular榜单中的item
popular_analysis = []
for item_id in popular_items:
    item_stats = stats.loc[item_id] if item_id in stats.index else None
    item_features = features.loc[item_id] if item_id in features.index else None

    if item_stats is None or item_features is None:
        print(f"\n⚠️  Item {item_id}: 数据缺失")
        continue

    # 提取关键指标
    dataset_name = item_features.get('dataset_name', 'Unknown')
    price = item_features.get('price', 0)
    interaction_count = item_features.get('interaction_count', 0)
    total_weight = item_features.get('total_weight', 0)
    popularity_score = item_features.get('popularity_score', 0)
    freshness_score = item_features.get('freshness_score', 0)
    days_since_last_purchase = item_features.get('days_since_last_purchase', 999)

    popular_analysis.append({
        'id': item_id,
        'name': dataset_name,
        'price': price,
        'interaction': interaction_count,
        'weight': total_weight,
        'popularity': popularity_score,
        'freshness': freshness_score,
        'days_inactive': days_since_last_purchase
    })

# 转为DataFrame
df = pd.DataFrame(popular_analysis)

print(f"\n前20个Popular Item详情:")
print(df.head(20)[['id', 'name', 'price', 'interaction', 'popularity', 'freshness', 'days_inactive']].to_string(index=False))

print(f"\n\n统计汇总:")
print(f"  价格分布:")
print(f"    最低: {df['price'].min():.2f}")
print(f"    最高: {df['price'].max():.2f}")
print(f"    平均: {df['price'].mean():.2f}")
print(f"    中位数: {df['price'].median():.2f}")

print(f"\n  互动量分布:")
print(f"    最低: {df['interaction'].min():.0f}")
print(f"    最高: {df['interaction'].max():.0f}")
print(f"    平均: {df['interaction'].mean():.0f}")

print(f"\n  Popularity Score分布:")
print(f"    最低: {df['popularity'].min():.2f}")
print(f"    最高: {df['popularity'].max():.2f}")
print(f"    平均: {df['popularity'].mean():.2f}")

print(f"\n  活跃度（days_inactive）:")
print(f"    最近: {df['days_inactive'].min():.0f} 天")
print(f"    最久: {df['days_inactive'].max():.0f} 天")
print(f"    平均: {df['days_inactive'].mean():.0f} 天")

# 识别低质量item
print(f"\n" + "="*80)
print("低质量Item识别（建议过滤）")
print("="*80)

low_quality = df[
    (df['price'] < 1.0) |  # 低价
    (df['interaction'] < 50) |  # 互动太少（低于平均值的1/4）
    (df['days_inactive'] > 180)  # 超过半年未购买
]

print(f"\n符合以下任一条件的item (共{len(low_quality)}个):")
print(f"  - 价格 < 1.0元")
print(f"  - 互动量 < 50")
print(f"  - 超过180天未购买")

print(f"\n低质量Item列表:")
for _, row in low_quality.iterrows():
    name_display = row['name'][:40] if len(row['name']) > 40 else row['name']
    print(f"  [{row['id']:5d}] {name_display:<40} | "
          f"价格:{row['price']:>6.2f} | 互动:{row['interaction']:>4.0f} | 不活跃:{row['days_inactive']:>4.0f}天")

# 高质量item
high_quality = df[
    (df['price'] >= 1.0) &
    (df['interaction'] >= 50) &
    (df['days_inactive'] <= 180)
]

print(f"\n" + "="*80)
print(f"高质量Item (共{len(high_quality)}个，建议保留)")
print("="*80)
for _, row in high_quality.head(10).iterrows():
    name_display = row['name'][:40] if len(row['name']) > 40 else row['name']
    print(f"  [{row['id']:5d}] {name_display:<40} | "
          f"价格:{row['price']:>6.2f} | 互动:{row['interaction']:>4.0f} | 不活跃:{row['days_inactive']:>4.0f}天")

print(f"\n\n过滤规则建议（基于数据分布）:")
print(f"  规则1: price >= {max(1.0, df['price'].quantile(0.25)):.2f}元  (25分位数)")
print(f"  规则2: interaction_count >= {max(50, df['interaction'].quantile(0.25)):.0f}")
print(f"  规则3: days_since_last_purchase <= 180天")
print(f"\n或者更严格的规则（中位数）:")
print(f"  规则1: price >= {df['price'].median():.2f}元")
print(f"  规则2: interaction_count >= {df['interaction'].median():.0f}")
print(f"  规则3: popularity_score >= {df['popularity'].median():.2f}")
