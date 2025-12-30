# 子任务02: 实现Tags增强脚本

## 目标
实现`pipeline/enhance_tags.py`脚本，支持批量处理、进度显示、错误处理、断点续传。

## 成功标准
- [ ] 脚本可成功加载dataset_features.parquet
- [ ] 支持批量推理（batch_size可配置）
- [ ] 有清晰的进度条显示（使用tqdm）
- [ ] 支持断点续传（已处理的item可跳过）
- [ ] 输出格式正确的enhanced数据文件
- [ ] 包含完整的错误处理和日志

## 实施步骤

### 1. 创建脚本文件
创建 `pipeline/enhance_tags.py`:

```python
"""
Tags增强脚本: 使用零样本分类模型从description提取标准化大类标签
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import json
import logging
from pathlib import Path
from typing import List, Dict
import pandas as pd
from tqdm import tqdm
from transformers import pipeline

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 11个候选大类
CATEGORIES = [
    "金融", "医疗健康", "政府政务", "交通运输",
    "教育培训", "能源环保", "农业", "科技",
    "商业零售", "文化娱乐", "社会民生"
]

# 配置
DATA_DIR = Path(__file__).parent.parent / "data"
MODELS_DIR = Path(__file__).parent.parent / "models"
INPUT_FILE = DATA_DIR / "cleaned" / "dataset_features.parquet"
OUTPUT_FILE = DATA_DIR / "processed" / "dataset_features_enhanced.parquet"
CHECKPOINT_FILE = DATA_DIR / "processed" / "enhance_tags_checkpoint.json"


def load_checkpoint() -> Dict[int, List[str]]:
    """加载断点续传数据"""
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_checkpoint(enhanced_tags: Dict[int, List[str]]):
    """保存断点数据"""
    with open(CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
        json.dump(enhanced_tags, f, ensure_ascii=False, indent=2)


def parse_original_tags(tag_str: str) -> List[str]:
    """解析原始tags字符串"""
    if pd.isna(tag_str) or not tag_str:
        return []
    return [t.strip() for t in str(tag_str).split(";") if t.strip()]


def enhance_single_item(
    description: str,
    original_tags: str,
    classifier,
    confidence_threshold: float = 0.3
) -> List[str]:
    """
    增强单个item的tags

    Args:
        description: item描述文本
        original_tags: 原始用户打的tags (分号分隔)
        classifier: 零样本分类器
        confidence_threshold: 置信度阈值

    Returns:
        合并后的tags列表
    """
    # 解析原始tags
    orig_tags_list = parse_original_tags(original_tags)

    # 如果没有description，只返回原始tags
    if pd.isna(description) or not str(description).strip():
        return orig_tags_list

    # 零样本分类
    try:
        result = classifier(
            str(description)[:512],  # 截断到512字符避免超长
            CATEGORIES,
            multi_label=True
        )

        # 取置信度>阈值的前3个大类
        model_tags = [
            label for label, score in zip(result["labels"], result["scores"])
            if score > confidence_threshold
        ][:3]

    except Exception as e:
        logger.warning(f"分类失败: {e}")
        model_tags = []

    # 合并原始tags和模型tags，去重
    combined = list(set(orig_tags_list + model_tags))
    return combined


def enhance_tags_batch(
    df: pd.DataFrame,
    classifier,
    batch_size: int = 1,
    resume_from_checkpoint: bool = True
) -> Dict[int, List[str]]:
    """
    批量增强tags

    Args:
        df: dataset_features数据
        classifier: 零样本分类器
        batch_size: 批处理大小 (当前版本=1，未来可优化)
        resume_from_checkpoint: 是否从断点续传

    Returns:
        {dataset_id: enhanced_tags_list}
    """
    # 加载checkpoint
    enhanced_tags = load_checkpoint() if resume_from_checkpoint else {}
    processed_ids = set(int(k) for k in enhanced_tags.keys())

    logger.info(f"总计 {len(df)} 个items，已处理 {len(processed_ids)} 个")

    # 过滤已处理的items
    if processed_ids:
        df = df[~df["dataset_id"].isin(processed_ids)]
        logger.info(f"剩余 {len(df)} 个items待处理")

    # 批量处理
    save_interval = 100  # 每100个保存一次checkpoint

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="增强Tags"):
        dataset_id = int(row["dataset_id"])
        description = row.get("description", "")
        original_tags = row.get("tag", "")

        # 增强tags
        enhanced = enhance_single_item(
            description,
            original_tags,
            classifier
        )

        enhanced_tags[dataset_id] = enhanced

        # 定期保存checkpoint
        if len(enhanced_tags) % save_interval == 0:
            save_checkpoint(enhanced_tags)

    # 最终保存
    save_checkpoint(enhanced_tags)
    logger.info(f"Tags增强完成，共处理 {len(enhanced_tags)} 个items")

    return enhanced_tags


def build_tag_indices(enhanced_tags: Dict[int, List[str]]) -> tuple:
    """
    构建tag倒排索引

    Returns:
        (tag_to_items, item_to_tags)
    """
    from collections import defaultdict

    tag_to_items = defaultdict(set)
    item_to_tags = {}

    for dataset_id, tags in enhanced_tags.items():
        item_to_tags[dataset_id] = tags
        for tag in tags:
            tag_lower = tag.lower().strip()
            if tag_lower:
                tag_to_items[tag_lower].add(int(dataset_id))

    # 转换set为list以便JSON序列化
    tag_to_items_serializable = {
        tag: list(items) for tag, items in tag_to_items.items()
    }

    return tag_to_items_serializable, item_to_tags


def main():
    """主函数"""
    logger.info("=== Tags增强流程开始 ===")

    # 1. 加载数据
    logger.info(f"加载数据: {INPUT_FILE}")
    df = pd.read_parquet(INPUT_FILE)
    logger.info(f"加载完成: {len(df)} 行数据")

    # 2. 加载零样本分类模型
    logger.info("加载Erlangshen-Roberta-110M-NLI模型...")
    classifier = pipeline(
        "zero-shot-classification",
        model="IDEA-CCNL/Erlangshen-Roberta-110M-NLI",
        device=-1  # CPU
    )
    logger.info("模型加载成功！")

    # 3. 批量增强tags
    enhanced_tags = enhance_tags_batch(df, classifier)

    # 4. 保存增强后的数据
    logger.info("保存增强后的数据...")

    # 4.1 添加enhanced_tags列到DataFrame
    df["enhanced_tags"] = df["dataset_id"].map(
        lambda x: ";".join(enhanced_tags.get(int(x), []))
    )

    # 4.2 保存parquet
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_FILE, index=False)
    logger.info(f"保存parquet: {OUTPUT_FILE}")

    # 5. 构建并保存tag索引
    logger.info("构建tag索引...")
    tag_to_items, item_to_tags = build_tag_indices(enhanced_tags)

    # 5.1 保存tag_to_items_enhanced.json
    tag_to_items_file = MODELS_DIR / "tag_to_items_enhanced.json"
    with open(tag_to_items_file, 'w', encoding='utf-8') as f:
        json.dump(tag_to_items, f, ensure_ascii=False, indent=2)
    logger.info(f"保存tag_to_items: {tag_to_items_file}")

    # 5.2 保存item_to_tags_enhanced.json (转为字符串key以便JSON)
    item_to_tags_serializable = {str(k): v for k, v in item_to_tags.items()}
    item_to_tags_file = MODELS_DIR / "item_to_tags_enhanced.json"
    with open(item_to_tags_file, 'w', encoding='utf-8') as f:
        json.dump(item_to_tags_serializable, f, ensure_ascii=False, indent=2)
    logger.info(f"保存item_to_tags: {item_to_tags_file}")

    # 6. 统计信息
    logger.info("=== 增强统计 ===")
    total_items = len(enhanced_tags)
    items_with_enhanced = sum(1 for tags in enhanced_tags.values() if tags)
    coverage = items_with_enhanced / total_items * 100 if total_items > 0 else 0

    avg_tags = sum(len(tags) for tags in enhanced_tags.values()) / total_items

    logger.info(f"总item数: {total_items}")
    logger.info(f"有tags的item数: {items_with_enhanced}")
    logger.info(f"Tags覆盖率: {coverage:.2f}%")
    logger.info(f"平均tags数/item: {avg_tags:.2f}")
    logger.info(f"唯一tag数: {len(tag_to_items)}")

    # 7. 清理checkpoint
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
        logger.info("清理checkpoint文件")

    logger.info("=== Tags增强流程完成 ===")


if __name__ == "__main__":
    main()
```

## 验证清单
- [ ] 脚本可正常导入所有依赖
- [ ] 使用少量数据（如100条）测试运行无误
- [ ] 进度条正常显示
- [ ] checkpoint机制正常工作（手动中断后可续传）
- [ ] 日志信息清晰完整

## 测试方法

### 小规模测试
```bash
# 备份原始数据
cp data/cleaned/dataset_features.parquet data/cleaned/dataset_features.parquet.bak

# 创建测试数据（只取前100行）
python3 -c "
import pandas as pd
df = pd.read_parquet('data/cleaned/dataset_features.parquet')
df.head(100).to_parquet('data/cleaned/dataset_features_test.parquet')
"

# 修改脚本INPUT_FILE指向测试数据
# 运行测试
python3 -m pipeline.enhance_tags
```

### 验证输出
```bash
# 检查输出文件
ls -lh data/processed/dataset_features_enhanced.parquet
ls -lh models/tag_to_items_enhanced.json
ls -lh models/item_to_tags_enhanced.json

# 验证覆盖率
python3 -c "
import json
with open('models/item_to_tags_enhanced.json') as f:
    data = json.load(f)
    total = len(data)
    with_tags = sum(1 for v in data.values() if v)
    print(f'覆盖率: {with_tags/total*100:.2f}%')
"
```

## 下一步
完成后进入子任务03：执行全量tags增强
