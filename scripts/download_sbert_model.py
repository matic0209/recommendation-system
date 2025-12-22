#!/usr/bin/env python3
"""
SBERT模型下载脚本 - 用于离线部署

使用说明：
1. 在有网络的机器上运行此脚本
2. 下载完成后，将模型目录打包传输到生产环境
3. 在生产环境解压到指定目录

运行方式：
    python3 download_sbert_model.py
    
或指定其他模型：
    python3 download_sbert_model.py --model moka-ai/m3e-base --output ./models/sbert/
"""
import argparse
import os
import shutil
import sys
from pathlib import Path


def download_model(model_name: str, output_dir: str):
    """下载Sentence-BERT模型到本地目录"""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("❌ 错误: sentence-transformers未安装")
        print("请先安装: pip3 install sentence-transformers")
        sys.exit(1)

    output_path = Path(output_dir) / model_name.replace("/", "_")
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"📥 开始下载模型: {model_name}")
    print(f"📂 保存路径: {output_path}")
    print()

    try:
        # 下载模型（会自动缓存到 ~/.cache/torch/sentence_transformers/）
        print("⏳ 下载中，请稍候...")
        model = SentenceTransformer(model_name)
        
        # 保存到指定目录
        print(f"💾 保存模型到: {output_path}")
        model.save(str(output_path))
        
        # 获取模型信息
        dimension = model.get_sentence_embedding_dimension()
        max_seq_length = model.max_seq_length
        
        print()
        print("✅ 模型下载成功!")
        print(f"   - 模型名称: {model_name}")
        print(f"   - 向量维度: {dimension}")
        print(f"   - 最大序列长度: {max_seq_length}")
        print(f"   - 本地路径: {output_path.absolute()}")
        
        # 计算大小
        total_size = sum(f.stat().st_size for f in output_path.rglob('*') if f.is_file())
        size_mb = total_size / (1024 * 1024)
        print(f"   - 模型大小: {size_mb:.1f} MB")
        
        print()
        print("📦 后续步骤:")
        print(f"   1. 打包模型: tar -czf {model_name.replace('/', '_')}.tar.gz -C {output_path.parent} {output_path.name}")
        print(f"   2. 传输到生产环境: scp {model_name.replace('/', '_')}.tar.gz production:/path/")
        print(f"   3. 解压: tar -xzf {model_name.replace('/', '_')}.tar.gz -C /home/ubuntu/recommend/models/sbert/")
        print(f"   4. 配置环境变量: SBERT_MODEL=/home/ubuntu/recommend/models/sbert/{output_path.name}")
        
        return True
        
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="下载Sentence-BERT模型用于离线部署",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 下载默认多语言模型
  python3 download_sbert_model.py
  
  # 下载中文专用模型
  python3 download_sbert_model.py --model moka-ai/m3e-base
  
  # 指定输出目录
  python3 download_sbert_model.py --output ./my_models/
        """
    )
    
    parser.add_argument(
        '--model',
        default='paraphrase-multilingual-MiniLM-L12-v2',
        help='模型名称 (默认: paraphrase-multilingual-MiniLM-L12-v2)'
    )
    
    parser.add_argument(
        '--output',
        default='./models/sbert',
        help='输出目录 (默认: ./models/sbert)'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("SBERT模型离线下载工具")
    print("=" * 70)
    print()
    
    success = download_model(args.model, args.output)
    
    if success:
        print()
        print("=" * 70)
        print("✅ 所有操作完成!")
        print("=" * 70)
        sys.exit(0)
    else:
        print()
        print("=" * 70)
        print("❌ 下载失败")
        print("=" * 70)
        sys.exit(1)


if __name__ == '__main__':
    main()
