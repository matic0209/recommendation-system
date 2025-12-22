# Sentence-BERT 模型离线部署指南

## 📋 概述

本文档说明如何在**有防火墙限制、无法访问HuggingFace**的生产环境中部署Sentence-BERT文本embedding特征。

---

## 🏗️ 架构说明

```
有网络的机器              生产环境（无网络）
─────────────            ────────────────
1. 下载SBERT模型  ──→    传输模型文件
2. 打包模型      ──→    解压到本地
3. 测试验证      ──→    配置环境变量
                        训练/推理正常使用
```

---

## 📦 需要准备的内容

### 模型信息
- **推荐模型**: `paraphrase-multilingual-MiniLM-L12-v2`
- **大小**: 约 420MB
- **维度**: 384维
- **优势**: 支持中文+英文，效果好，速度快

### 可选模型（根据需求选择）
| 模型名称 | 大小 | 维度 | 语言 | 说明 |
|---------|------|------|------|------|
| `paraphrase-multilingual-MiniLM-L12-v2` | 420MB | 384 | 多语言 | 推荐，平衡性能和速度 |
| `moka-ai/m3e-base` | 800MB | 768 | 中文 | 中文专用，效果最好 |
| `paraphrase-MiniLM-L6-v2` | 90MB | 384 | 英文 | 最快，仅英文 |

---

## 🚀 部署步骤

### Step 1: 在有网络的机器上下载模型

```bash
# 1. 确保已安装sentence-transformers
pip3 install sentence-transformers

# 2. 运行下载脚本
cd /path/to/recommend
python3 scripts/download_sbert_model.py

# 输出示例：
# ======================================================================
# SBERT模型离线下载工具
# ======================================================================
# 
# 📥 开始下载模型: paraphrase-multilingual-MiniLM-L12-v2
# 📂 保存路径: ./models/sbert/paraphrase-multilingual-MiniLM-L12-v2
# 
# ⏳ 下载中，请稍候...
# 💾 保存模型到: ./models/sbert/paraphrase-multilingual-MiniLM-L12-v2
# 
# ✅ 模型下载成功!
#    - 模型名称: paraphrase-multilingual-MiniLM-L12-v2
#    - 向量维度: 384
#    - 最大序列长度: 128
#    - 本地路径: /path/to/models/sbert/paraphrase-multilingual-MiniLM-L12-v2
#    - 模型大小: 420.5 MB
```

### Step 2: 打包模型

```bash
# 进入模型目录
cd models/sbert

# 打包模型（压缩）
tar -czf paraphrase-multilingual-MiniLM-L12-v2.tar.gz paraphrase-multilingual-MiniLM-L12-v2/

# 验证压缩包
ls -lh paraphrase-multilingual-MiniLM-L12-v2.tar.gz
# 输出: -rw-rw-r-- 1 user user 380M Dec 22 16:00 paraphrase-multilingual-MiniLM-L12-v2.tar.gz
```

### Step 3: 传输到生产环境

```bash
# 方法1: SCP传输（如果可以直接访问生产服务器）
scp paraphrase-multilingual-MiniLM-L12-v2.tar.gz \
    user@production-server:/home/ubuntu/recommend/models/sbert/

# 方法2: 通过跳板机
scp paraphrase-multilingual-MiniLM-L12-v2.tar.gz \
    user@jumphost:/tmp/
# 然后从跳板机传到生产
ssh user@jumphost
scp /tmp/paraphrase-multilingual-MiniLM-L12-v2.tar.gz \
    user@production:/home/ubuntu/recommend/models/sbert/

# 方法3: 物理介质（如果完全隔离）
# 将文件拷贝到U盘，然后手动上传到生产服务器
```

### Step 4: 在生产环境解压

```bash
# SSH登录生产环境
ssh user@production-server

# 创建目录
mkdir -p /home/ubuntu/recommend/models/sbert
cd /home/ubuntu/recommend/models/sbert

# 解压模型
tar -xzf paraphrase-multilingual-MiniLM-L12-v2.tar.gz

# 验证目录结构
ls -la paraphrase-multilingual-MiniLM-L12-v2/
# 应该看到:
# config.json
# config_sentence_transformers.json
# modules.json
# pytorch_model.bin
# sentence_bert_config.json
# special_tokens_map.json
# tokenizer.json
# tokenizer_config.json
# vocab.txt
# 等文件

# 验证文件完整性
du -sh paraphrase-multilingual-MiniLM-L12-v2/
# 输出: 420M    paraphrase-multilingual-MiniLM-L12-v2/
```

### Step 5: 配置环境变量

```bash
# 编辑生产环境配置
vim /home/ubuntu/recommend/.env.prod

# 确认或修改以下配置:
SBERT_MODEL=/home/ubuntu/recommend/models/sbert/paraphrase-multilingual-MiniLM-L12-v2
TEXT_PCA_COMPONENTS=10
```

### Step 6: 验证模型加载

```bash
# 测试模型是否能正常加载
cd /home/ubuntu/recommend
python3 << 'EOF_TEST'
import os
os.environ['SBERT_MODEL'] = '/home/ubuntu/recommend/models/sbert/paraphrase-multilingual-MiniLM-L12-v2'

from sentence_transformers import SentenceTransformer

model_path = os.environ['SBERT_MODEL']
print(f"Loading model from: {model_path}")

model = SentenceTransformer(model_path)
print(f"✓ Model loaded successfully!")
print(f"  Dimension: {model.get_sentence_embedding_dimension()}")

# 测试编码
texts = ["机器学习", "深度学习", "人工智能"]
embeddings = model.encode(texts)
print(f"✓ Encoding test passed!")
print(f"  Embeddings shape: {embeddings.shape}")
EOF_TEST

# 预期输出:
# Loading model from: /home/ubuntu/recommend/models/sbert/paraphrase-multilingual-MiniLM-L12-v2
# ✓ Model loaded successfully!
#   Dimension: 384
# ✓ Encoding test passed!
#   Embeddings shape: (3, 384)
```

### Step 7: 运行训练生成embeddings

```bash
cd /root/recommendation-system

# 使用本地模型训练
PYTHONPATH=/home/ubuntu/recommend \
SBERT_MODEL=/home/ubuntu/recommend/models/sbert/paraphrase-multilingual-MiniLM-L12-v2 \
python3 -m pipeline.train_models

# 应该看到:
# Loading Sentence-BERT model from local path: /home/ubuntu/recommend/models/sbert/paraphrase-multilingual-MiniLM-L12-v2
# Generating text embeddings for 12948 items...
# Batches: 100%|████████████████████| 405/405 [XX:XX<00:00]
# Text embeddings generated: shape=(12948, 384)
# ...
```

---

## ✅ 验证清单

完成部署后，请检查：

- [ ] 模型文件已上传到 `/home/ubuntu/recommend/models/sbert/paraphrase-multilingual-MiniLM-L12-v2/`
- [ ] 模型目录包含所有必要文件（pytorch_model.bin等）
- [ ] 环境变量 `SBERT_MODEL` 已配置
- [ ] 能够成功加载模型（运行验证脚本）
- [ ] 训练脚本正常运行，生成了 `dataset_features_with_embeddings.parquet`
- [ ] 生成了新的 `rank_model.pkl`
- [ ] 推理服务能正常加载embeddings特征

---

## 🔍 故障排除

### 问题1: 模型加载失败

**症状**:
```
FileNotFoundError: [Errno 2] No such file or directory: '.../pytorch_model.bin'
```

**解决**:
```bash
# 检查模型目录是否完整
ls -la /home/ubuntu/recommend/models/sbert/paraphrase-multilingual-MiniLM-L12-v2/

# 重新解压
cd /home/ubuntu/recommend/models/sbert
rm -rf paraphrase-multilingual-MiniLM-L12-v2
tar -xzf paraphrase-multilingual-MiniLM-L12-v2.tar.gz
```

### 问题2: 训练时仍尝试下载模型

**症状**:
```
ConnectionError: Can't reach HuggingFace
```

**解决**:
```bash
# 确认环境变量已设置
echo $SBERT_MODEL

# 确认路径存在
ls -la $SBERT_MODEL

# 检查代码中的路径解析逻辑
# 确保传入的是绝对路径
SBERT_MODEL=/home/ubuntu/recommend/models/sbert/paraphrase-multilingual-MiniLM-L12-v2 \
python3 -m pipeline.train_models
```

### 问题3: 权限问题

**症状**:
```
PermissionError: Permission denied
```

**解决**:
```bash
# 修改目录权限
sudo chown -R ubuntu:ubuntu /home/ubuntu/recommend/models/sbert/
chmod -R 755 /home/ubuntu/recommend/models/sbert/
```

---

## 📊 性能说明

| 阶段 | 首次（下载） | 后续（使用本地） |
|------|-------------|-----------------|
| 下载时间 | 5-15分钟 | 0秒（无需下载） |
| 模型加载 | 2-5秒 | 2-5秒（相同） |
| Embedding生成 | 5-15分钟 | 5-15分钟（相同） |

---

## 🔄 更新流程

当需要使用新模型时：

```bash
# 1. 在有网络的机器下载新模型
python3 scripts/download_sbert_model.py --model moka-ai/m3e-base

# 2. 打包
tar -czf m3e-base.tar.gz moka-ai_m3e-base/

# 3. 传输到生产环境
scp m3e-base.tar.gz production:/path/

# 4. 解压并更新配置
tar -xzf m3e-base.tar.gz -C /home/ubuntu/recommend/models/sbert/
# 更新 SBERT_MODEL=/home/ubuntu/recommend/models/sbert/moka-ai_m3e-base

# 5. 重新训练
python3 -m pipeline.train_models
```

---

## 💡 最佳实践

1. **模型选择建议**：
   - 推荐使用 `paraphrase-multilingual-MiniLM-L12-v2`（平衡性能）
   - 如果只有中文，可用 `moka-ai/m3e-base`（更好但更大）
   - 追求速度可用 `paraphrase-MiniLM-L6-v2`（仅英文）

2. **存储管理**：
   - 定期清理旧版本模型
   - 保留至少一个备份模型
   - 记录模型版本和训练时间

3. **安全性**：
   - 模型文件应有适当权限（755）
   - 定期验证文件完整性（MD5/SHA256）

---

## 📝 附录

### A. 目录结构

```
/home/ubuntu/recommend/
├── models/
│   └── sbert/
│       └── paraphrase-multilingual-MiniLM-L12-v2/
│           ├── config.json
│           ├── pytorch_model.bin      # 主要模型文件
│           ├── tokenizer.json
│           └── ...
├── data/
│   └── processed/
│       └── dataset_features_with_embeddings.parquet  # 生成的特征
└── scripts/
    └── download_sbert_model.py       # 下载脚本
```

### B. 环境变量完整列表

```bash
# 必需
SBERT_MODEL=/home/ubuntu/recommend/models/sbert/paraphrase-multilingual-MiniLM-L12-v2

# 可选
TEXT_PCA_COMPONENTS=10                # PCA降维组件数
SENTENCE_TRANSFORMERS_HOME=/custom   # 自定义缓存目录（不推荐修改）
```

---

**文档更新时间**: 2025-12-22  
**适用版本**: v1.0+
