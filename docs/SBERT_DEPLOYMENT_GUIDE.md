# Sentence-BERT 模型部署指南（生产环境）

## 🎯 两种部署方式

### 🔥 方式一：使用HuggingFace国内镜像（推荐）

**优点**：
- ✅ 配置简单（只需设置环境变量）
- ✅ 无需手动下载和传输
- ✅ 自动缓存，后续无需重复下载
- ✅ 速度快（国内镜像）

**适用场景**：生产环境可以访问 hf-mirror.com

#### 部署步骤：

```bash
# 1. 配置环境变量（已在 .env.prod 中配置）
HF_ENDPOINT=https://hf-mirror.com
SBERT_MODEL=paraphrase-multilingual-MiniLM-L12-v2

# 2. 直接运行训练（会自动从镜像下载）
cd /root/recommendation-system
PYTHONPATH=/home/ubuntu/recommend python3 -m pipeline.train_models

# 首次运行会下载模型（约490MB），日志显示：
# Downloading from https://hf-mirror.com/...
# Model cached to ~/.cache/huggingface/
# 
# 后续运行直接使用缓存，无需重复下载
```

**就这么简单！** 无需复杂的离线传输流程。

---

### 📦 方式二：完全离线部署

**适用场景**：生产环境完全无法访问外网（包括镜像站）

详见：`SBERT_OFFLINE_DEPLOYMENT.md`

---

## 🌐 可用的HuggingFace镜像

| 镜像站 | 地址 | 速度 | 稳定性 |
|--------|------|------|--------|
| **hf-mirror.com** | https://hf-mirror.com | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| ModelScope | https://www.modelscope.cn | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| 阿里云 | https://mirrors.aliyun.com/huggingface | ⭐⭐⭐ | ⭐⭐⭐⭐ |

**推荐使用 hf-mirror.com**（最快最稳定）

---

## 🔧 配置说明

### .env.prod 配置

```bash
# 方式一：使用镜像（推荐）
HF_ENDPOINT=https://hf-mirror.com
SBERT_MODEL=paraphrase-multilingual-MiniLM-L12-v2

# 方式二：使用本地路径（完全离线）
# HF_ENDPOINT=  # 留空或注释掉
# SBERT_MODEL=/home/ubuntu/recommend/models/sbert/paraphrase-multilingual-MiniLM-L12-v2
```

---

## 📊 模型信息

| 项目 | 内容 |
|------|------|
| **模型名称** | paraphrase-multilingual-MiniLM-L12-v2 |
| **实际大小** | ~490MB |
| **缓存位置** | ~/.cache/huggingface/hub/ |
| **向量维度** | 384 |
| **支持语言** | 50+语言（包括中文） |

---

## ⚡ 快速验证

### 测试镜像连接

```bash
# 测试是否能访问HF镜像
curl -I https://hf-mirror.com
# 应该返回 HTTP/2 200

# 或者
ping hf-mirror.com
```

### 测试模型下载

```bash
cd /home/ubuntu/recommend

# 使用镜像下载测试
HF_ENDPOINT=https://hf-mirror.com python3 << 'EOF'
from sentence_transformers import SentenceTransformer
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

print("开始下载模型...")
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
print(f"✓ 模型下载成功！维度: {model.get_sentence_embedding_dimension()}")

# 测试编码
texts = ["推荐系统", "机器学习"]
embeddings = model.encode(texts)
print(f"✓ 编码测试通过！Shape: {embeddings.shape}")
