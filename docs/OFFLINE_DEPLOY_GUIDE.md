# 离线安装 Sentence-Transformers 与 Faiss 指南

本指南帮助在离线环境（含 Docker 部署）中部署 `sentence-transformers` 与 `faiss-cpu`，确保推荐系统的语义召回和向量召回功能可用。

---

## 1. 准备依赖（联网机器）

1. **创建 wheel 仓库**
   ```bash
   mkdir -p wheelhouse
   ```

2. **下载所需 wheel**
   ```bash
   pip download --only-binary=:all: \
       faiss-cpu==1.7.4 \
       sentence-transformers==2.3.1 \
       torch==2.1.2 \
       torchvision==0.16.2 \
       "transformers>=4.6.0,<5" \
       "huggingface-hub>=0.4.0" \
       numpy==1.24.4 \
       scipy==1.10.1 \
       scikit-learn==1.3.2 \
       sentencepiece==0.2.0 \
       tqdm \
       nltk \
       safetensors \
       tokenizers \
       requests \
       pyyaml \
       typing-extensions
   ```

3. **可选：下载 Sentence-BERT 模型**
   ```bash
   python - <<'PY'
   from sentence_transformers import SentenceTransformer
   SentenceTransformer('all-MiniLM-L6-v2')
   PY
   ```
   模型缓存会位于 `~/.cache/sentence-transformers/all-MiniLM-L6-v2/`。

4. **整理文件**
   ```
   wheelhouse/                 # 所有 wheel 包
   cache/sentence-transformers # 预下载模型（如需要）
   cache/huggingface           # 其他模型缓存（可选）
   ```
   将上述目录拷贝到代码仓库（例如 `docker/wheels/` 和 `cache/`）。

---

## 2. 修改 Dockerfile（本地完成）

1. **builder 阶段引用 wheelhouse**
   ```dockerfile
   COPY docker/wheels/ /tmp/wheels/
   RUN pip install --no-cache-dir --upgrade pip && \
       pip install --no-index --find-links=/tmp/wheels \
           faiss-cpu==1.7.4 sentence-transformers==2.3.1 && \
       pip install --no-index --find-links=/tmp/wheels \
           -r /tmp/requirements.txt
   ```

2. **runtime 阶段补齐依赖**
   ```dockerfile
   RUN apt-get update && apt-get install -y --no-install-recommends \
       libopenblas-dev libomp5 && \
       rm -rf /var/lib/apt/lists/*
   ```

3. **可选：拷贝模型缓存入镜像**
   ```dockerfile
   COPY cache/sentence-transformers /opt/recommend/cache/sentence-transformers
   COPY cache/huggingface /opt/recommend/cache/huggingface
   ```
  （或在 Docker Compose 中挂载宿主机目录）

---

## 3. 本地验证

1. 构建并启动：
   ```bash
   docker compose build recommendation-api
   docker compose up -d recommendation-api
   ```

2. 进入容器验证：
   ```bash
   docker compose exec recommendation-api bash -lc "python -c 'import faiss; import sentence_transformers; print(\"OK\")'"
   ```

3. 运行召回训练：
   ```bash
   docker compose exec recommendation-api python -m pipeline.recall_engine_v2
   ```
   若成功，会看到 “Training Faiss vector recall…” 等日志。

通过后，将 Dockerfile 与离线依赖提交至版本库。

---

## 4. 生产部署步骤

1. 拉取最新代码：
   ```bash
   git fetch origin
   git checkout feature/request-id-tracking   # 或 master
   git pull
   ```

2. 构建并重启相关容器：
   ```bash
   docker compose build recommendation-api airflow-scheduler airflow-worker airflow-webserver
   docker compose up -d recommendation-api airflow-scheduler airflow-worker airflow-webserver
   ```

3. 若镜像未内置模型缓存，确保 compose 挂载：
   ```yaml
   - /opt/recommend/cache/sentence-transformers:/opt/recommend/cache/sentence-transformers
   - /opt/recommend/cache/huggingface:/opt/recommend/cache/huggingface
   ```
   并将 `cache/` 内的模型复制到宿主机对应目录。

4. 在生产容器内验证：
   ```bash
   docker compose exec recommendation-api bash -lc "python -c 'import faiss; import sentence_transformers; print(\"OK\")'"
   docker compose exec recommendation-api python -m pipeline.recall_engine_v2
   ```

---

## 5. 虚拟环境同步（可选）

若本地也使用虚拟环境，可复用同一 wheelhouse：
```bash
python -m venv .venv
source .venv/bin/activate
pip install --no-index --find-links=docker/wheels \
    faiss-cpu==1.7.4 sentence-transformers==2.3.1 \
    -r requirements.txt
```

如此，本地和 Docker 环境保持一致，避免版本差异。

---

## 6. 常见验证命令

- 召回训练：`python -m pipeline.recall_engine_v2`
- 检查模型文件：`ls models/faiss_recall*`（模型会在召回训练后保存）
- 设置环境变量禁用 Faiss（调试用）：`export USE_FAISS_RECALL=0`

按照以上流程，离线环境即可稳定使用 Sentence-Transformers 与 Faiss，实现语义召回和多路召回闭环。好运！
