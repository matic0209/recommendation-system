# 环境配置说明

## 概述

项目支持多环境配置，通过不同的环境变量文件管理开发环境和生产环境。

## 配置文件

| 文件 | 用途 | 使用场景 |
|------|------|---------|
| `.env` | 开发环境配置 | 本地开发、测试 |
| `.env.prod` | 生产环境配置 | 生产环境部署 |
| `.env.example` | 配置模板 | 参考示例 |

## 使用方式

### 1. 本地开发（使用 .env）

#### 方式1: 直接运行 Python 脚本

```bash
# 自动加载 .env 文件
python3 -m pipeline.train_models

# 或显式指定环境文件
ENV_FILE=.env python3 -m pipeline.train_models
```

#### 方式2: Docker Compose 开发环境

```bash
# 使用 .env 配置启动
docker-compose up -d

# 查看日志
docker-compose logs -f airflow-scheduler
```

### 2. 生产环境（使用 .env.prod）

#### 方式1: 使用部署脚本（推荐）

```bash
# 一键部署生产环境
bash scripts/deploy_prod.sh

# 部署并拉取最新代码
bash scripts/deploy_prod.sh --pull
```

#### 方式2: 手动部署

```bash
# 使用 .env.prod 配置启动
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# 重启特定服务
docker-compose -f docker-compose.yml -f docker-compose.prod.yml restart airflow-scheduler

# 查看日志
docker-compose -f docker-compose.yml -f docker-compose.prod.yml logs -f airflow-scheduler

# 停止服务
docker-compose -f docker-compose.yml -f docker-compose.prod.yml down
```

#### 方式3: 本地测试生产配置

```bash
# 使用 .env.prod 但不通过 Docker
ENV_FILE=.env.prod python3 -m pipeline.train_models
```

## 环境变量加载优先级

### 在 Docker 容器内

```
Docker Compose env_file (.env.prod) > 容器环境变量 > Python load_dotenv
```

**说明**：
- Docker Compose 会将 env_file 中的变量注入到容器环境
- Python 代码使用 `override=False`，不会覆盖已存在的环境变量
- 因此容器内运行时，完全使用 docker-compose 指定的配置文件

### 直接运行 Python 脚本

```
系统环境变量 > ENV_FILE 指定的文件 > 默认 .env 文件
```

**示例**：
```bash
# 1. 使用默认 .env
python3 -m pipeline.train_models

# 2. 使用 .env.prod
ENV_FILE=.env.prod python3 -m pipeline.train_models

# 3. 直接设置环境变量（最高优先级）
HF_ENDPOINT=https://hf-mirror.com ENV_FILE=.env.prod python3 -m pipeline.train_models
```

## 关键配置项对比

### HuggingFace 镜像配置

```bash
# .env (开发环境 - 使用国内镜像)
HF_ENDPOINT=https://hf-mirror.com
SBERT_MODEL=paraphrase-multilingual-MiniLM-L12-v2

# .env.prod (生产环境)
HF_ENDPOINT=https://hf-mirror.com  # 生产环境也使用镜像
SBERT_MODEL=paraphrase-multilingual-MiniLM-L12-v2
```

### 端口映射

```bash
# .env (开发环境 - 避免端口冲突)
RECOMMEND_API_HOST_PORT=8000
AIRFLOW_WEB_HOST_PORT=8080

# .env.prod (生产环境 - 使用标准端口或自定义)
RECOMMEND_API_HOST_PORT=8090
AIRFLOW_WEB_HOST_PORT=8080
```

### 数据源配置

```bash
# .env (开发环境 - 使用 JSON 文件)
DATA_SOURCE=json
MATOMO_DATA_SOURCE=database

# .env.prod (生产环境 - 使用数据库)
DATA_SOURCE=json
MATOMO_DATA_SOURCE=database
```

## 验证配置

### 检查环境变量是否生效

```bash
# 在容器内检查
docker-compose -f docker-compose.yml -f docker-compose.prod.yml exec airflow-scheduler env | grep HF_ENDPOINT

# 或查看服务日志
docker-compose -f docker-compose.yml -f docker-compose.prod.yml logs airflow-scheduler | grep "HuggingFace endpoint"
```

### 测试 HuggingFace 镜像配置

```bash
# 本地测试
ENV_FILE=.env.prod python3 scripts/test_hf_endpoint.py

# 容器内测试
docker-compose -f docker-compose.yml -f docker-compose.prod.yml exec airflow-scheduler python3 scripts/test_hf_endpoint.py
```

## 常见问题

### Q1: 为什么 HF_ENDPOINT 没有生效？

**A**: 检查以下几点：
1. 确认配置文件中设置了 `HF_ENDPOINT=https://hf-mirror.com`
2. 如果在容器内，确认使用了正确的 docker-compose 命令
3. 如果直接运行 Python，确认 `ENV_FILE` 指向了正确的文件
4. 检查是否有其他地方设置了 `HF_HUB_OFFLINE=1`（会阻止下载）

### Q2: 如何切换环境？

**A**:
- **开发 → 生产**: 使用 `docker-compose -f docker-compose.yml -f docker-compose.prod.yml up`
- **生产 → 开发**: 使用 `docker-compose up`
- **本地脚本**: 设置 `ENV_FILE=.env.prod` 或 `ENV_FILE=.env`

### Q3: .env.prod 配置没有被 Docker 使用？

**A**: 确保使用了 `-f docker-compose.prod.yml` 参数：
```bash
# ✗ 错误 - 只会使用 .env
docker-compose up

# ✓ 正确 - 使用 .env.prod
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up
```

### Q4: 如何验证当前使用的配置文件？

**A**: 查看容器启动日志：
```bash
docker-compose -f docker-compose.yml -f docker-compose.prod.yml logs airflow-scheduler 2>&1 | head -50
```

应该看到：
```
[INFO] Loaded environment from: .env.prod  # (如果在容器外运行 Python)
[INFO] HuggingFace endpoint set to: https://hf-mirror.com
```

## 最佳实践

1. **不要提交敏感信息**
   - `.env` 和 `.env.prod` 已在 `.gitignore` 中
   - 使用 `.env.example` 作为模板

2. **生产环境部署**
   - 始终使用 `docker-compose.prod.yml` 覆盖文件
   - 使用部署脚本 `scripts/deploy_prod.sh` 简化流程

3. **本地测试生产配置**
   - 使用 `ENV_FILE=.env.prod` 在本地测试生产配置
   - 避免直接修改生产环境

4. **环境变量命名规范**
   - 使用大写字母和下划线
   - 相关变量使用相同前缀（如 `HF_*`, `SBERT_*`）
