# 推荐系统迁移指南

本文档说明如何将推荐系统从一台服务器迁移到另一台服务器。

## 📦 需要迁移的内容

### 必须迁移
1. **代码仓库** - 整个项目目录
2. **JSON 数据文件** - `data/dianshu_data/`
3. **配置文件** - `.env`

### 建议迁移（可选）
4. **训练好的模型** - `models/` 目录（避免重新训练）
5. **处理后的数据** - `data/processed/` 目录
6. **Redis 数据** - 缓存数据（可选，会自动重建）

### 不需要迁移
- Docker 容器和镜像（新机器上重新构建）
- Python 虚拟环境（新机器上重新创建）
- 日志文件（临时数据）

---

## 🚀 方式 A：完整迁移（推荐）

适用于：希望保留所有数据和模型，快速启动服务

### 在源服务器上

```bash
cd /home/ubuntu

# 1. 打包整个项目（包含数据和模型）
tar -czf recommend_full.tar.gz \
    --exclude='recommend/venv' \
    --exclude='recommend/.git' \
    --exclude='recommend/logs' \
    --exclude='recommend/mlruns' \
    recommend/

# 2. 查看打包大小
ls -lh recommend_full.tar.gz

# 3. 传输到新服务器
scp recommend_full.tar.gz user@new-server:/tmp/
```

### 在目标服务器上

```bash
cd /opt

# 1. 解压
sudo tar -xzf /tmp/recommend_full.tar.gz
sudo chown -R $USER:$USER recommend

# 2. 进入项目目录
cd recommend

# 3. 检查配置
cat .env

# 4. 更新路径配置（如果部署路径不同）
# 编辑 .env，更新 DATA_JSON_DIR 路径
vim .env

# 5. 安装 Docker（如果未安装）
# 参考 DEPLOYMENT_GUIDE_JSON.md

# 6. 创建 Python 虚拟环境
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 7. 启动 Docker 服务
docker compose up -d

# 8. 验证
curl http://localhost:8000/health
```

**优点：** 快速，保留所有数据和模型
**缺点：** 传输文件较大（可能几百 MB 到几 GB）

---

## 🎯 方式 B：最小化迁移

适用于：网络带宽有限，或希望在新服务器上重新训练模型

### 在源服务器上

```bash
cd /home/ubuntu

# 1. 打包必要文件（代码 + JSON 数据 + 配置）
tar -czf recommend_minimal.tar.gz \
    --exclude='recommend/venv' \
    --exclude='recommend/.git' \
    --exclude='recommend/logs' \
    --exclude='recommend/mlruns' \
    --exclude='recommend/models' \
    --exclude='recommend/data/business' \
    --exclude='recommend/data/cleaned' \
    --exclude='recommend/data/processed' \
    --exclude='recommend/data/evaluation' \
    recommend/

# 2. 查看打包大小（应该小很多）
ls -lh recommend_minimal.tar.gz

# 3. 传输到新服务器
scp recommend_minimal.tar.gz user@new-server:/tmp/
```

### 在目标服务器上

```bash
cd /opt

# 1. 解压
sudo tar -xzf /tmp/recommend_minimal.tar.gz
sudo chown -R $USER:$USER recommend

# 2. 进入项目目录
cd recommend

# 3. 检查 JSON 数据是否完整
ls -lh data/dianshu_data/

# 4. 更新配置
vim .env

# 5. 安装依赖
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 6. 运行 Pipeline 生成模型（30-60 分钟）
export PYTHONPATH=/opt/recommend:$PYTHONPATH
bash scripts/run_pipeline.sh

# 7. 启动服务
docker compose up -d

# 8. 验证
curl http://localhost:8000/health
```

**优点：** 传输文件小，适合网络较慢的情况
**缺点：** 需要重新训练模型（耗时 30-60 分钟）

---

## 📤 方式 C：仅迁移模型（快速恢复）

适用于：代码已在新服务器上，只需要迁移训练好的模型

### 在源服务器上

```bash
cd /home/ubuntu/recommend

# 打包模型和配置
tar -czf models_and_config.tar.gz models/ .env
```

### 在目标服务器上

```bash
cd /opt/recommend

# 解压并覆盖
tar -xzf /tmp/models_and_config.tar.gz

# 检查模型文件
ls -lh models/

# 重启 API
docker compose restart recommendation-api

# 验证
curl http://localhost:8000/health
```

---

## 🔐 配置文件处理

### .env 文件需要更新的配置项

迁移后，根据新服务器环境更新以下配置：

```ini
# 1. 数据路径（如果部署目录不同）
DATA_JSON_DIR=/opt/recommend/data/dianshu_data

# 2. 企业微信 IP 白名单（如果 IP 变化）
# 需要在企业微信后台重新配置新服务器的公网 IP

# 3. Redis URL（如果使用外部 Redis）
REDIS_URL=redis://redis:6379/0

# 4. 其他路径配置
MLFLOW_TRACKING_URI=http://mlflow:5000
```

---

## 📋 迁移检查清单

### 迁移前（源服务器）

- [ ] 停止服务：`docker compose down`
- [ ] 备份数据库（如果使用数据库模式）
- [ ] 打包项目文件
- [ ] 验证打包完整性：`tar -tzf recommend_full.tar.gz | head`

### 迁移中

- [ ] 传输文件到新服务器
- [ ] 验证文件完整性：`md5sum recommend_full.tar.gz`

### 迁移后（目标服务器）

- [ ] 解压文件
- [ ] 检查 JSON 数据：`ls -lh data/dianshu_data/`
- [ ] 检查模型文件：`ls -lh models/` （完整迁移）
- [ ] 更新 .env 配置
- [ ] 运行 Pipeline（最小化迁移）
- [ ] 启动 Docker 服务
- [ ] 健康检查：`curl http://localhost:8000/health`
- [ ] 测试推荐接口：`curl http://localhost:8000/similar/123?top_n=10`
- [ ] 配置企业微信 IP 白名单（如果 IP 变化）
- [ ] 测试企业微信通知：`curl -X POST http://localhost:9000/test`

---

## 🔧 常见问题

### Q1: 迁移后 API 返回 "models_loaded": false

**原因：** 模型文件未迁移或路径不正确

**解决：**
```bash
# 检查模型文件
ls -lh models/

# 如果缺失，重新训练
bash scripts/run_pipeline.sh

# 或从源服务器复制
scp -r source-server:/home/ubuntu/recommend/models /opt/recommend/
```

### Q2: JSON 数据路径错误

**错误：** `FileNotFoundError: data/dianshu_data/user.json`

**解决：**
```bash
# 检查文件
ls -lh data/dianshu_data/

# 更新 .env 中的路径
vim .env
# 确保 DATA_JSON_DIR 是正确的绝对路径

# 重启服务
docker compose restart recommendation-api
```

### Q3: Docker 构建失败

**错误：** `Error response from daemon: no space left on device`

**解决：**
```bash
# 清理 Docker 缓存
docker system prune -a

# 检查磁盘空间
df -h
```

### Q4: 企业微信通知失败（IP 白名单）

**错误：** `errcode: 60020 - not allow to access from your ip`

**解决：**
1. 查看新服务器公网 IP：`curl ifconfig.me`
2. 登录企业微信管理后台
3. 更新应用的"企业可信 IP"配置
4. 重试测试：`curl -X POST http://localhost:9000/test`

---

## 🔄 回滚方案

如果迁移失败需要回滚：

### 保留源服务器

在确认新服务器正常运行前，不要删除源服务器上的数据。

### 快速回滚

```bash
# 在源服务器上重新启动服务
cd /home/ubuntu/recommend
docker compose up -d
```

---

## 📊 迁移时间估算

| 迁移方式 | 传输时间 | 安装配置 | Pipeline | 总计 |
|---------|---------|----------|----------|------|
| 完整迁移 | 10-30 分钟 | 10 分钟 | 0 分钟 | 20-40 分钟 |
| 最小化迁移 | 5-10 分钟 | 10 分钟 | 30-60 分钟 | 45-80 分钟 |
| 仅迁移模型 | 2-5 分钟 | 5 分钟 | 0 分钟 | 7-10 分钟 |

*注：传输时间取决于网络带宽和文件大小*

---

## 💡 最佳实践

1. **先在新服务器上测试**
   - 部署到测试端口（如 8001）
   - 验证所有功能正常
   - 再切换生产流量

2. **使用版本控制**
   - 提交代码到 Git
   - 在新服务器上 `git clone`
   - 只迁移数据和配置文件

3. **保留备份**
   - 至少保留 3 个版本的模型备份
   - 定期备份 JSON 数据

4. **自动化迁移**
   - 编写迁移脚本
   - 使用 Ansible 或 Terraform

---

**相关文档：**
- [部署指南](DEPLOYMENT_GUIDE_JSON.md)
- [快速开始](../QUICKSTART_JSON.md)
- [运维手册](OPERATIONS_SOP.md)

**维护者：** 推荐系统团队
**最后更新：** 2025-10-16
