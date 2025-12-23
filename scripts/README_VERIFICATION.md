# 数据源配置验证脚本使用说明

## 概述

本目录包含用于验证数据源配置的脚本，确保：
- Business 数据从 JSON 文件加载
- Matomo 数据从 MySQL 数据库加载

## 验证脚本列表

### 1. `verify_data_source_quick.sh` - 快速验证（推荐）

快速检查数据源配置是否正确。

**使用方法：**
```bash
bash scripts/verify_data_source_quick.sh
```

**检查项：**
- ✓ 环境变量设置
- ✓ 配置值验证（BUSINESS=json, MATOMO=database）
- ✓ 路径检查
- ✓ JSON 文件检查
- ✓ 数据库配置

### 2. `test_data_source_integration.sh` - 集成测试（完整验证）

完整的集成测试，实际加载数据验证配置生效。

**使用方法：**
```bash
bash scripts/test_data_source_integration.sh
```

**测试内容：**
- ✓ 配置值验证
- ✓ JSON 数据实际加载（user.json, dataset.json 等）
- ✓ 数据库配置和连接串生成
- ✓ 端到端验证

### 3. `verify_data_source.py` - Python 验证脚本

独立的 Python 验证脚本（可在本地或容器内运行）。

**使用方法：**
```bash
# 本地运行
python3 scripts/verify_data_source.py

# Docker 容器内运行
docker compose exec recommendation-api python3 /app/scripts/verify_data_source.py
```

## 验证结果示例

### 成功输出

```
======================================================================
  🎉 所有集成测试通过！
======================================================================

✅ 数据源配置已正确生效:
  • Business 数据: 从 JSON 文件加载 ✓
  • Matomo 数据: 从 MySQL 数据库加载 ✓
```

### 失败输出

如果配置有问题，脚本会显示详细的错误信息和未通过的检查项。

## 验证报告

完整的验证报告请查看：[../DATA_SOURCE_VERIFICATION_REPORT.md](../DATA_SOURCE_VERIFICATION_REPORT.md)

## 配置文件

相关配置文件：
- `.env.prod` - 生产环境配置
- `docker-compose.yml` - Docker 服务配置
- `config/settings.py` - 应用配置加载

## 关键配置说明

### Business 数据源（JSON）

```python
BUSINESS_DATA_SOURCE=json
```

- 从 JSON 文件读取业务数据
- 文件位置: `/app/data/dianshu_data/jsons/`
- 包含: user.json, dataset.json, task.json, api_order.json 等

### Matomo 数据源（Database）

```python
MATOMO_DATA_SOURCE=database
```

- 从 MySQL 数据库读取 Matomo 分析数据
- 配置: `MATOMO_DB_HOST`, `MATOMO_DB_NAME`, etc.
- 连接: host.docker.internal:3306/matomo

## 故障排查

### JSON 文件未找到

检查文件是否存在：
```bash
docker compose exec recommendation-api ls -la /app/data/dianshu_data/jsons/
```

### 数据库连接失败

检查数据库配置：
```bash
docker compose exec recommendation-api env | grep MATOMO_DB
```

### 配置未生效

重启服务：
```bash
docker compose restart recommendation-api
```

## 更新日期

2025-12-23
