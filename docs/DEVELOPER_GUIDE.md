# 推荐系统开发人员使用指南

## 1. 概览

该推荐系统面向数据集详情页，提供“看了此数据集的用户还会看哪些数据集”的推荐结果。整体分为四个核心模块：

1. 数据抽取 (`pipeline/extract_load.py`)
2. 特征构建 (`pipeline/build_features.py`)
3. 模型训练 (`pipeline/train_models.py`)
4. 在线服务 (`app/main.py`)

项目的详细架构参见 `docs/ARCHITECTURE.md`，本文档聚焦开发/运行流程。

```
app/                FastAPI 服务
config/             配置模块（环境变量读取、路径定义）
pipeline/           数据处理与模型训练脚本
models/             产出的模型文件
scripts/            辅助脚本（如 run_pipeline.sh）
data/               原始/处理后的数据（运行时生成）
```

## 2. 环境准备

### 2.1 拉起虚拟环境

```bash
python3 -m virtualenv .venv
source .venv/bin/activate  # Windows 下使用 .venv\Scriptsctivate
pip install --upgrade pip
pip install -r requirements.txt
```

> 如果所在系统未安装 `virtualenv`，先执行 `python3 -m pip install --user virtualenv`。

### 2.2 数据库配置

推荐通过环境变量提供业务库与 Matomo 库的连接信息：

```bash
export BUSINESS_DB_HOST=localhost
export BUSINESS_DB_PORT=3306
export BUSINESS_DB_NAME=dianshu_backend
export BUSINESS_DB_USER=root
export BUSINESS_DB_PASSWORD=***

export MATOMO_DB_HOST=localhost
export MATOMO_DB_PORT=3306
export MATOMO_DB_NAME=matomo
export MATOMO_DB_USER=root
export MATOMO_DB_PASSWORD=***
```

如需持久化，可拷贝 `.env.example` 新建 `.env`，并在 shell 中 `source .env`。

## 3. 数据与模型流水线

流水线包含抽取 → 构建特征 → 训练模型三个步骤。可以手动逐步执行，也可以使用脚本一次性运行。

### 3.1 单步执行

```bash
# 仅检查将要抽取的表（不连接数据库）
.venv/bin/python -m pipeline.extract_load --dry-run

# 真正抽取并写入 data/business / data/matomo
.venv/bin/python -m pipeline.extract_load

# 构建交互数据、数据集特征、用户画像
.venv/bin/python -m pipeline.build_features

# 训练行为/内容相似模型与热门榜
.venv/bin/python pipeline/train_models.py
```

管线生成的文件：

- `data/business/*.parquet`、`data/matomo/*.parquet`：原始表导出
- `data/processed/interactions.parquet`：用户-数据集交互权重
- `data/processed/dataset_features.parquet`：数据集内容特征
- `models/item_sim_behavior.pkl`：ItemCF 行为相似矩阵
- `models/item_sim_content.pkl`：内容相似矩阵
- `models/top_items.json`：热门数据集列表

### 3.2 使用脚本一键运行

### 3.3 数据来源说明

- `order_tab`：以 `create_user` (用户 ID)、`dataset_id`、`price`、`create_time` 构造行为权重。
- `api_order`：将 `creator_id` 视为用户 ID，`api_id` 等同于 `dataset_id`，其余字段与 `order_tab` 同处理后合并至交互表。
- 行为权重统一采用 \(\log(1+price)\) 并按用户-数据集累加，同时记录最近一次行为时间。

``scripts/run_pipeline.sh`` 会自动执行上述合并逻辑，生成 `data/processed/interactions.parquet`，供模型训练使用。

```bash
scripts/run_pipeline.sh --dry-run  # 查看计划
scripts/run_pipeline.sh            # 执行全流程
```

脚本内部按顺序调用 `extract_load` → `build_features` → `train_models`。

## 4. 启动在线服务

1. 确保模型文件已生成（`models/` 目录存在）
2. 激活虚拟环境后运行：

```bash
uvicorn app.main:app --reload --port 8000
```

成功启动后默认提供以下接口：

- `GET /health`：存活检测
- `GET /similar/{dataset_id}?limit=10`
- `GET /recommend/detail/{dataset_id}?user_id=123&limit=10`

示例调用：

```bash
curl 'http://127.0.0.1:8000/recommend/detail/13196?user_id=123&limit=5'
```

返回字段：

```json
{
  "dataset_id": 13196,
  "recommendations": [
    {
      "dataset_id": 13004,
      "title": "示例数据集",
      "price": 99.0,
      "cover_image": "https://...",
      "score": 0.67,
      "reason": "behavior"  // 可能为 behavior/content/popular
    }
  ]
}
```

如果数据集中缺少相关信息，接口会自动以热门榜兜底。

## 5. 常见问题

| 问题 | 处理办法 |
| ---- | -------- |
| `ModuleNotFoundError: No module named 'config'` | 运行脚本时使用 `python -m package.module` 或确保当前路径为项目根目录 |
| 生成的特征为空 | 确认前一步抽取数据是否成功；确保数据库中存在对应表数据 |
| FastAPI 返回 404 | 检查请求的 `dataset_id` 是否在模型矩阵中；必要时重新跑管线 |

## 6. 后续扩展建议

- 在 `pipeline/evaluate.py` 中实现真实的推荐效果评估，结合 Matomo 行为统计 CTR/CVR
- 引入调度工具（如 Airflow）定期执行 pipeline
- 丰富特征（标签、行业、价格段等）、引入更强的排序模型
- 将推荐结果埋点写回 Matomo 自定义维度，实现线上效果监控

如有任何疑问，可参考 `docs/ARCHITECTURE.md` 或联系推荐系统开发负责人。

## 7. 推荐效果评估

运行 `python -m pipeline.evaluate` 可基于 Matomo 行为日志计算数据集的浏览/转化指标，结果输出至 `data/evaluation/`，包含：
- dataset_metrics.csv：各数据集的浏览量、转化次数、转化率、收入等汇总
- summary.json：整体指标（总浏览、总转化、平均转化率等）

建议在生成推荐结果后配合该报告验证推荐覆盖度和业务指标表现。
