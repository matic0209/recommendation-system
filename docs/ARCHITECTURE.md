# 推荐系统架构文档（数据集详情页推荐最小可用版）

## 1. 数据层

- **数据源**
  - 业务库（MySQL）：`user`、`dataset`、`order_tab`、`api_order`、`dataset_image`
  - 行为库（Matomo MySQL）：`matomo_log_visit`、`matomo_log_link_visit_action`、`matomo_log_action`、`matomo_log_conversion`
- **处理流程**
  - 订单与行为日志融合，生成用户-数据集交互表（浏览/购买打分 + 时间衰减）
  - `dataset` 表清洗文本、标签、行业、价格等字段，构建内容向量与数值特征
  - `user` 表提取行业、价格区间、消费状态等画像信息
- **产出文件**
  - `data/interactions.parquet`
  - `data/dataset_features.parquet`
  - `data/user_profile.parquet`

## 2. 模型层

- **行为相似（ItemCF）**：依据交互表计算物品共现/余弦相似度，输出 `models/item_sim_behavior.pkl`
- **内容相似**：TF-IDF + 余弦相似度（标题、描述、标签等），输出 `models/item_sim_content.pkl`
- **热门榜单**：按最近浏览/成交统计生成 `models/top_items.json`
- **训练脚本**：`pipeline/train_models.py` 读取特征 → 训练模型 → 持久化到 `models/`

## 3. 服务层

- **框架**：FastAPI（使用 Uvicorn 部署）
- **启动阶段加载**：行为相似矩阵、内容相似矩阵、热门榜、用户画像
- **API 接口**
  - `GET /recommend/detail/{dataset_id}`（可选 `user_id`、`n`）
  - `GET /similar/{dataset_id}`
- **推荐策略**
  - 登录用户：优先行为相似结果，结合用户画像加权，缺量时补充内容相似与热门
  - 未登录用户：内容相似 + 热门榜单
- **返回字段**：`dataset_id`、`title`、`cover_image`、`price`、`score`、`reason`

## 4. 监控与评估

- 将推荐请求信息写入 Matomo 自定义维度（算法版本、请求 ID）
- `pipeline/evaluate.py` 对照 Matomo 行为日志统计点击/转化效果
- 评估结果输出为 CSV 报表，支持后续迭代分析

## 5. 行动顺序

1. 重组工程目录（`app/`、`pipeline/`、`models/`、`config/`、`tests/` 等），更新 `requirements.txt`
2. 编写 `pipeline/extract_load.py`：连接两套数据库，抽取原始表，保存为 Parquet
3. 实现 `pipeline/build_features.py`：构建交互表、数据集特征、用户画像
4. 开发 `pipeline/train_models.py`：训练 ItemCF、内容相似、热门榜，持久化模型
5. 搭建 FastAPI 服务 `app/main.py`：加载模型并完成 `/recommend/detail`、`/similar` 接口逻辑
6. 添加运行脚本（Makefile 或 `scripts/`）串联 Extract → Features → Train → Serve，并编写基础测试
7. 手动跑 pipeline、启动服务，调用接口验证输出，并使用 `pipeline/evaluate.py` 生成评估报告
8. 更新 README，说明配置、运行步骤及 API 示例；准备 Dockerfile/部署脚本（可选）以支撑上线
