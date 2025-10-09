# 数据处理与推荐流程概览

本文档总结当前推荐系统从数据到服务的完整流程，便于开发与运营人员理解各阶段职责。

## 1. 数据抽取（pipeline/extract_load.py）

1. 从业务库 `dianshu_backend` 导出：`user`、`dataset`、`order_tab`、`api_order`、`dataset_image`
2. 从 Matomo 库 `matomo` 导出：`matomo_log_visit`、`matomo_log_link_visit_action`、`matomo_log_action`、`matomo_log_conversion`
3. 导出结果写入 `data/business/*.parquet` 与 `data/matomo/*.parquet`

> 配置通过 `.env` 或环境变量提供，脚本会自动读取。

## 2. 特征构建（pipeline/build_features.py）

### 2.1 用户-数据集交互

- `order_tab`：使用 `create_user` → 用户 ID、`dataset_id` → 数据集 ID、`price`、`create_time`
- `api_order`：将 `creator_id` → 用户 ID、`api_id` → 数据集 ID、其余字段同上
- 对价格取 `log(1 + price)` 作为权重，按用户-数据集汇总，并记录最近一次交互时间
- 结果保存至 `data/processed/interactions.parquet`

### 2.2 数据集特征

- 从 `dataset` 表提取 `dataset_id`、`dataset_name`、`description`、`tag`、`price`、`create_company_name`
- 输出 `data/processed/dataset_features.parquet`

### 2.3 用户画像

- 从 `user` 表提取 `company_name`、`province`、`city`、`is_consumption`
- 输出 `data/processed/user_profile.parquet`

## 3. 模型训练（pipeline/train_models.py）

1. **行为相似模型**：基于交互表计算物品共现概率 → `models/item_sim_behavior.pkl`
2. **内容相似模型**：对数据集描述/标签做 TF-IDF → `models/item_sim_content.pkl`
3. **热门榜单**：按交互权重排序 → `models/top_items.json`

## 4. 在线服务（app/main.py）

- 启动时加载上述模型及数据集元信息- 同时载入 `interactions.parquet` 构建 `user_history` 与标签偏好，用于登录用户的个性化重排

- 关键接口：
  - `GET /similar/{dataset_id}`：返回行为/内容相似集合
  - `GET /recommend/detail/{dataset_id}`：详情页推荐，自动融合行为相似、内容相似、热门兜底
- 返回字段包含推荐理由（behavior/content/popular），便于前端展示或调试

## 5. 推荐效果评估

- `pipeline/evaluate.py`（待实现）：建议对比推荐结果与 Matomo 行为日志，统计 CTR/CVR
- 推荐请求应埋点到 Matomo 自定义维度，便于线上监控

## 6. 常用命令

```bash
# 评估推荐效果
python -m pipeline.evaluate
```


## 7. 评估与监控

- 运行 `python -m pipeline.evaluate` 生成 Matomo 行为与转化统计，结果写入 `data/evaluation/dataset_metrics.csv` 与 `summary.json`。
- 报告包含每个数据集的浏览量、转化次数、转化率，可辅助验证推荐覆盖情况。
- 建议将该脚本纳入日常调度，并结合可视化/告警工具。


```bash
# 一键跑完整流水线
tscripts/run_pipeline.sh

# 单独重训模型（数据处理已完成的情况下）
.venv/bin/python pipeline/train_models.py

# 启动接口服务
uvicorn app.main:app --reload
```

该流程确保 API 订单 (`api_order`) 与常规订单统一纳入交互数据，为后续推荐效果提供更全面的依据。
