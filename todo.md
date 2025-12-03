## 推荐链路优化 TODO

1. **打通 request_id 标签闭环**  
   - [ ] 在 `pipeline.aggregate_matomo_events` 输出中新增 CTR/CVR 统计字段，按 `(dataset_id, pos)` 聚合；  
   - [ ] Weekly DAG 里使用这些曝光/点击数据生成 ranking labels，替换当前只看交互次数的方式。

2. **强化特征工程与模型**  
   - [ ] 构建统一特征库：价格区间、行业、卖家信誉、文本嵌入、Matomo 行为率等；  
   - [ ] 将当前混合推荐的随机森林 meta model 替换为 GBDT（LightGBM/XGBoost），并补齐缺失值处理与冷启动策略。

3. **上下文与实验能力**  
   - [ ] Matomo 记录已包含 variant/pos，API 层需要透传 request context（入口、设备），让模型能按场景调权；  
   - [ ] 在推荐 API 和前端之间增加实验参数（自定义维度），配合 Airflow 报表评估各算法权重，形成自动化调权流程。

4. **监控与验证**  
   - [ ] 为 `pipeline.aggregate_matomo_events` 补充单元测试，确保新维度变更时报警；  
   - [ ] Airflow DAG 成功后自动抽样校验 `data/processed` 文件行数，避免出现 0 行仍继续训练。

