## 推荐链路优化 TODO（下一阶段）

1. **SOTA 特征/模型升级**
   - [ ] **丰富用户画像**：将 Feature V2 中的用户行业、公司、偏好标签、活跃度等特征纳入 `_prepare_ranking_dataset`，并确保线上也能实时更新。
   - [ ] **多模/语义信号**：对 `dataset_features` 增加文本/图像 Embedding（可通过 `dataset_image_embeddings` 或外部模型），并在召回/排序中使用。
   - [ ] **多目标训练**：在 `train_models` 中引入 CVR/GMV 相关标签（利用 Matomo 转化数据），考虑多任务或加权训练，提升商业指标。
   - [ ] **实时/增量特征**：设计 Kafka/Flink（或 Redis stream）驱动的近实时特征更新方案，减少离线延迟。

2. **召回与排序策略进化**
   - [ ] **Embedding 召回**：实现双塔/Graph Embedding（Faiss/HNSW）召回模块，补充现有 CF/TF-IDF。
   - [ ] **粗排/重排**：在 `recommendation_api` 中加入粗排或 rerank（如小型 Transformer），提升候选质量。
   - [ ] **场景化模型**：按入口（detail/API/home）训练独立模型或在排序中引入场景特征，让不同业务场景定制化。
   - [ ] **实验体系升级**：自动生成实验计划、显著性检测和灰度策略（如 Thompson Sampling），减少手工调参数。

