## 推荐链路优化 TODO（下一阶段）

1. **算法与特征性能提升**
   - [ ] Embedding 召回（双塔/Graph + Faiss/HNSW）
   - [ ] 粗排/重排（Transformer、小模型 rerank）
   - [ ] 场景化模型（按入口定制、特征动态）
   - [ ] 实时/增量特征管道（Kafka/Flink/Redis Stream）

2. **多模/多目标能力**
   - [x] 丰富用户画像（Feature V2 → ranking）
   - [x] 多目标训练（CTR + CVR/GMV）
   - [ ] 多模信号扩展：图像/音频等后续引入

3. **实验与自动化闭环**
   - [ ] 实验自动化：显著性检测、Thompson Sampling、自动 rollout
   - [ ] 监控闭环：实时 CTR/GMV 看板，pipeline 必要报警/自恢复
