# 生产环境优化 TODO 清单

**文档创建时间**：2025-10-10
**系统**：数据交易推荐系统
**负责人**：待分配

---

## 📋 总览

| 优先级 | 类别 | 任务数 | 预计工时 | 预期收益 |
|--------|------|--------|----------|----------|
| P0 | 数据库优化 | 2 | 2-3天 | Pipeline速度↑60-80% |
| P1 | 缓存优化 | 2 | 1-2周 | 延迟↓20-30ms，命中率↑15-25% |
| P2 | 模型优化 | 3 | 3-4周 | 召回耗时↓40-60%，推理速度↑2-3倍 |
| P3 | 部署优化 | 3 | 4-6周 | 自动扩缩容，可用性↑ |
| P4 | 监控增强 | 2 | 持续 | 问题及时发现，降低故障时间 |
| P5 | 成本优化 | 2 | 2-3个月 | 成本↓60-70% |

**总计**：14个优化项

---

## 🔴 P0 - 数据库性能优化（立即执行）

### ✅ TODO-01: MySQL 索引优化

**优先级**：🔴 P0 - Critical
**预计工时**：1-2天
**负责人**：[ ]
**依赖项**：无

#### 任务描述
为业务库和 Matomo 库的时间列添加索引，优化 CDC 增量抽取性能。

#### 实施步骤
- [ ] **步骤1**：在测试环境验证索引创建
  ```sql
  -- 在从库或测试库执行
  EXPLAIN SELECT * FROM user WHERE update_time > '2025-10-01';
  ```

- [ ] **步骤2**：创建业务库索引
  ```sql
  -- 连接到业务库
  USE dianshu_backend;

  -- 添加索引（非高峰期执行）
  ALTER TABLE user ADD INDEX idx_update_time (update_time);
  ALTER TABLE dataset ADD INDEX idx_update_time (update_time);
  ALTER TABLE order_tab ADD INDEX idx_create_time (create_time);
  ALTER TABLE api_order ADD INDEX idx_create_time (create_time);
  ALTER TABLE dataset_image ADD INDEX idx_update_time (update_time);

  -- 验证索引是否创建成功
  SHOW INDEX FROM user WHERE Key_name = 'idx_update_time';
  ```

- [ ] **步骤3**：创建 Matomo 库索引
  ```sql
  -- 连接到 Matomo 库
  USE matomo;

  -- 添加索引
  ALTER TABLE matomo_log_visit
    ADD INDEX idx_visit_time (visit_last_action_time);

  ALTER TABLE matomo_log_link_visit_action
    ADD INDEX idx_server_time (server_time);

  ALTER TABLE matomo_log_conversion
    ADD INDEX idx_server_time (server_time);

  -- 联合索引（支持用户行为查询）
  ALTER TABLE matomo_log_visit
    ADD INDEX idx_user_time (idvisitor, visit_last_action_time);
  ```

- [ ] **步骤4**：验证性能提升
  ```bash
  # 执行索引优化前的 pipeline 基线测试
  time python -m pipeline.extract_load

  # 记录执行时间：____ 秒

  # 创建索引后重新测试
  time python -m pipeline.extract_load

  # 记录执行时间：____ 秒
  # 计算提升比例：____ %
  ```

- [ ] **步骤5**：监控索引使用情况
  ```sql
  -- 查看索引使用统计（运行1-2天后）
  SELECT TABLE_NAME, INDEX_NAME,
         ROWS_READ, ROWS_INSERTED, ROWS_UPDATED
  FROM performance_schema.table_io_waits_summary_by_index_usage
  WHERE INDEX_NAME LIKE 'idx_%time';
  ```

#### 验收标准
- [x] 所有索引创建成功，无错误
- [x] CDC 增量抽取时间缩短 ≥50%
- [x] 数据库写入性能无明显下降（<10%）
- [x] 索引在 EXPLAIN 分析中被正确使用

#### 回滚方案
```sql
-- 如果索引导致性能问题，可快速删除
DROP INDEX idx_update_time ON user;
DROP INDEX idx_update_time ON dataset;
-- ... 其他索引
```

#### 参考文档
- 位置：`pipeline/extract_load.py:114-124`（增量抽取逻辑）
- 文档：MySQL 索引优化最佳实践

---

### ✅ TODO-02: 数据库连接池优化

**优先级**：🔴 P0 - Critical
**预计工时**：0.5-1天
**负责人**：[ ]
**依赖项**：无

#### 任务描述
优化 SQLAlchemy 连接池配置，防止连接泄漏和超时。

#### 实施步骤
- [ ] **步骤1**：修改 `config/settings.py`
  ```python
  # 在 DatabaseConfig 类中修改 sqlalchemy_url 方法
  def sqlalchemy_url(
      self,
      pool_size: int = 10,           # 连接池大小
      max_overflow: int = 20,        # 最大溢出连接
      pool_recycle: int = 3600,      # 连接回收时间（秒）
      pool_pre_ping: bool = True     # 连接前 ping 测试
  ) -> str:
      base_url = (
          f"mysql+pymysql://{self.user}:{self.password}"
          f"@{self.host}:{self.port}/{self.name}"
      )
      params = (
          f"?pool_size={pool_size}"
          f"&max_overflow={max_overflow}"
          f"&pool_recycle={pool_recycle}"
          f"&pool_pre_ping={'true' if pool_pre_ping else 'false'}"
      )
      return base_url + params
  ```

- [ ] **步骤2**：更新 `pipeline/extract_load.py`
  ```python
  # 在 _export_table 函数中（约 182 行）
  # 修改 engine 创建逻辑
  from sqlalchemy.pool import QueuePool

  engine = create_engine(
      engine_url,
      poolclass=QueuePool,
      pool_size=10,
      max_overflow=20,
      pool_recycle=3600,
      pool_pre_ping=True,
      echo=False  # 生产环境关闭 SQL 日志
  )
  ```

- [ ] **步骤3**：添加连接池监控
  ```python
  # 在 pipeline/extract_load.py 添加监控函数
  def log_pool_status(engine):
      pool = engine.pool
      LOGGER.info(
          "Connection pool status: size=%d, checked_in=%d, "
          "checked_out=%d, overflow=%d",
          pool.size(),
          pool.checkedin(),
          pool.checkedout(),
          pool.overflow()
      )
  ```

- [ ] **步骤4**：压力测试
  ```bash
  # 并发执行 pipeline，观察连接池状态
  for i in {1..5}; do
    python -m pipeline.extract_load &
  done
  wait

  # 检查是否有连接泄漏
  # MySQL: SHOW PROCESSLIST;
  ```

#### 验收标准
- [x] 连接池配置正确生效
- [x] 并发执行无连接泄漏
- [x] 连接回收机制正常工作
- [x] pool_pre_ping 避免 "MySQL has gone away" 错误

#### 环境变量配置
```bash
# 可选：在 .env 中配置连接池参数
DB_POOL_SIZE=10
DB_MAX_OVERFLOW=20
DB_POOL_RECYCLE=3600
```

---

## 🟠 P1 - 缓存策略优化（2周内）

### ✅ TODO-03: Redis 分层缓存架构

**优先级**：🟠 P1 - High
**预计工时**：1周
**负责人**：[ ]
**依赖项**：无

#### 任务描述
实现 Redis 三层缓存架构：热缓存、特征库、冷缓存。

#### 实施步骤
- [ ] **步骤1**：创建缓存抽象层 `app/tiered_cache.py`
  ```python
  """分层缓存实现"""
  import json
  from typing import Optional, Any
  import redis.asyncio as aioredis

  class TieredCache:
      def __init__(self, redis_url: str):
          self.hot = aioredis.from_url(f"{redis_url}/0")      # 热缓存
          self.feature = aioredis.from_url(f"{redis_url}/1")  # 特征库
          self.cold = aioredis.from_url(f"{redis_url}/2")     # 冷缓存

      async def get_recommendation(
          self,
          key: str
      ) -> Optional[dict]:
          """获取推荐结果，自动提升热度"""
          # 1. 先查热缓存
          result = await self.hot.get(key)
          if result:
              return json.loads(result)

          # 2. 查冷缓存
          result = await self.cold.get(key)
          if result:
              # 提升到热缓存（5分钟）
              await self.hot.setex(key, 300, result)
              return json.loads(result)

          return None

      async def set_recommendation(
          self,
          key: str,
          value: dict,
          hot: bool = True
      ):
          """保存推荐结果"""
          data = json.dumps(value)
          if hot:
              # 热数据：5分钟 hot + 1小时 cold
              await self.hot.setex(key, 300, data)
              await self.cold.setex(key, 3600, data)
          else:
              # 冷数据：仅保存到 cold（24小时）
              await self.cold.setex(key, 86400, data)

      async def get_feature(
          self,
          key: str
      ) -> Optional[dict]:
          """获取特征数据（从 feature db）"""
          result = await self.feature.get(key)
          return json.loads(result) if result else None

      async def set_feature(
          self,
          key: str,
          value: dict,
          ttl: int = 3600
      ):
          """保存特征数据（1小时 TTL）"""
          await self.feature.setex(key, ttl, json.dumps(value))
  ```

- [ ] **步骤2**：集成到 `app/main.py`
  ```python
  # 修改 load_models() 函数
  from app.tiered_cache import TieredCache

  @app.on_event("startup")
  def load_models() -> None:
      # ... 现有代码 ...

      # 替换原有的 cache 初始化
      redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
      app.state.tiered_cache = TieredCache(redis_url)

      LOGGER.info("Tiered cache initialized (hot/feature/cold)")
  ```

- [ ] **步骤3**：修改推荐接口使用分层缓存
  ```python
  # 在 get_similar() 函数中
  @app.get("/similar/{dataset_id}", response_model=SimilarResponse)
  async def get_similar(...):
      tiered_cache = getattr(state, "tiered_cache", None)

      if tiered_cache:
          cache_key = f"similar:{dataset_id}:{limit}"
          cached = await tiered_cache.get_recommendation(cache_key)
          if cached:
              metrics_tracker.track_cache_hit()
              return SimilarResponse(**cached)

          # ... 计算推荐 ...

          # 保存结果（判断是否为热点）
          is_hot = dataset_id in state.popular[:100]  # 热门物品
          await tiered_cache.set_recommendation(
              cache_key,
              response.dict(),
              hot=is_hot
          )
  ```

- [ ] **步骤4**：配置 Redis 实例
  ```yaml
  # docker-compose.yml 修改
  redis:
    image: redis:7-alpine
    command: >
      redis-server
      --appendonly yes
      --maxmemory 4gb
      --databases 16
      --save 900 1
      --save 300 10
    # 为不同 db 配置不同策略（需要 Redis 7+）
    # 通过 CONFIG SET 动态配置
  ```

  ```bash
  # 启动后配置各 db 的淘汰策略
  redis-cli CONFIG SET maxmemory-policy-db-0 allkeys-lru   # 热缓存
  redis-cli CONFIG SET maxmemory-policy-db-1 volatile-lru  # 特征库
  redis-cli CONFIG SET maxmemory-policy-db-2 allkeys-lfu   # 冷缓存
  ```

- [ ] **步骤5**：监控缓存命中率
  ```python
  # app/metrics.py 添加分层缓存指标
  cache_hit_by_tier = Counter(
      'cache_hit_by_tier_total',
      'Cache hits by tier',
      ['tier']  # hot, feature, cold
  )
  ```

#### 验收标准
- [x] 三层缓存正常工作
- [x] 热点数据自动提升到热缓存
- [x] 缓存命中率提升 ≥15%
- [x] P95 延迟降低 ≥20ms

#### 监控指标
```promql
# Grafana 查询
# 各层缓存命中率
sum(rate(cache_hit_by_tier_total{tier="hot"}[5m])) /
sum(rate(cache_requests_total[5m]))
```

---

### ✅ TODO-04: 本地内存缓存

**优先级**：🟠 P1 - High
**预计工时**：2-3天
**负责人**：[ ]
**依赖项**：无

#### 任务描述
对热榜等高频访问数据增加进程内缓存，减少 Redis 访问。

#### 实施步骤
- [ ] **步骤1**：创建本地缓存模块 `app/local_cache.py`
  ```python
  """本地内存缓存（进程级别）"""
  from functools import lru_cache
  from datetime import datetime, timedelta
  from typing import Optional, Any
  import threading

  class LocalCache:
      """支持 TTL 的本地缓存"""
      def __init__(self, ttl_seconds: int = 60):
          self.cache = {}
          self.ttl = ttl_seconds
          self.lock = threading.Lock()

      def get(self, key: str) -> Optional[Any]:
          with self.lock:
              entry = self.cache.get(key)
              if not entry:
                  return None

              value, expire_time = entry
              if datetime.now() > expire_time:
                  del self.cache[key]
                  return None

              return value

      def set(self, key: str, value: Any, ttl: Optional[int] = None):
          expire_time = datetime.now() + timedelta(
              seconds=ttl or self.ttl
          )
          with self.lock:
              self.cache[key] = (value, expire_time)

      def clear(self):
          with self.lock:
              self.cache.clear()

      def size(self) -> int:
          return len(self.cache)

  # 全局单例
  _local_cache = LocalCache(ttl_seconds=60)

  def get_local_cache() -> LocalCache:
      return _local_cache
  ```

- [ ] **步骤2**：在 `app/main.py` 中使用
  ```python
  from app.local_cache import get_local_cache

  @app.get("/hot/trending")
  async def get_trending(...):
      local_cache = get_local_cache()

      # 1. 先查本地缓存（60秒）
      cache_key = f"hot:trending:{timeframe}:{limit}"
      cached = local_cache.get(cache_key)
      if cached:
          return cached

      # 2. 再查 Redis
      # ... 现有逻辑 ...

      # 3. 保存到本地缓存
      result = {"timeframe": timeframe, "items": items}
      local_cache.set(cache_key, result, ttl=60)

      return result
  ```

- [ ] **步骤3**：为模型元数据添加缓存
  ```python
  # 缓存数据集元数据（减少 SQLite 查询）
  @lru_cache(maxsize=10000)
  def get_dataset_metadata(dataset_id: int) -> dict:
      """LRU 缓存数据集元数据"""
      return state.metadata.get(dataset_id, {})

  # 缓存用户画像（减少特征查询）
  @lru_cache(maxsize=5000)
  def get_user_profile(user_id: int) -> dict:
      """LRU 缓存用户画像"""
      return state.user_profiles.get(user_id, {})
  ```

- [ ] **步骤4**：添加缓存预热
  ```python
  @app.on_event("startup")
  def load_models() -> None:
      # ... 现有代码 ...

      # 缓存预热：加载热门数据
      local_cache = get_local_cache()

      # 预热热门榜单
      for timeframe in ["1h", "24h"]:
          hot_items = get_hot_items_from_redis(timeframe, 20)
          local_cache.set(
              f"hot:trending:{timeframe}:20",
              {"timeframe": timeframe, "items": hot_items},
              ttl=300  # 5分钟
          )

      LOGGER.info("Local cache preheated with %d entries",
                  local_cache.size())
  ```

- [ ] **步骤5**：添加监控指标
  ```python
  # app/metrics.py
  local_cache_size = Gauge(
      'local_cache_size',
      'Number of entries in local cache'
  )

  local_cache_hit_rate = Gauge(
      'local_cache_hit_rate',
      'Local cache hit rate'
  )

  # 定期更新指标
  @app.middleware("http")
  async def update_cache_metrics(request, call_next):
      response = await call_next(request)

      local_cache = get_local_cache()
      local_cache_size.set(local_cache.size())

      return response
  ```

#### 验收标准
- [x] 本地缓存正常工作
- [x] 热榜接口延迟降低 ≥50%
- [x] Redis 访问量减少 ≥30%
- [x] 内存使用增长 <100MB

#### 注意事项
⚠️ **多实例部署问题**：本地缓存在多个 pod 之间不共享，可能导致数据不一致
**解决方案**：
1. 仅缓存准实时数据（热榜、统计数据）
2. TTL 设置较短（≤60秒）
3. 通过 Redis pub/sub 实现缓存失效通知

---

## 🟡 P2 - 模型和召回优化（1个月）

### ✅ TODO-05: 并行召回实现

**优先级**：🟡 P2 - Medium
**预计工时**：1周
**负责人**：[ ]
**依赖项**：无

#### 任务描述
将多路召回从串行改为异步并行执行，提升召回速度。

#### 实施步骤
- [ ] **步骤1**：重构召回函数为异步
  ```python
  # app/recall_parallel.py（新建文件）
  import asyncio
  from typing import Dict, List, Tuple

  async def behavior_recall(
      target_id: int,
      behavior_index: Dict,
      limit: int,
      weight: float
  ) -> Tuple[Dict[int, float], Dict[int, str]]:
      """行为召回（异步）"""
      scores = {}
      reasons = {}

      neighbors = behavior_index.get(target_id, {})
      for item_id, score in neighbors.items():
          scores[int(item_id)] = float(score) * weight
          reasons[int(item_id)] = "behavior"

      return scores, reasons

  async def content_recall(
      target_id: int,
      content_index: Dict,
      limit: int,
      weight: float
  ) -> Tuple[Dict[int, float], Dict[int, str]]:
      """内容召回（异步）"""
      # ... 类似实现 ...
      pass

  async def vector_recall(
      target_id: int,
      vector_index: Dict,
      limit: int,
      weight: float
  ) -> Tuple[Dict[int, float], Dict[int, str]]:
      """向量召回（异步）"""
      # ... 类似实现 ...
      pass

  async def parallel_recall(
      target_id: int,
      bundle: ModelBundle,
      limit: int,
      weights: Dict[str, float]
  ) -> Tuple[Dict[int, float], Dict[int, str]]:
      """并行执行多路召回"""
      tasks = [
          behavior_recall(
              target_id, bundle.behavior, limit, weights['behavior']
          ),
          content_recall(
              target_id, bundle.content, limit, weights['content']
          ),
          vector_recall(
              target_id, bundle.vector, limit, weights['vector']
          ),
      ]

      # 并行执行
      results = await asyncio.gather(*tasks)

      # 合并结果
      merged_scores = {}
      merged_reasons = {}

      for scores, reasons in results:
          for item_id, score in scores.items():
              if item_id not in merged_scores:
                  merged_scores[item_id] = score
                  merged_reasons[item_id] = reasons[item_id]
              else:
                  merged_scores[item_id] += score
                  # 合并原因
                  if reasons[item_id] not in merged_reasons[item_id]:
                      merged_reasons[item_id] += f"+{reasons[item_id]}"

      return merged_scores, merged_reasons
  ```

- [ ] **步骤2**：在 `app/main.py` 中使用
  ```python
  from app.recall_parallel import parallel_recall

  @app.get("/similar/{dataset_id}")
  async def get_similar(...):
      # 替换原有的 _combine_scores_with_weights
      scores, reasons = await parallel_recall(
          target_id=dataset_id,
          bundle=bundle,
          limit=limit,
          weights=channel_weights
      )

      # ... 后续处理 ...
  ```

- [ ] **步骤3**：性能基准测试
  ```python
  # tests/benchmark_recall.py
  import time
  import asyncio

  async def benchmark_recall():
      # 串行召回
      start = time.perf_counter()
      for _ in range(100):
          await sequential_recall(dataset_id=123)
      sequential_time = time.perf_counter() - start

      # 并行召回
      start = time.perf_counter()
      for _ in range(100):
          await parallel_recall(dataset_id=123)
      parallel_time = time.perf_counter() - start

      print(f"Sequential: {sequential_time:.3f}s")
      print(f"Parallel: {parallel_time:.3f}s")
      print(f"Speedup: {sequential_time/parallel_time:.2f}x")

  if __name__ == "__main__":
      asyncio.run(benchmark_recall())
  ```

- [ ] **步骤4**：添加监控
  ```python
  # app/metrics.py
  recall_duration_by_channel = Histogram(
      'recall_duration_seconds',
      'Duration of recall by channel',
      ['channel']  # behavior, content, vector
  )
  ```

#### 验收标准
- [x] 并行召回功能正常
- [x] 召回结果与串行版本一致（A/B 测试）
- [x] 召回耗时降低 ≥40%
- [x] CPU 使用率无显著上升

---

### ✅ TODO-06: Faiss HNSW 索引优化

**优先级**：🟡 P2 - Medium
**预计工时**：3-5天
**负责人**：[ ]
**依赖项**：TODO-05（可并行）

#### 任务描述
将 Faiss 向量索引从 IVF 替换为 HNSW，提升查询速度和召回质量。

#### 实施步骤
- [ ] **步骤1**：修改 `pipeline/vector_recall_faiss.py`
  ```python
  import faiss
  import numpy as np

  def build_hnsw_index(
      embeddings: np.ndarray,
      M: int = 32,              # 每个节点的连接数
      efConstruction: int = 200, # 构建时的搜索深度
      efSearch: int = 64         # 查询时的搜索深度
  ):
      """构建 HNSW 索引"""
      dimension = embeddings.shape[1]

      # 创建 HNSW 索引
      index = faiss.IndexHNSWFlat(dimension, M)
      index.hnsw.efConstruction = efConstruction

      # 添加向量
      index.add(embeddings)

      # 设置查询参数
      index.hnsw.efSearch = efSearch

      return index

  def search_hnsw_index(
      index: faiss.IndexHNSWFlat,
      query: np.ndarray,
      k: int = 50
  ) -> Tuple[np.ndarray, np.ndarray]:
      """在 HNSW 索引中搜索"""
      distances, indices = index.search(query, k)
      return distances, indices
  ```

- [ ] **步骤2**：对比 IVF 和 HNSW 性能
  ```python
  # tests/test_faiss_index.py
  def benchmark_faiss_index():
      # 准备测试数据
      embeddings = np.random.rand(10000, 128).astype('float32')
      queries = np.random.rand(100, 128).astype('float32')

      # IVF 索引
      ivf_index = build_ivf_index(embeddings)
      start = time.time()
      for q in queries:
          ivf_index.search(q.reshape(1, -1), 50)
      ivf_time = time.time() - start

      # HNSW 索引
      hnsw_index = build_hnsw_index(embeddings)
      start = time.time()
      for q in queries:
          hnsw_index.search(q.reshape(1, -1), 50)
      hnsw_time = time.time() - start

      print(f"IVF: {ivf_time:.3f}s, HNSW: {hnsw_time:.3f}s")
      print(f"Speedup: {ivf_time/hnsw_time:.2f}x")
  ```

- [ ] **步骤3**：评估召回质量
  ```python
  # 计算 Recall@K
  def evaluate_recall_quality(index, ground_truth, k=50):
      recalls = []
      for query_id, true_neighbors in ground_truth.items():
          query_vec = get_embedding(query_id)
          _, predicted = index.search(query_vec, k)

          hit = len(set(predicted[0]) & set(true_neighbors[:k]))
          recall = hit / min(k, len(true_neighbors))
          recalls.append(recall)

      return np.mean(recalls)
  ```

- [ ] **步骤4**：更新模型训练 pipeline
  ```python
  # pipeline/train_models.py 或 recall_engine_v2.py
  if USE_FAISS:
      # 使用 HNSW 替代 IVF
      index = build_hnsw_index(
          embeddings,
          M=32,                 # 可通过配置调整
          efConstruction=200,
          efSearch=64
      )

      # 保存索引
      faiss.write_index(index, "models/faiss_hnsw.index")

      # 保存元数据
      metadata = {
          "index_type": "HNSW",
          "M": 32,
          "efConstruction": 200,
          "efSearch": 64,
          "dimension": dimension,
          "num_vectors": len(embeddings)
      }
      save_json(metadata, "models/faiss_recall.meta.json")
  ```

- [ ] **步骤5**：在线服务加载 HNSW 索引
  ```python
  # app/main.py 启动时加载
  def _load_faiss_index(base_dir: Path) -> Optional[faiss.Index]:
      index_path = base_dir / "faiss_hnsw.index"
      if not index_path.exists():
          return None

      try:
          index = faiss.read_index(str(index_path))

          # 设置查询参数（运行时可调）
          if hasattr(index, 'hnsw'):
              index.hnsw.efSearch = int(
                  os.getenv("FAISS_EF_SEARCH", "64")
              )

          LOGGER.info("Loaded HNSW index with %d vectors",
                      index.ntotal)
          return index
      except Exception as exc:
          LOGGER.error("Failed to load Faiss index: %s", exc)
          return None
  ```

#### 验收标准
- [x] HNSW 索引构建成功
- [x] 查询速度提升 ≥2倍
- [x] Recall@50 指标 ≥90%（与 IVF 对比）
- [x] 内存使用合理（<2GB）

#### 参数调优建议
| 参数 | 推荐值 | 说明 |
|------|--------|------|
| M | 16-32 | 连接数越大，精度越高但内存占用也越大 |
| efConstruction | 200-400 | 构建质量，仅影响离线训练时间 |
| efSearch | 32-128 | 查询精度，可在线调整 |

---

### ✅ TODO-07: LightGBM 模型轻量化

**优先级**：🟡 P2 - Medium
**预计工时**：3-5天
**负责人**：[ ]
**依赖项**：无

#### 任务描述
优化 LightGBM 模型参数，在保持精度的前提下提升推理速度和减小模型大小。

#### 实施步骤
- [ ] **步骤1**：创建模型对比配置 `config/model_configs.yaml`
  ```yaml
  # 当前配置（baseline）
  baseline:
    n_estimators: 100
    max_depth: 6
    num_leaves: 31
    learning_rate: 0.05
    min_child_samples: 20
    subsample: 1.0
    colsample_bytree: 1.0

  # 轻量化配置
  lightweight:
    n_estimators: 50          # 减少树数量
    max_depth: 4              # 降低树深度
    num_leaves: 15            # 减少叶子节点
    learning_rate: 0.08       # 提高学习率补偿树数量
    min_child_samples: 100    # 增加最小样本（防过拟合）
    subsample: 0.8            # 数据采样
    colsample_bytree: 0.8     # 特征采样
    reg_alpha: 0.1            # L1 正则
    reg_lambda: 0.1           # L2 正则

  # 极简配置（极致速度）
  minimal:
    n_estimators: 30
    max_depth: 3
    num_leaves: 8
    learning_rate: 0.1
    min_child_samples: 200
    subsample: 0.7
    colsample_bytree: 0.7
    reg_alpha: 0.2
    reg_lambda: 0.2
  ```

- [ ] **步骤2**：修改 `pipeline/train_models.py`
  ```python
  import yaml
  from pathlib import Path

  def load_model_config(config_name: str = "lightweight"):
      """加载模型配置"""
      config_path = Path("config/model_configs.yaml")
      with open(config_path) as f:
          configs = yaml.safe_load(f)
      return configs.get(config_name, configs['baseline'])

  def train_ranking_model_with_config(config_name: str):
      # 加载配置
      params = load_model_config(config_name)

      # 训练模型
      model = LGBMClassifier(**params, random_state=42)
      model.fit(X_train, y_train)

      # 评估
      metrics = evaluate_model(model, X_test, y_test)
      metrics['config_name'] = config_name
      metrics['model_size_mb'] = get_model_size(model)
      metrics['inference_time_ms'] = benchmark_inference(model)

      return model, metrics
  ```

- [ ] **步骤3**：对比实验
  ```python
  # 训练并对比三个配置
  configs = ['baseline', 'lightweight', 'minimal']
  results = []

  for config in configs:
      model, metrics = train_ranking_model_with_config(config)
      results.append(metrics)

      # 保存模型
      model_path = f"models/rank_model_{config}.pkl"
      save_pickle(model, model_path)

  # 生成对比报告
  comparison = pd.DataFrame(results)
  print(comparison[['config_name', 'auc', 'model_size_mb',
                    'inference_time_ms']])

  # 示例输出：
  # config_name    auc    model_size_mb  inference_time_ms
  # baseline      0.79      12.5           8.2
  # lightweight   0.77       6.3           3.5
  # minimal       0.74       3.1           1.8
  ```

- [ ] **步骤4**：特征重要性分析（优化特征）
  ```python
  def feature_importance_analysis(model, feature_names):
      """分析特征重要性，移除冗余特征"""
      importance = pd.DataFrame({
          'feature': feature_names,
          'importance': model.feature_importances_
      }).sort_values('importance', ascending=False)

      # 仅保留 Top-K 特征
      top_k = 30  # 从 70+ 特征降到 30 个
      important_features = importance.head(top_k)['feature'].tolist()

      return important_features

  # 使用精简特征重新训练
  important_features = feature_importance_analysis(
      baseline_model,
      feature_names
  )

  X_train_slim = X_train[important_features]
  X_test_slim = X_test[important_features]

  slim_model = LGBMClassifier(**lightweight_params)
  slim_model.fit(X_train_slim, y_train)
  ```

- [ ] **步骤5**：ONNX 导出和验证
  ```python
  from skl2onnx import to_onnx
  import onnxruntime as rt

  def export_and_benchmark_onnx(model, X_sample):
      # 导出 ONNX
      onnx_model = to_onnx(
          model,
          X_sample[:1],
          target_opset=12
      )

      # 保存
      with open("models/rank_model_lightweight.onnx", "wb") as f:
          f.write(onnx_model.SerializeToString())

      # 性能测试
      sess = rt.InferenceSession(onnx_model.SerializeToString())

      # Benchmark
      import time
      start = time.time()
      for _ in range(1000):
          sess.run(None, {'X': X_sample.values})
      onnx_time = time.time() - start

      # 对比 sklearn
      start = time.time()
      for _ in range(1000):
          model.predict_proba(X_sample)
      sklearn_time = time.time() - start

      print(f"sklearn: {sklearn_time:.3f}s")
      print(f"ONNX: {onnx_time:.3f}s")
      print(f"Speedup: {sklearn_time/onnx_time:.2f}x")
  ```

- [ ] **步骤6**：在线服务集成 ONNX
  ```python
  # app/main.py
  import onnxruntime as rt

  def _load_onnx_model(path: Path) -> Optional[rt.InferenceSession]:
      if not path.exists():
          return None

      try:
          sess = rt.InferenceSession(
              str(path),
              providers=['CPUExecutionProvider']
          )
          LOGGER.info("Loaded ONNX model from %s", path)
          return sess
      except Exception as exc:
          LOGGER.error("Failed to load ONNX model: %s", exc)
          return None

  # 启动时加载
  @app.on_event("startup")
  def load_models():
      # 优先加载 ONNX 模型
      onnx_path = MODELS_DIR / "rank_model_lightweight.onnx"
      app.state.onnx_ranker = _load_onnx_model(onnx_path)

      # fallback 到 pkl 模型
      if not app.state.onnx_ranker:
          app.state.rank_model = _load_rank_model(
              MODELS_DIR / "rank_model.pkl"
          )
  ```

#### 验收标准
- [x] 轻量化模型 AUC 下降 <3%（如 0.79→0.76+）
- [x] 模型大小减少 ≥50%
- [x] 推理速度提升 ≥2倍
- [x] ONNX 导出成功且功能正常

#### A/B 测试方案
```python
# 使用影子模型进行 A/B 测试
curl -X POST http://localhost:8000/models/reload \
  -H 'Content-Type: application/json' \
  -d '{
    "mode": "shadow",
    "source": "models/lightweight",
    "rollout": 0.1
  }'

# 观察 1-2 天后对比指标
# - 延迟改善
# - 业务指标（CTR/CVR）是否下降
```

---

## 🟢 P3 - 部署和扩缩容优化（1-2个月）

### ✅ TODO-08: HPA 配置优化

**优先级**：🟢 P3 - Low
**预计工时**：2-3天
**负责人**：[ ]
**依赖项**：Kubernetes 集群

#### 实施步骤
- [ ] **步骤1**：修改 `k8s/hpa.yaml`
  ```yaml
  apiVersion: autoscaling/v2
  kind: HorizontalPodAutoscaler
  metadata:
    name: recommendation-api-hpa
    namespace: recommendation
  spec:
    scaleTargetRef:
      apiVersion: apps/v1
      kind: Deployment
      name: recommendation-api
    minReplicas: 3              # 提高最小副本数（从 2→3）
    maxReplicas: 20             # 增加最大副本数（从 10→20）
    metrics:
    # CPU 扩缩容
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 60  # 降低阈值（从 70→60）

    # 内存扩缩容
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 70

    # 自定义指标：P95 延迟
    - type: Pods
      pods:
        metric:
          name: recommendation_latency_p95_milliseconds
        target:
          type: AverageValue
          averageValue: "80"  # P95 < 80ms

    # 自定义指标：请求 QPS
    - type: Pods
      pods:
        metric:
          name: recommendation_requests_per_second
        target:
          type: AverageValue
          averageValue: "100"  # 每个 pod 处理 100 QPS

    behavior:
      scaleUp:
        stabilizationWindowSeconds: 60   # 缩短扩容稳定窗口
        policies:
        - type: Percent
          value: 50                      # 每次扩容 50%
          periodSeconds: 60
        - type: Pods
          value: 2                       # 或每次增加 2 个 pod
          periodSeconds: 60
        selectPolicy: Max                # 选择更激进的策略

      scaleDown:
        stabilizationWindowSeconds: 300  # 延长缩容稳定窗口（5分钟）
        policies:
        - type: Percent
          value: 10                      # 每次缩容 10%
          periodSeconds: 60
        selectPolicy: Min                # 选择更保守的策略
  ```

- [ ] **步骤2**：配置 Prometheus Adapter（暴露自定义指标）
  ```yaml
  # k8s/prometheus-adapter.yaml
  apiVersion: v1
  kind: ConfigMap
  metadata:
    name: adapter-config
    namespace: monitoring
  data:
    config.yaml: |
      rules:
      # P95 延迟指标
      - seriesQuery: 'recommendation_latency_seconds_bucket'
        resources:
          overrides:
            namespace: {resource: "namespace"}
            pod: {resource: "pod"}
        name:
          as: "recommendation_latency_p95_milliseconds"
        metricsQuery: 'histogram_quantile(0.95, sum(rate(recommendation_latency_seconds_bucket[2m])) by (le, pod)) * 1000'

      # QPS 指标
      - seriesQuery: 'recommendation_requests_total'
        resources:
          overrides:
            namespace: {resource: "namespace"}
            pod: {resource: "pod"}
        name:
          as: "recommendation_requests_per_second"
        metricsQuery: 'sum(rate(recommendation_requests_total[1m])) by (pod)'
  ```

- [ ] **步骤3**：部署并验证
  ```bash
  # 应用 HPA 配置
  kubectl apply -f k8s/hpa.yaml

  # 查看 HPA 状态
  kubectl get hpa recommendation-api-hpa -n recommendation

  # 查看自定义指标是否正常
  kubectl get --raw "/apis/custom.metrics.k8s.io/v1beta1/namespaces/recommendation/pods/*/recommendation_latency_p95_milliseconds" | jq .

  # 压力测试验证扩容
  hey -z 60s -c 100 http://<ingress>/api/similar/123

  # 观察 pod 数量变化
  kubectl get pods -n recommendation -w
  ```

#### 验收标准
- [x] HPA 配置正确部署
- [x] 自定义指标正常采集
- [x] 压力测试时自动扩容
- [x] 负载降低时平滑缩容

---

### ✅ TODO-09: 资源限制优化

**优先级**：🟢 P3 - Low
**预计工时**：1-2天
**负责人**：[ ]
**依赖项**：TODO-08

#### 实施步骤
- [ ] **步骤1**：修改 `k8s/deployment.yaml`
  ```yaml
  containers:
  - name: recommendation-api
    image: recommendation-api:latest
    resources:
      requests:
        memory: "2Gi"      # 提高请求（确保特征加载）
        cpu: "1000m"       # 1 核 CPU
      limits:
        memory: "4Gi"      # 限制 4GB（防止 OOM）
        cpu: "2000m"       # 2 核 CPU（峰值）

    # 就绪探针
    readinessProbe:
      httpGet:
        path: /health
        port: 8000
      initialDelaySeconds: 15
      periodSeconds: 5
      timeoutSeconds: 3
      successThreshold: 1
      failureThreshold: 3

    # 存活探针
    livenessProbe:
      httpGet:
        path: /health
        port: 8000
      initialDelaySeconds: 30
      periodSeconds: 10
      timeoutSeconds: 5
      failureThreshold: 3

    # 优雅关闭
    lifecycle:
      preStop:
        exec:
          command: ["/bin/sh", "-c", "sleep 15"]
  ```

- [ ] **步骤2**：添加 PodDisruptionBudget
  ```yaml
  # k8s/pdb.yaml
  apiVersion: policy/v1
  kind: PodDisruptionBudget
  metadata:
    name: recommendation-api-pdb
    namespace: recommendation
  spec:
    minAvailable: 2      # 至少保持 2 个 pod 运行
    selector:
      matchLabels:
        app: recommendation-api
  ```

- [ ] **步骤3**：配置 QoS 等级
  ```yaml
  # 确保 requests == limits 以获得 Guaranteed QoS
  # 或者 requests < limits 获得 Burstable QoS

  # Guaranteed（推荐生产环境）
  resources:
    requests:
      memory: "3Gi"
      cpu: "1500m"
    limits:
      memory: "3Gi"      # 相同
      cpu: "1500m"       # 相同
  ```

- [ ] **步骤4**：监控资源使用
  ```bash
  # 查看实际资源使用
  kubectl top pods -n recommendation

  # 查看 OOM 事件
  kubectl get events -n recommendation | grep OOM

  # 查看 pod 重启次数
  kubectl get pods -n recommendation -o json | \
    jq '.items[] | {name:.metadata.name, restarts:.status.containerStatuses[].restartCount}'
  ```

#### 验收标准
- [x] 资源配置合理（无 OOM，CPU 不过高）
- [x] 就绪和存活探针正常工作
- [x] 优雅关闭无请求丢失
- [x] QoS 等级为 Guaranteed

---

### ✅ TODO-10: CDN 边缘缓存

**优先级**：🟢 P3 - Low
**预计工时**：3-5天
**负责人**：[ ]
**依赖项**：CDN 服务商账号

#### 实施步骤
- [ ] **步骤1**：配置 Nginx 缓存层
  ```nginx
  # k8s/nginx-cache.conf
  http {
      # 缓存路径配置
      proxy_cache_path /var/cache/nginx/hot
                       levels=1:2
                       keys_zone=hot_cache:10m
                       max_size=1g
                       inactive=10m;

      upstream recommendation_backend {
          server recommendation-api-service:8000;
      }

      server {
          listen 80;

          # 热榜接口缓存
          location /api/hot/trending {
              proxy_pass http://recommendation_backend;

              # 启用缓存
              proxy_cache hot_cache;
              proxy_cache_valid 200 5m;
              proxy_cache_use_stale error timeout updating;
              proxy_cache_lock on;

              # 缓存 key
              proxy_cache_key "$request_uri";

              # 返回缓存状态
              add_header X-Cache-Status $upstream_cache_status;
              add_header Cache-Control "public, max-age=300";
          }

          # 相似推荐（匿名用户可缓存）
          location /api/similar {
              proxy_pass http://recommendation_backend;

              # 仅缓存匿名请求
              proxy_cache hot_cache;
              proxy_cache_valid 200 3m;
              proxy_no_cache $http_authorization;

              add_header X-Cache-Status $upstream_cache_status;
          }

          # 其他接口不缓存
          location / {
              proxy_pass http://recommendation_backend;
          }
      }
  }
  ```

- [ ] **步骤2**：部署 Nginx 缓存层
  ```yaml
  # k8s/nginx-cache-deployment.yaml
  apiVersion: apps/v1
  kind: Deployment
  metadata:
    name: nginx-cache
    namespace: recommendation
  spec:
    replicas: 2
    template:
      spec:
        containers:
        - name: nginx
          image: nginx:alpine
          volumeMounts:
          - name: nginx-config
            mountPath: /etc/nginx/nginx.conf
            subPath: nginx.conf
          - name: cache-volume
            mountPath: /var/cache/nginx
        volumes:
        - name: nginx-config
          configMap:
            name: nginx-cache-config
        - name: cache-volume
          emptyDir:
            sizeLimit: 1Gi
  ```

- [ ] **步骤3**：CDN 配置（以 CloudFlare 为例）
  ```yaml
  # Cloudflare Page Rules
  rules:
    - url: "https://api.example.com/api/hot/*"
      settings:
        cache_level: "cache_everything"
        edge_cache_ttl: 300        # 5 分钟
        browser_cache_ttl: 180     # 3 分钟

    - url: "https://api.example.com/api/similar/*"
      settings:
        cache_level: "cache_everything"
        edge_cache_ttl: 180
        browser_cache_ttl: 120
  ```

- [ ] **步骤4**：监控缓存命中率
  ```python
  # app/metrics.py
  cdn_cache_hit = Counter(
      'cdn_cache_hit_total',
      'CDN cache hits',
      ['endpoint']
  )

  # 解析 X-Cache-Status 头
  @app.middleware("http")
  async def track_cdn_cache(request, call_next):
      response = await call_next(request)

      cache_status = response.headers.get("X-Cache-Status")
      if cache_status in ["HIT", "STALE"]:
          endpoint = request.url.path
          cdn_cache_hit.labels(endpoint=endpoint).inc()

      return response
  ```

#### 验收标准
- [x] Nginx 缓存层正常工作
- [x] CDN 缓存命中率 >60%
- [x] 热榜接口延迟 <20ms（CDN 命中时）
- [x] 缓存失效机制正常

---

## 🔵 P4 - 监控和告警增强（持续优化）

### ✅ TODO-11: 实时数据质量监控

**优先级**：🔵 P4 - Nice to have
**预计工时**：1周
**负责人**：[ ]
**依赖项**：Prometheus + AlertManager

#### 实施步骤
- [ ] **步骤1**：增强 `pipeline/data_quality_v2.py`
  ```python
  def check_data_freshness():
      """检查数据新鲜度"""
      state_path = DATA_DIR / "_metadata" / "extract_state.json"

      if not state_path.exists():
          send_alert("ERROR", "Extract state file not found")
          return

      state = json.loads(state_path.read_text())

      for source, tables in state.items():
          for table, info in tables.items():
              watermark = info.get('watermark')
              if not watermark:
                  continue

              last_update = datetime.fromisoformat(watermark)
              hours_ago = (datetime.now() - last_update).total_seconds() / 3600

              if hours_ago > 6:  # 超过 6 小时
                  send_alert(
                      "WARNING",
                      f"Table {source}.{table} not updated for {hours_ago:.1f}h"
                  )

  def check_null_ratios():
      """检查空值比例"""
      # 读取最新的质量报告
      report_path = DATA_DIR / "evaluation" / "data_quality_report_v2.json"
      report = json.loads(report_path.read_text())

      for table, stats in report['tables'].items():
          null_ratio = stats.get('null_ratio', 0)

          if null_ratio > 0.3:  # 超过 30%
              send_alert(
                  "WARNING",
                  f"Table {table} has {null_ratio:.1%} null values"
              )

  def check_data_distribution():
      """检查数据分布异常"""
      # 检测数据倾斜
      df = pd.read_parquet(DATA_DIR / "processed" / "interactions.parquet")

      # 用户交互分布
      user_counts = df.groupby('user_id').size()

      # 检测异常用户（交互次数 > 平均值 + 3*标准差）
      mean = user_counts.mean()
      std = user_counts.std()
      threshold = mean + 3 * std

      abnormal_users = user_counts[user_counts > threshold]

      if len(abnormal_users) > 0:
          send_alert(
              "INFO",
              f"Found {len(abnormal_users)} users with abnormal behavior"
          )

  def send_alert(severity: str, message: str):
      """发送告警（集成 AlertManager）"""
      import requests

      alert_url = os.getenv("ALERTMANAGER_URL", "http://localhost:9093")

      payload = [{
          "labels": {
              "alertname": "DataQualityIssue",
              "severity": severity.lower(),
              "service": "recommendation"
          },
          "annotations": {
              "summary": message,
              "description": f"Data quality check failed: {message}"
          }
      }]

      try:
          requests.post(f"{alert_url}/api/v1/alerts", json=payload)
      except Exception as exc:
          LOGGER.error("Failed to send alert: %s", exc)
  ```

- [ ] **步骤2**：添加定时任务
  ```python
  # scripts/data_quality_monitor.py
  import schedule
  import time

  def run_quality_checks():
      """运行所有质量检查"""
      check_data_freshness()
      check_null_ratios()
      check_data_distribution()

  # 每小时检查一次
  schedule.every(1).hours.do(run_quality_checks)

  if __name__ == "__main__":
      while True:
          schedule.run_pending()
          time.sleep(60)
  ```

- [ ] **步骤3**：配置 Prometheus 告警规则
  ```yaml
  # monitoring/alerts/data_quality.yml
  groups:
  - name: data_quality
    interval: 5m
    rules:
    # 数据新鲜度告警
    - alert: DataStale
      expr: (time() - data_quality_last_run_timestamp) > 21600
      for: 5m
      labels:
        severity: warning
        team: data
      annotations:
        summary: "数据抽取超过 6 小时未更新"
        description: "Pipeline 可能卡住，请检查日志"

    # 数据质量得分告警
    - alert: LowDataQuality
      expr: data_quality_score < 70
      for: 10m
      labels:
        severity: warning
        team: data
      annotations:
        summary: "数据质量得分低于 70: {{ $value }}"
        description: "请检查 data/evaluation/data_quality_report_v2.json"

    # 空值率告警
    - alert: HighNullRatio
      expr: data_quality_null_ratio > 0.3
      for: 10m
      labels:
        severity: warning
        table: "{{ $labels.table }}"
      annotations:
        summary: "表 {{ $labels.table }} 空值率超过 30%"

    # 数据倾斜告警
    - alert: DataSkewDetected
      expr: data_quality_skew_ratio > 10
      for: 15m
      labels:
        severity: info
      annotations:
        summary: "检测到数据倾斜"
        description: "部分用户/物品的交互次数异常偏高"
  ```

- [ ] **步骤4**：Grafana 面板
  ```json
  {
    "dashboard": {
      "title": "Data Quality Dashboard",
      "panels": [
        {
          "title": "Data Freshness",
          "targets": [{
            "expr": "(time() - data_quality_last_run_timestamp) / 3600"
          }],
          "alert": {
            "conditions": [{"value": 6, "operator": "gt"}]
          }
        },
        {
          "title": "Null Ratio by Table",
          "targets": [{
            "expr": "data_quality_null_ratio"
          }]
        },
        {
          "title": "Data Quality Score",
          "targets": [{
            "expr": "data_quality_score"
          }]
        }
      ]
    }
  }
  ```

#### 验收标准
- [x] 质量检查定时任务运行正常
- [x] 告警规则正确触发
- [x] Grafana 面板展示正常
- [x] 问题发现时间 <30 分钟

---

### ✅ TODO-12: 业务指标监控

**优先级**：🔵 P4 - Nice to have
**预计工时**：3-5天
**负责人**：[ ]
**依赖项**：TODO-11

#### 实施步骤
- [ ] **步骤1**：添加业务指标 `app/business_metrics.py`
  ```python
  """业务指标追踪"""
  from prometheus_client import Histogram, Counter, Gauge
  from typing import List, Set

  # 推荐多样性
  recommendation_diversity = Histogram(
      'recommendation_diversity_score',
      'Diversity score of recommendations',
      ['endpoint'],
      buckets=[0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
  )

  # 新物品曝光
  new_item_exposure = Counter(
      'new_item_exposure_total',
      'Number of new items exposed in recommendations',
      ['endpoint']
  )

  # 用户覆盖率
  user_coverage = Gauge(
      'user_coverage_ratio',
      'Ratio of users receiving personalized recommendations'
  )

  # 推荐位置分布
  recommendation_position = Histogram(
      'recommendation_click_position',
      'Position of clicked recommendations',
      buckets=[1, 2, 3, 5, 10, 20]
  )

  def calculate_diversity(items: List[int], metadata: dict) -> float:
      """计算推荐多样性（基于类别/标签）"""
      if not items:
          return 0.0

      categories = set()
      for item_id in items:
          item_meta = metadata.get(item_id, {})
          category = item_meta.get('company')
          if category:
              categories.add(category)

      # 香农熵 or 简单的唯一类别比例
      diversity = len(categories) / len(items)
      return diversity

  def track_new_item_exposure(
      items: List[int],
      new_threshold_days: int = 7
  ):
      """追踪新物品曝光"""
      # 读取数据集创建时间
      dataset_df = pd.read_parquet("data/processed/dataset_features.parquet")

      cutoff_date = datetime.now() - timedelta(days=new_threshold_days)
      new_items = dataset_df[
          dataset_df['create_time'] > cutoff_date
      ]['dataset_id'].tolist()

      new_item_set = set(new_items)
      exposed_new_items = [i for i in items if i in new_item_set]

      return len(exposed_new_items)
  ```

- [ ] **步骤2**：集成到推荐接口
  ```python
  # app/main.py
  from app.business_metrics import (
      calculate_diversity,
      track_new_item_exposure,
      recommendation_diversity,
      new_item_exposure
  )

  @app.get("/similar/{dataset_id}")
  async def get_similar(...):
      # ... 现有逻辑 ...

      # 计算多样性
      diversity = calculate_diversity(
          [item.dataset_id for item in items],
          state.metadata
      )
      recommendation_diversity.labels(endpoint="similar").observe(diversity)

      # 追踪新物品
      new_count = track_new_item_exposure(
          [item.dataset_id for item in items]
      )
      new_item_exposure.labels(endpoint="similar").inc(new_count)

      return response
  ```

- [ ] **步骤3**：用户覆盖率统计
  ```python
  # scripts/calculate_user_coverage.py
  import pandas as pd
  from prometheus_client import push_to_gateway

  def calculate_user_coverage():
      """计算用户覆盖率"""
      # 读取曝光日志
      exposure_df = pd.read_json(
          "data/evaluation/exposure_log.jsonl",
          lines=True
      )

      # 统计有推荐的用户数
      users_with_reco = exposure_df['user_id'].nunique()

      # 统计总用户数
      user_df = pd.read_parquet("data/business/user.parquet")
      total_users = len(user_df)

      coverage = users_with_reco / total_users if total_users > 0 else 0

      # 推送到 Prometheus
      from prometheus_client import CollectorRegistry, Gauge
      registry = CollectorRegistry()
      g = Gauge(
          'user_coverage_ratio',
          'User coverage ratio',
          registry=registry
      )
      g.set(coverage)

      push_to_gateway(
          'localhost:9091',
          job='user_coverage',
          registry=registry
      )

      return coverage

  # 每天运行一次
  if __name__ == "__main__":
      coverage = calculate_user_coverage()
      print(f"User coverage: {coverage:.2%}")
  ```

- [ ] **步骤4**：Grafana 业务面板
  ```json
  {
    "panels": [
      {
        "title": "Recommendation Diversity",
        "targets": [{
          "expr": "histogram_quantile(0.5, sum(rate(recommendation_diversity_score_bucket[5m])) by (le, endpoint))"
        }]
      },
      {
        "title": "New Item Exposure Rate",
        "targets": [{
          "expr": "sum(rate(new_item_exposure_total[5m])) / sum(rate(recommendation_count[5m]))"
        }]
      },
      {
        "title": "User Coverage",
        "targets": [{
          "expr": "user_coverage_ratio"
        }]
      }
    ]
  }
  ```

#### 验收标准
- [x] 业务指标正确采集
- [x] Grafana 面板展示正常
- [x] 多样性得分合理（>0.3）
- [x] 新物品曝光率 >15%

---

## ⚪ P5 - 成本优化（长期）

### ✅ TODO-13: 数据分层存储

**优先级**：⚪ P5 - Future
**预计工时**：2-3周
**负责人**：[ ]
**依赖项**：S3/OSS 存储账号

#### 实施步骤
- [ ] **步骤1**：配置对象存储
  ```python
  # config/storage.py
  import boto3
  from pathlib import Path

  class TieredStorage:
      def __init__(self):
          self.s3_client = boto3.client(
              's3',
              endpoint_url=os.getenv('S3_ENDPOINT'),
              aws_access_key_id=os.getenv('S3_ACCESS_KEY'),
              aws_secret_access_key=os.getenv('S3_SECRET_KEY')
          )
          self.bucket = os.getenv('S3_BUCKET', 'recommendation-data')

      def archive_old_data(
          self,
          source_table: str,
          cutoff_days: int = 30
      ):
          """归档旧数据到 S3"""
          cutoff_date = datetime.now() - timedelta(days=cutoff_days)

          # 读取数据
          df = pd.read_parquet(f"data/{source_table}.parquet")

          # 分割热/温数据
          hot_df = df[df['create_time'] > cutoff_date]
          warm_df = df[df['create_time'] <= cutoff_date]

          # 保留热数据在 MySQL
          hot_df.to_parquet(f"data/{source_table}_hot.parquet")

          # 温数据上传到 S3
          warm_path = f"warm/{source_table}/{cutoff_date.strftime('%Y%m')}.parquet"
          warm_df.to_parquet('/tmp/warm.parquet')

          self.s3_client.upload_file(
              '/tmp/warm.parquet',
              self.bucket,
              warm_path
          )

          return len(hot_df), len(warm_df)
  ```

- [ ] **步骤2**：修改 pipeline 支持分层读取
  ```python
  # pipeline/extract_load.py
  def load_historical_data(
      table: str,
      start_date: datetime,
      end_date: datetime
  ) -> pd.DataFrame:
      """加载历史数据（支持从 S3 读取）"""
      frames = []

      # 读取热数据（MySQL）
      if end_date > datetime.now() - timedelta(days=30):
          hot_df = read_from_mysql(table, start_date, end_date)
          frames.append(hot_df)

      # 读取温数据（S3）
      if start_date < datetime.now() - timedelta(days=30):
          warm_df = read_from_s3(table, start_date, end_date)
          frames.append(warm_df)

      return pd.concat(frames, ignore_index=True)
  ```

- [ ] **步骤3**：定时归档任务
  ```python
  # scripts/archive_old_data.py
  def archive_pipeline():
      """归档旧数据的定时任务"""
      storage = TieredStorage()

      tables = [
          'matomo_log_visit',
          'matomo_log_link_visit_action',
          'order_tab'
      ]

      for table in tables:
          hot_count, warm_count = storage.archive_old_data(
              table,
              cutoff_days=30
          )

          LOGGER.info(
              "Archived %s: hot=%d, warm=%d",
              table, hot_count, warm_count
          )

  # Cron: 每周运行一次
  # 0 2 * * 0 python scripts/archive_old_data.py
  ```

- [ ] **步骤4**：MySQL 数据清理
  ```sql
  -- 删除已归档的旧数据
  DELETE FROM matomo_log_visit
  WHERE server_time < DATE_SUB(NOW(), INTERVAL 30 DAY);

  -- 优化表
  OPTIMIZE TABLE matomo_log_visit;
  ```

#### 验收标准
- [x] 数据成功归档到 S3
- [x] 历史数据可正常读取
- [x] MySQL 存储空间减少 ≥50%
- [x] 存储成本降低 ≥60%

---

### ✅ TODO-14: 特征存储压缩

**优先级**：⚪ P5 - Future
**预计工时**：2-3天
**负责人**：[ ]
**依赖项**：无

#### 实施步骤
- [ ] **步骤1**：安装压缩库
  ```bash
  pip install lz4
  ```

- [ ] **步骤2**：创建压缩工具 `pipeline/compression.py`
  ```python
  """特征压缩存储"""
  import pickle
  import lz4.frame
  from pathlib import Path
  from typing import Any

  def save_compressed(data: Any, path: Path):
      """压缩保存"""
      # 序列化
      serialized = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)

      # 压缩
      compressed = lz4.frame.compress(
          serialized,
          compression_level=lz4.frame.COMPRESSIONLEVEL_MINHC
      )

      # 保存
      path.write_bytes(compressed)

      # 输出压缩比
      ratio = len(compressed) / len(serialized)
      print(f"Compression ratio: {ratio:.2%}")

  def load_compressed(path: Path) -> Any:
      """解压加载"""
      compressed = path.read_bytes()
      serialized = lz4.frame.decompress(compressed)
      return pickle.loads(serialized)
  ```

- [ ] **步骤3**：修改特征保存
  ```python
  # pipeline/build_features.py
  from pipeline.compression import save_compressed, load_compressed

  # 保存特征到 SQLite（保持原样）
  features_df.to_sql('user_features_v2', conn, if_exists='replace')

  # 额外保存压缩版本（用于快速加载）
  save_compressed(
      features_df,
      DATA_DIR / "processed" / "user_features_v2.lz4"
  )
  ```

- [ ] **步骤4**：修改 API 加载逻辑
  ```python
  # app/main.py
  from pipeline.compression import load_compressed

  def _load_user_features() -> pd.DataFrame:
      # 优先加载压缩版本（更快）
      compressed_path = DATA_DIR / "processed" / "user_features_v2.lz4"
      if compressed_path.exists():
          return load_compressed(compressed_path)

      # fallback 到 Parquet
      parquet_path = DATA_DIR / "processed" / "user_features_v2.parquet"
      if parquet_path.exists():
          return pd.read_parquet(parquet_path)

      # fallback 到 SQLite
      return _read_feature_store("SELECT * FROM user_features_v2")
  ```

- [ ] **步骤5**：批量压缩现有文件
  ```python
  # scripts/compress_features.py
  from pathlib import Path
  from pipeline.compression import save_compressed, load_compressed

  def compress_all_features():
      """压缩所有特征文件"""
      processed_dir = Path("data/processed")

      for parquet_file in processed_dir.glob("*.parquet"):
          # 读取
          df = pd.read_parquet(parquet_file)

          # 压缩保存
          compressed_path = parquet_file.with_suffix(".lz4")
          save_compressed(df, compressed_path)

          # 对比大小
          original_size = parquet_file.stat().st_size / 1024 / 1024
          compressed_size = compressed_path.stat().st_size / 1024 / 1024

          print(f"{parquet_file.name}:")
          print(f"  Original: {original_size:.1f} MB")
          print(f"  Compressed: {compressed_size:.1f} MB")
          print(f"  Ratio: {compressed_size/original_size:.1%}")

  if __name__ == "__main__":
      compress_all_features()
  ```

#### 验收标准
- [x] 压缩率 ≥60%（文件大小减少）
- [x] 加载速度无明显下降（<10%）
- [x] 功能正常，无数据丢失
- [x] 磁盘空间节省 ≥150MB

---

## 📊 追踪和验收

### 进度追踪表

| ID | 任务 | 优先级 | 状态 | 负责人 | 开始日期 | 完成日期 | 备注 |
|----|------|--------|------|--------|----------|----------|------|
| TODO-01 | MySQL 索引优化 | P0 | ⬜ Not Started | | | | |
| TODO-02 | 连接池优化 | P0 | ⬜ Not Started | | | | |
| TODO-03 | Redis 分层缓存 | P1 | ⬜ Not Started | | | | |
| TODO-04 | 本地内存缓存 | P1 | ⬜ Not Started | | | | |
| TODO-05 | 并行召回 | P2 | ⬜ Not Started | | | | |
| TODO-06 | Faiss HNSW | P2 | ⬜ Not Started | | | | |
| TODO-07 | LightGBM 轻量化 | P2 | ⬜ Not Started | | | | |
| TODO-08 | HPA 优化 | P3 | ⬜ Not Started | | | | |
| TODO-09 | 资源限制优化 | P3 | ⬜ Not Started | | | | |
| TODO-10 | CDN 缓存 | P3 | ⬜ Not Started | | | | |
| TODO-11 | 数据质量监控 | P4 | ⬜ Not Started | | | | |
| TODO-12 | 业务指标监控 | P4 | ⬜ Not Started | | | | |
| TODO-13 | 数据分层存储 | P5 | ⬜ Not Started | | | | |
| TODO-14 | 特征压缩 | P5 | ⬜ Not Started | | | | |

### 关键指标基线和目标

| 指标 | 当前值 | 目标值 | 优化后 | 达成日期 |
|------|--------|--------|--------|----------|
| API P95 延迟 | ~80ms | <50ms | | |
| 缓存命中率 | ~60% | >80% | | |
| Pipeline 执行时间 | 未知 | -40% | | |
| 召回耗时 | 未知 | -50% | | |
| 模型推理速度 | 未知 | +200% | | |
| 数据库存储成本 | 基线 | -60% | | |
| 可用性 | 未知 | >99.9% | | |

---

## 📝 更新日志

| 日期 | 更新内容 | 更新人 |
|------|----------|--------|
| 2025-10-10 | 创建 TODO 文档 | Claude Code |
| | | |
| | | |

---

## 🔗 参考资料

- 架构文档：`docs/ARCHITECTURE.md`
- 原始优化建议：基于 2025-10-10 系统分析
- Prometheus 最佳实践：https://prometheus.io/docs/practices/
- LightGBM 调优：https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html
- Faiss 索引选择：https://github.com/facebookresearch/faiss/wiki/Faiss-indexes

---

**备注**：
- 本文档是动态的，随着优化进展持续更新
- 每个 TODO 完成后更新状态和完成日期
- 重要决策和变更记录在备注列中
