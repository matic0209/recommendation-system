# P0 优化执行指南

本指南详细说明如何执行 P0 级优化（数据库性能优化），包括索引优化和连接池配置。

---

## 📋 优化概览

| 项目 | 优化内容 | 预期收益 | 预计时间 |
|------|---------|---------|----------|
| TODO-01 | MySQL 索引优化 | Pipeline 速度↑60-80% | 1-2天 |
| TODO-02 | 数据库连接池优化 | 稳定性提升，避免连接泄漏 | 0.5-1天 |

---

## ✅ TODO-01: MySQL 索引优化

### 目标
为业务库和 Matomo 库的时间列添加索引，优化 CDC 增量抽取性能。

### 前置准备

#### 1. 备份数据库
```bash
# 备份业务库
mysqldump -u root -p --single-transaction dianshu_backend > dianshu_backend_backup_$(date +%Y%m%d).sql

# 备份 Matomo 库
mysqldump -u root -p --single-transaction matomo > matomo_backup_$(date +%Y%m%d).sql
```

#### 2. 检查磁盘空间
```bash
# 索引创建需要额外空间，确保至少有 10GB 可用空间
df -h /var/lib/mysql
```

#### 3. 确认执行时间
- **建议时间**：凌晨 2-5 点（业务低峰期）
- **预计耗时**：
  - 小表（<10万行）：1-5 分钟
  - 中表（10-100万行）：5-30 分钟
  - 大表（>100万行）：30-120 分钟

### 执行步骤

#### 步骤 1: 基准测试（执行前）
```bash
# 记录当前 pipeline 执行时间
cd /home/ubuntu/recommend
time python -m pipeline.extract_load

# 记录输出时间，例如：
# real    15m23.456s
# user    2m15.123s
# sys     0m32.567s
```

#### 步骤 2: 在测试环境验证（可选但强烈推荐）
```bash
# 如果有测试数据库，先在测试库执行
mysql -u root -p -h <test-db-host> < scripts/p0_01_add_indexes.sql
```

#### 步骤 3: 生产环境执行索引创建
```bash
# 连接到生产数据库
mysql -u root -p -h <production-db-host>

# 或者直接执行 SQL 文件
mysql -u root -p < scripts/p0_01_add_indexes.sql
```

**执行过程中的监控**：
```bash
# 在另一个终端监控 MySQL 进程
watch -n 2 'mysql -u root -p -e "SHOW PROCESSLIST"'

# 监控表状态
mysql -u root -p -e "SHOW STATUS LIKE 'Threads%'"
```

#### 步骤 4: 验证索引创建
```bash
# 运行验证脚本
python scripts/p0_02_verify_indexes.py --full

# 检查输出，确保所有索引都显示 "✓ OK"
```

#### 步骤 5: 性能测试（执行后）
```bash
# 再次执行 pipeline，对比时间
time python -m pipeline.extract_load

# 计算提升比例
# 预期：执行时间应减少 50-80%
```

### 验收标准

✅ **成功标准**：
- [ ] 所有索引创建成功（验证脚本显示 0 missing）
- [ ] EXPLAIN 查询显示使用了新索引
- [ ] Pipeline 执行时间减少 ≥50%
- [ ] 数据库写入性能下降 <10%（可通过监控确认）

❌ **失败标准**：
- 索引创建失败或部分失败
- 查询仍然不使用索引（EXPLAIN 显示 Full Table Scan）
- 数据库负载显著增加

### 回滚方案

如果索引导致问题（如写入性能下降严重），可以快速回滚：

```sql
-- 删除所有新创建的索引（详见 scripts/p0_01_add_indexes.sql 底部）

-- 业务库
USE dianshu_backend;
DROP INDEX idx_update_time ON user;
DROP INDEX idx_id_update_time ON user;
DROP INDEX idx_update_time ON dataset;
DROP INDEX idx_create_time ON dataset;
DROP INDEX idx_status_update_time ON dataset;
DROP INDEX idx_create_time ON order_tab;
DROP INDEX idx_update_time ON order_tab;
DROP INDEX idx_user_create_time ON order_tab;
DROP INDEX idx_dataset_create_time ON order_tab;
DROP INDEX idx_create_time ON api_order;
DROP INDEX idx_update_time ON api_order;
DROP INDEX idx_update_time ON dataset_image;
DROP INDEX idx_create_time ON dataset_image;

-- Matomo 库
USE matomo;
DROP INDEX idx_visit_last_action_time ON matomo_log_visit;
DROP INDEX idx_server_time ON matomo_log_visit;
DROP INDEX idx_visitor_time ON matomo_log_visit;
DROP INDEX idx_server_time ON matomo_log_link_visit_action;
DROP INDEX idx_visit_time ON matomo_log_link_visit_action;
DROP INDEX idx_server_time ON matomo_log_conversion;
DROP INDEX idx_visit_conversion_time ON matomo_log_conversion;
```

### 常见问题

**Q1: 索引创建时间过长怎么办？**
- A: 大表索引创建可能需要 30-120 分钟，这是正常的。可以使用 `SHOW PROCESSLIST` 监控进度。

**Q2: 创建索引期间数据库是否可用？**
- A: 默认情况下，MySQL 5.7+ 使用在线 DDL，表仍然可读写，但性能会有所下降。

**Q3: 如何判断索引是否被使用？**
- A: 运行 `python scripts/p0_02_verify_indexes.py --test-queries`，查看 EXPLAIN 结果。

**Q4: 索引占用多少磁盘空间？**
- A: 通常为原表大小的 10-30%。可以通过以下 SQL 查询：
```sql
SELECT
    TABLE_NAME,
    ROUND(DATA_LENGTH/1024/1024, 2) AS 'Data Size (MB)',
    ROUND(INDEX_LENGTH/1024/1024, 2) AS 'Index Size (MB)'
FROM information_schema.TABLES
WHERE TABLE_SCHEMA = 'dianshu_backend';
```

---

## ✅ TODO-02: 数据库连接池优化

### 目标
优化 SQLAlchemy 连接池配置，防止连接泄漏和超时错误。

### 执行步骤

#### 步骤 1: 更新环境变量
```bash
# 编辑 .env 文件
vi .env

# 添加或修改以下配置
DB_POOL_SIZE=10                  # 连接池大小
DB_MAX_OVERFLOW=20               # 最大溢出连接
DB_POOL_RECYCLE=3600             # 连接回收时间（秒）
DB_POOL_PRE_PING=true            # 连接前 ping 测试
DB_CONNECT_TIMEOUT=10            # 连接超时（秒）
```

**参数说明**：
- `DB_POOL_SIZE`: 保持在池中的连接数，建议 5-15
- `DB_MAX_OVERFLOW`: 超出 pool_size 的额外连接数，建议为 pool_size 的 2 倍
- `DB_POOL_RECYCLE`: 连接回收时间，建议 3600（1小时），避免 MySQL "gone away" 错误
- `DB_POOL_PRE_PING`: 使用连接前测试，强烈推荐开启
- `DB_CONNECT_TIMEOUT`: 连接超时，建议 10 秒

#### 步骤 2: 代码已自动应用

配置已在 `config/settings.py` 中自动应用，无需额外修改代码。

#### 步骤 3: 测试连接池
```bash
# 测试连接池配置
python -c "
from config.settings import load_database_configs
from sqlalchemy import create_engine

configs = load_database_configs()
business_config = configs['business']
engine = create_engine(business_config.sqlalchemy_url())

print('✓ Connection pool configured successfully')
print(f'Pool size: {engine.pool.size()}')
engine.dispose()
"
```

#### 步骤 4: 运行连接池监控
```bash
# 在 pipeline 执行时监控连接池
python -c "
from pipeline.connection_pool_monitor import monitor_engine_with_context
from config.settings import load_database_configs
from sqlalchemy import create_engine

configs = load_database_configs()
business_config = configs['business']
engine = create_engine(business_config.sqlalchemy_url())

with monitor_engine_with_context(engine, 'business') as monitor:
    # 执行一些查询
    with engine.connect() as conn:
        result = conn.execute('SELECT COUNT(*) FROM user')
        print(result.fetchone())
        monitor.record_snapshot()

# 自动打印报告
"
```

#### 步骤 5: 压力测试（检测连接泄漏）
```bash
# 并发执行 pipeline，观察连接池状态
for i in {1..5}; do
    python -m pipeline.extract_load --dry-run &
done
wait

# 检查是否有连接泄漏
mysql -u root -p -e "SHOW PROCESSLIST" | grep Sleep | wc -l
# 预期：Sleep 连接数应该 <= DB_POOL_SIZE
```

### 验收标准

✅ **成功标准**：
- [ ] 环境变量配置正确
- [ ] 连接池参数生效（通过 engine.pool.size() 验证）
- [ ] 并发测试无连接泄漏
- [ ] 无 "MySQL server has gone away" 错误

❌ **失败标准**：
- 连接池配置未生效
- 并发测试时连接数异常增长
- 出现连接超时或泄漏

### 监控和调优

#### 持续监控
```bash
# 定期检查 MySQL 连接数
watch -n 5 'mysql -u root -p -e "SHOW STATUS LIKE \"Threads_connected\""'

# 查看最大连接数配置
mysql -u root -p -e "SHOW VARIABLES LIKE 'max_connections'"
```

#### 调优建议

根据实际使用情况调整参数：

**场景 1：单进程运行**
```bash
DB_POOL_SIZE=5
DB_MAX_OVERFLOW=10
```

**场景 2：多进程并发（如 Airflow）**
```bash
DB_POOL_SIZE=10
DB_MAX_OVERFLOW=20
```

**场景 3：高并发场景（如 API 服务）**
```bash
DB_POOL_SIZE=20
DB_MAX_OVERFLOW=40
```

**注意**：总连接数 = 进程数 × (pool_size + max_overflow)，不应超过 MySQL `max_connections` 的 80%

### 常见问题

**Q1: 如何确定合适的 pool_size？**
- A: 根据并发度设置。公式：pool_size = 并发查询数 × 平均查询时间（秒）/ 查询间隔

**Q2: 什么时候需要增大 pool_size？**
- A: 当连接池利用率长期 >80% 时，或频繁看到 "QueuePool limit of size X overflow Y reached" 警告时。

**Q3: pool_recycle 设置多少合适？**
- A: 建议小于 MySQL `wait_timeout`（默认 8 小时）。生产环境建议 3600（1小时）。

**Q4: 如何检测连接泄漏？**
- A: 使用 `python scripts/p0_02_verify_indexes.py --monitor`，观察 checked_out 连接数是否持续增长。

---

## 📊 整体验收

### 完成 P0 优化后的检查清单

- [ ] TODO-01: 所有索引创建成功
- [ ] TODO-01: Pipeline 执行时间减少 ≥50%
- [ ] TODO-01: 查询使用索引（EXPLAIN 验证）
- [ ] TODO-02: 连接池配置生效
- [ ] TODO-02: 无连接泄漏
- [ ] TODO-02: 无 "gone away" 错误
- [ ] 更新 `docs/PRODUCTION_OPTIMIZATION_TODO.md` 中的进度

### 性能指标对比

| 指标 | 优化前 | 优化后 | 提升 |
|------|--------|--------|------|
| Pipeline 执行时间 | ___分钟 | ___分钟 | __% |
| 增量抽取速度 | ___行/秒 | ___行/秒 | __% |
| 数据库连接数 | ___ | ___ | -- |
| 连接泄漏次数 | ___ | 0 | -- |

---

## 🔧 故障排查

### 问题 1: 索引创建失败

**症状**：
```
ERROR 1062 (23000): Duplicate entry ...
ERROR 1091 (42000): Can't DROP 'idx_name'; check that column/key exists
```

**解决方案**：
1. 检查索引是否已存在：
```sql
SHOW INDEX FROM table_name WHERE Key_name = 'idx_name';
```

2. 删除冲突的索引后重试

### 问题 2: 连接池不工作

**症状**：
```
sqlalchemy.exc.TimeoutError: QueuePool limit exceeded
```

**解决方案**：
1. 检查环境变量是否加载：
```python
import os
print(os.getenv('DB_POOL_SIZE'))
```

2. 检查 .env 文件是否被 load_dotenv() 正确加载

3. 增大 pool_size 或 max_overflow

### 问题 3: MySQL "gone away" 错误

**症状**：
```
MySQL server has gone away
```

**解决方案**：
1. 确保 `DB_POOL_PRE_PING=true`
2. 减小 `DB_POOL_RECYCLE` 值（例如从 3600 降到 1800）
3. 增大 MySQL `wait_timeout` 配置

---

## 📝 记录和报告

### 优化执行记录

```
执行人：___________
执行日期：___________
执行环境：[ ] 测试环境  [ ] 生产环境

TODO-01 执行结果：
- 索引创建数量：_____ 个
- 执行耗时：_____ 分钟
- Pipeline 性能提升：_____ %
- 问题记录：_____________________

TODO-02 执行结果：
- 连接池配置：pool_size=___, max_overflow=___
- 并发测试结果：[ ] 通过  [ ] 失败
- 连接泄漏检测：[ ] 无泄漏  [ ] 发现泄漏
- 问题记录：_____________________

验收结果：[ ] 通过  [ ] 不通过
备注：_____________________
```

### 更新文档

完成后，在 `docs/PRODUCTION_OPTIMIZATION_TODO.md` 中更新：
1. 进度追踪表中的状态 ✅
2. 关键指标基线和目标表中的"优化后"列
3. 更新日志

---

## 🎯 下一步

P0 优化完成后，可以继续执行：
- **P1**: Redis 分层缓存优化（预计 1-2 周）
- **P2**: 模型和召回优化（预计 3-4 周）

详见 `docs/PRODUCTION_OPTIMIZATION_TODO.md`

---

## 🔗 参考资料

- 索引优化脚本：`scripts/p0_01_add_indexes.sql`
- 验证脚本：`scripts/p0_02_verify_indexes.py`
- 连接池监控：`pipeline/connection_pool_monitor.py`
- MySQL 索引最佳实践：https://dev.mysql.com/doc/refman/8.0/en/optimization-indexes.html
- SQLAlchemy 连接池文档：https://docs.sqlalchemy.org/en/14/core/pooling.html
