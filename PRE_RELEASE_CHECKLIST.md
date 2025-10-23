# Request ID追踪功能 - 发版前准备清单

**准备日期**: 2025-10-19
**功能版本**: v1.0 (Request ID Tracking)
**Git分支**: feature/request-id-tracking
**状态**: ✅ 准备就绪

---

## 一、功能开发状态

### 1.1 后端API改动 ✅

- [x] **推荐接口增加request_id字段**
  - 文件: `app/main.py`
  - 接口: `GET /recommend/detail/{dataset_id}`
  - 返回: `request_id` + `algorithm_version`
  - 验证: ✅ 通过 (2025-10-19 15:07)

- [x] **相似推荐接口增加request_id字段**
  - 文件: `app/main.py`
  - 接口: `GET /similar/{dataset_id}`
  - 返回: `request_id`
  - 验证: ✅ 通过 (2025-10-19 15:13)

- [x] **曝光日志记录功能**
  - 文件: `app/telemetry.py`
  - 路径: `data/evaluation/exposure_log.jsonl`
  - 格式: JSONL (每行一个JSON对象)
  - 字段: request_id, user_id, page_id, algorithm_version, items, timestamp, context
  - 验证: ✅ 通过 (2025-10-19 15:13)

### 1.2 评估脚本 ✅

- [x] **新版评估脚本 (evaluate_v2.py)**
  - 文件: `pipeline/evaluate_v2.py`
  - 功能: 基于request_id精确匹配曝光和点击
  - 输出: `data/evaluation/tracking_report_v2.json`
  - 状态: 已完成

### 1.3 验证工具 ✅

- [x] **自动化验证脚本**
  - 文件: `scripts/verify_tracking.py`
  - 功能: 验证API、曝光日志、数据结构
  - 测试结果: 9/9 通过 (100% 通过率)
  - 运行命令: `no_proxy=localhost python3 scripts/verify_tracking.py --api-url http://localhost:8090`

---

## 二、文档完整性检查

### 2.1 核心文档 ✅

| 文档名称 | 路径 | 大小 | 状态 | 用途 |
|---------|------|------|------|------|
| 团队协作指南 | `TEAM_COLLABORATION_GUIDE.md` | 11KB | ✅ | 跨团队协作总览 |
| Request ID追踪指南 | `docs/REQUEST_ID_TRACKING_GUIDE.md` | 17KB | ✅ | 完整部署方案 |
| 前端集成指南 | `docs/FRONTEND_INTEGRATION.md` | 19KB | ✅ | 前端开发文档 |
| 后端集成指南 | `docs/BACKEND_INTEGRATION.md` | 18KB | ✅ | 后端调用文档 |
| 验证测试指南 | `docs/TRACKING_VALIDATION_GUIDE.md` | 7KB | ✅ | 发版前验证方案 |

### 2.2 文档内容检查 ✅

- [x] **前端接入文档**
  - React/Vue/原生JS示例代码 ✓
  - Matomo埋点说明 ✓
  - 验证清单 ✓
  - FAQ ✓

- [x] **后端接入文档**
  - Java/Go/PHP/Python示例 ✓
  - 错误处理和降级策略 ✓
  - 超时配置建议 ✓
  - 验证清单 ✓

- [x] **验证指南**
  - 手动模拟Matomo数据方法 ✓
  - 自动化测试脚本说明 ✓
  - 前端本地测试环境 ✓

---

## 三、功能验证状态

### 3.1 API功能验证 ✅

**测试环境**: Docker容器 (localhost:8090)
**测试时间**: 2025-10-19 15:13

| 测试项 | 状态 | 详情 |
|--------|------|------|
| API健康检查 | ✅ PASS | status=healthy, models_loaded=True |
| 推荐接口返回request_id | ✅ PASS | request_id格式正确 (UUID) |
| 曝光日志写入 | ✅ PASS | 成功记录2条曝光日志 |
| 曝光日志格式 | ✅ PASS | 包含所有必需字段 |
| 相似推荐接口 | ✅ PASS | 返回request_id |

### 3.2 数据质量验证 ✅

| 检查项 | 状态 | 详情 |
|--------|------|------|
| Matomo核心表 | ✅ PASS | 3/3个表已抽取 |
| Matomo数据新鲜度 | ✅ PASS | 22.9小时前更新 |
| 评估输出文件 | ✅ PASS | 3/3个文件存在 |
| 数据目录结构 | ✅ PASS | 所有必需目录完整 |

### 3.3 日志完整性 ✅

**曝光日志**: `data/evaluation/exposure_log.jsonl`

- 总行数: 48条
- 有效行数: 48条 (100%)
- 包含字段: request_id, user_id, page_id, algorithm_version, items, timestamp, context
- 最新记录时间: 2025-10-19 15:13

**示例记录**:
```json
{
  "request_id": "3a253234-d113-446b-9a56-e59ce97a4c91",
  "user_id": 123,
  "page_id": 1,
  "algorithm_version": "721801873b8f40afb7e7ba617e1fff55",
  "items": [...],
  "timestamp": "2025-10-19T15:07:47.735207+00:00",
  "context": {
    "endpoint": "recommend_detail",
    "variant": "primary",
    "experiment_variant": "content_boost"
  }
}
```

---

## 四、Git版本管理

### 4.1 版本标签 ✅

- **生产版本**: v1.0 (commit: c7bab3e)
  - 标签信息: "Production release v1.0 - Deployed to production environment"
  - 功能: 完整推荐系统基线版本

- **功能分支**: feature/request-id-tracking
  - 最新提交: 42d09e0 (Add deployment improvements)
  - 包含改动: Request ID追踪功能 + 文档

### 4.2 Git状态

```
Current branch: feature/request-id-tracking
Untracked files:
- PRE_RELEASE_CHECKLIST.md (本文件)
- send_weixin_v2/
```

---

## 五、Docker部署状态

### 5.1 容器运行状态 ✅

| 服务名称 | 容器名 | 端口 | 健康状态 |
|---------|--------|------|----------|
| 推荐API | recommendation-api | 8090 | ✅ healthy |
| Airflow Web | airflow-webserver | 8080 | ✅ running |
| Airflow Scheduler | airflow-scheduler | - | ✅ running |
| Redis | redis | 6379 | ✅ healthy |
| MLflow | mlflow | 5000 | ✅ running |
| PostgreSQL | postgres-airflow | - | ✅ healthy |
| Prometheus | prometheus | 9090 | ✅ running |
| Grafana | grafana | 3000 | ✅ running |
| Alertmanager | alertmanager | 9093 | ✅ running |
| Notification Gateway | notification-gateway | 9000 | ✅ running |

### 5.2 镜像版本

- `recommend-recommendation-api:latest` ✅
- `recommend-airflow:latest` ✅
- `recommend-notification-gateway:latest` ✅

---

## 六、待前后端团队完成的工作

### 6.1 Matomo管理员 (P0 - 必须先做)

- [ ] 配置Matomo自定义维度
  - 维度名称: `recommendation_request_id`
  - 作用域: `Visit`
  - 记录维度ID (通常是1)
  - 通知前端团队使用此ID

### 6.2 后端团队 (如适用)

- [ ] 更新代码透传request_id
  - 参考: `docs/BACKEND_INTEGRATION.md`
  - 确保调用推荐API后将request_id传给前端
  - 添加超时配置 (建议5秒)
  - 实施错误处理和降级策略

### 6.3 前端团队

- [ ] 实施推荐埋点
  - 参考: `docs/FRONTEND_INTEGRATION.md`
  - 保存API返回的request_id
  - 在推荐链接中携带追踪参数
  - 点击时发送Matomo事件
  - 设置自定义维度 (dimension1=request_id)

### 6.4 验证清单 (前端完成后)

- [ ] API返回request_id ✓ (推荐系统已完成)
- [ ] 推荐链接包含 `?from=recommend&rid=xxx&pos=0`
- [ ] 点击时浏览器Network能看到Matomo请求
- [ ] Matomo请求包含 `dimension1=req_xxx` 参数
- [ ] Matomo数据库有custom_dimension_3数据
- [ ] 运行evaluate_v2.py能生成报告
- [ ] CTR降到合理范围 (0.03-0.15)

---

## 七、上线计划

### 阶段1: 准备阶段 (Day 1)

- [x] **推荐系统**: 部署API改动 ✅ (已完成)
- [ ] **Matomo管理员**: 配置自定义维度
- [ ] **后端团队**: 更新代码透传request_id
- [ ] **前端团队**: 开始开发

### 阶段2: 测试阶段 (Day 2-3)

- [ ] **前端团队**: 完成开发，自测验证清单
- [ ] **推荐系统**: 提供测试环境验证
- [ ] **联调测试**: 端到端验证

### 阶段3: 灰度上线 (Day 4)

- [ ] **前端团队**: 10%流量灰度
- [ ] **推荐系统**: 运行evaluate_v2.py监控
- [ ] **验证指标**:
  - unique_request_ids > 0
  - overall_ctr 在 0.03-0.15 范围
  - 追踪覆盖率 > 80%

### 阶段4: 全量上线 (Day 5)

- [ ] **前端团队**: 全量上线
- [ ] **推荐系统**: 每天运行evaluate_v2.py
- [ ] **对比分析**: 新旧CTR差异

---

## 八、关键指标基线

### 8.1 当前状态 (基于旧版evaluate.py)

- CTR: 高估 (包含所有访问来源)
- CVR: 高估 (包含所有转化来源)
- 曝光数: 400 (截至2025-10-18)

### 8.2 预期改进 (基于新版evaluate_v2.py)

- **CTR合理范围**: 0.03 - 0.15 (3% - 15%)
- **CVR合理范围**: 0.005 - 0.05 (0.5% - 5%)
- **追踪覆盖率目标**: > 80%

---

## 九、风险与应对

### 9.1 已知风险

| 风险 | 影响 | 概率 | 应对措施 |
|------|------|------|----------|
| 前端未正确埋点 | CTR/CVR无法计算 | 中 | 提供详细文档 + 示例代码 + 验证工具 |
| Matomo自定义维度未配置 | 无法关联点击 | 中 | 提前沟通 + 验证SQL脚本 |
| 后端未透传request_id | 前端无法获取 | 低 | 提供多语言示例 + 验证清单 |
| 追踪覆盖率低 | 数据不完整 | 中 | 灰度上线 + 监控 + 逐步提升 |

### 9.2 降级方案

1. **Request ID追踪失败**
   - 继续使用旧版evaluate.py
   - 保持现有CTR/CVR计算
   - 不影响推荐服务本身

2. **前端埋点失败**
   - 推荐功能正常运行
   - 仅影响效果评估
   - 可后续补充

---

## 十、验证命令参考

### 10.1 验证推荐API

```bash
# 测试推荐接口
no_proxy=localhost curl -s "http://localhost:8090/recommend/detail/1?user_id=123&limit=5" | jq .request_id

# 测试相似推荐接口
no_proxy=localhost curl -s "http://localhost:8090/similar/1?limit=3" | jq .request_id
```

### 10.2 检查曝光日志

```bash
# 查看最新曝光日志
tail -n 3 data/evaluation/exposure_log.jsonl | python3 -m json.tool

# 统计日志条数
wc -l data/evaluation/exposure_log.jsonl
```

### 10.3 运行验证脚本

```bash
# 完整验证
no_proxy=localhost python3 scripts/verify_tracking.py --api-url http://localhost:8090

# 查看验证报告
cat data/evaluation/tracking_verification.json | jq .
```

### 10.4 运行评估脚本 (前端埋点后)

```bash
# 运行新版评估
python3 -m pipeline.evaluate_v2

# 查看评估报告
cat data/evaluation/tracking_report_v2.json | jq .summary
```

---

## 十一、联系方式

### 推荐系统团队

- **负责人**: [你的名字]
- **邮箱**: [你的邮箱]
- **文档**: `TEAM_COLLABORATION_GUIDE.md`

### 问题反馈

- **前端问题**: 参考 `docs/FRONTEND_INTEGRATION.md` FAQ
- **后端问题**: 参考 `docs/BACKEND_INTEGRATION.md` FAQ
- **验证问题**: 参考 `docs/TRACKING_VALIDATION_GUIDE.md`
- **数据问题**: 联系推荐系统团队

---

## 总结

**✅ 推荐系统侧准备就绪**

所有后端改动已完成并验证通过，文档齐全，验证工具可用。现在可以：

1. 将文档分发给相关团队
2. 等待Matomo/后端/前端完成各自改动
3. 进行联调测试
4. 按阶段上线

**下一步行动**:
1. 提交当前分支代码
2. 创建Pull Request合并到主分支
3. 分发文档给相关团队
4. 安排启动会议

---

**文档版本**: v1.0
**最后更新**: 2025-10-19
**状态**: ✅ Ready for Release
