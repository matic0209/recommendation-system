# FastAPI 服务结构说明

## 1. 总览

项目通过 FastAPI 暴露推荐接口，入口文件为 `app/main.py`。服务负责加载训练产物（相似度矩阵、热门榜、元数据）并对外提供查询接口。部署时通常搭配 Uvicorn/Gunicorn 运行。

关键依赖：
- FastAPI / Pydantic：API 定义与数据模型
- pandas：读取数据集元信息
- pickle / json：加载模型文件

启动命令示例：
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## 2. 文件结构

```
app/
  __init__.py
  main.py          # FastAPI 应用定义
```

`app/main.py` 中的重要模块：
- `RecommendationItem`, `RecommendationResponse`, `SimilarResponse`：接口输出的 Pydantic 模型
- `load_models()`：应用启动事件，加载模型与元数据
- `_combine_scores()`、`_build_response_items()`：推荐结果合并与格式化逻辑
- API 路由：`/health`、`/similar/{dataset_id}`、`/recommend/detail/{dataset_id}`

## 3. 启动流程

1. 服务启动时触发
## 3.1 用户画像与个性化

- 从 `data/processed/interactions.parquet` 构建用户最近行为列表（按时间排序，权重归一化）。
- 解析数据集标签，统计用户偏好标签，用于个性化加权。
- `app.state.user_history`、`app.state.user_tag_preferences` 在推荐阶段用于：
  1. 过滤用户已浏览/购买的数据集；
  2. 根据最近行为的相似度和标签偏好对候选项提升得分，并将 `reason` 标记为 `+personalized`。
- 未登录用户则直接返回基础候选（行为+内容+热门）。

 `@app.on_event("startup")`，依次加载：
   - `models/item_sim_behavior.pkl`
   - `models/item_sim_content.pkl`
   - `models/top_items.json`
   - `data/processed/dataset_features.parquet`（用于补全标题、价格、图片等）
2. 加载结果缓存在 `app.state`：
   - `behavior`：行为相似矩阵（`dict[int, dict[int, float]]`）
   - `content`：内容相似矩阵
   - `popular`：热门数据集列表
   - `metadata`：数据集元信息（标题、价格、封面）
3. 若缺少模型文件或元数据，服务会打印 warning；相关接口可能返回空结果或 404。

## 4. 接口说明

### 4.1 健康检查
- **路径**：`GET /health`
- **返回**：`{"status": "ok"}`
- **用途**：服务存活检测

### 4.2 相似数据集
- **路径**：`GET /similar/{dataset_id}`
- **参数**：
  - `dataset_id` (path)：目标数据集 ID
  - `limit` (query，可选，默认 10)：返回条数
- **逻辑**：
  1. 调用 `_combine_scores` 结合行为相似、内容相似与热门兜底
  2. 使用 `_build_response_items` 整理基础信息与推荐理由
  3. 若无结果，返回 404
- **响应**：`SimilarResponse`

### 4.3 详情页推荐
- **路径**：`GET /recommend/detail/{dataset_id}`
- **参数**：
  - `dataset_id` (path)：当前页面数据集 ID
  - `user_id` (query，可选)：预留字段，目前未使用
  - `limit` (query，可选，默认 10)：返回条数
- **逻辑**：与 `/similar` 类似，目前以物品相似为主；用户画像可在后续迭代中加权
- **响应**：`RecommendationResponse`

### 4.4 响应字段
`RecommendationItem` 与 `SimilarResponse` 中的字段：
- `dataset_id`：推荐的数据集
- `title`：标题（若元数据缺失可能为空）
- `price`：价格
- `cover_image`：封面 URL（目前 `dataset_features` 中需包含该字段）
- `score`：归一化后的相似度/权重
- `reason`：推荐来源（behavior / content / popular）

## 5. 常见问题

| 问题 | 排查建议 |
| ---- | -------- |
| 启动日志显示模型文件缺失 | 确认 `models/` 目录内的文件已生成（运行 `pipeline/train_models.py`） |
| 返回结果为空或 404 | 检查 `data/processed` 与 `models/` 是否最新；`dataset_id` 是否存在于模型矩阵中 |
| 修改模型后服务仍返回旧结果 | 重新运行 `uvicorn`，或触发自定义的热更新逻辑 |

## 6. 扩展建议

- 接入用户画像与个性化排序：在 `_combine_scores` 中引入用户权重、实时行为等
- 增加异常处理与日志：记录请求耗时、命中/兜底比例
- 集成身份/权限控制：如需将 `/recommend` 暴露给第三方，需要加入鉴权

## 7. 开发调试提示

- 使用 `curl` 或 Postman 调试接口，观察返回结构是否符合预期
- 在 VSCode/IDE 中可直接运行 `uvicorn`，结合 FastAPI 默认的 Swagger UI (`/docs`) 进行调试
- 若需单元测试，可使用 FastAPI 的 `TestClient` 模拟请求

以上即为 FastAPI 服务的主要结构与使用说明，帮助新成员快速理解在线推荐接口的行为与拓展方式。
