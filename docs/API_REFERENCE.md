# 在线推荐服务 API 参考

本文档描述 `app/main.py` 暴露的 REST 接口，涵盖入参、返回值、匿名用户处理方式以及示例。

## 基础信息

- 基础 URL：`http://<host>:8000`
- 内容类型：`application/json`
- 认证：当前无需鉴权（可在后续版本扩展）
- Swagger UI：`/docs`
- ReDoc：`/redoc`

所有时间均为服务器当前时区，数字类型默认以浮点数传输。

## 1. 健康检查

### `GET /health`

确认服务是否可用。

#### 响应

```json
{
  "status": "ok"
}
```

## 2. 相似数据集

### `GET /similar/{dataset_id}`

返回与目标数据集最相似的列表。对匿名用户与登录用户效果一致（此接口不使用用户上下文）。

#### 路径参数
| 参数 | 类型 | 必填 | 说明 |
| ---- | ---- | ---- | ---- |
| `dataset_id` | int | 是 | 目标数据集 ID |

#### 查询参数
| 参数 | 类型 | 默认值 | 说明 |
| ---- | ---- | ---- | ---- |
| `limit` | int | 10 | 返回条数，范围 1~50 |

#### 响应
```json
{
  "dataset_id": 13196,
  "similar_items": [
    {
      "dataset_id": 13004,
      "title": "示例数据集",
      "price": 99.0,
      "cover_image": "https://...",
      "score": 0.67,
      "reason": "behavior"
    }
  ]
}
```
- `reason` 取值：`behavior`（行为共现）、`content`（内容相似）、`popular`（热门兜底），当出现 `+personalized` 后缀时表示额外叠加了用户画像或最近行为的加权。

若找不到任何结果，接口返回 `404 Not Found`。

#### 示例
```
curl 'http://127.0.0.1:8000/similar/13196?limit=5'
```

## 3. 数据集详情页推荐

### `GET /recommend/detail/{dataset_id}`

在详情页展示“你可能还喜欢”列表。传入 `user_id` 时会基于该用户最近行为与标签偏好做个性化加权，并过滤已互动的数据集。

- 未登录用户 (`user_id` 未传或为空)：依据行为相似、内容相似和热门榜组合结果。
- 登录用户 (`user_id` 提供)：在上述基础上，提升与用户最近浏览/购买数据集相似、或标签偏好匹配的候选项得分。

#### 路径参数
| 参数 | 类型 | 必填 | 说明 |
| ---- | ---- | ---- | ---- |
| `dataset_id` | int | 是 | 当前页面数据集 ID |

#### 查询参数
| 参数 | 类型 | 默认值 | 说明 |
| ---- | ---- | ---- | ---- |
| `user_id` | int | 无 | 可选，表示当前登录用户；未登录时无需传递 |
| `limit` | int | 10 | 返回条数，范围 1~50 |

#### 响应
```json
{
  "dataset_id": 13196,
  "recommendations": [
    {
      "dataset_id": 13004,
      "title": "示例数据集",
      "price": 99.0,
      "cover_image": "https://...",
      "score": 0.72,
      "reason": "behavior"
    }
  ]
}
```

#### 示例
```
# 未登录用户示例（不传 user_id）
curl 'http://127.0.0.1:8000/recommend/detail/13196?limit=8'

# 登录用户示例（user_id=123）
curl 'http://127.0.0.1:8000/recommend/detail/13196?user_id=123&limit=8'
```

## 4. 响应字段说明
| 字段 | 类型 | 说明 |
| ---- | ---- | ---- |
| `dataset_id` | int | 推荐数据集 ID |
| `title` | string/null | 数据集标题，若缺失则为空 |
| `price` | float/null | 数据集价格，若无价格则为空 |
| `cover_image` | string/null | 封面图 URL，需在元数据中预先提供 |
| `score` | float | 综合得分（行为/内容相似度 + 热门兜底） |
| `reason` | string | 来源标签：`behavior` / `content` / `popular` |

## 5. 错误码
| 状态码 | 说明 |
| ------ | ---- |
| 200 | 成功 |
| 404 | 未找到相关推荐（常见于 `dataset_id` 不存在或模型未覆盖） |
| 500 | 内部错误（查看 `logs/uvicorn.log` 定位原因） |

## 6. 开发调试建议
1. 启动服务后访问 Swagger UI (`/docs`)，可直接填写参数并执行请求。
2. 如果模型或数据更新，应在重新训练后重启 FastAPI 服务以加载最新文件。
3. 日志输出默认保存到 `logs/uvicorn.log`，可用于排查异常。
4. 若未来接入用户画像，请在接口响应中新增必要字段并同步更新本文档。

如有新增接口或参数调整，请同步维护此文档，确保前后端沟通一致。
