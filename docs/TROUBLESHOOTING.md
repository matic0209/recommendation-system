# 故障排查手册（中文）

本文列出项目在部署与运行过程中常见错误及建议的解决办法。遇到问题时可根据提示逐项排查。

---

## 1. 权限与文件路径

### 1.1 `Permission denied` 写入模型/数据
**现象**：`models/`、`data/`、`/mlflow/artifacts` 等目录写入失败。  
**原因**：Docker 容器以 UID 50000（`appuser`）运行，宿主机目录属主不同导致写入失败。  
**处理**：
```bash
sudo chown -R 50000:50000 models data
sudo chown -R 50000:50000 /var/lib/docker/volumes/recommend_mlflow-data/_data
sudo chown -R 50000:50000 /opt/recommend/cache
```

### 1.2 `JSON data directory not found`
**现象**：Airflow 日志提示 `JSON data directory not found: ...`。  
**原因**：`.env` 中 `DATA_JSON_DIR` 与容器挂载目录不一致。  
**处理**：
1. 在 Compose 中为 Airflow 相关服务增加挂载：
   ```yaml
   volumes:
     - /path/on/host/jsons:/dianshu/backup/data/dianshu_data/jsons:ro
   ```
2. `.env` 中 `DATA_JSON_DIR` 设置为容器内实际路径（如 `/dianshu/backup/data/dianshu_data/jsons`）。
3. 重启 Airflow 容器：`docker compose up -d airflow-webserver airflow-scheduler`。

### 1.3 Docker Volume 名称确认
**现象**：执行 `docker volume inspect mlflow-data` 报 “no such volume”。  
**原因**：Compose 项目名导致实际卷名带前缀（如 `recommend_mlflow-data`）。  
**处理**：使用 `docker volume ls` 查找真实卷名，再 `docker volume inspect <name>`。

---

## 2. 模型与图片特征

### 2.1 CLIP 模型仍访问 Hugging Face
**现象**：日志出现 `https://huggingface.co` 相关错误。  
**处理**：
- 设置以下环境变量：
  ```ini
  CLIP_MODEL_PATH=/opt/recommend/cache/sentence-transformers/clip-ViT-B-32
  SENTENCE_TRANSFORMERS_HOME=/opt/recommend/cache/sentence-transformers
  HF_HOME=/opt/recommend/cache/huggingface
  TRANSFORMERS_CACHE=/opt/recommend/cache/huggingface/hub
  HF_HUB_OFFLINE=1
  ```
- 确保模型目录结构完整（含 `0_CLIPModel/`、`modules.json`）；
- 授权目录：`sudo chown -R 50000:50000 /opt/recommend/cache`。

### 2.2 评估缺少图像特征
**现象**：`evaluate` 报 “feature names should match … missing has_cover/has_images”。  
**处理**：
- 重新执行 `pipeline.build_features` 确保图片特征生成；
- 更新到最新代码，评估阶段会自动补齐可选列；
- 如果确实没有图片数据，可考虑在训练阶段移除这些特征或手动填充 0。

### 2.3 下载图片多次失败
**现象**：日志出现 `Giving up downloading ... after 3 attempts`。  
**说明**：离线环境无法访问原始图片 URL，此警告可忽略；仍会使用文本特征。若需完整图像信息，可提前离线下载图片或提供内网镜像。

---

## 3. 训练与评估异常

### 3.1 LightGBM “No further splits with positive gain”
**说明**：LightGBM 找不到更优的分裂点，训练仍成功，可忽略。若想优化，可增加特征或调整超参。

### 3.2 评估阶段比较时间时报错
**错误**：`Invalid comparison between dtype=datetime64[ns] and Timestamp`。  
**原因**：Matomo 行为日志带时区，曝光日志无时区。最新代码已统一转 UTC → naive 时间；如仍异常，检查数据中是否存在空值或格式不一致。

### 3.3 排序模型评估缺少字段
**现象**：`Ranking model evaluation failed: feature names missing`。  
**处理**：确认 `dataset_features_v2` 包含 `has_cover/has_images/image_richness_score`；若缺失，请先修复 `build_features` 流程或补值。

---

## 4. 告警与通知

### 4.1 企业微信未收到告警
**排查清单**：
1. `docker compose logs notification-gateway` 是否有 401/403；
2. `curl http://<host>:<port>/health`；
3. `curl -X POST http://<host>:<port>/test` 是否返回成功；
4. `.env` 中 `WEIXIN_*` 是否正确；
5. 企业微信后台 IP 白名单是否开放；
6. 若离线环境需走代理，请配置出口。

### 4.2 Alertmanager Webhook 失败
**现象**：Alertmanager 日志无法访问 `notification-gateway`。  
**处理**：确认 Compose 网络下主机名 `notification-gateway` 可达，`monitoring/alertmanager.yml` 指向正确端口。

---

## 5. Airflow 常见问题

### 5.1 `DagRunNotFound`
**解决**：先 `airflow dags list-runs -d recommendation_pipeline` 查看可用 `execution_date`，再使用完整 ISO 字符串执行 `tasks run/test`。

### 5.2 `JSON data directory not found`
**同 1.2。**

### 5.3 DAG 卡滞
**排查**：
- 查看 `scheduler`、`webserver` 日志；
- 检查 Postgres（Airflow 元数据库）状态；
- 确认任务队列是否过长，可暂停 DAG 或手动重试。

---

## 6. API 与服务

### 6.1 `/recommend/detail` 404
**原因**：接口路径为 `/recommend/detail/{dataset_id}`，请更新调用方式并可附带 `user_id`。

### 6.2 `/health` 状态异常
**排查**：
- 模型文件是否存在且可读取；
- Redis、MLflow 服务是否启动；
- 日志中是否有 “Ranking model failed” 等警告。

---

## 7. 离线部署常见坑

### 7.1 镜像缺少 `sentence-transformers`
**处理**：在联网环境 `docker compose build recommendation-api`，`docker save` → 拷贝到生产机 `docker load`，确保镜像内依赖齐全。

### 7.2 MLflow artifacts 权限
**步骤**：
```bash
docker compose down
sudo chown -R 50000:50000 /var/lib/docker/volumes/recommend_mlflow-data/_data
docker compose up -d
```

---

## 8. 其他提示

- `docker compose` 提示 `version` 过期：不影响运行，可忽略；若想消除，可删除该行；
- 定期执行 `./smoke_test.sh` 验证链路；
- `scripts/run_pipeline.sh --dry-run` 可预览将处理的表；
- 保留 `extract_state.json`、`extract_metrics.json` 快照便于回滚；
- 未覆盖的问题可结合《运维手册》《部署指南》进一步排查，并将总结写回文档。

祝排查顺利！
