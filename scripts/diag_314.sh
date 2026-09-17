#!/usr/bin/env bash
# ============================================================================
# diag_314.sh —— “3 月 14 日之后推荐就不再更新” 一次性排查脚本（只读，不改任何生产文件）
#
# 用法（在服务器 dn051 上执行）：
#     cd /root/recommendation-system && bash scripts/diag_314.sh 2>&1 | tee /tmp/diag_314.txt
#     然后把 /tmp/diag_314.txt 的内容贴回给助手
#
# 排查思路（三层时间差）：
#     ① 服务进程什么时候启动的         → 决定它“认识”哪些数据
#     ② 流水线哪一步失败、报什么错     → 决定索引/权重为什么不更新
#     ③ 关键产物文件的时间戳           → 哪些步骤真的没跑成功
# ============================================================================
set -u

REPO="/root/recommendation-system"
API="recommendation-api"
SCHED="airflow-scheduler"
PG="postgres-airflow"

section() { echo; echo "================================================================"; echo "## $*"; echo "================================================================"; }

cd "$REPO" 2>/dev/null || { echo "找不到 $REPO，请先 cd 到仓库目录"; exit 1; }

section "① 容器状态与启动时间（关键：服务是否长期未重启）"
docker ps --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}' | head -15
echo
for c in "$API" "$SCHED" airflow-webserver redis postgres-airflow mlflow; do
  started=$(docker inspect "$c" --format '{{.State.StartedAt}}' 2>/dev/null)
  [ -n "$started" ] && printf '  %-22s 启动于 %s\n' "$c" "$started" || printf '  %-22s (未运行)\n' "$c"
done

section "② 关键产物文件时间戳（哪些步骤停了）"
echo "--- models/（按时间倒序）---"
ls -lt models/ 2>/dev/null | head -20
echo
echo "--- data/processed/（只看最近 10 个）---"
ls -lt data/processed/ 2>/dev/null | head -11
echo
echo "--- 线上代码版本对比（容器内 vs 工作目录）---"
docker exec "$API" md5sum /app/app/main.py 2>/dev/null
md5sum app/main.py 2>/dev/null

section "③ 流水线近 14 天各任务成败统计"
docker exec "$PG" psql -U airflow -d airflow -c "
select task_id, state, count(*) n
from task_instance
where dag_id='recommendation_pipeline' and start_date > now() - interval '14 days'
group by 1,2 order by 1,3 desc;" 2>&1 | grep -v "collation\|HINT\|DETAIL\|WARNING"

section "④ 最近一次运行的逐任务状态（找出第一个失败的任务）"
RUN=$(docker exec "$PG" psql -U airflow -d airflow -t -A -c \
  "select run_id from dag_run where dag_id='recommendation_pipeline' order by start_date desc limit 1;" 2>/dev/null | tr -d '\r')
echo "最近一次 run_id: $RUN"
docker exec "$PG" psql -U airflow -d airflow -c "
select task_id, state,
       to_char(start_date,'MM-DD HH24:MI') started,
       to_char(end_date,'MM-DD HH24:MI') ended,
       try_number
from task_instance
where dag_id='recommendation_pipeline' and run_id='$RUN'
order by start_date nulls first;" 2>&1 | grep -v "collation\|HINT\|DETAIL\|WARNING"

section "⑤ 失败任务 train_models 的报错日志（★最关键★）"
# Airflow 2.x 日志目录结构：logs/dag_id=X/run_id=Y/task_id=Z/attempt=1.log
FOUND=0
for base in "$REPO/airflow/logs" "/opt/airflow/logs"; do
  # 注意：scheduler/ 目录下是“DAG 解析日志”，不是任务日志，必须排除；
  # 否则按字典序排序时 scheduler 开头的路径会挤掉真正的任务日志（第一次采集就踩了这个坑）。
  hits=$(docker exec "$SCHED" sh -c "find $base -path '*recommendation_pipeline*' -name '*.log' -newermt '-7 days' 2>/dev/null | grep -v '/scheduler/' | sort -r | head -5" 2>/dev/null)
  if [ -n "$hits" ]; then
    echo "--- 日志文件（$base）---"
    echo "$hits"
    LAST=$(echo "$hits" | tail -1)
    echo
    echo "--- 最新日志文件尾部 80 行（关键报错）---"
    docker exec "$SCHED" sh -c "tail -80 '$LAST'" 2>/dev/null
    FOUND=1
    break
  fi
done
if [ "$FOUND" -eq 0 ]; then
  echo "! 未在容器内找到日志文件，尝试用 CLI 取（语法可能随版本不同）："
  docker exec "$SCHED" airflow tasks logs recommendation_pipeline train_models "$RUN" 2>&1 | tail -60
fi

section "⑥ 依赖服务可达性（★我怀疑 train_models 是卡在 MLflow★）"
echo "--- 相关环境变量 ---"
docker exec "$SCHED" env 2>/dev/null | grep -iE "mlflow|tracking|redis_url" | sort
echo "--- 从 airflow-scheduler 容器内测 MLflow 可达性 ---"
# 注意：必须带 -i，否则 heredoc 传不进容器，python 会读到空脚本而静默无输出。
docker exec -i "$SCHED" python - <<'PY' 2>&1 | tail -5
import os, urllib.request
uri = os.getenv("MLFLOW_TRACKING_URI") or os.getenv("MLFLOW_TRACKING_URI ") or "http://mlflow:5000"
print("MLFLOW_TRACKING_URI =", uri)
try:
    with urllib.request.urlopen(uri.rstrip("/") + "/health", timeout=6) as r:
        print("MLflow /health ->", r.status, r.read()[:120])
except Exception as e:
    print("!! MLflow 不可达:", type(e).__name__, e)
PY
echo "--- 从 airflow-scheduler 容器内测 Redis / 业务库 ---"
docker exec "$SCHED" sh -c 'getent hosts mlflow redis postgres-airflow 2>/dev/null | head -5' 2>/dev/null

section "⑦ 资源情况（怀疑 OOM 就看这里）"
free -h 2>/dev/null | head -3
echo
docker stats --no-stream --format 'table {{.Name}}\t{{.MemUsage}}\t{{.MemPerc}}\t{{.CPUPerc}}' 2>/dev/null | head -12
echo
df -h / /var/lib/docker 2>/dev/null | head -4
echo
echo "--- 有没有被系统 OOM killer 杀过（最近）---"
dmesg 2>/dev/null | grep -iE "oom|killed process" | tail -10 || echo "(无法读取 dmesg，可跳过)"

section "⑧ 最近一次运行的 Airflow 调度器日志（看任务为什么被判失败）"
docker logs --tail 60 "$SCHED" 2>&1 | grep -viE "collation|HINT|DETAIL" | tail -30

echo
echo "================================================================"
echo "完成。请把以上完整输出（或 /tmp/diag_314.txt）发给助手。"
echo "重点看：③④ 哪个任务失败、⑤ 的报错内容、⑥ MLflow 是否可达、⑦ 是否 OOM。"
echo "================================================================"
