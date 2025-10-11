# Recommendation Service Operations SOP

This playbook covers the day-2 procedures promised in Phase 1.5 of the production upgrade plan.

## 1. Health Checks & Dashboards

1. Visit Grafana (`monitoring/grafana/README.md`) and open the *Recommendation SLO* dashboard.
2. Validate the following panels after each deployment:
   - Request success rate ≥ 99.5%
   - `recommendation_latency_seconds` P95 < 80ms
   - `recommendation_degraded_total` steady under 5/min.
3. Confirm data quality gauges (`data_quality_score`) expose the latest pipeline run timestamp. Out-of-date metrics imply the pipeline stalled.

## 2. Cache / Feature Store Failover

### Redis cache outage
1. Tail application logs for `Redis connection failed` events.
2. Observe `recommendation_degraded_total{reason="timeout:redis"}` – if it spikes, execute:
   ```bash
   kubectl scale deployment recommendation-api --replicas=0
   kubectl scale deployment recommendation-api --replicas=3
   ```
   This forces processes to start without Redis (automatic SQLite fallback).
3. Once Redis is restored, redeploy or call `/models/reload` to re-enable cache.

### Redis feature store failover
1. Check `/health` endpoint: `cache` can be disabled, but `models_loaded` must remain `true`.
2. If latency increases (>120ms), verify SQLite fallback by checking logs for `Failed to connect to Redis feature store` warnings (expected).
3. After Redis is back online, run:
   ```bash
   scripts/run_pipeline.sh --sync-only
   ```
   to resync features. (`--sync-only` is accepted by the script to skip ETL.)

## 3. Model Deployment / Shadow Testing

1. Stage artifacts with `scripts/stage_model.py <mlflow_run_dir>`.
2. Load as shadow model, 5% rollout:
   ```bash
   curl -X POST http://<host>/models/reload \
     -H 'Content-Type: application/json' \
     -d '{"mode": "shadow", "source": "models/staging/run", "rollout": 0.05}'
   ```
3. Monitor Grafana panels `Shadow hit ratio` (from Prometheus label `variant="shadow"`).
4. Promote to primary once metrics stable:
   ```bash
   curl -X POST http://<host>/models/reload \
     -H 'Content-Type: application/json' \
     -d '{"mode": "primary"}'
   ```

## 4. Chaos / Failure Drills

### Simulate Redis outage
1. Block Redis via firewall or `kubectl scale deployment redis-cache --replicas=0`.
2. Ensure API stays available; check fallback metrics.
3. Restore Redis and run `scripts/run_pipeline.sh --sync-only`.

### Simulate slow ranking model
1. Temporarily set env `RANKING_TIMEOUT=0.01` (via K8s configmap) and reload pods.
2. Verify `recommendation_timeouts_total{operation="model_inference"}` increases while app still serves popular fallbacks.
3. Revert configuration.

## 5. Alert Response Matrix

| Alert | Severity | Action |
|-------|----------|--------|
| `HighErrorRate` | Critical | Check API pods, roll back recent deploy, inspect logs.
| `HighLatency` | Warning | Inspect Redis/feature store, verify fallback metrics.
| `data_quality_score < 70` | Warning | Re-run pipeline, inspect `data/evaluation/data_quality_report_v2.json`.
| `CircuitBreakerOpen` | Warning | Inspect downstream dependency, consider toggling feature flag.
| `recommendation_degraded_total` spike | Warning | Check fallback reason label, follow relevant SOP above.

---

**Revision history**
- 2025-10-09: Initial SOP covering cache failover, model rollout, and chaos drills.
