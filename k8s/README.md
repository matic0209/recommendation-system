# Kubernetes Deployment Guide

## Prerequisites

- Kubernetes cluster (v1.24+)
- kubectl configured
- Docker registry access
- Persistent volume provisioner
- Nginx Ingress Controller
- Cert-Manager (for TLS)

## Deployment Steps

### 1. Build and Push Docker Image

```bash
# Build the image
docker build -t recommendation-api:v1.0 .

# Tag for your registry
docker tag recommendation-api:v1.0 your-registry/recommendation-api:v1.0

# Push to registry
docker push your-registry/recommendation-api:v1.0
```

### 2. Create Namespace and Secrets

```bash
# Create namespace
kubectl apply -f k8s/namespace.yaml

# Update secrets with actual values
# Edit k8s/secret.yaml and replace with base64-encoded credentials
echo -n 'your-database-user' | base64
echo -n 'your-database-password' | base64

# Apply secrets
kubectl apply -f k8s/secret.yaml
```

### 3. Apply ConfigMaps

```bash
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/redis-deployment.yaml
```

### 4. Create Persistent Volumes

```bash
kubectl apply -f k8s/pvc.yaml
```

### 5. Deploy Application

```bash
# Deploy the application
kubectl apply -f k8s/deployment.yaml

# Create service
kubectl apply -f k8s/service.yaml

# Configure ingress
kubectl apply -f k8s/ingress.yaml

# Enable autoscaling
kubectl apply -f k8s/hpa.yaml
```

### 6. Verify Deployment

```bash
# Check pods
kubectl get pods -n recommendation

# Check services
kubectl get svc -n recommendation

# Check ingress
kubectl get ingress -n recommendation

# View logs
kubectl logs -f deployment/recommendation-api -n recommendation
```

## Rolling Updates

```bash
# Update image version
kubectl set image deployment/recommendation-api \
  recommendation-api=your-registry/recommendation-api:v1.1 \
  -n recommendation

# Monitor rollout
kubectl rollout status deployment/recommendation-api -n recommendation

# Rollback if needed
kubectl rollout undo deployment/recommendation-api -n recommendation
```

## Scaling

```bash
# Manual scaling
kubectl scale deployment recommendation-api --replicas=5 -n recommendation

# HPA is configured for automatic scaling based on CPU/Memory
```

## Monitoring

```bash
# Check HPA status
kubectl get hpa -n recommendation

# View metrics
kubectl top pods -n recommendation

# Describe deployment
kubectl describe deployment recommendation-api -n recommendation
```

## Troubleshooting

```bash
# Check pod status
kubectl describe pod <pod-name> -n recommendation

# View logs
kubectl logs <pod-name> -n recommendation

# Execute commands in pod
kubectl exec -it <pod-name> -n recommendation -- /bin/bash

# Check events
kubectl get events -n recommendation --sort-by='.lastTimestamp'
```

## Clean Up

```bash
# Delete all resources
kubectl delete -f k8s/

# Or delete namespace (this removes everything)
kubectl delete namespace recommendation
```

## Production Considerations

1. **Security**:
   - Use sealed-secrets or external secret management (Vault, AWS Secrets Manager)
   - Enable Pod Security Policies
   - Configure Network Policies
   - Use RBAC for access control

2. **High Availability**:
   - Deploy Redis in cluster mode (3 masters + 3 replicas)
   - Use multiple availability zones
   - Configure pod disruption budgets

3. **Monitoring**:
   - Deploy Prometheus and Grafana
   - Configure AlertManager
   - Set up log aggregation (ELK/Loki)

4. **Backup**:
   - Schedule regular backups of PVCs
   - Backup Redis AOF files
   - Version model artifacts

5. **Performance**:
   - Tune resource requests/limits
   - Configure pod affinity/anti-affinity
   - Use node selectors for GPU nodes (if using deep learning)
