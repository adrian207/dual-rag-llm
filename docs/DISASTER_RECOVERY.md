# Disaster Recovery Guide

**Comprehensive disaster recovery procedures for Dual RAG LLM**

Author: Adrian Johnson <adrian207@gmail.com>

---

## Table of Contents

1. [Overview](#overview)
2. [Disaster Recovery Architecture](#disaster-recovery-architecture)
3. [Recovery Objectives](#recovery-objectives)
4. [Backup Strategy](#backup-strategy)
5. [Automated Health Monitoring](#automated-health-monitoring)
6. [Disaster Scenarios](#disaster-scenarios)
7. [Recovery Procedures](#recovery-procedures)
8. [Testing DR Plans](#testing-dr-plans)
9. [Maintenance](#maintenance)

---

## Overview

The Dual RAG LLM system includes enterprise-grade disaster recovery capabilities:

- **PostgreSQL Vector Database Backend**: Scalable vector storage with pgvector
- **Automated Backups**: Scheduled backups via Kubernetes CronJobs
- **Health Monitoring**: Continuous health checks with automatic alerting
- **Recovery Automation**: Automated recovery procedures
- **Manual Recovery Tools**: Scripts for backup, restore, and DR operations

### Key Features

✅ Automated 6-hour backup schedule  
✅ 30-day backup retention with configurable policies  
✅ Checksum verification for backup integrity  
✅ Point-in-time recovery capabilities  
✅ Automated failover for critical failures  
✅ Manual and automatic recovery modes  
✅ Comprehensive health monitoring  

---

## Disaster Recovery Architecture

### Components

```
┌─────────────────────────────────────────────────────────┐
│                   Application Layer                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │   FastAPI    │  │   Frontend   │  │    Ollama    │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────┘
                         │
┌─────────────────────────────────────────────────────────┐
│                     Data Layer                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │  PostgreSQL  │  │    Redis     │  │  ChromaDB    │ │
│  │  (pgvector)  │  │   (Cache)    │  │  (Vectors)   │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────┘
                         │
┌─────────────────────────────────────────────────────────┐
│                   Backup & DR Layer                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │   CronJob    │  │  PVC Storage │  │  S3 Backup   │ │
│  │  (6-hourly)  │  │  (100 GB)    │  │  (Optional)  │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────┘
                         │
┌─────────────────────────────────────────────────────────┐
│                 Monitoring & Recovery                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │Health Checks │  │ Auto-Failover│  │   Alerting   │ │
│  │  (60 sec)    │  │  (Automatic) │  │  (Webhooks)  │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────┘
```

### Database Backend

**PostgreSQL with pgvector** provides:

- **Vector Storage**: Native vector operations with IVFFLAT indexing
- **ACID Compliance**: Guaranteed data consistency
- **Point-in-Time Recovery**: Transaction log-based recovery
- **Replication**: Streaming replication for high availability
- **Backup Integration**: Standard pg_dump/pg_restore tooling

---

## Recovery Objectives

### RPO (Recovery Point Objective)

**Target: 60 minutes**

Maximum acceptable data loss in case of disaster.

- Automated backups every 6 hours
- Transaction logs for point-in-time recovery
- Continuous replication (if configured)

### RTO (Recovery Time Objective)

**Target: 15 minutes**

Maximum acceptable downtime in case of disaster.

- Automated health monitoring (60-second intervals)
- Automatic failover for critical failures
- Pre-configured recovery procedures
- Fast restore from recent backups

### Backup Retention

| Backup Type | Retention Period | Frequency |
|-------------|------------------|-----------|
| Full Backup | 30 days | Every 6 hours |
| Incremental | 7 days | Every 1 hour |
| Point-in-Time Logs | 7 days | Continuous |

---

## Backup Strategy

### Automated Backups

Kubernetes CronJob performs automated backups:

```yaml
schedule: "0 */6 * * *"  # Every 6 hours
retentionDays: 30
maxBackups: 100
```

**Backup Process:**

1. CronJob triggers at scheduled time
2. Connect to PostgreSQL with credentials
3. Execute `pg_dump` with compression
4. Calculate SHA-256 checksum
5. Save to PVC storage (/backups)
6. Update backup metadata
7. Clean up old backups per retention policy
8. (Optional) Upload to S3

### Manual Backups

Create on-demand backup:

```bash
# Using the backup script
export NAMESPACE=default
./scripts/backup.sh

# Or using kubectl directly
kubectl create job --from=cronjob/dual-rag-llm-backup manual-backup-$(date +%Y%m%d%H%M%S)
```

### Backup Verification

Every backup includes:

- **SHA-256 Checksum**: File integrity verification
- **Metadata**: Timestamp, size, database stats
- **Table Counts**: Row counts for validation
- **Database Size**: Total storage utilization

Verify a backup:

```bash
# Check backup integrity
cd /tmp/dualrag_backups
sha256sum -c <(grep checksum backup_*.json | cut -d'"' -f4)

# View backup metadata
cat backup_20241031_120000.json
```

---

## Automated Health Monitoring

### Health Check System

The Disaster Recovery Manager performs continuous health monitoring:

**Monitored Components:**

1. **Database Health**
   - PostgreSQL connectivity
   - Query response time
   - Connection pool status
   
2. **Backup System Health**
   - Backup directory accessibility
   - Write permissions
   - Backup age compliance (< RPO)
   
3. **Disk Space**
   - Available storage
   - Alert threshold: 90% usage
   
4. **Application Health**
   - Pod readiness
   - API responsiveness
   - Cache connectivity

### Health Status Levels

| Status | Description | Action |
|--------|-------------|--------|
| **HEALTHY** | All systems operational | None |
| **DEGRADED** | Minor issues detected | Monitor closely |
| **CRITICAL** | Major system failure | Automatic recovery |
| **RECOVERING** | Recovery in progress | Wait for completion |
| **FAILED** | Recovery failed | Manual intervention |

### Automatic Recovery

When status is CRITICAL, the system automatically:

1. **Creates Incident**: Logs incident with metrics
2. **Analyzes Failure**: Determines root cause
3. **Executes Recovery**:
   - Database down → Restore from backup
   - Backup system down → Restart service
   - Application down → Restart pods
4. **Verifies Recovery**: Re-checks health
5. **Sends Alerts**: Notifies via webhooks

---

## Disaster Scenarios

### Scenario 1: Database Corruption

**Symptoms:**
- Database connection failures
- Data integrity errors
- Query failures

**Recovery Procedure:**

```bash
# 1. Verify the issue
kubectl logs -n default statefulset/dual-rag-llm-postgresql

# 2. List available backups
./scripts/disaster-recovery.sh
# Select option 3: List Backups

# 3. Restore from most recent backup
./scripts/restore.sh backup_20241031_120000.sql.gz

# 4. Verify restoration
kubectl exec -n default deployment/dual-rag-llm -- \
  python -c "from rag.database import get_database; import asyncio; \
  db = asyncio.run(get_database()); \
  print(asyncio.run(db.health_check()))"
```

**Expected RTO:** 10-15 minutes

---

### Scenario 2: Complete Data Loss

**Symptoms:**
- PostgreSQL pod failure
- Persistent volume corruption
- All data unavailable

**Recovery Procedure:**

```bash
# 1. Delete corrupted resources
kubectl delete pvc dual-rag-llm-postgresql-pvc -n default

# 2. Recreate PostgreSQL
helm upgrade dual-rag-llm k8s/helm/dual-rag-llm -n default

# 3. Wait for PostgreSQL to be ready
kubectl wait --for=condition=ready pod \
  -l app.kubernetes.io/name=postgresql -n default --timeout=300s

# 4. Restore from backup
./scripts/restore.sh backup_20241031_120000.sql.gz

# 5. Verify all collections
kubectl exec -n default deployment/dual-rag-llm -- \
  python -c "from rag.database import get_database; import asyncio; \
  db = asyncio.run(get_database()); \
  print(asyncio.run(db.list_collections()))"
```

**Expected RTO:** 15-20 minutes

---

### Scenario 3: Application Failure

**Symptoms:**
- Application pods crashing
- API unavailable
- 5xx errors

**Recovery Procedure:**

```bash
# 1. Use DR automation
./scripts/disaster-recovery.sh
# Select option 5: Perform Recovery

# OR manually:

# 2. Check pod status
kubectl get pods -n default -l app.kubernetes.io/name=dual-rag-llm

# 3. View logs
kubectl logs -n default -l app.kubernetes.io/name=dual-rag-llm --tail=100

# 4. Restart deployment
kubectl rollout restart deployment/dual-rag-llm -n default

# 5. Wait for recovery
kubectl rollout status deployment/dual-rag-llm -n default

# 6. Verify health
curl https://dual-rag.example.com/health
```

**Expected RTO:** 5-10 minutes

---

### Scenario 4: Backup System Failure

**Symptoms:**
- No recent backups (> RPO)
- CronJob failures
- Storage full

**Recovery Procedure:**

```bash
# 1. Check CronJob status
kubectl get cronjob dual-rag-llm-backup -n default
kubectl describe cronjob dual-rag-llm-backup -n default

# 2. Check recent job runs
kubectl get jobs -n default | grep backup

# 3. View job logs
kubectl logs -n default job/dual-rag-llm-backup-<timestamp>

# 4. Check storage space
kubectl exec -n default deployment/dual-rag-llm -- df -h /backups

# 5. If storage full, clean old backups
kubectl exec -n default deployment/dual-rag-llm -- \
  find /backups -name "*.sql.gz" -mtime +30 -delete

# 6. Trigger manual backup
kubectl create job --from=cronjob/dual-rag-llm-backup \
  manual-backup-$(date +%Y%m%d%H%M%S) -n default
```

**Expected RTO:** 5 minutes

---

## Recovery Procedures

### Using DR Scripts

The system includes three comprehensive scripts:

#### 1. backup.sh

Creates manual database backup:

```bash
export NAMESPACE=default
./scripts/backup.sh

# Output:
# Backup completed: backup_20241031_120000.sql.gz
# Size: 2.5G
# SHA-256: abc123...
```

#### 2. restore.sh

Restores database from backup:

```bash
export NAMESPACE=default
./scripts/restore.sh backup_20241031_120000.sql.gz

# Interactive prompts:
# - Checksum verification
# - Confirmation warning
# - Automatic scaling down/up
```

**Warning:** This will overwrite the current database!

#### 3. disaster-recovery.sh

Interactive DR management:

```bash
./scripts/disaster-recovery.sh

# Menu options:
# 1) Show DR Status
# 2) Create Backup
# 3) List Backups
# 4) Restore from Backup
# 5) Perform Recovery
# 6) Exit
```

### Using Kubernetes Jobs

#### Create Backup Job

```bash
kubectl create job --from=cronjob/dual-rag-llm-backup \
  manual-backup-$(date +%Y%m%d%H%M%S) -n default

# Monitor progress
kubectl logs -f job/manual-backup-* -n default
```

#### Create Restore Job

```bash
# Edit values.yaml
cat <<EOF >> values-restore.yaml
restore:
  enabled: true
  backupId: "backup_20241031_120000"
EOF

# Apply restore job
helm upgrade dual-rag-llm k8s/helm/dual-rag-llm \
  -f values-restore.yaml -n default

# Monitor restore
kubectl logs -f job/dual-rag-llm-restore-* -n default
```

---

## Testing DR Plans

### Monthly DR Test

Perform monthly disaster recovery testing:

**Test Plan:**

1. **Week 1**: Backup verification
   ```bash
   # Verify all backups
   ./scripts/disaster-recovery.sh
   # Option 3: List Backups
   
   # Check integrity
   kubectl exec -n default deployment/dual-rag-llm -- \
     sha256sum -c /backups/*.json
   ```

2. **Week 2**: Restore test (non-production)
   ```bash
   # Create test namespace
   kubectl create namespace dr-test
   
   # Deploy test instance
   helm install dr-test k8s/helm/dual-rag-llm -n dr-test
   
   # Restore latest backup
   export NAMESPACE=dr-test
   ./scripts/restore.sh backup_20241031_120000.sql.gz
   
   # Validate data
   # Cleanup
   kubectl delete namespace dr-test
   ```

3. **Week 3**: Failover test
   ```bash
   # Simulate database failure
   kubectl delete pod dual-rag-llm-postgresql-0 -n default
   
   # Monitor automatic recovery
   watch kubectl get pods -n default
   
   # Verify recovery time
   ```

4. **Week 4**: Full DR drill
   ```bash
   # Execute complete DR procedure
   ./scripts/disaster-recovery.sh
   # Option 5: Perform Recovery
   
   # Document RTO/RPO
   # Update runbooks
   ```

---

## Maintenance

### Regular Tasks

**Daily:**
- Monitor health check status
- Review alerting logs
- Verify backup job completion

**Weekly:**
- Review backup storage usage
- Analyze incident reports
- Test backup integrity (sampling)

**Monthly:**
- Perform DR drill
- Update DR documentation
- Review and update RPO/RTO targets

**Quarterly:**
- Full-scale DR test
- Update recovery procedures
- Train team on DR processes

### Backup Retention Management

```bash
# Check current backups
kubectl exec -n default deployment/dual-rag-llm -- ls -lh /backups/

# Update retention policy
helm upgrade dual-rag-llm k8s/helm/dual-rag-llm \
  --set backup.retentionDays=60 \
  --set backup.maxBackups=200 \
  -n default

# Manual cleanup
kubectl exec -n default deployment/dual-rag-llm -- \
  find /backups -name "*.sql.gz" -mtime +60 -delete
```

### Monitoring Backup Health

```bash
# Check CronJob schedule
kubectl get cronjob dual-rag-llm-backup -n default -o yaml

# View recent backup jobs
kubectl get jobs -n default --sort-by=.metadata.creationTimestamp

# Check backup PVC usage
kubectl exec -n default deployment/dual-rag-llm -- df -h /backups
```

---

## Troubleshooting

### Backup Failures

**Issue:** CronJob fails to create backup

**Debug Steps:**
```bash
# 1. Check job logs
kubectl logs -n default job/dual-rag-llm-backup-<timestamp>

# 2. Check PostgreSQL connectivity
kubectl exec -n default deployment/dual-rag-llm -- \
  pg_isready -h dual-rag-llm-postgresql -p 5432

# 3. Verify credentials
kubectl get secret dual-rag-llm-postgresql -n default -o yaml

# 4. Check storage permissions
kubectl exec -n default deployment/dual-rag-llm -- ls -la /backups
```

### Restore Failures

**Issue:** Restore job fails or data is incomplete

**Debug Steps:**
```bash
# 1. Verify backup integrity
kubectl exec -n default deployment/dual-rag-llm -- \
  sha256sum /backups/backup_20241031_120000.sql.gz

# 2. Check PostgreSQL logs
kubectl logs -n default statefulset/dual-rag-llm-postgresql

# 3. Verify database permissions
kubectl exec -n default statefulset/dual-rag-llm-postgresql -- \
  psql -U dualrag -d dual_rag -c "\du"

# 4. Test restore manually
kubectl exec -it -n default statefulset/dual-rag-llm-postgresql -- bash
gunzip -c /backups/backup_20241031_120000.sql.gz | \
  psql -U dualrag -d dual_rag
```

---

## Emergency Contacts

| Role | Contact | Availability |
|------|---------|--------------|
| **Platform Team** | platform@example.com | 24/7 |
| **Database Admin** | dba@example.com | Business hours |
| **DevOps Lead** | Adrian Johnson<br>adrian207@gmail.com | On-call |
| **Security Team** | security@example.com | 24/7 |

---

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2024-10-31 | Adrian Johnson | Initial DR guide for v1.20.0 |

---

## References

- [PostgreSQL Backup Documentation](https://www.postgresql.org/docs/current/backup.html)
- [pgvector Documentation](https://github.com/pgvector/pgvector)
- [Kubernetes CronJobs](https://kubernetes.io/docs/concepts/workloads/controllers/cron-jobs/)
- [Helm Chart Documentation](../k8s/README.md)

---

**For immediate assistance with disasters, contact the Platform Team or use the automated recovery scripts.**

