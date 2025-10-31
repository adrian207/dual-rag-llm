#!/bin/bash
#
# Database Restore Script
# Restores PostgreSQL database from backup
#
# Author: Adrian Johnson <adrian207@gmail.com>

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
NAMESPACE="${NAMESPACE:-default}"
BACKUP_DIR="/tmp/dualrag_backups"

echo -e "${GREEN}Dual RAG LLM - Database Restore${NC}"
echo "=================================="
echo ""

# Check if kubectl is installed
if ! command -v kubectl &> /dev/null; then
    echo -e "${RED}Error: kubectl is not installed${NC}"
    exit 1
fi

# Check backup file argument
if [ -z "$1" ]; then
    echo -e "${YELLOW}Available backups:${NC}"
    echo ""
    ls -lh ${BACKUP_DIR}/*.sql.gz 2>/dev/null | awk '{print "  " $9}'
    echo ""
    echo "Usage: $0 <backup_file>"
    echo "Example: $0 backup_20241031_120000.sql.gz"
    exit 1
fi

BACKUP_FILE="$1"

# Check if file exists
if [ ! -f "${BACKUP_DIR}/${BACKUP_FILE}" ]; then
    echo -e "${RED}Error: Backup file not found: ${BACKUP_DIR}/${BACKUP_FILE}${NC}"
    exit 1
fi

# Verify checksum if metadata exists
BACKUP_NAME="${BACKUP_FILE%.sql.gz}"
if [ -f "${BACKUP_DIR}/${BACKUP_NAME}.json" ]; then
    echo -e "${YELLOW}Verifying backup integrity...${NC}"
    EXPECTED_CHECKSUM=$(grep -o '"checksum": "[^"]*' ${BACKUP_DIR}/${BACKUP_NAME}.json | cut -d'"' -f4)
    ACTUAL_CHECKSUM=$(sha256sum ${BACKUP_DIR}/${BACKUP_FILE} | cut -d' ' -f1)
    
    if [ "${EXPECTED_CHECKSUM}" != "${ACTUAL_CHECKSUM}" ]; then
        echo -e "${RED}Error: Backup checksum mismatch!${NC}"
        echo "  Expected: ${EXPECTED_CHECKSUM}"
        echo "  Actual:   ${ACTUAL_CHECKSUM}"
        echo ""
        echo -e "${YELLOW}The backup file may be corrupted.${NC}"
        read -p "Do you want to continue anyway? (yes/no): " CONTINUE
        if [ "${CONTINUE}" != "yes" ]; then
            exit 1
        fi
    else
        echo -e "${GREEN}✓ Checksum verified${NC}"
    fi
fi

# Warning
echo ""
echo -e "${RED}WARNING: This will overwrite the current database!${NC}"
echo ""
echo "  Backup file: ${BACKUP_FILE}"
echo "  Database: dual_rag"
echo "  Namespace: ${NAMESPACE}"
echo ""
read -p "Are you sure you want to continue? (yes/no): " CONFIRM

if [ "${CONFIRM}" != "yes" ]; then
    echo "Restore cancelled."
    exit 0
fi

# Get PostgreSQL credentials
echo ""
echo -e "${YELLOW}Fetching PostgreSQL credentials...${NC}"
POSTGRES_PASSWORD=$(kubectl get secret --namespace ${NAMESPACE} dual-rag-llm-postgresql -o jsonpath="{.data.password}" | base64 --decode)
POSTGRES_HOST="dual-rag-llm-postgresql.${NAMESPACE}.svc.cluster.local"
POSTGRES_DB="dual_rag"
POSTGRES_USER="dualrag"

# Scale down application to prevent connections
echo -e "${YELLOW}Scaling down application...${NC}"
kubectl scale deployment/dual-rag-llm --replicas=0 -n ${NAMESPACE}
sleep 5

# Drop existing connections
echo -e "${YELLOW}Terminating existing connections...${NC}"
kubectl exec -n ${NAMESPACE} statefulset/dual-rag-llm-postgresql -- bash -c "
export PGPASSWORD='${POSTGRES_PASSWORD}'
psql -h localhost -U ${POSTGRES_USER} -d postgres -c \"
SELECT pg_terminate_backend(pg_stat_activity.pid)
FROM pg_stat_activity
WHERE pg_stat_activity.datname = '${POSTGRES_DB}'
  AND pid <> pg_backend_pid();
\"
" || true

# Drop and recreate database
echo -e "${YELLOW}Recreating database...${NC}"
kubectl exec -n ${NAMESPACE} statefulset/dual-rag-llm-postgresql -- bash -c "
export PGPASSWORD='${POSTGRES_PASSWORD}'
psql -h localhost -U ${POSTGRES_USER} -d postgres -c 'DROP DATABASE IF EXISTS ${POSTGRES_DB};'
psql -h localhost -U ${POSTGRES_USER} -d postgres -c 'CREATE DATABASE ${POSTGRES_DB};'
"

# Restore from backup
echo -e "${YELLOW}Restoring from backup...${NC}"
gunzip -c ${BACKUP_DIR}/${BACKUP_FILE} | kubectl exec -i -n ${NAMESPACE} statefulset/dual-rag-llm-postgresql -- bash -c "
export PGPASSWORD='${POSTGRES_PASSWORD}'
psql -h localhost -U ${POSTGRES_USER} -d ${POSTGRES_DB}
"

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ Restore completed successfully${NC}"
else
    echo -e "${RED}✗ Restore failed${NC}"
    exit 1
fi

# Scale up application
echo -e "${YELLOW}Scaling up application...${NC}"
kubectl scale deployment/dual-rag-llm --replicas=2 -n ${NAMESPACE}

# Wait for pods to be ready
echo -e "${YELLOW}Waiting for pods to be ready...${NC}"
kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=dual-rag-llm -n ${NAMESPACE} --timeout=300s

echo ""
echo -e "${GREEN}Done!${NC}"
echo ""
echo -e "${YELLOW}Verify the restored data:${NC}"
echo "  kubectl exec -it -n ${NAMESPACE} deployment/dual-rag-llm -- python -c 'from rag.database import get_database; import asyncio; db = asyncio.run(get_database()); print(asyncio.run(db.list_collections()))'"

