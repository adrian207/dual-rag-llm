#!/bin/bash
#
# Database Backup Script
# Performs manual backup of PostgreSQL database
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
BACKUP_NAME="backup_$(date +%Y%m%d_%H%M%S)"
BACKUP_TYPE="${BACKUP_TYPE:-full}"

echo -e "${GREEN}Dual RAG LLM - Database Backup${NC}"
echo "=================================="
echo ""

# Check if kubectl is installed
if ! command -v kubectl &> /dev/null; then
    echo -e "${RED}Error: kubectl is not installed${NC}"
    exit 1
fi

# Get PostgreSQL credentials
echo -e "${YELLOW}Fetching PostgreSQL credentials...${NC}"
POSTGRES_PASSWORD=$(kubectl get secret --namespace ${NAMESPACE} dual-rag-llm-postgresql -o jsonpath="{.data.password}" | base64 --decode)
POSTGRES_HOST="dual-rag-llm-postgresql.${NAMESPACE}.svc.cluster.local"
POSTGRES_DB="dual_rag"
POSTGRES_USER="dualrag"

# Create backup directory
BACKUP_DIR="/tmp/dualrag_backups"
mkdir -p ${BACKUP_DIR}

echo -e "${YELLOW}Starting backup: ${BACKUP_NAME}${NC}"
echo "  Type: ${BACKUP_TYPE}"
echo "  Database: ${POSTGRES_DB}"
echo "  Host: ${POSTGRES_HOST}"
echo ""

# Execute backup via kubectl exec
echo -e "${YELLOW}Running pg_dump...${NC}"
kubectl exec -n ${NAMESPACE} deployment/dual-rag-llm -- bash -c "
export PGPASSWORD='${POSTGRES_PASSWORD}'
pg_dump \
  --host=${POSTGRES_HOST} \
  --port=5432 \
  --username=${POSTGRES_USER} \
  --dbname=${POSTGRES_DB} \
  --format=plain \
  --no-owner \
  --no-acl \
  --verbose
" | gzip > ${BACKUP_DIR}/${BACKUP_NAME}.sql.gz

if [ $? -eq 0 ]; then
    BACKUP_SIZE=$(du -h ${BACKUP_DIR}/${BACKUP_NAME}.sql.gz | cut -f1)
    echo ""
    echo -e "${GREEN}✓ Backup completed successfully${NC}"
    echo "  File: ${BACKUP_DIR}/${BACKUP_NAME}.sql.gz"
    echo "  Size: ${BACKUP_SIZE}"
    echo ""
    
    # Calculate checksum
    CHECKSUM=$(sha256sum ${BACKUP_DIR}/${BACKUP_NAME}.sql.gz | cut -d' ' -f1)
    echo "  SHA-256: ${CHECKSUM}"
    echo ""
    
    # Save metadata
    cat > ${BACKUP_DIR}/${BACKUP_NAME}.json <<EOF
{
  "backup_id": "${BACKUP_NAME}",
  "backup_type": "${BACKUP_TYPE}",
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "database": "${POSTGRES_DB}",
  "checksum": "${CHECKSUM}",
  "file": "${BACKUP_NAME}.sql.gz",
  "size_bytes": $(stat -f%z ${BACKUP_DIR}/${BACKUP_NAME}.sql.gz 2>/dev/null || stat -c%s ${BACKUP_DIR}/${BACKUP_NAME}.sql.gz)
}
EOF
    
    echo -e "${GREEN}Backup metadata saved${NC}"
else
    echo -e "${RED}✗ Backup failed${NC}"
    exit 1
fi

# List recent backups
echo ""
echo -e "${YELLOW}Recent backups:${NC}"
ls -lh ${BACKUP_DIR}/ | tail -5

echo ""
echo -e "${GREEN}Done!${NC}"

