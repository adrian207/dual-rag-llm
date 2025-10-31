#!/bin/bash
#
# Disaster Recovery Script
# Performs complete DR procedures including health check, backup, and recovery
#
# Author: Adrian Johnson <adrian207@gmail.com>

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
NAMESPACE="${NAMESPACE:-default}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Functions
check_health() {
    echo -e "${BLUE}Checking system health...${NC}"
    
    # Check PostgreSQL
    echo -n "  PostgreSQL: "
    if kubectl exec -n ${NAMESPACE} statefulset/dual-rag-llm-postgresql -- pg_isready -U dualrag > /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC}"
        DB_HEALTHY=true
    else
        echo -e "${RED}✗${NC}"
        DB_HEALTHY=false
    fi
    
    # Check Redis
    echo -n "  Redis: "
    if kubectl exec -n ${NAMESPACE} statefulset/dual-rag-llm-redis-master-0 -- redis-cli ping > /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC}"
        REDIS_HEALTHY=true
    else
        echo -e "${RED}✗${NC}"
        REDIS_HEALTHY=false
    fi
    
    # Check Application
    echo -n "  Application: "
    if kubectl get pods -n ${NAMESPACE} -l app.kubernetes.io/name=dual-rag-llm -o jsonpath='{.items[*].status.conditions[?(@.type=="Ready")].status}' | grep -q "True"; then
        echo -e "${GREEN}✓${NC}"
        APP_HEALTHY=true
    else
        echo -e "${RED}✗${NC}"
        APP_HEALTHY=false
    fi
    
    # Check disk space
    echo -n "  Disk Space: "
    DISK_USAGE=$(kubectl exec -n ${NAMESPACE} deployment/dual-rag-llm -- df -h /backups | tail -1 | awk '{print $5}' | tr -d '%')
    if [ ${DISK_USAGE} -lt 90 ]; then
        echo -e "${GREEN}✓ (${DISK_USAGE}% used)${NC}"
        DISK_HEALTHY=true
    else
        echo -e "${RED}✗ (${DISK_USAGE}% used)${NC}"
        DISK_HEALTHY=false
    fi
    
    echo ""
}

create_backup() {
    echo -e "${BLUE}Creating backup...${NC}"
    ${SCRIPT_DIR}/backup.sh
    echo ""
}

list_backups() {
    echo -e "${BLUE}Available backups:${NC}"
    kubectl exec -n ${NAMESPACE} deployment/dual-rag-llm -- ls -lh /backups/*.sql.gz 2>/dev/null | awk '{print "  " $9 "  (" $5 ")"}'
    echo ""
}

perform_restore() {
    local backup_file=$1
    echo -e "${BLUE}Performing restore...${NC}"
    ${SCRIPT_DIR}/restore.sh ${backup_file}
    echo ""
}

show_dr_status() {
    echo -e "${GREEN}Disaster Recovery Status${NC}"
    echo "========================="
    echo ""
    
    check_health
    
    # Get backup statistics
    echo -e "${BLUE}Backup Statistics:${NC}"
    BACKUP_COUNT=$(kubectl exec -n ${NAMESPACE} deployment/dual-rag-llm -- ls /backups/*.sql.gz 2>/dev/null | wc -l || echo "0")
    echo "  Total backups: ${BACKUP_COUNT}"
    
    if [ ${BACKUP_COUNT} -gt 0 ]; then
        LATEST_BACKUP=$(kubectl exec -n ${NAMESPACE} deployment/dual-rag-llm -- ls -t /backups/*.sql.gz 2>/dev/null | head -1)
        LATEST_BACKUP_TIME=$(kubectl exec -n ${NAMESPACE} deployment/dual-rag-llm -- stat -c %y ${LATEST_BACKUP} 2>/dev/null | cut -d'.' -f1)
        echo "  Latest backup: $(basename ${LATEST_BACKUP})"
        echo "  Backup time: ${LATEST_BACKUP_TIME}"
    fi
    
    echo ""
    
    # Overall status
    if [ "${DB_HEALTHY}" = true ] && [ "${REDIS_HEALTHY}" = true ] && [ "${APP_HEALTHY}" = true ] && [ "${DISK_HEALTHY}" = true ]; then
        echo -e "${GREEN}Overall Status: HEALTHY ✓${NC}"
    elif [ "${DB_HEALTHY}" = false ]; then
        echo -e "${RED}Overall Status: CRITICAL - Database down${NC}"
    else
        echo -e "${YELLOW}Overall Status: DEGRADED${NC}"
    fi
    
    echo ""
}

perform_recovery() {
    echo -e "${RED}INITIATING DISASTER RECOVERY${NC}"
    echo "=============================="
    echo ""
    
    # Step 1: Health Check
    check_health
    
    # Step 2: Determine recovery action
    if [ "${DB_HEALTHY}" = false ]; then
        echo -e "${YELLOW}Database is down. Attempting recovery...${NC}"
        echo ""
        
        # Try to restart PostgreSQL
        echo "Step 1: Restarting PostgreSQL..."
        kubectl rollout restart statefulset/dual-rag-llm-postgresql -n ${NAMESPACE}
        kubectl rollout status statefulset/dual-rag-llm-postgresql -n ${NAMESPACE} --timeout=300s
        
        # Re-check health
        sleep 10
        check_health
        
        if [ "${DB_HEALTHY}" = false ]; then
            echo -e "${RED}PostgreSQL restart failed. Manual intervention required.${NC}"
            echo ""
            echo "Consider restoring from backup:"
            list_backups
            exit 1
        else
            echo -e "${GREEN}PostgreSQL recovered successfully${NC}"
        fi
    fi
    
    if [ "${APP_HEALTHY}" = false ]; then
        echo -e "${YELLOW}Application is down. Attempting recovery...${NC}"
        echo ""
        
        # Restart application
        echo "Restarting application..."
        kubectl rollout restart deployment/dual-rag-llm -n ${NAMESPACE}
        kubectl rollout status deployment/dual-rag-llm -n ${NAMESPACE} --timeout=300s
        
        # Re-check health
        sleep 10
        check_health
        
        if [ "${APP_HEALTHY}" = false ]; then
            echo -e "${RED}Application restart failed. Check logs:${NC}"
            echo "  kubectl logs -n ${NAMESPACE} -l app.kubernetes.io/name=dual-rag-llm --tail=50"
            exit 1
        else
            echo -e "${GREEN}Application recovered successfully${NC}"
        fi
    fi
    
    echo ""
    echo -e "${GREEN}Recovery completed!${NC}"
    show_dr_status
}

# Main menu
echo -e "${GREEN}Dual RAG LLM - Disaster Recovery${NC}"
echo "=================================="
echo ""
echo "1) Show DR Status"
echo "2) Create Backup"
echo "3) List Backups"
echo "4) Restore from Backup"
echo "5) Perform Recovery"
echo "6) Exit"
echo ""
read -p "Select option: " OPTION

case ${OPTION} in
    1)
        show_dr_status
        ;;
    2)
        create_backup
        ;;
    3)
        list_backups
        ;;
    4)
        list_backups
        read -p "Enter backup filename: " BACKUP_FILE
        perform_restore ${BACKUP_FILE}
        ;;
    5)
        perform_recovery
        ;;
    6)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo -e "${RED}Invalid option${NC}"
        exit 1
        ;;
esac

