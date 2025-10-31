"""
Disaster Recovery System
Automated DR procedures, failover, and recovery automation

Author: Adrian Johnson <adrian207@gmail.com>
"""

import os
import asyncio
from typing import Optional, List, Dict, Any, Callable
from datetime import datetime, timedelta
from enum import Enum

from pydantic import BaseModel
import structlog

from rag.backup import BackupManager, BackupType, get_backup_manager
from rag.database import VectorDatabase, DatabaseConfig

logger = structlog.get_logger()


class DRStatus(str, Enum):
    """Disaster recovery status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    RECOVERING = "recovering"
    FAILED = "failed"


class FailoverMode(str, Enum):
    """Failover mode"""
    AUTOMATIC = "automatic"
    MANUAL = "manual"


class RecoveryAction(str, Enum):
    """Recovery action type"""
    RESTORE_BACKUP = "restore_backup"
    FAILOVER = "failover"
    REBUILD_INDEX = "rebuild_index"
    RESTART_SERVICE = "restart_service"
    ALERT = "alert"


class DRConfig(BaseModel):
    """Disaster recovery configuration"""
    enabled: bool = True
    auto_backup_enabled: bool = True
    backup_interval_hours: int = 6
    health_check_interval_seconds: int = 60
    failover_mode: FailoverMode = FailoverMode.AUTOMATIC
    max_recovery_attempts: int = 3
    recovery_timeout_seconds: int = 600
    rpo_minutes: int = 60  # Recovery Point Objective
    rto_minutes: int = 15  # Recovery Time Objective
    alert_webhooks: List[str] = []
    secondary_db_config: Optional[Dict[str, str]] = None


class DRIncident(BaseModel):
    """Disaster recovery incident"""
    id: str
    incident_type: str
    severity: str
    status: DRStatus
    detected_at: datetime
    resolved_at: Optional[datetime] = None
    actions_taken: List[RecoveryAction] = []
    success: bool = False
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = {}


class HealthCheckResult(BaseModel):
    """Health check result"""
    timestamp: datetime
    database_healthy: bool
    backup_system_healthy: bool
    disk_space_available: bool
    replication_lag_seconds: Optional[float] = None
    last_backup_age_hours: Optional[float] = None
    issues: List[str] = []
    status: DRStatus


class DisasterRecoveryManager:
    """Disaster recovery orchestrator"""
    
    def __init__(
        self, 
        config: Optional[DRConfig] = None,
        backup_manager: Optional[BackupManager] = None
    ):
        self.config = config or DRConfig()
        self.backup_manager = backup_manager or get_backup_manager()
        
        self.incidents: List[DRIncident] = []
        self.last_backup_time: Optional[datetime] = None
        self.is_running = False
        self._health_check_task: Optional[asyncio.Task] = None
        self._backup_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start DR monitoring"""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start health check loop
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        
        # Start automatic backup loop
        if self.config.auto_backup_enabled:
            self._backup_task = asyncio.create_task(self._backup_loop())
        
        logger.info("dr_manager_started",
                   health_check_interval=self.config.health_check_interval_seconds,
                   backup_interval=self.config.backup_interval_hours)
    
    async def stop(self):
        """Stop DR monitoring"""
        self.is_running = False
        
        if self._health_check_task:
            self._health_check_task.cancel()
        
        if self._backup_task:
            self._backup_task.cancel()
        
        logger.info("dr_manager_stopped")
    
    async def _health_check_loop(self):
        """Continuous health checking"""
        while self.is_running:
            try:
                result = await self.perform_health_check()
                
                if result.status in [DRStatus.CRITICAL, DRStatus.DEGRADED]:
                    await self._handle_unhealthy_state(result)
                
                await asyncio.sleep(self.config.health_check_interval_seconds)
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("health_check_loop_error", error=str(e))
                await asyncio.sleep(self.config.health_check_interval_seconds)
    
    async def _backup_loop(self):
        """Automatic backup scheduling"""
        while self.is_running:
            try:
                # Calculate time until next backup
                if self.last_backup_time:
                    next_backup = self.last_backup_time + timedelta(
                        hours=self.config.backup_interval_hours
                    )
                    wait_seconds = (next_backup - datetime.utcnow()).total_seconds()
                    
                    if wait_seconds > 0:
                        await asyncio.sleep(wait_seconds)
                
                # Create backup
                await self.create_scheduled_backup()
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("backup_loop_error", error=str(e))
                await asyncio.sleep(3600)  # Wait 1 hour on error
    
    async def perform_health_check(self) -> HealthCheckResult:
        """Perform comprehensive health check"""
        timestamp = datetime.utcnow()
        issues = []
        
        # Check database health
        db_healthy = await self._check_database_health()
        if not db_healthy:
            issues.append("Database connectivity issues detected")
        
        # Check backup system
        backup_healthy = self._check_backup_system()
        if not backup_healthy:
            issues.append("Backup system issues detected")
        
        # Check disk space
        disk_available = self._check_disk_space()
        if not disk_available:
            issues.append("Low disk space detected")
        
        # Check last backup age
        last_backup_age = self._get_last_backup_age_hours()
        if last_backup_age and last_backup_age > self.config.rpo_minutes / 60:
            issues.append(f"Last backup is {last_backup_age:.1f} hours old (RPO: {self.config.rpo_minutes} minutes)")
        
        # Determine overall status
        if len(issues) >= 3 or not db_healthy:
            status = DRStatus.CRITICAL
        elif len(issues) > 0:
            status = DRStatus.DEGRADED
        else:
            status = DRStatus.HEALTHY
        
        result = HealthCheckResult(
            timestamp=timestamp,
            database_healthy=db_healthy,
            backup_system_healthy=backup_healthy,
            disk_space_available=disk_available,
            last_backup_age_hours=last_backup_age,
            issues=issues,
            status=status
        )
        
        logger.info("health_check_completed",
                   status=status.value,
                   issues_count=len(issues))
        
        return result
    
    async def _check_database_health(self) -> bool:
        """Check database connectivity and health"""
        try:
            from rag.database import get_database
            
            db = await get_database()
            health = await db.health_check()
            return health.get("healthy", False)
        
        except Exception as e:
            logger.error("database_health_check_failed", error=str(e))
            return False
    
    def _check_backup_system(self) -> bool:
        """Check backup system health"""
        try:
            # Check if backup directory is accessible
            backup_dir = self.backup_manager.backup_dir
            if not backup_dir.exists():
                return False
            
            # Check if we can write to backup directory
            test_file = backup_dir / ".health_check"
            test_file.touch()
            test_file.unlink()
            
            return True
        
        except Exception as e:
            logger.error("backup_system_check_failed", error=str(e))
            return False
    
    def _check_disk_space(self, min_gb: float = 10.0) -> bool:
        """Check available disk space"""
        try:
            import shutil
            
            backup_dir = self.backup_manager.backup_dir
            stat = shutil.disk_usage(backup_dir)
            available_gb = stat.free / (1024 ** 3)
            
            return available_gb >= min_gb
        
        except Exception as e:
            logger.error("disk_space_check_failed", error=str(e))
            return False
    
    def _get_last_backup_age_hours(self) -> Optional[float]:
        """Get age of last successful backup in hours"""
        backups = self.backup_manager.list_backups()
        if not backups:
            return None
        
        completed = [b for b in backups if b.status.value == "completed"]
        if not completed:
            return None
        
        latest = max(completed, key=lambda x: x.started_at)
        age = datetime.utcnow() - latest.started_at
        return age.total_seconds() / 3600
    
    async def _handle_unhealthy_state(self, health_result: HealthCheckResult):
        """Handle unhealthy system state"""
        incident_id = f"incident_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        incident = DRIncident(
            id=incident_id,
            incident_type="health_check_failure",
            severity="critical" if health_result.status == DRStatus.CRITICAL else "warning",
            status=health_result.status,
            detected_at=health_result.timestamp,
            metrics={
                "database_healthy": health_result.database_healthy,
                "backup_system_healthy": health_result.backup_system_healthy,
                "disk_space_available": health_result.disk_space_available,
                "issues": health_result.issues
            }
        )
        
        logger.warning("unhealthy_state_detected",
                      incident_id=incident_id,
                      status=health_result.status.value,
                      issues=health_result.issues)
        
        # Attempt automatic recovery if enabled
        if self.config.failover_mode == FailoverMode.AUTOMATIC:
            await self._attempt_recovery(incident)
        
        # Send alerts
        await self._send_alerts(incident)
        
        self.incidents.append(incident)
    
    async def _attempt_recovery(self, incident: DRIncident):
        """Attempt automatic recovery"""
        logger.info("recovery_attempt_started", incident_id=incident.id)
        
        incident.status = DRStatus.RECOVERING
        
        try:
            # Try recovery actions in order
            if not incident.metrics.get("database_healthy"):
                # Database issue - try to restore from backup
                incident.actions_taken.append(RecoveryAction.RESTORE_BACKUP)
                await self._restore_latest_backup()
            
            if not incident.metrics.get("backup_system_healthy"):
                # Backup system issue - try to restart
                incident.actions_taken.append(RecoveryAction.RESTART_SERVICE)
                # In production, this would restart the backup service
                logger.info("backup_service_restart_simulated")
            
            # Verify recovery
            health_result = await self.perform_health_check()
            
            if health_result.status == DRStatus.HEALTHY:
                incident.success = True
                incident.status = DRStatus.HEALTHY
                incident.resolved_at = datetime.utcnow()
                logger.info("recovery_successful", incident_id=incident.id)
            else:
                incident.success = False
                incident.status = DRStatus.FAILED
                incident.error_message = "Recovery attempts did not resolve all issues"
                logger.error("recovery_failed", incident_id=incident.id)
        
        except Exception as e:
            incident.success = False
            incident.status = DRStatus.FAILED
            incident.error_message = str(e)
            logger.error("recovery_error", incident_id=incident.id, error=str(e))
    
    async def _restore_latest_backup(self):
        """Restore from the latest backup"""
        restore_points = self.backup_manager.get_restore_points()
        if not restore_points:
            raise Exception("No restore points available")
        
        latest = max(restore_points, key=lambda x: x.timestamp)
        
        # This would perform actual restoration
        logger.info("restore_initiated", backup_id=latest.backup_id)
        # await self.backup_manager.restore_backup(latest.backup_id, db_config)
    
    async def _send_alerts(self, incident: DRIncident):
        """Send alerts via configured channels"""
        if not self.config.alert_webhooks:
            return
        
        alert_data = {
            "incident_id": incident.id,
            "type": incident.incident_type,
            "severity": incident.severity,
            "status": incident.status.value,
            "detected_at": incident.detected_at.isoformat(),
            "metrics": incident.metrics
        }
        
        # In production, this would send to Slack, PagerDuty, etc.
        logger.info("alert_sent", incident_id=incident.id, alert_data=alert_data)
    
    async def create_scheduled_backup(self) -> bool:
        """Create a scheduled backup"""
        try:
            logger.info("scheduled_backup_started")
            
            # Get database config from environment
            db_config = {
                "host": os.getenv("POSTGRES_HOST", "localhost"),
                "port": os.getenv("POSTGRES_PORT", "5432"),
                "database": os.getenv("POSTGRES_DB", "dual_rag"),
                "user": os.getenv("POSTGRES_USER", "postgres"),
                "password": os.getenv("POSTGRES_PASSWORD", "postgres")
            }
            
            # Create backup
            metadata = await self.backup_manager.create_backup(
                db_config, 
                BackupType.FULL
            )
            
            if metadata and metadata.status.value == "completed":
                self.last_backup_time = datetime.utcnow()
                logger.info("scheduled_backup_completed",
                           backup_id=metadata.id,
                           size_mb=metadata.size_bytes / 1024 / 1024)
                return True
            else:
                logger.error("scheduled_backup_failed")
                return False
        
        except Exception as e:
            logger.error("scheduled_backup_error", error=str(e))
            return False
    
    def get_dr_status(self) -> Dict[str, Any]:
        """Get disaster recovery status"""
        recent_incidents = [
            i for i in self.incidents 
            if i.detected_at > datetime.utcnow() - timedelta(days=7)
        ]
        
        return {
            "enabled": self.config.enabled,
            "is_running": self.is_running,
            "last_backup_time": self.last_backup_time.isoformat() if self.last_backup_time else None,
            "rpo_minutes": self.config.rpo_minutes,
            "rto_minutes": self.config.rto_minutes,
            "failover_mode": self.config.failover_mode.value,
            "recent_incidents_count": len(recent_incidents),
            "unresolved_incidents": len([i for i in recent_incidents if not i.resolved_at]),
            "backup_stats": self.backup_manager.get_backup_stats()
        }
    
    def get_incidents(
        self, 
        days: int = 7,
        unresolved_only: bool = False
    ) -> List[DRIncident]:
        """Get DR incidents"""
        cutoff = datetime.utcnow() - timedelta(days=days)
        incidents = [i for i in self.incidents if i.detected_at > cutoff]
        
        if unresolved_only:
            incidents = [i for i in incidents if not i.resolved_at]
        
        return incidents


# Global DR manager
_dr_manager: Optional[DisasterRecoveryManager] = None


def get_dr_manager() -> DisasterRecoveryManager:
    """Get or create DR manager instance"""
    global _dr_manager
    
    if _dr_manager is None:
        _dr_manager = DisasterRecoveryManager()
    
    return _dr_manager

