"""
Backup and Restore System
Automated backups with retention policies and restoration

Author: Adrian Johnson <adrian207@gmail.com>
"""

import os
import asyncio
import shutil
import gzip
import json
import tempfile
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum

from pydantic import BaseModel
import structlog
import asyncpg

logger = structlog.get_logger()


class BackupType(str, Enum):
    """Backup type"""
    FULL = "full"
    INCREMENTAL = "incremental"
    DIFFERENTIAL = "differential"


class BackupStatus(str, Enum):
    """Backup status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class BackupMetadata(BaseModel):
    """Backup metadata"""
    id: str
    backup_type: BackupType
    status: BackupStatus
    size_bytes: int
    file_path: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    error: Optional[str] = None
    database_size: Optional[int] = None
    table_counts: Dict[str, int] = {}
    checksum: Optional[str] = None


class BackupConfig(BaseModel):
    """Backup configuration"""
    backup_dir: str = os.getenv("BACKUP_DIR", "/backups")
    retention_days: int = int(os.getenv("BACKUP_RETENTION_DAYS", "30"))
    max_backups: int = int(os.getenv("MAX_BACKUPS", "100"))
    compression_enabled: bool = True
    incremental_enabled: bool = False
    s3_enabled: bool = os.getenv("S3_BACKUP_ENABLED", "false").lower() == "true"
    s3_bucket: Optional[str] = os.getenv("S3_BACKUP_BUCKET")
    s3_region: Optional[str] = os.getenv("S3_BACKUP_REGION", "us-east-1")


class RestorePoint(BaseModel):
    """Restore point information"""
    backup_id: str
    timestamp: datetime
    backup_type: BackupType
    size_bytes: int
    database_size: int
    table_counts: Dict[str, int]
    file_path: str


class BackupManager:
    """Backup and restore manager"""
    
    def __init__(self, config: Optional[BackupConfig] = None):
        self.config = config or BackupConfig()
        self.backup_dir = Path(self.config.backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Metadata file
        self.metadata_file = self.backup_dir / "backup_metadata.json"
        self.metadata: List[BackupMetadata] = []
        self._load_metadata()
    
    def _load_metadata(self):
        """Load backup metadata"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    data = json.load(f)
                    self.metadata = [BackupMetadata(**item) for item in data]
            except Exception as e:
                logger.error("metadata_load_failed", error=str(e))
                self.metadata = []
    
    def _save_metadata(self):
        """Save backup metadata"""
        try:
            with open(self.metadata_file, 'w') as f:
                data = [m.dict() for m in self.metadata]
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.error("metadata_save_failed", error=str(e))
    
    async def create_backup(
        self, 
        db_config: Dict[str, str],
        backup_type: BackupType = BackupType.FULL
    ) -> Optional[BackupMetadata]:
        """Create a database backup"""
        backup_id = f"backup_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        started_at = datetime.utcnow()
        
        metadata = BackupMetadata(
            id=backup_id,
            backup_type=backup_type,
            status=BackupStatus.IN_PROGRESS,
            size_bytes=0,
            file_path="",
            started_at=started_at
        )
        
        try:
            logger.info("backup_started", backup_id=backup_id, backup_type=backup_type)
            
            # Create backup file path
            suffix = ".sql.gz" if self.config.compression_enabled else ".sql"
            backup_file = self.backup_dir / f"{backup_id}{suffix}"
            
            # Get database statistics
            table_counts = await self._get_table_counts(db_config)
            db_size = await self._get_database_size(db_config)
            
            # Create backup using pg_dump
            await self._pg_dump(db_config, backup_file)
            
            # Calculate size and checksum
            size_bytes = backup_file.stat().st_size
            checksum = self._calculate_checksum(backup_file)
            
            # Update metadata
            completed_at = datetime.utcnow()
            duration = (completed_at - started_at).total_seconds()
            
            metadata.status = BackupStatus.COMPLETED
            metadata.completed_at = completed_at
            metadata.duration_seconds = duration
            metadata.size_bytes = size_bytes
            metadata.file_path = str(backup_file)
            metadata.database_size = db_size
            metadata.table_counts = table_counts
            metadata.checksum = checksum
            
            # Save metadata
            self.metadata.append(metadata)
            self._save_metadata()
            
            logger.info("backup_completed",
                       backup_id=backup_id,
                       size_mb=size_bytes / 1024 / 1024,
                       duration_sec=duration)
            
            # Cleanup old backups
            await self._cleanup_old_backups()
            
            # Upload to S3 if enabled
            if self.config.s3_enabled:
                await self._upload_to_s3(backup_file)
            
            return metadata
        
        except Exception as e:
            logger.error("backup_failed", backup_id=backup_id, error=str(e))
            metadata.status = BackupStatus.FAILED
            metadata.error = str(e)
            metadata.completed_at = datetime.utcnow()
            self.metadata.append(metadata)
            self._save_metadata()
            return metadata
    
    async def _pg_dump(self, db_config: Dict[str, str], output_file: Path):
        """Execute pg_dump to create backup"""
        host = db_config.get("host", "localhost")
        port = db_config.get("port", "5432")
        database = db_config.get("database", "dual_rag")
        user = db_config.get("user", "postgres")
        password = db_config.get("password", "")
        
        # Set environment variable for password
        env = os.environ.copy()
        env["PGPASSWORD"] = password
        
        # Build pg_dump command
        cmd = [
            "pg_dump",
            f"--host={host}",
            f"--port={port}",
            f"--username={user}",
            f"--dbname={database}",
            "--format=plain",
            "--verbose",
            "--no-owner",
            "--no-acl"
        ]
        
        # Execute pg_dump
        if self.config.compression_enabled:
            # Pipe through gzip
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                raise Exception(f"pg_dump failed: {stderr.decode()}")
            
            # Compress output
            with gzip.open(output_file, 'wb') as f:
                f.write(stdout)
        else:
            # Direct output
            with open(output_file, 'wb') as f:
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=f,
                    stderr=asyncio.subprocess.PIPE,
                    env=env
                )
                
                _, stderr = await process.communicate()
                
                if process.returncode != 0:
                    raise Exception(f"pg_dump failed: {stderr.decode()}")
    
    async def _get_table_counts(self, db_config: Dict[str, str]) -> Dict[str, int]:
        """Get row counts for all tables"""
        try:
            conn = await asyncpg.connect(
                host=db_config.get("host", "localhost"),
                port=int(db_config.get("port", "5432")),
                database=db_config.get("database", "dual_rag"),
                user=db_config.get("user", "postgres"),
                password=db_config.get("password", "")
            )
            
            rows = await conn.fetch("""
                SELECT tablename, 
                       (xpath('/row/count/text()', 
                              xml_count))[1]::text::int as row_count
                FROM (
                    SELECT tablename, 
                           query_to_xml(format('SELECT COUNT(*) AS count FROM %I.%I',
                                              schemaname, tablename), false, true, '') as xml_count
                    FROM pg_tables
                    WHERE schemaname = 'public'
                ) t
            """)
            
            counts = {row['tablename']: row['row_count'] for row in rows}
            
            await conn.close()
            return counts
        
        except Exception as e:
            logger.error("get_table_counts_failed", error=str(e))
            return {}
    
    async def _get_database_size(self, db_config: Dict[str, str]) -> int:
        """Get database size in bytes"""
        try:
            conn = await asyncpg.connect(
                host=db_config.get("host", "localhost"),
                port=int(db_config.get("port", "5432")),
                database=db_config.get("database", "dual_rag"),
                user=db_config.get("user", "postgres"),
                password=db_config.get("password", "")
            )
            
            size = await conn.fetchval(
                "SELECT pg_database_size($1)", 
                db_config.get("database", "dual_rag")
            )
            
            await conn.close()
            return size
        
        except Exception as e:
            logger.error("get_database_size_failed", error=str(e))
            return 0
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA-256 checksum"""
        import hashlib
        
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                sha256.update(chunk)
        
        return sha256.hexdigest()
    
    async def restore_backup(
        self, 
        backup_id: str,
        db_config: Dict[str, str],
        force: bool = False
    ) -> bool:
        """Restore from a backup"""
        try:
            # Find backup metadata
            metadata = next((m for m in self.metadata if m.id == backup_id), None)
            if not metadata:
                raise ValueError(f"Backup {backup_id} not found")
            
            if metadata.status != BackupStatus.COMPLETED:
                raise ValueError(f"Backup {backup_id} is not completed")
            
            backup_file = Path(metadata.file_path)
            if not backup_file.exists():
                raise FileNotFoundError(f"Backup file not found: {backup_file}")
            
            # Verify checksum
            if metadata.checksum:
                current_checksum = self._calculate_checksum(backup_file)
                if current_checksum != metadata.checksum:
                    raise ValueError("Backup checksum mismatch - file may be corrupted")
            
            logger.info("restore_started", backup_id=backup_id)
            
            # Restore using psql
            await self._psql_restore(db_config, backup_file)
            
            logger.info("restore_completed", backup_id=backup_id)
            return True
        
        except Exception as e:
            logger.error("restore_failed", backup_id=backup_id, error=str(e))
            return False
    
    async def _psql_restore(self, db_config: Dict[str, str], backup_file: Path):
        """Execute psql to restore backup"""
        host = db_config.get("host", "localhost")
        port = db_config.get("port", "5432")
        database = db_config.get("database", "dual_rag")
        user = db_config.get("user", "postgres")
        password = db_config.get("password", "")
        
        # Set environment variable for password
        env = os.environ.copy()
        env["PGPASSWORD"] = password
        
        # Build psql command
        cmd = [
            "psql",
            f"--host={host}",
            f"--port={port}",
            f"--username={user}",
            f"--dbname={database}",
            "--quiet"
        ]
        
        # Execute psql
        if backup_file.suffix == ".gz":
            # Decompress and pipe to psql
            with gzip.open(backup_file, 'rb') as f:
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdin=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=env
                )
                
                _, stderr = await process.communicate(input=f.read())
                
                if process.returncode != 0:
                    raise Exception(f"psql restore failed: {stderr.decode()}")
        else:
            # Direct input
            with open(backup_file, 'rb') as f:
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdin=f,
                    stderr=asyncio.subprocess.PIPE,
                    env=env
                )
                
                _, stderr = await process.communicate()
                
                if process.returncode != 0:
                    raise Exception(f"psql restore failed: {stderr.decode()}")
    
    async def _cleanup_old_backups(self):
        """Remove old backups based on retention policy"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=self.config.retention_days)
            
            # Filter completed backups
            completed_backups = [
                m for m in self.metadata 
                if m.status == BackupStatus.COMPLETED
            ]
            
            # Sort by date
            completed_backups.sort(key=lambda x: x.started_at, reverse=True)
            
            # Remove backups beyond max count or retention
            to_remove = []
            for i, backup in enumerate(completed_backups):
                if i >= self.config.max_backups or backup.started_at < cutoff_date:
                    to_remove.append(backup)
            
            for backup in to_remove:
                try:
                    # Remove file
                    backup_file = Path(backup.file_path)
                    if backup_file.exists():
                        backup_file.unlink()
                    
                    # Remove from metadata
                    self.metadata.remove(backup)
                    
                    logger.info("backup_removed", backup_id=backup.id)
                
                except Exception as e:
                    logger.error("backup_remove_failed", 
                                backup_id=backup.id, 
                                error=str(e))
            
            if to_remove:
                self._save_metadata()
        
        except Exception as e:
            logger.error("cleanup_failed", error=str(e))
    
    async def _upload_to_s3(self, backup_file: Path):
        """Upload backup to S3 (placeholder for S3 integration)"""
        logger.info("s3_upload_skipped", 
                   message="S3 integration not yet implemented",
                   file=str(backup_file))
    
    def list_backups(self, status: Optional[BackupStatus] = None) -> List[BackupMetadata]:
        """List all backups"""
        if status:
            return [m for m in self.metadata if m.status == status]
        return self.metadata.copy()
    
    def get_backup(self, backup_id: str) -> Optional[BackupMetadata]:
        """Get backup metadata by ID"""
        return next((m for m in self.metadata if m.id == backup_id), None)
    
    def get_restore_points(self) -> List[RestorePoint]:
        """Get all available restore points"""
        completed = [m for m in self.metadata if m.status == BackupStatus.COMPLETED]
        
        return [
            RestorePoint(
                backup_id=m.id,
                timestamp=m.started_at,
                backup_type=m.backup_type,
                size_bytes=m.size_bytes,
                database_size=m.database_size or 0,
                table_counts=m.table_counts,
                file_path=m.file_path
            )
            for m in completed
        ]
    
    def get_backup_stats(self) -> Dict[str, Any]:
        """Get backup statistics"""
        completed = [m for m in self.metadata if m.status == BackupStatus.COMPLETED]
        
        if not completed:
            return {
                "total_backups": 0,
                "total_size_bytes": 0,
                "avg_backup_size_bytes": 0,
                "oldest_backup": None,
                "newest_backup": None
            }
        
        total_size = sum(m.size_bytes for m in completed)
        oldest = min(completed, key=lambda x: x.started_at)
        newest = max(completed, key=lambda x: x.started_at)
        
        return {
            "total_backups": len(completed),
            "total_size_bytes": total_size,
            "avg_backup_size_bytes": total_size // len(completed),
            "oldest_backup": oldest.started_at.isoformat(),
            "newest_backup": newest.started_at.isoformat(),
            "retention_days": self.config.retention_days,
            "max_backups": self.config.max_backups
        }


# Global backup manager
_backup_manager: Optional[BackupManager] = None


def get_backup_manager() -> BackupManager:
    """Get or create backup manager instance"""
    global _backup_manager
    
    if _backup_manager is None:
        _backup_manager = BackupManager()
    
    return _backup_manager

