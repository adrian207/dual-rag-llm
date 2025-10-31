"""
Unit Tests for Backup Module
Tests for automated backup and restore functionality

Author: Adrian Johnson <adrian207@gmail.com>
"""

import pytest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, mock_open
from rag.backup import (
    BackupManager,
    BackupConfig,
    BackupMetadata,
    BackupType,
    BackupStatus,
    get_backup_manager
)


@pytest.mark.unit
class TestBackupConfig:
    """Test backup configuration"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = BackupConfig()
        assert config.backup_dir == "/backups"
        assert config.retention_days == 30
        assert config.max_backups == 100
        assert config.compression_enabled is True
    
    def test_custom_config(self):
        """Test custom configuration"""
        config = BackupConfig(
            backup_dir="/custom/backups",
            retention_days=60,
            max_backups=200
        )
        assert config.backup_dir == "/custom/backups"
        assert config.retention_days == 60
        assert config.max_backups == 200


@pytest.mark.unit
class TestBackupMetadata:
    """Test backup metadata model"""
    
    def test_create_metadata(self):
        """Test metadata creation"""
        metadata = BackupMetadata(
            id="backup_20241031_120000",
            backup_type=BackupType.FULL,
            status=BackupStatus.COMPLETED,
            size_bytes=1024000,
            file_path="/backups/backup_20241031_120000.sql.gz",
            started_at=datetime.utcnow(),
            completed_at=datetime.utcnow(),
            duration_seconds=120.5
        )
        assert metadata.id == "backup_20241031_120000"
        assert metadata.backup_type == BackupType.FULL
        assert metadata.status == BackupStatus.COMPLETED
        assert metadata.duration_seconds == 120.5


@pytest.mark.unit
class TestBackupManager:
    """Test backup manager operations"""
    
    def test_initialization(self, tmp_path):
        """Test backup manager initialization"""
        config = BackupConfig(backup_dir=str(tmp_path))
        manager = BackupManager(config)
        
        assert manager.backup_dir.exists()
        assert manager.metadata == []
    
    def test_load_metadata(self, tmp_path):
        """Test loading backup metadata"""
        config = BackupConfig(backup_dir=str(tmp_path))
        
        # Create mock metadata file
        metadata_file = tmp_path / "backup_metadata.json"
        metadata_file.write_text('[{"id": "test", "backup_type": "full", "status": "completed", "size_bytes": 1024, "file_path": "/test", "started_at": "2024-10-31T12:00:00"}]')
        
        manager = BackupManager(config)
        assert len(manager.metadata) == 1
        assert manager.metadata[0].id == "test"
    
    def test_calculate_checksum(self, tmp_path):
        """Test checksum calculation"""
        config = BackupConfig(backup_dir=str(tmp_path))
        manager = BackupManager(config)
        
        # Create test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")
        
        checksum = manager._calculate_checksum(test_file)
        assert checksum is not None
        assert len(checksum) == 64  # SHA-256 hex digest
    
    def test_list_backups(self, tmp_path):
        """Test listing backups"""
        config = BackupConfig(backup_dir=str(tmp_path))
        manager = BackupManager(config)
        
        # Add test backups
        manager.metadata = [
            BackupMetadata(
                id=f"backup_{i}",
                backup_type=BackupType.FULL,
                status=BackupStatus.COMPLETED,
                size_bytes=1024,
                file_path=f"/backups/backup_{i}.sql.gz",
                started_at=datetime.utcnow()
            )
            for i in range(3)
        ]
        
        backups = manager.list_backups()
        assert len(backups) == 3
        
        # Filter by status
        backups_completed = manager.list_backups(status=BackupStatus.COMPLETED)
        assert len(backups_completed) == 3
    
    def test_get_backup(self, tmp_path):
        """Test getting specific backup"""
        config = BackupConfig(backup_dir=str(tmp_path))
        manager = BackupManager(config)
        
        test_backup = BackupMetadata(
            id="backup_test",
            backup_type=BackupType.FULL,
            status=BackupStatus.COMPLETED,
            size_bytes=1024,
            file_path="/backups/test.sql.gz",
            started_at=datetime.utcnow()
        )
        manager.metadata = [test_backup]
        
        backup = manager.get_backup("backup_test")
        assert backup is not None
        assert backup.id == "backup_test"
        
        not_found = manager.get_backup("nonexistent")
        assert not_found is None
    
    def test_get_restore_points(self, tmp_path):
        """Test getting restore points"""
        config = BackupConfig(backup_dir=str(tmp_path))
        manager = BackupManager(config)
        
        # Add completed and failed backups
        manager.metadata = [
            BackupMetadata(
                id="backup_1",
                backup_type=BackupType.FULL,
                status=BackupStatus.COMPLETED,
                size_bytes=1024,
                file_path="/backups/backup_1.sql.gz",
                started_at=datetime.utcnow(),
                database_size=5000,
                table_counts={"users": 100}
            ),
            BackupMetadata(
                id="backup_2",
                backup_type=BackupType.FULL,
                status=BackupStatus.FAILED,
                size_bytes=0,
                file_path="/backups/backup_2.sql.gz",
                started_at=datetime.utcnow()
            )
        ]
        
        restore_points = manager.get_restore_points()
        assert len(restore_points) == 1  # Only completed backups
        assert restore_points[0].backup_id == "backup_1"
    
    def test_get_backup_stats(self, tmp_path):
        """Test backup statistics"""
        config = BackupConfig(backup_dir=str(tmp_path))
        manager = BackupManager(config)
        
        now = datetime.utcnow()
        manager.metadata = [
            BackupMetadata(
                id=f"backup_{i}",
                backup_type=BackupType.FULL,
                status=BackupStatus.COMPLETED,
                size_bytes=1024 * i,
                file_path=f"/backups/backup_{i}.sql.gz",
                started_at=now - timedelta(hours=i)
            )
            for i in range(1, 4)
        ]
        
        stats = manager.get_backup_stats()
        assert stats["total_backups"] == 3
        assert stats["total_size_bytes"] == 1024 + 2048 + 3072
        assert "oldest_backup" in stats
        assert "newest_backup" in stats
    
    @pytest.mark.asyncio
    async def test_cleanup_old_backups(self, tmp_path):
        """Test cleanup of old backups"""
        config = BackupConfig(
            backup_dir=str(tmp_path),
            retention_days=7,
            max_backups=2
        )
        manager = BackupManager(config)
        
        # Create old and new backups
        old_date = datetime.utcnow() - timedelta(days=10)
        new_date = datetime.utcnow()
        
        old_file = tmp_path / "old_backup.sql.gz"
        old_file.write_text("old backup")
        
        new_file = tmp_path / "new_backup.sql.gz"
        new_file.write_text("new backup")
        
        manager.metadata = [
            BackupMetadata(
                id="old_backup",
                backup_type=BackupType.FULL,
                status=BackupStatus.COMPLETED,
                size_bytes=1024,
                file_path=str(old_file),
                started_at=old_date
            ),
            BackupMetadata(
                id="new_backup",
                backup_type=BackupType.FULL,
                status=BackupStatus.COMPLETED,
                size_bytes=1024,
                file_path=str(new_file),
                started_at=new_date
            )
        ]
        
        await manager._cleanup_old_backups()
        
        # Old backup should be removed
        assert not old_file.exists()
        assert new_file.exists()
        assert len(manager.metadata) == 1


@pytest.mark.unit
def test_get_backup_manager_singleton():
    """Test backup manager singleton pattern"""
    with patch('rag.backup._backup_manager', None):
        manager1 = get_backup_manager()
        manager2 = get_backup_manager()
        assert manager1 is manager2

