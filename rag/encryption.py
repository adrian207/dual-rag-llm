"""
Enterprise Encryption Module
Provides data-at-rest and data-in-transit encryption capabilities

Author: Adrian Johnson <adrian207@gmail.com>
"""

import os
import base64
import hashlib
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
from cryptography.hazmat.backends import default_backend
import structlog

logger = structlog.get_logger()


class EncryptionManager:
    """
    Manages encryption keys and provides encryption/decryption services.
    
    Features:
    - AES-256 encryption via Fernet
    - Key derivation from passwords
    - Key rotation support
    - Encrypted field handling
    """
    
    def __init__(self, master_key: Optional[str] = None):
        """
        Initialize encryption manager.
        
        Args:
            master_key: Master encryption key (base64-encoded). 
                       If None, generates a new key.
        """
        if master_key:
            try:
                self.master_key = master_key.encode() if isinstance(master_key, str) else master_key
                self.fernet = Fernet(self.master_key)
                logger.info("encryption_manager_initialized", source="provided_key")
            except Exception as e:
                logger.error("invalid_master_key", error=str(e))
                raise ValueError(f"Invalid master key: {e}")
        else:
            # Generate new key
            self.master_key = Fernet.generate_key()
            self.fernet = Fernet(self.master_key)
            logger.info("encryption_manager_initialized", source="generated_key")
        
        self.key_created_at = datetime.utcnow()
        self.encryption_count = 0
        self.decryption_count = 0
    
    @staticmethod
    def generate_key() -> bytes:
        """Generate a new Fernet encryption key."""
        return Fernet.generate_key()
    
    @staticmethod
    def derive_key_from_password(password: str, salt: Optional[bytes] = None) -> tuple[bytes, bytes]:
        """
        Derive an encryption key from a password using PBKDF2.
        
        Args:
            password: User password
            salt: Salt for key derivation. If None, generates new salt.
        
        Returns:
            Tuple of (derived_key, salt)
        """
        if salt is None:
            salt = os.urandom(16)
        
        kdf = PBKDF2(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key, salt
    
    def encrypt(self, data: str) -> str:
        """
        Encrypt string data.
        
        Args:
            data: Plain text to encrypt
        
        Returns:
            Base64-encoded encrypted data
        """
        try:
            encrypted = self.fernet.encrypt(data.encode())
            self.encryption_count += 1
            return encrypted.decode()
        except Exception as e:
            logger.error("encryption_failed", error=str(e))
            raise
    
    def decrypt(self, encrypted_data: str) -> str:
        """
        Decrypt encrypted data.
        
        Args:
            encrypted_data: Base64-encoded encrypted data
        
        Returns:
            Decrypted plain text
        """
        try:
            decrypted = self.fernet.decrypt(encrypted_data.encode())
            self.decryption_count += 1
            return decrypted.decode()
        except Exception as e:
            logger.error("decryption_failed", error=str(e))
            raise
    
    def encrypt_dict(self, data: Dict[str, Any], fields_to_encrypt: list[str]) -> Dict[str, Any]:
        """
        Encrypt specific fields in a dictionary.
        
        Args:
            data: Dictionary to encrypt
            fields_to_encrypt: List of field names to encrypt
        
        Returns:
            Dictionary with encrypted fields (prefixed with 'encrypted_')
        """
        result = data.copy()
        
        for field in fields_to_encrypt:
            if field in result and result[field] is not None:
                encrypted_value = self.encrypt(str(result[field]))
                result[f"encrypted_{field}"] = encrypted_value
                del result[field]  # Remove plaintext
        
        return result
    
    def decrypt_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decrypt encrypted fields in a dictionary.
        
        Args:
            data: Dictionary with encrypted fields
        
        Returns:
            Dictionary with decrypted fields
        """
        result = data.copy()
        
        for key in list(result.keys()):
            if key.startswith("encrypted_"):
                original_field = key.replace("encrypted_", "")
                try:
                    result[original_field] = self.decrypt(result[key])
                    del result[key]
                except Exception as e:
                    logger.warning("field_decryption_failed", field=key, error=str(e))
        
        return result
    
    def rotate_key(self, new_key: Optional[bytes] = None) -> bytes:
        """
        Rotate encryption key.
        
        Args:
            new_key: New key to use. If None, generates new key.
        
        Returns:
            New encryption key
        """
        if new_key is None:
            new_key = Fernet.generate_key()
        
        old_key = self.master_key
        self.master_key = new_key
        self.fernet = Fernet(self.master_key)
        self.key_created_at = datetime.utcnow()
        
        logger.info("encryption_key_rotated", 
                   old_key_hash=hashlib.sha256(old_key).hexdigest()[:8],
                   new_key_hash=hashlib.sha256(new_key).hexdigest()[:8])
        
        return new_key
    
    def get_key_info(self) -> Dict[str, Any]:
        """
        Get information about the current encryption key.
        
        Returns:
            Dictionary with key metadata
        """
        return {
            "key_hash": hashlib.sha256(self.master_key).hexdigest()[:16],
            "created_at": self.key_created_at.isoformat(),
            "age_days": (datetime.utcnow() - self.key_created_at).days,
            "encryption_count": self.encryption_count,
            "decryption_count": self.decryption_count,
            "algorithm": "AES-256 (Fernet)"
        }


class FieldEncryptor:
    """
    Helper for encrypting/decrypting specific fields in data models.
    """
    
    def __init__(self, encryption_manager: EncryptionManager):
        self.encryption_manager = encryption_manager
    
    def encrypt_sensitive_fields(
        self, 
        obj: Any, 
        sensitive_fields: list[str]
    ) -> Any:
        """
        Encrypt sensitive fields in a Pydantic model or dict.
        
        Args:
            obj: Object to encrypt
            sensitive_fields: List of field names to encrypt
        
        Returns:
            Object with encrypted fields
        """
        if hasattr(obj, 'dict'):
            # Pydantic model
            data = obj.dict()
        else:
            data = dict(obj)
        
        for field in sensitive_fields:
            if field in data and data[field] is not None:
                data[field] = self.encryption_manager.encrypt(str(data[field]))
        
        return data
    
    def decrypt_sensitive_fields(
        self,
        data: Dict[str, Any],
        sensitive_fields: list[str]
    ) -> Dict[str, Any]:
        """
        Decrypt sensitive fields in data.
        
        Args:
            data: Data dictionary
            sensitive_fields: List of field names to decrypt
        
        Returns:
            Data with decrypted fields
        """
        result = data.copy()
        
        for field in sensitive_fields:
            if field in result and result[field] is not None:
                try:
                    result[field] = self.encryption_manager.decrypt(result[field])
                except Exception as e:
                    logger.warning("field_decryption_failed", 
                                 field=field, 
                                 error=str(e))
        
        return result


def hash_password(password: str, salt: Optional[bytes] = None) -> tuple[str, str]:
    """
    Hash a password using PBKDF2.
    
    Args:
        password: Password to hash
        salt: Salt for hashing. If None, generates new salt.
    
    Returns:
        Tuple of (hash, salt) both base64-encoded
    """
    if salt is None:
        salt = os.urandom(16)
    elif isinstance(salt, str):
        salt = base64.b64decode(salt)
    
    kdf = PBKDF2(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
        backend=default_backend()
    )
    
    password_hash = base64.b64encode(kdf.derive(password.encode())).decode()
    salt_b64 = base64.b64encode(salt).decode()
    
    return password_hash, salt_b64


def verify_password(password: str, password_hash: str, salt: str) -> bool:
    """
    Verify a password against a hash.
    
    Args:
        password: Password to verify
        password_hash: Stored password hash (base64)
        salt: Stored salt (base64)
    
    Returns:
        True if password matches
    """
    computed_hash, _ = hash_password(password, salt)
    return computed_hash == password_hash

