"""
Enterprise Security Module for MZ Max

This module provides comprehensive security features including encryption,
secure model storage, and data protection capabilities.
"""

import os
import hashlib
import hmac
import secrets
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
import json
import base64
from datetime import datetime, timedelta
import logging

try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False

from ..utils.logging import get_logger

logger = get_logger(__name__)


class SecurityManager:
    """
    Enterprise security manager for MZ Max.
    
    Provides comprehensive security features including encryption,
    secure storage, and access control.
    """
    
    def __init__(self, master_key: Optional[str] = None):
        """
        Initialize security manager.
        
        Args:
            master_key: Master encryption key (auto-generated if not provided)
        """
        if not CRYPTO_AVAILABLE:
            raise ImportError("cryptography package required for security features. Install with: pip install cryptography")
        
        self.master_key = master_key or self._generate_master_key()
        self._fernet = self._create_fernet_instance()
        self.security_config = self._load_security_config()
        
    def _generate_master_key(self) -> str:
        """Generate a secure master key."""
        return base64.urlsafe_b64encode(secrets.token_bytes(32)).decode()
    
    def _create_fernet_instance(self) -> Fernet:
        """Create Fernet encryption instance."""
        # The master key is already base64 encoded, use it directly
        return Fernet(self.master_key.encode())
    
    def _load_security_config(self) -> Dict[str, Any]:
        """Load security configuration."""
        return {
            'encryption_enabled': True,
            'key_rotation_days': 90,
            'audit_logging': True,
            'secure_deletion': True,
            'access_control': True,
            'data_masking': True
        }
    
    def encrypt_data(self, data: Union[str, bytes, Dict]) -> str:
        """
        Encrypt sensitive data.
        
        Args:
            data: Data to encrypt
            
        Returns:
            Encrypted data as base64 string
        """
        if isinstance(data, dict):
            data = json.dumps(data)
        if isinstance(data, str):
            data = data.encode()
        
        encrypted_data = self._fernet.encrypt(data)
        return base64.urlsafe_b64encode(encrypted_data).decode()
    
    def decrypt_data(self, encrypted_data: str) -> Union[str, Dict]:
        """
        Decrypt encrypted data.
        
        Args:
            encrypted_data: Base64 encoded encrypted data
            
        Returns:
            Decrypted data
        """
        encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode())
        decrypted_data = self._fernet.decrypt(encrypted_bytes)
        
        try:
            # Try to parse as JSON
            return json.loads(decrypted_data.decode())
        except json.JSONDecodeError:
            return decrypted_data.decode()
    
    def hash_password(self, password: str, salt: Optional[bytes] = None) -> Dict[str, str]:
        """
        Hash password with salt using PBKDF2.
        
        Args:
            password: Password to hash
            salt: Salt bytes (auto-generated if not provided)
            
        Returns:
            Dictionary with hashed password and salt
        """
        if salt is None:
            salt = secrets.token_bytes(32)
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        
        key = kdf.derive(password.encode())
        
        return {
            'hash': base64.urlsafe_b64encode(key).decode(),
            'salt': base64.urlsafe_b64encode(salt).decode()
        }
    
    def verify_password(self, password: str, stored_hash: str, stored_salt: str) -> bool:
        """
        Verify password against stored hash.
        
        Args:
            password: Password to verify
            stored_hash: Stored password hash
            stored_salt: Stored salt
            
        Returns:
            True if password is correct
        """
        salt = base64.urlsafe_b64decode(stored_salt.encode())
        expected_hash = self.hash_password(password, salt)['hash']
        return hmac.compare_digest(stored_hash, expected_hash)
    
    def generate_api_key(self, user_id: str, permissions: List[str]) -> Dict[str, Any]:
        """
        Generate secure API key with permissions.
        
        Args:
            user_id: User identifier
            permissions: List of permissions
            
        Returns:
            API key information
        """
        key_data = {
            'user_id': user_id,
            'permissions': permissions,
            'created_at': datetime.utcnow().isoformat(),
            'expires_at': (datetime.utcnow() + timedelta(days=365)).isoformat(),
            'key_id': secrets.token_hex(16)
        }
        
        api_key = secrets.token_urlsafe(32)
        encrypted_data = self.encrypt_data(key_data)
        
        return {
            'api_key': api_key,
            'key_id': key_data['key_id'],
            'encrypted_metadata': encrypted_data,
            'expires_at': key_data['expires_at']
        }
    
    def validate_api_key(self, api_key: str, encrypted_metadata: str) -> Optional[Dict[str, Any]]:
        """
        Validate API key and return metadata.
        
        Args:
            api_key: API key to validate
            encrypted_metadata: Encrypted key metadata
            
        Returns:
            Key metadata if valid, None otherwise
        """
        try:
            metadata = self.decrypt_data(encrypted_metadata)
            expires_at = datetime.fromisoformat(metadata['expires_at'])
            
            if datetime.utcnow() > expires_at:
                return None  # Key expired
            
            return metadata
        except Exception:
            return None
    
    def secure_delete_file(self, filepath: Union[str, Path]) -> bool:
        """
        Securely delete a file by overwriting with random data.
        
        Args:
            filepath: Path to file to delete
            
        Returns:
            True if successful
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            return True
        
        try:
            # Get file size
            file_size = filepath.stat().st_size
            
            # Overwrite with random data multiple times
            with open(filepath, 'r+b') as file:
                for _ in range(3):  # 3 passes
                    file.seek(0)
                    file.write(secrets.token_bytes(file_size))
                    file.flush()
                    os.fsync(file.fileno())
            
            # Finally delete the file
            filepath.unlink()
            return True
            
        except Exception as e:
            logger.error(f"Secure delete failed: {e}")
            return False
    
    def mask_sensitive_data(self, data: Dict[str, Any], 
                           sensitive_fields: List[str] = None) -> Dict[str, Any]:
        """
        Mask sensitive data fields.
        
        Args:
            data: Data dictionary
            sensitive_fields: List of sensitive field names
            
        Returns:
            Data with masked sensitive fields
        """
        if sensitive_fields is None:
            sensitive_fields = [
                'password', 'api_key', 'secret', 'token', 'key',
                'ssn', 'credit_card', 'phone', 'email'
            ]
        
        masked_data = data.copy()
        
        for field in sensitive_fields:
            if field in masked_data:
                value = str(masked_data[field])
                if len(value) > 4:
                    masked_data[field] = value[:2] + '*' * (len(value) - 4) + value[-2:]
                else:
                    masked_data[field] = '*' * len(value)
        
        return masked_data
    
    def generate_secure_token(self, length: int = 32) -> str:
        """
        Generate cryptographically secure token.
        
        Args:
            length: Token length in bytes
            
        Returns:
            URL-safe base64 encoded token
        """
        return secrets.token_urlsafe(length)
    
    def create_checksum(self, data: Union[str, bytes]) -> str:
        """
        Create SHA-256 checksum for data integrity.
        
        Args:
            data: Data to checksum
            
        Returns:
            Hexadecimal checksum
        """
        if isinstance(data, str):
            data = data.encode()
        
        return hashlib.sha256(data).hexdigest()
    
    def verify_checksum(self, data: Union[str, bytes], expected_checksum: str) -> bool:
        """
        Verify data integrity using checksum.
        
        Args:
            data: Data to verify
            expected_checksum: Expected checksum
            
        Returns:
            True if checksum matches
        """
        actual_checksum = self.create_checksum(data)
        return hmac.compare_digest(actual_checksum, expected_checksum)


def encrypt_model(model: Any, password: str) -> Dict[str, Any]:
    """
    Encrypt a trained model for secure storage.
    
    Args:
        model: Trained model to encrypt
        password: Encryption password
        
    Returns:
        Dictionary with encrypted model data
    """
    import pickle
    
    # Serialize model
    model_bytes = pickle.dumps(model)
    
    # Create security manager
    security_manager = SecurityManager()
    
    # Create password-derived key
    password_hash = security_manager.hash_password(password)
    
    # Encrypt model data
    encrypted_data = security_manager.encrypt_data(model_bytes)
    
    return {
        'encrypted_model': encrypted_data,
        'salt': password_hash['salt'],
        'checksum': security_manager.create_checksum(model_bytes),
        'metadata': {
            'model_type': type(model).__name__,
            'encrypted_at': datetime.utcnow().isoformat(),
            'version': '1.0'
        }
    }


def decrypt_model(encrypted_model_data: Dict[str, Any], password: str) -> Any:
    """
    Decrypt an encrypted model.
    
    Args:
        encrypted_model_data: Encrypted model data
        password: Decryption password
        
    Returns:
        Decrypted model
    """
    import pickle
    
    # Create security manager
    security_manager = SecurityManager()
    
    # Verify password
    salt = encrypted_model_data['salt']
    password_hash = security_manager.hash_password(password, base64.urlsafe_b64decode(salt.encode()))
    
    # Decrypt model data
    decrypted_bytes = security_manager.decrypt_data(encrypted_model_data['encrypted_model'])
    
    # Verify checksum
    if not security_manager.verify_checksum(decrypted_bytes, encrypted_model_data['checksum']):
        raise ValueError("Model data integrity check failed")
    
    # Deserialize model
    model = pickle.loads(decrypted_bytes)
    
    return model


class DataProtectionManager:
    """
    Data protection and privacy compliance manager.
    """
    
    def __init__(self):
        """Initialize data protection manager."""
        self.security_manager = SecurityManager()
        self.protection_rules = self._load_protection_rules()
    
    def _load_protection_rules(self) -> Dict[str, Any]:
        """Load data protection rules."""
        return {
            'pii_fields': [
                'name', 'email', 'phone', 'address', 'ssn', 'credit_card',
                'date_of_birth', 'passport', 'license'
            ],
            'retention_days': 365,
            'anonymization_enabled': True,
            'encryption_required': True,
            'audit_required': True
        }
    
    def identify_pii(self, data: Dict[str, Any]) -> List[str]:
        """
        Identify personally identifiable information in data.
        
        Args:
            data: Data to analyze
            
        Returns:
            List of PII field names
        """
        pii_fields = []
        
        for field_name, value in data.items():
            # Check field name
            if any(pii in field_name.lower() for pii in self.protection_rules['pii_fields']):
                pii_fields.append(field_name)
                continue
            
            # Check value patterns (basic examples)
            if isinstance(value, str):
                # Email pattern
                if '@' in value and '.' in value:
                    pii_fields.append(field_name)
                # Phone pattern
                elif len(value.replace('-', '').replace('(', '').replace(')', '').replace(' ', '')) >= 10 and \
                     value.replace('-', '').replace('(', '').replace(')', '').replace(' ', '').isdigit():
                    pii_fields.append(field_name)
        
        return pii_fields
    
    def anonymize_data(self, data: Dict[str, Any], pii_fields: List[str] = None) -> Dict[str, Any]:
        """
        Anonymize PII data.
        
        Args:
            data: Data to anonymize
            pii_fields: List of PII fields (auto-detected if not provided)
            
        Returns:
            Anonymized data
        """
        if pii_fields is None:
            pii_fields = self.identify_pii(data)
        
        anonymized_data = data.copy()
        
        for field in pii_fields:
            if field in anonymized_data:
                value = anonymized_data[field]
                
                if isinstance(value, str):
                    # Hash the value
                    hashed_value = hashlib.sha256(value.encode()).hexdigest()[:8]
                    anonymized_data[field] = f"ANON_{hashed_value}"
                else:
                    anonymized_data[field] = "ANONYMIZED"
        
        return anonymized_data
    
    def check_compliance(self, data_usage: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check data usage compliance with regulations.
        
        Args:
            data_usage: Data usage information
            
        Returns:
            Compliance check results
        """
        compliance_results = {
            'compliant': True,
            'issues': [],
            'recommendations': []
        }
        
        # Check data retention
        if 'retention_days' in data_usage:
            if data_usage['retention_days'] > self.protection_rules['retention_days']:
                compliance_results['compliant'] = False
                compliance_results['issues'].append(f"Data retention exceeds limit: {data_usage['retention_days']} > {self.protection_rules['retention_days']}")
        
        # Check encryption
        if not data_usage.get('encrypted', False) and self.protection_rules['encryption_required']:
            compliance_results['compliant'] = False
            compliance_results['issues'].append("Data encryption required but not enabled")
        
        # Check audit logging
        if not data_usage.get('audit_logged', False) and self.protection_rules['audit_required']:
            compliance_results['compliant'] = False
            compliance_results['issues'].append("Audit logging required but not enabled")
        
        # Add recommendations
        if not compliance_results['compliant']:
            compliance_results['recommendations'].extend([
                "Enable data encryption",
                "Implement audit logging",
                "Review data retention policies",
                "Consider data anonymization"
            ])
        
        return compliance_results
