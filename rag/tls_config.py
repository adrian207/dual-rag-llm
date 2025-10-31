"""
TLS/HTTPS Configuration for Data-in-Transit Encryption

Author: Adrian Johnson <adrian207@gmail.com>
"""

import os
import ssl
from pathlib import Path
from typing import Optional
import structlog

logger = structlog.get_logger()


class TLSConfiguration:
    """
    Manages TLS/HTTPS configuration for securing data in transit.
    """
    
    def __init__(
        self,
        cert_path: Optional[str] = None,
        key_path: Optional[str] = None,
        ca_cert_path: Optional[str] = None,
        require_client_cert: bool = False
    ):
        """
        Initialize TLS configuration.
        
        Args:
            cert_path: Path to SSL certificate file
            key_path: Path to SSL private key file
            ca_cert_path: Path to CA certificate (for client verification)
            require_client_cert: Require client certificates (mTLS)
        """
        self.cert_path = cert_path or os.getenv("SSL_CERT_PATH")
        self.key_path = key_path or os.getenv("SSL_KEY_PATH")
        self.ca_cert_path = ca_cert_path or os.getenv("SSL_CA_CERT_PATH")
        self.require_client_cert = require_client_cert
        
        self.validate_config()
    
    def validate_config(self):
        """Validate TLS configuration"""
        if self.cert_path and not Path(self.cert_path).exists():
            logger.warning("ssl_cert_not_found", path=self.cert_path)
        
        if self.key_path and not Path(self.key_path).exists():
            logger.warning("ssl_key_not_found", path=self.key_path)
        
        if self.require_client_cert and not self.ca_cert_path:
            logger.warning("client_cert_required_but_no_ca", 
                         msg="Client certificate verification requires CA cert")
    
    def create_ssl_context(self) -> Optional[ssl.SSLContext]:
        """
        Create SSL context for uvicorn.
        
        Returns:
            SSL context or None if TLS not configured
        """
        if not (self.cert_path and self.key_path):
            logger.info("tls_disabled", reason="no_cert_or_key")
            return None
        
        try:
            # Create SSL context
            ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            
            # Load certificate and private key
            ssl_context.load_cert_chain(
                certfile=self.cert_path,
                keyfile=self.key_path
            )
            
            # Configure minimum TLS version
            ssl_context.minimum_version = ssl.TLSVersion.TLSv1_2
            
            # Set secure cipher suites
            ssl_context.set_ciphers(
                'ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:!aNULL:!MD5:!DSS'
            )
            
            # Client certificate verification (mTLS)
            if self.require_client_cert and self.ca_cert_path:
                ssl_context.verify_mode = ssl.CERT_REQUIRED
                ssl_context.load_verify_locations(cafile=self.ca_cert_path)
                logger.info("mtls_enabled", ca_cert=self.ca_cert_path)
            else:
                ssl_context.verify_mode = ssl.CERT_NONE
            
            # Security options
            ssl_context.options |= ssl.OP_NO_TLSv1 | ssl.OP_NO_TLSv1_1
            ssl_context.options |= ssl.OP_SINGLE_DH_USE | ssl.OP_SINGLE_ECDH_USE
            
            logger.info("tls_enabled", 
                       cert=self.cert_path,
                       mtls=self.require_client_cert,
                       min_version="TLSv1.2")
            
            return ssl_context
            
        except Exception as e:
            logger.error("ssl_context_creation_failed", error=str(e))
            raise
    
    def get_uvicorn_ssl_config(self) -> dict:
        """
        Get SSL configuration for uvicorn server.
        
        Returns:
            Dictionary with ssl_keyfile and ssl_certfile
        """
        if not (self.cert_path and self.key_path):
            return {}
        
        config = {
            "ssl_keyfile": self.key_path,
            "ssl_certfile": self.cert_path,
            "ssl_version": ssl.PROTOCOL_TLS_SERVER,
            "ssl_cert_reqs": ssl.CERT_REQUIRED if self.require_client_cert else ssl.CERT_NONE,
        }
        
        if self.require_client_cert and self.ca_cert_path:
            config["ssl_ca_certs"] = self.ca_cert_path
        
        return config
    
    @staticmethod
    def generate_self_signed_cert(
        output_dir: str = ".",
        domain: str = "localhost",
        days_valid: int = 365
    ):
        """
        Generate self-signed certificate for development/testing.
        
        Args:
            output_dir: Directory to save certificate files
            domain: Domain name for certificate
            days_valid: Certificate validity in days
        
        Returns:
            Tuple of (cert_path, key_path)
        """
        try:
            from cryptography import x509
            from cryptography.x509.oid import NameOID
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.asymmetric import rsa
            from cryptography.hazmat.primitives import serialization
            from datetime import datetime, timedelta
            
            # Generate private key
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048,
            )
            
            # Generate certificate
            subject = issuer = x509.Name([
                x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
                x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "State"),
                x509.NameAttribute(NameOID.LOCALITY_NAME, "City"),
                x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Dual RAG LLM"),
                x509.NameAttribute(NameOID.COMMON_NAME, domain),
            ])
            
            cert = x509.CertificateBuilder().subject_name(
                subject
            ).issuer_name(
                issuer
            ).public_key(
                private_key.public_key()
            ).serial_number(
                x509.random_serial_number()
            ).not_valid_before(
                datetime.utcnow()
            ).not_valid_after(
                datetime.utcnow() + timedelta(days=days_valid)
            ).add_extension(
                x509.SubjectAlternativeName([
                    x509.DNSName(domain),
                    x509.DNSName(f"*.{domain}"),
                ]),
                critical=False,
            ).sign(private_key, hashes.SHA256())
            
            # Save certificate
            cert_path = Path(output_dir) / "cert.pem"
            with open(cert_path, "wb") as f:
                f.write(cert.public_bytes(serialization.Encoding.PEM))
            
            # Save private key
            key_path = Path(output_dir) / "key.pem"
            with open(key_path, "wb") as f:
                f.write(private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.TraditionalOpenSSL,
                    encryption_algorithm=serialization.NoEncryption()
                ))
            
            logger.info("self_signed_cert_generated", 
                       cert=str(cert_path),
                       key=str(key_path),
                       domain=domain,
                       valid_days=days_valid)
            
            return str(cert_path), str(key_path)
            
        except ImportError:
            logger.error("cryptography_not_installed",
                        msg="Install cryptography package to generate certificates")
            raise
        except Exception as e:
            logger.error("cert_generation_failed", error=str(e))
            raise


def get_tls_config_from_env() -> TLSConfiguration:
    """
    Create TLS configuration from environment variables.
    
    Environment Variables:
        SSL_CERT_PATH: Path to SSL certificate
        SSL_KEY_PATH: Path to SSL private key
        SSL_CA_CERT_PATH: Path to CA certificate
        SSL_REQUIRE_CLIENT_CERT: Require client certificates (true/false)
    
    Returns:
        TLSConfiguration instance
    """
    require_client = os.getenv("SSL_REQUIRE_CLIENT_CERT", "false").lower() == "true"
    
    return TLSConfiguration(
        cert_path=os.getenv("SSL_CERT_PATH"),
        key_path=os.getenv("SSL_KEY_PATH"),
        ca_cert_path=os.getenv("SSL_CA_CERT_PATH"),
        require_client_cert=require_client
    )

