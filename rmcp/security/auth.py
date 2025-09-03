"""
Authentication module for RMCP
"""

import hashlib
import secrets
import time
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import jwt
from fastapi import HTTPException, status
from ..storage.database import DatabaseManager
from .models import User, APIKey, Tenant


class APIKeyAuth:
    """API Key authentication handler"""
    
    def __init__(self, secret_key: str):
        self.secret_key = secret_key
    
    def generate_api_key(self) -> str:
        """Generate a new API key"""
        return f"rmcp_{secrets.token_urlsafe(32)}"
    
    def hash_api_key(self, api_key: str) -> str:
        """Hash an API key for storage"""
        return hashlib.sha256(api_key.encode()).hexdigest()
    
    def verify_api_key(self, api_key: str, stored_hash: str) -> bool:
        """Verify an API key against its hash"""
        return self.hash_api_key(api_key) == stored_hash
    
    def create_jwt_token(self, user_id: str, tenant_id: str, permissions: list, expires_in: int = 3600) -> str:
        """Create a JWT token for authenticated user"""
        payload = {
            "user_id": user_id,
            "tenant_id": tenant_id,
            "permissions": permissions,
            "exp": datetime.utcnow() + timedelta(seconds=expires_in),
            "iat": datetime.utcnow()
        }
        return jwt.encode(payload, self.secret_key, algorithm="HS256")
    
    def verify_jwt_token(self, token: str) -> Dict[str, Any]:
        """Verify and decode a JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )
        except jwt.InvalidTokenError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )


class AuthManager:
    """Authentication manager"""
    
    def __init__(self, db_manager: DatabaseManager, secret_key: str):
        self.db_manager = db_manager
        self.auth = APIKeyAuth(secret_key)
        self._init_security_tables()
    
    def _init_security_tables(self):
        """Initialize security-related database tables"""
        from ..storage.schema import get_connection
        
        with get_connection(self.db_manager.db_path) as conn:
            cursor = conn.cursor()
            
            # Users table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    tenant_id TEXT NOT NULL,
                    roles TEXT NOT NULL DEFAULT '[]',
                    is_active BOOLEAN NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                    FOREIGN KEY (tenant_id) REFERENCES tenants(id) ON DELETE CASCADE
                )
            """)
            
            # Tenants table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tenants (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    is_active BOOLEAN NOT NULL DEFAULT 1,
                    settings TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
                )
            """)
            
            # API Keys table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS api_keys (
                    id TEXT PRIMARY KEY,
                    key_hash TEXT UNIQUE NOT NULL,
                    user_id TEXT NOT NULL,
                    tenant_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    permissions TEXT NOT NULL DEFAULT '[]',
                    expires_at TEXT,
                    last_used_at TEXT,
                    is_active BOOLEAN NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    FOREIGN KEY (tenant_id) REFERENCES tenants(id) ON DELETE CASCADE
                )
            """)
            
            # Roles table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS roles (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL,
                    permissions TEXT NOT NULL DEFAULT '[]',
                    tenant_id TEXT,
                    is_system BOOLEAN NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                    FOREIGN KEY (tenant_id) REFERENCES tenants(id) ON DELETE CASCADE
                )
            """)
            
            # Permissions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS permissions (
                    id TEXT PRIMARY KEY,
                    name TEXT UNIQUE NOT NULL,
                    resource_type TEXT NOT NULL,
                    action TEXT NOT NULL,
                    description TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
                )
            """)
            
            # Audit events table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS audit_events (
                    id TEXT PRIMARY KEY,
                    user_id TEXT,
                    tenant_id TEXT,
                    action TEXT NOT NULL,
                    resource_type TEXT NOT NULL,
                    resource_id TEXT,
                    details TEXT NOT NULL DEFAULT '{}',
                    ip_address TEXT,
                    user_agent TEXT,
                    timestamp TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                    success BOOLEAN NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL,
                    FOREIGN KEY (tenant_id) REFERENCES tenants(id) ON DELETE SET NULL
                )
            """)
            
            conn.commit()
    
    def create_tenant(self, tenant_id: str, name: str, description: str = None) -> Tenant:
        """Create a new tenant"""
        tenant = Tenant(
            id=tenant_id,
            name=name,
            description=description
        )
        
        from ..storage.schema import get_connection
        with get_connection(self.db_manager.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO tenants (id, name, description)
                VALUES (?, ?, ?)
            """, (tenant.id, tenant.name, tenant.description))
            conn.commit()
        
        return tenant
    
    def create_user(self, user_id: str, username: str, email: str, tenant_id: str, roles: list = None) -> User:
        """Create a new user"""
        import json
        
        user = User(
            id=user_id,
            username=username,
            email=email,
            tenant_id=tenant_id,
            roles=roles or []
        )
        
        from ..storage.schema import get_connection
        with get_connection(self.db_manager.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO users (id, username, email, tenant_id, roles)
                VALUES (?, ?, ?, ?, ?)
            """, (user.id, user.username, user.email, user.tenant_id, json.dumps(user.roles)))
            conn.commit()
        
        return user
    
    def create_api_key(self, user_id: str, tenant_id: str, name: str, permissions: list = None) -> tuple[str, APIKey]:
        """Create a new API key and return both the key and the model"""
        import json
        import uuid
        
        # Generate API key
        api_key = self.auth.generate_api_key()
        key_hash = self.auth.hash_api_key(api_key)
        
        # Create API key record
        api_key_id = str(uuid.uuid4())
        api_key_model = APIKey(
            id=api_key_id,
            key_hash=key_hash,
            user_id=user_id,
            tenant_id=tenant_id,
            name=name,
            permissions=permissions or []
        )
        
        from ..storage.schema import get_connection
        with get_connection(self.db_manager.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO api_keys (id, key_hash, user_id, tenant_id, name, permissions)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                api_key_model.id,
                api_key_model.key_hash,
                api_key_model.user_id,
                api_key_model.tenant_id,
                api_key_model.name,
                json.dumps(api_key_model.permissions)
            ))
            conn.commit()
        
        return api_key, api_key_model
    
    def authenticate_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Authenticate an API key and return user info"""
        from ..storage.schema import get_connection
        import json
        
        key_hash = self.auth.hash_api_key(api_key)
        
        with get_connection(self.db_manager.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT ak.id, ak.key_hash, ak.user_id, ak.tenant_id, ak.name, ak.permissions, 
                       ak.expires_at, ak.last_used_at, ak.is_active, ak.created_at,
                       u.username, u.email, u.roles, t.name as tenant_name
                FROM api_keys ak
                JOIN users u ON ak.user_id = u.id
                JOIN tenants t ON ak.tenant_id = t.id
                WHERE ak.key_hash = ? AND ak.is_active = 1 AND u.is_active = 1 AND t.is_active = 1
                AND (ak.expires_at IS NULL OR ak.expires_at > datetime('now'))
            """, (key_hash,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            # Update last used timestamp
            cursor.execute("""
                UPDATE api_keys SET last_used_at = datetime('now') WHERE id = ?
            """, (row[0],))
            conn.commit()
            
            # Safely parse JSON fields
            try:
                roles = json.loads(row[9]) if row[9] else []
            except (json.JSONDecodeError, TypeError):
                roles = []
            
            try:
                permissions = json.loads(row[5]) if row[5] else []
            except (json.JSONDecodeError, TypeError):
                permissions = []
            
            return {
                "user_id": row[2],  # user_id
                "tenant_id": row[3],  # tenant_id
                "username": row[10],  # username
                "email": row[11],  # email
                "roles": roles,
                "tenant_name": row[13],  # tenant_name
                "permissions": permissions
            }
    
    def get_user_permissions(self, user_id: str, tenant_id: str) -> list:
        """Get all permissions for a user"""
        from ..storage.schema import get_connection
        import json
        
        with get_connection(self.db_manager.db_path) as conn:
            cursor = conn.cursor()
            
            # Get user roles
            cursor.execute("""
                SELECT roles FROM users WHERE id = ? AND tenant_id = ?
            """, (user_id, tenant_id))
            user_row = cursor.fetchone()
            if not user_row:
                return []
            
            try:
                user_roles = json.loads(user_row[0]) if user_row[0] else []
            except (json.JSONDecodeError, TypeError):
                user_roles = []
            
            # Get permissions from roles
            permissions = set()
            for role_name in user_roles:
                cursor.execute("""
                    SELECT permissions FROM roles WHERE name = ? AND (tenant_id = ? OR is_system = 1)
                """, (role_name, tenant_id))
                role_row = cursor.fetchone()
                if role_row:
                    try:
                        role_permissions = json.loads(role_row[0]) if role_row[0] else []
                    except (json.JSONDecodeError, TypeError):
                        role_permissions = []
                    permissions.update(role_permissions)
            
            return list(permissions)
