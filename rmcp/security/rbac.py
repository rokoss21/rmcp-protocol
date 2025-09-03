"""
Role-Based Access Control (RBAC) module for RMCP
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
import json
from ..storage.database import DatabaseManager
from .models import Role, Permission, ResourceType, PermissionLevel


class RBACManager:
    """Role-Based Access Control manager"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self._init_default_permissions()
        self._init_default_roles()
    
    def _init_default_permissions(self):
        """Initialize default permissions"""
        from ..storage.schema import get_connection
        
        default_permissions = [
            # Tool permissions
            ("tool:read", ResourceType.TOOL, PermissionLevel.READ, "Read tool information"),
            ("tool:execute", ResourceType.TOOL, PermissionLevel.EXECUTE, "Execute tools"),
            ("tool:admin", ResourceType.TOOL, PermissionLevel.ADMIN, "Administer tools"),
            
            # Server permissions
            ("server:read", ResourceType.SERVER, PermissionLevel.READ, "Read server information"),
            ("server:admin", ResourceType.SERVER, PermissionLevel.ADMIN, "Administer servers"),
            
            # Execution permissions
            ("execution:read", ResourceType.EXECUTION, PermissionLevel.READ, "Read execution history"),
            ("execution:execute", ResourceType.EXECUTION, PermissionLevel.EXECUTE, "Execute tasks"),
            ("execution:admin", ResourceType.EXECUTION, PermissionLevel.ADMIN, "Administer executions"),
            
            # Telemetry permissions
            ("telemetry:read", ResourceType.TELEMETRY, PermissionLevel.READ, "Read telemetry data"),
            ("telemetry:admin", ResourceType.TELEMETRY, PermissionLevel.ADMIN, "Administer telemetry"),
            
            # System permissions
            ("system:read", ResourceType.SYSTEM, PermissionLevel.READ, "Read system information"),
            ("system:admin", ResourceType.SYSTEM, PermissionLevel.ADMIN, "Administer system"),
        ]
        
        with get_connection(self.db_manager.db_path) as conn:
            cursor = conn.cursor()
            
            for perm_id, resource_type, action, description in default_permissions:
                cursor.execute("""
                    INSERT OR IGNORE INTO permissions (id, name, resource_type, action, description)
                    VALUES (?, ?, ?, ?, ?)
                """, (perm_id, perm_id, resource_type.value, action.value, description))
            
            conn.commit()
    
    def _init_default_roles(self):
        """Initialize default system roles"""
        from ..storage.schema import get_connection
        
        default_roles = [
            ("admin", "System Administrator", [
                "tool:admin", "server:admin", "execution:admin", 
                "telemetry:admin", "system:admin"
            ], True),
            ("user", "Regular User", [
                "tool:read", "tool:execute", "execution:execute", 
                "execution:read", "telemetry:read", "system:read"
            ], True),
            ("readonly", "Read-Only User", [
                "tool:read", "execution:read", "telemetry:read", "system:read"
            ], True),
        ]
        
        with get_connection(self.db_manager.db_path) as conn:
            cursor = conn.cursor()
            
            for role_id, description, permissions, is_system in default_roles:
                cursor.execute("""
                    INSERT OR IGNORE INTO roles (id, name, description, permissions, is_system)
                    VALUES (?, ?, ?, ?, ?)
                """, (role_id, role_id, description, json.dumps(permissions), is_system))
            
            conn.commit()
    
    def create_role(self, role_id: str, name: str, description: str, 
                   permissions: List[str], tenant_id: Optional[str] = None) -> Role:
        """Create a new role"""
        role = Role(
            id=role_id,
            name=name,
            description=description,
            permissions=permissions,
            tenant_id=tenant_id
        )
        
        from ..storage.schema import get_connection
        with get_connection(self.db_manager.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO roles (id, name, description, permissions, tenant_id)
                VALUES (?, ?, ?, ?, ?)
            """, (role.id, role.name, role.description, json.dumps(role.permissions), role.tenant_id))
            conn.commit()
        
        return role
    
    def get_role(self, role_id: str, tenant_id: Optional[str] = None) -> Optional[Role]:
        """Get a role by ID"""
        from ..storage.schema import get_connection
        
        with get_connection(self.db_manager.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, name, description, permissions, tenant_id, is_system, created_at
                FROM roles 
                WHERE id = ? AND (tenant_id = ? OR is_system = 1)
            """, (role_id, tenant_id))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            try:
                permissions = json.loads(row[3]) if row[3] else []
            except (json.JSONDecodeError, TypeError):
                permissions = []
            
            return Role(
                id=row[0],
                name=row[1],
                description=row[2],
                permissions=permissions,
                tenant_id=row[4],
                is_system=bool(row[5]),
                created_at=datetime.fromisoformat(row[6].replace('Z', '+00:00'))
            )
    
    def list_roles(self, tenant_id: Optional[str] = None) -> List[Role]:
        """List all roles for a tenant"""
        from ..storage.schema import get_connection
        
        with get_connection(self.db_manager.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, name, description, permissions, tenant_id, is_system, created_at
                FROM roles 
                WHERE tenant_id = ? OR is_system = 1
                ORDER BY is_system DESC, name
            """, (tenant_id,))
            
            roles = []
            for row in cursor.fetchall():
                try:
                    permissions = json.loads(row[3]) if row[3] else []
                except (json.JSONDecodeError, TypeError):
                    permissions = []
                
                roles.append(Role(
                    id=row[0],
                    name=row[1],
                    description=row[2],
                    permissions=permissions,
                    tenant_id=row[4],
                    is_system=bool(row[5]),
                    created_at=datetime.fromisoformat(row[6].replace('Z', '+00:00'))
                ))
            
            return roles
    
    def check_permission(self, user_permissions: List[str], required_permission: str) -> bool:
        """Check if user has required permission"""
        return required_permission in user_permissions
    
    def check_resource_permission(self, user_permissions: List[str], resource_type: ResourceType, 
                                action: PermissionLevel) -> bool:
        """Check if user has permission for specific resource and action"""
        permission_name = f"{resource_type.value}:{action.value}"
        return self.check_permission(user_permissions, permission_name)
    
    def filter_resources_by_permission(self, resources: List[Dict[str, Any]], 
                                     user_permissions: List[str], 
                                     resource_type: ResourceType, 
                                     action: PermissionLevel) -> List[Dict[str, Any]]:
        """Filter resources based on user permissions"""
        if self.check_resource_permission(user_permissions, resource_type, action):
            return resources
        else:
            return []
    
    def get_user_effective_permissions(self, user_id: str, tenant_id: str) -> List[str]:
        """Get all effective permissions for a user (from roles + direct permissions)"""
        from ..storage.schema import get_connection
        
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
            
            # Get direct API key permissions
            cursor.execute("""
                SELECT permissions FROM api_keys WHERE user_id = ? AND tenant_id = ? AND is_active = 1
            """, (user_id, tenant_id))
            for row in cursor.fetchall():
                try:
                    api_key_permissions = json.loads(row[0]) if row[0] else []
                except (json.JSONDecodeError, TypeError):
                    api_key_permissions = []
                permissions.update(api_key_permissions)
            
            return list(permissions)
