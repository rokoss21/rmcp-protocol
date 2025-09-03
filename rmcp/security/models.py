"""
Security models for RMCP
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum


class PermissionLevel(str, Enum):
    """Permission levels"""
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    ADMIN = "admin"


class ResourceType(str, Enum):
    """Resource types for RBAC"""
    TOOL = "tool"
    SERVER = "server"
    EXECUTION = "execution"
    TELEMETRY = "telemetry"
    SYSTEM = "system"


class User(BaseModel):
    """User model"""
    id: str = Field(..., description="Unique user identifier")
    username: str = Field(..., description="Username")
    email: str = Field(..., description="User email")
    tenant_id: str = Field(..., description="Tenant identifier")
    roles: List[str] = Field(default_factory=list, description="User roles")
    is_active: bool = Field(default=True, description="Whether user is active")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        use_enum_values = True


class Tenant(BaseModel):
    """Tenant model for multi-tenancy"""
    id: str = Field(..., description="Unique tenant identifier")
    name: str = Field(..., description="Tenant name")
    description: Optional[str] = Field(None, description="Tenant description")
    is_active: bool = Field(default=True, description="Whether tenant is active")
    settings: Dict[str, Any] = Field(default_factory=dict, description="Tenant-specific settings")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class APIKey(BaseModel):
    """API Key model"""
    id: str = Field(..., description="Unique API key identifier")
    key_hash: str = Field(..., description="Hashed API key")
    user_id: str = Field(..., description="Associated user ID")
    tenant_id: str = Field(..., description="Associated tenant ID")
    name: str = Field(..., description="API key name/description")
    permissions: List[str] = Field(default_factory=list, description="API key permissions")
    expires_at: Optional[datetime] = Field(None, description="Expiration date")
    last_used_at: Optional[datetime] = Field(None, description="Last usage timestamp")
    is_active: bool = Field(default=True, description="Whether API key is active")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        use_enum_values = True


class Role(BaseModel):
    """Role model"""
    id: str = Field(..., description="Unique role identifier")
    name: str = Field(..., description="Role name")
    description: str = Field(..., description="Role description")
    permissions: List[str] = Field(default_factory=list, description="Role permissions")
    tenant_id: Optional[str] = Field(None, description="Tenant-specific role")
    is_system: bool = Field(default=False, description="Whether this is a system role")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        use_enum_values = True


class Permission(BaseModel):
    """Permission model"""
    id: str = Field(..., description="Unique permission identifier")
    name: str = Field(..., description="Permission name")
    resource_type: ResourceType = Field(..., description="Resource type")
    action: PermissionLevel = Field(..., description="Permission level")
    description: str = Field(..., description="Permission description")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        use_enum_values = True


class AuditEvent(BaseModel):
    """Audit event model"""
    id: str = Field(..., description="Unique event identifier")
    user_id: Optional[str] = Field(None, description="User who performed the action")
    tenant_id: Optional[str] = Field(None, description="Tenant context")
    action: str = Field(..., description="Action performed")
    resource_type: ResourceType = Field(..., description="Type of resource affected")
    resource_id: Optional[str] = Field(None, description="ID of affected resource")
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional details")
    ip_address: Optional[str] = Field(None, description="Client IP address")
    user_agent: Optional[str] = Field(None, description="Client user agent")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    success: bool = Field(..., description="Whether action was successful")
    
    class Config:
        use_enum_values = True

