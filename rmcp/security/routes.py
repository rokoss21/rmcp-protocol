"""
Security API routes for RMCP
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends, status, Request
from pydantic import BaseModel, Field

from .auth import AuthManager
from .rbac import RBACManager
from .audit import AuditLogger
from .approval import ApprovalGateway
from .circuit_breaker import CircuitBreakerManager
from .models import User, Tenant, APIKey, Role, ApprovalRequest, ResourceType, PermissionLevel
from .middleware import get_current_user


router = APIRouter(prefix="/security", tags=["security"])


# Request/Response models
class CreateTenantRequest(BaseModel):
    """Request to create a tenant"""
    tenant_id: str = Field(..., description="Unique tenant identifier")
    name: str = Field(..., description="Tenant name")
    description: Optional[str] = Field(None, description="Tenant description")


class CreateUserRequest(BaseModel):
    """Request to create a user"""
    user_id: str = Field(..., description="Unique user identifier")
    username: str = Field(..., description="Username")
    email: str = Field(..., description="User email")
    tenant_id: str = Field(..., description="Tenant identifier")
    roles: List[str] = Field(default_factory=list, description="User roles")


class CreateAPIKeyRequest(BaseModel):
    """Request to create an API key"""
    user_id: str = Field(..., description="User identifier")
    tenant_id: str = Field(..., description="Tenant identifier")
    name: str = Field(..., description="API key name")
    permissions: List[str] = Field(default_factory=list, description="API key permissions")
    expires_in_hours: Optional[int] = Field(None, description="Expiration in hours")


class CreateAPIKeyResponse(BaseModel):
    """Response for API key creation"""
    api_key: str = Field(..., description="Generated API key")
    api_key_info: APIKey = Field(..., description="API key information")


class CreateRoleRequest(BaseModel):
    """Request to create a role"""
    role_id: str = Field(..., description="Unique role identifier")
    name: str = Field(..., description="Role name")
    description: str = Field(..., description="Role description")
    permissions: List[str] = Field(..., description="Role permissions")
    tenant_id: Optional[str] = Field(None, description="Tenant-specific role")


class ApprovalActionRequest(BaseModel):
    """Request to approve/reject an approval request"""
    action: str = Field(..., description="Action: 'approve' or 'reject'")


# Dependency injection
def get_auth_manager(request: Request) -> AuthManager:
    """Get auth manager from app state"""
    return request.app.state.auth_manager


def get_rbac_manager(request: Request) -> RBACManager:
    """Get RBAC manager from app state"""
    return request.app.state.rbac_manager


def get_audit_logger(request: Request) -> AuditLogger:
    """Get audit logger from app state"""
    return request.app.state.audit_logger


def get_approval_gateway(request: Request) -> ApprovalGateway:
    """Get approval gateway from app state"""
    return request.app.state.approval_gateway


def get_circuit_breaker_manager(request: Request) -> CircuitBreakerManager:
    """Get circuit breaker manager from app state"""
    return request.app.state.circuit_breaker_manager


# Tenant management
@router.post("/tenants", response_model=Tenant)
async def create_tenant(
    request: CreateTenantRequest,
    auth_manager: AuthManager = Depends(get_auth_manager),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user)
):
    """Create a new tenant"""
    if not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    
    # Check if user has system admin permissions
    if "system:admin" not in current_user.get("permissions", []):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="System admin permission required")
    
    try:
        tenant = auth_manager.create_tenant(
            tenant_id=request.tenant_id,
            name=request.name,
            description=request.description
        )
        return tenant
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


# User management
@router.post("/users", response_model=User)
async def create_user(
    request: CreateUserRequest,
    auth_manager: AuthManager = Depends(get_auth_manager),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user)
):
    """Create a new user"""
    if not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    
    # Check if user has system admin permissions or is in the same tenant
    user_permissions = current_user.get("permissions", [])
    if "system:admin" not in user_permissions and current_user.get("tenant_id") != request.tenant_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient permissions")
    
    try:
        user = auth_manager.create_user(
            user_id=request.user_id,
            username=request.username,
            email=request.email,
            tenant_id=request.tenant_id,
            roles=request.roles
        )
        return user
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


# API Key management
@router.post("/api-keys", response_model=CreateAPIKeyResponse)
async def create_api_key(
    request: CreateAPIKeyRequest,
    auth_manager: AuthManager = Depends(get_auth_manager),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user)
):
    """Create a new API key"""
    if not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    
    # Check if user has system admin permissions or is creating for themselves
    user_permissions = current_user.get("permissions", [])
    if "system:admin" not in user_permissions and current_user.get("user_id") != request.user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient permissions")
    
    try:
        api_key, api_key_info = auth_manager.create_api_key(
            user_id=request.user_id,
            tenant_id=request.tenant_id,
            name=request.name,
            permissions=request.permissions
        )
        return CreateAPIKeyResponse(api_key=api_key, api_key_info=api_key_info)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


# Role management
@router.post("/roles", response_model=Role)
async def create_role(
    request: CreateRoleRequest,
    rbac_manager: RBACManager = Depends(get_rbac_manager),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user)
):
    """Create a new role"""
    if not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    
    # Check if user has system admin permissions or is in the same tenant
    user_permissions = current_user.get("permissions", [])
    if "system:admin" not in user_permissions and current_user.get("tenant_id") != request.tenant_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient permissions")
    
    try:
        role = rbac_manager.create_role(
            role_id=request.role_id,
            name=request.name,
            description=request.description,
            permissions=request.permissions,
            tenant_id=request.tenant_id
        )
        return role
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.get("/roles", response_model=List[Role])
async def list_roles(
    tenant_id: Optional[str] = None,
    rbac_manager: RBACManager = Depends(get_rbac_manager),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user)
):
    """List roles"""
    if not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    
    # Use current user's tenant if not specified
    if not tenant_id:
        tenant_id = current_user.get("tenant_id")
    
    # Check if user has system admin permissions or is in the same tenant
    user_permissions = current_user.get("permissions", [])
    if "system:admin" not in user_permissions and current_user.get("tenant_id") != tenant_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient permissions")
    
    roles = rbac_manager.list_roles(tenant_id)
    return roles


# Approval management
@router.get("/approvals", response_model=List[ApprovalRequest])
async def list_pending_approvals(
    approval_gateway: ApprovalGateway = Depends(get_approval_gateway),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user)
):
    """List pending approval requests"""
    if not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    
    # Check if user has admin permissions
    user_permissions = current_user.get("permissions", [])
    if not any(perm.endswith(":admin") for perm in user_permissions):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin permission required")
    
    tenant_id = current_user.get("tenant_id")
    approvals = approval_gateway.get_pending_requests(tenant_id)
    return approvals


@router.post("/approvals/{approval_id}/action")
async def handle_approval(
    approval_id: str,
    request: ApprovalActionRequest,
    approval_gateway: ApprovalGateway = Depends(get_approval_gateway),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user)
):
    """Approve or reject an approval request"""
    if not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    
    # Check if user has admin permissions
    user_permissions = current_user.get("permissions", [])
    if not any(perm.endswith(":admin") for perm in user_permissions):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin permission required")
    
    approver_id = current_user.get("user_id")
    
    if request.action == "approve":
        success = approval_gateway.approve_request(approval_id, approver_id)
    elif request.action == "reject":
        success = approval_gateway.reject_request(approval_id, approver_id)
    else:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid action")
    
    if not success:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Failed to process approval")
    
    return {"message": f"Approval request {request.action}d successfully"}


# Circuit breaker management
@router.get("/circuit-breakers")
async def get_circuit_breaker_stats(
    circuit_breaker_manager: CircuitBreakerManager = Depends(get_circuit_breaker_manager),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user)
):
    """Get circuit breaker statistics"""
    if not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    
    # Check if user has system admin permissions
    user_permissions = current_user.get("permissions", [])
    if "system:admin" not in user_permissions:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="System admin permission required")
    
    stats = circuit_breaker_manager.get_all_stats()
    return stats


@router.post("/circuit-breakers/{server_id}/reset")
async def reset_circuit_breaker(
    server_id: str,
    circuit_breaker_manager: CircuitBreakerManager = Depends(get_circuit_breaker_manager),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user)
):
    """Reset a circuit breaker"""
    if not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    
    # Check if user has system admin permissions
    user_permissions = current_user.get("permissions", [])
    if "system:admin" not in user_permissions:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="System admin permission required")
    
    circuit_breaker_manager.reset_circuit_breaker(server_id)
    return {"message": f"Circuit breaker for {server_id} reset successfully"}


# Audit management
@router.get("/audit/events")
async def get_audit_events(
    user_id: Optional[str] = None,
    resource_type: Optional[ResourceType] = None,
    action: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    audit_logger: AuditLogger = Depends(get_audit_logger),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user)
):
    """Get audit events"""
    if not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    
    # Check if user has system admin permissions
    user_permissions = current_user.get("permissions", [])
    if "system:admin" not in user_permissions:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="System admin permission required")
    
    tenant_id = current_user.get("tenant_id")
    events = audit_logger.get_audit_events(
        user_id=user_id,
        tenant_id=tenant_id,
        resource_type=resource_type,
        action=action,
        limit=limit,
        offset=offset
    )
    return events


@router.get("/audit/summary")
async def get_audit_summary(
    days: int = 30,
    audit_logger: AuditLogger = Depends(get_audit_logger),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user)
):
    """Get audit summary"""
    if not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    
    # Check if user has system admin permissions
    user_permissions = current_user.get("permissions", [])
    if "system:admin" not in user_permissions:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="System admin permission required")
    
    tenant_id = current_user.get("tenant_id")
    summary = audit_logger.get_audit_summary(tenant_id, days)
    return summary

