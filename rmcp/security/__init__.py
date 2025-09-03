"""
Security module for RMCP
Provides authentication, authorization, audit, approval, and circuit breaker capabilities
"""

from .auth import AuthManager, APIKeyAuth
from .rbac import RBACManager, Role, Permission
from .audit import AuditLogger
from .approval import ApprovalGateway, ApprovalRequest, ApprovalStatus
from .circuit_breaker import CircuitBreakerManager, CircuitBreaker, CircuitBreakerConfig, CircuitState
from .middleware import SecurityMiddleware, get_current_user, require_permission
from .models import User, Tenant, APIKey, AuditEvent, ResourceType, PermissionLevel

__all__ = [
    "AuthManager",
    "APIKeyAuth", 
    "RBACManager",
    "Role",
    "Permission",
    "AuditLogger",
    "ApprovalGateway",
    "ApprovalRequest",
    "ApprovalStatus",
    "CircuitBreakerManager",
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitState",
    "SecurityMiddleware",
    "get_current_user",
    "require_permission",
    "User",
    "Tenant",
    "APIKey",
    "AuditEvent",
    "ResourceType",
    "PermissionLevel"
]
