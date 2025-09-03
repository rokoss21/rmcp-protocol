"""
Security middleware for FastAPI
"""

from typing import Optional, Dict, Any
from fastapi import Request, HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
import time
from .auth import AuthManager
from .rbac import RBACManager
from .audit import AuditLogger
from .approval import ApprovalGateway
from .circuit_breaker import CircuitBreakerManager
from .models import ResourceType, PermissionLevel


class SecurityMiddleware:
    """Security middleware for RMCP"""
    
    def __init__(self, auth_manager: AuthManager, rbac_manager: RBACManager, 
                 audit_logger: AuditLogger, approval_gateway: ApprovalGateway,
                 circuit_breaker_manager: CircuitBreakerManager):
        self.auth_manager = auth_manager
        self.rbac_manager = rbac_manager
        self.audit_logger = audit_logger
        self.approval_gateway = approval_gateway
        self.circuit_breaker_manager = circuit_breaker_manager
        self.security = HTTPBearer(auto_error=False)
    
    async def authenticate_request(self, request: Request) -> Optional[Dict[str, Any]]:
        """Authenticate incoming request"""
        # Skip authentication for health checks and docs
        if request.url.path in ["/health", "/docs", "/openapi.json", "/redoc"]:
            return None
        
        # Get API key from header
        api_key = request.headers.get("X-API-Key")
        if not api_key:
            # Try Bearer token
            authorization = request.headers.get("Authorization")
            if authorization and authorization.startswith("Bearer "):
                token = authorization[7:]
                try:
                    payload = self.auth_manager.auth.verify_jwt_token(token)
                    return {
                        "user_id": payload["user_id"],
                        "tenant_id": payload["tenant_id"],
                        "permissions": payload["permissions"]
                    }
                except HTTPException:
                    pass
            
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API key or Bearer token required"
            )
        
        # Authenticate API key
        user_info = self.auth_manager.authenticate_api_key(api_key)
        if not user_info:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key"
            )
        
        # Log authentication
        self.audit_logger.log_authentication(
            user_id=user_info["user_id"],
            tenant_id=user_info["tenant_id"],
            success=True,
            request=request
        )
        
        return user_info
    
    async def authorize_request(self, user_info: Dict[str, Any], 
                              resource_type: ResourceType, 
                              action: PermissionLevel,
                              resource_id: Optional[str] = None) -> bool:
        """Authorize request based on user permissions"""
        user_permissions = user_info["permissions"]
        
        # Check permission
        has_permission = self.rbac_manager.check_resource_permission(
            user_permissions, resource_type, action
        )
        
        # Log authorization attempt
        self.audit_logger.log_authorization(
            user_id=user_info["user_id"],
            tenant_id=user_info["tenant_id"],
            action=action.value,
            resource_type=resource_type,
            resource_id=resource_id,
            success=has_permission
        )
        
        if not has_permission:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions for {resource_type.value}:{action.value}"
            )
        
        return True
    
    async def check_approval_required(self, user_info: Dict[str, Any],
                                    action: str, resource_type: str,
                                    resource_id: Optional[str] = None) -> bool:
        """Check if action requires approval"""
        user_permissions = user_info["permissions"]
        
        if self.approval_gateway.requires_approval(action, resource_type, user_permissions):
            # Create approval request
            approval_request = self.approval_gateway.create_approval_request(
                user_id=user_info["user_id"],
                tenant_id=user_info["tenant_id"],
                action=action,
                resource_type=resource_type,
                resource_id=resource_id,
                details={"requires_approval": True},
                justification="Dangerous operation requiring approval"
            )
            
            # Log approval request
            self.audit_logger.log_system_event(
                action="approval_request_created",
                user_id=user_info["user_id"],
                tenant_id=user_info["tenant_id"],
                details={"approval_request_id": approval_request.id}
            )
            
            raise HTTPException(
                status_code=status.HTTP_202_ACCEPTED,
                detail={
                    "message": "Action requires approval",
                    "approval_request_id": approval_request.id,
                    "status": "pending"
                }
            )
        
        return True
    
    async def check_circuit_breaker(self, server_id: str) -> bool:
        """Check circuit breaker for server"""
        if not self.circuit_breaker_manager.can_execute(server_id):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Server {server_id} is currently unavailable (circuit breaker open)"
            )
        
        return True
    
    async def process_request(self, request: Request, call_next):
        """Process request through security middleware"""
        start_time = time.time()
        
        try:
            # Authenticate request
            user_info = await self.authenticate_request(request)
            
            # Add user info to request state
            request.state.user_info = user_info
            
            # Process request
            response = await call_next(request)
            
            # Record success metrics
            if hasattr(request.state, 'server_id'):
                self.circuit_breaker_manager.record_success(request.state.server_id)
            
            # Log successful request
            if user_info:
                self.audit_logger.log_system_event(
                    action="request_processed",
                    user_id=user_info["user_id"],
                    tenant_id=user_info["tenant_id"],
                    success=True,
                    request=request,
                    details={
                        "path": request.url.path,
                        "method": request.method,
                        "duration_ms": (time.time() - start_time) * 1000
                    }
                )
            
            return response
            
        except HTTPException as e:
            # Record failure metrics
            if hasattr(request.state, 'server_id'):
                self.circuit_breaker_manager.record_failure(request.state.server_id)
            
            # Log failed request
            if hasattr(request.state, 'user_info') and request.state.user_info:
                user_info = request.state.user_info
                self.audit_logger.log_system_event(
                    action="request_failed",
                    user_id=user_info["user_id"],
                    tenant_id=user_info["tenant_id"],
                    success=False,
                    request=request,
                    details={
                        "path": request.url.path,
                        "method": request.method,
                        "error": str(e.detail),
                        "status_code": e.status_code
                    }
                )
            
            return JSONResponse(
                status_code=e.status_code,
                content={"detail": e.detail}
            )
        
        except Exception as e:
            # Record failure metrics
            if hasattr(request.state, 'server_id'):
                self.circuit_breaker_manager.record_failure(request.state.server_id)
            
            # Log system error
            if hasattr(request.state, 'user_info') and request.state.user_info:
                user_info = request.state.user_info
                self.audit_logger.log_system_event(
                    action="system_error",
                    user_id=user_info["user_id"],
                    tenant_id=user_info["tenant_id"],
                    success=False,
                    request=request,
                    details={
                        "path": request.url.path,
                        "method": request.method,
                        "error": str(e)
                    }
                )
            
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"detail": "Internal server error"}
            )


def get_current_user(request: Request) -> Optional[Dict[str, Any]]:
    """Get current authenticated user from request"""
    return getattr(request.state, 'user_info', None)


def require_permission(resource_type: ResourceType, action: PermissionLevel):
    """Decorator to require specific permission"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # This would be used in route handlers
            # Implementation depends on how you want to integrate with FastAPI
            pass
        return wrapper
    return decorator

