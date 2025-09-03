"""
Approval workflow API routes for RMCP
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends, status, Request
from pydantic import BaseModel, Field

from .approval import ApprovalGateway
from .audit import AuditLogger
from .models import ResourceType, PermissionLevel
from .middleware import get_current_user
from ..core.approval_executor import ApprovalExecutor
from ..models.execution import ExecutionStatus


router = APIRouter(prefix="/approval", tags=["approval"])


# Request/Response models
class ApprovalActionRequest(BaseModel):
    """Request to approve/reject an approval request"""
    action: str = Field(..., description="Action: 'approve' or 'reject'")
    comment: Optional[str] = Field(None, description="Optional comment from approver")


class ExecutionApprovalRequest(BaseModel):
    """Request to approve/reject an execution"""
    action: str = Field(..., description="Action: 'approve' or 'reject'")
    comment: Optional[str] = Field(None, description="Optional comment from approver")


class ExecutionStatusResponse(BaseModel):
    """Response for execution status"""
    execution_request_id: str = Field(..., description="Execution request ID")
    status: str = Field(..., description="Current status")
    approval_request_id: Optional[str] = Field(None, description="Associated approval request ID")
    plan_summary: Optional[str] = Field(None, description="Plan summary")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    started_at: Optional[datetime] = Field(None, description="Start timestamp")
    completed_at: Optional[datetime] = Field(None, description="Completion timestamp")
    result: Optional[Dict[str, Any]] = Field(None, description="Execution result")
    error: Optional[str] = Field(None, description="Error message if failed")


# Dependency injection
def get_approval_gateway(request: Request) -> ApprovalGateway:
    """Get approval gateway from app state"""
    return request.app.state.approval_gateway


def get_audit_logger(request: Request) -> AuditLogger:
    """Get audit logger from app state"""
    return request.app.state.audit_logger


def get_approval_executor(request: Request) -> ApprovalExecutor:
    """Get approval executor from app state"""
    return request.app.state.approval_executor


# Approval management endpoints
@router.get("/requests", response_model=List[Dict[str, Any]])
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
    
    # Convert to dict format for response
    return [approval.dict() for approval in approvals]


@router.post("/requests/{approval_id}/action")
async def handle_approval(
    approval_id: str,
    request: ApprovalActionRequest,
    approval_gateway: ApprovalGateway = Depends(get_approval_gateway),
    audit_logger: AuditLogger = Depends(get_audit_logger),
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
    
    # Log approval action
    audit_logger.log_system_event(
        action=f"approval_{request.action}",
        user_id=approver_id,
        tenant_id=current_user.get("tenant_id"),
        details={
            "approval_request_id": approval_id,
            "comment": request.comment
        }
    )
    
    return {"message": f"Approval request {request.action}d successfully"}


# Execution management endpoints
@router.get("/executions/{execution_id}/status", response_model=ExecutionStatusResponse)
async def get_execution_status(
    execution_id: str,
    approval_executor: ApprovalExecutor = Depends(get_approval_executor),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user)
):
    """Get execution request status"""
    if not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    
    # Get execution request
    execution_request = await approval_executor._get_execution_request(execution_id)
    if not execution_request:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Execution request not found")
    
    # Check if user has permission to view this execution
    user_permissions = current_user.get("permissions", [])
    if (execution_request.user_id != current_user.get("user_id") and 
        "system:admin" not in user_permissions and
        execution_request.tenant_id != current_user.get("tenant_id")):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient permissions")
    
    return ExecutionStatusResponse(
        execution_request_id=execution_request.id,
        status=execution_request.status.value,
        approval_request_id=execution_request.approval_request_id,
        plan_summary="Plan summary",  # TODO: Get from stored plan data
        created_at=execution_request.created_at,
        updated_at=execution_request.updated_at,
        started_at=execution_request.started_at,
        completed_at=execution_request.completed_at,
        result=execution_request.result,
        error=execution_request.error
    )


@router.post("/executions/{execution_id}/approve")
async def approve_execution(
    execution_id: str,
    request: ExecutionApprovalRequest,
    approval_executor: ApprovalExecutor = Depends(get_approval_executor),
    audit_logger: AuditLogger = Depends(get_audit_logger),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user)
):
    """Approve an execution request"""
    if not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    
    # Check if user has admin permissions
    user_permissions = current_user.get("permissions", [])
    if not any(perm.endswith(":admin") for perm in user_permissions):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin permission required")
    
    if request.action == "approve":
        # Continue execution
        result = await approval_executor.continue_execution(execution_id)
        
        # Log approval
        audit_logger.log_system_event(
            action="execution_approved",
            user_id=current_user.get("user_id"),
            tenant_id=current_user.get("tenant_id"),
            details={
                "execution_request_id": execution_id,
                "comment": request.comment
            }
        )
        
        return {
            "message": "Execution approved and started",
            "execution_result": result.dict()
        }
    
    elif request.action == "reject":
        # TODO: Implement rejection logic
        audit_logger.log_system_event(
            action="execution_rejected",
            user_id=current_user.get("user_id"),
            tenant_id=current_user.get("tenant_id"),
            details={
                "execution_request_id": execution_id,
                "comment": request.comment
            }
        )
        
        return {"message": "Execution rejected"}
    
    else:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid action")


@router.get("/executions", response_model=List[ExecutionStatusResponse])
async def list_executions(
    status_filter: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    approval_executor: ApprovalExecutor = Depends(get_approval_executor),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user)
):
    """List execution requests"""
    if not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    
    # TODO: Implement execution listing with filtering
    # For now, return empty list
    return []


@router.get("/dashboard")
async def get_approval_dashboard(
    approval_gateway: ApprovalGateway = Depends(get_approval_gateway),
    audit_logger: AuditLogger = Depends(get_audit_logger),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user)
):
    """Get approval dashboard data"""
    if not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    
    # Check if user has admin permissions
    user_permissions = current_user.get("permissions", [])
    if not any(perm.endswith(":admin") for perm in user_permissions):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin permission required")
    
    tenant_id = current_user.get("tenant_id")
    
    # Get pending approvals
    pending_approvals = approval_gateway.get_pending_requests(tenant_id)
    
    # Get audit summary
    audit_summary = audit_logger.get_audit_summary(tenant_id, days=7)
    
    return {
        "pending_approvals_count": len(pending_approvals),
        "pending_approvals": [approval.dict() for approval in pending_approvals[:5]],  # Top 5
        "audit_summary": audit_summary,
        "dashboard_updated_at": datetime.utcnow().isoformat()
    }

