"""
Models for execution states and approval workflow
"""

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime


class ExecutionStatus(str, Enum):
    """Execution status"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    AWAITING_APPROVAL = "awaiting_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


class ExecutionRequest(BaseModel):
    """Execution request model"""
    id: str = Field(..., description="Unique execution request ID")
    plan_id: str = Field(..., description="Associated execution plan ID")
    user_id: str = Field(..., description="User who requested execution")
    tenant_id: str = Field(..., description="Tenant context")
    status: ExecutionStatus = Field(default=ExecutionStatus.PENDING, description="Execution status")
    approval_request_id: Optional[str] = Field(None, description="Associated approval request ID")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = Field(None, description="When execution started")
    completed_at: Optional[datetime] = Field(None, description="When execution completed")
    result: Optional[Dict[str, Any]] = Field(None, description="Execution result")
    error: Optional[str] = Field(None, description="Error message if failed")
    
    class Config:
        use_enum_values = True


class ApprovalContext(BaseModel):
    """Context for approval decisions"""
    user_id: str = Field(..., description="User requesting approval")
    tenant_id: str = Field(..., description="Tenant context")
    plan_summary: str = Field(..., description="Summary of execution plan")
    risk_level: str = Field(default="medium", description="Risk level (low, medium, high)")
    justification: str = Field(..., description="User justification for the operation")
    estimated_impact: str = Field(default="unknown", description="Estimated impact of the operation")
    
    class Config:
        use_enum_values = True

