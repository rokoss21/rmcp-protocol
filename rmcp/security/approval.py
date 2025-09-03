"""
Approval Gateway for dangerous operations
"""

import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from enum import Enum
from pydantic import BaseModel, Field
from ..storage.database import DatabaseManager


class ApprovalStatus(str, Enum):
    """Approval status"""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"


class ApprovalRequest(BaseModel):
    """Approval request model"""
    id: str = Field(..., description="Unique approval request ID")
    user_id: str = Field(..., description="User who requested approval")
    tenant_id: str = Field(..., description="Tenant context")
    action: str = Field(..., description="Action requiring approval")
    resource_type: str = Field(..., description="Type of resource")
    resource_id: Optional[str] = Field(None, description="ID of resource")
    details: Dict[str, Any] = Field(default_factory=dict, description="Request details")
    justification: str = Field(..., description="User justification for the action")
    approver_id: Optional[str] = Field(None, description="User who approved/rejected")
    status: ApprovalStatus = Field(default=ApprovalStatus.PENDING, description="Approval status")
    expires_at: datetime = Field(..., description="Expiration time")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        use_enum_values = True


class ApprovalGateway:
    """Approval Gateway for dangerous operations"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self._init_approval_tables()
    
    def _init_approval_tables(self):
        """Initialize approval-related database tables"""
        from ..storage.schema import get_connection
        
        with get_connection(self.db_manager.db_path) as conn:
            cursor = conn.cursor()
            
            # Approval requests table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS approval_requests (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    tenant_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    resource_type TEXT NOT NULL,
                    resource_id TEXT,
                    details TEXT NOT NULL DEFAULT '{}',
                    justification TEXT NOT NULL,
                    approver_id TEXT,
                    status TEXT NOT NULL DEFAULT 'pending',
                    expires_at TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    FOREIGN KEY (tenant_id) REFERENCES tenants(id) ON DELETE CASCADE,
                    FOREIGN KEY (approver_id) REFERENCES users(id) ON DELETE SET NULL
                )
            """)
            
            conn.commit()
    
    def create_approval_request(self, 
                              user_id: str,
                              tenant_id: str,
                              action: str,
                              resource_type: str,
                              resource_id: Optional[str],
                              details: Dict[str, Any],
                              justification: str,
                              expires_in_hours: int = 24) -> ApprovalRequest:
        """Create a new approval request"""
        import json
        
        request_id = str(uuid.uuid4())
        expires_at = datetime.utcnow() + timedelta(hours=expires_in_hours)
        
        request = ApprovalRequest(
            id=request_id,
            user_id=user_id,
            tenant_id=tenant_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            details=details,
            justification=justification,
            expires_at=expires_at
        )
        
        from ..storage.schema import get_connection
        with get_connection(self.db_manager.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO approval_requests (
                    id, user_id, tenant_id, action, resource_type, resource_id,
                    details, justification, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                request.id,
                request.user_id,
                request.tenant_id,
                request.action,
                request.resource_type,
                request.resource_id,
                json.dumps(request.details),
                request.justification,
                request.expires_at.isoformat()
            ))
            conn.commit()
        
        return request
    
    def get_approval_request(self, request_id: str) -> Optional[ApprovalRequest]:
        """Get an approval request by ID"""
        from ..storage.schema import get_connection
        import json
        
        with get_connection(self.db_manager.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, user_id, tenant_id, action, resource_type, resource_id,
                       details, justification, approver_id, status, expires_at,
                       created_at, updated_at
                FROM approval_requests WHERE id = ?
            """, (request_id,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            try:
                details = json.loads(row[6]) if row[6] else {}
            except (json.JSONDecodeError, TypeError):
                details = {}
            
            return ApprovalRequest(
                id=row[0],
                user_id=row[1],
                tenant_id=row[2],
                action=row[3],
                resource_type=row[4],
                resource_id=row[5],
                details=details,
                justification=row[7],
                approver_id=row[8],
                status=ApprovalStatus(row[9]),
                expires_at=datetime.fromisoformat(row[10].replace('Z', '+00:00')),
                created_at=datetime.fromisoformat(row[11].replace('Z', '+00:00')),
                updated_at=datetime.fromisoformat(row[12].replace('Z', '+00:00'))
            )
    
    def approve_request(self, request_id: str, approver_id: str) -> bool:
        """Approve an approval request"""
        from ..storage.schema import get_connection
        
        with get_connection(self.db_manager.db_path) as conn:
            cursor = conn.cursor()
            
            # Check if request exists and is pending
            cursor.execute("""
                SELECT status, expires_at FROM approval_requests WHERE id = ?
            """, (request_id,))
            row = cursor.fetchone()
            
            if not row:
                return False
            
            status, expires_at = row
            if status != ApprovalStatus.PENDING.value:
                return False
            
            # Check if expired
            if datetime.fromisoformat(expires_at.replace('Z', '+00:00')) < datetime.utcnow():
                # Mark as expired
                cursor.execute("""
                    UPDATE approval_requests 
                    SET status = ?, approver_id = ?, updated_at = datetime('now')
                    WHERE id = ?
                """, (ApprovalStatus.EXPIRED, approver_id, request_id))
                conn.commit()
                return False
            
            # Approve the request
            cursor.execute("""
                UPDATE approval_requests 
                SET status = ?, approver_id = ?, updated_at = datetime('now')
                WHERE id = ?
            """, (ApprovalStatus.APPROVED, approver_id, request_id))
            conn.commit()
            
            return True
    
    def reject_request(self, request_id: str, approver_id: str) -> bool:
        """Reject an approval request"""
        from ..storage.schema import get_connection
        
        with get_connection(self.db_manager.db_path) as conn:
            cursor = conn.cursor()
            
            # Check if request exists and is pending
            cursor.execute("""
                SELECT status FROM approval_requests WHERE id = ?
            """, (request_id,))
            row = cursor.fetchone()
            
            if not row or row[0] != ApprovalStatus.PENDING.value:
                return False
            
            # Reject the request
            cursor.execute("""
                UPDATE approval_requests 
                SET status = ?, approver_id = ?, updated_at = datetime('now')
                WHERE id = ?
            """, (ApprovalStatus.REJECTED, approver_id, request_id))
            conn.commit()
            
            return True
    
    def get_pending_requests(self, tenant_id: str, limit: int = 50) -> List[ApprovalRequest]:
        """Get pending approval requests for a tenant"""
        from ..storage.schema import get_connection
        import json
        
        with get_connection(self.db_manager.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, user_id, tenant_id, action, resource_type, resource_id,
                       details, justification, approver_id, status, expires_at,
                       created_at, updated_at
                FROM approval_requests 
                WHERE tenant_id = ? AND status = 'pending' AND expires_at > datetime('now')
                ORDER BY created_at ASC
                LIMIT ?
            """, (tenant_id, limit))
            
            requests = []
            for row in cursor.fetchall():
                try:
                    details = json.loads(row[6]) if row[6] else {}
                except (json.JSONDecodeError, TypeError):
                    details = {}
                
                requests.append(ApprovalRequest(
                    id=row[0],
                    user_id=row[1],
                    tenant_id=row[2],
                    action=row[3],
                    resource_type=row[4],
                    resource_id=row[5],
                    details=details,
                    justification=row[7],
                    approver_id=row[8],
                    status=ApprovalStatus(row[9]),
                    expires_at=datetime.fromisoformat(row[10].replace('Z', '+00:00')),
                    created_at=datetime.fromisoformat(row[11].replace('Z', '+00:00')),
                    updated_at=datetime.fromisoformat(row[12].replace('Z', '+00:00'))
                ))
            
            return requests
    
    def cleanup_expired_requests(self) -> int:
        """Clean up expired approval requests"""
        from ..storage.schema import get_connection
        
        with get_connection(self.db_manager.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE approval_requests 
                SET status = 'expired', updated_at = datetime('now')
                WHERE status = 'pending' AND expires_at <= datetime('now')
            """)
            updated_count = cursor.rowcount
            conn.commit()
            
            return updated_count
    
    def requires_approval(self, action: str, resource_type: str, 
                         user_permissions: List[str]) -> bool:
        """Check if an action requires approval"""
        # Define actions that require approval
        dangerous_actions = {
            "tool": ["shell.execute", "terraform.apply", "kubectl.delete"],
            "tool_execution": ["shell.execute", "terraform.apply", "kubectl.delete"],
            "system": ["user.delete", "tenant.delete", "api_key.revoke"],
            "server": ["server.delete", "server.modify"]
        }
        
        # Check if user has admin permissions (admins don't need approval)
        if any(perm.endswith(":admin") for perm in user_permissions):
            return False
        
        # Check if action is in dangerous actions list
        if resource_type in dangerous_actions:
            dangerous_tools = dangerous_actions[resource_type]
            if action in dangerous_tools:
                return True
        
        return False
