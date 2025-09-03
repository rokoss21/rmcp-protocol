"""
Audit logging module for RMCP
"""

import json
import uuid
from typing import Dict, Any, Optional
from datetime import datetime
from fastapi import Request
from ..storage.database import DatabaseManager
from .models import AuditEvent, ResourceType


class AuditLogger:
    """Audit logger for security events"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
    
    def log_event(self, 
                  action: str,
                  resource_type: ResourceType,
                  user_id: Optional[str] = None,
                  tenant_id: Optional[str] = None,
                  resource_id: Optional[str] = None,
                  details: Optional[Dict[str, Any]] = None,
                  request: Optional[Request] = None,
                  success: bool = True) -> str:
        """Log an audit event"""
        
        event_id = str(uuid.uuid4())
        
        # Extract request information if available
        ip_address = None
        user_agent = None
        if request:
            ip_address = request.client.host if request.client else None
            user_agent = request.headers.get("user-agent")
        
        event = AuditEvent(
            id=event_id,
            user_id=user_id,
            tenant_id=tenant_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            details=details or {},
            ip_address=ip_address,
            user_agent=user_agent,
            success=success
        )
        
        from ..storage.schema import get_connection
        with get_connection(self.db_manager.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO audit_events (
                    id, user_id, tenant_id, action, resource_type, resource_id,
                    details, ip_address, user_agent, timestamp, success
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                event.id,
                event.user_id,
                event.tenant_id,
                event.action,
                event.resource_type.value if hasattr(event.resource_type, 'value') else str(event.resource_type),
                event.resource_id,
                json.dumps(event.details),
                event.ip_address,
                event.user_agent,
                event.timestamp.isoformat(),
                event.success
            ))
            conn.commit()
        
        return event_id
    
    def log_authentication(self, user_id: str, tenant_id: str, success: bool, 
                          request: Optional[Request] = None, details: Optional[Dict[str, Any]] = None):
        """Log authentication event"""
        return self.log_event(
            action="authentication",
            resource_type=ResourceType.SYSTEM,
            user_id=user_id,
            tenant_id=tenant_id,
            details=details,
            request=request,
            success=success
        )
    
    def log_authorization(self, user_id: str, tenant_id: str, action: str, 
                         resource_type: ResourceType, resource_id: Optional[str] = None,
                         success: bool = True, request: Optional[Request] = None,
                         details: Optional[Dict[str, Any]] = None):
        """Log authorization event"""
        return self.log_event(
            action=f"authorization:{action}",
            resource_type=resource_type,
            user_id=user_id,
            tenant_id=tenant_id,
            resource_id=resource_id,
            details=details,
            request=request,
            success=success
        )
    
    def log_tool_execution(self, user_id: str, tenant_id: str, tool_id: str,
                          success: bool, request: Optional[Request] = None,
                          details: Optional[Dict[str, Any]] = None):
        """Log tool execution event"""
        return self.log_event(
            action="tool_execution",
            resource_type=ResourceType.TOOL,
            user_id=user_id,
            tenant_id=tenant_id,
            resource_id=tool_id,
            details=details,
            request=request,
            success=success
        )
    
    def log_system_event(self, action: str, user_id: Optional[str] = None,
                        tenant_id: Optional[str] = None, success: bool = True,
                        request: Optional[Request] = None, details: Optional[Dict[str, Any]] = None):
        """Log system event"""
        return self.log_event(
            action=action,
            resource_type=ResourceType.SYSTEM,
            user_id=user_id,
            tenant_id=tenant_id,
            details=details,
            request=request,
            success=success
        )
    
    def get_audit_events(self, 
                        user_id: Optional[str] = None,
                        tenant_id: Optional[str] = None,
                        resource_type: Optional[ResourceType] = None,
                        action: Optional[str] = None,
                        start_date: Optional[datetime] = None,
                        end_date: Optional[datetime] = None,
                        limit: int = 100,
                        offset: int = 0) -> list:
        """Get audit events with filtering"""
        from ..storage.schema import get_connection
        
        with get_connection(self.db_manager.db_path) as conn:
            cursor = conn.cursor()
            
            # Build query with filters
            query = "SELECT * FROM audit_events WHERE 1=1"
            params = []
            
            if user_id:
                query += " AND user_id = ?"
                params.append(user_id)
            
            if tenant_id:
                query += " AND tenant_id = ?"
                params.append(tenant_id)
            
            if resource_type:
                query += " AND resource_type = ?"
                params.append(resource_type.value)
            
            if action:
                query += " AND action LIKE ?"
                params.append(f"%{action}%")
            
            if start_date:
                query += " AND timestamp >= ?"
                params.append(start_date.isoformat())
            
            if end_date:
                query += " AND timestamp <= ?"
                params.append(end_date.isoformat())
            
            query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])
            
            cursor.execute(query, params)
            
            events = []
            for row in cursor.fetchall():
                events.append({
                    "id": row[0],
                    "user_id": row[1],
                    "tenant_id": row[2],
                    "action": row[3],
                    "resource_type": row[4],
                    "resource_id": row[5],
                    "details": json.loads(row[6]) if row[6] else {},
                    "ip_address": row[7],
                    "user_agent": row[8],
                    "timestamp": row[9],
                    "success": bool(row[10])
                })
            
            return events
    
    def get_audit_summary(self, tenant_id: str, days: int = 30) -> Dict[str, Any]:
        """Get audit summary for a tenant"""
        from ..storage.schema import get_connection
        
        with get_connection(self.db_manager.db_path) as conn:
            cursor = conn.cursor()
            
            # Get total events
            cursor.execute("""
                SELECT COUNT(*) FROM audit_events 
                WHERE tenant_id = ? AND timestamp >= datetime('now', '-{} days')
            """.format(days), (tenant_id,))
            total_events = cursor.fetchone()[0]
            
            # Get events by type
            cursor.execute("""
                SELECT resource_type, COUNT(*) FROM audit_events 
                WHERE tenant_id = ? AND timestamp >= datetime('now', '-{} days')
                GROUP BY resource_type
            """.format(days), (tenant_id,))
            events_by_type = dict(cursor.fetchall())
            
            # Get events by action
            cursor.execute("""
                SELECT action, COUNT(*) FROM audit_events 
                WHERE tenant_id = ? AND timestamp >= datetime('now', '-{} days')
                GROUP BY action
                ORDER BY COUNT(*) DESC
                LIMIT 10
            """.format(days), (tenant_id,))
            top_actions = dict(cursor.fetchall())
            
            # Get success rate
            cursor.execute("""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful
                FROM audit_events 
                WHERE tenant_id = ? AND timestamp >= datetime('now', '-{} days')
            """.format(days), (tenant_id,))
            success_row = cursor.fetchone()
            success_rate = (success_row[1] / success_row[0]) * 100 if success_row[0] > 0 else 0
            
            return {
                "total_events": total_events,
                "events_by_type": events_by_type,
                "top_actions": top_actions,
                "success_rate": round(success_rate, 2),
                "period_days": days
            }
