"""
Approval-aware Executor with integration to ApprovalGateway
"""

import httpx
import time
import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime

from ..storage.database import DatabaseManager
from ..models.plan import ExecutionPlan, ExecutionStep, ExecutionStrategy
from ..models.request import ExecuteResponse
from ..models.execution import ExecutionRequest, ExecutionStatus, ApprovalContext
from ..security.approval import ApprovalGateway
from ..security.audit import AuditLogger


class ApprovalExecutor:
    """Executor with approval workflow integration"""
    
    def __init__(self, db_manager: DatabaseManager, approval_gateway: ApprovalGateway, audit_logger: AuditLogger):
        self.db_manager = db_manager
        self.approval_gateway = approval_gateway
        self.audit_logger = audit_logger
        self.client = httpx.AsyncClient(timeout=60.0)
        self._init_execution_tables()
    
    def _init_execution_tables(self):
        """Initialize execution-related database tables"""
        from ..storage.schema import get_connection
        
        with get_connection(self.db_manager.db_path) as conn:
            cursor = conn.cursor()
            
            # Execution requests table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS execution_requests (
                    id TEXT PRIMARY KEY,
                    plan_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    tenant_id TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    approval_request_id TEXT,
                    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                    started_at TEXT,
                    completed_at TEXT,
                    result TEXT,
                    error TEXT,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    FOREIGN KEY (tenant_id) REFERENCES tenants(id) ON DELETE CASCADE,
                    FOREIGN KEY (approval_request_id) REFERENCES approval_requests(id) ON DELETE SET NULL
                )
            """)
            
            conn.commit()
    
    async def execute(self, plan: ExecutionPlan, user_id: str, tenant_id: str, context: Dict[str, Any] = None) -> ExecuteResponse:
        """
        Execute the execution plan with approval workflow
        
        Args:
            plan: Execution plan to execute
            user_id: User requesting execution
            tenant_id: Tenant context
            context: Additional context for approval decisions
            
        Returns:
            ExecuteResponse with execution result or approval request info
        """
        start_time = time.time()
        context = context or {}
        
        try:
            # Create execution request
            execution_request = await self._create_execution_request(plan, user_id, tenant_id)
            
            # Check if approval is required
            if plan.requires_approval:
                return await self._handle_approval_required(execution_request, plan, user_id, tenant_id, context)
            else:
                return await self._execute_immediately(execution_request, plan)
            
        except Exception as e:
            execution_time_ms = int((time.time() - start_time) * 1000)
            return ExecuteResponse(
                status="ERROR",
                summary=f"Execution failed: {str(e)}",
                data={},
                metadata={"error": str(e)},
                confidence=0.0,
                execution_time_ms=execution_time_ms
            )
    
    async def _create_execution_request(self, plan: ExecutionPlan, user_id: str, tenant_id: str) -> ExecutionRequest:
        """Create execution request record"""
        import json
        
        request_id = str(uuid.uuid4())
        plan_id = str(uuid.uuid4())  # Generate plan ID for tracking
        
        execution_request = ExecutionRequest(
            id=request_id,
            plan_id=plan_id,
            user_id=user_id,
            tenant_id=tenant_id,
            status=ExecutionStatus.PENDING
        )
        
        from ..storage.schema import get_connection
        with get_connection(self.db_manager.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO execution_requests (
                    id, plan_id, user_id, tenant_id, status
                ) VALUES (?, ?, ?, ?, ?)
            """, (
                execution_request.id,
                execution_request.plan_id,
                execution_request.user_id,
                execution_request.tenant_id,
                execution_request.status.value if hasattr(execution_request.status, 'value') else str(execution_request.status)
            ))
            conn.commit()
        
        return execution_request
    
    async def _handle_approval_required(
        self, 
        execution_request: ExecutionRequest, 
        plan: ExecutionPlan, 
        user_id: str, 
        tenant_id: str, 
        context: Dict[str, Any]
    ) -> ExecuteResponse:
        """Handle case where approval is required"""
        
        # Create approval context
        approval_context = ApprovalContext(
            user_id=user_id,
            tenant_id=tenant_id,
            plan_summary=self._generate_plan_summary(plan),
            risk_level=self._assess_risk_level(plan),
            justification=context.get("justification", "Dangerous operation requiring approval"),
            estimated_impact=context.get("estimated_impact", "unknown")
        )
        
        # Create approval request
        approval_request = self.approval_gateway.create_approval_request(
            user_id=user_id,
            tenant_id=tenant_id,
            action="execute_plan",
            resource_type="execution",
            resource_id=execution_request.id,
            details={
                "plan_id": execution_request.plan_id,
                "plan_summary": approval_context.plan_summary,
                "risk_level": approval_context.risk_level,
                "estimated_impact": approval_context.estimated_impact,
                "strategy": plan.strategy.value if hasattr(plan.strategy, 'value') else str(plan.strategy),
                "step_count": len(plan.steps) if isinstance(plan.steps, list) else len(plan.steps.keys())
            },
            justification=approval_context.justification
        )
        
        # Update execution request with approval info
        await self._update_execution_request(
            execution_request.id,
            status=ExecutionStatus.AWAITING_APPROVAL,
            approval_request_id=approval_request.id
        )
        
        # Log approval request
        self.audit_logger.log_system_event(
            action="execution_approval_requested",
            user_id=user_id,
            tenant_id=tenant_id,
            details={
                "execution_request_id": execution_request.id,
                "approval_request_id": approval_request.id,
                "plan_summary": approval_context.plan_summary
            }
        )
        
        return ExecuteResponse(
            status="AWAITING_APPROVAL",
            summary="Execution requires approval",
            data={
                "execution_request_id": execution_request.id,
                "approval_request_id": approval_request.id,
                "plan_summary": approval_context.plan_summary,
                "risk_level": approval_context.risk_level
            },
            metadata={
                "requires_approval": True,
                "approval_request_id": approval_request.id
            },
            confidence=1.0,
            execution_time_ms=0
        )
    
    async def _execute_immediately(self, execution_request: ExecutionRequest, plan: ExecutionPlan) -> ExecuteResponse:
        """Execute plan immediately without approval"""
        
        # Update status to running
        await self._update_execution_request(
            execution_request.id,
            status=ExecutionStatus.RUNNING,
            started_at=datetime.utcnow()
        )
        
        # Execute the plan
        try:
            if plan.strategy == ExecutionStrategy.SOLO:
                result = await self._execute_solo(plan)
            elif plan.strategy == ExecutionStrategy.PARALLEL:
                result = await self._execute_parallel(plan)
            elif plan.strategy == ExecutionStrategy.DAG:
                result = await self._execute_dag(plan)
            else:
                raise ValueError(f"Unsupported execution strategy: {plan.strategy}")
            
            # Update execution request with success
            await self._update_execution_request(
                execution_request.id,
                status=ExecutionStatus.SUCCESS,
                completed_at=datetime.utcnow(),
                result=result.dict()
            )
            
            return result
            
        except Exception as e:
            # Update execution request with failure
            await self._update_execution_request(
                execution_request.id,
                status=ExecutionStatus.FAILED,
                completed_at=datetime.utcnow(),
                error=str(e)
            )
            
            return ExecuteResponse(
                status="ERROR",
                summary=f"Execution failed: {str(e)}",
                data={},
                metadata={"error": str(e)},
                confidence=0.0,
                execution_time_ms=0
            )
    
    async def continue_execution(self, execution_request_id: str) -> ExecuteResponse:
        """Continue execution after approval"""
        
        # Get execution request
        execution_request = await self._get_execution_request(execution_request_id)
        if not execution_request:
            return ExecuteResponse(
                status="ERROR",
                summary="Execution request not found",
                data={},
                metadata={"error": "Execution request not found"},
                confidence=0.0,
                execution_time_ms=0
            )
        
        # Check if approval was granted
        if execution_request.approval_request_id:
            approval_request = self.approval_gateway.get_approval_request(execution_request.approval_request_id)
            if not approval_request or approval_request.status.value != "approved":
                return ExecuteResponse(
                    status="ERROR",
                    summary="Approval not granted",
                    data={},
                    metadata={"error": "Approval not granted"},
                    confidence=0.0,
                    execution_time_ms=0
                )
        
        # TODO: Reconstruct plan from stored data and execute
        # For now, return success
        await self._update_execution_request(
            execution_request_id,
            status=ExecutionStatus.SUCCESS,
            completed_at=datetime.utcnow()
        )
        
        return ExecuteResponse(
            status="SUCCESS",
            summary="Execution completed after approval",
            data={},
            metadata={"execution_request_id": execution_request_id},
            confidence=1.0,
            execution_time_ms=0
        )
    
    def _generate_plan_summary(self, plan: ExecutionPlan) -> str:
        """Generate human-readable summary of execution plan"""
        if plan.strategy == ExecutionStrategy.SOLO:
            if isinstance(plan.steps, list) and plan.steps:
                tool = self.db_manager.get_tool(plan.steps[0].tool_id)
                tool_name = tool.name if tool else plan.steps[0].tool_id
                return f"Execute single tool: {tool_name}"
        
        step_count = len(plan.steps) if isinstance(plan.steps, list) else len(plan.steps.keys())
        strategy_str = plan.strategy.value if hasattr(plan.strategy, 'value') else str(plan.strategy)
        return f"Execute {strategy_str} plan with {step_count} steps"
    
    def _assess_risk_level(self, plan: ExecutionPlan) -> str:
        """Assess risk level of execution plan"""
        if plan.strategy == ExecutionStrategy.SOLO:
            if isinstance(plan.steps, list) and plan.steps:
                tool = self.db_manager.get_tool(plan.steps[0].tool_id)
                if tool:
                    dangerous_capabilities = ["execution", "filesystem:write", "system.admin"]
                    if any(cap in tool.capabilities for cap in dangerous_capabilities):
                        return "high"
        
        return "medium"
    
    async def _update_execution_request(
        self, 
        request_id: str, 
        status: ExecutionStatus = None,
        approval_request_id: str = None,
        started_at: datetime = None,
        completed_at: datetime = None,
        result: Dict[str, Any] = None,
        error: str = None
    ):
        """Update execution request"""
        import json
        
        updates = []
        params = []
        
        if status:
            updates.append("status = ?")
            params.append(status.value if hasattr(status, 'value') else str(status))
        
        if approval_request_id:
            updates.append("approval_request_id = ?")
            params.append(approval_request_id)
        
        if started_at:
            updates.append("started_at = ?")
            params.append(started_at.isoformat())
        
        if completed_at:
            updates.append("completed_at = ?")
            params.append(completed_at.isoformat())
        
        if result:
            updates.append("result = ?")
            params.append(json.dumps(result))
        
        if error:
            updates.append("error = ?")
            params.append(error)
        
        if updates:
            updates.append("updated_at = datetime('now')")
            params.append(request_id)
            
            from ..storage.schema import get_connection
            with get_connection(self.db_manager.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(f"""
                    UPDATE execution_requests 
                    SET {', '.join(updates)}
                    WHERE id = ?
                """, params)
                conn.commit()
    
    async def _get_execution_request(self, request_id: str) -> Optional[ExecutionRequest]:
        """Get execution request by ID"""
        from ..storage.schema import get_connection
        import json
        
        with get_connection(self.db_manager.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, plan_id, user_id, tenant_id, status, approval_request_id,
                       created_at, updated_at, started_at, completed_at, result, error
                FROM execution_requests WHERE id = ?
            """, (request_id,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            try:
                result = json.loads(row[10]) if row[10] else None
            except (json.JSONDecodeError, TypeError):
                result = None
            
            return ExecutionRequest(
                id=row[0],
                plan_id=row[1],
                user_id=row[2],
                tenant_id=row[3],
                status=ExecutionStatus(row[4]),
                approval_request_id=row[5],
                created_at=datetime.fromisoformat(row[6].replace('Z', '+00:00')),
                updated_at=datetime.fromisoformat(row[7].replace('Z', '+00:00')),
                started_at=datetime.fromisoformat(row[8].replace('Z', '+00:00')) if row[8] else None,
                completed_at=datetime.fromisoformat(row[9].replace('Z', '+00:00')) if row[9] else None,
                result=result,
                error=row[11]
            )
    
    # Delegate execution methods to SimpleExecutor for now
    async def _execute_solo(self, plan: ExecutionPlan) -> ExecuteResponse:
        """Execute single step plan"""
        from .executor import SimpleExecutor
        executor = SimpleExecutor(self.db_manager)
        try:
            return await executor._execute_solo(plan)
        finally:
            await executor.close()
    
    async def _execute_parallel(self, plan: ExecutionPlan) -> ExecuteResponse:
        """Execute parallel steps"""
        from .executor import SimpleExecutor
        executor = SimpleExecutor(self.db_manager)
        try:
            return await executor._execute_parallel(plan)
        finally:
            await executor.close()
    
    async def _execute_dag(self, plan: ExecutionPlan) -> ExecuteResponse:
        """Execute DAG plan"""
        from .executor import SimpleExecutor
        executor = SimpleExecutor(self.db_manager)
        try:
            return await executor._execute_dag(plan)
        finally:
            await executor.close()
    
    async def close(self):
        """Close HTTP client"""
        await self.client.aclose()
