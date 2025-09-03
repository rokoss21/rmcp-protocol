"""
Circuit Breaker-aware Executor with fault tolerance
"""

import httpx
import time
from typing import Dict, List, Any, Optional
from datetime import datetime

from ..storage.database import DatabaseManager
from ..models.plan import ExecutionPlan, ExecutionStep, ExecutionStrategy
from ..models.request import ExecuteResponse
from ..models.execution import ExecutionRequest, ExecutionStatus
from ..security.circuit_breaker import CircuitBreakerManager, CircuitBreakerConfig
from ..security.audit import AuditLogger
from ..observability.metrics import metrics
from ..logging.config import get_logger, log_tool_execution, log_circuit_breaker


class CircuitBreakerExecutor:
    """Executor with circuit breaker integration for fault tolerance"""
    
    def __init__(self, db_manager: DatabaseManager, circuit_breaker_manager: CircuitBreakerManager, audit_logger: AuditLogger):
        self.db_manager = db_manager
        self.circuit_breaker_manager = circuit_breaker_manager
        self.audit_logger = audit_logger
        self.client = httpx.AsyncClient(timeout=60.0)
        self.logger = get_logger(__name__)
        self._init_execution_tables()
    
    def _init_execution_tables(self):
        """Initialize execution-related database tables"""
        from ..storage.schema import get_connection
        
        with get_connection(self.db_manager.db_path) as conn:
            cursor = conn.cursor()
            
            # Execution requests table (if not exists)
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
        Execute the execution plan with circuit breaker protection
        
        Args:
            plan: Execution plan to execute
            user_id: User requesting execution
            tenant_id: Tenant context
            context: Additional context
            
        Returns:
            ExecuteResponse with execution result
        """
        start_time = time.time()
        context = context or {}
        
        try:
            # Execute the plan with circuit breaker protection
            if plan.strategy == ExecutionStrategy.SOLO:
                result = await self._execute_solo_with_circuit_breaker(plan, user_id, tenant_id, context)
            elif plan.strategy == ExecutionStrategy.PARALLEL:
                result = await self._execute_parallel_with_circuit_breaker(plan, user_id, tenant_id, context)
            elif plan.strategy == ExecutionStrategy.DAG:
                result = await self._execute_dag_with_circuit_breaker(plan, user_id, tenant_id, context)
            else:
                raise ValueError(f"Unsupported execution strategy: {plan.strategy}")
            
            execution_time_ms = int((time.time() - start_time) * 1000)
            result.execution_time_ms = execution_time_ms
            
            # Record execution metrics
            strategy_str = plan.strategy.value if hasattr(plan.strategy, 'value') else str(plan.strategy)
            metrics.record_execution_plan(strategy_str, result.status, tenant_id)
            
            return result
            
        except Exception as e:
            execution_time_ms = int((time.time() - start_time) * 1000)
            
            # Record execution metrics for error case
            strategy_str = plan.strategy.value if hasattr(plan.strategy, 'value') else str(plan.strategy)
            metrics.record_execution_plan(strategy_str, "ERROR", tenant_id)
            
            return ExecuteResponse(
                status="ERROR",
                summary=f"Execution failed: {str(e)}",
                data={},
                metadata={"error": str(e)},
                confidence=0.0,
                execution_time_ms=execution_time_ms
            )
    
    async def _execute_solo_with_circuit_breaker(self, plan: ExecutionPlan, user_id: str, tenant_id: str, context: Dict[str, Any]) -> ExecuteResponse:
        """Execute single step plan with circuit breaker protection"""
        if not plan.steps or len(plan.steps) == 0:
            return ExecuteResponse(
                status="SUCCESS",
                summary="No steps to execute",
                data={},
                metadata={},
                confidence=1.0
            )
        
        step = plan.steps[0]
        return await self._execute_step_with_circuit_breaker(step, user_id, tenant_id, context)
    
    async def _execute_parallel_with_circuit_breaker(self, plan: ExecutionPlan, user_id: str, tenant_id: str, context: Dict[str, Any]) -> ExecuteResponse:
        """Execute parallel steps with circuit breaker protection"""
        results = []
        
        for step in plan.steps:
            try:
                result = await self._execute_step_with_circuit_breaker(step, user_id, tenant_id, context)
                results.append(result)
            except Exception as e:
                results.append(ExecuteResponse(
                    status="ERROR",
                    summary=f"Step failed: {str(e)}",
                    data={},
                    metadata={"error": str(e)},
                    confidence=0.0
                ))
        
        # Simple merge policy: first successful result
        for result in results:
            if result.status == "SUCCESS":
                return result
        
        # If no successful results, return the first error
        return results[0] if results else ExecuteResponse(
            status="ERROR",
            summary="No steps executed",
            data={},
            metadata={},
            confidence=0.0
        )
    
    async def _execute_dag_with_circuit_breaker(self, plan: ExecutionPlan, user_id: str, tenant_id: str, context: Dict[str, Any]) -> ExecuteResponse:
        """Execute DAG plan with circuit breaker protection and full fault tolerance"""
        import asyncio
        from collections import defaultdict, deque
        
        if not plan.steps:
            return ExecuteResponse(
                status="SUCCESS",
                summary="No steps to execute",
                data={},
                metadata={},
                confidence=1.0
            )
        
        # Convert steps to dict if it's a list (for backward compatibility)
        if isinstance(plan.steps, list):
            steps_dict = {f"step_{i}": step for i, step in enumerate(plan.steps)}
        else:
            steps_dict = plan.steps
        
        # Build dependency graph and topological order with circuit breaker awareness
        try:
            execution_order = self._resolve_dependencies_with_circuit_breaker(steps_dict)
        except Exception as e:
            return ExecuteResponse(
                status="ERROR",
                summary=f"Dependency resolution failed: {str(e)}",
                data={},
                metadata={"error": str(e)},
                confidence=0.0
            )
        
        # Execute steps with circuit breaker protection
        execution_results = {}
        step_outputs = {}
        circuit_breaker_stats = {}
        
        start_time = time.time()
        
        try:
            for level_idx, level in enumerate(execution_order):
                # Execute all steps in this level in parallel with circuit breaker protection
                if len(level) == 1:
                    # Single step
                    step_id = level[0]
                    step = steps_dict[step_id]
                    
                    # Inject dependency results
                    enhanced_args = self._inject_dependency_data_cb(step, step_outputs)
                    step_with_args = ExecutionStep(
                        tool_id=step.tool_id,
                        args=enhanced_args,
                        dependencies=step.dependencies,
                        outputs=step.outputs,
                        timeout_ms=step.timeout_ms,
                        retry_count=step.retry_count
                    )
                    
                    result = await self._execute_step_with_circuit_breaker(step_with_args, user_id, tenant_id, context)
                    execution_results[step_id] = result
                    circuit_breaker_stats[step_id] = self._get_circuit_breaker_stats(step.tool_id)
                    
                    # Store step outputs
                    if step.outputs and result.status == "SUCCESS":
                        for output_name in step.outputs:
                            step_outputs[output_name] = result.data
                
                else:
                    # Multiple steps in parallel
                    tasks = []
                    step_ids = []
                    
                    for step_id in level:
                        step = steps_dict[step_id]
                        enhanced_args = self._inject_dependency_data_cb(step, step_outputs)
                        step_with_args = ExecutionStep(
                            tool_id=step.tool_id,
                            args=enhanced_args,
                            dependencies=step.dependencies,
                            outputs=step.outputs,
                            timeout_ms=step.timeout_ms,
                            retry_count=step.retry_count
                        )
                        
                        tasks.append(self._execute_step_with_circuit_breaker(step_with_args, user_id, tenant_id, context))
                        step_ids.append(step_id)
                    
                    # Wait for all parallel steps with circuit breaker protection
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    # Process results and circuit breaker stats
                    for i, result in enumerate(results):
                        step_id = step_ids[i]
                        step = steps_dict[step_id]
                        
                        if isinstance(result, Exception):
                            execution_results[step_id] = ExecuteResponse(
                                status="ERROR",
                                summary=f"Step execution failed: {str(result)}",
                                data={},
                                metadata={"error": str(result)},
                                confidence=0.0
                            )
                        else:
                            execution_results[step_id] = result
                            
                            # Store step outputs
                            if step.outputs and result.status == "SUCCESS":
                                for output_name in step.outputs:
                                    step_outputs[output_name] = result.data
                        
                        circuit_breaker_stats[step_id] = self._get_circuit_breaker_stats(step.tool_id)
                
                # Check for failures considering circuit breaker states
                level_failed = any(
                    execution_results[step_id].status == "ERROR" 
                    for step_id in level
                )
                
                if level_failed and plan.merge_policy != "continue_on_error":
                    break
        
        except Exception as e:
            return ExecuteResponse(
                status="ERROR",
                summary=f"DAG execution failed: {str(e)}",
                data={},
                metadata={"error": str(e)},
                confidence=0.0
            )
        
        # Aggregate results with circuit breaker information
        return self._aggregate_dag_results_with_circuit_breaker(
            execution_results, step_outputs, circuit_breaker_stats, plan, start_time
        )
    
    async def _execute_step_with_circuit_breaker(self, step: ExecutionStep, user_id: str, tenant_id: str, context: Dict[str, Any]) -> ExecuteResponse:
        """Execute single step with circuit breaker protection"""
        start_time = time.time()
        
        try:
            # Get tool information
            tool = self.db_manager.get_tool(step.tool_id)
            if not tool:
                raise ValueError(f"Tool {step.tool_id} not found")
            
            # Get server information
            server = self.db_manager.get_server(tool.server_id)
            if not server:
                raise ValueError(f"Server {tool.server_id} not found")
            
            # Check circuit breaker state before execution
            if not self.circuit_breaker_manager.is_server_available(server.id):
                # Circuit is open - fail fast
                execution_time_ms = int((time.time() - start_time) * 1000)
                
                # Log circuit breaker event
                self.audit_logger.log_system_event(
                    action="circuit_breaker_open",
                    user_id=user_id,
                    tenant_id=tenant_id,
                    details={
                        "server_id": server.id,
                        "tool_id": step.tool_id,
                        "reason": "Circuit breaker is open"
                    }
                )
                
                return ExecuteResponse(
                    status="ERROR",
                    summary=f"Server {server.id} is unavailable (circuit breaker open)",
                    data={},
                    metadata={
                        "tool_id": step.tool_id,
                        "server_id": server.id,
                        "error_type": "CIRCUIT_OPEN",
                        "execution_time_ms": execution_time_ms
                    },
                    confidence=0.0,
                    execution_time_ms=execution_time_ms
                )
            
            # Execute the tool
            result = await self._call_mcp_tool_with_circuit_breaker(server.base_url, tool.name, step.args, server.id)
            
            execution_time_ms = int((time.time() - start_time) * 1000)
            
            # Record success in circuit breaker
            self.circuit_breaker_manager.record_success(server.id, tenant_id)
            
            # Structured logging
            log_tool_execution(
                tool_id=step.tool_id,
                server_id=server.id,
                success=True,
                duration_ms=execution_time_ms,
                request_id=context.get("request_id"),
                user_id=user_id,
                tenant_id=tenant_id
            )
            
            # Record metrics
            metrics.record_tool_execution(
                tool_id=step.tool_id,
                server_id=server.id,
                status="SUCCESS",
                tenant_id=tenant_id,
                duration=execution_time_ms / 1000.0
            )
            
            # Record telemetry
            self.db_manager.add_telemetry_event(
                tool_id=step.tool_id,
                success=True,
                latency_ms=execution_time_ms,
                cost=0.0  # MVP version doesn't track costs
            )
            
            return ExecuteResponse(
                status="SUCCESS",
                summary=f"Successfully executed {tool.name}",
                data=result,
                metadata={
                    "tool_id": step.tool_id,
                    "tool_name": tool.name,
                    "server_id": server.id,
                    "execution_time_ms": execution_time_ms
                },
                confidence=0.9
            )
            
        except Exception as e:
            execution_time_ms = int((time.time() - start_time) * 1000)
            
            # Record failure in circuit breaker
            if 'server' in locals():
                self.circuit_breaker_manager.record_failure(server.id, tenant_id)
            
            # Structured logging
            server_id = server.id if 'server' in locals() else "unknown"
            log_tool_execution(
                tool_id=step.tool_id,
                server_id=server_id,
                success=False,
                duration_ms=execution_time_ms,
                request_id=context.get("request_id"),
                user_id=user_id,
                tenant_id=tenant_id,
                error=str(e)
            )
            
            # Record metrics
            metrics.record_tool_execution(
                tool_id=step.tool_id,
                server_id=server_id,
                status="ERROR",
                tenant_id=tenant_id,
                duration=execution_time_ms / 1000.0
            )
            
            # Record failed telemetry
            self.db_manager.add_telemetry_event(
                tool_id=step.tool_id,
                success=False,
                latency_ms=execution_time_ms,
                cost=0.0
            )
            
            # Log failure event
            self.audit_logger.log_system_event(
                action="tool_execution_failed",
                user_id=user_id,
                tenant_id=tenant_id,
                details={
                    "tool_id": step.tool_id,
                    "server_id": server.id if 'server' in locals() else "unknown",
                    "error": str(e)
                }
            )
            
            return ExecuteResponse(
                status="ERROR",
                summary=f"Failed to execute {step.tool_id}: {str(e)}",
                data={},
                metadata={
                    "tool_id": step.tool_id,
                    "server_id": server.id if 'server' in locals() else "unknown",
                    "error": str(e),
                    "execution_time_ms": execution_time_ms
                },
                confidence=0.0,
                execution_time_ms=execution_time_ms
            )
    
    async def _call_mcp_tool_with_circuit_breaker(self, base_url: str, tool_name: str, args: Dict[str, Any], server_id: str) -> Dict[str, Any]:
        """Call MCP tool via HTTP with circuit breaker protection"""
        try:
            # Prepare the request payload
            if 'agent-wrapper' in base_url:
                # For agent wrapper, pass tool name as first argument
                goal = args.get("goal", "")
                arguments = [tool_name]
                if goal:
                    arguments.append(goal)
                
                payload = {
                    "arguments": arguments,
                    "parameters": args
                }
            else:
                payload = {
                    "tool_name": tool_name,
                    "parameters": args
                }
            
            # Make HTTP request to MCP server
            response = await self.client.post(
                f"{base_url}/execute",
                json=payload,
                timeout=30.0
            )
            response.raise_for_status()
            
            result = response.json()
            return result
            
        except httpx.RequestError as e:
            # Network/connection error - record failure
            self.circuit_breaker_manager.record_failure(server_id, "default")
            raise Exception(f"Failed to connect to MCP server {base_url}: {str(e)}")
        except httpx.HTTPStatusError as e:
            # HTTP error - record failure
            self.circuit_breaker_manager.record_failure(server_id, "default")
            raise Exception(f"MCP server returned error {e.response.status_code}: {e.response.text}")
        except Exception as e:
            # Other error - record failure
            self.circuit_breaker_manager.record_failure(server_id, "default")
            raise Exception(f"Unexpected error calling MCP tool: {str(e)}")
    
    def get_circuit_breaker_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics for all servers"""
        return self.circuit_breaker_manager.get_all_stats()
    
    def get_server_health(self, server_id: str) -> Dict[str, Any]:
        """Get health status for a specific server"""
        stats = self.circuit_breaker_manager.get_stats(server_id)
        if not stats:
            return {"status": "unknown", "server_id": server_id}
        
        return {
            "server_id": server_id,
            "status": stats.state.value if hasattr(stats.state, 'value') else str(stats.state),
            "failure_count": stats.failure_count,
            "success_count": stats.success_count,
            "total_requests": stats.total_requests,
            "total_failures": stats.total_failures,
            "total_successes": stats.total_successes,
            "last_failure_time": stats.last_failure_time.isoformat() if stats.last_failure_time else None,
            "last_success_time": stats.last_success_time.isoformat() if stats.last_success_time else None,
            "is_available": self.circuit_breaker_manager.is_server_available(server_id)
        }
    
    async def close(self):
        """Close HTTP client"""
        await self.client.aclose()
    
    def _resolve_dependencies_with_circuit_breaker(self, steps_dict: Dict[str, ExecutionStep]) -> List[List[str]]:
        """Resolve dependencies with circuit breaker state awareness"""
        from collections import defaultdict, deque
        
        # Build dependency graph
        in_degree = defaultdict(int)
        graph = defaultdict(list)
        
        # Initialize all steps
        for step_id in steps_dict.keys():
            in_degree[step_id] = 0
        
        # Build graph based on dependencies
        for step_id, step in steps_dict.items():
            for dep in step.dependencies:
                # Find step that provides this dependency
                provider_step = None
                for other_id, other_step in steps_dict.items():
                    if dep in other_step.outputs:
                        provider_step = other_id
                        break
                
                if provider_step:
                    graph[provider_step].append(step_id)
                    in_degree[step_id] += 1
        
        # Topological sort with level detection
        execution_order = []
        queue = deque([step_id for step_id in steps_dict.keys() if in_degree[step_id] == 0])
        
        while queue:
            # Current level - all steps with no remaining dependencies
            current_level = list(queue)
            queue.clear()
            
            if not current_level:
                break
                
            execution_order.append(current_level)
            
            # Process each step in current level
            for step_id in current_level:
                for neighbor in graph[step_id]:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)
        
        # Check for circular dependencies
        remaining_steps = [step_id for step_id in steps_dict.keys() 
                          if step_id not in [s for level in execution_order for s in level]]
        
        if remaining_steps:
            raise ValueError(f"Circular dependency detected in steps: {remaining_steps}")
        
        return execution_order
    
    def _inject_dependency_data_cb(self, step: ExecutionStep, step_outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Inject dependency data with circuit breaker context"""
        enhanced_args = step.args.copy()
        
        # For each dependency, try to inject the output data
        for dep in step.dependencies:
            if dep in step_outputs:
                # Add dependency data
                enhanced_args[f"dependency_{dep}"] = step_outputs[dep]
                
                # Smart field mapping
                dep_data = step_outputs[dep]
                if isinstance(dep_data, dict):
                    if "content" in dep_data and "content" not in enhanced_args:
                        enhanced_args["content"] = dep_data["content"]
                    if "path" in dep_data and "path" not in enhanced_args:
                        enhanced_args["path"] = dep_data["path"]
                    if "output" in dep_data and "input" not in enhanced_args:
                        enhanced_args["input"] = dep_data["output"]
        
        return enhanced_args
    
    def _get_circuit_breaker_stats(self, tool_id: str) -> Dict[str, Any]:
        """Get circuit breaker statistics for a tool"""
        try:
            # Get tool to find its server
            tool = self.db_manager.get_tool_by_id(tool_id)
            if not tool:
                return {"status": "unknown", "tool_id": tool_id}
            
            server_id = tool.id.split('.')[0] if '.' in tool.id else tool_id
            stats = self.circuit_breaker_manager.get_stats(server_id)
            
            if not stats:
                return {"status": "unknown", "server_id": server_id, "tool_id": tool_id}
            
            return {
                "tool_id": tool_id,
                "server_id": server_id,
                "status": stats.state.value if hasattr(stats.state, 'value') else str(stats.state),
                "failure_count": stats.failure_count,
                "is_available": self.circuit_breaker_manager.is_server_available(server_id)
            }
        except Exception:
            return {"status": "error", "tool_id": tool_id}
    
    def _aggregate_dag_results_with_circuit_breaker(
        self, 
        execution_results: Dict[str, ExecuteResponse], 
        step_outputs: Dict[str, Any],
        circuit_breaker_stats: Dict[str, Dict[str, Any]],
        plan: ExecutionPlan, 
        start_time: float
    ) -> ExecuteResponse:
        """Aggregate DAG results with circuit breaker information"""
        execution_time_ms = int((time.time() - start_time) * 1000)
        
        if not execution_results:
            return ExecuteResponse(
                status="SUCCESS",
                summary="No steps executed",
                data={},
                metadata={},
                confidence=1.0,
                execution_time_ms=execution_time_ms
            )
        
        # Count successful and failed steps
        successful_steps = [r for r in execution_results.values() if r.status == "SUCCESS"]
        failed_steps = [r for r in execution_results.values() if r.status == "ERROR"]
        
        # Determine overall status
        if not failed_steps:
            status = "SUCCESS"
            summary = f"Successfully executed all {len(successful_steps)} steps with circuit breaker protection"
        elif not successful_steps:
            status = "ERROR"
            summary = f"All {len(failed_steps)} steps failed despite circuit breaker protection"
        else:
            status = "PARTIAL"
            summary = f"{len(successful_steps)} succeeded, {len(failed_steps)} failed with circuit breaker protection"
        
        # Enhanced data aggregation with circuit breaker info
        aggregated_data = {
            "workflow_results": {
                "final_outputs": step_outputs,
                "step_results": {step_id: result.data for step_id, result in execution_results.items()},
                "execution_summary": {
                    "total_steps": len(execution_results),
                    "successful_steps": len(successful_steps),
                    "failed_steps": len(failed_steps),
                    "merge_policy": plan.merge_policy
                },
                "circuit_breaker_stats": circuit_breaker_stats
            }
        }
        
        # Calculate average confidence
        confidences = [r.confidence for r in execution_results.values() if r.confidence is not None]
        avg_confidence = sum(confidences) / len(confidences) if confidences else None
        
        return ExecuteResponse(
            status=status,
            summary=summary,
            data=aggregated_data,
            metadata={
                "execution_type": "DAG_CircuitBreaker",
                "total_steps": len(execution_results),
                "successful_steps": len(successful_steps),
                "failed_steps": len(failed_steps),
                "merge_policy": plan.merge_policy,
                "circuit_breaker_protection": True,
                "fault_tolerance": True
            },
            confidence=avg_confidence,
            execution_time_ms=execution_time_ms
        )
