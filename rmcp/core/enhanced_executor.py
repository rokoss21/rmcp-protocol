"""
Enhanced Executor - supports both atomic tools and AI agents
Implements strategic execution with agent delegation capabilities
"""

import httpx
import time
from typing import Dict, List, Any, Optional
from datetime import datetime

from ..storage.database import DatabaseManager
from ..models.plan import ExecutionPlan, ExecutionStep, ExecutionStrategy
from ..models.request import ExecuteResponse
from ..models.tool import Tool
from ..security.circuit_breaker import CircuitBreakerManager
from ..telemetry.engine import TelemetryEngine
from ..logging.config import get_logger, log_tool_execution


class EnhancedExecutor:
    """
    Enhanced executor supporting both atomic tools and AI agents
    
    Features:
    - Agent delegation via HTTP API calls
    - Circuit breaker protection for both tools and agents
    - Telemetry tracking for all execution types
    - Strategic execution with fallback mechanisms
    """
    
    def __init__(
        self, 
        db_manager: DatabaseManager,
        circuit_breaker_manager: Optional[CircuitBreakerManager] = None,
        telemetry_engine: Optional[TelemetryEngine] = None
    ):
        self.db_manager = db_manager
        self.circuit_breaker_manager = circuit_breaker_manager or CircuitBreakerManager()
        self.telemetry_engine = telemetry_engine or TelemetryEngine(db_manager)
        self.client = httpx.AsyncClient(timeout=60.0)
        self.logger = get_logger(__name__)
    
    async def execute(self, plan: ExecutionPlan) -> ExecuteResponse:
        """
        Execute the execution plan with enhanced capabilities
        
        Supports:
        - SOLO strategy (single tool/agent)
        - PARALLEL strategy (multiple tools/agents)
        - DAG strategy (complex workflows)
        """
        start_time = time.time()
        
        try:
            self.logger.info(
                "Execution started",
                strategy=plan.strategy.value if hasattr(plan.strategy, 'value') else str(plan.strategy),
                steps_count=len(plan.steps),
                requires_approval=plan.requires_approval
            )
            
            if plan.strategy == ExecutionStrategy.SOLO:
                result = await self._execute_solo(plan)
            elif plan.strategy == ExecutionStrategy.PARALLEL:
                result = await self._execute_parallel(plan)
            elif plan.strategy == ExecutionStrategy.DAG:
                result = await self._execute_dag_enhanced(plan)
            else:
                raise ValueError(f"Unsupported execution strategy: {plan.strategy}")
            
            execution_time_ms = int((time.time() - start_time) * 1000)
            result.execution_time_ms = execution_time_ms
            
            self.logger.info(
                "Execution completed",
                strategy=plan.strategy.value if hasattr(plan.strategy, 'value') else str(plan.strategy),
                status=result.status,
                execution_time_ms=execution_time_ms
            )
            
            return result
            
        except Exception as e:
            execution_time_ms = int((time.time() - start_time) * 1000)
            self.logger.error(
                "Execution failed",
                strategy=plan.strategy.value if hasattr(plan.strategy, 'value') else str(plan.strategy),
                error=str(e),
                execution_time_ms=execution_time_ms
            )
            
            return ExecuteResponse(
                status="ERROR",
                summary=f"Execution failed: {str(e)}",
                data={},
                metadata={"error": str(e), "execution_time_ms": execution_time_ms},
                confidence=0.0,
                execution_time_ms=execution_time_ms
            )
    
    async def _execute_solo(self, plan: ExecutionPlan) -> ExecuteResponse:
        """Execute single step plan (tool or agent)"""
        if not plan.steps or len(plan.steps) == 0:
            return ExecuteResponse(
                status="SUCCESS",
                summary="No steps to execute",
                data={},
                metadata={},
                confidence=1.0
            )
        
        step = plan.steps[0]
        return await self._execute_step_with_circuit_breaker(step)
    
    async def _execute_parallel(self, plan: ExecutionPlan) -> ExecuteResponse:
        """Execute parallel steps (tools and/or agents)"""
        results = []
        
        for step in plan.steps:
            try:
                result = await self._execute_step_with_circuit_breaker(step)
                results.append(result)
            except Exception as e:
                self.logger.error(
                    "Parallel step failed",
                    tool_id=step.tool_id,
                    error=str(e)
                )
                results.append(ExecuteResponse(
                    status="ERROR",
                    summary=f"Step failed: {str(e)}",
                    data={},
                    metadata={"error": str(e), "tool_id": step.tool_id},
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
    
    async def _execute_dag(self, plan: ExecutionPlan) -> ExecuteResponse:
        """Execute DAG plan (complex workflows with dependencies) - legacy method"""
        return await self._execute_dag_enhanced(plan)
    
    async def _execute_dag_enhanced(self, plan: ExecutionPlan) -> ExecuteResponse:
        """Execute DAG plan with enhanced capabilities, circuit breakers, and agent support"""
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
        
        self.logger.info(f"Starting DAG execution with {len(steps_dict)} steps")
        
        # Build dependency graph and topological order
        try:
            execution_order = self._resolve_dependencies_enhanced(steps_dict)
            self.logger.info(f"Dependency resolution complete: {len(execution_order)} levels")
        except Exception as e:
            self.logger.error(f"Dependency resolution failed: {e}")
            return ExecuteResponse(
                status="ERROR",
                summary=f"Dependency resolution failed: {str(e)}",
                data={},
                metadata={"error": str(e)},
                confidence=0.0
            )
        
        # Execute steps in topological order with parallelization and circuit breakers
        execution_results = {}
        step_outputs = {}  # Store outputs from each step
        
        start_time = time.time()
        
        try:
            for level_idx, level in enumerate(execution_order):
                self.logger.info(f"Executing level {level_idx + 1}/{len(execution_order)} with {len(level)} steps")
                
                # Execute all steps in this level in parallel
                if len(level) == 1:
                    # Single step - execute directly with circuit breaker
                    step_id = level[0]
                    step = steps_dict[step_id]
                    
                    # Inject dependency results into step arguments
                    enhanced_args = self._inject_dependency_data_enhanced(step, step_outputs)
                    step_with_args = ExecutionStep(
                        tool_id=step.tool_id,
                        args=enhanced_args,
                        dependencies=step.dependencies,
                        outputs=step.outputs,
                        timeout_ms=step.timeout_ms,
                        retry_count=step.retry_count
                    )
                    
                    result = await self._execute_step_with_circuit_breaker(step_with_args)
                    execution_results[step_id] = result
                    
                    # Store step outputs for dependent steps
                    if step.outputs and result.status == "SUCCESS":
                        for output_name in step.outputs:
                            step_outputs[output_name] = result.data
                
                else:
                    # Multiple steps - execute in parallel with circuit breakers
                    tasks = []
                    step_ids = []
                    
                    for step_id in level:
                        step = steps_dict[step_id]
                        enhanced_args = self._inject_dependency_data_enhanced(step, step_outputs)
                        step_with_args = ExecutionStep(
                            tool_id=step.tool_id,
                            args=enhanced_args,
                            dependencies=step.dependencies,
                            outputs=step.outputs,
                            timeout_ms=step.timeout_ms,
                            retry_count=step.retry_count
                        )
                        
                        tasks.append(self._execute_step_with_circuit_breaker(step_with_args))
                        step_ids.append(step_id)
                    
                    # Wait for all parallel steps to complete
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    # Process results
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
                            
                            # Store step outputs for dependent steps
                            if step.outputs and result.status == "SUCCESS":
                                for output_name in step.outputs:
                                    step_outputs[output_name] = result.data
                
                # Check for failures that should halt execution
                level_failed = any(
                    execution_results[step_id].status == "ERROR" 
                    for step_id in level
                )
                
                if level_failed:
                    if plan.merge_policy == "continue_on_error":
                        self.logger.warning(f"Level {level_idx + 1} had failures but continuing due to merge policy")
                    else:
                        self.logger.error(f"Level {level_idx + 1} failed, halting execution")
                        break
        
        except Exception as e:
            self.logger.error(f"DAG execution failed: {e}")
            return ExecuteResponse(
                status="ERROR",
                summary=f"DAG execution failed: {str(e)}",
                data={},
                metadata={"error": str(e)},
                confidence=0.0
            )
        
        # Aggregate results based on merge policy
        return self._aggregate_dag_results_enhanced(execution_results, step_outputs, plan, start_time)
    
    def _resolve_dependencies_enhanced(self, steps_dict: Dict[str, ExecutionStep]) -> List[List[str]]:
        """Enhanced dependency resolution with better error handling and logging"""
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
                    self.logger.debug(f"Dependency: {step_id} depends on {provider_step} for '{dep}'")
                else:
                    # Dependency not found - might be external data
                    self.logger.warning(f"External dependency '{dep}' for step {step_id} - assuming provided via context")
        
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
    
    def _inject_dependency_data_enhanced(self, step: ExecutionStep, step_outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced dependency data injection with smart field mapping"""
        enhanced_args = step.args.copy()
        
        # For each dependency, try to inject the output data
        for dep in step.dependencies:
            if dep in step_outputs:
                # Add dependency data with clear naming
                enhanced_args[f"dependency_{dep}"] = step_outputs[dep]
                
                # Smart field mapping based on tool type and dependency data
                dep_data = step_outputs[dep]
                if isinstance(dep_data, dict):
                    # Intelligent field mapping
                    self._smart_field_mapping(enhanced_args, dep_data, step.tool_id)
                elif isinstance(dep_data, str):
                    # If dependency is a string, try to use it as content or path
                    if "content" not in enhanced_args and dep_data:
                        enhanced_args["content"] = dep_data
        
        return enhanced_args
    
    def _smart_field_mapping(self, enhanced_args: Dict[str, Any], dep_data: Dict[str, Any], tool_id: str):
        """Smart field mapping based on tool type and available data"""
        tool_name = tool_id.lower()
        
        # File operation tools
        if "create_file" in tool_name or "append_to_file" in tool_name:
            if "path" in dep_data and "path" not in enhanced_args:
                enhanced_args["path"] = dep_data["path"]
            if "content" in dep_data and "content" not in enhanced_args:
                enhanced_args["content"] = dep_data["content"]
            if "output" in dep_data and "path" not in enhanced_args:
                enhanced_args["path"] = dep_data["output"]
        
        # Search/analysis tools
        elif "grep" in tool_name or "search" in tool_name or "wc" in tool_name:
            if "path" in dep_data and "text" not in enhanced_args:
                enhanced_args["text"] = dep_data["path"]  # Use file path as input
            if "content" in dep_data and "text" not in enhanced_args:
                enhanced_args["text"] = dep_data["content"]
        
        # Generic fallback mappings
        if "result" in dep_data and "input" not in enhanced_args:
            enhanced_args["input"] = dep_data["result"]
        if "value" in dep_data and "value" not in enhanced_args:
            enhanced_args["value"] = dep_data["value"]
    
    def _aggregate_dag_results_enhanced(
        self, 
        execution_results: Dict[str, ExecuteResponse], 
        step_outputs: Dict[str, Any],
        plan: ExecutionPlan, 
        start_time: float
    ) -> ExecuteResponse:
        """Enhanced result aggregation with detailed metrics and telemetry"""
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
        partial_steps = [r for r in execution_results.values() if r.status == "PARTIAL"]
        
        # Determine overall status
        if not failed_steps and not partial_steps:
            status = "SUCCESS"
            summary = f"Successfully executed all {len(successful_steps)} steps"
        elif not successful_steps:
            status = "ERROR"
            summary = f"All {len(failed_steps)} steps failed"
        else:
            status = "PARTIAL"
            summary = f"{len(successful_steps)} succeeded, {len(failed_steps)} failed, {len(partial_steps)} partial"
        
        # Enhanced data aggregation
        aggregated_data = {
            "workflow_results": {
                "final_outputs": step_outputs,
                "step_results": {step_id: {
                    "status": result.status,
                    "summary": result.summary,
                    "data": result.data,
                    "confidence": result.confidence,
                    "execution_time_ms": result.execution_time_ms
                } for step_id, result in execution_results.items()},
                "execution_flow": {
                    "total_steps": len(execution_results),
                    "successful_steps": len(successful_steps),
                    "failed_steps": len(failed_steps),
                    "partial_steps": len(partial_steps),
                    "merge_policy": plan.merge_policy
                }
            }
        }
        
        # If there's a clear final result, include it at the top level
        if plan.merge_policy == "last_success" and successful_steps:
            aggregated_data["final_result"] = successful_steps[-1].data
        elif plan.merge_policy == "first_success" and successful_steps:
            aggregated_data["final_result"] = successful_steps[0].data
        
        # Calculate weighted confidence
        confidences = [r.confidence for r in execution_results.values() if r.confidence is not None]
        if confidences:
            # Weight by success
            weights = [1.0 if r.status == "SUCCESS" else 0.5 if r.status == "PARTIAL" else 0.1 
                      for r in execution_results.values() if r.confidence is not None]
            weighted_confidence = sum(c * w for c, w in zip(confidences, weights)) / sum(weights)
        else:
            weighted_confidence = None
        
        # Enhanced metadata
        metadata = {
            "execution_type": "DAG_Enhanced",
            "total_steps": len(execution_results),
            "successful_steps": len(successful_steps),
            "failed_steps": len(failed_steps),
            "partial_steps": len(partial_steps),
            "merge_policy": plan.merge_policy,
            "parallel_execution": True,
            "circuit_breaker_protection": True,
            "agent_support": True
        }
        
        self.logger.info(
            "DAG execution completed",
            status=status,
            total_steps=len(execution_results),
            successful=len(successful_steps),
            failed=len(failed_steps),
            execution_time_ms=execution_time_ms
        )
        
        return ExecuteResponse(
            status=status,
            summary=summary,
            data=aggregated_data,
            metadata=metadata,
            confidence=weighted_confidence,
            execution_time_ms=execution_time_ms
        )
    
    async def _execute_step_with_circuit_breaker(self, step: ExecutionStep) -> ExecuteResponse:
        """
        Execute single step with circuit breaker protection
        
        This is the core method that handles both atomic tools and agents
        """
        start_time = time.time()
        
        try:
            # Get tool information
            tool = self.db_manager.get_tool(step.tool_id)
            if not tool:
                raise ValueError(f"Tool {step.tool_id} not found")
            
            # Check circuit breaker status
            if not self.circuit_breaker_manager.is_server_available(tool.server_id):
                raise Exception(f"Circuit breaker is open for server {tool.server_id}")
            
            # Execute based on tool type
            if tool.tool_type == 'agent':
                result = await self._execute_agent(tool, step.args)
            else:
                result = await self._execute_atomic_tool(tool, step.args)
            
            execution_time_ms = int((time.time() - start_time) * 1000)
            
            # Record success in circuit breaker
            self.circuit_breaker_manager.record_success(tool.server_id, execution_time_ms)
            
            # Record telemetry
            await self.telemetry_engine.record_tool_execution(
                tool_id=step.tool_id,
                success=True,
                latency_ms=execution_time_ms,
                request_text=step.args.get('goal', ''),
                cost=0.0  # Can be enhanced later
            )
            
            # Log execution
            log_tool_execution(
                tool_id=step.tool_id,
                server_id=tool.server_id,
                success=True,
                duration_ms=execution_time_ms,
                tool_name=tool.name,
                tool_type=tool.tool_type,
                goal=step.args.get('goal', '')
            )
            
            return ExecuteResponse(
                status="SUCCESS",
                summary=f"Successfully executed {tool.name} ({tool.tool_type})",
                data=result,
                metadata={
                    "tool_id": step.tool_id,
                    "tool_name": tool.name,
                    "tool_type": tool.tool_type,
                    "execution_time_ms": execution_time_ms
                },
                confidence=0.9
            )
            
        except Exception as e:
            execution_time_ms = int((time.time() - start_time) * 1000)
            
            # Record failure in circuit breaker
            self.circuit_breaker_manager.record_failure(step.tool_id, execution_time_ms)
            
            # Record failed telemetry
            await self.telemetry_engine.record_tool_execution(
                tool_id=step.tool_id,
                success=False,
                latency_ms=execution_time_ms,
                request_text=step.args.get('goal', ''),
                cost=0.0
            )
            
            # Log execution failure
            log_tool_execution(
                tool_id=step.tool_id,
                server_id=tool.server_id if tool else "unknown",
                success=False,
                duration_ms=execution_time_ms,
                tool_name=tool.name if tool else "unknown",
                tool_type=tool.tool_type if tool else "unknown",
                goal=step.args.get('goal', ''),
                error=str(e)
            )
            
            return ExecuteResponse(
                status="ERROR",
                summary=f"Failed to execute {step.tool_id}: {str(e)}",
                data={},
                metadata={
                    "tool_id": step.tool_id,
                    "error": str(e),
                    "execution_time_ms": execution_time_ms
                },
                confidence=0.0
            )
    
    async def _execute_atomic_tool(self, tool: Tool, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute atomic tool via MCP protocol
        
        This is the existing logic for atomic tools
        """
        # Get server information
        server = self.db_manager.get_server(tool.server_id)
        if not server:
            raise ValueError(f"Server {tool.server_id} not found")
        
        # Prepare the request payload for MCP
        if 'agent-wrapper' in server.base_url:
            # For agent wrapper, pass tool name as first argument
            goal = args.get("goal", "")
            arguments = [tool.name]
            if goal:
                arguments.append(goal)
            
            payload = {
                "arguments": arguments,
                "parameters": args
            }
        else:
            # For regular MCP servers, extract tool name without prefix
            tool_name = tool.name.split('.')[-1] if '.' in tool.name else tool.name
            payload = {
                "tool_name": tool_name,
                "parameters": args
            }
        
        # Make HTTP request to MCP server
        response = await self.client.post(
            f"{server.base_url}/execute",
            json=payload,
            timeout=30.0
        )
        response.raise_for_status()
        
        result = response.json()
        return result
    
    async def _execute_agent(self, agent: Tool, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute AI agent via HTTP API
        
        This is the new logic for agent delegation
        """
        # Get server information (agent endpoint)
        server = self.db_manager.get_server(agent.server_id)
        if not server:
            raise ValueError(f"Agent server {agent.server_id} not found")
        
        # Prepare the request payload for agent
        # Agents expect a different format than MCP tools
        payload = {
            "goal": args.get('goal', ''),
            "context": args.get('context', {}),
            "specialization": agent.specialization if hasattr(agent, 'specialization') and agent.specialization else 'general',
            "abstraction_level": agent.abstraction_level if hasattr(agent, 'abstraction_level') and agent.abstraction_level else 'medium',
            "parameters": args
        }
        
        self.logger.info(
            "Delegating to agent",
            agent_id=agent.id,
            agent_name=agent.name,
            endpoint=server.base_url,
            goal=payload.get('goal', '')
        )
        
        # Make HTTP request to agent endpoint
        response = await self.client.post(
            f"{server.base_url}/execute",
            json=payload,
            timeout=getattr(agent, 'avg_execution_time_ms', 30000) / 1000.0  # Convert ms to seconds
        )
        response.raise_for_status()
        
        result = response.json()
        
        self.logger.info(
            "Agent execution completed",
            agent_id=agent.id,
            agent_name=agent.name,
            success=True
        )
        
        return result
    
    async def close(self):
        """Close HTTP client"""
        await self.client.aclose()
