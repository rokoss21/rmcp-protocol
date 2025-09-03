"""
Simple Executor - basic execution logic for MVP
"""

import httpx
import time
from typing import Dict, List, Any, Optional
from datetime import datetime

from ..storage.database import DatabaseManager
from ..models.plan import ExecutionPlan, ExecutionStep, ExecutionStrategy
from ..models.request import ExecuteResponse


class SimpleExecutor:
    """Simple executor for MVP version of RMCP"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.client = httpx.AsyncClient(timeout=60.0)
    
    async def execute(self, plan: ExecutionPlan) -> ExecuteResponse:
        """
        Execute the execution plan
        
        In MVP version supports only SOLO strategy
        """
        start_time = time.time()
        
        try:
            if plan.strategy == ExecutionStrategy.SOLO:
                result = await self._execute_solo(plan)
            elif plan.strategy == ExecutionStrategy.PARALLEL:
                result = await self._execute_parallel(plan)
            elif plan.strategy == ExecutionStrategy.DAG:
                result = await self._execute_dag(plan)
            else:
                raise ValueError(f"Unsupported execution strategy: {plan.strategy}")
            
            execution_time_ms = int((time.time() - start_time) * 1000)
            result.execution_time_ms = execution_time_ms
            
            return result
            
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
    
    async def _execute_solo(self, plan: ExecutionPlan) -> ExecuteResponse:
        """Execute single step plan"""
        if not plan.steps or len(plan.steps) == 0:
            return ExecuteResponse(
                status="SUCCESS",
                summary="No steps to execute",
                data={},
                metadata={},
                confidence=1.0
            )
        
        step = plan.steps[0]
        result = await self._execute_step(step)
        # Normalize into step_results structure for diagnostics
        step_id = getattr(step, 'id', 'step_0')
        return ExecuteResponse(
            status=result.status,
            summary=result.summary,
            data={
                "step_results": {
                    step_id: {
                        "success": result.status == "SUCCESS",
                        "output": result.data.get("output", "") if isinstance(result.data, dict) else "",
                        "error": result.metadata.get("error") if isinstance(result.metadata, dict) else None,
                        "exit_code": 0 if result.status == "SUCCESS" else 1,
                        "tool_name": result.metadata.get("tool_name") if isinstance(result.metadata, dict) and result.metadata.get("tool_name") else None,
                        "metadata": result.metadata
                    }
                },
                "execution_summary": {
                    "total_steps": 1,
                    "successful_steps": 1 if result.status == "SUCCESS" else 0,
                    "failed_steps": 0 if result.status == "SUCCESS" else 1,
                    "steps_details": {
                        step_id: {
                            "status": result.status,
                            "summary": result.summary,
                            "confidence": result.confidence
                        }
                    }
                }
            },
            metadata=result.metadata,
            confidence=result.confidence
        )
    
    async def _execute_parallel(self, plan: ExecutionPlan) -> ExecuteResponse:
        """Execute parallel steps (MVP version - sequential execution)"""
        results = []
        
        for step in plan.steps:
            try:
                result = await self._execute_step(step)
                results.append(result)
            except Exception as e:
                results.append(ExecuteResponse(
                    status="ERROR",
                    summary=f"Step failed: {str(e)}",
                    data={},
                    metadata={"error": str(e)},
                    confidence=0.0
                ))
        
        # Build step_results map
        step_results_map = {}
        steps_details = {}
        successful = 0
        failed = 0
        for idx, res in enumerate(results):
            step_id = f"step_{idx}"
            is_success = res.status == "SUCCESS"
            successful += 1 if is_success else 0
            failed += 0 if is_success else 1
            step_results_map[step_id] = {
                "success": is_success,
                "output": res.data.get("output", "") if isinstance(res.data, dict) else "",
                "error": res.metadata.get("error") if isinstance(res.metadata, dict) else None,
                "exit_code": 0 if is_success else 1,
                "tool_name": res.metadata.get("tool_name") if isinstance(res.metadata, dict) and res.metadata.get("tool_name") else None,
                "metadata": res.metadata
            }
            steps_details[step_id] = {
                "status": res.status,
                "summary": res.summary,
                "confidence": res.confidence
            }
        overall_status = "SUCCESS" if failed == 0 else ("ERROR" if successful == 0 else "PARTIAL")
        overall_summary = (
            "Successfully executed all steps" if overall_status == "SUCCESS" else
            ("All steps failed" if overall_status == "ERROR" else f"{successful} steps succeeded, {failed} steps failed")
        )
        return ExecuteResponse(
            status=overall_status,
            summary=overall_summary,
            data={
                "step_results": step_results_map,
                "execution_summary": {
                    "total_steps": len(results),
                    "successful_steps": successful,
                    "failed_steps": failed,
                    "steps_details": steps_details
                }
            },
            metadata={},
            confidence=sum([r.confidence or 0.0 for r in results]) / len(results) if results else 0.0
        )
    
    async def _execute_dag(self, plan: ExecutionPlan) -> ExecuteResponse:
        """Execute DAG plan with dependency resolution and parallel execution"""
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
        
        # Build dependency graph and topological order
        try:
            execution_order = self._resolve_dependencies(steps_dict)
        except Exception as e:
            return ExecuteResponse(
                status="ERROR",
                summary=f"Dependency resolution failed: {str(e)}",
                data={},
                metadata={"error": str(e)},
                confidence=0.0
            )
        
        # Execute steps in topological order with parallelization
        execution_results = {}
        step_outputs = {}  # Store outputs from each step
        
        start_time = time.time()
        
        try:
            for level in execution_order:
                # Execute all steps in this level in parallel
                if len(level) == 1:
                    # Single step - execute directly
                    step_id = level[0]
                    step = steps_dict[step_id]
                    
                    # Inject dependency results into step arguments
                    enhanced_args = self._inject_dependency_data(step, step_outputs)
                    step_with_args = ExecutionStep(
                        tool_id=step.tool_id,
                        args=enhanced_args,
                        dependencies=step.dependencies,
                        outputs=step.outputs,
                        timeout_ms=step.timeout_ms,
                        retry_count=step.retry_count
                    )
                    
                    result = await self._execute_step(step_with_args)
                    execution_results[step_id] = result
                    
                    # Store step outputs for dependent steps
                    if step.outputs and result.status == "SUCCESS":
                        for output_name in step.outputs:
                            step_outputs[output_name] = result.data
                
                else:
                    # Multiple steps - execute in parallel
                    tasks = []
                    step_ids = []
                    
                    for step_id in level:
                        step = steps_dict[step_id]
                        enhanced_args = self._inject_dependency_data(step, step_outputs)
                        step_with_args = ExecutionStep(
                            tool_id=step.tool_id,
                            args=enhanced_args,
                            dependencies=step.dependencies,
                            outputs=step.outputs,
                            timeout_ms=step.timeout_ms,
                            retry_count=step.retry_count
                        )
                        
                        tasks.append(self._execute_step(step_with_args))
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
        
        # Aggregate results based on merge policy
        return self._aggregate_dag_results(execution_results, step_outputs, plan, start_time)
    
    def _resolve_dependencies(self, steps_dict: Dict[str, ExecutionStep]) -> List[List[str]]:
        """
        Resolve step dependencies using topological sort.
        Returns list of levels, where each level contains steps that can be executed in parallel.
        """
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
                else:
                    # Dependency not found - this might be external data
                    # For now, we'll allow it and assume it's provided via context
                    pass
        
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
    
    def _inject_dependency_data(self, step: ExecutionStep, step_outputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Inject data from dependency outputs into step arguments.
        """
        enhanced_args = step.args.copy()
        
        # For each dependency, try to inject the output data
        for dep in step.dependencies:
            if dep in step_outputs:
                # Simple injection strategy: add dependency data to args
                enhanced_args[f"dependency_{dep}"] = step_outputs[dep]
                
                # Also try to inject into common parameter names
                dep_data = step_outputs[dep]
                if isinstance(dep_data, dict):
                    # If dependency output is a dict, try to merge useful fields
                    if "content" in dep_data and "content" not in enhanced_args:
                        enhanced_args["content"] = dep_data["content"]
                    if "path" in dep_data and "path" not in enhanced_args:
                        enhanced_args["path"] = dep_data["path"]
                    if "output" in dep_data and "input" not in enhanced_args:
                        enhanced_args["input"] = dep_data["output"]
        
        return enhanced_args
    
    def _aggregate_dag_results(
        self, 
        execution_results: Dict[str, ExecuteResponse], 
        step_outputs: Dict[str, Any],
        plan: ExecutionPlan, 
        start_time: float
    ) -> ExecuteResponse:
        """
        Aggregate results from all DAG steps into a single response.
        """
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
            summary = f"Successfully executed {len(successful_steps)} steps"
        elif not successful_steps:
            status = "ERROR"
            summary = f"All {len(failed_steps)} steps failed"
        else:
            status = "PARTIAL"
            summary = f"{len(successful_steps)} steps succeeded, {len(failed_steps)} steps failed"
        
        # Aggregate data based on merge policy
        aggregated_data = {}
        
        if plan.merge_policy == "last_success":
            # Use data from the last successful step
            for result in successful_steps:
                if result.data:
                    aggregated_data.update(result.data)
        elif plan.merge_policy == "first_success":
            # Use data from the first successful step
            for result in successful_steps:
                if result.data:
                    aggregated_data = result.data
                    break
        else:  # "all" or default
            # Combine all step results
            aggregated_data = {
                "step_results": {
                    step_id: {
                        "success": result.status == "SUCCESS",
                        "output": result.data.get("output", "") if isinstance(result.data, dict) else "",
                        "error": result.metadata.get("error") if isinstance(result.metadata, dict) else None,
                        "exit_code": 0 if result.status == "SUCCESS" else 1,
                        "tool_name": result.metadata.get("tool_name") if isinstance(result.metadata, dict) and result.metadata.get("tool_name") else self.db_manager.get_tool(step_id).name if self.db_manager.get_tool(step_id) else "unknown",
                        "metadata": result.metadata
                    }
                    for step_id, result in execution_results.items()
                },
                "step_outputs": step_outputs,
                "execution_summary": {
                    "total_steps": len(execution_results),
                    "successful_steps": len(successful_steps),
                    "failed_steps": len(failed_steps),
                    "steps_details": {
                        step_id: {
                            "status": result.status,
                            "summary": result.summary,
                            "confidence": result.confidence
                        }
                        for step_id, result in execution_results.items()
                    }
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
                "execution_type": "DAG",
                "total_steps": len(execution_results),
                "successful_steps": len(successful_steps),
                "failed_steps": len(failed_steps),
                "merge_policy": plan.merge_policy
            },
            confidence=avg_confidence,
            execution_time_ms=execution_time_ms
        )
    
    async def _execute_step(self, step: ExecutionStep) -> ExecuteResponse:
        """Execute single step"""
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
            
            # Execute the tool - use full tool name for agent wrapper
            if 'agent-wrapper' in server.base_url:
                # For agent wrapper, use full tool name and pass goal in arguments
                result = await self._call_mcp_tool_with_args(server.base_url, tool.name, step.args)
            else:
                # For regular MCP servers, extract tool name without prefix
                tool_name = tool.name.split('.')[-1] if '.' in tool.name else tool.name
                result = await self._call_mcp_tool(server.base_url, tool_name, step.args)
            
            execution_time_ms = int((time.time() - start_time) * 1000)
            
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
                    "execution_time_ms": execution_time_ms
                },
                confidence=0.9
            )
            
        except Exception as e:
            execution_time_ms = int((time.time() - start_time) * 1000)
            
            # Record failed telemetry
            self.db_manager.add_telemetry_event(
                tool_id=step.tool_id,
                success=False,
                latency_ms=execution_time_ms,
                cost=0.0
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
    
    async def _call_mcp_tool(self, base_url: str, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Call MCP tool via HTTP"""
        try:
            # Prepare the request payload
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
            raise Exception(f"Failed to connect to MCP server {base_url}: {str(e)}")
        except httpx.HTTPStatusError as e:
            raise Exception(f"MCP server returned error {e.response.status_code}: {e.response.text}")
        except Exception as e:
            raise Exception(f"Unexpected error calling MCP tool: {str(e)}")
    
    async def _call_mcp_tool_with_args(self, base_url: str, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Call MCP tool with arguments array (for agent wrapper)"""
        try:
            # For agent wrapper, pass tool name as first argument
            goal = args.get("goal", "")
            arguments = [tool_name]
            if goal:
                arguments.append(goal)
            
            # Prepare the request payload with both arguments and parameters
            payload = {
                "arguments": arguments,
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
            raise Exception(f"Failed to connect to MCP server {base_url}: {str(e)}")
        except httpx.HTTPStatusError as e:
            raise Exception(f"MCP server returned error {e.response.status_code}: {e.response.text}")
        except Exception as e:
            raise Exception(f"Unexpected error calling MCP tool: {str(e)}")
    
    async def close(self):
        """Close HTTP client"""
        await self.client.aclose()

