"""
Stage 3: The Judge - Adaptive orchestration
Implements final plan formation using LLM for complex decision making
"""

import time
from typing import List, Dict, Any, Optional, Tuple
from ..storage.database import DatabaseManager
from ..models.tool import Tool
from ..models.plan import ExecutionPlan, ExecutionStrategy, ExecutionStep
from ..llm.manager import LLMManager
from ..observability.metrics import metrics
from ..logging.config import get_logger, log_planning


class JudgeStage:
    """
    Stage 3: The Judge - Adaptive orchestration (5-200ms)
    
    Performs final plan formation using:
    - Short Path: Simple single-tool execution
    - Deep Path: Complex multi-tool orchestration with LLM
    - Adaptive decision making based on complexity
    """
    
    def __init__(self, db_manager: DatabaseManager, llm_manager: LLMManager):
        self.db_manager = db_manager
        self.llm_manager = llm_manager
        self.complexity_threshold = 0.4  # Threshold for choosing Deep Path (lowered for multi-step tasks)
        self.max_simple_tools = 3  # Maximum tools for Short Path
        self.logger = get_logger(__name__)
    
    async def create_execution_plan(
        self, 
        goal: str, 
        context: Dict[str, Any],
        ranked_candidates: List[Tuple[Tool, float]]
    ) -> ExecutionPlan:
        """
        Create final execution plan using adaptive orchestration
        
        Args:
            goal: Task goal/description
            context: Additional context
            ranked_candidates: List of (tool, score) tuples from Compass stage
            
        Returns:
            Final execution plan
        """
        start_time = time.time()
        
        if not ranked_candidates:
            return self._create_empty_plan()
        
        # Step 1: Analyze complexity
        complexity_score = await self._analyze_complexity(goal, context, ranked_candidates)
        
        # Step 2: Choose execution strategy
        candidate_count = len(ranked_candidates)
        use_short_path = complexity_score < self.complexity_threshold and candidate_count <= self.max_simple_tools
        
        self.logger.info(f"Strategy decision: complexity={complexity_score:.3f} < {self.complexity_threshold} = {complexity_score < self.complexity_threshold}")
        self.logger.info(f"Strategy decision: candidates={candidate_count} <= {self.max_simple_tools} = {candidate_count <= self.max_simple_tools}")
        self.logger.info(f"Strategy decision: use_short_path = {use_short_path}")
        
        if use_short_path:
            # Short Path: Simple execution
            self.logger.info("Using Short Path (simple execution)")
            plan = await self._create_short_path_plan(goal, context, ranked_candidates)
        else:
            # Deep Path: Complex orchestration with LLM
            self.logger.info("Using Deep Path (LLM orchestration)")
            plan = await self._create_deep_path_plan(goal, context, ranked_candidates)
        
        # Step 3: Validate and optimize plan
        plan = await self._validate_plan(plan, goal, context)
        
        # Log performance
        elapsed_ms = (time.time() - start_time) * 1000
        tenant_id = context.get("tenant_id", "default")
        strategy_str = plan.strategy.value if hasattr(plan.strategy, 'value') else str(plan.strategy)
        
        # Structured logging
        log_planning(
            strategy=strategy_str,
            tool_count=len(plan.steps),
            requires_approval=plan.requires_approval,
            duration_ms=elapsed_ms,
            request_id=context.get("request_id"),
            user_id=context.get("user_id"),
            tenant_id=tenant_id,
            complexity_score=complexity_score
        )
        
        # Record metrics
        metrics.record_plan_created(strategy_str, tenant_id, plan.requires_approval)
        metrics.record_planning_duration(strategy_str, tenant_id, elapsed_ms / 1000.0)
        
        return plan
    
    async def _analyze_complexity(
        self, 
        goal: str, 
        context: Dict[str, Any], 
        candidates: List[Tuple[Tool, float]]
    ) -> float:
        """
        Analyze task complexity to determine execution strategy
        
        Args:
            goal: Task goal
            context: Additional context
            candidates: List of candidate tools
            
        Returns:
            Complexity score (0.0 to 1.0)
        """
        complexity_factors = []
        
        # Factor 1: Number of candidate tools
        tool_count_factor = min(1.0, len(candidates) / 10.0)  # Normalize to 10 tools max
        complexity_factors.append(tool_count_factor)
        
        # Factor 2: Goal length and complexity
        goal_complexity = min(1.0, len(goal.split()) / 50.0)  # Normalize to 50 words max
        complexity_factors.append(goal_complexity)
        
        # Factor 3: Context complexity
        context_keys = len(context.keys())
        context_complexity = min(1.0, context_keys / 20.0)  # Normalize to 20 keys max
        complexity_factors.append(context_complexity)
        
        # Factor 4: Tool diversity (different capabilities)
        if candidates:
            all_capabilities = set()
            for tool, _ in candidates:
                all_capabilities.update(tool.capabilities)
            diversity_factor = min(1.0, len(all_capabilities) / 10.0)  # Normalize to 10 capabilities max
            complexity_factors.append(diversity_factor)
        
        # Factor 5: Multi-step task indicators
        multistep_indicators = [
            'затем', 'потом', 'после', 'далее', 'также', 'и', 'then', 'after', 'next', 
            'also', 'additionally', 'furthermore', 'first', 'second', 'finally',
            'проанализируй', 'analyze', 'найди', 'find', 'создай', 'create', 'сделай'
        ]
        goal_lower = goal.lower()
        multistep_count = sum(1 for indicator in multistep_indicators if indicator in goal_lower)
        multistep_factor = min(1.0, multistep_count / 5.0)  # Normalize to 5 indicators max
        complexity_factors.append(multistep_factor)
        
        # Factor 6: Complex task keywords
        complex_keywords = [
            'подробный', 'comprehensive', 'анализ', 'analysis', 'отчет', 'report',
            'сводка', 'summary', 'исследование', 'research', 'многоступенчатый', 'multi-step',
            'workflow', 'процесс', 'process', 'pipeline'
        ]
        complex_count = sum(1 for keyword in complex_keywords if keyword in goal_lower)
        complex_factor = min(1.0, complex_count / 3.0)  # Normalize to 3 keywords max
        complexity_factors.append(complex_factor)
        
        # Factor 7: Multi-agent task detection (removed hardcoded keywords)
        # LLM will handle intelligent agent selection, not keyword matching
        multi_agent_factor = 0.0
        
        # Only boost complexity if explicitly requested complex workflow
        if context.get("requires_multiple_agents", False) or context.get("complexity", 0.0) > 0.5:
            multi_agent_factor = 0.8
            self.logger.info(f"Multi-agent task detected from context, setting complexity to {multi_agent_factor}")
        
        complexity_factors.append(multi_agent_factor)
        
        # Factor 8: Explicit complexity hint from context
        explicit_complexity = context.get("complexity", 0.0)
        if context.get("requires_multiple_steps", False):
            explicit_complexity = max(explicit_complexity, 0.8)
        complexity_factors.append(explicit_complexity)
        
        # Calculate weighted average (updated weights for new factors)
        weights = [0.15, 0.12, 0.12, 0.12, 0.12, 0.12, 0.2, 0.05]  # Tool count, goal, context, diversity, multistep, complex, web, explicit
        complexity_score = sum(factor * weight for factor, weight in zip(complexity_factors, weights))
        
        # Debug logging
        factor_names = ["tool_count", "goal_length", "context", "diversity", "multistep", "complex_keywords", "multi_agent", "explicit"]
        self.logger.info(f"Complexity analysis: {dict(zip(factor_names, complexity_factors))}")
        self.logger.info(f"Final complexity_score: {complexity_score:.3f} (threshold: {self.complexity_threshold})")
        
        return min(1.0, complexity_score)
    
    async def _create_short_path_plan(
        self, 
        goal: str, 
        context: Dict[str, Any], 
        candidates: List[Tuple[Tool, float]]
    ) -> ExecutionPlan:
        """
        Create simple execution plan (Short Path)
        
        Args:
            goal: Task goal
            context: Additional context
            candidates: List of candidate tools
            
        Returns:
            Simple execution plan
        """
        # Select best tool
        best_tool, best_score = candidates[0]
        
        # Create single execution step with enhanced argument extraction
        self.logger.info(f"Extracting args for {best_tool.name}")
        self.logger.info(f"Tool input_schema: {best_tool.input_schema}")
        args = self._extract_tool_args(goal, context, best_tool)
        self.logger.info(f"Extracted args: {args}")
        
        step = ExecutionStep(
            tool_id=best_tool.id,
            args=args,
            timeout_ms=int(best_tool.p95_latency_ms * 2),  # 2x P95 latency as timeout
            retry_count=1 if best_tool.success_rate < 0.9 else 0
        )
        
        return ExecutionPlan(
            strategy=ExecutionStrategy.SOLO,
            steps=[step],
            merge_policy="first_good",
            max_execution_time_ms=int(best_tool.p95_latency_ms * 3),
            requires_approval=self._requires_approval(best_tool, context)
        )
    
    async def _create_deep_path_plan(
        self, 
        goal: str, 
        context: Dict[str, Any], 
        candidates: List[Tuple[Tool, float]]
    ) -> ExecutionPlan:
        """
        Create complex execution plan using LLM (Deep Path)
        
        Args:
            goal: Task goal
            context: Additional context
            candidates: List of candidate tools
            
        Returns:
            Complex execution plan
        """
        # Prepare candidate data for LLM
        candidate_data = []
        for tool, score in candidates[:10]:  # Limit to top 10 candidates
            candidate_data.append({
                "id": tool.id,
                "name": tool.name,
                "description": tool.description,
                "capabilities": tool.capabilities,
                "tags": tool.tags,
                "success_rate": tool.success_rate,
                "p95_latency_ms": tool.p95_latency_ms,
                "cost_hint": tool.cost_hint,
                "affinity_score": score
            })
        
        # Use LLM to create execution plan
        try:
            llm_plan = await self.llm_manager.plan_execution(goal, context, candidate_data)
            
            # Log LLM plan for debugging
            self.logger.info(f"LLM created plan: {llm_plan}")
            
            # Convert LLM plan to ExecutionPlan
            plan = self._convert_llm_plan_to_execution_plan(llm_plan, candidates)
            
            # Log final execution plan
            self.logger.info(f"Final ExecutionPlan: strategy={plan.strategy}, steps={len(plan.steps)}")
            for i, step in enumerate(plan.steps):
                self.logger.info(f"  Step {i+1}: {step.tool_id} with args {step.args}")
            
            return plan
            
        except Exception as e:
            print(f"LLM planning failed, falling back to simple plan: {e}")
            return await self._create_short_path_plan(goal, context, candidates)
    
    def _convert_llm_plan_to_execution_plan(
        self, 
        llm_plan: Dict[str, Any], 
        candidates: List[Tuple[Tool, float]]
    ) -> ExecutionPlan:
        """
        Convert LLM-generated plan to ExecutionPlan object
        
        Args:
            llm_plan: Plan generated by LLM
            candidates: List of candidate tools for reference
            
        Returns:
            ExecutionPlan object
        """
        # Create tool lookup
        tool_lookup = {tool.id: tool for tool, _ in candidates}
        
        # Determine strategy
        strategy_str = llm_plan.get("strategy", "solo")
        try:
            strategy = ExecutionStrategy(strategy_str)
        except ValueError:
            strategy = ExecutionStrategy.SOLO
        
        # Create execution steps
        steps_data = llm_plan.get("steps", [])
        steps = []
        
        for step_data in steps_data:
            # LLM can use either "tool_id" or "tool" field
            tool_name = step_data.get("tool_id") or step_data.get("tool")
            if not tool_name:
                continue  # Skip steps without tool reference
            
            # Find tool by name in candidates
            tool = None
            tool_id = None
            for candidate_tool, _ in candidates:
                if candidate_tool.name == tool_name or candidate_tool.id == tool_name:
                    tool = candidate_tool
                    tool_id = candidate_tool.id
                    break
            
            if not tool:
                self.logger.warning(f"Tool {tool_name} not found in candidates")
                continue  # Skip invalid tool references
            
            # Extract arguments from step data (support multiple formats)
            args = step_data.get("args", {})
            if not args and "input" in step_data:
                args = step_data["input"]
            if not args and "parameters" in step_data:
                args = step_data["parameters"]
            
            # Map LLM field names to tool parameter names
            if args:
                mapped_args = {}
                for key, value in args.items():
                    # Map common LLM field names to tool parameters
                    if key == "file_name" or key == "file_path":
                        mapped_args["path"] = value
                    elif key == "content":
                        mapped_args["content"] = value
                    else:
                        mapped_args[key] = value
                args = mapped_args
            
            # If no args specified, try to extract from action description
            if not args and "action" in step_data:
                action = step_data["action"]
                
                # For create_file tools, extract path and content
                if "create_file" in tool_name:
                    if "output" in step_data:
                        args["path"] = step_data["output"]
                    args["content"] = action
                    
                # For append_to_file tools
                elif "append_to_file" in tool_name:
                    if "input" in step_data:
                        args["path"] = step_data["input"]
                    args["content"] = action
            
            step = ExecutionStep(
                tool_id=tool_id,
                args=args,
                dependencies=step_data.get("dependencies", []),
                outputs=step_data.get("outputs", []),
                timeout_ms=step_data.get("timeout_ms", int(tool.p95_latency_ms * 2)),
                retry_count=step_data.get("retry_count", 1 if tool.success_rate < 0.9 else 0)
            )
            steps.append(step)
        
        # If no valid steps, create a simple plan
        if not steps:
            best_tool, _ = candidates[0]
            step = ExecutionStep(
                tool_id=best_tool.id,
                args={},
                timeout_ms=int(best_tool.p95_latency_ms * 2)
            )
            steps = [step]
            strategy = ExecutionStrategy.SOLO
        
        # Check if any step requires approval
        requires_approval = llm_plan.get("requires_approval", False)
        if not requires_approval:
            for step in steps:
                tool = tool_lookup.get(step.tool_id)
                if tool and self._requires_approval(tool, {}):
                    requires_approval = True
                    break
        
        return ExecutionPlan(
            strategy=strategy,
            steps=steps,
            merge_policy=llm_plan.get("merge_policy", "first_good"),
            max_execution_time_ms=llm_plan.get("max_execution_time_ms"),
            requires_approval=requires_approval
        )
    
    def _extract_tool_args(self, goal: str, context: Dict[str, Any], tool: Tool) -> Dict[str, Any]:
        """
        Extract tool arguments from goal and context with enhanced logic
        
        Args:
            goal: Task goal
            context: Additional context
            tool: Tool to extract arguments for
            
        Returns:
            Dictionary of tool arguments
        """
        args = {}
        tool_name = tool.name.lower()
        goal_lower = goal.lower()
        
        # First, extract from context
        context_args = ["path", "content", "text", "pattern", "query", "url", "file_path"]
        for arg in context_args:
            if arg in context and context[arg]:
                args[arg] = context[arg]
        
        # Enhanced extraction based on tool type and goal
        import re
        
        # File creation tools
        if "create_file" in tool_name:
            if "path" not in args:
                # Try to extract file path from goal
                path_patterns = [
                    r'(?:create|file)\s+(\S+\.\w+)',  # "create report.txt"
                    r'(?:document|file)\s+(\S+)',     # "document specs"
                    r'(\w+\.\w+)',                    # any filename.ext
                ]
                for pattern in path_patterns:
                    match = re.search(pattern, goal_lower)
                    if match:
                        args["path"] = match.group(1)
                        break
                else:
                    # Generate logical filename
                    if "document" in goal_lower:
                        args["path"] = "document.txt"
                    elif "spec" in goal_lower or "technical" in goal_lower:
                        args["path"] = "technical_specifications.txt"
                    elif "report" in goal_lower:
                        args["path"] = "report.txt"
                    else:
                        args["path"] = "file.txt"
            
            if "content" not in args:
                # Extract content or generate based on goal
                content_patterns = [
                    r'(?:with|content|containing)\s+(.+)',
                    r'(?:spec|specification).*?([^.]+)',
                ]
                for pattern in content_patterns:
                    match = re.search(pattern, goal)
                    if match:
                        args["content"] = match.group(1).strip()
                        break
                else:
                    # Generate content based on goal
                    if "technical" in goal_lower and "spec" in goal_lower:
                        args["content"] = "Technical Specifications\n\n1. Overview\n2. Requirements\n3. Implementation Details"
                    elif "document" in goal_lower:
                        args["content"] = f"Document: {goal}\n\nGenerated content based on request."
                    else:
                        args["content"] = goal
        
        # Search/grep tools
        elif "grep" in tool_name or "search" in tool_name:
            if "pattern" not in args and "query" not in args:
                # Extract search pattern
                pattern_match = re.search(r'(?:search|find|grep).*?(?:for\s+)?(\w+)', goal_lower)
                if pattern_match:
                    args["pattern"] = pattern_match.group(1)
                else:
                    args["pattern"] = goal
            
            if "text" not in args:
                args["text"] = goal
        
        # Echo tools
        elif "echo" in tool_name:
            if "text" not in args:
                # Extract text to echo
                echo_match = re.search(r'echo\s+(.+)', goal_lower)
                if echo_match:
                    args["text"] = echo_match.group(1)
                else:
                    args["text"] = goal
        
        # Word count tools
        elif "wc" in tool_name:
            if "text" not in args:
                args["text"] = goal
        
        # List tools
        elif "ls" in tool_name or "list" in tool_name:
            if "path" not in args:
                path_match = re.search(r'(?:list|ls)\s+(\S+)', goal_lower)
                if path_match:
                    args["path"] = path_match.group(1)
                else:
                    args["path"] = "."  # current directory
        
        # Agent tools (backend, architect, tester, devops, etc.)
        elif any(agent in tool_name for agent in ["backend.", "architect.", "tester.", "devops.", "orchestrator.", "validator."]):
            # Always include the goal for agent tools
            if "goal" not in args:
                args["goal"] = goal
            
            # Extract file_path from context or infer from goal
            if "file_path" not in args:
                # Try to extract file path from goal
                file_patterns = [
                    r'(\w+\.\w+)',  # any filename.ext
                    r'файл\s+(\w+\.\w+)',  # "файл calculator.py"
                    r'создай\s+(\w+\.\w+)',  # "создай calculator.py"
                ]
                for pattern in file_patterns:
                    match = re.search(pattern, goal_lower)
                    if match:
                        args["file_path"] = match.group(1)
                        break
                else:
                    # Generate filename based on tool and goal
                    if "python" in goal_lower or "script" in goal_lower:
                        if "калькулятор" in goal_lower or "calculator" in goal_lower:
                            args["file_path"] = "calculator.py"
                        elif "игра" in goal_lower or "game" in goal_lower:
                            args["file_path"] = "game.py"
                        else:
                            args["file_path"] = "script.py"
                    elif "html" in goal_lower or "web" in goal_lower or "крестики" in goal_lower:
                        args["file_path"] = "index.html"
                    elif "dockerfile" in goal_lower:
                        args["file_path"] = "Dockerfile"
                    elif "docker-compose" in goal_lower:
                        args["file_path"] = "docker-compose.yml"
                    else:
                        args["file_path"] = "generated_file.txt"
        
        # Fallback: use tool's input schema if available
        if not args and tool.input_schema:
            schema_props = tool.input_schema.get("properties", {})
            required_props = tool.input_schema.get("required", [])
            
            # Try to map goal to the first required parameter
            if required_props:
                first_param = required_props[0]
                args[first_param] = goal
            elif schema_props:
                # Use first property
                first_param = list(schema_props.keys())[0]
                args[first_param] = goal
        
        # Final fallback
        if not args:
            args["goal"] = goal
        
        return args
    
    def _requires_approval(self, tool: Tool, context: Dict[str, Any]) -> bool:
        """
        Determine if plan requires human approval
        
        Args:
            tool: Tool to check
            context: Additional context
            
        Returns:
            True if approval is required
        """
        # Check tool capabilities
        if "requires_human_approval" in tool.capabilities:
            return True
        
        # Check context requirements
        if context.get("requires_approval", False):
            return True
        
        # Check for dangerous capabilities
        dangerous_capabilities = [
            "execution", 
            "filesystem:write", 
            "network:http",
            "shell.execute",
            "terraform.apply",
            "kubectl.delete",
            "docker.run",
            "system.admin"
        ]
        if any(cap in tool.capabilities for cap in dangerous_capabilities):
            return True
        
        # Check tool name patterns for dangerous operations
        dangerous_patterns = [
            "shell", "exec", "run", "delete", "remove", "destroy",
            "terraform", "kubectl", "docker", "system"
        ]
        tool_name_lower = tool.name.lower()
        if any(pattern in tool_name_lower for pattern in dangerous_patterns):
            return True
        
        return False
    
    async def _validate_plan(self, plan: ExecutionPlan, goal: str, context: Dict[str, Any]) -> ExecutionPlan:
        """
        Validate and optimize execution plan
        
        Args:
            plan: Plan to validate
            goal: Task goal
            context: Additional context
            
        Returns:
            Validated and optimized plan
        """
        # Validate that all referenced tools exist
        valid_steps = []
        for step in plan.steps:
            tool = self.db_manager.get_tool(step.tool_id)
            if tool:
                valid_steps.append(step)
            else:
                print(f"Warning: Tool {step.tool_id} not found, skipping step")
        
        # Update plan with valid steps
        plan.steps = valid_steps
        
        # Ensure we have at least one step
        if not plan.steps:
            return self._create_empty_plan()
        
        # Optimize timeouts based on tool metrics
        for step in plan.steps:
            tool = self.db_manager.get_tool(step.tool_id)
            if tool and not step.timeout_ms:
                step.timeout_ms = int(tool.p95_latency_ms * 2)
        
        return plan
    
    def _create_empty_plan(self) -> ExecutionPlan:
        """
        Create empty execution plan for error cases
        
        Returns:
            Empty execution plan
        """
        return ExecutionPlan(
            strategy=ExecutionStrategy.SOLO,
            steps=[],
            merge_policy="first_good",
            requires_approval=True
        )
    
    def get_judge_stats(self) -> Dict[str, Any]:
        """
        Get statistics about judge performance
        
        Returns:
            Dictionary with judge statistics
        """
        return {
            "complexity_threshold": self.complexity_threshold,
            "max_simple_tools": self.max_simple_tools,
            "stage": "judge",
            "description": "Adaptive orchestration"
        }
