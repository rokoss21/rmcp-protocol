"""
Simple Planner - basic planning logic for MVP
"""

import re
from typing import Dict, List, Any, Optional
from datetime import datetime

from ..storage.database import DatabaseManager
from ..models.tool import Tool
from ..models.plan import ExecutionPlan, ExecutionStep, ExecutionStrategy


class SimplePlanner:
    """Simple planner for RMCP MVP version"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
    
    async def plan(self, goal: str, context: Dict[str, Any]) -> ExecutionPlan:
        """
        Plan task execution
        
        In MVP version uses simple logic:
        1. Search tools by keywords
        2. Select best tool based on metrics
        3. Create simple plan
        """
        try:
            # Stage 1: Find suitable tools
            candidates = await self._find_candidates(goal, context)
            
            if not candidates:
                # If no suitable tools found, return empty plan
                return ExecutionPlan(
                    strategy=ExecutionStrategy.SOLO,
                    steps=[],
                    requires_approval=False
                )
            
            # Stage 2: Select best tool
            best_tool = self._select_best_tool(candidates)
            
            # Stage 3: Create plan
            plan = self._create_plan(best_tool, goal, context)
            
            return plan
            
        except Exception as e:
            print(f"Planning failed: {e}")
            # Return empty plan in case of error
            return ExecutionPlan(
                strategy=ExecutionStrategy.SOLO,
                steps=[],
                requires_approval=False
            )
    
    async def _find_candidates(self, goal: str, context: Dict[str, Any]) -> List[Tool]:
        """Find candidates for task execution"""
        candidates = []
        
        # Extract keywords from goal
        keywords = self._extract_keywords(goal)
        
        # Search by full-text index
        if keywords:
            query = " OR ".join(keywords)
            candidates.extend(self.db_manager.search_tools(query, limit=20))
        
        # Additional search by capabilities from context
        if "capability" in context:
            capability = context["capability"]
            candidates.extend(self.db_manager.get_tools_by_capability(capability))
        
        # Remove duplicates
        seen_ids = set()
        unique_candidates = []
        for tool in candidates:
            if tool.id not in seen_ids:
                seen_ids.add(tool.id)
                unique_candidates.append(tool)
        
        return unique_candidates
    
    def _extract_keywords(self, goal: str) -> List[str]:
        """Extract keywords from goal"""
        # Simple keyword extraction logic
        words = re.findall(r'\b\w+\b', goal.lower())
        
        # Filter stop words and short words
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        keywords = [word for word in words if len(word) > 2 and word not in stop_words]
        
        return keywords[:10]  # Limit number of keywords
    
    def _select_best_tool(self, candidates: List[Tool]) -> Tool:
        """Select best tool from candidates"""
        if not candidates:
            raise ValueError("No candidates provided")
        
        # Simple heuristic for MVP:
        # 1. Priority by success_rate
        # 2. Priority by low latency
        # 3. Priority by number of tags (more tags = more versatile)
        
        def score_tool(tool: Tool) -> float:
            success_score = tool.success_rate
            latency_score = max(0, 1 - (tool.p95_latency_ms / 10000))  # Normalize to 10 seconds
            versatility_score = min(1.0, len(tool.tags) / 5.0)  # Normalize to 5 tags
            
            # Weighted sum
            return 0.5 * success_score + 0.3 * latency_score + 0.2 * versatility_score
        
        # Sort by descending score
        candidates.sort(key=score_tool, reverse=True)
        
        return candidates[0]
    
    def _create_plan(self, tool: Tool, goal: str, context: Dict[str, Any]) -> ExecutionPlan:
        """Create execution plan"""
        # Check if approval is required
        requires_approval = "requires_human_approval" in tool.capabilities
        
        # Create execution step
        step = ExecutionStep(
            tool_id=tool.id,
            args=self._prepare_args(tool, goal, context),
            timeout_ms=min(int(tool.p95_latency_ms * 3), 30000),  # 3x from p95 or max 30 sec
            retry_count=1 if requires_approval else 0
        )
        
        # Create plan
        plan = ExecutionPlan(
            strategy=ExecutionStrategy.SOLO,
            steps=[step],
            requires_approval=requires_approval,
            max_execution_time_ms=int(tool.p95_latency_ms * 5)  # 5x from p95
        )
        
        return plan
    
    def _prepare_args(self, tool: Tool, goal: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare arguments for tool"""
        args = {}
        
        # Basic arguments from context
        if "query" in context:
            args["query"] = context["query"]
        elif "pattern" in context:
            args["pattern"] = context["pattern"]
        else:
            # Use goal as query
            args["query"] = goal
        
        # Additional arguments from context
        for key, value in context.items():
            if key not in ["query", "pattern", "capability"]:
                args[key] = value
        
        # Check required parameters from schema
        required_params = tool.input_schema.get("required", [])
        for param in required_params:
            if param not in args:
                # Set default value
                if param in ["path", "file_path"]:
                    args[param] = context.get("path", ".")
                elif param in ["limit", "max_results"]:
                    args[param] = 10
                else:
                    args[param] = ""
        
        return args
