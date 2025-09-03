"""
Three-Stage Planner for RMCP
Implements the complete Three-Stage Decision Funnel
"""

import time
from typing import Dict, Any, Optional
from ..storage.database import DatabaseManager
from ..models.plan import ExecutionPlan
from ..llm.manager import LLMManager
from ..embeddings.manager import EmbeddingManager
from ..embeddings.store import EmbeddingStore
from .sieve import SieveStage
from .compass import CompassStage
from .judge import JudgeStage


class ThreeStagePlanner:
    """
    Three-Stage Decision Funnel for RMCP
    
    Implements the complete decision pipeline:
    1. The Sieve: Fast lexical-declarative filtering (< 1ms)
    2. The Compass: Semantic guidance and ranking (2-5ms)
    3. The Judge: Adaptive orchestration (5-200ms)
    """
    
    def __init__(
        self, 
        db_manager: DatabaseManager,
        llm_manager: Optional[LLMManager] = None,
        embedding_manager: Optional[EmbeddingManager] = None
    ):
        self.db_manager = db_manager
        self.llm_manager = llm_manager
        self.embedding_manager = embedding_manager
        
        # Initialize embedding store
        if embedding_manager:
            self.embedding_store = EmbeddingStore(embedding_manager)
        else:
            self.embedding_store = None
        
        # Initialize stages with intelligent selection
        self.sieve = SieveStage(db_manager)
        
        # Use intelligent compass if both LLM and embedding managers are available
        if llm_manager and embedding_manager:
            from .intelligent_compass import HybridCompassStage
            self.compass = HybridCompassStage(db_manager, embedding_manager, self.embedding_store, llm_manager)
            print("🧠 Initialized Intelligent Compass Stage")
        elif embedding_manager:
            from .compass import CompassStage
            self.compass = CompassStage(db_manager, embedding_manager, self.embedding_store)
            print("📍 Initialized Traditional Compass Stage")
        else:
            self.compass = None
            
        self.judge = JudgeStage(db_manager, llm_manager) if llm_manager else None
        
        # Configuration
        self.enable_semantic_ranking = embedding_manager is not None
        self.enable_llm_planning = llm_manager is not None
        self.enable_intelligent_selection = llm_manager is not None and embedding_manager is not None
    
    async def plan(self, goal: str, context: Dict[str, Any]) -> ExecutionPlan:
        """
        Create execution plan using Three-Stage Decision Funnel
        
        Args:
            goal: Task goal/description
            context: Additional context
            
        Returns:
            Execution plan
        """
        start_time = time.time()
        
        print(f"ThreeStagePlanner: Starting planning for goal: {goal[:100]}...")
        
        # Stage 1: The Sieve - Fast filtering
        sieve_start = time.time()
        candidates = await self.sieve.filter_candidates(goal, context)
        sieve_time = (time.time() - sieve_start) * 1000
        
        print(f"Stage 1 (Sieve): {len(candidates)} candidates in {sieve_time:.2f}ms")
        
        if not candidates:
            return self._create_no_tools_plan()
        
        # Stage 2: The Compass - Semantic ranking
        ranked_candidates = []
        compass_time = 0
        
        if self.enable_semantic_ranking and self.compass:
            compass_start = time.time()
            ranked_candidates = await self.compass.rank_candidates(goal, context, candidates)
            compass_time = (time.time() - compass_start) * 1000
            
            if self.enable_intelligent_selection:
                print(f"Stage 2 (Intelligent Compass): Ranked {len(ranked_candidates)} candidates in {compass_time:.2f}ms")
            else:
                print(f"Stage 2 (Compass): Ranked {len(ranked_candidates)} candidates in {compass_time:.2f}ms")
        else:
            # Fallback to simple ranking
            ranked_candidates = [(tool, 0.5) for tool in candidates]
            print("Stage 2 (Compass): Skipped - no embedding manager available")
        
        # Stage 3: The Judge - Adaptive orchestration
        judge_start = time.time()
        
        if self.enable_llm_planning and self.judge:
            plan = await self.judge.create_execution_plan(goal, context, ranked_candidates)
        else:
            # Fallback to simple planning
            plan = await self._create_simple_plan(goal, context, ranked_candidates)
        
        judge_time = (time.time() - judge_start) * 1000
        
        print(f"Stage 3 (Judge): Created {plan.strategy} plan in {judge_time:.2f}ms")
        
        # Log total performance
        total_time = (time.time() - start_time) * 1000
        print(f"ThreeStagePlanner: Total planning time: {total_time:.2f}ms")
        
        # Add performance metadata
        plan.metadata = {
            "planning_time_ms": total_time,
            "sieve_time_ms": sieve_time,
            "compass_time_ms": compass_time,
            "judge_time_ms": judge_time,
            "candidates_found": len(candidates),
            "semantic_ranking_enabled": self.enable_semantic_ranking,
            "llm_planning_enabled": self.enable_llm_planning
        }
        
        return plan
    
    async def _create_simple_plan(
        self, 
        goal: str, 
        context: Dict[str, Any], 
        ranked_candidates: list
    ) -> ExecutionPlan:
        """
        Create simple execution plan as fallback
        
        Args:
            goal: Task goal
            context: Additional context
            ranked_candidates: List of (tool, score) tuples
            
        Returns:
            Simple execution plan
        """
        from ..models.plan import ExecutionStrategy, ExecutionStep
        
        if not ranked_candidates:
            return self._create_no_tools_plan()
        
        # Select best tool
        best_tool, best_score = ranked_candidates[0]
        
        # Create single execution step
        step = ExecutionStep(
            tool_id=best_tool.id,
            args=self._extract_simple_args(goal, context, best_tool),
            timeout_ms=int(best_tool.p95_latency_ms * 2),
            retry_count=1 if best_tool.success_rate < 0.9 else 0
        )
        
        return ExecutionPlan(
            strategy=ExecutionStrategy.SOLO,
            steps=[step],
            merge_policy="first_good",
            max_execution_time_ms=int(best_tool.p95_latency_ms * 3),
            requires_approval=self._requires_approval(best_tool, context)
        )
    
    def _extract_simple_args(self, goal: str, context: Dict[str, Any], tool) -> Dict[str, Any]:
        """
        Extract simple tool arguments with improved logic
        
        Args:
            goal: Task goal
            context: Additional context
            tool: Tool to extract arguments for
            
        Returns:
            Dictionary of tool arguments
        """
        args = {}
        tool_name = tool.name.lower()
        
        # Enhanced argument extraction based on tool type and context
        
        # For echo tools
        if "echo" in tool_name:
            if "text" in context:
                args["text"] = context["text"]
            elif "message" in context:
                args["text"] = context["message"]
            else:
                # Extract text from goal (remove "echo" prefix)
                text = goal.replace("echo", "").replace("the text", "").strip()
                args["text"] = text or goal
        
        # For file creation tools
        elif "create_file" in tool_name:
            if "path" in context:
                args["path"] = context["path"]
            if "content" in context:
                args["content"] = context["content"]
            # Try to extract from goal
            if not args.get("path"):
                import re
                path_match = re.search(r'create file (\S+)', goal)
                if path_match:
                    args["path"] = path_match.group(1)
            if not args.get("content"):
                content_match = re.search(r'with content (.+)', goal)
                if content_match:
                    args["content"] = content_match.group(1)
        
        # For search/grep tools
        elif "grep" in tool_name or "search" in tool_name:
            if "pattern" in context:
                args["pattern"] = context["pattern"]
            if "text" in context:
                args["text"] = context["text"]
            # Extract pattern from goal
            if not args.get("pattern"):
                pattern_match = re.search(r'search for (\w+)', goal)
                if pattern_match:
                    args["pattern"] = pattern_match.group(1)
                    args["text"] = goal  # Use full goal as text
        
        # For word count tools
        elif "wc" in tool_name:
            if "file_path" in context:
                args["file_path"] = context["file_path"]
            elif "text" in context:
                args["text"] = context["text"]
            else:
                args["text"] = goal  # Count words in the goal itself
        
        # For cat tools
        elif "cat" in tool_name:
            if "file_path" in context:
                args["file_path"] = context["file_path"]
            # Try to extract file path from goal
            elif not args.get("file_path"):
                import re
                file_match = re.search(r'show.*content.*of (\S+)', goal)
                if file_match:
                    args["file_path"] = file_match.group(1)
        
        # Fallback: extract from context first, then goal
        if not args:
            # Try common context fields
            for field in ["query", "path", "url", "text", "pattern", "file_path"]:
                if field in context:
                    args[field] = context[field]
                    break
            
            # If still no args, use goal
            if not args:
                args["goal"] = goal
        
        return args
    
    def _requires_approval(self, tool, context: Dict[str, Any]) -> bool:
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
        dangerous_capabilities = ["execution", "filesystem:write", "network:http"]
        if any(cap in tool.capabilities for cap in dangerous_capabilities):
            return True
        
        return False
    
    def _create_no_tools_plan(self) -> ExecutionPlan:
        """
        Create plan for when no tools are available
        
        Returns:
            Empty execution plan
        """
        from ..models.plan import ExecutionStrategy
        
        return ExecutionPlan(
            strategy=ExecutionStrategy.SOLO,
            steps=[],
            merge_policy="first_good",
            requires_approval=True
        )
    
    async def update_tool_affinity(self, tool_id: str, request_text: str, success: bool) -> None:
        """
        Update tool's affinity embeddings with execution result
        
        Args:
            tool_id: Tool identifier
            request_text: Text of the request
            success: Whether the execution was successful
        """
        if self.compass:
            await self.compass.update_tool_affinity(tool_id, request_text, success)
    
    def get_planner_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive planner statistics
        
        Returns:
            Dictionary with planner statistics
        """
        stats = {
            "planner_type": "three_stage",
            "semantic_ranking_enabled": self.enable_semantic_ranking,
            "llm_planning_enabled": self.enable_llm_planning,
            "stages": {
                "sieve": self.sieve.get_filtering_stats(),
                "compass": self.compass.get_ranking_stats() if self.compass else {"enabled": False},
                "judge": self.judge.get_judge_stats() if self.judge else {"enabled": False}
            }
        }
        
        if self.embedding_manager:
            stats["embedding_cache"] = self.embedding_manager.get_cache_stats()
        
        return stats
    
    async def analyze_tool_affinity(self, tool_id: str) -> Dict[str, Any]:
        """
        Analyze tool's affinity embeddings
        
        Args:
            tool_id: Tool identifier
            
        Returns:
            Dictionary with affinity analysis
        """
        if self.compass:
            return await self.compass.analyze_tool_affinity(tool_id)
        else:
            return {
                "tool_id": tool_id,
                "embedding_count": 0,
                "has_embeddings": False,
                "stats": {},
                "error": "Compass stage not available"
            }
