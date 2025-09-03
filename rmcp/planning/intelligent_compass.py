"""
Intelligent Compass Stage - LLM-powered semantic tool ranking
Replaces keyword-based ranking with deep semantic understanding
"""

import time
from typing import List, Dict, Any, Optional, Tuple
from ..storage.database import DatabaseManager
from ..models.tool import Tool
from ..embeddings.manager import EmbeddingManager
from ..embeddings.store import EmbeddingStore
from ..llm.manager import LLMManager
from ..logging.config import get_logger
from .intelligent_selector import IntelligentToolSelector


class IntelligentCompassStage:
    """
    Enhanced Compass Stage with LLM-powered intelligent tool selection
    
    Replaces simple keyword matching and cosine similarity with:
    - Deep semantic understanding of tasks
    - Intelligent analysis of tool capabilities  
    - Natural language processing of user intent
    - Context-aware tool ranking
    """
    
    def __init__(
        self, 
        db_manager: DatabaseManager, 
        embedding_manager: EmbeddingManager,
        embedding_store: EmbeddingStore,
        llm_manager: LLMManager
    ):
        self.db_manager = db_manager
        self.embedding_manager = embedding_manager
        self.embedding_store = embedding_store
        self.llm_manager = llm_manager
        self.logger = get_logger(__name__)
        
        # Initialize intelligent selector
        self.intelligent_selector = IntelligentToolSelector(llm_manager)
        
        # Fallback to traditional compass if LLM fails
        from .compass import CompassStage
        self.fallback_compass = CompassStage(db_manager, embedding_manager, embedding_store)
        
        # Configuration
        self.use_intelligent_selection = True
        self.max_candidates = 10
    
    async def rank_candidates(
        self, 
        goal: str, 
        context: Dict[str, Any],
        candidates: List[Tool],
        max_candidates: Optional[int] = None
    ) -> List[Tuple[Tool, float]]:
        """
        Perform intelligent semantic ranking of tool candidates
        
        Args:
            goal: Task goal/description
            context: Additional context
            candidates: List of tool candidates from Sieve stage
            max_candidates: Maximum number of candidates to return
            
        Returns:
            List of (tool, confidence_score) tuples, ranked by relevance
        """
        start_time = time.time()
        max_results = max_candidates or self.max_candidates
        
        self.logger.info(f"Intelligent ranking {len(candidates)} candidates for: {goal[:50]}...")
        
        try:
            if self.use_intelligent_selection and self.llm_manager:
                # Use LLM-powered intelligent selection
                ranked_candidates = await self._intelligent_ranking(
                    goal, context, candidates, max_results
                )
            else:
                # Fallback to traditional ranking
                ranked_candidates = await self.fallback_compass.rank_candidates(
                    goal, context, candidates, max_results
                )
                
        except Exception as e:
            self.logger.error(f"Intelligent ranking failed, using fallback: {e}")
            # Fallback to traditional compass
            ranked_candidates = await self.fallback_compass.rank_candidates(
                goal, context, candidates, max_results
            )
        
        elapsed_ms = (time.time() - start_time) * 1000
        self.logger.info(f"Intelligent ranking completed in {elapsed_ms:.2f}ms")
        
        return ranked_candidates[:max_results]
    
    async def _intelligent_ranking(
        self,
        goal: str,
        context: Dict[str, Any], 
        candidates: List[Tool],
        max_results: int
    ) -> List[Tuple[Tool, float]]:
        """
        Perform intelligent ranking using LLM analysis
        """
        # Use intelligent selector to get ranked tools with reasoning
        selected_tools = await self.intelligent_selector.select_best_tools(
            task_description=goal,
            context=context,
            available_tools=candidates,
            max_tools=max_results
        )
        
        # Convert format: (tool, confidence, reasoning) -> (tool, confidence)
        ranked_candidates = [(tool, confidence) for tool, confidence, _ in selected_tools]
        
        # Log selection reasoning for debugging
        if selected_tools:
            self.logger.info("Intelligent selection reasoning:")
            for tool, confidence, reasoning in selected_tools[:3]:
                self.logger.info(f"  {tool.name} ({confidence:.2f}): {reasoning[:100]}...")
        
        return ranked_candidates
    
    async def get_selection_explanation(
        self,
        goal: str,
        context: Dict[str, Any],
        candidates: List[Tool]
    ) -> str:
        """
        Get detailed explanation of tool selection process
        """
        try:
            selected_tools = await self.intelligent_selector.select_best_tools(
                task_description=goal,
                context=context,
                available_tools=candidates,
                max_tools=5
            )
            
            return await self.intelligent_selector.explain_selection(goal, selected_tools)
            
        except Exception as e:
            return f"Selection explanation failed: {e}"
    
    async def extract_tool_arguments(
        self,
        goal: str,
        context: Dict[str, Any],
        selected_tool: Tool
    ) -> Dict[str, Any]:
        """
        Extract arguments for selected tool using intelligent analysis
        """
        if self.intelligent_selector:
            try:
                return await self.intelligent_selector.extract_arguments_for_tool(
                    goal, context, selected_tool
                )
            except Exception as e:
                self.logger.error(f"Intelligent argument extraction failed: {e}")
        
        # Fallback to simple argument extraction
        return self._simple_argument_fallback(goal, context, selected_tool)
    
    def _simple_argument_fallback(self, goal: str, context: Dict[str, Any], tool: Tool) -> Dict[str, Any]:
        """Simple fallback for argument extraction"""
        # Extract from context first
        args = {}
        for key in ["path", "content", "text", "pattern", "query", "url"]:
            if key in context:
                args[key] = context[key]
        
        # If no args from context, use goal
        if not args:
            args["goal"] = goal
            
        return args
    
    def enable_intelligent_selection(self, enabled: bool):
        """Enable or disable intelligent selection (for testing/fallback)"""
        self.use_intelligent_selection = enabled
        self.logger.info(f"Intelligent selection {'enabled' if enabled else 'disabled'}")
    
    async def analyze_selection_quality(
        self,
        goal: str,
        selected_tools: List[Tuple[Tool, float]],
        execution_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze the quality of tool selection based on execution results
        This can be used to improve the selection process over time
        """
        analysis = {
            "task": goal,
            "selected_tools": [tool.name for tool, _ in selected_tools],
            "selection_accuracy": 0.0,
            "execution_success_rate": 0.0,
            "feedback": []
        }
        
        if execution_results:
            successful_executions = sum(1 for result in execution_results if result.get("success", False))
            analysis["execution_success_rate"] = successful_executions / len(execution_results)
            
            # Provide feedback for improvement
            if analysis["execution_success_rate"] < 0.5:
                analysis["feedback"].append("Low success rate - consider improving tool selection criteria")
            
            if analysis["execution_success_rate"] > 0.8:
                analysis["feedback"].append("High success rate - selection strategy is effective")
        
        return analysis


class HybridCompassStage:
    """
    Hybrid approach that combines intelligent selection with traditional ranking
    Uses LLM for complex tasks and falls back to traditional methods for simple ones
    """
    
    def __init__(
        self,
        db_manager: DatabaseManager,
        embedding_manager: EmbeddingManager, 
        embedding_store: EmbeddingStore,
        llm_manager: Optional[LLMManager] = None
    ):
        self.db_manager = db_manager
        self.llm_manager = llm_manager
        self.logger = get_logger(__name__)
        
        # Initialize both approaches
        if llm_manager:
            self.intelligent_compass = IntelligentCompassStage(
                db_manager, embedding_manager, embedding_store, llm_manager
            )
        else:
            self.intelligent_compass = None
            
        from .compass import CompassStage
        self.traditional_compass = CompassStage(db_manager, embedding_manager, embedding_store)
        
        # Thresholds for selecting approach
        self.complexity_threshold = 0.6  # Use intelligent for complex tasks
        self.candidate_threshold = 5     # Use intelligent when many candidates
    
    async def rank_candidates(
        self,
        goal: str,
        context: Dict[str, Any],
        candidates: List[Tool],
        max_candidates: Optional[int] = None
    ) -> List[Tuple[Tool, float]]:
        """
        Rank candidates using hybrid approach
        """
        # Determine complexity
        complexity = self._estimate_task_complexity(goal, context)
        use_intelligent = (
            self.intelligent_compass is not None and
            (complexity > self.complexity_threshold or len(candidates) > self.candidate_threshold)
        )
        
        self.logger.info(f"Hybrid selection: complexity={complexity:.2f}, "
                        f"candidates={len(candidates)}, using_intelligent={use_intelligent}")
        
        if use_intelligent:
            return await self.intelligent_compass.rank_candidates(
                goal, context, candidates, max_candidates
            )
        else:
            return await self.traditional_compass.rank_candidates(
                goal, context, candidates, max_candidates
            )
    
    def _estimate_task_complexity(self, goal: str, context: Dict[str, Any]) -> float:
        """
        Estimate task complexity to decide which ranking approach to use
        """
        complexity = 0.0
        
        # Length factor
        complexity += min(len(goal) / 200, 0.3)
        
        # Context richness
        complexity += min(len(context) / 10, 0.2)
        
        # Keyword complexity indicators
        complex_words = ['analyze', 'process', 'transform', 'generate', 'complex', 'multi', 'workflow']
        complexity += sum(0.1 for word in complex_words if word in goal.lower())
        
        # Multi-step indicators
        if any(indicator in goal.lower() for indicator in ['then', 'after', 'first', 'next', 'finally']):
            complexity += 0.3
            
        return min(complexity, 1.0)
