"""
Strategic Three-Stage Planner - Enhanced planning with strategic decision making
Combines StrategicCompassStage and StrategicJudgeStage for meta-orchestration
"""

import time
from typing import List, Dict, Any, Optional
from ..storage.database import DatabaseManager
from ..models.tool import Tool
from ..models.plan import ExecutionPlan, ExecutionStrategy
from ..embeddings.manager import EmbeddingManager
from ..embeddings.store import EmbeddingStore
from ..llm.manager import LLMManager
from ..planning.sieve import SieveStage
from ..planning.strategic_compass import StrategicCompassStage
from ..planning.strategic_judge import StrategicJudgeStage
from ..logging.config import get_logger, log_planning


class StrategicThreeStagePlanner:
    """
    Strategic Three-Stage Planner with Meta-Orchestration capabilities
    
    Enhanced version of ThreeStagePlanner with:
    - Strategic abstraction level matching
    - Agent vs atomic tool selection
    - Specialization-based routing
    - Strategic decision making
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
        
        # Initialize strategic stages
        self.sieve = SieveStage(db_manager)
        self.compass = StrategicCompassStage(db_manager, embedding_manager, embedding_store)
        self.judge = StrategicJudgeStage(db_manager, llm_manager)
        
        # Strategic planning configuration
        self.enable_strategic_planning = True
        self.agent_registry_enabled = True
    
    async def create_plan(
        self, 
        goal: str, 
        context: Dict[str, Any] = None
    ) -> ExecutionPlan:
        """
        Create strategic execution plan using enhanced three-stage process
        
        Args:
            goal: Task goal/description
            context: Additional context (request_id, user_id, tenant_id, etc.)
            
        Returns:
            Strategic execution plan
        """
        if context is None:
            context = {}
        
        start_time = time.time()
        
        self.logger.info(
            f"Strategic planning started: goal='{goal[:50]}...', "
            f"context={context}"
        )
        
        try:
            # Stage 1: Sieve - Filter candidates
            sieve_start = time.time()
            candidates = await self.sieve.filter_candidates(goal, context)
            sieve_duration = (time.time() - sieve_start) * 1000
            
            self.logger.info(
                f"Sieve stage: {len(candidates)} candidates, {sieve_duration:.1f}ms"
            )
            
            if not candidates:
                self.logger.warning("No candidates found in sieve stage")
                return ExecutionPlan(
                    strategy=ExecutionStrategy.SOLO,
                    steps=[],
                    max_execution_time_ms=0,
                    requires_approval=False,
                    metadata={'complexity_score': 0.0}
                )
            
            # Stage 2: Strategic Compass - Semantic ranking with strategic enhancements
            compass_start = time.time()
            ranked_candidates = await self.compass.rank_candidates(
                goal, context, candidates
            )
            compass_duration = (time.time() - compass_start) * 1000
            
            self.logger.info(
                f"Strategic Compass stage: {len(ranked_candidates)} ranked candidates, "
                f"{compass_duration:.1f}ms"
            )
            
            # Log strategic insights
            self._log_strategic_insights(goal, ranked_candidates)
            
            # Stage 3: Strategic Judge - Strategic plan formation
            judge_start = time.time()
            plan = await self.judge.create_execution_plan(
                goal, context, ranked_candidates
            )
            judge_duration = (time.time() - judge_start) * 1000
            
            total_duration = (time.time() - start_time) * 1000
            
            self.logger.info(
                f"Strategic Judge stage: strategy={plan.strategy}, "
                f"steps={len(plan.steps)}, {judge_duration:.1f}ms"
            )
            
            # Log final strategic plan
            log_planning(
                "strategic_plan_completed",
                goal=goal,
                tool_count=len(plan.steps),
                duration_ms=total_duration,
                complexity_score=plan.metadata.get('complexity_score', 0.0),
                requires_approval=plan.requires_approval,
                request_id=context.get('request_id'),
                user_id=context.get('user_id'),
                tenant_id=context.get('tenant_id')
            )
            
            return plan
            
        except Exception as e:
            self.logger.error(f"Strategic planning failed: {e}")
            
            # Fallback to simple plan
            return ExecutionPlan(
                strategy=ExecutionStrategy.SOLO,
                steps=[],
                max_execution_time_ms=0,
                requires_approval=True,  # Require approval for failed planning
                metadata={'complexity_score': 1.0}
            )
    
    def _log_strategic_insights(
        self, 
        goal: str, 
        ranked_candidates: List[tuple]
    ) -> None:
        """
        Log strategic insights about the planning process
        
        Args:
            goal: Task goal
            ranked_candidates: Ranked list of (tool, score) tuples
        """
        if not ranked_candidates:
            return
        
        # Analyze candidate distribution
        agents = [tool for tool, score in ranked_candidates if tool.tool_type == 'agent']
        atomic_tools = [tool for tool, score in ranked_candidates if tool.tool_type == 'atomic']
        
        # Get top candidates
        top_3 = ranked_candidates[:3]
        top_agent = next((tool for tool, score in top_3 if tool.tool_type == 'agent'), None)
        top_atomic = next((tool for tool, score in top_3 if tool.tool_type == 'atomic'), None)
        
        # Log strategic insights
        insights = {
            'total_candidates': len(ranked_candidates),
            'agent_count': len(agents),
            'atomic_tool_count': len(atomic_tools),
            'top_score': ranked_candidates[0][1] if ranked_candidates else 0.0,
            'top_agent': top_agent.name if top_agent else None,
            'top_atomic': top_atomic.name if top_atomic else None,
            'strategic_balance': len(agents) / len(ranked_candidates) if ranked_candidates else 0.0
        }
        
        self.logger.info(f"Strategic insights: {insights}")
        
        # Log individual top candidates
        for i, (tool, score) in enumerate(top_3):
            tool_type = tool.tool_type
            specialization = getattr(tool, 'specialization', 'general')
            abstraction_level = getattr(tool, 'abstraction_level', 'low')
            
            self.logger.info(
                f"Top {i+1}: {tool.name} ({tool_type}, {specialization}, "
                f"{abstraction_level}) - Score: {score:.3f}"
            )
    
    async def get_planning_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about strategic planning performance
        
        Returns:
            Dictionary with planning statistics
        """
        # Get database statistics
        total_tools = len(self.db_manager.get_all_tools())
        agents = self.db_manager.get_agents()
        atomic_tools = self.db_manager.get_atomic_tools()
        
        # Calculate strategic metrics
        agent_ratio = len(agents) / total_tools if total_tools > 0 else 0.0
        
        # Get agent specializations
        specializations = {}
        for agent in agents:
            spec = getattr(agent, 'specialization', 'general')
            specializations[spec] = specializations.get(spec, 0) + 1
        
        # Get abstraction level distribution
        abstraction_levels = {'low': 0, 'medium': 0, 'high': 0}
        for agent in agents:
            level = getattr(agent, 'abstraction_level', 'low')
            abstraction_levels[level] = abstraction_levels.get(level, 0) + 1
        
        return {
            'total_tools': total_tools,
            'agents': len(agents),
            'atomic_tools': len(atomic_tools),
            'agent_ratio': agent_ratio,
            'specializations': specializations,
            'abstraction_levels': abstraction_levels,
            'strategic_planning_enabled': self.enable_strategic_planning,
            'agent_registry_enabled': self.agent_registry_enabled
        }
    
    def enable_strategic_mode(self, enabled: bool = True) -> None:
        """
        Enable or disable strategic planning mode
        
        Args:
            enabled: Whether to enable strategic planning
        """
        self.enable_strategic_planning = enabled
        self.logger.info(f"Strategic planning mode: {'enabled' if enabled else 'disabled'}")
    
    def configure_strategic_parameters(
        self, 
        abstraction_boost_factor: float = None,
        specialization_boost_factor: float = None,
        agent_preference_factor: float = None,
        agent_delegation_threshold: float = None
    ) -> None:
        """
        Configure strategic planning parameters
        
        Args:
            abstraction_boost_factor: Boost for abstraction level matching
            specialization_boost_factor: Boost for specialization matching
            agent_preference_factor: Preference for agents over atomic tools
            agent_delegation_threshold: Threshold for agent delegation
        """
        if abstraction_boost_factor is not None:
            self.compass.abstraction_boost_factor = abstraction_boost_factor
        
        if specialization_boost_factor is not None:
            self.compass.specialization_boost_factor = specialization_boost_factor
        
        if agent_preference_factor is not None:
            self.compass.agent_preference_factor = agent_preference_factor
        
        if agent_delegation_threshold is not None:
            self.judge.agent_delegation_threshold = agent_delegation_threshold
        
        self.logger.info("Strategic planning parameters updated")
