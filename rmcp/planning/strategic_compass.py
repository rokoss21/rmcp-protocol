"""
Strategic Compass Stage - Enhanced semantic guidance with abstraction level matching
Implements strategic planning by considering tool types, abstraction levels, and specializations
"""

import time
import re
from typing import List, Dict, Any, Optional, Tuple
from ..storage.database import DatabaseManager
from ..models.tool import Tool
from ..embeddings.manager import EmbeddingManager
from ..embeddings.store import EmbeddingStore
from ..embeddings.similarity import CosineSimilarity
from ..planning.compass import CompassStage
from ..logging.config import get_logger


class StrategicCompassStage(CompassStage):
    """
    Enhanced Compass Stage with Strategic Planning capabilities
    
    Extends the base CompassStage with:
    - Abstraction level matching
    - Specialization-based scoring
    - Strategic vs tactical decision making
    - Agent vs atomic tool selection
    """
    
    def __init__(
        self, 
        db_manager: DatabaseManager, 
        embedding_manager: EmbeddingManager,
        embedding_store: EmbeddingStore
    ):
        super().__init__(db_manager, embedding_manager, embedding_store)
        self.logger = get_logger(__name__)
        
        # Strategic planning configuration
        self.abstraction_boost_factor = 0.3  # Boost for abstraction level matching
        self.specialization_boost_factor = 0.4  # Boost for specialization matching
        self.agent_preference_factor = 0.2  # Preference for agents over atomic tools
        
        # Keywords that indicate high-level strategic tasks
        self.strategic_keywords = {
            'audit', 'review', 'analyze', 'assess', 'evaluate', 'deploy', 'orchestrate',
            'coordinate', 'manage', 'plan', 'strategy', 'comprehensive', 'full', 'complete',
            'end-to-end', 'automated', 'intelligent', 'smart', 'advanced', 'sophisticated'
        }
        
        # Keywords that indicate low-level tactical tasks
        self.tactical_keywords = {
            'find', 'search', 'grep', 'list', 'show', 'display', 'get', 'fetch',
            'read', 'write', 'copy', 'move', 'delete', 'create', 'simple', 'basic',
            'containing', 'word', 'files'
        }
    
    async def rank_candidates(
        self, 
        goal: str, 
        context: Dict[str, Any],
        candidates: List[Tool],
        max_candidates: Optional[int] = None
    ) -> List[Tuple[Tool, float]]:
        """
        Perform strategic semantic ranking of tool candidates
        
        Enhanced with abstraction level matching and specialization scoring
        """
        start_time = time.time()
        
        if not candidates:
            return []
        
        # Determine task abstraction level
        task_abstraction_level = self._analyze_task_abstraction_level(goal)
        task_specializations = self._extract_task_specializations(goal)
        
        self.logger.info(
            f"Strategic ranking: task_level={task_abstraction_level}, "
            f"specializations={task_specializations}, candidates={len(candidates)}"
        )
        
        # Calculate strategic scores for all candidates
        scored_candidates = []
        for tool in candidates:
            # Get base affinity score from parent class
            base_affinity = await self._calculate_affinity_score(tool, goal, context)
            
            # Calculate strategic enhancements
            abstraction_boost = self._calculate_abstraction_boost(
                tool, task_abstraction_level
            )
            specialization_boost = self._calculate_specialization_boost(
                tool, task_specializations
            )
            agent_preference_boost = self._calculate_agent_preference_boost(tool)
            
            # Combine scores
            strategic_score = (
                base_affinity + 
                abstraction_boost + 
                specialization_boost + 
                agent_preference_boost
            )
            
            # Ensure score is within bounds
            strategic_score = max(0.0, min(1.0, strategic_score))
            
            scored_candidates.append((tool, strategic_score))
            
            self.logger.debug(
                f"Tool {tool.name}: base={base_affinity:.3f}, "
                f"abstraction={abstraction_boost:.3f}, "
                f"specialization={specialization_boost:.3f}, "
                f"agent_pref={agent_preference_boost:.3f}, "
                f"final={strategic_score:.3f}"
            )
        
        # Sort by strategic score (descending)
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Apply max candidates limit
        if max_candidates:
            scored_candidates = scored_candidates[:max_candidates]
        
        duration_ms = (time.time() - start_time) * 1000
        self.logger.info(
            f"Strategic ranking completed: {len(scored_candidates)} candidates, "
            f"{duration_ms:.1f}ms"
        )
        
        return scored_candidates
    
    def _analyze_task_abstraction_level(self, goal: str) -> str:
        """
        Analyze the abstraction level of a task based on goal description
        
        Returns: 'low', 'medium', or 'high'
        """
        goal_lower = goal.lower()
        
        # Count strategic vs tactical keywords
        strategic_count = sum(1 for keyword in self.strategic_keywords if keyword in goal_lower)
        tactical_count = sum(1 for keyword in self.tactical_keywords if keyword in goal_lower)
        
        # Analyze goal characteristics
        goal_length = len(goal.split())
        has_complex_verbs = any(verb in goal_lower for verb in [
            'orchestrate', 'coordinate', 'manage', 'deploy', 'audit', 'analyze'
        ])
        has_scope_indicators = any(indicator in goal_lower for indicator in [
            'comprehensive', 'full', 'complete', 'end-to-end', 'entire', 'all'
        ])
        
        # Determine abstraction level
        # Tactical keywords have higher weight for low-level tasks
        if tactical_count > 0 and tactical_count >= strategic_count:
            return 'low'
        elif strategic_count > tactical_count or has_complex_verbs or has_scope_indicators:
            if goal_length > 10 or strategic_count >= 2:
                return 'high'
            else:
                return 'medium'
        else:
            # Default to medium if no clear indicators
            return 'medium'
    
    def _extract_task_specializations(self, goal: str) -> List[str]:
        """
        Extract specializations from task goal
        
        Returns: List of specializations (e.g., ['security', 'deployment', 'testing'])
        """
        goal_lower = goal.lower()
        specializations = []
        
        # Security-related keywords
        if any(keyword in goal_lower for keyword in [
            'security', 'audit', 'vulnerability', 'compliance', 'cis', 'nist',
            'secure', 'protection', 'threat', 'risk', 'penetration'
        ]):
            specializations.append('security')
        
        # Deployment-related keywords
        if any(keyword in goal_lower for keyword in [
            'deploy', 'deployment', 'release', 'production', 'staging', 'environment',
            'blue-green', 'canary', 'rollout', 'infrastructure', 'kubernetes', 'docker'
        ]):
            specializations.append('deployment')
        
        # Testing-related keywords
        if any(keyword in goal_lower for keyword in [
            'test', 'testing', 'unit', 'integration', 'e2e', 'coverage', 'quality',
            'automation', 'ci/cd', 'pipeline', 'validation'
        ]):
            specializations.append('testing')
        
        # Code quality-related keywords
        if any(keyword in goal_lower for keyword in [
            'review', 'code', 'quality', 'style', 'lint', 'format', 'refactor',
            'architecture', 'design', 'pattern', 'best practice'
        ]):
            specializations.append('code_quality')
        
        # Documentation-related keywords
        if any(keyword in goal_lower for keyword in [
            'documentation', 'docs', 'api', 'guide', 'tutorial', 'manual',
            'readme', 'changelog', 'diagram', 'specification'
        ]):
            specializations.append('documentation')
        
        return specializations
    
    def _calculate_abstraction_boost(
        self, 
        tool: Tool, 
        task_abstraction_level: str
    ) -> float:
        """
        Calculate boost based on abstraction level matching
        
        Args:
            tool: Tool to score
            task_abstraction_level: 'low', 'medium', or 'high'
            
        Returns:
            Boost score (0.0 to 1.0)
        """
        # Get tool abstraction level
        tool_abstraction_level = getattr(tool, 'abstraction_level', 'low')
        
        # Perfect match gets full boost
        if tool_abstraction_level == task_abstraction_level:
            return self.abstraction_boost_factor
        
        # Adjacent levels get partial boost
        level_hierarchy = {'low': 0, 'medium': 1, 'high': 2}
        task_level = level_hierarchy.get(task_abstraction_level, 0)
        tool_level = level_hierarchy.get(tool_abstraction_level, 0)
        
        level_diff = abs(task_level - tool_level)
        
        if level_diff == 1:
            return self.abstraction_boost_factor * 0.5  # Adjacent levels
        else:
            return 0.0  # Too far apart
    
    def _calculate_specialization_boost(
        self, 
        tool: Tool, 
        task_specializations: List[str]
    ) -> float:
        """
        Calculate boost based on specialization matching
        
        Args:
            tool: Tool to score
            task_specializations: List of specializations from task
            
        Returns:
            Boost score (0.0 to 1.0)
        """
        if not task_specializations:
            return 0.0
        
        # Get tool specializations
        tool_specialization = getattr(tool, 'specialization', None)
        if not tool_specialization:
            return 0.0
        
        # Check for exact match
        if tool_specialization in task_specializations:
            return self.specialization_boost_factor
        
        # Check for partial match in capabilities
        tool_capabilities = [cap.lower() for cap in tool.capabilities]
        for specialization in task_specializations:
            if any(specialization in cap for cap in tool_capabilities):
                return self.specialization_boost_factor * 0.7
        
        return 0.0
    
    def _calculate_agent_preference_boost(self, tool: Tool) -> float:
        """
        Calculate boost for agents over atomic tools
        
        Args:
            tool: Tool to score
            
        Returns:
            Boost score (0.0 to 1.0)
        """
        # Only apply boost to agents
        if tool.tool_type == 'agent':
            return self.agent_preference_factor
        
        return 0.0
    
    def _fallback_affinity_score(
        self, 
        tool: Tool, 
        goal: str, 
        context: Dict[str, Any]
    ) -> float:
        """
        Enhanced fallback affinity scoring with strategic considerations
        
        Extends the base fallback scoring with strategic enhancements
        """
        # Get base score from parent class
        base_score = super()._fallback_affinity_score(tool, goal, context)
        
        # Add strategic enhancements
        task_abstraction_level = self._analyze_task_abstraction_level(goal)
        task_specializations = self._extract_task_specializations(goal)
        
        abstraction_boost = self._calculate_abstraction_boost(tool, task_abstraction_level)
        specialization_boost = self._calculate_specialization_boost(tool, task_specializations)
        agent_preference_boost = self._calculate_agent_preference_boost(tool)
        
        # Combine scores
        strategic_score = (
            base_score + 
            abstraction_boost + 
            specialization_boost + 
            agent_preference_boost
        )
        
        return max(0.0, min(1.0, strategic_score))
