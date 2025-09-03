"""
Strategic Judge Stage - Enhanced plan formation with strategic decision making
Implements strategic planning by considering agents vs atomic tools and abstraction levels
"""

import time
from typing import List, Dict, Any, Optional, Tuple
from ..storage.database import DatabaseManager
from ..models.tool import Tool
from ..models.plan import ExecutionPlan, ExecutionStrategy, ExecutionStep
from ..llm.manager import LLMManager
from ..observability.metrics import metrics
from ..logging.config import get_logger, log_planning
from .judge import JudgeStage


class StrategicJudgeStage(JudgeStage):
    """
    Enhanced Judge Stage with Strategic Planning capabilities
    
    Extends the base JudgeStage with:
    - Strategic vs tactical decision making
    - Agent delegation vs atomic tool orchestration
    - Abstraction level-aware planning
    - Specialization-based routing
    """
    
    def __init__(self, db_manager: DatabaseManager, llm_manager: LLMManager):
        super().__init__(db_manager, llm_manager)
        self.logger = get_logger(__name__)
        
        # Strategic planning configuration
        self.agent_delegation_threshold = 0.8  # Threshold for choosing agent delegation
        self.strategic_complexity_threshold = 0.6  # Threshold for strategic planning
        
        # Keywords that indicate strategic tasks suitable for agent delegation
        self.strategic_delegation_keywords = {
            'comprehensive', 'full', 'complete', 'end-to-end', 'automated', 'intelligent',
            'audit', 'review', 'analyze', 'assess', 'evaluate', 'deploy', 'orchestrate',
            'coordinate', 'manage', 'plan', 'strategy', 'sophisticated', 'advanced'
        }
    
    async def create_execution_plan(
        self, 
        goal: str, 
        context: Dict[str, Any],
        ranked_candidates: List[Tuple[Tool, float]]
    ) -> ExecutionPlan:
        """
        Create strategic execution plan using enhanced decision making
        
        Enhanced with strategic planning capabilities
        """
        start_time = time.time()
        
        if not ranked_candidates:
            return ExecutionPlan(
                strategy=ExecutionStrategy.SOLO,
                steps=[],
                estimated_duration_ms=0,
                requires_approval=False,
                complexity_score=0.0
            )
        
        # Analyze strategic context
        strategic_context = self._analyze_strategic_context(goal, ranked_candidates)
        
        self.logger.info(
            f"Strategic planning: goal='{goal[:50]}...', "
            f"context={strategic_context}, candidates={len(ranked_candidates)}"
        )
        
        # Choose strategic planning approach
        if self._should_use_agent_delegation(goal, ranked_candidates, strategic_context):
            plan = await self._create_agent_delegation_plan(
                goal, context, ranked_candidates, strategic_context
            )
        elif self._should_use_strategic_orchestration(goal, ranked_candidates, strategic_context):
            plan = await self._create_strategic_orchestration_plan(
                goal, context, ranked_candidates, strategic_context
            )
        else:
            # Fall back to standard planning
            plan = await super().create_execution_plan(goal, context, ranked_candidates)
        
        # Log strategic planning decision
        duration_ms = (time.time() - start_time) * 1000
        log_planning(
            plan.strategy.value if hasattr(plan.strategy, 'value') else str(plan.strategy),
            tool_count=len(plan.steps),
            requires_approval=plan.requires_approval,
            duration_ms=duration_ms,
            request_id=context.get('request_id'),
            user_id=context.get('user_id'),
            tenant_id=context.get('tenant_id'),
            goal=goal,
            complexity_score=plan.metadata.get('complexity_score', 0.0)
        )
        
        return plan
    
    def _analyze_strategic_context(
        self, 
        goal: str, 
        ranked_candidates: List[Tuple[Tool, float]]
    ) -> Dict[str, Any]:
        """
        Analyze strategic context of the task and available candidates
        
        Returns:
            Dictionary with strategic context information
        """
        # Analyze task characteristics
        goal_lower = goal.lower()
        task_length = len(goal.split())
        
        # Count strategic keywords
        strategic_keyword_count = sum(
            1 for keyword in self.strategic_delegation_keywords 
            if keyword in goal_lower
        )
        
        # Analyze candidates
        agents = [tool for tool, score in ranked_candidates if tool.tool_type == 'agent']
        atomic_tools = [tool for tool, score in ranked_candidates if tool.tool_type == 'atomic']
        
        # Get top candidates
        top_candidates = ranked_candidates[:3]
        top_agent = next((tool for tool, score in top_candidates if tool.tool_type == 'agent'), None)
        top_atomic = next((tool for tool, score in top_candidates if tool.tool_type == 'atomic'), None)
        
        # Calculate strategic indicators
        has_high_abstraction_agents = any(
            getattr(tool, 'abstraction_level', 'low') == 'high' 
            for tool, score in ranked_candidates
        )
        
        has_specialized_agents = any(
            getattr(tool, 'specialization', None) is not None
            for tool, score in ranked_candidates
        )
        
        return {
            'task_length': task_length,
            'strategic_keyword_count': strategic_keyword_count,
            'agent_count': len(agents),
            'atomic_tool_count': len(atomic_tools),
            'top_agent': top_agent,
            'top_atomic': top_atomic,
            'has_high_abstraction_agents': has_high_abstraction_agents,
            'has_specialized_agents': has_specialized_agents,
            'top_agent_score': ranked_candidates[0][1] if ranked_candidates else 0.0
        }
    
    def _should_use_agent_delegation(
        self, 
        goal: str, 
        ranked_candidates: List[Tuple[Tool, float]],
        strategic_context: Dict[str, Any]
    ) -> bool:
        """
        Determine if task should be delegated to a single agent
        
        Returns:
            True if agent delegation is recommended
        """
        # Must have agents available
        if strategic_context['agent_count'] == 0:
            return False
        
        # Must have a high-scoring agent
        if strategic_context['top_agent_score'] < self.agent_delegation_threshold:
            return False
        
        # Check strategic indicators
        strategic_indicators = (
            strategic_context['strategic_keyword_count'] >= 2 or
            strategic_context['task_length'] > 8 or
            strategic_context['has_high_abstraction_agents'] or
            strategic_context['has_specialized_agents']
        )
        
        return strategic_indicators
    
    def _should_use_strategic_orchestration(
        self, 
        goal: str, 
        ranked_candidates: List[Tuple[Tool, float]],
        strategic_context: Dict[str, Any]
    ) -> bool:
        """
        Determine if task should use strategic orchestration (multiple agents/tools)
        
        Returns:
            True if strategic orchestration is recommended
        """
        # Must have multiple candidates
        if len(ranked_candidates) < 2:
            return False
        
        # Check if we have both agents and atomic tools
        has_agents = strategic_context['agent_count'] > 0
        has_atomic_tools = strategic_context['atomic_tool_count'] > 0
        
        # Strategic orchestration is suitable for complex tasks with mixed resources
        return (
            has_agents and has_atomic_tools and
            strategic_context['strategic_keyword_count'] >= 1
        )
    
    async def _create_agent_delegation_plan(
        self, 
        goal: str, 
        context: Dict[str, Any],
        ranked_candidates: List[Tuple[Tool, float]],
        strategic_context: Dict[str, Any]
    ) -> ExecutionPlan:
        """
        Create execution plan that delegates to a single agent
        
        Returns:
            Execution plan with agent delegation
        """
        # Find the best agent
        best_agent = None
        best_score = 0.0
        
        for tool, score in ranked_candidates:
            if tool.tool_type == 'agent' and score > best_score:
                best_agent = tool
                best_score = score
        
        if not best_agent:
            # Fallback to standard planning
            return await super().create_execution_plan(goal, context, ranked_candidates)
        
        # Create single-step plan with agent
        step = ExecutionStep(
            tool_id=best_agent.id,
            args={
                'goal': goal,
                'context': context,
                'specialization': getattr(best_agent, 'specialization', 'general'),
                'abstraction_level': getattr(best_agent, 'abstraction_level', 'medium')
            },
            timeout_ms=getattr(best_agent, 'avg_execution_time_ms', 30000)
        )
        
        # Calculate complexity based on agent capabilities
        complexity_score = self._calculate_agent_complexity(best_agent, goal)
        
        return ExecutionPlan(
            strategy=ExecutionStrategy.SOLO,
            steps=[step],
            max_execution_time_ms=step.timeout_ms,
            requires_approval=self._requires_approval_for_agent(best_agent, goal),
            metadata={'complexity_score': complexity_score}
        )
    
    async def _create_strategic_orchestration_plan(
        self, 
        goal: str, 
        context: Dict[str, Any],
        ranked_candidates: List[Tuple[Tool, float]],
        strategic_context: Dict[str, Any]
    ) -> ExecutionPlan:
        """
        Create execution plan using strategic orchestration (LLM-based)
        
        Returns:
            Execution plan with strategic orchestration
        """
        # Use LLM to create strategic orchestration plan
        plan = await self._llm_strategic_planning(goal, context, ranked_candidates)
        
        return plan
    
    async def _llm_strategic_planning(
        self, 
        goal: str, 
        context: Dict[str, Any],
        ranked_candidates: List[Tuple[Tool, float]]
    ) -> ExecutionPlan:
        """
        Use LLM for strategic planning with enhanced context
        
        Returns:
            Execution plan created by LLM
        """
        # Prepare enhanced context for LLM
        enhanced_context = self._prepare_strategic_llm_context(goal, ranked_candidates)
        
        # Create strategic planning prompt
        prompt = self._create_strategic_planning_prompt(goal, enhanced_context)
        
        try:
            # Get LLM response
            response = await self.llm_manager.generate_response(
                prompt=prompt,
                max_tokens=1000,
                temperature=0.3
            )
            
            # Parse LLM response into execution plan
            plan = self._parse_strategic_llm_response(response, ranked_candidates)
            
            return plan
            
        except Exception as e:
            self.logger.error(f"Strategic LLM planning failed: {e}")
            # Fallback to standard planning
            return await super().create_execution_plan(goal, context, ranked_candidates)
    
    def _prepare_strategic_llm_context(
        self, 
        goal: str, 
        ranked_candidates: List[Tuple[Tool, float]]
    ) -> Dict[str, Any]:
        """
        Prepare enhanced context for strategic LLM planning
        
        Returns:
            Enhanced context dictionary
        """
        # Separate agents and atomic tools
        agents = []
        atomic_tools = []
        
        for tool, score in ranked_candidates:
            tool_info = {
                'id': tool.id,
                'name': tool.name,
                'description': tool.description,
                'score': score,
                'capabilities': tool.capabilities,
                'tags': tool.tags
            }
            
            if tool.tool_type == 'agent':
                tool_info.update({
                    'specialization': getattr(tool, 'specialization', 'general'),
                    'abstraction_level': getattr(tool, 'abstraction_level', 'medium'),
                    'max_complexity': getattr(tool, 'max_complexity', 1.0)
                })
                agents.append(tool_info)
            else:
                atomic_tools.append(tool_info)
        
        return {
            'agents': agents,
            'atomic_tools': atomic_tools,
            'total_candidates': len(ranked_candidates),
            'top_score': ranked_candidates[0][1] if ranked_candidates else 0.0
        }
    
    def _create_strategic_planning_prompt(
        self, 
        goal: str, 
        enhanced_context: Dict[str, Any]
    ) -> str:
        """
        Create strategic planning prompt for LLM
        
        Returns:
            Strategic planning prompt
        """
        agents = enhanced_context['agents']
        atomic_tools = enhanced_context['atomic_tools']
        
        prompt = f"""
You are a strategic AI planner for RMCP (Routing & Memory Control Plane). Your task is to create an optimal execution plan for the given goal.

GOAL: {goal}

AVAILABLE RESOURCES:
"""
        
        if agents:
            prompt += "\nAGENTS (High-level specialists):\n"
            for agent in agents:
                prompt += f"- {agent['name']} (Score: {agent['score']:.3f})\n"
                prompt += f"  Specialization: {agent['specialization']}\n"
                prompt += f"  Abstraction Level: {agent['abstraction_level']}\n"
                prompt += f"  Description: {agent['description']}\n"
                prompt += f"  Capabilities: {', '.join(agent['capabilities'])}\n\n"
        
        if atomic_tools:
            prompt += "\nATOMIC TOOLS (Low-level utilities):\n"
            for tool in atomic_tools[:5]:  # Limit to top 5 atomic tools
                prompt += f"- {tool['name']} (Score: {tool['score']:.3f})\n"
                prompt += f"  Description: {tool['description']}\n"
                prompt += f"  Capabilities: {', '.join(tool['capabilities'])}\n\n"
        
        prompt += """
STRATEGIC PLANNING RULES:
1. PREFER AGENT DELEGATION: If a specialized agent can handle the entire task, delegate to that agent
2. USE STRATEGIC ORCHESTRATION: For complex tasks, combine agents and atomic tools intelligently
3. CONSIDER ABSTRACTION LEVELS: Match task complexity with appropriate abstraction levels
4. MINIMIZE COORDINATION: Prefer fewer, more capable tools over many simple tools
5. ENSURE COMPLETENESS: The plan should fully address the goal

RESPONSE FORMAT:
Return a JSON object with:
{
  "strategy": "SOLO" | "SEQUENTIAL" | "PARALLEL",
  "steps": [
    {
      "tool_id": "tool_id",
      "tool_name": "tool_name",
      "parameters": {...},
      "estimated_duration_ms": 5000,
      "requires_approval": false
    }
  ],
  "estimated_duration_ms": 10000,
  "requires_approval": false,
  "complexity_score": 0.8
}

Create the optimal strategic plan:
"""
        
        return prompt
    
    def _parse_strategic_llm_response(
        self, 
        response: str, 
        ranked_candidates: List[Tuple[Tool, float]]
    ) -> ExecutionPlan:
        """
        Parse LLM response into execution plan
        
        Returns:
            Parsed execution plan
        """
        try:
            import json
            
            # Extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON found in response")
            
            json_str = response[json_start:json_end]
            plan_data = json.loads(json_str)
            
            # Create execution steps
            steps = []
            for step_data in plan_data.get('steps', []):
                step = ExecutionStep(
                    tool_id=step_data['tool_id'],
                    args=step_data.get('parameters', {}),
                    timeout_ms=step_data.get('estimated_duration_ms', 5000)
                )
                steps.append(step)
            
            # Create execution plan
            plan = ExecutionPlan(
                strategy=ExecutionStrategy(plan_data.get('strategy', 'solo')),
                steps=steps,
                max_execution_time_ms=plan_data.get('estimated_duration_ms', 10000),
                requires_approval=plan_data.get('requires_approval', False),
                metadata={'complexity_score': plan_data.get('complexity_score', 0.5)}
            )
            
            return plan
            
        except Exception as e:
            self.logger.error(f"Failed to parse strategic LLM response: {e}")
            # Fallback to simple plan with top candidate
            if ranked_candidates:
                top_tool, top_score = ranked_candidates[0]
                step = ExecutionStep(
                    tool_id=top_tool.id,
                    args={'goal': response},  # Use response as goal
                    timeout_ms=5000
                )
                
                return ExecutionPlan(
                    strategy=ExecutionStrategy.SOLO,
                    steps=[step],
                    max_execution_time_ms=5000,
                    requires_approval=False,
                    metadata={'complexity_score': top_score}
                )
            else:
                return ExecutionPlan(
                    strategy=ExecutionStrategy.SOLO,
                    steps=[],
                    max_execution_time_ms=0,
                    requires_approval=False,
                    metadata={'complexity_score': 0.0}
                )
    
    def _requires_approval_for_agent(self, agent: Tool, goal: str) -> bool:
        """
        Determine if agent execution requires approval
        
        Returns:
            True if approval is required
        """
        # High-level agents with sensitive operations require approval
        abstraction_level = getattr(agent, 'abstraction_level', 'low')
        specialization = getattr(agent, 'specialization', 'general')
        
        sensitive_specializations = {'security', 'deployment', 'testing'}
        sensitive_keywords = {'production', 'delete', 'remove', 'destroy', 'critical'}
        
        goal_lower = goal.lower()
        
        return (
            abstraction_level == 'high' or
            specialization in sensitive_specializations or
            any(keyword in goal_lower for keyword in sensitive_keywords)
        )
    
    def _calculate_agent_complexity(self, agent: Tool, goal: str) -> float:
        """
        Calculate complexity score for agent execution
        
        Returns:
            Complexity score (0.0 to 1.0)
        """
        base_complexity = getattr(agent, 'max_complexity', 0.5)
        
        # Adjust based on goal characteristics
        goal_lower = goal.lower()
        if any(keyword in goal_lower for keyword in ['comprehensive', 'full', 'complete']):
            base_complexity += 0.2
        
        if any(keyword in goal_lower for keyword in ['production', 'critical', 'important']):
            base_complexity += 0.1
        
        return min(1.0, base_complexity)
