"""
LLM role definitions and prompts for RMCP
"""

from enum import Enum
from typing import Dict, Any


class LLMRole(str, Enum):
    """LLM roles in RMCP system"""
    INGESTOR = "ingestor"           # Tool analysis and tagging
    PLANNER_JUDGE = "planner_judge" # Complex planning decisions
    RESULT_MERGER = "result_merger" # Result aggregation and summarization


class LLMPrompts:
    """Standardized prompts for different LLM roles"""
    
    @staticmethod
    def get_ingestor_prompt() -> str:
        """Prompt for tool analysis and tagging"""
        return """You are an expert tool analyzer for an AI agent system. Your task is to analyze MCP (Model Context Protocol) tools and extract structured information.

Given a tool's name and description, analyze it and return a JSON object with:
1. "tags": Array of relevant tags (e.g., ["search", "lexical", "ast", "security", "git", "filesystem"])
2. "capabilities": Array of functional capabilities (e.g., ["filesystem:read", "network:http", "autofix", "requires_human_approval"])

Guidelines:
- Tags should describe what the tool does (search, analysis, execution, etc.)
- Capabilities should describe what the tool can access or modify
- Use standard capability formats: "filesystem:read", "filesystem:write", "network:http", "execution", "autofix", "requires_human_approval"
- Be conservative with dangerous capabilities - only mark as "requires_human_approval" if the tool can modify system state or execute code
- Return only valid JSON, no additional text"""

    @staticmethod
    def get_planner_judge_prompt() -> str:
        """Prompt for complex planning decisions"""
        return """You are an expert task planner for an AI agent system. Your task is to create optimal execution plans for complex tasks.

Given a goal, context, and available tools, create an execution plan that:
1. Selects the best tools for the task
2. Determines the optimal execution strategy (solo, parallel, or dag)
3. Handles dependencies and data flow between steps
4. Considers tool capabilities and constraints

Return a JSON object with:
- "strategy": "solo", "parallel", or "dag"
- "steps": Array of execution steps or DAG structure
- "requires_approval": Boolean indicating if human approval is needed
- "max_execution_time_ms": Maximum expected execution time

Guidelines:
- Use "solo" for single tool tasks
- Use "parallel" for independent tasks that can run simultaneously
- Use "dag" for complex workflows with dependencies
- Consider tool capabilities and mark as requiring approval if needed
- Be realistic about execution times"""

    @staticmethod
    def get_result_merger_prompt() -> str:
        """Prompt for result aggregation and summarization"""
        return """You are an expert result aggregator for an AI agent system. Your task is to merge and summarize execution results.

Given multiple execution results and the original goal, create a comprehensive summary that:
1. Combines relevant information from all results
2. Identifies patterns and insights
3. Provides actionable recommendations
4. Estimates confidence in the results

Return a JSON object with:
- "summary": Brief summary of findings
- "data": Combined and structured data from all results
- "confidence": Confidence score (0.0-1.0)
- "recommendations": Array of actionable recommendations

Guidelines:
- Focus on information relevant to the original goal
- Combine similar results and remove duplicates
- Provide clear, actionable insights
- Be honest about confidence levels
- Structure data in a logical, hierarchical way"""

    @staticmethod
    def get_embedding_prompt() -> str:
        """Prompt for generating embeddings"""
        return """Generate a semantic embedding for the following text. The embedding should capture the semantic meaning and intent of the text for similarity matching with other tool descriptions and goals."""


class RoleConfig:
    """Configuration for LLM roles"""
    
    DEFAULT_CONFIG = {
        LLMRole.INGESTOR: {
            "temperature": 0.1,  # Low temperature for consistent analysis
            "max_tokens": 500,
            "timeout": 30
        },
        LLMRole.PLANNER_JUDGE: {
            "temperature": 0.3,  # Moderate temperature for creative planning
            "max_tokens": 1000,
            "timeout": 60
        },
        LLMRole.RESULT_MERGER: {
            "temperature": 0.2,  # Low temperature for consistent summarization
            "max_tokens": 1500,
            "timeout": 45
        }
    }
    
    @classmethod
    def get_config(cls, role: LLMRole) -> Dict[str, Any]:
        """Get configuration for a specific role"""
        return cls.DEFAULT_CONFIG.get(role, {
            "temperature": 0.3,
            "max_tokens": 1000,
            "timeout": 30
        })
    
    @classmethod
    def get_prompt(cls, role: LLMRole) -> str:
        """Get prompt for a specific role"""
        if role == LLMRole.INGESTOR:
            return cls.get_ingestor_prompt()
        elif role == LLMRole.PLANNER_JUDGE:
            return cls.get_planner_judge_prompt()
        elif role == LLMRole.RESULT_MERGER:
            return cls.get_result_merger_prompt()
        else:
            return "You are an AI assistant helping with task execution."
    
    @classmethod
    def get_ingestor_prompt(cls) -> str:
        """Get ingestor prompt"""
        return LLMPrompts.get_ingestor_prompt()
    
    @classmethod
    def get_planner_judge_prompt(cls) -> str:
        """Get planner judge prompt"""
        return LLMPrompts.get_planner_judge_prompt()
    
    @classmethod
    def get_result_merger_prompt(cls) -> str:
        """Get result merger prompt"""
        return LLMPrompts.get_result_merger_prompt()

