"""
Architect Agent - System Design and Planning
"""

import json
import asyncio
import os
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

from ..base.agent import BaseAgent
from ..base.models import AgentRequest, AgentResponse, TaskStatus, TaskPriority, Task, DevelopmentPlan
from ..llm_manager import LLMManager
from .prompts import SYSTEM_DESIGN_PROMPT, UTILITY_RESEARCH_PROMPT, ARCHITECTURE_DESIGN_PROMPT


class ArchitectAgent(BaseAgent):
    """
    Architect Agent - System Design and Planning
    
    Responsibilities:
    - Analyze high-level goals
    - Research target utilities
    - Design system architecture
    - Create detailed development plans
    - Generate task DAGs for execution
    """
    
    def __init__(self, rmcp_endpoint: str = "http://localhost:8000"):
        super().__init__(
            name="Architect Agent",
            specialization="system_design",
            rmcp_endpoint=rmcp_endpoint
        )
        # Initialize LLM Manager with OpenAI
        self.llm_manager = LLMManager(
            api_key=os.getenv("OPENAI_API_KEY"),
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        )
    
    async def _execute_specialized(self, request: AgentRequest) -> Dict[str, Any]:
        """
        Execute architectural planning
        
        Args:
            request: The agent request
            
        Returns:
            Dict containing development plan
        """
        goal = request.goal
        context = request.context
        parameters = request.parameters
        
        # Step 1: Research the target utility
        research_result = await self._research_utility(goal, context)
        
        # Step 2: Design system architecture
        architecture = await self._design_architecture(research_result, goal)
        
        # Step 3: Create development plan
        development_plan = await self._create_development_plan(
            goal, research_result, architecture, parameters
        )
        
        return {
            "development_plan": development_plan,
            "research_result": research_result,
            "architecture": architecture,
            "artifacts": [
                {
                    "type": "development_plan",
                    "content": development_plan,
                    "format": "json"
                }
            ]
        }
    
    async def _research_utility(self, goal: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Research the target utility to understand its functionality
        
        Args:
            goal: The project goal
            context: Additional context
            
        Returns:
            Dict containing research results
        """
        # Extract utility name from goal
        utility_name = self._extract_utility_name(goal)
        
        # Use RMCP to research the utility
        research_goal = f"Research the '{utility_name}' command-line utility. Find out what it does, its arguments, input/output format, and usage examples."
        
        rmcp_result = await self.delegate_to_rmcp(research_goal, {
            "research_type": "utility_analysis",
            "utility_name": utility_name,
            "context": context
        })
        
        if rmcp_result.get("success"):
            # Parse RMCP result
            research_data = rmcp_result.get("result", {})
            return {
                "utility_name": utility_name,
                "description": research_data.get("summary", f"Command-line utility: {utility_name}"),
                "command_line_args": self._extract_arguments(research_data),
                "input_format": "text/string",
                "output_format": "text/string",
                "dependencies": [utility_name],
                "examples": self._extract_examples(research_data),
                "research_source": "rmcp_delegation"
            }
        else:
            # Fallback to mock research
            return self._create_mock_research(utility_name)
    
    async def _design_architecture(self, research: Dict[str, Any], goal: str) -> Dict[str, Any]:
        """
        Design system architecture based on research
        
        Args:
            research: Research results
            goal: Project goal
            
        Returns:
            Dict containing architecture design
        """
        utility_name = research.get("utility_name", "unknown")
        
        # Create standard MCP server architecture
        architecture = {
            "project_structure": {
                "main.py": "FastAPI application with MCP tool endpoints",
                "models.py": "Pydantic models for input/output validation",
                "requirements.txt": "Python dependencies",
                "Dockerfile": "Container configuration",
                "README.md": "Project documentation"
            },
            "api_endpoints": [
                {
                    "path": "/tools",
                    "method": "GET",
                    "description": "List available MCP tools"
                },
                {
                    "path": "/execute",
                    "method": "POST",
                    "description": "Execute MCP tool"
                },
                {
                    "path": "/health",
                    "method": "GET",
                    "description": "Health check endpoint"
                }
            ],
            "tool_definitions": [
                {
                    "name": f"{utility_name}.execute",
                    "description": f"Execute {utility_name} command with specified parameters",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "text": {
                                "type": "string",
                                "description": "Input text for the command"
                            },
                            "options": {
                                "type": "object",
                                "description": "Additional command options",
                                "additionalProperties": True
                            }
                        },
                        "required": ["text"]
                    },
                    "output_schema": {
                        "type": "object",
                        "properties": {
                            "result": {
                                "type": "string",
                                "description": "Command output"
                            },
                            "success": {
                                "type": "boolean",
                                "description": "Whether command succeeded"
                            },
                            "error": {
                                "type": "string",
                                "description": "Error message if failed"
                            }
                        }
                    }
                }
            ],
            "dependencies": [
                "fastapi>=0.104.0",
                "uvicorn>=0.24.0",
                "pydantic>=2.0.0"
            ]
        }
        
        return architecture
    
    async def _create_development_plan(
        self, 
        goal: str, 
        research: Dict[str, Any], 
        architecture: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> DevelopmentPlan:
        """
        Create detailed development plan with task DAG
        
        Args:
            goal: Project goal
            research: Research results
            architecture: Architecture design
            parameters: Additional parameters
            
        Returns:
            DevelopmentPlan: Complete development plan
        """
        project_id = f"mcp-{research.get('utility_name', 'unknown')}-{int(asyncio.get_event_loop().time())}"
        
        # Create tasks based on architecture
        tasks = []
        
        # Task 1: Create main.py
        tasks.append(Task(
            id="task_1",
            name="Create main.py",
            description="Create FastAPI application with MCP tool endpoints",
            task_type="code_generation",
            agent_type="backend",
            parameters={
                "file_path": "main.py",
                "template": "fastapi_mcp_server",
                "utility_name": research.get("utility_name"),
                "tool_definitions": architecture["tool_definitions"]
            },
            outputs=["main.py"]
        ))
        
        # Task 2: Create models.py
        tasks.append(Task(
            id="task_2",
            name="Create models.py",
            description="Create Pydantic models for input/output validation",
            task_type="code_generation",
            agent_type="backend",
            parameters={
                "file_path": "models.py",
                "template": "pydantic_models",
                "input_schema": architecture["tool_definitions"][0]["input_schema"],
                "output_schema": architecture["tool_definitions"][0]["output_schema"]
            },
            dependencies=["task_1"],
            outputs=["models.py"]
        ))
        
        # Task 3: Create requirements.txt
        tasks.append(Task(
            id="task_3",
            name="Create requirements.txt",
            description="Create Python dependencies file",
            task_type="file_creation",
            agent_type="backend",
            parameters={
                "file_path": "requirements.txt",
                "dependencies": architecture["dependencies"]
            },
            outputs=["requirements.txt"]
        ))
        
        # Task 4: Create Dockerfile
        tasks.append(Task(
            id="task_4",
            name="Create Dockerfile",
            description="Create container configuration",
            task_type="file_creation",
            agent_type="devops",
            parameters={
                "file_path": "Dockerfile",
                "base_image": "python:3.11-slim",
                "utility_name": research.get("utility_name")
            },
            dependencies=["task_1", "task_2", "task_3"],
            outputs=["Dockerfile"]
        ))
        
        # Task 5: Create tests
        tasks.append(Task(
            id="task_5",
            name="Create tests",
            description="Create pytest tests for the MCP server",
            task_type="test_generation",
            agent_type="tester",
            parameters={
                "file_path": "test_main.py",
                "target_file": "main.py",
                "test_cases": ["health_check", "tool_execution", "error_handling"]
            },
            dependencies=["task_1", "task_2"],
            outputs=["test_main.py"]
        ))
        
        # Task 6: Validate and build
        tasks.append(Task(
            id="task_6",
            name="Validate and build",
            description="Run tests and build Docker image",
            task_type="validation",
            agent_type="validator",
            parameters={
                "test_file": "test_main.py",
                "dockerfile": "Dockerfile",
                "image_name": f"mcp-{research.get('utility_name', 'unknown')}"
            },
            dependencies=["task_4", "task_5"],
            outputs=["docker_image", "test_results"]
        ))
        
        # Task 7: Deploy
        tasks.append(Task(
            id="task_7",
            name="Deploy MCP server",
            description="Deploy the MCP server container",
            task_type="deployment",
            agent_type="devops",
            parameters={
                "image_name": f"mcp-{research.get('utility_name', 'unknown')}",
                "container_name": f"mcp-{research.get('utility_name', 'unknown')}-server",
                "port": 8000
            },
            dependencies=["task_6"],
            outputs=["deployed_container"]
        ))
        
        # Create dependencies mapping
        dependencies = {
            "task_1": [],
            "task_2": ["task_1"],
            "task_3": [],
            "task_4": ["task_1", "task_2", "task_3"],
            "task_5": ["task_1", "task_2"],
            "task_6": ["task_4", "task_5"],
            "task_7": ["task_6"]
        }
        
        return DevelopmentPlan(
            project_id=project_id,
            goal=goal,
            analysis=research,
            architecture=architecture,
            tasks=tasks,
            dependencies=dependencies,
            estimated_duration_ms=600000  # 10 minutes
        )
    
    def _extract_utility_name(self, goal: str) -> str:
        """Extract utility name from goal"""
        goal_lower = goal.lower()
        
        # Common patterns
        if "cowsay" in goal_lower:
            return "cowsay"
        elif "figlet" in goal_lower:
            return "figlet"
        elif "fortune" in goal_lower:
            return "fortune"
        elif "lolcat" in goal_lower:
            return "lolcat"
        else:
            # Try to extract from "Create MCP for X" pattern
            import re
            match = re.search(r'for\s+([a-zA-Z0-9_-]+)', goal_lower)
            if match:
                return match.group(1)
            return "unknown"
    
    def _extract_arguments(self, research_data: Dict[str, Any]) -> List[str]:
        """Extract command-line arguments from research data"""
        # This would parse the research data to extract arguments
        # For now, return common arguments
        return ["text", "options"]
    
    def _extract_examples(self, research_data: Dict[str, Any]) -> List[str]:
        """Extract usage examples from research data"""
        # This would parse the research data to extract examples
        # For now, return mock examples
        return [
            "Basic usage example",
            "Advanced usage example"
        ]
    
    def _create_mock_research(self, utility_name: str) -> Dict[str, Any]:
        """Create mock research data when RMCP delegation fails"""
        return {
            "utility_name": utility_name,
            "description": f"Command-line utility: {utility_name}",
            "command_line_args": ["text", "options"],
            "input_format": "text/string",
            "output_format": "text/string",
            "dependencies": [utility_name],
            "examples": [
                f"{utility_name} 'Hello World'",
                f"{utility_name} --help"
            ],
            "research_source": "mock_fallback"
        }


# FastAPI application
app = FastAPI(
    title="Architect Agent API",
    description="System Design and Planning Agent",
    version="1.0.0"
)

# Global agent instance
agent = ArchitectAgent()


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "agent_name": agent.name,
        "specialization": agent.specialization,
        "execution_count": agent.execution_count
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "agent": agent.name}


@app.post("/execute", response_model=AgentResponse)
async def execute(request: AgentRequest):
    """
    Execute architect request
    
    This endpoint receives high-level goals and returns detailed development plans.
    """
    try:
        response = await agent.execute(request)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats():
    """Get agent statistics"""
    return agent.get_stats()


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8006,
        reload=True,
        log_level="info"
    )

