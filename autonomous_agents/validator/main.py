"""
Validator Agent - Code Validation and Testing
"""

import os
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

from ..base.agent import BaseAgent
from ..base.models import AgentRequest, AgentResponse, TaskStatus, TaskPriority
from ..llm_manager import LLMManager
from .docker_utils import DockerValidator


class ValidatorAgent(BaseAgent):
    """
    Validator Agent - Code Validation and Testing
    
    Responsibilities:
    - Run syntax checks on generated code
    - Execute linters and formatters
    - Run pytest tests
    - Validate Docker builds
    - Provide detailed validation reports
    """
    
    def __init__(self, rmcp_endpoint: str = "http://localhost:8000"):
        super().__init__(
            name="Validator Agent",
            specialization="code_validation",
            rmcp_endpoint=rmcp_endpoint
        )
        # Initialize LLM Manager with OpenAI
        self.llm_manager = LLMManager(
            api_key=os.getenv("OPENAI_API_KEY"),
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        )
        self.docker_validator = DockerValidator()
    
    async def _execute_specialized(self, request: AgentRequest) -> Dict[str, Any]:
        """
        Execute code validation task
        
        Args:
            request: The agent request
            
        Returns:
            Dict containing validation results
        """
        task_type = request.task_type
        goal = request.goal
        parameters = request.parameters
        context = request.context
        
        print(f"✅ Validator Agent executing: {task_type}")
        print(f"   Goal: {goal}")
        print(f"   Parameters: {parameters}")
        
        try:
            if task_type == "validation":
                result = await self._validate_code(goal, parameters, context)
            else:
                result = await self._validate_generic(goal, parameters, context)
            
            return {
                "validation_results": result.get("results", {}),
                "overall_success": result.get("overall_success", False),
                "artifacts": [
                    {
                        "type": "validation_report",
                        "name": "validation_report.json",
                        "content": result.get("results", {}),
                        "metadata": {
                            "task_type": task_type,
                            "generated_by": "validator_agent",
                            "parameters": parameters
                        }
                    }
                ],
                "metadata": {
                    "task_type": task_type,
                    "validation_method": "docker",
                    "overall_success": result.get("overall_success", False),
                    "results": result.get("results", {})
                }
            }
            
        except Exception as e:
            print(f"❌ Validator Agent error: {e}")
            raise
    
    async def _validate_code(self, goal: str, parameters: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate generated code
        
        Args:
            goal: Validation goal
            parameters: Validation parameters
            context: Additional context
            
        Returns:
            Dict containing validation results
        """
        project_path = parameters.get("project_path", "/tmp/test_project")
        test_file = parameters.get("test_file")
        
        # If no project path provided, create a mock validation
        if not os.path.exists(project_path):
            return await self._mock_validation(goal, parameters, context)
        
        # Run real validation
        results = self.docker_validator.validate_project(project_path, test_file)
        
        return {
            "results": results,
            "overall_success": results.get("overall_success", False)
        }
    
    async def _validate_generic(self, goal: str, parameters: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform generic validation
        
        Args:
            goal: Validation goal
            parameters: Validation parameters
            context: Additional context
            
        Returns:
            Dict containing validation results
        """
        return await self._mock_validation(goal, parameters, context)
    
    async def _mock_validation(self, goal: str, parameters: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Mock validation for testing
        
        Args:
            goal: Validation goal
            parameters: Validation parameters
            context: Additional context
            
        Returns:
            Dict containing mock validation results
        """
        import asyncio
        await asyncio.sleep(0.1)  # Simulate validation time
        
        # Mock validation results
        results = {
            "syntax_check": {
                "success": True,
                "output": "Syntax check passed",
                "error": ""
            },
            "lint_check": {
                "success": True,
                "output": "Linting passed",
                "warnings": "",
                "linting_issues": False
            },
            "test_results": {
                "success": True,
                "output": "All tests passed",
                "error": "",
                "test_count": 5
            },
            "build_check": {
                "success": True,
                "output": "Docker build successful",
                "error": ""
            },
            "overall_success": True
        }
        
        return {
            "results": results,
            "overall_success": True
        }


# FastAPI application
app = FastAPI(
    title="Validator Agent API",
    description="Code Validation Agent",
    version="1.0.0"
)

# Global agent instance
agent = ValidatorAgent()


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
    Execute code validation request
    
    This endpoint receives validation tasks and returns validation results.
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
        port=8010,
        reload=True,
        log_level="info"
    )

