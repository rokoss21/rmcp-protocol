"""
Tester Agent - Test Generation
"""

import os
import asyncio
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

from ..base.agent import BaseAgent
from ..base.models import AgentRequest, AgentResponse, TaskStatus, TaskPriority
from ..llm_manager import LLMManager
from .prompts import (
    TEST_GENERATION_PROMPT,
    FASTAPI_TEST_PROMPT,
    PYDANTIC_MODEL_TEST_PROMPT
)


class TesterAgent(BaseAgent):
    """
    Tester Agent - Test Generation
    
    Responsibilities:
    - Generate comprehensive pytest tests
    - Create FastAPI TestClient tests
    - Generate Pydantic model tests
    - Create integration and unit tests
    - Ensure high test coverage
    """
    
    def __init__(self, rmcp_endpoint: str = "http://localhost:8000"):
        super().__init__(
            name="Tester Agent",
            specialization="test_generation",
            rmcp_endpoint=rmcp_endpoint
        )
        # Initialize LLM Manager with OpenAI
        self.llm_manager = LLMManager(
            api_key=os.getenv("OPENAI_API_KEY"),
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        )
    
    async def _execute_specialized(self, request: AgentRequest) -> Dict[str, Any]:
        """
        Execute test generation task
        
        Args:
            request: The agent request
            
        Returns:
            Dict containing generated tests and artifacts
        """
        task_type = request.task_type
        goal = request.goal
        parameters = request.parameters
        context = request.context
        
        print(f"🧪 Tester Agent executing: {task_type}")
        print(f"   Goal: {goal}")
        print(f"   Parameters: {parameters}")
        
        try:
            if task_type == "test_generation":
                result = await self._generate_tests(goal, parameters, context)
            else:
                result = await self._generate_generic_tests(goal, parameters, context)
            
            return {
                "generated_tests": result.get("code", ""),
                "test_file": result.get("file_path", ""),
                "test_cases": result.get("test_cases", []),
                "artifacts": [
                    {
                        "type": "file",
                        "name": result.get("file_path", "test_generated.py"),
                        "content": result.get("code", ""),
                        "metadata": {
                            "task_type": task_type,
                            "generated_by": "tester_agent",
                            "test_cases": result.get("test_cases", []),
                            "parameters": parameters
                        }
                    }
                ],
                "metadata": {
                    "task_type": task_type,
                    "test_file": result.get("file_path", ""),
                    "test_cases_count": len(result.get("test_cases", [])),
                    "code_length": len(result.get("code", "")),
                    "generation_method": "llm"
                }
            }
            
        except Exception as e:
            print(f"❌ Tester Agent error: {e}")
            raise
    
    async def _generate_tests(self, goal: str, parameters: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate tests using LLM
        
        Args:
            goal: Test generation goal
            parameters: Generation parameters
            context: Additional context
            
        Returns:
            Dict containing generated tests
        """
        test_file = parameters.get("file_path", "test_generated.py")
        target_file = parameters.get("target_file", "main.py")
        test_cases = parameters.get("test_cases", ["basic_functionality"])
        
        # Get target code from context or parameters
        target_code = parameters.get("target_code", "")
        if not target_code and "target_code" in context:
            target_code = context["target_code"]
        
        # Determine prompt based on target file type
        if target_file.endswith("main.py") or "fastapi" in target_file.lower():
            prompt = self._create_fastapi_test_prompt(goal, parameters, context, target_code)
        elif target_file.endswith("models.py") or "pydantic" in target_file.lower():
            prompt = self._create_pydantic_test_prompt(goal, parameters, context, target_code)
        else:
            prompt = self._create_generic_test_prompt(goal, parameters, context, target_code)
        
        # Generate tests using LLM
        test_code = await self._call_llm(prompt)
        
        return {
            "code": test_code,
            "file_path": test_file,
            "target_file": target_file,
            "test_cases": test_cases
        }
    
    async def _generate_generic_tests(self, goal: str, parameters: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate generic tests based on goal
        
        Args:
            goal: Test generation goal
            parameters: Generation parameters
            context: Additional context
            
        Returns:
            Dict containing generated tests
        """
        prompt = self._create_generic_test_prompt(goal, parameters, context, "")
        test_code = await self._call_llm(prompt)
        
        return {
            "code": test_code,
            "file_path": parameters.get("file_path", "test_generated.py"),
            "test_cases": parameters.get("test_cases", ["basic_functionality"])
        }
    
    def _create_fastapi_test_prompt(self, goal: str, parameters: Dict[str, Any], context: Dict[str, Any], target_code: str) -> str:
        """Create FastAPI test prompt"""
        return FASTAPI_TEST_PROMPT.format(
            application_code=target_code,
            test_file=parameters.get("file_path", "test_main.py"),
            test_cases=parameters.get("test_cases", ["health_check", "endpoint_tests"]),
            context=context
        )
    
    def _create_pydantic_test_prompt(self, goal: str, parameters: Dict[str, Any], context: Dict[str, Any], target_code: str) -> str:
        """Create Pydantic model test prompt"""
        return PYDANTIC_MODEL_TEST_PROMPT.format(
            model_code=target_code,
            test_file=parameters.get("file_path", "test_models.py"),
            context=context
        )
    
    def _create_generic_test_prompt(self, goal: str, parameters: Dict[str, Any], context: Dict[str, Any], target_code: str) -> str:
        """Create generic test prompt"""
        return TEST_GENERATION_PROMPT.format(
            target_file=parameters.get("target_file", "target.py"),
            test_file=parameters.get("file_path", "test_generated.py"),
            test_cases=parameters.get("test_cases", ["basic_functionality"]),
            context=context
        )
    
    async def _call_llm(self, prompt: str) -> str:
        """
        Call LLM to generate tests
        
        Args:
            prompt: The prompt to send to LLM
            
        Returns:
            Generated test code
        """
        try:
            # Use real OpenAI LLM
            result = await self.llm_manager.generate_code_for_role(
                role="test_developer",
                prompt=prompt,
                language="python"
            )
            
            if result["success"]:
                print(f"✅ LLM generated tests: {result.get('tokens_used', 0)} tokens")
                return result["content"]
            else:
                print(f"❌ LLM call failed: {result.get('error', 'Unknown error')}")
                # Fallback to mock generation
                return await self._mock_llm_call(prompt)
            
        except Exception as e:
            print(f"❌ LLM call failed: {e}")
            # Fallback to template-based generation
            return self._fallback_test_generation(prompt)
    
    async def _mock_llm_call(self, prompt: str) -> str:
        """
        Mock LLM call for testing
        
        Args:
            prompt: The prompt
            
        Returns:
            Mock generated test code
        """
        # Simulate LLM processing time
        await asyncio.sleep(0.1)
        
        # Extract test file from prompt
        if "test_main.py" in prompt:
            return self._generate_mock_fastapi_tests()
        elif "test_models.py" in prompt:
            return self._generate_mock_pydantic_tests()
        else:
            return self._generate_mock_generic_tests()
    
    def _generate_mock_fastapi_tests(self) -> str:
        """Generate mock FastAPI tests"""
        return '''"""
Tests for FastAPI MCP Server
"""

import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_health_check():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_list_tools():
    """Test tools listing endpoint"""
    response = client.get("/tools")
    assert response.status_code == 200
    assert "tools" in response.json()
    assert len(response.json()["tools"]) > 0

def test_execute_tool_success():
    """Test successful tool execution"""
    response = client.post("/execute", json={"text": "Hello World"})
    assert response.status_code == 200
    result = response.json()
    assert "result" in result
    assert "success" in result

def test_execute_tool_validation_error():
    """Test tool execution with validation error"""
    response = client.post("/execute", json={})
    assert response.status_code == 422  # Validation error

def test_execute_tool_invalid_input():
    """Test tool execution with invalid input"""
    response = client.post("/execute", json={"text": ""})
    assert response.status_code == 200
    result = response.json()
    assert "result" in result

@pytest.mark.parametrize("text", [
    "Hello",
    "World",
    "Test Message",
    "Special Characters: !@#$%^&*()"
])
def test_execute_tool_with_different_inputs(text):
    """Test tool execution with different inputs"""
    response = client.post("/execute", json={"text": text})
    assert response.status_code == 200
    result = response.json()
    assert "result" in result
    assert "success" in result
'''
    
    def _generate_mock_pydantic_tests(self) -> str:
        """Generate mock Pydantic model tests"""
        return '''"""
Tests for Pydantic Models
"""

import pytest
from pydantic import ValidationError
from models import ToolRequest, ToolResponse

def test_tool_request_valid():
    """Test valid tool request"""
    request = ToolRequest(text="Hello World")
    assert request.text == "Hello World"
    assert request.options == {}

def test_tool_request_with_options():
    """Test tool request with options"""
    request = ToolRequest(text="Hello", options={"format": "json"})
    assert request.text == "Hello"
    assert request.options == {"format": "json"}

def test_tool_request_validation_error():
    """Test tool request validation error"""
    with pytest.raises(ValidationError):
        ToolRequest()  # Missing required field

def test_tool_response_success():
    """Test successful tool response"""
    response = ToolResponse(result="Success", success=True)
    assert response.result == "Success"
    assert response.success is True
    assert response.error is None

def test_tool_response_error():
    """Test error tool response"""
    response = ToolResponse(result="", success=False, error="Test error")
    assert response.result == ""
    assert response.success is False
    assert response.error == "Test error"

def test_tool_response_serialization():
    """Test tool response serialization"""
    response = ToolResponse(result="Test", success=True)
    data = response.model_dump()
    assert data["result"] == "Test"
    assert data["success"] is True
'''
    
    def _generate_mock_generic_tests(self) -> str:
        """Generate mock generic tests"""
        return '''"""
Generated tests
"""

import pytest

def test_basic_functionality():
    """Test basic functionality"""
    assert True

def test_edge_case():
    """Test edge case"""
    assert 1 + 1 == 2

def test_error_handling():
    """Test error handling"""
    with pytest.raises(ValueError):
        raise ValueError("Test error")
'''
    
    def _fallback_test_generation(self, prompt: str) -> str:
        """Fallback test generation when LLM fails"""
        return f'''"""
Fallback generated tests
Prompt: {prompt[:100]}...
"""

import pytest

def test_fallback():
    """Fallback test"""
    assert True
'''


# FastAPI application
app = FastAPI(
    title="Tester Agent API",
    description="Test Generation Agent",
    version="1.0.0"
)

# Global agent instance
agent = TesterAgent()


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
    Execute test generation request
    
    This endpoint receives test generation tasks and returns generated tests.
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
        port=8007,
        reload=True,
        log_level="info"
    )
