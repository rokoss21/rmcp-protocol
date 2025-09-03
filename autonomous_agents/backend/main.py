"""
Backend Agent - Code Generation
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
    BACKEND_DEVELOPER_PROMPT,
    PYDANTIC_MODELS_PROMPT,
    FASTAPI_APPLICATION_PROMPT,
    REQUIREMENTS_FILE_PROMPT
)


class BackendAgent(BaseAgent):
    """
    Backend Agent - Code Generation
    
    Responsibilities:
    - Generate Python code for FastAPI applications
    - Create Pydantic models
    - Generate requirements.txt files
    - Create MCP server implementations
    - Write production-ready, maintainable code
    """
    
    def __init__(self, rmcp_endpoint: str = "http://localhost:8000"):
        super().__init__(
            name="Backend Agent",
            specialization="code_generation",
            rmcp_endpoint=rmcp_endpoint
        )
        # Initialize LLM Manager with OpenAI
        self.llm_manager = LLMManager(
            api_key=os.getenv("OPENAI_API_KEY"),
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        )
    
    async def _execute_specialized(self, request: AgentRequest) -> Dict[str, Any]:
        """
        Execute backend development task
        
        Args:
            request: The agent request
            
        Returns:
            Dict containing generated code and artifacts
        """
        task_type = request.task_type
        goal = request.goal
        parameters = request.parameters
        context = request.context
        
        print(f"🔧 Backend Agent executing: {task_type}")
        print(f"   Goal: {goal}")
        print(f"   Parameters: {parameters}")
        
        try:
            if task_type == "code_generation":
                result = await self._generate_code(goal, parameters, context)
            elif task_type == "file_creation":
                result = await self._create_file(goal, parameters, context)
            else:
                result = await self._generate_generic_code(goal, parameters, context)
            
            return {
                "generated_code": result.get("code", ""),
                "file_path": result.get("file_path", ""),
                "artifacts": [
                    {
                        "type": "file",
                        "name": result.get("file_path", "generated_file.py"),
                        "content": result.get("code", ""),
                        "metadata": {
                            "task_type": task_type,
                            "generated_by": "backend_agent",
                            "parameters": parameters
                        }
                    }
                ],
                "metadata": {
                    "task_type": task_type,
                    "file_path": result.get("file_path", ""),
                    "code_length": len(result.get("code", "")),
                    "generation_method": "llm"
                }
            }
            
        except Exception as e:
            print(f"❌ Backend Agent error: {e}")
            raise
    
    async def _generate_code(self, goal: str, parameters: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate Python code using LLM
        
        Args:
            goal: Code generation goal
            parameters: Generation parameters
            context: Additional context
            
        Returns:
            Dict containing generated code
        """
        file_path = parameters.get("file_path", "generated_file.py")
        template = parameters.get("template", "generic")
        
        # Determine prompt based on template
        if template == "fastapi_mcp_server":
            prompt = self._create_fastapi_prompt(goal, parameters, context)
        elif template == "pydantic_models":
            prompt = self._create_pydantic_prompt(goal, parameters, context)
        elif template == "requirements":
            prompt = self._create_requirements_prompt(goal, parameters, context)
        else:
            prompt = self._create_generic_prompt(goal, parameters, context)
        
        # Generate code using LLM
        code = await self._call_llm(prompt)
        
        return {
            "code": code,
            "file_path": file_path,
            "template": template
        }
    
    async def _create_file(self, goal: str, parameters: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a file with specified content
        
        Args:
            goal: File creation goal
            parameters: File parameters
            context: Additional context
            
        Returns:
            Dict containing file content
        """
        file_path = parameters.get("file_path", "generated_file.txt")
        content = parameters.get("content", "")
        
        # If no content provided, generate it
        if not content:
            if file_path.endswith(".py"):
                content = await self._generate_python_file(file_path, goal, parameters, context)
            elif file_path.endswith(".txt"):
                content = await self._generate_text_file(file_path, goal, parameters, context)
            else:
                content = f"# Generated file: {file_path}\n# Goal: {goal}\n"
        
        return {
            "code": content,
            "file_path": file_path,
            "template": "file_creation"
        }
    
    async def _generate_generic_code(self, goal: str, parameters: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate generic code based on goal
        
        Args:
            goal: Code generation goal
            parameters: Generation parameters
            context: Additional context
            
        Returns:
            Dict containing generated code
        """
        prompt = self._create_generic_prompt(goal, parameters, context)
        code = await self._call_llm(prompt)
        
        return {
            "code": code,
            "file_path": parameters.get("file_path", "generated_file.py"),
            "template": "generic"
        }
    
    def _create_fastapi_prompt(self, goal: str, parameters: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Create FastAPI application prompt"""
        return FASTAPI_APPLICATION_PROMPT.format(
            file_path=parameters.get("file_path", "main.py"),
            tool_definitions=parameters.get("tool_definitions", []),
            utility_name=parameters.get("utility_name", "unknown"),
            context=context
        )
    
    def _create_pydantic_prompt(self, goal: str, parameters: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Create Pydantic models prompt"""
        return PYDANTIC_MODELS_PROMPT.format(
            file_path=parameters.get("file_path", "models.py"),
            input_schema=parameters.get("input_schema", {}),
            output_schema=parameters.get("output_schema", {}),
            context=context
        )
    
    def _create_requirements_prompt(self, goal: str, parameters: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Create requirements.txt prompt"""
        return REQUIREMENTS_FILE_PROMPT.format(
            project_type=parameters.get("project_type", "mcp_server"),
            dependencies=parameters.get("dependencies", []),
            python_version=parameters.get("python_version", "3.11"),
            additional_requirements=parameters.get("additional_requirements", [])
        )
    
    def _create_generic_prompt(self, goal: str, parameters: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Create generic code generation prompt"""
        return BACKEND_DEVELOPER_PROMPT.format(
            file_path=parameters.get("file_path", "generated_file.py"),
            task_description=goal,
            context=context,
            code_requirements=parameters.get("code_requirements", "Generate clean, production-ready Python code")
        )
    
    async def _generate_python_file(self, file_path: str, goal: str, parameters: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Generate Python file content"""
        prompt = self._create_generic_prompt(goal, parameters, context)
        return await self._call_llm(prompt)
    
    async def _generate_text_file(self, file_path: str, goal: str, parameters: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Generate text file content"""
        if file_path == "requirements.txt":
            return await self._generate_requirements_content(parameters)
        else:
            return f"# Generated file: {file_path}\n# Goal: {goal}\n"
    
    async def _generate_requirements_content(self, parameters: Dict[str, Any]) -> str:
        """Generate requirements.txt content"""
        dependencies = parameters.get("dependencies", [])
        if not dependencies:
            dependencies = ["fastapi>=0.104.0", "uvicorn>=0.24.0", "pydantic>=2.0.0"]
        
        return "\n".join(dependencies)
    
    async def _call_llm(self, prompt: str) -> str:
        """
        Call LLM to generate code
        
        Args:
            prompt: The prompt to send to LLM
            
        Returns:
            Generated code
        """
        try:
            # Use real OpenAI LLM
            result = await self.llm_manager.generate_code_for_role(
                role="backend_developer",
                prompt=prompt,
                language="python"
            )
            
            if result["success"]:
                print(f"✅ LLM generated code: {result.get('tokens_used', 0)} tokens")
                return result["content"]
            else:
                print(f"❌ LLM call failed: {result.get('error', 'Unknown error')}")
                # Fallback to mock generation
                return await self._mock_llm_call(prompt)
            
        except Exception as e:
            print(f"❌ LLM call failed: {e}")
            # Fallback to template-based generation
            return self._fallback_code_generation(prompt)
    
    async def _mock_llm_call(self, prompt: str) -> str:
        """
        Mock LLM call for testing
        
        Args:
            prompt: The prompt
            
        Returns:
            Mock generated code
        """
        # Simulate LLM processing time
        await asyncio.sleep(0.1)
        
        # Extract file path from prompt
        if "main.py" in prompt:
            return self._generate_mock_fastapi_code()
        elif "models.py" in prompt:
            return self._generate_mock_pydantic_code()
        elif "requirements.txt" in prompt:
            return self._generate_mock_requirements()
        else:
            return self._generate_mock_generic_code()
    
    def _generate_mock_fastapi_code(self) -> str:
        """Generate mock FastAPI code"""
        return '''"""
MCP Server - Auto-generated by Backend Agent
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import subprocess
import uvicorn

app = FastAPI(title="MCP Server", version="1.0.0")

class ToolRequest(BaseModel):
    text: str
    options: dict = {}

class ToolResponse(BaseModel):
    result: str
    success: bool
    error: str = None

@app.get("/tools")
async def list_tools():
    """List available MCP tools"""
    return {
        "tools": [
            {
                "name": "tool.execute",
                "description": "Execute tool with specified parameters",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "Input text"},
                        "options": {"type": "object", "description": "Additional options"}
                    },
                    "required": ["text"]
                }
            }
        ]
    }

@app.post("/execute")
async def execute_tool(request: ToolRequest):
    """Execute tool"""
    try:
        # Execute command
        result = subprocess.run(
            ["echo", request.text],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            return ToolResponse(
                result=result.stdout,
                success=True
            )
        else:
            return ToolResponse(
                result="",
                success=False,
                error=result.stderr
            )
    except Exception as e:
        return ToolResponse(
            result="",
            success=False,
            error=str(e)
        )

@app.get("/health")
async def health():
    """Health check"""
    return {"status": "healthy", "service": "mcp-server"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''
    
    def _generate_mock_pydantic_code(self) -> str:
        """Generate mock Pydantic models"""
        return '''"""
Pydantic models for MCP Server
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class ToolRequest(BaseModel):
    """Request model for tool execution"""
    text: str = Field(..., description="Input text for the tool")
    options: Dict[str, Any] = Field(default_factory=dict, description="Additional options")

class ToolResponse(BaseModel):
    """Response model for tool execution"""
    result: str = Field(..., description="Tool output")
    success: bool = Field(..., description="Whether execution succeeded")
    error: Optional[str] = Field(None, description="Error message if failed")
'''
    
    def _generate_mock_requirements(self) -> str:
        """Generate mock requirements.txt"""
        return '''fastapi>=0.104.0
uvicorn>=0.24.0
pydantic>=2.0.0
'''
    
    def _generate_mock_generic_code(self) -> str:
        """Generate mock generic code"""
        return '''"""
Generated Python code
"""

def main():
    """Main function"""
    print("Hello, World!")

if __name__ == "__main__":
    main()
'''
    
    def _fallback_code_generation(self, prompt: str) -> str:
        """Fallback code generation when LLM fails"""
        return f'''"""
Fallback generated code
Prompt: {prompt[:100]}...
"""

def main():
    """Main function"""
    print("Fallback code generated")

if __name__ == "__main__":
    main()
'''


# FastAPI application
app = FastAPI(
    title="Backend Agent API",
    description="Code Generation Agent",
    version="1.0.0"
)

# Global agent instance
agent = BackendAgent()


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

@app.get("/capabilities")
async def get_capabilities():
    """Rich metadata about agent capabilities"""
    return {
        "agent_type": "backend",
        "role": "Full-Stack Developer", 
        "name": "Backend Agent",
        "version": "1.0.0",
        "description": "Experienced full-stack developer specialized in Python, web development, and code generation. Can build complete web applications, games, APIs, and interactive tools using modern technologies.",
        "core_competencies": [
            "Python development and scripting",
            "Web application development (full-stack)",
            "Frontend development (HTML/CSS/JavaScript)", 
            "API and backend services",
            "Interactive games and applications",
            "Code generation and automation"
        ],
        "professional_focus": "full-stack web developer",
        "can_handle": [
            "Complete web applications from frontend to backend",
            "Interactive web games and tools",
            "Python scripts and automation",
            "API development and integration",
            "Modern web technologies (HTML5, CSS3, JavaScript ES6+)"
        ],
        "output_formats": ["python", "html", "css", "javascript", "json", "yaml"],
        "complexity_level": "intermediate_to_advanced",
        "requires_llm": True,
        "suitable_for_tasks": [
            "Building web applications",
            "Creating interactive games", 
            "Developing APIs and backends",
            "Full-stack development projects",
            "Code generation tasks"
        ]
    }


@app.post("/execute", response_model=AgentResponse)
async def execute(request: Dict[str, Any]):
    """
    Execute backend development request (flexible input)
    
    Accepts either full AgentRequest schema or a simplified payload like:
    {"goal": "...", "file_path": "...", "task_type": "file_creation|code_generation"}
    """
    try:
        # If payload already matches AgentRequest, pass through
        if all(key in request for key in ["goal", "task_type", "parameters", "context"]):
            # Ensure task_id exists
            if "task_id" not in request or not request["task_id"]:
                import uuid as _uuid
                request["task_id"] = _uuid.uuid4().hex
            agent_request = AgentRequest(**request)
        else:
            # Build AgentRequest from simplified payload
            goal: str = request.get("goal", "")
            if not goal:
                raise HTTPException(status_code=422, detail="'goal' is required")
            task_type: str = request.get("task_type") or ("file_creation" if request.get("file_path") else "code_generation")
            parameters: Dict[str, Any] = request.get("parameters", {})
            if "file_path" in request:
                parameters.setdefault("file_path", request["file_path"]) 
            context: Dict[str, Any] = request.get("context", {})
            import uuid as _uuid
            agent_request = AgentRequest(
                task_id=_uuid.uuid4().hex,
                goal=goal,
                task_type=task_type,
                parameters=parameters,
                context=context
            )
        response = await agent.execute(agent_request)
        return response
    except HTTPException:
        raise
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
        port=8005,
        reload=True,
        log_level="info"
    )
