"""
MCP Agent Wrapper - Bridges RMCP and Autonomous Agents
Exposes autonomous agents as MCP tools
"""

import httpx
import asyncio
import os
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Agent Wrapper MCP Server",
    description="MCP Server that wraps autonomous agents",
    version="1.0.0"
)

class ToolRequest(BaseModel):
    """MCP tool request format"""
    arguments: List[str] = Field(default_factory=list, description="Tool arguments (legacy)")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Tool parameters (modern format)")

class ToolResponse(BaseModel):
    """MCP tool response format"""
    success: bool
    output: str
    error: Optional[str] = None
    exit_code: int = 0
    tool_name: str
    metadata: Optional[Dict[str, Any]] = None

# Agent endpoints configuration
AGENT_ENDPOINTS = {
    "backend": os.getenv("BACKEND_AGENT_URL", "http://backend-agent:8005"),
    "architect": os.getenv("ARCHITECT_AGENT_URL", "http://architect-agent:8006"),
    "tester": os.getenv("TESTER_AGENT_URL", "http://tester-agent:8007"),
    "devops": os.getenv("DEVOPS_AGENT_URL", "http://devops-agent:8008"),
    "orchestrator": os.getenv("ORCHESTRATOR_AGENT_URL", "http://orchestrator-agent:8009"),
    "validator": os.getenv("VALIDATOR_AGENT_URL", "http://validator-agent:8010"),
}

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Agent Wrapper MCP Server",
        "version": "1.0.0",
        "wrapped_agents": list(AGENT_ENDPOINTS.keys())
    }

@app.get("/health")
async def health():
    """Health check"""
    return {"status": "healthy", "service": "agent-wrapper-mcp"}

@app.get("/tools")
async def list_tools():
    """List available MCP tools (wrapped agents)"""
    tools = []
    
    # Backend Agent Tools
    tools.extend([
        {
            "name": "backend.generate_python_script",
            "description": "Generate Python script using backend agent with LLM",
            "input_schema": {
                "type": "object",
                "properties": {
                    "goal": {"type": "string", "description": "What to generate"},
                    "file_path": {"type": "string", "description": "Output file path"},
                    "script_type": {"type": "string", "description": "Type of script (basic, fastapi, etc.)"}
                },
                "required": ["goal", "file_path"]
            }
        },
        {
            "name": "backend.create_fastapi_app",
            "description": "Create FastAPI application using backend agent",
            "input_schema": {
                "type": "object",
                "properties": {
                    "goal": {"type": "string", "description": "Application description"},
                    "app_name": {"type": "string", "description": "Application name"},
                    "features": {"type": "array", "description": "Required features"}
                },
                "required": ["goal", "app_name"]
            }
        },
        {
            "name": "backend.generate_code",
            "description": "Generate any Python code using backend agent LLM",
            "input_schema": {
                "type": "object",
                "properties": {
                    "goal": {"type": "string", "description": "Code generation goal"},
                    "file_path": {"type": "string", "description": "Output file path"},
                    "template": {"type": "string", "description": "Code template type"}
                },
                "required": ["goal", "file_path"]
            }
        }
    ])
    
    # Architect Agent Tools
    tools.extend([
        {
            "name": "architect.design_system",
            "description": "Design system architecture using architect agent",
            "input_schema": {
                "type": "object",
                "properties": {
                    "goal": {"type": "string", "description": "System to design"},
                    "requirements": {"type": "array", "description": "System requirements"}
                },
                "required": ["goal"]
            }
        },
        {
            "name": "architect.create_development_plan",
            "description": "Create detailed development plan using architect agent",
            "input_schema": {
                "type": "object",
                "properties": {
                    "goal": {"type": "string", "description": "Project goal"},
                    "complexity": {"type": "string", "description": "Project complexity"}
                },
                "required": ["goal"]
            }
        }
    ])
    
    # Tester Agent Tools
    tools.extend([
        {
            "name": "tester.generate_tests",
            "description": "Generate comprehensive tests using tester agent",
            "input_schema": {
                "type": "object",
                "properties": {
                    "goal": {"type": "string", "description": "Testing goal"},
                    "code_path": {"type": "string", "description": "Path to code to test"},
                    "test_type": {"type": "string", "description": "Type of test (unit, integration, e2e)"}
                },
                "required": ["goal", "code_path"]
            }
        }
    ])
    
    # DevOps Agent Tools
    tools.extend([
        {
            "name": "devops.create_dockerfile",
            "description": "Create optimized Dockerfile using devops agent",
            "input_schema": {
                "type": "object",
                "properties": {
                    "goal": {"type": "string", "description": "Deployment goal"},
                    "project_type": {"type": "string", "description": "Type of project"},
                    "dependencies": {"type": "array", "description": "Project dependencies"}
                },
                "required": ["goal", "project_type"]
            }
        },
        {
            "name": "devops.create_docker_compose",
            "description": "Create docker-compose configuration using devops agent",
            "input_schema": {
                "type": "object",
                "properties": {
                    "goal": {"type": "string", "description": "Orchestration goal"},
                    "services": {"type": "array", "description": "Required services"}
                },
                "required": ["goal"]
            }
        }
    ])
    
    # Orchestrator Agent Tools
    tools.extend([
        {
            "name": "orchestrator.manage_workflow",
            "description": "Orchestrate multi-agent development workflow",
            "input_schema": {
                "type": "object",
                "properties": {
                    "goal": {"type": "string", "description": "Development goal"},
                    "agents": {"type": "array", "description": "Required agents"}
                },
                "required": ["goal"]
            }
        }
    ])
    
    # Validator Agent Tools
    tools.extend([
        {
            "name": "validator.validate_code",
            "description": "Validate and test code using validator agent",
            "input_schema": {
                "type": "object",
                "properties": {
                    "goal": {"type": "string", "description": "Validation goal"},
                    "code_path": {"type": "string", "description": "Path to code"},
                    "validation_type": {"type": "string", "description": "Type of validation"}
                },
                "required": ["goal", "code_path"]
            }
        }
    ])
    
    return {"tools": tools}

@app.post("/execute")
async def execute_tool(request: ToolRequest):
    """Execute wrapped agent tool"""
    
    logger.info(f"Received request: args={request.arguments}, params={request.parameters}")
    
    # RMCP sends tool name as first argument in legacy format
    if request.arguments and len(request.arguments) > 0:
        tool_name = request.arguments[0]
        # Extract parameters from either format
        if request.parameters:
            params = request.parameters.copy()
        else:
            params = {}
        
        # Get goal from parameters or second argument
        goal = params.get("goal", "") 
        if not goal and len(request.arguments) > 1:
            goal = request.arguments[1]
        if not goal:
            goal = params.get("task_description", "")
    else:
        return ToolResponse(
            success=False,
            output="",
            error="Tool name is required in arguments[0]",
            exit_code=1,
            tool_name="unknown"
        )
    
    # Ensure we have a goal
    if not goal:
        return ToolResponse(
            success=False,
            output="",
            error="Goal is required",
            exit_code=1,
            tool_name=tool_name
        )
    
    # Set goal in params
    params["goal"] = goal
    
    # Extract file path from various sources
    file_path = (params.get("file_path") or 
                params.get("output_file") or 
                params.get("path"))
    
    # Smart file path detection based on goal content
    if not file_path:
        if "крестики" in goal.lower() or "tic.tac.toe" in goal.lower():
            file_path = "tic_tac_toe.html"
        elif "калькулятор" in goal.lower() or "calculator" in goal.lower():
            file_path = "calculator.py"
        elif "dockerfile" in goal.lower():
            file_path = "Dockerfile"
        elif "docker.compose" in goal.lower():
            file_path = "docker-compose.yml"
        elif goal.lower().endswith(".py") or "python" in goal.lower():
            file_path = "generated_script.py"
        elif goal.lower().endswith(".html") or "html" in goal.lower():
            file_path = "index.html"
        else:
            file_path = "generated_file.txt"
    
    params["file_path"] = file_path
    
    logger.info(f"Processed: tool_name={tool_name}, goal={goal}, file_path={file_path}")
    
    try:
        if tool_name.startswith("backend."):
            return await _execute_backend_agent(tool_name, params)
        elif tool_name.startswith("architect."):
            return await _execute_architect_agent(tool_name, params)
        elif tool_name.startswith("tester."):
            return await _execute_tester_agent(tool_name, params)
        elif tool_name.startswith("devops."):
            return await _execute_devops_agent(tool_name, params)
        elif tool_name.startswith("orchestrator."):
            return await _execute_orchestrator_agent(tool_name, params)
        elif tool_name.startswith("validator."):
            return await _execute_validator_agent(tool_name, params)
        else:
            return ToolResponse(
                success=False,
                output="",
                error=f"Unknown tool: {tool_name}",
                exit_code=1,
                tool_name=tool_name
            )
            
    except Exception as e:
        logger.error(f"Error executing {tool_name}: {e}")
        return ToolResponse(
            success=False,
            output="",
            error=f"Execution failed: {str(e)}",
            exit_code=1,
            tool_name=tool_name
        )

async def _execute_backend_agent(tool_name: str, params: Dict[str, Any]) -> ToolResponse:
    """Execute backend agent tool"""
    
    agent_endpoint = AGENT_ENDPOINTS["backend"]
    goal = params.get("goal", "")
    if not goal:
        return ToolResponse(
            success=False,
            output="",
            error="Goal parameter is required",
            exit_code=1,
            tool_name=tool_name
        )
    
    file_path = params.get("file_path", "generated_file.py")
    
    # Map tool to agent task type
    if "python_script" in tool_name:
        task_type = "code_generation"
        agent_params = {
            "file_path": file_path,
            "template": "generic",
            "code_requirements": "Generate clean, working Python code"
        }
    elif "fastapi" in tool_name:
        task_type = "code_generation"
        agent_params = {
            "file_path": file_path,
            "template": "fastapi_mcp_server",
            "app_name": params.get("app_name", "FastAPI App")
        }
    else:
        task_type = "code_generation"
        agent_params = {
            "file_path": file_path,
            "template": params.get("template", "generic")
        }
    
    # Create agent request
    agent_request = {
        "task_id": f"mcp-wrapper-{int(asyncio.get_event_loop().time())}",
        "task_type": task_type,
        "goal": goal,
        "parameters": agent_params,
        "context": params.get("context", {}),
        "priority": "medium",
        "timeout_ms": 30000
    }
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            logger.info(f"Calling backend agent at {agent_endpoint}")
            response = await client.post(
                f"{agent_endpoint}/execute",
                json=agent_request,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                agent_response = response.json()
                
                if agent_response.get("status") == "completed":
                    result = agent_response.get("result", {})
                    generated_code = result.get("generated_code", "")
                    
                    return ToolResponse(
                        success=True,
                        output=f"Generated {file_path}: {len(generated_code)} characters",
                        error=None,
                        exit_code=0,
                        tool_name=tool_name,
                        metadata={
                            "generated_code": generated_code,
                            "file_path": file_path,
                            "agent": "backend"
                        }
                    )
                else:
                    return ToolResponse(
                        success=False,
                        output="",
                        error=agent_response.get("error", "Agent execution failed"),
                        exit_code=1,
                        tool_name=tool_name
                    )
            else:
                return ToolResponse(
                    success=False,
                    output="",
                    error=f"Backend agent returned {response.status_code}: {response.text}",
                    exit_code=1,
                    tool_name=tool_name
                )
                
    except httpx.TimeoutException:
        return ToolResponse(
            success=False,
            output="",
            error="Backend agent request timed out",
            exit_code=1,
            tool_name=tool_name
        )
    except Exception as e:
        return ToolResponse(
            success=False,
            output="",
            error=f"Failed to call backend agent: {str(e)}",
            exit_code=1,
            tool_name=tool_name
        )

async def _execute_architect_agent(tool_name: str, params: Dict[str, Any]) -> ToolResponse:
    """Execute architect agent tool"""
    return await _execute_generic_agent("architect", tool_name, params)

async def _execute_tester_agent(tool_name: str, params: Dict[str, Any]) -> ToolResponse:
    """Execute tester agent tool"""
    return await _execute_generic_agent("tester", tool_name, params)

async def _execute_devops_agent(tool_name: str, params: Dict[str, Any]) -> ToolResponse:
    """Execute devops agent tool"""
    return await _execute_generic_agent("devops", tool_name, params)

async def _execute_orchestrator_agent(tool_name: str, params: Dict[str, Any]) -> ToolResponse:
    """Execute orchestrator agent tool"""
    return await _execute_generic_agent("orchestrator", tool_name, params)

async def _execute_validator_agent(tool_name: str, params: Dict[str, Any]) -> ToolResponse:
    """Execute validator agent tool"""
    return await _execute_generic_agent("validator", tool_name, params)

async def _execute_generic_agent(agent_type: str, tool_name: str, params: Dict[str, Any]) -> ToolResponse:
    """Execute generic agent tool"""
    
    agent_endpoint = AGENT_ENDPOINTS.get(agent_type)
    if not agent_endpoint:
        return ToolResponse(
            success=False,
            output="",
            error=f"Agent endpoint not configured for {agent_type}",
            exit_code=1,
            tool_name=tool_name
        )
    
    goal = params.get("goal", "")
    if not goal:
        return ToolResponse(
            success=False,
            output="",
            error="Goal parameter is required",
            exit_code=1,
            tool_name=tool_name
        )
    
    # Map tool to agent task type
    if "design" in tool_name or "plan" in tool_name:
        task_type = "system_design"
    elif "test" in tool_name:
        task_type = "test_generation"
    elif "dockerfile" in tool_name or "docker_compose" in tool_name:
        task_type = "deployment_configuration"
    elif "workflow" in tool_name:
        task_type = "workflow_management"
    elif "validate" in tool_name:
        task_type = "code_validation"
    else:
        task_type = "general"
    
    # Create agent request
    agent_request = {
        "task_id": f"mcp-wrapper-{agent_type}-{int(asyncio.get_event_loop().time())}",
        "task_type": task_type,
        "goal": goal,
        "parameters": params,
        "context": params.get("context", {}),
        "priority": "medium",
        "timeout_ms": 30000
    }
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            logger.info(f"Calling {agent_type} agent at {agent_endpoint}")
            response = await client.post(
                f"{agent_endpoint}/execute",
                json=agent_request,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                agent_response = response.json()
                
                if agent_response.get("status") == "completed":
                    result = agent_response.get("result", {})
                    output_summary = f"{agent_type.title()} agent completed task successfully"
                    
                    return ToolResponse(
                        success=True,
                        output=output_summary,
                        error=None,
                        exit_code=0,
                        tool_name=tool_name,
                        metadata={
                            "agent_result": result,
                            "agent": agent_type,
                            "task_type": task_type
                        }
                    )
                else:
                    return ToolResponse(
                        success=False,
                        output="",
                        error=agent_response.get("error", f"{agent_type} agent execution failed"),
                        exit_code=1,
                        tool_name=tool_name
                    )
            else:
                return ToolResponse(
                    success=False,
                    output="",
                    error=f"{agent_type} agent returned {response.status_code}: {response.text}",
                    exit_code=1,
                    tool_name=tool_name
                )
                
    except httpx.TimeoutException:
        return ToolResponse(
            success=False,
            output="",
            error=f"{agent_type} agent request timed out",
            exit_code=1,
            tool_name=tool_name
        )
    except Exception as e:
        return ToolResponse(
            success=False,
            output="",
            error=f"Failed to call {agent_type} agent: {str(e)}",
            exit_code=1,
            tool_name=tool_name
        )

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8003,
        log_level="info"
    )
