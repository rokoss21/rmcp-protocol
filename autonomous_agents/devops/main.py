"""
DevOps Agent - Deployment and Infrastructure
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
from .deployment import DeploymentManager


class DevOpsAgent(BaseAgent):
    """
    DevOps Agent - Deployment and Infrastructure
    
    Responsibilities:
    - Deploy MCP servers as Docker containers
    - Generate Docker configurations
    - Manage container lifecycle
    - Monitor deployment status
    - Handle infrastructure tasks
    """
    
    def __init__(self, rmcp_endpoint: str = "http://localhost:8000"):
        super().__init__(
            name="DevOps Agent",
            specialization="deployment",
            rmcp_endpoint=rmcp_endpoint
        )
        self.deployment_manager = DeploymentManager()
        # Initialize LLM Manager with OpenAI
        self.llm_manager = LLMManager(
            api_key=os.getenv("OPENAI_API_KEY"),
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        )
    
    async def _execute_specialized(self, request: AgentRequest) -> Dict[str, Any]:
        """
        Execute DevOps task
        
        Args:
            request: The agent request
            
        Returns:
            Dict containing deployment results
        """
        task_type = request.task_type
        goal = request.goal
        parameters = request.parameters
        context = request.context
        
        print(f"🚀 DevOps Agent executing: {task_type}")
        print(f"   Goal: {goal}")
        print(f"   Parameters: {parameters}")
        
        try:
            if task_type == "deployment":
                result = await self._deploy_service(goal, parameters, context)
            elif task_type == "file_creation":
                result = await self._create_infrastructure_file(goal, parameters, context)
            else:
                result = await self._handle_generic_devops_task(goal, parameters, context)
            
            return {
                "deployment_results": result.get("results", {}),
                "deployment_success": result.get("success", False),
                "artifacts": [
                    {
                        "type": "deployment_report",
                        "name": "deployment_report.json",
                        "content": result.get("results", {}),
                        "metadata": {
                            "task_type": task_type,
                            "generated_by": "devops_agent",
                            "parameters": parameters
                        }
                    }
                ],
                "metadata": {
                    "task_type": task_type,
                    "deployment_method": "docker",
                    "deployment_success": result.get("success", False),
                    "results": result.get("results", {})
                }
            }
            
        except Exception as e:
            print(f"❌ DevOps Agent error: {e}")
            raise
    
    async def _deploy_service(self, goal: str, parameters: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deploy MCP service
        
        Args:
            goal: Deployment goal
            parameters: Deployment parameters
            context: Additional context
            
        Returns:
            Dict containing deployment results
        """
        project_path = parameters.get("project_path", "/tmp/test_project")
        image_name = parameters.get("image_name", "mcp-server")
        container_name = parameters.get("container_name", "mcp-server-container")
        port = parameters.get("port", 8000)
        
        # If no real project path, create mock deployment
        if not os.path.exists(project_path):
            return await self._mock_deployment(goal, parameters, context)
        
        # Run real deployment
        results = self.deployment_manager.deploy_mcp_server(
            project_path, image_name, container_name, port
        )
        
        return {
            "results": results,
            "success": results.get("success", False)
        }
    
    async def _create_infrastructure_file(self, goal: str, parameters: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create infrastructure files (Dockerfile, docker-compose.yml)
        
        Args:
            goal: File creation goal
            parameters: File parameters
            context: Additional context
            
        Returns:
            Dict containing file creation results
        """
        file_path = parameters.get("file_path", "Dockerfile")
        file_type = parameters.get("file_type", "dockerfile")
        
        if file_type == "dockerfile":
            content = await self._generate_dockerfile(parameters, context)
        elif file_type == "docker_compose":
            content = await self._generate_docker_compose(parameters, context)
        else:
            content = await self._generate_generic_infrastructure_file(parameters, context)
        
        return {
            "results": {
                "file_path": file_path,
                "content": content,
                "file_type": file_type
            },
            "success": True
        }
    
    async def _handle_generic_devops_task(self, goal: str, parameters: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle generic DevOps tasks
        
        Args:
            goal: Task goal
            parameters: Task parameters
            context: Additional context
            
        Returns:
            Dict containing task results
        """
        return await self._mock_deployment(goal, parameters, context)
    
    async def _mock_deployment(self, goal: str, parameters: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Mock deployment for testing
        
        Args:
            goal: Deployment goal
            parameters: Deployment parameters
            context: Additional context
            
        Returns:
            Dict containing mock deployment results
        """
        await asyncio.sleep(0.1)  # Simulate deployment time
        
        image_name = parameters.get("image_name", "mcp-server")
        container_name = parameters.get("container_name", "mcp-server-container")
        port = parameters.get("port", 8000)
        
        results = {
            "success": True,
            "output": f"Successfully deployed {container_name}",
            "error": "",
            "container_name": container_name,
            "port": port,
            "image_name": image_name,
            "status": "running"
        }
        
        return {
            "results": results,
            "success": True
        }
    
    async def _generate_dockerfile(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Generate Dockerfile content using LLM"""
        try:
            prompt = f"""Generate a production-ready Dockerfile for a FastAPI MCP server with the following requirements:

- Base image: {parameters.get("base_image", "python:3.11-slim")}
- Utility: {parameters.get("utility_name", "unknown")}
- Port: 8000
- Health check endpoint: /health
- Application entry point: uvicorn main:app

Requirements:
1. Install system dependencies including the utility
2. Install Python dependencies from requirements.txt
3. Copy application code
4. Expose port 8000
5. Add health check
6. Use proper security practices
7. Optimize for production

Return ONLY the Dockerfile content, no explanations."""
            
            result = await self.llm_manager.generate_code_for_role(
                role="devops_engineer",
                prompt=prompt,
                language="dockerfile"
            )
            
            if result["success"]:
                print(f"✅ LLM generated Dockerfile: {result.get('tokens_used', 0)} tokens")
                return result["content"]
            else:
                # Fallback to template
                return self._generate_fallback_dockerfile(parameters)
                
        except Exception as e:
            print(f"❌ LLM Dockerfile generation failed: {e}")
            return self._generate_fallback_dockerfile(parameters)
    
    def _generate_fallback_dockerfile(self, parameters: Dict[str, Any]) -> str:
        """Fallback Dockerfile generation"""
        base_image = parameters.get("base_image", "python:3.11-slim")
        utility_name = parameters.get("utility_name", "unknown")
        
        return f'''FROM {base_image}

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    {utility_name} \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
'''
    
    async def _generate_docker_compose(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Generate docker-compose.yml content"""
        service_name = parameters.get("service_name", "mcp-server")
        port = parameters.get("port", 8000)
        
        return f'''version: '3.8'

services:
  {service_name}:
    build: .
    ports:
      - "{port}:8000"
    environment:
      - PYTHONUNBUFFERED=1
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
'''
    
    async def _generate_generic_infrastructure_file(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Generate generic infrastructure file"""
        file_type = parameters.get("file_type", "unknown")
        
        return f'''# Generated {file_type} file
# Created by DevOps Agent

# Add your configuration here
'''


# FastAPI application
app = FastAPI(
    title="DevOps Agent API",
    description="Deployment and Infrastructure Agent",
    version="1.0.0"
)

# Global agent instance
agent = DevOpsAgent()


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
    Execute DevOps request
    
    This endpoint receives deployment tasks and returns deployment results.
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
        port=8008,
        reload=True,
        log_level="info"
    )
