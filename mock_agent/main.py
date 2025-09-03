"""
Mock Agent - Test AI Agent for RMCP Meta-Orchestration
This is a simple FastAPI-based agent that demonstrates the agent protocol
"""

import json
import time
import httpx
import asyncio
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn


class AgentRequest(BaseModel):
    """Request model for agent execution"""
    goal: str = Field(..., description="The goal to achieve")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context")
    specialization: str = Field(default="general", description="Agent specialization")
    abstraction_level: str = Field(default="medium", description="Abstraction level")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Additional parameters")


class AgentResponse(BaseModel):
    """Response model for agent execution"""
    status: str = Field(..., description="Execution status")
    summary: str = Field(..., description="Summary of execution")
    data: Dict[str, Any] = Field(default_factory=dict, description="Execution results")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Execution metadata")
    execution_time_ms: int = Field(..., description="Execution time in milliseconds")


class MockAgent:
    """
    Mock AI Agent for testing RMCP meta-orchestration
    
    This agent demonstrates the agent protocol by:
    - Accepting structured requests from RMCP
    - Processing goals based on specialization
    - Using RMCP API for sub-task execution (Fractal Orchestration)
    - Returning structured responses
    - Simulating realistic execution times
    """
    
    def __init__(self, name: str = "Mock Agent", specialization: str = "general", rmcp_endpoint: str = "http://localhost:8000"):
        self.name = name
        self.specialization = specialization
        self.rmcp_endpoint = rmcp_endpoint
        self.execution_count = 0
        self.http_client = httpx.AsyncClient(timeout=30.0)
        
        # Predefined responses for different types of goals
        self.response_templates = {
            "security": {
                "status": "completed",
                "summary": "Security audit completed successfully",
                "data": {
                    "vulnerabilities_found": 3,
                    "critical_issues": 1,
                    "medium_issues": 2,
                    "low_issues": 0,
                    "recommendations": [
                        "Update authentication mechanism",
                        "Implement input validation",
                        "Enable security headers"
                    ],
                    "compliance_score": 85
                },
                "metadata": {
                    "audit_duration_minutes": 15,
                    "files_scanned": 47,
                    "security_tools_used": ["semgrep", "bandit", "safety"]
                }
            },
            "deployment": {
                "status": "completed", 
                "summary": "Deployment orchestration completed successfully",
                "data": {
                    "deployment_status": "successful",
                    "services_deployed": 3,
                    "rollback_available": True,
                    "health_checks_passed": True,
                    "deployment_time_minutes": 8
                },
                "metadata": {
                    "deployment_strategy": "blue-green",
                    "environment": "production",
                    "version": "v1.2.3"
                }
            },
            "testing": {
                "status": "completed",
                "summary": "Test automation completed successfully", 
                "data": {
                    "tests_executed": 156,
                    "tests_passed": 152,
                    "tests_failed": 4,
                    "coverage_percentage": 87.5,
                    "execution_time_minutes": 12
                },
                "metadata": {
                    "test_framework": "pytest",
                    "parallel_execution": True,
                    "test_types": ["unit", "integration", "e2e"]
                }
            },
            "documentation": {
                "status": "completed",
                "summary": "Documentation generation completed successfully",
                "data": {
                    "documents_generated": 5,
                    "api_endpoints_documented": 23,
                    "code_examples_added": 12,
                    "documentation_quality_score": 92
                },
                "metadata": {
                    "documentation_format": "markdown",
                    "generation_time_minutes": 6,
                    "sources_analyzed": ["code", "comments", "tests"]
                }
            },
            "general": {
                "status": "completed",
                "summary": "Task completed successfully",
                "data": {
                    "task_type": "general",
                    "complexity": "medium",
                    "success_rate": 0.95
                },
                "metadata": {
                    "processing_time_minutes": 5,
                    "resources_used": ["cpu", "memory"]
                }
            }
        }
    
    async def execute(self, request: AgentRequest) -> AgentResponse:
        """
        Execute the agent request
        
        Args:
            request: The agent request containing goal, context, etc.
            
        Returns:
            AgentResponse: The execution result
        """
        start_time = time.time()
        self.execution_count += 1
        
        try:
            # Check if this goal requires RMCP delegation (Fractal Orchestration)
            if self._should_delegate_to_rmcp(request.goal):
                # Delegate to RMCP for sub-task execution
                rmcp_result = await self._delegate_to_rmcp(request)
                
                # Create response based on RMCP result
                response = AgentResponse(
                    status="completed" if rmcp_result["success"] else "error",
                    summary=f"Task completed via RMCP delegation: {rmcp_result.get('result', {}).get('summary', 'No summary available')}",
                    data={
                        "delegated_to_rmcp": True,
                        "rmcp_result": rmcp_result,
                        "agent_name": self.name,
                        "specialization": request.specialization
                    },
                    metadata={
                        "agent_name": self.name,
                        "specialization": request.specialization,
                        "abstraction_level": request.abstraction_level,
                        "execution_count": self.execution_count,
                        "fractal_orchestration": True,
                        "rmcp_endpoint": self.rmcp_endpoint
                    },
                    execution_time_ms=int((time.time() - start_time) * 1000)
                )
                
                return response
            
            # Standard processing for non-delegated tasks
            response_template = self._select_response_template(request.goal, request.specialization)
            
            # Simulate processing time based on goal complexity
            processing_time = self._calculate_processing_time(request.goal)
            await asyncio.sleep(processing_time)
            
            # Create response
            response = AgentResponse(
                status=response_template["status"],
                summary=response_template["summary"],
                data=response_template["data"],
                metadata={
                    **response_template["metadata"],
                    "agent_name": self.name,
                    "specialization": request.specialization,
                    "abstraction_level": request.abstraction_level,
                    "execution_count": self.execution_count,
                    "processing_time_seconds": processing_time,
                    "fractal_orchestration": False
                },
                execution_time_ms=int((time.time() - start_time) * 1000)
            )
            
            return response
            
        except Exception as e:
            return AgentResponse(
                status="error",
                summary=f"Agent execution failed: {str(e)}",
                data={},
                metadata={
                    "agent_name": self.name,
                    "error": str(e),
                    "execution_count": self.execution_count
                },
                execution_time_ms=int((time.time() - start_time) * 1000)
            )
    
    def _select_response_template(self, goal: str, specialization: str) -> Dict[str, Any]:
        """Select appropriate response template based on goal and specialization"""
        goal_lower = goal.lower()
        
        # Check for specific keywords in goal
        if any(keyword in goal_lower for keyword in ["security", "audit", "vulnerability", "secure"]):
            return self.response_templates["security"]
        elif any(keyword in goal_lower for keyword in ["deploy", "deployment", "release", "production"]):
            return self.response_templates["deployment"]
        elif any(keyword in goal_lower for keyword in ["test", "testing", "coverage", "quality"]):
            return self.response_templates["testing"]
        elif any(keyword in goal_lower for keyword in ["document", "documentation", "docs", "api"]):
            return self.response_templates["documentation"]
        elif specialization in self.response_templates:
            return self.response_templates[specialization]
        else:
            return self.response_templates["general"]
    
    def _calculate_processing_time(self, goal: str) -> float:
        """Calculate realistic processing time based on goal complexity"""
        # Base processing time
        base_time = 0.5
        
        # Add time based on goal length (complexity)
        complexity_factor = len(goal.split()) * 0.1
        
        # Add some randomness for realism
        import random
        random_factor = random.uniform(0.1, 0.5)
        
        return min(base_time + complexity_factor + random_factor, 3.0)  # Cap at 3 seconds
    
    def _should_delegate_to_rmcp(self, goal: str) -> bool:
        """
        Determine if this goal should be delegated to RMCP for fractal orchestration
        
        This creates the fractal structure where agents can use RMCP for sub-tasks
        """
        goal_lower = goal.lower()
        
        # Delegate tasks that require specific tools or complex operations
        delegation_keywords = [
            "search", "find", "grep", "analyze code", "scan files",
            "execute command", "run tool", "process data", "generate report"
        ]
        
        return any(keyword in goal_lower for keyword in delegation_keywords)
    
    async def _delegate_to_rmcp(self, request: AgentRequest) -> Dict[str, Any]:
        """
        Delegate task to RMCP API for fractal orchestration
        
        This demonstrates how agents can use RMCP as a service
        """
        try:
            # Prepare request for RMCP API
            rmcp_request = {
                "goal": request.goal,
                "context": request.context,
                "max_execution_time_ms": 15000,  # 15 seconds for sub-tasks
                "strategy": "solo",
                "agent_id": f"{self.name}-{self.execution_count}"
            }
            
            # Make HTTP request to RMCP API
            response = await self.http_client.post(
                f"{self.rmcp_endpoint}/api/v1/agent/execute",
                json=rmcp_request,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "success": False,
                    "error": f"RMCP API returned status {response.status_code}: {response.text}"
                }
                
        except httpx.TimeoutException:
            return {
                "success": False,
                "error": "RMCP API request timed out"
            }
        except httpx.ConnectError:
            return {
                "success": False,
                "error": f"Could not connect to RMCP at {self.rmcp_endpoint}"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"RMCP delegation failed: {str(e)}"
            }
    
    async def close(self):
        """Close HTTP client"""
        await self.http_client.aclose()


# FastAPI application
app = FastAPI(
    title="Mock Agent API",
    description="Test AI Agent for RMCP Meta-Orchestration",
    version="1.0.0"
)

# Global agent instance
agent = MockAgent(name="Security Auditor Agent", specialization="security")


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
    Execute agent request
    
    This is the main endpoint that RMCP will call to delegate tasks to this agent.
    """
    try:
        response = await agent.execute(request)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats():
    """Get agent statistics"""
    return {
        "agent_name": agent.name,
        "specialization": agent.specialization,
        "execution_count": agent.execution_count,
        "status": "active"
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )
