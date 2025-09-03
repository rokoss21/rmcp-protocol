"""
Base class for autonomous agents
"""

import asyncio
import time
import httpx
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod
from datetime import datetime

from .models import AgentRequest, AgentResponse, TaskStatus, TaskPriority


class BaseAgent(ABC):
    """
    Base class for all autonomous agents
    
    Provides common functionality:
    - HTTP client for RMCP communication
    - Standardized request/response handling
    - Error handling and retry logic
    - Telemetry and logging
    """
    
    def __init__(
        self, 
        name: str, 
        specialization: str,
        rmcp_endpoint: str = "http://localhost:8000",
        timeout: float = 30.0
    ):
        self.name = name
        self.specialization = specialization
        self.rmcp_endpoint = rmcp_endpoint
        self.timeout = timeout
        self.http_client = httpx.AsyncClient(timeout=timeout)
        self.execution_count = 0
        
    async def execute(self, request: AgentRequest) -> AgentResponse:
        """
        Execute agent request with standardized error handling
        
        Args:
            request: The agent request
            
        Returns:
            AgentResponse: The execution result
        """
        start_time = time.time()
        self.execution_count += 1
        
        try:
            # Validate request
            self._validate_request(request)
            
            # Execute specialized logic
            result = await self._execute_specialized(request)
            
            # Create response
            response = AgentResponse(
                task_id=request.task_id,
                status=TaskStatus.COMPLETED,
                result=result,
                artifacts=result.get("artifacts", []),
                metadata={
                    "agent_name": self.name,
                    "specialization": self.specialization,
                    "execution_count": self.execution_count,
                    "task_type": request.task_type,
                    "priority": request.priority.value
                },
                execution_time_ms=int((time.time() - start_time) * 1000)
            )
            
            return response
            
        except Exception as e:
            return AgentResponse(
                task_id=request.task_id,
                status=TaskStatus.FAILED,
                error=str(e),
                metadata={
                    "agent_name": self.name,
                    "specialization": self.specialization,
                    "execution_count": self.execution_count,
                    "error_type": type(e).__name__
                },
                execution_time_ms=int((time.time() - start_time) * 1000)
            )
    
    @abstractmethod
    async def _execute_specialized(self, request: AgentRequest) -> Dict[str, Any]:
        """
        Execute specialized agent logic
        
        Args:
            request: The agent request
            
        Returns:
            Dict containing execution result
        """
        pass
    
    def _validate_request(self, request: AgentRequest):
        """Validate agent request"""
        if not request.task_id:
            raise ValueError("Task ID is required")
        if not request.goal:
            raise ValueError("Goal is required")
        if not request.task_type:
            raise ValueError("Task type is required")
    
    async def delegate_to_rmcp(self, goal: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Delegate task to RMCP for execution
        
        Args:
            goal: The goal to achieve
            context: Additional context
            
        Returns:
            Dict containing RMCP execution result
        """
        try:
            rmcp_request = {
                "goal": goal,
                "context": context or {},
                "max_execution_time_ms": 30000,
                "strategy": "solo",
                "agent_id": f"{self.name}-{self.execution_count}"
            }
            
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
    
    async def call_other_agent(
        self, 
        agent_endpoint: str, 
        task_type: str, 
        goal: str, 
        parameters: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Call another agent directly
        
        Args:
            agent_endpoint: Endpoint of the target agent
            task_type: Type of task
            goal: Goal to achieve
            parameters: Task parameters
            
        Returns:
            Dict containing agent execution result
        """
        try:
            agent_request = {
                "task_id": f"{self.name}-{self.execution_count}-{int(time.time())}",
                "task_type": task_type,
                "goal": goal,
                "context": {},
                "parameters": parameters or {},
                "priority": "medium",
                "timeout_ms": 30000
            }
            
            response = await self.http_client.post(
                f"{agent_endpoint}/execute",
                json=agent_request,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "success": False,
                    "error": f"Agent API returned status {response.status_code}: {response.text}"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Agent call failed: {str(e)}"
            }
    
    async def close(self):
        """Close HTTP client"""
        await self.http_client.aclose()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics"""
        return {
            "name": self.name,
            "specialization": self.specialization,
            "execution_count": self.execution_count,
            "rmcp_endpoint": self.rmcp_endpoint,
            "status": "active"
        }

