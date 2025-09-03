"""
Brigade Orchestrator - Development Workflow Management
"""

import asyncio
import httpx
import time
import os
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

from ..base.models import AgentRequest, AgentResponse, TaskStatus, TaskPriority, Task, TaskResult
from ..llm_manager import LLMManager
from .project_manager import ProjectManager


class BrigadeOrchestrator:
    """
    Brigade Orchestrator - Manages autonomous development workflow
    
    Responsibilities:
    - Coordinate between different agents
    - Manage project state and task execution
    - Handle retry logic and error recovery
    - Execute development plans from architect
    """
    
    def __init__(self, rmcp_endpoint: str = "http://localhost:8000"):
        self.rmcp_endpoint = rmcp_endpoint
        self.http_client = httpx.AsyncClient(timeout=60.0)
        self.project_manager = ProjectManager()
        
        # Initialize LLM Manager with OpenAI
        self.llm_manager = LLMManager(
            api_key=os.getenv("OPENAI_API_KEY"),
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        )
        
        # Agent endpoints
        self.agent_endpoints = {
            "architect": "http://localhost:8003",
            "backend": "http://localhost:8004",
            "tester": "http://localhost:8005",
            "validator": "http://localhost:8006",
            "debugger": "http://localhost:8007",
            "devops": "http://localhost:8008"
        }
        
        self.execution_count = 0
    
    async def execute_development_plan(self, goal: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute a complete development plan from goal to deployment
        
        Args:
            goal: High-level development goal
            context: Additional context
            
        Returns:
            Dict containing execution results
        """
        start_time = time.time()
        self.execution_count += 1
        
        try:
            # Step 1: Get development plan from architect
            print(f"🏗️  Step 1: Getting development plan from architect...")
            plan = await self._get_development_plan(goal, context or {})
            
            if not plan:
                return {
                    "success": False,
                    "error": "Failed to get development plan from architect"
                }
            
            # Step 2: Create project
            print(f"📁 Step 2: Creating project...")
            project = self.project_manager.create_project(plan)
            print(f"   Project ID: {project.project_id}")
            print(f"   Tasks: {len(project.tasks)}")
            
            # Step 3: Execute development loop
            print(f"🔄 Step 3: Executing development loop...")
            execution_result = await self._execute_development_loop(project)
            
            # Step 4: Get final project state
            final_project = self.project_manager.get_project(project.project_id)
            summary = self.project_manager.get_project_summary(project.project_id)
            
            return {
                "success": execution_result["success"],
                "project_id": project.project_id,
                "project_summary": summary,
                "execution_result": execution_result,
                "development_plan": plan.dict(),
                "execution_time_ms": int((time.time() - start_time) * 1000)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Development plan execution failed: {str(e)}",
                "execution_time_ms": int((time.time() - start_time) * 1000)
            }
    
    async def _get_development_plan(self, goal: str, context: Dict[str, Any]) -> Optional[Any]:
        """
        Get development plan from architect agent
        
        Args:
            goal: Development goal
            context: Additional context
            
        Returns:
            DevelopmentPlan or None
        """
        try:
            request = AgentRequest(
                task_id=f"orchestrator-{self.execution_count}-plan",
                task_type="system_design",
                goal=goal,
                context=context,
                parameters={
                    "include_tests": True,
                    "include_docker": True,
                    "include_documentation": True
                },
                priority=TaskPriority.HIGH
            )
            
            response = await self.http_client.post(
                f"{self.agent_endpoints['architect']}/execute",
                json=request.dict(),
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get("status") == "completed" and "development_plan" in result.get("result", {}):
                    from ..base.models import DevelopmentPlan
                    plan_data = result["result"]["development_plan"]
                    return DevelopmentPlan(**plan_data)
            
            return None
            
        except Exception as e:
            print(f"❌ Failed to get development plan: {e}")
            return None
    
    async def _execute_development_loop(self, project) -> Dict[str, Any]:
        """
        Execute the main development loop
        
        Args:
            project: Project state
            
        Returns:
            Dict containing execution results
        """
        max_iterations = 50  # Prevent infinite loops
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            print(f"   Iteration {iteration}")
            
            # Get ready tasks
            ready_tasks = self.project_manager.get_ready_tasks(project.project_id)
            
            if not ready_tasks:
                # Check if all tasks are completed
                all_completed = all(
                    task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED] 
                    for task in project.tasks
                )
                
                if all_completed:
                    print(f"✅ All tasks completed!")
                    break
                else:
                    print(f"⚠️  No ready tasks, but not all completed. Waiting...")
                    await asyncio.sleep(1)
                    continue
            
            # Execute ready tasks
            for task in ready_tasks:
                print(f"   Executing task: {task.name} ({task.agent_type})")
                
                # Update task status to in_progress
                self.project_manager.update_task_status(
                    project.project_id, 
                    task.id, 
                    TaskStatus.IN_PROGRESS
                )
                
                # Execute task
                task_result = await self._execute_task(task)
                
                # Update task status
                self.project_manager.update_task_status(
                    project.project_id,
                    task.id,
                    TaskStatus.COMPLETED if task_result["success"] else TaskStatus.FAILED,
                    task_result.get("result")
                )
                
                # Add artifacts
                if task_result.get("artifacts"):
                    for artifact in task_result["artifacts"]:
                        self.project_manager.add_artifact(project.project_id, artifact)
                
                if not task_result["success"]:
                    print(f"❌ Task failed: {task_result.get('error', 'Unknown error')}")
                    # For now, continue with other tasks
                    # In a full implementation, we might want to retry or handle failures differently
        
        # Get final project state
        final_project = self.project_manager.get_project(project.project_id)
        completed_tasks = sum(1 for task in final_project.tasks if task.status == TaskStatus.COMPLETED)
        failed_tasks = sum(1 for task in final_project.tasks if task.status == TaskStatus.FAILED)
        
        return {
            "success": failed_tasks == 0,
            "completed_tasks": completed_tasks,
            "failed_tasks": failed_tasks,
            "total_tasks": len(final_project.tasks),
            "iterations": iteration
        }
    
    async def _execute_task(self, task: Task) -> Dict[str, Any]:
        """
        Execute a single task by delegating to appropriate agent
        
        Args:
            task: Task to execute
            
        Returns:
            Dict containing task execution result
        """
        try:
            agent_endpoint = self.agent_endpoints.get(task.agent_type)
            if not agent_endpoint:
                return {
                    "success": False,
                    "error": f"No agent endpoint found for type: {task.agent_type}"
                }
            
            # Create agent request
            request = AgentRequest(
                task_id=task.id,
                task_type=task.task_type,
                goal=task.description,
                context={"project_id": task.id.split('-')[0]},  # Extract project ID
                parameters=task.parameters,
                priority=task.priority,
                dependencies=task.dependencies,
                outputs=task.outputs
            )
            
            # Call agent
            response = await self.http_client.post(
                f"{agent_endpoint}/execute",
                json=request.dict(),
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "success": result.get("status") == "completed",
                    "result": TaskResult(
                        task_id=task.id,
                        status=TaskStatus.COMPLETED if result.get("status") == "completed" else TaskStatus.FAILED,
                        artifacts=result.get("artifacts", []),
                        metadata=result.get("metadata", {}),
                        error=result.get("error"),
                        execution_time_ms=result.get("execution_time_ms", 0)
                    ),
                    "artifacts": result.get("artifacts", [])
                }
            else:
                return {
                    "success": False,
                    "error": f"Agent returned status {response.status_code}: {response.text}"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Task execution failed: {str(e)}"
            }
    
    async def close(self):
        """Close HTTP client"""
        await self.http_client.aclose()


# FastAPI application
app = FastAPI(
    title="Brigade Orchestrator API",
    description="Autonomous Development Workflow Management",
    version="1.0.0"
)

# Global orchestrator instance
orchestrator = BrigadeOrchestrator()


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "orchestrator": "Brigade Orchestrator",
        "execution_count": orchestrator.execution_count
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "orchestrator": "Brigade Orchestrator"}


@app.post("/execute")
async def execute_development(request: Dict[str, Any]):
    """
    Execute autonomous development workflow
    
    This endpoint receives high-level goals and orchestrates the entire
    development process from planning to deployment.
    """
    try:
        goal = request.get("goal")
        context = request.get("context", {})
        
        if not goal:
            raise HTTPException(status_code=400, detail="Goal is required")
        
        result = await orchestrator.execute_development_plan(goal, context)
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/projects")
async def list_projects():
    """List all active projects"""
    return {
        "projects": list(orchestrator.project_manager.active_projects.keys()),
        "total": len(orchestrator.project_manager.active_projects)
    }


@app.get("/projects/{project_id}")
async def get_project(project_id: str):
    """Get project details"""
    project = orchestrator.project_manager.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    summary = orchestrator.project_manager.get_project_summary(project_id)
    return {
        "project": project.dict(),
        "summary": summary
    }


@app.get("/stats")
async def get_stats():
    """Get orchestrator statistics"""
    return {
        "orchestrator": "Brigade Orchestrator",
        "execution_count": orchestrator.execution_count,
        "active_projects": len(orchestrator.project_manager.active_projects),
        "agent_endpoints": orchestrator.agent_endpoints
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8009,
        reload=True,
        log_level="info"
    )

