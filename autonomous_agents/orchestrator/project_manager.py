"""
Project State Management for Brigade Orchestrator
"""

import os
import json
import shutil
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime

from ..base.models import ProjectState, Task, TaskStatus, TaskResult, DevelopmentPlan


class ProjectManager:
    """
    Manages project state and file system operations for autonomous development
    
    Responsibilities:
    - Create and manage project directories
    - Track project state and task progress
    - Manage file artifacts
    - Persist project state
    """
    
    def __init__(self, base_dir: str = "/tmp/autonomous_projects"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.active_projects: Dict[str, ProjectState] = {}
    
    def create_project(self, plan: DevelopmentPlan) -> ProjectState:
        """
        Create a new project from development plan
        
        Args:
            plan: Development plan from architect
            
        Returns:
            ProjectState: Created project state
        """
        project_id = plan.project_id
        project_dir = self.base_dir / project_id
        project_dir.mkdir(exist_ok=True)
        
        # Create project state
        project_state = ProjectState(
            project_id=project_id,
            name=f"MCP Server for {plan.goal}",
            goal=plan.goal,
            status="initialized",
            tasks=plan.tasks,
            metadata={
                "plan": plan.dict(),
                "project_dir": str(project_dir),
                "created_by": "brigade_orchestrator"
            }
        )
        
        # Save initial state
        self._save_project_state(project_state)
        self.active_projects[project_id] = project_state
        
        return project_state
    
    def get_project(self, project_id: str) -> Optional[ProjectState]:
        """Get project state by ID"""
        if project_id in self.active_projects:
            return self.active_projects[project_id]
        
        # Try to load from disk
        project_dir = self.base_dir / project_id
        state_file = project_dir / "project_state.json"
        
        if state_file.exists():
            with open(state_file, 'r') as f:
                data = json.load(f)
                project_state = ProjectState(**data)
                self.active_projects[project_id] = project_state
                return project_state
        
        return None
    
    def update_task_status(
        self, 
        project_id: str, 
        task_id: str, 
        status: TaskStatus,
        result: Optional[TaskResult] = None
    ) -> bool:
        """
        Update task status in project
        
        Args:
            project_id: Project identifier
            task_id: Task identifier
            status: New task status
            result: Task result if completed
            
        Returns:
            bool: True if updated successfully
        """
        project = self.get_project(project_id)
        if not project:
            return False
        
        # Find and update task
        for task in project.tasks:
            if task.id == task_id:
                task.status = status
                if result:
                    task.result = result
                
                if status == TaskStatus.IN_PROGRESS:
                    task.started_at = datetime.utcnow()
                elif status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                    task.completed_at = datetime.utcnow()
                
                break
        
        # Update project status
        self._update_project_status(project)
        
        # Save state
        self._save_project_state(project)
        return True
    
    def add_artifact(
        self, 
        project_id: str, 
        artifact: Dict[str, Any]
    ) -> bool:
        """
        Add artifact to project
        
        Args:
            project_id: Project identifier
            artifact: Artifact data
            
        Returns:
            bool: True if added successfully
        """
        project = self.get_project(project_id)
        if not project:
            return False
        
        project.artifacts.append(artifact)
        project.updated_at = datetime.utcnow()
        
        # Save artifact to file system
        self._save_artifact(project, artifact)
        
        # Save state
        self._save_project_state(project)
        return True
    
    def get_ready_tasks(self, project_id: str) -> List[Task]:
        """
        Get tasks that are ready to execute (dependencies satisfied)
        
        Args:
            project_id: Project identifier
            
        Returns:
            List of ready tasks
        """
        project = self.get_project(project_id)
        if not project:
            return []
        
        ready_tasks = []
        
        for task in project.tasks:
            if task.status != TaskStatus.PENDING:
                continue
            
            # Check if all dependencies are completed
            dependencies_satisfied = True
            for dep_id in task.dependencies:
                dep_task = next((t for t in project.tasks if t.id == dep_id), None)
                if not dep_task or dep_task.status != TaskStatus.COMPLETED:
                    dependencies_satisfied = False
                    break
            
            if dependencies_satisfied:
                ready_tasks.append(task)
        
        return ready_tasks
    
    def get_project_directory(self, project_id: str) -> Optional[Path]:
        """Get project directory path"""
        project = self.get_project(project_id)
        if not project:
            return None
        
        return Path(project.metadata.get("project_dir", self.base_dir / project_id))
    
    def cleanup_project(self, project_id: str) -> bool:
        """
        Clean up project directory and state
        
        Args:
            project_id: Project identifier
            
        Returns:
            bool: True if cleaned successfully
        """
        project = self.get_project(project_id)
        if not project:
            return False
        
        # Remove from active projects
        if project_id in self.active_projects:
            del self.active_projects[project_id]
        
        # Remove project directory
        project_dir = self.get_project_directory(project_id)
        if project_dir and project_dir.exists():
            shutil.rmtree(project_dir)
        
        return True
    
    def _save_project_state(self, project: ProjectState):
        """Save project state to disk"""
        project_dir = Path(project.metadata.get("project_dir", self.base_dir / project.project_id))
        state_file = project_dir / "project_state.json"
        
        with open(state_file, 'w') as f:
            json.dump(project.dict(), f, indent=2, default=str)
    
    def _save_artifact(self, project: ProjectState, artifact: Dict[str, Any]):
        """Save artifact to file system"""
        project_dir = Path(project.metadata.get("project_dir", self.base_dir / project.project_id))
        
        artifact_type = artifact.get("type", "unknown")
        artifact_name = artifact.get("name", "artifact")
        
        if artifact_type == "file":
            file_path = project_dir / artifact_name
            content = artifact.get("content", "")
            
            # Ensure directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'w') as f:
                f.write(content)
            
            artifact["file_path"] = str(file_path)
        
        elif artifact_type == "json":
            json_path = project_dir / f"{artifact_name}.json"
            
            with open(json_path, 'w') as f:
                json.dump(artifact.get("data", {}), f, indent=2)
            
            artifact["file_path"] = str(json_path)
    
    def _update_project_status(self, project: ProjectState):
        """Update overall project status based on task states"""
        total_tasks = len(project.tasks)
        completed_tasks = sum(1 for task in project.tasks if task.status == TaskStatus.COMPLETED)
        failed_tasks = sum(1 for task in project.tasks if task.status == TaskStatus.FAILED)
        in_progress_tasks = sum(1 for task in project.tasks if task.status == TaskStatus.IN_PROGRESS)
        
        if failed_tasks > 0:
            project.status = "failed"
        elif completed_tasks == total_tasks:
            project.status = "completed"
        elif in_progress_tasks > 0 or completed_tasks > 0:
            project.status = "in_progress"
        else:
            project.status = "pending"
        
        project.updated_at = datetime.utcnow()
    
    def get_project_summary(self, project_id: str) -> Dict[str, Any]:
        """Get project summary statistics"""
        project = self.get_project(project_id)
        if not project:
            return {}
        
        total_tasks = len(project.tasks)
        completed_tasks = sum(1 for task in project.tasks if task.status == TaskStatus.COMPLETED)
        failed_tasks = sum(1 for task in project.tasks if task.status == TaskStatus.FAILED)
        pending_tasks = sum(1 for task in project.tasks if task.status == TaskStatus.PENDING)
        in_progress_tasks = sum(1 for task in project.tasks if task.status == TaskStatus.IN_PROGRESS)
        
        return {
            "project_id": project_id,
            "name": project.name,
            "status": project.status,
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "failed_tasks": failed_tasks,
            "pending_tasks": pending_tasks,
            "in_progress_tasks": in_progress_tasks,
            "progress_percentage": (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0,
            "artifacts_count": len(project.artifacts),
            "created_at": project.created_at,
            "updated_at": project.updated_at
        }

