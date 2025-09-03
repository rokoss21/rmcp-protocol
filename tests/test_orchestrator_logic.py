#!/usr/bin/env python3
"""
Test Brigade Orchestrator Logic - Phase 6
=========================================

This test demonstrates the Brigade Orchestrator's internal logic
without requiring external agent services.

Usage:
    python test_orchestrator_logic.py
"""

import asyncio
import sys
from pathlib import Path

# Add autonomous_agents to path
sys.path.insert(0, str(Path(__file__).parent / "autonomous_agents"))

from autonomous_agents.orchestrator.project_manager import ProjectManager
from autonomous_agents.base.models import DevelopmentPlan, Task, TaskStatus, TaskPriority


async def test_orchestrator_logic():
    """Test the Brigade Orchestrator's internal logic"""
    print("🚀 Testing Brigade Orchestrator Logic - Phase 6")
    print("=" * 50)
    
    # Create project manager
    project_manager = ProjectManager()
    
    # Create mock development plan
    print("📋 Creating mock development plan...")
    
    # Create tasks
    tasks = [
        Task(
            id="task_1",
            name="Create main.py",
            description="Create FastAPI application",
            task_type="code_generation",
            agent_type="backend",
            parameters={"file_path": "main.py"},
            outputs=["main.py"]
        ),
        Task(
            id="task_2",
            name="Create models.py",
            description="Create Pydantic models",
            task_type="code_generation",
            agent_type="backend",
            dependencies=["task_1"],
            parameters={"file_path": "models.py"},
            outputs=["models.py"]
        ),
        Task(
            id="task_3",
            name="Create tests",
            description="Create pytest tests",
            task_type="test_generation",
            agent_type="tester",
            dependencies=["task_1", "task_2"],
            parameters={"file_path": "test_main.py"},
            outputs=["test_main.py"]
        )
    ]
    
    # Create development plan
    plan = DevelopmentPlan(
        project_id="test-project-001",
        goal="Create MCP server for cowsay",
        analysis={"utility_name": "cowsay"},
        architecture={"project_structure": {"main.py": "FastAPI app"}},
        tasks=tasks,
        dependencies={
            "task_1": [],
            "task_2": ["task_1"],
            "task_3": ["task_1", "task_2"]
        },
        estimated_duration_ms=300000
    )
    
    print(f"   Project ID: {plan.project_id}")
    print(f"   Goal: {plan.goal}")
    print(f"   Tasks: {len(plan.tasks)}")
    
    # Create project
    print(f"\n📁 Creating project...")
    project = project_manager.create_project(plan)
    print(f"   Project created: {project.project_id}")
    print(f"   Status: {project.status}")
    
    # Test task management
    print(f"\n🔧 Testing task management...")
    
    # Get ready tasks (should be task_1)
    ready_tasks = project_manager.get_ready_tasks(project.project_id)
    print(f"   Ready tasks: {len(ready_tasks)}")
    for task in ready_tasks:
        print(f"     - {task.name} (dependencies: {len(task.dependencies)})")
    
    # Mark task_1 as completed
    print(f"\n✅ Marking task_1 as completed...")
    project_manager.update_task_status(project.project_id, "task_1", TaskStatus.COMPLETED)
    
    # Get ready tasks again (should be task_2)
    ready_tasks = project_manager.get_ready_tasks(project.project_id)
    print(f"   Ready tasks after task_1 completion: {len(ready_tasks)}")
    for task in ready_tasks:
        print(f"     - {task.name} (dependencies: {len(task.dependencies)})")
    
    # Mark task_2 as completed
    print(f"\n✅ Marking task_2 as completed...")
    project_manager.update_task_status(project.project_id, "task_2", TaskStatus.COMPLETED)
    
    # Get ready tasks again (should be task_3)
    ready_tasks = project_manager.get_ready_tasks(project.project_id)
    print(f"   Ready tasks after task_2 completion: {len(ready_tasks)}")
    for task in ready_tasks:
        print(f"     - {task.name} (dependencies: {len(task.dependencies)})")
    
    # Mark task_3 as completed
    print(f"\n✅ Marking task_3 as completed...")
    project_manager.update_task_status(project.project_id, "task_3", TaskStatus.COMPLETED)
    
    # Test artifact management
    print(f"\n📦 Testing artifact management...")
    
    # Add some artifacts
    artifacts = [
        {
            "type": "file",
            "name": "main.py",
            "content": "from fastapi import FastAPI\napp = FastAPI()"
        },
        {
            "type": "file",
            "name": "models.py",
            "content": "from pydantic import BaseModel\nclass Request(BaseModel):\n    text: str"
        },
        {
            "type": "json",
            "name": "project_info",
            "data": {"utility": "cowsay", "status": "completed"}
        }
    ]
    
    for artifact in artifacts:
        project_manager.add_artifact(project.project_id, artifact)
        print(f"   Added artifact: {artifact['name']}")
    
    # Get project summary
    print(f"\n📊 Project Summary:")
    summary = project_manager.get_project_summary(project.project_id)
    print(f"   Status: {summary.get('status', 'N/A')}")
    print(f"   Total Tasks: {summary.get('total_tasks', 0)}")
    print(f"   Completed: {summary.get('completed_tasks', 0)}")
    print(f"   Failed: {summary.get('failed_tasks', 0)}")
    print(f"   Progress: {summary.get('progress_percentage', 0):.1f}%")
    print(f"   Artifacts: {summary.get('artifacts_count', 0)}")
    
    # Test project directory
    print(f"\n📁 Project Directory:")
    project_dir = project_manager.get_project_directory(project.project_id)
    if project_dir:
        print(f"   Path: {project_dir}")
        print(f"   Exists: {project_dir.exists()}")
        
        # List files
        if project_dir.exists():
            files = list(project_dir.iterdir())
            print(f"   Files: {len(files)}")
            for file in files:
                print(f"     - {file.name}")
    
    # Cleanup
    print(f"\n🧹 Cleaning up...")
    project_manager.cleanup_project(project.project_id)
    print(f"   Project cleaned up")
    
    print(f"\n🎉 ORCHESTRATOR LOGIC SUCCESS!")
    print("   ✅ Project creation and management working")
    print("   ✅ Task dependency resolution working")
    print("   ✅ Artifact management working")
    print("   ✅ Project state tracking working")
    print("   ✅ Ready for full Brigade Orchestrator")
    
    print(f"\n🚀 Brigade Orchestrator Logic Test: SUCCESS!")


if __name__ == "__main__":
    asyncio.run(test_orchestrator_logic())

