#!/usr/bin/env python3
"""
Test Architect Agent - Phase 6
==============================

This test demonstrates the Architect Agent's ability to create detailed
development plans for MCP server creation.

Usage:
    python test_architect_agent.py
"""

import asyncio
import sys
from pathlib import Path

# Add autonomous_agents to path
sys.path.insert(0, str(Path(__file__).parent / "autonomous_agents"))

from autonomous_agents.architect.main import ArchitectAgent
from autonomous_agents.base.models import AgentRequest, TaskPriority


async def test_architect_agent():
    """Test the Architect Agent"""
    print("🚀 Testing Architect Agent - Phase 6")
    print("=" * 50)
    
    # Create agent instance
    agent = ArchitectAgent()
    
    # Create test request
    request = AgentRequest(
        task_id="test-architect-001",
        task_type="system_design",
        goal="Create and deploy an MCP server for the 'cowsay' command-line utility. It should accept a 'text' parameter and return the ASCII art as a string.",
        context={
            "project_type": "mcp_server",
            "target_utility": "cowsay",
            "deployment_target": "docker"
        },
        parameters={
            "include_tests": True,
            "include_docker": True,
            "include_documentation": True
        },
        priority=TaskPriority.HIGH
    )
    
    print(f"📋 Goal: {request.goal}")
    print(f"   Task ID: {request.task_id}")
    print(f"   Priority: {request.priority}")
    
    # Execute the request
    print("\n🚀 Executing architect request...")
    response = await agent.execute(request)
    
    # Analyze results
    print(f"\n📊 Results:")
    print(f"   Status: {response.status}")
    print(f"   Execution Time: {response.execution_time_ms}ms")
    
    if response.status == "completed":
        print("✅ Architect Agent executed successfully!")
        
        # Display development plan
        if response.result and "development_plan" in response.result:
            plan = response.result["development_plan"]
            print(f"\n📋 Development Plan:")
            print(f"   Project ID: {plan.project_id}")
            print(f"   Goal: {plan.goal}")
            print(f"   Tasks: {len(plan.tasks)}")
            print(f"   Estimated Duration: {plan.estimated_duration_ms}ms")
            
            print(f"\n🔧 Tasks:")
            for task in plan.tasks:
                print(f"   - {task.name} ({task.agent_type})")
                if task.dependencies:
                    print(f"     Dependencies: {', '.join(task.dependencies)}")
            
            print(f"\n🏗️ Architecture:")
            if "architecture" in response.result:
                arch = response.result["architecture"]
                print(f"   Project Structure: {len(arch.get('project_structure', {}))} files")
                print(f"   API Endpoints: {len(arch.get('api_endpoints', []))}")
                print(f"   Tool Definitions: {len(arch.get('tool_definitions', []))}")
            
            print(f"\n🔍 Research Results:")
            if "research_result" in response.result:
                research = response.result["research_result"]
                print(f"   Utility: {research.get('utility_name', 'unknown')}")
                print(f"   Description: {research.get('description', 'N/A')}")
                print(f"   Dependencies: {research.get('dependencies', [])}")
            
            print(f"\n🎉 ARCHITECT AGENT SUCCESS!")
            print("   ✅ Successfully analyzed the goal")
            print("   ✅ Created detailed development plan")
            print("   ✅ Generated task DAG for execution")
            print("   ✅ Ready for Brigade Orchestrator")
            
        else:
            print("⚠️  No development plan found in result")
    else:
        print(f"❌ Architect Agent failed: {response.error}")
    
    # Cleanup
    await agent.close()
    
    print(f"\n🧹 Test completed")
    print(f"🚀 Architect Agent Test: {'SUCCESS' if response.status == 'completed' else 'FAILED'}")


if __name__ == "__main__":
    asyncio.run(test_architect_agent())

