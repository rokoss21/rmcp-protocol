#!/usr/bin/env python3
"""
Test Brigade Orchestrator - Phase 6
===================================

This test demonstrates the Brigade Orchestrator's ability to coordinate
autonomous development workflow.

Usage:
    python test_brigade_orchestrator.py
"""

import asyncio
import sys
from pathlib import Path

# Add autonomous_agents to path
sys.path.insert(0, str(Path(__file__).parent / "autonomous_agents"))

from autonomous_agents.orchestrator.main import BrigadeOrchestrator


async def test_brigade_orchestrator():
    """Test the Brigade Orchestrator"""
    print("🚀 Testing Brigade Orchestrator - Phase 6")
    print("=" * 50)
    
    # Create orchestrator instance
    orchestrator = BrigadeOrchestrator()
    
    # Test goal
    goal = "Create and deploy an MCP server for the 'cowsay' command-line utility. It should accept a 'text' parameter and return the ASCII art as a string."
    context = {
        "project_type": "mcp_server",
        "target_utility": "cowsay",
        "deployment_target": "docker"
    }
    
    print(f"📋 Goal: {goal}")
    print(f"   Context: {context}")
    
    # Execute development plan
    print(f"\n🚀 Executing autonomous development workflow...")
    result = await orchestrator.execute_development_plan(goal, context)
    
    # Analyze results
    print(f"\n📊 Results:")
    print(f"   Success: {result.get('success', False)}")
    print(f"   Project ID: {result.get('project_id', 'N/A')}")
    print(f"   Execution Time: {result.get('execution_time_ms', 0)}ms")
    
    if result.get("success"):
        print("✅ Brigade Orchestrator executed successfully!")
        
        # Display project summary
        if "project_summary" in result:
            summary = result["project_summary"]
            print(f"\n📋 Project Summary:")
            print(f"   Name: {summary.get('name', 'N/A')}")
            print(f"   Status: {summary.get('status', 'N/A')}")
            print(f"   Total Tasks: {summary.get('total_tasks', 0)}")
            print(f"   Completed: {summary.get('completed_tasks', 0)}")
            print(f"   Failed: {summary.get('failed_tasks', 0)}")
            print(f"   Progress: {summary.get('progress_percentage', 0):.1f}%")
            print(f"   Artifacts: {summary.get('artifacts_count', 0)}")
        
        # Display execution result
        if "execution_result" in result:
            exec_result = result["execution_result"]
            print(f"\n🔄 Execution Result:")
            print(f"   Completed Tasks: {exec_result.get('completed_tasks', 0)}")
            print(f"   Failed Tasks: {exec_result.get('failed_tasks', 0)}")
            print(f"   Total Tasks: {exec_result.get('total_tasks', 0)}")
            print(f"   Iterations: {exec_result.get('iterations', 0)}")
        
        print(f"\n🎉 BRIGADE ORCHESTRATOR SUCCESS!")
        print("   ✅ Successfully coordinated development workflow")
        print("   ✅ Managed project state and task execution")
        print("   ✅ Ready for Prometheus Experiment")
        
    else:
        print(f"❌ Brigade Orchestrator failed: {result.get('error', 'Unknown error')}")
    
    # Cleanup
    await orchestrator.close()
    
    print(f"\n🧹 Test completed")
    print(f"🚀 Brigade Orchestrator Test: {'SUCCESS' if result.get('success') else 'FAILED'}")


if __name__ == "__main__":
    asyncio.run(test_brigade_orchestrator())

