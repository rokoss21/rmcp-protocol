#!/usr/bin/env python3
"""
Test All Agents - Phase 6 Final
===============================

This test demonstrates all agents working together in the Brigade.

Usage:
    python test_all_agents.py
"""

import asyncio
import sys
from pathlib import Path

# Add autonomous_agents to path
sys.path.insert(0, str(Path(__file__).parent / "autonomous_agents"))

from autonomous_agents.architect.main import ArchitectAgent
from autonomous_agents.backend.main import BackendAgent
from autonomous_agents.tester.main import TesterAgent
from autonomous_agents.validator.main import ValidatorAgent
from autonomous_agents.devops.main import DevOpsAgent
from autonomous_agents.base.models import AgentRequest, TaskPriority


async def test_all_agents():
    """Test all agents in the Brigade"""
    print("🚀 Testing All Agents - Phase 6 Final")
    print("=" * 50)
    
    # Create all agents
    agents = {
        "architect": ArchitectAgent(),
        "backend": BackendAgent(),
        "tester": TesterAgent(),
        "validator": ValidatorAgent(),
        "devops": DevOpsAgent()
    }
    
    # Test each agent
    results = {}
    
    for agent_name, agent in agents.items():
        print(f"\n🔧 Testing {agent_name.upper()} Agent")
        print("-" * 40)
        
        try:
            # Create test request
            request = AgentRequest(
                task_id=f"test-{agent_name}-001",
                task_type="test",
                goal=f"Test {agent_name} agent functionality",
                context={"test": True},
                parameters={"test_mode": True},
                priority=TaskPriority.MEDIUM
            )
            
            # Execute agent
            response = await agent.execute(request)
            
            results[agent_name] = {
                "success": response.status == "completed",
                "execution_time": response.execution_time_ms,
                "error": response.error
            }
            
            status_icon = "✅" if response.status == "completed" else "❌"
            print(f"   {status_icon} Status: {response.status}")
            print(f"   ⏱️  Execution Time: {response.execution_time_ms}ms")
            
            if response.status != "completed":
                print(f"   ❌ Error: {response.error}")
            
        except Exception as e:
            results[agent_name] = {
                "success": False,
                "execution_time": 0,
                "error": str(e)
            }
            print(f"   ❌ Exception: {e}")
    
    # Summary
    print(f"\n📊 Test Summary:")
    print(f"   Total Agents: {len(agents)}")
    print(f"   Successful: {sum(1 for r in results.values() if r['success'])}")
    print(f"   Failed: {sum(1 for r in results.values() if not r['success'])}")
    
    print(f"\n🔧 Agent Results:")
    for agent_name, result in results.items():
        status_icon = "✅" if result["success"] else "❌"
        print(f"   {status_icon} {agent_name}: {result['execution_time']}ms")
        if not result["success"]:
            print(f"      Error: {result['error']}")
    
    # Cleanup
    for agent in agents.values():
        await agent.close()
    
    success_count = sum(1 for r in results.values() if r["success"])
    
    if success_count == len(agents):
        print(f"\n🎉 ALL AGENTS SUCCESS!")
        print("   ✅ Architect Agent working")
        print("   ✅ Backend Agent working")
        print("   ✅ Tester Agent working")
        print("   ✅ Validator Agent working")
        print("   ✅ DevOps Agent working")
        print("   ✅ Brigade ready for live experiment")
    else:
        print(f"\n❌ PARTIAL SUCCESS")
        print(f"   {success_count}/{len(agents)} agents working")
    
    print(f"\n🚀 All Agents Test: {'SUCCESS' if success_count == len(agents) else 'PARTIAL'}")


if __name__ == "__main__":
    asyncio.run(test_all_agents())

