#!/usr/bin/env python3
"""
Test Backend Agent - Phase 6 Final
==================================

This test demonstrates the Backend Agent's ability to generate
production-ready Python code using LLM integration.

Usage:
    python test_backend_agent.py
"""

import asyncio
import sys
from pathlib import Path

# Add autonomous_agents to path
sys.path.insert(0, str(Path(__file__).parent / "autonomous_agents"))

from autonomous_agents.backend.main import BackendAgent
from autonomous_agents.base.models import AgentRequest, TaskPriority


async def test_backend_agent():
    """Test the Backend Agent"""
    print("🚀 Testing Backend Agent - Phase 6 Final")
    print("=" * 50)
    
    # Create agent instance
    agent = BackendAgent()
    
    # Test 1: Generate FastAPI application
    print("🔧 Test 1: Generate FastAPI MCP Server")
    print("-" * 40)
    
    request1 = AgentRequest(
        task_id="backend-test-001",
        task_type="code_generation",
        goal="Create a FastAPI MCP server for cowsay utility",
        context={
            "project_type": "mcp_server",
            "target_utility": "cowsay"
        },
        parameters={
            "file_path": "main.py",
            "template": "fastapi_mcp_server",
            "utility_name": "cowsay",
            "tool_definitions": [
                {
                    "name": "cowsay.execute",
                    "description": "Execute cowsay command",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string", "description": "Text to display"}
                        },
                        "required": ["text"]
                    }
                }
            ]
        },
        priority=TaskPriority.HIGH
    )
    
    print(f"   Goal: {request1.goal}")
    print(f"   Template: {request1.parameters.get('template')}")
    
    response1 = await agent.execute(request1)
    
    print(f"   Status: {response1.status}")
    print(f"   Execution Time: {response1.execution_time_ms}ms")
    
    if response1.status == "completed":
        print("   ✅ FastAPI code generated successfully!")
        
        # Display generated code snippet
        if response1.result and "generated_code" in response1.result:
            code = response1.result["generated_code"]
            print(f"   Code Length: {len(code)} characters")
            print(f"   First 200 chars: {code[:200]}...")
        
        # Display artifacts
        if response1.artifacts:
            print(f"   Artifacts: {len(response1.artifacts)}")
            for artifact in response1.artifacts:
                print(f"     - {artifact.get('name', 'unknown')}")
    else:
        print(f"   ❌ Failed: {response1.error}")
    
    # Test 2: Generate Pydantic models
    print(f"\n🔧 Test 2: Generate Pydantic Models")
    print("-" * 40)
    
    request2 = AgentRequest(
        task_id="backend-test-002",
        task_type="code_generation",
        goal="Create Pydantic models for cowsay MCP server",
        context={
            "project_type": "mcp_server",
            "target_utility": "cowsay"
        },
        parameters={
            "file_path": "models.py",
            "template": "pydantic_models",
            "input_schema": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to display"}
                },
                "required": ["text"]
            },
            "output_schema": {
                "type": "object",
                "properties": {
                    "result": {"type": "string", "description": "Cowsay output"},
                    "success": {"type": "boolean", "description": "Success status"}
                }
            }
        },
        priority=TaskPriority.MEDIUM
    )
    
    print(f"   Goal: {request2.goal}")
    print(f"   Template: {request2.parameters.get('template')}")
    
    response2 = await agent.execute(request2)
    
    print(f"   Status: {response2.status}")
    print(f"   Execution Time: {response2.execution_time_ms}ms")
    
    if response2.status == "completed":
        print("   ✅ Pydantic models generated successfully!")
        
        # Display generated code snippet
        if response2.result and "generated_code" in response2.result:
            code = response2.result["generated_code"]
            print(f"   Code Length: {len(code)} characters")
            print(f"   First 200 chars: {code[:200]}...")
    else:
        print(f"   ❌ Failed: {response2.error}")
    
    # Test 3: Generate requirements.txt
    print(f"\n🔧 Test 3: Generate Requirements File")
    print("-" * 40)
    
    request3 = AgentRequest(
        task_id="backend-test-003",
        task_type="file_creation",
        goal="Create requirements.txt for MCP server",
        context={
            "project_type": "mcp_server"
        },
        parameters={
            "file_path": "requirements.txt",
            "project_type": "mcp_server",
            "dependencies": ["fastapi>=0.104.0", "uvicorn>=0.24.0", "pydantic>=2.0.0"],
            "python_version": "3.11"
        },
        priority=TaskPriority.LOW
    )
    
    print(f"   Goal: {request3.goal}")
    print(f"   File: {request3.parameters.get('file_path')}")
    
    response3 = await agent.execute(request3)
    
    print(f"   Status: {response3.status}")
    print(f"   Execution Time: {response3.execution_time_ms}ms")
    
    if response3.status == "completed":
        print("   ✅ Requirements file generated successfully!")
        
        # Display generated content
        if response3.result and "generated_code" in response3.result:
            content = response3.result["generated_code"]
            print(f"   Content: {content}")
    else:
        print(f"   ❌ Failed: {response3.error}")
    
    # Summary
    print(f"\n📊 Test Summary:")
    print(f"   Total Tests: 3")
    print(f"   Successful: {sum(1 for r in [response1, response2, response3] if r.status == 'completed')}")
    print(f"   Failed: {sum(1 for r in [response1, response2, response3] if r.status != 'completed')}")
    
    # Cleanup
    await agent.close()
    
    success_count = sum(1 for r in [response1, response2, response3] if r.status == "completed")
    
    if success_count == 3:
        print(f"\n🎉 BACKEND AGENT SUCCESS!")
        print("   ✅ FastAPI code generation working")
        print("   ✅ Pydantic models generation working")
        print("   ✅ Requirements file generation working")
        print("   ✅ Ready for Brigade integration")
    else:
        print(f"\n❌ BACKEND AGENT PARTIAL SUCCESS")
        print(f"   {success_count}/3 tests passed")
    
    print(f"\n🚀 Backend Agent Test: {'SUCCESS' if success_count == 3 else 'PARTIAL'}")


if __name__ == "__main__":
    asyncio.run(test_backend_agent())

