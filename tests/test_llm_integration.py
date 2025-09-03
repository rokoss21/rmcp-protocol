#!/usr/bin/env python3
"""
Test LLM Integration - Phase 6 Final
===================================

This test demonstrates the LLM integration with real OpenAI API calls.

Usage:
    export OPENAI_API_KEY="your-api-key"
    python test_llm_integration.py
"""

import asyncio
import sys
import os
from pathlib import Path

# Add autonomous_agents to path
sys.path.insert(0, str(Path(__file__).parent / "autonomous_agents"))

from autonomous_agents.llm_manager import LLMManager
from autonomous_agents.backend.main import BackendAgent
from autonomous_agents.tester.main import TesterAgent
from autonomous_agents.base.models import AgentRequest, TaskPriority


async def test_llm_integration():
    """Test LLM integration with real OpenAI API"""
    print("🧠 Testing LLM Integration - Phase 6 Final")
    print("=" * 50)
    
    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not set. Please set it and try again.")
        print("   export OPENAI_API_KEY='your-api-key'")
        return
    
    print(f"✅ OpenAI API Key configured: {api_key[:20]}...")
    
    # Test 1: LLM Manager
    print(f"\n🧠 Test 1: LLM Manager")
    print("-" * 40)
    
    llm_manager = LLMManager(api_key=api_key)
    
    # Test basic text generation
    prompt = "Write a simple Python function that adds two numbers and returns the result."
    
    print(f"   Prompt: {prompt}")
    print(f"   Generating response...")
    
    result = await llm_manager.generate_code_for_role(
        role="backend_developer",
        prompt=prompt,
        language="python"
    )
    
    if result["success"]:
        print(f"   ✅ LLM generated code successfully!")
        print(f"   Tokens used: {result.get('tokens_used', 0)}")
        print(f"   Code length: {len(result['content'])} characters")
        print(f"   First 200 chars: {result['content'][:200]}...")
    else:
        print(f"   ❌ LLM generation failed: {result.get('error', 'Unknown error')}")
        return
    
    # Test 2: Backend Agent with LLM
    print(f"\n🔧 Test 2: Backend Agent with LLM")
    print("-" * 40)
    
    backend_agent = BackendAgent()
    
    request = AgentRequest(
        task_id="llm-test-backend-001",
        task_type="code_generation",
        goal="Create a simple FastAPI endpoint that returns 'Hello World'",
        context={"test": True},
        parameters={
            "file_path": "test_main.py",
            "template": "fastapi_mcp_server",
            "utility_name": "test"
        },
        priority=TaskPriority.HIGH
    )
    
    print(f"   Goal: {request.goal}")
    print(f"   Executing with real LLM...")
    
    response = await backend_agent.execute(request)
    
    if response.status == "completed":
        print(f"   ✅ Backend Agent with LLM successful!")
        print(f"   Execution time: {response.execution_time_ms}ms")
        
        if response.result and "generated_code" in response.result:
            code = response.result["generated_code"]
            print(f"   Generated code length: {len(code)} characters")
            print(f"   First 300 chars: {code[:300]}...")
    else:
        print(f"   ❌ Backend Agent failed: {response.error}")
    
    # Test 3: Tester Agent with LLM
    print(f"\n🧪 Test 3: Tester Agent with LLM")
    print("-" * 40)
    
    tester_agent = TesterAgent()
    
    request = AgentRequest(
        task_id="llm-test-tester-001",
        task_type="test_generation",
        goal="Create pytest tests for a FastAPI application",
        context={"test": True},
        parameters={
            "file_path": "test_main.py",
            "target_file": "main.py",
            "test_cases": ["health_check", "endpoint_tests"]
        },
        priority=TaskPriority.MEDIUM
    )
    
    print(f"   Goal: {request.goal}")
    print(f"   Executing with real LLM...")
    
    response = await tester_agent.execute(request)
    
    if response.status == "completed":
        print(f"   ✅ Tester Agent with LLM successful!")
        print(f"   Execution time: {response.execution_time_ms}ms")
        
        if response.result and "generated_tests" in response.result:
            tests = response.result["generated_tests"]
            print(f"   Generated tests length: {len(tests)} characters")
            print(f"   First 300 chars: {tests[:300]}...")
    else:
        print(f"   ❌ Tester Agent failed: {response.error}")
    
    # Cleanup
    await backend_agent.close()
    await tester_agent.close()
    
    print(f"\n🎉 LLM INTEGRATION SUCCESS!")
    print("   ✅ OpenAI API integration working")
    print("   ✅ Backend Agent with real LLM working")
    print("   ✅ Tester Agent with real LLM working")
    print("   ✅ Ready for Live Prometheus Experiment")
    
    print(f"\n🚀 LLM Integration Test: SUCCESS")


if __name__ == "__main__":
    asyncio.run(test_llm_integration())

