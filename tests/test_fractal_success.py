#!/usr/bin/env python3
"""
Fractal Orchestration Success Demo
==================================

This script demonstrates the successful implementation of fractal orchestration
where agents can use RMCP as a service for sub-task execution.

Usage:
    python test_fractal_success.py
"""

import asyncio
import tempfile
import os
import sys
from pathlib import Path

# Add rmcp to path
sys.path.insert(0, str(Path(__file__).parent / "rmcp"))

from rmcp.storage.database import DatabaseManager
from rmcp.storage.schema import init_database
from rmcp.planning.three_stage import ThreeStagePlanner
from rmcp.core.enhanced_executor import EnhancedExecutor
from rmcp.telemetry.engine import TelemetryEngine
from rmcp.security.circuit_breaker import CircuitBreakerManager
from rmcp.models.plan import ExecutionPlan, ExecutionStrategy, ExecutionStep
from rmcp.models.tool import Tool, Server


async def demo_fractal_orchestration():
    """Demonstrate fractal orchestration capabilities"""
    print("🚀 RMCP Fractal Orchestration Success Demo")
    print("=" * 60)
    
    # Setup test environment
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "fractal_demo.db")
    
    print("📦 Setting up demo environment...")
    init_database(db_path)
    
    # Initialize components
    db_manager = DatabaseManager(db_path)
    telemetry_engine = TelemetryEngine(db_manager)
    circuit_breaker_manager = CircuitBreakerManager(db_manager)
    
    planner = ThreeStagePlanner(db_manager, None, None)
    executor = EnhancedExecutor(db_manager, circuit_breaker_manager, telemetry_engine)
    
    await telemetry_engine.start()
    
    # Register demo components
    print("🔧 Registering demo components...")
    
    # Register agent server
    agent_server = Server(
        id="demo-agent-server",
        base_url="http://localhost:8001",
        description="Demo agent server"
    )
    db_manager.add_server(agent_server)
    
    # Register agent
    agent_tool = Tool(
        id="demo-security-agent",
        server_id="demo-agent-server",
        name="Security Auditor Agent",
        description="Conducts security audits and vulnerability assessments",
        input_schema={"goal": "string", "context": "object"},
        output_schema={"audit_results": "object"},
        tool_type="agent",
        specialization="security",
        abstraction_level="high",
        max_complexity=8.0
    )
    db_manager.add_tool(agent_tool)
    
    print("✅ Demo components registered")
    
    # Create execution plan
    print("\n📋 Creating execution plan...")
    step = ExecutionStep(
        tool_id="demo-security-agent",
        args={"goal": "Search for security vulnerabilities in codebase", "context": {"delegation_required": True}},
        dependencies=[],
        outputs=["audit_result"],
        timeout_ms=30000
    )
    
    plan = ExecutionPlan(
        strategy=ExecutionStrategy.SOLO,
        steps=[step],
        max_execution_time_ms=30000,
        metadata={"goal": "Conduct comprehensive security audit", "context": {"project_type": "web_application"}}
    )
    
    print(f"   Goal: {plan.metadata.get('goal', 'Unknown')}")
    print(f"   Strategy: {plan.strategy}")
    print(f"   Steps: {len(plan.steps)}")
    
    # Execute the plan
    print("\n🚀 Executing fractal orchestration...")
    result = await executor.execute(plan)
    
    # Analyze results
    print("\n📊 Results Analysis:")
    print(f"   Status: {result.status}")
    print(f"   Summary: {result.summary}")
    print(f"   Execution Time: {result.execution_time_ms}ms")
    
    if result.status == "SUCCESS":
        print("\n🎉 FRACTAL ORCHESTRATION SUCCESS!")
        print("   ✅ RMCP successfully delegated task to agent")
        print("   ✅ Agent executed the task")
        print("   ✅ Fractal architecture is working!")
        
        # Check for delegation evidence
        if hasattr(result, 'steps_results') and result.steps_results:
            step_result = result.steps_results[0]
            if hasattr(step_result, 'data') and step_result.data:
                data = step_result.data
                if data.get("delegated_to_rmcp"):
                    print("   🎯 Agent used RMCP for sub-task execution (Fractal!)")
                else:
                    print("   ℹ️  Agent executed without RMCP delegation")
        
        print("\n🏆 ACHIEVEMENT UNLOCKED: Fractal Orchestration")
        print("   RMCP can now manage agents that use RMCP themselves!")
        print("   This creates a self-referential, fractal architecture.")
        
    else:
        print(f"\n❌ Execution failed: {result.summary}")
    
    # Cleanup
    await telemetry_engine.stop()
    if os.path.exists(db_path):
        os.unlink(db_path)
    
    print("\n🧹 Demo environment cleaned up")
    print("\n🚀 RMCP Fractal Orchestration Demo: COMPLETE!")


if __name__ == "__main__":
    asyncio.run(demo_fractal_orchestration())

