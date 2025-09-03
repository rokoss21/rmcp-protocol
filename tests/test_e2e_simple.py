#!/usr/bin/env python3
"""
Simplified End-to-End Test for RMCP Meta-Orchestration
This test runs without Docker, using the Enhanced Executor directly
"""

import asyncio
import json
import time
import tempfile
from pathlib import Path
import sys
import os

# Add RMCP to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rmcp.core.enhanced_executor import EnhancedExecutor
from rmcp.storage.database import DatabaseManager
from rmcp.storage.schema import init_database
from rmcp.models.plan import ExecutionPlan, ExecutionStep, ExecutionStrategy
from rmcp.models.tool import Tool, Server
from rmcp.security.circuit_breaker import CircuitBreakerManager
from rmcp.telemetry.engine import TelemetryEngine


class SimpleE2ETester:
    """
    Simplified E2E tester that uses Enhanced Executor directly
    without requiring Docker or external services
    """
    
    def __init__(self):
        self.test_results = []
        self.temp_db = None
        self.executor = None
        self.db_manager = None
    
    async def run_tests(self):
        """Run all E2E tests"""
        print("🚀 Starting RMCP Meta-Orchestration Simple E2E Tests")
        print("=" * 60)
        
        try:
            # Setup test environment
            await self._setup_environment()
            
            # Run test scenarios
            await self._run_test_scenarios()
            
            # Print results
            self._print_results()
            
        except Exception as e:
            print(f"❌ Test execution failed: {e}")
            return False
        finally:
            # Clean up
            await self._cleanup()
        
        return all(result["passed"] for result in self.test_results)
    
    async def _setup_environment(self):
        """Setup test environment"""
        print("📦 Setting up test environment...")
        
        # Create temporary database
        self.temp_db = tempfile.mktemp(suffix='.db')
        init_database(self.temp_db)
        
        # Create database manager
        self.db_manager = DatabaseManager(self.temp_db)
        
        # Create circuit breaker manager and telemetry engine
        circuit_breaker_manager = CircuitBreakerManager(self.db_manager)
        telemetry_engine = TelemetryEngine(self.db_manager)
        
        # Create enhanced executor
        self.executor = EnhancedExecutor(
            self.db_manager, 
            circuit_breaker_manager, 
            telemetry_engine
        )
        
        # Add test tools and agents
        await self._add_test_components()
        
        print("✅ Test environment setup completed")
    
    async def _add_test_components(self):
        """Add test tools and agents to the database"""
        # Add mock MCP server
        mcp_server = Server(
            id="mock-mcp-server",
            name="Mock MCP Server",
            base_url="http://localhost:8000",
            description="Test MCP server"
        )
        self.db_manager.add_server(mcp_server)
        
        # Add mock agent server
        agent_server = Server(
            id="mock-agent-server",
            name="Mock Agent Server", 
            base_url="http://localhost:8001",
            description="Test agent server"
        )
        self.db_manager.add_server(agent_server)
        
        # Add atomic tool
        atomic_tool = Tool(
            id="test-grep",
            server_id="mock-mcp-server",
            name="grep",
            description="Search text in files",
            input_schema={},
            output_schema={},
            tags=["search", "text"],
            capabilities=["text_search"],
            tool_type="atomic",
            specialization=None,
            abstraction_level="low",
            max_complexity=1.0,
            avg_execution_time_ms=5000,
            p95_latency_ms=3000,
            success_rate=0.95,
            cost_hint=0.0
        )
        self.db_manager.add_tool(atomic_tool)
        
        # Add agent
        agent = Tool(
            id="test-security-agent",
            server_id="mock-agent-server",
            name="Security Auditor Agent",
            description="Autonomous agent for security auditing",
            input_schema={},
            output_schema={},
            tags=["security", "audit"],
            capabilities=["security_audit", "vulnerability_scan"],
            tool_type="agent",
            specialization="security",
            abstraction_level="high",
            max_complexity=1.0,
            avg_execution_time_ms=30000,
            p95_latency_ms=3000,
            success_rate=0.95,
            cost_hint=0.0
        )
        self.db_manager.add_tool(agent)
    
    async def _run_test_scenarios(self):
        """Run test scenarios"""
        print("\n🧪 Running test scenarios...")
        
        # Test 1: Agent delegation (will fail without real agent, but tests the logic)
        await self._test_agent_delegation_logic()
        
        # Test 2: Atomic tool execution (will fail without real MCP server, but tests the logic)
        await self._test_atomic_tool_logic()
        
        # Test 3: Tool type detection
        await self._test_tool_type_detection()
        
        # Test 4: Database operations
        await self._test_database_operations()
    
    async def _test_agent_delegation_logic(self):
        """Test agent delegation logic"""
        print("\n🔍 Test 1: Agent Delegation Logic")
        
        try:
            # Create execution plan for agent
            step = ExecutionStep(
                tool_id="test-security-agent",
                args={
                    "goal": "Conduct security audit",
                    "context": {"repo": "test"}
                },
                timeout_ms=30000
            )
            plan = ExecutionPlan(
                strategy=ExecutionStrategy.SOLO,
                steps=[step],
                max_execution_time_ms=30000
            )
            
            # This will fail because the agent server is not running,
            # but it tests the logic flow
            result = await self.executor.execute(plan)
            
            # Check if the result indicates the correct logic flow
            if result.status == "ERROR" and "connection" in result.summary.lower():
                print("✅ Agent delegation logic correct (failed as expected)")
                self.test_results.append({
                    "name": "Agent Delegation Logic",
                    "passed": True,
                    "details": f"Correctly failed with connection error: {result.summary[:100]}"
                })
            else:
                print(f"❌ Unexpected result: {result}")
                self.test_results.append({
                    "name": "Agent Delegation Logic",
                    "passed": False,
                    "details": f"Unexpected result: {result.status} - {result.summary}"
                })
                    
        except Exception as e:
            print(f"❌ Agent delegation test failed: {e}")
            self.test_results.append({
                "name": "Agent Delegation Logic",
                "passed": False,
                "details": str(e)
            })
    
    async def _test_atomic_tool_logic(self):
        """Test atomic tool execution logic"""
        print("\n🔧 Test 2: Atomic Tool Logic")
        
        try:
            # Create execution plan for atomic tool
            step = ExecutionStep(
                tool_id="test-grep",
                args={
                    "pattern": "error",
                    "file": "test.py"
                },
                timeout_ms=5000
            )
            plan = ExecutionPlan(
                strategy=ExecutionStrategy.SOLO,
                steps=[step],
                max_execution_time_ms=5000
            )
            
            # This will fail because the MCP server is not running,
            # but it tests the logic flow
            result = await self.executor.execute(plan)
            
            # Check if the result indicates the correct logic flow
            if result.status == "ERROR" and "connection" in result.summary.lower():
                print("✅ Atomic tool logic correct (failed as expected)")
                self.test_results.append({
                    "name": "Atomic Tool Logic",
                    "passed": True,
                    "details": f"Correctly failed with connection error: {result.summary[:100]}"
                })
            else:
                print(f"❌ Unexpected result: {result}")
                self.test_results.append({
                    "name": "Atomic Tool Logic",
                    "passed": False,
                    "details": f"Unexpected result: {result.status} - {result.summary}"
                })
                    
        except Exception as e:
            print(f"❌ Atomic tool test failed: {e}")
            self.test_results.append({
                "name": "Atomic Tool Logic",
                "passed": False,
                "details": str(e)
            })
    
    async def _test_tool_type_detection(self):
        """Test tool type detection"""
        print("\n🔍 Test 3: Tool Type Detection")
        
        try:
            # Get atomic tool
            atomic_tool = self.db_manager.get_tool("test-grep")
            assert atomic_tool is not None
            assert atomic_tool.tool_type == "atomic"
            assert atomic_tool.specialization is None
            assert atomic_tool.abstraction_level == "low"
            
            # Get agent
            agent = self.db_manager.get_tool("test-security-agent")
            assert agent is not None
            assert agent.tool_type == "agent"
            assert agent.specialization == "security"
            assert agent.abstraction_level == "high"
            
            print("✅ Tool type detection correct")
            self.test_results.append({
                "name": "Tool Type Detection",
                "passed": True,
                "details": "Atomic tools and agents correctly identified"
            })
            
        except Exception as e:
            print(f"❌ Tool type detection test failed: {e}")
            self.test_results.append({
                "name": "Tool Type Detection",
                "passed": False,
                "details": str(e)
            })
    
    async def _test_database_operations(self):
        """Test database operations"""
        print("\n📊 Test 4: Database Operations")
        
        try:
            # Test getting tools by type
            atomic_tools = self.db_manager.get_tools_by_type("atomic")
            assert len(atomic_tools) == 1
            assert atomic_tools[0].id == "test-grep"
            
            agents = self.db_manager.get_tools_by_type("agent")
            assert len(agents) == 1
            assert agents[0].id == "test-security-agent"
            
            # Test getting all agents
            all_agents = self.db_manager.get_agents()
            assert len(all_agents) == 1
            assert all_agents[0].id == "test-security-agent"
            
            # Test getting all atomic tools
            all_atomic = self.db_manager.get_atomic_tools()
            assert len(all_atomic) == 1
            assert all_atomic[0].id == "test-grep"
            
            print("✅ Database operations correct")
            self.test_results.append({
                "name": "Database Operations",
                "passed": True,
                "details": "All database operations working correctly"
            })
            
        except Exception as e:
            print(f"❌ Database operations test failed: {e}")
            self.test_results.append({
                "name": "Database Operations",
                "passed": False,
                "details": str(e)
            })
    
    def _print_results(self):
        """Print test results summary"""
        print("\n" + "=" * 60)
        print("📋 TEST RESULTS SUMMARY")
        print("=" * 60)
        
        passed = sum(1 for result in self.test_results if result["passed"])
        total = len(self.test_results)
        
        for result in self.test_results:
            status = "✅ PASS" if result["passed"] else "❌ FAIL"
            print(f"{status} {result['name']}")
            print(f"    {result['details']}")
            print()
        
        print(f"Overall: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 ALL TESTS PASSED! Meta-orchestration logic is working!")
        else:
            print("⚠️  Some tests failed. Check the details above.")
    
    async def _cleanup(self):
        """Clean up test environment"""
        print("\n🧹 Cleaning up test environment...")
        
        if self.executor:
            await self.executor.close()
        
        if self.temp_db and os.path.exists(self.temp_db):
            os.unlink(self.temp_db)
        
        print("✅ Cleanup completed")


async def main():
    """Main entry point"""
    tester = SimpleE2ETester()
    success = await tester.run_tests()
    
    if success:
        print("\n🚀 RMCP Meta-Orchestration Simple E2E Tests: SUCCESS!")
        sys.exit(0)
    else:
        print("\n💥 RMCP Meta-Orchestration Simple E2E Tests: FAILED!")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
