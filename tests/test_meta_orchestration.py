#!/usr/bin/env python3
"""
End-to-End Test for RMCP Meta-Orchestration
This script tests the complete meta-orchestration flow with real agents and MCP servers.
"""

import asyncio
import json
import time
import requests
import subprocess
import sys
from typing import Dict, Any, List
from pathlib import Path


class MetaOrchestrationTester:
    """
    End-to-End tester for RMCP Meta-Orchestration
    
    This class:
    1. Starts the test environment using docker-compose
    2. Waits for all services to be healthy
    3. Executes test scenarios
    4. Validates results
    5. Cleans up the environment
    """
    
    def __init__(self):
        self.compose_file = "docker-compose.test.yml"
        self.services = {
            "mock-mcp-server": "http://localhost:8000",
            "mock-agent": "http://localhost:8001", 
            "rmcp": "http://localhost:8080"
        }
        self.test_results = []
    
    async def run_tests(self):
        """Run all E2E tests"""
        print("🚀 Starting RMCP Meta-Orchestration E2E Tests")
        print("=" * 60)
        
        try:
            # Start test environment
            await self._start_environment()
            
            # Wait for services to be ready
            await self._wait_for_services()
            
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
    
    async def _start_environment(self):
        """Start the test environment using docker-compose"""
        print("📦 Starting test environment...")
        
        # Stop any existing containers
        subprocess.run([
            "docker-compose", "-f", self.compose_file, "down", "-v"
        ], capture_output=True)
        
        # Start services
        result = subprocess.run([
            "docker-compose", "-f", self.compose_file, "up", "-d", "--build"
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            raise Exception(f"Failed to start environment: {result.stderr}")
        
        print("✅ Test environment started successfully")
    
    async def _wait_for_services(self):
        """Wait for all services to be healthy"""
        print("⏳ Waiting for services to be ready...")
        
        max_wait_time = 120  # 2 minutes
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            all_healthy = True
            
            for service_name, base_url in self.services.items():
                try:
                    response = requests.get(f"{base_url}/health", timeout=5)
                    if response.status_code != 200:
                        all_healthy = False
                        break
                except requests.RequestException:
                    all_healthy = False
                    break
            
            if all_healthy:
                print("✅ All services are healthy")
                return
            
            await asyncio.sleep(5)
            print(".", end="", flush=True)
        
        raise Exception("Services did not become healthy within timeout")
    
    async def _run_test_scenarios(self):
        """Run test scenarios"""
        print("\n🧪 Running test scenarios...")
        
        # Test 1: High-level task delegation to agent
        await self._test_agent_delegation()
        
        # Test 2: Low-level task execution with atomic tools
        await self._test_atomic_tool_execution()
        
        # Test 3: Mixed execution (agent + atomic tools)
        await self._test_mixed_execution()
        
        # Test 4: Service health and statistics
        await self._test_service_health()
    
    async def _test_agent_delegation(self):
        """Test delegation of high-level task to agent"""
        print("\n🔍 Test 1: Agent Delegation")
        
        try:
            # Send high-level security audit request
            request = {
                "goal": "Conduct comprehensive security audit of our Terraform infrastructure",
                "context": {
                    "repo_path": "/path/to/terraform",
                    "environment": "production"
                },
                "user_id": "test-user",
                "tenant_id": "test-tenant"
            }
            
            response = requests.post(
                f"{self.services['rmcp']}/execute",
                json=request,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Validate response structure
                assert "status" in result
                assert "summary" in result
                assert "data" in result
                
                # Check if agent was used (should contain agent-specific data)
                if "vulnerabilities_found" in result.get("data", {}):
                    print("✅ Agent delegation successful")
                    self.test_results.append({
                        "name": "Agent Delegation",
                        "passed": True,
                        "details": f"Agent returned: {result['summary']}"
                    })
                else:
                    print("⚠️  Agent delegation may not have been used")
                    self.test_results.append({
                        "name": "Agent Delegation",
                        "passed": False,
                        "details": "Agent-specific data not found in response"
                    })
            else:
                raise Exception(f"Request failed with status {response.status_code}: {response.text}")
                
        except Exception as e:
            print(f"❌ Agent delegation test failed: {e}")
            self.test_results.append({
                "name": "Agent Delegation",
                "passed": False,
                "details": str(e)
            })
    
    async def _test_atomic_tool_execution(self):
        """Test execution of low-level task with atomic tools"""
        print("\n🔧 Test 2: Atomic Tool Execution")
        
        try:
            # Send low-level grep request
            request = {
                "goal": "Find all files containing the word 'error'",
                "context": {
                    "file_pattern": "*.py"
                },
                "user_id": "test-user",
                "tenant_id": "test-tenant"
            }
            
            response = requests.post(
                f"{self.services['rmcp']}/execute",
                json=request,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Validate response structure
                assert "status" in result
                assert "summary" in result
                
                print("✅ Atomic tool execution successful")
                self.test_results.append({
                    "name": "Atomic Tool Execution",
                    "passed": True,
                    "details": f"Tool returned: {result['summary']}"
                })
            else:
                raise Exception(f"Request failed with status {response.status_code}: {response.text}")
                
        except Exception as e:
            print(f"❌ Atomic tool execution test failed: {e}")
            self.test_results.append({
                "name": "Atomic Tool Execution",
                "passed": False,
                "details": str(e)
            })
    
    async def _test_mixed_execution(self):
        """Test mixed execution scenario"""
        print("\n🔄 Test 3: Mixed Execution")
        
        try:
            # Send complex request that might use both agents and tools
            request = {
                "goal": "Deploy new version and run security tests",
                "context": {
                    "version": "v1.2.3",
                    "environment": "staging"
                },
                "user_id": "test-user",
                "tenant_id": "test-tenant"
            }
            
            response = requests.post(
                f"{self.services['rmcp']}/execute",
                json=request,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                
                print("✅ Mixed execution successful")
                self.test_results.append({
                    "name": "Mixed Execution",
                    "passed": True,
                    "details": f"Execution completed: {result['summary']}"
                })
            else:
                raise Exception(f"Request failed with status {response.status_code}: {response.text}")
                
        except Exception as e:
            print(f"❌ Mixed execution test failed: {e}")
            self.test_results.append({
                "name": "Mixed Execution",
                "passed": False,
                "details": str(e)
            })
    
    async def _test_service_health(self):
        """Test service health and statistics"""
        print("\n📊 Test 4: Service Health")
        
        try:
            all_healthy = True
            
            for service_name, base_url in self.services.items():
                try:
                    # Test health endpoint
                    health_response = requests.get(f"{base_url}/health", timeout=5)
                    if health_response.status_code != 200:
                        all_healthy = False
                        print(f"❌ {service_name} health check failed")
                        continue
                    
                    # Test stats endpoint if available
                    try:
                        stats_response = requests.get(f"{base_url}/stats", timeout=5)
                        if stats_response.status_code == 200:
                            stats = stats_response.json()
                            print(f"✅ {service_name}: {stats}")
                    except requests.RequestException:
                        # Stats endpoint might not be available
                        pass
                    
                    print(f"✅ {service_name} is healthy")
                    
                except requests.RequestException as e:
                    all_healthy = False
                    print(f"❌ {service_name} health check failed: {e}")
            
            self.test_results.append({
                "name": "Service Health",
                "passed": all_healthy,
                "details": "All services healthy" if all_healthy else "Some services unhealthy"
            })
            
        except Exception as e:
            print(f"❌ Service health test failed: {e}")
            self.test_results.append({
                "name": "Service Health",
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
            print("🎉 ALL TESTS PASSED! Meta-orchestration is working!")
        else:
            print("⚠️  Some tests failed. Check the details above.")
    
    async def _cleanup(self):
        """Clean up test environment"""
        print("\n🧹 Cleaning up test environment...")
        
        subprocess.run([
            "docker-compose", "-f", self.compose_file, "down", "-v"
        ], capture_output=True)
        
        print("✅ Cleanup completed")


async def main():
    """Main entry point"""
    tester = MetaOrchestrationTester()
    success = await tester.run_tests()
    
    if success:
        print("\n🚀 RMCP Meta-Orchestration E2E Tests: SUCCESS!")
        sys.exit(0)
    else:
        print("\n💥 RMCP Meta-Orchestration E2E Tests: FAILED!")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

