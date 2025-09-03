#!/usr/bin/env python3
"""
Comprehensive RMCP System Diagnosis Test

This test performs deep inspection of the entire RMCP pipeline:
1. Capability ingestion and discovery
2. Three-stage planning (Sieve → Compass → Judge)
3. DAG execution and step coordination
4. Agent communication and argument passing
5. Error propagation and reporting

Goal: Identify all issues in the execution chain with detailed logging.
"""

import asyncio
import json
import httpx
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class TestStep:
    """Represents a single test step with timing and results"""
    name: str
    description: str
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    success: bool = False
    result: Any = None
    error: Optional[str] = None
    details: Dict[str, Any] = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}

    @property
    def duration_ms(self) -> float:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time) * 1000
        return 0.0


class ComprehensiveSystemDiagnosis:
    """
    Comprehensive system diagnosis with detailed logging
    """
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.test_steps: List[TestStep] = []
        self.client = httpx.AsyncClient(timeout=60.0)
        
        # Test configuration
        self.test_goal = "Создай простую HTML страницу с кнопкой"
        self.expected_files = ["index.html", "button.html", "simple.html"]
        
    async def run_comprehensive_diagnosis(self) -> Dict[str, Any]:
        """Run complete system diagnosis"""
        print("🔍 Starting Comprehensive RMCP System Diagnosis")
        print("=" * 80)
        
        start_time = time.time()
        
        # Phase 1: System Health Check
        await self._test_system_health()
        
        # Phase 2: Capability Discovery
        await self._test_capability_discovery()
        
        # Phase 3: Planning Pipeline
        await self._test_planning_pipeline()
        
        # Phase 4: Execution Analysis
        await self._test_execution_analysis()
        
        # Phase 5: Agent Communication
        await self._test_agent_communication()
        
        # Phase 6: End-to-End Integration
        await self._test_e2e_integration()
        
        total_time = time.time() - start_time
        
        # Generate comprehensive report
        report = self._generate_diagnosis_report(total_time)
        
        await self.client.aclose()
        return report
        
    async def _test_system_health(self):
        """Test basic system health and connectivity"""
        print("\n🏥 Phase 1: System Health Check")
        print("-" * 40)
        
        # Test RMCP health
        step = TestStep("rmcp_health", "Check RMCP main service health")
        step.start_time = time.time()
        
        try:
            response = await self.client.get(f"{self.base_url}/health")
            step.end_time = time.time()
            step.success = response.status_code == 200
            step.result = response.json()
            print(f"✅ RMCP Health: {step.result}")
        except Exception as e:
            step.end_time = time.time()
            step.error = str(e)
            print(f"❌ RMCP Health Failed: {e}")
            
        self.test_steps.append(step)
        
        # Test agent endpoints
        agent_endpoints = [
            ("http://localhost:8005", "Backend Agent"),
            ("http://localhost:8003", "Agent Wrapper"),
            ("http://localhost:8001", "Basic Tools"),
            ("http://localhost:8002", "Filesystem Write")
        ]
        
        for url, name in agent_endpoints:
            step = TestStep(f"agent_health_{name.lower().replace(' ', '_')}", f"Check {name} health")
            step.start_time = time.time()
            
            try:
                response = await self.client.get(f"{url}/health")
                step.end_time = time.time()
                step.success = response.status_code == 200
                step.result = response.json()
                print(f"✅ {name}: {step.result}")
            except Exception as e:
                step.end_time = time.time()
                step.error = str(e)
                print(f"❌ {name} Failed: {e}")
                
            self.test_steps.append(step)
    
    async def _test_capability_discovery(self):
        """Test capability discovery and ingestion"""
        print("\n🔍 Phase 2: Capability Discovery")
        print("-" * 40)
        
        # Test capability ingestion
        step = TestStep("capability_ingestion", "Ingest capabilities from all MCP servers")
        step.start_time = time.time()
        
        try:
            response = await self.client.post(f"{self.base_url}/api/v1/ingest")
            step.end_time = time.time()
            step.success = response.status_code == 200
            step.result = response.json()
            
            print(f"✅ Capability Ingestion:")
            print(f"   Servers Scanned: {step.result.get('servers_scanned', 0)}")
            print(f"   Tools Discovered: {step.result.get('tools_discovered', 0)}")
            
        except Exception as e:
            step.end_time = time.time()
            step.error = str(e)
            print(f"❌ Capability Ingestion Failed: {e}")
            
        self.test_steps.append(step)
        
        # Test tool listing
        step = TestStep("tool_listing", "List all available tools")
        step.start_time = time.time()
        
        try:
            response = await self.client.get(f"{self.base_url}/api/v1/tools")
            step.end_time = time.time()
            step.success = response.status_code == 200
            step.result = response.json()
            
            tools = step.result.get('tools', [])
            print(f"✅ Available Tools: {len(tools)}")
            
            # Categorize tools
            agent_tools = [t for t in tools if any(x in t['name'] for x in ['backend', 'architect', 'devops'])]
            fs_tools = [t for t in tools if 'filesystem' in t['name']]
            basic_tools = [t for t in tools if any(x in t['name'] for x in ['ls', 'cat', 'wc'])]
            
            print(f"   Agent Tools: {len(agent_tools)}")
            print(f"   Filesystem Tools: {len(fs_tools)}")
            print(f"   Basic Tools: {len(basic_tools)}")
            
            step.details = {
                "total_tools": len(tools),
                "agent_tools": len(agent_tools),
                "fs_tools": len(fs_tools),
                "basic_tools": len(basic_tools)
            }
            
        except Exception as e:
            step.end_time = time.time()
            step.error = str(e)
            print(f"❌ Tool Listing Failed: {e}")
            
        self.test_steps.append(step)
    
    async def _test_planning_pipeline(self):
        """Test the three-stage planning pipeline in detail"""
        print("\n🧠 Phase 3: Planning Pipeline Analysis")
        print("-" * 40)
        
        # Test basic routing
        step = TestStep("planning_basic", "Test basic planning with rmcp.route")
        step.start_time = time.time()
        
        try:
            # Make the request
            response = await self.client.post(
                f"{self.base_url}/api/v1/execute",
                json={
                    "tool_name": "rmcp.route",
                    "parameters": {
                        "goal": self.test_goal,
                        "context": {"debug": True}
                    }
                }
            )
            
            step.end_time = time.time()
            step.success = response.status_code == 200
            step.result = response.json()
            
            print(f"✅ Planning Response Status: {response.status_code}")
            print(f"   Execution Status: {step.result.get('status', 'UNKNOWN')}")
            print(f"   Summary: {step.result.get('summary', 'No summary')}")
            
            # Analyze execution details
            data = step.result.get('data', {})
            if 'step_results' in data:
                step_results = data['step_results']
                print(f"   Total Steps Planned: {len(step_results)}")
                
                success_count = sum(1 for r in step_results.values() if r.get('success', False))
                print(f"   Successful Steps: {success_count}/{len(step_results)}")
                
                # Show first few step details
                for i, (step_id, result) in enumerate(list(step_results.items())[:3]):
                    print(f"   Step {i}: {result.get('tool_name', 'unknown')} - "
                          f"{'✅' if result.get('success') else '❌'} {result.get('error', '')}")
            
            step.details = step.result
            
        except Exception as e:
            step.end_time = time.time()
            step.error = str(e)
            print(f"❌ Planning Failed: {e}")
            
        self.test_steps.append(step)
    
    async def _test_execution_analysis(self):
        """Analyze execution step by step"""
        print("\n⚙️ Phase 4: Execution Analysis")
        print("-" * 40)
        
        # Get the last planning result
        last_planning = None
        for step in reversed(self.test_steps):
            if step.name == "planning_basic" and step.success:
                last_planning = step.result
                break
        
        if not last_planning:
            print("❌ No successful planning result to analyze")
            return
        
        data = last_planning.get('data', {})
        step_results = data.get('step_results', {})
        
        if not step_results:
            print("❌ No step results to analyze")
            return
        
        print(f"📊 Analyzing {len(step_results)} execution steps:")
        
        for step_id, result in step_results.items():
            step = TestStep(f"exec_analysis_{step_id}", f"Analyze execution step {step_id}")
            step.start_time = time.time()
            
            tool_name = result.get('tool_name', 'unknown')
            success = result.get('success', False)
            error = result.get('error', '')
            
            print(f"\n   🔍 Step {step_id}: {tool_name}")
            print(f"      Status: {'✅ SUCCESS' if success else '❌ FAILED'}")
            
            if not success and error:
                print(f"      Error: {error}")
                
                # Analyze common error patterns
                if "Goal is required" in error:
                    print(f"      🔧 Issue: Agent wrapper not receiving goal parameter")
                elif "Parameter 'path' is required" in error:
                    print(f"      🔧 Issue: Filesystem tool missing required path")
                elif "Usage:" in error:
                    print(f"      🔧 Issue: Tool receiving wrong argument format")
                else:
                    print(f"      🔧 Issue: Unknown error pattern")
            
            step.end_time = time.time()
            step.success = success
            step.result = result
            step.error = error if not success else None
            
            self.test_steps.append(step)
    
    async def _test_agent_communication(self):
        """Test direct agent communication"""
        print("\n🤖 Phase 5: Agent Communication Testing")
        print("-" * 40)
        
        # Test direct backend agent call
        step = TestStep("direct_backend_test", "Direct backend agent communication")
        step.start_time = time.time()
        
        try:
            response = await self.client.post(
                "http://localhost:8005/execute",
                json={
                    "goal": "Create simple HTML page",
                    "file_path": "test_direct.html"
                }
            )
            
            step.end_time = time.time()
            step.success = response.status_code == 200
            step.result = response.json()
            
            print(f"✅ Direct Backend Agent: {response.status_code}")
            if step.success:
                print(f"   Response: {step.result.get('message', 'No message')[:100]}...")
            
        except Exception as e:
            step.end_time = time.time()
            step.error = str(e)
            print(f"❌ Direct Backend Agent Failed: {e}")
            
        self.test_steps.append(step)
        
        # Test agent wrapper
        step = TestStep("agent_wrapper_test", "Agent wrapper communication")
        step.start_time = time.time()
        
        try:
            response = await self.client.post(
                "http://localhost:8003/execute",
                json={
                    "arguments": ["backend.generate_code", "Create simple HTML page"],
                    "parameters": {
                        "goal": "Create simple HTML page",
                        "file_path": "test_wrapper.html"
                    }
                }
            )
            
            step.end_time = time.time()
            step.success = response.status_code == 200
            step.result = response.json()
            
            print(f"✅ Agent Wrapper: {response.status_code}")
            if step.success:
                print(f"   Success: {step.result.get('success', False)}")
                print(f"   Output: {step.result.get('output', 'No output')[:100]}...")
            
        except Exception as e:
            step.end_time = time.time()
            step.error = str(e)
            print(f"❌ Agent Wrapper Failed: {e}")
            
        self.test_steps.append(step)
    
    async def _test_e2e_integration(self):
        """Test end-to-end integration with file verification"""
        print("\n🔄 Phase 6: End-to-End Integration")
        print("-" * 40)
        
        # Test complete workflow
        step = TestStep("e2e_integration", "Complete workflow with file verification")
        step.start_time = time.time()
        
        try:
            # Execute the workflow
            response = await self.client.post(
                f"{self.base_url}/api/v1/execute",
                json={
                    "tool_name": "rmcp.route",
                    "parameters": {
                        "goal": "Создай файл test_integration.html с простой HTML страницей",
                        "context": {"test_mode": True}
                    }
                }
            )
            
            step.end_time = time.time()
            step.success = response.status_code == 200
            step.result = response.json()
            
            print(f"✅ E2E Execution: {response.status_code}")
            print(f"   Status: {step.result.get('status', 'UNKNOWN')}")
            
            # Check for created files
            try:
                file_check_response = await self.client.post(
                    f"{self.base_url}/api/v1/execute",
                    json={
                        "tool_name": "ls",
                        "parameters": {"path": "."}
                    }
                )
                
                if file_check_response.status_code == 200:
                    file_result = file_check_response.json()
                    output = file_result.get('data', {}).get('output', '')
                    
                    created_files = []
                    for expected_file in self.expected_files + ["test_integration.html"]:
                        if expected_file in output:
                            created_files.append(expected_file)
                    
                    print(f"   Created Files: {created_files}")
                    step.details = {"created_files": created_files}
                    
            except Exception as e:
                print(f"   File Check Failed: {e}")
            
        except Exception as e:
            step.end_time = time.time()
            step.error = str(e)
            print(f"❌ E2E Integration Failed: {e}")
            
        self.test_steps.append(step)
    
    def _generate_diagnosis_report(self, total_time: float) -> Dict[str, Any]:
        """Generate comprehensive diagnosis report"""
        print("\n📊 COMPREHENSIVE DIAGNOSIS REPORT")
        print("=" * 80)
        
        # Calculate statistics
        total_steps = len(self.test_steps)
        successful_steps = sum(1 for step in self.test_steps if step.success)
        failed_steps = total_steps - successful_steps
        
        print(f"📈 Overall Statistics:")
        print(f"   Total Test Steps: {total_steps}")
        print(f"   Successful Steps: {successful_steps}")
        print(f"   Failed Steps: {failed_steps}")
        print(f"   Success Rate: {(successful_steps/total_steps)*100:.1f}%")
        print(f"   Total Execution Time: {total_time:.2f}s")
        
        # Categorize issues
        issues = {
            "argument_passing": [],
            "agent_communication": [],
            "planning_logic": [],
            "execution_flow": [],
            "system_health": []
        }
        
        for step in self.test_steps:
            if not step.success and step.error:
                error = step.error.lower()
                if "goal is required" in error or "parameter" in error:
                    issues["argument_passing"].append(step)
                elif "connection" in error or "timeout" in error:
                    issues["agent_communication"].append(step)
                elif "planning" in error or "strategy" in error:
                    issues["planning_logic"].append(step)
                elif "execution" in error:
                    issues["execution_flow"].append(step)
                else:
                    issues["system_health"].append(step)
        
        print(f"\n🔍 Issue Analysis:")
        for category, steps in issues.items():
            if steps:
                print(f"   {category.replace('_', ' ').title()}: {len(steps)} issues")
                for step in steps[:2]:  # Show first 2 issues
                    print(f"      - {step.name}: {step.error[:80]}...")
        
        # Performance analysis
        step_times = [step.duration_ms for step in self.test_steps if step.duration_ms > 0]
        if step_times:
            avg_time = sum(step_times) / len(step_times)
            print(f"\n⏱️ Performance Analysis:")
            print(f"   Average Step Time: {avg_time:.2f}ms")
            print(f"   Slowest Step: {max(step_times):.2f}ms")
            print(f"   Fastest Step: {min(step_times):.2f}ms")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        
        if issues["argument_passing"]:
            print("   🔧 Fix argument passing between DAG executor and agents")
            print("      - Ensure goal parameter is properly propagated")
            print("      - Standardize parameter format across all tools")
        
        if issues["agent_communication"]:
            print("   🔧 Improve agent communication reliability")
            print("      - Add retry logic for agent calls")
            print("      - Implement better error handling")
        
        if failed_steps > successful_steps:
            print("   🔧 Critical: System has more failures than successes")
            print("      - Review entire execution pipeline")
            print("      - Add comprehensive error handling")
        
        # Generate detailed report
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_time_seconds": total_time,
            "statistics": {
                "total_steps": total_steps,
                "successful_steps": successful_steps,
                "failed_steps": failed_steps,
                "success_rate": (successful_steps/total_steps)*100
            },
            "issues": {k: len(v) for k, v in issues.items()},
            "performance": {
                "average_step_time_ms": sum(step_times) / len(step_times) if step_times else 0,
                "slowest_step_ms": max(step_times) if step_times else 0,
                "fastest_step_ms": min(step_times) if step_times else 0
            },
            "detailed_steps": [
                {
                    "name": step.name,
                    "description": step.description,
                    "success": step.success,
                    "duration_ms": step.duration_ms,
                    "error": step.error,
                    "details": step.details
                }
                for step in self.test_steps
            ]
        }
        
        return report


async def main():
    """Run comprehensive system diagnosis"""
    diagnosis = ComprehensiveSystemDiagnosis()
    
    try:
        report = await diagnosis.run_comprehensive_diagnosis()
        
        # Save detailed report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"system_diagnosis_report_{timestamp}.json"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Detailed report saved to: {report_file}")
        print("\n🎯 DIAGNOSIS COMPLETE!")
        
        return report
        
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR in diagnosis: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    asyncio.run(main())
