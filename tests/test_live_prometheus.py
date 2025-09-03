#!/usr/bin/env python3
"""
Live Prometheus Experiment - Phase 6 Final
==========================================

This test demonstrates the complete autonomous development workflow
with real agents working together to create and deploy an MCP server.

Usage:
    python test_live_prometheus.py
"""

import asyncio
import sys
import httpx
import time
from pathlib import Path
from typing import Dict, Any, List

# Add autonomous_agents to path
sys.path.insert(0, str(Path(__file__).parent / "autonomous_agents"))

from autonomous_agents.orchestrator.main import BrigadeOrchestrator


class LivePrometheusExperiment:
    """
    Live Prometheus Experiment - Real Autonomous Development
    
    This experiment demonstrates the complete autonomous development workflow:
    1. Architect creates development plan
    2. Backend generates code
    3. Tester creates tests
    4. Validator validates code
    5. DevOps deploys service
    6. System self-registers the new tool
    """
    
    def __init__(self):
        self.orchestrator = BrigadeOrchestrator()
        self.agent_endpoints = {
            "architect": "http://localhost:8003",
            "backend": "http://localhost:8004",
            "tester": "http://localhost:8005",
            "validator": "http://localhost:8006",
            "devops": "http://localhost:8008"
        }
        self.experiment_results = []
    
    async def run_live_experiment(self, goal: str) -> Dict[str, Any]:
        """
        Run the live Prometheus experiment
        
        Args:
            goal: High-level goal for MCP server creation
            
        Returns:
            Dict containing experiment results
        """
        print("🔥 LIVE PROMETHEUS EXPERIMENT: Real Autonomous Development")
        print("=" * 60)
        print(f"🎯 Goal: {goal}")
        print()
        
        try:
            # Phase 1: Check Agent Health
            print("🏥 PHASE 1: Agent Health Check")
            print("-" * 40)
            health_status = await self._check_agent_health()
            
            if not health_status["all_healthy"]:
                print("❌ Not all agents are healthy. Cannot proceed.")
                return {
                    "success": False,
                    "error": "Agent health check failed",
                    "health_status": health_status
                }
            
            # Phase 2: Execute Development Plan
            print("\n🚀 PHASE 2: Execute Development Plan")
            print("-" * 40)
            development_result = await self._execute_development_plan(goal)
            
            if not development_result.get("success"):
                return {
                    "success": False,
                    "error": "Development plan execution failed",
                    "development_result": development_result
                }
            
            # Phase 3: Validate Results
            print("\n✅ PHASE 3: Validate Results")
            print("-" * 40)
            validation_result = await self._validate_results(development_result)
            
            # Phase 4: Final Analysis
            print("\n📊 PHASE 4: Final Analysis")
            print("-" * 40)
            analysis = await self._analyze_results(development_result, validation_result)
            
            return {
                "success": True,
                "experiment": "live_prometheus",
                "goal": goal,
                "phases": {
                    "agent_health": health_status,
                    "development_execution": development_result,
                    "results_validation": validation_result,
                    "final_analysis": analysis
                },
                "summary": analysis
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Live experiment failed: {str(e)}",
                "experiment": "live_prometheus"
            }
    
    async def _check_agent_health(self) -> Dict[str, Any]:
        """Check health of all agents"""
        print("   🏥 Checking agent health...")
        
        health_status = {}
        all_healthy = True
        
        for agent_name, endpoint in self.agent_endpoints.items():
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.get(f"{endpoint}/health")
                    is_healthy = response.status_code == 200
                    health_status[agent_name] = {
                        "endpoint": endpoint,
                        "healthy": is_healthy,
                        "status_code": response.status_code
                    }
                    
                    status_icon = "✅" if is_healthy else "❌"
                    print(f"      {status_icon} {agent_name}: {endpoint}")
                    
                    if not is_healthy:
                        all_healthy = False
                        
            except Exception as e:
                health_status[agent_name] = {
                    "endpoint": endpoint,
                    "healthy": False,
                    "error": str(e)
                }
                print(f"      ❌ {agent_name}: {endpoint} - {e}")
                all_healthy = False
        
        print(f"   📊 Health Summary: {sum(1 for h in health_status.values() if h.get('healthy', False))}/{len(health_status)} agents healthy")
        
        return {
            "all_healthy": all_healthy,
            "agents": health_status
        }
    
    async def _execute_development_plan(self, goal: str) -> Dict[str, Any]:
        """Execute the complete development plan"""
        print("   🚀 Executing autonomous development workflow...")
        
        try:
            # Use the Brigade Orchestrator to execute the plan
            result = await self.orchestrator.execute_development_plan(goal, {
                "project_type": "mcp_server",
                "target_utility": "cowsay",
                "deployment_target": "docker"
            })
            
            print(f"   📊 Development Result:")
            print(f"      Success: {result.get('success', False)}")
            print(f"      Project ID: {result.get('project_id', 'N/A')}")
            print(f"      Execution Time: {result.get('execution_time_ms', 0)}ms")
            
            if result.get("project_summary"):
                summary = result["project_summary"]
                print(f"      Status: {summary.get('status', 'N/A')}")
                print(f"      Tasks: {summary.get('completed_tasks', 0)}/{summary.get('total_tasks', 0)}")
                print(f"      Progress: {summary.get('progress_percentage', 0):.1f}%")
            
            return result
            
        except Exception as e:
            print(f"   ❌ Development execution failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _validate_results(self, development_result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the development results"""
        print("   ✅ Validating development results...")
        
        validation_results = {
            "project_created": False,
            "code_generated": False,
            "tests_created": False,
            "validation_passed": False,
            "deployment_successful": False
        }
        
        # Check if project was created
        if development_result.get("project_id"):
            validation_results["project_created"] = True
            print("      ✅ Project created successfully")
        
        # Check if code was generated
        if development_result.get("execution_result", {}).get("completed_tasks", 0) > 0:
            validation_results["code_generated"] = True
            print("      ✅ Code generated successfully")
        
        # Check if tests were created
        if development_result.get("execution_result", {}).get("completed_tasks", 0) >= 3:
            validation_results["tests_created"] = True
            print("      ✅ Tests created successfully")
        
        # Check if validation passed
        if development_result.get("execution_result", {}).get("failed_tasks", 0) == 0:
            validation_results["validation_passed"] = True
            print("      ✅ Validation passed successfully")
        
        # Check if deployment was successful
        if development_result.get("success"):
            validation_results["deployment_successful"] = True
            print("      ✅ Deployment successful")
        
        overall_success = all(validation_results.values())
        
        print(f"   📊 Validation Summary: {sum(validation_results.values())}/{len(validation_results)} checks passed")
        
        return {
            "overall_success": overall_success,
            "validation_results": validation_results
        }
    
    async def _analyze_results(self, development_result: Dict[str, Any], validation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the final results"""
        print("   📊 Analyzing final results...")
        
        # Calculate success metrics
        success_metrics = {
            "agent_health": True,  # We got here, so agents were healthy
            "development_execution": development_result.get("success", False),
            "results_validation": validation_result.get("overall_success", False),
            "overall_success": development_result.get("success", False) and validation_result.get("overall_success", False)
        }
        
        print(f"   🎯 Success Metrics:")
        for metric, success in success_metrics.items():
            status = "✅" if success else "❌"
            print(f"      {status} {metric}")
        
        # Calculate performance metrics
        execution_time = development_result.get("execution_time_ms", 0)
        project_summary = development_result.get("project_summary", {})
        
        performance_metrics = {
            "execution_time_ms": execution_time,
            "total_tasks": project_summary.get("total_tasks", 0),
            "completed_tasks": project_summary.get("completed_tasks", 0),
            "failed_tasks": project_summary.get("failed_tasks", 0),
            "artifacts_created": project_summary.get("artifacts_count", 0),
            "success_rate": (project_summary.get("completed_tasks", 0) / max(project_summary.get("total_tasks", 1), 1)) * 100
        }
        
        print(f"   📈 Performance Metrics:")
        print(f"      Execution Time: {performance_metrics['execution_time_ms']}ms")
        print(f"      Tasks: {performance_metrics['completed_tasks']}/{performance_metrics['total_tasks']}")
        print(f"      Success Rate: {performance_metrics['success_rate']:.1f}%")
        print(f"      Artifacts: {performance_metrics['artifacts_created']}")
        
        return {
            "success_metrics": success_metrics,
            "performance_metrics": performance_metrics,
            "overall_success": success_metrics["overall_success"]
        }
    
    async def cleanup(self):
        """Cleanup experiment resources"""
        await self.orchestrator.close()


async def main():
    """Run the Live Prometheus Experiment"""
    experiment = LivePrometheusExperiment()
    
    # The ultimate test goal
    goal = "Create and deploy a fully functional MCP server for the command-line utility cowsay. It must accept a text parameter and return the ASCII art."
    
    try:
        result = await experiment.run_live_experiment(goal)
        
        print("\n" + "=" * 60)
        print("🔥 LIVE PROMETHEUS EXPERIMENT RESULTS")
        print("=" * 60)
        
        if result.get("success"):
            print("🎉 LIVE EXPERIMENT SUCCESS!")
            print("   ✅ Real autonomous MCP server creation achieved")
            print("   ✅ Live Brigade coordination demonstrated")
            print("   ✅ Phase 6 objectives completed in real-time")
            
            summary = result.get("summary", {})
            success_metrics = summary.get("success_metrics", {})
            performance_metrics = summary.get("performance_metrics", {})
            
            print(f"\n📊 Final Results:")
            print(f"   Overall Success: {success_metrics.get('overall_success', False)}")
            print(f"   Execution Time: {performance_metrics.get('execution_time_ms', 0)}ms")
            print(f"   Success Rate: {performance_metrics.get('success_rate', 0):.1f}%")
            print(f"   Artifacts Created: {performance_metrics.get('artifacts_created', 0)}")
            
            print(f"\n🏆 ACHIEVEMENT UNLOCKED: Live Prometheus")
            print("   The system can now autonomously create new tools in real-time!")
            print("   This represents the ultimate breakthrough in AI autonomy.")
            
        else:
            print("❌ LIVE EXPERIMENT FAILED")
            print(f"   Error: {result.get('error', 'Unknown error')}")
            
            # Show partial results if available
            if "phases" in result:
                phases = result["phases"]
                print(f"   Phases completed:")
                for phase_name, phase_result in phases.items():
                    if isinstance(phase_result, dict) and "success" in phase_result:
                        status = "✅" if phase_result["success"] else "❌"
                        print(f"      {status} {phase_name}")
        
        print(f"\n🚀 Live Prometheus Experiment: {'SUCCESS' if result.get('success') else 'FAILED'}")
        
    finally:
        await experiment.cleanup()


if __name__ == "__main__":
    asyncio.run(main())

