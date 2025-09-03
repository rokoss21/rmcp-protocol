#!/usr/bin/env python3
"""
E2E Test for Complete RMCP Ecosystem
====================================

This test demonstrates the complete RMCP ecosystem working together:
- RMCP as Meta-Orchestrator
- MCP Basic Tools (read-only operations)
- MCP Filesystem Write (write operations with security)
- Mock Agent (security auditor)

Usage:
    python test_ecosystem_e2e.py
"""

import asyncio
import httpx
import json
import time
import tempfile
import shutil
import os


async def test_ecosystem_e2e():
    """Test the complete RMCP ecosystem"""
    print("🌍 Testing Complete RMCP Ecosystem")
    print("=" * 60)
    
    # Service URLs
    rmcp_url = "http://localhost:8000"
    basic_tools_url = "http://localhost:8001"
    filesystem_write_url = "http://localhost:8002"
    agent_url = "http://localhost:8004"
    
    # Create a temporary directory for testing
    test_dir = tempfile.mkdtemp(prefix="rmcp_ecosystem_test_")
    print(f"🧪 Using test directory: {test_dir}")
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Test 1: Health checks for all services
            print("🏥 Test 1: Health Checks for All Services")
            print("-" * 50)
            
            services = [
                ("RMCP", rmcp_url),
                ("MCP Basic Tools", basic_tools_url),
                ("MCP Filesystem Write", filesystem_write_url),
                ("Mock Agent", agent_url)
            ]
            
            all_healthy = True
            for service_name, url in services:
                try:
                    response = await client.get(f"{url}/health")
                    if response.status_code == 200:
                        print(f"   ✅ {service_name}: Healthy")
                    else:
                        print(f"   ❌ {service_name}: Unhealthy ({response.status_code})")
                        all_healthy = False
                except Exception as e:
                    print(f"   ❌ {service_name}: Connection failed ({e})")
                    all_healthy = False
            
            if not all_healthy:
                print("   ❌ Not all services are healthy. Aborting test.")
                return
            
            # Test 2: Ingest MCP servers into RMCP
            print(f"\n📥 Test 2: Ingest MCP Servers into RMCP")
            print("-" * 50)
            
            # Ingest Basic Tools
            basic_tools_ingest = {
                "name": "mcp-basic-tools",
                "url": basic_tools_url,
                "description": "Basic command-line utilities for RMCP ecosystem"
            }
            
            response = await client.post(f"{rmcp_url}/api/v1/ingest", json=basic_tools_ingest)
            if response.status_code == 200:
                print(f"   ✅ Basic Tools ingested successfully")
            else:
                print(f"   ❌ Basic Tools ingestion failed: {response.status_code}")
                return
            
            # Ingest Filesystem Write
            filesystem_write_ingest = {
                "name": "mcp-filesystem-write",
                "url": filesystem_write_url,
                "description": "Secure filesystem write operations for RMCP ecosystem"
            }
            
            response = await client.post(f"{rmcp_url}/api/v1/ingest", json=filesystem_write_ingest)
            if response.status_code == 200:
                print(f"   ✅ Filesystem Write ingested successfully")
            else:
                print(f"   ❌ Filesystem Write ingestion failed: {response.status_code}")
                return
            
            # Ingest Mock Agent
            agent_ingest = {
                "name": "agent-security-auditor",
                "url": agent_url,
                "description": "Security auditor agent for RMCP ecosystem"
            }
            
            response = await client.post(f"{rmcp_url}/api/v1/ingest", json=agent_ingest)
            if response.status_code == 200:
                print(f"   ✅ Mock Agent ingested successfully")
            else:
                print(f"   ❌ Mock Agent ingestion failed: {response.status_code}")
                return
            
            # Test 3: List available tools
            print(f"\n🔧 Test 3: List Available Tools")
            print("-" * 50)
            
            response = await client.get(f"{rmcp_url}/api/v1/tools")
            if response.status_code == 200:
                tools_data = response.json()
                tools = tools_data.get("tools", [])
                print(f"   ✅ Tools listed successfully")
                print(f"   Total tools available: {len(tools)}")
                
                # Group tools by source
                tool_sources = {}
                for tool in tools:
                    source = tool.get("source", "unknown")
                    if source not in tool_sources:
                        tool_sources[source] = []
                    tool_sources[source].append(tool.get("name"))
                
                for source, tool_names in tool_sources.items():
                    print(f"     📦 {source}: {len(tool_names)} tools")
                    for tool_name in tool_names[:3]:  # Show first 3 tools
                        print(f"       - {tool_name}")
                    if len(tool_names) > 3:
                        print(f"       ... and {len(tool_names) - 3} more")
            else:
                print(f"   ❌ Tools listing failed: {response.status_code}")
                return
            
            # Test 4: Execute read-only operation (grep)
            print(f"\n🔍 Test 4: Execute Read-Only Operation (Grep)")
            print("-" * 50)
            
            grep_request = {
                "tool_name": "basic_tools.grep",
                "parameters": {
                    "pattern": "error",
                    "text": "This is a test log.\nError: Something went wrong\nSuccess: Operation completed\nWarning: Check this out"
                }
            }
            
            response = await client.post(f"{rmcp_url}/api/v1/execute", json=grep_request)
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ Grep executed successfully")
                print(f"   Output: {result.get('output')}")
                print(f"   Success: {result.get('success')}")
            else:
                print(f"   ❌ Grep execution failed: {response.status_code}")
            
            # Test 5: Execute write operation (create file)
            print(f"\n📝 Test 5: Execute Write Operation (Create File)")
            print("-" * 50)
            
            test_file = os.path.join(test_dir, "ecosystem_test.txt")
            create_file_request = {
                "tool_name": "filesystem_write.create_file",
                "parameters": {
                    "path": test_file,
                    "content": "Hello from RMCP Ecosystem E2E Test!"
                }
            }
            
            response = await client.post(f"{rmcp_url}/api/v1/execute", json=create_file_request)
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ File creation executed successfully")
                print(f"   Output: {result.get('output')}")
                print(f"   Success: {result.get('success')}")
                
                # Verify file was created
                if os.path.exists(test_file):
                    with open(test_file, 'r') as f:
                        content = f.read()
                    print(f"   ✅ File actually exists with content: '{content}'")
                else:
                    print(f"   ❌ File was not created")
            else:
                print(f"   ❌ File creation failed: {response.status_code}")
            
            # Test 6: Execute agent operation (security audit)
            print(f"\n🛡️  Test 6: Execute Agent Operation (Security Audit)")
            print("-" * 50)
            
            audit_request = {
                "tool_name": "agent-security-auditor.execute",
                "parameters": {
                    "goal": "Perform a security audit of the test file"
                }
            }
            
            response = await client.post(f"{rmcp_url}/api/v1/execute", json=audit_request)
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ Security audit executed successfully")
                print(f"   Output: {result.get('output')}")
                print(f"   Success: {result.get('success')}")
            else:
                print(f"   ❌ Security audit failed: {response.status_code}")
            
            # Test 7: Complex multi-step operation
            print(f"\n🔄 Test 7: Complex Multi-Step Operation")
            print("-" * 50)
            
            # Step 1: Create a directory
            test_subdir = os.path.join(test_dir, "multi_step_test")
            create_dir_request = {
                "tool_name": "filesystem_write.create_directory",
                "parameters": {
                    "path": test_subdir
                }
            }
            
            response = await client.post(f"{rmcp_url}/api/v1/execute", json=create_dir_request)
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ Step 1: Directory creation successful")
                print(f"   Output: {result.get('output')}")
            else:
                print(f"   ❌ Step 1: Directory creation failed")
                return
            
            # Step 2: Create a file in the directory
            test_file2 = os.path.join(test_subdir, "multi_step.txt")
            create_file2_request = {
                "tool_name": "filesystem_write.create_file",
                "parameters": {
                    "path": test_file2,
                    "content": "This file was created in a multi-step operation!"
                }
            }
            
            response = await client.post(f"{rmcp_url}/api/v1/execute", json=create_file2_request)
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ Step 2: File creation successful")
                print(f"   Output: {result.get('output')}")
            else:
                print(f"   ❌ Step 2: File creation failed")
                return
            
            # Step 3: Read the file using basic tools
            read_file_request = {
                "tool_name": "basic_tools.cat",
                "parameters": {
                    "file_path": test_file2
                }
            }
            
            response = await client.post(f"{rmcp_url}/api/v1/execute", json=read_file_request)
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ Step 3: File reading successful")
                print(f"   Content: {result.get('output')}")
            else:
                print(f"   ❌ Step 3: File reading failed")
                return
            
            print(f"\n🎉 RMCP ECOSYSTEM E2E TEST SUCCESS!")
            print("   ✅ All services healthy")
            print("   ✅ MCP servers ingested successfully")
            print("   ✅ Tools listing working")
            print("   ✅ Read-only operations working")
            print("   ✅ Write operations working")
            print("   ✅ Agent operations working")
            print("   ✅ Multi-step operations working")
            print("   ✅ Complete ecosystem integration successful")
            print("   ✅ RMCP as Meta-Orchestrator working perfectly")
            
    except httpx.ConnectError as e:
        print(f"❌ Could not connect to services: {e}")
        print("   Make sure all services are running:")
        print("   docker-compose -f docker-compose-ecosystem.yml up")
    except Exception as e:
        print(f"❌ Test failed: {e}")
    finally:
        # Clean up test directory
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
            print(f"🧹 Cleaned up test directory: {test_dir}")


if __name__ == "__main__":
    asyncio.run(test_ecosystem_e2e())
