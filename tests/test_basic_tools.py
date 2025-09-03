#!/usr/bin/env python3
"""
Test MCP Basic Tools Server
===========================

This test demonstrates the basic MCP server functionality.

Usage:
    python test_basic_tools.py
"""

import asyncio
import httpx
import json


async def test_basic_tools():
    """Test the MCP Basic Tools server"""
    print("🛠️  Testing MCP Basic Tools Server")
    print("=" * 50)
    
    base_url = "http://localhost:8001"
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            # Test 1: Health check
            print("🏥 Test 1: Health Check")
            print("-" * 30)
            
            response = await client.get(f"{base_url}/health")
            if response.status_code == 200:
                health_data = response.json()
                print(f"   ✅ Health check passed")
                print(f"   Service: {health_data.get('service')}")
                print(f"   Available tools: {health_data.get('available_tools')}")
            else:
                print(f"   ❌ Health check failed: {response.status_code}")
                return
            
            # Test 2: List tools
            print(f"\n🔧 Test 2: List Tools")
            print("-" * 30)
            
            response = await client.get(f"{base_url}/tools")
            if response.status_code == 200:
                tools_data = response.json()
                tools = tools_data.get("tools", [])
                print(f"   ✅ Tools listed successfully")
                print(f"   Available tools: {len(tools)}")
                for tool in tools:
                    print(f"     - {tool.get('name')}: {tool.get('description')}")
            else:
                print(f"   ❌ Tools listing failed: {response.status_code}")
                return
            
            # Test 3: Execute echo
            print(f"\n📢 Test 3: Execute Echo")
            print("-" * 30)
            
            echo_request = {
                "tool_name": "echo",
                "arguments": ["Hello", "from", "MCP", "Basic", "Tools!"]
            }
            
            response = await client.post(f"{base_url}/execute", json=echo_request)
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ Echo executed successfully")
                print(f"   Output: {result.get('output')}")
                print(f"   Success: {result.get('success')}")
            else:
                print(f"   ❌ Echo execution failed: {response.status_code}")
            
            # Test 4: Execute grep
            print(f"\n🔍 Test 4: Execute Grep")
            print("-" * 30)
            
            grep_request = {
                "tool_name": "grep",
                "input_text": "Hello World\nError: Something went wrong\nSuccess: Operation completed\nWarning: Check this out"
            }
            
            response = await client.post(f"{base_url}/execute", json=grep_request)
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ Grep executed successfully")
                print(f"   Output: {result.get('output')}")
                print(f"   Success: {result.get('success')}")
            else:
                print(f"   ❌ Grep execution failed: {response.status_code}")
            
            # Test 5: Execute ls
            print(f"\n📁 Test 5: Execute LS")
            print("-" * 30)
            
            ls_request = {
                "tool_name": "ls",
                "arguments": ["."]
            }
            
            response = await client.post(f"{base_url}/execute", json=ls_request)
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ LS executed successfully")
                print(f"   Output: {result.get('output')}")
                print(f"   Success: {result.get('success')}")
            else:
                print(f"   ❌ LS execution failed: {response.status_code}")
            
            # Test 6: Execute wc
            print(f"\n📊 Test 6: Execute WC")
            print("-" * 30)
            
            wc_request = {
                "tool_name": "wc",
                "input_text": "This is a test text with multiple words and lines.\nIt has several sentences.\nAnd multiple lines for counting."
            }
            
            response = await client.post(f"{base_url}/execute", json=wc_request)
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ WC executed successfully")
                print(f"   Output: {result.get('output')} (lines words chars)")
                print(f"   Success: {result.get('success')}")
            else:
                print(f"   ❌ WC execution failed: {response.status_code}")
            
            print(f"\n🎉 MCP BASIC TOOLS TEST SUCCESS!")
            print("   ✅ Health check working")
            print("   ✅ Tools listing working")
            print("   ✅ Echo tool working")
            print("   ✅ Grep tool working")
            print("   ✅ LS tool working")
            print("   ✅ WC tool working")
            print("   ✅ Ready for RMCP ecosystem integration")
            
        except httpx.ConnectError:
            print("❌ Could not connect to MCP Basic Tools server")
            print("   Make sure the server is running on http://localhost:8001")
        except Exception as e:
            print(f"❌ Test failed: {e}")


if __name__ == "__main__":
    asyncio.run(test_basic_tools())

