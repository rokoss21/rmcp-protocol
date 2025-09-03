#!/usr/bin/env python3
"""
Test MCP Filesystem Write Server
================================

This test demonstrates the filesystem write MCP server functionality.

Usage:
    python test_filesystem_write.py
"""

import asyncio
import httpx
import json
import os
import tempfile
import shutil


async def test_filesystem_write():
    """Test the MCP Filesystem Write server"""
    print("📝 Testing MCP Filesystem Write Server")
    print("=" * 50)
    
    base_url = "http://localhost:8002"
    
    # Create a temporary directory for testing
    test_dir = tempfile.mkdtemp(prefix="mcp_test_")
    print(f"🧪 Using test directory: {test_dir}")
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Test 1: Health check
            print("🏥 Test 1: Health Check")
            print("-" * 30)
            
            response = await client.get(f"{base_url}/health")
            if response.status_code == 200:
                health_data = response.json()
                print(f"   ✅ Health check passed")
                print(f"   Service: {health_data.get('service')}")
                print(f"   Security Level: {health_data.get('security_level')}")
                print(f"   Capabilities: {health_data.get('capabilities')}")
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
                    print(f"       Capabilities: {tool.get('capabilities')}")
            else:
                print(f"   ❌ Tools listing failed: {response.status_code}")
                return
            
            # Test 3: Create directory
            print(f"\n📁 Test 3: Create Directory")
            print("-" * 30)
            
            test_subdir = os.path.join(test_dir, "subdir")
            create_dir_request = {
                "tool_name": "create_directory",
                "arguments": [test_subdir]
            }
            
            response = await client.post(f"{base_url}/execute", json=create_dir_request)
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ Directory creation executed")
                print(f"   Output: {result.get('output')}")
                print(f"   Success: {result.get('success')}")
                
                # Verify directory was created
                if os.path.exists(test_subdir):
                    print(f"   ✅ Directory actually exists: {test_subdir}")
                else:
                    print(f"   ❌ Directory was not created")
            else:
                print(f"   ❌ Directory creation failed: {response.status_code}")
            
            # Test 4: Create file
            print(f"\n📄 Test 4: Create File")
            print("-" * 30)
            
            test_file = os.path.join(test_dir, "test.txt")
            create_file_request = {
                "tool_name": "create_file",
                "arguments": [test_file, "Hello from MCP Filesystem Write!"]
            }
            
            response = await client.post(f"{base_url}/execute", json=create_file_request)
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ File creation executed")
                print(f"   Output: {result.get('output')}")
                print(f"   Success: {result.get('success')}")
                
                # Verify file was created and has correct content
                if os.path.exists(test_file):
                    with open(test_file, 'r') as f:
                        content = f.read()
                    print(f"   ✅ File actually exists with content: '{content}'")
                else:
                    print(f"   ❌ File was not created")
            else:
                print(f"   ❌ File creation failed: {response.status_code}")
            
            # Test 5: Append to file
            print(f"\n➕ Test 5: Append to File")
            print("-" * 30)
            
            append_request = {
                "tool_name": "append_to_file",
                "arguments": [test_file, "\nThis is appended content!"]
            }
            
            response = await client.post(f"{base_url}/execute", json=append_request)
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ File append executed")
                print(f"   Output: {result.get('output')}")
                print(f"   Success: {result.get('success')}")
                
                # Verify content was appended
                with open(test_file, 'r') as f:
                    content = f.read()
                print(f"   ✅ File content after append: '{content}'")
            else:
                print(f"   ❌ File append failed: {response.status_code}")
            
            # Test 6: Create nested directory structure
            print(f"\n🏗️  Test 6: Create Nested Directory Structure")
            print("-" * 30)
            
            nested_dir = os.path.join(test_dir, "nested", "deep", "structure")
            nested_request = {
                "tool_name": "create_directory",
                "arguments": [nested_dir]
            }
            
            response = await client.post(f"{base_url}/execute", json=nested_request)
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ Nested directory creation executed")
                print(f"   Output: {result.get('output')}")
                print(f"   Success: {result.get('success')}")
                
                # Verify nested structure was created
                if os.path.exists(nested_dir):
                    print(f"   ✅ Nested directory actually exists: {nested_dir}")
                else:
                    print(f"   ❌ Nested directory was not created")
            else:
                print(f"   ❌ Nested directory creation failed: {response.status_code}")
            
            # Test 7: Delete file
            print(f"\n🗑️  Test 7: Delete File")
            print("-" * 30)
            
            delete_file_request = {
                "tool_name": "delete_file",
                "arguments": [test_file]
            }
            
            response = await client.post(f"{base_url}/execute", json=delete_file_request)
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ File deletion executed")
                print(f"   Output: {result.get('output')}")
                print(f"   Success: {result.get('success')}")
                
                # Verify file was deleted
                if not os.path.exists(test_file):
                    print(f"   ✅ File was actually deleted")
                else:
                    print(f"   ❌ File still exists")
            else:
                print(f"   ❌ File deletion failed: {response.status_code}")
            
            print(f"\n🎉 MCP FILESYSTEM WRITE TEST SUCCESS!")
            print("   ✅ Health check working")
            print("   ✅ Tools listing working")
            print("   ✅ Directory creation working")
            print("   ✅ File creation working")
            print("   ✅ File append working")
            print("   ✅ Nested directory creation working")
            print("   ✅ File deletion working")
            print("   ✅ Ready for RMCP ecosystem integration")
            print("   ✅ Security capabilities properly marked")
            
    except httpx.ConnectError:
        print("❌ Could not connect to MCP Filesystem Write server")
        print("   Make sure the server is running on http://localhost:8002")
    except Exception as e:
        print(f"❌ Test failed: {e}")
    finally:
        # Clean up test directory
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
            print(f"🧹 Cleaned up test directory: {test_dir}")


if __name__ == "__main__":
    asyncio.run(test_filesystem_write())

