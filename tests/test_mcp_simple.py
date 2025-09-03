#!/usr/bin/env python3
"""
Simple test for the mock MCP server
"""

import requests
import json
import time
import subprocess
import sys

def test_mcp_server():
    """Test the mock MCP server"""
    print("🧪 Testing Mock MCP Server")
    
    # Start MCP server in background
    print("Starting MCP server...")
    process = subprocess.Popen([
        sys.executable, "-c", 
        """
import uvicorn
from main import app
uvicorn.run(app, host='0.0.0.0', port=8000, log_level='info')
"""
    ], cwd="mock_mcp_server")
    
    # Wait for server to start
    print("Waiting for MCP server to start...")
    time.sleep(5)
    
    try:
        # Test health endpoint
        print("Testing health endpoint...")
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ Health check passed")
            print(f"Response: {response.json()}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
        
        # Test tools endpoint
        print("Testing tools endpoint...")
        response = requests.get("http://localhost:8000/tools", timeout=5)
        if response.status_code == 200:
            tools = response.json()
            print("✅ Tools endpoint passed")
            print(f"Available tools: {list(tools['tools'].keys())}")
        else:
            print(f"❌ Tools endpoint failed: {response.status_code}")
            return False
        
        # Test execute endpoint with grep
        print("Testing grep tool execution...")
        request_data = {
            "tool_name": "grep",
            "parameters": {
                "pattern": "error",
                "file": "test.py"
            }
        }
        
        response = requests.post(
            "http://localhost:8000/execute",
            json=request_data,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Grep execution passed")
            print(f"Response: {json.dumps(result, indent=2)}")
            
            # Validate response structure
            if "result" in result and "metadata" in result:
                print("✅ Response structure valid")
                return True
            else:
                print("❌ Invalid response structure")
                return False
        else:
            print(f"❌ Grep execution failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.RequestException as e:
        print(f"❌ Request failed: {e}")
        return False
    finally:
        # Clean up
        print("Stopping MCP server...")
        process.terminate()
        process.wait()

if __name__ == "__main__":
    success = test_mcp_server()
    if success:
        print("🎉 MCP server test passed!")
        sys.exit(0)
    else:
        print("💥 MCP server test failed!")
        sys.exit(1)

