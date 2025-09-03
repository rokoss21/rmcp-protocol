#!/usr/bin/env python3
"""
Simple test for the mock agent
"""

import requests
import json
import time
import subprocess
import sys
from pathlib import Path

def test_agent():
    """Test the mock agent"""
    print("🧪 Testing Mock Agent")
    
    # Start agent in background
    print("Starting agent...")
    process = subprocess.Popen([
        sys.executable, "-c", 
        """
import uvicorn
from main import app
uvicorn.run(app, host='0.0.0.0', port=8001, log_level='info')
"""
    ], cwd="mock_agent")
    
    # Wait for agent to start
    print("Waiting for agent to start...")
    time.sleep(5)
    
    try:
        # Test health endpoint
        print("Testing health endpoint...")
        response = requests.get("http://localhost:8001/health", timeout=5)
        if response.status_code == 200:
            print("✅ Health check passed")
            print(f"Response: {response.json()}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
        
        # Test execute endpoint
        print("Testing execute endpoint...")
        request_data = {
            "goal": "Conduct security audit",
            "context": {"repo": "test"},
            "specialization": "security"
        }
        
        response = requests.post(
            "http://localhost:8001/execute",
            json=request_data,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Execute endpoint passed")
            print(f"Response: {json.dumps(result, indent=2)}")
            
            # Validate response structure
            required_fields = ["status", "summary", "data", "metadata", "execution_time_ms"]
            for field in required_fields:
                if field not in result:
                    print(f"❌ Missing field: {field}")
                    return False
            
            print("✅ All required fields present")
            return True
        else:
            print(f"❌ Execute endpoint failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.RequestException as e:
        print(f"❌ Request failed: {e}")
        return False
    finally:
        # Clean up
        print("Stopping agent...")
        process.terminate()
        process.wait()

if __name__ == "__main__":
    success = test_agent()
    if success:
        print("🎉 Agent test passed!")
        sys.exit(0)
    else:
        print("💥 Agent test failed!")
        sys.exit(1)

