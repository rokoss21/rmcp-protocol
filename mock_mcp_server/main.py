"""
Mock MCP Server - Test MCP Server for RMCP
This is a simple FastAPI-based MCP server that demonstrates the MCP protocol
"""

import json
import time
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn


class MCPRequest(BaseModel):
    """Request model for MCP tool execution"""
    tool_name: str = Field(..., description="Name of the tool to execute")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Tool parameters")


class MCPResponse(BaseModel):
    """Response model for MCP tool execution"""
    result: Dict[str, Any] = Field(..., description="Tool execution result")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Execution metadata")


class MockMCPServer:
    """
    Mock MCP Server for testing RMCP
    
    This server demonstrates the MCP protocol by:
    - Accepting tool execution requests
    - Simulating tool behavior
    - Returning structured results
    """
    
    def __init__(self):
        self.execution_count = 0
        self.available_tools = {
            "grep": {
                "description": "Search text in files",
                "parameters": ["pattern", "file"],
                "simulation": self._simulate_grep
            },
            "find": {
                "description": "Find files by name",
                "parameters": ["name", "path"],
                "simulation": self._simulate_find
            },
            "cat": {
                "description": "Display file contents",
                "parameters": ["file"],
                "simulation": self._simulate_cat
            }
        }
    
    def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> MCPResponse:
        """Execute a tool with given parameters"""
        self.execution_count += 1
        start_time = time.time()
        
        if tool_name not in self.available_tools:
            raise HTTPException(
                status_code=404, 
                detail=f"Tool '{tool_name}' not found. Available tools: {list(self.available_tools.keys())}"
            )
        
        try:
            # Simulate tool execution
            tool_info = self.available_tools[tool_name]
            result = tool_info["simulation"](parameters)
            
            execution_time = int((time.time() - start_time) * 1000)
            
            return MCPResponse(
                result=result,
                metadata={
                    "tool_name": tool_name,
                    "execution_time_ms": execution_time,
                    "execution_count": self.execution_count,
                    "server": "mock-mcp-server"
                }
            )
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Tool execution failed: {str(e)}")
    
    def _simulate_grep(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate grep tool execution"""
        pattern = parameters.get("pattern", "")
        file = parameters.get("file", "")
        
        # Simulate processing time
        time.sleep(0.1)
        
        # Mock results based on pattern
        if "error" in pattern.lower():
            matches = [
                {"line": 15, "content": "ERROR: Database connection failed"},
                {"line": 23, "content": "ERROR: Invalid input parameter"},
                {"line": 45, "content": "ERROR: File not found"}
            ]
        elif "todo" in pattern.lower():
            matches = [
                {"line": 12, "content": "TODO: Implement user authentication"},
                {"line": 28, "content": "TODO: Add error handling"},
                {"line": 67, "content": "TODO: Optimize database queries"}
            ]
        else:
            matches = [
                {"line": 5, "content": f"Found pattern '{pattern}' in {file}"},
                {"line": 12, "content": f"Another match for '{pattern}'"},
                {"line": 18, "content": f"Third occurrence of '{pattern}'"}
            ]
        
        return {
            "matches": matches,
            "total_matches": len(matches),
            "file": file,
            "pattern": pattern
        }
    
    def _simulate_find(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate find tool execution"""
        name = parameters.get("name", "*")
        path = parameters.get("path", ".")
        
        # Simulate processing time
        time.sleep(0.2)
        
        # Mock results
        files = [
            f"{path}/file1.py",
            f"{path}/file2.js",
            f"{path}/config.json",
            f"{path}/README.md"
        ]
        
        return {
            "files": files,
            "total_files": len(files),
            "search_path": path,
            "search_pattern": name
        }
    
    def _simulate_cat(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate cat tool execution"""
        file = parameters.get("file", "")
        
        # Simulate processing time
        time.sleep(0.05)
        
        # Mock file content
        content = f"""# {file}
This is a mock file content for testing purposes.

Line 1: Some content here
Line 2: More content
Line 3: Even more content

End of file.
"""
        
        return {
            "content": content,
            "file": file,
            "lines": len(content.split('\n')),
            "size_bytes": len(content.encode('utf-8'))
        }


# FastAPI application
app = FastAPI(
    title="Mock MCP Server API",
    description="Test MCP Server for RMCP",
    version="1.0.0"
)

# Global MCP server instance
mcp_server = MockMCPServer()


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "server": "mock-mcp-server",
        "available_tools": list(mcp_server.available_tools.keys()),
        "execution_count": mcp_server.execution_count
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "server": "mock-mcp-server"}


@app.get("/tools")
async def list_tools():
    """List available tools"""
    return {
        "tools": {
            name: {
                "description": info["description"],
                "parameters": info["parameters"]
            }
            for name, info in mcp_server.available_tools.items()
        }
    }


@app.post("/execute", response_model=MCPResponse)
async def execute(request: MCPRequest):
    """
    Execute MCP tool
    
    This is the main endpoint that RMCP will call to execute tools.
    """
    try:
        response = mcp_server.execute_tool(request.tool_name, request.parameters)
        return response
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats():
    """Get server statistics"""
    return {
        "server": "mock-mcp-server",
        "execution_count": mcp_server.execution_count,
        "available_tools": len(mcp_server.available_tools),
        "status": "active"
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

