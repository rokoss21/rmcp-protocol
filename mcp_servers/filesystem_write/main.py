"""
MCP Filesystem Write Server
===========================

A secure MCP server providing filesystem write operations:
- create_directory: Create directories
- create_file: Create files with content
- append_to_file: Append content to files
- delete_file: Delete files
- delete_directory: Delete directories (dangerous)

All operations are marked with appropriate security capabilities.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import os
import shutil
import uvicorn
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="MCP Filesystem Write Server",
    description="Secure filesystem write operations for RMCP ecosystem",
    version="1.0.0"
)

class ToolRequest(BaseModel):
    """Request model for tool execution"""
    tool_name: str = Field(..., description="Name of the tool to execute")
    arguments: List[str] = Field(default_factory=list, description="Command line arguments (legacy)")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Tool parameters (modern format)")
    input_text: Optional[str] = Field(None, description="Input text for tools that need it")
    working_directory: Optional[str] = Field(None, description="Working directory for execution")

class ToolResponse(BaseModel):
    """Response model for tool execution"""
    success: bool = Field(..., description="Whether execution was successful")
    output: str = Field(..., description="Tool output")
    error: Optional[str] = Field(None, description="Error message if failed")
    exit_code: int = Field(..., description="Exit code of the command")
    tool_name: str = Field(..., description="Name of the executed tool")

# Available tools with security capabilities
AVAILABLE_TOOLS = {
    "create_directory": {
        "name": "create_directory",
        "description": "Create a new directory (mkdir -p)",
        "capabilities": ["filesystem:write", "mutates_state"],
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory path to create"}
            },
            "required": ["path"]
        }
    },
    "create_file": {
        "name": "create_file",
        "description": "Create a new file with content",
        "capabilities": ["filesystem:write", "mutates_state"],
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to create"},
                "content": {"type": "string", "description": "Content to write to the file"}
            },
            "required": ["path", "content"]
        }
    },
    "append_to_file": {
        "name": "append_to_file",
        "description": "Append content to an existing file",
        "capabilities": ["filesystem:write", "mutates_state"],
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to append to"},
                "content": {"type": "string", "description": "Content to append"}
            },
            "required": ["path", "content"]
        }
    },
    "delete_file": {
        "name": "delete_file",
        "description": "Delete a file",
        "capabilities": ["filesystem:write", "mutates_state", "dangerous"],
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to delete"}
            },
            "required": ["path"]
        }
    },
    "delete_directory": {
        "name": "delete_directory",
        "description": "Delete a directory and all its contents (rm -rf)",
        "capabilities": ["filesystem:write", "mutates_state", "dangerous", "requires_human_approval"],
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory path to delete"}
            },
            "required": ["path"]
        }
    }
}

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "MCP Filesystem Write Server",
        "version": "1.0.0",
        "status": "running",
        "available_tools": list(AVAILABLE_TOOLS.keys()),
        "security_note": "This server provides filesystem write operations with security capabilities"
    }

@app.get("/tools")
async def list_tools():
    """List available MCP tools with security capabilities"""
    return {
        "tools": [
            {
                "name": f"filesystem_write.{tool_name}",
                "description": tool_info["description"],
                "capabilities": tool_info["capabilities"],
                "input_schema": tool_info["input_schema"]
            }
            for tool_name, tool_info in AVAILABLE_TOOLS.items()
        ]
    }

@app.post("/execute")
async def execute_tool(request: ToolRequest):
    """Execute a filesystem write tool"""
    try:
        logger.info(f"Executing filesystem write tool: {request.tool_name}")
        
        # Validate tool name
        if request.tool_name not in AVAILABLE_TOOLS:
            raise HTTPException(
                status_code=400, 
                detail=f"Unknown tool: {request.tool_name}. Available tools: {list(AVAILABLE_TOOLS.keys())}"
            )
        
        # Execute the tool
        result = await _execute_tool_command(request)
        
        return ToolResponse(
            success=result["success"],
            output=result["output"],
            error=result.get("error"),
            exit_code=result["exit_code"],
            tool_name=request.tool_name
        )
        
    except Exception as e:
        logger.error(f"Tool execution failed: {e}")
        return ToolResponse(
            success=False,
            output="",
            error=str(e),
            exit_code=1,
            tool_name=request.tool_name
        )

async def _execute_tool_command(request: ToolRequest) -> Dict[str, Any]:
    """Execute the actual tool command"""
    
    if request.tool_name == "create_directory":
        return await _execute_create_directory(request)
    elif request.tool_name == "create_file":
        return await _execute_create_file(request)
    elif request.tool_name == "append_to_file":
        return await _execute_append_to_file(request)
    elif request.tool_name == "delete_file":
        return await _execute_delete_file(request)
    elif request.tool_name == "delete_directory":
        return await _execute_delete_directory(request)
    else:
        return {
            "success": False,
            "output": "",
            "error": f"Tool {request.tool_name} not implemented",
            "exit_code": 1
        }

async def _execute_create_directory(request: ToolRequest) -> Dict[str, Any]:
    """Execute create_directory command"""
    try:
        path = request.arguments[0] if request.arguments else ""
        
        if not path:
            return {
                "success": False,
                "output": "",
                "error": "No path provided for directory creation",
                "exit_code": 1
            }
        
        # Create directory with parents (mkdir -p)
        os.makedirs(path, exist_ok=True)
        
        return {
            "success": True,
            "output": f"Directory created successfully: {path}",
            "exit_code": 0
        }
        
    except Exception as e:
        return {
            "success": False,
            "output": "",
            "error": str(e),
            "exit_code": 1
        }

async def _execute_create_file(request: ToolRequest) -> Dict[str, Any]:
    """Execute create_file command"""
    try:
        logger.info(f"create_file request: args={request.arguments}, params={request.parameters}")
        
        # Support both modern parameters and legacy arguments
        if request.parameters:
            # Modern format: use parameters dict
            path = request.parameters.get("path")
            content = request.parameters.get("content", "")
            
            if not path:
                return {
                    "success": False,
                    "output": "",
                    "error": "Parameter 'path' is required",
                    "exit_code": 1
                }
        else:
            # Legacy format: use command line arguments
            if len(request.arguments) < 2:
                return {
                    "success": False,
                    "output": "",
                    "error": "Usage: create_file <path> <content>",
                    "exit_code": 1
                }
            
            path = request.arguments[0]
            content = " ".join(request.arguments[1:])
        
        # Create parent directories if they don't exist
        parent_dir = os.path.dirname(path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        
        # Write file
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return {
            "success": True,
            "output": f"File created successfully: {path}",
            "exit_code": 0
        }
        
    except Exception as e:
        return {
            "success": False,
            "output": "",
            "error": str(e),
            "exit_code": 1
        }

async def _execute_append_to_file(request: ToolRequest) -> Dict[str, Any]:
    """Execute append_to_file command"""
    try:
        if len(request.arguments) < 2:
            return {
                "success": False,
                "output": "",
                "error": "Usage: append_to_file <path> <content>",
                "exit_code": 1
            }
        
        path = request.arguments[0]
        content = " ".join(request.arguments[1:])
        
        # Append to file
        with open(path, 'a', encoding='utf-8') as f:
            f.write(content)
        
        return {
            "success": True,
            "output": f"Content appended successfully to: {path}",
            "exit_code": 0
        }
        
    except Exception as e:
        return {
            "success": False,
            "output": "",
            "error": str(e),
            "exit_code": 1
        }

async def _execute_delete_file(request: ToolRequest) -> Dict[str, Any]:
    """Execute delete_file command"""
    try:
        path = request.arguments[0] if request.arguments else ""
        
        if not path:
            return {
                "success": False,
                "output": "",
                "error": "No path provided for file deletion",
                "exit_code": 1
            }
        
        if not os.path.exists(path):
            return {
                "success": False,
                "output": "",
                "error": f"File does not exist: {path}",
                "exit_code": 1
            }
        
        if not os.path.isfile(path):
            return {
                "success": False,
                "output": "",
                "error": f"Path is not a file: {path}",
                "exit_code": 1
            }
        
        # Delete file
        os.remove(path)
        
        return {
            "success": True,
            "output": f"File deleted successfully: {path}",
            "exit_code": 0
        }
        
    except Exception as e:
        return {
            "success": False,
            "output": "",
            "error": str(e),
            "exit_code": 1
        }

async def _execute_delete_directory(request: ToolRequest) -> Dict[str, Any]:
    """Execute delete_directory command (dangerous operation)"""
    try:
        path = request.arguments[0] if request.arguments else ""
        
        if not path:
            return {
                "success": False,
                "output": "",
                "error": "No path provided for directory deletion",
                "exit_code": 1
            }
        
        if not os.path.exists(path):
            return {
                "success": False,
                "output": "",
                "error": f"Directory does not exist: {path}",
                "exit_code": 1
            }
        
        if not os.path.isdir(path):
            return {
                "success": False,
                "output": "",
                "error": f"Path is not a directory: {path}",
                "exit_code": 1
            }
        
        # Delete directory and all contents (rm -rf)
        shutil.rmtree(path)
        
        return {
            "success": True,
            "output": f"Directory deleted successfully: {path}",
            "exit_code": 0
        }
        
    except Exception as e:
        return {
            "success": False,
            "output": "",
            "error": str(e),
            "exit_code": 1
        }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "mcp-filesystem-write",
        "version": "1.0.0",
        "available_tools": len(AVAILABLE_TOOLS),
        "security_level": "high",
        "capabilities": ["filesystem:write", "mutates_state"]
    }

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8002,
        log_level="info"
    )

