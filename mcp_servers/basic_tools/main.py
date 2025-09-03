"""
MCP Basic Tools Server
=====================

A simple MCP server providing basic command-line utilities:
- grep: Search for patterns in text
- ls: List directory contents
- echo: Print text
- cat: Display file contents
- wc: Word count
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import subprocess
import os
import uvicorn
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="MCP Basic Tools Server",
    description="Basic command-line utilities for RMCP ecosystem",
    version="1.0.0"
)

class ToolRequest(BaseModel):
    """Request model for tool execution"""
    tool_name: str = Field(..., description="Name of the tool to execute")
    arguments: List[str] = Field(default_factory=list, description="Command line arguments")
    input_text: Optional[str] = Field(None, description="Input text for tools that need it")
    working_directory: Optional[str] = Field(None, description="Working directory for execution")

class ToolResponse(BaseModel):
    """Response model for tool execution"""
    success: bool = Field(..., description="Whether execution was successful")
    output: str = Field(..., description="Tool output")
    error: Optional[str] = Field(None, description="Error message if failed")
    exit_code: int = Field(..., description="Exit code of the command")
    tool_name: str = Field(..., description="Name of the executed tool")

# Available tools
AVAILABLE_TOOLS = {
    "grep": {
        "name": "grep",
        "description": "Search for patterns in text or files",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Pattern to search for"},
                "text": {"type": "string", "description": "Text to search in"},
                "case_sensitive": {"type": "boolean", "description": "Case sensitive search", "default": False}
            },
            "required": ["pattern", "text"]
        }
    },
    "ls": {
        "name": "ls",
        "description": "List directory contents",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory path to list", "default": "."},
                "detailed": {"type": "boolean", "description": "Show detailed information", "default": False}
            }
        }
    },
    "echo": {
        "name": "echo",
        "description": "Print text to output",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to print"}
            },
            "required": ["text"]
        }
    },
    "cat": {
        "name": "cat",
        "description": "Display file contents",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Path to the file to display"}
            },
            "required": ["file_path"]
        }
    },
    "wc": {
        "name": "wc",
        "description": "Word count for text or files",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to count words in"},
                "file_path": {"type": "string", "description": "File path to count words in"}
            }
        }
    }
}

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "MCP Basic Tools Server",
        "version": "1.0.0",
        "status": "running",
        "available_tools": list(AVAILABLE_TOOLS.keys())
    }

@app.get("/tools")
async def list_tools():
    """List available MCP tools"""
    return {
        "tools": [
            {
                "name": f"basic_tools.{tool_name}",
                "description": tool_info["description"],
                "input_schema": tool_info["input_schema"]
            }
            for tool_name, tool_info in AVAILABLE_TOOLS.items()
        ]
    }

@app.post("/execute")
async def execute_tool(request: ToolRequest):
    """Execute a basic tool"""
    try:
        logger.info(f"Executing tool: {request.tool_name}")
        
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
    
    if request.tool_name == "grep":
        return await _execute_grep(request)
    elif request.tool_name == "ls":
        return await _execute_ls(request)
    elif request.tool_name == "echo":
        return await _execute_echo(request)
    elif request.tool_name == "cat":
        return await _execute_cat(request)
    elif request.tool_name == "wc":
        return await _execute_wc(request)
    else:
        return {
            "success": False,
            "output": "",
            "error": f"Tool {request.tool_name} not implemented",
            "exit_code": 1
        }

async def _execute_grep(request: ToolRequest) -> Dict[str, Any]:
    """Execute grep command"""
    try:
        # Extract pattern and text from arguments or input_text
        if request.arguments:
            pattern = request.arguments[0]
            text = " ".join(request.arguments[1:]) if len(request.arguments) > 1 else request.input_text or ""
        else:
            # Parse from input_text if no arguments
            text = request.input_text or ""
            pattern = ".*"  # Default pattern
        
        if not text:
            return {
                "success": False,
                "output": "",
                "error": "No text provided for grep",
                "exit_code": 1
            }
        
        # Use Python's re module for grep functionality
        import re
        
        lines = text.split('\n')
        matches = []
        
        for line in lines:
            if re.search(pattern, line, re.IGNORECASE):
                matches.append(line)
        
        return {
            "success": True,
            "output": "\n".join(matches),
            "exit_code": 0
        }
        
    except Exception as e:
        return {
            "success": False,
            "output": "",
            "error": str(e),
            "exit_code": 1
        }

async def _execute_ls(request: ToolRequest) -> Dict[str, Any]:
    """Execute ls command"""
    try:
        path = request.arguments[0] if request.arguments else "."
        
        if not os.path.exists(path):
            return {
                "success": False,
                "output": "",
                "error": f"Path does not exist: {path}",
                "exit_code": 1
            }
        
        if os.path.isfile(path):
            return {
                "success": True,
                "output": path,
                "exit_code": 0
            }
        
        # List directory contents
        items = os.listdir(path)
        items.sort()
        
        return {
            "success": True,
            "output": "\n".join(items),
            "exit_code": 0
        }
        
    except Exception as e:
        return {
            "success": False,
            "output": "",
            "error": str(e),
            "exit_code": 1
        }

async def _execute_echo(request: ToolRequest) -> Dict[str, Any]:
    """Execute echo command"""
    try:
        text = " ".join(request.arguments) if request.arguments else request.input_text or ""
        
        return {
            "success": True,
            "output": text,
            "exit_code": 0
        }
        
    except Exception as e:
        return {
            "success": False,
            "output": "",
            "error": str(e),
            "exit_code": 1
        }

async def _execute_cat(request: ToolRequest) -> Dict[str, Any]:
    """Execute cat command"""
    try:
        file_path = request.arguments[0] if request.arguments else ""
        
        if not file_path:
            return {
                "success": False,
                "output": "",
                "error": "No file path provided",
                "exit_code": 1
            }
        
        if not os.path.exists(file_path):
            return {
                "success": False,
                "output": "",
                "error": f"File does not exist: {file_path}",
                "exit_code": 1
            }
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return {
            "success": True,
            "output": content,
            "exit_code": 0
        }
        
    except Exception as e:
        return {
            "success": False,
            "output": "",
            "error": str(e),
            "exit_code": 1
        }

async def _execute_wc(request: ToolRequest) -> Dict[str, Any]:
    """Execute wc command"""
    try:
        text = ""
        
        if request.arguments:
            file_path = request.arguments[0]
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            else:
                return {
                    "success": False,
                    "output": "",
                    "error": f"File does not exist: {file_path}",
                    "exit_code": 1
                }
        else:
            text = request.input_text or ""
        
        # Count words, lines, characters
        words = len(text.split())
        lines = len(text.split('\n'))
        chars = len(text)
        
        result = f"{lines} {words} {chars}"
        
        return {
            "success": True,
            "output": result,
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
        "service": "mcp-basic-tools",
        "version": "1.0.0",
        "available_tools": len(AVAILABLE_TOOLS)
    }

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        log_level="info"
    )

