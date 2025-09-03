"""
Mock Agent for RMCP Ecosystem Testing
Simple agent that responds to health checks and basic requests
"""

import asyncio
import os
import sys
from typing import Dict, Any
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Mock Security Auditor Agent",
    description="Mock agent for RMCP ecosystem testing",
    version="1.0.0"
)

class AgentRequest(BaseModel):
    task: str
    context: Dict[str, Any] = {}

class AgentResponse(BaseModel):
    status: str
    result: str
    agent_name: str

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Mock Security Auditor Agent",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "mock-security-auditor"}

@app.post("/execute", response_model=AgentResponse)
async def execute_task(request: AgentRequest):
    """Execute a mock task"""
    try:
        # Simulate some processing
        await asyncio.sleep(0.1)
        
        result = f"Mock security audit completed for task: {request.task}"
        
        return AgentResponse(
            status="completed",
            result=result,
            agent_name="security-auditor"
        )
    except Exception as e:
        logger.error(f"Error executing task: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
async def get_status():
    """Get agent status"""
    return {
        "agent_name": "security-auditor",
        "status": "active",
        "capabilities": ["security_audit", "compliance_check", "vulnerability_scan"]
    }

async def connect_to_rmcp():
    """Connect to RMCP server"""
    rmcp_url = os.getenv("RMCP_URL", "http://rmcp:8000")
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{rmcp_url}/health")
            if response.status_code == 200:
                logger.info(f"Successfully connected to RMCP at {rmcp_url}")
                return True
            else:
                logger.warning(f"RMCP health check failed: {response.status_code}")
                return False
    except Exception as e:
        logger.warning(f"Failed to connect to RMCP: {e}")
        return False

@app.on_event("startup")
async def startup_event():
    """Startup event handler"""
    logger.info("Mock Security Auditor Agent starting up...")
    
    # Try to connect to RMCP
    await connect_to_rmcp()
    
    logger.info("Mock Security Auditor Agent ready!")

if __name__ == "__main__":
    port = int(os.getenv("AGENT_PORT", "8004"))
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )

