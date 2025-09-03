#!/usr/bin/env python3
"""
RMCP API Server - Fractal Orchestration
=======================================

This module provides a REST API for RMCP, allowing agents to use RMCP
as a service for executing their own sub-tasks. This enables fractal
orchestration where agents can delegate work back to RMCP.

Usage:
    python rmcp_api.py --port 8000 --config config/test_config.yaml
"""

import asyncio
import argparse
import logging
from typing import Dict, Any, Optional
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rmcp.core.engine import RMCPEngine
from rmcp.models.execution import ExecutionPlan, ExecutionResult, ExecutionStrategy
from rmcp.models.tool import ToolPassport
from rmcp.storage.database import DatabaseManager
from rmcp.ingestion.capability_ingestor import CapabilityIngestor
from rmcp.planning.strategic_planner import StrategicThreeStagePlanner
from rmcp.execution.enhanced_executor import EnhancedExecutor
from rmcp.telemetry.telemetry_engine import TelemetryEngine
from rmcp.circuit_breaker.circuit_breaker_manager import CircuitBreakerManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for API
class ExecuteRequest(BaseModel):
    """Request model for agent execution via RMCP API"""
    goal: str = Field(..., description="The goal to achieve")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context")
    max_execution_time_ms: int = Field(default=30000, description="Maximum execution time in milliseconds")
    strategy: str = Field(default="solo", description="Execution strategy: solo, parallel, or dag")
    agent_id: Optional[str] = Field(default=None, description="ID of the requesting agent")

class ExecuteResponse(BaseModel):
    """Response model for agent execution via RMCP API"""
    success: bool
    result: Optional[ExecutionResult] = None
    error: Optional[str] = None
    execution_id: Optional[str] = None

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    agents_registered: int
    tools_registered: int

class RMCPAPIServer:
    """RMCP API Server for fractal orchestration"""
    
    def __init__(self, config_path: str, port: int = 8000):
        self.config_path = config_path
        self.port = port
        self.engine: Optional[RMCPEngine] = None
        self.app = FastAPI(
            title="RMCP API Server",
            description="Fractal Orchestration API for AI Agents",
            version="1.0.0"
        )
        self._setup_middleware()
        self._setup_routes()
    
    def _setup_middleware(self):
        """Setup CORS and other middleware"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # In production, specify actual origins
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/health", response_model=HealthResponse)
        async def health_check():
            """Health check endpoint"""
            if not self.engine:
                raise HTTPException(status_code=503, detail="Engine not initialized")
            
            # Get stats from engine
            stats = await self.engine.get_stats()
            
            return HealthResponse(
                status="healthy",
                version="1.0.0",
                agents_registered=stats.get("agents_count", 0),
                tools_registered=stats.get("tools_count", 0)
            )
        
        @self.app.post("/execute", response_model=ExecuteResponse)
        async def execute_for_agent(request: ExecuteRequest, background_tasks: BackgroundTasks):
            """Execute a task for an agent via RMCP"""
            if not self.engine:
                raise HTTPException(status_code=503, detail="Engine not initialized")
            
            try:
                logger.info(f"Received execution request from agent {request.agent_id}: {request.goal}")
                
                # Convert strategy string to enum
                strategy_map = {
                    "solo": ExecutionStrategy.SOLO,
                    "parallel": ExecutionStrategy.PARALLEL,
                    "dag": ExecutionStrategy.DAG
                }
                strategy = strategy_map.get(request.strategy, ExecutionStrategy.SOLO)
                
                # Create execution plan
                plan = ExecutionPlan(
                    goal=request.goal,
                    context=request.context or {},
                    strategy=strategy,
                    max_execution_time_ms=request.max_execution_time_ms
                )
                
                # Execute via RMCP engine
                result = await self.engine.execute(plan)
                
                # Log execution in background
                background_tasks.add_task(
                    self._log_agent_execution,
                    request.agent_id,
                    request.goal,
                    result
                )
                
                return ExecuteResponse(
                    success=result.status == "SUCCESS",
                    result=result,
                    execution_id=result.execution_id
                )
                
            except Exception as e:
                logger.error(f"Execution failed for agent {request.agent_id}: {e}")
                return ExecuteResponse(
                    success=False,
                    error=str(e)
                )
        
        @self.app.get("/agents")
        async def list_agents():
            """List all registered agents"""
            if not self.engine:
                raise HTTPException(status_code=503, detail="Engine not initialized")
            
            agents = await self.engine.list_agents()
            return {"agents": agents}
        
        @self.app.get("/tools")
        async def list_tools():
            """List all registered tools"""
            if not self.engine:
                raise HTTPException(status_code=503, detail="Engine not initialized")
            
            tools = await self.engine.list_tools()
            return {"tools": tools}
    
    async def _log_agent_execution(self, agent_id: Optional[str], goal: str, result: ExecutionResult):
        """Log agent execution for analytics"""
        if self.engine and hasattr(self.engine, 'telemetry'):
            await self.engine.telemetry.record_agent_execution(
                agent_id=agent_id,
                goal=goal,
                result=result
            )
    
    async def initialize(self):
        """Initialize the RMCP engine"""
        try:
            logger.info(f"Initializing RMCP API Server with config: {self.config_path}")
            
            # Initialize RMCP engine
            self.engine = RMCPEngine(self.config_path)
            await self.engine.initialize()
            
            logger.info("RMCP API Server initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RMCP API Server: {e}")
            raise
    
    async def shutdown(self):
        """Shutdown the RMCP engine"""
        if self.engine:
            await self.engine.shutdown()
            logger.info("RMCP API Server shutdown complete")
    
    def run(self):
        """Run the API server"""
        uvicorn.run(
            self.app,
            host="0.0.0.0",
            port=self.port,
            log_level="info"
        )

async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="RMCP API Server")
    parser.add_argument("--config", required=True, help="Path to RMCP config file")
    parser.add_argument("--port", type=int, default=8000, help="Port to run on")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    
    args = parser.parse_args()
    
    # Create and initialize server
    server = RMCPAPIServer(args.config, args.port)
    
    try:
        await server.initialize()
        server.run()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        await server.shutdown()

if __name__ == "__main__":
    asyncio.run(main())

