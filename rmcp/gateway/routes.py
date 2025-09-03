"""
API routes for RMCP
"""

from fastapi import APIRouter, HTTPException, Depends, Request
from typing import Dict, Any

from ..models.request import RouteRequest, ExecuteResponse
from ..models.plan import ExecutionPlan, ExecutionStrategy
from ..storage.database import DatabaseManager
from ..core.planner import SimplePlanner
from ..core.executor import SimpleExecutor
from ..core.ingestor import CapabilityIngestor
from ..planning.three_stage import ThreeStagePlanner
from ..llm.manager import LLMManager
from ..embeddings.manager import EmbeddingManager

router = APIRouter()


def get_db_manager(request: Request) -> DatabaseManager:
    """Dependency for getting DB manager from app state"""
    return request.app.state.db_manager


def get_planner(request: Request):
    """Dependency for getting planner from app state"""
    return request.app.state.three_stage_planner


def get_telemetry_engine(request: Request):
    """Dependency for getting telemetry engine from app state"""
    return getattr(request.app.state, 'telemetry_engine', None)


def get_executor(request: Request) -> SimpleExecutor:
    """Dependency for getting executor"""
    return SimpleExecutor(request.app.state.db_manager)


def get_ingestor(request: Request) -> CapabilityIngestor:
    """Dependency for getting ingestor"""
    llm_manager = getattr(request.app.state, 'llm_manager', None)
    config_manager = getattr(request.app.state, 'config_manager', None)
    return CapabilityIngestor(request.app.state.db_manager, config_manager, llm_manager)


@router.post("/execute", response_model=ExecuteResponse)
async def execute_route(
    route_request: RouteRequest,
    http_request: Request,
    planner = Depends(get_planner),
    executor: SimpleExecutor = Depends(get_executor),
    telemetry_engine = Depends(get_telemetry_engine)
) -> ExecuteResponse:
    """
    Main endpoint for task execution through RMCP
    
    Accepts request from LLM agent and returns execution result
    """
    try:
        # Request validation
        if route_request.tool_name != "rmcp.route":
            raise HTTPException(
                status_code=400,
                detail="Invalid tool_name. Must be 'rmcp.route'"
            )
        
        if not route_request.goal:
            raise HTTPException(
                status_code=400,
                detail="Missing 'goal' in parameters"
            )
        
        # Planning
        plan = await planner.plan(route_request.goal, route_request.context)
        
        # Execution
        result = await executor.execute(plan)
        
        # Record telemetry if available
        if telemetry_engine and plan.steps:
            # Record execution telemetry for each step
            for step in plan.steps:
                await telemetry_engine.record_tool_execution(
                    tool_id=step.tool_id,
                    success=result.status == "SUCCESS",
                    latency_ms=result.execution_time_ms or 0,
                    cost=0.0,  # TODO: Extract from result
                    request_text=route_request.goal,
                    priority=1  # High priority for execution events
                )
        
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Execution failed: {str(e)}"
        )


@router.post("/ingest")
async def ingest_capabilities(
    ingestor: CapabilityIngestor = Depends(get_ingestor)
) -> Dict[str, Any]:
    """
    Scan MCP servers and update tool catalog
    """
    try:
        result = await ingestor.ingest_all()
        return {
            "status": "success",
            "message": "Capabilities ingested successfully",
            "servers_scanned": result.get("servers_scanned", 0),
            "tools_discovered": result.get("tools_discovered", 0)
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ingestion failed: {str(e)}"
        )


@router.get("/tools")
async def list_tools(
    db_manager: DatabaseManager = Depends(get_db_manager)
) -> Dict[str, Any]:
    """
    Get list of all available tools
    """
    try:
        tools = db_manager.get_all_tools()
        return {
            "status": "success",
            "tools": [
                {
                    "id": tool.id,
                    "name": tool.name,
                    "description": tool.description,
                    "tags": tool.tags,
                    "capabilities": tool.capabilities,
                    "success_rate": tool.success_rate,
                    "p95_latency_ms": tool.p95_latency_ms
                }
                for tool in tools
            ],
            "total": len(tools)
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list tools: {str(e)}"
        )


@router.get("/tools/{tool_id}")
async def get_tool(
    tool_id: str,
    db_manager: DatabaseManager = Depends(get_db_manager)
) -> Dict[str, Any]:
    """
    Get information about specific tool
    """
    try:
        tool = db_manager.get_tool(tool_id)
        if not tool:
            raise HTTPException(
                status_code=404,
                detail=f"Tool {tool_id} not found"
            )
        
        return {
            "status": "success",
            "tool": {
                "id": tool.id,
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema,
                "output_schema": tool.output_schema,
                "tags": tool.tags,
                "capabilities": tool.capabilities,
                "success_rate": tool.success_rate,
                "p95_latency_ms": tool.p95_latency_ms,
                "cost_hint": tool.cost_hint
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get tool: {str(e)}"
        )


@router.get("/search")
async def search_tools(
    q: str,
    limit: int = 50,
    db_manager: DatabaseManager = Depends(get_db_manager)
) -> Dict[str, Any]:
    """
    Search tools by query
    """
    try:
        tools = db_manager.search_tools(q, limit)
        return {
            "status": "success",
            "query": q,
            "tools": [
                {
                    "id": tool.id,
                    "name": tool.name,
                    "description": tool.description,
                    "tags": tool.tags,
                    "capabilities": tool.capabilities
                }
                for tool in tools
            ],
            "total": len(tools)
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )


@router.get("/planner/stats")
async def get_planner_stats(
    planner: ThreeStagePlanner = Depends(get_planner)
) -> Dict[str, Any]:
    """
    Get three-stage planner statistics
    """
    try:
        stats = planner.get_planner_stats()
        return {
            "status": "success",
            "stats": stats
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get planner stats: {str(e)}"
        )


@router.get("/tools/{tool_id}/affinity")
async def get_tool_affinity(
    tool_id: str,
    planner: ThreeStagePlanner = Depends(get_planner)
) -> Dict[str, Any]:
    """
    Get tool affinity analysis
    """
    try:
        analysis = await planner.analyze_tool_affinity(tool_id)
        return {
            "status": "success",
            "tool_id": tool_id,
            "analysis": analysis
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to analyze tool affinity: {str(e)}"
        )


@router.post("/tools/{tool_id}/affinity/update")
async def update_tool_affinity(
    tool_id: str,
    request_data: Dict[str, Any],
    planner: ThreeStagePlanner = Depends(get_planner)
) -> Dict[str, Any]:
    """
    Update tool affinity with execution result
    """
    try:
        request_text = request_data.get("request_text", "")
        success = request_data.get("success", False)
        
        await planner.update_tool_affinity(tool_id, request_text, success)
        
        return {
            "status": "success",
            "message": f"Affinity updated for tool {tool_id}",
            "tool_id": tool_id
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update tool affinity: {str(e)}"
        )


@router.get("/system/status")
async def get_system_status(
    request: Request
) -> Dict[str, Any]:
    """
    Get comprehensive system status
    """
    try:
        status = {
            "status": "success",
            "version": "0.1.0",
            "phase": "2",
            "components": {}
        }
        
        # Check database
        db_manager = request.app.state.db_manager
        tools = db_manager.get_all_tools()
        status["components"]["database"] = {
            "status": "healthy",
            "tools_count": len(tools)
        }
        
        # Check planner
        planner = request.app.state.three_stage_planner
        if hasattr(planner, 'get_planner_stats'):
            planner_stats = planner.get_planner_stats()
            status["components"]["planner"] = {
                "status": "healthy",
                "type": planner_stats.get("planner_type", "unknown"),
                "semantic_ranking": planner_stats.get("semantic_ranking_enabled", False),
                "llm_planning": planner_stats.get("llm_planning_enabled", False)
            }
        else:
            status["components"]["planner"] = {
                "status": "healthy",
                "type": "simple",
                "semantic_ranking": False,
                "llm_planning": False
            }
        
        # Check telemetry engine
        telemetry_engine = getattr(request.app.state, 'telemetry_engine', None)
        if telemetry_engine:
            telemetry_stats = telemetry_engine.get_stats()
            status["components"]["telemetry"] = {
                "status": "healthy" if telemetry_stats["is_running"] else "stopped",
                "events_processed": telemetry_stats["events_processed"],
                "queue_sizes": telemetry_stats["queue_sizes"]
            }
        else:
            status["components"]["telemetry"] = {
                "status": "not_available"
            }
        
        # Check background curator
        background_curator = getattr(request.app.state, 'background_curator', None)
        if background_curator:
            curator_stats = background_curator.get_stats()
            status["components"]["curator"] = {
                "status": "healthy" if curator_stats["is_running"] else "stopped",
                "processing_cycles": curator_stats["processing_cycles"],
                "active_aggregators": curator_stats["active_tool_aggregators"]
            }
        else:
            status["components"]["curator"] = {
                "status": "not_available"
            }
        
        return status
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get system status: {str(e)}"
        )


@router.get("/telemetry/stats")
async def get_telemetry_stats(
    request: Request
) -> Dict[str, Any]:
    """
    Get telemetry engine statistics
    """
    try:
        telemetry_engine = getattr(request.app.state, 'telemetry_engine', None)
        if not telemetry_engine:
            raise HTTPException(
                status_code=404,
                detail="Telemetry engine not available"
            )
        
        stats = telemetry_engine.get_stats()
        return {
            "status": "success",
            "stats": stats
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get telemetry stats: {str(e)}"
        )


@router.get("/curator/stats")
async def get_curator_stats(
    request: Request
) -> Dict[str, Any]:
    """
    Get background curator statistics
    """
    try:
        background_curator = getattr(request.app.state, 'background_curator', None)
        if not background_curator:
            raise HTTPException(
                status_code=404,
                detail="Background curator not available"
            )
        
        stats = background_curator.get_stats()
        health = background_curator.get_health_status()
        
        return {
            "status": "success",
            "stats": stats,
            "health": health
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get curator stats: {str(e)}"
        )


# ============================================================================
# FRACTAL ORCHESTRATION ENDPOINTS (Phase 5)
# ============================================================================

@router.post("/agent/execute")
async def execute_for_agent(
    request_data: Dict[str, Any],
    http_request: Request,
    planner = Depends(get_planner),
    executor: SimpleExecutor = Depends(get_executor),
    telemetry_engine = Depends(get_telemetry_engine)
) -> Dict[str, Any]:
    """
    Execute a task for an agent via RMCP (Fractal Orchestration)
    
    This endpoint allows agents to use RMCP as a service for executing
    their own sub-tasks, enabling fractal orchestration.
    """
    try:
        # Extract request parameters
        goal = request_data.get("goal")
        context = request_data.get("context", {})
        max_execution_time_ms = request_data.get("max_execution_time_ms", 30000)
        strategy = request_data.get("strategy", "solo")
        agent_id = request_data.get("agent_id")
        
        if not goal:
            raise HTTPException(
                status_code=400,
                detail="Missing 'goal' in request data"
            )
        
        # Convert strategy string to enum
        strategy_map = {
            "solo": ExecutionStrategy.SOLO,
            "parallel": ExecutionStrategy.PARALLEL,
            "dag": ExecutionStrategy.DAG
        }
        execution_strategy = strategy_map.get(strategy, ExecutionStrategy.SOLO)
        
        # Create execution plan
        plan = ExecutionPlan(
            strategy=execution_strategy,
            steps=[],  # Will be populated by planner
            max_execution_time_ms=max_execution_time_ms,
            metadata={"goal": goal, "context": context}
        )
        
        # Execute via planner and executor
        result = await executor.execute(plan)
        
        # Record telemetry if available
        if telemetry_engine and plan.steps:
            for step in plan.steps:
                await telemetry_engine.record_tool_execution(
                    tool_id=step.tool_id,
                    success=result.status == "SUCCESS",
                    latency_ms=result.execution_time_ms or 0,
                    cost=0.0,
                    request_text=goal,
                    priority=2  # Medium priority for agent requests
                )
        
        return {
            "success": result.status == "SUCCESS",
            "result": result,
            "execution_id": result.execution_id,
            "agent_id": agent_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Agent execution failed: {str(e)}"
        )


@router.get("/agents")
async def list_agents(
    db_manager: DatabaseManager = Depends(get_db_manager)
) -> Dict[str, Any]:
    """
    List all registered agents (tools with tool_type='agent')
    """
    try:
        # Get all tools and filter for agents
        all_tools = db_manager.get_all_tools()
        agents = [
            {
                "id": tool.id,
                "name": tool.name,
                "description": tool.description,
                "specialization": getattr(tool, 'specialization', None),
                "abstraction_level": getattr(tool, 'abstraction_level', 'low'),
                "max_complexity": getattr(tool, 'max_complexity', 1.0),
                "endpoint": getattr(tool, 'endpoint', None),
                "success_rate": tool.success_rate,
                "p95_latency_ms": tool.p95_latency_ms
            }
            for tool in all_tools
            if getattr(tool, 'tool_type', 'atomic') == 'agent'
        ]
        
        return {
            "status": "success",
            "agents": agents,
            "total": len(agents)
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list agents: {str(e)}"
        )
