"""
Models for MCP tools and servers
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class Server(BaseModel):
    """MCP server"""
    id: str = Field(..., description="Unique server identifier")
    base_url: str = Field(..., description="Server base URL")
    description: Optional[str] = Field(None, description="Server description")
    added_at: datetime = Field(default_factory=datetime.utcnow, description="Addition time")


class Tool(BaseModel):
    """MCP tool"""
    id: str = Field(..., description="Unique tool identifier")
    server_id: str = Field(..., description="ID of server that owns the tool")
    name: str = Field(..., description="Tool name")
    description: Optional[str] = Field(None, description="Tool description")
    
    # Static data ("Visa")
    input_schema: Optional[Dict[str, Any]] = Field(None, description="JSON Schema for input parameters")
    output_schema: Optional[Dict[str, Any]] = Field(None, description="JSON Schema for output data")
    tags: List[str] = Field(default_factory=list, description="Tool tags")
    capabilities: List[str] = Field(default_factory=list, description="Tool capabilities")
    tool_type: str = Field(default="atomic", description="Tool type: 'atomic' for low-level tools, 'agent' for AI agents")
    
    # Agent-specific fields (optional, only for agents)
    specialization: Optional[str] = Field(default=None, description="Agent specialization (e.g., 'security', 'deployment', 'testing')")
    abstraction_level: str = Field(default="low", description="Abstraction level: 'low', 'medium', 'high'")
    max_complexity: float = Field(default=1.0, description="Maximum task complexity this agent can handle")
    avg_execution_time_ms: int = Field(default=5000, description="Average execution time in ms")
    
    # Dynamic data ("Border stamps")
    p95_latency_ms: float = Field(default=3000.0, description="95th percentile latency in ms")
    success_rate: float = Field(default=0.95, description="Percentage of successful executions")
    cost_hint: float = Field(default=0.0, description="Estimated execution cost")
    
    # Semantic affinity ("Golden element")
    affinity_embeddings: Optional[bytes] = Field(None, description="Binary embedding data")


class Agent(BaseModel):
    """AI Agent for meta-orchestration"""
    id: str = Field(..., description="Unique agent identifier")
    name: str = Field(..., description="Agent name")
    description: str = Field(..., description="Agent mission and capabilities")
    endpoint: str = Field(..., description="Agent API endpoint")
    input_schema: Dict[str, Any] = Field(default_factory=dict, description="Expected input schema")
    output_schema: Dict[str, Any] = Field(default_factory=dict, description="Expected output schema")
    capabilities: List[str] = Field(default_factory=list, description="Agent capabilities")
    tags: List[str] = Field(default_factory=list, description="Agent tags")
    
    # Agent-specific metadata
    specialization: str = Field(..., description="Agent specialization (e.g., 'security', 'deployment', 'testing')")
    abstraction_level: str = Field(default="high", description="Abstraction level: 'low', 'medium', 'high'")
    max_complexity: float = Field(default=1.0, description="Maximum task complexity this agent can handle")
    
    # Performance metrics
    success_rate: float = Field(default=0.95, description="Agent success rate")
    avg_execution_time_ms: int = Field(default=30000, description="Average execution time in ms")
    cost_per_execution: float = Field(default=0.1, description="Cost per execution")


class ToolPassport(BaseModel):
    """Tool passport for planner"""
    tool: Tool
    affinity_score: float = Field(default=0.0, description="Semantic affinity with request")
    final_score: float = Field(default=0.0, description="Final score for ranking")
    
    class Config:
        arbitrary_types_allowed = True


class AgentPassport(BaseModel):
    """Agent passport for strategic planner"""
    agent: Agent
    affinity_score: float = Field(default=0.0, description="Semantic affinity with request")
    strategic_score: float = Field(default=0.0, description="Strategic planning score")
    final_score: float = Field(default=0.0, description="Final score for ranking")
    
    class Config:
        arbitrary_types_allowed = True
