"""
Request and response models for RMCP API
"""

from typing import Any, Dict, Optional, List
from pydantic import BaseModel, Field


class RouteRequest(BaseModel):
    """Request for task routing"""
    tool_name: str = Field(..., description="Tool name (always 'rmcp.route')")
    parameters: Dict[str, Any] = Field(..., description="Request parameters")
    
    @property
    def goal(self) -> str:
        """Extract goal from parameters"""
        return self.parameters.get("goal", "")
    
    @property
    def context(self) -> Dict[str, Any]:
        """Extract context from parameters"""
        return self.parameters.get("context", {})


class ExecuteResponse(BaseModel):
    """Response to task execution"""
    status: str = Field(..., description="Execution status: SUCCESS, ERROR, PARTIAL")
    summary: str = Field(..., description="Brief summary of results")
    data: Dict[str, Any] = Field(default_factory=dict, description="Result data")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Execution metadata")
    confidence: Optional[float] = Field(None, description="Confidence in result (0.0-1.0)")
    execution_time_ms: Optional[int] = Field(None, description="Execution time in milliseconds")
    
    class Config:
        json_encoders = {
            # Add custom encoders if needed
        }
