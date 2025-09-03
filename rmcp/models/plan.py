"""
Models for execution plans
"""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field
from enum import Enum


class ExecutionStrategy(str, Enum):
    """Execution plan strategy"""
    SOLO = "solo"           # Single tool
    PARALLEL = "parallel"   # Parallel execution
    DAG = "dag"            # Dependency graph


class ExecutionStep(BaseModel):
    """Execution plan step"""
    tool_id: str = Field(..., description="ID of tool to execute")
    args: Dict[str, Any] = Field(default_factory=dict, description="Arguments for the tool")
    dependencies: List[str] = Field(default_factory=list, description="Dependencies on other steps")
    outputs: List[str] = Field(default_factory=list, description="Output variable names")
    timeout_ms: Optional[int] = Field(None, description="Execution timeout in ms")
    retry_count: int = Field(default=0, description="Number of retries on error")


class ExecutionPlan(BaseModel):
    """Task execution plan"""
    strategy: ExecutionStrategy = Field(..., description="Execution strategy")
    steps: Union[List[ExecutionStep], Dict[str, ExecutionStep]] = Field(
        ..., 
        description="Execution steps (list for solo/parallel, dict for DAG)"
    )
    merge_policy: str = Field(default="first_good", description="Result merging policy")
    max_execution_time_ms: Optional[int] = Field(None, description="Maximum execution time")
    requires_approval: bool = Field(default=False, description="Whether plan requires approval")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Planning metadata")
    
    class Config:
        use_enum_values = True
