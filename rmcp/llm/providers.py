"""
Base classes and interfaces for LLM providers
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from pydantic import BaseModel
from enum import Enum


class LLMRole(str, Enum):
    """LLM roles in RMCP system"""
    INGESTOR = "ingestor"           # Tool analysis and tagging
    PLANNER_JUDGE = "planner_judge" # Complex planning decisions
    RESULT_MERGER = "result_merger" # Result aggregation and summarization


class LLMResponse(BaseModel):
    """Standardized response from LLM provider"""
    content: str
    model: str
    usage: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    def __init__(self, api_key: str, model: str, max_tokens: int = 1000):
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
    
    @abstractmethod
    async def generate_text(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> LLMResponse:
        """Generate text completion"""
        pass
    
    @abstractmethod
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text"""
        pass
    
    @abstractmethod
    async def analyze_tool(
        self, 
        name: str, 
        description: str, 
        input_schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze tool and extract tags and capabilities"""
        pass
    
    @abstractmethod
    async def plan_execution(
        self,
        goal: str,
        context: Dict[str, Any],
        candidates: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Create execution plan using LLM"""
        pass
    
    @abstractmethod
    async def merge_results(
        self,
        results: List[Dict[str, Any]],
        goal: str
    ) -> Dict[str, Any]:
        """Merge and summarize execution results"""
        pass
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Name of the provider"""
        pass
    
    @property
    @abstractmethod
    def supports_embeddings(self) -> bool:
        """Whether provider supports embeddings"""
        pass


class MockLLMProvider(LLMProvider):
    """Mock LLM provider for testing"""
    
    def __init__(self, api_key: str = "mock", model: str = "mock-model", max_tokens: int = 1000):
        super().__init__(api_key, model, max_tokens)
    
    async def generate_text(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> LLMResponse:
        """Mock text generation"""
        return LLMResponse(
            content="Mock response",
            model=self.model,
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        )
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Mock embedding generation"""
        # Return a simple mock embedding
        return [0.1] * 384  # Common embedding dimension
    
    async def analyze_tool(
        self, 
        name: str, 
        description: str, 
        input_schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Mock tool analysis"""
        return {
            "tags": ["mock", "test"],
            "capabilities": ["mock:read"]
        }
    
    async def plan_execution(
        self,
        goal: str,
        context: Dict[str, Any],
        candidates: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Mock execution planning"""
        return {
            "strategy": "solo",
            "steps": [{"tool_id": candidates[0]["id"] if candidates else "mock_tool"}],
            "requires_approval": False
        }
    
    async def merge_results(
        self,
        results: List[Dict[str, Any]],
        goal: str
    ) -> Dict[str, Any]:
        """Mock result merging"""
        return {
            "summary": "Mock summary",
            "data": results[0] if results else {},
            "confidence": 0.8
        }
    
    @property
    def provider_name(self) -> str:
        return "mock"
    
    @property
    def supports_embeddings(self) -> bool:
        return True

