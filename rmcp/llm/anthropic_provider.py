"""
Anthropic provider implementation for RMCP
"""

import json
import asyncio
from typing import Dict, Any, Optional, List
import anthropic

from .providers import LLMProvider, LLMResponse
from .roles import LLMRole, RoleConfig


class AnthropicProvider(LLMProvider):
    """Anthropic provider implementation"""
    
    def __init__(self, api_key: str, model: str = "claude-3-sonnet-20240229", max_tokens: int = 1000):
        super().__init__(api_key, model, max_tokens)
        self.client = anthropic.AsyncAnthropic(api_key=api_key)
    
    async def generate_text(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> LLMResponse:
        """Generate text completion using Anthropic API"""
        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens or self.max_tokens,
                temperature=temperature,
                system=system_prompt or "",
                messages=[{"role": "user", "content": prompt}]
            )
            
            return LLMResponse(
                content=response.content[0].text,
                model=response.model,
                usage={
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens
                },
                metadata={"stop_reason": response.stop_reason}
            )
            
        except Exception as e:
            raise Exception(f"Anthropic API error: {str(e)}")
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding - Anthropic doesn't support embeddings directly"""
        # For now, return a mock embedding
        # In production, you might want to use a separate embedding service
        raise NotImplementedError("Anthropic doesn't support embeddings directly. Use a separate embedding service.")
    
    async def analyze_tool(
        self, 
        name: str, 
        description: str, 
        input_schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze tool and extract tags and capabilities"""
        config = RoleConfig.get_config(LLMRole.INGESTOR)
        prompt = RoleConfig.get_ingestor_prompt()
        
        user_prompt = f"""Tool Name: {name}
Tool Description: {description}
Input Schema: {json.dumps(input_schema, indent=2)}

Please analyze this tool and return the JSON with tags and capabilities."""
        
        response = await self.generate_text(
            prompt=user_prompt,
            system_prompt=prompt,
            temperature=config["temperature"],
            max_tokens=config["max_tokens"]
        )
        
        try:
            # Try to extract JSON from response
            content = response.content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            
            result = json.loads(content)
            return result
            
        except json.JSONDecodeError:
            # Fallback to simple parsing if JSON parsing fails
            return {
                "tags": ["unknown"],
                "capabilities": []
            }
    
    async def plan_execution(
        self,
        goal: str,
        context: Dict[str, Any],
        candidates: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Create execution plan using Anthropic"""
        config = RoleConfig.get_config(LLMRole.PLANNER_JUDGE)
        prompt = RoleConfig.get_planner_judge_prompt()
        
        user_prompt = f"""Goal: {goal}
Context: {json.dumps(context, indent=2)}
Available Tools: {json.dumps(candidates, indent=2)}

Please create an optimal execution plan."""
        
        response = await self.generate_text(
            prompt=user_prompt,
            system_prompt=prompt,
            temperature=config["temperature"],
            max_tokens=config["max_tokens"]
        )
        
        try:
            # Try to extract JSON from response
            content = response.content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            
            result = json.loads(content)
            return result
            
        except json.JSONDecodeError:
            # Fallback to simple plan
            return {
                "strategy": "solo",
                "steps": [{"tool_id": candidates[0]["id"] if candidates else "unknown"}],
                "requires_approval": False
            }
    
    async def merge_results(
        self,
        results: List[Dict[str, Any]],
        goal: str
    ) -> Dict[str, Any]:
        """Merge and summarize execution results"""
        config = RoleConfig.get_config(LLMRole.RESULT_MERGER)
        prompt = RoleConfig.get_result_merger_prompt()
        
        user_prompt = f"""Original Goal: {goal}
Execution Results: {json.dumps(results, indent=2)}

Please merge and summarize these results."""
        
        response = await self.generate_text(
            prompt=user_prompt,
            system_prompt=prompt,
            temperature=config["temperature"],
            max_tokens=config["max_tokens"]
        )
        
        try:
            # Try to extract JSON from response
            content = response.content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            
            result = json.loads(content)
            return result
            
        except json.JSONDecodeError:
            # Fallback to simple summary
            return {
                "summary": f"Executed {len(results)} tasks for goal: {goal}",
                "data": results,
                "confidence": 0.5
            }
    
    @property
    def provider_name(self) -> str:
        return "anthropic"
    
    @property
    def supports_embeddings(self) -> bool:
        return False  # Anthropic doesn't support embeddings directly
    
    async def close(self):
        """Close the Anthropic client"""
        # Anthropic client doesn't need explicit closing
        pass

