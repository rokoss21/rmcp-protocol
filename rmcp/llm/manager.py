"""
LLM Manager for RMCP - manages multiple LLM providers and roles
"""

import asyncio
from typing import Dict, Any, Optional, List
from .providers import LLMProvider, LLMResponse, MockLLMProvider
from .openai_provider import OpenAIProvider
from .anthropic_provider import AnthropicProvider
from .roles import LLMRole, RoleConfig


class LLMManager:
    """Manages multiple LLM providers and assigns them to specific roles"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.providers: Dict[str, LLMProvider] = {}
        self.role_assignments: Dict[LLMRole, str] = {}
        self._initialize_providers()
        self._assign_roles()
    
    def _initialize_providers(self):
        """Initialize LLM providers from configuration"""
        llm_config = self.config.get("llm_providers", {})
        
        # Initialize OpenAI provider
        if "openai" in llm_config:
            openai_config = llm_config["openai"]
            if openai_config.get("api_key"):
                self.providers["openai"] = OpenAIProvider(
                    api_key=openai_config["api_key"],
                    model=openai_config.get("model", "gpt-3.5-turbo"),
                    max_tokens=openai_config.get("max_tokens", 1000)
                )
        
        # Initialize Anthropic provider
        if "anthropic" in llm_config:
            anthropic_config = llm_config["anthropic"]
            if anthropic_config.get("api_key"):
                self.providers["anthropic"] = AnthropicProvider(
                    api_key=anthropic_config["api_key"],
                    model=anthropic_config.get("model", "claude-3-sonnet-20240229"),
                    max_tokens=anthropic_config.get("max_tokens", 1000)
                )
        
        # Add mock provider for testing if no real providers are available
        if not self.providers:
            self.providers["mock"] = MockLLMProvider()
    
    def _assign_roles(self):
        """Assign providers to specific roles"""
        role_config = self.config.get("llm_roles", {})
        
        # Default role assignments
        default_assignments = {
            LLMRole.INGESTOR: "openai",
            LLMRole.PLANNER_JUDGE: "openai", 
            LLMRole.RESULT_MERGER: "openai"
        }
        
        # Override with configuration
        for role_name, provider_name in role_config.items():
            try:
                role = LLMRole(role_name)
                if provider_name in self.providers:
                    self.role_assignments[role] = provider_name
                else:
                    # Fallback to available provider
                    available_provider = list(self.providers.keys())[0]
                    self.role_assignments[role] = available_provider
            except ValueError:
                # Invalid role name, skip
                continue
        
        # Fill in any missing assignments
        for role, default_provider in default_assignments.items():
            if role not in self.role_assignments:
                if default_provider in self.providers:
                    self.role_assignments[role] = default_provider
                else:
                    # Use first available provider
                    available_provider = list(self.providers.keys())[0]
                    self.role_assignments[role] = available_provider
    
    def get_provider_for_role(self, role: LLMRole) -> LLMProvider:
        """Get the provider assigned to a specific role"""
        provider_name = self.role_assignments.get(role)
        if not provider_name or provider_name not in self.providers:
            # Fallback to first available provider
            provider_name = list(self.providers.keys())[0]
        
        return self.providers[provider_name]
    
    async def generate_text_for_role(
        self,
        role: LLMRole,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate text using the provider assigned to a specific role"""
        provider = self.get_provider_for_role(role)
        config = RoleConfig.get_config(role)
        
        # Merge role config with provided kwargs
        final_kwargs = {**config, **kwargs}
        
        return await provider.generate_text(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=final_kwargs.get("temperature", 0.7),
            max_tokens=final_kwargs.get("max_tokens", 1000)
        )
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding using the first available provider that supports embeddings"""
        for provider in self.providers.values():
            if provider.supports_embeddings:
                return await provider.generate_embedding(text)
        
        # Fallback to mock embedding if no provider supports embeddings
        return [0.1] * 384
    
    async def analyze_tool(
        self,
        name: str,
        description: str,
        input_schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze tool using the ingestor role provider"""
        provider = self.get_provider_for_role(LLMRole.INGESTOR)
        return await provider.analyze_tool(name, description, input_schema)
    
    async def plan_execution(
        self,
        goal: str,
        context: Dict[str, Any],
        candidates: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Create execution plan using the planner judge role provider"""
        provider = self.get_provider_for_role(LLMRole.PLANNER_JUDGE)
        return await provider.plan_execution(goal, context, candidates)
    
    async def merge_results(
        self,
        results: List[Dict[str, Any]],
        goal: str
    ) -> Dict[str, Any]:
        """Merge results using the result merger role provider"""
        provider = self.get_provider_for_role(LLMRole.RESULT_MERGER)
        return await provider.merge_results(results, goal)
    
    def get_available_providers(self) -> List[str]:
        """Get list of available provider names"""
        return list(self.providers.keys())
    
    def get_role_assignments(self) -> Dict[str, str]:
        """Get current role assignments"""
        return {role.value: provider for role, provider in self.role_assignments.items()}
    
    async def close(self):
        """Close all providers"""
        for provider in self.providers.values():
            if hasattr(provider, 'close'):
                await provider.close()
    
    def __del__(self):
        """Cleanup on deletion"""
        # Note: This won't work reliably in async context, but it's good practice
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self.close())
        except:
            pass

