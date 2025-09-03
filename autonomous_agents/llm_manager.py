"""
LLM Manager for Autonomous Agents
"""

import os
import asyncio
from typing import Dict, Any, Optional, List
from openai import AsyncOpenAI
import json
import re


class LLMManager:
    """
    Centralized LLM Manager for all autonomous agents
    
    Responsibilities:
    - Manage OpenAI API connections
    - Provide role-specific prompts
    - Clean and extract code from LLM responses
    - Handle rate limiting and error recovery
    """
    
    def __init__(self, api_key: str = None, model: str = "gpt-4o-mini"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.client = None
        
        if self.api_key:
            self.client = AsyncOpenAI(api_key=self.api_key)
        
        # Role-specific system prompts
        self.role_prompts = {
            "architect": "You are an expert software architect specializing in MCP (Model Context Protocol) server development. You create detailed, actionable development plans.",
            "backend_developer": "You are an expert Python backend developer specializing in FastAPI and MCP server development. You write clean, production-ready code.",
            "test_developer": "You are an expert Python test developer specializing in pytest and FastAPI testing. You write comprehensive, high-quality tests.",
            "devops_engineer": "You are an expert DevOps engineer specializing in Docker and containerization. You create deployment configurations and manage infrastructure.",
            "debugger": "You are an expert software debugger. You analyze code, identify issues, and provide fixes.",
            "default": "You are an expert software developer. You provide high-quality, production-ready solutions."
        }
    
    async def generate_text_for_role(
        self, 
        role: str, 
        prompt: str, 
        max_tokens: int = 4000,
        temperature: float = 0.1
    ) -> Dict[str, Any]:
        """
        Generate text using OpenAI API for specific role
        
        Args:
            role: The role/context for the LLM
            prompt: The user prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Dict containing generated text and metadata
        """
        if not self.client:
            return {
                "success": False,
                "error": "OpenAI client not initialized. Check API key.",
                "content": ""
            }
        
        try:
            system_prompt = self.role_prompts.get(role, self.role_prompts["default"])
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            content = response.choices[0].message.content
            
            return {
                "success": True,
                "content": content,
                "role": role,
                "model": self.model,
                "tokens_used": response.usage.total_tokens if response.usage else 0
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "content": ""
            }
    
    def extract_code_from_response(self, response: str, language: str = "python") -> str:
        """
        Extract code from LLM response
        
        Args:
            response: LLM response text
            language: Programming language to extract
            
        Returns:
            Extracted code
        """
        # Pattern to match code blocks
        patterns = [
            rf"```{language}\s*\n(.*?)\n```",
            rf"```\s*\n(.*?)\n```",
            rf"```{language}(.*?)```",
            rf"```(.*?)```"
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response, re.DOTALL)
            if matches:
                # Return the longest match (usually the main code)
                return max(matches, key=len).strip()
        
        # If no code blocks found, return the whole response
        return response.strip()
    
    def extract_json_from_response(self, response: str) -> Optional[Dict[str, Any]]:
        """
        Extract JSON from LLM response
        
        Args:
            response: LLM response text
            
        Returns:
            Extracted JSON or None
        """
        # Pattern to match JSON blocks
        json_pattern = r"```json\s*\n(.*?)\n```"
        matches = re.findall(json_pattern, response, re.DOTALL)
        
        if matches:
            try:
                return json.loads(matches[0])
            except json.JSONDecodeError:
                pass
        
        # Try to find JSON in the response
        try:
            # Look for JSON-like structures
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
        
        return None
    
    def clean_code_response(self, response: str, language: str = "python") -> str:
        """
        Clean and format code response
        
        Args:
            response: Raw LLM response
            language: Programming language
            
        Returns:
            Cleaned code
        """
        # Extract code
        code = self.extract_code_from_response(response, language)
        
        # Remove common LLM artifacts
        code = re.sub(r'^# Generated.*?\n', '', code, flags=re.MULTILINE)
        code = re.sub(r'^# Created by.*?\n', '', code, flags=re.MULTILINE)
        code = re.sub(r'^# Auto-generated.*?\n', '', code, flags=re.MULTILINE)
        
        # Ensure proper formatting
        code = code.strip()
        
        return code
    
    async def generate_code_for_role(
        self, 
        role: str, 
        prompt: str, 
        language: str = "python"
    ) -> Dict[str, Any]:
        """
        Generate code for specific role and extract clean code
        
        Args:
            role: The role/context for the LLM
            prompt: The user prompt
            language: Programming language
            
        Returns:
            Dict containing generated code and metadata
        """
        response = await self.generate_text_for_role(role, prompt)
        
        if not response["success"]:
            return response
        
        # Extract and clean code
        clean_code = self.clean_code_response(response["content"], language)
        
        return {
            "success": True,
            "content": clean_code,
            "raw_content": response["content"],
            "role": role,
            "language": language,
            "tokens_used": response.get("tokens_used", 0)
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get LLM Manager statistics"""
        return {
            "api_key_configured": bool(self.api_key),
            "model": self.model,
            "client_initialized": bool(self.client),
            "available_roles": list(self.role_prompts.keys())
        }


# Global LLM Manager instance
llm_manager = LLMManager()

