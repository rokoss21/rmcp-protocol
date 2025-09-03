"""
Capability Ingestor - scanning MCP servers and creating tool catalog
"""

import httpx
import json
import uuid
import hashlib
from typing import Dict, List, Any, Optional
from datetime import datetime

from ..storage.database import DatabaseManager
from ..models.tool import Tool, Server
from ..llm.manager import LLMManager


class CapabilityIngestor:
    """Ingestor for scanning MCP servers and creating tool catalog"""
    
    def __init__(self, db_manager: DatabaseManager, config_manager, llm_manager: Optional[LLMManager] = None):
        self.db_manager = db_manager
        self.config_manager = config_manager
        self.client = httpx.AsyncClient(timeout=30.0)
        self.llm_manager = llm_manager
    
    async def ingest_all(self) -> Dict[str, Any]:
        """Scan all configured MCP servers"""
        # Get servers from configuration
        config = self.config_manager.get_config()
        servers = config.get("mcp_servers", [])
        
        total_servers = 0
        total_tools = 0
        
        for server_config in servers:
            try:
                server_id = await self._ingest_server(server_config)
                if server_id:
                    total_servers += 1
                    tools_count = await self._ingest_tools(server_id, server_config["base_url"])
                    total_tools += tools_count
            except Exception as e:
                print(f"Failed to ingest server {server_config['base_url']}: {e}")
        
        return {
            "servers_scanned": total_servers,
            "tools_discovered": total_tools
        }
    
    async def _ingest_server(self, server_config: Dict[str, str]) -> Optional[str]:
        """Scan single MCP server"""
        base_url = server_config["base_url"]
        description = server_config["description"]
        
        # Generate a stable server ID based on base_url to avoid mismatches across restarts
        # Use md5 hash of the base_url for deterministic ID while keeping it short
        url_hash = hashlib.md5(base_url.encode("utf-8")).hexdigest()[:8]
        server_id = f"server_{url_hash}"
        
        # Create server object
        server = Server(
            id=server_id,
            base_url=base_url,
            description=description,
            added_at=datetime.utcnow()
        )
        
        # Save to database
        self.db_manager.add_server(server)
        
        return server_id
    
    async def _ingest_tools(self, server_id: str, base_url: str) -> int:
        """Scan tools from single server"""
        try:
            # Request to /tools endpoint
            response = await self.client.get(f"{base_url}/tools")
            response.raise_for_status()
            
            tools_data = response.json()
            tools_count = 0
            
            # Process each tool
            for tool_data in tools_data.get("tools", []):
                try:
                    tool = await self._process_tool(server_id, tool_data)
                    if tool:
                        self.db_manager.add_tool(tool)
                        tools_count += 1
                except Exception as e:
                    print(f"Failed to process tool {tool_data.get('name', 'unknown')}: {e}")
            
            return tools_count
            
        except httpx.RequestError as e:
            print(f"Failed to connect to {base_url}: {e}")
            return 0
        except Exception as e:
            print(f"Failed to ingest tools from {base_url}: {e}")
            return 0
    
    async def _process_tool(self, server_id: str, tool_data: Dict[str, Any]) -> Optional[Tool]:
        """Process data for single tool"""
        try:
            # Extract basic information
            name = tool_data.get("name", "")
            description = tool_data.get("description", "")
            input_schema = tool_data.get("input_schema", tool_data.get("inputSchema", {}))
            output_schema = tool_data.get("output_schema", tool_data.get("outputSchema", {}))
            
            # Debug: print schema information (removed for production)
            
            if not name:
                return None
            
            # Generate tool ID
            tool_id = f"{server_id}_{name}"
            
            # Extract capabilities from MCP server if available
            mcp_capabilities = tool_data.get("capabilities", [])
            
            # Use LLM for analysis if available, otherwise fallback to simple extraction
            if self.llm_manager:
                try:
                    analysis = await self.llm_manager.analyze_tool(name, description, input_schema)
                    tags = analysis.get("tags", [])
                    llm_capabilities = analysis.get("capabilities", [])
                    # Combine MCP and LLM capabilities
                    capabilities = list(set(mcp_capabilities + llm_capabilities))
                except Exception as e:
                    print(f"LLM analysis failed for {name}, using fallback: {e}")
                    tags = self._extract_tags(name, description)
                    fallback_capabilities = self._extract_capabilities(name, description, input_schema)
                    capabilities = list(set(mcp_capabilities + fallback_capabilities))
            else:
                tags = self._extract_tags(name, description)
                fallback_capabilities = self._extract_capabilities(name, description, input_schema)
                capabilities = list(set(mcp_capabilities + fallback_capabilities))
            
            # Create tool object
            tool = Tool(
                id=tool_id,
                server_id=server_id,
                name=name,
                description=description,
                input_schema=input_schema,
                output_schema=output_schema,
                tags=tags,
                capabilities=capabilities,
                p95_latency_ms=3000,  # Default value
                success_rate=0.95,    # Default value
                cost_hint=0.0
            )
            
            return tool
            
        except Exception as e:
            print(f"Error processing tool: {e}")
            return None
    
    def _extract_tags(self, name: str, description: str) -> List[str]:
        """Extract tags from tool name and description (MVP version)"""
        tags = []
        text = f"{name} {description}".lower()
        
        # Simple tag extraction logic
        if any(word in text for word in ["search", "find", "lookup"]):
            tags.append("search")
        if any(word in text for word in ["git", "commit", "branch", "diff"]):
            tags.append("git")
        if any(word in text for word in ["file", "directory", "path"]):
            tags.append("filesystem")
        if any(word in text for word in ["execute", "run", "command"]):
            tags.append("execution")
        if any(word in text for word in ["read", "get", "fetch"]):
            tags.append("read")
        if any(word in text for word in ["write", "create", "update", "delete"]):
            tags.append("write")
        
        return tags
    
    def _extract_capabilities(self, name: str, description: str, input_schema: Dict[str, Any]) -> List[str]:
        """Extract tool capabilities (MVP version)"""
        capabilities = []
        text = f"{name} {description}".lower()
        
        # Simple capability extraction logic
        if any(word in text for word in ["file", "directory", "path"]):
            if any(word in text for word in ["read", "get", "fetch"]):
                capabilities.append("filesystem:read")
            if any(word in text for word in ["write", "create", "update", "delete"]):
                capabilities.append("filesystem:write")
        
        if any(word in text for word in ["execute", "run", "command", "shell"]):
            capabilities.append("execution")
        
        if any(word in text for word in ["dangerous", "destructive", "delete", "remove"]):
            capabilities.append("requires_human_approval")
        
        # Check schema for state mutation
        if self._schema_suggests_mutation(input_schema):
            capabilities.append("mutates_state")
        
        return capabilities
    
    def _schema_suggests_mutation(self, input_schema: Dict[str, Any]) -> bool:
        """Check schema for state mutation indicators"""
        # Simple heuristic for MVP
        schema_str = json.dumps(input_schema).lower()
        mutation_keywords = ["write", "create", "update", "delete", "remove", "modify"]
        return any(keyword in schema_str for keyword in mutation_keywords)
    
    async def close(self):
        """Close HTTP client"""
        await self.client.aclose()
