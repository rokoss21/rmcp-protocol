"""
Agent Registry for RMCP Meta-Orchestration
"""

import yaml
import os
from typing import List, Dict, Any, Optional
from pathlib import Path

from ..models.tool import Agent
from ..storage.database import DatabaseManager
from ..logging.config import get_logger


class AgentRegistry:
    """Registry for managing AI agents in RMCP"""
    
    def __init__(self, db_manager: DatabaseManager, config_path: Optional[str] = None):
        self.db_manager = db_manager
        self.logger = get_logger(__name__)
        
        # Default config path
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "agents.yaml"
        
        self.config_path = config_path
        self.agents: Dict[str, Agent] = {}
        
        # Load agents from config
        self._load_agents_from_config()
    
    def _load_agents_from_config(self) -> None:
        """Load agents from YAML configuration file"""
        try:
            if not os.path.exists(self.config_path):
                self.logger.warning(f"Agent config file not found: {self.config_path}")
                return
            
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            agent_configs = config.get('agent_registry', [])
            
            for agent_config in agent_configs:
                try:
                    agent = Agent(**agent_config)
                    self.agents[agent.id] = agent
                    self.logger.info(f"Loaded agent: {agent.name} ({agent.id})")
                except Exception as e:
                    self.logger.error(f"Failed to load agent {agent_config.get('id', 'unknown')}: {e}")
            
            self.logger.info(f"Loaded {len(self.agents)} agents from config")
            
        except Exception as e:
            self.logger.error(f"Failed to load agent config: {e}")
    
    def register_agent_in_database(self, agent: Agent) -> None:
        """Register an agent in the database as a tool with tool_type='agent'"""
        try:
            # Create a server entry for the agent
            server_id = f"agent-server-{agent.id}"
            
            # Check if server already exists
            existing_server = self.db_manager.get_server_by_url(agent.endpoint)
            if not existing_server:
                from ..models.tool import Server
                server = Server(
                    id=server_id,
                    base_url=agent.endpoint,
                    description=f"Agent server for {agent.name}"
                )
                self.db_manager.add_server(server)
                self.logger.info(f"Created server entry for agent: {agent.name}")
            else:
                server_id = existing_server.id
                self.logger.info(f"Using existing server entry for agent: {agent.name}")
            
            # Create tool entry for the agent
            tool_id = f"agent-{agent.id}"
            
            # Check if tool already exists
            existing_tool = self.db_manager.get_tool_by_id(tool_id)
            if existing_tool:
                # Update existing tool
                self.db_manager.update_tool(
                    tool_id=tool_id,
                    name=agent.name,
                    description=agent.description,
                    input_schema=agent.input_schema,
                    output_schema=agent.output_schema,
                    tags=agent.tags,
                    capabilities=agent.capabilities,
                    tool_type="agent"
                )
                self.logger.info(f"Updated existing agent tool: {agent.name}")
            else:
                # Create new tool
                from ..models.tool import Tool
                tool = Tool(
                    id=tool_id,
                    server_id=server_id,
                    name=agent.name,
                    description=agent.description,
                    input_schema=agent.input_schema,
                    output_schema=agent.output_schema,
                    tags=agent.tags,
                    capabilities=agent.capabilities,
                    tool_type="agent"
                )
                self.db_manager.add_tool(tool)
                self.logger.info(f"Registered new agent tool: {agent.name}")
            
        except Exception as e:
            self.logger.error(f"Failed to register agent {agent.name} in database: {e}")
    
    def register_all_agents(self) -> None:
        """Register all loaded agents in the database"""
        for agent in self.agents.values():
            self.register_agent_in_database(agent)
        
        self.logger.info(f"Registered {len(self.agents)} agents in database")
    
    def get_agent(self, agent_id: str) -> Optional[Agent]:
        """Get an agent by ID"""
        return self.agents.get(agent_id)
    
    def get_agents_by_specialization(self, specialization: str) -> List[Agent]:
        """Get all agents with a specific specialization"""
        return [
            agent for agent in self.agents.values()
            if agent.specialization == specialization
        ]
    
    def get_agents_by_abstraction_level(self, level: str) -> List[Agent]:
        """Get all agents with a specific abstraction level"""
        return [
            agent for agent in self.agents.values()
            if agent.abstraction_level == level
        ]
    
    def get_agents_for_complexity(self, complexity: float) -> List[Agent]:
        """Get all agents that can handle a given complexity level"""
        return [
            agent for agent in self.agents.values()
            if agent.max_complexity >= complexity
        ]
    
    def search_agents(self, query: str) -> List[Agent]:
        """Search agents by name, description, or capabilities"""
        query_lower = query.lower()
        matching_agents = []
        
        for agent in self.agents.values():
            # Search in name
            if query_lower in agent.name.lower():
                matching_agents.append(agent)
                continue
            
            # Search in description
            if query_lower in agent.description.lower():
                matching_agents.append(agent)
                continue
            
            # Search in capabilities
            for capability in agent.capabilities:
                if query_lower in capability.lower():
                    matching_agents.append(agent)
                    break
            
            # Search in tags
            for tag in agent.tags:
                if query_lower in tag.lower():
                    matching_agents.append(agent)
                    break
        
        return matching_agents
    
    def get_agent_statistics(self) -> Dict[str, Any]:
        """Get statistics about registered agents"""
        if not self.agents:
            return {
                "total_agents": 0,
                "by_specialization": {},
                "by_abstraction_level": {},
                "avg_success_rate": 0.0,
                "avg_execution_time": 0
            }
        
        by_specialization = {}
        by_abstraction_level = {}
        total_success_rate = 0.0
        total_execution_time = 0
        
        for agent in self.agents.values():
            # Count by specialization
            spec = agent.specialization
            by_specialization[spec] = by_specialization.get(spec, 0) + 1
            
            # Count by abstraction level
            level = agent.abstraction_level
            by_abstraction_level[level] = by_abstraction_level.get(level, 0) + 1
            
            # Sum metrics
            total_success_rate += agent.success_rate
            total_execution_time += agent.avg_execution_time_ms
        
        return {
            "total_agents": len(self.agents),
            "by_specialization": by_specialization,
            "by_abstraction_level": by_abstraction_level,
            "avg_success_rate": total_success_rate / len(self.agents),
            "avg_execution_time": total_execution_time // len(self.agents)
        }
    
    def reload_config(self) -> None:
        """Reload agents from configuration file"""
        self.agents.clear()
        self._load_agents_from_config()
        self.logger.info("Reloaded agent configuration")
    
    def list_agents(self) -> List[Agent]:
        """Get list of all registered agents"""
        return list(self.agents.values())
