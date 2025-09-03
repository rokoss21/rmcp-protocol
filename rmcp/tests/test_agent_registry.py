"""
Tests for Agent Registry functionality
"""

import pytest
import tempfile
import os
from pathlib import Path

from rmcp.agents.registry import AgentRegistry
from rmcp.storage.database import DatabaseManager
from rmcp.storage.schema import init_database
from rmcp.models.tool import Agent


class TestAgentRegistry:
    """Test Agent Registry functionality"""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing"""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        # Initialize database
        init_database(db_path)
        
        yield db_path
        
        # Cleanup
        os.unlink(db_path)
    
    @pytest.fixture
    def db_manager(self, temp_db):
        """Create database manager"""
        return DatabaseManager(temp_db)
    
    @pytest.fixture
    def agent_config_file(self):
        """Create temporary agent config file"""
        config_content = """
agent_registry:
  - id: "test-agent-1"
    name: "Test Agent 1"
    description: "Test agent for unit testing"
    endpoint: "http://test-agent-1:8000/execute"
    specialization: "testing"
    abstraction_level: "high"
    max_complexity: 0.8
    capabilities:
      - "test_execution"
      - "test_generation"
    tags:
      - "test"
      - "automation"
    input_schema:
      type: "object"
      properties:
        goal:
          type: "string"
      required: ["goal"]
    output_schema:
      type: "object"
      properties:
        status:
          type: "string"
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            config_path = f.name
        
        yield config_path
        
        # Cleanup
        os.unlink(config_path)
    
    def test_agent_registry_initialization(self, db_manager, agent_config_file):
        """Test agent registry initialization"""
        registry = AgentRegistry(db_manager, agent_config_file)
        
        assert len(registry.agents) == 1
        assert "test-agent-1" in registry.agents
        
        agent = registry.agents["test-agent-1"]
        assert agent.name == "Test Agent 1"
        assert agent.specialization == "testing"
        assert agent.abstraction_level == "high"
        assert agent.max_complexity == 0.8
    
    def test_agent_registration_in_database(self, db_manager, agent_config_file):
        """Test registering agents in database"""
        registry = AgentRegistry(db_manager, agent_config_file)
        registry.register_all_agents()
        
        # Check that agent was registered as a tool
        agents = db_manager.get_agents()
        assert len(agents) == 1
        
        agent_tool = agents[0]
        assert agent_tool.name == "Test Agent 1"
        assert agent_tool.tool_type == "agent"
        assert agent_tool.server_id.startswith("agent-server-")
        
        # Check that server was created
        server = db_manager.get_server(agent_tool.server_id)
        assert server is not None
        assert server.base_url == "http://test-agent-1:8000/execute"
    
    def test_get_agent_by_id(self, db_manager, agent_config_file):
        """Test getting agent by ID"""
        registry = AgentRegistry(db_manager, agent_config_file)
        
        agent = registry.get_agent("test-agent-1")
        assert agent is not None
        assert agent.name == "Test Agent 1"
        
        # Test non-existent agent
        agent = registry.get_agent("non-existent")
        assert agent is None
    
    def test_get_agents_by_specialization(self, db_manager, agent_config_file):
        """Test getting agents by specialization"""
        registry = AgentRegistry(db_manager, agent_config_file)
        
        testing_agents = registry.get_agents_by_specialization("testing")
        assert len(testing_agents) == 1
        assert testing_agents[0].name == "Test Agent 1"
        
        # Test non-existent specialization
        security_agents = registry.get_agents_by_specialization("security")
        assert len(security_agents) == 0
    
    def test_get_agents_by_abstraction_level(self, db_manager, agent_config_file):
        """Test getting agents by abstraction level"""
        registry = AgentRegistry(db_manager, agent_config_file)
        
        high_level_agents = registry.get_agents_by_abstraction_level("high")
        assert len(high_level_agents) == 1
        assert high_level_agents[0].name == "Test Agent 1"
        
        # Test non-existent level
        low_level_agents = registry.get_agents_by_abstraction_level("low")
        assert len(low_level_agents) == 0
    
    def test_get_agents_for_complexity(self, db_manager, agent_config_file):
        """Test getting agents for complexity level"""
        registry = AgentRegistry(db_manager, agent_config_file)
        
        # Test complexity within range
        agents = registry.get_agents_for_complexity(0.5)
        assert len(agents) == 1
        assert agents[0].name == "Test Agent 1"
        
        # Test complexity too high
        agents = registry.get_agents_for_complexity(0.9)
        assert len(agents) == 0
    
    def test_search_agents(self, db_manager, agent_config_file):
        """Test searching agents"""
        registry = AgentRegistry(db_manager, agent_config_file)
        
        # Search by name
        agents = registry.search_agents("Test Agent")
        assert len(agents) == 1
        assert agents[0].name == "Test Agent 1"
        
        # Search by capability
        agents = registry.search_agents("test_execution")
        assert len(agents) == 1
        assert agents[0].name == "Test Agent 1"
        
        # Search by tag
        agents = registry.search_agents("automation")
        assert len(agents) == 1
        assert agents[0].name == "Test Agent 1"
        
        # Search non-existent
        agents = registry.search_agents("non-existent")
        assert len(agents) == 0
    
    def test_agent_statistics(self, db_manager, agent_config_file):
        """Test agent statistics"""
        registry = AgentRegistry(db_manager, agent_config_file)
        
        stats = registry.get_agent_statistics()
        
        assert stats["total_agents"] == 1
        assert stats["by_specialization"]["testing"] == 1
        assert stats["by_abstraction_level"]["high"] == 1
        assert stats["avg_success_rate"] == 0.95  # Default value
        assert stats["avg_execution_time"] == 30000  # Default value
    
    def test_empty_registry(self, db_manager):
        """Test empty registry"""
        # Create empty config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("agent_registry: []")
            config_path = f.name
        
        try:
            registry = AgentRegistry(db_manager, config_path)
            
            assert len(registry.agents) == 0
            assert registry.get_agent_statistics()["total_agents"] == 0
        finally:
            os.unlink(config_path)
    
    def test_invalid_config_file(self, db_manager):
        """Test handling of invalid config file"""
        # Create invalid config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [")
            config_path = f.name
        
        try:
            registry = AgentRegistry(db_manager, config_path)
            
            # Should handle error gracefully
            assert len(registry.agents) == 0
        finally:
            os.unlink(config_path)
    
    def test_missing_config_file(self, db_manager):
        """Test handling of missing config file"""
        registry = AgentRegistry(db_manager, "/non/existent/path.yaml")
        
        # Should handle missing file gracefully
        assert len(registry.agents) == 0
    
    def test_reload_config(self, db_manager, agent_config_file):
        """Test reloading configuration"""
        registry = AgentRegistry(db_manager, agent_config_file)
        
        assert len(registry.agents) == 1
        
        # Modify config file
        with open(agent_config_file, 'w') as f:
            f.write("agent_registry: []")
        
        registry.reload_config()
        
        assert len(registry.agents) == 0
    
    def test_list_agents(self, db_manager, agent_config_file):
        """Test listing all agents"""
        registry = AgentRegistry(db_manager, agent_config_file)
        
        agents = registry.list_agents()
        assert len(agents) == 1
        assert agents[0].name == "Test Agent 1"

