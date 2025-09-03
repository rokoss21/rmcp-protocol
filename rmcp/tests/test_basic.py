"""
Basic tests for RMCP MVP
"""

import pytest
import tempfile
import os
from rmcp.storage.database import DatabaseManager
from rmcp.storage.schema import init_database
from rmcp.models.tool import Tool, Server
from rmcp.models.request import RouteRequest, ExecuteResponse


class TestDatabase:
    """Test database functionality"""
    
    def setup_method(self):
        """Setup test database"""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False)
        self.temp_db.close()
        self.db_path = self.temp_db.name
        init_database(self.db_path)
        self.db_manager = DatabaseManager(self.db_path)
    
    def teardown_method(self):
        """Cleanup test database"""
        os.unlink(self.db_path)
    
    def test_add_server(self):
        """Test adding a server"""
        server = Server(
            id="test_server",
            base_url="http://localhost:3001",
            description="Test MCP Server"
        )
        
        self.db_manager.add_server(server)
        retrieved_server = self.db_manager.get_server("test_server")
        
        assert retrieved_server is not None
        assert retrieved_server.id == "test_server"
        assert retrieved_server.base_url == "http://localhost:3001"
    
    def test_add_tool(self):
        """Test adding a tool"""
        # First add a server
        server = Server(
            id="test_server",
            base_url="http://localhost:3001",
            description="Test MCP Server"
        )
        self.db_manager.add_server(server)
        
        # Then add a tool
        tool = Tool(
            id="test_tool",
            server_id="test_server",
            name="test_tool",
            description="Test tool for testing",
            tags=["test", "search"],
            capabilities=["filesystem:read"]
        )
        
        self.db_manager.add_tool(tool)
        retrieved_tool = self.db_manager.get_tool("test_tool")
        
        assert retrieved_tool is not None
        assert retrieved_tool.id == "test_tool"
        assert retrieved_tool.name == "test_tool"
        assert "test" in retrieved_tool.tags
        assert "filesystem:read" in retrieved_tool.capabilities


class TestModels:
    """Test Pydantic models"""
    
    def test_route_request(self):
        """Test RouteRequest model"""
        request = RouteRequest(
            tool_name="rmcp.route",
            parameters={
                "goal": "Find all Python files",
                "context": {"path": "/home/user"}
            }
        )
        
        assert request.tool_name == "rmcp.route"
        assert request.goal == "Find all Python files"
        assert request.context["path"] == "/home/user"
    
    def test_execute_response(self):
        """Test ExecuteResponse model"""
        response = ExecuteResponse(
            status="SUCCESS",
            summary="Task completed successfully",
            data={"files": ["file1.py", "file2.py"]},
            confidence=0.95
        )
        
        assert response.status == "SUCCESS"
        assert response.summary == "Task completed successfully"
        assert len(response.data["files"]) == 2
        assert response.confidence == 0.95


if __name__ == "__main__":
    pytest.main([__file__])

