"""
Tests for circuit breaker integration
"""

import pytest
import tempfile
import os
from datetime import datetime
from unittest.mock import AsyncMock, patch, Mock

from rmcp.storage.database import DatabaseManager
from rmcp.storage.schema import init_database
from rmcp.security import CircuitBreakerManager, AuditLogger
from rmcp.core.circuit_breaker_executor import CircuitBreakerExecutor
from rmcp.models.plan import ExecutionPlan, ExecutionStrategy, ExecutionStep
from rmcp.security.circuit_breaker import CircuitBreakerConfig, CircuitState


@pytest.fixture
def test_db_path():
    """Fixture for a temporary database file path."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name
    init_database(db_path)
    yield db_path
    os.unlink(db_path)


@pytest.fixture
def db_manager(test_db_path):
    """Fixture for database manager."""
    return DatabaseManager(test_db_path)


@pytest.fixture
def circuit_breaker_manager(db_manager):
    """Fixture for circuit breaker manager."""
    return CircuitBreakerManager(db_manager)


@pytest.fixture
def audit_logger(db_manager):
    """Fixture for audit logger."""
    return AuditLogger(db_manager)


@pytest.fixture
def circuit_breaker_executor(db_manager, circuit_breaker_manager, audit_logger):
    """Fixture for circuit breaker executor."""
    return CircuitBreakerExecutor(db_manager, circuit_breaker_manager, audit_logger)


@pytest.fixture
def test_tool():
    """Fixture for a test tool."""
    from rmcp.models.tool import Tool
    return Tool(
        id="test-tool",
        server_id="test-server",
        name="test.tool",
        description="Test tool",
        input_schema={"param": {"type": "string"}},
        output_schema={"result": {"type": "string"}},
        tags=["test"],
        capabilities=["test"],
        p95_latency_ms=1000,
        success_rate=0.95,
        cost_hint=0.01
    )


@pytest.fixture
def test_server():
    """Fixture for a test server."""
    from rmcp.models.tool import Server
    return Server(
        id="test-server",
        base_url="http://localhost:8000",
        description="Test MCP server"
    )


class TestCircuitBreakerIntegration:
    """Test circuit breaker integration"""
    
    def test_circuit_breaker_creation(self, circuit_breaker_manager):
        """Test circuit breaker creation"""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=30,
            success_threshold=2
        )
        
        circuit_breaker = circuit_breaker_manager.get_circuit_breaker("test-server", config)
        assert circuit_breaker is not None
        assert circuit_breaker.server_id == "test-server"
        assert circuit_breaker.config.failure_threshold == 3
    
    def test_circuit_breaker_normal_operation(self, circuit_breaker_manager):
        """Test circuit breaker in normal operation"""
        circuit_breaker = circuit_breaker_manager.get_circuit_breaker("test-server")
        
        # Initially closed
        assert circuit_breaker.is_available() is True
        
        # Record some successes
        for _ in range(5):
            circuit_breaker.record_success()
        
        # Should still be available
        assert circuit_breaker.is_available() is True
        assert circuit_breaker.stats.success_count == 5
    
    def test_circuit_breaker_failure_threshold(self, circuit_breaker_manager):
        """Test circuit breaker opening after failure threshold"""
        config = CircuitBreakerConfig(failure_threshold=3, recovery_timeout=1)
        circuit_breaker = circuit_breaker_manager.get_circuit_breaker("test-server", config)
        
        # Record failures up to threshold
        for _ in range(3):
            circuit_breaker.record_failure()
        
        # Circuit should be open
        assert circuit_breaker.is_available() is False
        assert circuit_breaker.stats.state == CircuitState.OPEN
    
    def test_circuit_breaker_recovery(self, circuit_breaker_manager):
        """Test circuit breaker recovery"""
        config = CircuitBreakerConfig(failure_threshold=2, recovery_timeout=1, success_threshold=2)
        circuit_breaker = circuit_breaker_manager.get_circuit_breaker("test-server", config)
        
        # Open the circuit
        for _ in range(2):
            circuit_breaker.record_failure()
        assert circuit_breaker.is_available() is False
        
        # Wait for recovery timeout (simulate)
        import time
        time.sleep(1.1)
        
        # Check availability to trigger state transition
        circuit_breaker.is_available()
        
        # Should be in half-open state
        assert circuit_breaker.stats.state == CircuitState.HALF_OPEN
        
        # Record successes to close circuit
        for _ in range(2):
            circuit_breaker.record_success()
        
        assert circuit_breaker.is_available() is True
        assert circuit_breaker.stats.state == CircuitState.CLOSED
    
    @pytest.mark.asyncio
    async def test_executor_with_working_server(self, circuit_breaker_executor, test_tool, test_server, db_manager):
        """Test executor with working server"""
        # Add test tool and server to database
        db_manager.add_server(test_server)
        db_manager.add_tool(test_tool)
        
        # Create execution plan
        plan = ExecutionPlan(
            strategy=ExecutionStrategy.SOLO,
            steps=[ExecutionStep(
                tool_id="test-tool",
                args={"param": "test"}
            )]
        )
        
        # Mock successful HTTP response
        with patch('httpx.AsyncClient.post') as mock_post:
            mock_response = Mock()
            mock_response.json.return_value = {"result": "success"}  # json() is sync
            mock_response.raise_for_status.return_value = None  # raise_for_status() is sync
            mock_post.return_value = mock_response
            
            result = await circuit_breaker_executor.execute(
                plan=plan,
                user_id="test-user",
                tenant_id="test-tenant"
            )
        
        # Should succeed
        assert result.status == "SUCCESS"
        assert result.data == {"result": "success"}
        
        # Circuit breaker should record success
        stats = circuit_breaker_executor.circuit_breaker_manager.get_stats("test-server")
        assert stats.success_count >= 1
    
    @pytest.mark.asyncio
    async def test_executor_with_failing_server(self, circuit_breaker_executor, test_tool, test_server, db_manager):
        """Test executor with failing server"""
        # Add test tool and server to database
        db_manager.add_server(test_server)
        db_manager.add_tool(test_tool)
        
        # Create execution plan
        plan = ExecutionPlan(
            strategy=ExecutionStrategy.SOLO,
            steps=[ExecutionStep(
                tool_id="test-tool",
                args={"param": "test"}
            )]
        )
        
        # Mock HTTP error
        with patch('httpx.AsyncClient.post') as mock_post:
            mock_post.side_effect = Exception("Connection failed")
            
            result = await circuit_breaker_executor.execute(
                plan=plan,
                user_id="test-user",
                tenant_id="test-tenant"
            )
        
        # Should fail
        assert result.status == "ERROR"
        assert "Connection failed" in result.summary
        
        # Circuit breaker should record failure
        stats = circuit_breaker_executor.circuit_breaker_manager.get_stats("test-server")
        assert stats.failure_count >= 1
    
    @pytest.mark.asyncio
    async def test_executor_circuit_open_fail_fast(self, circuit_breaker_executor, test_tool, test_server, db_manager):
        """Test executor fail-fast when circuit is open"""
        # Add test tool and server to database
        db_manager.add_server(test_server)
        db_manager.add_tool(test_tool)
        
        # Open the circuit by recording failures
        config = CircuitBreakerConfig(failure_threshold=2, recovery_timeout=60)
        circuit_breaker = circuit_breaker_executor.circuit_breaker_manager.get_circuit_breaker("test-server", config)
        
        for _ in range(2):
            circuit_breaker.record_failure()
        
        # Circuit should be open
        assert circuit_breaker.is_available() is False
        
        # Create execution plan
        plan = ExecutionPlan(
            strategy=ExecutionStrategy.SOLO,
            steps=[ExecutionStep(
                tool_id="test-tool",
                args={"param": "test"}
            )]
        )
        
        # Execute - should fail fast
        result = await circuit_breaker_executor.execute(
            plan=plan,
            user_id="test-user",
            tenant_id="test-tenant"
        )
        
        # Should fail fast with circuit open error
        assert result.status == "ERROR"
        assert "circuit breaker open" in result.summary
        assert result.metadata.get("error_type") == "CIRCUIT_OPEN"
    
    def test_server_health_status(self, circuit_breaker_executor, test_server, db_manager):
        """Test server health status reporting"""
        # Add test server to database
        db_manager.add_server(test_server)
        
        # Get health status
        health = circuit_breaker_executor.get_server_health("test-server")
        
        assert health["server_id"] == "test-server"
        assert health["status"] == "closed"  # Initially closed
        assert health["is_available"] is True
        assert health["failure_count"] == 0
        assert health["success_count"] == 0
    
    def test_circuit_breaker_stats(self, circuit_breaker_executor):
        """Test circuit breaker statistics"""
        stats = circuit_breaker_executor.get_circuit_breaker_stats()
        
        # Should be empty initially
        assert isinstance(stats, dict)
    
    @pytest.mark.asyncio
    async def test_parallel_execution_with_mixed_results(self, circuit_breaker_executor, test_tool, test_server, db_manager):
        """Test parallel execution with mixed success/failure results"""
        # Add test tool and server to database
        db_manager.add_server(test_server)
        db_manager.add_tool(test_tool)
        
        # Create parallel execution plan
        plan = ExecutionPlan(
            strategy=ExecutionStrategy.PARALLEL,
            steps=[
                ExecutionStep(tool_id="test-tool", args={"param": "test1"}),
                ExecutionStep(tool_id="test-tool", args={"param": "test2"})
            ]
        )
        
        # Mock mixed responses
        with patch('httpx.AsyncClient.post') as mock_post:
            # First call succeeds, second fails
            mock_response1 = Mock()
            mock_response1.json.return_value = {"result": "success1"}  # json() is sync
            mock_response1.raise_for_status.return_value = None  # raise_for_status() is sync
            
            mock_response2 = Mock()
            mock_response2.raise_for_status.side_effect = Exception("Server error")
            
            mock_post.side_effect = [mock_response1, mock_response2]
            
            result = await circuit_breaker_executor.execute(
                plan=plan,
                user_id="test-user",
                tenant_id="test-tenant"
            )
        
        # Should succeed (first successful result)
        assert result.status == "SUCCESS"
        assert result.data == {"result": "success1"}
        
        # Circuit breaker should have both success and failure recorded
        stats = circuit_breaker_executor.circuit_breaker_manager.get_stats("test-server")
        assert stats.success_count >= 1
        assert stats.failure_count >= 1
    
    def test_circuit_breaker_configuration(self, circuit_breaker_manager):
        """Test circuit breaker configuration options"""
        # Test default configuration
        circuit_breaker = circuit_breaker_manager.get_circuit_breaker("test-server")
        assert circuit_breaker.config.failure_threshold == 5
        assert circuit_breaker.config.recovery_timeout == 60
        assert circuit_breaker.config.success_threshold == 3
        
        # Test custom configuration
        custom_config = CircuitBreakerConfig(
            failure_threshold=2,
            recovery_timeout=30,
            success_threshold=1
        )
        circuit_breaker = circuit_breaker_manager.get_circuit_breaker("test-server-2", custom_config)
        assert circuit_breaker.config.failure_threshold == 2
        assert circuit_breaker.config.recovery_timeout == 30
        assert circuit_breaker.config.success_threshold == 1
    
    @pytest.mark.asyncio
    async def test_audit_logging_integration(self, circuit_breaker_executor, test_tool, test_server, db_manager):
        """Test audit logging integration"""
        # Add test tool and server to database
        db_manager.add_server(test_server)
        db_manager.add_tool(test_tool)
        
        # Open the circuit
        config = CircuitBreakerConfig(failure_threshold=1, recovery_timeout=60)
        circuit_breaker = circuit_breaker_executor.circuit_breaker_manager.get_circuit_breaker("test-server", config)
        circuit_breaker.record_failure()
        
        # Create execution plan
        plan = ExecutionPlan(
            strategy=ExecutionStrategy.SOLO,
            steps=[ExecutionStep(tool_id="test-tool", args={"param": "test"})]
        )
        
        # Execute - should trigger audit logging
        result = await circuit_breaker_executor.execute(
            plan=plan,
            user_id="test-user",
            tenant_id="test-tenant"
        )
        
        # Should fail with circuit open
        assert result.status == "ERROR"
        assert "circuit breaker open" in result.summary
        
        # Check that audit event was logged
        # Note: In a real test, we would verify the audit log entries
        # For now, we just ensure no exceptions were thrown during logging
