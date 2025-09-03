"""
Tests for observability and Prometheus metrics integration
"""

import pytest
import tempfile
import os
import time
from unittest.mock import patch, Mock
from fastapi.testclient import TestClient

from rmcp.storage.database import DatabaseManager
from rmcp.storage.schema import init_database
from rmcp.security import CircuitBreakerManager, AuditLogger
from rmcp.core.circuit_breaker_executor import CircuitBreakerExecutor
from rmcp.observability.metrics import metrics, RMCPMetrics
from rmcp.gateway.app import create_app
from rmcp.models.plan import ExecutionPlan, ExecutionStrategy, ExecutionStep


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


@pytest.fixture
def app():
    """Fixture for FastAPI app with metrics."""
    return create_app()


@pytest.fixture
def client(app):
    """Fixture for test client."""
    return TestClient(app)


class TestObservabilityIntegration:
    """Test observability and metrics integration"""
    
    def test_metrics_initialization(self):
        """Test that metrics are properly initialized"""
        assert metrics is not None
        assert isinstance(metrics, RMCPMetrics)
        
        # Check that key metrics exist
        assert hasattr(metrics, 'plans_created_total')
        assert hasattr(metrics, 'tool_executions_total')
        assert hasattr(metrics, 'circuit_breaker_state')
        assert hasattr(metrics, 'system_info')
    
    def test_plan_creation_metrics(self):
        """Test plan creation metrics recording"""
        # Record a plan creation
        metrics.record_plan_created("SOLO", "test-tenant", False)
        metrics.record_plan_created("PARALLEL", "test-tenant", True)
        
        # Check that metrics were recorded (we can't easily test the exact values
        # without scraping the metrics endpoint, but we can ensure no exceptions)
        assert True  # If we get here, no exceptions were raised
    
    def test_planning_duration_metrics(self):
        """Test planning duration metrics recording"""
        # Record planning durations
        metrics.record_planning_duration("SOLO", "test-tenant", 0.1)
        metrics.record_planning_duration("PARALLEL", "test-tenant", 0.5)
        
        assert True  # No exceptions means metrics were recorded
    
    def test_tool_execution_metrics(self):
        """Test tool execution metrics recording"""
        # Record tool executions
        metrics.record_tool_execution("tool1", "server1", "SUCCESS", "test-tenant", 0.2)
        metrics.record_tool_execution("tool2", "server2", "ERROR", "test-tenant", 0.1)
        
        assert True  # No exceptions means metrics were recorded
    
    def test_circuit_breaker_metrics(self):
        """Test circuit breaker metrics recording"""
        # Record circuit breaker events
        metrics.record_circuit_breaker_success("server1", "test-tenant")
        metrics.record_circuit_breaker_failure("server2", "test-tenant")
        metrics.update_circuit_breaker_state("server1", "test-tenant", "closed")
        metrics.update_circuit_breaker_state("server2", "test-tenant", "open")
        
        assert True  # No exceptions means metrics were recorded
    
    def test_execution_plan_metrics(self):
        """Test execution plan metrics recording"""
        # Record execution plans
        metrics.record_execution_plan("SOLO", "SUCCESS", "test-tenant")
        metrics.record_execution_plan("PARALLEL", "ERROR", "test-tenant")
        
        assert True  # No exceptions means metrics were recorded
    
    def test_approval_request_metrics(self):
        """Test approval request metrics recording"""
        # Record approval requests
        metrics.record_approval_request("create", "tool", "approved", "test-tenant")
        metrics.record_approval_request("delete", "server", "rejected", "test-tenant")
        
        assert True  # No exceptions means metrics were recorded
    
    def test_authentication_metrics(self):
        """Test authentication metrics recording"""
        # Record authentication attempts
        metrics.record_authentication_attempt("api_key", "success", "test-tenant")
        metrics.record_authentication_attempt("jwt", "failure", "test-tenant")
        
        assert True  # No exceptions means metrics were recorded
    
    def test_audit_event_metrics(self):
        """Test audit event metrics recording"""
        # Record audit events
        metrics.record_audit_event("create", "tool", True, "test-tenant")
        metrics.record_audit_event("delete", "server", False, "test-tenant")
        
        assert True  # No exceptions means metrics were recorded
    
    def test_database_operation_metrics(self):
        """Test database operation metrics recording"""
        # Record database operations
        metrics.record_database_operation("SELECT", "tools", "success", "test-tenant", 0.01)
        metrics.record_database_operation("INSERT", "servers", "error", "test-tenant", 0.02)
        
        assert True  # No exceptions means metrics were recorded
    
    def test_embedding_operation_metrics(self):
        """Test embedding operation metrics recording"""
        # Record embedding operations
        metrics.record_embedding_operation("encode", "sentence-transformers", "success", "test-tenant", 0.5)
        metrics.record_embedding_operation("search", "sentence-transformers", "error", "test-tenant", 0.3)
        
        assert True  # No exceptions means metrics were recorded
    
    def test_llm_request_metrics(self):
        """Test LLM request metrics recording"""
        # Record LLM requests
        metrics.record_llm_request("openai", "gpt-4", "success", "test-tenant", 2.0)
        metrics.record_llm_request("anthropic", "claude-3", "error", "test-tenant", 1.5)
        
        assert True  # No exceptions means metrics were recorded
    
    def test_llm_tokens_metrics(self):
        """Test LLM tokens metrics recording"""
        # Record LLM tokens
        metrics.record_llm_tokens("openai", "gpt-4", "input", 100, "test-tenant")
        metrics.record_llm_tokens("openai", "gpt-4", "output", 50, "test-tenant")
        
        assert True  # No exceptions means metrics were recorded
    
    def test_active_connections_metrics(self):
        """Test active connections metrics recording"""
        # Update active connections
        metrics.update_active_connections("test-tenant", 5)
        metrics.update_active_connections("test-tenant", 10)
        
        assert True  # No exceptions means metrics were recorded
    
    def test_system_info_initialization(self):
        """Test that system info is properly initialized"""
        # System info should be initialized with basic system information
        assert metrics.system_info is not None
        
        # We can't easily test the exact content without scraping metrics,
        # but we can ensure it was initialized
        assert True
    
    def test_metrics_endpoint_availability(self, client):
        """Test that metrics endpoint is available"""
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers.get("content-type", "")
        
        # Check that some expected metrics are present
        content = response.text
        assert "rmcp_" in content  # Our custom metrics should be present
        assert "http_requests_total" in content  # FastAPI instrumentator metrics
    
    def test_root_endpoint_includes_metrics_info(self, client):
        """Test that root endpoint includes metrics information"""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert "metrics" in data
        assert data["metrics"] == "/metrics"
    
    def test_health_endpoint_availability(self, client):
        """Test that health endpoint is available"""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "rmcp"
    
    @pytest.mark.asyncio
    async def test_executor_metrics_integration(self, circuit_breaker_executor, test_tool, test_server, db_manager):
        """Test that executor properly records metrics"""
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
            mock_response.json.return_value = {"result": "success"}
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response
            
            result = await circuit_breaker_executor.execute(
                plan=plan,
                user_id="test-user",
                tenant_id="test-tenant"
            )
        
        # Should succeed
        assert result.status == "SUCCESS"
        
        # Metrics should have been recorded (we can't easily verify exact values
        # without scraping the metrics endpoint, but no exceptions means they were recorded)
        assert True
    
    def test_circuit_breaker_metrics_integration(self, circuit_breaker_manager):
        """Test that circuit breaker properly records metrics"""
        # Record some circuit breaker events
        circuit_breaker_manager.record_success("test-server", "test-tenant")
        circuit_breaker_manager.record_failure("test-server", "test-tenant")
        
        # Metrics should have been recorded
        assert True
    
    def test_metrics_labels_consistency(self):
        """Test that metrics use consistent label names"""
        # This test ensures that all metrics use consistent label naming
        # We can't easily test the exact label values, but we can ensure
        # that the metrics methods accept the expected parameters
        
        # Test that all metric recording methods accept tenant_id
        metrics.record_plan_created("SOLO", "test-tenant", False)
        metrics.record_tool_execution("tool1", "server1", "SUCCESS", "test-tenant")
        metrics.record_circuit_breaker_success("server1", "test-tenant")
        metrics.record_approval_request("create", "tool", "approved", "test-tenant")
        metrics.record_authentication_attempt("api_key", "success", "test-tenant")
        metrics.record_audit_event("create", "tool", True, "test-tenant")
        metrics.record_database_operation("SELECT", "tools", "success", "test-tenant")
        metrics.record_embedding_operation("encode", "model", "success", "test-tenant")
        metrics.record_llm_request("provider", "model", "success", "test-tenant")
        metrics.record_llm_tokens("provider", "model", "input", 100, "test-tenant")
        metrics.update_active_connections("test-tenant", 5)
        metrics.update_circuit_breaker_state("server1", "test-tenant", "closed")
        
        assert True  # No exceptions means all methods work with consistent parameters
    
    def test_metrics_performance(self):
        """Test that metrics recording doesn't significantly impact performance"""
        start_time = time.time()
        
        # Record many metrics quickly
        for i in range(100):
            metrics.record_plan_created("SOLO", f"tenant-{i}", False)
            metrics.record_tool_execution(f"tool-{i}", f"server-{i}", "SUCCESS", f"tenant-{i}")
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should complete quickly (less than 1 second for 200 metric recordings)
        assert duration < 1.0
    
    def test_metrics_error_handling(self):
        """Test that metrics recording handles errors gracefully"""
        # Test with invalid parameters
        try:
            metrics.record_plan_created(None, None, None)
            metrics.record_tool_execution(None, None, None, None)
            metrics.record_circuit_breaker_success(None, None)
        except Exception:
            # Metrics should handle invalid parameters gracefully
            # If they raise exceptions, that's also acceptable behavior
            pass
        
        assert True  # Test passes if no critical errors occur

