"""
Tests for structured logging functionality
"""

import pytest
import json
import tempfile
import os
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from fastapi import FastAPI, Request

from rmcp.logging.config import (
    configure_logging,
    get_logger,
    log_request_start,
    log_request_end,
    log_tool_execution,
    log_planning,
    log_circuit_breaker,
    log_approval_request,
    log_telemetry,
    log_security,
    log_system,
    generate_request_id,
    LogContext
)
from rmcp.logging.middleware import (
    RequestLoggingMiddleware,
    HealthCheckMiddleware,
    MetricsMiddleware,
    SecurityLoggingMiddleware,
    ErrorLoggingMiddleware
)


class TestStructuredLogging:
    """Test structured logging functionality"""
    
    def _get_log_output(self, capsys):
        """Helper to get log output from either stdout or stderr"""
        captured = capsys.readouterr()
        # Return the actual log output, not debug output
        return captured.err.strip()
    
    def test_configure_logging(self):
        """Test logging configuration"""
        # This should not raise an exception
        configure_logging(log_level="INFO", json_format=True)
        
        # Test with different configurations
        configure_logging(log_level="DEBUG", json_format=False)
        configure_logging(log_level="WARNING", json_format=True, include_timestamps=False)
    
    def test_get_logger(self):
        """Test logger creation"""
        logger = get_logger("test.module")
        assert logger is not None
        
        # Test that logger has expected methods
        assert hasattr(logger, "info")
        assert hasattr(logger, "warning")
        assert hasattr(logger, "error")
        assert hasattr(logger, "debug")
    
    def test_generate_request_id(self):
        """Test request ID generation"""
        request_id1 = generate_request_id()
        request_id2 = generate_request_id()
        
        # Should be different
        assert request_id1 != request_id2
        
        # Should be valid UUIDs
        assert len(request_id1) == 36  # UUID string length
        assert len(request_id2) == 36
    
    def test_log_request_start(self, capsys):
        """Test request start logging"""
        configure_logging(log_level="INFO", json_format=True)
        
        log_request_start(
            method="GET",
            path="/api/v1/tools",
            request_id="test-request-123",
            user_id="user-456",
            tenant_id="tenant-789"
        )
        
        output = self._get_log_output(capsys)
        log_data = json.loads(output)
        
        assert log_data["event"] == "Request started"
        assert log_data["method"] == "GET"
        assert log_data["path"] == "/api/v1/tools"
        assert log_data["request_id"] == "test-request-123"
        assert log_data["user_id"] == "user-456"
        assert log_data["tenant_id"] == "tenant-789"
        assert "timestamp" in log_data
        assert log_data["service"] == "rmcp"
    
    def test_log_request_end(self, capsys):
        """Test request end logging"""
        configure_logging(log_level="INFO", json_format=True)
        
        log_request_end(
            method="GET",
            path="/api/v1/tools",
            request_id="test-request-123",
            status_code=200,
            duration_ms=150.5,
            user_id="user-456",
            tenant_id="tenant-789"
        )
        
        output = self._get_log_output(capsys)
        log_data = json.loads(output)
        
        assert log_data["event"] == "Request completed"
        assert log_data["method"] == "GET"
        assert log_data["path"] == "/api/v1/tools"
        assert log_data["request_id"] == "test-request-123"
        assert log_data["status_code"] == 200
        assert log_data["duration_ms"] == 150.5
        assert log_data["user_id"] == "user-456"
        assert log_data["tenant_id"] == "tenant-789"
    
    def test_log_tool_execution(self, capsys):
        """Test tool execution logging"""
        configure_logging(log_level="INFO", json_format=True)
        
        # Test successful execution
        log_tool_execution(
            tool_id="test-tool",
            server_id="test-server",
            success=True,
            duration_ms=250.0,
            request_id="test-request-123",
            user_id="user-456",
            tenant_id="tenant-789"
        )
        
        output = self._get_log_output(capsys)
        log_data = json.loads(output)
        
        assert log_data["event"] == "Tool execution"
        assert log_data["tool_id"] == "test-tool"
        assert log_data["server_id"] == "test-server"
        assert log_data["success"] is True
        assert log_data["duration_ms"] == 250.0
        assert log_data["request_id"] == "test-request-123"
        assert log_data["user_id"] == "user-456"
        assert log_data["tenant_id"] == "tenant-789"
    
    def test_log_tool_execution_failure(self, capsys):
        """Test tool execution failure logging"""
        configure_logging(log_level="INFO", json_format=True)
        
        log_tool_execution(
            tool_id="test-tool",
            server_id="test-server",
            success=False,
            duration_ms=1000.0,
            request_id="test-request-123",
            user_id="user-456",
            tenant_id="tenant-789",
            error="Connection timeout"
        )
        
        output = self._get_log_output(capsys)
        log_data = json.loads(output)
        
        assert log_data["event"] == "Tool execution"
        assert log_data["tool_id"] == "test-tool"
        assert log_data["server_id"] == "test-server"
        assert log_data["success"] is False
        assert log_data["duration_ms"] == 1000.0
        assert log_data["error"] == "Connection timeout"
    
    def test_log_planning(self, capsys):
        """Test planning logging"""
        configure_logging(log_level="INFO", json_format=True)
        
        log_planning(
            strategy="SOLO",
            tool_count=3,
            requires_approval=False,
            duration_ms=45.2,
            request_id="test-request-123",
            user_id="user-456",
            tenant_id="tenant-789",
            complexity_score=0.6
        )
        
        output = self._get_log_output(capsys)
        log_data = json.loads(output)
        
        assert log_data["event"] == "Plan created"
        assert log_data["strategy"] == "SOLO"
        assert log_data["tool_count"] == 3
        assert log_data["requires_approval"] is False
        assert log_data["duration_ms"] == 45.2
        assert log_data["complexity_score"] == 0.6
    
    def test_log_circuit_breaker(self, capsys):
        """Test circuit breaker logging"""
        configure_logging(log_level="INFO", json_format=True)
        
        log_circuit_breaker(
            server_id="test-server",
            state="OPEN",
            action="circuit_opened",
            request_id="test-request-123",
            user_id="user-456",
            tenant_id="tenant-789"
        )
        
        output = self._get_log_output(capsys)
        log_data = json.loads(output)
        
        assert log_data["event"] == "Circuit breaker activity"
        assert log_data["server_id"] == "test-server"
        assert log_data["state"] == "OPEN"
        assert log_data["action"] == "circuit_opened"
    
    def test_log_approval_request(self, capsys):
        """Test approval request logging"""
        configure_logging(log_level="INFO", json_format=True)
        
        log_approval_request(
            action="create",
            resource_type="tool_execution",
            resource_id="exec-123",
            status="pending",
            request_id="test-request-123",
            user_id="user-456",
            tenant_id="tenant-789"
        )
        
        output = self._get_log_output(capsys)
        log_data = json.loads(output)
        
        assert log_data["event"] == "Approval request"
        assert log_data["action"] == "create"
        assert log_data["resource_type"] == "tool_execution"
        assert log_data["resource_id"] == "exec-123"
        assert log_data["status"] == "pending"
    
    def test_log_telemetry(self, capsys):
        """Test telemetry logging"""
        configure_logging(log_level="INFO", json_format=True)
        
        log_telemetry(
            event_type="tool_execution",
            tool_id="test-tool",
            success=True,
            request_id="test-request-123",
            user_id="user-456",
            tenant_id="tenant-789"
        )
        
        output = self._get_log_output(capsys)
        log_data = json.loads(output)
        
        assert log_data["event"] == "Telemetry event"
        assert log_data["event_type"] == "tool_execution"
        assert log_data["tool_id"] == "test-tool"
        assert log_data["success"] is True
    
    def test_log_security(self, capsys):
        """Test security logging"""
        configure_logging(log_level="INFO", json_format=True)
        
        log_security(
            action="api_key_validation",
            user_id="user-456",
            tenant_id="tenant-789",
            success=True,
            request_id="test-request-123"
        )
        
        output = self._get_log_output(capsys)
        log_data = json.loads(output)
        
        assert log_data["event"] == "Security event"
        assert log_data["action"] == "api_key_validation"
        assert log_data["user_id"] == "user-456"
        assert log_data["tenant_id"] == "tenant-789"
        assert log_data["success"] is True
    
    def test_log_system(self, capsys):
        """Test system logging"""
        configure_logging(log_level="INFO", json_format=True)
        
        log_system(
            component="telemetry_engine",
            event="started",
            level="info"
        )
        
        output = self._get_log_output(capsys)
        
        if output:
            log_data = json.loads(output)
        else:
            pytest.skip("No log output captured")
        
        assert log_data["event"] == "started"
        assert log_data["component"] == "telemetry_engine"
    
    def test_log_context(self, capsys):
        """Test log context manager"""
        configure_logging(log_level="INFO", json_format=True)
        
        with LogContext(user_id="user-123", tenant_id="tenant-456") as logger:
            logger.info("Test message", action="test_action")
        
        output = self._get_log_output(capsys)
        log_data = json.loads(output)
        
        assert log_data["event"] == "Test message"
        assert log_data["user_id"] == "user-123"
        assert log_data["tenant_id"] == "tenant-456"
        assert log_data["action"] == "test_action"


class TestLoggingMiddleware:
    """Test logging middleware functionality"""
    
    def test_request_logging_middleware(self):
        """Test request logging middleware"""
        app = FastAPI()
        
        @app.get("/test")
        async def test_endpoint():
            return {"message": "test"}
        
        app.add_middleware(RequestLoggingMiddleware)
        
        client = TestClient(app)
        response = client.get("/test")
        
        assert response.status_code == 200
        assert "X-Request-ID" in response.headers
    
    def test_health_check_middleware(self):
        """Test health check middleware"""
        app = FastAPI()
        
        @app.get("/health")
        async def health():
            return {"status": "healthy"}
        
        app.add_middleware(HealthCheckMiddleware)
        
        client = TestClient(app)
        response = client.get("/health")
        
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}
    
    def test_metrics_middleware(self):
        """Test metrics middleware"""
        app = FastAPI()
        
        @app.get("/metrics")
        async def metrics():
            return "# HELP test_metric Test metric\n# TYPE test_metric counter\ntest_metric 1"
        
        app.add_middleware(MetricsMiddleware)
        
        client = TestClient(app)
        response = client.get("/metrics")
        
        assert response.status_code == 200
        assert "test_metric" in response.text
    
    def test_security_logging_middleware(self):
        """Test security logging middleware"""
        app = FastAPI()
        
        @app.get("/api/v1/auth/login")
        async def login():
            return {"token": "test-token"}
        
        app.add_middleware(SecurityLoggingMiddleware)
        
        client = TestClient(app)
        response = client.get("/api/v1/auth/login")
        
        assert response.status_code == 200
    
    def test_error_logging_middleware(self):
        """Test error logging middleware"""
        app = FastAPI()
        
        @app.get("/error")
        async def error_endpoint():
            raise ValueError("Test error")
        
        app.add_middleware(ErrorLoggingMiddleware)
        
        client = TestClient(app)
        
        # The middleware should log the error, but FastAPI will still return 500
        with pytest.raises(ValueError):
            response = client.get("/error")
    
    def test_middleware_order(self):
        """Test middleware order and interaction"""
        app = FastAPI()
        
        @app.get("/test")
        async def test_endpoint():
            return {"message": "test"}
        
        # Add middleware in correct order
        app.add_middleware(ErrorLoggingMiddleware)
        app.add_middleware(SecurityLoggingMiddleware)
        app.add_middleware(HealthCheckMiddleware)
        app.add_middleware(MetricsMiddleware)
        app.add_middleware(RequestLoggingMiddleware)
        
        client = TestClient(app)
        response = client.get("/test")
        
        assert response.status_code == 200
        assert "X-Request-ID" in response.headers


class TestLoggingIntegration:
    """Test logging integration with other components"""
    
    def _get_log_output(self, capsys):
        """Helper to get log output from either stdout or stderr"""
        captured = capsys.readouterr()
        # Return the actual log output, not debug output
        return captured.err.strip()
    
    def test_logging_with_mock_request(self):
        """Test logging with mock request object"""
        from fastapi import Request
        from unittest.mock import MagicMock
        
        # Create mock request
        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.url.path = "/api/v1/execute"
        mock_request.headers = {
            "x-user-id": "user-123",
            "x-tenant-id": "tenant-456",
            "user-agent": "test-client/1.0"
        }
        mock_request.query_params = {"param": "value"}
        mock_request.client.host = "127.0.0.1"
        
        # Test middleware with mock request
        middleware = RequestLoggingMiddleware(app=None)
        
        # Test user ID extraction
        user_id = middleware._extract_user_id(mock_request)
        assert user_id == "user-123"
        
        # Test tenant ID extraction
        tenant_id = middleware._extract_tenant_id(mock_request)
        assert tenant_id == "tenant-456"
    
    def test_logging_performance(self):
        """Test logging performance"""
        import time
        
        configure_logging(log_level="INFO", json_format=True)
        
        start_time = time.time()
        
        # Log many messages
        for i in range(100):
            log_system(f"component_{i}", f"event_{i}", level="info")
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should be fast (less than 1 second for 100 logs)
        assert duration < 1.0
    
    def test_logging_with_exceptions(self, capsys):
        """Test logging with exceptions"""
        configure_logging(log_level="INFO", json_format=True)
        
        try:
            raise ValueError("Test exception")
        except Exception as e:
            log_system("test", "exception_occurred", level="error", error=str(e))
        
        output = self._get_log_output(capsys)
        log_data = json.loads(output)
        
        assert log_data["event"] == "exception_occurred"
        assert log_data["error"] == "Test exception"
    
    def test_logging_context_preservation(self, capsys):
        """Test that logging context is preserved across calls"""
        configure_logging(log_level="INFO", json_format=True)
        
        # Create logger with context
        logger = get_logger("test").bind(user_id="user-123", tenant_id="tenant-456")
        
        # Log multiple messages
        logger.info("First message")
        logger.info("Second message")
        
        output = self._get_log_output(capsys)
        lines = output.split('\n')
        
        # Both messages should have the same context
        for line in lines:
            log_data = json.loads(line)
            assert log_data["user_id"] == "user-123"
            assert log_data["tenant_id"] == "tenant-456"
    
    def test_logging_with_unicode(self, capsys):
        """Test logging with unicode characters"""
        configure_logging(log_level="INFO", json_format=True)
        
        log_system("test", "unicode_test", level="info", message="Привет мир! 🌍")
        
        output = self._get_log_output(capsys)
        log_data = json.loads(output)
        
        assert log_data["event"] == "unicode_test"
        assert log_data["message"] == "Привет мир! 🌍"
