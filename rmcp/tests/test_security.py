"""
Tests for security module
"""

import pytest
import tempfile
import os
import json
from datetime import datetime, timedelta

from rmcp.storage.database import DatabaseManager
from rmcp.storage.schema import init_database
from rmcp.security import (
    AuthManager, RBACManager, AuditLogger, ApprovalGateway, 
    CircuitBreakerManager, SecurityMiddleware
)
from rmcp.security.models import ResourceType, PermissionLevel
from rmcp.security.approval import ApprovalStatus


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
def auth_manager(db_manager):
    """Fixture for auth manager."""
    return AuthManager(db_manager, "test-secret-key")


@pytest.fixture
def rbac_manager(db_manager):
    """Fixture for RBAC manager."""
    return RBACManager(db_manager)


@pytest.fixture
def audit_logger(db_manager):
    """Fixture for audit logger."""
    return AuditLogger(db_manager)


@pytest.fixture
def approval_gateway(db_manager):
    """Fixture for approval gateway."""
    return ApprovalGateway(db_manager)


@pytest.fixture
def circuit_breaker_manager(db_manager):
    """Fixture for circuit breaker manager."""
    return CircuitBreakerManager(db_manager)


class TestAuthManager:
    """Test authentication manager"""
    
    def test_create_tenant(self, auth_manager):
        """Test tenant creation"""
        tenant = auth_manager.create_tenant("test-tenant", "Test Tenant", "Test description")
        assert tenant.id == "test-tenant"
        assert tenant.name == "Test Tenant"
        assert tenant.description == "Test description"
        assert tenant.is_active is True
    
    def test_create_user(self, auth_manager):
        """Test user creation"""
        # Create tenant first
        auth_manager.create_tenant("test-tenant", "Test Tenant")
        
        user = auth_manager.create_user(
            "test-user", "testuser", "test@example.com", "test-tenant", ["user"]
        )
        assert user.id == "test-user"
        assert user.username == "testuser"
        assert user.email == "test@example.com"
        assert user.tenant_id == "test-tenant"
        assert user.roles == ["user"]
    
    def test_create_api_key(self, auth_manager):
        """Test API key creation"""
        # Create tenant and user first
        auth_manager.create_tenant("test-tenant", "Test Tenant")
        auth_manager.create_user("test-user", "testuser", "test@example.com", "test-tenant")
        
        api_key, api_key_info = auth_manager.create_api_key(
            "test-user", "test-tenant", "Test API Key", ["tool:execute"]
        )
        
        assert api_key.startswith("rmcp_")
        assert api_key_info.user_id == "test-user"
        assert api_key_info.tenant_id == "test-tenant"
        assert api_key_info.name == "Test API Key"
        assert api_key_info.permissions == ["tool:execute"]
    
    def test_authenticate_api_key(self, auth_manager):
        """Test API key authentication"""
        # Create tenant, user, and API key
        auth_manager.create_tenant("test-tenant", "Test Tenant")
        auth_manager.create_user("test-user", "testuser", "test@example.com", "test-tenant")
        api_key, _ = auth_manager.create_api_key("test-user", "test-tenant", "Test API Key")
        
        # Test authentication
        user_info = auth_manager.authenticate_api_key(api_key)
        assert user_info is not None
        assert user_info["user_id"] == "test-user"
        assert user_info["tenant_id"] == "test-tenant"
        assert user_info["username"] == "testuser"
        
        # Test invalid API key
        invalid_user_info = auth_manager.authenticate_api_key("invalid-key")
        assert invalid_user_info is None


class TestRBACManager:
    """Test RBAC manager"""
    
    def test_check_permission(self, rbac_manager):
        """Test permission checking"""
        user_permissions = ["tool:read", "tool:execute", "system:admin"]
        
        assert rbac_manager.check_permission(user_permissions, "tool:read") is True
        assert rbac_manager.check_permission(user_permissions, "tool:execute") is True
        assert rbac_manager.check_permission(user_permissions, "system:admin") is True
        assert rbac_manager.check_permission(user_permissions, "tool:admin") is False
    
    def test_check_resource_permission(self, rbac_manager):
        """Test resource permission checking"""
        user_permissions = ["tool:read", "tool:execute", "system:admin"]
        
        assert rbac_manager.check_resource_permission(
            user_permissions, ResourceType.TOOL, PermissionLevel.READ
        ) is True
        assert rbac_manager.check_resource_permission(
            user_permissions, ResourceType.TOOL, PermissionLevel.EXECUTE
        ) is True
        assert rbac_manager.check_resource_permission(
            user_permissions, ResourceType.SYSTEM, PermissionLevel.ADMIN
        ) is True
        assert rbac_manager.check_resource_permission(
            user_permissions, ResourceType.TOOL, PermissionLevel.ADMIN
        ) is False
    
    def test_create_role(self, rbac_manager):
        """Test role creation"""
        role = rbac_manager.create_role(
            "custom-role", "Custom Role", "A custom role", 
            ["tool:read", "tool:execute"], "test-tenant"
        )
        
        assert role.id == "custom-role"
        assert role.name == "Custom Role"
        assert role.description == "A custom role"
        assert role.permissions == ["tool:read", "tool:execute"]
        assert role.tenant_id == "test-tenant"
    
    def test_list_roles(self, rbac_manager):
        """Test role listing"""
        roles = rbac_manager.list_roles("test-tenant")
        
        # Should have default system roles
        role_names = [role.name for role in roles]
        assert "admin" in role_names
        assert "user" in role_names
        assert "readonly" in role_names


class TestAuditLogger:
    """Test audit logger"""
    
    def test_log_event(self, audit_logger):
        """Test event logging"""
        event_id = audit_logger.log_event(
            action="test_action",
            resource_type=ResourceType.SYSTEM,
            user_id="test-user",
            tenant_id="test-tenant",
            success=True
        )
        
        assert event_id is not None
        
        # Get events
        events = audit_logger.get_audit_events(user_id="test-user")
        assert len(events) == 1
        assert events[0]["action"] == "test_action"
        assert events[0]["user_id"] == "test-user"
        assert events[0]["success"] is True
    
    def test_log_authentication(self, audit_logger):
        """Test authentication logging"""
        event_id = audit_logger.log_authentication(
            user_id="test-user",
            tenant_id="test-tenant",
            success=True
        )
        
        assert event_id is not None
        
        events = audit_logger.get_audit_events(action="authentication")
        assert len(events) == 1
        assert events[0]["action"] == "authentication"
    
    def test_log_tool_execution(self, audit_logger):
        """Test tool execution logging"""
        event_id = audit_logger.log_tool_execution(
            user_id="test-user",
            tenant_id="test-tenant",
            tool_id="test-tool",
            success=True
        )
        
        assert event_id is not None
        
        events = audit_logger.get_audit_events(action="tool_execution")
        assert len(events) == 1
        assert events[0]["resource_id"] == "test-tool"


class TestApprovalGateway:
    """Test approval gateway"""
    
    def test_create_approval_request(self, approval_gateway):
        """Test approval request creation"""
        request = approval_gateway.create_approval_request(
            user_id="test-user",
            tenant_id="test-tenant",
            action="dangerous_action",
            resource_type="tool",
            resource_id="dangerous-tool",
            details={"risk_level": "high"},
            justification="Testing dangerous operation"
        )
        
        assert request.id is not None
        assert request.user_id == "test-user"
        assert request.tenant_id == "test-tenant"
        assert request.action == "dangerous_action"
        assert request.status == ApprovalStatus.PENDING
        assert request.justification == "Testing dangerous operation"
    
    def test_approve_request(self, approval_gateway):
        """Test approval request approval"""
        # Create request
        request = approval_gateway.create_approval_request(
            user_id="test-user",
            tenant_id="test-tenant",
            action="dangerous_action",
            resource_type="tool",
            resource_id="test-tool",
            details={"risk_level": "high"},
            justification="Testing"
        )
        
        # Approve request
        success = approval_gateway.approve_request(request.id, "approver-user")
        assert success is True
        
        # Check status
        updated_request = approval_gateway.get_approval_request(request.id)
        assert updated_request.status == ApprovalStatus.APPROVED
        assert updated_request.approver_id == "approver-user"
    
    def test_reject_request(self, approval_gateway):
        """Test approval request rejection"""
        # Create request
        request = approval_gateway.create_approval_request(
            user_id="test-user",
            tenant_id="test-tenant",
            action="dangerous_action",
            resource_type="tool",
            resource_id="test-tool",
            details={"risk_level": "high"},
            justification="Testing"
        )
        
        # Reject request
        success = approval_gateway.reject_request(request.id, "approver-user")
        assert success is True
        
        # Check status
        updated_request = approval_gateway.get_approval_request(request.id)
        assert updated_request.status == ApprovalStatus.REJECTED
        assert updated_request.approver_id == "approver-user"
    
    def test_requires_approval(self, approval_gateway):
        """Test approval requirement checking"""
        # Admin user should not require approval
        admin_permissions = ["system:admin"]
        assert approval_gateway.requires_approval("shell.execute", "tool", admin_permissions) is False
        
        # Regular user should require approval for dangerous actions
        user_permissions = ["tool:execute"]
        assert approval_gateway.requires_approval("shell.execute", "tool", user_permissions) is True
        
        # Regular user should not require approval for safe actions
        assert approval_gateway.requires_approval("ripgrep.search", "tool", user_permissions) is False


class TestCircuitBreakerManager:
    """Test circuit breaker manager"""
    
    def test_circuit_breaker_creation(self, circuit_breaker_manager):
        """Test circuit breaker creation"""
        circuit_breaker = circuit_breaker_manager.get_circuit_breaker("test-server")
        
        assert circuit_breaker.server_id == "test-server"
        assert circuit_breaker.can_execute() is True  # Should start in closed state
    
    def test_circuit_breaker_success(self, circuit_breaker_manager):
        """Test circuit breaker success recording"""
        circuit_breaker_manager.record_success("test-server")
        
        circuit_breaker = circuit_breaker_manager.get_circuit_breaker("test-server")
        stats = circuit_breaker.get_stats()
        
        assert stats.total_requests == 1
        assert stats.total_successes == 1
        assert stats.total_failures == 0
    
    def test_circuit_breaker_failure(self, circuit_breaker_manager):
        """Test circuit breaker failure recording"""
        # Record multiple failures to trigger circuit breaker
        for _ in range(6):  # Default threshold is 5
            circuit_breaker_manager.record_failure("test-server")
        
        circuit_breaker = circuit_breaker_manager.get_circuit_breaker("test-server")
        assert circuit_breaker.can_execute() is False  # Circuit should be open
    
    def test_circuit_breaker_recovery(self, circuit_breaker_manager):
        """Test circuit breaker recovery"""
        # Open the circuit
        for _ in range(6):
            circuit_breaker_manager.record_failure("test-server")
        
        circuit_breaker = circuit_breaker_manager.get_circuit_breaker("test-server")
        assert circuit_breaker.can_execute() is False
        
        # Reset circuit breaker
        circuit_breaker_manager.reset_circuit_breaker("test-server")
        assert circuit_breaker.can_execute() is True


class TestSecurityIntegration:
    """Test security components integration"""
    
    def test_security_middleware_creation(self, auth_manager, rbac_manager, 
                                        audit_logger, approval_gateway, 
                                        circuit_breaker_manager):
        """Test security middleware creation"""
        middleware = SecurityMiddleware(
            auth_manager=auth_manager,
            rbac_manager=rbac_manager,
            audit_logger=audit_logger,
            approval_gateway=approval_gateway,
            circuit_breaker_manager=circuit_breaker_manager
        )
        
        assert middleware.auth_manager == auth_manager
        assert middleware.rbac_manager == rbac_manager
        assert middleware.audit_logger == audit_logger
        assert middleware.approval_gateway == approval_gateway
        assert middleware.circuit_breaker_manager == circuit_breaker_manager
    
    def test_end_to_end_security_flow(self, auth_manager, rbac_manager, 
                                     audit_logger, approval_gateway):
        """Test end-to-end security flow"""
        # Create tenant and user
        auth_manager.create_tenant("test-tenant", "Test Tenant")
        auth_manager.create_user("test-user", "testuser", "test@example.com", "test-tenant", ["user"])
        
        # Create API key
        api_key, _ = auth_manager.create_api_key("test-user", "test-tenant", "Test Key")
        
        # Authenticate
        user_info = auth_manager.authenticate_api_key(api_key)
        assert user_info is not None
        
        # Check permissions
        user_permissions = rbac_manager.get_user_effective_permissions("test-user", "test-tenant")
        assert len(user_permissions) > 0
        
        # Log audit event
        event_id = audit_logger.log_tool_execution(
            user_id="test-user",
            tenant_id="test-tenant",
            tool_id="test-tool",
            success=True
        )
        assert event_id is not None
        
        # Check approval requirement
        requires_approval = approval_gateway.requires_approval(
            "shell.execute", "tool", user_permissions
        )
        assert requires_approval is True
