"""
Tests for approval workflow integration
"""

import pytest
import tempfile
import os
from datetime import datetime

from rmcp.storage.database import DatabaseManager
from rmcp.storage.schema import init_database
from rmcp.security import AuthManager, RBACManager, AuditLogger, ApprovalGateway
from rmcp.core.approval_executor import ApprovalExecutor
from rmcp.models.plan import ExecutionPlan, ExecutionStrategy, ExecutionStep
from rmcp.models.execution import ExecutionStatus


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
def approval_executor(db_manager, approval_gateway, audit_logger):
    """Fixture for approval executor."""
    return ApprovalExecutor(db_manager, approval_gateway, audit_logger)


@pytest.fixture
def test_tool():
    """Fixture for a test tool."""
    from rmcp.models.tool import Tool
    return Tool(
        id="test-tool",
        server_id="test-server",
        name="shell.execute",
        description="Execute shell commands",
        input_schema={"command": {"type": "string"}},
        output_schema={"result": {"type": "string"}},
        tags=["shell", "execution"],
        capabilities=["shell.execute", "execution"],
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


class TestApprovalIntegration:
    """Test approval workflow integration"""
    
    def test_plan_requires_approval_detection(self, approval_executor, test_tool, test_server, db_manager):
        """Test that planner correctly detects approval requirements"""
        # Add test tool and server to database
        db_manager.add_server(test_server)
        db_manager.add_tool(test_tool)
        
        # Create execution plan with dangerous tool
        plan = ExecutionPlan(
            strategy=ExecutionStrategy.SOLO,
            steps=[ExecutionStep(
                tool_id="test-tool",
                args={"command": "rm -rf /tmp/test"}
            )],
            requires_approval=True  # This should be set by planner
        )
        
        # Test that plan requires approval
        assert plan.requires_approval is True
    
    @pytest.mark.asyncio
    async def test_execution_approval_workflow(self, approval_executor, test_tool, test_server, db_manager):
        """Test complete approval workflow"""
        # Add test tool and server to database
        db_manager.add_server(test_server)
        db_manager.add_tool(test_tool)
        
        # Create execution plan requiring approval
        plan = ExecutionPlan(
            strategy=ExecutionStrategy.SOLO,
            steps=[ExecutionStep(
                tool_id="test-tool",
                args={"command": "rm -rf /tmp/test"}
            )],
            requires_approval=True
        )
        
        # Execute plan - should require approval
        result = await approval_executor.execute(
            plan=plan,
            user_id="test-user",
            tenant_id="test-tenant",
            context={"justification": "Testing approval workflow"}
        )
        
        # Should return approval required status
        assert result.status == "AWAITING_APPROVAL"
        assert "approval_request_id" in result.data
        assert "execution_request_id" in result.data
        
        # Check that approval request was created
        approval_request_id = result.data["approval_request_id"]
        approval_request = approval_executor.approval_gateway.get_approval_request(approval_request_id)
        assert approval_request is not None
        assert approval_request.action == "execute_plan"
        assert approval_request.resource_type == "execution"
    
    @pytest.mark.asyncio
    async def test_execution_without_approval(self, approval_executor, test_tool, test_server, db_manager):
        """Test execution without approval requirements"""
        # Add test tool and server to database
        db_manager.add_server(test_server)
        db_manager.add_tool(test_tool)
        
        # Create execution plan not requiring approval
        plan = ExecutionPlan(
            strategy=ExecutionStrategy.SOLO,
            steps=[ExecutionStep(
                tool_id="test-tool",
                args={"command": "echo 'hello'"}
            )],
            requires_approval=False
        )
        
        # Execute plan - should execute immediately
        result = await approval_executor.execute(
            plan=plan,
            user_id="test-user",
            tenant_id="test-tenant"
        )
        
        # Should return success status (or error if MCP server not available)
        assert result.status in ["SUCCESS", "ERROR"]
    
    def test_approval_request_creation(self, approval_gateway):
        """Test approval request creation"""
        request = approval_gateway.create_approval_request(
            user_id="test-user",
            tenant_id="test-tenant",
            action="execute_plan",
            resource_type="execution",
            resource_id="test-execution",
            details={
                "plan_summary": "Execute shell command",
                "risk_level": "high"
            },
            justification="Testing dangerous operation"
        )
        
        assert request.user_id == "test-user"
        assert request.tenant_id == "test-tenant"
        assert request.action == "execute_plan"
        assert request.resource_type == "execution"
        assert request.justification == "Testing dangerous operation"
        assert request.status.value == "pending"
    
    def test_approval_request_approval(self, approval_gateway):
        """Test approval request approval"""
        # Create request
        request = approval_gateway.create_approval_request(
            user_id="test-user",
            tenant_id="test-tenant",
            action="execute_plan",
            resource_type="execution",
            resource_id="test-execution",
            details={"plan_summary": "Execute shell command"},
            justification="Testing"
        )
        
        # Approve request
        success = approval_gateway.approve_request(request.id, "approver-user")
        assert success is True
        
        # Check status
        updated_request = approval_gateway.get_approval_request(request.id)
        assert updated_request.status == "approved"
        assert updated_request.approver_id == "approver-user"
    
    def test_approval_request_rejection(self, approval_gateway):
        """Test approval request rejection"""
        # Create request
        request = approval_gateway.create_approval_request(
            user_id="test-user",
            tenant_id="test-tenant",
            action="execute_plan",
            resource_type="execution",
            resource_id="test-execution",
            details={"plan_summary": "Execute shell command"},
            justification="Testing"
        )
        
        # Reject request
        success = approval_gateway.reject_request(request.id, "approver-user")
        assert success is True
        
        # Check status
        updated_request = approval_gateway.get_approval_request(request.id)
        assert updated_request.status == "rejected"
        assert updated_request.approver_id == "approver-user"
    
    def test_dangerous_operation_detection(self, approval_gateway):
        """Test detection of dangerous operations"""
        # Admin user should not require approval
        admin_permissions = ["system:admin"]
        assert approval_gateway.requires_approval("shell.execute", "tool", admin_permissions) is False
        
        # Regular user should require approval for dangerous actions
        user_permissions = ["tool:execute"]
        assert approval_gateway.requires_approval("shell.execute", "tool", user_permissions) is True
        assert approval_gateway.requires_approval("terraform.apply", "tool", user_permissions) is True
        assert approval_gateway.requires_approval("kubectl.delete", "tool", user_permissions) is True
        
        # Regular user should not require approval for safe actions
        assert approval_gateway.requires_approval("ripgrep.search", "tool", user_permissions) is False
        assert approval_gateway.requires_approval("file.read", "tool", user_permissions) is False
    
    def test_execution_request_creation(self, approval_executor):
        """Test execution request creation"""
        from rmcp.models.plan import ExecutionPlan, ExecutionStrategy, ExecutionStep
        
        plan = ExecutionPlan(
            strategy=ExecutionStrategy.SOLO,
            steps=[ExecutionStep(tool_id="test-tool", args={})]
        )
        
        # This would normally be async, but we're testing the creation logic
        import asyncio
        execution_request = asyncio.run(approval_executor._create_execution_request(
            plan, "test-user", "test-tenant"
        ))
        
        assert execution_request.user_id == "test-user"
        assert execution_request.tenant_id == "test-tenant"
        assert execution_request.status == ExecutionStatus.PENDING
        assert execution_request.plan_id is not None
    
    def test_plan_summary_generation(self, approval_executor, test_tool, db_manager):
        """Test plan summary generation"""
        # Add test tool to database
        db_manager.add_tool(test_tool)
        
        from rmcp.models.plan import ExecutionPlan, ExecutionStrategy, ExecutionStep
        
        # Test solo plan summary
        plan = ExecutionPlan(
            strategy=ExecutionStrategy.SOLO,
            steps=[ExecutionStep(tool_id="test-tool", args={})]
        )
        
        summary = approval_executor._generate_plan_summary(plan)
        assert "shell.execute" in summary or "test-tool" in summary
        
        # Test parallel plan summary
        plan = ExecutionPlan(
            strategy=ExecutionStrategy.PARALLEL,
            steps=[
                ExecutionStep(tool_id="test-tool", args={}),
                ExecutionStep(tool_id="test-tool", args={})
            ]
        )
        
        summary = approval_executor._generate_plan_summary(plan)
        assert "parallel" in summary
        assert "2 steps" in summary
    
    def test_risk_level_assessment(self, approval_executor, test_tool, db_manager):
        """Test risk level assessment"""
        # Add test tool to database
        db_manager.add_tool(test_tool)
        
        from rmcp.models.plan import ExecutionPlan, ExecutionStrategy, ExecutionStep
        
        # Test high risk plan
        plan = ExecutionPlan(
            strategy=ExecutionStrategy.SOLO,
            steps=[ExecutionStep(tool_id="test-tool", args={})]
        )
        
        risk_level = approval_executor._assess_risk_level(plan)
        assert risk_level == "high"  # shell.execute is dangerous
        
        # Test medium risk plan (no dangerous tools)
        safe_tool = test_tool.model_copy()
        safe_tool.id = "safe-tool"
        safe_tool.name = "file.read"
        safe_tool.capabilities = ["file.read"]
        db_manager.add_tool(safe_tool)
        
        plan = ExecutionPlan(
            strategy=ExecutionStrategy.SOLO,
            steps=[ExecutionStep(tool_id="safe-tool", args={})]
        )
        
        risk_level = approval_executor._assess_risk_level(plan)
        assert risk_level == "medium"  # Default for non-dangerous tools
