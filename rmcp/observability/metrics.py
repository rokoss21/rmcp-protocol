"""
Prometheus metrics definitions for RMCP
"""

from prometheus_client import Counter, Gauge, Histogram, Info
from typing import Dict, Any


class RMCPMetrics:
    """RMCP Prometheus metrics collection"""
    
    def __init__(self):
        # Planning metrics
        self.plans_created_total = Counter(
            'rmcp_plans_created_total',
            'Total number of execution plans created',
            ['strategy', 'tenant_id', 'requires_approval']
        )
        
        self.planning_duration_seconds = Histogram(
            'rmcp_planning_duration_seconds',
            'Time spent on planning execution',
            ['strategy', 'tenant_id'],
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
        )
        
        # Execution metrics
        self.tool_executions_total = Counter(
            'rmcp_tool_executions_total',
            'Total number of tool executions',
            ['tool_id', 'server_id', 'status', 'tenant_id']
        )
        
        self.tool_execution_duration_seconds = Histogram(
            'rmcp_tool_execution_duration_seconds',
            'Time spent executing tools',
            ['tool_id', 'server_id', 'tenant_id'],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
        )
        
        self.execution_plans_total = Counter(
            'rmcp_execution_plans_total',
            'Total number of execution plans processed',
            ['strategy', 'status', 'tenant_id']
        )
        
        # Circuit breaker metrics
        self.circuit_breaker_state = Gauge(
            'rmcp_circuit_breaker_state',
            'Circuit breaker state (0=closed, 1=open, 2=half_open)',
            ['server_id', 'tenant_id']
        )
        
        self.circuit_breaker_failures_total = Counter(
            'rmcp_circuit_breaker_failures_total',
            'Total number of circuit breaker failures',
            ['server_id', 'tenant_id']
        )
        
        self.circuit_breaker_successes_total = Counter(
            'rmcp_circuit_breaker_successes_total',
            'Total number of circuit breaker successes',
            ['server_id', 'tenant_id']
        )
        
        # Security metrics
        self.approval_requests_total = Counter(
            'rmcp_approval_requests_total',
            'Total number of approval requests',
            ['action', 'resource_type', 'status', 'tenant_id']
        )
        
        self.authentication_attempts_total = Counter(
            'rmcp_authentication_attempts_total',
            'Total number of authentication attempts',
            ['method', 'status', 'tenant_id']
        )
        
        self.audit_events_total = Counter(
            'rmcp_audit_events_total',
            'Total number of audit events',
            ['action', 'resource_type', 'success', 'tenant_id']
        )
        
        # System metrics
        self.active_connections = Gauge(
            'rmcp_active_connections',
            'Number of active connections',
            ['tenant_id']
        )
        
        self.database_operations_total = Counter(
            'rmcp_database_operations_total',
            'Total number of database operations',
            ['operation', 'table', 'status', 'tenant_id']
        )
        
        self.database_operation_duration_seconds = Histogram(
            'rmcp_database_operation_duration_seconds',
            'Time spent on database operations',
            ['operation', 'table', 'tenant_id'],
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
        )
        
        # Embedding metrics
        self.embedding_operations_total = Counter(
            'rmcp_embedding_operations_total',
            'Total number of embedding operations',
            ['operation', 'model', 'status', 'tenant_id']
        )
        
        self.embedding_operation_duration_seconds = Histogram(
            'rmcp_embedding_operation_duration_seconds',
            'Time spent on embedding operations',
            ['operation', 'model', 'tenant_id'],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
        )
        
        # LLM metrics
        self.llm_requests_total = Counter(
            'rmcp_llm_requests_total',
            'Total number of LLM requests',
            ['provider', 'model', 'status', 'tenant_id']
        )
        
        self.llm_request_duration_seconds = Histogram(
            'rmcp_llm_request_duration_seconds',
            'Time spent on LLM requests',
            ['provider', 'model', 'tenant_id'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0]
        )
        
        self.llm_tokens_total = Counter(
            'rmcp_llm_tokens_total',
            'Total number of LLM tokens processed',
            ['provider', 'model', 'token_type', 'tenant_id']
        )
        
        # System info
        self.system_info = Info(
            'rmcp_system_info',
            'RMCP system information'
        )
        
        # Initialize system info
        self._initialize_system_info()
    
    def _initialize_system_info(self):
        """Initialize system information metric"""
        import platform
        import sys
        
        self.system_info.info({
            'version': '1.0.0',
            'python_version': sys.version.split()[0],
            'platform': platform.platform(),
            'architecture': platform.architecture()[0]
        })
    
    def record_plan_created(self, strategy: str, tenant_id: str, requires_approval: bool):
        """Record a plan creation"""
        self.plans_created_total.labels(
            strategy=strategy,
            tenant_id=tenant_id,
            requires_approval=str(requires_approval)
        ).inc()
    
    def record_planning_duration(self, strategy: str, tenant_id: str, duration: float):
        """Record planning duration"""
        self.planning_duration_seconds.labels(
            strategy=strategy,
            tenant_id=tenant_id
        ).observe(duration)
    
    def record_tool_execution(self, tool_id: str, server_id: str, status: str, tenant_id: str, duration: float = None):
        """Record a tool execution"""
        self.tool_executions_total.labels(
            tool_id=tool_id,
            server_id=server_id,
            status=status,
            tenant_id=tenant_id
        ).inc()
        
        if duration is not None:
            self.tool_execution_duration_seconds.labels(
                tool_id=tool_id,
                server_id=server_id,
                tenant_id=tenant_id
            ).observe(duration)
    
    def record_execution_plan(self, strategy: str, status: str, tenant_id: str):
        """Record an execution plan processing"""
        self.execution_plans_total.labels(
            strategy=strategy,
            status=status,
            tenant_id=tenant_id
        ).inc()
    
    def update_circuit_breaker_state(self, server_id: str, tenant_id: str, state: str):
        """Update circuit breaker state"""
        state_value = {'closed': 0, 'open': 1, 'half_open': 2}.get(state, 0)
        self.circuit_breaker_state.labels(
            server_id=server_id,
            tenant_id=tenant_id
        ).set(state_value)
    
    def record_circuit_breaker_failure(self, server_id: str, tenant_id: str):
        """Record a circuit breaker failure"""
        self.circuit_breaker_failures_total.labels(
            server_id=server_id,
            tenant_id=tenant_id
        ).inc()
    
    def record_circuit_breaker_success(self, server_id: str, tenant_id: str):
        """Record a circuit breaker success"""
        self.circuit_breaker_successes_total.labels(
            server_id=server_id,
            tenant_id=tenant_id
        ).inc()
    
    def record_approval_request(self, action: str, resource_type: str, status: str, tenant_id: str):
        """Record an approval request"""
        self.approval_requests_total.labels(
            action=action,
            resource_type=resource_type,
            status=status,
            tenant_id=tenant_id
        ).inc()
    
    def record_authentication_attempt(self, method: str, status: str, tenant_id: str):
        """Record an authentication attempt"""
        self.authentication_attempts_total.labels(
            method=method,
            status=status,
            tenant_id=tenant_id
        ).inc()
    
    def record_audit_event(self, action: str, resource_type: str, success: bool, tenant_id: str):
        """Record an audit event"""
        self.audit_events_total.labels(
            action=action,
            resource_type=resource_type,
            success=str(success),
            tenant_id=tenant_id
        ).inc()
    
    def update_active_connections(self, tenant_id: str, count: int):
        """Update active connections count"""
        self.active_connections.labels(tenant_id=tenant_id).set(count)
    
    def record_database_operation(self, operation: str, table: str, status: str, tenant_id: str, duration: float = None):
        """Record a database operation"""
        self.database_operations_total.labels(
            operation=operation,
            table=table,
            status=status,
            tenant_id=tenant_id
        ).inc()
        
        if duration is not None:
            self.database_operation_duration_seconds.labels(
                operation=operation,
                table=table,
                tenant_id=tenant_id
            ).observe(duration)
    
    def record_embedding_operation(self, operation: str, model: str, status: str, tenant_id: str, duration: float = None):
        """Record an embedding operation"""
        self.embedding_operations_total.labels(
            operation=operation,
            model=model,
            status=status,
            tenant_id=tenant_id
        ).inc()
        
        if duration is not None:
            self.embedding_operation_duration_seconds.labels(
                operation=operation,
                model=model,
                tenant_id=tenant_id
            ).observe(duration)
    
    def record_llm_request(self, provider: str, model: str, status: str, tenant_id: str, duration: float = None):
        """Record an LLM request"""
        self.llm_requests_total.labels(
            provider=provider,
            model=model,
            status=status,
            tenant_id=tenant_id
        ).inc()
        
        if duration is not None:
            self.llm_request_duration_seconds.labels(
                provider=provider,
                model=model,
                tenant_id=tenant_id
            ).observe(duration)
    
    def record_llm_tokens(self, provider: str, model: str, token_type: str, count: int, tenant_id: str):
        """Record LLM token usage"""
        self.llm_tokens_total.labels(
            provider=provider,
            model=model,
            token_type=token_type,
            tenant_id=tenant_id
        ).inc(count)


# Global metrics instance
metrics = RMCPMetrics()

