"""
Structured logging configuration for RMCP
"""

import sys
import json
import uuid
from typing import Any, Dict, Optional
from datetime import datetime

import structlog
from structlog.stdlib import LoggerFactory


class RMCPJSONRenderer:
    """Custom JSON renderer for RMCP logs"""
    
    def __call__(self, logger, name, event_dict):
        """Render log event as JSON"""
        # Add timestamp if not present
        if "timestamp" not in event_dict:
            event_dict["timestamp"] = datetime.utcnow().isoformat() + "Z"
        
        # Add service name
        event_dict["service"] = "rmcp"
        
        # Add version
        event_dict["version"] = "1.0.0"
        
        # Ensure event is present
        if "event" not in event_dict:
            event_dict["event"] = event_dict.get("msg", "log_event")
        
        # Remove msg field if present (we use event instead)
        event_dict.pop("msg", None)
        
        # Convert to JSON
        return json.dumps(event_dict, ensure_ascii=False, separators=(',', ':'))


def configure_logging(
    log_level: str = "INFO",
    json_format: bool = True,
    include_timestamps: bool = True
) -> None:
    """
    Configure structured logging for RMCP
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_format: Whether to use JSON format
        include_timestamps: Whether to include timestamps
    """
    
    # Configure structlog
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]
    
    if include_timestamps:
        processors.append(structlog.processors.TimeStamper(fmt="iso"))
    
    if json_format:
        processors.append(RMCPJSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())
    
    # Reset structlog configuration
    structlog.reset_defaults()
    
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=LoggerFactory(),
        context_class=dict,
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    import logging
    
    # Clear existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Set log level
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(message)s",
        stream=sys.stderr,  # structlog outputs to stderr by default
        force=True  # Force reconfiguration
    )
    
    # Reduce noise from third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Get a structured logger instance
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Configured structured logger
    """
    return structlog.get_logger(name)


class LogContext:
    """Context manager for adding structured context to logs"""
    
    def __init__(self, **context):
        self.context = context
        self.logger = structlog.get_logger()
    
    def __enter__(self):
        """Enter context and bind context to logger"""
        self.logger = self.logger.bind(**self.context)
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context"""
        pass


def log_request_start(
    method: str,
    path: str,
    request_id: str,
    user_id: Optional[str] = None,
    tenant_id: Optional[str] = None,
    **kwargs
) -> None:
    """Log the start of a request"""
    logger = structlog.get_logger("rmcp.request")
    logger.info(
        "Request started",
        method=method,
        path=path,
        request_id=request_id,
        user_id=user_id,
        tenant_id=tenant_id,
        **kwargs
    )


def log_request_end(
    method: str,
    path: str,
    request_id: str,
    status_code: int,
    duration_ms: float,
    user_id: Optional[str] = None,
    tenant_id: Optional[str] = None,
    **kwargs
) -> None:
    """Log the end of a request"""
    logger = structlog.get_logger("rmcp.request")
    logger.info(
        "Request completed",
        method=method,
        path=path,
        request_id=request_id,
        status_code=status_code,
        duration_ms=duration_ms,
        user_id=user_id,
        tenant_id=tenant_id,
        **kwargs
    )


def log_tool_execution(
    tool_id: str,
    server_id: str,
    success: bool,
    duration_ms: float,
    request_id: Optional[str] = None,
    user_id: Optional[str] = None,
    tenant_id: Optional[str] = None,
    error: Optional[str] = None,
    **kwargs
) -> None:
    """Log tool execution"""
    logger = structlog.get_logger("rmcp.tool")
    level = "info" if success else "error"
    
    log_data = {
        "tool_id": tool_id,
        "server_id": server_id,
        "success": success,
        "duration_ms": duration_ms,
        "request_id": request_id,
        "user_id": user_id,
        "tenant_id": tenant_id,
        **kwargs
    }
    
    if error:
        log_data["error"] = error
    
    getattr(logger, level)("Tool execution", **log_data)


def log_planning(
    strategy: str,
    tool_count: int,
    requires_approval: bool,
    duration_ms: float,
    request_id: Optional[str] = None,
    user_id: Optional[str] = None,
    tenant_id: Optional[str] = None,
    **kwargs
) -> None:
    """Log planning activity"""
    logger = structlog.get_logger("rmcp.planning")
    logger.info(
        "Plan created",
        strategy=strategy,
        tool_count=tool_count,
        requires_approval=requires_approval,
        duration_ms=duration_ms,
        request_id=request_id,
        user_id=user_id,
        tenant_id=tenant_id,
        **kwargs
    )


def log_circuit_breaker(
    server_id: str,
    state: str,
    action: str,
    request_id: Optional[str] = None,
    user_id: Optional[str] = None,
    tenant_id: Optional[str] = None,
    **kwargs
) -> None:
    """Log circuit breaker activity"""
    logger = structlog.get_logger("rmcp.circuit_breaker")
    logger.info(
        "Circuit breaker activity",
        server_id=server_id,
        state=state,
        action=action,
        request_id=request_id,
        user_id=user_id,
        tenant_id=tenant_id,
        **kwargs
    )


def log_approval_request(
    action: str,
    resource_type: str,
    resource_id: str,
    status: str,
    request_id: Optional[str] = None,
    user_id: Optional[str] = None,
    tenant_id: Optional[str] = None,
    **kwargs
) -> None:
    """Log approval request activity"""
    logger = structlog.get_logger("rmcp.approval")
    logger.info(
        "Approval request",
        action=action,
        resource_type=resource_type,
        resource_id=resource_id,
        status=status,
        request_id=request_id,
        user_id=user_id,
        tenant_id=tenant_id,
        **kwargs
    )


def log_telemetry(
    event_type: str,
    tool_id: str,
    success: bool,
    request_id: Optional[str] = None,
    user_id: Optional[str] = None,
    tenant_id: Optional[str] = None,
    **kwargs
) -> None:
    """Log telemetry activity"""
    logger = structlog.get_logger("rmcp.telemetry")
    level = "info" if success else "warning"
    
    log_data = {
        "event_type": event_type,
        "tool_id": tool_id,
        "success": success,
        "request_id": request_id,
        "user_id": user_id,
        "tenant_id": tenant_id,
        **kwargs
    }
    
    getattr(logger, level)("Telemetry event", **log_data)


def log_security(
    action: str,
    user_id: Optional[str] = None,
    tenant_id: Optional[str] = None,
    success: bool = True,
    request_id: Optional[str] = None,
    **kwargs
) -> None:
    """Log security-related events"""
    logger = structlog.get_logger("rmcp.security")
    level = "info" if success else "warning"
    
    log_data = {
        "action": action,
        "user_id": user_id,
        "tenant_id": tenant_id,
        "success": success,
        "request_id": request_id,
        **kwargs
    }
    
    getattr(logger, level)("Security event", **log_data)


def log_system(
    component: str,
    event: str,
    level: str = "info",
    **kwargs
) -> None:
    """Log system-level events"""
    logger = structlog.get_logger("rmcp.system")
    
    log_data = {
        "component": component,
        **kwargs
    }
    
    getattr(logger, level)(event, **log_data)


def generate_request_id() -> str:
    """Generate a unique request ID"""
    return str(uuid.uuid4())


# Initialize logging on import (commented out to allow test configuration)
# configure_logging()
