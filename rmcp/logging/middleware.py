"""
FastAPI middleware for structured request logging
"""

import time
import uuid
from typing import Callable, Optional
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from .config import (
    log_request_start,
    log_request_end,
    generate_request_id,
    get_logger
)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging HTTP requests with structured logging"""
    
    def __init__(
        self,
        app: ASGIApp,
        log_requests: bool = True,
        log_responses: bool = True,
        exclude_paths: Optional[list] = None
    ):
        super().__init__(app)
        self.log_requests = log_requests
        self.log_responses = log_responses
        self.exclude_paths = exclude_paths or ["/health", "/metrics"]
        self.logger = get_logger("rmcp.middleware")
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and response with logging"""
        
        # Skip logging for excluded paths
        if request.url.path in self.exclude_paths:
            return await call_next(request)
        
        # Generate request ID
        request_id = generate_request_id()
        
        # Add request ID to request state
        request.state.request_id = request_id
        
        # Extract user context from headers or auth
        user_id = self._extract_user_id(request)
        tenant_id = self._extract_tenant_id(request)
        
        # Log request start
        if self.log_requests:
            log_request_start(
                method=request.method,
                path=request.url.path,
                request_id=request_id,
                user_id=user_id,
                tenant_id=tenant_id,
                query_params=dict(request.query_params),
                user_agent=request.headers.get("user-agent"),
                content_length=request.headers.get("content-length"),
                content_type=request.headers.get("content-type")
            )
        
        # Record start time
        start_time = time.time()
        
        # Process request
        try:
            response = await call_next(request)
            
            # Calculate duration
            duration_ms = (time.time() - start_time) * 1000
            
            # Log successful response
            if self.log_responses:
                log_request_end(
                    method=request.method,
                    path=request.url.path,
                    request_id=request_id,
                    status_code=response.status_code,
                    duration_ms=duration_ms,
                    user_id=user_id,
                    tenant_id=tenant_id,
                    response_size=response.headers.get("content-length")
                )
            
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            
            return response
            
        except Exception as e:
            # Calculate duration for failed requests
            duration_ms = (time.time() - start_time) * 1000
            
            # Log failed request
            self.logger.error(
                "Request failed",
                method=request.method,
                path=request.url.path,
                request_id=request_id,
                duration_ms=duration_ms,
                user_id=user_id,
                tenant_id=tenant_id,
                error=str(e),
                error_type=type(e).__name__
            )
            
            # Re-raise the exception
            raise
    
    def _extract_user_id(self, request: Request) -> Optional[str]:
        """Extract user ID from request"""
        # Try to get from Authorization header
        auth_header = request.headers.get("authorization")
        if auth_header and auth_header.startswith("Bearer "):
            # In a real implementation, you would decode the JWT token
            # For now, we'll just return None
            pass
        
        # Try to get from X-User-ID header
        return request.headers.get("x-user-id")
    
    def _extract_tenant_id(self, request: Request) -> Optional[str]:
        """Extract tenant ID from request"""
        # Try to get from X-Tenant-ID header
        return request.headers.get("x-tenant-id")


class HealthCheckMiddleware(BaseHTTPMiddleware):
    """Middleware for health check endpoints"""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.logger = get_logger("rmcp.health")
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process health check requests"""
        
        if request.url.path == "/health":
            start_time = time.time()
            response = await call_next(request)
            duration_ms = (time.time() - start_time) * 1000
            
            # Log health check (but don't include full request context)
            self.logger.debug(
                "Health check",
                status_code=response.status_code,
                duration_ms=duration_ms
            )
            
            return response
        
        return await call_next(request)


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware for metrics endpoints"""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.logger = get_logger("rmcp.metrics")
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process metrics requests"""
        
        if request.url.path == "/metrics":
            start_time = time.time()
            response = await call_next(request)
            duration_ms = (time.time() - start_time) * 1000
            
            # Log metrics request (but don't include full request context)
            self.logger.debug(
                "Metrics request",
                status_code=response.status_code,
                duration_ms=duration_ms
            )
            
            return response
        
        return await call_next(request)


class SecurityLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for security-related logging"""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.logger = get_logger("rmcp.security")
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process requests with security logging"""
        
        # Log authentication attempts
        if request.url.path.startswith("/api/v1/auth"):
            self.logger.info(
                "Authentication attempt",
                method=request.method,
                path=request.url.path,
                user_agent=request.headers.get("user-agent"),
                ip_address=request.client.host if request.client else None
            )
        
        # Log API key usage
        api_key = request.headers.get("x-api-key")
        if api_key:
            self.logger.info(
                "API key usage",
                method=request.method,
                path=request.url.path,
                api_key_prefix=api_key[:8] + "..." if len(api_key) > 8 else api_key,
                ip_address=request.client.host if request.client else None
            )
        
        response = await call_next(request)
        
        # Log security-related responses
        if response.status_code in [401, 403, 429]:
            self.logger.warning(
                "Security response",
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                ip_address=request.client.host if request.client else None
            )
        
        return response


class ErrorLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for error logging"""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.logger = get_logger("rmcp.error")
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process requests with error logging"""
        
        try:
            response = await call_next(request)
            
            # Log client errors (4xx)
            if 400 <= response.status_code < 500:
                self.logger.warning(
                    "Client error",
                    method=request.method,
                    path=request.url.path,
                    status_code=response.status_code,
                    request_id=getattr(request.state, 'request_id', None)
                )
            
            # Log server errors (5xx)
            elif response.status_code >= 500:
                self.logger.error(
                    "Server error",
                    method=request.method,
                    path=request.url.path,
                    status_code=response.status_code,
                    request_id=getattr(request.state, 'request_id', None)
                )
            
            return response
            
        except Exception as e:
            # Log unhandled exceptions
            self.logger.error(
                "Unhandled exception",
                method=request.method,
                path=request.url.path,
                error=str(e),
                error_type=type(e).__name__,
                request_id=getattr(request.state, 'request_id', None)
            )
            
            # Re-raise the exception
            raise

