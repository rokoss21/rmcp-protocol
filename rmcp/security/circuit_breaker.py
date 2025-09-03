"""
Circuit Breaker pattern implementation for RMCP
"""

import time
from typing import Dict, Any, Optional, Callable
from enum import Enum
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from ..storage.database import DatabaseManager
from ..observability.metrics import metrics


class CircuitState(str, Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Circuit is open, requests fail fast
    HALF_OPEN = "half_open"  # Testing if service is back


class CircuitBreakerConfig(BaseModel):
    """Circuit breaker configuration"""
    failure_threshold: int = Field(default=5, description="Number of failures before opening")
    recovery_timeout: int = Field(default=60, description="Seconds before trying half-open")
    success_threshold: int = Field(default=3, description="Successes needed to close from half-open")
    timeout: int = Field(default=30, description="Request timeout in seconds")


class CircuitBreakerStats(BaseModel):
    """Circuit breaker statistics"""
    server_id: str = Field(..., description="Server identifier")
    state: CircuitState = Field(..., description="Current circuit state")
    failure_count: int = Field(default=0, description="Current failure count")
    success_count: int = Field(default=0, description="Current success count")
    last_failure_time: Optional[datetime] = Field(None, description="Last failure timestamp")
    last_success_time: Optional[datetime] = Field(None, description="Last success timestamp")
    total_requests: int = Field(default=0, description="Total requests made")
    total_failures: int = Field(default=0, description="Total failures")
    total_successes: int = Field(default=0, description="Total successes")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        use_enum_values = True


class CircuitBreaker:
    """Circuit breaker for individual servers"""
    
    def __init__(self, server_id: str, config: CircuitBreakerConfig):
        self.server_id = server_id
        self.config = config
        self.stats = CircuitBreakerStats(
            server_id=server_id,
            state=CircuitState.CLOSED
        )
    
    def can_execute(self) -> bool:
        """Check if requests can be executed"""
        if self.stats.state == CircuitState.CLOSED:
            return True
        
        if self.stats.state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            if self.stats.last_failure_time:
                time_since_failure = datetime.utcnow() - self.stats.last_failure_time
                if time_since_failure.total_seconds() >= self.config.recovery_timeout:
                    self.stats.state = CircuitState.HALF_OPEN
                    self.stats.success_count = 0
                    return True
            return False
        
        if self.stats.state == CircuitState.HALF_OPEN:
            return True
        
        return False
    
    def is_available(self) -> bool:
        """Check if the circuit breaker allows requests (alias for can_execute)"""
        return self.can_execute()
    
    def record_success(self):
        """Record a successful request"""
        self.stats.total_requests += 1
        self.stats.total_successes += 1
        self.stats.success_count += 1
        self.stats.last_success_time = datetime.utcnow()
        self.stats.updated_at = datetime.utcnow()
        
        if self.stats.state == CircuitState.HALF_OPEN:
            if self.stats.success_count >= self.config.success_threshold:
                self.stats.state = CircuitState.CLOSED
                self.stats.failure_count = 0
                self.stats.success_count = 0
    
    def record_failure(self):
        """Record a failed request"""
        self.stats.total_requests += 1
        self.stats.total_failures += 1
        self.stats.failure_count += 1
        self.stats.last_failure_time = datetime.utcnow()
        self.stats.updated_at = datetime.utcnow()
        
        if self.stats.state == CircuitState.CLOSED:
            if self.stats.failure_count >= self.config.failure_threshold:
                self.stats.state = CircuitState.OPEN
                self.stats.failure_count = 0
        
        elif self.stats.state == CircuitState.HALF_OPEN:
            self.stats.state = CircuitState.OPEN
            self.stats.failure_count = 0
            self.stats.success_count = 0
    
    def get_stats(self) -> CircuitBreakerStats:
        """Get current circuit breaker statistics"""
        return self.stats


class CircuitBreakerManager:
    """Manager for multiple circuit breakers"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._init_circuit_breaker_tables()
        self._load_circuit_breakers()
    
    def _init_circuit_breaker_tables(self):
        """Initialize circuit breaker database tables"""
        from ..storage.schema import get_connection
        
        with get_connection(self.db_manager.db_path) as conn:
            cursor = conn.cursor()
            
            # Circuit breaker stats table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS circuit_breaker_stats (
                    server_id TEXT PRIMARY KEY,
                    state TEXT NOT NULL DEFAULT 'closed',
                    failure_count INTEGER NOT NULL DEFAULT 0,
                    success_count INTEGER NOT NULL DEFAULT 0,
                    last_failure_time TEXT,
                    last_success_time TEXT,
                    total_requests INTEGER NOT NULL DEFAULT 0,
                    total_failures INTEGER NOT NULL DEFAULT 0,
                    total_successes INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
                )
            """)
            
            conn.commit()
    
    def _load_circuit_breakers(self):
        """Load circuit breaker stats from database"""
        from ..storage.schema import get_connection
        
        with get_connection(self.db_manager.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT server_id, state, failure_count, success_count,
                       last_failure_time, last_success_time, total_requests,
                       total_failures, total_successes, created_at, updated_at
                FROM circuit_breaker_stats
            """)
            
            for row in cursor.fetchall():
                server_id = row[0]
                config = CircuitBreakerConfig()  # Use default config
                circuit_breaker = CircuitBreaker(server_id, config)
                
                circuit_breaker.stats.state = CircuitState(row[1])
                circuit_breaker.stats.failure_count = row[2]
                circuit_breaker.stats.success_count = row[3]
                circuit_breaker.stats.last_failure_time = (
                    datetime.fromisoformat(row[4].replace('Z', '+00:00')) if row[4] else None
                )
                circuit_breaker.stats.last_success_time = (
                    datetime.fromisoformat(row[5].replace('Z', '+00:00')) if row[5] else None
                )
                circuit_breaker.stats.total_requests = row[6]
                circuit_breaker.stats.total_failures = row[7]
                circuit_breaker.stats.total_successes = row[8]
                circuit_breaker.stats.created_at = datetime.fromisoformat(row[9].replace('Z', '+00:00'))
                circuit_breaker.stats.updated_at = datetime.fromisoformat(row[10].replace('Z', '+00:00'))
                
                self.circuit_breakers[server_id] = circuit_breaker
    
    def get_circuit_breaker(self, server_id: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
        """Get or create a circuit breaker for a server"""
        if server_id not in self.circuit_breakers:
            if config is None:
                config = CircuitBreakerConfig()
            self.circuit_breakers[server_id] = CircuitBreaker(server_id, config)
        
        return self.circuit_breakers[server_id]
    
    def can_execute(self, server_id: str) -> bool:
        """Check if requests can be executed for a server"""
        circuit_breaker = self.get_circuit_breaker(server_id)
        return circuit_breaker.can_execute()
    
    def record_success(self, server_id: str, tenant_id: str = "default"):
        """Record a successful request for a server"""
        circuit_breaker = self.get_circuit_breaker(server_id)
        circuit_breaker.record_success()
        self._save_circuit_breaker_stats(circuit_breaker)
        
        # Record metrics
        metrics.record_circuit_breaker_success(server_id, tenant_id)
        state_value = circuit_breaker.stats.state.value if hasattr(circuit_breaker.stats.state, 'value') else str(circuit_breaker.stats.state)
        metrics.update_circuit_breaker_state(server_id, tenant_id, state_value)
    
    def record_failure(self, server_id: str, tenant_id: str = "default"):
        """Record a failed request for a server"""
        circuit_breaker = self.get_circuit_breaker(server_id)
        circuit_breaker.record_failure()
        self._save_circuit_breaker_stats(circuit_breaker)
        
        # Record metrics
        metrics.record_circuit_breaker_failure(server_id, tenant_id)
        state_value = circuit_breaker.stats.state.value if hasattr(circuit_breaker.stats.state, 'value') else str(circuit_breaker.stats.state)
        metrics.update_circuit_breaker_state(server_id, tenant_id, state_value)
    
    def _save_circuit_breaker_stats(self, circuit_breaker: CircuitBreaker):
        """Save circuit breaker stats to database"""
        from ..storage.schema import get_connection
        
        stats = circuit_breaker.stats
        
        with get_connection(self.db_manager.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO circuit_breaker_stats (
                    server_id, state, failure_count, success_count,
                    last_failure_time, last_success_time, total_requests,
                    total_failures, total_successes, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                stats.server_id,
                stats.state.value if hasattr(stats.state, 'value') else str(stats.state),
                stats.failure_count,
                stats.success_count,
                stats.last_failure_time.isoformat() if stats.last_failure_time else None,
                stats.last_success_time.isoformat() if stats.last_success_time else None,
                stats.total_requests,
                stats.total_failures,
                stats.total_successes,
                stats.created_at.isoformat(),
                stats.updated_at.isoformat()
            ))
            conn.commit()
    
    def get_all_stats(self) -> Dict[str, CircuitBreakerStats]:
        """Get statistics for all circuit breakers"""
        return {server_id: cb.get_stats() for server_id, cb in self.circuit_breakers.items()}
    
    def get_stats(self, server_id: str) -> Optional[CircuitBreakerStats]:
        """Get statistics for a specific server"""
        circuit_breaker = self.get_circuit_breaker(server_id)
        return circuit_breaker.get_stats()
    
    def is_server_available(self, server_id: str) -> bool:
        """Check if a server is available (circuit breaker is closed)"""
        circuit_breaker = self.get_circuit_breaker(server_id)
        return circuit_breaker.is_available()
    
    def reset_circuit_breaker(self, server_id: str):
        """Reset a circuit breaker to closed state"""
        if server_id in self.circuit_breakers:
            circuit_breaker = self.circuit_breakers[server_id]
            circuit_breaker.stats.state = CircuitState.CLOSED
            circuit_breaker.stats.failure_count = 0
            circuit_breaker.stats.success_count = 0
            circuit_breaker.stats.updated_at = datetime.utcnow()
            self._save_circuit_breaker_stats(circuit_breaker)
    
    def cleanup_old_stats(self, days: int = 30):
        """Clean up old circuit breaker statistics"""
        from ..storage.schema import get_connection
        
        with get_connection(self.db_manager.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                DELETE FROM circuit_breaker_stats 
                WHERE updated_at < datetime('now', '-{} days')
            """.format(days))
            deleted_count = cursor.rowcount
            conn.commit()
            
            return deleted_count
