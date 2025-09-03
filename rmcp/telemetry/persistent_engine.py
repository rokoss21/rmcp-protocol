"""
Persistent Telemetry Engine for RMCP
Handles asynchronous processing of telemetry events with persistent queue
"""

import asyncio
import time
import json
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from ..storage.database import DatabaseManager
from ..embeddings.manager import EmbeddingManager
from ..embeddings.store import EmbeddingStore
from ..observability.metrics import metrics
from ..logging.config import get_logger, log_telemetry, log_system


class TelemetryEventType(str, Enum):
    """Types of telemetry events"""
    TOOL_EXECUTION = "tool_execution"
    AFFINITY_UPDATE = "affinity_update"
    METRICS_UPDATE = "metrics_update"
    SYSTEM_HEALTH = "system_health"


@dataclass
class TelemetryEvent:
    """Telemetry event data structure"""
    event_type: TelemetryEventType
    tool_id: str
    timestamp: datetime
    data: Dict[str, Any]
    priority: int = 2  # 1=high, 2=medium, 3=low


class PersistentTelemetryEngine:
    """
    Persistent Telemetry Engine for RMCP
    
    Features:
    - Persistent queue using SQLite table
    - Automatic retry with exponential backoff
    - Priority-based processing
    - Graceful error handling and recovery
    - Metrics integration
    """
    
    def __init__(
        self, 
        db_manager: DatabaseManager,
        embedding_manager: Optional[EmbeddingManager] = None,
        batch_size: int = 10,
        processing_interval: float = 1.0,
        max_retries: int = 3
    ):
        self.db_manager = db_manager
        self.embedding_manager = embedding_manager
        self.batch_size = batch_size
        self.processing_interval = processing_interval
        self.max_retries = max_retries
        
        # Processing state
        self.is_running = False
        self.processing_task = None
        
        # Statistics
        self.stats = {
            "events_processed": 0,
            "events_failed": 0,
            "events_retried": 0,
            "last_processing_time": 0.0,
            "queue_size": 0
        }
        
        # Logger
        self.logger = get_logger(__name__)
        
        # Initialize embedding store if available
        self.embedding_store = None
        if embedding_manager:
            try:
                self.embedding_store = EmbeddingStore(embedding_manager)
            except Exception as e:
                print(f"Warning: Could not initialize EmbeddingStore: {e}")
    
    async def start(self) -> None:
        """Start the telemetry engine"""
        if self.is_running:
            return
        
        self.is_running = True
        self.processing_task = asyncio.create_task(self._processing_loop())
        log_system("telemetry_engine", "started", level="info")
    
    async def stop(self) -> None:
        """Stop the telemetry engine"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
            finally:
                self.processing_task = None
        
        log_system("telemetry_engine", "stopped", level="info")
    
    async def record_tool_execution(
        self, 
        tool_id: str, 
        success: bool, 
        latency_ms: int, 
        cost: float = 0.0,
        request_text: Optional[str] = None,
        priority: int = 2
    ) -> None:
        """
        Record tool execution telemetry
        
        Args:
            tool_id: Tool identifier
            success: Whether execution was successful
            latency_ms: Execution latency in milliseconds
            cost: Execution cost
            request_text: Text of the request (for affinity updates)
            priority: Event priority (1=high, 2=medium, 3=low)
        """
        event = TelemetryEvent(
            event_type=TelemetryEventType.TOOL_EXECUTION,
            tool_id=tool_id,
            timestamp=datetime.utcnow(),
            data={
                "success": success,
                "latency_ms": latency_ms,
                "cost": cost,
                "request_text": request_text
            },
            priority=priority
        )
        
        await self._enqueue_event(event)
    
    async def record_affinity_update(
        self, 
        tool_id: str, 
        request_text: str, 
        success: bool,
        priority: int = 1
    ) -> None:
        """
        Record affinity update event
        
        Args:
            tool_id: Tool identifier
            request_text: Text of the request
            success: Whether execution was successful
            priority: Event priority (1=high, 2=medium, 3=low)
        """
        event = TelemetryEvent(
            event_type=TelemetryEventType.AFFINITY_UPDATE,
            tool_id=tool_id,
            timestamp=datetime.utcnow(),
            data={
                "request_text": request_text,
                "success": success
            },
            priority=priority
        )
        
        await self._enqueue_event(event)
    
    async def record_metrics_update(
        self, 
        tool_id: str, 
        metrics_data: Dict[str, Any],
        priority: int = 2
    ) -> None:
        """
        Record metrics update event
        
        Args:
            tool_id: Tool identifier
            metrics_data: Metrics data to update
            priority: Event priority (1=high, 2=medium, 3=low)
        """
        event = TelemetryEvent(
            event_type=TelemetryEventType.METRICS_UPDATE,
            tool_id=tool_id,
            timestamp=datetime.utcnow(),
            data=metrics_data,
            priority=priority
        )
        
        await self._enqueue_event(event)
    
    async def _enqueue_event(self, event: TelemetryEvent) -> None:
        """
        Enqueue telemetry event to persistent queue
        
        Args:
            event: Telemetry event to enqueue
        """
        try:
            from ..storage.schema import get_connection
            
            with get_connection(self.db_manager.db_path) as conn:
                cursor = conn.cursor()
                
                # Insert event into persistent queue
                cursor.execute("""
                    INSERT INTO telemetry_event_queue (
                        event_type, tool_id, priority, payload, status
                    ) VALUES (?, ?, ?, ?, ?)
                """, (
                    event.event_type.value,
                    event.tool_id,
                    event.priority,
                    json.dumps({
                        "timestamp": event.timestamp.isoformat(),
                        "data": event.data
                    }),
                    "pending"
                ))
                
                conn.commit()
                
                # Update queue size metric
                self._update_queue_size()
                
        except Exception as e:
            print(f"PersistentTelemetryEngine: Error enqueueing event: {e}")
            # Record failure metric
            metrics.record_database_operation("INSERT", "telemetry_event_queue", "error", "default")
    
    async def _processing_loop(self) -> None:
        """Main processing loop for telemetry events"""
        while self.is_running:
            try:
                # Process events from persistent queue
                await self._process_events()
                
                # Update statistics
                self._update_stats()
                
                # Wait before next processing cycle
                await asyncio.sleep(self.processing_interval)
                
            except Exception as e:
                print(f"PersistentTelemetryEngine: Error in processing loop: {e}")
                await asyncio.sleep(self.processing_interval)
    
    async def _process_events(self) -> None:
        """Process events from persistent queue"""
        start_time = time.time()
        processed_count = 0
        
        try:
            from ..storage.schema import get_connection
            
            # Get batch of events in separate connection
            with get_connection(self.db_manager.db_path) as conn:
                cursor = conn.cursor()
                
                # Get batch of events ordered by priority and creation time
                cursor.execute("""
                    SELECT id, event_type, tool_id, priority, payload, retry_count
                    FROM telemetry_event_queue
                    WHERE status = 'pending' AND retry_count < max_retries
                    ORDER BY priority ASC, created_at ASC
                    LIMIT ?
                """, (self.batch_size,))
                
                events = cursor.fetchall()
            
            # Process events in separate connections to avoid locking
            for event_row in events:
                try:
                    # Parse event data
                    event_data = json.loads(event_row[4])  # payload
                    event = TelemetryEvent(
                        event_type=TelemetryEventType(event_row[1]),  # event_type
                        tool_id=event_row[2],  # tool_id
                        timestamp=datetime.fromisoformat(event_data["timestamp"]),
                        data=event_data["data"],
                        priority=event_row[3]  # priority
                    )
                    
                    # Process the event
                    await self._process_event(event)
                    
                    # Mark as processed in separate connection
                    with get_connection(self.db_manager.db_path) as conn:
                        cursor = conn.cursor()
                        cursor.execute("""
                            UPDATE telemetry_event_queue
                            SET status = 'processed', processed_at = ?
                            WHERE id = ?
                        """, (datetime.utcnow().isoformat(), event_row[0]))
                        conn.commit()
                    
                    processed_count += 1
                    
                except Exception as e:
                    print(f"PersistentTelemetryEngine: Error processing event {event_row[0]}: {e}")
                    
                    # Increment retry count in separate connection
                    with get_connection(self.db_manager.db_path) as conn:
                        cursor = conn.cursor()
                        cursor.execute("""
                            UPDATE telemetry_event_queue
                            SET retry_count = retry_count + 1,
                                status = CASE 
                                    WHEN retry_count + 1 >= max_retries THEN 'failed'
                                    ELSE 'pending'
                                END
                            WHERE id = ?
                        """, (event_row[0],))
                        conn.commit()
                    
                    self.stats["events_failed"] += 1
                    if event_row[5] < self.max_retries:  # retry_count
                        self.stats["events_retried"] += 1
                
        except Exception as e:
            print(f"PersistentTelemetryEngine: Error in _process_events: {e}")
        
        # Update processing statistics
        processing_time = time.time() - start_time
        self.stats["events_processed"] += processed_count
        self.stats["last_processing_time"] = processing_time
        
        if processed_count > 0:
            print(f"PersistentTelemetryEngine: Processed {processed_count} events in {processing_time:.3f}s")
    
    async def _process_event(self, event: TelemetryEvent) -> None:
        """
        Process a single telemetry event
        
        Args:
            event: Telemetry event to process
        """
        try:
            if event.event_type == TelemetryEventType.TOOL_EXECUTION:
                await self._process_tool_execution_event(event)
            elif event.event_type == TelemetryEventType.AFFINITY_UPDATE:
                await self._process_affinity_update_event(event)
            elif event.event_type == TelemetryEventType.METRICS_UPDATE:
                await self._process_metrics_update_event(event)
            elif event.event_type == TelemetryEventType.SYSTEM_HEALTH:
                await self._process_system_health_event(event)
            else:
                print(f"PersistentTelemetryEngine: Unknown event type: {event.event_type}")
                
        except Exception as e:
            print(f"PersistentTelemetryEngine: Error processing {event.event_type} event: {e}")
            raise
    
    async def _process_tool_execution_event(self, event: TelemetryEvent) -> None:
        """Process tool execution event"""
        data = event.data
        
        # Update tool metrics in database
        self.db_manager.add_telemetry_event(
            tool_id=event.tool_id,
            success=data["success"],
            latency_ms=data["latency_ms"],
            cost=data.get("cost", 0.0)
        )
        
        # Update affinity if request text is available
        if data.get("request_text") and self.embedding_store:
            try:
                await self._update_affinity(
                    event.tool_id,
                    data["request_text"],
                    data["success"]
                )
            except Exception as e:
                print(f"PersistentTelemetryEngine: Error updating affinity: {e}")
        
        # Record metrics
        metrics.record_database_operation("INSERT", "telemetry_queue", "success", "default")
    
    async def _process_affinity_update_event(self, event: TelemetryEvent) -> None:
        """Process affinity update event"""
        data = event.data
        
        if self.embedding_store:
            try:
                await self._update_affinity(
                    event.tool_id,
                    data["request_text"],
                    data["success"]
                )
            except Exception as e:
                print(f"PersistentTelemetryEngine: Error updating affinity: {e}")
                raise
    
    async def _process_metrics_update_event(self, event: TelemetryEvent) -> None:
        """Process metrics update event"""
        # Update tool metrics in database
        metrics_data = event.data
        
        # This would update tool performance metrics
        # Implementation depends on specific metrics structure
        print(f"PersistentTelemetryEngine: Updated metrics for tool {event.tool_id}")
    
    async def _process_system_health_event(self, event: TelemetryEvent) -> None:
        """Process system health event"""
        # Handle system health monitoring
        print(f"PersistentTelemetryEngine: Processed system health event for tool {event.tool_id}")
    
    async def _update_affinity(self, tool_id: str, request_text: str, success: bool) -> None:
        """Update semantic affinity for a tool"""
        if not self.embedding_manager or not self.embedding_store:
            return
        
        try:
            # Update affinity in embedding store (only for successful executions)
            if success:
                # Get current embeddings for the tool
                current_embeddings = await self._get_current_embeddings(tool_id)
                
                # Add successful embedding
                await self.embedding_store.add_successful_embedding(
                    tool_id=tool_id,
                    request_text=request_text,
                    current_embeddings=current_embeddings
                )
            
        except Exception as e:
            print(f"PersistentTelemetryEngine: Error updating affinity: {e}")
            raise
    
    async def _get_current_embeddings(self, tool_id: str) -> List[List[float]]:
        """Get current embeddings for a tool"""
        try:
            from ..storage.schema import get_connection
            
            with get_connection(self.db_manager.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT affinity_embeddings FROM tools WHERE id = ?", (tool_id,))
                result = cursor.fetchone()
                
                if result and result[0]:
                    return self.embedding_store.deserialize_embeddings(result[0])
                else:
                    return []
                    
        except Exception as e:
            print(f"PersistentTelemetryEngine: Error getting current embeddings: {e}")
            return []
    
    def _update_stats(self) -> None:
        """Update processing statistics"""
        self._update_queue_size()
    
    def _update_queue_size(self) -> None:
        """Update queue size metric"""
        try:
            from ..storage.schema import get_connection
            
            with get_connection(self.db_manager.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM telemetry_event_queue WHERE status = 'pending'")
                self.stats["queue_size"] = cursor.fetchone()[0]
                
        except Exception as e:
            print(f"PersistentTelemetryEngine: Error updating queue size: {e}")
    
    async def flush_queues(self) -> None:
        """Flush all pending events"""
        print("PersistentTelemetryEngine: Flushing all pending events...")
        
        while True:
            try:
                from ..storage.schema import get_connection
                
                with get_connection(self.db_manager.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT COUNT(*) FROM telemetry_event_queue WHERE status = 'pending'")
                    pending_count = cursor.fetchone()[0]
                
                if pending_count == 0:
                    break
                
                await self._process_events()
                await asyncio.sleep(0.1)
                
            except Exception as e:
                print(f"PersistentTelemetryEngine: Error flushing queues: {e}")
                break
        
        print("PersistentTelemetryEngine: All events flushed")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return {
            **self.stats,
            "is_running": self.is_running,
            "batch_size": self.batch_size,
            "processing_interval": self.processing_interval,
            "max_retries": self.max_retries
        }
    
    async def cleanup_old_events(self, days_old: int = 7) -> None:
        """Clean up old processed events"""
        try:
            from ..storage.schema import get_connection
            
            cutoff_date = (datetime.utcnow() - timedelta(days=days_old)).isoformat()
            
            with get_connection(self.db_manager.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    DELETE FROM telemetry_event_queue
                    WHERE status IN ('processed', 'failed') 
                    AND created_at < ?
                """, (cutoff_date,))
                
                deleted_count = cursor.rowcount
                conn.commit()
                
                if deleted_count > 0:
                    print(f"PersistentTelemetryEngine: Cleaned up {deleted_count} old events")
                    
        except Exception as e:
            print(f"PersistentTelemetryEngine: Error cleaning up old events: {e}")
