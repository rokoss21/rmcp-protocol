"""
Telemetry Engine for RMCP
Handles asynchronous processing of telemetry events and continuous learning
"""

import asyncio
import time
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from ..storage.database import DatabaseManager
from ..embeddings.manager import EmbeddingManager
from ..embeddings.store import EmbeddingStore


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
    priority: int = 1  # 1=high, 2=medium, 3=low


class TelemetryEngine:
    """
    Telemetry Engine for RMCP
    
    Handles:
    - Asynchronous telemetry event processing
    - Continuous learning from execution results
    - Background metrics updates
    - Semantic affinity updates
    """
    
    def __init__(
        self, 
        db_manager: DatabaseManager,
        embedding_manager: Optional[EmbeddingManager] = None,
        max_queue_size: int = 10000,
        batch_size: int = 100,
        processing_interval: float = 5.0
    ):
        self.db_manager = db_manager
        self.embedding_manager = embedding_manager
        self.embedding_store = EmbeddingStore(embedding_manager) if embedding_manager else None
        
        # Queue management
        self.max_queue_size = max_queue_size
        self.batch_size = batch_size
        self.processing_interval = processing_interval
        
        # Event queues (priority-based) - initialize lazily to avoid event loop issues
        self.high_priority_queue: Optional[asyncio.Queue] = None
        self.medium_priority_queue: Optional[asyncio.Queue] = None
        self.low_priority_queue: Optional[asyncio.Queue] = None
        
        # Processing state
        self.is_running = False
        self.processing_task: Optional[asyncio.Task] = None
        self.stats = {
            "events_processed": 0,
            "events_dropped": 0,
            "last_processing_time": None,
            "queue_sizes": {"high": 0, "medium": 0, "low": 0}
        }
    
    def _initialize_queues(self):
        """Initialize asyncio queues if not already done"""
        if self.high_priority_queue is None:
            self.high_priority_queue = asyncio.Queue(maxsize=self.max_queue_size // 3)
            self.medium_priority_queue = asyncio.Queue(maxsize=self.max_queue_size // 3)
            self.low_priority_queue = asyncio.Queue(maxsize=self.max_queue_size // 3)
    
    async def start(self) -> None:
        """Start the telemetry engine"""
        if self.is_running:
            return
        
        # Initialize queues
        self._initialize_queues()
        
        self.is_running = True
        self.processing_task = asyncio.create_task(self._processing_loop())
        print("TelemetryEngine: Started")
    
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
        
        print("TelemetryEngine: Stopped")
    
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
            priority: Event priority
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
        metrics: Dict[str, Any],
        priority: int = 2
    ) -> None:
        """
        Record metrics update event
        
        Args:
            tool_id: Tool identifier
            metrics: Metrics data to update
            priority: Event priority
        """
        event = TelemetryEvent(
            event_type=TelemetryEventType.METRICS_UPDATE,
            tool_id=tool_id,
            timestamp=datetime.utcnow(),
            data=metrics,
            priority=priority
        )
        
        await self._enqueue_event(event)
    
    async def _enqueue_event(self, event: TelemetryEvent) -> None:
        """
        Enqueue telemetry event based on priority
        
        Args:
            event: Telemetry event to enqueue
        """
        # Initialize queues if not done
        self._initialize_queues()
        
        try:
            if event.priority == 1:
                await self.high_priority_queue.put(event)
            elif event.priority == 2:
                await self.medium_priority_queue.put(event)
            else:
                await self.low_priority_queue.put(event)
        except asyncio.QueueFull:
            self.stats["events_dropped"] += 1
            print(f"TelemetryEngine: Queue full, dropped event for tool {event.tool_id}")
    
    async def _processing_loop(self) -> None:
        """Main processing loop for telemetry events"""
        while self.is_running:
            try:
                # Process events from all queues
                await self._process_events()
                
                # Update statistics
                self._update_stats()
                
                # Wait before next processing cycle
                await asyncio.sleep(self.processing_interval)
                
            except Exception as e:
                print(f"TelemetryEngine: Error in processing loop: {e}")
                await asyncio.sleep(self.processing_interval)
    
    async def _process_events(self) -> None:
        """Process events from all priority queues"""
        # Initialize queues if not done
        self._initialize_queues()
        
        start_time = time.time()
        processed_count = 0
        
        # Process high priority events first
        processed_count += await self._process_queue(self.high_priority_queue, "high")
        
        # Process medium priority events
        processed_count += await self._process_queue(self.medium_priority_queue, "medium")
        
        # Process low priority events
        processed_count += await self._process_queue(self.low_priority_queue, "low")
        
        # Update processing statistics
        processing_time = time.time() - start_time
        self.stats["events_processed"] += processed_count
        self.stats["last_processing_time"] = processing_time
        
        if processed_count > 0:
            print(f"TelemetryEngine: Processed {processed_count} events in {processing_time:.3f}s")
    
    async def _process_queue(self, queue: asyncio.Queue, queue_name: str) -> int:
        """
        Process events from a specific queue
        
        Args:
            queue: Queue to process
            queue_name: Name of the queue for logging
            
        Returns:
            Number of events processed
        """
        processed_count = 0
        batch = []
        
        # Collect batch of events
        while len(batch) < self.batch_size:
            try:
                event = queue.get_nowait()
                batch.append(event)
            except asyncio.QueueEmpty:
                break
        
        # Process batch
        for event in batch:
            try:
                await self._process_event(event)
                processed_count += 1
            except Exception as e:
                print(f"TelemetryEngine: Error processing {queue_name} priority event: {e}")
        
        return processed_count
    
    async def _process_event(self, event: TelemetryEvent) -> None:
        """
        Process individual telemetry event
        
        Args:
            event: Telemetry event to process
        """
        if event.event_type == TelemetryEventType.TOOL_EXECUTION:
            await self._process_tool_execution_event(event)
        elif event.event_type == TelemetryEventType.AFFINITY_UPDATE:
            await self._process_affinity_update_event(event)
        elif event.event_type == TelemetryEventType.METRICS_UPDATE:
            await self._process_metrics_update_event(event)
        elif event.event_type == TelemetryEventType.SYSTEM_HEALTH:
            await self._process_system_health_event(event)
    
    async def _process_tool_execution_event(self, event: TelemetryEvent) -> None:
        """Process tool execution telemetry event"""
        data = event.data
        
        # Update tool metrics in database
        self.db_manager.update_tool_metrics(
            event.tool_id,
            data["success"],
            data["latency_ms"],
            data["cost"]
        )
        
        # Add to telemetry queue for background processing
        self.db_manager.add_telemetry_event(
            event.tool_id,
            data["success"],
            data["latency_ms"],
            data["cost"]
        )
        
        # Update affinity if request text is available and execution was successful
        if data.get("request_text") and data["success"] and self.embedding_store:
            await self._update_tool_affinity(event.tool_id, data["request_text"])
    
    async def _process_affinity_update_event(self, event: TelemetryEvent) -> None:
        """Process affinity update telemetry event"""
        if not self.embedding_store:
            return
        
        data = event.data
        if data["success"]:
            await self._update_tool_affinity(event.tool_id, data["request_text"])
    
    async def _process_metrics_update_event(self, event: TelemetryEvent) -> None:
        """Process metrics update telemetry event"""
        # This would be used for more complex metrics updates
        # For now, we rely on the tool execution events
        pass
    
    async def _process_system_health_event(self, event: TelemetryEvent) -> None:
        """Process system health telemetry event"""
        # This would be used for system monitoring
        pass
    
    async def _update_tool_affinity(self, tool_id: str, request_text: str) -> None:
        """
        Update tool's affinity embeddings
        
        Args:
            tool_id: Tool identifier
            request_text: Text of the successful request
        """
        try:
            # Get current embeddings
            embeddings_data, embedding_count = self.db_manager.get_tool_embeddings(tool_id)
            
            # Deserialize current embeddings
            current_embeddings = []
            if embeddings_data:
                current_embeddings = self.embedding_store.deserialize_embeddings(embeddings_data)
            
            # Add new successful embedding
            updated_embeddings = await self.embedding_store.add_successful_embedding(
                tool_id, 
                request_text, 
                current_embeddings
            )
            
            # Serialize and store updated embeddings
            if updated_embeddings != current_embeddings:
                new_embeddings_data = self.embedding_store.serialize_embeddings(updated_embeddings)
                new_embedding_count = len(updated_embeddings)
                
                self.db_manager.update_tool_embeddings(tool_id, new_embeddings_data, new_embedding_count)
                
        except Exception as e:
            print(f"TelemetryEngine: Error updating affinity for tool {tool_id}: {e}")
    
    def _update_stats(self) -> None:
        """Update engine statistics"""
        # Initialize queues if not done
        self._initialize_queues()
        
        self.stats["queue_sizes"] = {
            "high": self.high_priority_queue.qsize(),
            "medium": self.medium_priority_queue.qsize(),
            "low": self.low_priority_queue.qsize()
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get telemetry engine statistics
        
        Returns:
            Dictionary with engine statistics
        """
        return {
            **self.stats,
            "is_running": self.is_running,
            "max_queue_size": self.max_queue_size,
            "batch_size": self.batch_size,
            "processing_interval": self.processing_interval,
            "embedding_manager_available": self.embedding_manager is not None
        }
    
    async def flush_queues(self) -> None:
        """Flush all pending events (for shutdown)"""
        print("TelemetryEngine: Flushing pending events...")
        
        # Initialize queues if not done
        self._initialize_queues()
        
        # Process remaining events
        while (self.high_priority_queue.qsize() > 0 or 
               self.medium_priority_queue.qsize() > 0 or 
               self.low_priority_queue.qsize() > 0):
            await self._process_events()
            await asyncio.sleep(0.1)
        
        print("TelemetryEngine: All events flushed")

