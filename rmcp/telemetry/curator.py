"""
Background Curator for RMCP
Implements background processing for metrics updates and continuous learning
"""

import asyncio
import time
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import sqlite3

from ..storage.database import DatabaseManager
from ..embeddings.manager import EmbeddingManager
from ..embeddings.store import EmbeddingStore
from .metrics import MetricsAggregator


class BackgroundCurator:
    """
    Background Curator for RMCP
    
    Performs background processing:
    - Processes telemetry queue
    - Updates tool metrics using EMA and P-Square
    - Updates semantic affinity embeddings
    - Maintains system health
    """
    
    def __init__(
        self, 
        db_manager: DatabaseManager,
        embedding_manager: Optional[EmbeddingManager] = None,
        processing_interval: float = 30.0,
        batch_size: int = 100,
        max_processing_time: float = 300.0  # 5 minutes max
    ):
        self.db_manager = db_manager
        self.embedding_manager = embedding_manager
        self.embedding_store = EmbeddingStore(embedding_manager) if embedding_manager else None
        
        # Processing configuration
        self.processing_interval = processing_interval
        self.batch_size = batch_size
        self.max_processing_time = max_processing_time
        
        # Processing state
        self.is_running = False
        self.curator_task: Optional[asyncio.Task] = None
        
        # Metrics aggregators for each tool
        self.tool_aggregators: Dict[str, MetricsAggregator] = {}
        
        # Statistics
        self.stats = {
            "processing_cycles": 0,
            "events_processed": 0,
            "tools_updated": 0,
            "last_processing_time": None,
            "last_successful_cycle": None,
            "errors": 0
        }
    
    async def start(self) -> None:
        """Start the background curator"""
        if self.is_running:
            return
        
        self.is_running = True
        self.curator_task = asyncio.create_task(self._curation_loop())
        print("BackgroundCurator: Started")
    
    async def stop(self) -> None:
        """Stop the background curator"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        if self.curator_task:
            self.curator_task.cancel()
            try:
                await self.curator_task
            except asyncio.CancelledError:
                pass
        
        print("BackgroundCurator: Stopped")
    
    async def _curation_loop(self) -> None:
        """Main curation loop"""
        while self.is_running:
            try:
                start_time = time.time()
                
                # Process telemetry queue
                await self._process_telemetry_queue()
                
                # Update tool metrics
                await self._update_tool_metrics()
                
                # Update semantic affinity
                if self.embedding_store:
                    await self._update_semantic_affinity()
                
                # Cleanup old data
                await self._cleanup_old_data()
                
                # Update statistics
                processing_time = time.time() - start_time
                self._update_stats(processing_time)
                
                # Wait before next cycle
                await asyncio.sleep(self.processing_interval)
                
            except Exception as e:
                print(f"BackgroundCurator: Error in curation loop: {e}")
                self.stats["errors"] += 1
                await asyncio.sleep(self.processing_interval)
    
    async def _process_telemetry_queue(self) -> None:
        """Process events from telemetry queue"""
        try:
            from ..storage.schema import get_connection
            with get_connection(self.db_manager.db_path) as conn:
                cursor = conn.cursor()
                
                # Get batch of telemetry events
                cursor.execute("""
                    SELECT tool_id, success, latency_ms, cost, request_embedding, timestamp
                    FROM telemetry_queue
                    ORDER BY timestamp ASC
                    LIMIT ?
                """, (self.batch_size,))
                
                events = cursor.fetchall()
                
                if not events:
                    return
                
                # Process events
                for event in events:
                    tool_id, success, latency_ms, cost, request_embedding, timestamp = event
                    
                    # Get or create metrics aggregator for this tool
                    if tool_id not in self.tool_aggregators:
                        self.tool_aggregators[tool_id] = MetricsAggregator()
                    
                    # Update metrics
                    aggregator = self.tool_aggregators[tool_id]
                    metrics = aggregator.update(
                        success=bool(success),
                        latency_ms=latency_ms,
                        cost=cost,
                        timestamp=timestamp
                    )
                    
                    self.stats["events_processed"] += 1
                
                # Remove processed events
                if events:
                    event_ids = [event[0] for event in events]  # Using tool_id as identifier
                    cursor.execute("""
                        DELETE FROM telemetry_queue
                        WHERE tool_id IN ({})
                    """.format(','.join('?' * len(event_ids))), event_ids)
                    
                    conn.commit()
                
        except Exception as e:
            print(f"BackgroundCurator: Error processing telemetry queue: {e}")
            self.stats["errors"] += 1
    
    async def _update_tool_metrics(self) -> None:
        """Update tool metrics in database"""
        try:
            for tool_id, aggregator in self.tool_aggregators.items():
                metrics = aggregator.get_metrics()
                
                if metrics["sample_count"] > 0:
                    # Update tool metrics in database
                    self.db_manager.update_tool_metrics(
                        tool_id=tool_id,
                        success=True,  # This is just for the update call
                        latency_ms=int(metrics["latency_p95_ms"] or 3000),
                        cost=metrics["cost_avg"] or 0.0
                    )
                    
                    # Update success rate separately (this would need a new method)
                    await self._update_tool_success_rate(tool_id, metrics["success_rate"])
                    
                    self.stats["tools_updated"] += 1
                
        except Exception as e:
            print(f"BackgroundCurator: Error updating tool metrics: {e}")
            self.stats["errors"] += 1
    
    async def _update_tool_success_rate(self, tool_id: str, success_rate: float) -> None:
        """Update tool success rate in database"""
        try:
            from ..storage.schema import get_connection
            with get_connection(self.db_manager.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE tools 
                    SET success_rate = ?
                    WHERE id = ?
                """, (success_rate, tool_id))
                conn.commit()
        except Exception as e:
            print(f"BackgroundCurator: Error updating success rate for {tool_id}: {e}")
    
    async def _update_semantic_affinity(self) -> None:
        """Update semantic affinity embeddings"""
        if not self.embedding_store:
            return
        
        try:
            # Get tools that have recent successful executions
            from ..storage.schema import get_connection
            with get_connection(self.db_manager.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT DISTINCT tool_id, success, latency_ms, cost, request_embedding
                    FROM telemetry_queue
                    WHERE success = 1 AND request_embedding IS NOT NULL
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (self.batch_size,))
                
                affinity_events = cursor.fetchall()
                
                for event in affinity_events:
                    tool_id, success, latency_ms, cost, request_embedding = event
                    
                    if request_embedding:
                        # Deserialize request embedding (this would need proper deserialization)
                        # For now, we'll skip this complex part
                        pass
                
        except Exception as e:
            print(f"BackgroundCurator: Error updating semantic affinity: {e}")
            self.stats["errors"] += 1
    
    async def _cleanup_old_data(self) -> None:
        """Clean up old telemetry data"""
        try:
            # Keep only last 7 days of telemetry data
            cutoff_date = datetime.utcnow() - timedelta(days=7)
            cutoff_str = cutoff_date.isoformat()
            
            from ..storage.schema import get_connection
            with get_connection(self.db_manager.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    DELETE FROM telemetry_queue
                    WHERE timestamp < ?
                """, (cutoff_str,))
                
                deleted_count = cursor.rowcount
                conn.commit()
                
                if deleted_count > 0:
                    print(f"BackgroundCurator: Cleaned up {deleted_count} old telemetry events")
                
        except Exception as e:
            print(f"BackgroundCurator: Error cleaning up old data: {e}")
            self.stats["errors"] += 1
    
    def _update_stats(self, processing_time: float) -> None:
        """Update curator statistics"""
        self.stats["processing_cycles"] += 1
        self.stats["last_processing_time"] = processing_time
        self.stats["last_successful_cycle"] = datetime.utcnow().isoformat()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get curator statistics"""
        return {
            **self.stats,
            "is_running": self.is_running,
            "processing_interval": self.processing_interval,
            "batch_size": self.batch_size,
            "max_processing_time": self.max_processing_time,
            "active_tool_aggregators": len(self.tool_aggregators),
            "embedding_store_available": self.embedding_store is not None
        }
    
    def get_tool_metrics(self, tool_id: str) -> Optional[Dict[str, Any]]:
        """Get metrics for a specific tool"""
        if tool_id in self.tool_aggregators:
            return self.tool_aggregators[tool_id].get_metrics()
        return None
    
    def get_all_tool_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all tools"""
        return {
            tool_id: aggregator.get_metrics()
            for tool_id, aggregator in self.tool_aggregators.items()
        }
    
    async def force_update_tool(self, tool_id: str) -> None:
        """Force update metrics for a specific tool"""
        if tool_id in self.tool_aggregators:
            aggregator = self.tool_aggregators[tool_id]
            metrics = aggregator.get_metrics()
            
            if metrics["sample_count"] > 0:
                await self._update_tool_metrics()
                print(f"BackgroundCurator: Force updated metrics for tool {tool_id}")
    
    async def reset_tool_metrics(self, tool_id: str) -> None:
        """Reset metrics for a specific tool"""
        if tool_id in self.tool_aggregators:
            self.tool_aggregators[tool_id].reset()
            print(f"BackgroundCurator: Reset metrics for tool {tool_id}")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get curator health status"""
        is_healthy = (
            self.is_running and
            self.stats["errors"] < 10 and  # Less than 10 errors
            self.stats["last_successful_cycle"] is not None
        )
        
        return {
            "healthy": is_healthy,
            "is_running": self.is_running,
            "error_count": self.stats["errors"],
            "last_successful_cycle": self.stats["last_successful_cycle"],
            "processing_cycles": self.stats["processing_cycles"],
            "active_aggregators": len(self.tool_aggregators)
        }

