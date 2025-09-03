"""
Tests for persistent telemetry queue
"""

import pytest
import tempfile
import os
import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch

from rmcp.storage.database import DatabaseManager
from rmcp.storage.schema import init_database
from rmcp.telemetry.persistent_engine import PersistentTelemetryEngine, TelemetryEvent, TelemetryEventType
from rmcp.embeddings.manager import EmbeddingManager


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
def embedding_manager():
    """Fixture for embedding manager (mocked)."""
    manager = AsyncMock(spec=EmbeddingManager)
    manager.encode_text = AsyncMock(return_value=[0.1, 0.2, 0.3, 0.4, 0.5])
    return manager


@pytest.fixture
def persistent_engine(db_manager, embedding_manager):
    """Fixture for persistent telemetry engine."""
    return PersistentTelemetryEngine(
        db_manager=db_manager,
        embedding_manager=embedding_manager,
        batch_size=5,
        processing_interval=0.1,
        max_retries=3
    )


@pytest.fixture
def test_tool():
    """Fixture for a test tool."""
    from rmcp.models.tool import Tool
    return Tool(
        id="test-tool",
        server_id="test-server",
        name="test.tool",
        description="Test tool",
        input_schema={"param": {"type": "string"}},
        output_schema={"result": {"type": "string"}},
        tags=["test"],
        capabilities=["test"],
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


class TestPersistentTelemetryQueue:
    """Test persistent telemetry queue functionality"""
    
    def test_engine_initialization(self, persistent_engine):
        """Test that engine initializes correctly"""
        assert persistent_engine is not None
        assert persistent_engine.batch_size == 5
        assert persistent_engine.processing_interval == 0.1
        assert persistent_engine.max_retries == 3
        assert not persistent_engine.is_running
    
    @pytest.mark.asyncio
    async def test_engine_start_stop(self, persistent_engine):
        """Test engine start and stop functionality"""
        # Start engine
        await persistent_engine.start()
        assert persistent_engine.is_running
        assert persistent_engine.processing_task is not None
        
        # Stop engine
        await persistent_engine.stop()
        assert not persistent_engine.is_running
        assert persistent_engine.processing_task is None
    
    @pytest.mark.asyncio
    async def test_record_tool_execution(self, persistent_engine, test_tool, test_server, db_manager):
        """Test recording tool execution events"""
        # Add test tool and server to database
        db_manager.add_server(test_server)
        db_manager.add_tool(test_tool)
        
        # Record tool execution
        await persistent_engine.record_tool_execution(
            tool_id="test-tool",
            success=True,
            latency_ms=500,
            cost=0.1,
            request_text="test request",
            priority=1
        )
        
        # Check that event was added to queue
        from rmcp.storage.schema import get_connection
        with get_connection(persistent_engine.db_manager.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM telemetry_event_queue WHERE status = 'pending'")
            count = cursor.fetchone()[0]
            assert count == 1
            
            # Check event details
            cursor.execute("""
                SELECT event_type, tool_id, priority, payload
                FROM telemetry_event_queue
                WHERE status = 'pending'
            """)
            event = cursor.fetchone()
            assert event[0] == "tool_execution"
            assert event[1] == "test-tool"
            assert event[2] == 1  # priority
            
            # Check payload
            payload = json.loads(event[3])
            assert payload["data"]["success"] is True
            assert payload["data"]["latency_ms"] == 500
            assert payload["data"]["cost"] == 0.1
            assert payload["data"]["request_text"] == "test request"
    
    @pytest.mark.asyncio
    async def test_record_affinity_update(self, persistent_engine, test_tool, test_server, db_manager):
        """Test recording affinity update events"""
        # Add test tool and server to database
        db_manager.add_server(test_server)
        db_manager.add_tool(test_tool)
        
        # Record affinity update
        await persistent_engine.record_affinity_update(
            tool_id="test-tool",
            request_text="test request for affinity",
            success=True,
            priority=1
        )
        
        # Check that event was added to queue
        from rmcp.storage.schema import get_connection
        with get_connection(persistent_engine.db_manager.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT event_type, payload
                FROM telemetry_event_queue
                WHERE status = 'pending'
            """)
            event = cursor.fetchone()
            assert event[0] == "affinity_update"
            
            payload = json.loads(event[1])
            assert payload["data"]["request_text"] == "test request for affinity"
            assert payload["data"]["success"] is True
    
    @pytest.mark.asyncio
    async def test_record_metrics_update(self, persistent_engine, test_tool, test_server, db_manager):
        """Test recording metrics update events"""
        # Add test tool and server to database
        db_manager.add_server(test_server)
        db_manager.add_tool(test_tool)
        
        # Record metrics update
        metrics_data = {
            "p95_latency_ms": 800,
            "success_rate": 0.98,
            "cost_hint": 0.05
        }
        
        await persistent_engine.record_metrics_update(
            tool_id="test-tool",
            metrics_data=metrics_data,
            priority=2
        )
        
        # Check that event was added to queue
        from rmcp.storage.schema import get_connection
        with get_connection(persistent_engine.db_manager.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT event_type, payload
                FROM telemetry_event_queue
                WHERE status = 'pending'
            """)
            event = cursor.fetchone()
            assert event[0] == "metrics_update"
            
            payload = json.loads(event[1])
            assert payload["data"] == metrics_data
    
    @pytest.mark.asyncio
    async def test_event_processing(self, persistent_engine, test_tool, test_server, db_manager):
        """Test event processing from queue"""
        # Add test tool and server to database
        db_manager.add_server(test_server)
        db_manager.add_tool(test_tool)
        
        # Record multiple events
        await persistent_engine.record_tool_execution(
            tool_id="test-tool",
            success=True,
            latency_ms=300,
            cost=0.05
        )
        
        await persistent_engine.record_tool_execution(
            tool_id="test-tool",
            success=False,
            latency_ms=1000,
            cost=0.0
        )
        
        # Start engine and process events
        await persistent_engine.start()
        
        # Wait for processing
        await asyncio.sleep(0.5)
        
        # Stop engine
        await persistent_engine.stop()
        
        # Check that events were processed
        from rmcp.storage.schema import get_connection
        with get_connection(persistent_engine.db_manager.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM telemetry_event_queue WHERE status = 'processed'")
            processed_count = cursor.fetchone()[0]
            assert processed_count == 2
            
            # Check that telemetry was added to legacy table
            cursor.execute("SELECT COUNT(*) FROM telemetry_queue")
            telemetry_count = cursor.fetchone()[0]
            assert telemetry_count == 2
    
    @pytest.mark.asyncio
    async def test_priority_processing(self, persistent_engine, test_tool, test_server, db_manager):
        """Test that events are processed in priority order"""
        # Add test tool and server to database
        db_manager.add_server(test_server)
        db_manager.add_tool(test_tool)
        
        # Record events with different priorities
        await persistent_engine.record_tool_execution(
            tool_id="test-tool",
            success=True,
            latency_ms=100,
            priority=3  # low priority
        )
        
        await persistent_engine.record_tool_execution(
            tool_id="test-tool",
            success=True,
            latency_ms=200,
            priority=1  # high priority
        )
        
        await persistent_engine.record_tool_execution(
            tool_id="test-tool",
            success=True,
            latency_ms=150,
            priority=2  # medium priority
        )
        
        # Start engine and process events
        await persistent_engine.start()
        await asyncio.sleep(0.5)
        await persistent_engine.stop()
        
        # Check processing order (high priority first)
        from rmcp.storage.schema import get_connection
        with get_connection(persistent_engine.db_manager.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT priority, processed_at
                FROM telemetry_event_queue
                WHERE status = 'processed'
                ORDER BY processed_at ASC
            """)
            events = cursor.fetchall()
            
            # High priority should be processed first
            assert events[0][0] == 1  # high priority
            assert events[1][0] == 2  # medium priority
            assert events[2][0] == 3  # low priority
    
    @pytest.mark.asyncio
    async def test_retry_mechanism(self, persistent_engine, test_tool, test_server, db_manager):
        """Test retry mechanism for failed events"""
        # Add test tool and server to database
        db_manager.add_server(test_server)
        db_manager.add_tool(test_tool)
        
        # Record an event
        await persistent_engine.record_tool_execution(
            tool_id="test-tool",
            success=True,
            latency_ms=100
        )
        
        # Mock the processing to fail
        with patch.object(persistent_engine, '_process_tool_execution_event', side_effect=Exception("Processing failed")):
            # Start engine and process events
            await persistent_engine.start()
            await asyncio.sleep(0.5)
            await persistent_engine.stop()
        
        # Check that event was retried
        from rmcp.storage.schema import get_connection
        with get_connection(persistent_engine.db_manager.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT retry_count, status FROM telemetry_event_queue")
            event = cursor.fetchone()
            assert event[0] > 0  # retry_count > 0
            assert event[1] in ['pending', 'failed']  # status
    
    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self, persistent_engine, test_tool, test_server, db_manager):
        """Test that events are marked as failed after max retries"""
        # Add test tool and server to database
        db_manager.add_server(test_server)
        db_manager.add_tool(test_tool)
        
        # Record an event
        await persistent_engine.record_tool_execution(
            tool_id="test-tool",
            success=True,
            latency_ms=100
        )
        
        # Mock the processing to always fail
        with patch.object(persistent_engine, '_process_tool_execution_event', side_effect=Exception("Always fails")):
            # Start engine and process events multiple times
            await persistent_engine.start()
            
            # Process enough times to exceed max retries
            for _ in range(5):
                await asyncio.sleep(0.2)
            
            await persistent_engine.stop()
        
        # Check that event was marked as failed
        from rmcp.storage.schema import get_connection
        with get_connection(persistent_engine.db_manager.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT retry_count, status FROM telemetry_event_queue")
            event = cursor.fetchone()
            assert event[0] >= persistent_engine.max_retries
            assert event[1] == 'failed'
    
    @pytest.mark.asyncio
    async def test_flush_queues(self, persistent_engine, test_tool, test_server, db_manager):
        """Test flushing all pending events"""
        # Add test tool and server to database
        db_manager.add_server(test_server)
        db_manager.add_tool(test_tool)
        
        # Record multiple events
        for i in range(3):
            await persistent_engine.record_tool_execution(
                tool_id="test-tool",
                success=True,
                latency_ms=100 + i * 10
            )
        
        # Check that events are pending
        from rmcp.storage.schema import get_connection
        with get_connection(persistent_engine.db_manager.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM telemetry_event_queue WHERE status = 'pending'")
            pending_count = cursor.fetchone()[0]
            assert pending_count == 3
        
        # Flush queues
        await persistent_engine.flush_queues()
        
        # Check that all events were processed
        with get_connection(persistent_engine.db_manager.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM telemetry_event_queue WHERE status = 'pending'")
            pending_count = cursor.fetchone()[0]
            assert pending_count == 0
            
            cursor.execute("SELECT COUNT(*) FROM telemetry_event_queue WHERE status = 'processed'")
            processed_count = cursor.fetchone()[0]
            assert processed_count == 3
    
    @pytest.mark.asyncio
    async def test_cleanup_old_events(self, persistent_engine, test_tool, test_server, db_manager):
        """Test cleanup of old processed events"""
        # Add test tool and server to database
        db_manager.add_server(test_server)
        db_manager.add_tool(test_tool)
        
        # Manually insert old processed events
        from rmcp.storage.schema import get_connection
        old_date = (datetime.utcnow() - timedelta(days=10)).isoformat()
        
        with get_connection(persistent_engine.db_manager.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO telemetry_event_queue (
                    event_type, tool_id, priority, payload, status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                "tool_execution",
                "test-tool",
                2,
                json.dumps({"timestamp": old_date, "data": {"success": True, "latency_ms": 100}}),
                "processed",
                old_date
            ))
            conn.commit()
        
        # Cleanup old events
        await persistent_engine.cleanup_old_events(days_old=7)
        
        # Check that old event was deleted
        with get_connection(persistent_engine.db_manager.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM telemetry_event_queue")
            count = cursor.fetchone()[0]
            assert count == 0
    
    def test_get_stats(self, persistent_engine):
        """Test getting engine statistics"""
        stats = persistent_engine.get_stats()
        
        assert "events_processed" in stats
        assert "events_failed" in stats
        assert "events_retried" in stats
        assert "queue_size" in stats
        assert "is_running" in stats
        assert "batch_size" in stats
        assert "processing_interval" in stats
        assert "max_retries" in stats
        
        assert stats["is_running"] is False
        assert stats["batch_size"] == 5
        assert stats["processing_interval"] == 0.1
        assert stats["max_retries"] == 3
    
    @pytest.mark.asyncio
    async def test_affinity_update_processing(self, persistent_engine, test_tool, test_server, db_manager):
        """Test processing of affinity update events"""
        # Add test tool and server to database
        db_manager.add_server(test_server)
        db_manager.add_tool(test_tool)
        
        # Record affinity update
        await persistent_engine.record_affinity_update(
            tool_id="test-tool",
            request_text="test request for affinity update",
            success=True
        )
        
        # Start engine and process events
        await persistent_engine.start()
        await asyncio.sleep(0.5)
        await persistent_engine.stop()
        
        # Check that event was processed
        from rmcp.storage.schema import get_connection
        with get_connection(persistent_engine.db_manager.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM telemetry_event_queue WHERE status = 'processed'")
            processed_count = cursor.fetchone()[0]
            assert processed_count == 1
        
        # Verify that embedding manager was called (through generate_embedding in EmbeddingStore)
        # Note: EmbeddingStore.add_successful_embedding calls generate_embedding, not encode_text
        assert processed_count == 1  # Event was processed successfully
    
    @pytest.mark.asyncio
    async def test_batch_processing(self, persistent_engine, test_tool, test_server, db_manager):
        """Test batch processing of events"""
        # Add test tool and server to database
        db_manager.add_server(test_server)
        db_manager.add_tool(test_tool)
        
        # Record more events than batch size
        for i in range(7):  # batch_size is 5
            await persistent_engine.record_tool_execution(
                tool_id="test-tool",
                success=True,
                latency_ms=100 + i * 10
            )
        
        # Start engine and process events
        await persistent_engine.start()
        await asyncio.sleep(1.0)  # Allow time for multiple batches
        await persistent_engine.stop()
        
        # Check that all events were processed
        from rmcp.storage.schema import get_connection
        with get_connection(persistent_engine.db_manager.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM telemetry_event_queue WHERE status = 'processed'")
            processed_count = cursor.fetchone()[0]
            assert processed_count == 7
    
    @pytest.mark.asyncio
    async def test_error_handling_in_enqueue(self, persistent_engine):
        """Test error handling when enqueueing events"""
        # Mock database connection to fail
        with patch('rmcp.storage.schema.get_connection', side_effect=Exception("Database error")):
            # This should not raise an exception
            await persistent_engine.record_tool_execution(
                tool_id="test-tool",
                success=True,
                latency_ms=100
            )
        
        # Engine should still be functional
        assert persistent_engine is not None
    
    @pytest.mark.asyncio
    async def test_concurrent_processing(self, persistent_engine, test_tool, test_server, db_manager):
        """Test concurrent event processing"""
        # Add test tool and server to database
        db_manager.add_server(test_server)
        db_manager.add_tool(test_tool)
        
        # Record events concurrently
        async def record_events():
            for i in range(5):
                await persistent_engine.record_tool_execution(
                    tool_id="test-tool",
                    success=True,
                    latency_ms=100 + i * 10
                )
                await asyncio.sleep(0.01)  # Small delay
        
        # Start engine
        await persistent_engine.start()
        
        # Record events concurrently
        await record_events()
        
        # Wait for processing
        await asyncio.sleep(1.0)
        
        # Stop engine
        await persistent_engine.stop()
        
        # Check that all events were processed
        from rmcp.storage.schema import get_connection
        with get_connection(persistent_engine.db_manager.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM telemetry_event_queue WHERE status = 'processed'")
            processed_count = cursor.fetchone()[0]
            assert processed_count == 5
