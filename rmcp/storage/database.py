"""
Database manager for RMCP
"""

import json
import sqlite3
from typing import List, Optional, Dict, Any
from datetime import datetime

from ..models.tool import Tool, Server
from .schema import get_connection


class DatabaseManager:
    """Manager for RMCP database operations"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
    
    def add_server(self, server: Server) -> None:
        """Add MCP server"""
        with get_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO servers (id, base_url, description, added_at)
                VALUES (?, ?, ?, ?)
            """, (server.id, server.base_url, server.description, server.added_at.isoformat()))
            conn.commit()
    
    def get_server(self, server_id: str) -> Optional[Server]:
        """Get server by ID"""
        with get_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM servers WHERE id = ?", (server_id,))
            row = cursor.fetchone()
            
            if row:
                return Server(
                    id=row['id'],
                    base_url=row['base_url'],
                    description=row['description'],
                    added_at=datetime.fromisoformat(row['added_at'])
                )
            return None
    
    def add_tool(self, tool: Tool) -> None:
        """Add tool"""
        with get_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO tools (
                    id, server_id, name, description,
                    input_schema, output_schema, tags, capabilities, tool_type,
                    specialization, abstraction_level, max_complexity, avg_execution_time_ms,
                    p95_latency_ms, success_rate, cost_hint, affinity_embeddings, embedding_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                tool.id,
                tool.server_id,
                tool.name,
                tool.description,
                json.dumps(tool.input_schema) if tool.input_schema else None,
                json.dumps(tool.output_schema) if tool.output_schema else None,
                json.dumps(tool.tags),
                json.dumps(tool.capabilities),
                tool.tool_type,
                getattr(tool, 'specialization', None),
                getattr(tool, 'abstraction_level', 'low'),
                getattr(tool, 'max_complexity', 1.0),
                getattr(tool, 'avg_execution_time_ms', 5000),
                tool.p95_latency_ms,
                tool.success_rate,
                tool.cost_hint,
                tool.affinity_embeddings,
                getattr(tool, 'embedding_count', 0)  # Default to 0 if not set
            ))
            conn.commit()
    
    def get_tool(self, tool_id: str) -> Optional[Tool]:
        """Get tool by ID"""
        with get_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM tools WHERE id = ?", (tool_id,))
            row = cursor.fetchone()
            
            if row:
                return self._row_to_tool(row)
            return None
    
    def search_tools(self, query: str, limit: int = 50) -> List[Tool]:
        """Search tools by simple text matching (FTS5 fallback) - English only"""
        with get_connection(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Simple English keyword extraction (LLM should handle natural language translation)
            import re
            keywords = re.findall(r'\b[a-zA-Z]{3,}\b', query.lower())
            
            if keywords:
                # Build search conditions for each keyword
                conditions = []
                params = []
                for keyword in keywords[:5]:  # Limit to first 5 keywords
                    pattern = f"%{keyword}%"
                    conditions.append("(name LIKE ? OR description LIKE ? OR tags LIKE ?)")
                    params.extend([pattern, pattern, pattern])
                
                where_clause = " OR ".join(conditions)
                params.append(limit)
                
                cursor.execute(f"""
                    SELECT * FROM tools 
                    WHERE {where_clause}
                    ORDER BY success_rate DESC, name
                    LIMIT ?
                """, params)
            else:
                # Fallback to simple pattern search
                search_pattern = f"%{query}%"
                cursor.execute("""
                    SELECT * FROM tools 
                    WHERE name LIKE ? OR description LIKE ? OR tags LIKE ?
                    ORDER BY success_rate DESC, name
                    LIMIT ?
                """, (search_pattern, search_pattern, search_pattern, limit))
            
            rows = cursor.fetchall()
            tools = [self._row_to_tool(row) for row in rows if self._row_to_tool(row)]
            print(f"Fallback search found {len(tools)} tools for query: {query}")
            return tools
    
    def get_tools_by_capability(self, capability: str) -> List[Tool]:
        """Get tools by capability"""
        with get_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM tools 
                WHERE capabilities LIKE ?
                ORDER BY success_rate DESC, p95_latency_ms ASC
            """, (f'%"{capability}"%',))
            
            rows = cursor.fetchall()
            return [self._row_to_tool(row) for row in rows]
    
    def get_all_tools(self) -> List[Tool]:
        """Get all tools"""
        with get_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM tools ORDER BY name")
            rows = cursor.fetchall()
            return [self._row_to_tool(row) for row in rows]
    
    def update_tool_metrics(self, tool_id: str, success: bool, latency_ms: int, cost: float = 0.0) -> None:
        """Update tool metrics"""
        with get_connection(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get current metrics
            cursor.execute("SELECT success_rate, p95_latency_ms FROM tools WHERE id = ?", (tool_id,))
            row = cursor.fetchone()
            
            if row:
                # Simple update (in MVP version)
                # In full version there will be more complex logic with EMA and percentiles
                new_success_rate = (row['success_rate'] + (1.0 if success else 0.0)) / 2
                new_latency = (row['p95_latency_ms'] + latency_ms) / 2
                
                cursor.execute("""
                    UPDATE tools 
                    SET success_rate = ?, p95_latency_ms = ?, cost_hint = ?
                    WHERE id = ?
                """, (new_success_rate, new_latency, cost, tool_id))
                conn.commit()
    
    def add_telemetry_event(self, tool_id: str, success: bool, latency_ms: int, 
                          cost: float = 0.0, request_embedding: Optional[bytes] = None) -> None:
        """Add telemetry event to queue"""
        with get_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO telemetry_queue (tool_id, success, latency_ms, cost, request_embedding)
                VALUES (?, ?, ?, ?, ?)
            """, (tool_id, success, latency_ms, cost, request_embedding))
            conn.commit()
    
    def _row_to_tool(self, row: sqlite3.Row) -> Tool:
        """Convert database row to Tool object"""
        return Tool(
            id=row['id'],
            server_id=row['server_id'],
            name=row['name'],
            description=row['description'],
            input_schema=json.loads(row['input_schema']) if row['input_schema'] and row['input_schema'] != '{}' else None,
            output_schema=json.loads(row['output_schema']) if row['output_schema'] and row['output_schema'] != '{}' else None,
            tags=json.loads(row['tags'] or '[]'),
            capabilities=json.loads(row['capabilities'] or '[]'),
            tool_type=row['tool_type'] if 'tool_type' in row.keys() else 'atomic',  # Default to 'atomic' for backward compatibility
            specialization=row['specialization'] if 'specialization' in row.keys() else None,
            abstraction_level=row['abstraction_level'] if 'abstraction_level' in row.keys() else 'low',
            max_complexity=row['max_complexity'] if 'max_complexity' in row.keys() else 1.0,
            avg_execution_time_ms=row['avg_execution_time_ms'] if 'avg_execution_time_ms' in row.keys() else 5000,
            p95_latency_ms=row['p95_latency_ms'],
            success_rate=row['success_rate'],
            cost_hint=row['cost_hint'],
            affinity_embeddings=row['affinity_embeddings']
        )
    
    def update_tool_embeddings(self, tool_id: str, embeddings_data: bytes, embedding_count: int) -> None:
        """Update tool's affinity embeddings"""
        with get_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE tools 
                SET affinity_embeddings = ?, embedding_count = ?
                WHERE id = ?
            """, (embeddings_data, embedding_count, tool_id))
            conn.commit()
    
    def get_tool_embeddings(self, tool_id: str) -> tuple[Optional[bytes], int]:
        """Get tool's affinity embeddings and count"""
        with get_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT affinity_embeddings, embedding_count 
                FROM tools 
                WHERE id = ?
            """, (tool_id,))
            row = cursor.fetchone()
            
            if row:
                return row['affinity_embeddings'], row['embedding_count']
            return None, 0
    
    def get_tools_by_type(self, tool_type: str) -> List[Tool]:
        """Get tools by type (atomic or agent)"""
        with get_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM tools 
                WHERE tool_type = ?
                ORDER BY success_rate DESC, p95_latency_ms ASC
            """, (tool_type,))
            
            rows = cursor.fetchall()
            return [self._row_to_tool(row) for row in rows]
    
    def get_agents(self) -> List[Tool]:
        """Get all agents (tools with tool_type='agent')"""
        return self.get_tools_by_type('agent')
    
    def get_atomic_tools(self) -> List[Tool]:
        """Get all atomic tools (tools with tool_type='atomic')"""
        return self.get_tools_by_type('atomic')
    
    def get_server_by_url(self, url: str) -> Optional[Server]:
        """Get server by base URL"""
        with get_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM servers WHERE base_url = ?", (url,))
            row = cursor.fetchone()
            
            if row:
                return Server(
                    id=row['id'],
                    base_url=row['base_url'],
                    description=row['description'],
                    added_at=datetime.fromisoformat(row['added_at'])
                )
            return None
    
    def get_tool_by_id(self, tool_id: str) -> Optional[Tool]:
        """Get tool by ID (alias for get_tool)"""
        return self.get_tool(tool_id)
    
    def update_tool(self, tool_id: str, **kwargs) -> None:
        """Update tool fields"""
        # Build dynamic update query
        fields = []
        values = []
        
        for field, value in kwargs.items():
            if field in ['input_schema', 'output_schema', 'tags', 'capabilities']:
                # JSON fields
                fields.append(f"{field} = ?")
                values.append(json.dumps(value))
            else:
                fields.append(f"{field} = ?")
                values.append(value)
        
        if not fields:
            return
        
        values.append(tool_id)
        
        with get_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(f"""
                UPDATE tools 
                SET {', '.join(fields)}
                WHERE id = ?
            """, values)
            conn.commit()
