"""
SQLite database schema for RMCP
"""

import sqlite3
from typing import Optional


def init_database(db_path: str) -> None:
    """Initialize database with table creation"""
    
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        
        # Enable FTS5 for full-text search
        cursor.execute("PRAGMA compile_options")
        compile_options = [row[0] for row in cursor.fetchall()]
        if 'ENABLE_FTS5' not in compile_options:
            print("Warning: FTS5 not enabled in SQLite. Full-text search will not work.")
        
        # MCP servers table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS servers (
                id TEXT PRIMARY KEY,
                base_url TEXT NOT NULL UNIQUE,
                description TEXT,
                added_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
            )
        """)
        
        # Tools table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tools (
                id TEXT PRIMARY KEY,
                server_id TEXT NOT NULL,
                name TEXT NOT NULL,
                description TEXT,
                
                -- "Visa" (Static data)
                input_schema TEXT NOT NULL DEFAULT '{}',
                output_schema TEXT NOT NULL DEFAULT '{}',
                tags TEXT,
                capabilities TEXT,
                tool_type TEXT NOT NULL DEFAULT 'atomic',  -- 'atomic' or 'agent'
                
                -- Agent-specific fields
                specialization TEXT,  -- Agent specialization (e.g., 'security', 'deployment')
                abstraction_level TEXT DEFAULT 'low',  -- 'low', 'medium', 'high'
                max_complexity REAL DEFAULT 1.0,  -- Maximum task complexity this agent can handle
                avg_execution_time_ms INTEGER DEFAULT 5000,  -- Average execution time in ms
                
                -- "Border stamps" (Dynamic data)
                p95_latency_ms INTEGER DEFAULT 3000,
                success_rate REAL DEFAULT 0.95,
                cost_hint REAL DEFAULT 0.0,
                
                -- "Golden element" (Semantic Affinity)
                affinity_embeddings BLOB,  -- Serialized embeddings for successful precedents
                embedding_count INTEGER DEFAULT 0,  -- Number of stored embeddings
                
                UNIQUE(server_id, name),
                FOREIGN KEY(server_id) REFERENCES servers(id) ON DELETE CASCADE
            )
        """)
        
        # Virtual table for full-text search
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS tool_fts USING fts5(
                tool_id, name, description, tags, tool_type, specialization, abstraction_level
            )
        """)
        
        # Triggers for FTS5 synchronization
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS tools_ai AFTER INSERT ON tools BEGIN
                INSERT INTO tool_fts(tool_id, name, description, tags, tool_type, specialization, abstraction_level)
                VALUES (new.id, new.name, new.description, new.tags, new.tool_type, new.specialization, new.abstraction_level);
            END
        """)
        
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS tools_au AFTER UPDATE ON tools BEGIN
                UPDATE tool_fts SET 
                    tool_id = new.id,
                    name = new.name,
                    description = new.description,
                    tags = new.tags,
                    tool_type = new.tool_type,
                    specialization = new.specialization,
                    abstraction_level = new.abstraction_level
                WHERE tool_id = old.id;
            END
        """)
        
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS tools_ad AFTER DELETE ON tools BEGIN
                DELETE FROM tool_fts WHERE tool_id = old.id;
            END
        """)
        
        # Telemetry queue table (legacy)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS telemetry_queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tool_id TEXT NOT NULL,
                success BOOLEAN NOT NULL,
                latency_ms INTEGER NOT NULL,
                cost REAL DEFAULT 0.0,
                request_embedding BLOB,
                timestamp TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                FOREIGN KEY(tool_id) REFERENCES tools(id) ON DELETE CASCADE
            )
        """)
        
        # Persistent telemetry event queue table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS telemetry_event_queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                tool_id TEXT NOT NULL,
                priority INTEGER NOT NULL DEFAULT 2,
                payload TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                processed_at TEXT,
                retry_count INTEGER DEFAULT 0,
                max_retries INTEGER DEFAULT 3,
                status TEXT DEFAULT 'pending',
                FOREIGN KEY(tool_id) REFERENCES tools(id) ON DELETE CASCADE
            )
        """)
        
        # ===== SECURITY TABLES =====
        
        # Tenants table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tenants (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                is_active BOOLEAN NOT NULL DEFAULT 1,
                settings TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
            )
        """)
        
        # Users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                tenant_id TEXT NOT NULL,
                roles TEXT NOT NULL DEFAULT '[]',
                is_active BOOLEAN NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                FOREIGN KEY (tenant_id) REFERENCES tenants(id) ON DELETE CASCADE
            )
        """)
        
        # API Keys table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS api_keys (
                id TEXT PRIMARY KEY,
                key_hash TEXT UNIQUE NOT NULL,
                user_id TEXT NOT NULL,
                tenant_id TEXT NOT NULL,
                name TEXT NOT NULL,
                permissions TEXT NOT NULL DEFAULT '[]',
                expires_at TEXT,
                last_used_at TEXT,
                is_active BOOLEAN NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                FOREIGN KEY (tenant_id) REFERENCES tenants(id) ON DELETE CASCADE
            )
        """)
        
        # Roles table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS roles (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT NOT NULL,
                permissions TEXT NOT NULL DEFAULT '[]',
                tenant_id TEXT,
                is_system BOOLEAN NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                FOREIGN KEY (tenant_id) REFERENCES tenants(id) ON DELETE CASCADE
            )
        """)
        
        # Permissions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS permissions (
                id TEXT PRIMARY KEY,
                name TEXT UNIQUE NOT NULL,
                resource_type TEXT NOT NULL,
                action TEXT NOT NULL,
                description TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
            )
        """)
        
        # Audit events table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_events (
                id TEXT PRIMARY KEY,
                user_id TEXT,
                tenant_id TEXT,
                action TEXT NOT NULL,
                resource_type TEXT NOT NULL,
                resource_id TEXT,
                details TEXT NOT NULL DEFAULT '{}',
                ip_address TEXT,
                user_agent TEXT,
                timestamp TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                success BOOLEAN NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL,
                FOREIGN KEY (tenant_id) REFERENCES tenants(id) ON DELETE SET NULL
            )
        """)
        
        # Approval requests table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS approval_requests (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                tenant_id TEXT NOT NULL,
                action TEXT NOT NULL,
                resource_type TEXT NOT NULL,
                resource_id TEXT,
                details TEXT NOT NULL DEFAULT '{}',
                justification TEXT NOT NULL,
                approver_id TEXT,
                status TEXT NOT NULL DEFAULT 'pending',
                expires_at TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                FOREIGN KEY (tenant_id) REFERENCES tenants(id) ON DELETE CASCADE,
                FOREIGN KEY (approver_id) REFERENCES users(id) ON DELETE SET NULL
            )
        """)
        
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
        
        # Execution requests table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS execution_requests (
                id TEXT PRIMARY KEY,
                plan_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                tenant_id TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                approval_request_id TEXT,
                created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                started_at TEXT,
                completed_at TEXT,
                result TEXT,
                error TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                FOREIGN KEY (tenant_id) REFERENCES tenants(id) ON DELETE CASCADE,
                FOREIGN KEY (approval_request_id) REFERENCES approval_requests(id) ON DELETE SET NULL
            )
        """)
        
        # Performance indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tools_server_id ON tools(server_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tools_tags ON tools(tags)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tools_capabilities ON tools(capabilities)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_telemetry_tool_id ON telemetry_queue(tool_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_telemetry_timestamp ON telemetry_queue(timestamp)")
        
        # Telemetry event queue indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_telemetry_event_queue_status ON telemetry_event_queue(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_telemetry_event_queue_priority ON telemetry_event_queue(priority)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_telemetry_event_queue_created_at ON telemetry_event_queue(created_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_telemetry_event_queue_tool_id ON telemetry_event_queue(tool_id)")
        
        # Security indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_tenant_id ON users(tenant_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_api_keys_user_id ON api_keys(user_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_api_keys_tenant_id ON api_keys(tenant_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_roles_tenant_id ON roles(tenant_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_events_user_id ON audit_events(user_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_events_tenant_id ON audit_events(tenant_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_events_timestamp ON audit_events(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_approval_requests_user_id ON approval_requests(user_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_approval_requests_tenant_id ON approval_requests(tenant_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_approval_requests_status ON approval_requests(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_execution_requests_user_id ON execution_requests(user_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_execution_requests_tenant_id ON execution_requests(tenant_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_execution_requests_status ON execution_requests(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_execution_requests_approval_id ON execution_requests(approval_request_id)")
        
        conn.commit()
        print(f"Database initialized at {db_path}")


def get_connection(db_path: str) -> sqlite3.Connection:
    """Get database connection"""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row  # For column access by name
    return conn
