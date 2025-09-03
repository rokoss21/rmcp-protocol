"""
FastAPI application for RMCP
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import yaml
import os

from ..storage.database import DatabaseManager
from ..storage.schema import init_database
from ..llm.manager import LLMManager
from ..embeddings.manager import EmbeddingManager
from ..planning.three_stage import ThreeStagePlanner
from ..telemetry.engine import TelemetryEngine
from ..telemetry.curator import BackgroundCurator
from ..observability.metrics import metrics
from ..logging.config import configure_logging, log_system
from ..logging.middleware import (
    RequestLoggingMiddleware,
    HealthCheckMiddleware,
    MetricsMiddleware,
    SecurityLoggingMiddleware,
    ErrorLoggingMiddleware
)
from .routes import router

# Prometheus integration
from prometheus_fastapi_instrumentator import Instrumentator


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management"""
    # Initialize on startup
    config = load_config()
    db_path = config.get("database", {}).get("path", "rmcp.db")
    
    # Initialize database
    init_database(db_path)
    
    # Initialize core components
    db_manager = DatabaseManager(db_path)
    app.state.db_manager = db_manager
    
    # Initialize config manager
    app.state.config_manager = type('ConfigManager', (), {'get_config': lambda self: config})()
    
    # Initialize Prometheus metrics
    app.state.metrics = metrics
    log_system("app", "prometheus_metrics_initialized", level="info")
    
    # Initialize LLM and embedding components
    llm_manager = None
    embedding_manager = None
    three_stage_planner = None
    telemetry_engine = None
    background_curator = None
    
    try:
        # Try to initialize LLM manager
        llm_manager = LLMManager(config)
        app.state.llm_manager = llm_manager
        log_system("app", "llm_manager_initialized", level="info")
        
        # Initialize embedding manager
        embedding_manager = EmbeddingManager(llm_manager)
        app.state.embedding_manager = embedding_manager
        log_system("app", "embedding_manager_initialized", level="info")
        
        # Initialize three-stage planner
        three_stage_planner = ThreeStagePlanner(db_manager, llm_manager, embedding_manager)
        app.state.three_stage_planner = three_stage_planner
        log_system("app", "three_stage_planner_initialized", level="info")
        
        # Initialize telemetry engine
        telemetry_engine = TelemetryEngine(db_manager, embedding_manager)
        app.state.telemetry_engine = telemetry_engine
        await telemetry_engine.start()
        log_system("app", "telemetry_engine_started", level="info")
        
        # Initialize background curator
        background_curator = BackgroundCurator(db_manager, embedding_manager)
        app.state.background_curator = background_curator
        await background_curator.start()
        log_system("app", "background_curator_started", level="info")
        
        log_system("app", "rmcp_phase2_fully_initialized", level="info")
        
        # Auto-ingestion of capabilities on startup
        try:
            from .routes import ingest_capabilities
            log_system("app", "starting_auto_ingestion", level="info")
            result = await ingest_capabilities(db_manager)
            log_system("app", "auto_ingestion_completed", level="info", 
                      servers_scanned=result.get("servers_scanned", 0),
                      tools_discovered=result.get("tools_discovered", 0))
        except Exception as e:
            log_system("app", "auto_ingestion_failed", level="warning", error=str(e))
        
    except Exception as e:
        log_system("app", "phase2_components_failed", level="warning", error=str(e))
        log_system("app", "falling_back_to_mvp_mode", level="info")
        
        # Fallback to simple planner
        from ..core.planner import SimplePlanner
        simple_planner = SimplePlanner(db_manager)
        app.state.three_stage_planner = simple_planner
        print("✅ Simple Planner initialized (fallback mode)")
    
    yield
    
    # Cleanup on shutdown
    print("🛑 Shutting down RMCP components...")
    
    if background_curator:
        try:
            await background_curator.stop()
            print("✅ Background Curator stopped")
        except Exception as e:
            print(f"❌ Error stopping Background Curator: {e}")
    
    if telemetry_engine:
        try:
            await telemetry_engine.flush_queues()
            await telemetry_engine.stop()
            print("✅ Telemetry Engine stopped")
        except Exception as e:
            print(f"❌ Error stopping Telemetry Engine: {e}")
    
    if llm_manager:
        try:
            await llm_manager.close()
            print("✅ LLM Manager closed")
        except Exception as e:
            print(f"❌ Error closing LLM Manager: {e}")
    
    print("👋 RMCP shutdown complete")


def load_config() -> dict:
    """Load configuration from YAML file"""
    config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # Substitute environment variables
            content = os.path.expandvars(content)
            config = yaml.safe_load(content)
        return config
    except FileNotFoundError:
        print(f"Warning: Config file not found at {config_path}")
        return {}
    except yaml.YAMLError as e:
        print(f"Error loading config: {e}")
        return {}


def get_db_manager() -> DatabaseManager:
    """Dependency for getting DB manager"""
    # In real application this would be from app.state
    # For MVP we use simple approach
    return DatabaseManager("rmcp.db")


def create_app() -> FastAPI:
    """Create FastAPI application"""
    
    # Configure structured logging
    configure_logging(log_level="INFO", json_format=True)
    
    app = FastAPI(
        title="RMCP - Routing & Memory Control Plane",
        description="Operating system for AI agents",
        version="0.1.0",
        lifespan=lifespan
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # In production, restrict this
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Structured logging middleware
    app.add_middleware(ErrorLoggingMiddleware)
    app.add_middleware(SecurityLoggingMiddleware)
    app.add_middleware(HealthCheckMiddleware)
    app.add_middleware(MetricsMiddleware)
    app.add_middleware(RequestLoggingMiddleware)
    
    # Prometheus instrumentation
    instrumentator = Instrumentator()
    instrumentator.instrument(app).expose(app)
    log_system("app", "prometheus_instrumentation_enabled", level="info")
    
    # Include routes
    app.include_router(router, prefix="/api/v1")
    
    # Root endpoint
    @app.get("/")
    async def root():
        return {
            "message": "RMCP - Routing & Memory Control Plane",
            "version": "0.1.0",
            "status": "running",
            "metrics": "/metrics"
        }
    
    # Health check endpoint
    @app.get("/health")
    async def health_check():
        return {"status": "healthy", "service": "rmcp"}
    
    return app
