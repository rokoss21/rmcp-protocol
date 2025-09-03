"""
RMCP Main Entry Point
FastAPI server startup for RMCP
"""

import uvicorn
from rmcp.gateway.app import create_app

# Create app instance for uvicorn
app = create_app()

def main():
    """Start RMCP server"""
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    main()
