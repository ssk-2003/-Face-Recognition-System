"""
Startup script for FRS API server
Adds project root to Python path and starts the API
"""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Now import and run the API service
if __name__ == "__main__":
    import uvicorn
    from src.api_service import app
    
    print("=" * 60)
    print("🚀 Starting Face Recognition System API")
    print("=" * 60)
    print(f"Server: http://localhost:8000")
    print(f"Docs: http://localhost:8000/docs")
    print(f"ReDoc: http://localhost:8000/redoc")
    print("=" * 60)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False
    )
