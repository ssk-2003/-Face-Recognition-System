"""
Main FastAPI application for Face Recognition Service
"""
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import time

from src.config import settings
from src.database import init_db, close_db, get_session, Identity
from src.api import routes
from src.detect_faces import FaceDetector
from src.extract_embeddings import FeatureExtractor
from src.match_faces import FaceMatcher
from src.logger import log
from src.schemas import HealthResponse, StatsResponse
from sqlalchemy import select


# Startup time for uptime calculation
start_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager for startup and shutdown
    """
    # Startup
    log.info("Starting Face Recognition Service...")
    
    # Initialize database
    await init_db()
    log.info("Database initialized")
    
    # Initialize models
    routes.detector = FaceDetector(
        model_name=settings.DETECTION_MODEL,
        device="cpu"
    )
    routes.extractor = FeatureExtractor(
        model_name=settings.EMBEDDING_MODEL,
        device="cpu"
    )
    routes.matcher = FaceMatcher(use_faiss=settings.USE_FAISS)
    
    # Load existing identities into matcher
    async for session in get_session():
        result = await session.execute(select(Identity))
        identities = result.scalars().all()
        
        for identity in identities:
            embedding = identity.get_embedding()
            routes.matcher.add_embedding(embedding, identity.id, identity.name)
        
        log.info(f"Loaded {len(identities)} identities from database")
        break
    
    log.info(f"Face Recognition Service started successfully on {settings.HOST}:{settings.PORT}")
    
    yield
    
    # Shutdown
    log.info("Shutting down Face Recognition Service...")
    await close_db()
    log.info("Face Recognition Service stopped")


# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Production-ready Face Recognition Service with detection, recognition, and gallery management",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    log.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"success": False, "error": "Internal server error", "detail": str(exc)}
    )


# Include routers
app.include_router(routes.router, prefix="/api/v1", tags=["Face Recognition"])


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "docs": "/docs"
    }


# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        app_name=settings.APP_NAME,
        version=settings.APP_VERSION,
        models={
            "detector": routes.detector is not None,
            "extractor": routes.extractor is not None,
            "matcher": routes.matcher is not None
        }
    )


# Stats endpoint
@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get system statistics"""
    async for session in get_session():
        from sqlalchemy import func
        from database import DetectionLog
        
        # Get identity count
        result = await session.execute(select(func.count(Identity.id)))
        identity_count = result.scalar()
        
        # Get detection log count
        result = await session.execute(select(func.count(DetectionLog.id)))
        detection_count = result.scalar()
        
        # Get matcher stats
        matcher_stats = routes.matcher.get_stats() if routes.matcher else {}
        
        return StatsResponse(
            num_identities=identity_count,
            num_embeddings=matcher_stats.get("num_embeddings", 0),
            matcher_stats=matcher_stats,
            total_detections=detection_count,
            uptime_seconds=time.time() - start_time
        )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )
