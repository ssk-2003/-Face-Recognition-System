"""
Database models and operations for Face Recognition Service
"""
from sqlalchemy import Column, Integer, String, Float, DateTime, LargeBinary, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from typing import Optional, List
import numpy as np
import json
from config import settings

Base = declarative_base()


class Identity(Base):
    """Database model for storing face identities and embeddings"""
    __tablename__ = "identities"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True, nullable=False)
    embedding = Column(LargeBinary, nullable=False)  # Stored as numpy array bytes
    image_path = Column(String, nullable=False)
    extra_metadata = Column(String, nullable=True)  # JSON string for additional info
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def set_embedding(self, embedding: np.ndarray):
        """Convert numpy array to bytes for storage"""
        self.embedding = embedding.tobytes()
    
    def get_embedding(self) -> np.ndarray:
        """Convert bytes back to numpy array"""
        return np.frombuffer(self.embedding, dtype=np.float32)
    
    def set_metadata(self, metadata: dict):
        """Set metadata as JSON string"""
        self.extra_metadata = json.dumps(metadata)
    
    def get_metadata(self) -> dict:
        """Get metadata from JSON string"""
        return json.loads(self.extra_metadata) if self.extra_metadata else {}


class DetectionLog(Base):
    """Log table for face detection and recognition attempts"""
    __tablename__ = "detection_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    image_path = Column(String, nullable=True)
    num_faces_detected = Column(Integer, default=0)
    recognized_identity = Column(String, nullable=True)
    confidence = Column(Float, nullable=True)
    processing_time_ms = Column(Float, nullable=True)
    extra_metadata = Column(String, nullable=True)


# Database engine and session management
engine = None
async_session_maker = None


async def init_db():
    """Initialize database and create tables"""
    global engine, async_session_maker
    
    engine = create_async_engine(
        settings.DATABASE_URL,
        echo=settings.DEBUG,
        future=True
    )
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    async_session_maker = async_sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )


async def get_session() -> AsyncSession:
    """Get database session"""
    async with async_session_maker() as session:
        yield session


async def close_db():
    """Close database connections"""
    if engine:
        await engine.dispose()
