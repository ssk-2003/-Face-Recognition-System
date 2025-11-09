"""
Configuration management for Face Recognition Service
"""
from pydantic_settings import BaseSettings
from pathlib import Path
from typing import Optional


class Settings(BaseSettings):
    """Application settings"""
    
    # Application
    APP_NAME: str = "Face Recognition Service"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./frs_database.db"
    
    # Model Settings
    DETECTION_MODEL: str = "retinaface"
    EMBEDDING_MODEL: str = "arcface"
    DETECTION_THRESHOLD: float = 0.8
    RECOGNITION_THRESHOLD: float = 0.6
    TOP_K_RESULTS: int = 5
    
    # Optimization
    USE_ONNX: bool = True
    USE_FAISS: bool = True
    BATCH_SIZE: int = 1
    NUM_WORKERS: int = 4
    
    # Storage Paths
    GALLERY_DIR: Path = Path("./data/gallery")
    UPLOAD_DIR: Path = Path("./data/uploads")
    MODEL_DIR: Path = Path("./models")
    LOG_DIR: Path = Path("./logs")
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "./logs/frs.log"
    
    # Face Detection Parameters
    MIN_FACE_SIZE: int = 20
    NMS_THRESHOLD: float = 0.4
    
    # Face Quality Thresholds
    MIN_FACE_QUALITY: float = 0.5
    MIN_BLUR_THRESHOLD: float = 100.0
    
    class Config:
        env_file = ".env"
        case_sensitive = True
    
    def create_directories(self):
        """Create required directories if they don't exist"""
        self.GALLERY_DIR.mkdir(parents=True, exist_ok=True)
        self.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        self.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self.LOG_DIR.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()
settings.create_directories()
