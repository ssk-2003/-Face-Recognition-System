"""
Pydantic schemas for API request/response models
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime


class BoundingBox(BaseModel):
    """Bounding box coordinates"""
    x1: float
    y1: float
    x2: float
    y2: float


class DetectionResult(BaseModel):
    """Face detection result"""
    bbox: BoundingBox
    confidence: float
    quality_score: Optional[float] = None


class RecognitionMatch(BaseModel):
    """Face recognition match result"""
    identity_id: int
    identity_name: str
    similarity: float
    confidence: float


class RecognitionResult(BaseModel):
    """Complete recognition result for a detected face"""
    bbox: BoundingBox
    detection_confidence: float
    matches: List[RecognitionMatch] = []
    best_match: Optional[RecognitionMatch] = None


class DetectResponse(BaseModel):
    """Response for /detect endpoint"""
    success: bool
    num_faces: int
    detections: List[DetectionResult]
    processing_time_ms: float


class RecognizeResponse(BaseModel):
    """Response for /recognize endpoint"""
    success: bool
    num_faces: int
    results: List[RecognitionResult]
    processing_time_ms: float


class IdentityCreate(BaseModel):
    """Request to add new identity"""
    name: str = Field(..., description="Unique identity name")
    metadata: Optional[Dict] = Field(None, description="Additional metadata")


class IdentityResponse(BaseModel):
    """Response for identity operations"""
    id: int
    name: str
    image_path: str
    metadata: Optional[Dict] = None
    created_at: datetime
    
    class Config:
        from_attributes = True


class IdentityListResponse(BaseModel):
    """Response for listing identities"""
    success: bool
    count: int
    identities: List[IdentityResponse]


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    app_name: str
    version: str
    models: Dict[str, bool]


class StatsResponse(BaseModel):
    """System statistics response"""
    num_identities: int
    num_embeddings: int
    matcher_stats: Dict
    total_detections: int
    uptime_seconds: float


class ErrorResponse(BaseModel):
    """Error response"""
    success: bool = False
    error: str
    detail: Optional[str] = None
