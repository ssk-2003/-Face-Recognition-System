"""
FastAPI routes for Face Recognition Service
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Form
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from typing import Optional
import numpy as np
import io
import time
from pathlib import Path
import uuid

from database import get_session, Identity, DetectionLog
from schemas import (
    DetectResponse, RecognizeResponse, IdentityCreate, IdentityResponse,
    IdentityListResponse, ErrorResponse, DetectionResult, RecognitionResult,
    RecognitionMatch, BoundingBox
)
from models.face_detector import FaceDetector
from models.feature_extractor import FeatureExtractor
from models.matcher import FaceMatcher
from utils.image_processing import load_image, save_image, crop_face, draw_bbox
from utils.logger import log
from config import settings

# Initialize router
router = APIRouter()

# Global model instances (initialized on startup)
detector: Optional[FaceDetector] = None
extractor: Optional[FeatureExtractor] = None
matcher: Optional[FaceMatcher] = None


def get_detector() -> FaceDetector:
    """Get face detector instance"""
    if detector is None:
        raise HTTPException(status_code=500, detail="Face detector not initialized")
    return detector


def get_extractor() -> FeatureExtractor:
    """Get feature extractor instance"""
    if extractor is None:
        raise HTTPException(status_code=500, detail="Feature extractor not initialized")
    return extractor


def get_matcher() -> FaceMatcher:
    """Get face matcher instance"""
    if matcher is None:
        raise HTTPException(status_code=500, detail="Face matcher not initialized")
    return matcher


async def load_image_from_upload(file: UploadFile) -> np.ndarray:
    """Load image from uploaded file"""
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        import cv2
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to decode image")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")


@router.post("/detect", response_model=DetectResponse)
async def detect_faces(
    file: UploadFile = File(..., description="Image file containing faces"),
    threshold: Optional[float] = None
):
    """
    Detect faces in uploaded image
    
    Returns bounding boxes and confidence scores for all detected faces.
    """
    start_time = time.time()
    
    try:
        # Load image
        image = await load_image_from_upload(file)
        
        # Detect faces
        det = get_detector()
        boxes, landmarks, confidences = det.detect(image, return_landmarks=True, return_confidence=True)
        
        # Filter by quality
        if len(boxes) > 0:
            boxes, qualities = det.filter_by_quality(image, boxes)
        
        # Build response
        detections = []
        for i, (box, conf) in enumerate(zip(boxes, confidences)):
            detections.append(DetectionResult(
                bbox=BoundingBox(x1=box[0], y1=box[1], x2=box[2], y2=box[3]),
                confidence=float(conf),
                quality_score=float(qualities[i]) if len(boxes) > 0 else None
            ))
        
        processing_time = (time.time() - start_time) * 1000
        
        return DetectResponse(
            success=True,
            num_faces=len(detections),
            detections=detections,
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        log.error(f"Error in detect endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/recognize", response_model=RecognizeResponse)
async def recognize_faces(
    file: UploadFile = File(..., description="Image file containing faces"),
    threshold: Optional[float] = None,
    top_k: Optional[int] = None,
    session: AsyncSession = Depends(get_session)
):
    """
    Detect and recognize faces in uploaded image
    
    Returns detected faces with matching identities from the gallery.
    """
    start_time = time.time()
    
    try:
        # Load image
        image = await load_image_from_upload(file)
        
        # Detect faces
        det = get_detector()
        boxes, landmarks, confidences = det.detect(image, return_landmarks=True, return_confidence=True)
        
        if len(boxes) == 0:
            return RecognizeResponse(
                success=True,
                num_faces=0,
                results=[],
                processing_time_ms=(time.time() - start_time) * 1000
            )
        
        # Extract features and match
        ext = get_extractor()
        mat = get_matcher()
        
        results = []
        for box, landmark, conf in zip(boxes, landmarks, confidences):
            # Crop and align face
            face = crop_face(image, box)
            
            # Extract embedding
            embedding = ext.extract(face)
            
            # Match against gallery
            matches = mat.match(embedding, threshold=threshold, top_k=top_k)
            
            # Build result
            match_list = [
                RecognitionMatch(
                    identity_id=m[0],
                    identity_name=m[1],
                    similarity=m[2],
                    confidence=m[2]
                )
                for m in matches
            ]
            
            results.append(RecognitionResult(
                bbox=BoundingBox(x1=box[0], y1=box[1], x2=box[2], y2=box[3]),
                detection_confidence=float(conf),
                matches=match_list,
                best_match=match_list[0] if match_list else None
            ))
        
        processing_time = (time.time() - start_time) * 1000
        
        # Log detection
        log_entry = DetectionLog(
            num_faces_detected=len(results),
            recognized_identity=results[0].best_match.identity_name if results and results[0].best_match else None,
            confidence=results[0].best_match.confidence if results and results[0].best_match else None,
            processing_time_ms=processing_time
        )
        session.add(log_entry)
        await session.commit()
        
        return RecognizeResponse(
            success=True,
            num_faces=len(results),
            results=results,
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        log.error(f"Error in recognize endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/add_identity", response_model=IdentityResponse)
async def add_identity(
    name: str = Form(..., description="Unique identity name"),
    file: UploadFile = File(..., description="Face image for the identity"),
    metadata: Optional[str] = Form(None, description="JSON metadata"),
    session: AsyncSession = Depends(get_session)
):
    """
    Add new identity to the gallery
    
    Detects face in the image, extracts embedding, and stores in database.
    """
    try:
        # Check if identity already exists
        result = await session.execute(select(Identity).where(Identity.name == name))
        if result.scalar_one_or_none():
            raise HTTPException(status_code=400, detail=f"Identity '{name}' already exists")
        
        # Load image
        image = await load_image_from_upload(file)
        
        # Detect single face
        det = get_detector()
        box, landmark, conf = det.detect_single(image, return_landmarks=True)
        
        if box is None:
            raise HTTPException(status_code=400, detail="No face detected in image")
        
        # Crop face
        face = crop_face(image, box)
        
        # Extract embedding
        ext = get_extractor()
        embedding = ext.extract(face)
        
        # Save image
        image_filename = f"{uuid.uuid4()}.jpg"
        image_path = settings.GALLERY_DIR / name / image_filename
        image_path.parent.mkdir(parents=True, exist_ok=True)
        save_image(face, str(image_path))
        
        # Create identity record
        identity = Identity(
            name=name,
            image_path=str(image_path)
        )
        identity.set_embedding(embedding)
        
        if metadata:
            import json
            identity.set_metadata(json.loads(metadata))
        
        session.add(identity)
        await session.commit()
        await session.refresh(identity)
        
        # Add to matcher
        mat = get_matcher()
        mat.add_embedding(embedding, identity.id, identity.name)
        
        log.info(f"Added identity: {name} (ID: {identity.id})")
        
        return IdentityResponse(
            id=identity.id,
            name=identity.name,
            image_path=identity.image_path,
            metadata=identity.get_metadata(),
            created_at=identity.created_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Error adding identity: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list_identities", response_model=IdentityListResponse)
async def list_identities(
    skip: int = 0,
    limit: int = 100,
    session: AsyncSession = Depends(get_session)
):
    """
    List all registered identities
    """
    try:
        result = await session.execute(
            select(Identity).offset(skip).limit(limit)
        )
        identities = result.scalars().all()
        
        identity_list = [
            IdentityResponse(
                id=identity.id,
                name=identity.name,
                image_path=identity.image_path,
                metadata=identity.get_metadata(),
                created_at=identity.created_at
            )
            for identity in identities
        ]
        
        return IdentityListResponse(
            success=True,
            count=len(identity_list),
            identities=identity_list
        )
        
    except Exception as e:
        log.error(f"Error listing identities: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/delete_identity/{identity_id}")
async def delete_identity(
    identity_id: int,
    session: AsyncSession = Depends(get_session)
):
    """
    Delete identity from gallery
    """
    try:
        result = await session.execute(
            select(Identity).where(Identity.id == identity_id)
        )
        identity = result.scalar_one_or_none()
        
        if not identity:
            raise HTTPException(status_code=404, detail="Identity not found")
        
        # Delete from database
        await session.delete(identity)
        await session.commit()
        
        # Rebuild matcher (reload all embeddings)
        await reload_gallery(session)
        
        log.info(f"Deleted identity: {identity.name} (ID: {identity_id})")
        
        return {"success": True, "message": f"Identity {identity.name} deleted"}
        
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Error deleting identity: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


async def reload_gallery(session: AsyncSession):
    """Reload all embeddings into matcher"""
    try:
        result = await session.execute(select(Identity))
        identities = result.scalars().all()
        
        mat = get_matcher()
        mat.clear()
        
        for identity in identities:
            embedding = identity.get_embedding()
            mat.add_embedding(embedding, identity.id, identity.name)
        
        log.info(f"Reloaded {len(identities)} identities into matcher")
        
    except Exception as e:
        log.error(f"Error reloading gallery: {str(e)}")
        raise
