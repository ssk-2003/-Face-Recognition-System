"""
Face detection module using RetinaFace
"""
import cv2
import numpy as np
import torch
from typing import List, Tuple, Optional
from pathlib import Path
import time

try:
    from facenet_pytorch import MTCNN
except ImportError:
    MTCNN = None

from config import settings
from utils.logger import log
from utils.image_processing import apply_nms, calculate_face_quality


class FaceDetector:
    """Face detector using RetinaFace/MTCNN"""
    
    def __init__(self, model_name: str = "retinaface", device: str = "cpu"):
        """
        Initialize face detector
        
        Args:
            model_name: Model to use ('retinaface' or 'mtcnn')
            device: Device to run inference on ('cpu' or 'cuda')
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self.load_model()
        log.info(f"Initialized {model_name} face detector on {device}")
    
    def load_model(self):
        """Load face detection model"""
        if self.model_name == "mtcnn" or self.model_name == "retinaface":
            # Use MTCNN as it's readily available and production-ready
            if MTCNN is None:
                raise ImportError("facenet-pytorch not installed. Install with: pip install facenet-pytorch")
            
            self.model = MTCNN(
                image_size=160,
                margin=0,
                min_face_size=settings.MIN_FACE_SIZE,
                thresholds=[0.6, 0.7, settings.DETECTION_THRESHOLD],
                factor=0.709,
                post_process=False,
                device=self.device,
                keep_all=True
            )
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
    
    def detect(
        self,
        image: np.ndarray,
        return_landmarks: bool = True,
        return_confidence: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Detect faces in image
        
        Args:
            image: Input image (RGB format)
            return_landmarks: Whether to return facial landmarks
            return_confidence: Whether to return confidence scores
            
        Returns:
            Tuple of (bounding_boxes, landmarks, confidences)
            - bounding_boxes: [N, 4] array of [x1, y1, x2, y2]
            - landmarks: [N, 5, 2] array of 5-point landmarks (if return_landmarks=True)
            - confidences: [N] array of confidence scores (if return_confidence=True)
        """
        start_time = time.time()
        
        # Detect faces
        boxes, probs, landmarks = self.model.detect(image, landmarks=True)
        
        # Handle no detections
        if boxes is None:
            return np.array([]), None, None
        
        # Filter by confidence threshold
        valid_idx = probs >= settings.DETECTION_THRESHOLD
        boxes = boxes[valid_idx]
        probs = probs[valid_idx]
        if landmarks is not None:
            landmarks = landmarks[valid_idx]
        
        # Apply NMS
        if len(boxes) > 1:
            keep_idx = apply_nms(boxes, probs, settings.NMS_THRESHOLD)
            boxes = boxes[keep_idx]
            probs = probs[keep_idx]
            if landmarks is not None:
                landmarks = landmarks[keep_idx]
        
        elapsed = (time.time() - start_time) * 1000
        log.debug(f"Detected {len(boxes)} faces in {elapsed:.2f}ms")
        
        return (
            boxes,
            landmarks if return_landmarks else None,
            probs if return_confidence else None
        )
    
    def detect_single(
        self,
        image: np.ndarray,
        return_landmarks: bool = True
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[float]]:
        """
        Detect single largest/most confident face
        
        Args:
            image: Input image (RGB format)
            return_landmarks: Whether to return facial landmarks
            
        Returns:
            Tuple of (bbox, landmarks, confidence) for single face, or None if no face found
        """
        boxes, landmarks, confidences = self.detect(image, return_landmarks, True)
        
        if len(boxes) == 0:
            return None, None, None
        
        # Select face with highest confidence
        best_idx = np.argmax(confidences)
        bbox = boxes[best_idx]
        landmark = landmarks[best_idx] if landmarks is not None else None
        confidence = confidences[best_idx]
        
        return bbox, landmark, confidence
    
    def filter_by_quality(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
        min_quality: float = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filter detections by face quality
        
        Args:
            image: Input image
            boxes: Detected bounding boxes
            min_quality: Minimum quality threshold
            
        Returns:
            Filtered boxes and quality scores
        """
        if min_quality is None:
            min_quality = settings.MIN_FACE_QUALITY
        
        qualities = []
        for box in boxes:
            x1, y1, x2, y2 = [int(v) for v in box]
            face = image[y1:y2, x1:x2]
            quality = calculate_face_quality(face, box)
            qualities.append(quality)
        
        qualities = np.array(qualities)
        valid_idx = qualities >= min_quality
        
        return boxes[valid_idx], qualities[valid_idx]


class ONNXFaceDetector:
    """ONNX-optimized face detector for faster CPU inference"""
    
    def __init__(self, onnx_path: str):
        """
        Initialize ONNX face detector
        
        Args:
            onnx_path: Path to ONNX model file
        """
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("onnxruntime not installed. Install with: pip install onnxruntime")
        
        self.session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        log.info(f"Loaded ONNX face detector from {onnx_path}")
    
    def detect(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect faces using ONNX model
        
        Args:
            image: Input image (RGB format)
            
        Returns:
            Tuple of (bounding_boxes, confidences)
        """
        # This is a placeholder - actual implementation depends on the ONNX model format
        # You would need to export your trained model to ONNX and implement preprocessing
        raise NotImplementedError("ONNX detection requires specific model format")
