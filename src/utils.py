"""
Image processing utilities for face detection and recognition
"""
import cv2
import numpy as np
from typing import Tuple, Optional, List
from pathlib import Path
from PIL import Image
import torch


def load_image(image_path: str) -> np.ndarray:
    """
    Load image from file path
    
    Args:
        image_path: Path to image file
        
    Returns:
        Image as numpy array in RGB format
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Failed to load image from {image_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_image(image: np.ndarray, save_path: str):
    """
    Save image to file
    
    Args:
        image: Image as numpy array in RGB format
        save_path: Path to save image
    """
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(save_path), img_bgr)


def resize_image(image: np.ndarray, max_size: int = 1024) -> np.ndarray:
    """
    Resize image while maintaining aspect ratio
    
    Args:
        image: Input image
        max_size: Maximum dimension size
        
    Returns:
        Resized image
    """
    h, w = image.shape[:2]
    if max(h, w) <= max_size:
        return image
    
    scale = max_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def crop_face(image: np.ndarray, bbox: List[float], margin: float = 0.2) -> np.ndarray:
    """
    Crop face from image with margin
    
    Args:
        image: Input image
        bbox: Bounding box [x1, y1, x2, y2]
        margin: Margin to add around face (relative to face size)
        
    Returns:
        Cropped face image
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox
    
    # Calculate margin
    face_w = x2 - x1
    face_h = y2 - y1
    margin_w = int(face_w * margin)
    margin_h = int(face_h * margin)
    
    # Add margin and clip to image boundaries
    x1 = max(0, int(x1 - margin_w))
    y1 = max(0, int(y1 - margin_h))
    x2 = min(w, int(x2 + margin_w))
    y2 = min(h, int(y2 + margin_h))
    
    return image[y1:y2, x1:x2]


def align_face(image: np.ndarray, landmarks: np.ndarray, output_size: Tuple[int, int] = (112, 112)) -> np.ndarray:
    """
    Align face using 5-point landmarks
    
    Args:
        image: Input image
        landmarks: 5 facial landmarks [left_eye, right_eye, nose, left_mouth, right_mouth]
        output_size: Output image size
        
    Returns:
        Aligned face image
    """
    # Standard 5-point landmarks for 112x112 face
    standard_landmarks = np.array([
        [38.2946, 51.6963],  # left eye
        [73.5318, 51.5014],  # right eye
        [56.0252, 71.7366],  # nose
        [41.5493, 92.3655],  # left mouth
        [70.7299, 92.2041]   # right mouth
    ], dtype=np.float32)
    
    # Scale standard landmarks to output size
    if output_size != (112, 112):
        scale_x = output_size[0] / 112
        scale_y = output_size[1] / 112
        standard_landmarks[:, 0] *= scale_x
        standard_landmarks[:, 1] *= scale_y
    
    # Calculate transformation matrix
    tform = cv2.estimateAffinePartial2D(landmarks, standard_landmarks)[0]
    
    # Apply transformation
    aligned_face = cv2.warpAffine(image, tform, output_size, flags=cv2.INTER_LINEAR)
    
    return aligned_face


def normalize_face(face: np.ndarray) -> np.ndarray:
    """
    Normalize face image for model input
    
    Args:
        face: Face image (RGB)
        
    Returns:
        Normalized face image
    """
    # Convert to float and normalize to [0, 1]
    face = face.astype(np.float32) / 255.0
    
    # Apply standard normalization (ImageNet stats)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    face = (face - mean) / std
    
    return face


def calculate_blur_score(image: np.ndarray) -> float:
    """
    Calculate Laplacian variance to measure image blur
    
    Args:
        image: Input image
        
    Returns:
        Blur score (higher = sharper)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def calculate_face_quality(face: np.ndarray, bbox: List[float]) -> float:
    """
    Calculate face quality score based on size, blur, and other factors
    
    Args:
        face: Face image
        bbox: Face bounding box
        
    Returns:
        Quality score between 0 and 1
    """
    # Size score (prefer larger faces)
    x1, y1, x2, y2 = bbox
    face_size = (x2 - x1) * (y2 - y1)
    size_score = min(face_size / (100 * 100), 1.0)  # Normalize by 100x100 pixels
    
    # Blur score
    blur_score = min(calculate_blur_score(face) / 200.0, 1.0)  # Normalize
    
    # Combined quality score
    quality = (size_score * 0.5 + blur_score * 0.5)
    
    return quality


def draw_bbox(image: np.ndarray, bbox: List[float], label: str = "", confidence: float = 0.0, color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
    """
    Draw bounding box and label on image
    
    Args:
        image: Input image
        bbox: Bounding box [x1, y1, x2, y2]
        label: Label text
        confidence: Confidence score
        color: Box color (RGB)
        
    Returns:
        Image with drawn bounding box
    """
    img = image.copy()
    x1, y1, x2, y2 = [int(v) for v in bbox]
    
    # Draw rectangle
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    
    # Draw label
    if label:
        text = f"{label}: {confidence:.2f}" if confidence > 0 else label
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        
        # Get text size for background
        (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Draw background
        cv2.rectangle(img, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1)
        
        # Draw text
        cv2.putText(img, text, (x1, y1 - 5), font, font_scale, (255, 255, 255), thickness)
    
    return img


def apply_nms(boxes: np.ndarray, scores: np.ndarray, threshold: float = 0.4) -> List[int]:
    """
    Apply Non-Maximum Suppression
    
    Args:
        boxes: Bounding boxes [N, 4] (x1, y1, x2, y2)
        scores: Confidence scores [N]
        threshold: IoU threshold
        
    Returns:
        Indices of boxes to keep
    """
    if len(boxes) == 0:
        return []
    
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        
        inds = np.where(iou <= threshold)[0]
        order = order[inds + 1]
    
    return keep
