"""
Liveness Detection Module
Detect if a face is real (live person) or spoofed (photo/video/mask)
"""

import cv2
import numpy as np
from typing import Tuple, Dict
import torch
import torch.nn as nn

class LivenessDetector:
    """
    Liveness detection using multiple techniques:
    1. Texture analysis (LBP - Local Binary Patterns)
    2. Motion detection (optical flow)
    3. Blink detection
    4. Color analysis
    """
    
    def __init__(self, method='texture'):
        """
        Initialize liveness detector
        
        Args:
            method: 'texture', 'motion', 'blink', or 'multi'
        """
        self.method = method
        self.blink_counter = 0
        self.blink_threshold = 0.21  # Eye aspect ratio threshold
        
    def detect_liveness(self, image: np.ndarray) -> Tuple[bool, float, Dict]:
        """
        Detect if face is live or spoofed
        
        Args:
            image: Face image (BGR format)
            
        Returns:
            is_live: True if face is live
            confidence: Confidence score (0-1)
            details: Dictionary with analysis details
        """
        if self.method == 'texture':
            return self._texture_analysis(image)
        elif self.method == 'motion':
            return self._motion_analysis(image)
        elif self.method == 'blink':
            return self._blink_detection(image)
        elif self.method == 'multi':
            return self._multi_method(image)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _texture_analysis(self, image: np.ndarray) -> Tuple[bool, float, Dict]:
        """
        Analyze image texture using LBP
        Real faces have different texture than printed photos
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate LBP histogram
        lbp = self._compute_lbp(gray)
        
        # Calculate texture features
        variance = np.var(lbp)
        edge_density = self._calculate_edge_density(gray)
        
        # Simple heuristic: real faces have higher texture variance
        # and more complex edges
        score = (variance / 1000.0) * 0.6 + edge_density * 0.4
        score = min(score, 1.0)
        
        is_live = score > 0.5
        
        details = {
            'method': 'texture',
            'variance': float(variance),
            'edge_density': float(edge_density),
            'lbp_score': float(score)
        }
        
        return is_live, score, details
    
    def _compute_lbp(self, gray: np.ndarray, radius: int = 1, neighbors: int = 8) -> np.ndarray:
        """Compute Local Binary Pattern"""
        h, w = gray.shape
        lbp = np.zeros_like(gray)
        
        for i in range(radius, h - radius):
            for j in range(radius, w - radius):
                center = gray[i, j]
                binary = 0
                
                # 8 neighbors
                for k in range(neighbors):
                    angle = 2 * np.pi * k / neighbors
                    x = int(j + radius * np.cos(angle))
                    y = int(i - radius * np.sin(angle))
                    
                    if 0 <= x < w and 0 <= y < h:
                        binary = (binary << 1) | (1 if gray[y, x] >= center else 0)
                
                lbp[i, j] = binary
        
        return lbp
    
    def _calculate_edge_density(self, gray: np.ndarray) -> float:
        """Calculate edge density in image"""
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        return edge_density
    
    def _motion_analysis(self, image: np.ndarray) -> Tuple[bool, float, Dict]:
        """
        Analyze motion patterns
        Note: Requires multiple frames (video)
        This is a placeholder - needs frame buffer
        """
        # Simplified: Check for subtle movements
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate image gradient (motion indicator)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        motion_score = np.mean(np.sqrt(grad_x**2 + grad_y**2)) / 255.0
        
        is_live = motion_score > 0.1
        
        details = {
            'method': 'motion',
            'motion_score': float(motion_score),
            'note': 'Requires video frames for accurate detection'
        }
        
        return is_live, motion_score, details
    
    def _blink_detection(self, image: np.ndarray) -> Tuple[bool, float, Dict]:
        """
        Detect eye blinks (indicates live person)
        Requires face landmarks
        """
        # Simplified: Use eye region analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Assume eyes are in upper half
        eye_region = gray[:h//2, :]
        
        # Detect dark regions (closed eyes)
        _, thresh = cv2.threshold(eye_region, 50, 255, cv2.THRESH_BINARY_INV)
        dark_ratio = np.sum(thresh > 0) / thresh.size
        
        # Simple heuristic
        blink_detected = dark_ratio > 0.3
        confidence = dark_ratio if blink_detected else 1 - dark_ratio
        
        details = {
            'method': 'blink',
            'dark_ratio': float(dark_ratio),
            'blink_detected': blink_detected,
            'note': 'Requires facial landmarks for accurate blink detection'
        }
        
        return blink_detected, confidence, details
    
    def _multi_method(self, image: np.ndarray) -> Tuple[bool, float, Dict]:
        """
        Combine multiple methods for robust detection
        """
        texture_live, texture_score, texture_details = self._texture_analysis(image)
        motion_live, motion_score, motion_details = self._motion_analysis(image)
        
        # Weighted combination
        combined_score = texture_score * 0.7 + motion_score * 0.3
        is_live = combined_score > 0.55
        
        details = {
            'method': 'multi',
            'texture': texture_details,
            'motion': motion_details,
            'combined_score': float(combined_score)
        }
        
        return is_live, combined_score, details

class DeepLivenessDetector(nn.Module):
    """
    Deep learning based liveness detector
    Simple CNN for demonstration
    """
    
    def __init__(self):
        super(DeepLivenessDetector, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128 * 20 * 20, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)  # live vs spoof
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def check_liveness(image: np.ndarray, method: str = 'texture') -> Dict:
    """
    Convenience function to check liveness
    
    Args:
        image: Face image (BGR)
        method: Detection method
        
    Returns:
        Dictionary with results
    """
    detector = LivenessDetector(method=method)
    is_live, confidence, details = detector.detect_liveness(image)
    
    return {
        'is_live': is_live,
        'confidence': confidence,
        'details': details,
        'recommendation': 'Accept' if is_live else 'Reject (possible spoof)'
    }

# Example usage
if __name__ == "__main__":
    # Test with a sample image
    import cv2
    
    # Create a dummy face image
    test_image = np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8)
    
    # Test different methods
    print("Testing Liveness Detection:")
    print("-" * 50)
    
    for method in ['texture', 'motion', 'multi']:
        result = check_liveness(test_image, method=method)
        print(f"\nMethod: {method}")
        print(f"  Is Live: {result['is_live']}")
        print(f"  Confidence: {result['confidence']:.2%}")
        print(f"  Recommendation: {result['recommendation']}")
