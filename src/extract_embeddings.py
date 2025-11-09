"""
Feature extraction module for face recognition using ArcFace/AdaFace
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Union, List
import time
from pathlib import Path

try:
    from facenet_pytorch import InceptionResnetV1
except ImportError:
    InceptionResnetV1 = None

from src.config import settings
from src.logger import log
from src.utils import normalize_face


class FeatureExtractor:
    """Feature extractor for face recognition"""
    
    def __init__(self, model_name: str = "arcface", device: str = "cpu"):
        """
        Initialize feature extractor
        
        Args:
            model_name: Model to use ('arcface', 'adaface', or 'facenet')
            device: Device to run inference on
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self.embedding_size = 512
        self.load_model()
        log.info(f"Initialized {model_name} feature extractor on {device}")
    
    def load_model(self):
        """Load feature extraction model"""
        if self.model_name == "facenet" or self.model_name == "arcface":
            # Use FaceNet as baseline (pretrained on VGGFace2)
            if InceptionResnetV1 is None:
                raise ImportError("facenet-pytorch not installed")
            
            self.model = InceptionResnetV1(
                pretrained='vggface2',
                classify=False,
                device=self.device
            ).eval()
            self.embedding_size = 512
            
        elif self.model_name == "adaface":
            # Placeholder for AdaFace - would require loading custom weights
            log.warning("AdaFace not implemented, falling back to FaceNet")
            self.model_name = "facenet"
            self.load_model()
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
    
    def preprocess(self, face: np.ndarray) -> torch.Tensor:
        """
        Preprocess face image for model input
        
        Args:
            face: Face image (RGB format, uint8)
            
        Returns:
            Preprocessed tensor
        """
        # Resize to 160x160 for FaceNet
        import cv2
        face = cv2.resize(face, (160, 160))
        
        # Normalize to [-1, 1]
        face = (face.astype(np.float32) - 127.5) / 128.0
        
        # Convert to tensor [C, H, W]
        face_tensor = torch.from_numpy(face).permute(2, 0, 1).unsqueeze(0)
        
        return face_tensor.to(self.device)
    
    def extract(self, face: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Extract embedding from face image
        
        Args:
            face: Face image or preprocessed tensor
            
        Returns:
            Normalized embedding vector
        """
        start_time = time.time()
        
        # Preprocess if needed
        if isinstance(face, np.ndarray):
            face_tensor = self.preprocess(face)
        else:
            face_tensor = face
        
        # Extract embedding
        with torch.no_grad():
            embedding = self.model(face_tensor)
        
        # Convert to numpy and normalize
        embedding = embedding.cpu().numpy().flatten()
        embedding = embedding / np.linalg.norm(embedding)
        
        elapsed = (time.time() - start_time) * 1000
        log.debug(f"Extracted embedding in {elapsed:.2f}ms")
        
        return embedding.astype(np.float32)
    
    def extract_batch(self, faces: List[np.ndarray]) -> np.ndarray:
        """
        Extract embeddings from multiple faces
        
        Args:
            faces: List of face images
            
        Returns:
            Array of normalized embeddings [N, embedding_size]
        """
        if len(faces) == 0:
            return np.array([])
        
        # Preprocess all faces
        face_tensors = [self.preprocess(face) for face in faces]
        batch_tensor = torch.cat(face_tensors, dim=0)
        
        # Extract embeddings
        with torch.no_grad():
            embeddings = self.model(batch_tensor)
        
        # Convert to numpy and normalize
        embeddings = embeddings.cpu().numpy()
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        return embeddings.astype(np.float32)


class ONNXFeatureExtractor:
    """ONNX-optimized feature extractor for faster CPU inference"""
    
    def __init__(self, onnx_path: str):
        """
        Initialize ONNX feature extractor
        
        Args:
            onnx_path: Path to ONNX model file
        """
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("onnxruntime not installed")
        
        self.session = ort.InferenceSession(
            onnx_path,
            providers=['CPUExecutionProvider']
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        log.info(f"Loaded ONNX feature extractor from {onnx_path}")
    
    def extract(self, face: np.ndarray) -> np.ndarray:
        """
        Extract embedding using ONNX model
        
        Args:
            face: Preprocessed face tensor [1, C, H, W]
            
        Returns:
            Normalized embedding vector
        """
        # Run inference
        outputs = self.session.run(
            [self.output_name],
            {self.input_name: face}
        )
        
        # Normalize
        embedding = outputs[0].flatten()
        embedding = embedding / np.linalg.norm(embedding)
        
        return embedding.astype(np.float32)


def export_to_onnx(model: nn.Module, save_path: str, input_size: tuple = (1, 3, 160, 160)):
    """
    Export PyTorch model to ONNX format
    
    Args:
        model: PyTorch model
        save_path: Path to save ONNX model
        input_size: Input tensor size
    """
    model.eval()
    dummy_input = torch.randn(*input_size)
    
    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    log.info(f"Exported model to ONNX: {save_path}")
