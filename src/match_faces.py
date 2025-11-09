"""
Matching pipeline for face recognition using similarity search
"""
import numpy as np
from typing import List, Tuple, Optional
import time

try:
    import faiss
except ImportError:
    faiss = None

from src.config import settings
from src.logger import log


class FaceMatcher:
    """Face matcher using cosine similarity"""
    
    def __init__(self, use_faiss: bool = None):
        """
        Initialize face matcher
        
        Args:
            use_faiss: Whether to use Faiss for fast similarity search
        """
        self.use_faiss = use_faiss if use_faiss is not None else settings.USE_FAISS
        self.embeddings = []
        self.identity_ids = []
        self.identity_names = []
        self.index = None
        
        if self.use_faiss and faiss is None:
            log.warning("Faiss not installed, falling back to numpy")
            self.use_faiss = False
        
        log.info(f"Initialized face matcher (use_faiss={self.use_faiss})")
    
    def add_embedding(self, embedding: np.ndarray, identity_id: int, identity_name: str):
        """
        Add single embedding to gallery
        
        Args:
            embedding: Face embedding vector
            identity_id: Unique identity ID
            identity_name: Identity name
        """
        self.embeddings.append(embedding)
        self.identity_ids.append(identity_id)
        self.identity_names.append(identity_name)
        
        # Rebuild index
        if self.use_faiss:
            self._build_faiss_index()
    
    def add_embeddings(
        self,
        embeddings: List[np.ndarray],
        identity_ids: List[int],
        identity_names: List[str]
    ):
        """
        Add multiple embeddings to gallery
        
        Args:
            embeddings: List of face embedding vectors
            identity_ids: List of identity IDs
            identity_names: List of identity names
        """
        self.embeddings.extend(embeddings)
        self.identity_ids.extend(identity_ids)
        self.identity_names.extend(identity_names)
        
        # Rebuild index
        if self.use_faiss:
            self._build_faiss_index()
        
        log.info(f"Added {len(embeddings)} embeddings to gallery")
    
    def _build_faiss_index(self):
        """Build Faiss index for fast similarity search"""
        if len(self.embeddings) == 0:
            return
        
        embeddings_array = np.array(self.embeddings).astype(np.float32)
        dimension = embeddings_array.shape[1]
        
        # Use IndexFlatIP for cosine similarity (assumes normalized embeddings)
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings_array)
        
        log.debug(f"Built Faiss index with {len(self.embeddings)} embeddings")
    
    def match(
        self,
        query_embedding: np.ndarray,
        threshold: float = None,
        top_k: int = None
    ) -> List[Tuple[int, str, float]]:
        """
        Match query embedding against gallery
        
        Args:
            query_embedding: Query face embedding
            threshold: Similarity threshold (default from config)
            top_k: Number of top matches to return (default from config)
            
        Returns:
            List of (identity_id, identity_name, similarity) tuples
        """
        if len(self.embeddings) == 0:
            return []
        
        threshold = threshold if threshold is not None else settings.RECOGNITION_THRESHOLD
        top_k = top_k if top_k is not None else settings.TOP_K_RESULTS
        
        start_time = time.time()
        
        if self.use_faiss:
            results = self._match_faiss(query_embedding, threshold, top_k)
        else:
            results = self._match_numpy(query_embedding, threshold, top_k)
        
        elapsed = (time.time() - start_time) * 1000
        log.debug(f"Matching completed in {elapsed:.2f}ms, found {len(results)} matches")
        
        return results
    
    def _match_numpy(
        self,
        query_embedding: np.ndarray,
        threshold: float,
        top_k: int
    ) -> List[Tuple[int, str, float]]:
        """Match using numpy cosine similarity"""
        embeddings_array = np.array(self.embeddings)
        
        # Compute cosine similarities
        similarities = np.dot(embeddings_array, query_embedding)
        
        # Filter by threshold
        valid_idx = similarities >= threshold
        valid_similarities = similarities[valid_idx]
        
        if len(valid_similarities) == 0:
            return []
        
        # Sort by similarity (descending)
        sorted_idx = np.argsort(valid_similarities)[::-1][:top_k]
        
        # Get original indices
        valid_positions = np.where(valid_idx)[0]
        
        # Build results
        results = []
        for idx in sorted_idx:
            original_idx = valid_positions[idx]
            results.append((
                self.identity_ids[original_idx],
                self.identity_names[original_idx],
                float(valid_similarities[idx])
            ))
        
        return results
    
    def _match_faiss(
        self,
        query_embedding: np.ndarray,
        threshold: float,
        top_k: int
    ) -> List[Tuple[int, str, float]]:
        """Match using Faiss index"""
        query = query_embedding.reshape(1, -1).astype(np.float32)
        
        # Search
        distances, indices = self.index.search(query, min(top_k, len(self.embeddings)))
        
        # Filter by threshold and build results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if dist >= threshold:
                results.append((
                    self.identity_ids[idx],
                    self.identity_names[idx],
                    float(dist)
                ))
        
        return results
    
    def clear(self):
        """Clear all embeddings from gallery"""
        self.embeddings = []
        self.identity_ids = []
        self.identity_names = []
        self.index = None
        log.info("Cleared gallery")
    
    def get_stats(self) -> dict:
        """Get matcher statistics"""
        return {
            "num_embeddings": len(self.embeddings),
            "num_identities": len(set(self.identity_ids)),
            "use_faiss": self.use_faiss,
            "embedding_dim": len(self.embeddings[0]) if len(self.embeddings) > 0 else 0
        }


def calculate_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two embeddings
    
    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector
        
    Returns:
        Cosine similarity score
    """
    return float(np.dot(embedding1, embedding2))


def calculate_distance(embedding1: np.ndarray, embedding2: np.ndarray, metric: str = "cosine") -> float:
    """
    Calculate distance between two embeddings
    
    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector
        metric: Distance metric ('cosine', 'euclidean', or 'l2')
        
    Returns:
        Distance score
    """
    if metric == "cosine":
        return 1.0 - calculate_similarity(embedding1, embedding2)
    elif metric in ["euclidean", "l2"]:
        return float(np.linalg.norm(embedding1 - embedding2))
    else:
        raise ValueError(f"Unknown metric: {metric}")
