"""
Face Clustering Module
Clusters unknown faces to identify unique individuals
"""

import numpy as np
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Tuple
import pickle
from pathlib import Path

class FaceClustering:
    """
    Cluster unknown faces to identify unique individuals
    Useful for grouping unknown people without manual labeling
    """
    
    def __init__(self, method='dbscan', eps=0.5, min_samples=2):
        """
        Initialize clustering
        
        Args:
            method: 'dbscan' or 'hierarchical'
            eps: DBSCAN epsilon (distance threshold)
            min_samples: Minimum samples per cluster
        """
        self.method = method
        self.eps = eps
        self.min_samples = min_samples
        self.clusterer = None
        self.scaler = StandardScaler()
        self.pca = None
        
    def fit(self, embeddings: np.ndarray, use_pca: bool = False, n_components: int = 128):
        """
        Fit clustering model on embeddings
        
        Args:
            embeddings: Face embeddings (N x D)
            use_pca: Whether to use PCA for dimensionality reduction
            n_components: Number of PCA components
            
        Returns:
            cluster_labels: Cluster assignment for each face
        """
        if len(embeddings) == 0:
            return np.array([])
        
        # Normalize embeddings
        X = self.scaler.fit_transform(embeddings)
        
        # Optional PCA
        if use_pca and embeddings.shape[1] > n_components:
            self.pca = PCA(n_components=n_components)
            X = self.pca.fit_transform(X)
        
        # Clustering
        if self.method == 'dbscan':
            self.clusterer = DBSCAN(
                eps=self.eps,
                min_samples=self.min_samples,
                metric='cosine'
            )
        elif self.method == 'hierarchical':
            n_clusters = max(2, len(embeddings) // 5)  # Estimate clusters
            self.clusterer = AgglomerativeClustering(
                n_clusters=n_clusters,
                metric='cosine',
                linkage='average'
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        labels = self.clusterer.fit_predict(X)
        
        return labels
    
    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Predict cluster for new embeddings
        Note: DBSCAN doesn't support predict, so we find nearest cluster
        
        Args:
            embeddings: New face embeddings
            
        Returns:
            cluster_labels: Predicted cluster for each face
        """
        if self.clusterer is None:
            raise ValueError("Model not fitted yet")
        
        X = self.scaler.transform(embeddings)
        
        if self.pca is not None:
            X = self.pca.transform(X)
        
        # For DBSCAN, find nearest existing cluster
        if self.method == 'dbscan':
            # This is a simplified approach
            # In practice, you'd compare against cluster centroids
            return np.array([-1] * len(embeddings))  # Mark as unknown
        else:
            return self.clusterer.fit_predict(X)
    
    def get_cluster_stats(self, labels: np.ndarray) -> Dict:
        """
        Get clustering statistics
        
        Returns:
            Dictionary with cluster statistics
        """
        unique_labels = set(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise = list(labels).count(-1)
        
        cluster_sizes = {}
        for label in unique_labels:
            if label != -1:
                cluster_sizes[f"Cluster {label}"] = list(labels).count(label)
        
        return {
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'n_samples': len(labels),
            'cluster_sizes': cluster_sizes
        }
    
    def save(self, path: str):
        """Save clustering model"""
        with open(path, 'wb') as f:
            pickle.dump({
                'clusterer': self.clusterer,
                'scaler': self.scaler,
                'pca': self.pca,
                'method': self.method,
                'eps': self.eps,
                'min_samples': self.min_samples
            }, f)
    
    def load(self, path: str):
        """Load clustering model"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.clusterer = data['clusterer']
            self.scaler = data['scaler']
            self.pca = data.get('pca')
            self.method = data['method']
            self.eps = data['eps']
            self.min_samples = data['min_samples']

def cluster_unknown_faces(
    embeddings: List[np.ndarray],
    method: str = 'dbscan',
    eps: float = 0.5,
    min_samples: int = 2
) -> Tuple[np.ndarray, Dict]:
    """
    Convenience function to cluster faces
    
    Args:
        embeddings: List of face embeddings
        method: Clustering method
        eps: DBSCAN epsilon
        min_samples: Minimum samples per cluster
        
    Returns:
        labels: Cluster assignments
        stats: Clustering statistics
    """
    if len(embeddings) == 0:
        return np.array([]), {}
    
    embeddings_array = np.array(embeddings)
    
    clustering = FaceClustering(method=method, eps=eps, min_samples=min_samples)
    labels = clustering.fit(embeddings_array)
    stats = clustering.get_cluster_stats(labels)
    
    return labels, stats

# Example usage
if __name__ == "__main__":
    # Generate some example embeddings (512-dim)
    np.random.seed(42)
    
    # Create 3 clusters of faces
    cluster1 = np.random.randn(10, 512) + np.array([1.0] * 512)
    cluster2 = np.random.randn(15, 512) + np.array([-1.0] * 512)
    cluster3 = np.random.randn(8, 512) + np.array([0.5] * 512)
    noise = np.random.randn(5, 512) * 3
    
    all_embeddings = np.vstack([cluster1, cluster2, cluster3, noise])
    
    # Cluster
    clustering = FaceClustering(method='dbscan', eps=0.5, min_samples=2)
    labels = clustering.fit(all_embeddings)
    stats = clustering.get_cluster_stats(labels)
    
    print("Clustering Results:")
    print(f"  Number of clusters: {stats['n_clusters']}")
    print(f"  Number of noise points: {stats['n_noise']}")
    print(f"  Cluster sizes: {stats['cluster_sizes']}")
