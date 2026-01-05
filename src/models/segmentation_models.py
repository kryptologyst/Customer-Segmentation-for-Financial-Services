"""
Advanced segmentation models for customer analysis.

This module implements various clustering algorithms and segmentation techniques
including K-means, Gaussian Mixture Models, Hierarchical Clustering, and DBSCAN.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from abc import ABC, abstractmethod
import logging

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

logger = logging.getLogger(__name__)


class BaseSegmenter(ABC):
    """Abstract base class for customer segmentation models."""
    
    def __init__(self, random_state: int = 42):
        """Initialize the segmenter."""
        self.random_state = random_state
        self.model = None
        self.labels_ = None
        self.n_clusters_ = None
        self.is_fitted = False
        
    @abstractmethod
    def fit(self, X: np.ndarray) -> 'BaseSegmenter':
        """Fit the segmentation model."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster labels."""
        pass
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit the model and predict labels."""
        self.fit(X)
        return self.predict(X)
    
    def get_cluster_centers(self) -> Optional[np.ndarray]:
        """Get cluster centers if available."""
        return None
    
    def get_cluster_info(self, X: np.ndarray, feature_names: List[str]) -> pd.DataFrame:
        """Get detailed information about each cluster."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting cluster info")
        
        df = pd.DataFrame(X, columns=feature_names)
        df['cluster'] = self.labels_
        
        cluster_info = df.groupby('cluster').agg({
            **{col: ['mean', 'std', 'count'] for col in feature_names},
            'cluster': 'count'
        }).round(3)
        
        return cluster_info


class KMeansSegmenter(BaseSegmenter):
    """K-means clustering for customer segmentation."""
    
    def __init__(self, n_clusters: int = 5, init: str = 'k-means++', 
                 max_iter: int = 300, random_state: int = 42):
        """Initialize K-means segmenter."""
        super().__init__(random_state)
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.model = KMeans(
            n_clusters=n_clusters,
            init=init,
            max_iter=max_iter,
            random_state=random_state
        )
    
    def fit(self, X: np.ndarray) -> 'KMeansSegmenter':
        """Fit K-means model."""
        logger.info(f"Fitting K-means with {self.n_clusters} clusters")
        self.model.fit(X)
        self.labels_ = self.model.labels_
        self.n_clusters_ = len(np.unique(self.labels_))
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster labels."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)
    
    def get_cluster_centers(self) -> np.ndarray:
        """Get cluster centers."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting cluster centers")
        return self.model.cluster_centers_
    
    def get_inertia(self) -> float:
        """Get inertia (within-cluster sum of squares)."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting inertia")
        return self.model.inertia_


class GaussianMixtureSegmenter(BaseSegmenter):
    """Gaussian Mixture Model for customer segmentation."""
    
    def __init__(self, n_components: int = 5, covariance_type: str = 'full',
                 random_state: int = 42):
        """Initialize GMM segmenter."""
        super().__init__(random_state)
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.model = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            random_state=random_state
        )
    
    def fit(self, X: np.ndarray) -> 'GaussianMixtureSegmenter':
        """Fit GMM model."""
        logger.info(f"Fitting GMM with {self.n_components} components")
        self.model.fit(X)
        self.labels_ = self.model.predict(X)
        self.n_clusters_ = len(np.unique(self.labels_))
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster labels."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict_proba(X)
    
    def get_aic_bic(self) -> Tuple[float, float]:
        """Get AIC and BIC scores."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting AIC/BIC")
        return self.model.aic(X), self.model.bic(X)


class HierarchicalSegmenter(BaseSegmenter):
    """Hierarchical clustering for customer segmentation."""
    
    def __init__(self, n_clusters: int = 5, linkage: str = 'ward',
                 random_state: int = 42):
        """Initialize hierarchical segmenter."""
        super().__init__(random_state)
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.model = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage
        )
    
    def fit(self, X: np.ndarray) -> 'HierarchicalSegmenter':
        """Fit hierarchical clustering model."""
        logger.info(f"Fitting hierarchical clustering with {self.n_clusters} clusters")
        self.model.fit(X)
        self.labels_ = self.model.labels_
        self.n_clusters_ = len(np.unique(self.labels_))
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster labels."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        # Note: AgglomerativeClustering doesn't have predict method
        # This is a limitation - would need to refit or use different approach
        raise NotImplementedError("Hierarchical clustering doesn't support prediction on new data")


class DBSegmenter(BaseSegmenter):
    """DBSCAN clustering for customer segmentation."""
    
    def __init__(self, eps: float = 0.5, min_samples: int = 5):
        """Initialize DBSCAN segmenter."""
        super().__init__()
        self.eps = eps
        self.min_samples = min_samples
        self.model = DBSCAN(eps=eps, min_samples=min_samples)
    
    def fit(self, X: np.ndarray) -> 'DBSegmenter':
        """Fit DBSCAN model."""
        logger.info(f"Fitting DBSCAN with eps={self.eps}, min_samples={self.min_samples}")
        self.model.fit(X)
        self.labels_ = self.model.labels_
        self.n_clusters_ = len(set(self.labels_)) - (1 if -1 in self.labels_ else 0)
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster labels."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        # DBSCAN doesn't have predict method, would need to use nearest neighbors
        raise NotImplementedError("DBSCAN doesn't support prediction on new data")
    
    def get_noise_points(self) -> int:
        """Get number of noise points (outliers)."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting noise points")
        return np.sum(self.labels_ == -1)


class SegmentationEnsemble:
    """Ensemble of segmentation models for robust clustering."""
    
    def __init__(self, models: List[BaseSegmenter]):
        """Initialize ensemble with list of models."""
        self.models = models
        self.consensus_labels_ = None
        self.is_fitted = False
    
    def fit(self, X: np.ndarray) -> 'SegmentationEnsemble':
        """Fit all models in the ensemble."""
        logger.info(f"Fitting ensemble of {len(self.models)} models")
        
        for model in self.models:
            model.fit(X)
        
        # Create consensus labels using majority voting
        all_labels = np.array([model.labels_ for model in self.models]).T
        self.consensus_labels_ = self._majority_vote(all_labels)
        
        self.is_fitted = True
        return self
    
    def _majority_vote(self, labels_matrix: np.ndarray) -> np.ndarray:
        """Perform majority voting on cluster labels."""
        consensus = []
        for labels in labels_matrix:
            unique, counts = np.unique(labels, return_counts=True)
            consensus.append(unique[np.argmax(counts)])
        return np.array(consensus)
    
    def get_model_scores(self, X: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Get evaluation scores for each model."""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before getting scores")
        
        scores = {}
        for i, model in enumerate(self.models):
            model_name = f"model_{i}_{type(model).__name__}"
            scores[model_name] = self._evaluate_model(X, model.labels_)
        
        return scores
    
    def _evaluate_model(self, X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Evaluate a single model."""
        n_clusters = len(np.unique(labels))
        
        if n_clusters < 2:
            return {"silhouette": 0, "calinski_harabasz": 0, "davies_bouldin": float('inf')}
        
        try:
            silhouette = silhouette_score(X, labels)
            calinski_harabasz = calinski_harabasz_score(X, labels)
            davies_bouldin = davies_bouldin_score(X, labels)
            
            return {
                "silhouette": silhouette,
                "calinski_harabasz": calinski_harabasz,
                "davies_bouldin": davies_bouldin
            }
        except Exception as e:
            logger.warning(f"Error evaluating model: {e}")
            return {"silhouette": 0, "calinski_harabasz": 0, "davies_bouldin": float('inf')}


class SegmentationEvaluator:
    """Evaluates segmentation models using various metrics."""
    
    @staticmethod
    def evaluate_clustering(X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Evaluate clustering results."""
        n_clusters = len(np.unique(labels))
        
        if n_clusters < 2:
            return {
                "silhouette_score": 0,
                "calinski_harabasz_score": 0,
                "davies_bouldin_score": float('inf'),
                "n_clusters": n_clusters
            }
        
        try:
            silhouette = silhouette_score(X, labels)
            calinski_harabasz = calinski_harabasz_score(X, labels)
            davies_bouldin = davies_bouldin_score(X, labels)
            
            return {
                "silhouette_score": silhouette,
                "calinski_harabasz_score": calinski_harabasz,
                "davies_bouldin_score": davies_bouldin,
                "n_clusters": n_clusters
            }
        except Exception as e:
            logger.warning(f"Error evaluating clustering: {e}")
            return {
                "silhouette_score": 0,
                "calinski_harabasz_score": 0,
                "davies_bouldin_score": float('inf'),
                "n_clusters": n_clusters
            }
    
    @staticmethod
    def find_optimal_clusters(X: np.ndarray, max_clusters: int = 10, 
                            method: str = "kmeans") -> Dict[str, Any]:
        """Find optimal number of clusters using elbow method and silhouette analysis."""
        logger.info(f"Finding optimal clusters for {method}")
        
        silhouette_scores = []
        inertias = []
        cluster_range = range(2, max_clusters + 1)
        
        for n_clusters in cluster_range:
            if method == "kmeans":
                model = KMeans(n_clusters=n_clusters, random_state=42)
            elif method == "gmm":
                model = GaussianMixture(n_components=n_clusters, random_state=42)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            labels = model.fit_predict(X)
            
            # Calculate silhouette score
            if len(np.unique(labels)) > 1:
                silhouette = silhouette_score(X, labels)
                silhouette_scores.append(silhouette)
            else:
                silhouette_scores.append(0)
            
            # Calculate inertia (for K-means)
            if method == "kmeans":
                inertias.append(model.inertia_)
        
        # Find optimal number of clusters
        optimal_silhouette = cluster_range[np.argmax(silhouette_scores)]
        
        result = {
            "optimal_clusters_silhouette": optimal_silhouette,
            "silhouette_scores": silhouette_scores,
            "cluster_range": list(cluster_range)
        }
        
        if method == "kmeans":
            result["inertias"] = inertias
        
        return result


def create_segmenter(model_type: str, **kwargs) -> BaseSegmenter:
    """Factory function to create segmenters."""
    segmenters = {
        "kmeans": KMeansSegmenter,
        "gmm": GaussianMixtureSegmenter,
        "hierarchical": HierarchicalSegmenter,
        "dbscan": DBSegmenter
    }
    
    if model_type not in segmenters:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    return segmenters[model_type](**kwargs)
