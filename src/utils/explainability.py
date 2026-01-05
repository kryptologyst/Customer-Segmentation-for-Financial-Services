"""
Explainability module for customer segmentation.

This module provides SHAP explanations, feature importance analysis,
and interpretability tools for customer segmentation models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP not available. Install with: pip install shap")

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class SegmentationExplainer:
    """Provides explanations for customer segmentation models."""
    
    def __init__(self, random_state: int = 42):
        """Initialize the explainer."""
        self.random_state = random_state
        self.explainer = None
        self.feature_names = None
        self.is_fitted = False
        
    def fit_explainer(self, X: np.ndarray, labels: np.ndarray, 
                     feature_names: List[str], method: str = "tree") -> 'SegmentationExplainer':
        """Fit the explainer model."""
        logger.info(f"Fitting explainer using {method} method")
        
        self.feature_names = feature_names
        
        if method == "tree" and SHAP_AVAILABLE:
            # Use tree explainer with random forest
            rf = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
            rf.fit(X, labels)
            self.explainer = shap.TreeExplainer(rf)
            self.is_fitted = True
            
        elif method == "kernel" and SHAP_AVAILABLE:
            # Use kernel explainer
            rf = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
            rf.fit(X, labels)
            self.explainer = shap.KernelExplainer(rf.predict_proba, X[:100])  # Sample for speed
            self.is_fitted = True
            
        else:
            logger.warning(f"Method {method} not available or SHAP not installed")
            self.explainer = None
            self.is_fitted = False
        
        return self
    
    def get_feature_importance(self, X: np.ndarray, labels: np.ndarray) -> pd.Series:
        """Get feature importance using random forest."""
        logger.info("Calculating feature importance")
        
        rf = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        rf.fit(X, labels)
        
        importance = pd.Series(rf.feature_importances_, index=self.feature_names)
        return importance.sort_values(ascending=False)
    
    def get_shap_values(self, X: np.ndarray, max_samples: int = 1000) -> Optional[np.ndarray]:
        """Get SHAP values for the given data."""
        if not self.is_fitted or self.explainer is None:
            logger.warning("Explainer not fitted or SHAP not available")
            return None
        
        # Limit samples for performance
        if len(X) > max_samples:
            indices = np.random.choice(len(X), max_samples, replace=False)
            X_sample = X[indices]
        else:
            X_sample = X
        
        try:
            shap_values = self.explainer.shap_values(X_sample)
            return shap_values
        except Exception as e:
            logger.error(f"Error calculating SHAP values: {e}")
            return None
    
    def create_feature_importance_plot(self, importance: pd.Series, 
                                     top_n: int = 15) -> plt.Figure:
        """Create feature importance plot."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        top_features = importance.head(top_n)
        
        sns.barplot(x=top_features.values, y=top_features.index, ax=ax)
        ax.set_title(f'Top {top_n} Most Important Features for Segmentation')
        ax.set_xlabel('Feature Importance')
        
        plt.tight_layout()
        return fig
    
    def create_shap_summary_plot(self, X: np.ndarray, shap_values: np.ndarray, 
                                max_display: int = 15) -> Optional[plt.Figure]:
        """Create SHAP summary plot."""
        if not SHAP_AVAILABLE or shap_values is None:
            logger.warning("SHAP not available or values not calculated")
            return None
        
        try:
            fig = plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X, feature_names=self.feature_names, 
                            max_display=max_display, show=False)
            plt.title('SHAP Summary Plot for Customer Segmentation')
            return fig
        except Exception as e:
            logger.error(f"Error creating SHAP summary plot: {e}")
            return None
    
    def create_segment_explanations(self, X: np.ndarray, labels: np.ndarray, 
                                  shap_values: Optional[np.ndarray] = None) -> Dict[int, Dict[str, Any]]:
        """Create explanations for each segment."""
        logger.info("Creating segment explanations")
        
        explanations = {}
        
        for segment in np.unique(labels):
            segment_mask = labels == segment
            segment_data = X[segment_mask]
            
            # Calculate segment statistics
            segment_stats = {
                'size': np.sum(segment_mask),
                'percentage': np.sum(segment_mask) / len(labels) * 100,
                'mean_features': np.mean(segment_data, axis=0),
                'std_features': np.std(segment_data, axis=0)
            }
            
            # Add SHAP values if available
            if shap_values is not None:
                segment_shap = shap_values[segment_mask]
                segment_stats['mean_shap'] = np.mean(segment_shap, axis=0)
                segment_stats['std_shap'] = np.std(segment_shap, axis=0)
            
            explanations[segment] = segment_stats
        
        return explanations
    
    def create_segment_comparison_plot(self, explanations: Dict[int, Dict[str, Any]], 
                                      top_n: int = 10) -> plt.Figure:
        """Create comparison plot between segments."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        segments = list(explanations.keys())
        
        # Feature means comparison
        feature_means = pd.DataFrame({
            f'Segment {seg}': explanations[seg]['mean_features'] 
            for seg in segments
        }, index=self.feature_names)
        
        # Top features by variance across segments
        feature_variance = feature_means.var(axis=1).sort_values(ascending=False)
        top_features = feature_variance.head(top_n).index
        
        # Plot 1: Feature means heatmap
        sns.heatmap(feature_means.loc[top_features].T, annot=True, fmt='.2f', 
                   ax=axes[0, 0], cmap='viridis')
        axes[0, 0].set_title('Feature Means by Segment')
        
        # Plot 2: Segment sizes
        segment_sizes = [explanations[seg]['size'] for seg in segments]
        sns.barplot(x=[f'Segment {seg}' for seg in segments], y=segment_sizes, ax=axes[0, 1])
        axes[0, 1].set_title('Segment Sizes')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Plot 3: Feature importance by segment (if SHAP available)
        if SHAP_AVAILABLE and 'mean_shap' in explanations[segments[0]]:
            shap_means = pd.DataFrame({
                f'Segment {seg}': explanations[seg]['mean_shap'] 
                for seg in segments
            }, index=self.feature_names)
            
            sns.heatmap(shap_means.loc[top_features].T, annot=True, fmt='.2f', 
                       ax=axes[1, 0], cmap='RdBu_r', center=0)
            axes[1, 0].set_title('SHAP Values by Segment')
        
        # Plot 4: Segment characteristics radar chart (simplified)
        if len(segments) <= 4:  # Limit for readability
            for i, seg in enumerate(segments[:4]):
                means = explanations[seg]['mean_features'][:6]  # Top 6 features
                axes[1, 1].plot(range(len(means)), means, marker='o', label=f'Segment {seg}')
            
            axes[1, 1].set_title('Segment Characteristics (Top 6 Features)')
            axes[1, 1].set_xlabel('Feature Index')
            axes[1, 1].set_ylabel('Mean Value')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        return fig


class CustomerProfileExplainer:
    """Explains individual customer profiles and segment assignments."""
    
    def __init__(self, model, scaler, feature_names: List[str]):
        """Initialize with trained model and scaler."""
        self.model = model
        self.scaler = scaler
        self.feature_names = feature_names
    
    def explain_customer(self, customer_data: np.ndarray, customer_id: str = None) -> Dict[str, Any]:
        """Explain why a customer was assigned to a specific segment."""
        logger.info(f"Explaining customer {customer_id}")
        
        # Scale the customer data
        customer_scaled = self.scaler.transform(customer_data.reshape(1, -1))
        
        # Get segment assignment
        segment = self.model.predict(customer_scaled)[0]
        
        # Get feature values
        feature_values = dict(zip(self.feature_names, customer_data))
        
        # Calculate feature importance for this customer (if model supports it)
        explanation = {
            'customer_id': customer_id,
            'assigned_segment': int(segment),
            'feature_values': feature_values,
            'scaled_values': dict(zip(self.feature_names, customer_scaled[0]))
        }
        
        # Add SHAP explanation if available
        if hasattr(self.model, 'predict_proba'):
            segment_probabilities = self.model.predict_proba(customer_scaled)[0]
            explanation['segment_probabilities'] = dict(enumerate(segment_probabilities))
        
        return explanation
    
    def create_customer_profile_plot(self, explanation: Dict[str, Any], 
                                   top_n: int = 10) -> plt.Figure:
        """Create visualization for individual customer profile."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        customer_id = explanation.get('customer_id', 'Unknown')
        segment = explanation['assigned_segment']
        feature_values = explanation['feature_values']
        
        # Plot 1: Feature values bar chart
        features = list(feature_values.keys())[:top_n]
        values = list(feature_values.values())[:top_n]
        
        sns.barplot(x=values, y=features, ax=axes[0, 0])
        axes[0, 0].set_title(f'Customer {customer_id} - Feature Values')
        
        # Plot 2: Segment probabilities (if available)
        if 'segment_probabilities' in explanation:
            probs = explanation['segment_probabilities']
            segments = list(probs.keys())
            probabilities = list(probs.values())
            
            sns.barplot(x=segments, y=probabilities, ax=axes[0, 1])
            axes[0, 1].set_title('Segment Assignment Probabilities')
            axes[0, 1].set_xlabel('Segment')
            axes[0, 1].set_ylabel('Probability')
        
        # Plot 3: Scaled values
        scaled_values = explanation['scaled_values']
        scaled_features = list(scaled_values.keys())[:top_n]
        scaled_vals = list(scaled_values.values())[:top_n]
        
        sns.barplot(x=scaled_vals, y=scaled_features, ax=axes[1, 0])
        axes[1, 0].set_title('Scaled Feature Values')
        
        # Plot 4: Summary
        axes[1, 1].text(0.1, 0.8, f'Customer ID: {customer_id}', fontsize=12, weight='bold')
        axes[1, 1].text(0.1, 0.7, f'Assigned Segment: {segment}', fontsize=12)
        axes[1, 1].text(0.1, 0.6, f'Number of Features: {len(feature_values)}', fontsize=12)
        
        if 'segment_probabilities' in explanation:
            max_prob = max(explanation['segment_probabilities'].values())
            axes[1, 1].text(0.1, 0.5, f'Max Probability: {max_prob:.3f}', fontsize=12)
        
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        return fig


def create_explainability_report(df: pd.DataFrame, labels: np.ndarray, 
                               model, scaler, feature_names: List[str]) -> Dict[str, Any]:
    """Create comprehensive explainability report."""
    logger.info("Creating explainability report")
    
    # Initialize explainer
    explainer = SegmentationExplainer()
    explainer.fit_explainer(df[feature_names].values, labels, feature_names)
    
    # Get feature importance
    feature_importance = explainer.get_feature_importance(df[feature_names].values, labels)
    
    # Get SHAP values
    shap_values = explainer.get_shap_values(df[feature_names].values)
    
    # Create segment explanations
    segment_explanations = explainer.create_segment_explanations(
        df[feature_names].values, labels, shap_values
    )
    
    # Create plots
    importance_plot = explainer.create_feature_importance_plot(feature_importance)
    comparison_plot = explainer.create_segment_comparison_plot(segment_explanations)
    
    shap_summary_plot = None
    if shap_values is not None:
        shap_summary_plot = explainer.create_shap_summary_plot(
            df[feature_names].values, shap_values
        )
    
    report = {
        'feature_importance': feature_importance.to_dict(),
        'segment_explanations': segment_explanations,
        'plots': {
            'importance_plot': importance_plot,
            'comparison_plot': comparison_plot,
            'shap_summary_plot': shap_summary_plot
        },
        'summary': {
            'top_features': feature_importance.head(10).to_dict(),
            'n_segments': len(np.unique(labels)),
            'total_customers': len(df)
        }
    }
    
    return report
