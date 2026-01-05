"""
Evaluation module for customer segmentation.

This module provides comprehensive evaluation metrics for customer segmentation
including both ML metrics and business metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging

from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)


class SegmentationEvaluator:
    """Comprehensive evaluator for customer segmentation models."""
    
    def __init__(self):
        """Initialize the evaluator."""
        self.results = {}
    
    def evaluate_ml_metrics(self, X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Evaluate machine learning metrics for clustering."""
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
            logger.warning(f"Error evaluating ML metrics: {e}")
            return {
                "silhouette_score": 0,
                "calinski_harabasz_score": 0,
                "davies_bouldin_score": float('inf'),
                "n_clusters": n_clusters
            }
    
    def evaluate_business_metrics(self, df: pd.DataFrame, labels: np.ndarray, 
                                segment_column: str = 'segment') -> Dict[str, Any]:
        """Evaluate business metrics for customer segments."""
        logger.info("Evaluating business metrics")
        
        df_eval = df.copy()
        df_eval[segment_column] = labels
        
        business_metrics = {}
        
        # Segment size distribution
        segment_sizes = df_eval[segment_column].value_counts().sort_index()
        business_metrics['segment_sizes'] = segment_sizes.to_dict()
        business_metrics['segment_size_std'] = segment_sizes.std()
        business_metrics['segment_size_cv'] = segment_sizes.std() / segment_sizes.mean()
        
        # Segment profitability analysis
        if 'customer_lifetime_value' in df_eval.columns:
            clv_by_segment = df_eval.groupby(segment_column)['customer_lifetime_value'].agg([
                'mean', 'std', 'sum', 'count'
            ]).round(2)
            business_metrics['clv_by_segment'] = clv_by_segment.to_dict()
            
            # Total CLV by segment
            total_clv_by_segment = df_eval.groupby(segment_column)['customer_lifetime_value'].sum()
            business_metrics['total_clv_by_segment'] = total_clv_by_segment.to_dict()
            
            # CLV concentration (Gini coefficient)
            business_metrics['clv_concentration'] = self._calculate_gini_coefficient(total_clv_by_segment.values)
        
        # Segment characteristics
        numeric_cols = df_eval.select_dtypes(include=[np.number]).columns
        segment_characteristics = {}
        
        for col in numeric_cols:
            if col != segment_column:
                segment_stats = df_eval.groupby(segment_column)[col].agg(['mean', 'std', 'min', 'max']).round(2)
                segment_characteristics[col] = segment_stats.to_dict()
        
        business_metrics['segment_characteristics'] = segment_characteristics
        
        # Churn analysis by segment
        if 'is_churned' in df_eval.columns:
            churn_by_segment = df_eval.groupby(segment_column)['is_churned'].agg(['mean', 'sum', 'count'])
            churn_by_segment['churn_rate'] = churn_by_segment['mean']
            business_metrics['churn_by_segment'] = churn_by_segment.to_dict()
        
        # Segment diversity (entropy)
        business_metrics['segment_diversity'] = self._calculate_entropy(segment_sizes.values)
        
        return business_metrics
    
    def evaluate_segment_stability(self, df: pd.DataFrame, labels1: np.ndarray, 
                                 labels2: np.ndarray) -> Dict[str, float]:
        """Evaluate stability between two segmentations."""
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
        
        ari = adjusted_rand_score(labels1, labels2)
        nmi = normalized_mutual_info_score(labels1, labels2)
        
        return {
            "adjusted_rand_index": ari,
            "normalized_mutual_info": nmi,
            "stability_score": (ari + nmi) / 2
        }
    
    def evaluate_feature_importance(self, df: pd.DataFrame, labels: np.ndarray) -> Dict[str, float]:
        """Evaluate feature importance for segmentation."""
        from sklearn.ensemble import RandomForestClassifier
        
        # Select numeric features
        feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'customer_id' in feature_cols:
            feature_cols.remove('customer_id')
        
        X = df[feature_cols]
        y = labels
        
        # Train random forest to predict segments
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        # Get feature importance
        importance = pd.Series(rf.feature_importances_, index=feature_cols)
        importance = importance.sort_values(ascending=False)
        
        return importance.to_dict()
    
    def _calculate_gini_coefficient(self, values: np.ndarray) -> float:
        """Calculate Gini coefficient for inequality measurement."""
        if len(values) == 0:
            return 0
        
        values = np.sort(values)
        n = len(values)
        cumsum = np.cumsum(values)
        
        return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
    
    def _calculate_entropy(self, values: np.ndarray) -> float:
        """Calculate entropy for diversity measurement."""
        if len(values) == 0:
            return 0
        
        probabilities = values / np.sum(values)
        probabilities = probabilities[probabilities > 0]  # Remove zeros
        
        return -np.sum(probabilities * np.log2(probabilities))
    
    def create_segment_profiles(self, df: pd.DataFrame, labels: np.ndarray, 
                              segment_column: str = 'segment') -> pd.DataFrame:
        """Create detailed profiles for each segment."""
        df_profiles = df.copy()
        df_profiles[segment_column] = labels
        
        # Calculate comprehensive statistics for each segment
        numeric_cols = df_profiles.select_dtypes(include=[np.number]).columns
        if segment_column in numeric_cols:
            numeric_cols = numeric_cols.drop(segment_column)
        
        segment_profiles = df_profiles.groupby(segment_column)[numeric_cols].agg([
            'count', 'mean', 'std', 'min', 'max', 'median'
        ]).round(2)
        
        # Add categorical column summaries
        categorical_cols = df_profiles.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if col != segment_column:
                cat_summary = df_profiles.groupby(segment_column)[col].value_counts().unstack(fill_value=0)
                cat_summary.columns = [f"{col}_{cat}" for cat in cat_summary.columns]
                segment_profiles = pd.concat([segment_profiles, cat_summary], axis=1)
        
        return segment_profiles
    
    def generate_evaluation_report(self, df: pd.DataFrame, labels: np.ndarray, 
                                 model_name: str = "Unknown") -> Dict[str, Any]:
        """Generate comprehensive evaluation report."""
        logger.info(f"Generating evaluation report for {model_name}")
        
        # Prepare data for evaluation
        feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'customer_id' in feature_cols:
            feature_cols.remove('customer_id')
        
        X = df[feature_cols].values
        
        # Evaluate all metrics
        ml_metrics = self.evaluate_ml_metrics(X, labels)
        business_metrics = self.evaluate_business_metrics(df, labels)
        feature_importance = self.evaluate_feature_importance(df, labels)
        segment_profiles = self.create_segment_profiles(df, labels)
        
        report = {
            "model_name": model_name,
            "ml_metrics": ml_metrics,
            "business_metrics": business_metrics,
            "feature_importance": feature_importance,
            "segment_profiles": segment_profiles.to_dict(),
            "evaluation_summary": self._create_summary(ml_metrics, business_metrics)
        }
        
        return report
    
    def _create_summary(self, ml_metrics: Dict[str, float], 
                       business_metrics: Dict[str, Any]) -> Dict[str, str]:
        """Create human-readable summary of evaluation results."""
        summary = {}
        
        # ML metrics summary
        silhouette = ml_metrics.get('silhouette_score', 0)
        if silhouette > 0.7:
            summary['clustering_quality'] = "Excellent"
        elif silhouette > 0.5:
            summary['clustering_quality'] = "Good"
        elif silhouette > 0.3:
            summary['clustering_quality'] = "Fair"
        else:
            summary['clustering_quality'] = "Poor"
        
        # Business metrics summary
        n_segments = ml_metrics.get('n_clusters', 0)
        summary['n_segments'] = f"{n_segments} customer segments identified"
        
        if 'segment_size_cv' in business_metrics:
            cv = business_metrics['segment_size_cv']
            if cv < 0.3:
                summary['segment_balance'] = "Well-balanced segments"
            elif cv < 0.6:
                summary['segment_balance'] = "Moderately balanced segments"
            else:
                summary['segment_balance'] = "Unbalanced segments"
        
        if 'clv_concentration' in business_metrics:
            concentration = business_metrics['clv_concentration']
            if concentration < 0.3:
                summary['clv_distribution'] = "CLV well-distributed across segments"
            elif concentration < 0.6:
                summary['clv_distribution'] = "CLV moderately concentrated"
            else:
                summary['clv_distribution'] = "CLV highly concentrated"
        
        return summary


class SegmentationComparator:
    """Compare multiple segmentation models."""
    
    def __init__(self):
        """Initialize the comparator."""
        self.results = {}
    
    def compare_models(self, df: pd.DataFrame, model_results: Dict[str, np.ndarray]) -> pd.DataFrame:
        """Compare multiple segmentation models."""
        logger.info(f"Comparing {len(model_results)} models")
        
        comparison_results = []
        
        for model_name, labels in model_results.items():
            evaluator = SegmentationEvaluator()
            report = evaluator.generate_evaluation_report(df, labels, model_name)
            
            # Extract key metrics for comparison
            ml_metrics = report['ml_metrics']
            business_metrics = report['business_metrics']
            
            comparison_results.append({
                'model': model_name,
                'silhouette_score': ml_metrics['silhouette_score'],
                'calinski_harabasz_score': ml_metrics['calinski_harabasz_score'],
                'davies_bouldin_score': ml_metrics['davies_bouldin_score'],
                'n_clusters': ml_metrics['n_clusters'],
                'segment_size_cv': business_metrics.get('segment_size_cv', 0),
                'clv_concentration': business_metrics.get('clv_concentration', 0),
                'segment_diversity': business_metrics.get('segment_diversity', 0)
            })
        
        comparison_df = pd.DataFrame(comparison_results)
        
        # Add rankings
        comparison_df['silhouette_rank'] = comparison_df['silhouette_score'].rank(ascending=False)
        comparison_df['calinski_harabasz_rank'] = comparison_df['calinski_harabasz_score'].rank(ascending=False)
        comparison_df['davies_bouldin_rank'] = comparison_df['davies_bouldin_score'].rank(ascending=True)
        
        # Overall ranking (lower is better)
        comparison_df['overall_rank'] = (
            comparison_df['silhouette_rank'] + 
            comparison_df['calinski_harabasz_rank'] + 
            comparison_df['davies_bouldin_rank']
        ) / 3
        
        return comparison_df.sort_values('overall_rank')
    
    def create_comparison_plot(self, comparison_df: pd.DataFrame) -> Dict[str, Any]:
        """Create comparison plots for model evaluation."""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Silhouette scores
        sns.barplot(data=comparison_df, x='model', y='silhouette_score', ax=axes[0, 0])
        axes[0, 0].set_title('Silhouette Scores by Model')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Calinski-Harabasz scores
        sns.barplot(data=comparison_df, x='model', y='calinski_harabasz_score', ax=axes[0, 1])
        axes[0, 1].set_title('Calinski-Harabasz Scores by Model')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Davies-Bouldin scores
        sns.barplot(data=comparison_df, x='model', y='davies_bouldin_score', ax=axes[1, 0])
        axes[1, 0].set_title('Davies-Bouldin Scores by Model')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Overall ranking
        sns.barplot(data=comparison_df, x='model', y='overall_rank', ax=axes[1, 1])
        axes[1, 1].set_title('Overall Ranking by Model')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        return {
            'figure': fig,
            'axes': axes
        }
