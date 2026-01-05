"""
Main execution script for customer segmentation analysis.

This script orchestrates the entire customer segmentation pipeline including
data generation, feature engineering, modeling, evaluation, and visualization.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.data_generator import CustomerDataGenerator, CustomerDataConfig, load_customer_data
from features.feature_engineering import FeatureEngineer
from models.segmentation_models import (
    KMeansSegmenter, GaussianMixtureSegmenter, HierarchicalSegmenter, 
    SegmentationEnsemble, SegmentationEvaluator as ModelEvaluator
)
from evaluation.evaluation import SegmentationEvaluator, SegmentationComparator
from utils.explainability import create_explainability_report

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CustomerSegmentationPipeline:
    """Main pipeline for customer segmentation analysis."""
    
    def __init__(self, config_path: str):
        """Initialize pipeline with configuration."""
        self.config = self._load_config(config_path)
        self.data = None
        self.features = None
        self.models = {}
        self.results = {}
        self.evaluator = SegmentationEvaluator()
        
        # Set random seeds for reproducibility
        np.random.seed(self.config['data']['random_seed'])
        
        # Create output directories
        self._create_output_dirs()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    
    def _create_output_dirs(self):
        """Create output directories for results."""
        dirs = [
            self.config['output']['results_dir'],
            self.config['output']['models_dir'],
            self.config['output']['logs_dir'],
            self.config['visualization']['output_dir']
        ]
        
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def generate_data(self) -> pd.DataFrame:
        """Generate or load customer data."""
        logger.info("Generating customer data")
        
        # Create data configuration
        data_config = CustomerDataConfig(
            n_customers=self.config['data']['n_customers'],
            random_seed=self.config['data']['random_seed']
        )
        
        # Generate data
        generator = CustomerDataGenerator(data_config)
        self.data = generator.generate_complete_dataset()
        
        logger.info(f"Generated dataset with {len(self.data)} customers and {len(self.data.columns)} features")
        return self.data
    
    def engineer_features(self) -> pd.DataFrame:
        """Engineer features for segmentation."""
        logger.info("Engineering features")
        
        if self.data is None:
            raise ValueError("Data must be generated first")
        
        # Initialize feature engineer
        feature_engineer = FeatureEngineer(
            scaling_method=self.config['features']['scaling_method']
        )
        
        # Create all engineered features
        self.features = feature_engineer.create_all_features(self.data)
        
        # Select features for modeling
        self.features = feature_engineer.select_features(
            self.features, 
            target_column='is_churned',
            method='numeric'
        )
        
        logger.info(f"Created {len(self.features.columns)} features for modeling")
        return self.features
    
    def train_models(self) -> Dict[str, Any]:
        """Train multiple segmentation models."""
        logger.info("Training segmentation models")
        
        if self.features is None:
            raise ValueError("Features must be engineered first")
        
        # Prepare data for modeling
        feature_cols = [col for col in self.features.columns if col != 'customer_id']
        X = self.features[feature_cols].values
        
        # Scale features
        feature_engineer = FeatureEngineer(self.config['features']['scaling_method'])
        X_scaled = feature_engineer.scale_features(
            self.features[feature_cols + ['customer_id']], 
            fit=True
        )
        X_scaled = X_scaled[feature_cols].values
        
        # Train different models
        models_config = self.config['models']
        
        # K-means
        kmeans = KMeansSegmenter(
            n_clusters=models_config['kmeans']['n_clusters'],
            init=models_config['kmeans']['init'],
            max_iter=models_config['kmeans']['max_iter'],
            random_state=models_config['kmeans']['random_state']
        )
        kmeans_labels = kmeans.fit_predict(X_scaled)
        self.models['kmeans'] = {
            'model': kmeans,
            'labels': kmeans_labels,
            'scaler': feature_engineer.scaler
        }
        
        # Gaussian Mixture Model
        gmm = GaussianMixtureSegmenter(
            n_components=models_config['gmm']['n_components'],
            covariance_type=models_config['gmm']['covariance_type'],
            random_state=models_config['gmm']['random_state']
        )
        gmm_labels = gmm.fit_predict(X_scaled)
        self.models['gmm'] = {
            'model': gmm,
            'labels': gmm_labels,
            'scaler': feature_engineer.scaler
        }
        
        # Hierarchical Clustering
        hierarchical = HierarchicalSegmenter(
            n_clusters=models_config['hierarchical']['n_clusters'],
            linkage=models_config['hierarchical']['linkage']
        )
        hierarchical_labels = hierarchical.fit_predict(X_scaled)
        self.models['hierarchical'] = {
            'model': hierarchical,
            'labels': hierarchical_labels,
            'scaler': feature_engineer.scaler
        }
        
        logger.info(f"Trained {len(self.models)} segmentation models")
        return self.models
    
    def evaluate_models(self) -> Dict[str, Any]:
        """Evaluate all trained models."""
        logger.info("Evaluating models")
        
        if not self.models:
            raise ValueError("Models must be trained first")
        
        # Prepare evaluation data
        feature_cols = [col for col in self.features.columns if col != 'customer_id']
        X = self.features[feature_cols].values
        
        # Evaluate each model
        evaluation_results = {}
        
        for model_name, model_data in self.models.items():
            logger.info(f"Evaluating {model_name}")
            
            labels = model_data['labels']
            
            # Generate evaluation report
            report = self.evaluator.generate_evaluation_report(
                self.features, labels, model_name
            )
            
            evaluation_results[model_name] = report
        
        # Compare models
        comparator = SegmentationComparator()
        model_labels = {name: data['labels'] for name, data in self.models.items()}
        comparison_df = comparator.compare_models(self.features, model_labels)
        
        self.results = {
            'evaluation_results': evaluation_results,
            'model_comparison': comparison_df,
            'best_model': comparison_df.iloc[0]['model']
        }
        
        logger.info(f"Best model: {self.results['best_model']}")
        return self.results
    
    def create_visualizations(self):
        """Create comprehensive visualizations."""
        logger.info("Creating visualizations")
        
        if not self.results:
            raise ValueError("Models must be evaluated first")
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Create comparison plots
        self._create_model_comparison_plots()
        
        # Create segment analysis plots
        self._create_segment_analysis_plots()
        
        # Create feature importance plots
        self._create_feature_importance_plots()
        
        logger.info("Visualizations created and saved")
    
    def _create_model_comparison_plots(self):
        """Create model comparison visualizations."""
        comparison_df = self.results['model_comparison']
        
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
        plt.savefig(f"{self.config['visualization']['output_dir']}/model_comparison.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_segment_analysis_plots(self):
        """Create segment analysis visualizations."""
        best_model_name = self.results['best_model']
        best_model_data = self.models[best_model_name]
        labels = best_model_data['labels']
        
        # Add labels to data
        df_with_labels = self.features.copy()
        df_with_labels['segment'] = labels
        
        # Create segment analysis plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Segment size distribution
        segment_sizes = df_with_labels['segment'].value_counts().sort_index()
        sns.barplot(x=segment_sizes.index, y=segment_sizes.values, ax=axes[0, 0])
        axes[0, 0].set_title('Segment Size Distribution')
        axes[0, 0].set_xlabel('Segment')
        axes[0, 0].set_ylabel('Number of Customers')
        
        # CLV by segment
        if 'customer_lifetime_value' in df_with_labels.columns:
            sns.boxplot(data=df_with_labels, x='segment', y='customer_lifetime_value', ax=axes[0, 1])
            axes[0, 1].set_title('Customer Lifetime Value by Segment')
            axes[0, 1].set_xlabel('Segment')
            axes[0, 1].set_ylabel('CLV')
        
        # Account balance by segment
        if 'account_balance' in df_with_labels.columns:
            sns.boxplot(data=df_with_labels, x='segment', y='account_balance', ax=axes[1, 0])
            axes[1, 0].set_title('Account Balance by Segment')
            axes[1, 0].set_xlabel('Segment')
            axes[1, 0].set_ylabel('Account Balance')
        
        # Transaction count by segment
        if 'num_transactions' in df_with_labels.columns:
            sns.boxplot(data=df_with_labels, x='segment', y='num_transactions', ax=axes[1, 1])
            axes[1, 1].set_title('Transaction Count by Segment')
            axes[1, 1].set_xlabel('Segment')
            axes[1, 1].set_ylabel('Number of Transactions')
        
        plt.tight_layout()
        plt.savefig(f"{self.config['visualization']['output_dir']}/segment_analysis.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_feature_importance_plots(self):
        """Create feature importance visualizations."""
        best_model_name = self.results['best_model']
        best_model_data = self.models[best_model_name]
        labels = best_model_data['labels']
        
        # Get feature importance
        feature_cols = [col for col in self.features.columns if col != 'customer_id']
        X = self.features[feature_cols].values
        
        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, labels)
        
        importance = pd.Series(rf.feature_importances_, index=feature_cols)
        importance = importance.sort_values(ascending=False)
        
        # Create plot
        plt.figure(figsize=(12, 8))
        sns.barplot(x=importance.head(15).values, y=importance.head(15).index)
        plt.title('Top 15 Most Important Features for Segmentation')
        plt.xlabel('Feature Importance')
        plt.tight_layout()
        plt.savefig(f"{self.config['visualization']['output_dir']}/feature_importance.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_explainability_report(self):
        """Generate explainability report for the best model."""
        logger.info("Generating explainability report")
        
        best_model_name = self.results['best_model']
        best_model_data = self.models[best_model_name]
        
        feature_cols = [col for col in self.features.columns if col != 'customer_id']
        
        # Create explainability report
        explainability_report = create_explainability_report(
            self.features,
            best_model_data['labels'],
            best_model_data['model'],
            best_model_data['scaler'],
            feature_cols
        )
        
        # Save plots
        plots_dir = Path(self.config['visualization']['output_dir']) / 'explainability'
        plots_dir.mkdir(exist_ok=True)
        
        if explainability_report['plots']['importance_plot']:
            explainability_report['plots']['importance_plot'].savefig(
                plots_dir / 'feature_importance.png', dpi=300, bbox_inches='tight'
            )
        
        if explainability_report['plots']['comparison_plot']:
            explainability_report['plots']['comparison_plot'].savefig(
                plots_dir / 'segment_comparison.png', dpi=300, bbox_inches='tight'
            )
        
        self.results['explainability'] = explainability_report
        logger.info("Explainability report generated")
    
    def save_results(self):
        """Save all results to files."""
        logger.info("Saving results")
        
        # Save model comparison
        comparison_df = self.results['model_comparison']
        comparison_df.to_csv(f"{self.config['output']['results_dir']}/model_comparison.csv", index=False)
        
        # Save evaluation results
        for model_name, report in self.results['evaluation_results'].items():
            # Save segment profiles
            segment_profiles = pd.DataFrame(report['segment_profiles'])
            segment_profiles.to_csv(f"{self.config['output']['results_dir']}/{model_name}_segment_profiles.csv")
            
            # Save evaluation summary
            summary = report['evaluation_summary']
            with open(f"{self.config['output']['results_dir']}/{model_name}_summary.txt", 'w') as f:
                for key, value in summary.items():
                    f.write(f"{key}: {value}\n")
        
        # Save best model labels
        best_model_name = self.results['best_model']
        best_model_data = self.models[best_model_name]
        
        df_with_labels = self.features.copy()
        df_with_labels['segment'] = best_model_data['labels']
        df_with_labels.to_csv(f"{self.config['output']['results_dir']}/best_model_segments.csv", index=False)
        
        logger.info("Results saved successfully")
    
    def run_full_pipeline(self):
        """Run the complete customer segmentation pipeline."""
        logger.info("Starting customer segmentation pipeline")
        
        try:
            # Generate data
            self.generate_data()
            
            # Engineer features
            self.engineer_features()
            
            # Train models
            self.train_models()
            
            # Evaluate models
            self.evaluate_models()
            
            # Create visualizations
            self.create_visualizations()
            
            # Generate explainability report
            self.generate_explainability_report()
            
            # Save results
            self.save_results()
            
            logger.info("Pipeline completed successfully")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise


def main():
    """Main function to run the customer segmentation pipeline."""
    parser = argparse.ArgumentParser(description='Customer Segmentation Analysis')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to configuration file')
    parser.add_argument('--output-dir', type=str, default='assets',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Initialize and run pipeline
    pipeline = CustomerSegmentationPipeline(args.config)
    pipeline.run_full_pipeline()
    
    print(f"\nCustomer segmentation analysis completed!")
    print(f"Best model: {pipeline.results['best_model']}")
    print(f"Results saved to: {args.output_dir}")
    print("\nDISCLAIMER: This is a research demonstration only. Not for investment advice.")


if __name__ == "__main__":
    main()
