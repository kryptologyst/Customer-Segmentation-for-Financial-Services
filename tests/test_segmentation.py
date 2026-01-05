"""
Test module for customer segmentation project.

This module contains basic tests to ensure the project components work correctly.
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.data_generator import CustomerDataGenerator, CustomerDataConfig
from features.feature_engineering import FeatureEngineer
from models.segmentation_models import KMeansSegmenter
from evaluation.evaluation import SegmentationEvaluator


class TestDataGenerator:
    """Test data generation functionality."""
    
    def test_customer_data_config(self):
        """Test CustomerDataConfig initialization."""
        config = CustomerDataConfig(n_customers=100, random_seed=42)
        assert config.n_customers == 100
        assert config.random_seed == 42
    
    def test_data_generation(self):
        """Test customer data generation."""
        config = CustomerDataConfig(n_customers=100, random_seed=42)
        generator = CustomerDataGenerator(config)
        data = generator.generate_complete_dataset()
        
        assert len(data) == 100
        assert 'customer_id' in data.columns
        assert 'account_balance' in data.columns
        assert 'customer_lifetime_value' in data.columns
    
    def test_data_validation(self):
        """Test data validation."""
        config = CustomerDataConfig(n_customers=50, random_seed=42)
        generator = CustomerDataGenerator(config)
        data = generator.generate_complete_dataset()
        
        # Check for required columns
        required_columns = ['customer_id', 'age', 'annual_income', 'account_balance']
        for col in required_columns:
            assert col in data.columns
        
        # Check for reasonable values
        assert data['age'].min() >= 18
        assert data['age'].max() <= 80
        assert data['account_balance'].min() >= 0


class TestFeatureEngineering:
    """Test feature engineering functionality."""
    
    def test_feature_engineer_initialization(self):
        """Test FeatureEngineer initialization."""
        engineer = FeatureEngineer(scaling_method="standard")
        assert engineer.scaling_method == "standard"
    
    def test_rfm_features(self):
        """Test RFM feature creation."""
        # Create sample data
        data = pd.DataFrame({
            'recency_days': [10, 20, 30, 40, 50],
            'frequency_score': [5, 4, 3, 2, 1],
            'monetary_value': [100, 200, 300, 400, 500]
        })
        
        engineer = FeatureEngineer()
        result = engineer.create_rfm_features(data)
        
        assert 'recency_score' in result.columns
        assert 'frequency_score' in result.columns
        assert 'monetary_score' in result.columns
        assert 'rfm_score' in result.columns
        assert 'rfm_segment' in result.columns


class TestSegmentationModels:
    """Test segmentation models."""
    
    def test_kmeans_initialization(self):
        """Test KMeansSegmenter initialization."""
        model = KMeansSegmenter(n_clusters=3, random_state=42)
        assert model.n_clusters == 3
        assert model.random_state == 42
    
    def test_kmeans_fit_predict(self):
        """Test KMeansSegmenter fit and predict."""
        # Create sample data
        X = np.random.randn(100, 5)
        
        model = KMeansSegmenter(n_clusters=3, random_state=42)
        labels = model.fit_predict(X)
        
        assert len(labels) == 100
        assert len(np.unique(labels)) <= 3
        assert model.is_fitted == True


class TestEvaluation:
    """Test evaluation functionality."""
    
    def test_evaluator_initialization(self):
        """Test SegmentationEvaluator initialization."""
        evaluator = SegmentationEvaluator()
        assert evaluator is not None
    
    def test_ml_metrics_evaluation(self):
        """Test ML metrics evaluation."""
        # Create sample data
        X = np.random.randn(100, 5)
        labels = np.random.randint(0, 3, 100)
        
        evaluator = SegmentationEvaluator()
        metrics = evaluator.evaluate_ml_metrics(X, labels)
        
        assert 'silhouette_score' in metrics
        assert 'calinski_harabasz_score' in metrics
        assert 'davies_bouldin_score' in metrics
        assert 'n_clusters' in metrics


def test_integration():
    """Test end-to-end integration."""
    # Generate data
    config = CustomerDataConfig(n_customers=100, random_seed=42)
    generator = CustomerDataGenerator(config)
    data = generator.generate_complete_dataset()
    
    # Engineer features
    engineer = FeatureEngineer()
    features = engineer.create_all_features(data)
    features = engineer.select_features(features, method='numeric')
    
    # Train model
    feature_cols = [col for col in features.columns if col != 'customer_id']
    X = features[feature_cols].values
    
    model = KMeansSegmenter(n_clusters=3, random_state=42)
    labels = model.fit_predict(X)
    
    # Evaluate
    evaluator = SegmentationEvaluator()
    metrics = evaluator.evaluate_ml_metrics(X, labels)
    
    assert metrics['n_clusters'] <= 3
    assert len(labels) == 100


if __name__ == "__main__":
    pytest.main([__file__])
