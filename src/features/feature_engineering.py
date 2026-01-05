"""
Feature engineering module for customer segmentation.

This module provides utilities for creating advanced features
for customer segmentation including RFM analysis, customer lifetime value,
and behavioral patterns.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Handles feature engineering for customer segmentation."""
    
    def __init__(self, scaling_method: str = "standard"):
        """Initialize feature engineer with scaling method."""
        self.scaling_method = scaling_method
        self.scaler = self._get_scaler()
        self.feature_names: List[str] = []
        self.is_fitted = False
        
    def _get_scaler(self):
        """Get the appropriate scaler based on method."""
        scalers = {
            "standard": StandardScaler(),
            "minmax": MinMaxScaler(),
            "robust": RobustScaler()
        }
        return scalers.get(self.scaling_method, StandardScaler())
    
    def create_rfm_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create RFM (Recency, Frequency, Monetary) features."""
        logger.info("Creating RFM features")
        
        # Recency scoring (lower days = higher score)
        df['recency_score'] = pd.qcut(df['recency_days'], 5, labels=[5, 4, 3, 2, 1]).astype(int)
        
        # Frequency scoring (higher frequency = higher score)
        df['frequency_score'] = pd.qcut(df['frequency_score'], 5, labels=[1, 2, 3, 4, 5]).astype(int)
        
        # Monetary scoring (higher value = higher score)
        df['monetary_score'] = pd.qcut(df['monetary_value'], 5, labels=[1, 2, 3, 4, 5]).astype(int)
        
        # RFM score combination
        df['rfm_score'] = df['recency_score'] * 100 + df['frequency_score'] * 10 + df['monetary_score']
        
        # RFM segments
        df['rfm_segment'] = self._create_rfm_segments(df)
        
        return df
    
    def _create_rfm_segments(self, df: pd.DataFrame) -> pd.Series:
        """Create RFM segments based on scores."""
        def assign_segment(row):
            r, f, m = row['recency_score'], row['frequency_score'], row['monetary_score']
            
            if r >= 4 and f >= 4 and m >= 4:
                return 'Champions'
            elif r >= 3 and f >= 3 and m >= 4:
                return 'Loyal Customers'
            elif r >= 4 and f <= 2 and m >= 3:
                return 'Potential Loyalists'
            elif r >= 4 and f <= 2 and m <= 2:
                return 'New Customers'
            elif r >= 3 and f >= 3 and m >= 3:
                return 'Promising'
            elif r <= 2 and f >= 3 and m >= 3:
                return 'Need Attention'
            elif r <= 2 and f >= 4 and m >= 4:
                return 'About to Sleep'
            elif r <= 2 and f <= 2 and m >= 3:
                return 'At Risk'
            elif r <= 2 and f <= 2 and m <= 2:
                return 'Cannot Lose Them'
            else:
                return 'Others'
        
        return df.apply(assign_segment, axis=1)
    
    def create_customer_lifetime_value_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced CLV features."""
        logger.info("Creating CLV features")
        
        # CLV categories
        df['clv_category'] = pd.qcut(df['customer_lifetime_value'], 5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
        
        # CLV to income ratio
        df['clv_income_ratio'] = df['customer_lifetime_value'] / df['annual_income']
        
        # CLV to balance ratio
        df['clv_balance_ratio'] = df['customer_lifetime_value'] / (df['account_balance'] + 1)
        
        # Predicted CLV growth (simple heuristic)
        df['predicted_clv_growth'] = df['customer_lifetime_value'] * (1 + df['years_with_bank'] * 0.1)
        
        return df
    
    def create_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create behavioral pattern features."""
        logger.info("Creating behavioral features")
        
        # Digital engagement score
        df['digital_engagement'] = (df['online_banking_usage'] + df['mobile_app_usage']) / 2
        
        # Service intensity
        df['service_intensity'] = df['service_calls'] / (df['years_with_bank'] + 1)
        
        # Product diversity
        df['product_diversity'] = df['has_investments'] + df['has_insurance'] + (df['num_credit_cards'] > 0).astype(int)
        
        # Transaction efficiency (spending per transaction)
        df['transaction_efficiency'] = df['monthly_spending'] / (df['num_transactions'] + 1)
        
        # Branch dependency (visits vs digital usage)
        df['branch_dependency'] = df['branch_visits'] / (df['digital_engagement'] + 0.1)
        
        # Customer maturity (age + years with bank)
        df['customer_maturity'] = df['age'] + df['years_with_bank']
        
        return df
    
    def create_financial_health_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create financial health indicators."""
        logger.info("Creating financial health features")
        
        # Debt to income ratio
        df['debt_income_ratio'] = df['loan_amount'] / (df['annual_income'] + 1)
        
        # Savings rate (balance relative to income)
        df['savings_rate'] = df['account_balance'] / (df['annual_income'] + 1)
        
        # Credit utilization (spending vs credit capacity)
        df['credit_utilization'] = df['monthly_spending'] / (df['account_balance'] + 1)
        
        # Financial stability score
        df['financial_stability'] = (
            (df['credit_score'] / 850) * 0.4 +
            (1 - df['debt_income_ratio']) * 0.3 +
            df['savings_rate'] * 0.3
        )
        
        # Income growth potential (based on age and education)
        education_multiplier = df['education'].map({
            'High School': 1.0,
            'Bachelor': 1.2,
            'Master': 1.4,
            'PhD': 1.6
        })
        df['income_growth_potential'] = education_multiplier * (1 - df['age'] / 80)
        
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between variables."""
        logger.info("Creating interaction features")
        
        # Age-income interaction
        df['age_income_interaction'] = df['age'] * df['annual_income'] / 1000000
        
        # Balance-transaction interaction
        df['balance_transaction_interaction'] = df['account_balance'] * df['num_transactions'] / 1000
        
        # Credit score-loan interaction
        df['credit_loan_interaction'] = df['credit_score'] * df['loan_amount'] / 1000000
        
        # Digital usage-age interaction
        df['digital_age_interaction'] = df['digital_engagement'] * (80 - df['age']) / 80
        
        return df
    
    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create all engineered features."""
        logger.info("Creating all engineered features")
        
        # Create copies to avoid modifying original
        df_engineered = df.copy()
        
        # Apply all feature engineering steps
        df_engineered = self.create_rfm_features(df_engineered)
        df_engineered = self.create_customer_lifetime_value_features(df_engineered)
        df_engineered = self.create_behavioral_features(df_engineered)
        df_engineered = self.create_financial_health_features(df_engineered)
        df_engineered = self.create_interaction_features(df_engineered)
        
        logger.info(f"Created {len(df_engineered.columns) - len(df.columns)} new features")
        return df_engineered
    
    def select_features(self, df: pd.DataFrame, target_column: Optional[str] = None, 
                       method: str = "all", k: int = 20) -> pd.DataFrame:
        """Select features for modeling."""
        logger.info(f"Selecting features using method: {method}")
        
        # Define feature columns (exclude non-numeric and target columns)
        exclude_columns = ['customer_id', 'gender', 'education', 'employment_status', 
                          'rfm_segment', 'clv_category']
        if target_column:
            exclude_columns.append(target_column)
        
        feature_columns = [col for col in df.columns if col not in exclude_columns]
        
        if method == "all":
            selected_features = feature_columns
        elif method == "numeric":
            selected_features = df[feature_columns].select_dtypes(include=[np.number]).columns.tolist()
        elif method == "kbest" and target_column:
            selector = SelectKBest(score_func=f_classif, k=k)
            X = df[feature_columns].select_dtypes(include=[np.number])
            y = df[target_column]
            selector.fit(X, y)
            selected_features = X.columns[selector.get_support()].tolist()
        else:
            selected_features = feature_columns
        
        self.feature_names = selected_features
        logger.info(f"Selected {len(selected_features)} features")
        
        return df[selected_features + ['customer_id']]
    
    def scale_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Scale features for modeling."""
        logger.info(f"Scaling features using {self.scaling_method} method")
        
        feature_columns = [col for col in df.columns if col != 'customer_id']
        
        if fit:
            scaled_features = self.scaler.fit_transform(df[feature_columns])
            self.is_fitted = True
        else:
            if not self.is_fitted:
                raise ValueError("Scaler must be fitted before transforming")
            scaled_features = self.scaler.transform(df[feature_columns])
        
        # Create new dataframe with scaled features
        df_scaled = pd.DataFrame(scaled_features, columns=feature_columns)
        df_scaled['customer_id'] = df['customer_id'].values
        
        return df_scaled
    
    def reduce_dimensionality(self, df: pd.DataFrame, n_components: int = 10, 
                            method: str = "pca") -> pd.DataFrame:
        """Reduce dimensionality of features."""
        logger.info(f"Reducing dimensionality using {method}")
        
        feature_columns = [col for col in df.columns if col != 'customer_id']
        X = df[feature_columns]
        
        if method == "pca":
            reducer = PCA(n_components=n_components)
            reduced_features = reducer.fit_transform(X)
            
            # Create column names
            column_names = [f"PC_{i+1}" for i in range(n_components)]
            
            logger.info(f"Explained variance ratio: {reducer.explained_variance_ratio_.sum():.3f}")
        
        # Create new dataframe
        df_reduced = pd.DataFrame(reduced_features, columns=column_names)
        df_reduced['customer_id'] = df['customer_id'].values
        
        return df_reduced
    
    def get_feature_importance(self, df: pd.DataFrame, target_column: str) -> pd.Series:
        """Get feature importance scores."""
        from sklearn.ensemble import RandomForestClassifier
        
        feature_columns = [col for col in df.columns if col not in ['customer_id', target_column]]
        X = df[feature_columns]
        y = df[target_column]
        
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        importance = pd.Series(rf.feature_importances_, index=feature_columns)
        return importance.sort_values(ascending=False)


def create_feature_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Create a summary of all features."""
    summary = pd.DataFrame({
        'feature': df.columns,
        'dtype': df.dtypes,
        'missing_count': df.isnull().sum(),
        'missing_pct': df.isnull().sum() / len(df) * 100,
        'unique_count': df.nunique(),
        'unique_pct': df.nunique() / len(df) * 100
    })
    
    # Add statistics for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        summary.loc[summary['feature'] == col, 'mean'] = df[col].mean()
        summary.loc[summary['feature'] == col, 'std'] = df[col].std()
        summary.loc[summary['feature'] == col, 'min'] = df[col].min()
        summary.loc[summary['feature'] == col, 'max'] = df[col].max()
    
    return summary
