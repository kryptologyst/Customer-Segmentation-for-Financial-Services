"""
Utility functions for customer segmentation project.

This module provides common utility functions used across the project.
"""

import os
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Union
import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Set up logging configuration."""
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Set up root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    logger.info(f"Logging configured with level: {log_level}")


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Loaded configuration from {config_path}")
    return config


def save_config(config: Dict[str, Any], config_path: Union[str, Path]) -> None:
    """Save configuration to YAML file."""
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    logger.info(f"Saved configuration to {config_path}")


def save_model(model: Any, model_path: Union[str, Path]) -> None:
    """Save a trained model to file."""
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    logger.info(f"Saved model to {model_path}")


def load_model(model_path: Union[str, Path]) -> Any:
    """Load a trained model from file."""
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    logger.info(f"Loaded model from {model_path}")
    return model


def create_output_dirs(base_dir: Union[str, Path], subdirs: list) -> None:
    """Create output directories."""
    base_dir = Path(base_dir)
    
    for subdir in subdirs:
        dir_path = base_dir / subdir
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Created directory: {dir_path}")


def ensure_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all columns are numeric, converting where possible."""
    df_clean = df.copy()
    
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            # Try to convert to numeric
            try:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            except:
                # If conversion fails, keep as object
                pass
    
    return df_clean


def remove_outliers_iqr(df: pd.DataFrame, columns: list, factor: float = 1.5) -> pd.DataFrame:
    """Remove outliers using IQR method."""
    df_clean = df.copy()
    
    for col in columns:
        if col in df_clean.columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            
            # Remove outliers
            mask = (df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)
            df_clean = df_clean[mask]
    
    logger.info(f"Removed outliers from {len(columns)} columns")
    return df_clean


def calculate_statistical_significance(group1: np.ndarray, group2: np.ndarray) -> Dict[str, float]:
    """Calculate statistical significance between two groups."""
    from scipy import stats
    
    # T-test
    t_stat, t_pvalue = stats.ttest_ind(group1, group2)
    
    # Mann-Whitney U test (non-parametric)
    u_stat, u_pvalue = stats.mannwhitneyu(group1, group2, alternative='two-sided')
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt(((len(group1) - 1) * np.var(group1, ddof=1) + 
                          (len(group2) - 1) * np.var(group2, ddof=1)) / 
                         (len(group1) + len(group2) - 2))
    cohens_d = (np.mean(group1) - np.mean(group2)) / pooled_std
    
    return {
        't_statistic': t_stat,
        't_pvalue': t_pvalue,
        'u_statistic': u_stat,
        'u_pvalue': u_pvalue,
        'cohens_d': cohens_d,
        'significant': t_pvalue < 0.05
    }


def format_currency(value: float, currency: str = "USD") -> str:
    """Format currency values."""
    if currency == "USD":
        return f"${value:,.2f}"
    else:
        return f"{value:,.2f} {currency}"


def format_percentage(value: float, decimals: int = 2) -> str:
    """Format percentage values."""
    return f"{value:.{decimals}f}%"


def create_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Create comprehensive summary statistics."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    summary = df[numeric_cols].describe()
    
    # Add additional statistics
    additional_stats = pd.DataFrame(index=numeric_cols)
    additional_stats['skewness'] = df[numeric_cols].skew()
    additional_stats['kurtosis'] = df[numeric_cols].kurtosis()
    additional_stats['missing_count'] = df[numeric_cols].isnull().sum()
    additional_stats['missing_pct'] = df[numeric_cols].isnull().sum() / len(df) * 100
    
    # Combine statistics
    combined_stats = pd.concat([summary.T, additional_stats], axis=1)
    
    return combined_stats


def validate_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """Validate data quality and return report."""
    report = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'missing_percentage': df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100,
        'duplicate_rows': df.duplicated().sum(),
        'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
        'categorical_columns': len(df.select_dtypes(include=['object', 'category']).columns),
        'issues': []
    }
    
    # Check for issues
    if report['missing_percentage'] > 10:
        report['issues'].append("High percentage of missing values")
    
    if report['duplicate_rows'] > 0:
        report['issues'].append("Duplicate rows found")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > len(df) * 0.5:
            report['issues'].append(f"Column '{col}' has >50% missing values")
    
    return report


def get_project_root() -> Path:
    """Get the project root directory."""
    current_file = Path(__file__)
    return current_file.parent.parent


def setup_project_environment() -> None:
    """Set up the project environment."""
    project_root = get_project_root()
    
    # Create necessary directories
    dirs_to_create = [
        'data',
        'assets',
        'assets/results',
        'assets/models',
        'assets/logs',
        'assets/plots',
        'configs',
        'scripts',
        'tests'
    ]
    
    for dir_name in dirs_to_create:
        dir_path = project_root / dir_name
        dir_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("Project environment set up successfully")


def print_project_info() -> None:
    """Print project information."""
    project_root = get_project_root()
    
    print("=" * 60)
    print("Customer Segmentation for Financial Services")
    print("=" * 60)
    print(f"Project Root: {project_root}")
    print(f"Python Version: {sys.version}")
    print("=" * 60)
    print("DISCLAIMER: This is a research demonstration only.")
    print("Not intended for investment advice or commercial use.")
    print("=" * 60)


# Import sys at module level
import sys
