# Customer Segmentation for Financial Services - Research Demo

**DISCLAIMER: This is a research and educational demonstration only. This software is NOT intended for investment advice, financial planning, or commercial use. Results may be inaccurate and should not be used for making financial decisions. Backtests are hypothetical and do not guarantee future performance.**

## Overview

This project implements advanced customer segmentation techniques for financial services using machine learning. It demonstrates various clustering algorithms, RFM analysis, and explainable AI techniques for understanding customer behavior patterns.

## Features

- **Multiple Segmentation Algorithms**: K-means, Gaussian Mixture Models, Hierarchical Clustering, DBSCAN
- **RFM Analysis**: Recency, Frequency, Monetary value analysis for customer segmentation
- **Advanced Features**: Customer lifetime value, churn prediction, behavioral patterns
- **Explainable AI**: SHAP explanations, feature importance, cluster interpretability
- **Interactive Demo**: Streamlit-based web interface for exploring segments
- **Comprehensive Evaluation**: Silhouette analysis, cluster stability, business metrics

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the interactive demo:
```bash
streamlit run demo/app.py
```

3. Or run the main analysis:
```bash
python scripts/run_segmentation.py --config configs/default.yaml
```

## Project Structure

```
├── src/                    # Source code
│   ├── data/              # Data processing modules
│   ├── features/           # Feature engineering
│   ├── models/            # ML models and algorithms
│   ├── evaluation/        # Evaluation metrics
│   └── utils/             # Utility functions
├── configs/               # Configuration files
├── scripts/               # Execution scripts
├── demo/                  # Streamlit demo
├── tests/                 # Unit tests
├── assets/                # Output artifacts
└── data/                  # Data storage
```

## Dataset

The project uses synthetic financial customer data including:
- Account balances and transaction history
- Loan amounts and credit scores
- Demographic information
- Behavioral patterns and preferences

## Evaluation Metrics

- **ML Metrics**: Silhouette score, Calinski-Harabasz index, Davies-Bouldin index
- **Business Metrics**: Segment profitability, customer lifetime value, churn rates
- **Stability**: Cluster consistency across time periods

## Configuration

Modify `configs/default.yaml` to adjust:
- Dataset parameters
- Model hyperparameters
- Evaluation settings
- Visualization options

## License

This project is for educational and research purposes only. See LICENSE file for details.
# Customer-Segmentation-for-Financial-Services
