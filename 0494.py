"""
Project 494: Customer Segmentation for Financial Services - Simple Demo

This is a simplified demonstration of the customer segmentation project.
For the full implementation with advanced features, see the main project structure.

DISCLAIMER: This is a research and educational demonstration only.
This software is NOT intended for investment advice or financial planning.
Results may be inaccurate and should not be used for financial decisions.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

def generate_customer_data(n_customers=1000):
    """Generate synthetic customer data for demonstration."""
    print("Generating synthetic customer data...")
    
    data = {
        'customer_id': range(1, n_customers + 1),
        'account_balance': np.random.normal(5000, 1500, n_customers),
        'num_transactions': np.random.randint(1, 50, n_customers),
        'loan_amount': np.random.normal(20000, 5000, n_customers),
        'age': np.random.randint(18, 70, n_customers),
        'annual_income': np.random.normal(60000, 15000, n_customers),
        'credit_score': np.random.normal(650, 100, n_customers)
    }
    
    df = pd.DataFrame(data)
    
    # Ensure positive values
    df['account_balance'] = np.abs(df['account_balance'])
    df['loan_amount'] = np.abs(df['loan_amount'])
    df['annual_income'] = np.abs(df['annual_income'])
    df['credit_score'] = np.clip(df['credit_score'], 300, 850)
    
    return df

def perform_customer_segmentation(df, n_clusters=4):
    """Perform customer segmentation using K-means clustering."""
    print(f"Performing customer segmentation with {n_clusters} clusters...")
    
    # Select features for clustering
    feature_columns = ['account_balance', 'num_transactions', 'loan_amount', 'age', 'annual_income', 'credit_score']
    X = df[feature_columns]
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['segment'] = kmeans.fit_predict(X_scaled)
    
    return df, kmeans, scaler

def visualize_segments(df):
    """Create visualizations of customer segments."""
    print("Creating visualizations...")
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Account Balance vs Loan Amount
    scatter = axes[0, 0].scatter(df['account_balance'], df['loan_amount'], 
                               c=df['segment'], cmap='viridis', alpha=0.6)
    axes[0, 0].set_title('Customer Segmentation: Account Balance vs Loan Amount')
    axes[0, 0].set_xlabel('Account Balance (USD)')
    axes[0, 0].set_ylabel('Loan Amount (USD)')
    plt.colorbar(scatter, ax=axes[0, 0], label='Segment')
    
    # Plot 2: Age vs Annual Income
    scatter = axes[0, 1].scatter(df['age'], df['annual_income'], 
                               c=df['segment'], cmap='viridis', alpha=0.6)
    axes[0, 1].set_title('Customer Segmentation: Age vs Annual Income')
    axes[0, 1].set_xlabel('Age')
    axes[0, 1].set_ylabel('Annual Income (USD)')
    plt.colorbar(scatter, ax=axes[0, 1], label='Segment')
    
    # Plot 3: Segment size distribution
    segment_counts = df['segment'].value_counts().sort_index()
    axes[1, 0].bar(range(len(segment_counts)), segment_counts.values, color='skyblue')
    axes[1, 0].set_title('Segment Size Distribution')
    axes[1, 0].set_xlabel('Segment')
    axes[1, 0].set_ylabel('Number of Customers')
    axes[1, 0].set_xticks(range(len(segment_counts)))
    axes[1, 0].set_xticklabels([f'Segment {i}' for i in segment_counts.index])
    
    # Plot 4: Average values by segment
    segment_means = df.groupby('segment')[['account_balance', 'annual_income', 'credit_score']].mean()
    segment_means.plot(kind='bar', ax=axes[1, 1])
    axes[1, 1].set_title('Average Values by Segment')
    axes[1, 1].set_xlabel('Segment')
    axes[1, 1].set_ylabel('Average Value')
    axes[1, 1].legend(['Account Balance', 'Annual Income', 'Credit Score'])
    axes[1, 1].tick_params(axis='x', rotation=0)
    
    plt.tight_layout()
    plt.show()

def analyze_segments(df):
    """Analyze and display segment characteristics."""
    print("\n" + "="*60)
    print("CUSTOMER SEGMENT ANALYSIS")
    print("="*60)
    
    # Segment summary
    segment_summary = df.groupby('segment').agg({
        'account_balance': ['mean', 'std'],
        'num_transactions': ['mean', 'std'],
        'loan_amount': ['mean', 'std'],
        'age': ['mean', 'std'],
        'annual_income': ['mean', 'std'],
        'credit_score': ['mean', 'std']
    }).round(2)
    
    print("Segment Summary (Mean ± Std):")
    print(segment_summary)
    
    # Segment sizes
    print(f"\nSegment Sizes:")
    segment_sizes = df['segment'].value_counts().sort_index()
    for segment, size in segment_sizes.items():
        percentage = (size / len(df)) * 100
        print(f"Segment {segment}: {size} customers ({percentage:.1f}%)")
    
    # Business insights
    print(f"\nBusiness Insights:")
    for segment in sorted(df['segment'].unique()):
        segment_data = df[df['segment'] == segment]
        avg_clv = segment_data['account_balance'].mean() + segment_data['loan_amount'].mean()
        
        print(f"\nSegment {segment}:")
        print(f"  - Average Account Balance: ${segment_data['account_balance'].mean():,.0f}")
        print(f"  - Average Annual Income: ${segment_data['annual_income'].mean():,.0f}")
        print(f"  - Average Credit Score: {segment_data['credit_score'].mean():.0f}")
        print(f"  - Average Transactions: {segment_data['num_transactions'].mean():.1f}")
        
        # Simple segment characterization
        if segment_data['account_balance'].mean() > df['account_balance'].quantile(0.75):
            print(f"  - Characterization: High-value customers")
        elif segment_data['age'].mean() < df['age'].quantile(0.25):
            print(f"  - Characterization: Young customers")
        elif segment_data['credit_score'].mean() > df['credit_score'].quantile(0.75):
            print(f"  - Characterization: High-credit customers")
        else:
            print(f"  - Characterization: Standard customers")

def main():
    """Main function to run the customer segmentation demo."""
    print("="*60)
    print("CUSTOMER SEGMENTATION FOR FINANCIAL SERVICES")
    print("="*60)
    print("DISCLAIMER: This is a research demonstration only.")
    print("Not intended for investment advice or commercial use.")
    print("="*60)
    
    # Generate customer data
    df = generate_customer_data(n_customers=1000)
    print(f"Generated data for {len(df)} customers")
    
    # Perform segmentation
    df_segmented, model, scaler = perform_customer_segmentation(df, n_clusters=4)
    
    # Analyze segments
    analyze_segments(df_segmented)
    
    # Create visualizations
    visualize_segments(df_segmented)
    
    print("\n" + "="*60)
    print("DEMO COMPLETED SUCCESSFULLY")
    print("="*60)
    print("For advanced features and production-ready implementation,")
    print("see the main project structure with:")
    print("- Advanced ML models (GMM, Hierarchical, DBSCAN)")
    print("- RFM analysis and customer lifetime value")
    print("- SHAP explainability")
    print("- Interactive Streamlit demo")
    print("- Comprehensive evaluation metrics")
    print("="*60)

if __name__ == "__main__":
    main()
