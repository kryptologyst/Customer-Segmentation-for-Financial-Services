"""
Streamlit demo for customer segmentation analysis.

This interactive demo allows users to explore customer segmentation results,
analyze different models, and understand customer profiles.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.data_generator import CustomerDataGenerator, CustomerDataConfig
from features.feature_engineering import FeatureEngineer
from models.segmentation_models import KMeansSegmenter, GaussianMixtureSegmenter
from evaluation.evaluation import SegmentationEvaluator
from utils.explainability import SegmentationExplainer

# Page configuration
st.set_page_config(
    page_title="Customer Segmentation Analysis",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .disclaimer {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 1rem;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🏦 Customer Segmentation Analysis</h1>', unsafe_allow_html=True)

# Disclaimer
st.markdown("""
<div class="disclaimer">
    <h4>⚠️ IMPORTANT DISCLAIMER</h4>
    <p><strong>This is a research and educational demonstration only.</strong></p>
    <ul>
        <li>This software is NOT intended for investment advice or financial planning</li>
        <li>Results may be inaccurate and should not be used for financial decisions</li>
        <li>All analyses are hypothetical and do not guarantee future performance</li>
        <li>This software is not intended for commercial use without proper validation</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Configuration")

# Data parameters
st.sidebar.header("Data Parameters")
n_customers = st.sidebar.slider("Number of Customers", 1000, 10000, 5000)
random_seed = st.sidebar.number_input("Random Seed", value=42, min_value=0, max_value=1000)

# Model parameters
st.sidebar.header("Model Parameters")
n_clusters = st.sidebar.slider("Number of Clusters", 3, 10, 5)
model_type = st.sidebar.selectbox("Model Type", ["K-means", "Gaussian Mixture Model"])

# Feature engineering options
st.sidebar.header("Feature Engineering")
scaling_method = st.sidebar.selectbox("Scaling Method", ["standard", "minmax", "robust"])
include_rfm = st.sidebar.checkbox("Include RFM Analysis", value=True)
include_clv = st.sidebar.checkbox("Include Customer Lifetime Value", value=True)

# Initialize session state
if 'data_generated' not in st.session_state:
    st.session_state.data_generated = False
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'results' not in st.session_state:
    st.session_state.results = {}

# Main content
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Data Overview", "🤖 Model Analysis", "👥 Customer Segments", "🔍 Explainability", "📈 Business Insights"])

with tab1:
    st.header("Data Overview")
    
    if st.button("Generate Customer Data", type="primary"):
        with st.spinner("Generating customer data..."):
            # Generate data
            config = CustomerDataConfig(
                n_customers=n_customers,
                random_seed=random_seed
            )
            generator = CustomerDataGenerator(config)
            data = generator.generate_complete_dataset()
            
            # Engineer features
            feature_engineer = FeatureEngineer(scaling_method)
            features = feature_engineer.create_all_features(data)
            features = feature_engineer.select_features(features, method='numeric')
            
            st.session_state.data = data
            st.session_state.features = features
            st.session_state.data_generated = True
            
        st.success("Data generated successfully!")
    
    if st.session_state.data_generated:
        data = st.session_state.data
        features = st.session_state.features
        
        # Data summary
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Customers", len(data))
        with col2:
            st.metric("Features", len(features.columns))
        with col3:
            st.metric("Avg Account Balance", f"${data['account_balance'].mean():,.0f}")
        with col4:
            st.metric("Avg CLV", f"${data['customer_lifetime_value'].mean():,.0f}")
        
        # Data distribution plots
        st.subheader("Data Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(data, x='account_balance', nbins=50, title='Account Balance Distribution')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.histogram(data, x='customer_lifetime_value', nbins=50, title='Customer Lifetime Value Distribution')
            st.plotly_chart(fig, use_container_width=True)
        
        # Feature correlation
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = features[numeric_cols].corr()
            fig = px.imshow(corr_matrix, title='Feature Correlation Matrix')
            st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Model Analysis")
    
    if st.session_state.data_generated:
        if st.button("Train Segmentation Models", type="primary"):
            with st.spinner("Training models..."):
                features = st.session_state.features
                feature_cols = [col for col in features.columns if col != 'customer_id']
                X = features[feature_cols].values
                
                # Scale features
                feature_engineer = FeatureEngineer(scaling_method)
                X_scaled = feature_engineer.scale_features(
                    features[feature_cols + ['customer_id']], fit=True
                )
                X_scaled = X_scaled[feature_cols].values
                
                # Train models
                models = {}
                
                if model_type == "K-means":
                    model = KMeansSegmenter(n_clusters=n_clusters, random_state=random_seed)
                else:
                    model = GaussianMixtureSegmenter(n_components=n_clusters, random_state=random_seed)
                
                labels = model.fit_predict(X_scaled)
                
                # Evaluate model
                evaluator = SegmentationEvaluator()
                report = evaluator.generate_evaluation_report(features, labels, model_type)
                
                st.session_state.model = model
                st.session_state.labels = labels
                st.session_state.scaler = feature_engineer.scaler
                st.session_state.evaluation_report = report
                st.session_state.models_trained = True
                
            st.success("Models trained successfully!")
    
    if st.session_state.models_trained:
        report = st.session_state.evaluation_report
        
        # Model metrics
        st.subheader("Model Performance")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Silhouette Score", f"{report['ml_metrics']['silhouette_score']:.3f}")
        with col2:
            st.metric("Calinski-Harabasz Score", f"{report['ml_metrics']['calinski_harabasz_score']:.1f}")
        with col3:
            st.metric("Davies-Bouldin Score", f"{report['ml_metrics']['davies_bouldin_score']:.3f}")
        
        # Evaluation summary
        st.subheader("Evaluation Summary")
        summary = report['evaluation_summary']
        for key, value in summary.items():
            st.write(f"**{key.replace('_', ' ').title()}**: {value}")

with tab3:
    st.header("Customer Segments")
    
    if st.session_state.models_trained:
        features = st.session_state.features
        labels = st.session_state.labels
        
        # Add labels to features
        df_with_labels = features.copy()
        df_with_labels['segment'] = labels
        
        # Segment overview
        st.subheader("Segment Overview")
        
        segment_sizes = df_with_labels['segment'].value_counts().sort_index()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(values=segment_sizes.values, names=[f'Segment {i}' for i in segment_sizes.index],
                        title='Segment Size Distribution')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(x=[f'Segment {i}' for i in segment_sizes.index], y=segment_sizes.values,
                        title='Segment Sizes')
            st.plotly_chart(fig, use_container_width=True)
        
        # Segment characteristics
        st.subheader("Segment Characteristics")
        
        # Select features to analyze
        numeric_cols = df_with_labels.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['customer_id', 'segment']]
        
        selected_features = st.multiselect(
            "Select features to analyze",
            numeric_cols,
            default=numeric_cols[:4] if len(numeric_cols) >= 4 else numeric_cols
        )
        
        if selected_features:
            # Create box plots for selected features
            fig = make_subplots(
                rows=len(selected_features), cols=1,
                subplot_titles=selected_features,
                vertical_spacing=0.05
            )
            
            for i, feature in enumerate(selected_features):
                for segment in sorted(df_with_labels['segment'].unique()):
                    segment_data = df_with_labels[df_with_labels['segment'] == segment][feature]
                    fig.add_trace(
                        go.Box(y=segment_data, name=f'Segment {segment}', showlegend=(i==0)),
                        row=i+1, col=1
                    )
            
            fig.update_layout(height=300*len(selected_features), title="Feature Distribution by Segment")
            st.plotly_chart(fig, use_container_width=True)
        
        # Segment profiles table
        st.subheader("Segment Profiles")
        
        segment_profiles = df_with_labels.groupby('segment')[numeric_cols].agg(['mean', 'std', 'count']).round(2)
        st.dataframe(segment_profiles, use_container_width=True)

with tab4:
    st.header("Explainability Analysis")
    
    if st.session_state.models_trained:
        features = st.session_state.features
        labels = st.session_state.labels
        model = st.session_state.model
        
        # Feature importance
        st.subheader("Feature Importance")
        
        feature_cols = [col for col in features.columns if col != 'customer_id']
        X = features[feature_cols].values
        
        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, labels)
        
        importance = pd.Series(rf.feature_importances_, index=feature_cols)
        importance = importance.sort_values(ascending=False)
        
        fig = px.bar(x=importance.head(15).values, y=importance.head(15).index,
                    orientation='h', title='Top 15 Most Important Features')
        st.plotly_chart(fig, use_container_width=True)
        
        # Individual customer analysis
        st.subheader("Individual Customer Analysis")
        
        customer_id = st.selectbox("Select Customer ID", features['customer_id'].tolist())
        
        if customer_id:
            customer_data = features[features['customer_id'] == customer_id][feature_cols].values[0]
            customer_segment = labels[features['customer_id'] == customer_id][0]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Customer ID**: {customer_id}")
                st.write(f"**Assigned Segment**: {customer_segment}")
            
            with col2:
                # Customer feature values
                customer_df = pd.DataFrame({
                    'Feature': feature_cols,
                    'Value': customer_data
                })
                customer_df = customer_df.sort_values('Value', ascending=False)
                
                fig = px.bar(customer_df.head(10), x='Value', y='Feature',
                           orientation='h', title=f'Customer {customer_id} - Top 10 Features')
                st.plotly_chart(fig, use_container_width=True)

with tab5:
    st.header("Business Insights")
    
    if st.session_state.models_trained:
        features = st.session_state.features
        labels = st.session_state.labels
        
        # Add labels to original data for business analysis
        df_business = st.session_state.data.copy()
        df_business['segment'] = labels
        
        # Business metrics by segment
        st.subheader("Business Metrics by Segment")
        
        # CLV analysis
        if 'customer_lifetime_value' in df_business.columns:
            clv_by_segment = df_business.groupby('segment')['customer_lifetime_value'].agg([
                'mean', 'sum', 'count'
            ]).round(2)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(x=[f'Segment {i}' for i in clv_by_segment.index],
                           y=clv_by_segment['mean'],
                           title='Average CLV by Segment')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.bar(x=[f'Segment {i}' for i in clv_by_segment.index],
                           y=clv_by_segment['sum'],
                           title='Total CLV by Segment')
                st.plotly_chart(fig, use_container_width=True)
        
        # Churn analysis
        if 'is_churned' in df_business.columns:
            st.subheader("Churn Analysis by Segment")
            
            churn_by_segment = df_business.groupby('segment')['is_churned'].agg(['mean', 'sum', 'count'])
            churn_by_segment['churn_rate'] = churn_by_segment['mean']
            
            fig = px.bar(x=[f'Segment {i}' for i in churn_by_segment.index],
                       y=churn_by_segment['churn_rate'],
                       title='Churn Rate by Segment')
            st.plotly_chart(fig, use_container_width=True)
        
        # Segment recommendations
        st.subheader("Segment Recommendations")
        
        segment_analysis = df_business.groupby('segment').agg({
            'customer_lifetime_value': 'mean',
            'account_balance': 'mean',
            'num_transactions': 'mean',
            'credit_score': 'mean'
        }).round(2)
        
        # Generate recommendations
        recommendations = []
        for segment in segment_analysis.index:
            clv = segment_analysis.loc[segment, 'customer_lifetime_value']
            balance = segment_analysis.loc[segment, 'account_balance']
            transactions = segment_analysis.loc[segment, 'num_transactions']
            credit = segment_analysis.loc[segment, 'credit_score']
            
            if clv > segment_analysis['customer_lifetime_value'].quantile(0.75):
                rec = "High-value segment - focus on retention and upselling"
            elif balance > segment_analysis['account_balance'].quantile(0.75):
                rec = "High-balance segment - offer premium services"
            elif transactions > segment_analysis['num_transactions'].quantile(0.75):
                rec = "Active segment - promote new products"
            elif credit > segment_analysis['credit_score'].quantile(0.75):
                rec = "High-credit segment - offer credit products"
            else:
                rec = "Standard segment - general marketing approach"
            
            recommendations.append({
                'Segment': f'Segment {segment}',
                'Avg CLV': f'${clv:,.0f}',
                'Avg Balance': f'${balance:,.0f}',
                'Avg Transactions': f'{transactions:.0f}',
                'Avg Credit Score': f'{credit:.0f}',
                'Recommendation': rec
            })
        
        recommendations_df = pd.DataFrame(recommendations)
        st.dataframe(recommendations_df, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.8rem;">
    <p>Customer Segmentation Analysis - Research Demo Only</p>
    <p>⚠️ Not for investment advice or commercial use</p>
</div>
""", unsafe_allow_html=True)
