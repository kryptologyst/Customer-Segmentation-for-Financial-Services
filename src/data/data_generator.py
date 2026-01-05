"""
Data generation and processing module for customer segmentation.

This module handles the creation of synthetic financial customer data
and provides utilities for data preprocessing and validation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class CustomerDataConfig:
    """Configuration for customer data generation."""
    n_customers: int = 5000
    random_seed: int = 42
    age_range: Tuple[int, int] = (18, 80)
    income_range: Tuple[float, float] = (20000, 150000)
    balance_range: Tuple[float, float] = (100, 50000)
    transaction_range: Tuple[int, int] = (1, 100)
    loan_range: Tuple[float, float] = (1000, 100000)
    credit_score_range: Tuple[int, int] = (300, 850)


class CustomerDataGenerator:
    """Generates synthetic financial customer data for segmentation analysis."""
    
    def __init__(self, config: CustomerDataConfig):
        """Initialize the data generator with configuration."""
        self.config = config
        np.random.seed(config.random_seed)
        
    def generate_demographics(self) -> pd.DataFrame:
        """Generate demographic features for customers."""
        n = self.config.n_customers
        
        # Age with realistic distribution (skewed towards middle-aged)
        age = np.random.beta(2, 5, n) * (self.config.age_range[1] - self.config.age_range[0]) + self.config.age_range[0]
        age = age.astype(int)
        
        # Gender (slightly more females in financial services)
        gender = np.random.choice(['Male', 'Female'], n, p=[0.45, 0.55])
        
        # Income with log-normal distribution
        income_log = np.random.normal(np.log(50000), 0.8, n)
        income = np.exp(income_log)
        income = np.clip(income, self.config.income_range[0], self.config.income_range[1])
        
        # Education level
        education = np.random.choice(
            ['High School', 'Bachelor', 'Master', 'PhD'], 
            n, 
            p=[0.3, 0.4, 0.25, 0.05]
        )
        
        # Employment status
        employment = np.random.choice(
            ['Employed', 'Self-employed', 'Unemployed', 'Retired'], 
            n, 
            p=[0.7, 0.15, 0.1, 0.05]
        )
        
        return pd.DataFrame({
            'customer_id': range(1, n + 1),
            'age': age,
            'gender': gender,
            'annual_income': income,
            'education': education,
            'employment_status': employment
        })
    
    def generate_financial_features(self, demographics: pd.DataFrame) -> pd.DataFrame:
        """Generate financial features based on demographics."""
        n = len(demographics)
        
        # Account balance correlated with income
        income_factor = demographics['annual_income'] / demographics['annual_income'].mean()
        base_balance = np.random.lognormal(8, 1, n)  # Base balance
        account_balance = base_balance * income_factor
        account_balance = np.clip(account_balance, self.config.balance_range[0], self.config.balance_range[1])
        
        # Number of transactions (correlated with balance and age)
        balance_factor = account_balance / account_balance.mean()
        age_factor = demographics['age'] / demographics['age'].mean()
        transaction_intensity = balance_factor * (2 - age_factor)  # Younger = more transactions
        num_transactions = np.random.poisson(transaction_intensity * 10)
        num_transactions = np.clip(num_transactions, self.config.transaction_range[0], self.config.transaction_range[1])
        
        # Loan amount (correlated with income and age)
        income_factor = demographics['annual_income'] / demographics['annual_income'].mean()
        age_factor = np.clip(demographics['age'] / 50, 0.5, 2)  # Peak around 50
        loan_factor = income_factor * age_factor
        base_loan = np.random.lognormal(9, 1, n)
        loan_amount = base_loan * loan_factor
        loan_amount = np.clip(loan_amount, self.config.loan_range[0], self.config.loan_range[1])
        
        # Credit score (correlated with income, age, and employment)
        base_score = np.random.normal(650, 100, n)
        
        # Adjust based on income
        income_adjustment = (demographics['annual_income'] - demographics['annual_income'].mean()) / demographics['annual_income'].std() * 20
        
        # Adjust based on age (older = higher score)
        age_adjustment = (demographics['age'] - demographics['age'].mean()) / demographics['age'].std() * 10
        
        # Adjust based on employment
        employment_adjustment = demographics['employment_status'].map({
            'Employed': 20,
            'Self-employed': 10,
            'Unemployed': -30,
            'Retired': 15
        })
        
        credit_score = base_score + income_adjustment + age_adjustment + employment_adjustment
        credit_score = np.clip(credit_score, self.config.credit_score_range[0], self.config.credit_score_range[1])
        
        # Monthly spending (correlated with income and balance)
        spending_factor = income_factor * balance_factor
        monthly_spending = np.random.lognormal(6, 0.8, n) * spending_factor
        
        # Number of credit cards
        num_credit_cards = np.random.poisson(2, n)
        num_credit_cards = np.clip(num_credit_cards, 0, 10)
        
        # Years with bank
        years_with_bank = np.random.exponential(5, n)
        years_with_bank = np.clip(years_with_bank, 0, 50)
        
        return pd.DataFrame({
            'account_balance': account_balance,
            'num_transactions': num_transactions,
            'loan_amount': loan_amount,
            'credit_score': credit_score,
            'monthly_spending': monthly_spending,
            'num_credit_cards': num_credit_cards,
            'years_with_bank': years_with_bank
        })
    
    def generate_behavioral_features(self, demographics: pd.DataFrame, financial: pd.DataFrame) -> pd.DataFrame:
        """Generate behavioral features based on demographics and financial data."""
        n = len(demographics)
        
        # Online banking usage (correlated with age - younger = more usage)
        age_factor = (80 - demographics['age']) / 62  # Normalize to 0-1
        online_usage = np.random.beta(2, 2, n) * age_factor
        online_usage = np.clip(online_usage, 0, 1)
        
        # Mobile app usage (similar to online but higher for younger)
        mobile_usage = np.random.beta(3, 2, n) * age_factor
        mobile_usage = np.clip(mobile_usage, 0, 1)
        
        # Branch visits (inversely correlated with online usage)
        branch_visits = np.random.poisson(2, n) * (1 - online_usage)
        
        # Customer service calls (random but slightly correlated with problems)
        service_calls = np.random.poisson(1, n)
        
        # Investment products (correlated with income and age)
        income_factor = demographics['annual_income'] / demographics['annual_income'].mean()
        age_factor = demographics['age'] / demographics['age'].mean()
        investment_prob = np.clip(income_factor * age_factor * 0.3, 0, 1)
        has_investments = np.random.binomial(1, investment_prob, n)
        
        # Insurance products (correlated with age and income)
        insurance_prob = np.clip(age_factor * income_factor * 0.4, 0, 1)
        has_insurance = np.random.binomial(1, insurance_prob, n)
        
        return pd.DataFrame({
            'online_banking_usage': online_usage,
            'mobile_app_usage': mobile_usage,
            'branch_visits': branch_visits,
            'service_calls': service_calls,
            'has_investments': has_investments,
            'has_insurance': has_insurance
        })
    
    def generate_rfm_features(self, financial: pd.DataFrame, behavioral: pd.DataFrame) -> pd.DataFrame:
        """Generate RFM (Recency, Frequency, Monetary) features."""
        n = len(financial)
        
        # Recency: Days since last transaction (inversely correlated with transaction frequency)
        transaction_factor = financial['num_transactions'] / financial['num_transactions'].max()
        recency_days = np.random.exponential(30, n) * (1 - transaction_factor)
        recency_days = np.clip(recency_days, 1, 365)
        
        # Frequency: Transaction frequency (already have this)
        frequency_score = financial['num_transactions']
        
        # Monetary: Average transaction value
        avg_transaction_value = financial['monthly_spending'] / np.maximum(financial['num_transactions'], 1)
        
        return pd.DataFrame({
            'recency_days': recency_days,
            'frequency_score': frequency_score,
            'monetary_value': avg_transaction_value
        })
    
    def generate_customer_lifetime_value(self, demographics: pd.DataFrame, financial: pd.DataFrame) -> pd.Series:
        """Calculate customer lifetime value."""
        # Simple CLV calculation: (Average Order Value * Purchase Frequency * Customer Lifespan) - Acquisition Cost
        avg_order_value = financial['monthly_spending']
        purchase_frequency = financial['num_transactions'] / 12  # Monthly frequency
        customer_lifespan = financial['years_with_bank'] + 5  # Projected future years
        acquisition_cost = 100  # Fixed acquisition cost
        
        clv = (avg_order_value * purchase_frequency * customer_lifespan) - acquisition_cost
        return clv
    
    def generate_complete_dataset(self) -> pd.DataFrame:
        """Generate complete customer dataset with all features."""
        logger.info(f"Generating dataset for {self.config.n_customers} customers")
        
        # Generate base features
        demographics = self.generate_demographics()
        financial = self.generate_financial_features(demographics)
        behavioral = self.generate_behavioral_features(demographics, financial)
        rfm = self.generate_rfm_features(financial, behavioral)
        
        # Calculate CLV
        clv = self.generate_customer_lifetime_value(demographics, financial)
        
        # Combine all features
        dataset = pd.concat([
            demographics,
            financial,
            behavioral,
            rfm
        ], axis=1)
        
        dataset['customer_lifetime_value'] = clv
        
        # Add churn probability (for evaluation)
        churn_prob = self._calculate_churn_probability(dataset)
        dataset['churn_probability'] = churn_prob
        dataset['is_churned'] = np.random.binomial(1, churn_prob, len(dataset))
        
        logger.info(f"Generated dataset with {len(dataset)} customers and {len(dataset.columns)} features")
        return dataset
    
    def _calculate_churn_probability(self, dataset: pd.DataFrame) -> np.ndarray:
        """Calculate churn probability based on customer features."""
        # Factors that increase churn probability
        low_balance_factor = (dataset['account_balance'] < dataset['account_balance'].quantile(0.25)).astype(int) * 0.2
        high_service_calls = (dataset['service_calls'] > dataset['service_calls'].quantile(0.75)).astype(int) * 0.15
        low_transactions = (dataset['num_transactions'] < dataset['num_transactions'].quantile(0.25)).astype(int) * 0.1
        recent_customer = (dataset['years_with_bank'] < 1).astype(int) * 0.1
        
        # Factors that decrease churn probability
        high_clv_factor = (dataset['customer_lifetime_value'] > dataset['customer_lifetime_value'].quantile(0.75)).astype(int) * -0.2
        multiple_products = (dataset['has_investments'] + dataset['has_insurance']).astype(int) * -0.1
        
        base_churn = 0.1  # Base 10% churn rate
        churn_prob = base_churn + low_balance_factor + high_service_calls + low_transactions + recent_customer + high_clv_factor + multiple_products
        
        return np.clip(churn_prob, 0, 1)


def load_customer_data(file_path: Optional[str] = None, config: Optional[CustomerDataConfig] = None) -> pd.DataFrame:
    """Load customer data from file or generate synthetic data."""
    if file_path and pd.io.common.file_exists(file_path):
        logger.info(f"Loading customer data from {file_path}")
        return pd.read_csv(file_path)
    else:
        if config is None:
            config = CustomerDataConfig()
        generator = CustomerDataGenerator(config)
        return generator.generate_complete_dataset()


def validate_customer_data(df: pd.DataFrame) -> bool:
    """Validate customer data for completeness and consistency."""
    required_columns = [
        'customer_id', 'age', 'gender', 'annual_income', 'account_balance',
        'num_transactions', 'loan_amount', 'credit_score', 'customer_lifetime_value'
    ]
    
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        return False
    
    # Check for negative values where they shouldn't exist
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if col in ['age', 'annual_income', 'account_balance', 'loan_amount', 'credit_score']:
            if (df[col] < 0).any():
                logger.error(f"Negative values found in {col}")
                return False
    
    logger.info("Data validation passed")
    return True
