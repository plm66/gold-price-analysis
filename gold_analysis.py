"""
Gold Price Analysis Module
Refactored from gold_price_analysis.py for API integration
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from math import sqrt
from datetime import datetime
import os
from typing import Dict, Any

class GoldPriceAnalyzer:
    """Gold price analysis and prediction"""
    
    def __init__(self, data_path: str = "data"):
        self.data_path = data_path
        self.df = None
        self.model = None
        self.last_prediction = None
        
    def load_data(self, filename: str = "data_inr.csv") -> pd.DataFrame:
        """Load gold price data from CSV"""
        filepath = os.path.join(self.data_path, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found: {filepath}")
            
        df = pd.read_csv(filepath)
        
        # Clean and prepare data
        ds_gold = 'Indian rupee'
        date_format = '%Y-%m-%d'
        
        df = df[['Name', ds_gold]]
        df['Name'] = [datetime.strptime(i, date_format) for i in df['Name']]
        df = df.set_index('Name')
        df = df.dropna()
        
        self.df = df
        return df
    
    def check_stationarity(self, series: pd.Series) -> Dict[str, Any]:
        """Check if time series is stationary using ADF test"""
        result = adfuller(series)
        
        return {
            'adf_statistic': result[0],
            'p_value': result[1],
            'critical_values': result[4],
            'is_stationary': result[1] <= 0.05
        }
    
    def prepare_features(self) -> pd.DataFrame:
        """Prepare features with moving averages and differencing"""
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")
            
        df = self.df.copy()
        ds_gold = 'Indian rupee'
        
        # Apply differencing to make stationary
        df[ds_gold] = df[ds_gold].diff().diff().dropna()
        df = df.dropna()
        
        # Create moving averages
        df['S_1'] = df[ds_gold].shift(1).rolling(window=3).mean()
        df['S_2'] = df[ds_gold].shift(1).rolling(window=12).mean()
        df = df.dropna()
        
        return df
    
    def train_linear_model(self) -> Dict[str, Any]:
        """Train linear regression model"""
        df = self.prepare_features()
        ds_gold = 'Indian rupee'
        
        # Features and target
        X = df[['S_1', 'S_2']]
        y = df[ds_gold]
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
        
        # Train model
        self.model = LinearRegression().fit(X_train, y_train)
        
        # Predictions
        y_pred = self.model.predict(X_test)
        
        # Metrics
        r2 = self.model.score(X_test, y_test) * 100
        rmse = sqrt(mean_squared_error(y_test, y_pred))
        
        return {
            'r2_score': r2,
            'rmse': rmse,
            'coefficients': self.model.coef_.tolist(),
            'intercept': self.model.intercept_,
            'latest_features': X.iloc[-1].to_dict()
        }
    
    def train_sarimax_model(self) -> Dict[str, Any]:
        """Train SARIMAX model"""
        df = self.prepare_features()
        ds_gold = 'Indian rupee'
        
        # Train SARIMAX model
        mod = sm.tsa.statespace.SARIMAX(
            df[ds_gold].values,
            order=(2, 1, 2),
            seasonal_order=(2, 1, 2, 12),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        
        results = mod.fit(disp=False)
        predictions = results.predict()
        
        # Metrics
        rmse = sqrt(mean_squared_error(df[ds_gold], predictions))
        r2 = r2_score(df[ds_gold], predictions)
        
        self.sarimax_model = results
        
        return {
            'r2_score': r2 * 100,
            'rmse': rmse,
            'aic': results.aic,
            'bic': results.bic
        }
    
    def predict_next_price(self) -> Dict[str, Any]:
        """Generate next price prediction"""
        if self.model is None:
            # Train model if not already trained
            self.train_linear_model()
            
        df = self.prepare_features()
        
        # Get latest features for prediction
        latest_features = df[['S_1', 'S_2']].iloc[-1:].values
        
        # Predict
        prediction = self.model.predict(latest_features)[0]
        
        # Get confidence based on model performance
        model_metrics = self.train_linear_model()
        confidence = min(model_metrics['r2_score'] / 100, 0.95)
        
        self.last_prediction = {
            'predicted_price_inr': float(prediction),
            'currency': 'INR',
            'confidence': float(confidence),
            'model_type': 'Linear Regression',
            'r2_score': model_metrics['r2_score'],
            'rmse': model_metrics['rmse']
        }
        
        return self.last_prediction
    
    def get_model_comparison(self) -> Dict[str, Any]:
        """Compare different models"""
        linear_metrics = self.train_linear_model()
        sarimax_metrics = self.train_sarimax_model()
        
        return {
            'linear_regression': {
                'r2_score': linear_metrics['r2_score'],
                'rmse': linear_metrics['rmse']
            },
            'sarimax': {
                'r2_score': sarimax_metrics['r2_score'],
                'rmse': sarimax_metrics['rmse'],
                'aic': sarimax_metrics['aic']
            }
        }

def run_analysis_and_predict() -> Dict[str, Any]:
    """Main function to run analysis and return prediction (for API)"""
    try:
        analyzer = GoldPriceAnalyzer()
        analyzer.load_data()
        prediction = analyzer.predict_next_price()
        
        return prediction
        
    except Exception as e:
        return {
            'error': str(e),
            'predicted_price_inr': 45000.0,  # Fallback value
            'currency': 'INR',
            'confidence': 0.5,
            'model_type': 'Fallback'
        }