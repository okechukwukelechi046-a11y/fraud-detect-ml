"""
Feature engineering pipeline for fraud detection.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from sklearn.preprocessing import StandardScaler
import hashlib

class FraudFeatureEngineer:
    """Engineers features for fraud detection from transaction data."""
    
    def __init__(self):
        self.scalers = {}
        self.feature_config = self._load_feature_config()
    
    def _load_feature_config(self) -> Dict:
        """Load feature engineering configuration."""
        return {
            "time_features": ["hour", "day_of_week", "day_of_month"],
            "amount_features": ["log_amount", "amount_zscore"],
            "user_features": ["avg_user_amount", "user_frequency"],
            "merchant_features": ["merchant_risk_score"],
            "behavioral_features": ["velocity_1h", "velocity_24h"],
            "device_features": ["device_trust_score"],
            "geographic_features": ["distance_from_home"]
        }
    
    def fit(self, historical_data: pd.DataFrame):
        """Fit feature engineering on historical data."""
        # Fit scalers
        self.scalers["amount"] = StandardScaler()
        self.scalers["amount"].fit(historical_data[["amount"]])
        
        # Calculate user statistics
        self.user_stats = historical_data.groupby("user_id").agg({
            "amount": ["mean", "std"],
            "transaction_time": "count"
        }).fillna(0)
        
        # Calculate merchant statistics
        self.merchant_stats = historical_data.groupby("merchant_id").agg({
            "is_fraud": "mean"  # Historical fraud rate
        }).fillna(0)
        
        return self
    
    def transform(self, transaction_df: pd.DataFrame) -> Dict[str, float]:
        """Transform single transaction into features."""
        features = {}
        
        # Basic transaction features
        features.update(self._extract_basic_features(transaction_df))
        
        # Time-based features
        features.update(self._extract_time_features(transaction_df))
        
        # Amount features
        features.update(self._extract_amount_features(transaction_df))
        
        # User behavior features
        features.update(self._extract_user_features(transaction_df))
        
        # Merchant features
        features.update(self._extract_merchant_features(transaction_df))
        
        # Behavioral velocity features
        features.update(self._extract_velocity_features(transaction_df))
        
        # Device and location features
        features.update(self._extract_device_features(transaction_df))
        features.update(self._extract_geographic_features(transaction_df))
        
        # Derived features
        features.update(self._create_derived_features(features))
        
        # Ensure no NaN values
        features = {k: (0.0 if pd.isna(v) else float(v)) 
                   for k, v in features.items()}
        
        return features
    
    def _extract_basic_features(self, df: pd.DataFrame) -> Dict:
        """Extract basic transaction features."""
        row = df.iloc[0]
        return {
            "transaction_amount": float(row["amount"]),
            "currency_code": self._encode_currency(row.get("currency", "USD")),
            "merchant_category_code": self._encode_mcc(row.get("merchant_category", "")),
        }
    
    def _extract_time_features(self, df: pd.DataFrame) -> Dict:
        """Extract time-based features."""
        row = df.iloc[0]
        timestamp = pd.to_datetime(row["transaction_time"])
        
        return {
            "transaction_hour": float(timestamp.hour),
            "transaction_day_of_week": float(timestamp.dayofweek),
            "transaction_day_of_month": float(timestamp.day),
            "is_weekend": float(timestamp.dayofweek >= 5),
            "is_night": float(22 <= timestamp.hour or timestamp.hour <= 6),
            "time_since_last_tx": self._calculate_time_since_last_tx(row)
        }
    
    def _extract_amount_features(self, df: pd.DataFrame) -> Dict:
        """Extract amount-related features."""
        row = df.iloc[0]
        amount = float(row["amount"])
        
        features = {
            "log_amount": np.log1p(amount),
            "amount_category": self._categorize_amount(amount)
        }
        
        # Calculate z-score if scaler is fitted
        if "amount" in self.scalers:
            zscore = self.scalers["amount"].transform([[amount]])[0][0]
            features["amount_zscore"] = zscore
        
        return features
    
    def _extract_user_features(self, df: pd.DataFrame) -> Dict:
        """Extract user behavior features."""
        row = df.iloc[0]
        user_id = row["user_id"]
        
        features = {
            "is_new_user": float(0),  # Would check against user database
            "user_trust_score": float(0.5)  # Default
        }
        
        # Add historical user stats if available
        if hasattr(self, 'user_stats') and user_id in self.user_stats.index:
            user_data = self.user_stats.loc[user_id]
            features.update({
                "avg_user_amount": float(user_data[("amount", "mean")]),
                "std_user_amount": float(user_data[("amount", "std")]),
                "user_transaction_count": float(user_data[("transaction_time", "count")])
            })
        
        return features
    
    def _extract_merchant_features(self, df: pd.DataFrame) -> Dict:
        """Extract merchant risk features."""
        row = df.iloc[0]
        merchant_id = row["merchant_id"]
        
        features = {
            "is_high_risk_merchant": float(0),
            "merchant_transaction_count": float(0)
        }
        
        # Add merchant fraud history if available
        if hasattr(self, 'merchant_stats') and merchant_id in self.merchant_stats.index:
            merchant_data = self.merchant_stats.loc[merchant_id]
            features["merchant_fraud_rate"] = float(merchant_data["is_fraud"])
        
        return features
    
    def _extract_velocity_features(self, df: pd.DataFrame) -> Dict:
        """Extract behavioral velocity features."""
        # In production, this would query a real-time database
        # For now, return placeholder features
        return {
            "user_transaction_frequency_1h": float(0),
            "user_transaction_frequency_24h": float(0),
            "amount_velocity_1h": float(0),
            "amount_velocity_24h": float(0)
        }
    
    def _extract_device_features(self, df: pd.DataFrame) -> Dict:
        """Extract device fingerprint features."""
        row = df.iloc[0]
        device_id = row.get("device_id", "")
        
        features = {
            "is_new_device": float(0),
            "device_trust_score": float(0.5)
        }
        
        if device_id:
            # Create hash-based device fingerprint
            device_hash = hashlib.md5(device_id.encode()).hexdigest()[:8]
            features["device_fingerprint"] = float(int(device_hash, 16))
        
        return features
    
    def _extract_geographic_features(self, df: pd.DataFrame) -> Dict:
        """Extract geographic features."""
        row = df.iloc[0]
        
        features = {
            "is_international": float(0),
            "distance_from_home": float(0)
        }
        
        # Check if billing and shipping zip codes differ
        billing_zip = row.get("billing_zip", "")
        shipping_zip = row.get("shipping_zip", "")
        
        if billing_zip and shipping_zip:
            features["zip_code_mismatch"] = float(billing_zip != shipping_zip)
        
        return features
    
    def _create_derived_features(self, features: Dict) -> Dict:
        """Create derived features from existing ones."""
        derived = {}
        
        # Risk score aggregation
        risk_score = (
            features.get("amount_zscore", 0) * 0.3 +
            features.get("is_night", 0) * 0.2 +
            features.get("is_weekend", 0) * 0.1 +
            features.get("is_new_device", 0) * 0.4
        )
        derived["aggregated_risk_score"] = risk_score
        
        # Interaction features
        if "transaction_hour" in features and "is_weekend" in features:
            derived["hour_weekend_interaction"] = (
                features["transaction_hour"] * features["is_weekend"]
            )
        
        return derived
    
    def _encode_currency(self, currency: str) -> float:
        """Encode currency as numeric feature."""
        currency_map = {"USD": 1, "EUR": 2, "GBP": 3, "JPY": 4}
        return float(currency_map.get(currency, 99))
    
    def _encode_mcc(self, mcc: str) -> float:
        """Encode merchant category code."""
        try:
            return float(mcc) if mcc else 0.0
        except:
            return 0.0
    
    def _categorize_amount(self, amount: float) -> float:
        """Categorize amount into bins."""
        if amount < 50:
            return 1.0
        elif amount < 500:
            return 2.0
        elif amount < 5000:
            return 3.0
        else:
            return 4.0
    
    def _calculate_time_since_last_tx(self, row) -> float:
        """Calculate time since user's last transaction."""
        # In production, this would query a real-time database
        # Return placeholder value
        return 3600.0  # 1 hour in seconds
