import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from src.models.training_pipeline import FraudModelTrainer
from src.serving.model_server import FraudModelServer, TransactionRequest

class TestFraudModelServer:
    @pytest.fixture
    def model_server(self):
        config = {
            "xgb_model_path": "tests/test_data/xgb_test.json",
            "lgb_model_path": "tests/test_data/lgb_test.txt",
            "nn_model_path": "tests/test_data/nn_test.pth",
            "feature_store_path": "tests/test_feature_repo"
        }
        return FraudModelServer(ModelConfig(**config))
    
    @pytest.fixture
    def sample_transaction(self):
        return TransactionRequest(
            transaction_id="test_123",
            user_id="user_456",
            amount=150.75,
            currency="USD",
            merchant_id="merchant_789",
            merchant_category="5411",
            transaction_time="2024-01-15T14:30:00Z"
        )
    
    @pytest.mark.asyncio
    async def test_predict_fraud_returns_valid_response(self, model_server, sample_transaction):
        """Test that prediction returns valid response format."""
        with patch.object(model_server, '_generate_features') as mock_features:
            mock_features.return_value = {
                "transaction_amount": 150.75,
                "transaction_hour": 14.0,
                "amount_zscore": 1.2,
                "is_new_device": 0.0
            }
            
            with patch.object(model_server, '_predict_xgb') as mock_xgb:
                mock_xgb.return_value = 0.15
                
            with patch.object(model_server, '_predict_lgb') as mock_lgb:
                mock_lgb.return_value = 0.22
                
            with patch.object(model_server, '_predict_nn') as mock_nn:
                mock_nn.return_value = 0.18
            
            result = await model_server.predict_fraud(sample_transaction)
            
            assert result.transaction_id == "test_123"
            assert isinstance(result.is_fraud, bool)
            assert 0 <= result.fraud_score <= 1
            assert 0 <= result.confidence <= 1
            assert "xgb" in result.model_breakdown
            assert len(result.features_used) > 0
    
    def test_business_rules_thresholding(self, model_server):
        """Test business rule threshold adjustments."""
        # Test high amount gets lower threshold
        is_fraud_low, confidence_low = model_server._apply_business_rules(0.6, 20000)
        is_fraud_high, confidence_high = model_server._apply_business_rules(0.6, 100)
        
        # High amount should be flagged at lower score
        assert is_fraud_low != is_fraud_high
        
        # Confidence should reflect distance from threshold
        assert 0 <= confidence_low <= 1
        assert 0 <= confidence_high <= 1
    
    @pytest.mark.asyncio
    async def test_feature_generation(self, model_server, sample_transaction):
        """Test feature generation from transaction data."""
        features = await model_server._generate_features(sample_transaction)
        
        assert isinstance(features, dict)
        assert len(features) > 0
        
        # Check for expected features
        assert "transaction_amount" in features
        assert "transaction_hour" in features
        
        # All features should be numeric
        for value in features.values():
            assert isinstance(value, (int, float, np.number))
    
    def test_model_agreement_calculation(self, model_server):
        """Test calculation of model agreement."""
        scores_high_agreement = (0.25, 0.27, 0.26)
        scores_low_agreement = (0.15, 0.45, 0.30)
        
        agreement_high = model_server._calculate_model_agreement(scores_high_agreement)
        agreement_low = model_server._calculate_model_agreement(scores_low_agreement)
        
        assert agreement_high == "high"
        assert agreement_low == "low"

class TestFeatureEngineering:
    @pytest.fixture
    def feature_engineer(self):
        return FraudFeatureEngineer()
    
    @pytest.fixture
    def sample_transaction_df(self):
        return pd.DataFrame({
            "transaction_id": ["tx1"],
            "user_id": ["user1"],
            "amount": [150.75],
            "currency": ["USD"],
            "merchant_id": ["merchant1"],
            "merchant_category": ["5411"],
            "transaction_time": ["2024-01-15T14:30:00Z"],
            "device_id": ["device123"],
            "billing_zip": ["12345"],
            "shipping_zip": ["12345"]
        })
    
    def test_feature_transformation(self, feature_engineer, sample_transaction_df):
        """Test feature transformation produces correct output."""
        features = feature_engineer.transform(sample_transaction_df)
        
        assert isinstance(features, dict)
        assert len(features) > 20  # Should have many features
        
        # Check specific feature calculations
        assert "transaction_amount" in features
        assert features["transaction_amount"] == 150.75
        
        assert "log_amount" in features
        assert features["log_amount"] == np.log1p(150.75)
        
        assert "transaction_hour" in features
        assert features["transaction_hour"] == 14.0
        
        # No NaN values
        for value in features.values():
            assert not pd.isna(value)

class TestTrainingPipeline:
    @pytest.fixture
    def trainer(self):
        config = {
            "mlflow_uri": "http://localhost:5000",
            "experiment_name": "test_fraud_detection"
        }
        return FraudModelTrainer(config)
    
    def test_data_preparation(self, trainer):
        """Test data preparation and balancing."""
        # Create synthetic data
        n_samples = 1000
        n_features = 10
        
        X = np.random.randn(n_samples, n_features)
        y = np.zeros(n_samples)
        y[:50] = 1  # 5% fraud rate
        
        data = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
        data["is_fraud"] = y
        data["is_test"] = 0
        
        # Test feature preparation
        X_balanced, y_balanced, feature_names = trainer._prepare_features(data)
        
        # Check balancing worked
        fraud_count = np.sum(y_balanced == 1)
        non_fraud_count = np.sum(y_balanced == 0)
        
        # Should have reasonable ratio (not 10:1 due to small fraud sample)
        assert fraud_count > 0
        assert non_fraud_count > fraud_count
        assert len(feature_names) == n_features
    
    @pytest.mark.slow
    def test_xgboost_training(self, trainer):
        """Test XGBoost training with synthetic data."""
        # Generate synthetic data
        np.random.seed(42)
        n_samples = 500
        n_features = 20
        
        X = np.random.randn(n_samples, n_features)
        # Create synthetic fraud pattern
        fraud_mask = (
            (X[:, 0] > 1.5) &  # Feature 0 high
            (X[:, 5] < -1.0) &  # Feature 5 low
            (X[:, 12] > 2.0)    # Feature 12 high
        )
        y = fraud_mask.astype(int)
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        # Train
        model, metrics = trainer._train_xgboost(X_train, y_train, X_test, y_test)
        
        # Check model performance
        assert metrics["auc"] > 0.7  # Should learn the pattern
        assert "precision" in metrics
        assert "recall" in metrics
        
        # Check feature importance was calculated
        assert "top_features" in metrics
        assert len(metrics["top_features"]) > 0
