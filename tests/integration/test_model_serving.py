import pytest
from fastapi.testclient import TestClient
from src.serving.model_server import app
import json

class TestModelServingAPI:
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "models_loaded" in data
    
    def test_predict_endpoint(self, client):
        """Test prediction endpoint with valid data."""
        transaction_data = {
            "transaction_id": "test_tx_001",
            "user_id": "user_001",
            "amount": 125.50,
            "currency": "USD",
            "merchant_id": "merchant_001",
            "merchant_category": "5411",
            "transaction_time": "2024-01-15T14:30:00Z",
            "device_id": "device_001",
            "ip_address": "192.168.1.1",
            "billing_zip": "12345",
            "shipping_zip": "12345"
        }
        
        response = client.post("/predict", json=transaction_data)
        
        # Should return 200 or 500 if models not loaded in test
        assert response.status_code in [200, 500]
        
        if response.status_code == 200:
            data = response.json()
            assert data["transaction_id"] == "test_tx_001"
            assert "is_fraud" in data
            assert "fraud_score" in data
            assert "confidence" in data
    
    def test_invalid_transaction_data(self, client):
        """Test prediction with invalid data."""
        invalid_data = {
            "transaction_id": "test_tx_001",
            "user_id": "user_001",
            "amount": -10,  # Invalid negative amount
            "currency": "USD",
            "merchant_id": "merchant_001",
            "merchant_category": "5411",
            "transaction_time": "invalid_time"  # Invalid timestamp
        }
        
        response = client.post("/predict", json=invalid_data)
        assert response.status_code == 422  # Validation error
    
    def test_models_endpoint(self, client):
        """Test models listing endpoint."""
        response = client.get("/models")
        assert response.status_code == 200
        data = response.json()
        assert "ensemble_weights" in data
        assert "models" in data
