"""
Fraud detection model server with ensemble predictions.
"""
import asyncio
import json
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field, validator
import redis
import mlflow
from feast import FeatureStore
import xgboost as xgb
import lightgbm as lgb
import torch
import torch.nn as nn

from src.utils.logger import get_logger
from src.monitoring.drift_detector import ConceptDriftDetector
from src.features.feature_engineering import FraudFeatureEngineer

logger = get_logger(__name__)

@dataclass
class ModelConfig:
    """Model configuration for ensemble."""
    xgb_model_path: str = "models/xgb_v1.json"
    lgb_model_path: str = "models/lgb_v1.txt"
    nn_model_path: str = "models/nn_v1.pth"
    feature_store_path: str = "feature_repo"
    ensemble_weights: Dict[str, float] = None
    
    def __post_init__(self):
        if self.ensemble_weights is None:
            self.ensemble_weights = {"xgb": 0.4, "lgb": 0.3, "nn": 0.3}

class TransactionRequest(BaseModel):
    """Request schema for fraud prediction."""
    transaction_id: str
    user_id: str
    amount: float = Field(..., gt=0)
    currency: str
    merchant_id: str
    merchant_category: str
    transaction_time: str
    device_id: Optional[str] = None
    ip_address: Optional[str] = None
    billing_zip: Optional[str] = None
    shipping_zip: Optional[str] = None
    
    @validator('transaction_time')
    def validate_timestamp(cls, v):
        try:
            pd.to_datetime(v)
            return v
        except:
            raise ValueError('Invalid timestamp format')

class FraudPrediction(BaseModel):
    """Response schema for fraud prediction."""
    transaction_id: str
    is_fraud: bool
    fraud_score: float = Field(..., ge=0, le=1)
    confidence: float = Field(..., ge=0, le=1)
    model_breakdown: Dict[str, float]
    features_used: List[str]
    explanation: Optional[Dict] = None
    processing_time_ms: float

class NeuralNetworkModel(nn.Module):
    """PyTorch neural network for fraud detection."""
    def __init__(self, input_dim: int, hidden_dims: List[int] = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class FraudModelServer:
    """Main model server handling ensemble predictions."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.logger = logger
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
        self.feature_store = FeatureStore(repo_path=config.feature_store_path)
        self.drift_detector = ConceptDriftDetector()
        self.feature_engineer = FraudFeatureEngineer()
        
        # Thread pool for parallel model inference
        self.thread_pool = ThreadPoolExecutor(max_workers=3)
        
        # Load models
        self.models = self._load_models()
        
        # Track metrics
        self.metrics = {
            "total_predictions": 0,
            "fraud_predictions": 0,
            "avg_latency_ms": 0.0
        }
    
    def _load_models(self) -> Dict:
        """Load all ensemble models."""
        models = {}
        
        # Load XGBoost model
        try:
            models["xgb"] = xgb.Booster()
            models["xgb"].load_model(self.config.xgb_model_path)
            self.logger.info("XGBoost model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load XGBoost model: {e}")
            raise
        
        # Load LightGBM model
        try:
            models["lgb"] = lgb.Booster(model_file=self.config.lgb_model_path)
            self.logger.info("LightGBM model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load LightGBM model: {e}")
            raise
        
        # Load Neural Network
        try:
            # Load model architecture and weights
            checkpoint = torch.load(self.config.nn_model_path, map_location='cpu')
            input_dim = checkpoint['input_dim']
            hidden_dims = checkpoint['hidden_dims']
            
            models["nn"] = NeuralNetworkModel(input_dim, hidden_dims)
            models["nn"].load_state_dict(checkpoint['model_state_dict'])
            models["nn"].eval()
            self.logger.info("Neural network model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load neural network: {e}")
            raise
        
        return models
    
    async def predict_fraud(self, transaction: TransactionRequest) -> FraudPrediction:
        """
        Main prediction method with ensemble voting.
        """
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = f"prediction:{transaction.transaction_id}"
            cached_result = self.redis_client.get(cache_key)
            if cached_result:
                self.logger.info(f"Cache hit for {transaction.transaction_id}")
                return FraudPrediction(**json.loads(cached_result))
            
            # Generate features
            features = await self._generate_features(transaction)
            
            # Run ensemble predictions in parallel
            prediction_tasks = [
                self._predict_xgb(features),
                self._predict_lgb(features),
                self._predict_nn(features)
            ]
            
            results = await asyncio.gather(*prediction_tasks)
            xgb_score, lgb_score, nn_score = results
            
            # Apply ensemble weights
            ensemble_score = (
                xgb_score * self.config.ensemble_weights["xgb"] +
                lgb_score * self.config.ensemble_weights["lgb"] +
                nn_score * self.config.ensemble_weights["nn"]
            )
            
            # Apply business rules (threshold optimization)
            is_fraud, confidence = self._apply_business_rules(
                ensemble_score, transaction.amount
            )
            
            # Generate explanation
            explanation = await self._generate_explanation(
                features, xgb_score, lgb_score, nn_score
            )
            
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000
            
            # Prepare response
            prediction = FraudPrediction(
                transaction_id=transaction.transaction_id,
                is_fraud=is_fraud,
                fraud_score=round(ensemble_score, 4),
                confidence=round(confidence, 4),
                model_breakdown={
                    "xgb": round(xgb_score, 4),
                    "lgb": round(lgb_score, 4),
                    "nn": round(nn_score, 4)
                },
                features_used=list(features.keys())[:10],  # Top 10 features
                explanation=explanation,
                processing_time_ms=round(processing_time, 2)
            )
            
            # Cache result (5 minute TTL)
            self.redis_client.setex(
                cache_key,
                300,
                json.dumps(prediction.dict())
            )
            
            # Update metrics
            self._update_metrics(prediction, processing_time)
            
            # Check for concept drift
            self._check_concept_drift(features, ensemble_score)
            
            return prediction
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    async def _generate_features(self, transaction: TransactionRequest) -> Dict:
        """Generate features from transaction data."""
        # Convert to DataFrame for feature engineering
        transaction_df = pd.DataFrame([transaction.dict()])
        
        # Add engineered features
        engineered_features = self.feature_engineer.transform(transaction_df)
        
        # Get online features from Feast
        try:
            entity_df = pd.DataFrame({
                'user_id': [transaction.user_id],
                'merchant_id': [transaction.merchant_id],
                'event_timestamp': [pd.to_datetime(transaction.transaction_time)]
            })
            
            online_features = self.feature_store.get_online_features(
                entity_rows=[{
                    'user_id': transaction.user_id,
                    'merchant_id': transaction.merchant_id
                }]
            ).to_dict()
            
            # Combine all features
            all_features = {**engineered_features, **online_features}
            return all_features
            
        except Exception as e:
            self.logger.warning(f"Failed to fetch online features: {e}")
            return engineered_features
    
    async def _predict_xgb(self, features: Dict) -> float:
        """XGBoost prediction."""
        loop = asyncio.get_event_loop()
        
        # Convert features to DMatrix format
        feature_values = list(features.values())
        feature_array = np.array([feature_values], dtype=np.float32)
        
        # Run in thread pool
        score = await loop.run_in_executor(
            self.thread_pool,
            lambda: self.models["xgb"].predict(
                xgb.DMatrix(feature_array)
            )[0]
        )
        
        return float(score)
    
    async def _predict_lgb(self, features: Dict) -> float:
        """LightGBM prediction."""
        loop = asyncio.get_event_loop()
        
        # Convert features to correct format
        feature_values = list(features.values())
        feature_array = np.array([feature_values], dtype=np.float32)
        
        # Run in thread pool
        score = await loop.run_in_executor(
            self.thread_pool,
            lambda: self.models["lgb"].predict(feature_array)[0]
        )
        
        return float(score)
    
    async def _predict_nn(self, features: Dict) -> float:
        """Neural network prediction."""
        loop = asyncio.get_event_loop()
        
        # Convert features to tensor
        feature_values = list(features.values())
        feature_tensor = torch.tensor([feature_values], dtype=torch.float32)
        
        # Run in thread pool
        def predict():
            with torch.no_grad():
                return self.models["nn"](feature_tensor).item()
        
        score = await loop.run_in_executor(self.thread_pool, predict)
        return float(score)
    
    def _apply_business_rules(self, score: float, amount: float) -> Tuple[bool, float]:
        """
        Apply business rules and adaptive thresholding.
        Higher amounts get lower thresholds.
        """
        # Dynamic threshold based on transaction amount
        base_threshold = 0.85
        
        # Lower threshold for high-value transactions
        if amount > 10000:
            threshold = base_threshold * 0.7
        elif amount > 5000:
            threshold = base_threshold * 0.8
        elif amount > 1000:
            threshold = base_threshold * 0.9
        else:
            threshold = base_threshold
        
        is_fraud = score > threshold
        
        # Calculate confidence based on distance from threshold
        confidence = abs(score - threshold) / max(score, threshold)
        confidence = min(confidence, 1.0)
        
        return is_fraud, confidence
    
    async def _generate_explanation(self, features: Dict, *scores) -> Dict:
        """Generate SHAP explanations for model decisions."""
        # Simplified explanation - in production would use SHAP/LIME
        feature_importance = {
            "top_features": sorted(
                features.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:5]
        }
        
        return {
            "feature_importance": feature_importance,
            "model_agreement": self._calculate_model_agreement(scores),
            "risk_factors": self._identify_risk_factors(features)
        }
    
    def _calculate_model_agreement(self, scores: Tuple) -> str:
        """Calculate agreement level between models."""
        std = np.std(scores)
        if std < 0.1:
            return "high"
        elif std < 0.2:
            return "medium"
        else:
            return "low"
    
    def _identify_risk_factors(self, features: Dict) -> List[str]:
        """Identify key risk factors from features."""
        risk_factors = []
        
        # Example risk factors
        if features.get("transaction_amount_zscore", 0) > 3:
            risk_factors.append("Unusually high transaction amount")
        if features.get("user_transaction_frequency_1h", 0) > 10:
            risk_factors.append("High transaction frequency")
        if features.get("is_new_device", 0) == 1:
            risk_factors.append("New device used")
        
        return risk_factors
    
    def _update_metrics(self, prediction: FraudPrediction, processing_time: float):
        """Update server metrics."""
        self.metrics["total_predictions"] += 1
        
        if prediction.is_fraud:
            self.metrics["fraud_predictions"] += 1
        
        # Update rolling average latency
        prev_avg = self.metrics["avg_latency_ms"]
        n = self.metrics["total_predictions"]
        self.metrics["avg_latency_ms"] = (
            (prev_avg * (n - 1) + processing_time) / n
        )
    
    def _check_concept_drift(self, features: Dict, score: float):
        """Check for concept drift in incoming data."""
        try:
            drift_detected = self.drift_detector.check_drift(
                features=features,
                prediction=score
            )
            
            if drift_detected:
                self.logger.warning("Concept drift detected!")
                # Trigger retraining workflow
                asyncio.create_task(self._trigger_retraining())
        
        except Exception as e:
            self.logger.error(f"Drift detection failed: {e}")

# FastAPI Application
app = FastAPI(
    title="Fraud Detection API",
    description="Real-time fraud detection with ensemble ML models",
    version="1.0.0"
)

# Initialize server
model_server = FraudModelServer(ModelConfig())

@app.post("/predict", response_model=FraudPrediction)
async def predict(request: TransactionRequest):
    """Main prediction endpoint."""
    return await model_server.predict_fraud(request)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "models_loaded": len(model_server.models),
        "metrics": model_server.metrics
    }

@app.get("/models")
async def list_models():
    """List loaded models and their configurations."""
    return {
        "ensemble_weights": model_server.config.ensemble_weights,
        "models": list(model_server.models.keys()),
        "feature_store": model_server.feature_store.config.project
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
