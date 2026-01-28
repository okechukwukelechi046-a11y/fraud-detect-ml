"""
End-to-end training pipeline for fraud detection models.
"""
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
import mlflow.pytorch
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, average_precision_score,
    classification_report, confusion_matrix
)
import xgboost as xgb
import lightgbm as lgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib
import json

from src.features.feature_engineering import FraudFeatureEngineer
from src.utils.data_loader import FraudDataLoader
from src.monitoring.model_evaluator import ModelEvaluator

class FraudModelTrainer:
    """Orchestrates training of ensemble fraud detection models."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.feature_engineer = FraudFeatureEngineer()
        self.data_loader = FraudDataLoader()
        self.evaluator = ModelEvaluator()
        
        # MLflow tracking
        mlflow.set_tracking_uri(config.get("mlflow_uri", "http://localhost:5000"))
        mlflow.set_experiment(config.get("experiment_name", "fraud_detection"))
    
    def run_pipeline(self):
        """Run complete training pipeline."""
        with mlflow.start_run(run_name=f"fraud_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            
            # 1. Load and prepare data
            self.logger.info("Loading data...")
            data = self.data_loader.load_training_data()
            
            # 2. Feature engineering
            self.logger.info("Engineering features...")
            X, y, feature_names = self._prepare_features(data)
            
            # 3. Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=42
            )
            
            # 4. Train models
            self.logger.info("Training XGBoost...")
            xgb_model, xgb_metrics = self._train_xgboost(X_train, y_train, X_test, y_test)
            
            self.logger.info("Training LightGBM...")
            lgb_model, lgb_metrics = self._train_lightgbm(X_train, y_train, X_test, y_test)
            
            self.logger.info("Training Neural Network...")
            nn_model, nn_metrics = self._train_neural_network(X_train, y_train, X_test, y_test)
            
            # 5. Ensemble evaluation
            self.logger.info("Evaluating ensemble...")
            ensemble_metrics = self._evaluate_ensemble(
                [xgb_model, lgb_model, nn_model],
                X_test, y_test
            )
            
            # 6. Log everything to MLflow
            self._log_training_artifacts(
                xgb_model, lgb_model, nn_model,
                xgb_metrics, lgb_metrics, nn_metrics, ensemble_metrics,
                feature_names
            )
            
            # 7. Register best model
            best_model = self._select_best_model(
                [xgb_metrics, lgb_metrics, nn_metrics],
                [xgb_model, lgb_model, nn_model]
            )
            
            mlflow.register_model(
                f"runs:/{mlflow.active_run().info.run_id}/model",
                "fraud_detection"
            )
            
            self.logger.info("Training pipeline completed successfully!")
            
            return {
                "models": {
                    "xgb": xgb_model,
                    "lgb": lgb_model,
                    "nn": nn_model
                },
                "metrics": {
                    "xgb": xgb_metrics,
                    "lgb": lgb_metrics,
                    "nn": nn_metrics,
                    "ensemble": ensemble_metrics
                }
            }
    
    def _prepare_features(self, data: pd.DataFrame) -> Tuple:
        """Prepare features for training."""
        # Fit feature engineer on training data
        self.feature_engineer.fit(data[data["is_test"] == 0])
        
        # Transform all data
        features = []
        for _, row in data.iterrows():
            feature_dict = self.feature_engineer.transform(pd.DataFrame([row]))
            features.append(list(feature_dict.values()))
        
        X = np.array(features)
        y = data["is_fraud"].values
        feature_names = list(feature_dict.keys())
        
        # Handle class imbalance (fraud is rare)
        fraud_indices = np.where(y == 1)[0]
        non_fraud_indices = np.where(y == 0)[0]
        
        # Undersample majority class
        n_fraud = len(fraud_indices)
        sampled_non_fraud = np.random.choice(
            non_fraud_indices,
            size=min(n_fraud * 10, len(non_fraud_indices)),  # 10:1 ratio
            replace=False
        )
        
        balanced_indices = np.concatenate([fraud_indices, sampled_non_fraud])
        X_balanced = X[balanced_indices]
        y_balanced = y[balanced_indices]
        
        return X_balanced, y_balanced, feature_names
    
    def _train_xgboost(self, X_train, y_train, X_test, y_test):
        """Train XGBoost model with cross-validation."""
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'scale_pos_weight': len(y_train[y_train==0]) / len(y_train[y_train==1])
        }
        
        # Cross-validation
        cv_scores = []
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for train_idx, val_idx in kfold.split(X_train, y_train):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]
            
            dtrain = xgb.DMatrix(X_tr, label=y_tr)
            dval = xgb.DMatrix(X_val, label=y_val)
            
            model = xgb.train(
                params,
                dtrain,
                num_boost_round=1000,
                evals=[(dval, 'eval')],
                early_stopping_rounds=50,
                verbose_eval=False
            )
            
            preds = model.predict(dval)
            score = roc_auc_score(y_val, preds)
            cv_scores.append(score)
        
        # Final model
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        
        final_model = xgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            evals=[(dtest, 'test')],
            early_stopping_rounds=50,
            verbose_eval=False
        )
        
        # Evaluate
        test_preds = final_model.predict(dtest)
        metrics = self.evaluator.calculate_metrics(y_test, test_preds)
        metrics["cv_mean_auc"] = np.mean(cv_scores)
        metrics["cv_std_auc"] = np.std(cv_scores)
        
        # Feature importance
        importance = final_model.get_score(importance_type='gain')
        metrics["top_features"] = dict(sorted(
            importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10])
        
        return final_model, metrics
    
    def _train_lightgbm(self, X_train, y_train, X_test, y_test):
        """Train LightGBM model."""
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.1,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42,
            'is_unbalance': True
        }
        
        train_data = lgb.Dataset(X_train, label=y_train)
        test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[test_data],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        
        # Evaluate
        test_preds = model.predict(X_test)
        metrics = self.evaluator.calculate_metrics(y_test, test_preds)
        
        return model, metrics
    
    def _train_neural_network(self, X_train, y_train, X_test, y_test):
        """Train PyTorch neural network."""
        input_dim = X_train.shape[1]
        
        # Create datasets
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train).unsqueeze(1)
        )
        test_dataset = TensorDataset(
            torch.FloatTensor(X_test),
            torch.FloatTensor(y_test).unsqueeze(1)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64)
        
        # Define model
        class FraudNet(nn.Module):
            def __init__(self, input_dim):
                super().__init__()
                self.network = nn.Sequential(
                    nn.Linear(input_dim, 128),
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    
                    nn.Linear(128, 64),
                    nn.BatchNorm1d(64),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    
                    nn.Linear(64, 32),
                    nn.BatchNorm1d(32),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    
                    nn.Linear(32, 1),
                    nn.Sigmoid()
                )
            
            def forward(self, x):
                return self.network(x)
        
        model = FraudNet(input_dim)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Training loop
        n_epochs = 50
        for epoch in range(n_epochs):
            model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_preds = []
            val_labels = []
            
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    outputs = model(batch_X)
                    val_preds.extend(outputs.cpu().numpy())
                    val_labels.extend(batch_y.cpu().numpy())
            
            val_auc = roc_auc_score(val_labels, val_preds)
            
            if epoch % 10 == 0:
                self.logger.info(f"Epoch {epoch}: Loss={train_loss/len(train_loader):.4f}, AUC={val_auc:.4f}")
        
        # Final evaluation
        model.eval()
        all_preds = []
        with torch.no_grad():
            for batch_X, _ in test_loader:
                outputs = model(batch_X)
                all_preds.extend(outputs.cpu().numpy())
        
        metrics = self.evaluator.calculate_metrics(y_test, np.array(all_preds))
        
        # Save model checkpoint
        checkpoint = {
            'input_dim': input_dim,
            'model_state_dict': model.state_dict(),
            'hidden_dims': [128, 64, 32]
        }
        
        return checkpoint, metrics
    
    def _evaluate_ensemble(self, models, X_test, y_test):
        """Evaluate ensemble of models."""
        # Get predictions from all models
        predictions = []
        
        # XGBoost predictions
        dtest = xgb.DMatrix(X_test)
        xgb_preds = models[0].predict(dtest)
        predictions.append(xgb_preds)
        
        # LightGBM predictions
        lgb_preds = models[1].predict(X_test)
        predictions.append(lgb_preds)
        
        # Neural network predictions
        nn_checkpoint = models[2]
        nn_model = self._load_nn_from_checkpoint(nn_checkpoint)
        nn_model.eval()
        
        with torch.no_grad():
            nn_preds = nn_model(torch.FloatTensor(X_test)).cpu().numpy()
        predictions.append(nn_preds.flatten())
        
        # Simple average ensemble
        ensemble_preds = np.mean(predictions, axis=0)
        
        # Weighted ensemble (could be optimized)
        weights = [0.4, 0.3, 0.3]  # XGB, LGB, NN
        weighted_preds = sum(w * p for w, p in zip(weights, predictions))
        
        # Evaluate both
        avg_metrics = self.evaluator.calculate_metrics(y_test, ensemble_preds)
        weighted_metrics = self.evaluator.calculate_metrics(y_test, weighted_preds)
        
        return {
            "average_ensemble": avg_metrics,
            "weighted_ensemble": weighted_metrics,
            "individual_predictions": {
                "xgb": predictions[0].tolist()[:10],  # Sample
                "lgb": predictions[1].tolist()[:10],
                "nn": predictions[2].tolist()[:10]
            }
        }
    
    def _log_training_artifacts(self, *args):
        """Log all artifacts to MLflow."""
        # Implementation for logging models, metrics, and artifacts
        pass
    
    def _select_best_model(self, metrics_list, models):
        """Select best model based on validation metrics."""
        # Implementation for model selection
        pass
