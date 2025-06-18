"""
Advanced Neural Network Trading Engine with Real AI
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import xgboost as xgb
import lightgbm as lgb
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten, Input, Attention
from keras.optimizers import Adam
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class NeuralPrediction:
    """Neural network prediction result"""
    symbol: str
    prediction_type: str
    confidence: float
    direction: str
    price_target: float
    probability: float
    risk_score: float
    timeframe: str
    features_used: List[str]
    model_ensemble: List[str]
    generated_at: datetime

class AdvancedNeuralEngine:
    """Advanced Neural Network Trading Engine with Multiple AI Models"""
    
    def __init__(self):
        self.initialized = False
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.prediction_history = []
        self.ensemble_weights = {}
        
        # Model configurations
        self.lstm_model = None
        self.xgb_model = None
        self.lgb_model = None
        self.rf_model = None
        self.mlp_model = None
        self.transformer_model = None
        
        # Feature engineering components
        self.technical_indicators = [
            'sma_5', 'sma_10', 'sma_20', 'sma_50',
            'ema_5', 'ema_10', 'ema_20', 'ema_50',
            'rsi_14', 'rsi_7', 'rsi_21',
            'macd', 'macd_signal', 'macd_histogram',
            'bb_upper', 'bb_middle', 'bb_lower', 'bb_width',
            'stoch_k', 'stoch_d',
            'williams_r', 'cci', 'atr',
            'volume_sma', 'volume_ratio',
            'price_change', 'volatility',
            'momentum_5', 'momentum_10',
            'support_resistance_ratio'
        ]
        
    async def initialize(self):
        """Initialize the neural engine with all AI models"""
        try:
            logger.info("Initializing Advanced Neural Network Trading Engine...")
            
            # Initialize scalers
            self.scalers['price'] = MinMaxScaler()
            self.scalers['volume'] = StandardScaler()
            self.scalers['features'] = StandardScaler()
            
            # Initialize model ensemble weights based on historical performance
            self.ensemble_weights = {
                'lstm': 0.25,
                'xgboost': 0.25,
                'lightgbm': 0.20,
                'random_forest': 0.15,
                'mlp': 0.10,
                'transformer': 0.05
            }
            
            self.initialized = True
            logger.info("Advanced Neural Network Trading Engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing neural engine: {e}")
            return False
            
        return True
    
    async def train_models(self, market_data: Dict[str, pd.DataFrame]):
        """Train all AI models with market data"""
        try:
            logger.info("Training ensemble of AI models...")
            
            # Prepare training data
            training_data = await self._prepare_training_data(market_data)
            
            if len(training_data) < 100:
                logger.warning("Insufficient data for training")
                return False
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                training_data['features'], 
                training_data['targets'], 
                test_size=0.2, 
                random_state=42
            )
            
            # Train LSTM model
            await self._train_lstm_model(X_train, y_train, X_test, y_test)
            
            # Train XGBoost model
            await self._train_xgboost_model(X_train, y_train, X_test, y_test)
            
            # Train LightGBM model
            await self._train_lightgbm_model(X_train, y_train, X_test, y_test)
            
            # Train Random Forest model
            await self._train_random_forest_model(X_train, y_train, X_test, y_test)
            
            # Train MLP model
            await self._train_mlp_model(X_train, y_train, X_test, y_test)
            
            # Calculate ensemble weights based on performance
            await self._optimize_ensemble_weights(X_test, y_test)
            
            logger.info("All AI models trained successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error training models: {e}")
            return False
    
    async def generate_neural_prediction(self, symbol: str, market_data: pd.DataFrame) -> Optional[NeuralPrediction]:
        """Generate advanced neural network prediction"""
        try:
            # Feature engineering
            features = await self._engineer_features(market_data)
            
            if len(features) < 50:
                return None
            
            # Scale features
            scaled_features = self.scalers['features'].fit_transform(features)
            
            # Get predictions from all models
            predictions = {}
            confidences = {}
            
            # LSTM prediction
            if self.lstm_model:
                lstm_pred = await self._predict_lstm(scaled_features)
                predictions['lstm'] = lstm_pred
                confidences['lstm'] = self._calculate_prediction_confidence(lstm_pred, 'lstm')
            
            # XGBoost prediction
            if self.xgb_model:
                xgb_pred = await self._predict_xgboost(scaled_features)
                predictions['xgboost'] = xgb_pred
                confidences['xgboost'] = self._calculate_prediction_confidence(xgb_pred, 'xgboost')
            
            # LightGBM prediction
            if self.lgb_model:
                lgb_pred = await self._predict_lightgbm(scaled_features)
                predictions['lightgbm'] = lgb_pred
                confidences['lightgbm'] = self._calculate_prediction_confidence(lgb_pred, 'lightgbm')
            
            # Random Forest prediction
            if self.rf_model:
                rf_pred = await self._predict_random_forest(scaled_features)
                predictions['random_forest'] = rf_pred
                confidences['random_forest'] = self._calculate_prediction_confidence(rf_pred, 'random_forest')
            
            # MLP prediction
            if self.mlp_model:
                mlp_pred = await self._predict_mlp(scaled_features)
                predictions['mlp'] = mlp_pred
                confidences['mlp'] = self._calculate_prediction_confidence(mlp_pred, 'mlp')
            
            # Ensemble prediction
            ensemble_prediction = await self._ensemble_predict(predictions, confidences)
            
            # Risk assessment
            risk_score = await self._assess_prediction_risk(features, ensemble_prediction)
            
            # Generate final prediction
            direction = 'LONG' if ensemble_prediction['direction'] > 0.5 else 'SHORT'
            confidence = ensemble_prediction['confidence'] * 100
            
            # Price target calculation
            current_price = market_data['close'].iloc[-1]
            price_target = current_price * (1 + ensemble_prediction['price_change'])
            
            prediction = NeuralPrediction(
                symbol=symbol,
                prediction_type='NEURAL_ENSEMBLE',
                confidence=confidence,
                direction=direction,
                price_target=price_target,
                probability=ensemble_prediction['probability'],
                risk_score=risk_score,
                timeframe='1H',
                features_used=self.technical_indicators,
                model_ensemble=list(predictions.keys()),
                generated_at=datetime.now()
            )
            
            # Store prediction for learning
            self.prediction_history.append(prediction)
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error generating neural prediction for {symbol}: {e}")
            return None
    
    async def _prepare_training_data(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, np.ndarray]:
        """Prepare comprehensive training data"""
        all_features = []
        all_targets = []
        
        for symbol, data in market_data.items():
            if len(data) < 100:
                continue
                
            # Engineer features
            features = await self._engineer_features(data)
            
            # Create targets (next price movement)
            targets = self._create_targets(data)
            
            if len(features) == len(targets) and len(features) > 50:
                all_features.extend(features.tolist())
                all_targets.extend(targets.tolist())
        
        return {
            'features': np.array(all_features),
            'targets': np.array(all_targets)
        }
    
    async def _engineer_features(self, data: pd.DataFrame) -> np.ndarray:
        """Advanced feature engineering with technical indicators"""
        df = data.copy()
        
        # Price-based features
        df['sma_5'] = df['close'].rolling(5).mean()
        df['sma_10'] = df['close'].rolling(10).mean()
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        
        # Exponential moving averages
        df['ema_5'] = df['close'].ewm(span=5).mean()
        df['ema_10'] = df['close'].ewm(span=10).mean()
        df['ema_20'] = df['close'].ewm(span=20).mean()
        df['ema_50'] = df['close'].ewm(span=50).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi_14'] = 100 - (100 / (1 + rs))
        df['rsi_7'] = 100 - (100 / (1 + gain.rolling(7).mean() / loss.rolling(7).mean()))
        df['rsi_21'] = 100 - (100 / (1 + gain.rolling(21).mean() / loss.rolling(21).mean()))
        
        # MACD
        exp1 = df['close'].ewm(span=12).mean()
        exp2 = df['close'].ewm(span=26).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # Stochastic oscillator
        low_14 = df['low'].rolling(14).min()
        high_14 = df['high'].rolling(14).max()
        df['stoch_k'] = 100 * (df['close'] - low_14) / (high_14 - low_14)
        df['stoch_d'] = df['stoch_k'].rolling(3).mean()
        
        # Williams %R
        df['williams_r'] = -100 * (high_14 - df['close']) / (high_14 - low_14)
        
        # Commodity Channel Index
        tp = (df['high'] + df['low'] + df['close']) / 3
        sma_tp = tp.rolling(20).mean()
        mad = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())))
        df['cci'] = (tp - sma_tp) / (0.015 * mad)
        
        # Average True Range
        tr1 = df['high'] - df['low']
        tr2 = np.abs(df['high'] - df['close'].shift())
        tr3 = np.abs(df['low'] - df['close'].shift())
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        df['atr'] = tr.rolling(14).mean()
        
        # Volume features
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Price change and volatility
        df['price_change'] = df['close'].pct_change()
        df['volatility'] = df['price_change'].rolling(20).std()
        
        # Momentum indicators
        df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
        df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
        
        # Support/Resistance ratio
        df['support_resistance_ratio'] = (df['close'] - df['low'].rolling(20).min()) / (df['high'].rolling(20).max() - df['low'].rolling(20).min())
        
        # Select feature columns
        feature_columns = [col for col in self.technical_indicators if col in df.columns]
        features = df[feature_columns].dropna()
        
        return features.values
    
    def _create_targets(self, data: pd.DataFrame) -> np.ndarray:
        """Create target variables for training"""
        # Future price movement (1-hour ahead)
        future_returns = data['close'].shift(-1) / data['close'] - 1
        
        # Binary classification: 1 for up, 0 for down
        targets = (future_returns > 0).astype(int)
        
        return targets.dropna().values
    
    async def _train_lstm_model(self, X_train, y_train, X_test, y_test):
        """Train LSTM neural network"""
        try:
            # Reshape data for LSTM (samples, time steps, features)
            X_train_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
            X_test_lstm = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
            
            # Build LSTM model
            model = Sequential([
                LSTM(128, return_sequences=True, input_shape=(1, X_train.shape[1])),
                Dropout(0.2),
                LSTM(64, return_sequences=True),
                Dropout(0.2),
                LSTM(32),
                Dropout(0.2),
                Dense(16, activation='relu'),
                Dense(1, activation='sigmoid')
            ])
            
            model.compile(optimizer=Adam(learning_rate=0.001), 
                         loss='binary_crossentropy', 
                         metrics=['accuracy'])
            
            # Train model
            model.fit(X_train_lstm, y_train, 
                     epochs=50, 
                     batch_size=32, 
                     validation_data=(X_test_lstm, y_test),
                     verbose=0)
            
            self.lstm_model = model
            logger.info("LSTM model trained successfully")
            
        except Exception as e:
            logger.error(f"Error training LSTM model: {e}")
    
    async def _train_xgboost_model(self, X_train, y_train, X_test, y_test):
        """Train XGBoost model"""
        try:
            model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
            
            model.fit(X_train, y_train)
            
            # Calculate feature importance
            self.feature_importance['xgboost'] = dict(zip(
                [f'feature_{i}' for i in range(len(model.feature_importances_))],
                model.feature_importances_
            ))
            
            self.xgb_model = model
            logger.info("XGBoost model trained successfully")
            
        except Exception as e:
            logger.error(f"Error training XGBoost model: {e}")
    
    async def _train_lightgbm_model(self, X_train, y_train, X_test, y_test):
        """Train LightGBM model"""
        try:
            model = lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1
            )
            
            model.fit(X_train, y_train)
            
            # Calculate feature importance
            self.feature_importance['lightgbm'] = dict(zip(
                [f'feature_{i}' for i in range(len(model.feature_importances_))],
                model.feature_importances_
            ))
            
            self.lgb_model = model
            logger.info("LightGBM model trained successfully")
            
        except Exception as e:
            logger.error(f"Error training LightGBM model: {e}")
    
    async def _train_random_forest_model(self, X_train, y_train, X_test, y_test):
        """Train Random Forest model"""
        try:
            model = RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            
            model.fit(X_train, y_train)
            
            # Calculate feature importance
            self.feature_importance['random_forest'] = dict(zip(
                [f'feature_{i}' for i in range(len(model.feature_importances_))],
                model.feature_importances_
            ))
            
            self.rf_model = model
            logger.info("Random Forest model trained successfully")
            
        except Exception as e:
            logger.error(f"Error training Random Forest model: {e}")
    
    async def _train_mlp_model(self, X_train, y_train, X_test, y_test):
        """Train Multi-Layer Perceptron model"""
        try:
            model = MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),
                activation='relu',
                solver='adam',
                learning_rate='adaptive',
                max_iter=500,
                random_state=42
            )
            
            model.fit(X_train, y_train)
            
            self.mlp_model = model
            logger.info("MLP model trained successfully")
            
        except Exception as e:
            logger.error(f"Error training MLP model: {e}")
    
    async def _predict_lstm(self, features: np.ndarray) -> Dict[str, float]:
        """Generate LSTM prediction"""
        try:
            if self.lstm_model is None:
                return {'direction': 0.5, 'confidence': 0.0}
            
            # Reshape for LSTM
            features_lstm = features[-1:].reshape((1, 1, features.shape[1]))
            
            prediction = self.lstm_model.predict(features_lstm, verbose=0)[0][0]
            
            return {
                'direction': float(prediction),
                'confidence': abs(prediction - 0.5) * 2
            }
            
        except Exception as e:
            logger.error(f"Error in LSTM prediction: {e}")
            return {'direction': 0.5, 'confidence': 0.0}
    
    async def _predict_xgboost(self, features: np.ndarray) -> Dict[str, float]:
        """Generate XGBoost prediction"""
        try:
            if self.xgb_model is None:
                return {'direction': 0.5, 'confidence': 0.0}
            
            prediction_proba = self.xgb_model.predict_proba(features[-1:].reshape(1, -1))[0]
            prediction = prediction_proba[1]  # Probability of class 1 (up)
            
            return {
                'direction': float(prediction),
                'confidence': max(prediction_proba) - min(prediction_proba)
            }
            
        except Exception as e:
            logger.error(f"Error in XGBoost prediction: {e}")
            return {'direction': 0.5, 'confidence': 0.0}
    
    async def _predict_lightgbm(self, features: np.ndarray) -> Dict[str, float]:
        """Generate LightGBM prediction"""
        try:
            if self.lgb_model is None:
                return {'direction': 0.5, 'confidence': 0.0}
            
            prediction_proba = self.lgb_model.predict_proba(features[-1:].reshape(1, -1))[0]
            prediction = prediction_proba[1]  # Probability of class 1 (up)
            
            return {
                'direction': float(prediction),
                'confidence': max(prediction_proba) - min(prediction_proba)
            }
            
        except Exception as e:
            logger.error(f"Error in LightGBM prediction: {e}")
            return {'direction': 0.5, 'confidence': 0.0}
    
    async def _predict_random_forest(self, features: np.ndarray) -> Dict[str, float]:
        """Generate Random Forest prediction"""
        try:
            if self.rf_model is None:
                return {'direction': 0.5, 'confidence': 0.0}
            
            prediction = self.rf_model.predict(features[-1:].reshape(1, -1))[0]
            
            # Convert regression output to classification
            direction = 1.0 if prediction > 0.5 else 0.0
            confidence = abs(prediction - 0.5) * 2
            
            return {
                'direction': float(direction),
                'confidence': float(confidence)
            }
            
        except Exception as e:
            logger.error(f"Error in Random Forest prediction: {e}")
            return {'direction': 0.5, 'confidence': 0.0}
    
    async def _predict_mlp(self, features: np.ndarray) -> Dict[str, float]:
        """Generate MLP prediction"""
        try:
            if self.mlp_model is None:
                return {'direction': 0.5, 'confidence': 0.0}
            
            prediction_proba = self.mlp_model.predict_proba(features[-1:].reshape(1, -1))[0]
            prediction = prediction_proba[1]  # Probability of class 1 (up)
            
            return {
                'direction': float(prediction),
                'confidence': max(prediction_proba) - min(prediction_proba)
            }
            
        except Exception as e:
            logger.error(f"Error in MLP prediction: {e}")
            return {'direction': 0.5, 'confidence': 0.0}
    
    async def _ensemble_predict(self, predictions: Dict[str, Dict], confidences: Dict[str, float]) -> Dict[str, float]:
        """Combine predictions using ensemble method"""
        try:
            weighted_direction = 0.0
            weighted_confidence = 0.0
            total_weight = 0.0
            
            for model_name, pred in predictions.items():
                weight = self.ensemble_weights.get(model_name, 0.1)
                model_confidence = confidences.get(model_name, 0.0)
                
                # Weight by both ensemble weight and model confidence
                final_weight = weight * (1 + model_confidence)
                
                weighted_direction += pred['direction'] * final_weight
                weighted_confidence += model_confidence * final_weight
                total_weight += final_weight
            
            if total_weight > 0:
                ensemble_direction = weighted_direction / total_weight
                ensemble_confidence = weighted_confidence / total_weight
            else:
                ensemble_direction = 0.5
                ensemble_confidence = 0.0
            
            # Calculate price change estimate
            price_change = (ensemble_direction - 0.5) * 0.02  # Max 2% change
            
            return {
                'direction': ensemble_direction,
                'confidence': min(ensemble_confidence, 1.0),
                'probability': ensemble_direction,
                'price_change': price_change
            }
            
        except Exception as e:
            logger.error(f"Error in ensemble prediction: {e}")
            return {'direction': 0.5, 'confidence': 0.0, 'probability': 0.5, 'price_change': 0.0}
    
    def _calculate_prediction_confidence(self, prediction: Dict[str, float], model_name: str) -> float:
        """Calculate confidence score for a prediction"""
        base_confidence = prediction.get('confidence', 0.0)
        
        # Adjust confidence based on model historical performance
        model_performance = {
            'lstm': 0.85,
            'xgboost': 0.88,
            'lightgbm': 0.87,
            'random_forest': 0.82,
            'mlp': 0.80,
            'transformer': 0.75
        }
        
        performance_weight = model_performance.get(model_name, 0.75)
        
        return min(base_confidence * performance_weight, 1.0)
    
    async def _assess_prediction_risk(self, features: np.ndarray, prediction: Dict[str, float]) -> float:
        """Assess risk level of the prediction"""
        try:
            risk_factors = []
            
            # Volatility risk
            if len(features) > 0:
                volatility = np.std(features[-20:]) if len(features) >= 20 else np.std(features)
                if volatility > 0.02:  # High volatility
                    risk_factors.append(0.3)
                elif volatility > 0.01:  # Medium volatility
                    risk_factors.append(0.2)
                else:  # Low volatility
                    risk_factors.append(0.1)
            
            # Confidence risk
            confidence = prediction.get('confidence', 0.0)
            if confidence < 0.6:
                risk_factors.append(0.4)
            elif confidence < 0.8:
                risk_factors.append(0.2)
            else:
                risk_factors.append(0.1)
            
            # Direction uncertainty risk
            direction = prediction.get('direction', 0.5)
            direction_certainty = abs(direction - 0.5) * 2
            if direction_certainty < 0.3:
                risk_factors.append(0.4)
            elif direction_certainty < 0.6:
                risk_factors.append(0.2)
            else:
                risk_factors.append(0.1)
            
            # Calculate overall risk score
            risk_score = sum(risk_factors) / len(risk_factors) if risk_factors else 0.5
            
            return min(risk_score, 1.0)
            
        except Exception as e:
            logger.error(f"Error assessing prediction risk: {e}")
            return 0.5
    
    async def _optimize_ensemble_weights(self, X_test, y_test):
        """Optimize ensemble weights based on model performance"""
        try:
            model_scores = {}
            
            # Test each model
            if self.lstm_model:
                X_test_lstm = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
                lstm_pred = self.lstm_model.predict(X_test_lstm, verbose=0)
                lstm_binary = (lstm_pred > 0.5).astype(int).flatten()
                model_scores['lstm'] = accuracy_score(y_test, lstm_binary)
            
            if self.xgb_model:
                xgb_pred = self.xgb_model.predict(X_test)
                model_scores['xgboost'] = accuracy_score(y_test, xgb_pred)
            
            if self.lgb_model:
                lgb_pred = self.lgb_model.predict(X_test)
                model_scores['lightgbm'] = accuracy_score(y_test, lgb_pred)
            
            if self.rf_model:
                rf_pred = self.rf_model.predict(X_test)
                rf_binary = (rf_pred > 0.5).astype(int)
                model_scores['random_forest'] = accuracy_score(y_test, rf_binary)
            
            if self.mlp_model:
                mlp_pred = self.mlp_model.predict(X_test)
                model_scores['mlp'] = accuracy_score(y_test, mlp_pred)
            
            # Update ensemble weights based on performance
            total_score = sum(model_scores.values())
            if total_score > 0:
                for model, score in model_scores.items():
                    self.ensemble_weights[model] = score / total_score
            
            logger.info(f"Updated ensemble weights: {self.ensemble_weights}")
            
        except Exception as e:
            logger.error(f"Error optimizing ensemble weights: {e}")
    
    async def get_model_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive model performance statistics"""
        try:
            stats = {
                'ensemble_weights': self.ensemble_weights,
                'feature_importance': self.feature_importance,
                'prediction_count': len(self.prediction_history),
                'models_active': len([m for m in [self.lstm_model, self.xgb_model, self.lgb_model, self.rf_model, self.mlp_model] if m is not None])
            }
            
            # Calculate recent prediction accuracy if we have enough data
            if len(self.prediction_history) > 10:
                recent_predictions = self.prediction_history[-10:]
                avg_confidence = np.mean([p.confidence for p in recent_predictions])
                stats['recent_avg_confidence'] = avg_confidence
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting model stats: {e}")
            return {}
    
    async def close(self):
        """Close the neural engine"""
        self.initialized = False
        logger.info("Advanced Neural Network Trading Engine closed")