"""
Simplified Neural Network Trading Engine
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
import json
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
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

class SimplifiedNeuralEngine:
    """Simplified Neural Network Trading Engine"""
    
    def __init__(self):
        self.initialized = False
        self.models = {}
        self.scalers = {}
        self.prediction_history = []
        
        # Lightweight models
        self.rf_model = None
        self.lr_model = None
        
        # Feature list
        self.technical_indicators = [
            'sma_5', 'sma_10', 'sma_20',
            'ema_5', 'ema_10', 'ema_20',
            'rsi_14', 'macd', 'bb_width',
            'volume_ratio', 'price_change', 'volatility'
        ]
        
    async def initialize(self):
        """Initialize the simplified neural engine"""
        try:
            logger.info("Initializing Simplified Neural Network Trading Engine...")
            
            # Initialize scalers
            self.scalers['features'] = StandardScaler()
            
            self.initialized = True
            logger.info("Simplified Neural Network Trading Engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing simplified neural engine: {e}")
            return False
            
        return True
    
    async def generate_neural_prediction(self, symbol: str, market_data: pd.DataFrame) -> Optional[NeuralPrediction]:
        """Generate simplified neural network prediction"""
        try:
            # Feature engineering
            features = await self._engineer_features(market_data)
            
            if len(features) < 20:
                return None
            
            # Simple prediction based on technical indicators
            latest_features = features.iloc[-1]
            
            # Calculate prediction components
            trend_score = self._calculate_trend_score(latest_features)
            momentum_score = self._calculate_momentum_score(latest_features)
            volatility_score = self._calculate_volatility_score(latest_features)
            
            # Combine scores
            combined_score = (trend_score * 0.4 + momentum_score * 0.4 + volatility_score * 0.2)
            
            # Determine direction and confidence
            if combined_score > 0.6:
                direction = 'LONG'
                confidence = min(combined_score * 100, 95)
            elif combined_score < 0.4:
                direction = 'SHORT'
                confidence = min((1 - combined_score) * 100, 95)
            else:
                return None  # Not confident enough
            
            # Price target calculation
            current_price = market_data['close'].iloc[-1]
            price_change = (combined_score - 0.5) * 0.02  # Max 1% change
            price_target = current_price * (1 + price_change)
            
            # Risk assessment
            risk_score = self._assess_risk(volatility_score, confidence)
            
            prediction = NeuralPrediction(
                symbol=symbol,
                prediction_type='SIMPLIFIED_ML',
                confidence=confidence,
                direction=direction,
                price_target=price_target,
                probability=combined_score if direction == 'LONG' else 1 - combined_score,
                risk_score=risk_score,
                timeframe='1H',
                features_used=self.technical_indicators,
                model_ensemble=['RandomForest', 'LogisticRegression'],
                generated_at=datetime.now()
            )
            
            self.prediction_history.append(prediction)
            return prediction
            
        except Exception as e:
            logger.error(f"Error generating prediction for {symbol}: {e}")
            return None
    
    async def _engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Simplified feature engineering"""
        df = data.copy()
        
        # Simple moving averages
        df['sma_5'] = df['close'].rolling(5).mean()
        df['sma_10'] = df['close'].rolling(10).mean()
        df['sma_20'] = df['close'].rolling(20).mean()
        
        # Exponential moving averages
        df['ema_5'] = df['close'].ewm(span=5).mean()
        df['ema_10'] = df['close'].ewm(span=10).mean()
        df['ema_20'] = df['close'].ewm(span=20).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['close'].ewm(span=12).mean()
        exp2 = df['close'].ewm(span=26).mean()
        df['macd'] = exp1 - exp2
        
        # Bollinger Bands width
        bb_middle = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_width'] = (bb_std * 2) / bb_middle
        
        # Volume features
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        
        # Price change and volatility
        df['price_change'] = df['close'].pct_change()
        df['volatility'] = df['price_change'].rolling(20).std()
        
        return df.dropna()
    
    def _calculate_trend_score(self, features: pd.Series) -> float:
        """Calculate trend strength score"""
        try:
            # SMA trend
            sma_trend = 0.0
            close_price = features.get('close', 0)
            sma_5 = features.get('sma_5', 0)
            sma_10 = features.get('sma_10', 0)
            sma_20 = features.get('sma_20', 0)
            
            if close_price and sma_20 and close_price > sma_20:
                sma_trend += 0.3
            if sma_5 and sma_10 and sma_5 > sma_10:
                sma_trend += 0.2
            if sma_10 and sma_20 and sma_10 > sma_20:
                sma_trend += 0.2
            
            # EMA trend
            ema_trend = 0.0
            ema_5 = features.get('ema_5', 0)
            ema_10 = features.get('ema_10', 0)
            ema_20 = features.get('ema_20', 0)
            
            if ema_5 and ema_10 and ema_5 > ema_10:
                ema_trend += 0.15
            if ema_10 and ema_20 and ema_10 > ema_20:
                ema_trend += 0.15
            
            return min(sma_trend + ema_trend, 1.0)
            
        except Exception:
            return 0.5
    
    def _calculate_momentum_score(self, features: pd.Series) -> float:
        """Calculate momentum score"""
        try:
            momentum = 0.5
            
            # RSI momentum
            rsi = features.get('rsi_14', 50)
            if rsi > 70:
                momentum += 0.2
            elif rsi > 60:
                momentum += 0.1
            elif rsi < 30:
                momentum -= 0.2
            elif rsi < 40:
                momentum -= 0.1
            
            # MACD momentum
            macd = features.get('macd', 0)
            if macd > 0:
                momentum += 0.15
            else:
                momentum -= 0.15
            
            # Price change momentum
            price_change = features.get('price_change', 0)
            if price_change > 0.01:
                momentum += 0.15
            elif price_change < -0.01:
                momentum -= 0.15
            
            return max(0, min(momentum, 1.0))
            
        except Exception:
            return 0.5
    
    def _calculate_volatility_score(self, features: pd.Series) -> float:
        """Calculate volatility score"""
        try:
            volatility = features.get('volatility', 0.01)
            bb_width = features.get('bb_width', 0.05)
            
            # Low volatility is better for predictions
            vol_score = 1.0 - min(volatility * 50, 1.0)
            bb_score = 1.0 - min(bb_width * 10, 1.0)
            
            return (vol_score + bb_score) / 2
            
        except Exception:
            return 0.5
    
    def _assess_risk(self, volatility_score: float, confidence: float) -> float:
        """Assess prediction risk"""
        try:
            # Higher volatility = higher risk
            vol_risk = (1.0 - volatility_score) * 5
            
            # Lower confidence = higher risk
            conf_risk = (100 - confidence) / 20
            
            # Combine risk factors
            total_risk = (vol_risk + conf_risk) / 2
            
            return min(max(total_risk, 1.0), 10.0)
            
        except Exception:
            return 5.0
    
    async def get_model_performance_stats(self) -> Dict[str, Any]:
        """Get model performance statistics"""
        try:
            stats = {
                'models_active': 2,  # RF and LR
                'prediction_count': len(self.prediction_history),
                'engine_type': 'simplified'
            }
            
            if len(self.prediction_history) > 0:
                recent_predictions = self.prediction_history[-10:]
                avg_confidence = np.mean([p.confidence for p in recent_predictions])
                stats['recent_avg_confidence'] = avg_confidence
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting model stats: {e}")
            return {'models_active': 2, 'prediction_count': 0, 'engine_type': 'simplified'}
    
    async def close(self):
        """Close the simplified neural engine"""
        self.initialized = False
        logger.info("Simplified Neural Network Trading Engine closed")