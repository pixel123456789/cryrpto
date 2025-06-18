"""
Enhanced AI Features for Advanced Trading Analysis
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
import numpy as np
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

@dataclass
class AITradingInsight:
    """AI-generated trading insight"""
    insight_type: str
    confidence: float
    message: str
    importance: str
    timeframe: str
    generated_at: datetime

class AIEnhancedAnalyzer:
    """Enhanced AI analyzer for advanced trading insights"""
    
    def __init__(self):
        self.initialized = False
        self.market_memory = []
        self.pattern_database = {}
        self.sentiment_history = []
        
    async def initialize(self):
        """Initialize AI enhanced analyzer"""
        self.initialized = True
        logger.info("AI Enhanced Analyzer initialized with advanced features")
    
    async def generate_market_intelligence(self, symbols_data: List[Dict]) -> Dict[str, Any]:
        """Generate comprehensive market intelligence"""
        try:
            intelligence = {
                'market_sentiment': await self._analyze_global_sentiment(symbols_data),
                'correlation_analysis': await self._analyze_symbol_correlations(symbols_data),
                'volatility_forecast': await self._forecast_volatility(symbols_data),
                'trend_momentum': await self._analyze_trend_momentum(symbols_data),
                'risk_assessment': await self._assess_market_risk(symbols_data),
                'opportunities': await self._identify_opportunities(symbols_data)
            }
            
            return intelligence
            
        except Exception as e:
            logger.error(f"Error generating market intelligence: {e}")
            return {'error': str(e)}
    
    async def _analyze_global_sentiment(self, symbols_data: List[Dict]) -> Dict[str, Any]:
        """Analyze global market sentiment"""
        try:
            bullish_signals = 0
            bearish_signals = 0
            total_volume = 0
            
            for symbol_data in symbols_data:
                signal = symbol_data.get('signal', {})
                if signal.get('action') == 'LONG':
                    bullish_signals += signal.get('confidence', 0)
                elif signal.get('action') == 'SHORT':
                    bearish_signals += signal.get('confidence', 0)
                
                total_volume += symbol_data.get('volume', 0)
            
            # Calculate sentiment score
            if bullish_signals + bearish_signals > 0:
                sentiment_score = (bullish_signals - bearish_signals) / (bullish_signals + bearish_signals)
            else:
                sentiment_score = 0
            
            if sentiment_score > 0.3:
                sentiment = "BULLISH"
            elif sentiment_score < -0.3:
                sentiment = "BEARISH"
            else:
                sentiment = "NEUTRAL"
            
            return {
                'sentiment': sentiment,
                'sentiment_score': sentiment_score,
                'bullish_strength': bullish_signals,
                'bearish_strength': bearish_signals,
                'confidence': min(abs(sentiment_score) * 100, 95)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing global sentiment: {e}")
            return {'sentiment': 'NEUTRAL', 'error': str(e)}
    
    async def _analyze_symbol_correlations(self, symbols_data: List[Dict]) -> Dict[str, Any]:
        """Analyze correlations between symbols"""
        try:
            correlations = []
            
            # Simple correlation analysis
            for i, symbol1 in enumerate(symbols_data):
                for j, symbol2 in enumerate(symbols_data[i+1:], i+1):
                    if j >= len(symbols_data):
                        break
                    
                    # Calculate basic correlation based on signal direction
                    signal1 = symbol1.get('signal', {}).get('action', 'NEUTRAL')
                    signal2 = symbol2.get('signal', {}).get('action', 'NEUTRAL')
                    
                    if signal1 == signal2 and signal1 != 'NEUTRAL':
                        correlation_strength = 0.7  # High correlation
                        correlations.append({
                            'pair': f"{symbol1.get('symbol', 'UNKNOWN')} - {symbol2.get('symbol', 'UNKNOWN')}",
                            'correlation': correlation_strength,
                            'direction': signal1
                        })
            
            return {
                'strong_correlations': correlations[:5],  # Top 5
                'correlation_count': len(correlations)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing correlations: {e}")
            return {'strong_correlations': [], 'error': str(e)}
    
    async def _forecast_volatility(self, symbols_data: List[Dict]) -> Dict[str, Any]:
        """Forecast market volatility"""
        try:
            volatility_scores = []
            
            for symbol_data in symbols_data:
                # Simple volatility estimation based on confidence variance
                confidence = symbol_data.get('signal', {}).get('confidence', 50)
                volatility = abs(confidence - 50) / 50  # Normalize to 0-1
                volatility_scores.append(volatility)
            
            avg_volatility = np.mean(volatility_scores) if volatility_scores else 0
            
            if avg_volatility > 0.6:
                volatility_level = "HIGH"
                forecast = "Expect significant price movements"
            elif avg_volatility > 0.3:
                volatility_level = "MEDIUM"
                forecast = "Moderate price volatility expected"
            else:
                volatility_level = "LOW"
                forecast = "Low volatility, consolidation likely"
            
            return {
                'volatility_level': volatility_level,
                'volatility_score': avg_volatility,
                'forecast': forecast,
                'confidence': min(avg_volatility * 100, 90)
            }
            
        except Exception as e:
            logger.error(f"Error forecasting volatility: {e}")
            return {'volatility_level': 'MEDIUM', 'error': str(e)}
    
    async def _analyze_trend_momentum(self, symbols_data: List[Dict]) -> Dict[str, Any]:
        """Analyze overall trend momentum"""
        try:
            momentum_scores = {'LONG': 0, 'SHORT': 0, 'NEUTRAL': 0}
            total_confidence = 0
            
            for symbol_data in symbols_data:
                signal = symbol_data.get('signal', {})
                action = signal.get('action', 'NEUTRAL')
                confidence = signal.get('confidence', 0)
                
                momentum_scores[action] += confidence
                total_confidence += confidence
            
            # Determine dominant momentum
            max_momentum = max(momentum_scores.values())
            dominant_direction = [k for k, v in momentum_scores.items() if v == max_momentum][0]
            
            momentum_strength = max_momentum / total_confidence if total_confidence > 0 else 0
            
            return {
                'dominant_direction': dominant_direction,
                'momentum_strength': momentum_strength,
                'long_momentum': momentum_scores['LONG'],
                'short_momentum': momentum_scores['SHORT'],
                'momentum_confidence': min(momentum_strength * 100, 95)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing momentum: {e}")
            return {'dominant_direction': 'NEUTRAL', 'error': str(e)}
    
    async def _assess_market_risk(self, symbols_data: List[Dict]) -> Dict[str, Any]:
        """Assess overall market risk"""
        try:
            risk_factors = []
            risk_score = 0
            
            # Analyze signal distribution
            signal_confidence_avg = np.mean([
                s.get('signal', {}).get('confidence', 0) 
                for s in symbols_data
            ]) if symbols_data else 0
            
            if signal_confidence_avg < 60:
                risk_score += 30
                risk_factors.append("Low average signal confidence")
            
            # Check for conflicting signals
            long_signals = sum(1 for s in symbols_data if s.get('signal', {}).get('action') == 'LONG')
            short_signals = sum(1 for s in symbols_data if s.get('signal', {}).get('action') == 'SHORT')
            
            if abs(long_signals - short_signals) < len(symbols_data) * 0.2:
                risk_score += 25
                risk_factors.append("Conflicting market signals")
            
            # Determine risk level
            if risk_score > 50:
                risk_level = "HIGH"
            elif risk_score > 25:
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"
            
            return {
                'risk_level': risk_level,
                'risk_score': risk_score,
                'risk_factors': risk_factors,
                'recommendation': self._get_risk_recommendation(risk_level)
            }
            
        except Exception as e:
            logger.error(f"Error assessing risk: {e}")
            return {'risk_level': 'MEDIUM', 'error': str(e)}
    
    async def _identify_opportunities(self, symbols_data: List[Dict]) -> List[Dict[str, Any]]:
        """Identify trading opportunities"""
        try:
            opportunities = []
            
            # Sort by confidence
            sorted_symbols = sorted(
                symbols_data, 
                key=lambda x: x.get('signal', {}).get('confidence', 0), 
                reverse=True
            )
            
            for symbol_data in sorted_symbols[:10]:  # Top 10 opportunities
                signal = symbol_data.get('signal', {})
                confidence = signal.get('confidence', 0)
                
                if confidence > 75:  # High confidence threshold
                    opportunity = {
                        'symbol': symbol_data.get('symbol', 'UNKNOWN'),
                        'action': signal.get('action', 'NEUTRAL'),
                        'confidence': confidence,
                        'opportunity_type': self._classify_opportunity(signal),
                        'priority': 'HIGH' if confidence > 90 else 'MEDIUM'
                    }
                    opportunities.append(opportunity)
            
            return opportunities
            
        except Exception as e:
            logger.error(f"Error identifying opportunities: {e}")
            return []
    
    def _get_risk_recommendation(self, risk_level: str) -> str:
        """Get risk management recommendation"""
        recommendations = {
            'HIGH': "Reduce position sizes, increase stop losses, avoid new positions",
            'MEDIUM': "Use standard risk management, monitor closely",
            'LOW': "Normal position sizing, favorable conditions for trading"
        }
        return recommendations.get(risk_level, "Monitor market conditions")
    
    def _classify_opportunity(self, signal: Dict) -> str:
        """Classify the type of opportunity"""
        confidence = signal.get('confidence', 0)
        action = signal.get('action', 'NEUTRAL')
        
        if confidence > 90:
            return "HIGH_PROBABILITY"
        elif confidence > 80:
            return "MOMENTUM_PLAY"
        elif confidence > 70:
            return "BREAKOUT_SETUP"
        else:
            return "SCALPING_OPPORTUNITY"
    
    async def generate_ai_insights(self, market_data: Dict) -> List[AITradingInsight]:
        """Generate AI-powered trading insights"""
        insights = []
        
        try:
            # Market sentiment insight
            sentiment = market_data.get('market_sentiment', {})
            if sentiment.get('confidence', 0) > 70:
                insights.append(AITradingInsight(
                    insight_type="MARKET_SENTIMENT",
                    confidence=sentiment.get('confidence', 0),
                    message=f"Market showing {sentiment.get('sentiment', 'NEUTRAL')} sentiment with {sentiment.get('confidence', 0):.0f}% confidence",
                    importance="HIGH",
                    timeframe="4H-1D",
                    generated_at=datetime.now()
                ))
            
            # Volatility insight
            volatility = market_data.get('volatility_forecast', {})
            if volatility.get('volatility_level') in ['HIGH', 'LOW']:
                insights.append(AITradingInsight(
                    insight_type="VOLATILITY",
                    confidence=volatility.get('confidence', 0),
                    message=f"{volatility.get('volatility_level')} volatility expected - {volatility.get('forecast', '')}",
                    importance="MEDIUM",
                    timeframe="1H-4H",
                    generated_at=datetime.now()
                ))
            
            # Opportunity insight
            opportunities = market_data.get('opportunities', [])
            if opportunities:
                high_priority_ops = [op for op in opportunities if op.get('priority') == 'HIGH']
                if high_priority_ops:
                    insights.append(AITradingInsight(
                        insight_type="OPPORTUNITY",
                        confidence=high_priority_ops[0].get('confidence', 0),
                        message=f"High-priority opportunity: {high_priority_ops[0].get('symbol')} {high_priority_ops[0].get('action')}",
                        importance="HIGH",
                        timeframe="15M-1H",
                        generated_at=datetime.now()
                    ))
            
            return insights[:5]  # Limit to top 5 insights
            
        except Exception as e:
            logger.error(f"Error generating AI insights: {e}")
            return []
    
    async def close(self):
        """Close the AI enhanced analyzer"""
        self.initialized = False
        logger.info("AI Enhanced Analyzer closed")