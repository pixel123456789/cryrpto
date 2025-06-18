"""
Real-time Market Sentiment Analysis Engine with Social Media & News Integration
"""

import asyncio
import logging
import re
import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import aiohttp
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from textblob import TextBlob
import nltk
from collections import defaultdict

logger = logging.getLogger(__name__)

class SentimentEngine:
    """Real-time market sentiment analysis with multiple data sources"""
    
    def __init__(self):
        self.initialized = False
        self.sentiment_models = {}
        self.vectorizers = {}
        self.sentiment_history = defaultdict(list)
        
        # Cryptocurrency keywords and terms
        self.crypto_keywords = {
            'bullish': ['moon', 'pump', 'buy', 'hodl', 'bull', 'bullish', 'rocket', 'gains', 'long', 'calls'],
            'bearish': ['dump', 'sell', 'bear', 'bearish', 'short', 'crash', 'drop', 'puts', 'rekt', 'liquidated'],
            'neutral': ['hold', 'wait', 'sideways', 'consolidation', 'range', 'neutral']
        }
        
        # News sentiment keywords
        self.news_keywords = {
            'positive': ['adoption', 'partnership', 'breakthrough', 'innovation', 'surge', 'milestone', 'approval'],
            'negative': ['hack', 'regulation', 'ban', 'crash', 'scam', 'investigation', 'decline'],
            'neutral': ['announcement', 'report', 'update', 'analysis', 'review']
        }
        
    async def initialize(self):
        """Initialize sentiment analysis models"""
        try:
            # Initialize TF-IDF vectorizers
            self.vectorizers['crypto'] = TfidfVectorizer(max_features=5000, stop_words='english')
            self.vectorizers['news'] = TfidfVectorizer(max_features=3000, stop_words='english')
            
            # Initialize sentiment classifiers
            self.sentiment_models['crypto'] = LogisticRegression(random_state=42)
            self.sentiment_models['news'] = MultinomialNB()
            
            # Train with sample data (in production, use real labeled data)
            await self._train_sentiment_models()
            
            self.initialized = True
            logger.info("Market Sentiment Analysis Engine initialized")
            
        except Exception as e:
            logger.error(f"Error initializing sentiment engine: {e}")
            return False
            
        return True
    
    async def analyze_market_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Comprehensive market sentiment analysis for a symbol"""
        try:
            # Get social media sentiment
            social_sentiment = await self._analyze_social_sentiment(symbol)
            
            # Get news sentiment
            news_sentiment = await self._analyze_news_sentiment(symbol)
            
            # Get technical sentiment from price action
            technical_sentiment = await self._analyze_technical_sentiment(symbol)
            
            # Get fear & greed indicators
            fear_greed = await self._analyze_fear_greed_index()
            
            # Combine all sentiment sources
            combined_sentiment = await self._combine_sentiments(
                social_sentiment, news_sentiment, technical_sentiment, fear_greed
            )
            
            # Store historical data
            self.sentiment_history[symbol].append({
                'timestamp': datetime.now(),
                'sentiment': combined_sentiment,
                'social': social_sentiment,
                'news': news_sentiment,
                'technical': technical_sentiment
            })
            
            # Keep only last 100 records
            if len(self.sentiment_history[symbol]) > 100:
                self.sentiment_history[symbol] = self.sentiment_history[symbol][-100:]
            
            return combined_sentiment
            
        except Exception as e:
            logger.error(f"Error analyzing market sentiment for {symbol}: {e}")
            return {'sentiment': 'NEUTRAL', 'confidence': 0.0, 'score': 0.0}
    
    async def _analyze_social_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Analyze social media sentiment (simulated due to API limitations)"""
        try:
            # In a real implementation, this would connect to Twitter API, Reddit API, etc.
            # For now, we'll simulate based on market conditions
            
            # Simulate social media posts analysis
            simulated_posts = await self._simulate_social_posts(symbol)
            
            # Analyze sentiment of simulated posts
            sentiments = []
            for post in simulated_posts:
                sentiment_score = await self._analyze_text_sentiment(post, 'crypto')
                sentiments.append(sentiment_score)
            
            if not sentiments:
                return {'sentiment': 'NEUTRAL', 'confidence': 0.0, 'score': 0.0, 'volume': 0}
            
            # Calculate aggregate sentiment
            avg_sentiment = np.mean(sentiments)
            confidence = min(abs(avg_sentiment) * 2, 1.0)
            
            if avg_sentiment > 0.2:
                sentiment = 'BULLISH'
            elif avg_sentiment < -0.2:
                sentiment = 'BEARISH'
            else:
                sentiment = 'NEUTRAL'
            
            return {
                'sentiment': sentiment,
                'confidence': confidence,
                'score': avg_sentiment,
                'volume': len(simulated_posts)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing social sentiment: {e}")
            return {'sentiment': 'NEUTRAL', 'confidence': 0.0, 'score': 0.0, 'volume': 0}
    
    async def _analyze_news_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Analyze news sentiment for the symbol"""
        try:
            # Simulate news headlines analysis
            news_headlines = await self._simulate_news_headlines(symbol)
            
            # Analyze sentiment of news
            sentiments = []
            for headline in news_headlines:
                sentiment_score = await self._analyze_text_sentiment(headline, 'news')
                sentiments.append(sentiment_score)
            
            if not sentiments:
                return {'sentiment': 'NEUTRAL', 'confidence': 0.0, 'score': 0.0, 'articles': 0}
            
            # Calculate aggregate sentiment
            avg_sentiment = np.mean(sentiments)
            confidence = min(abs(avg_sentiment) * 1.5, 1.0)
            
            if avg_sentiment > 0.15:
                sentiment = 'POSITIVE'
            elif avg_sentiment < -0.15:
                sentiment = 'NEGATIVE'
            else:
                sentiment = 'NEUTRAL'
            
            return {
                'sentiment': sentiment,
                'confidence': confidence,
                'score': avg_sentiment,
                'articles': len(news_headlines)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing news sentiment: {e}")
            return {'sentiment': 'NEUTRAL', 'confidence': 0.0, 'score': 0.0, 'articles': 0}
    
    async def _analyze_technical_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Analyze technical sentiment from price action"""
        try:
            # This would normally get real price data
            # For now, simulate based on symbol patterns
            price_change_1h = np.random.uniform(-5, 5)
            price_change_24h = np.random.uniform(-15, 15)
            volume_change = np.random.uniform(-50, 100)
            
            # Calculate technical sentiment score
            technical_score = 0.0
            
            # Price momentum factor
            if price_change_1h > 2:
                technical_score += 0.3
            elif price_change_1h < -2:
                technical_score -= 0.3
            
            if price_change_24h > 5:
                technical_score += 0.4
            elif price_change_24h < -5:
                technical_score -= 0.4
            
            # Volume factor
            if volume_change > 30:
                technical_score += 0.2
            elif volume_change < -20:
                technical_score -= 0.1
            
            # Normalize score
            technical_score = max(-1, min(1, technical_score))
            
            if technical_score > 0.2:
                sentiment = 'BULLISH'
            elif technical_score < -0.2:
                sentiment = 'BEARISH'
            else:
                sentiment = 'NEUTRAL'
            
            return {
                'sentiment': sentiment,
                'score': technical_score,
                'confidence': abs(technical_score),
                'price_change_1h': price_change_1h,
                'price_change_24h': price_change_24h,
                'volume_change': volume_change
            }
            
        except Exception as e:
            logger.error(f"Error analyzing technical sentiment: {e}")
            return {'sentiment': 'NEUTRAL', 'score': 0.0, 'confidence': 0.0}
    
    async def _analyze_fear_greed_index(self) -> Dict[str, Any]:
        """Analyze overall market fear & greed"""
        try:
            # Simulate fear & greed index (0-100, where 0 = extreme fear, 100 = extreme greed)
            fear_greed_value = np.random.randint(10, 90)
            
            if fear_greed_value >= 75:
                sentiment = 'EXTREME_GREED'
                market_mood = 'Very bullish market conditions'
            elif fear_greed_value >= 55:
                sentiment = 'GREED'
                market_mood = 'Bullish market sentiment'
            elif fear_greed_value >= 45:
                sentiment = 'NEUTRAL'
                market_mood = 'Balanced market conditions'
            elif fear_greed_value >= 25:
                sentiment = 'FEAR'
                market_mood = 'Bearish market sentiment'
            else:
                sentiment = 'EXTREME_FEAR'
                market_mood = 'Very bearish market conditions'
            
            return {
                'sentiment': sentiment,
                'value': fear_greed_value,
                'mood': market_mood,
                'confidence': abs(fear_greed_value - 50) / 50
            }
            
        except Exception as e:
            logger.error(f"Error analyzing fear & greed index: {e}")
            return {'sentiment': 'NEUTRAL', 'value': 50, 'mood': 'Unknown', 'confidence': 0.0}
    
    async def _combine_sentiments(self, social: Dict, news: Dict, technical: Dict, fear_greed: Dict) -> Dict[str, Any]:
        """Combine all sentiment sources into final sentiment"""
        try:
            # Weight different sentiment sources
            weights = {
                'social': 0.25,
                'news': 0.30,
                'technical': 0.35,
                'fear_greed': 0.10
            }
            
            # Convert sentiments to numerical scores
            sentiment_scores = {
                'social': social.get('score', 0.0),
                'news': news.get('score', 0.0),
                'technical': technical.get('score', 0.0),
                'fear_greed': (fear_greed.get('value', 50) - 50) / 50  # Normalize to -1 to 1
            }
            
            # Calculate weighted average
            weighted_score = sum(
                sentiment_scores[source] * weights[source] 
                for source in weights.keys()
            )
            
            # Calculate confidence based on agreement between sources
            confidences = [
                social.get('confidence', 0.0),
                news.get('confidence', 0.0),
                technical.get('confidence', 0.0),
                fear_greed.get('confidence', 0.0)
            ]
            avg_confidence = np.mean(confidences)
            
            # Determine final sentiment
            if weighted_score > 0.3:
                final_sentiment = 'VERY_BULLISH'
            elif weighted_score > 0.1:
                final_sentiment = 'BULLISH'
            elif weighted_score > -0.1:
                final_sentiment = 'NEUTRAL'
            elif weighted_score > -0.3:
                final_sentiment = 'BEARISH'
            else:
                final_sentiment = 'VERY_BEARISH'
            
            # Generate insights
            insights = await self._generate_sentiment_insights(social, news, technical, fear_greed)
            
            return {
                'sentiment': final_sentiment,
                'score': weighted_score,
                'confidence': avg_confidence,
                'breakdown': {
                    'social': social,
                    'news': news,
                    'technical': technical,
                    'fear_greed': fear_greed
                },
                'insights': insights,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error combining sentiments: {e}")
            return {'sentiment': 'NEUTRAL', 'score': 0.0, 'confidence': 0.0}
    
    async def _analyze_text_sentiment(self, text: str, model_type: str) -> float:
        """Analyze sentiment of a single text using multiple methods"""
        try:
            # Method 1: TextBlob sentiment
            blob = TextBlob(text)
            textblob_score = blob.sentiment.polarity
            
            # Method 2: Keyword-based sentiment
            keyword_score = await self._keyword_sentiment_analysis(text, model_type)
            
            # Method 3: Pattern-based sentiment
            pattern_score = await self._pattern_sentiment_analysis(text)
            
            # Combine scores
            combined_score = (textblob_score * 0.4 + keyword_score * 0.4 + pattern_score * 0.2)
            
            return max(-1, min(1, combined_score))
            
        except Exception as e:
            logger.error(f"Error analyzing text sentiment: {e}")
            return 0.0
    
    async def _keyword_sentiment_analysis(self, text: str, model_type: str) -> float:
        """Analyze sentiment based on keyword matching"""
        try:
            text_lower = text.lower()
            
            if model_type == 'crypto':
                keywords = self.crypto_keywords
            else:
                keywords = self.news_keywords
            
            bullish_count = sum(1 for word in keywords.get('bullish', []) + keywords.get('positive', []) if word in text_lower)
            bearish_count = sum(1 for word in keywords.get('bearish', []) + keywords.get('negative', []) if word in text_lower)
            neutral_count = sum(1 for word in keywords.get('neutral', []) if word in text_lower)
            
            total_words = len(text_lower.split())
            if total_words == 0:
                return 0.0
            
            # Calculate sentiment score
            sentiment_score = (bullish_count - bearish_count) / max(total_words, 1)
            
            return max(-1, min(1, sentiment_score * 10))  # Amplify and normalize
            
        except Exception as e:
            logger.error(f"Error in keyword sentiment analysis: {e}")
            return 0.0
    
    async def _pattern_sentiment_analysis(self, text: str) -> float:
        """Analyze sentiment based on text patterns"""
        try:
            text_lower = text.lower()
            
            # Positive patterns
            positive_patterns = [
                r'\b(to the moon|mooning|pumping|bullish|buy the dip)\b',
                r'\b(hodl|diamond hands|rocket|gains)\b',
                r'\b(breakout|surge|rally|boom)\b'
            ]
            
            # Negative patterns
            negative_patterns = [
                r'\b(dump|crash|rekt|liquidated)\b',
                r'\b(bearish|sell|short|puts)\b',
                r'\b(drop|fall|decline|bear market)\b'
            ]
            
            positive_matches = sum(len(re.findall(pattern, text_lower)) for pattern in positive_patterns)
            negative_matches = sum(len(re.findall(pattern, text_lower)) for pattern in negative_patterns)
            
            if positive_matches + negative_matches == 0:
                return 0.0
            
            sentiment_score = (positive_matches - negative_matches) / (positive_matches + negative_matches)
            
            return sentiment_score
            
        except Exception as e:
            logger.error(f"Error in pattern sentiment analysis: {e}")
            return 0.0
    
    async def _simulate_social_posts(self, symbol: str) -> List[str]:
        """Simulate social media posts for testing"""
        base_symbol = symbol.replace('_USDT', '').replace('USDT', '')
        
        # Simulate different types of posts based on symbol
        posts = [
            f"{base_symbol} is pumping hard! Moon mission activated 🚀",
            f"Just bought more {base_symbol}, diamond hands 💎",
            f"{base_symbol} looking bearish, might dump soon",
            f"HODL {base_symbol} for long term gains",
            f"{base_symbol} breaking resistance, bullish AF",
            f"Selling my {base_symbol} bags, too risky",
            f"{base_symbol} to the moon! Buy the dip!",
            f"Technical analysis shows {base_symbol} bullish pattern",
            f"{base_symbol} partnership announcement coming soon",
            f"Market makers dumping {base_symbol}, be careful"
        ]
        
        # Return random subset
        num_posts = np.random.randint(3, 8)
        return np.random.choice(posts, num_posts, replace=False).tolist()
    
    async def _simulate_news_headlines(self, symbol: str) -> List[str]:
        """Simulate news headlines for testing"""
        base_symbol = symbol.replace('_USDT', '').replace('USDT', '')
        
        headlines = [
            f"{base_symbol} Sees Major Partnership with Tech Giant",
            f"{base_symbol} Price Surges Following Adoption News",
            f"Regulatory Concerns Impact {base_symbol} Trading",
            f"{base_symbol} Network Upgrade Boosts Investor Confidence",
            f"Whale Activity Detected in {base_symbol} Markets",
            f"{base_symbol} Integration Announced by Major Exchange",
            f"Market Analysis: {base_symbol} Shows Strong Fundamentals",
            f"{base_symbol} Faces Selling Pressure Amid Market Uncertainty",
            f"Institutional Interest Growing in {base_symbol}",
            f"{base_symbol} Technical Indicators Signal Bullish Momentum"
        ]
        
        # Return random subset
        num_headlines = np.random.randint(2, 6)
        return np.random.choice(headlines, num_headlines, replace=False).tolist()
    
    async def _generate_sentiment_insights(self, social: Dict, news: Dict, technical: Dict, fear_greed: Dict) -> List[str]:
        """Generate actionable sentiment insights"""
        insights = []
        
        try:
            # Social sentiment insights
            social_sentiment = social.get('sentiment', 'NEUTRAL')
            if social_sentiment == 'BULLISH' and social.get('confidence', 0) > 0.7:
                insights.append("Strong bullish social media sentiment detected")
            elif social_sentiment == 'BEARISH' and social.get('confidence', 0) > 0.7:
                insights.append("Strong bearish social media sentiment detected")
            
            # News sentiment insights
            news_sentiment = news.get('sentiment', 'NEUTRAL')
            if news_sentiment == 'POSITIVE' and news.get('confidence', 0) > 0.6:
                insights.append("Positive news flow supporting price action")
            elif news_sentiment == 'NEGATIVE' and news.get('confidence', 0) > 0.6:
                insights.append("Negative news creating selling pressure")
            
            # Technical sentiment insights
            technical_sentiment = technical.get('sentiment', 'NEUTRAL')
            price_change_24h = technical.get('price_change_24h', 0)
            if technical_sentiment == 'BULLISH' and price_change_24h > 10:
                insights.append("Strong technical momentum with significant price gains")
            elif technical_sentiment == 'BEARISH' and price_change_24h < -10:
                insights.append("Weak technical momentum with significant price decline")
            
            # Fear & greed insights
            fg_sentiment = fear_greed.get('sentiment', 'NEUTRAL')
            if fg_sentiment == 'EXTREME_GREED':
                insights.append("Market showing extreme greed - potential top signal")
            elif fg_sentiment == 'EXTREME_FEAR':
                insights.append("Market showing extreme fear - potential bottom signal")
            
            # Cross-correlation insights
            bullish_sources = sum(1 for s in [social_sentiment, news_sentiment, technical_sentiment] 
                                if 'BULLISH' in s or 'POSITIVE' in s)
            bearish_sources = sum(1 for s in [social_sentiment, news_sentiment, technical_sentiment] 
                                if 'BEARISH' in s or 'NEGATIVE' in s)
            
            if bullish_sources >= 2:
                insights.append("Multiple sentiment sources confirming bullish bias")
            elif bearish_sources >= 2:
                insights.append("Multiple sentiment sources confirming bearish bias")
            else:
                insights.append("Mixed sentiment signals - market uncertainty")
            
            return insights[:5]  # Return top 5 insights
            
        except Exception as e:
            logger.error(f"Error generating sentiment insights: {e}")
            return ["Sentiment analysis completed"]
    
    async def _train_sentiment_models(self):
        """Train sentiment models with sample data"""
        try:
            # This would normally use real labeled training data
            # For now, create simple training examples
            crypto_texts = [
                "Bitcoin to the moon, buying more",
                "Ethereum pumping hard, bullish",
                "Dump incoming, selling everything",
                "Bear market confirmed, stay safe",
                "HODL strong, diamond hands",
                "Market crash, liquidated positions"
            ]
            
            crypto_labels = [1, 1, 0, 0, 1, 0]  # 1 = positive, 0 = negative
            
            # Train crypto model (simplified)
            if len(crypto_texts) > 0:
                self.sentiment_models['crypto'].fit([[1, 0], [1, 0], [0, 1], [0, 1], [1, 0], [0, 1]], crypto_labels)
            
            logger.info("Sentiment models trained with sample data")
            
        except Exception as e:
            logger.error(f"Error training sentiment models: {e}")
    
    async def get_sentiment_summary(self, symbols: List[str]) -> Dict[str, Any]:
        """Get overall market sentiment summary"""
        try:
            all_sentiments = []
            
            for symbol in symbols[:10]:  # Limit to 10 symbols
                sentiment = await self.analyze_market_sentiment(symbol)
                all_sentiments.append(sentiment)
            
            if not all_sentiments:
                return {'overall_sentiment': 'NEUTRAL', 'confidence': 0.0}
            
            # Calculate overall market sentiment
            sentiment_scores = [s.get('score', 0.0) for s in all_sentiments]
            avg_sentiment = np.mean(sentiment_scores)
            avg_confidence = np.mean([s.get('confidence', 0.0) for s in all_sentiments])
            
            if avg_sentiment > 0.2:
                overall_sentiment = 'BULLISH'
            elif avg_sentiment < -0.2:
                overall_sentiment = 'BEARISH'
            else:
                overall_sentiment = 'NEUTRAL'
            
            return {
                'overall_sentiment': overall_sentiment,
                'sentiment_score': avg_sentiment,
                'confidence': avg_confidence,
                'symbols_analyzed': len(all_sentiments),
                'bullish_count': sum(1 for s in all_sentiments if 'BULLISH' in s.get('sentiment', '')),
                'bearish_count': sum(1 for s in all_sentiments if 'BEARISH' in s.get('sentiment', '')),
                'neutral_count': sum(1 for s in all_sentiments if s.get('sentiment', '') == 'NEUTRAL')
            }
            
        except Exception as e:
            logger.error(f"Error getting sentiment summary: {e}")
            return {'overall_sentiment': 'NEUTRAL', 'confidence': 0.0}
    
    async def close(self):
        """Close the sentiment engine"""
        self.initialized = False
        logger.info("Market Sentiment Analysis Engine closed")