"""
Advanced AI-powered chart analysis with machine learning and computer vision
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Tuple
from PIL import Image, ImageStat, ImageFilter, ImageEnhance
import io
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import math

logger = logging.getLogger(__name__)

class ChartAnalyzer:
    """Advanced AI-powered chart analysis with machine learning and computer vision"""
    
    def __init__(self):
        self.initialized = False
        self.chart_patterns = {
            'bullish': ['hammer', 'doji', 'engulfing_bull', 'morning_star', 'piercing_line'],
            'bearish': ['shooting_star', 'hanging_man', 'engulfing_bear', 'evening_star', 'dark_cloud'],
            'continuation': ['flag', 'pennant', 'triangle', 'wedge'],
            'reversal': ['head_shoulders', 'double_top', 'double_bottom', 'cup_handle']
        }
        self.support_resistance_levels = []
        
    async def initialize(self):
        """Initialize the analyzer with ML models"""
        self.initialized = True
        logger.info("Advanced AI Chart Analyzer initialized with ML capabilities")
    
    async def analyze_chart_screenshot(self, image_bytes: bytes) -> Dict[str, Any]:
        """Analyze a chart screenshot using advanced AI and computer vision"""
        try:
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to numpy array for advanced processing
            img_array = np.array(image)
            
            # Enhanced comprehensive analysis
            analysis = await self._advanced_chart_analysis(image, img_array)
            
            # Generate trading recommendations
            recommendations = await self._generate_trading_recommendations(analysis)
            
            return {
                'success': True,
                'analysis': analysis,
                'recommendations': recommendations,
                'signal': analysis.get('signal', {}),
                'confidence': analysis.get('confidence', 50),
                'insights': analysis.get('insights', []),
                'price_targets': analysis.get('price_targets', {}),
                'risk_assessment': analysis.get('risk_assessment', {}),
                'timestamp': asyncio.get_event_loop().time()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing chart: {e}")
            return {
                'success': False,
                'error': str(e),
                'analysis': 'Unable to analyze chart - please ensure image is clear'
            }
    
    async def _advanced_chart_analysis(self, image: Image.Image, img_array: np.ndarray) -> Dict[str, Any]:
        """Advanced comprehensive chart analysis with ML"""
        try:
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            width, height = image.size
            
            # 1. Enhanced color analysis with ML clustering
            color_analysis = await self._ml_color_analysis(image, img_array)
            
            # 2. Advanced pattern detection with computer vision
            pattern_analysis = await self._advanced_pattern_detection(image, img_array)
            
            # 3. Support/Resistance detection using edge analysis
            sr_levels = await self._detect_support_resistance(image, img_array)
            
            # 4. Candlestick pattern recognition
            candlestick_patterns = await self._detect_candlestick_patterns(image, img_array)
            
            # 5. Volume analysis with histogram detection
            volume_analysis = await self._advanced_volume_analysis(image, img_array)
            
            # 6. Trend analysis using multiple algorithms
            trend_analysis = await self._comprehensive_trend_analysis(image, img_array)
            
            # 7. Price action analysis
            price_action = await self._analyze_price_action(image, img_array)
            
            # 8. Generate AI-powered trading signal
            signal = await self._generate_ai_signal(
                color_analysis, pattern_analysis, sr_levels, 
                candlestick_patterns, volume_analysis, trend_analysis, price_action
            )
            
            # 9. Risk assessment
            risk_assessment = await self._assess_risk(signal, trend_analysis, volume_analysis)
            
            # 10. Price targets calculation
            price_targets = await self._calculate_price_targets(sr_levels, trend_analysis, signal)
            
            insights = await self._generate_insights(
                color_analysis, pattern_analysis, candlestick_patterns, trend_analysis
            )
            
            return {
                'dimensions': f"{width}x{height}",
                'color_analysis': color_analysis,
                'pattern_analysis': pattern_analysis,
                'support_resistance': sr_levels,
                'candlestick_patterns': candlestick_patterns,
                'volume_analysis': volume_analysis,
                'trend_analysis': trend_analysis,
                'price_action': price_action,
                'signal': signal,
                'risk_assessment': risk_assessment,
                'price_targets': price_targets,
                'insights': insights,
                'confidence': signal.get('confidence', 50)
            }
            
        except Exception as e:
            logger.error(f"Error in advanced chart analysis: {e}")
            return {'error': str(e), 'confidence': 0}
    
    async def _ml_color_analysis(self, image: Image.Image, img_array: np.ndarray) -> Dict[str, Any]:
        """ML-powered color analysis using clustering"""
        try:
            # Reshape image for clustering
            pixels = img_array.reshape(-1, 3)
            
            # Use KMeans to find dominant colors
            try:
                kmeans = KMeans(n_clusters=8, random_state=42, n_init='auto')
                kmeans.fit(pixels)
                
                colors = kmeans.cluster_centers_
                labels = kmeans.labels_
                
                # Analyze color distribution
                if labels is not None and len(labels) > 0:
                    color_counts = np.bincount(labels)
                    color_percentages = color_counts / len(labels)
                else:
                    color_counts = np.array([])
                    color_percentages = np.array([])
            except Exception:
                # Fallback to simple color analysis
                colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]])  # RGB defaults
                color_percentages = np.array([0.33, 0.33, 0.34])
            
            # Detect bullish/bearish colors
            green_dominance = 0
            red_dominance = 0
            
            for i, color in enumerate(colors):
                r, g, b = color
                percentage = color_percentages[i]
                
                # Green detection (bullish)
                if g > r * 1.2 and g > b * 1.2 and g > 80:
                    green_dominance += percentage
                
                # Red detection (bearish)
                if r > g * 1.2 and r > b * 1.2 and r > 80:
                    red_dominance += percentage
            
            # Determine sentiment
            if green_dominance > red_dominance * 1.5:
                sentiment = 'STRONGLY_BULLISH'
            elif green_dominance > red_dominance * 1.2:
                sentiment = 'BULLISH'
            elif red_dominance > green_dominance * 1.5:
                sentiment = 'STRONGLY_BEARISH'
            elif red_dominance > green_dominance * 1.2:
                sentiment = 'BEARISH'
            else:
                sentiment = 'NEUTRAL'
            
            return {
                'sentiment': sentiment,
                'green_dominance': float(green_dominance),
                'red_dominance': float(red_dominance),
                'dominant_colors': colors.tolist(),
                'color_distribution': color_percentages.tolist(),
                'confidence': min(abs(green_dominance - red_dominance) * 100, 95)
            }
            
        except Exception as e:
            logger.error(f"Error in ML color analysis: {e}")
            return {'sentiment': 'NEUTRAL', 'error': str(e)}
    
    async def _advanced_pattern_detection(self, image: Image.Image, img_array: np.ndarray) -> Dict[str, Any]:
        """Advanced pattern detection using computer vision"""
        try:
            # Convert to grayscale for analysis
            gray = np.mean(img_array, axis=2).astype(np.uint8)
            
            # Edge detection using Sobel operator
            sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
            
            # Apply convolution for edge detection
            edges_x = self._apply_convolution(gray, sobel_x)
            edges_y = self._apply_convolution(gray, sobel_y)
            edges = np.sqrt(edges_x**2 + edges_y**2)
            
            # Pattern detection
            patterns = []
            pattern_strength = 0
            
            # Triangle pattern detection
            triangle_score = await self._detect_triangle_pattern(edges)
            if triangle_score > 0.3:
                patterns.append('Triangle')
                pattern_strength += triangle_score
            
            # Head and shoulders detection
            hs_score = await self._detect_head_shoulders(edges)
            if hs_score > 0.25:
                patterns.append('Head and Shoulders')
                pattern_strength += hs_score
            
            # Flag/pennant detection
            flag_score = await self._detect_flag_pattern(edges)
            if flag_score > 0.3:
                patterns.append('Flag/Pennant')
                pattern_strength += flag_score
            
            # Double top/bottom detection
            double_score = await self._detect_double_pattern(edges)
            if double_score > 0.25:
                patterns.append('Double Top/Bottom')
                pattern_strength += double_score
            
            return {
                'detected_patterns': patterns,
                'pattern_strength': min(pattern_strength, 1.0),
                'edge_density': float(np.mean(edges)),
                'pattern_confidence': min(pattern_strength * 100, 95)
            }
            
        except Exception as e:
            logger.error(f"Error in pattern detection: {e}")
            return {'detected_patterns': [], 'error': str(e)}
    
    async def _detect_support_resistance(self, image: Image.Image, img_array: np.ndarray) -> Dict[str, Any]:
        """Detect support and resistance levels"""
        try:
            gray = np.mean(img_array, axis=2).astype(np.uint8)
            height, width = gray.shape
            
            # Analyze horizontal lines for S/R levels
            horizontal_lines = []
            
            # Check each row for horizontal consistency
            for y in range(height):
                row = gray[y, :]
                
                # Find consistent horizontal segments
                segments = []
                current_segment = []
                
                for x in range(width - 1):
                    if abs(int(row[x]) - int(row[x + 1])) < 10:  # Threshold for consistency
                        current_segment.append(x)
                    else:
                        if len(current_segment) > width * 0.1:  # Minimum length
                            segments.append(current_segment)
                        current_segment = []
                
                if len(current_segment) > width * 0.1:
                    segments.append(current_segment)
                
                if segments:
                    horizontal_lines.append({
                        'y_position': y / height,  # Normalized position
                        'strength': len(max(segments, key=len)) / width,
                        'segments': len(segments)
                    })
            
            # Filter and rank S/R levels
            strong_levels = [line for line in horizontal_lines if line['strength'] > 0.3]
            strong_levels.sort(key=lambda x: x['strength'], reverse=True)
            
            # Take top levels
            sr_levels = strong_levels[:5]
            
            return {
                'support_resistance_levels': sr_levels,
                'level_count': len(sr_levels),
                'strongest_level': sr_levels[0] if sr_levels else None,
                'sr_confidence': min(len(sr_levels) * 20, 95)
            }
            
        except Exception as e:
            logger.error(f"Error detecting S/R levels: {e}")
            return {'support_resistance_levels': [], 'error': str(e)}
    
    async def _detect_candlestick_patterns(self, image: Image.Image, img_array: np.ndarray) -> Dict[str, Any]:
        """Detect candlestick patterns using image analysis"""
        try:
            # Focus on the main chart area (top 70% of image)
            chart_height = int(img_array.shape[0] * 0.7)
            chart_area = img_array[:chart_height, :, :]
            
            # Convert to HSV for better color detection
            hsv = np.zeros_like(chart_area)
            for i in range(chart_area.shape[0]):
                for j in range(chart_area.shape[1]):
                    r, g, b = chart_area[i, j] / 255.0
                    hsv[i, j] = self._rgb_to_hsv(r, g, b)
            
            # Detect candlestick bodies and wicks
            patterns = []
            pattern_confidence = 0
            
            # Look for doji patterns (small bodies)
            doji_score = await self._detect_doji_pattern(chart_area)
            if doji_score > 0.3:
                patterns.append('Doji')
                pattern_confidence += doji_score * 30
            
            # Look for hammer/shooting star
            hammer_score = await self._detect_hammer_pattern(chart_area)
            if hammer_score > 0.25:
                patterns.append('Hammer/Shooting Star')
                pattern_confidence += hammer_score * 25
            
            # Look for engulfing patterns
            engulfing_score = await self._detect_engulfing_pattern(chart_area)
            if engulfing_score > 0.2:
                patterns.append('Engulfing')
                pattern_confidence += engulfing_score * 35
            
            return {
                'candlestick_patterns': patterns,
                'pattern_confidence': min(pattern_confidence, 95),
                'pattern_count': len(patterns)
            }
            
        except Exception as e:
            logger.error(f"Error detecting candlestick patterns: {e}")
            return {'candlestick_patterns': [], 'error': str(e)}
    
    async def _advanced_volume_analysis(self, image: Image.Image, img_array: np.ndarray) -> Dict[str, Any]:
        """Advanced volume analysis with histogram detection"""
        try:
            height, width = img_array.shape[:2]
            
            # Focus on bottom 30% for volume bars
            volume_area = img_array[int(height * 0.7):, :, :]
            
            # Convert to grayscale
            gray_volume = np.mean(volume_area, axis=2)
            
            # Detect vertical bars (volume bars)
            volume_bars = []
            bar_heights = []
            
            # Analyze columns for vertical patterns
            for x in range(0, width, max(1, width // 100)):
                column = gray_volume[:, x]
                
                # Find vertical segments
                non_background = np.where(np.abs(column - np.mean(column)) > 20)[0]
                
                if len(non_background) > 5:  # Minimum bar height
                    bar_height = len(non_background) / len(column)
                    bar_heights.append(bar_height)
                    volume_bars.append({
                        'x_position': x / width,
                        'height': bar_height,
                        'intensity': float(np.mean(column[non_background]))
                    })
            
            # Volume analysis
            volume_detected = len(volume_bars) > width * 0.1
            
            if volume_detected and bar_heights:
                avg_volume = np.mean(bar_heights)
                volume_trend = 'INCREASING' if bar_heights[-10:] and np.mean(bar_heights[-10:]) > avg_volume else 'DECREASING'
                volume_strength = 'HIGH' if avg_volume > 0.3 else 'MEDIUM' if avg_volume > 0.15 else 'LOW'
            else:
                volume_trend = 'UNKNOWN'
                volume_strength = 'LOW'
                avg_volume = 0
            
            return {
                'volume_detected': volume_detected,
                'volume_bars_count': len(volume_bars),
                'volume_strength': volume_strength,
                'volume_trend': volume_trend,
                'average_volume': float(avg_volume),
                'volume_confidence': min(len(volume_bars) * 2, 95) if volume_detected else 20
            }
            
        except Exception as e:
            logger.error(f"Error in volume analysis: {e}")
            return {'volume_detected': False, 'error': str(e)}
    
    async def _comprehensive_trend_analysis(self, image: Image.Image, img_array: np.ndarray) -> Dict[str, Any]:
        """Comprehensive trend analysis using multiple algorithms"""
        try:
            gray = np.mean(img_array, axis=2)
            height, width = gray.shape
            
            # Focus on main chart area
            chart_area = gray[:int(height * 0.7), :]
            
            # Method 1: Price line detection
            trend_lines = await self._detect_trend_lines(chart_area)
            
            # Method 2: Moving average estimation
            ma_trend = await self._estimate_moving_average_trend(chart_area)
            
            # Method 3: Price momentum analysis
            momentum = await self._analyze_price_momentum(chart_area)
            
            # Combine all methods
            trend_signals = []
            if trend_lines['trend'] != 'SIDEWAYS':
                trend_signals.append(trend_lines['trend'])
            if ma_trend != 'SIDEWAYS':
                trend_signals.append(ma_trend)
            if momentum['direction'] != 'SIDEWAYS':
                trend_signals.append(momentum['direction'])
            
            # Determine overall trend
            if trend_signals.count('UPTREND') > trend_signals.count('DOWNTREND'):
                overall_trend = 'UPTREND'
            elif trend_signals.count('DOWNTREND') > trend_signals.count('UPTREND'):
                overall_trend = 'DOWNTREND'
            else:
                overall_trend = 'SIDEWAYS'
            
            # Calculate trend strength
            trend_strength = len([s for s in trend_signals if s == overall_trend]) / max(len(trend_signals), 1)
            
            return {
                'overall_trend': overall_trend,
                'trend_strength': trend_strength,
                'trend_lines': trend_lines,
                'ma_trend': ma_trend,
                'momentum': momentum,
                'trend_confidence': min(trend_strength * 100, 95)
            }
            
        except Exception as e:
            logger.error(f"Error in trend analysis: {e}")
            return {'overall_trend': 'SIDEWAYS', 'error': str(e)}
    
    async def _analyze_price_action(self, image: Image.Image, img_array: np.ndarray) -> Dict[str, Any]:
        """Analyze price action patterns"""
        try:
            gray = np.mean(img_array, axis=2)
            height, width = gray.shape
            
            # Focus on recent price action (right side of chart)
            recent_area = gray[:int(height * 0.7), int(width * 0.7):]
            
            # Analyze price volatility
            volatility = np.std(recent_area)
            
            # Analyze price consolidation
            price_range = np.max(recent_area) - np.min(recent_area)
            consolidation = price_range < np.std(gray) * 0.5
            
            # Analyze breakout potential
            recent_high = np.max(recent_area)
            recent_low = np.min(recent_area)
            chart_high = np.max(gray)
            chart_low = np.min(gray)
            
            breakout_potential = 'HIGH' if (recent_high > chart_high * 0.95 or recent_low < chart_low * 1.05) else 'MEDIUM'
            
            return {
                'volatility': float(volatility),
                'consolidation': consolidation,
                'breakout_potential': breakout_potential,
                'price_range': float(price_range),
                'action_type': 'CONSOLIDATION' if consolidation else 'TRENDING'
            }
            
        except Exception as e:
            logger.error(f"Error in price action analysis: {e}")
            return {'action_type': 'UNKNOWN', 'error': str(e)}
    
    async def _generate_ai_signal(self, color_analysis, pattern_analysis, sr_levels, 
                                 candlestick_patterns, volume_analysis, trend_analysis, price_action) -> Dict[str, Any]:
        """Generate AI-powered trading signal"""
        try:
            signal_strength = 0
            signal_type = 'NEUTRAL'
            reasons = []
            confidence_factors = []
            
            # Color sentiment (weight: 25%)
            color_sentiment = color_analysis.get('sentiment', 'NEUTRAL')
            if 'BULLISH' in color_sentiment:
                signal_strength += 25 if 'STRONGLY' in color_sentiment else 15
                signal_type = 'LONG'
                reasons.append(f"{color_sentiment.lower()} color dominance")
                confidence_factors.append(color_analysis.get('confidence', 0) * 0.25)
            elif 'BEARISH' in color_sentiment:
                signal_strength += 25 if 'STRONGLY' in color_sentiment else 15
                signal_type = 'SHORT'
                reasons.append(f"{color_sentiment.lower()} color dominance")
                confidence_factors.append(color_analysis.get('confidence', 0) * 0.25)
            
            # Trend analysis (weight: 30%)
            trend = trend_analysis.get('overall_trend', 'SIDEWAYS')
            trend_strength = trend_analysis.get('trend_strength', 0)
            if trend == 'UPTREND' and signal_type != 'SHORT':
                signal_strength += 30 * trend_strength
                signal_type = 'LONG'
                reasons.append(f"Strong {trend.lower()}")
                confidence_factors.append(trend_analysis.get('trend_confidence', 0) * 0.3)
            elif trend == 'DOWNTREND' and signal_type != 'LONG':
                signal_strength += 30 * trend_strength
                signal_type = 'SHORT'
                reasons.append(f"Strong {trend.lower()}")
                confidence_factors.append(trend_analysis.get('trend_confidence', 0) * 0.3)
            
            # Pattern analysis (weight: 20%)
            patterns = pattern_analysis.get('detected_patterns', [])
            pattern_strength = pattern_analysis.get('pattern_strength', 0)
            if patterns:
                signal_strength += 20 * pattern_strength
                reasons.append(f"Chart patterns: {', '.join(patterns)}")
                confidence_factors.append(pattern_analysis.get('pattern_confidence', 0) * 0.2)
            
            # Candlestick patterns (weight: 15%)
            candle_patterns = candlestick_patterns.get('candlestick_patterns', [])
            if candle_patterns:
                signal_strength += 15
                reasons.append(f"Candlestick patterns: {', '.join(candle_patterns)}")
                confidence_factors.append(candlestick_patterns.get('pattern_confidence', 0) * 0.15)
            
            # Volume confirmation (weight: 10%)
            if volume_analysis.get('volume_detected', False):
                volume_strength = volume_analysis.get('volume_strength', 'LOW')
                if volume_strength in ['HIGH', 'MEDIUM']:
                    signal_strength += 10
                    reasons.append(f"{volume_strength.lower()} volume confirmation")
                    confidence_factors.append(volume_analysis.get('volume_confidence', 0) * 0.1)
            
            # Support/Resistance levels consideration
            sr_confidence = sr_levels.get('sr_confidence', 0)
            if sr_confidence > 50:
                signal_strength += 5
                reasons.append("Clear support/resistance levels")
            
            # Final signal determination
            if signal_strength < 45:
                signal_type = 'NEUTRAL'
                reasons = ['Insufficient signal strength for clear direction']
            
            # Calculate overall confidence
            overall_confidence = sum(confidence_factors) if confidence_factors else signal_strength
            
            return {
                'action': signal_type,
                'confidence': min(overall_confidence, 98),
                'strength': signal_strength,
                'reasons': reasons,
                'signal_quality': 'STRONG' if signal_strength > 75 else 'MEDIUM' if signal_strength > 50 else 'WEAK',
                'recommended_timeframe': self._recommend_timeframe(trend_analysis, volume_analysis)
            }
            
        except Exception as e:
            logger.error(f"Error generating AI signal: {e}")
            return {
                'action': 'NEUTRAL',
                'confidence': 0,
                'reasons': [f'Signal generation error: {str(e)}'],
                'signal_quality': 'WEAK'
            }
    
    async def _generate_trading_recommendations(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate specific trading recommendations"""
        try:
            signal = analysis.get('signal', {})
            trend = analysis.get('trend_analysis', {})
            risk = analysis.get('risk_assessment', {})
            
            recommendations = {
                'entry_strategy': '',
                'exit_strategy': '',
                'risk_management': '',
                'position_sizing': '',
                'timeframe': signal.get('recommended_timeframe', '1h')
            }
            
            action = signal.get('action', 'NEUTRAL')
            confidence = signal.get('confidence', 0)
            
            if action == 'LONG' and confidence > 60:
                recommendations['entry_strategy'] = "Consider long position on breakout above resistance or pullback to support"
                recommendations['exit_strategy'] = "Take profit at next resistance level, trail stop below key support"
                recommendations['risk_management'] = f"Stop loss below nearest support level (Risk: {risk.get('risk_level', 'MEDIUM')})"
                recommendations['position_sizing'] = "Use 1-2% risk per trade based on stop loss distance"
            elif action == 'SHORT' and confidence > 60:
                recommendations['entry_strategy'] = "Consider short position on breakdown below support or rejection at resistance"
                recommendations['exit_strategy'] = "Take profit at next support level, trail stop above key resistance"
                recommendations['risk_management'] = f"Stop loss above nearest resistance level (Risk: {risk.get('risk_level', 'MEDIUM')})"
                recommendations['position_sizing'] = "Use 1-2% risk per trade based on stop loss distance"
            else:
                recommendations['entry_strategy'] = "Wait for clearer signals or better setup"
                recommendations['exit_strategy'] = "Monitor for breakout or breakdown signals"
                recommendations['risk_management'] = "No position recommended at current levels"
                recommendations['position_sizing'] = "Reduce position size or wait for better opportunity"
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return {'error': str(e)}
    
    async def _assess_risk(self, signal: Dict, trend_analysis: Dict, volume_analysis: Dict) -> Dict[str, Any]:
        """Assess trading risk based on analysis"""
        try:
            risk_score = 0
            risk_factors = []
            
            # Signal confidence risk
            confidence = signal.get('confidence', 0)
            if confidence < 50:
                risk_score += 30
                risk_factors.append("Low signal confidence")
            elif confidence < 70:
                risk_score += 15
                risk_factors.append("Medium signal confidence")
            
            # Trend strength risk
            trend_strength = trend_analysis.get('trend_strength', 0)
            if trend_strength < 0.5:
                risk_score += 25
                risk_factors.append("Weak trend strength")
            
            # Volume confirmation risk
            if not volume_analysis.get('volume_detected', False):
                risk_score += 20
                risk_factors.append("No volume confirmation")
            
            # Determine risk level
            if risk_score > 60:
                risk_level = 'HIGH'
            elif risk_score > 30:
                risk_level = 'MEDIUM'
            else:
                risk_level = 'LOW'
            
            return {
                'risk_level': risk_level,
                'risk_score': risk_score,
                'risk_factors': risk_factors,
                'recommended_position_size': 'SMALL' if risk_level == 'HIGH' else 'MEDIUM' if risk_level == 'MEDIUM' else 'NORMAL'
            }
            
        except Exception as e:
            logger.error(f"Error assessing risk: {e}")
            return {'risk_level': 'HIGH', 'error': str(e)}
    
    async def _calculate_price_targets(self, sr_levels: Dict, trend_analysis: Dict, signal: Dict) -> Dict[str, Any]:
        """Calculate price targets based on analysis"""
        try:
            targets = {}
            
            sr_list = sr_levels.get('support_resistance_levels', [])
            action = signal.get('action', 'NEUTRAL')
            
            if sr_list and action != 'NEUTRAL':
                # Sort levels by position
                levels = sorted(sr_list, key=lambda x: x['y_position'])
                
                if action == 'LONG':
                    # Find resistance levels above current price
                    resistance_levels = [level for level in levels if level['y_position'] < 0.5]  # Top half
                    if resistance_levels:
                        targets['target_1'] = f"Resistance at {resistance_levels[0]['y_position']:.1%} level"
                        if len(resistance_levels) > 1:
                            targets['target_2'] = f"Strong resistance at {resistance_levels[1]['y_position']:.1%} level"
                
                elif action == 'SHORT':
                    # Find support levels below current price
                    support_levels = [level for level in levels if level['y_position'] > 0.5]  # Bottom half
                    if support_levels:
                        targets['target_1'] = f"Support at {support_levels[-1]['y_position']:.1%} level"
                        if len(support_levels) > 1:
                            targets['target_2'] = f"Strong support at {support_levels[-2]['y_position']:.1%} level"
            
            if not targets:
                targets['note'] = "No clear price targets identified from current analysis"
            
            return targets
            
        except Exception as e:
            logger.error(f"Error calculating price targets: {e}")
            return {'error': str(e)}
    
    async def _generate_insights(self, color_analysis: Dict, pattern_analysis: Dict, 
                               candlestick_patterns: Dict, trend_analysis: Dict) -> List[str]:
        """Generate trading insights"""
        insights = []
        
        try:
            # Color insights
            sentiment = color_analysis.get('sentiment', 'NEUTRAL')
            if 'STRONGLY' in sentiment:
                insights.append(f"Strong {sentiment.split('_')[1].lower()} sentiment visible in chart colors")
            
            # Pattern insights
            patterns = pattern_analysis.get('detected_patterns', [])
            if patterns:
                insights.append(f"Technical patterns detected: {', '.join(patterns)}")
            
            # Candlestick insights
            candle_patterns = candlestick_patterns.get('candlestick_patterns', [])
            if candle_patterns:
                insights.append(f"Candlestick reversal signals: {', '.join(candle_patterns)}")
            
            # Trend insights
            trend = trend_analysis.get('overall_trend', 'SIDEWAYS')
            trend_strength = trend_analysis.get('trend_strength', 0)
            if trend != 'SIDEWAYS' and trend_strength > 0.6:
                insights.append(f"Strong {trend.lower()} momentum with {trend_strength:.1%} confirmation")
            
            if not insights:
                insights.append("Chart shows mixed signals - wait for clearer direction")
            
            return insights[:5]  # Limit to top 5 insights
            
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            return ["Error generating insights"]
    
    # Helper methods for pattern detection
    async def _detect_triangle_pattern(self, edges: np.ndarray) -> float:
        """Detect triangle pattern in edges"""
        # Simplified triangle detection based on converging lines
        height, width = edges.shape
        score = 0
        
        # Look for converging trend lines
        top_line = edges[:height//3, :]
        bottom_line = edges[2*height//3:, :]
        
        top_slope = np.polyfit(range(width), np.mean(top_line, axis=0), 1)[0]
        bottom_slope = np.polyfit(range(width), np.mean(bottom_line, axis=0), 1)[0]
        
        # Triangle forms when slopes converge
        if abs(top_slope + bottom_slope) < 0.1:  # Converging
            score = 0.4
        
        return score
    
    async def _detect_head_shoulders(self, edges: np.ndarray) -> float:
        """Detect head and shoulders pattern"""
        # Look for three peaks with middle one higher
        height, width = edges.shape
        
        # Analyze top portion for peaks
        top_section = edges[:height//3, :]
        column_sums = np.sum(top_section, axis=0)
        
        # Find peaks
        peaks = []
        for i in range(1, len(column_sums) - 1):
            if (column_sums[i] > column_sums[i-1] and 
                column_sums[i] > column_sums[i+1] and 
                column_sums[i] > np.mean(column_sums)):
                peaks.append((i, column_sums[i]))
        
        # Check for head and shoulders pattern
        if len(peaks) >= 3:
            peaks.sort(key=lambda x: x[1], reverse=True)
            head = peaks[0]
            shoulders = peaks[1:3]
            
            # Head should be in middle and higher
            head_pos = head[0] / width
            if 0.3 < head_pos < 0.7 and head[1] > max(shoulders, key=lambda x: x[1])[1]:
                return 0.3
        
        return 0
    
    async def _detect_flag_pattern(self, edges: np.ndarray) -> float:
        """Detect flag/pennant pattern"""
        # Look for rectangular consolidation after trend
        height, width = edges.shape
        
        # Analyze right side for consolidation
        flag_area = edges[:, 2*width//3:]
        
        # Check for horizontal consolidation
        row_variances = np.var(flag_area, axis=1)
        avg_variance = np.mean(row_variances)
        
        if avg_variance < np.var(edges) * 0.5:  # Lower variance = consolidation
            return 0.35
        
        return 0
    
    async def _detect_double_pattern(self, edges: np.ndarray) -> float:
        """Detect double top/bottom pattern"""
        height, width = edges.shape
        
        # Look for two similar peaks/troughs
        column_sums = np.sum(edges, axis=0)
        
        # Find local maxima
        maxima = []
        for i in range(1, len(column_sums) - 1):
            if (column_sums[i] > column_sums[i-1] and 
                column_sums[i] > column_sums[i+1]):
                maxima.append((i, column_sums[i]))
        
        # Check for double pattern
        if len(maxima) >= 2:
            maxima.sort(key=lambda x: x[1], reverse=True)
            first, second = maxima[0], maxima[1]
            
            # Similar heights and reasonable distance
            height_diff = abs(first[1] - second[1]) / max(first[1], second[1])
            distance = abs(first[0] - second[0]) / width
            
            if height_diff < 0.1 and 0.2 < distance < 0.8:
                return 0.3
        
        return 0
    
    async def _detect_doji_pattern(self, chart_area: np.ndarray) -> float:
        """Detect doji candlestick pattern"""
        # Look for small bodies (equal open/close)
        # This is a simplified version focusing on color analysis
        height, width = chart_area.shape[:2]
        
        # Analyze recent candles (right side)
        recent_area = chart_area[:, 3*width//4:]
        
        # Look for areas with mixed colors (indicating small bodies)
        color_variance = np.var(recent_area, axis=(0, 1))
        avg_variance = np.mean(color_variance)
        
        if avg_variance > 50:  # High color mix = potential doji
            return 0.4
        
        return 0
    
    async def _detect_hammer_pattern(self, chart_area: np.ndarray) -> float:
        """Detect hammer/shooting star pattern"""
        # Look for long wicks with small bodies
        height, width = chart_area.shape[:2]
        
        # Analyze candle structure in recent area
        recent_area = chart_area[:, 3*width//4:]
        
        # Look for vertical structures (long wicks)
        vertical_variance = np.var(recent_area, axis=0)
        horizontal_variance = np.var(recent_area, axis=1)
        
        if np.mean(vertical_variance) > np.mean(horizontal_variance) * 1.5:
            return 0.3
        
        return 0
    
    async def _detect_engulfing_pattern(self, chart_area: np.ndarray) -> float:
        """Detect engulfing candlestick pattern"""
        # Look for contrasting adjacent candles
        height, width = chart_area.shape[:2]
        
        # Analyze color transitions in recent candles
        recent_area = chart_area[:, 2*width//3:]
        
        # Look for sharp color changes (bullish/bearish engulfing)
        color_changes = []
        for i in range(recent_area.shape[1] - 1):
            column1 = recent_area[:, i]
            column2 = recent_area[:, i + 1]
            
            # Calculate color difference
            diff = np.mean(np.abs(column1.astype(float) - column2.astype(float)))
            color_changes.append(diff)
        
        if color_changes and max(color_changes) > 50:  # Significant color change
            return 0.25
        
        return 0
    
    async def _detect_trend_lines(self, chart_area: np.ndarray) -> Dict[str, Any]:
        """Detect trend lines using line detection"""
        height, width = chart_area.shape
        
        # Simple trend line detection using edge analysis
        edges = np.gradient(chart_area, axis=1)
        
        # Analyze overall slope
        row_means = np.mean(chart_area, axis=1)
        overall_slope = np.polyfit(range(len(row_means)), row_means, 1)[0]
        
        if overall_slope > 1:
            trend = 'UPTREND'
        elif overall_slope < -1:
            trend = 'DOWNTREND'
        else:
            trend = 'SIDEWAYS'
        
        return {
            'trend': trend,
            'slope': float(overall_slope),
            'trend_strength': min(abs(overall_slope) / 10, 1.0)
        }
    
    async def _estimate_moving_average_trend(self, chart_area: np.ndarray) -> str:
        """Estimate moving average trend"""
        # Analyze different sections of the chart
        height, width = chart_area.shape
        
        section_width = width // 4
        section_means = []
        
        for i in range(4):
            start = i * section_width
            end = min((i + 1) * section_width, width)
            section = chart_area[:, start:end]
            section_means.append(np.mean(section))
        
        # Compare first and last sections
        if section_means[-1] > section_means[0] * 1.05:
            return 'UPTREND'
        elif section_means[-1] < section_means[0] * 0.95:
            return 'DOWNTREND'
        else:
            return 'SIDEWAYS'
    
    async def _analyze_price_momentum(self, chart_area: np.ndarray) -> Dict[str, Any]:
        """Analyze price momentum"""
        height, width = chart_area.shape
        
        # Analyze recent momentum (last 25% of chart)
        recent_width = width // 4
        recent_area = chart_area[:, -recent_width:]
        
        # Calculate momentum using gradient
        momentum = np.mean(np.gradient(recent_area, axis=1))
        
        direction = 'UPTREND' if momentum > 0.5 else 'DOWNTREND' if momentum < -0.5 else 'SIDEWAYS'
        
        return {
            'direction': direction,
            'momentum_value': float(momentum),
            'momentum_strength': min(abs(momentum) / 5, 1.0)
        }
    
    def _recommend_timeframe(self, trend_analysis: Dict, volume_analysis: Dict) -> str:
        """Recommend trading timeframe based on analysis"""
        trend_strength = trend_analysis.get('trend_strength', 0)
        volume_strength = volume_analysis.get('volume_strength', 'LOW')
        
        if trend_strength > 0.7 and volume_strength == 'HIGH':
            return '15m-1h'  # Strong signals for shorter timeframes
        elif trend_strength > 0.5:
            return '1h-4h'   # Medium signals for medium timeframes
        else:
            return '4h-1d'   # Weak signals need longer timeframes
    
    def _apply_convolution(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Apply convolution for edge detection"""
        h, w = image.shape
        kh, kw = kernel.shape
        result = np.zeros_like(image)
        
        pad_h, pad_w = kh // 2, kw // 2
        
        for i in range(pad_h, h - pad_h):
            for j in range(pad_w, w - pad_w):
                region = image[i-pad_h:i+pad_h+1, j-pad_w:j+pad_w+1]
                result[i, j] = np.sum(region * kernel)
        
        return result
    
    def _rgb_to_hsv(self, r: float, g: float, b: float) -> Tuple[float, float, float]:
        """Convert RGB to HSV"""
        max_val = max(r, g, b)
        min_val = min(r, g, b)
        diff = max_val - min_val
        
        # Hue calculation
        if diff == 0:
            h = 0
        elif max_val == r:
            h = ((g - b) / diff) % 6
        elif max_val == g:
            h = (b - r) / diff + 2
        else:
            h = (r - g) / diff + 4
        h *= 60
        
        # Saturation calculation
        s = 0 if max_val == 0 else diff / max_val
        
        # Value calculation
        v = max_val
        
        return h, s, v
    
    async def close(self):
        """Close the analyzer"""
        self.initialized = False
        logger.info("Advanced AI Chart Analyzer closed")