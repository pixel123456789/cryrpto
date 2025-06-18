"""
Technical indicators calculation
"""

import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

class TechnicalIndicators:
    """Technical indicators calculation class"""
    
    @staticmethod
    def calculate_rsi(prices: List[float], period: int = 14) -> Optional[float]:
        """Calculate RSI (Relative Strength Index)"""
        try:
            if len(prices) < period + 1:
                return None
            
            df = pd.DataFrame({'price': prices})
            delta = df['price'].diff()
            
            gains = delta.where(delta > 0, 0)
            losses = -delta.where(delta < 0, 0)
            
            avg_gains = gains.rolling(window=period).mean()
            avg_losses = losses.rolling(window=period).mean()
            
            rs = avg_gains / avg_losses
            rsi = 100 - (100 / (1 + rs))
            
            return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else None
            
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            return None
    
    @staticmethod
    def calculate_macd(prices: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, Optional[float]]:
        """Calculate MACD (Moving Average Convergence Divergence)"""
        try:
            if len(prices) < slow + signal:
                return {'macd': None, 'signal': None, 'histogram': None}
            
            df = pd.DataFrame({'price': prices})
            
            # Calculate EMAs
            ema_fast = df['price'].ewm(span=fast).mean()
            ema_slow = df['price'].ewm(span=slow).mean()
            
            # Calculate MACD
            macd = ema_fast - ema_slow
            macd_signal = macd.ewm(span=signal).mean()
            macd_histogram = macd - macd_signal
            
            return {
                'macd': float(macd.iloc[-1]) if not pd.isna(macd.iloc[-1]) else None,
                'signal': float(macd_signal.iloc[-1]) if not pd.isna(macd_signal.iloc[-1]) else None,
                'histogram': float(macd_histogram.iloc[-1]) if not pd.isna(macd_histogram.iloc[-1]) else None
            }
            
        except Exception as e:
            logger.error(f"Error calculating MACD: {e}")
            return {'macd': None, 'signal': None, 'histogram': None}
    
    @staticmethod
    def calculate_ema(prices: List[float], period: int) -> Optional[float]:
        """Calculate Exponential Moving Average"""
        try:
            if len(prices) < period:
                return None
            
            df = pd.DataFrame({'price': prices})
            ema = df['price'].ewm(span=period).mean()
            
            return float(ema.iloc[-1]) if not pd.isna(ema.iloc[-1]) else None
            
        except Exception as e:
            logger.error(f"Error calculating EMA: {e}")
            return None
    
    @staticmethod
    def calculate_sma(prices: List[float], period: int) -> Optional[float]:
        """Calculate Simple Moving Average"""
        try:
            if len(prices) < period:
                return None
            
            return sum(prices[-period:]) / period
            
        except Exception as e:
            logger.error(f"Error calculating SMA: {e}")
            return None
    
    @staticmethod
    def calculate_bollinger_bands(prices: List[float], period: int = 20, std_dev: float = 2) -> Dict[str, Optional[float]]:
        """Calculate Bollinger Bands"""
        try:
            if len(prices) < period:
                return {'upper': None, 'middle': None, 'lower': None}
            
            df = pd.DataFrame({'price': prices})
            sma = df['price'].rolling(window=period).mean()
            std = df['price'].rolling(window=period).std()
            
            upper = sma + (std * std_dev)
            lower = sma - (std * std_dev)
            
            return {
                'upper': float(upper.iloc[-1]) if not pd.isna(upper.iloc[-1]) else None,
                'middle': float(sma.iloc[-1]) if not pd.isna(sma.iloc[-1]) else None,
                'lower': float(lower.iloc[-1]) if not pd.isna(lower.iloc[-1]) else None
            }
            
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {e}")
            return {'upper': None, 'middle': None, 'lower': None}
    
    @staticmethod
    def calculate_atr(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> Optional[float]:
        """Calculate Average True Range"""
        try:
            if len(highs) < period + 1 or len(lows) < period + 1 or len(closes) < period + 1:
                return None
            
            df = pd.DataFrame({
                'high': highs,
                'low': lows,
                'close': closes
            })
            
            # Calculate True Range
            tr1 = df['high'] - df['low']
            tr2 = abs(df['high'] - df['close'].shift())
            tr3 = abs(df['low'] - df['close'].shift())
            
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_range.rolling(window=period).mean()
            
            return float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else None
            
        except Exception as e:
            logger.error(f"Error calculating ATR: {e}")
            return None
    
    @staticmethod
    def calculate_stochastic(highs: List[float], lows: List[float], closes: List[float], 
                           k_period: int = 14, d_period: int = 3) -> Dict[str, Optional[float]]:
        """Calculate Stochastic Oscillator"""
        try:
            if len(highs) < k_period or len(lows) < k_period or len(closes) < k_period:
                return {'%k': None, '%d': None}
            
            df = pd.DataFrame({
                'high': highs,
                'low': lows,
                'close': closes
            })
            
            # Calculate %K
            lowest_low = df['low'].rolling(window=k_period).min()
            highest_high = df['high'].rolling(window=k_period).max()
            
            k_percent = 100 * ((df['close'] - lowest_low) / (highest_high - lowest_low))
            d_percent = k_percent.rolling(window=d_period).mean()
            
            return {
                '%k': float(k_percent.iloc[-1]) if not pd.isna(k_percent.iloc[-1]) else None,
                '%d': float(d_percent.iloc[-1]) if not pd.isna(d_percent.iloc[-1]) else None
            }
            
        except Exception as e:
            logger.error(f"Error calculating Stochastic: {e}")
            return {'%k': None, '%d': None}
    
    @staticmethod
    def calculate_volume_indicators(volumes: List[float], prices: List[float]) -> Dict[str, Any]:
        """Calculate volume-based indicators"""
        try:
            if len(volumes) < 20 or len(prices) < 20:
                return {'volume_sma': None, 'volume_ratio': None, 'obv': None}
            
            df = pd.DataFrame({
                'volume': volumes,
                'price': prices
            })
            
            # Volume SMA
            volume_sma = df['volume'].rolling(window=20).mean()
            
            # Volume ratio (current vs average)
            volume_ratio = df['volume'].iloc[-1] / volume_sma.iloc[-1] if volume_sma.iloc[-1] > 0 else None
            
            # On Balance Volume (OBV)
            price_change = df['price'].diff()
            obv_values = []
            obv = 0
            
            for i, (price_diff, volume) in enumerate(zip(price_change, volumes)):
                if i == 0:
                    obv_values.append(0)
                    continue
                
                if price_diff > 0:
                    obv += volume
                elif price_diff < 0:
                    obv -= volume
                
                obv_values.append(obv)
            
            return {
                'volume_sma': float(volume_sma.iloc[-1]) if not pd.isna(volume_sma.iloc[-1]) else None,
                'volume_ratio': float(volume_ratio) if volume_ratio else None,
                'obv': float(obv_values[-1]) if obv_values else None
            }
            
        except Exception as e:
            logger.error(f"Error calculating volume indicators: {e}")
            return {'volume_sma': None, 'volume_ratio': None, 'obv': None}
    
    @staticmethod
    def detect_patterns(highs: List[float], lows: List[float], closes: List[float]) -> List[str]:
        """Detect basic chart patterns"""
        patterns = []
        
        try:
            if len(closes) < 20:
                return patterns
            
            recent_closes = closes[-20:]
            recent_highs = highs[-20:]
            recent_lows = lows[-20:]
            
            # Double top pattern
            if TechnicalIndicators._detect_double_top(recent_highs):
                patterns.append("Double Top")
            
            # Double bottom pattern
            if TechnicalIndicators._detect_double_bottom(recent_lows):
                patterns.append("Double Bottom")
            
            # Rising wedge
            if TechnicalIndicators._detect_rising_wedge(recent_highs, recent_lows):
                patterns.append("Rising Wedge")
            
            # Falling wedge
            if TechnicalIndicators._detect_falling_wedge(recent_highs, recent_lows):
                patterns.append("Falling Wedge")
            
            # Breakout detection
            if TechnicalIndicators._detect_breakout(recent_closes):
                patterns.append("Breakout")
            
        except Exception as e:
            logger.error(f"Error detecting patterns: {e}")
        
        return patterns
    
    @staticmethod
    def _detect_double_top(highs: List[float]) -> bool:
        """Detect double top pattern"""
        if len(highs) < 10:
            return False
        
        # Find local maxima
        maxima = []
        for i in range(1, len(highs) - 1):
            if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
                maxima.append((i, highs[i]))
        
        if len(maxima) < 2:
            return False
        
        # Check if last two maxima are similar in height
        last_two = maxima[-2:]
        height_diff = abs(last_two[0][1] - last_two[1][1]) / max(last_two[0][1], last_two[1][1])
        
        return height_diff < 0.02  # Within 2%
    
    @staticmethod
    def _detect_double_bottom(lows: List[float]) -> bool:
        """Detect double bottom pattern"""
        if len(lows) < 10:
            return False
        
        # Find local minima
        minima = []
        for i in range(1, len(lows) - 1):
            if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
                minima.append((i, lows[i]))
        
        if len(minima) < 2:
            return False
        
        # Check if last two minima are similar in depth
        last_two = minima[-2:]
        depth_diff = abs(last_two[0][1] - last_two[1][1]) / max(last_two[0][1], last_two[1][1])
        
        return depth_diff < 0.02  # Within 2%
    
    @staticmethod
    def _detect_rising_wedge(highs: List[float], lows: List[float]) -> bool:
        """Detect rising wedge pattern"""
        if len(highs) < 10 or len(lows) < 10:
            return False
        
        # Check if highs are making higher highs but at a decreasing rate
        # and lows are making higher lows
        recent_highs = highs[-10:]
        recent_lows = lows[-10:]
        
        # Simple trend check
        high_trend = recent_highs[-1] > recent_highs[0]
        low_trend = recent_lows[-1] > recent_lows[0]
        
        # Check if the range is narrowing
        early_range = max(recent_highs[:5]) - min(recent_lows[:5])
        late_range = max(recent_highs[-5:]) - min(recent_lows[-5:])
        
        return high_trend and low_trend and late_range < early_range
    
    @staticmethod
    def _detect_falling_wedge(highs: List[float], lows: List[float]) -> bool:
        """Detect falling wedge pattern"""
        if len(highs) < 10 or len(lows) < 10:
            return False
        
        recent_highs = highs[-10:]
        recent_lows = lows[-10:]
        
        # Check if both highs and lows are declining but converging
        high_trend = recent_highs[-1] < recent_highs[0]
        low_trend = recent_lows[-1] < recent_lows[0]
        
        # Check if the range is narrowing
        early_range = max(recent_highs[:5]) - min(recent_lows[:5])
        late_range = max(recent_highs[-5:]) - min(recent_lows[-5:])
        
        return high_trend and low_trend and late_range < early_range
    
    @staticmethod
    def _detect_breakout(closes: List[float]) -> bool:
        """Detect breakout from consolidation"""
        if len(closes) < 15:
            return False
        
        # Check for consolidation followed by strong move
        consolidation = closes[-15:-5]
        recent = closes[-5:]
        
        # Calculate consolidation range
        cons_high = max(consolidation)
        cons_low = min(consolidation)
        cons_range = cons_high - cons_low
        
        # Check if recent price moved significantly outside consolidation
        recent_high = max(recent)
        recent_low = min(recent)
        
        breakout_up = recent_high > cons_high + (cons_range * 0.5)
        breakout_down = recent_low < cons_low - (cons_range * 0.5)
        
        return breakout_up or breakout_down
