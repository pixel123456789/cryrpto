"""
Chart generation for trading analysis
"""

import asyncio
import logging
import io
import base64
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from PIL import Image

from core.mexc_api import MEXCClient
from signals.indicators import TechnicalIndicators
from config import Config

logger = logging.getLogger(__name__)

class ChartGenerator:
    """Generate trading charts with technical indicators"""
    
    def __init__(self, mexc_client: MEXCClient):
        self.mexc_client = mexc_client
        self.config = Config()
        self.indicators = TechnicalIndicators()
        
        # Set matplotlib style
        plt.style.use('dark_background')
        
    async def generate_chart(self, symbol: str, timeframe: str = "1h", 
                           include_indicators: bool = True) -> Optional[bytes]:
        """Generate a complete trading chart"""
        try:
            # Get candlestick data
            kline_data = await self.mexc_client.get_kline_data(
                symbol, timeframe, self.config.CHART_CANDLES
            )
            
            if len(kline_data) < 20:
                logger.warning(f"Insufficient data for {symbol} chart")
                return None
            
            # Create the chart
            chart_bytes = await self._create_candlestick_chart(
                symbol, kline_data, timeframe, include_indicators
            )
            
            return chart_bytes
            
        except Exception as e:
            logger.error(f"Error generating chart for {symbol}: {e}")
            return None
    
    async def generate_chart_with_trades(self, symbol: str, trades: List[Dict], 
                                       timeframe: str = "1h") -> Optional[bytes]:
        """Generate chart with trade markers"""
        try:
            # Get candlestick data
            kline_data = await self.mexc_client.get_kline_data(symbol, timeframe, 200)
            
            if len(kline_data) < 20:
                return None
            
            # Create chart with trade overlays
            chart_bytes = await self._create_chart_with_trades(
                symbol, kline_data, trades, timeframe
            )
            
            return chart_bytes
            
        except Exception as e:
            logger.error(f"Error generating chart with trades for {symbol}: {e}")
            return None
    
    async def _create_candlestick_chart(self, symbol: str, kline_data: List[Dict], 
                                      timeframe: str, include_indicators: bool) -> bytes:
        """Create a candlestick chart with indicators"""
        # Prepare data
        df = pd.DataFrame(kline_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Create figure with subplots
        if include_indicators:
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), 
                                              gridspec_kw={'height_ratios': [3, 1, 1]})
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))
        
        # Main price chart
        await self._plot_candlesticks(ax1, df)
        
        if include_indicators:
            # Add moving averages
            await self._add_moving_averages(ax1, df)
            
            # Add Bollinger Bands
            await self._add_bollinger_bands(ax1, df)
            
            # RSI subplot
            await self._plot_rsi(ax2, df)
            
            # MACD subplot
            await self._plot_macd(ax3, df)
        
        # Format chart
        self._format_chart(fig, ax1, symbol, timeframe)
        
        if include_indicators:
            self._format_indicator_subplots(ax2, ax3)
        
        # Convert to bytes
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight',
                   facecolor='#1e1e1e', edgecolor='none')
        plt.close(fig)
        
        img_buffer.seek(0)
        return img_buffer.getvalue()
    
    async def _plot_candlesticks(self, ax, df: pd.DataFrame):
        """Plot candlestick chart"""
        # Calculate colors
        colors = ['#00ff88' if close >= open_price else '#ff4444' 
                 for close, open_price in zip(df['close'], df['open'])]
        
        # Plot candlesticks
        for i, (timestamp, row) in enumerate(df.iterrows()):
            color = colors[i]
            
            # Candlestick body
            body_height = abs(row['close'] - row['open'])
            body_bottom = min(row['close'], row['open'])
            
            rect = Rectangle((mdates.date2num(timestamp) - 0.3, body_bottom),
                           0.6, body_height, facecolor=color, edgecolor=color)
            ax.add_patch(rect)
            
            # Wicks
            ax.vlines(mdates.date2num(timestamp), row['low'], row['high'], 
                     colors=color, alpha=0.8, linewidth=1)
        
        # Set y-axis to price data
        ax.set_ylim(df['low'].min() * 0.995, df['high'].max() * 1.005)
    
    async def _add_moving_averages(self, ax, df: pd.DataFrame):
        """Add moving averages to the chart"""
        # Calculate EMAs
        ema_9 = df['close'].ewm(span=9).mean()
        ema_21 = df['close'].ewm(span=21).mean()
        sma_50 = df['close'].rolling(window=50).mean()
        
        # Plot EMAs
        ax.plot(df.index, ema_9, color='#ffaa00', linewidth=1, label='EMA 9', alpha=0.8)
        ax.plot(df.index, ema_21, color='#00aaff', linewidth=1, label='EMA 21', alpha=0.8)
        ax.plot(df.index, sma_50, color='#aa00ff', linewidth=1, label='SMA 50', alpha=0.8)
        
        ax.legend(loc='upper left', fontsize=8)
    
    async def _add_bollinger_bands(self, ax, df: pd.DataFrame):
        """Add Bollinger Bands to the chart"""
        # Calculate Bollinger Bands
        sma_20 = df['close'].rolling(window=20).mean()
        std_20 = df['close'].rolling(window=20).std()
        
        upper_band = sma_20 + (std_20 * 2)
        lower_band = sma_20 - (std_20 * 2)
        
        # Plot bands
        ax.plot(df.index, upper_band, color='#666666', linewidth=1, alpha=0.6)
        ax.plot(df.index, lower_band, color='#666666', linewidth=1, alpha=0.6)
        ax.fill_between(df.index, upper_band, lower_band, alpha=0.1, color='#666666')
    
    async def _plot_rsi(self, ax, df: pd.DataFrame):
        """Plot RSI indicator"""
        # Calculate RSI
        delta = df['close'].diff()
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        
        avg_gains = gains.rolling(window=14).mean()
        avg_losses = losses.rolling(window=14).mean()
        
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        
        # Plot RSI
        ax.plot(df.index, rsi, color='#ffaa00', linewidth=1)
        ax.axhline(y=70, color='#ff4444', linestyle='--', alpha=0.7, linewidth=0.8)
        ax.axhline(y=30, color='#00ff88', linestyle='--', alpha=0.7, linewidth=0.8)
        ax.axhline(y=50, color='#666666', linestyle='-', alpha=0.5, linewidth=0.8)
        
        ax.set_ylim(0, 100)
        ax.set_ylabel('RSI', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    async def _plot_macd(self, ax, df: pd.DataFrame):
        """Plot MACD indicator"""
        # Calculate MACD
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        
        macd = ema_12 - ema_26
        macd_signal = macd.ewm(span=9).mean()
        macd_histogram = macd - macd_signal
        
        # Plot MACD
        ax.plot(df.index, macd, color='#00aaff', linewidth=1, label='MACD')
        ax.plot(df.index, macd_signal, color='#ff4444', linewidth=1, label='Signal')
        
        # Plot histogram
        colors = ['#00ff88' if val >= 0 else '#ff4444' for val in macd_histogram]
        ax.bar(df.index, macd_histogram, color=colors, alpha=0.7, width=0.8)
        
        ax.axhline(y=0, color='#666666', linestyle='-', alpha=0.5, linewidth=0.8)
        ax.set_ylabel('MACD', fontsize=10)
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    async def _create_chart_with_trades(self, symbol: str, kline_data: List[Dict], 
                                      trades: List[Dict], timeframe: str) -> bytes:
        """Create chart with trade entry/exit markers"""
        # Create base chart
        chart_bytes = await self._create_candlestick_chart(symbol, kline_data, timeframe, True)
        
        # Note: For now, return base chart. Trade markers can be added in future iterations
        # This would require additional logic to overlay trade markers on the chart
        return chart_bytes
    
    def _format_chart(self, fig, ax, symbol: str, timeframe: str):
        """Format the main chart appearance"""
        # Title
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M UTC')
        ax.set_title(f'{symbol} - {timeframe} | {current_time}', 
                    fontsize=14, fontweight='bold', color='white', pad=20)
        
        # X-axis formatting
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # Grid and styling
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('#1e1e1e')
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#666666')
        ax.spines['bottom'].set_color('#666666')
        
        # Tick colors
        ax.tick_params(colors='white', which='both')
        ax.yaxis.label.set_color('white')
        ax.xaxis.label.set_color('white')
    
    def _format_indicator_subplots(self, ax2, ax3):
        """Format indicator subplots"""
        for ax in [ax2, ax3]:
            ax.set_facecolor('#1e1e1e')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#666666')
            ax.spines['bottom'].set_color('#666666')
            ax.tick_params(colors='white', which='both')
            ax.yaxis.label.set_color('white')
            ax.xaxis.label.set_color('white')
        
        # Format x-axis for bottom subplot only
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        ax3.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
        
        # Hide x-axis labels for middle subplot
        ax2.set_xticklabels([])
    
    async def generate_mini_chart(self, symbol: str, width: int = 400, height: int = 200) -> Optional[bytes]:
        """Generate a mini chart for quick preview"""
        try:
            # Get limited data for mini chart
            kline_data = await self.mexc_client.get_kline_data(symbol, "1h", 50)
            
            if len(kline_data) < 10:
                return None
            
            # Create simple line chart
            df = pd.DataFrame(kline_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            fig, ax = plt.subplots(1, 1, figsize=(width/100, height/100))
            
            # Simple line chart
            ax.plot(df['timestamp'], df['close'], color='#00aaff', linewidth=2)
            
            # Minimal formatting
            ax.set_facecolor('#1e1e1e')
            ax.grid(True, alpha=0.2)
            ax.set_title(symbol, color='white', fontsize=12)
            
            # Remove axes for clean look
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            
            # Convert to bytes
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=80, bbox_inches='tight',
                       facecolor='#1e1e1e', edgecolor='none')
            plt.close(fig)
            
            img_buffer.seek(0)
            return img_buffer.getvalue()
            
        except Exception as e:
            logger.error(f"Error generating mini chart for {symbol}: {e}")
            return None
    
    async def generate_heatmap(self, symbols: List[str]) -> Optional[bytes]:
        """Generate a market heatmap"""
        try:
            # Get ticker data for all symbols
            tickers = []
            for symbol in symbols[:20]:  # Limit to 20 symbols
                ticker = await self.mexc_client.get_ticker(symbol)
                if ticker:
                    tickers.append(ticker)
            
            if len(tickers) < 5:
                return None
            
            # Create heatmap data
            symbols_list = [t['symbol'] for t in tickers]
            changes = [t['change_24h'] for t in tickers]
            
            # Create heatmap
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            
            # Create grid layout
            cols = 5
            rows = (len(tickers) + cols - 1) // cols
            
            for i, (symbol, change) in enumerate(zip(symbols_list, changes)):
                row = i // cols
                col = i % cols
                
                # Color based on change
                if change > 0:
                    color = '#00ff88'
                elif change < 0:
                    color = '#ff4444'
                else:
                    color = '#666666'
                
                # Create rectangle
                rect = Rectangle((col, rows - row - 1), 1, 1, 
                               facecolor=color, alpha=abs(change) / 10 + 0.3)
                ax.add_patch(rect)
                
                # Add text
                ax.text(col + 0.5, rows - row - 0.5, f'{symbol}\n{change:+.1f}%',
                       ha='center', va='center', fontsize=8, color='white',
                       fontweight='bold')
            
            ax.set_xlim(0, cols)
            ax.set_ylim(0, rows)
            ax.set_aspect('equal')
            ax.set_facecolor('#1e1e1e')
            ax.set_title('Market Heatmap - 24h Change', color='white', 
                        fontsize=14, fontweight='bold', pad=20)
            
            # Remove axes
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            
            # Convert to bytes
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight',
                       facecolor='#1e1e1e', edgecolor='none')
            plt.close(fig)
            
            img_buffer.seek(0)
            return img_buffer.getvalue()
            
        except Exception as e:
            logger.error(f"Error generating heatmap: {e}")
            return None

