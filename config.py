"""
Configuration settings for the trading bot
"""

import os
from typing import Optional

class Config:
    """Configuration class for bot settings"""
    
    def __init__(self):
        # Telegram Bot Configuration
        self.TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
        if not self.TELEGRAM_BOT_TOKEN:
            raise ValueError("TELEGRAM_BOT_TOKEN environment variable is required")
        
        # MEXC API Configuration
        self.MEXC_API_KEY: str = os.getenv("MEXC_API_KEY", "")
        self.MEXC_SECRET_KEY: str = os.getenv("MEXC_SECRET_KEY", "")
        
        # Database Configuration
        self.DATABASE_PATH: str = os.getenv("DATABASE_PATH", "trading_bot.db")
        
        # Trading Configuration
        self.DEFAULT_LEVERAGE: int = int(os.getenv("DEFAULT_LEVERAGE", "10"))
        self.MAX_LEVERAGE: int = int(os.getenv("MAX_LEVERAGE", "100"))
        self.MIN_CAPITAL: float = float(os.getenv("MIN_CAPITAL", "10.0"))
        self.MAX_CAPITAL: float = float(os.getenv("MAX_CAPITAL", "10000.0"))
        
        # Risk Management
        self.DEFAULT_TP_PERCENTAGE: float = float(os.getenv("DEFAULT_TP_PERCENTAGE", "3.0"))
        self.DEFAULT_SL_PERCENTAGE: float = float(os.getenv("DEFAULT_SL_PERCENTAGE", "2.0"))
        self.MAX_OPEN_TRADES: int = int(os.getenv("MAX_OPEN_TRADES", "5"))
        
        # Signal Detection
        self.SIGNAL_CONFIDENCE_THRESHOLD: float = float(os.getenv("SIGNAL_CONFIDENCE_THRESHOLD", "70.0"))
        self.HIGH_CONFIDENCE_THRESHOLD: float = float(os.getenv("HIGH_CONFIDENCE_THRESHOLD", "85.0"))
        self.ULTRA_HIGH_CONFIDENCE_THRESHOLD: float = float(os.getenv("ULTRA_HIGH_CONFIDENCE_THRESHOLD", "90.0"))
        
        # Technical Indicators
        self.RSI_PERIOD: int = int(os.getenv("RSI_PERIOD", "14"))
        self.RSI_OVERSOLD: float = float(os.getenv("RSI_OVERSOLD", "30.0"))
        self.RSI_OVERBOUGHT: float = float(os.getenv("RSI_OVERBOUGHT", "70.0"))
        
        self.MACD_FAST: int = int(os.getenv("MACD_FAST", "12"))
        self.MACD_SLOW: int = int(os.getenv("MACD_SLOW", "26"))
        self.MACD_SIGNAL: int = int(os.getenv("MACD_SIGNAL", "9"))
        
        self.EMA_FAST: int = int(os.getenv("EMA_FAST", "9"))
        self.EMA_SLOW: int = int(os.getenv("EMA_SLOW", "21"))
        
        # Chart Configuration
        self.CHART_WIDTH: int = int(os.getenv("CHART_WIDTH", "800"))
        self.CHART_HEIGHT: int = int(os.getenv("CHART_HEIGHT", "600"))
        self.CHART_CANDLES: int = int(os.getenv("CHART_CANDLES", "100"))
        
        # API Rate Limits
        self.API_RATE_LIMIT: int = int(os.getenv("API_RATE_LIMIT", "10"))  # requests per second
        self.API_TIMEOUT: int = int(os.getenv("API_TIMEOUT", "30"))  # seconds
        
        # Logging
        self.LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
        self.LOG_FILE: Optional[str] = os.getenv("LOG_FILE", "trading_bot.log")
        
        # Notification Settings
        self.ENABLE_PNL_ALERTS: bool = os.getenv("ENABLE_PNL_ALERTS", "true").lower() == "true"
        self.PNL_ALERT_INTERVAL: int = int(os.getenv("PNL_ALERT_INTERVAL", "300"))  # seconds
        self.SIGNIFICANT_PNL_THRESHOLD: float = float(os.getenv("SIGNIFICANT_PNL_THRESHOLD", "5.0"))  # percentage
        
        # Data Retention
        self.KEEP_CLOSED_TRADES_DAYS: int = int(os.getenv("KEEP_CLOSED_TRADES_DAYS", "30"))
        self.KEEP_PRICE_DATA_DAYS: int = int(os.getenv("KEEP_PRICE_DATA_DAYS", "7"))
        
    def validate(self) -> bool:
        """Validate configuration settings"""
        errors = []
        
        if not self.TELEGRAM_BOT_TOKEN:
            errors.append("TELEGRAM_BOT_TOKEN is required")
        
        if self.DEFAULT_LEVERAGE < 1 or self.DEFAULT_LEVERAGE > self.MAX_LEVERAGE:
            errors.append(f"DEFAULT_LEVERAGE must be between 1 and {self.MAX_LEVERAGE}")
        
        if self.MIN_CAPITAL <= 0 or self.MIN_CAPITAL >= self.MAX_CAPITAL:
            errors.append("MIN_CAPITAL must be positive and less than MAX_CAPITAL")
        
        if self.DEFAULT_TP_PERCENTAGE <= 0 or self.DEFAULT_SL_PERCENTAGE <= 0:
            errors.append("TP and SL percentages must be positive")
        
        if errors:
            for error in errors:
                print(f"Configuration error: {error}")
            return False
        
        return True

# Global config instance
config = Config()
