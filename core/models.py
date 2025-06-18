"""
Data models for the trading bot
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

class TradeType(Enum):
    """Trade direction enumeration"""
    LONG = "LONG"
    SHORT = "SHORT"

class TradeStatus(Enum):
    """Trade status enumeration"""
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    CANCELLED = "CANCELLED"

@dataclass
class Trade:
    """Trade data model"""
    user_id: int
    symbol: str
    trade_type: TradeType
    entry_price: float
    leverage: int
    capital: float
    status: TradeStatus = TradeStatus.OPEN
    id: Optional[int] = None
    exit_price: Optional[float] = None
    take_profit: Optional[float] = None
    stop_loss: Optional[float] = None
    tp_percentage: Optional[float] = None
    sl_percentage: Optional[float] = None
    pnl_usd: float = 0.0
    pnl_percentage: float = 0.0
    created_at: Optional[datetime] = None
    closed_at: Optional[datetime] = None
    notes: Optional[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()

@dataclass
class PriceData:
    """Price data model"""
    symbol: str
    price: float
    volume: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    high_24h: Optional[float] = None
    low_24h: Optional[float] = None
    change_24h: Optional[float] = None
    change_percentage_24h: Optional[float] = None

@dataclass
class TechnicalIndicator:
    """Technical indicator data model"""
    symbol: str
    timeframe: str
    rsi: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_histogram: Optional[float] = None
    ema_fast: Optional[float] = None
    ema_slow: Optional[float] = None
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    bollinger_upper: Optional[float] = None
    bollinger_lower: Optional[float] = None
    atr: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)

@dataclass
class TradingSignal:
    """Trading signal data model"""
    symbol: str
    action: str  # LONG, SHORT, CLOSE
    confidence: float  # 0-100
    reason: str
    entry_zone: Optional[str] = None
    take_profit: Optional[float] = None
    stop_loss: Optional[float] = None
    risk_reward_ratio: Optional[float] = None
    timeframe: str = "1h"
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            'symbol': self.symbol,
            'action': self.action,
            'confidence': self.confidence,
            'reason': self.reason,
            'entry_zone': self.entry_zone,
            'take_profit': self.take_profit,
            'stop_loss': self.stop_loss,
            'risk_reward_ratio': self.risk_reward_ratio,
            'timeframe': self.timeframe,
            'timestamp': self.timestamp.isoformat()
        }

@dataclass
class UserSettings:
    """User settings data model"""
    user_id: int
    default_leverage: int = 10
    risk_percentage: float = 2.0
    notifications_enabled: bool = True
    pnl_alert_threshold: float = 5.0
    auto_close_enabled: bool = False
    max_open_trades: int = 5
    preferred_timeframe: str = "1h"
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class MarketData:
    """Market data aggregation"""
    symbol: str
    price: float
    volume_24h: float
    price_change_24h: float
    price_change_percentage_24h: float
    high_24h: float
    low_24h: float
    market_dominance: Optional[float] = None
    market_cap: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)

@dataclass
class PortfolioSummary:
    """Portfolio summary data model"""
    user_id: int
    total_capital: float
    total_pnl_usd: float
    total_pnl_percentage: float
    open_trades_count: int
    total_trades_count: int
    win_rate: float
    avg_trade_duration: Optional[float] = None  # in hours
    best_trade_pnl: Optional[float] = None
    worst_trade_pnl: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
