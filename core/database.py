"""
Database management for the trading bot
"""

import sqlite3
import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

from core.models import Trade, TradeStatus, TradeType

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages SQLite database operations for the trading bot"""
    
    def __init__(self, db_path: str = "trading_bot.db"):
        self.db_path = db_path
        self._lock = asyncio.Lock()
        
    async def initialize(self):
        """Initialize database tables"""
        async with self._lock:
            await self._execute_query("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    trade_type TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL,
                    leverage INTEGER NOT NULL,
                    capital REAL NOT NULL,
                    take_profit REAL,
                    stop_loss REAL,
                    tp_percentage REAL,
                    sl_percentage REAL,
                    status TEXT NOT NULL,
                    pnl_usd REAL DEFAULT 0,
                    pnl_percentage REAL DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    closed_at TIMESTAMP,
                    notes TEXT
                )
            """)
            
            await self._execute_query("""
                CREATE TABLE IF NOT EXISTS price_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    price REAL NOT NULL,
                    volume REAL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            await self._execute_query("""
                CREATE TABLE IF NOT EXISTS user_settings (
                    user_id INTEGER PRIMARY KEY,
                    default_leverage INTEGER DEFAULT 10,
                    risk_percentage REAL DEFAULT 2.0,
                    notifications_enabled BOOLEAN DEFAULT 1,
                    pnl_alert_threshold REAL DEFAULT 5.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            await self._execute_query("""
                CREATE TABLE IF NOT EXISTS signal_tracking (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    signal_type TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL,
                    result TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP NOT NULL,
                    closed_at TIMESTAMP,
                    pnl_percentage REAL,
                    is_winner BOOLEAN
                )
            """)
            
            await self._execute_query("""
                CREATE TABLE IF NOT EXISTS signal_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    total_signals INTEGER DEFAULT 0,
                    winning_signals INTEGER DEFAULT 0,
                    losing_signals INTEGER DEFAULT 0,
                    win_rate REAL DEFAULT 0.0,
                    avg_pnl REAL DEFAULT 0.0,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indices for better performance
            await self._execute_query("CREATE INDEX IF NOT EXISTS idx_trades_user_id ON trades(user_id)")
            await self._execute_query("CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status)")
            await self._execute_query("CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)")
            await self._execute_query("CREATE INDEX IF NOT EXISTS idx_price_data_symbol ON price_data(symbol)")
            
            logger.info("Database initialized successfully")
    
    async def create_trade(self, trade: Trade) -> int:
        """Create a new trade record"""
        query = """
            INSERT INTO trades (
                user_id, symbol, trade_type, entry_price, leverage, capital,
                take_profit, stop_loss, tp_percentage, sl_percentage, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        params = (
            trade.user_id, trade.symbol, trade.trade_type.value, trade.entry_price,
            trade.leverage, trade.capital, trade.take_profit, trade.stop_loss,
            trade.tp_percentage, trade.sl_percentage, trade.status.value
        )
        
        trade_id = await self._execute_query(query, params, fetch_id=True)
        logger.info(f"Created trade {trade_id} for user {trade.user_id}")
        return trade_id
    
    async def update_trade(self, trade: Trade):
        """Update an existing trade record"""
        query = """
            UPDATE trades SET
                exit_price = ?, status = ?, pnl_usd = ?, pnl_percentage = ?,
                closed_at = ?, take_profit = ?, stop_loss = ?, tp_percentage = ?, sl_percentage = ?
            WHERE id = ?
        """
        
        params = (
            trade.exit_price, trade.status.value, trade.pnl_usd, trade.pnl_percentage,
            trade.closed_at, trade.take_profit, trade.stop_loss,
            trade.tp_percentage, trade.sl_percentage, trade.id
        )
        
        await self._execute_query(query, params)
        logger.info(f"Updated trade {trade.id}")
    
    async def get_trade(self, trade_id: int) -> Optional[Trade]:
        """Get a specific trade by ID"""
        query = "SELECT * FROM trades WHERE id = ?"
        result = await self._execute_query(query, (trade_id,), fetch_one=True)
        
        if result:
            return self._row_to_trade(result)
        return None
    
    async def get_user_trades(self, user_id: int, status: Optional[TradeStatus] = None, limit: int = 50) -> List[Trade]:
        """Get trades for a specific user"""
        if status:
            query = "SELECT * FROM trades WHERE user_id = ? AND status = ? ORDER BY created_at DESC LIMIT ?"
            params = (user_id, status.value, limit)
        else:
            query = "SELECT * FROM trades WHERE user_id = ? ORDER BY created_at DESC LIMIT ?"
            params = (user_id, limit)
        
        results = await self._execute_query(query, params, fetch_all=True)
        return [self._row_to_trade(row) for row in results]
    
    async def get_all_open_trades(self) -> List[Trade]:
        """Get all open trades across all users"""
        query = "SELECT * FROM trades WHERE status = ? ORDER BY created_at DESC"
        results = await self._execute_query(query, (TradeStatus.OPEN.value,), fetch_all=True)
        return [self._row_to_trade(row) for row in results]
    
    async def close_trade(self, trade_id: int, exit_price: float, pnl_usd: float, pnl_percentage: float):
        """Close a trade with final PnL"""
        query = """
            UPDATE trades SET
                exit_price = ?, status = ?, pnl_usd = ?, pnl_percentage = ?, closed_at = ?
            WHERE id = ?
        """
        
        params = (exit_price, TradeStatus.CLOSED.value, pnl_usd, pnl_percentage, datetime.utcnow(), trade_id)
        await self._execute_query(query, params)
        logger.info(f"Closed trade {trade_id} with PnL: {pnl_percentage:.2f}%")
    
    async def store_price_data(self, symbol: str, price: float, volume: Optional[float] = None):
        """Store price data for historical analysis"""
        query = "INSERT INTO price_data (symbol, price, volume) VALUES (?, ?, ?)"
        await self._execute_query(query, (symbol, price, volume))
    
    async def get_price_history(self, symbol: str, hours: int = 24) -> List[Dict[str, Any]]:
        """Get price history for a symbol"""
        since = datetime.utcnow() - timedelta(hours=hours)
        query = """
            SELECT price, volume, timestamp FROM price_data 
            WHERE symbol = ? AND timestamp >= ? 
            ORDER BY timestamp ASC
        """
        
        results = await self._execute_query(query, (symbol, since), fetch_all=True)
        return [
            {
                'price': row[0],
                'volume': row[1],
                'timestamp': row[2]
            }
            for row in results
        ]
    
    async def get_user_settings(self, user_id: int) -> Dict[str, Any]:
        """Get user settings"""
        query = "SELECT * FROM user_settings WHERE user_id = ?"
        result = await self._execute_query(query, (user_id,), fetch_one=True)
        
        if result:
            return {
                'user_id': result[0],
                'default_leverage': result[1],
                'risk_percentage': result[2],
                'notifications_enabled': bool(result[3]),
                'pnl_alert_threshold': result[4]
            }
        else:
            # Return default settings
            return {
                'user_id': user_id,
                'default_leverage': 10,
                'risk_percentage': 2.0,
                'notifications_enabled': True,
                'pnl_alert_threshold': 5.0
            }
    
    async def update_user_settings(self, user_id: int, settings: Dict[str, Any]):
        """Update user settings"""
        query = """
            INSERT OR REPLACE INTO user_settings 
            (user_id, default_leverage, risk_percentage, notifications_enabled, pnl_alert_threshold, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """
        
        params = (
            user_id,
            settings.get('default_leverage', 10),
            settings.get('risk_percentage', 2.0),
            settings.get('notifications_enabled', True),
            settings.get('pnl_alert_threshold', 5.0),
            datetime.utcnow()
        )
        
        await self._execute_query(query, params)
        logger.info(f"Updated settings for user {user_id}")
    
    async def cleanup_old_data(self, days: int = 30):
        """Clean up old closed trades and price data"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        # Clean old closed trades
        query1 = "DELETE FROM trades WHERE status = ? AND closed_at < ?"
        await self._execute_query(query1, (TradeStatus.CLOSED.value, cutoff_date))
        
        # Clean old price data (keep only last 7 days)
        price_cutoff = datetime.utcnow() - timedelta(days=7)
        query2 = "DELETE FROM price_data WHERE timestamp < ?"
        await self._execute_query(query2, (price_cutoff,))
        
        logger.info(f"Cleaned up data older than {days} days")
    
    async def get_trading_stats(self, user_id: int) -> Dict[str, Any]:
        """Get trading statistics for a user"""
        queries = {
            'total_trades': "SELECT COUNT(*) FROM trades WHERE user_id = ?",
            'open_trades': "SELECT COUNT(*) FROM trades WHERE user_id = ? AND status = ?",
            'closed_trades': "SELECT COUNT(*) FROM trades WHERE user_id = ? AND status = ?",
            'winning_trades': "SELECT COUNT(*) FROM trades WHERE user_id = ? AND status = ? AND pnl_percentage > 0",
            'total_pnl': "SELECT SUM(pnl_usd) FROM trades WHERE user_id = ? AND status = ?",
            'avg_pnl': "SELECT AVG(pnl_percentage) FROM trades WHERE user_id = ? AND status = ?"
        }
        
        stats = {}
        
        # Total trades
        result = await self._execute_query(queries['total_trades'], (user_id,), fetch_one=True)
        stats['total_trades'] = result[0] if result else 0
        
        # Open trades
        result = await self._execute_query(queries['open_trades'], (user_id, TradeStatus.OPEN.value), fetch_one=True)
        stats['open_trades'] = result[0] if result else 0
        
        # Closed trades
        result = await self._execute_query(queries['closed_trades'], (user_id, TradeStatus.CLOSED.value), fetch_one=True)
        stats['closed_trades'] = result[0] if result else 0
        
        # Winning trades
        result = await self._execute_query(queries['winning_trades'], (user_id, TradeStatus.CLOSED.value), fetch_one=True)
        stats['winning_trades'] = result[0] if result else 0
        
        # Total PnL
        result = await self._execute_query(queries['total_pnl'], (user_id, TradeStatus.CLOSED.value), fetch_one=True)
        stats['total_pnl_usd'] = result[0] if result and result[0] else 0
        
        # Average PnL
        result = await self._execute_query(queries['avg_pnl'], (user_id, TradeStatus.CLOSED.value), fetch_one=True)
        stats['avg_pnl_percentage'] = result[0] if result and result[0] else 0
        
        # Calculate win rate
        if stats['closed_trades'] > 0:
            stats['win_rate'] = (stats['winning_trades'] / stats['closed_trades']) * 100
        else:
            stats['win_rate'] = 0
        
        return stats
    
    async def has_active_signal(self, symbol: str) -> bool:
        """Check if symbol already has an active signal"""
        # Use UTC now to match the Python datetime.utcnow() used in creation
        current_time = datetime.utcnow()
        
        query = """
            SELECT COUNT(*) FROM signal_tracking 
            WHERE symbol = ? AND expires_at > ? AND closed_at IS NULL
        """
        
        result = await self._execute_query(query, (symbol, current_time), fetch_one=True)
        has_active = result[0] > 0 if result else False
        
        if has_active:
            logger.debug(f"Found active signal for {symbol}, skipping duplicate")
        
        return has_active
    
    async def create_signal_tracking(self, symbol: str, signal_type: str, confidence: float, entry_price: float, expires_in_minutes: int = 5) -> Optional[int]:
        """Create a new signal tracking record (only if no active signal exists)"""
        # Check for existing active signal
        if await self.has_active_signal(symbol):
            logger.debug(f"Skipping duplicate signal for {symbol} - already has active signal")
            return None
        
        expires_at = datetime.utcnow() + timedelta(minutes=expires_in_minutes)
        
        query = """
            INSERT INTO signal_tracking (symbol, signal_type, confidence, entry_price, expires_at)
            VALUES (?, ?, ?, ?, ?)
        """
        
        signal_id = await self._execute_query(query, (symbol, signal_type, confidence, entry_price, expires_at), fetch_id=True)
        logger.info(f"Created signal tracking {signal_id} for {symbol}")
        return signal_id
    
    async def get_active_signals(self) -> List[Dict[str, Any]]:
        """Get all active (non-expired) signals"""
        query = """
            SELECT * FROM signal_tracking 
            WHERE expires_at > datetime('now') AND closed_at IS NULL
            ORDER BY created_at DESC
        """
        
        rows = await self._execute_query(query, fetch_all=True)
        return [dict(row) for row in rows] if rows else []
    
    async def get_expired_signals(self) -> List[Dict[str, Any]]:
        """Get all expired signals that need to be closed"""
        query = """
            SELECT * FROM signal_tracking 
            WHERE expires_at <= datetime('now') AND closed_at IS NULL
        """
        
        rows = await self._execute_query(query, fetch_all=True)
        return [dict(row) for row in rows] if rows else []
    
    async def close_signal(self, signal_id: int, exit_price: float):
        """Close a signal and calculate result"""
        # Get the signal first
        signal_query = "SELECT * FROM signal_tracking WHERE id = ?"
        signal_row = await self._execute_query(signal_query, (signal_id,), fetch_one=True)
        
        if not signal_row:
            logger.error(f"Signal {signal_id} not found")
            return
        
        signal = dict(signal_row)
        entry_price = signal['entry_price']
        signal_type = signal['signal_type']
        
        # Calculate PnL percentage
        if signal_type.upper() == 'LONG':
            pnl_percentage = ((exit_price - entry_price) / entry_price) * 100
        else:  # SHORT
            pnl_percentage = ((entry_price - exit_price) / entry_price) * 100
        
        # Determine if winner
        is_winner = pnl_percentage > 0
        result = 'WIN' if is_winner else 'LOSS'
        
        # Update signal
        update_query = """
            UPDATE signal_tracking 
            SET exit_price = ?, pnl_percentage = ?, is_winner = ?, result = ?, closed_at = datetime('now')
            WHERE id = ?
        """
        
        await self._execute_query(update_query, (exit_price, pnl_percentage, is_winner, result, signal_id))
        
        # Update statistics
        await self._update_signal_stats(is_winner, pnl_percentage)
        
        logger.info(f"Closed signal {signal_id}: {result} with {pnl_percentage:.2f}% PnL")
        
        return {
            'result': result,
            'pnl_percentage': pnl_percentage,
            'is_winner': is_winner
        }
    
    async def _update_signal_stats(self, is_winner: bool, pnl_percentage: float):
        """Update overall signal statistics"""
        # Get current stats or create if not exists
        stats_query = "SELECT * FROM signal_stats ORDER BY id DESC LIMIT 1"
        stats_row = await self._execute_query(stats_query, fetch_one=True)
        
        if stats_row:
            stats = dict(stats_row)
            total_signals = stats['total_signals'] + 1
            winning_signals = stats['winning_signals'] + (1 if is_winner else 0)
            losing_signals = stats['losing_signals'] + (1 if not is_winner else 0)
            
            # Calculate new averages
            current_avg_pnl = stats['avg_pnl'] or 0
            avg_pnl = ((current_avg_pnl * (total_signals - 1)) + pnl_percentage) / total_signals
            win_rate = (winning_signals / total_signals) * 100
            
            # Update existing record
            update_query = """
                UPDATE signal_stats 
                SET total_signals = ?, winning_signals = ?, losing_signals = ?, 
                    win_rate = ?, avg_pnl = ?, last_updated = datetime('now')
                WHERE id = ?
            """
            await self._execute_query(update_query, (total_signals, winning_signals, losing_signals, win_rate, avg_pnl, stats['id']))
        else:
            # Create first record
            total_signals = 1
            winning_signals = 1 if is_winner else 0
            losing_signals = 1 if not is_winner else 0
            win_rate = 100.0 if is_winner else 0.0
            avg_pnl = pnl_percentage
            
            insert_query = """
                INSERT INTO signal_stats (total_signals, winning_signals, losing_signals, win_rate, avg_pnl)
                VALUES (?, ?, ?, ?, ?)
            """
            await self._execute_query(insert_query, (total_signals, winning_signals, losing_signals, win_rate, avg_pnl))
    
    async def get_signal_stats(self) -> Dict[str, Any]:
        """Get overall signal performance statistics"""
        query = "SELECT * FROM signal_stats ORDER BY id DESC LIMIT 1"
        row = await self._execute_query(query, fetch_one=True)
        
        if row:
            stats = dict(row)
            return {
                'total_signals': stats['total_signals'],
                'winning_signals': stats['winning_signals'],
                'losing_signals': stats['losing_signals'],
                'win_rate': stats['win_rate'],
                'avg_pnl': stats['avg_pnl'],
                'last_updated': stats['last_updated']
            }
        else:
            return {
                'total_signals': 0,
                'winning_signals': 0,
                'losing_signals': 0,
                'win_rate': 0.0,
                'avg_pnl': 0.0,
                'last_updated': None
            }
    
    def _row_to_trade(self, row) -> Trade:
        """Convert database row to Trade object"""
        return Trade(
            id=row[0],
            user_id=row[1],
            symbol=row[2],
            trade_type=TradeType(row[3]),
            entry_price=row[4],
            exit_price=row[5],
            leverage=row[6],
            capital=row[7],
            take_profit=row[8],
            stop_loss=row[9],
            tp_percentage=row[10],
            sl_percentage=row[11],
            status=TradeStatus(row[12]),
            pnl_usd=row[13] or 0,
            pnl_percentage=row[14] or 0,
            created_at=datetime.fromisoformat(row[15]) if row[15] else None,
            closed_at=datetime.fromisoformat(row[16]) if row[16] else None,
            notes=row[17]
        )
    
    async def _execute_query(self, query: str, params: tuple = (), fetch_one: bool = False, 
                           fetch_all: bool = False, fetch_id: bool = False):
        """Execute database query with proper error handling"""
        try:
            loop = asyncio.get_event_loop()
            
            def _execute():
                conn = sqlite3.connect(self.db_path)
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                try:
                    cursor.execute(query, params)
                    
                    if fetch_id:
                        result = cursor.lastrowid
                    elif fetch_one:
                        result = cursor.fetchone()
                    elif fetch_all:
                        result = cursor.fetchall()
                    else:
                        result = None
                    
                    conn.commit()
                    return result
                    
                finally:
                    conn.close()
            
            return await loop.run_in_executor(None, _execute)
            
        except Exception as e:
            logger.error(f"Database error: {e}")
            raise
    
    async def clear_all_data(self):
        """Clear all signals, trades, and statistics from the database"""
        try:
            # Clear all signal tracking data
            await self._execute_query("DELETE FROM signal_tracking")
            
            # Clear all trades
            await self._execute_query("DELETE FROM trades")
            
            # Clear all price data
            await self._execute_query("DELETE FROM price_data")
            
            # Clear user settings
            await self._execute_query("DELETE FROM user_settings")
            
            # Reset signal statistics
            await self._execute_query("DELETE FROM signal_stats")
            
            logger.info("All database data cleared successfully")
            
        except Exception as e:
            logger.error(f"Error clearing database: {e}")
            raise
    
    async def close(self):
        """Close database connections"""
        # SQLite connections are closed after each operation
        logger.info("Database manager closed")
