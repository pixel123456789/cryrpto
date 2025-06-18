"""
Logging configuration for the trading bot
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None, 
                  max_file_size: int = 10 * 1024 * 1024, backup_count: int = 5):
    """
    Setup logging configuration for the application
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        max_file_size: Maximum size of log file in bytes
        backup_count: Number of backup files to keep
    """
    
    # Create logs directory if it doesn't exist
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(name)-20s | %(funcName)-15s:%(lineno)-3d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_file,
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(file_handler)
    
    # Set specific logger levels
    configure_specific_loggers()
    
    # Log startup message
    logger = logging.getLogger(__name__)
    logger.info("="*80)
    logger.info("🚀 MEXC Futures Trading Bot - Logging System Initialized")
    logger.info(f"📊 Log Level: {log_level.upper()}")
    if log_file:
        logger.info(f"📝 Log File: {log_file}")
    logger.info(f"🕐 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*80)

def configure_specific_loggers():
    """Configure specific loggers with appropriate levels"""
    
    # Reduce noise from external libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('aiohttp').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('telegram').setLevel(logging.INFO)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    
    # Set our application loggers to appropriate levels
    logging.getLogger('bot').setLevel(logging.INFO)
    logging.getLogger('core').setLevel(logging.INFO)
    logging.getLogger('signals').setLevel(logging.INFO)
    logging.getLogger('charts').setLevel(logging.INFO)
    logging.getLogger('utils').setLevel(logging.INFO)

class BotLogger:
    """Custom logger wrapper with bot-specific methods"""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.name = name
    
    def trade_opened(self, user_id: int, symbol: str, trade_type: str, 
                    entry_price: float, leverage: int, capital: float):
        """Log trade opening"""
        self.logger.info(
            f"🟢 TRADE OPENED | User: {user_id} | {symbol} {trade_type} | "
            f"Entry: ${entry_price} | Leverage: {leverage}x | Capital: ${capital}"
        )
    
    def trade_closed(self, user_id: int, symbol: str, trade_type: str, 
                    entry_price: float, exit_price: float, pnl_percentage: float, pnl_usd: float):
        """Log trade closing"""
        pnl_emoji = "📈" if pnl_percentage >= 0 else "📉"
        self.logger.info(
            f"🔴 TRADE CLOSED | User: {user_id} | {symbol} {trade_type} | "
            f"Entry: ${entry_price} → Exit: ${exit_price} | "
            f"{pnl_emoji} PnL: {pnl_percentage:.2f}% (${pnl_usd:.2f})"
        )
    
    def signal_generated(self, symbol: str, action: str, confidence: float, reason: str):
        """Log signal generation"""
        confidence_emoji = "🔥" if confidence >= 90 else "⭐" if confidence >= 80 else "✅"
        self.logger.info(
            f"🎯 SIGNAL | {symbol} {action} | {confidence_emoji} {confidence:.1f}% | {reason}"
        )
    
    def api_error(self, endpoint: str, error: str):
        """Log API errors"""
        self.logger.error(f"🚨 API ERROR | Endpoint: {endpoint} | Error: {error}")
    
    def user_action(self, user_id: int, username: str, action: str):
        """Log user actions"""
        self.logger.info(f"👤 USER ACTION | {username} ({user_id}) | {action}")
    
    def system_status(self, component: str, status: str, details: str = ""):
        """Log system status changes"""
        status_emoji = "✅" if status.lower() == "ok" else "⚠️" if status.lower() == "warning" else "❌"
        message = f"🔧 SYSTEM | {component}: {status_emoji} {status.upper()}"
        if details:
            message += f" | {details}"
        self.logger.info(message)
    
    def performance_metric(self, metric_name: str, value: float, unit: str = ""):
        """Log performance metrics"""
        self.logger.info(f"📊 METRIC | {metric_name}: {value}{unit}")
    
    def debug(self, message: str):
        """Debug logging"""
        self.logger.debug(message)
    
    def info(self, message: str):
        """Info logging"""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Warning logging"""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Error logging"""
        self.logger.error(message)
    
    def critical(self, message: str):
        """Critical logging"""
        self.logger.critical(message)

def get_bot_logger(name: str) -> BotLogger:
    """Get a bot-specific logger instance"""
    return BotLogger(name)

class PerformanceLogger:
    """Logger for performance monitoring"""
    
    def __init__(self):
        self.logger = logging.getLogger('performance')
        self.start_times = {}
    
    def start_timer(self, operation: str):
        """Start timing an operation"""
        self.start_times[operation] = datetime.now()
    
    def end_timer(self, operation: str):
        """End timing and log the duration"""
        if operation in self.start_times:
            duration = (datetime.now() - self.start_times[operation]).total_seconds()
            self.logger.info(f"⏱️ {operation}: {duration:.3f}s")
            del self.start_times[operation]
        else:
            self.logger.warning(f"Timer for '{operation}' was not started")
    
    def log_memory_usage(self):
        """Log current memory usage"""
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            self.logger.info(f"💾 Memory Usage: {memory_mb:.1f} MB")
        except ImportError:
            self.logger.debug("psutil not available for memory monitoring")
    
    def log_api_call(self, endpoint: str, status_code: int, duration: float):
        """Log API call performance"""
        status_emoji = "✅" if 200 <= status_code < 300 else "⚠️" if 400 <= status_code < 500 else "❌"
        self.logger.info(f"🌐 API | {endpoint} | {status_emoji} {status_code} | {duration:.3f}s")

# Global performance logger instance
perf_logger = PerformanceLogger()

