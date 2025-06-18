"""
Utility helper functions for the trading bot
"""

import re
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple, Union
import json
import hashlib
import hmac
import base64
from decimal import Decimal, ROUND_HALF_UP

logger = logging.getLogger(__name__)

def validate_symbol(symbol: str) -> Optional[str]:
    """
    Validate and normalize trading symbol format
    
    Args:
        symbol: Trading symbol (e.g., 'btcusdt', 'BTC-USDT', 'BTC_USDT')
    
    Returns:
        Normalized symbol in MEXC format (BTC_USDT) or None if invalid
    """
    if not symbol or not isinstance(symbol, str):
        return None
    
    # Remove spaces and convert to uppercase
    symbol = symbol.strip().upper()
    
    # Handle different formats
    if '_' in symbol:
        # Already in correct format
        parts = symbol.split('_')
    elif '-' in symbol:
        # Convert from BTC-USDT to BTC_USDT
        parts = symbol.split('-')
    elif symbol.endswith('USDT') and len(symbol) > 4:
        # Convert from BTCUSDT to BTC_USDT
        base = symbol[:-4]
        quote = 'USDT'
        parts = [base, quote]
    else:
        return None
    
    if len(parts) != 2 or not all(parts):
        return None
    
    base, quote = parts
    
    # Validate base and quote currencies
    if not re.match(r'^[A-Z0-9]{2,10}$', base) or not re.match(r'^[A-Z]{3,6}$', quote):
        return None
    
    # For MEXC futures, we primarily deal with USDT pairs
    if quote != 'USDT':
        return None
    
    return f"{base}_USDT"

def parse_trade_params(args: List[str]) -> Optional[Dict[str, Any]]:
    """
    Parse trade command parameters
    
    Args:
        args: List of command arguments
    
    Returns:
        Parsed parameters or None if invalid
    """
    try:
        if len(args) < 5:
            return None
        
        symbol = validate_symbol(args[0])
        if not symbol:
            return None
        
        trade_type = args[1].upper()
        if trade_type not in ['LONG', 'SHORT']:
            return None
        
        entry_price = float(args[2])
        leverage = int(args[3])
        capital = float(args[4])
        
        # Validate ranges
        if entry_price <= 0:
            return None
        
        if leverage < 1 or leverage > 100:
            return None
        
        if capital < 1:
            return None
        
        return {
            'symbol': symbol,
            'trade_type': trade_type,
            'entry_price': entry_price,
            'leverage': leverage,
            'capital': capital
        }
        
    except (ValueError, IndexError):
        return None

def format_price(price: float, precision: int = 6) -> str:
    """
    Format price with appropriate precision
    
    Args:
        price: Price value
        precision: Number of decimal places
    
    Returns:
        Formatted price string
    """
    if price is None:
        return "N/A"
    
    # Determine appropriate precision based on price magnitude
    if price >= 1000:
        precision = 2
    elif price >= 100:
        precision = 3
    elif price >= 10:
        precision = 4
    elif price >= 1:
        precision = 5
    else:
        precision = 8
    
    # Use Decimal for precise formatting
    decimal_price = Decimal(str(price))
    formatted = decimal_price.quantize(Decimal('0.' + '0' * precision), rounding=ROUND_HALF_UP)
    
    # Remove trailing zeros
    formatted_str = str(formatted).rstrip('0').rstrip('.')
    
    return formatted_str

def format_pnl(pnl: float, is_usd: bool = False) -> str:
    """
    Format PnL with appropriate colors and symbols
    
    Args:
        pnl: PnL value
        is_usd: Whether the value is in USD
    
    Returns:
        Formatted PnL string with emoji
    """
    if pnl is None:
        return "N/A"
    
    if is_usd:
        if pnl >= 0:
            return f"+${abs(pnl):.2f} 📈"
        else:
            return f"-${abs(pnl):.2f} 📉"
    else:
        if pnl >= 0:
            return f"+{pnl:.2f}% 📈"
        else:
            return f"{pnl:.2f}% 📉"

def format_percentage(value: float, precision: int = 2) -> str:
    """
    Format percentage value
    
    Args:
        value: Percentage value
        precision: Decimal precision
    
    Returns:
        Formatted percentage string
    """
    if value is None:
        return "N/A"
    
    return f"{value:.{precision}f}%"

def format_duration(start_time: datetime, end_time: Optional[datetime] = None) -> str:
    """
    Format duration between two timestamps
    
    Args:
        start_time: Start timestamp
        end_time: End timestamp (defaults to now)
    
    Returns:
        Formatted duration string
    """
    if not start_time:
        return "N/A"
    
    if end_time is None:
        end_time = datetime.utcnow()
    
    duration = end_time - start_time
    
    days = duration.days
    hours, remainder = divmod(duration.seconds, 3600)
    minutes, _ = divmod(remainder, 60)
    
    if days > 0:
        return f"{days}d {hours}h {minutes}m"
    elif hours > 0:
        return f"{hours}h {minutes}m"
    else:
        return f"{minutes}m"

def calculate_position_size(capital: float, leverage: int, entry_price: float) -> Dict[str, float]:
    """
    Calculate position size and related metrics
    
    Args:
        capital: Available capital
        leverage: Trading leverage
        entry_price: Entry price
    
    Returns:
        Dictionary with position metrics
    """
    try:
        notional_value = capital * leverage
        quantity = notional_value / entry_price
        
        return {
            'quantity': quantity,
            'notional_value': notional_value,
            'margin_used': capital,
            'leverage_ratio': leverage
        }
        
    except (ZeroDivisionError, TypeError):
        return {
            'quantity': 0,
            'notional_value': 0,
            'margin_used': 0,
            'leverage_ratio': 0
        }

def calculate_risk_metrics(entry_price: float, stop_loss: float, take_profit: float,
                          capital: float, leverage: int, trade_type: str) -> Dict[str, float]:
    """
    Calculate risk/reward metrics for a trade
    
    Args:
        entry_price: Entry price
        stop_loss: Stop loss price
        take_profit: Take profit price
        capital: Trading capital
        leverage: Leverage ratio
        trade_type: 'LONG' or 'SHORT'
    
    Returns:
        Dictionary with risk metrics
    """
    try:
        if trade_type.upper() == 'LONG':
            potential_profit = take_profit - entry_price
            potential_loss = entry_price - stop_loss
        else:  # SHORT
            potential_profit = entry_price - take_profit
            potential_loss = stop_loss - entry_price
        
        # Calculate percentages
        profit_percentage = (potential_profit / entry_price) * 100 * leverage
        loss_percentage = (potential_loss / entry_price) * 100 * leverage
        
        # Calculate USD amounts
        profit_usd = capital * (profit_percentage / 100)
        loss_usd = capital * (loss_percentage / 100)
        
        # Risk/reward ratio
        risk_reward_ratio = abs(potential_profit / potential_loss) if potential_loss != 0 else 0
        
        return {
            'potential_profit_pct': profit_percentage,
            'potential_loss_pct': abs(loss_percentage),
            'potential_profit_usd': profit_usd,
            'potential_loss_usd': abs(loss_usd),
            'risk_reward_ratio': risk_reward_ratio
        }
        
    except (ZeroDivisionError, TypeError):
        return {
            'potential_profit_pct': 0,
            'potential_loss_pct': 0,
            'potential_profit_usd': 0,
            'potential_loss_usd': 0,
            'risk_reward_ratio': 0
        }

def sanitize_user_input(text: str, max_length: int = 1000) -> str:
    """
    Sanitize user input to prevent injection attacks
    
    Args:
        text: Input text
        max_length: Maximum allowed length
    
    Returns:
        Sanitized text
    """
    if not isinstance(text, str):
        return ""
    
    # Limit length
    text = text[:max_length]
    
    # Remove potential harmful characters
    text = re.sub(r'[<>"\';\\]', '', text)
    
    # Strip whitespace
    text = text.strip()
    
    return text

def generate_trade_id(user_id: int, symbol: str, timestamp: datetime) -> str:
    """
    Generate a unique trade ID
    
    Args:
        user_id: User ID
        symbol: Trading symbol
        timestamp: Trade timestamp
    
    Returns:
        Unique trade ID
    """
    data = f"{user_id}_{symbol}_{timestamp.isoformat()}"
    return hashlib.md5(data.encode()).hexdigest()[:12].upper()

def is_market_hours() -> bool:
    """
    Check if it's currently market hours (crypto markets are 24/7)
    
    Returns:
        Always True for crypto markets
    """
    return True

def get_market_status() -> Dict[str, Any]:
    """
    Get current market status
    
    Returns:
        Market status information
    """
    now = datetime.utcnow()
    
    return {
        'is_open': True,  # Crypto markets are always open
        'current_time': now.isoformat(),
        'timezone': 'UTC',
        'next_close': None,  # No market close for crypto
        'session': 'CONTINUOUS'
    }

def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    Split a list into chunks of specified size
    
    Args:
        lst: Input list
        chunk_size: Size of each chunk
    
    Returns:
        List of chunks
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def safe_float(value: Any, default: float = 0.0) -> float:
    """
    Safely convert value to float
    
    Args:
        value: Value to convert
        default: Default value if conversion fails
    
    Returns:
        Float value or default
    """
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def safe_int(value: Any, default: int = 0) -> int:
    """
    Safely convert value to integer
    
    Args:
        value: Value to convert
        default: Default value if conversion fails
    
    Returns:
        Integer value or default
    """
    try:
        return int(value)
    except (ValueError, TypeError):
        return default

def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to specified length
    
    Args:
        text: Input text
        max_length: Maximum length
        suffix: Suffix to append if truncated
    
    Returns:
        Truncated text
    """
    if not text or len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix

def format_large_number(number: float, precision: int = 2) -> str:
    """
    Format large numbers with appropriate suffixes (K, M, B)
    
    Args:
        number: Number to format
        precision: Decimal precision
    
    Returns:
        Formatted number string
    """
    if abs(number) >= 1_000_000_000:
        return f"{number / 1_000_000_000:.{precision}f}B"
    elif abs(number) >= 1_000_000:
        return f"{number / 1_000_000:.{precision}f}M"
    elif abs(number) >= 1_000:
        return f"{number / 1_000:.{precision}f}K"
    else:
        return f"{number:.{precision}f}"

def create_pagination_info(total_items: int, page: int, per_page: int) -> Dict[str, Any]:
    """
    Create pagination information
    
    Args:
        total_items: Total number of items
        page: Current page (1-based)
        per_page: Items per page
    
    Returns:
        Pagination information
    """
    total_pages = (total_items + per_page - 1) // per_page
    start_item = (page - 1) * per_page + 1
    end_item = min(page * per_page, total_items)
    
    return {
        'current_page': page,
        'total_pages': total_pages,
        'per_page': per_page,
        'total_items': total_items,
        'start_item': start_item,
        'end_item': end_item,
        'has_previous': page > 1,
        'has_next': page < total_pages
    }

async def retry_async(func, max_retries: int = 3, delay: float = 1.0, 
                     backoff_factor: float = 2.0, exceptions: Tuple = (Exception,)):
    """
    Retry an async function with exponential backoff
    
    Args:
        func: Async function to retry
        max_retries: Maximum number of retries
        delay: Initial delay between retries
        backoff_factor: Backoff multiplier
        exceptions: Exceptions to catch and retry
    
    Returns:
        Function result or raises last exception
    """
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            return await func()
        except exceptions as e:
            last_exception = e
            if attempt < max_retries:
                wait_time = delay * (backoff_factor ** attempt)
                logger.warning(f"Attempt {attempt + 1} failed, retrying in {wait_time:.1f}s: {e}")
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"All {max_retries + 1} attempts failed: {e}")
    
    raise last_exception

def validate_leverage(leverage: int) -> bool:
    """
    Validate leverage value
    
    Args:
        leverage: Leverage value to validate
    
    Returns:
        True if valid, False otherwise
    """
    return isinstance(leverage, int) and 1 <= leverage <= 100

def validate_capital(capital: float, min_capital: float = 10.0, max_capital: float = 100000.0) -> bool:
    """
    Validate capital amount
    
    Args:
        capital: Capital amount to validate
        min_capital: Minimum allowed capital
        max_capital: Maximum allowed capital
    
    Returns:
        True if valid, False otherwise
    """
    return isinstance(capital, (int, float)) and min_capital <= capital <= max_capital

def get_emoji_by_percentage(percentage: float) -> str:
    """
    Get appropriate emoji based on percentage value
    
    Args:
        percentage: Percentage value
    
    Returns:
        Appropriate emoji
    """
    if percentage >= 10:
        return "🚀"
    elif percentage >= 5:
        return "📈"
    elif percentage >= 0:
        return "✅"
    elif percentage >= -5:
        return "📉"
    elif percentage >= -10:
        return "⚠️"
    else:
        return "🔥"

