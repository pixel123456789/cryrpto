"""
Telegram inline keyboards for bot interactions
"""

from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from typing import List, Dict, Any

def create_trade_keyboard() -> InlineKeyboardMarkup:
    """Create main trading action keyboard"""
    keyboard = [
        [
            InlineKeyboardButton("📈 Open Trade", callback_data="open_trade"),
            InlineKeyboardButton("📊 Status", callback_data="show_status")
        ],
        [
            InlineKeyboardButton("🔍 Signals", callback_data="show_signals"),
            InlineKeyboardButton("📋 Help", callback_data="show_help")
        ]
    ]
    return InlineKeyboardMarkup(keyboard)

def create_signals_keyboard(signals: List[Dict[str, Any]]) -> InlineKeyboardMarkup:
    """Create keyboard for trading signals"""
    keyboard = []
    
    for signal in signals[:5]:  # Show top 5 signals
        symbol = signal['symbol']
        action = signal['action'].upper()
        confidence = signal['confidence']
        
        button_text = f"{symbol} {action} ({confidence:.1f}%)"
        callback_data = f"signal_{symbol}_{action.lower()}"
        
        keyboard.append([InlineKeyboardButton(button_text, callback_data=callback_data)])
    
    # Add refresh button
    keyboard.append([InlineKeyboardButton("🔄 Refresh Signals", callback_data="refresh_signals")])
    
    return InlineKeyboardMarkup(keyboard)

def create_trade_actions_keyboard(trade_id: int, symbol: str) -> InlineKeyboardMarkup:
    """Create keyboard for individual trade actions"""
    keyboard = [
        [
            InlineKeyboardButton("📊 Chart", callback_data=f"chart_{symbol}"),
            InlineKeyboardButton("📈 Status", callback_data=f"status_{trade_id}")
        ],
        [
            InlineKeyboardButton("🎯 Modify", callback_data=f"modify_{trade_id}"),
            InlineKeyboardButton("❌ Close", callback_data=f"close_{trade_id}")
        ]
    ]
    return InlineKeyboardMarkup(keyboard)

def create_confirmation_keyboard(action: str, trade_id: int) -> InlineKeyboardMarkup:
    """Create confirmation keyboard for destructive actions"""
    keyboard = [
        [
            InlineKeyboardButton("✅ Confirm", callback_data=f"confirm_{action}_{trade_id}"),
            InlineKeyboardButton("❌ Cancel", callback_data="cancel")
        ]
    ]
    return InlineKeyboardMarkup(keyboard)

def create_leverage_keyboard() -> InlineKeyboardMarkup:
    """Create keyboard for leverage selection"""
    leverages = [5, 10, 20, 25, 50, 75, 100]
    keyboard = []
    
    row = []
    for leverage in leverages:
        row.append(InlineKeyboardButton(f"{leverage}x", callback_data=f"leverage_{leverage}"))
        if len(row) == 3:
            keyboard.append(row)
            row = []
    
    if row:
        keyboard.append(row)
    
    return InlineKeyboardMarkup(keyboard)

def create_timeframe_keyboard() -> InlineKeyboardMarkup:
    """Create keyboard for chart timeframe selection"""
    timeframes = [
        ("1m", "1m"), ("5m", "5m"), ("15m", "15m"), ("1h", "1h"),
        ("4h", "4h"), ("1d", "1d"), ("1w", "1w")
    ]
    
    keyboard = []
    row = []
    
    for tf_display, tf_value in timeframes:
        row.append(InlineKeyboardButton(tf_display, callback_data=f"timeframe_{tf_value}"))
        if len(row) == 4:
            keyboard.append(row)
            row = []
    
    if row:
        keyboard.append(row)
    
    return InlineKeyboardMarkup(keyboard)
