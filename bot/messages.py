"""
Message templates and formatting for Telegram bot
"""

from datetime import datetime
from typing import Dict, List, Any
from core.models import Trade, TradeType, TradeStatus
from utils.helpers import format_pnl, format_price

class Messages:
    """Message templates and formatting utilities"""
    
    def get_welcome_message(self) -> str:
        """Get welcome message for new users"""
        return """
🚀 **Welcome to AI-Powered MEXC Futures Bot!**

Your 24/7 crypto trading assistant for MEXC USDT-Margined Futures.

**Quick Start:**
• `/open SYMBOL LONG/SHORT PRICE LEVERAGE CAPITAL` - Log a trade
• `/status` - View your open trades
• `/signals` - Get AI trading signals
• `/chart SYMBOL` - Generate price charts

**Example:**
`/open BTC_USDT LONG 42000 10 100`

Ready to start trading? Use the buttons below! 📈
        """
    
    def get_help_message(self) -> str:
        """Get help message with all commands"""
        return """
📚 **MEXC Futures Bot Commands**

**Trading Commands:**
• `/open SYMBOL TYPE PRICE LEV CAPITAL` - Open trade
  Example: `/open BTC_USDT LONG 42000 10 100`
• `/close [SYMBOL/ID]` - Close trade(s)
• `/status` - View all open trades

**Analysis Commands:**
• `/signals` - Get trading signals
• `/chart SYMBOL` - Generate price chart

**Parameters:**
• **SYMBOL**: Trading pair (BTC_USDT, ETH_USDT, etc.)
• **TYPE**: LONG or SHORT
• **PRICE**: Entry price
• **LEV**: Leverage (1-100x)
• **CAPITAL**: Capital amount in USDT

**Features:**
✅ Real-time PnL tracking
✅ Automatic TP/SL calculation
✅ AI-powered signal detection
✅ Technical analysis charts
✅ Risk management alerts

Need help? Just send a message! 🤖
        """
    
    def format_trade_opened(self, trade: Trade, current_price: float, pnl_data: Dict) -> str:
        """Format message for newly opened trade"""
        direction_emoji = "🟢" if trade.trade_type == TradeType.LONG else "🔴"
        pnl_emoji = "📈" if pnl_data['pnl_percentage'] >= 0 else "📉"
        
        return f"""
{direction_emoji} **Trade Opened Successfully!**

📊 **Symbol:** {trade.symbol}
📍 **Type:** {trade.trade_type.value}
💰 **Entry Price:** ${format_price(trade.entry_price)}
⚡ **Leverage:** {trade.leverage}x
💵 **Capital:** ${format_price(trade.capital)}

🎯 **Take Profit:** ${format_price(trade.take_profit)} (+{trade.tp_percentage:.1f}%)
🛡️ **Stop Loss:** ${format_price(trade.stop_loss)} (-{trade.sl_percentage:.1f}%)

📈 **Current Price:** ${format_price(current_price)}
{pnl_emoji} **Current PnL:** {format_pnl(pnl_data['pnl_percentage'])} ({format_pnl(pnl_data['pnl_usd'], is_usd=True)})

🕐 **Time:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}
        """
    
    def format_trade_closed(self, trade: Trade, exit_price: float, pnl_data: Dict) -> str:
        """Format message for closed trade"""
        direction_emoji = "🟢" if trade.trade_type == TradeType.LONG else "🔴"
        result_emoji = "🎉" if pnl_data['pnl_percentage'] >= 0 else "😞"
        
        return f"""
{direction_emoji} **Trade Closed!** {result_emoji}

📊 **Symbol:** {trade.symbol}
📍 **Type:** {trade.trade_type.value}
💰 **Entry Price:** ${format_price(trade.entry_price)}
🚪 **Exit Price:** ${format_price(exit_price)}
⚡ **Leverage:** {trade.leverage}x

💵 **Capital:** ${format_price(trade.capital)}
📊 **Final PnL:** {format_pnl(pnl_data['pnl_percentage'])} ({format_pnl(pnl_data['pnl_usd'], is_usd=True)})

⏱️ **Duration:** {self._calculate_duration(trade)}
🕐 **Closed:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}
        """
    
    def format_trade_status(self, trade: Trade, current_price: float, pnl_data: Dict) -> str:
        """Format individual trade status"""
        direction_emoji = "🟢" if trade.trade_type == TradeType.LONG else "🔴"
        pnl_emoji = "📈" if pnl_data['pnl_percentage'] >= 0 else "📉"
        
        # Calculate distance to TP/SL
        if trade.trade_type == TradeType.LONG:
            tp_distance = ((trade.take_profit - current_price) / current_price) * 100
            sl_distance = ((current_price - trade.stop_loss) / current_price) * 100
        else:
            tp_distance = ((current_price - trade.take_profit) / current_price) * 100
            sl_distance = ((trade.stop_loss - current_price) / current_price) * 100
        
        status_line = ""
        if tp_distance <= 1:
            status_line = "🎯 **Near TP!**"
        elif sl_distance <= 1:
            status_line = "⚠️ **Near SL!**"
        
        return f"""
{direction_emoji} **{trade.symbol}** - {trade.trade_type.value}
💰 Entry: ${format_price(trade.entry_price)} | Current: ${format_price(current_price)}
{pnl_emoji} **PnL: {format_pnl(pnl_data['pnl_percentage'])} ({format_pnl(pnl_data['pnl_usd'], is_usd=True)})**
🎯 TP: ${format_price(trade.take_profit)} | 🛡️ SL: ${format_price(trade.stop_loss)}
{status_line}
        """
    
    def format_signals(self, signals: List[Dict[str, Any]]) -> str:
        """Format trading signals list"""
        if not signals:
            return "📊 No high-confidence signals found at the moment."
        
        message = "🔍 **AI Trading Signals**\n\n"
        
        for i, signal in enumerate(signals[:10], 1):
            confidence_emoji = self._get_confidence_emoji(signal['confidence'])
            action_emoji = "🟢" if signal['action'].upper() == 'LONG' else "🔴"
            
            message += f"{i}. {action_emoji} **{signal['symbol']}** - {signal['action'].upper()}\n"
            message += f"   {confidence_emoji} Confidence: {signal['confidence']:.1f}%\n"
            message += f"   💡 {signal['reason']}\n"
            
            if 'entry_zone' in signal:
                message += f"   🎯 Entry: ${signal['entry_zone']}\n"
            
            message += "\n"
        
        message += "⚡ **Signals update every 5 minutes**\n"
        message += "🎯 Only showing signals with >70% confidence"
        
        return message
    
    def format_pnl_alert(self, trade: Trade, current_price: float, pnl_data: Dict) -> str:
        """Format PnL alert message"""
        direction_emoji = "🟢" if trade.trade_type == TradeType.LONG else "🔴"
        pnl_emoji = "📈" if pnl_data['pnl_percentage'] >= 0 else "📉"
        
        alert_type = ""
        if pnl_data['pnl_percentage'] >= 10:
            alert_type = "🎉 **Profit Alert!**"
        elif pnl_data['pnl_percentage'] <= -5:
            alert_type = "⚠️ **Loss Alert!**"
        else:
            alert_type = "📊 **PnL Update**"
        
        return f"""
{alert_type}

{direction_emoji} **{trade.symbol}** - {trade.trade_type.value}
{pnl_emoji} **PnL: {format_pnl(pnl_data['pnl_percentage'])} ({format_pnl(pnl_data['pnl_usd'], is_usd=True)})**
💰 Entry: ${format_price(trade.entry_price)} → Current: ${format_price(current_price)}
        """
    
    def _get_confidence_emoji(self, confidence: float) -> str:
        """Get emoji based on confidence level"""
        if confidence >= 90:
            return "🔥"
        elif confidence >= 80:
            return "⭐"
        elif confidence >= 70:
            return "✅"
        else:
            return "📊"
    
    def _calculate_duration(self, trade: Trade) -> str:
        """Calculate trade duration"""
        if not trade.closed_at or not trade.created_at:
            return "Unknown"
        
        duration = trade.closed_at - trade.created_at
        hours = duration.total_seconds() / 3600
        
        if hours < 1:
            minutes = int(duration.total_seconds() / 60)
            return f"{minutes}m"
        elif hours < 24:
            return f"{hours:.1f}h"
        else:
            days = duration.days
            remaining_hours = (duration.total_seconds() % 86400) / 3600
            return f"{days}d {remaining_hours:.1f}h"
