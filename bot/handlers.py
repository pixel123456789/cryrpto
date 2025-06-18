"""
Telegram bot handlers for user interactions
"""

import asyncio
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes
from telegram.constants import ParseMode

from core.database import DatabaseManager
from core.mexc_api import MEXCClient
from core.trading import TradingManager
from core.models import Trade, TradeStatus, TradeType
from signals.detector import SignalDetector
from charts.generator import ChartGenerator
from bot.keyboards import create_trade_keyboard, create_signals_keyboard
from bot.messages import Messages
from utils.helpers import validate_symbol, parse_trade_params, format_pnl

logger = logging.getLogger(__name__)

class BotHandlers:
    def __init__(self, db_manager: DatabaseManager, mexc_client: MEXCClient, signal_detector: SignalDetector):
        self.db_manager = db_manager
        self.mexc_client = mexc_client
        self.signal_detector = signal_detector
        self.trading_manager = TradingManager(db_manager, mexc_client)
        self.chart_generator = ChartGenerator(mexc_client)
        self.messages = Messages()
        
        # Track user states for multi-step commands
        self.user_states: Dict[int, Dict[str, Any]] = {}
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        user_id = update.effective_user.id
        username = update.effective_user.username or "Unknown"
        
        logger.info(f"User {username} ({user_id}) started the bot")
        
        welcome_message = self.messages.get_welcome_message()
        keyboard = create_trade_keyboard()
        
        await update.message.reply_text(
            welcome_message,
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=keyboard
        )
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command"""
        help_message = self.messages.get_help_message()
        await update.message.reply_text(help_message, parse_mode=ParseMode.MARKDOWN)
    
    async def open_trade_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /open command to log a new trade"""
        user_id = update.effective_user.id
        
        try:
            # Parse command arguments
            args = context.args
            if len(args) < 5:
                await update.message.reply_text(
                    "❌ Invalid format. Use:\n"
                    "`/open SYMBOL LONG/SHORT ENTRY_PRICE LEVERAGE CAPITAL`\n\n"
                    "Example: `/open BTC_USDT LONG 42000 10 100`",
                    parse_mode=ParseMode.MARKDOWN
                )
                return
            
            symbol, trade_type_str, entry_price_str, leverage_str, capital_str = args[:5]
            
            # Validate and parse parameters
            symbol = validate_symbol(symbol)
            if not symbol:
                await update.message.reply_text("❌ Invalid symbol format. Use format like BTC_USDT")
                return
            
            try:
                trade_type = TradeType.LONG if trade_type_str.upper() == "LONG" else TradeType.SHORT
                entry_price = float(entry_price_str)
                leverage = int(leverage_str)
                capital = float(capital_str)
            except ValueError:
                await update.message.reply_text("❌ Invalid numeric values for price, leverage, or capital")
                return
            
            # Validate ranges
            if leverage < 1 or leverage > 100:
                await update.message.reply_text("❌ Leverage must be between 1 and 100")
                return
            
            if capital < 10:
                await update.message.reply_text("❌ Minimum capital is $10")
                return
            
            # Check if symbol exists on MEXC
            ticker = await self.mexc_client.get_ticker(symbol)
            if not ticker:
                await update.message.reply_text(f"❌ Symbol {symbol} not found on MEXC")
                return
            
            # Check for existing open trades for this symbol
            existing_trades = await self.db_manager.get_user_trades(user_id, status=TradeStatus.OPEN)
            symbol_trades = [t for t in existing_trades if t.symbol == symbol]
            
            if symbol_trades:
                await update.message.reply_text(
                    f"⚠️ You already have an open {symbol} trade. Close it first or modify the existing one."
                )
                return
            
            # Create new trade
            trade = Trade(
                user_id=user_id,
                symbol=symbol,
                trade_type=trade_type,
                entry_price=entry_price,
                leverage=leverage,
                capital=capital,
                status=TradeStatus.OPEN,
                created_at=datetime.utcnow()
            )
            
            # Calculate TP and SL
            await self.trading_manager.calculate_tp_sl(trade)
            
            # Save trade to database
            trade_id = await self.db_manager.create_trade(trade)
            trade.id = trade_id
            
            # Get current price for immediate PnL calculation
            current_price = ticker.get('price', entry_price)
            pnl_data = await self.trading_manager.calculate_pnl(trade, current_price)
            
            # Format response
            response = self.messages.format_trade_opened(trade, current_price, pnl_data)
            
            # Create inline keyboard for trade actions
            keyboard = InlineKeyboardMarkup([
                [
                    InlineKeyboardButton("📊 Chart", callback_data=f"chart_{symbol}"),
                    InlineKeyboardButton("📈 Status", callback_data=f"status_{trade_id}")
                ],
                [
                    InlineKeyboardButton("🎯 Modify TP/SL", callback_data=f"modify_{trade_id}"),
                    InlineKeyboardButton("❌ Close Trade", callback_data=f"close_{trade_id}")
                ]
            ])
            
            await update.message.reply_text(response, parse_mode=ParseMode.MARKDOWN, reply_markup=keyboard)
            
            logger.info(f"User {user_id} opened {trade_type.value} trade for {symbol} at {entry_price}")
            
        except Exception as e:
            logger.error(f"Error in open_trade_command: {e}")
            await update.message.reply_text("❌ An error occurred while opening the trade. Please try again.")
    
    async def close_trade_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /close command to close a trade"""
        user_id = update.effective_user.id
        
        try:
            args = context.args
            if not args:
                # Show list of open trades to close
                open_trades = await self.db_manager.get_user_trades(user_id, status=TradeStatus.OPEN)
                
                if not open_trades:
                    await update.message.reply_text("📝 You have no open trades to close.")
                    return
                
                keyboard_buttons = []
                for trade in open_trades:
                    current_price = await self.mexc_client.get_current_price(trade.symbol)
                    pnl_data = await self.trading_manager.calculate_pnl(trade, current_price)
                    pnl_str = format_pnl(pnl_data['pnl_percentage'])
                    
                    button_text = f"{trade.symbol} {trade.trade_type.value} ({pnl_str})"
                    keyboard_buttons.append([InlineKeyboardButton(button_text, callback_data=f"close_{trade.id}")])
                
                keyboard = InlineKeyboardMarkup(keyboard_buttons)
                await update.message.reply_text("Select a trade to close:", reply_markup=keyboard)
                return
            
            # Close specific trade by symbol or ID
            identifier = args[0]
            
            # Try to find trade by symbol first, then by ID
            open_trades = await self.db_manager.get_user_trades(user_id, status=TradeStatus.OPEN)
            trade = None
            
            for t in open_trades:
                if t.symbol.upper() == identifier.upper() or str(t.id) == identifier:
                    trade = t
                    break
            
            if not trade:
                await update.message.reply_text(f"❌ No open trade found for {identifier}")
                return
            
            # Get current price and close the trade
            current_price = await self.mexc_client.get_current_price(trade.symbol)
            if not current_price:
                await update.message.reply_text("❌ Unable to get current price. Please try again.")
                return
            
            # Close the trade
            await self.trading_manager.close_trade(trade, current_price)
            
            # Calculate final PnL
            pnl_data = await self.trading_manager.calculate_pnl(trade, current_price)
            
            # Format response
            response = self.messages.format_trade_closed(trade, current_price, pnl_data)
            await update.message.reply_text(response, parse_mode=ParseMode.MARKDOWN)
            
            logger.info(f"User {user_id} closed trade {trade.id} for {trade.symbol}")
            
        except Exception as e:
            logger.error(f"Error in close_trade_command: {e}")
            await update.message.reply_text("❌ An error occurred while closing the trade. Please try again.")
    
    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command to show all open trades"""
        user_id = update.effective_user.id
        
        try:
            open_trades = await self.db_manager.get_user_trades(user_id, status=TradeStatus.OPEN)
            
            if not open_trades:
                await update.message.reply_text("📝 You have no open trades.")
                return
            
            # Get current prices and calculate PnL for all trades
            status_messages = []
            total_pnl = 0
            
            for trade in open_trades:
                current_price = await self.mexc_client.get_current_price(trade.symbol)
                if current_price:
                    pnl_data = await self.trading_manager.calculate_pnl(trade, current_price)
                    total_pnl += pnl_data['pnl_usd']
                    
                    status_msg = self.messages.format_trade_status(trade, current_price, pnl_data)
                    status_messages.append(status_msg)
                else:
                    status_messages.append(f"❌ {trade.symbol}: Unable to get current price")
            
            # Combine all status messages
            response = "📊 **Your Open Trades:**\n\n" + "\n\n".join(status_messages)
            response += f"\n\n💰 **Total Portfolio PnL: {format_pnl(total_pnl, is_usd=True)}**"
            
            # Create action buttons
            keyboard = InlineKeyboardMarkup([
                [
                    InlineKeyboardButton("🔄 Refresh", callback_data="refresh_status"),
                    InlineKeyboardButton("📈 Signals", callback_data="show_signals")
                ]
            ])
            
            await update.message.reply_text(response, parse_mode=ParseMode.MARKDOWN, reply_markup=keyboard)
            
        except Exception as e:
            logger.error(f"Error in status_command: {e}")
            await update.message.reply_text("❌ An error occurred while fetching trade status. Please try again.")
    
    async def signals_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /signals command to show trading signals"""
        try:
            await update.message.reply_text("🔍 Scanning for trading signals... Please wait.")
            
            # Get trading signals
            signals = await self.signal_detector.scan_all_symbols()
            
            if not signals:
                await update.message.reply_text("📊 No high-confidence signals found at the moment.")
                return
            
            # Format signals response
            response = self.messages.format_signals(signals)
            
            # Create keyboard for signal actions
            keyboard = create_signals_keyboard(signals[:5])  # Show top 5 signals
            
            await update.message.reply_text(response, parse_mode=ParseMode.MARKDOWN, reply_markup=keyboard)
            
        except Exception as e:
            logger.error(f"Error in signals_command: {e}")
            await update.message.reply_text("❌ An error occurred while fetching signals. Please try again.")
    
    async def chart_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /chart command to generate price charts"""
        try:
            args = context.args
            if not args:
                await update.message.reply_text("Please specify a symbol. Example: `/chart BTC_USDT`", parse_mode=ParseMode.MARKDOWN)
                return
            
            symbol = validate_symbol(args[0])
            if not symbol:
                await update.message.reply_text("❌ Invalid symbol format. Use format like BTC_USDT")
                return
            
            await update.message.reply_text(f"📊 Generating chart for {symbol}...")
            
            # Generate chart
            chart_data = await self.chart_generator.generate_chart(symbol)
            
            if chart_data:
                await update.message.reply_photo(
                    photo=chart_data,
                    caption=f"📊 {symbol} Price Chart with Technical Indicators"
                )
            else:
                await update.message.reply_text(f"❌ Unable to generate chart for {symbol}")
                
        except Exception as e:
            logger.error(f"Error in chart_command: {e}")
            await update.message.reply_text("❌ An error occurred while generating the chart. Please try again.")
    
    async def handle_screenshot(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle screenshot analysis"""
        try:
            await update.message.reply_text("🧠 AI Chart Analysis is coming soon! This feature will analyze your screenshot and provide trading recommendations.")
        except Exception as e:
            logger.error(f"Error in handle_screenshot: {e}")
    
    async def handle_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle text messages"""
        text = update.message.text.strip()
        
        # Check if it's a symbol request
        if text.upper().endswith('_USDT') or text.upper().startswith('BTC') or text.upper().startswith('ETH'):
            # Treat as chart request
            symbol = validate_symbol(text)
            if symbol:
                await self.chart_command(update, context)
                return
        
        # Default response
        await update.message.reply_text(
            "I didn't understand that. Use /help to see available commands.",
            reply_markup=create_trade_keyboard()
        )
    
    async def handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle inline keyboard callbacks"""
        query = update.callback_query
        await query.answer()
        
        try:
            action, *params = query.data.split('_')
            
            if action == "chart":
                symbol = params[0] if params else "BTC_USDT"
                await self.generate_chart_callback(query, symbol)
                
            elif action == "status":
                trade_id = int(params[0]) if params else None
                await self.show_trade_status_callback(query, trade_id)
                
            elif action == "close":
                trade_id = int(params[0]) if params else None
                await self.close_trade_callback(query, trade_id)
                
            elif action == "refresh":
                await self.refresh_status_callback(query)
                
            elif action == "signals":
                await self.show_signals_callback(query)
                
        except Exception as e:
            logger.error(f"Error in handle_callback: {e}")
            await query.message.reply_text("❌ An error occurred processing your request.")
    
    async def generate_chart_callback(self, query, symbol: str):
        """Generate chart from callback"""
        await query.message.reply_text(f"📊 Generating chart for {symbol}...")
        
        chart_data = await self.chart_generator.generate_chart(symbol)
        if chart_data:
            await query.message.reply_photo(
                photo=chart_data,
                caption=f"📊 {symbol} Price Chart"
            )
    
    async def close_trade_callback(self, query, trade_id: int):
        """Close trade from callback"""
        user_id = query.from_user.id
        
        trade = await self.db_manager.get_trade(trade_id)
        if not trade or trade.user_id != user_id:
            await query.message.reply_text("❌ Trade not found or access denied.")
            return
        
        if trade.status != TradeStatus.OPEN:
            await query.message.reply_text("❌ Trade is already closed.")
            return
        
        current_price = await self.mexc_client.get_current_price(trade.symbol)
        if current_price:
            await self.trading_manager.close_trade(trade, current_price)
            pnl_data = await self.trading_manager.calculate_pnl(trade, current_price)
            
            response = self.messages.format_trade_closed(trade, current_price, pnl_data)
            await query.message.reply_text(response, parse_mode=ParseMode.MARKDOWN)
    
    async def error_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Global error handler"""
        logger.error(f"Update {update} caused error {context.error}")
        
        if update and update.effective_message:
            await update.effective_message.reply_text(
                "❌ An unexpected error occurred. Please try again or contact support."
            )
    
    async def update_all_positions(self):
        """Background task to update all open positions"""
        try:
            all_open_trades = await self.db_manager.get_all_open_trades()
            
            for trade in all_open_trades:
                current_price = await self.mexc_client.get_current_price(trade.symbol)
                if current_price:
                    pnl_data = await self.trading_manager.calculate_pnl(trade, current_price)
                    
                    # Check for significant PnL changes and send alerts
                    await self.check_pnl_alerts(trade, pnl_data)
                    
        except Exception as e:
            logger.error(f"Error updating positions: {e}")
    
    async def scan_for_signals(self):
        """Background task to scan for high-confidence signals"""
        try:
            signals = await self.signal_detector.scan_all_symbols()
            ultra_high_signals = [s for s in signals if s['confidence'] >= 90]
            
            if ultra_high_signals:
                # Send alerts to all users (implement user subscription system)
                logger.info(f"Found {len(ultra_high_signals)} ultra-high confidence signals")
                
        except Exception as e:
            logger.error(f"Error scanning for signals: {e}")
    
    async def check_pnl_alerts(self, trade: Trade, pnl_data: Dict):
        """Check if PnL alerts should be sent"""
        # Implement PnL alert logic based on user preferences
        pass
