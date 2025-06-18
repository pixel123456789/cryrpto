#!/usr/bin/env python3
"""
Working MEXC Futures Telegram Bot - Simplified Implementation
"""

import asyncio
import logging
import os
import sys
import json
from datetime import datetime
from typing import Dict, Any, Optional

# Simple telegram integration without complex imports
try:
    import telegram
    from telegram import Bot
    from telegram.ext import Application, CommandHandler, CallbackQueryHandler
    TELEGRAM_AVAILABLE = True
    print("Telegram integration available")
except ImportError as e:
    print(f"Telegram integration not available: {e}")
    TELEGRAM_AVAILABLE = False
    Bot = None
    Application = None
    CommandHandler = None
    CallbackQueryHandler = None

from apscheduler.schedulers.asyncio import AsyncIOScheduler

# Import our bot components
from core.database import DatabaseManager
from core.mexc_api import MEXCClient
from core.trading import TradingManager
from core.models import Trade, TradeStatus, TradeType
from signals.detector import SignalDetector
from charts.generator import ChartGenerator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

class WorkingMEXCBot:
    def __init__(self):
        # Hardcoded credentials
        self.bot_token = "8091002773:AAEcIdyAc6FGoMFu08DErj2DLmTyAjMEkTk"
        
        # Initialize core components
        self.db_manager = None
        self.mexc_client = None
        self.trading_manager = None
        self.signal_detector = None
        self.chart_generator = None
        self.scheduler = AsyncIOScheduler()
        
    async def initialize(self):
        """Initialize all bot components"""
        logger.info("Initializing MEXC Futures Trading Bot...")
        
        try:
            # Initialize database
            self.db_manager = DatabaseManager()
            await self.db_manager.initialize()
            logger.info("Database initialized")
            
            # Initialize MEXC client
            self.mexc_client = MEXCClient()
            await self.mexc_client.initialize()
            logger.info("MEXC API client initialized")
            
            # Initialize trading manager
            self.trading_manager = TradingManager(self.db_manager, self.mexc_client)
            logger.info("Trading manager initialized")
            
            # Initialize signal detector
            self.signal_detector = SignalDetector(self.mexc_client)
            logger.info("Signal detector initialized")
            
            # Initialize chart generator
            self.chart_generator = ChartGenerator(self.mexc_client)
            logger.info("Chart generator initialized")
            
            # Setup scheduler
            self.setup_scheduler()
            
            return True
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return False
    
    def setup_scheduler(self):
        """Setup background scheduled tasks"""
        # Update positions every 30 seconds
        self.scheduler.add_job(
            self.update_all_positions,
            'interval',
            seconds=30,
            id='update_positions'
        )
        
        # Scan for signals every 5 minutes
        self.scheduler.add_job(
            self.scan_for_signals,
            'interval',
            minutes=5,
            id='scan_signals'
        )
    
    async def start_command(self, update, context):
        """Handle /start command"""
        if not TELEGRAM_AVAILABLE:
            return
            
        user_id = update.effective_user.id
        username = update.effective_user.username or "Unknown"
        
        logger.info(f"User {username} ({user_id}) started the bot")
        
        welcome_message = """
🚀 **Welcome to AI-Powered MEXC Futures Bot!**

Your 24/7 crypto trading assistant for MEXC USDT-Margined Futures.

**Quick Start:**
• `/open SYMBOL LONG/SHORT PRICE LEVERAGE CAPITAL` - Log a trade
• `/status` - View your open trades
• `/signals` - Get AI trading signals
• `/chart SYMBOL` - Generate price charts

**Example:**
`/open BTC_USDT LONG 42000 10 100`

Ready to start trading! 📈
        """
        
        await update.message.reply_text(welcome_message, parse_mode='Markdown')
    
    async def help_command(self, update, context):
        """Handle /help command"""
        if not TELEGRAM_AVAILABLE:
            return
            
        help_message = """
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

Need help? Just send a message!
        """
        await update.message.reply_text(help_message, parse_mode='Markdown')
    
    async def signals_command(self, update, context):
        """Handle /signals command"""
        if not TELEGRAM_AVAILABLE:
            return
            
        try:
            await update.message.reply_text("🔍 Scanning for trading signals... Please wait.")
            
            # Get trading signals
            signals = await self.signal_detector.scan_all_symbols(10)
            
            if not signals:
                await update.message.reply_text("📊 No high-confidence signals found at the moment.")
                return
            
            # Format signals response
            message = "🔍 **AI Trading Signals**\n\n"
            
            for i, signal in enumerate(signals[:5], 1):
                confidence_emoji = "🔥" if signal['confidence'] >= 90 else "⭐" if signal['confidence'] >= 80 else "✅"
                action_emoji = "🟢" if signal['action'].upper() == 'LONG' else "🔴"
                
                message += f"{i}. {action_emoji} **{signal['symbol']}** - {signal['action'].upper()}\n"
                message += f"   {confidence_emoji} Confidence: {signal['confidence']:.1f}%\n"
                message += f"   💡 {signal['reason']}\n\n"
            
            message += "⚡ **Signals update every 5 minutes**\n"
            message += "🎯 Only showing signals with >70% confidence"
            
            await update.message.reply_text(message, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Error in signals_command: {e}")
            await update.message.reply_text("❌ An error occurred while fetching signals. Please try again.")
    
    async def status_command(self, update, context):
        """Handle /status command"""
        if not TELEGRAM_AVAILABLE:
            return
            
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
                    
                    direction_emoji = "🟢" if trade.trade_type == TradeType.LONG else "🔴"
                    pnl_emoji = "📈" if pnl_data['pnl_percentage'] >= 0 else "📉"
                    
                    status_msg = f"""
{direction_emoji} **{trade.symbol}** - {trade.trade_type.value}
💰 Entry: ${trade.entry_price:.2f} | Current: ${current_price:.2f}
{pnl_emoji} **PnL: {pnl_data['pnl_percentage']:+.2f}% (${pnl_data['pnl_usd']:+.2f})**
🎯 TP: ${trade.take_profit:.2f} | 🛡️ SL: ${trade.stop_loss:.2f}
                    """
                    status_messages.append(status_msg)
                else:
                    status_messages.append(f"❌ {trade.symbol}: Unable to get current price")
            
            # Combine all status messages
            response = "📊 **Your Open Trades:**\n" + "\n".join(status_messages)
            response += f"\n\n💰 **Total Portfolio PnL: ${total_pnl:+.2f}**"
            
            await update.message.reply_text(response, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Error in status_command: {e}")
            await update.message.reply_text("❌ An error occurred while fetching trade status. Please try again.")
    
    async def update_all_positions(self):
        """Background task to update all open positions"""
        try:
            open_trades = await self.db_manager.get_all_open_trades()
            for trade in open_trades:
                current_price = await self.mexc_client.get_current_price(trade.symbol)
                if current_price:
                    # Check for TP/SL hits
                    hit_type = await self.trading_manager.check_tp_sl_hit(trade, current_price)
                    if hit_type:
                        await self.trading_manager.close_trade(trade, current_price, f"{hit_type} Hit")
                        logger.info(f"Auto-closed trade {trade.id} due to {hit_type} hit")
        except Exception as e:
            logger.error(f"Error updating positions: {e}")
    
    async def scan_for_signals(self):
        """Background task to scan for high-confidence signals"""
        try:
            signals = await self.signal_detector.scan_all_symbols(20)
            ultra_high_signals = [s for s in signals if s['confidence'] >= 90]
            
            if ultra_high_signals:
                logger.info(f"Found {len(ultra_high_signals)} ultra-high confidence signals")
                
        except Exception as e:
            logger.error(f"Error scanning for signals: {e}")
    
    def create_application(self):
        """Create and configure Telegram application"""
        if not TELEGRAM_AVAILABLE or not self.bot_token:
            logger.warning("Telegram application cannot be created - missing requirements")
            return None
        
        try:
            application = Application.builder().token(self.bot_token).build()
            
            # Command handlers
            application.add_handler(CommandHandler("start", self.start_command))
            application.add_handler(CommandHandler("help", self.help_command))
            application.add_handler(CommandHandler("signals", self.signals_command))
            application.add_handler(CommandHandler("status", self.status_command))
            
            return application
        except Exception as e:
            logger.error(f"Failed to create Telegram application: {e}")
            return None
    
    async def run(self):
        """Run the complete bot"""
        try:
            # Initialize components
            if not await self.initialize():
                logger.error("Failed to initialize bot components")
                return
            
            # Start scheduler
            self.scheduler.start()
            logger.info("Background scheduler started")
            
            if not TELEGRAM_AVAILABLE or not self.bot_token:
                logger.info("Running core trading system without Telegram integration")
                logger.info("To enable Telegram features, set TELEGRAM_BOT_TOKEN environment variable")
                
                # Keep running and show system status
                while True:
                    try:
                        # Check system status every minute
                        btc_price = await self.mexc_client.get_current_price('BTC_USDT')
                        signals = await self.signal_detector.scan_all_symbols(5)
                        signal_count = len(signals) if signals else 0
                        open_trades = await self.db_manager.get_all_open_trades()
                        trade_count = len(open_trades) if open_trades else 0
                        
                        if btc_price:
                            logger.info(f"System Status - BTC: ${btc_price:.2f} | Signals: {signal_count} | Open Trades: {trade_count}")
                        else:
                            logger.info(f"System Status - API: Connecting... | Signals: {signal_count} | Open Trades: {trade_count}")
                        
                    except Exception as e:
                        logger.error(f"System check failed: {e}")
                    
                    await asyncio.sleep(60)
                return
            
            # Create Telegram application
            application = self.create_application()
            if not application:
                logger.error("Failed to create Telegram application")
                return
            
            # Start bot
            logger.info("Starting Telegram bot...")
            await application.initialize()
            await application.start()
            await application.updater.start_polling()
            
            logger.info("✅ MEXC Futures Telegram Bot is running and ready!")
            logger.info("Send /start to your bot to begin trading")
            
            # Keep the bot running
            try:
                await asyncio.Future()  # Run forever
            except KeyboardInterrupt:
                logger.info("Received interrupt signal, shutting down...")
            
        except Exception as e:
            logger.error(f"Fatal error: {e}")
            raise
        finally:
            await self.shutdown()
    
    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down bot...")
        
        if hasattr(self, 'scheduler') and self.scheduler.running:
            self.scheduler.shutdown()
        
        if self.db_manager:
            await self.db_manager.close()
            
        if self.mexc_client:
            await self.mexc_client.close()
            
        logger.info("Bot shutdown complete")

async def main():
    """Main entry point"""
    bot = WorkingMEXCBot()
    
    try:
        await bot.run()
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Bot crashed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())