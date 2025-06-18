#!/usr/bin/env python3
"""
AI-Powered MEXC Futures Telegram Signals Bot
Main application entry point
"""

import asyncio
import logging
import os
from datetime import datetime

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from config import Config
from bot.handlers import BotHandlers
from core.database import DatabaseManager
from core.mexc_api import MEXCClient
from signals.detector import SignalDetector
from utils.logger import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

class TradingBot:
    def __init__(self):
        self.config = Config()
        self.db_manager = DatabaseManager()
        self.mexc_client = MEXCClient()
        self.signal_detector = SignalDetector(self.mexc_client)
        self.handlers = BotHandlers(self.db_manager, self.mexc_client, self.signal_detector)
        self.scheduler = AsyncIOScheduler()
        
    async def initialize(self):
        """Initialize bot components"""
        logger.info("Initializing trading bot...")
        
        # Initialize database
        await self.db_manager.initialize()
        
        # Initialize MEXC client
        await self.mexc_client.initialize()
        
        # Setup scheduled tasks
        self.setup_scheduler()
        
        logger.info("Bot initialization complete")
    
    def setup_scheduler(self):
        """Setup background scheduled tasks"""
        # Update prices every 30 seconds
        self.scheduler.add_job(
            self.handlers.update_all_positions,
            'interval',
            seconds=30,
            id='update_positions'
        )
        
        # Scan for signals every 5 minutes
        self.scheduler.add_job(
            self.handlers.scan_for_signals,
            'interval',
            minutes=5,
            id='scan_signals'
        )
        
        # Cleanup old data daily
        self.scheduler.add_job(
            self.db_manager.cleanup_old_data,
            'cron',
            hour=0,
            minute=0,
            id='cleanup_data'
        )
    
    def create_application(self):
        """Create and configure Telegram application"""
        application = Application.builder().token(self.config.TELEGRAM_BOT_TOKEN).build()
        
        # Command handlers
        application.add_handler(CommandHandler("start", self.handlers.start_command))
        application.add_handler(CommandHandler("help", self.handlers.help_command))
        application.add_handler(CommandHandler("open", self.handlers.open_trade_command))
        application.add_handler(CommandHandler("close", self.handlers.close_trade_command))
        application.add_handler(CommandHandler("status", self.handlers.status_command))
        application.add_handler(CommandHandler("signals", self.handlers.signals_command))
        application.add_handler(CommandHandler("chart", self.handlers.chart_command))
        
        # Message handlers
        application.add_handler(MessageHandler(filters.PHOTO, self.handlers.handle_screenshot))
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handlers.handle_text))
        
        # Callback query handler for inline keyboards
        from telegram.ext import CallbackQueryHandler
        application.add_handler(CallbackQueryHandler(self.handlers.handle_callback))
        
        # Error handler
        application.add_error_handler(self.handlers.error_handler)
        
        return application
    
    async def run(self):
        """Run the bot"""
        try:
            await self.initialize()
            
            application = self.create_application()
            
            # Start scheduler
            self.scheduler.start()
            logger.info("Background scheduler started")
            
            # Start bot
            logger.info("Starting Telegram bot...")
            await application.initialize()
            await application.start()
            await application.updater.start_polling()
            
            logger.info("Bot is running. Press Ctrl+C to stop.")
            
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
            
        await self.db_manager.close()
        logger.info("Bot shutdown complete")

def main():
    """Main entry point"""
    bot = TradingBot()
    
    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Bot crashed: {e}")
        raise

if __name__ == "__main__":
    main()
