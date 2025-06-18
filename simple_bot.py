#!/usr/bin/env python3
"""
Simple MEXC Futures Trading Bot - Main Entry Point
"""

import asyncio
import logging
import os
import sys
from datetime import datetime

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

class SimpleTradingBot:
    def __init__(self):
        # Hardcoded credentials
        self.bot_token = "8091002773:AAEcIdyAc6FGoMFu08DErj2DLmTyAjMEkTk"
        
        logger.info("🚀 MEXC Futures Trading Bot - Starting...")
        logger.info(f"📊 Bot Token: {self.bot_token[:10]}...")
        
    async def initialize_components(self):
        """Initialize bot components without telegram imports"""
        try:
            # Initialize database
            from core.database import DatabaseManager
            self.db_manager = DatabaseManager()
            await self.db_manager.initialize()
            logger.info("✅ Database initialized")
            
            # Initialize MEXC client
            from core.mexc_api import MEXCClient
            self.mexc_client = MEXCClient()
            await self.mexc_client.initialize()
            logger.info("✅ MEXC API client initialized")
            
            # Initialize signal detector
            from signals.detector import SignalDetector
            self.signal_detector = SignalDetector(self.mexc_client)
            logger.info("✅ Signal detector initialized")
            
            # Test API connectivity
            symbols = await self.mexc_client.get_popular_symbols(5)
            logger.info(f"✅ API test successful - Got {len(symbols)} symbols")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Component initialization failed: {e}")
            return False
    
    async def test_signal_detection(self):
        """Test signal detection functionality"""
        try:
            logger.info("🔍 Testing signal detection...")
            signals = await self.signal_detector.scan_all_symbols(5)
            logger.info(f"📊 Generated {len(signals)} trading signals")
            
            for signal in signals[:3]:
                logger.info(f"🎯 {signal['symbol']} {signal['action']} - {signal['confidence']:.1f}% confidence")
            
        except Exception as e:
            logger.error(f"❌ Signal detection test failed: {e}")
    
    async def test_chart_generation(self):
        """Test chart generation functionality"""
        try:
            logger.info("📈 Testing chart generation...")
            from charts.generator import ChartGenerator
            chart_gen = ChartGenerator(self.mexc_client)
            
            chart_data = await chart_gen.generate_chart("BTC_USDT")
            if chart_data:
                logger.info(f"✅ Chart generated successfully - {len(chart_data)} bytes")
            else:
                logger.warning("⚠️ Chart generation returned no data")
                
        except Exception as e:
            logger.error(f"❌ Chart generation test failed: {e}")
    
    async def run_tests(self):
        """Run comprehensive tests of bot functionality"""
        logger.info("="*60)
        logger.info("🧪 RUNNING COMPREHENSIVE BOT TESTS")
        logger.info("="*60)
        
        # Test 1: Initialize components
        if not await self.initialize_components():
            logger.error("❌ Component initialization failed - stopping tests")
            return False
        
        # Test 2: Signal detection
        await self.test_signal_detection()
        
        # Test 3: Chart generation
        await self.test_chart_generation()
        
        # Test 4: Database operations
        await self.test_database_operations()
        
        logger.info("="*60)
        logger.info("✅ ALL TESTS COMPLETED")
        logger.info("="*60)
        return True
    
    async def test_database_operations(self):
        """Test database functionality"""
        try:
            logger.info("💾 Testing database operations...")
            
            # Test user settings
            user_settings = await self.db_manager.get_user_settings(12345)
            logger.info(f"✅ User settings retrieved: {user_settings['default_leverage']}x leverage")
            
            # Test trading stats
            stats = await self.db_manager.get_trading_stats(12345)
            logger.info(f"✅ Trading stats: {stats['total_trades']} total trades")
            
        except Exception as e:
            logger.error(f"❌ Database test failed: {e}")

async def main():
    """Main entry point"""
    bot = SimpleTradingBot()
    
    try:
        # Run comprehensive tests
        success = await bot.run_tests()
        
        if success:
            logger.info("🎉 Bot is ready for Telegram integration!")
            logger.info("💡 Next steps:")
            logger.info("   1. Resolve telegram import conflicts")
            logger.info("   2. Add Telegram bot handlers")
            logger.info("   3. Start accepting user commands")
        else:
            logger.error("❌ Bot tests failed - check configuration")
        
        # Keep running for monitoring
        logger.info("🔄 Bot will continue running for monitoring...")
        while True:
            await asyncio.sleep(60)
            logger.info(f"💓 Bot heartbeat - {datetime.now().strftime('%H:%M:%S')}")
            
    except KeyboardInterrupt:
        logger.info("👋 Bot stopped by user")
    except Exception as e:
        logger.error(f"💥 Bot crashed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())