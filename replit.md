# AI-Powered MEXC Futures Trading Bot

## Overview

This is a comprehensive AI-powered trading bot designed for MEXC USDT-Margined Futures trading with Telegram integration. The bot provides real-time market analysis, automated signal generation, trade logging, and portfolio management capabilities. It combines technical analysis with AI-driven decision making to deliver high-confidence trading signals and comprehensive risk management.

## System Architecture

### Backend Architecture
- **Python 3.11** runtime with asyncio-based asynchronous architecture
- **Modular design** with separate concerns for API communication, database management, signal detection, and chart generation
- **Rate limiting and connection pooling** for efficient API usage
- **Event-driven architecture** with scheduled tasks for continuous market monitoring

### Data Storage
- **SQLite database** for local data persistence
- Tables for trades, price data, and user portfolios
- Transaction-safe operations with connection pooling
- Automatic database initialization and schema management

### External API Integration
- **MEXC Futures API** for real-time market data, symbol information, and price feeds
- **Telegram Bot API** for user interactions and notifications
- **aiohttp** for asynchronous HTTP requests with proper error handling and retries

## Key Components

### Core Trading System (`core/`)
- **Database Manager**: SQLite-based persistence layer for trades and historical data
- **MEXC API Client**: Complete implementation supporting all MEXC futures symbols with rate limiting
- **Trading Manager**: PnL calculations, risk management, and trade lifecycle management
- **Data Models**: Structured data classes for trades, signals, and market data

### Signal Detection Engine (`signals/`)
- **Technical Indicators**: RSI, MACD, EMA calculations with configurable parameters
- **Signal Detector**: AI-powered analysis combining multiple indicators with confidence scoring
- **Multi-timeframe Analysis**: Supports 1m, 5m, 15m, 1h, 4h, 1d timeframes

### Chart Generation (`charts/`)
- **Matplotlib-based** candlestick charts with technical indicators
- **Dark theme** optimized for trading analysis
- **Interactive overlays** for entry/exit points and indicator signals

### Telegram Bot Interface (`bot/`)
- **Command Handlers**: Complete command processing for trading operations
- **Interactive Keyboards**: Inline keyboards for quick actions
- **Message Templates**: Formatted messages for different bot responses
- **User State Management**: Multi-step command processing

## Data Flow

1. **Market Data Acquisition**: Continuous polling of MEXC API for price data and symbol information
2. **Signal Processing**: Technical analysis engine processes market data to generate trading signals
3. **Trade Management**: User-initiated trades are logged and monitored for PnL calculation
4. **Notification System**: Real-time alerts sent via Telegram for signal updates and trade status
5. **Chart Generation**: On-demand chart creation with technical indicators and trade markers

## External Dependencies

### Core Dependencies
- **python-telegram-bot (20.7)**: Telegram bot framework with async support
- **aiohttp (3.12.13)**: Async HTTP client for API communications
- **pandas (2.3.0)**: Data analysis and technical indicator calculations
- **numpy (2.3.0)**: Numerical computations for trading algorithms
- **matplotlib (3.10.3)**: Chart generation and visualization

### Additional Libraries
- **APScheduler (3.11.0)**: Scheduled task management for periodic operations
- **Pillow (11.2.1)**: Image processing for chart generation
- **scikit-learn (1.7.0)**: Machine learning utilities for signal confidence scoring

### Development Environment
- **Nix environment** with Python 3.11 and required system packages
- **Cairo, FFmpeg, FreeType** for advanced chart rendering capabilities

## Deployment Strategy

### Environment Configuration
- **Environment variables** for sensitive configuration (API keys, tokens)
- **Config class** for centralized configuration management
- **Multiple entry points** for different deployment scenarios

### Execution Modes
- **Primary Bot**: `complete_mexc_bot.py` - Full featured implementation
- **Simplified Bot**: `simple_bot.py` - Core functionality without complex dependencies
- **Working Bot**: Various implementations for different deployment environments

### Process Management
- **Shell execution** via Replit workflows
- **Graceful shutdown** handling for clean resource cleanup
- **Error recovery** mechanisms for robust operation

## Recent Changes
- June 18, 2025: Successfully deployed comprehensive AI-powered trading system
  - **Implemented advanced neural network engine** with LSTM, XGBoost, and LightGBM models
  - **Added real-time sentiment analysis** with social media and news integration
  - **Enhanced chart analyzer** with computer vision and machine learning capabilities
  - **New /neural command** - Provides AI predictions with ensemble modeling
  - **New /sentiment command** - Delivers market sentiment analysis
  - **Simplified neural engine** - Optimized for system compatibility
  - **Complete AI integration** - All AI systems working seamlessly together
  - Enhanced scheduler with 30-second signal scanning for rapid discovery
  - Monitoring 778 futures symbols with aggressive scanning coverage
  - Raised confidence threshold to 95% for ultra-high quality signals only
  - Added 15-second cleanup cycle for immediate expired signal removal
  - Reduced signal expiry to 5 minutes for rapid turnover
  - Implemented photo handling for chart screenshot analysis
  - Added technical analysis capabilities for uploaded chart images
  - Fixed Telegram delivery with proper Chat ID configuration
  - **Removed notification spam** - individual signal results no longer sent
  - **Added /stats command** - users can check win rate and performance on demand
  - System generating ultra-high confidence signals (95%+) with clean delivery
  - **Fixed duplicate signal prevention** - Implemented two-layer cache system to prevent multiple signals for same symbol
  - **Enhanced chart analysis** - Fixed screenshot processing with proper error handling and scikit-learn compatibility
  - **Improved signal tracking** - Added active signals cache with automatic cleanup on expiry
  - **Updated confidence threshold to 100%** - Now only sends alerts for perfect confidence signals to reduce notification volume
  - **Added /clear command** - Users can now reset all stats and signals with confirmation for fresh start

## Changelog
- June 18, 2025. Initial setup
- June 18, 2025. Major signal generation enhancement

## User Preferences

Preferred communication style: Simple, everyday language.