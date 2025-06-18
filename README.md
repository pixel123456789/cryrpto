# AI-Powered MEXC Futures Telegram Bot

A fully autonomous AI-enhanced Telegram bot for MEXC USDT-Margined Futures trading with real-time market analysis, automated signal generation, trade logging, and comprehensive risk management.

## 🚀 Features

### Core Trading System
- **Real-time Trade Logging**: Log and track LONG/SHORT positions with automatic TP/SL calculation
- **Live PnL Monitoring**: Real-time profit/loss tracking with percentage and USD values
- **Risk Management**: Automatic position sizing and risk/reward ratio calculations
- **Portfolio Overview**: Complete portfolio management with trade history

### AI-Powered Analysis
- **Smart Signal Detection**: AI algorithms analyzing RSI, MACD, Moving Averages, and Volume
- **Confidence Scoring**: Each signal rated with confidence percentage (70%+ threshold)
- **Multi-timeframe Analysis**: Comprehensive technical analysis across different timeframes
- **Market Sentiment**: Real-time market sentiment indicators

### Advanced Charts
- **Technical Analysis Charts**: Candlestick charts with indicators (RSI, MACD, Bollinger Bands)
- **Trade Visualization**: Entry/exit points marked on charts
- **Multiple Timeframes**: 1m, 5m, 15m, 1h, 4h, 1d chart generation
- **Custom Indicators**: Moving averages, support/resistance levels

### Telegram Integration
- **Interactive Commands**: Easy-to-use command system
- **Real-time Notifications**: Instant alerts for TP/SL hits and high-confidence signals
- **Inline Keyboards**: Quick action buttons for trading operations
- **Multi-user Support**: Individual user portfolios and settings

## 📊 Trading Commands

### Basic Commands
```
/start          - Welcome message and quick start guide
/help           - Complete command reference
/status         - View all open trades and portfolio PnL
/signals        - Get latest AI trading signals
```

### Trade Management
```
/open SYMBOL TYPE PRICE LEVERAGE CAPITAL
Example: /open BTC_USDT LONG 42000 10 100

Parameters:
- SYMBOL: Trading pair (BTC_USDT, ETH_USDT, SOL_USDT, etc.)
- TYPE: LONG or SHORT
- PRICE: Entry price in USDT
- LEVERAGE: 1-100x leverage
- CAPITAL: Capital amount in USDT
```

### Analysis Tools
```
/chart SYMBOL   - Generate technical analysis chart
/chart BTC_USDT - Example: Bitcoin chart with indicators
```

## 🏗️ Architecture

### Core Components
- **Database Manager**: SQLite database for trade storage and user management
- **MEXC API Client**: Real-time market data and price feeds
- **Trading Manager**: PnL calculations, TP/SL logic, position management
- **Signal Detector**: AI-powered technical analysis and signal generation
- **Chart Generator**: Technical analysis visualization with matplotlib
- **Telegram Bot**: User interface and command handling

### Background Tasks
- **Position Monitoring**: Updates all open positions every 30 seconds
- **Signal Scanning**: Scans for high-confidence signals every 5 minutes
- **Auto TP/SL**: Automatically closes trades when targets are hit
- **Risk Alerts**: Notifications for significant PnL changes

## 🛠️ Setup Instructions

### Prerequisites
- Python 3.11+
- TELEGRAM_BOT_TOKEN environment variable

### Quick Start
1. **Get Your Telegram Bot Token**
   - Message @BotFather on Telegram
   - Create a new bot with `/newbot`
   - Copy the bot token

2. **Set Environment Variable**
   ```bash
   export TELEGRAM_BOT_TOKEN="your_bot_token_here"
   ```

3. **Run the Bot**
   ```bash
   python working_telegram_bot.py
   ```

4. **Start Trading**
   - Find your bot on Telegram
   - Send `/start` to begin
   - Use `/open BTC_USDT LONG 42000 10 100` to log your first trade

## 📈 Trading Examples

### Opening a Long Position
```
/open BTC_USDT LONG 42000 10 100
```
- Symbol: BTC/USDT
- Direction: LONG (bullish)
- Entry Price: $42,000
- Leverage: 10x
- Capital: $100

**Result**: Automatic TP/SL calculation, real-time PnL tracking

### Getting Trading Signals
```
/signals
```
**Sample Output**:
```
🔍 AI Trading Signals

1. 🟢 BTC_USDT - LONG
   🔥 Confidence: 92.5%
   💡 RSI oversold + MACD bullish crossover

2. 🔴 ETH_USDT - SHORT
   ⭐ Confidence: 87.3%
   💡 Resistance rejection + volume spike

⚡ Signals update every 5 minutes
🎯 Only showing signals with >70% confidence
```

### Checking Portfolio Status
```
/status
```
**Sample Output**:
```
📊 Your Open Trades:

🟢 BTC_USDT - LONG
💰 Entry: $42,000.00 | Current: $43,500.00
📈 PnL: +3.57% (+$35.70)
🎯 TP: $44,100.00 | 🛡️ SL: $40,320.00

💰 Total Portfolio PnL: +$35.70
```

## 🔧 Technical Details

### Signal Detection Algorithm
- **RSI Analysis**: Identifies overbought/oversold conditions
- **MACD Signals**: Bullish/bearish momentum crossovers
- **Moving Average Trends**: 20, 50, 200 EMA analysis
- **Volume Confirmation**: Volume spike validation
- **Confluence Scoring**: Multiple indicator agreement

### Risk Management
- **Automatic TP/SL**: 5% take profit, 4% stop loss (configurable)
- **Position Sizing**: Risk-based capital allocation
- **Leverage Limits**: 1-100x with risk warnings
- **Portfolio Monitoring**: Real-time exposure tracking

### Performance Features
- **Real-time Updates**: 30-second position monitoring
- **Background Processing**: Non-blocking signal detection
- **Error Handling**: Robust error recovery and logging
- **Rate Limiting**: API call optimization for stability

## 🚀 Current Status

**✅ FULLY OPERATIONAL**
- Core trading system: Running
- Database: Initialized and connected
- Signal detection: Active (scanning every 5 minutes)
- Position monitoring: Active (updating every 30 seconds)
- Chart generation: Ready
- Background tasks: Running

The bot is ready for live trading with Telegram integration available when TELEGRAM_BOT_TOKEN is provided.

## 📞 Support

For setup assistance or trading questions:
1. Check the `/help` command in the bot
2. Review this documentation
3. Ensure TELEGRAM_BOT_TOKEN is properly set

---

**Disclaimer**: This bot is for educational and trading assistance purposes. Always conduct your own research and risk management when trading cryptocurrency futures.