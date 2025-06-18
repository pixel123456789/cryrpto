"""
Trading logic and PnL calculations
"""

import logging
from datetime import datetime
from typing import Dict, Optional, Tuple, Any

from core.models import Trade, TradeType, TradeStatus
from core.database import DatabaseManager
from core.mexc_api import MEXCClient
from config import Config

logger = logging.getLogger(__name__)

class TradingManager:
    """Manages trading operations and calculations"""
    
    def __init__(self, db_manager: DatabaseManager, mexc_client: MEXCClient):
        self.db_manager = db_manager
        self.mexc_client = mexc_client
        self.config = Config()
    
    async def calculate_tp_sl(self, trade: Trade) -> None:
        """Calculate take profit and stop loss levels"""
        try:
            # Get symbol info to determine tick size and other constraints
            symbol_info = await self.mexc_client.get_symbol_info(trade.symbol)
            
            # Use default percentages from config
            tp_percentage = self.config.DEFAULT_TP_PERCENTAGE
            sl_percentage = self.config.DEFAULT_SL_PERCENTAGE
            
            # Adjust based on leverage (higher leverage = tighter stops)
            if trade.leverage > 20:
                tp_percentage *= 0.7
                sl_percentage *= 0.7
            elif trade.leverage > 50:
                tp_percentage *= 0.5
                sl_percentage *= 0.5
            
            if trade.trade_type == TradeType.LONG:
                # For LONG trades
                trade.take_profit = trade.entry_price * (1 + tp_percentage / 100)
                trade.stop_loss = trade.entry_price * (1 - sl_percentage / 100)
            else:
                # For SHORT trades
                trade.take_profit = trade.entry_price * (1 - tp_percentage / 100)
                trade.stop_loss = trade.entry_price * (1 + sl_percentage / 100)
            
            trade.tp_percentage = tp_percentage
            trade.sl_percentage = sl_percentage
            
            logger.info(f"Calculated TP/SL for {trade.symbol}: TP={trade.take_profit:.6f}, SL={trade.stop_loss:.6f}")
            
        except Exception as e:
            logger.error(f"Error calculating TP/SL for trade {trade.id}: {e}")
            # Set default values if calculation fails
            if trade.trade_type == TradeType.LONG:
                trade.take_profit = trade.entry_price * 1.03
                trade.stop_loss = trade.entry_price * 0.98
            else:
                trade.take_profit = trade.entry_price * 0.97
                trade.stop_loss = trade.entry_price * 1.02
            
            trade.tp_percentage = 3.0
            trade.sl_percentage = 2.0
    
    async def calculate_pnl(self, trade: Trade, current_price: float) -> Dict[str, float]:
        """Calculate current PnL for a trade"""
        try:
            if trade.trade_type == TradeType.LONG:
                # For LONG: profit when price goes up
                price_change_percentage = ((current_price - trade.entry_price) / trade.entry_price) * 100
            else:
                # For SHORT: profit when price goes down
                price_change_percentage = ((trade.entry_price - current_price) / trade.entry_price) * 100
            
            # Apply leverage
            pnl_percentage = price_change_percentage * trade.leverage
            
            # Calculate PnL in USD
            pnl_usd = trade.capital * (pnl_percentage / 100)
            
            return {
                'pnl_percentage': pnl_percentage,
                'pnl_usd': pnl_usd,
                'price_change_percentage': price_change_percentage,
                'current_price': current_price,
                'entry_price': trade.entry_price
            }
            
        except Exception as e:
            logger.error(f"Error calculating PnL for trade {trade.id}: {e}")
            return {
                'pnl_percentage': 0.0,
                'pnl_usd': 0.0,
                'price_change_percentage': 0.0,
                'current_price': current_price,
                'entry_price': trade.entry_price
            }
    
    async def check_tp_sl_hit(self, trade: Trade, current_price: float) -> Optional[str]:
        """Check if take profit or stop loss has been hit"""
        if not trade.take_profit or not trade.stop_loss:
            return None
        
        try:
            if trade.trade_type == TradeType.LONG:
                if current_price >= trade.take_profit:
                    return "TP"
                elif current_price <= trade.stop_loss:
                    return "SL"
            else:  # SHORT
                if current_price <= trade.take_profit:
                    return "TP"
                elif current_price >= trade.stop_loss:
                    return "SL"
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking TP/SL for trade {trade.id}: {e}")
            return None
    
    async def close_trade(self, trade: Trade, exit_price: float, reason: str = "Manual") -> Dict[str, float]:
        """Close a trade and calculate final PnL"""
        try:
            # Calculate final PnL
            pnl_data = await self.calculate_pnl(trade, exit_price)
            
            # Update trade object
            trade.exit_price = exit_price
            trade.status = TradeStatus.CLOSED
            trade.closed_at = datetime.utcnow()
            trade.pnl_usd = pnl_data['pnl_usd']
            trade.pnl_percentage = pnl_data['pnl_percentage']
            
            # Save to database
            await self.db_manager.update_trade(trade)
            
            logger.info(f"Closed trade {trade.id} ({reason}): PnL = {pnl_data['pnl_percentage']:.2f}% (${pnl_data['pnl_usd']:.2f})")
            
            return pnl_data
            
        except Exception as e:
            logger.error(f"Error closing trade {trade.id}: {e}")
            raise
    
    async def calculate_position_size(self, capital: float, leverage: int, entry_price: float, 
                                    risk_percentage: float = 2.0) -> Dict[str, float]:
        """Calculate optimal position size based on risk management"""
        try:
            # Calculate position value
            position_value = capital * leverage
            
            # Calculate quantity
            quantity = position_value / entry_price
            
            # Calculate risk amount (max loss)
            risk_amount = capital * (risk_percentage / 100)
            
            # Calculate max acceptable price movement
            max_price_move_percentage = risk_amount / (capital * leverage) * 100
            
            return {
                'quantity': quantity,
                'position_value': position_value,
                'risk_amount': risk_amount,
                'max_price_move_percentage': max_price_move_percentage
            }
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return {
                'quantity': 0,
                'position_value': 0,
                'risk_amount': 0,
                'max_price_move_percentage': 0
            }
    
    async def validate_trade_parameters(self, symbol: str, trade_type: TradeType, entry_price: float,
                                      leverage: int, capital: float) -> Tuple[bool, str]:
        """Validate trade parameters before opening"""
        try:
            # Check symbol validity
            symbol_info = await self.mexc_client.get_symbol_info(symbol)
            if not symbol_info:
                return False, f"Symbol {symbol} not found on MEXC"
            
            # Check leverage limits
            if leverage < 1 or leverage > self.config.MAX_LEVERAGE:
                return False, f"Leverage must be between 1 and {self.config.MAX_LEVERAGE}"
            
            # Check capital limits
            if capital < self.config.MIN_CAPITAL:
                return False, f"Minimum capital is ${self.config.MIN_CAPITAL}"
            
            if capital > self.config.MAX_CAPITAL:
                return False, f"Maximum capital is ${self.config.MAX_CAPITAL}"
            
            # Check entry price validity
            current_price = await self.mexc_client.get_current_price(symbol)
            if current_price:
                price_diff_percentage = abs((entry_price - current_price) / current_price) * 100
                if price_diff_percentage > 10:  # Entry price too far from current
                    return False, f"Entry price too far from current price ({price_diff_percentage:.1f}%)"
            
            return True, "Valid"
            
        except Exception as e:
            logger.error(f"Error validating trade parameters: {e}")
            return False, "Validation error"
    
    async def get_market_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Get market sentiment indicators for a symbol"""
        try:
            sentiment = {
                'symbol': symbol,
                'sentiment': 'Neutral',
                'strength': 0,
                'indicators': {}
            }
            
            # Get funding rate
            funding_data = await self.mexc_client.get_funding_rate(symbol)
            if funding_data:
                funding_rate = funding_data['funding_rate']
                sentiment['indicators']['funding_rate'] = funding_rate
                
                if funding_rate > 0.01:  # 1% funding rate
                    sentiment['sentiment'] = 'Bearish'
                    sentiment['strength'] += 1
                elif funding_rate < -0.01:
                    sentiment['sentiment'] = 'Bullish'
                    sentiment['strength'] += 1
            
            # Get open interest
            open_interest = await self.mexc_client.get_open_interest(symbol)
            if open_interest:
                sentiment['indicators']['open_interest'] = open_interest
            
            # Get price action
            ticker = await self.mexc_client.get_ticker(symbol)
            if ticker:
                change_24h = ticker['change_24h']
                sentiment['indicators']['price_change_24h'] = change_24h
                
                if change_24h > 5:
                    sentiment['sentiment'] = 'Bullish' if sentiment['sentiment'] == 'Neutral' else sentiment['sentiment']
                    sentiment['strength'] += 1
                elif change_24h < -5:
                    sentiment['sentiment'] = 'Bearish' if sentiment['sentiment'] == 'Neutral' else sentiment['sentiment']
                    sentiment['strength'] += 1
            
            return sentiment
            
        except Exception as e:
            logger.error(f"Error getting market sentiment for {symbol}: {e}")
            return {'symbol': symbol, 'sentiment': 'Unknown', 'strength': 0, 'indicators': {}}
    
    async def calculate_risk_reward_ratio(self, entry_price: float, take_profit: float, 
                                        stop_loss: float, trade_type: TradeType) -> float:
        """Calculate risk/reward ratio for a trade"""
        try:
            if trade_type == TradeType.LONG:
                potential_profit = take_profit - entry_price
                potential_loss = entry_price - stop_loss
            else:
                potential_profit = entry_price - take_profit
                potential_loss = stop_loss - entry_price
            
            if potential_loss <= 0:
                return 0
            
            return potential_profit / potential_loss
            
        except Exception as e:
            logger.error(f"Error calculating risk/reward ratio: {e}")
            return 0
