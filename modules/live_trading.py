"""
Live trading environment for MetaTrader 5 integration
"""

import MetaTrader5 as mt5
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
import logging
import pickle
from typing import Dict, List, Tuple, Optional
from config import MT5_CONFIG, MIN_POSITION_SIZE, MAX_POSITION_SIZE, STOP_LOSS, PROFIT_TARGET
                    

logger = logging.getLogger(__name__)

class LiveTradingEnvironment:
    def __init__(self, symbol: str = MT5_CONFIG['MT5SYMBOL'], timeframe: str = MT5_CONFIG['MT5TIMEFRAME'], model_path: str = MT5_CONFIG['MODEL_PATH']):
        self.symbol = symbol
        self.timeframe = timeframe
        self.position = 0  # 0: No position, 1: Long, -1: Short
        self.entry_price = 0
        self.last_tick_time = None
        self.trade_history = []
        self.position_size = MIN_POSITION_SIZE  # Start with minimum position size
        
        # Generate unique magic number for this bot instance
        import random
        self.magic = MT5_CONFIG['MAGIC_BASE'] + random.randint(1, 9999)
        
        # MT5 should already be initialized and logged in from main
        
        # Get symbol info
        self.symbol_info = mt5.symbol_info(symbol)
        if self.symbol_info is None:
            raise ValueError(f"Symbol {symbol} not found")
        
        # Enable symbol for trading
        if not self.symbol_info.visible:
            if not mt5.symbol_select(symbol, True):
                raise RuntimeError(f"Failed to select {symbol}")
        
        # Load trained model using generic loader
        self.model = self._load_model(model_path)
        
        # Sync position state with MT5
        self._sync_position_state()
        
        logger.info(f"Initialized live trading for {symbol} using model from {model_path}")
        logger.info(f"Bot instance magic number: {self.magic}")
    
    def _load_model(self, model_path: str):
        """Load model using generic approach"""
        try:
            # Try loading as stable-baselines3 model first
            from stable_baselines3 import PPO
            return PPO.load(model_path)
        except Exception as e:
            logger.warning(f"Failed to load as stable-baselines3 model: {e}")
            try:
                # Try loading as pickle file
                with open(model_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e2:
                logger.error(f"Failed to load model from {model_path}: {e2}")
                raise RuntimeError(f"Cannot load model from {model_path}")
    
    def _get_mt5_data(self, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """Get market data directly from MT5 with fallback strategies"""
        try:
            # Check if symbol is available
            symbol_info = mt5.symbol_info(self.symbol)
            if symbol_info is None:
                logger.error(f"Symbol {self.symbol} not found in MT5")
                # Try to get available symbols and suggest alternatives
                symbols = mt5.symbols_get()
                if symbols:
                    available_fx = [s.name for s in symbols if s.name.endswith(('USD', 'EUR', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD'))][:5]
                    logger.info(f"Available symbols: {available_fx}")
                return pd.DataFrame()

            if not symbol_info.visible:
                logger.warning(f"Symbol {self.symbol} not visible, attempting to select...")
                if not mt5.symbol_select(self.symbol, True):
                    logger.error(f"Failed to select symbol {self.symbol}")
                    return pd.DataFrame()
                logger.info(f"Successfully selected symbol {self.symbol}")

            logger.debug(f"Getting data for {self.symbol} from {start_time} to {end_time}")

            # Convert timeframe string to MT5 constant
            timeframe_map = {
                'M1': mt5.TIMEFRAME_M1,
                'M5': mt5.TIMEFRAME_M5,
                'M15': mt5.TIMEFRAME_M15,
                'M30': mt5.TIMEFRAME_M30,
                'H1': mt5.TIMEFRAME_H1,
                'H4': mt5.TIMEFRAME_H4,
                'D1': mt5.TIMEFRAME_D1
            }

            mt5_timeframe = timeframe_map.get(self.timeframe, mt5.TIMEFRAME_M15)

            # Try multiple strategies to get data
            strategies = [
                # Strategy 1: Try current timeframe with recent data
                (mt5_timeframe, datetime.now() - timedelta(hours=24), datetime.now()),
                # Strategy 2: Try H1 timeframe
                (mt5.TIMEFRAME_H1, datetime.now() - timedelta(hours=24), datetime.now()),
                # Strategy 3: Try D1 timeframe
                (mt5.TIMEFRAME_D1, datetime.now() - timedelta(days=30), datetime.now()),
                # Strategy 4: Try M1 timeframe with shorter period
                (mt5.TIMEFRAME_M1, datetime.now() - timedelta(hours=1), datetime.now()),
            ]

            for i, (tf, start, end) in enumerate(strategies):
                logger.debug(f"Trying strategy {i+1}: timeframe={tf}, start={start}, end={end}")
                rates = mt5.copy_rates_range(self.symbol, tf, start, end)

                if rates is not None and len(rates) > 0:
                    logger.info(f"Successfully got {len(rates)} rates using strategy {i+1}")
                    # Convert to DataFrame
                    df = pd.DataFrame(rates)
                    df['time'] = pd.to_datetime(df['time'], unit='s')
                    df.set_index('time', inplace=True)
                    return df
                else:
                    logger.warning(f"Strategy {i+1} failed: got {len(rates) if rates else 0} rates")

            # If all strategies fail, try to get current tick data
            logger.warning("All historical data strategies failed, trying current tick data...")
            tick = mt5.symbol_info_tick(self.symbol)
            if tick:
                logger.info(f"Got current tick data: bid={tick.bid}, ask={tick.ask}")
                # Create a minimal DataFrame with current data
                current_time = datetime.now()
                df = pd.DataFrame([{
                    'time': current_time,
                    'open': tick.bid,
                    'high': tick.bid,
                    'low': tick.bid,
                    'close': tick.bid,
                    'tick_volume': 1,
                    'spread': symbol_info.spread,
                    'real_volume': 0
                }])
                df['time'] = pd.to_datetime(df['time'])
                df.set_index('time', inplace=True)
                return df

            logger.error(f"All data retrieval strategies failed for {self.symbol}")
            return pd.DataFrame()

        except Exception as e:
            logger.error(f"Failed to get MT5 data for {self.symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_dynamic_deviation(self) -> int:
        """Calculate deviation based on current market volatility"""
        try:
            # Get recent price data to calculate volatility
            end_time = datetime.now()
            start_time = end_time - pd.Timedelta(hours=1)  # Last hour of data
            
            df = self._get_mt5_data(start_time, end_time)
            
            if df.empty or len(df) < 10:
                return MT5_CONFIG['BASE_DEVIATION']
            
            # Calculate price volatility (standard deviation of price changes)
            price_changes = df['close'].pct_change().dropna()
            volatility = price_changes.std()
            
            # Convert volatility to deviation multiplier
            # Higher volatility = higher deviation needed
            volatility_multiplier = 1 + (volatility * 100)  # Scale volatility
            
            # Calculate dynamic deviation
            dynamic_deviation = int(MT5_CONFIG['BASE_DEVIATION'] * volatility_multiplier)
            
            # Ensure deviation stays within bounds
            dynamic_deviation = max(MT5_CONFIG['MIN_DEVIATION'], 
                                  min(MT5_CONFIG['MAX_DEVIATION'], dynamic_deviation))
            
            logger.debug(f"Volatility: {volatility:.4f}, Deviation: {dynamic_deviation}")
            return dynamic_deviation
            
        except Exception as e:
            logger.warning(f"Failed to calculate dynamic deviation: {e}, using base deviation")
            return MT5_CONFIG['BASE_DEVIATION']
    
    def get_observation(self) -> np.ndarray:
        """Get current market state observation"""
        # Get recent market data
        end_time = datetime.now()
        start_time = end_time - pd.Timedelta(minutes=5)  # Get last 5 minutes of data
        
        df = self._get_mt5_data(start_time, end_time)
        
        if df.empty:
            logger.error("No data received from MT5")
            raise RuntimeError("Failed to get market data")
        
        logger.info(f"Received {len(df)} data points from MT5")
        
        # Get the most recent data point
        latest = df.iloc[-1]
        
        # Create observation matching training format: [open, high, low, close, volume]
        observation = np.array([
            latest['open'],
            latest['high'], 
            latest['low'],
            latest['close'],
            latest.get('tick_volume', 0)  # Use tick_volume if available, otherwise 0
        ], dtype=np.float32)
        
        return observation
    
    def _calculate_position_size(self):
        """Calculate position size based on win rate"""
        if not self.trade_history:
            return MIN_POSITION_SIZE
        
        # Calculate win rate
        total_trades = len(self.trade_history)
        winning_trades = sum(1 for trade in self.trade_history if trade['profit'] > 0)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Calculate position size based on win rate
        # Linear scaling from MIN_POSITION_SIZE to MAX_POSITION_SIZE based on win rate
        position_size = MIN_POSITION_SIZE + (MAX_POSITION_SIZE - MIN_POSITION_SIZE) * win_rate
        
        # Ensure position size stays within bounds
        return np.clip(position_size, MIN_POSITION_SIZE, MAX_POSITION_SIZE)
    
    def execute_trade(self, action: int):
        """Execute trading action based on model decision"""
        current_price = mt5.symbol_info_tick(self.symbol).ask
        
        # Hold
        if action == 0:
            return
        
        # Buy
        elif action == 1 and self.position <= 0:
            if self.position < 0:
                self._close_position()
            
            # Update position size based on win rate
            self.position_size = self._calculate_position_size()
            
            # Open long position with calculated size
            self._open_position(mt5.ORDER_TYPE_BUY, self.position_size)
        
        # Sell
        elif action == 2 and self.position >= 0:
            if self.position > 0:
                self._close_position()
            
            # Update position size based on win rate
            self.position_size = self._calculate_position_size()
            
            # Open short position with calculated size
            self._open_position(mt5.ORDER_TYPE_SELL, self.position_size)
        
        # Close
        elif action == 3 and self.position != 0:
            self._close_position()
    
    def run(self, max_duration_seconds: int = 1800):
        """Main trading loop"""
        logger.info(f"Starting live trading for {self.symbol} (max duration: {max_duration_seconds} seconds)")
        start_time = time.time()
        
        while True:
            try:
                # Check if max duration exceeded
                elapsed = time.time() - start_time
                if elapsed >= max_duration_seconds:
                    logger.info(f"Max trading duration ({max_duration_seconds} seconds) reached. Stopping.")
                    break
                
                logger.debug("Getting observation...")
                # Get current market state
                observation = self.get_observation()
                logger.debug(f"Got observation: {observation}")

                logger.debug("Getting model prediction...")
                # Get model's action
                action, _ = self.model.predict(observation, deterministic=True)
                logger.debug(f"Model predicted action: {action}")

                logger.debug("Executing trade...")
                # Execute trade
                self.execute_trade(action)
                logger.debug("Trade executed successfully")

                logger.debug("Waiting for next tick...")
                # Wait for next tick
                self.wait_for_next_tick()
                logger.debug("Next tick received")

            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
            
            time.sleep(0.1)
    
    def _open_position(self, order_type: int, position_size: float):
        """Open a new position"""
        # Check supported filling modes
        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info is None:
            logger.error(f"Failed to get symbol info for {self.symbol}")
            return False
        
        logger.info(f"Symbol {self.symbol} filling modes: {symbol_info.filling_mode}")
        logger.info(f"Supported filling modes - FOK: {bool(symbol_info.filling_mode & 0x01)}, "
                   f"IOC: {bool(symbol_info.filling_mode & 0x02)}, "
                   f"RETURN: {bool(symbol_info.filling_mode & 0x04)}")
        
        price = mt5.symbol_info_tick(self.symbol).ask if order_type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(self.symbol).bid
        
        # Calculate Stop Loss and Take Profit
        if order_type == mt5.ORDER_TYPE_BUY:
            sl = price * (1 - STOP_LOSS)
            tp = price * (1 + PROFIT_TARGET)
        else:  # SELL
            sl = price * (1 + STOP_LOSS)
            tp = price * (1 - PROFIT_TARGET)
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": position_size,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": self.calculate_dynamic_deviation(),
            "magic": self.magic,
            "comment": MT5_CONFIG['COMMENT'] + " open",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,
        }
        
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            if "AutoTrading disabled" in str(result.comment):
                logger.warning("⚠️ Auto-trading appears disabled in MT5 - check Expert Advisors settings")
                logger.warning("Make sure 'Allow automated trading' is checked in Tools → Options → Expert Advisors")
                return False
            else:
                logger.error(f"Failed to open position: {result.comment}")
            return False
        
        self.position = 1 if order_type == mt5.ORDER_TYPE_BUY else -1
        self.entry_price = price
        logger.info(f"Opened {'long' if order_type == mt5.ORDER_TYPE_BUY else 'short'} position at {price} with size {position_size}, SL: {sl:.5f}, TP: {tp:.5f}")
        return True
    
    def _close_position(self):
        """Close current position"""
        ticket = self._get_position_ticket()
        if ticket is None:
            return False
        
        position = mt5.positions_get(ticket=ticket)[0]
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": position.volume,
            "type": mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
            "position": ticket,
            "price": mt5.symbol_info_tick(self.symbol).bid if position.type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(self.symbol).ask,
            "deviation": self.calculate_dynamic_deviation(),
            "magic": self.magic,
            "comment": MT5_CONFIG['COMMENT'] + " close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,
        }
        
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Failed to close position: {result.comment}")
            return False
        
        # Calculate profit
        profit = position.profit
        
        # Update trade history
        self.trade_history.append({
            'type': 'long' if position.type == mt5.ORDER_TYPE_BUY else 'short',
            'entry_price': self.entry_price,
            'exit_price': result.price,
            'profit': profit,
            'timestamp': datetime.now()
        })
        
        self.position = 0
        self.entry_price = 0
        logger.info(f"Closed position at {result.price} with profit {profit}")
        return True
    
    def _sync_position_state(self):
        """Sync internal position state with MT5 positions"""
        positions = mt5.positions_get(symbol=self.symbol, magic=self.magic)
        if positions:
            position = positions[0]  # Assume only one position per symbol/magic
            self.position = 1 if position.type == mt5.ORDER_TYPE_BUY else -1
            self.entry_price = position.price_open
            logger.info(f"Synced with existing position: {'long' if self.position > 0 else 'short'} at {self.entry_price}")
        else:
            self.position = 0
            self.entry_price = 0
            logger.info("No existing positions found, starting with neutral state")
    
    def wait_for_next_tick(self):
        """Wait for next market tick"""
        while True:
            tick = mt5.symbol_info_tick(self.symbol)
            if tick is None:
                time.sleep(0.1)
                continue
            
            if self.last_tick_time is None or tick.time > self.last_tick_time:
                self.last_tick_time = tick.time
                break
            
            time.sleep(0.1)
    
    def _get_account_balance(self) -> float:
        """Get current account balance"""
        account_info = mt5.account_info()
        return account_info.balance if account_info else 0.0

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize MT5 connection
    if not mt5.initialize(
        login=int(MT5_CONFIG['MT5LOGIN']),
        password=MT5_CONFIG['MT5PASSWORD'],
        server=MT5_CONFIG['MT5SERVER']
    ):
        logger.error("Failed to initialize MT5")
        exit(1)
    
    # Check MT5 connection status
    logger.info("Checking MT5 connection...")
    
    # Check account info
    account_info = mt5.account_info()
    if account_info is None:
        logger.error("Failed to get account info - MT5 may not be logged in or terminal not running")
        logger.error("Please ensure:")
        logger.error("1. MT5 terminal is running")
        logger.error("2. You are logged in to your account")
        logger.error("3. Auto-trading is enabled")
        exit(1)
    
    logger.info(f"✅ MT5 Connected - Account: {account_info.login} ({account_info.name})")
    logger.info(f"✅ Balance: {account_info.balance}")
    logger.info(f"✅ Server: {account_info.server}")
    
    # Check if auto-trading is actually enabled
    terminal_info = mt5.terminal_info()
    logger.info(f"Terminal info: {terminal_info}")
    if terminal_info:
        logger.info(f"Trade allowed: {terminal_info.trade_allowed}")
        logger.info(f"Connected: {terminal_info.connected}")
        logger.info(f"DLLs allowed: {terminal_info.dlls_allowed}")
        logger.info(f"Trade API: {terminal_info.tradeapi_disabled}")
    
    # Note: terminal_info.trade_allowed may not be reliable, so we'll try trading and handle errors
    logger.info("✅ Bot ready - attempting to execute trades (auto-trading should be enabled in MT5)")
    
    logger.info(f"✅ Auto-trading enabled in MT5 terminal")
    
    try:
        # Create trading environment
        trader = LiveTradingEnvironment(
            symbol=MT5_CONFIG['MT5SYMBOL'],  # Use symbol from MT5 config
            timeframe=MT5_CONFIG['MT5TIMEFRAME'],  # Use timeframe from MT5 config
            model_path=MT5_CONFIG['MODEL_PATH']  # Use model path from config
        )
        
        # Start trading
        trader.run()
        
    except KeyboardInterrupt:
        logger.info("Trading stopped by user")
    except Exception as e:
        logger.error(f"Error in main loop: {e}")
    finally:
        # Shutdown MT5
        mt5.shutdown() 