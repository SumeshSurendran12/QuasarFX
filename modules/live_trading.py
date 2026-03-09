"""
Live trading environment for MetaTrader 5 integration
"""

from __future__ import annotations

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover - optional in paper fallback mode
    mt5 = None  # type: ignore[assignment]
    MT5_AVAILABLE = False
import numpy as np
try:
    import pandas as pd
except ModuleNotFoundError:  # pragma: no cover - optional in synthetic paper fallback
    pd = None  # type: ignore[assignment]
from datetime import datetime, timedelta, timezone
import time
import logging
import pickle
import os
import zlib
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple, Optional
try:
    from .config import (
        MT5_CONFIG,
        MIN_POSITION_SIZE,
        MAX_POSITION_SIZE,
        STOP_LOSS,
        PROFIT_TARGET,
        STRATEGY_1_NAME,
        STRATEGY_1_LIVE_MAX_SPREAD,
        DATA_FEED_HEARTBEAT_SECONDS,
    )
except ImportError:  # pragma: no cover - script mode fallback
    from config import (  # type: ignore[no-redef]
        MT5_CONFIG,
        MIN_POSITION_SIZE,
        MAX_POSITION_SIZE,
        STOP_LOSS,
        PROFIT_TARGET,
        STRATEGY_1_NAME,
        STRATEGY_1_LIVE_MAX_SPREAD,
        DATA_FEED_HEARTBEAT_SECONDS,
    )
                    

try:
    from .strategy_1_event_logger import Strategy1CanonicalEventLogger
except ImportError:  # pragma: no cover - script mode fallback
    from strategy_1_event_logger import Strategy1CanonicalEventLogger  # type: ignore[no-redef]


logger = logging.getLogger(__name__)

MT5_ORDER_TYPE_BUY = int(getattr(mt5, "ORDER_TYPE_BUY", 0))
MT5_ORDER_TYPE_SELL = int(getattr(mt5, "ORDER_TYPE_SELL", 1))


class _FallbackPaperModel:
    """Minimal deterministic policy to keep paper-event ingestion active when model loading fails."""

    def __init__(self) -> None:
        self._step = 0

    def predict(self, observation: Any, deterministic: bool = True) -> Tuple[np.ndarray, None]:
        self._step += 1
        if self._step % 7 == 0:
            action = 3  # close
        elif self._step % 3 == 0:
            action = 2  # sell
        elif self._step % 2 == 0:
            action = 1  # buy
        else:
            action = 0  # hold
        return np.array([action], dtype=np.int32), None


class LiveTradingEnvironment:
    def _resolve_magic_number(self) -> int:
        raw_magic = str(os.getenv("FX_MAGIC_NUMBER", "")).strip()
        if raw_magic:
            try:
                return int(raw_magic)
            except ValueError:
                logger.warning(f"Invalid FX_MAGIC_NUMBER={raw_magic!r}; using deterministic strategy magic.")

        strategy_key = f"{STRATEGY_1_NAME}:{self.symbol}".encode("utf-8")
        return int(MT5_CONFIG['MAGIC_BASE']) + int(zlib.crc32(strategy_key) % 1000)

    @staticmethod
    def _build_order_comment(prefix: str, action: str) -> str:
        return f"{prefix}|{action}"[:31]

    def _ensure_mt5_connection(self):
        """Initialize MT5 when instantiated outside the script entrypoint."""
        if not MT5_AVAILABLE:
            if self.execution_mode == "live":
                raise RuntimeError("MetaTrader5 package is required in live mode. Install with `pip install MetaTrader5`.")
            logger.warning("MetaTrader5 package not found. Running synthetic paper feed fallback.")
            return
        if self.execution_mode != "live":
            # Paper mode does not require authenticated MT5 connectivity.
            return

        terminal_info = mt5.terminal_info()
        if terminal_info is not None and terminal_info.connected:
            return

        login = MT5_CONFIG.get('MT5LOGIN')
        password = MT5_CONFIG.get('MT5PASSWORD')
        server = MT5_CONFIG.get('MT5SERVER')
        if not login or not password or not server:
            raise RuntimeError("MT5 credentials are missing (MT5_LOGIN/MT5_PASSWORD/MT5_SERVER)")

        if not mt5.initialize(login=int(login), password=password, server=server):
            raise RuntimeError(f"Failed to initialize MT5: {mt5.last_error()}")

    def __init__(self, symbol: str = MT5_CONFIG['MT5SYMBOL'], timeframe: str = MT5_CONFIG['MT5TIMEFRAME'], model_path: str = MT5_CONFIG['MODEL_PATH']):
        self.symbol = symbol
        self.timeframe = timeframe
        self.position = 0  # 0: No position, 1: Long, -1: Short
        self.entry_price = 0
        self.last_tick_time = None
        self.trade_history = []
        self.position_size = MIN_POSITION_SIZE  # Start with minimum position size
        self.last_action = None
        self.last_action_log = 0.0
        self.action_log_interval = 30.0
        self.execution_mode = str(os.getenv("FX_EXECUTION_MODE", "paper")).strip().lower()
        if self.execution_mode not in {"paper", "live"}:
            self.execution_mode = "paper"
        self.event_timezone = str(os.getenv("FX_EVENT_TIMEZONE", "America/Chicago")).strip() or "America/Chicago"
        self.session_label = "".join(ch for ch in str(os.getenv("FX_SESSION_LABEL", "LONDON")).upper() if ch.isalnum()) or "LONDON"
        self.broker_name = str(os.getenv("FX_BROKER_NAME", "gcapi_demo" if self.execution_mode == "paper" else "mt5")).strip()
        self.event_logger = Strategy1CanonicalEventLogger(
            session_label=self.session_label,
            mode=self.execution_mode,
            symbol=self.symbol,
            broker=self.broker_name,
            timezone_name=self.event_timezone,
        )
        self.max_trades_per_session = int(self.event_logger.max_trades_per_session)
        self.session_trade_count = 0
        self.session_net_pnl_usd = 0.0
        self.trade_counter = 0
        self.current_trade: Optional[Dict[str, Any]] = None
        self.session_closed = False
        self.pip_size = 0.01 if str(self.symbol).upper().endswith("JPY") else 0.0001
        self.paper_mid_price = float(str(os.getenv("FX_PAPER_START_PRICE", "1.08500")).strip() or "1.08500")
        self.paper_spread = max(self.pip_size * 0.2, float(str(os.getenv("FX_PAPER_SPREAD", "0.00008")).strip() or "0.00008"))
        self.paper_tick_step = max(self.pip_size * 0.05, float(str(os.getenv("FX_PAPER_TICK_STEP", "0.00003")).strip() or "0.00003"))
        self.synthetic_tick_seq = 0
        self._last_synthetic_tick: Optional[Any] = None
        self.data_feed_heartbeat_seconds = max(1, int(DATA_FEED_HEARTBEAT_SECONDS))
        self.last_feed_heartbeat = 0.0
        self.order_comment_prefix = str(os.getenv("FX_ORDER_COMMENT_PREFIX", "QFX-S1")).strip() or "QFX-S1"
        self.magic = self._resolve_magic_number()
        
        # If this class is used directly (e.g. ad-hoc smoke tests), initialize MT5 here.
        self._ensure_mt5_connection()
        
        if MT5_AVAILABLE and self.execution_mode == "live":
            # Get symbol info
            self.symbol_info = mt5.symbol_info(symbol)
            if self.symbol_info is None:
                raise ValueError(f"Symbol {symbol} not found")
            
            # Enable symbol for trading
            if not self.symbol_info.visible:
                if not mt5.symbol_select(symbol, True):
                    raise RuntimeError(f"Failed to select {symbol}")
        else:
            self.symbol_info = SimpleNamespace(
                visible=True,
                spread=int(round(self.paper_spread / self.pip_size)),
                filling_mode=0,
            )
        
        # Load trained model using generic loader
        self.model = self._load_model(model_path)
        
        # Sync position state with MT5
        self._sync_position_state()

        self.event_logger.append(
            "session_start",
            profile_name=self.event_logger.profile_name,
            symbol=self.symbol,
            mode=self.execution_mode,
            broker=self.broker_name,
            timezone=self.event_timezone,
        )
        
        logger.info(f"Initialized live trading for {symbol} using model from {model_path}")
        logger.info(f"Strategy magic number: {self.magic}")
        logger.info(f"[STRATEGY1] run_id={self.event_logger.run_id}")
        logger.info(f"Canonical mode={self.execution_mode} events={self.event_logger.events_path}")

    def _next_synthetic_tick(self) -> Any:
        self.synthetic_tick_seq += 1
        phase = self.synthetic_tick_seq % 40
        direction = 1.0 if phase < 20 else -1.0
        self.paper_mid_price = max(self.pip_size * 10.0, self.paper_mid_price + direction * self.paper_tick_step)
        half_spread = max(self.pip_size * 0.1, self.paper_spread / 2.0)
        bid = self.paper_mid_price - half_spread
        ask = self.paper_mid_price + half_spread
        return SimpleNamespace(
            bid=float(bid),
            ask=float(ask),
            time=int(time.time() * 1000) + int(self.synthetic_tick_seq),
        )

    def _get_tick(self, *, advance: bool = True) -> Any:
        if MT5_AVAILABLE:
            tick = mt5.symbol_info_tick(self.symbol)
            if tick is not None:
                return tick
            if self.execution_mode == "paper":
                if advance or self._last_synthetic_tick is None:
                    self._last_synthetic_tick = self._next_synthetic_tick()
                return self._last_synthetic_tick
            return None
        if self.execution_mode != "paper":
            return None
        if advance or self._last_synthetic_tick is None:
            self._last_synthetic_tick = self._next_synthetic_tick()
        return self._last_synthetic_tick
    
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
                if self.execution_mode == "paper":
                    logger.warning(f"Failed to load model from {model_path}: {e2}")
                    logger.warning("Using fallback paper policy for ingestion validation.")
                    return _FallbackPaperModel()
                logger.error(f"Failed to load model from {model_path}: {e2}")
                raise RuntimeError(f"Cannot load model from {model_path}")
    
    def _get_mt5_data(self, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """Get market data directly from MT5 with fallback strategies"""
        if pd is None:
            raise RuntimeError("pandas is required for MT5 data retrieval")
        if not MT5_AVAILABLE:
            return pd.DataFrame()
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
        if pd is None:
            return MT5_CONFIG['BASE_DEVIATION']
        try:
            # Get recent price data to calculate volatility
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=1)  # Last hour of data
            
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
        if self.execution_mode == "paper" and not MT5_AVAILABLE:
            tick = self._get_tick(advance=True)
            if tick is None:
                raise RuntimeError("Failed to generate synthetic paper tick")
            mid = float((float(tick.bid) + float(tick.ask)) / 2.0)
            return np.array([mid, mid + self.pip_size, mid - self.pip_size, mid, 100.0], dtype=np.float32)
        if pd is None:
            if self.execution_mode == "paper":
                tick = self._get_tick(advance=True)
                if tick is None:
                    raise RuntimeError("Failed to get fallback tick for paper observation")
                mid = float((float(tick.bid) + float(tick.ask)) / 2.0)
                return np.array([mid, mid + self.pip_size, mid - self.pip_size, mid, 100.0], dtype=np.float32)
            raise RuntimeError("pandas is required for live observation pipeline")

        # Get recent market data
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=5)  # Get last 5 minutes of data
        
        df = self._get_mt5_data(start_time, end_time)
        
        if df.empty:
            if self.execution_mode == "paper":
                tick = self._get_tick(advance=True)
                if tick is None:
                    raise RuntimeError("Failed to get fallback tick for paper observation")
                mid = float((float(tick.bid) + float(tick.ask)) / 2.0)
                return np.array([mid, mid + self.pip_size, mid - self.pip_size, mid, 100.0], dtype=np.float32)
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

    def _get_symbol_positions(self):
        """Get open positions for this bot instance using stable strategy identity."""
        if not MT5_AVAILABLE:
            return []
        positions = mt5.positions_get(symbol=self.symbol)
        if positions is None:
            logger.warning(f"positions_get failed for {self.symbol}: {mt5.last_error()}")
            return []

        managed_positions = []
        for position in positions:
            position_magic = int(getattr(position, "magic", 0) or 0)
            position_comment = str(getattr(position, "comment", "") or "")
            if position_magic == self.magic or position_comment.startswith(self.order_comment_prefix):
                managed_positions.append(position)
        return managed_positions

    @staticmethod
    def _utc_iso() -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def _spread_pips(self, tick: Any) -> float:
        if tick is None:
            return 0.0
        return float((float(tick.ask) - float(tick.bid)) / self.pip_size)

    def _max_spread_pips(self) -> float:
        return float(float(STRATEGY_1_LIVE_MAX_SPREAD) / self.pip_size)

    def _next_trade_id(self) -> str:
        self.trade_counter += 1
        trade_day = datetime.now().strftime("%Y%m%d")
        return f"S1-{trade_day}-{self.trade_counter:03d}"

    @staticmethod
    def _tick_last_price(tick: Any) -> Optional[float]:
        if tick is None:
            return None
        try:
            bid = float(getattr(tick, "bid", 0.0))
            ask = float(getattr(tick, "ask", 0.0))
        except (TypeError, ValueError):
            return None
        if bid > 0 and ask > 0:
            return float((bid + ask) / 2.0)
        if bid > 0:
            return bid
        if ask > 0:
            return ask
        return None

    def _emit_data_feed_alive(self, tick: Any) -> None:
        self.event_logger.append(
            "data_feed_alive",
            symbol=self.symbol,
            last_price=self._tick_last_price(tick),
        )

    def _emit_signal_candidate(self, action: int, tick: Any) -> None:
        if action not in {1, 2}:
            return
        side = "long" if action == 1 else "short"
        self.event_logger.append(
            "signal_evaluated",
            symbol=self.symbol,
            bar_ts=self._utc_iso(),
            side=side,
            signal_score=float(action),
            spread_pips=self._spread_pips(tick),
            decision="candidate",
            reason_code="signal_pass",
        )

    def _emit_skip(self, *, action: int, reason_code: str, tick: Any, decision: str = "skip") -> None:
        side = "long" if action == 1 else "short"
        self.event_logger.append(
            "trade_skipped",
            symbol=self.symbol,
            bar_ts=self._utc_iso(),
            side=side,
            decision=decision,
            reason_code=reason_code,
            spread_pips=self._spread_pips(tick),
            max_spread_pips=self._max_spread_pips(),
        )

    def execute_trade(self, action: int):
        """Execute trading action based on model decision"""
        tick = self._get_tick(advance=False)
        if tick is None:
            tick = self._get_tick(advance=True)
        if tick is None:
            logger.warning("No tick data available; skipping action.")
            return

        # Refresh current position state from MT5 (FIFO/no-hedge safe)
        if self.execution_mode == "live":
            positions = self._get_symbol_positions()
            if positions:
                oldest = min(positions, key=lambda p: p.time)
                self.position = 1 if oldest.type == MT5_ORDER_TYPE_BUY else -1
                self.entry_price = oldest.price_open
            else:
                self.position = 0
                self.entry_price = 0
        else:
            if self.current_trade is None:
                self.position = 0
                self.entry_price = 0

        # Hold
        if action == 0:
            return

        # Strategy 1 hard spread gate for new entries.
        live_spread = float(tick.ask - tick.bid)
        entry_blocked_by_spread = live_spread > float(STRATEGY_1_LIVE_MAX_SPREAD)

        if action in {1, 2} and self.position == 0 and self.session_trade_count >= self.max_trades_per_session:
            self._emit_skip(action=action, reason_code="session_cap", tick=tick)
            logger.info("Skip entry due to Strategy 1 session cap.")
            return
        
        # Buy
        if action == 1 and self.position <= 0:
            if self.position < 0:
                # FIFO/hedging-safe: close first, then wait for next tick to open
                closed = self._close_position(exit_reason="manual_disable")
                if closed:
                    logger.info("Closed short; will wait before opening long.")
                return
            if entry_blocked_by_spread:
                logger.info(
                    f"Skip BUY entry [{STRATEGY_1_NAME}] due to live spread "
                    f"{live_spread:.5f} > {float(STRATEGY_1_LIVE_MAX_SPREAD):.5f}"
                )
                self._emit_skip(action=action, reason_code="spread_gate", tick=tick)
                return

            # Update position size based on win rate
            self.position_size = self._calculate_position_size()

            # Open long position with calculated size
            self._open_position(MT5_ORDER_TYPE_BUY, self.position_size)
        elif action == 1 and self.position > 0:
            logger.info("Action BUY ignored (already long)")
            self._emit_skip(action=action, reason_code="duplicate_signal", tick=tick)
        
        # Sell
        elif action == 2 and self.position >= 0:
            if self.position > 0:
                # FIFO/hedging-safe: close first, then wait for next tick to open
                closed = self._close_position(exit_reason="manual_disable")
                if closed:
                    logger.info("Closed long; will wait before opening short.")
                return
            if entry_blocked_by_spread:
                logger.info(
                    f"Skip SELL entry [{STRATEGY_1_NAME}] due to live spread "
                    f"{live_spread:.5f} > {float(STRATEGY_1_LIVE_MAX_SPREAD):.5f}"
                )
                self._emit_skip(action=action, reason_code="spread_gate", tick=tick)
                return

            # Update position size based on win rate
            self.position_size = self._calculate_position_size()

            # Open short position with calculated size
            self._open_position(MT5_ORDER_TYPE_SELL, self.position_size)
        elif action == 2 and self.position < 0:
            logger.info("Action SELL ignored (already short)")
            self._emit_skip(action=action, reason_code="duplicate_signal", tick=tick)
        
        # Close
        elif action == 3 and self.position != 0:
            self._close_position(exit_reason="manual_disable")
        elif action == 3 and self.position == 0:
            logger.info("Action CLOSE ignored (no open position)")

    def _check_profit_targets(self):
        """Check profit/stop thresholds based on entry price and current price."""
        if self.execution_mode == "live":
            positions = self._get_symbol_positions()
            if not positions:
                self.position = 0
                self.entry_price = 0
                return False, None, None
            oldest = min(positions, key=lambda p: p.time)
            self.position = 1 if oldest.type == MT5_ORDER_TYPE_BUY else -1
            self.entry_price = oldest.price_open
        else:
            if self.current_trade is None:
                self.position = 0
                self.entry_price = 0
                return False, None, None
            self.position = 1 if str(self.current_trade.get("side")) == "long" else -1
            self.entry_price = float(self.current_trade.get("entry_price", 0.0))
        if self.entry_price == 0:
            return False, None, None
        tick = self._get_tick(advance=False)
        if tick is None:
            tick = self._get_tick(advance=True)
        if tick is None:
            return False, None, None

        current_price = tick.bid if self.position > 0 else tick.ask
        if self.position > 0:
            loss_price = self.entry_price * (1 - STOP_LOSS)
            profit_price = self.entry_price * (1 + PROFIT_TARGET)
            if current_price <= loss_price:
                return True, "stop_loss", current_price
            if current_price >= profit_price:
                return True, "take_profit", current_price
        else:
            loss_price = self.entry_price * (1 + STOP_LOSS)
            profit_price = self.entry_price * (1 - PROFIT_TARGET)
            if current_price >= loss_price:
                return True, "stop_loss", current_price
            if current_price <= profit_price:
                return True, "take_profit", current_price
        return False, None, current_price

    def _log_action(self, action: int, price: float):
        now = time.time()
        if self.last_action is None or action != self.last_action or (now - self.last_action_log) >= self.action_log_interval:
            action_names = {0: "HOLD", 1: "BUY", 2: "SELL", 3: "CLOSE"}
            name = action_names.get(action, f"UNKNOWN({action})")
            logger.info(f"Action: {name} | position={self.position} | price={price:.5f}")
            self.last_action = action
            self.last_action_log = now
    
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

                loop_now = time.time()
                if (loop_now - self.last_feed_heartbeat) >= float(self.data_feed_heartbeat_seconds):
                    heartbeat_tick = self._get_tick(advance=False)
                    if heartbeat_tick is None:
                        heartbeat_tick = self._get_tick(advance=True)
                    self._emit_data_feed_alive(heartbeat_tick)
                    self.last_feed_heartbeat = loop_now

                # Auto-close based on profit/stop thresholds
                should_close, reason, price = self._check_profit_targets()
                if should_close:
                    logger.info(f"Auto-close triggered ({reason}) at {price:.5f}")
                    self._close_position(exit_reason=str(reason or "risk_rule"))
                    self.wait_for_next_tick()
                    time.sleep(0.1)
                    continue
                
                logger.debug("Getting observation...")
                # Get current market state
                observation = self.get_observation()
                logger.debug(f"Got observation: {observation}")

                logger.debug("Getting model prediction...")
                # Get model's action
                action, _ = self.model.predict(observation, deterministic=True)
                action_arr = np.asarray(action)
                if action_arr.size == 1:
                    action = int(action_arr.item())
                else:
                    action = int(action_arr.reshape(-1)[0])
                self._log_action(action, float(observation[3]))
                self._emit_signal_candidate(action=action, tick=self._get_tick(advance=False))

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
        """Open a new position and emit canonical order events."""
        if self.execution_mode == "paper":
            symbol_info = self.symbol_info
            tick = self._get_tick(advance=False)
            if tick is None:
                tick = self._get_tick(advance=True)
            if tick is None:
                logger.error("No tick data available; cannot open paper position.")
                return False
        else:
            symbol_info = mt5.symbol_info(self.symbol)
            if symbol_info is None:
                logger.error(f"Failed to get symbol info for {self.symbol}")
                return False

            tick = self._get_tick(advance=False)
            if tick is None:
                tick = self._get_tick(advance=True)
            if tick is None:
                logger.error("No tick data available; cannot open position.")
                return False

            logger.info(f"Symbol {self.symbol} filling modes: {symbol_info.filling_mode}")
            logger.info(
                f"Supported filling modes - FOK: {bool(symbol_info.filling_mode & 0x01)}, "
                f"IOC: {bool(symbol_info.filling_mode & 0x02)}, "
                f"RETURN: {bool(symbol_info.filling_mode & 0x04)}"
            )

        price = float(tick.ask) if order_type == MT5_ORDER_TYPE_BUY else float(tick.bid)
        side = "long" if order_type == MT5_ORDER_TYPE_BUY else "short"
        qty = float(position_size)
        trade_id = self._next_trade_id()
        client_order_id = f"{self.execution_mode}-S1-{self.trade_counter:03d}"

        if order_type == MT5_ORDER_TYPE_BUY:
            sl = price * (1 - STOP_LOSS)
            tp = price * (1 + PROFIT_TARGET)
        else:
            sl = price * (1 + STOP_LOSS)
            tp = price * (1 - PROFIT_TARGET)

        self.event_logger.append(
            "order_submitted",
            symbol=self.symbol,
            trade_id=trade_id,
            client_order_id=client_order_id,
            side=side,
            qty=qty,
            order_type="market",
            expected_entry=price,
            sl=sl,
            tp=tp,
            risk_usd=0.0,
        )

        if self.execution_mode == "paper":
            self.position = 1 if order_type == MT5_ORDER_TYPE_BUY else -1
            self.entry_price = price
            self.current_trade = {
                "trade_id": trade_id,
                "side": side,
                "qty": qty,
                "entry_price": price,
                "entry_ts": datetime.now(timezone.utc),
            }
            self.session_trade_count += 1
            self.event_logger.append(
                "order_filled",
                symbol=self.symbol,
                trade_id=trade_id,
                broker_order_id=f"paper-{client_order_id}",
                side=side,
                qty=qty,
                fill_price=price,
                slippage_pips=0.0,
                spread_pips=self._spread_pips(tick),
            )
            logger.info(
                f"[PAPER] Opened {side} position at {price:.5f} size={qty:.2f}, "
                f"SL={sl:.5f}, TP={tp:.5f}, trade_id={trade_id}"
            )
            return True

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
            "comment": self._build_order_comment(self.order_comment_prefix, "open"),
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,
        }

        result = mt5.order_send(request)
        if result is None:
            logger.error(f"Failed to open position: order_send returned None ({mt5.last_error()})")
            return False
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Failed to open position: {result.comment}")
            return False

        fill_price = float(result.price if getattr(result, "price", 0.0) else price)
        self.position = 1 if order_type == MT5_ORDER_TYPE_BUY else -1
        self.entry_price = fill_price
        self.current_trade = {
            "trade_id": trade_id,
            "side": side,
            "qty": qty,
            "entry_price": fill_price,
            "entry_ts": datetime.now(timezone.utc),
        }
        self.session_trade_count += 1
        self.event_logger.append(
            "order_filled",
            symbol=self.symbol,
            trade_id=trade_id,
            broker_order_id=str(getattr(result, "order", "") or getattr(result, "deal", "") or ""),
            side=side,
            qty=qty,
            fill_price=fill_price,
            slippage_pips=abs(fill_price - price) / self.pip_size,
            spread_pips=self._spread_pips(self._get_tick(advance=False)),
        )
        logger.info(
            f"Opened {side} position at {fill_price:.5f} with size {position_size}, "
            f"SL: {sl:.5f}, TP: {tp:.5f}, trade_id={trade_id}"
        )
        return True
    def _close_position(self, exit_reason: str = "manual_disable"):
        """Close a position (FIFO-safe) and emit canonical close events."""
        now_utc = datetime.now(timezone.utc)

        if self.execution_mode == "paper":
            if self.current_trade is None:
                return False
            tick = self._get_tick(advance=False)
            if tick is None:
                tick = self._get_tick(advance=True)
            if tick is None:
                logger.error("No tick data available; cannot close paper position.")
                return False
            side = str(self.current_trade.get("side", "long"))
            entry_price = float(self.current_trade.get("entry_price", self.entry_price))
            qty = float(self.current_trade.get("qty", self.position_size))
            trade_id = str(self.current_trade.get("trade_id", self._next_trade_id()))
            entry_ts = self.current_trade.get("entry_ts", now_utc)
            exit_price = float(tick.bid) if side == "long" else float(tick.ask)
            pips = (exit_price - entry_price) / self.pip_size if side == "long" else (entry_price - exit_price) / self.pip_size
            pnl_usd = float(pips * 10.0 * qty)
            hold_seconds = max(0, int((now_utc - entry_ts).total_seconds()))

            self.trade_history.append(
                {
                    "type": side,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "profit": pnl_usd,
                    "timestamp": datetime.now(),
                }
            )
            self.session_net_pnl_usd += pnl_usd
            self.event_logger.append(
                "position_closed",
                symbol=self.symbol,
                trade_id=trade_id,
                exit_reason=exit_reason,
                entry_price=entry_price,
                exit_price=exit_price,
                pnl_usd=pnl_usd,
                hold_seconds=hold_seconds,
                fees_usd=0.0,
            )
            self.position = 0
            self.entry_price = 0
            self.current_trade = None
            logger.info(f"[PAPER] Closed {side} at {exit_price:.5f} pnl_usd={pnl_usd:.2f} trade_id={trade_id}")
            return True

        positions = self._get_symbol_positions()
        if not positions:
            return False

        position = min(positions, key=lambda p: p.time)
        ticket = position.ticket
        close_tick = self._get_tick(advance=False)
        if close_tick is None:
            logger.error("No tick data available; cannot close live position.")
            return False
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": position.volume,
            "type": mt5.ORDER_TYPE_SELL if position.type == MT5_ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
            "position": ticket,
            "price": close_tick.bid if position.type == MT5_ORDER_TYPE_BUY else close_tick.ask,
            "deviation": self.calculate_dynamic_deviation(),
            "magic": self.magic,
            "comment": self._build_order_comment(self.order_comment_prefix, "close"),
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,
        }

        result = mt5.order_send(request)
        if result is None:
            logger.error(f"Failed to close position: order_send returned None ({mt5.last_error()})")
            return False
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Failed to close position: {result.comment}")
            return False

        side = "long" if position.type == MT5_ORDER_TYPE_BUY else "short"
        entry_price = float(self.current_trade.get("entry_price", position.price_open) if self.current_trade else position.price_open)
        exit_price = float(result.price if getattr(result, "price", 0.0) else request["price"])
        profit = float(position.profit)
        hold_seconds = max(0, int(now_utc.timestamp() - int(position.time)))
        trade_id = str(self.current_trade.get("trade_id") if self.current_trade else f"S1-{datetime.now().strftime('%Y%m%d')}-{int(ticket)}")
        qty = float(position.volume)

        self.trade_history.append(
            {
                "type": side,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "profit": profit,
                "timestamp": datetime.now(),
            }
        )
        self.session_net_pnl_usd += profit
        self.event_logger.append(
            "position_closed",
            symbol=self.symbol,
            trade_id=trade_id,
            exit_reason=exit_reason,
            entry_price=entry_price,
            exit_price=exit_price,
            pnl_usd=profit,
            hold_seconds=hold_seconds,
            fees_usd=0.0,
        )

        self.position = 0
        self.entry_price = 0
        self.current_trade = None
        logger.info(f"Closed position at {exit_price} with profit {profit}, trade_id={trade_id}")
        return True
    def _get_position_ticket(self):
        """Get the current position ticket for this bot instance."""
        positions = self._get_symbol_positions()
        if positions:
            return positions[0].ticket
        return None
    
    def _sync_position_state(self):
        """Sync internal position state with MT5 positions"""
        if self.execution_mode != "live":
            self.position = 0
            self.entry_price = 0
            self.current_trade = None
            logger.info("Paper mode active: starting with neutral simulated state")
            return
        positions = self._get_symbol_positions()
        if positions:
            oldest = min(positions, key=lambda p: p.time)
            self.position = 1 if oldest.type == MT5_ORDER_TYPE_BUY else -1
            self.entry_price = oldest.price_open
            self.current_trade = {
                "trade_id": f"S1-{datetime.now().strftime('%Y%m%d')}-{int(oldest.ticket)}",
                "side": "long" if self.position > 0 else "short",
                "qty": float(getattr(oldest, "volume", 0.0) or 0.0),
                "entry_price": float(oldest.price_open),
                "entry_ts": datetime.fromtimestamp(int(oldest.time), tz=timezone.utc),
            }
            logger.info(f"Synced with existing position (FIFO oldest): {'long' if self.position > 0 else 'short'} at {self.entry_price}")
        else:
            self.position = 0
            self.entry_price = 0
            self.current_trade = None
            logger.info("No existing positions found, starting with neutral state")
    
    def wait_for_next_tick(self):
        """Wait for next market tick"""
        max_wait_seconds = 1.0 if self.execution_mode == "paper" else 10.0
        deadline = time.time() + max_wait_seconds

        while True:
            tick = self._get_tick(advance=True)
            if tick is not None and (self.last_tick_time is None or tick.time > self.last_tick_time):
                self.last_tick_time = tick.time
                return

            if time.time() >= deadline:
                if self.execution_mode == "paper":
                    synthetic = self._next_synthetic_tick()
                    self._last_synthetic_tick = synthetic
                    self.last_tick_time = synthetic.time
                return

            time.sleep(0.1)
    
    def _get_account_balance(self) -> float:
        """Get current account balance"""
        if not MT5_AVAILABLE:
            return 0.0
        account_info = mt5.account_info()
        return account_info.balance if account_info else 0.0

    def close_session(self, status: str = "completed") -> None:
        if self.session_closed:
            return
        self.session_closed = True
        self.event_logger.append(
            "session_end",
            symbol=self.symbol,
            trades=int(self.session_trade_count),
            net_pnl_usd=float(self.session_net_pnl_usd),
            status=status,
        )

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    execution_mode = str(os.getenv("FX_EXECUTION_MODE", "paper")).strip().lower()
    if execution_mode not in {"paper", "live"}:
        execution_mode = "paper"

    if execution_mode == "live":
        if not MT5_AVAILABLE:
            logger.error("MetaTrader5 package is not installed. Install with `pip install MetaTrader5` for live mode.")
            raise SystemExit(1)

        # Initialize MT5 connection
        if not mt5.initialize(
            login=int(MT5_CONFIG['MT5LOGIN']),
            password=MT5_CONFIG['MT5PASSWORD'],
            server=MT5_CONFIG['MT5SERVER']
        ):
            logger.error("Failed to initialize MT5")
            raise SystemExit(1)
        
        logger.info("Checking MT5 connection...")
        account_info = mt5.account_info()
        if account_info is None:
            logger.error("Failed to get account info - MT5 may not be logged in or terminal not running")
            logger.error("Please ensure:")
            logger.error("1. MT5 terminal is running")
            logger.error("2. You are logged in to your account")
            logger.error("3. Auto-trading is enabled")
            raise SystemExit(1)
        
        logger.info(f"MT5 Connected - Account: {account_info.login} ({account_info.name})")
        logger.info(f"Balance: {account_info.balance}")
        logger.info(f"Server: {account_info.server}")
        
        terminal_info = mt5.terminal_info()
        logger.info(f"Terminal info: {terminal_info}")
        if terminal_info:
            logger.info(f"Trade allowed: {terminal_info.trade_allowed}")
            logger.info(f"Connected: {terminal_info.connected}")
            logger.info(f"DLLs allowed: {terminal_info.dlls_allowed}")
            logger.info(f"Trade API: {terminal_info.tradeapi_disabled}")
    else:
        if MT5_AVAILABLE:
            logger.info("Paper mode selected: MT5 available; using MT5 market feed in simulation.")
        else:
            logger.info("Paper mode selected: MT5 unavailable; using synthetic market feed fallback.")
    
    trader: Optional[LiveTradingEnvironment] = None
    session_status = "completed"
    try:
        # Create trading environment
        trader = LiveTradingEnvironment(
            symbol=MT5_CONFIG['MT5SYMBOL'],  # Use symbol from MT5 config
            timeframe=MT5_CONFIG['MT5TIMEFRAME'],  # Use timeframe from MT5 config
            model_path=MT5_CONFIG['MODEL_PATH']  # Use model path from config
        )
        
        # Start trading
        max_duration = int(str(os.getenv("FX_MAX_DURATION_SECONDS", "1800")).strip() or "1800")
        trader.run(max_duration_seconds=max(30, max_duration))
        
    except KeyboardInterrupt:
        logger.info("Trading stopped by user")
        session_status = "stopped"
    except Exception as e:
        logger.error(f"Error in main loop: {e}")
        session_status = "error"
    finally:
        if trader is not None:
            try:
                trader.close_session(status=session_status)
            except Exception as e:
                logger.error(f"Failed to emit session_end event: {e}")
        # Shutdown MT5
        if MT5_AVAILABLE:
            mt5.shutdown() 
