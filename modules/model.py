"""
Model module for the Forex Trading Bot
- Handles reinforcement learning and hyperparameter optimization
"""

import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pickle
import os
from datetime import datetime
from collections import deque
from tqdm import tqdm
from typing import Dict, Tuple, Optional, Union, List, Any
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import logging
import optuna
from optuna.trial import Trial
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from shimmy import GymV21CompatibilityV0
from stable_baselines3 import PPO
import torch
try:
    from .data_fetcher import cov_to_corr
    from .config import (
        MIN_POSITION_SIZE,
        MAX_POSITION_SIZE,
        MODELS_DIRECTORY,
        RL_TRAINING_STEPS,
        RL_WINDOW_SIZE,
        GENETIC_POPULATION_SIZE,
        GENETIC_GENERATIONS,
        MAX_TRADES_PER_WEEK,
        ACTION_OPEN_REWARD,
        ACTION_NO_TRADE_PENALTY,
        ACTION_FORCE_CLOSE_DAYS,
        ACTION_HOLD_FLAT_PENALTY,
        ACTION_HOLD_POSITION_PENALTY,
        ACTION_IDLE_THRESHOLD_STEPS,
        ACTION_IDLE_STEP_PENALTY,
        ACTION_FORCE_TRADE_ENABLED,
        ACTION_FORCE_TRADE_IDLE_STEPS,
        ACTION_MIN_TRADES_WINDOW_STEPS,
        ACTION_MIN_TRADES_PER_WINDOW,
        ACTION_MIN_TRADE_DEFICIT_PENALTY,
        TRAIN_RANDOM_START,
        POSITION_BALANCE_SCALING,
        POSITION_BALANCE_FLOOR,
        EVAL_ACTION_SHAPING,
        REGIME_FILTER_ENABLED,
        REGIME_BLOCK_LOW_EXPECTANCY,
        REGIME_BLOCK_IN_EVAL,
        REGIME_FORCE_EXIT_ON_REVERSAL,
        REGIME_TREND_MIN_STRENGTH,
        REGIME_TREND_REVERSAL_MULT,
        REGIME_VOL_MIN_ATR_NORM,
        REGIME_VOL_MAX_ATR_NORM,
        ENTRY_EXPECTANCY_THRESHOLD,
        ENTRY_EXPECTANCY_REWARD_WEIGHT,
        LOW_EXPECTANCY_ENTRY_PENALTY,
        LOW_EXPECTANCY_CLOSE_PENALTY_SCALE,
        INITIAL_BALANCE,
        EPISODE_LENGTH,
        SPREAD,
        COMMISSION,
        SLIPPAGE,
        PROFIT_TARGET,
        STOP_LOSS,
        REWARD_BALANCE_CHANGE_WEIGHT,
        REWARD_POSITION_HOLDING_PROFIT,
        REWARD_POSITION_HOLDING_LOSS,
        REWARD_TRADE_PROFIT_WEIGHT,
        REWARD_TRADE_LOSS_WEIGHT,
        REWARD_TRADE_FREQUENCY,
        REWARD_MARK_TO_MARKET_WEIGHT,
        PORTFOLIO_EFFICIENT_FRONTIER_POINTS,
        PORTFOLIO_PLOT_HEIGHT,
        PORTFOLIO_PLOT_ASPECT,
        PORTFOLIO_PLOT_STYLE,
        PORTFOLIO_CORRELATION_CMAP,
        PORTFOLIO_CORRELATION_CENTER,
        PORTFOLIO_CORRELATION_FMT,
        PORTFOLIO_PLOT_TITLE_FONTSIZE,
        PORTFOLIO_AXIS_FONTSIZE,
        PPO_LEARNING_RATE,
        PPO_N_STEPS,
        PPO_BATCH_SIZE,
        PPO_N_EPOCHS,
        PPO_GAMMA,
        PPO_GAE_LAMBDA,
        PPO_CLIP_RANGE,
        PPO_ENT_COEF,
        PPO_VF_COEF,
        PPO_MAX_GRAD_NORM,
        PPO_TARGET_KL,
        PPO_USE_SDE,
        PPO_SDE_SAMPLE_FREQ,
        PPO_VERBOSE,
        HP_LEARNING_RATE_MIN,
        HP_LEARNING_RATE_MAX,
        HP_N_STEPS_MIN,
        HP_N_STEPS_MAX,
        HP_BATCH_SIZE_MIN,
        HP_BATCH_SIZE_MAX,
        HP_N_EPOCHS_MIN,
        HP_N_EPOCHS_MAX,
        HP_GAMMA_MIN,
        HP_GAMMA_MAX,
        HP_ENT_COEF_MIN,
        HP_ENT_COEF_MAX
    )
except ImportError:  # pragma: no cover - script mode fallback
    from data_fetcher import cov_to_corr
    from config import (
        MIN_POSITION_SIZE,
        MAX_POSITION_SIZE,
        MODELS_DIRECTORY,
        RL_TRAINING_STEPS,
        RL_WINDOW_SIZE,
        GENETIC_POPULATION_SIZE,
        GENETIC_GENERATIONS,
        MAX_TRADES_PER_WEEK,
        ACTION_OPEN_REWARD,
        ACTION_NO_TRADE_PENALTY,
        ACTION_FORCE_CLOSE_DAYS,
        ACTION_HOLD_FLAT_PENALTY,
        ACTION_HOLD_POSITION_PENALTY,
        ACTION_IDLE_THRESHOLD_STEPS,
        ACTION_IDLE_STEP_PENALTY,
        ACTION_FORCE_TRADE_ENABLED,
        ACTION_FORCE_TRADE_IDLE_STEPS,
        ACTION_MIN_TRADES_WINDOW_STEPS,
        ACTION_MIN_TRADES_PER_WINDOW,
        ACTION_MIN_TRADE_DEFICIT_PENALTY,
        TRAIN_RANDOM_START,
        POSITION_BALANCE_SCALING,
        POSITION_BALANCE_FLOOR,
        EVAL_ACTION_SHAPING,
        REGIME_FILTER_ENABLED,
        REGIME_BLOCK_LOW_EXPECTANCY,
        REGIME_BLOCK_IN_EVAL,
        REGIME_FORCE_EXIT_ON_REVERSAL,
        REGIME_TREND_MIN_STRENGTH,
        REGIME_TREND_REVERSAL_MULT,
        REGIME_VOL_MIN_ATR_NORM,
        REGIME_VOL_MAX_ATR_NORM,
        ENTRY_EXPECTANCY_THRESHOLD,
        ENTRY_EXPECTANCY_REWARD_WEIGHT,
        LOW_EXPECTANCY_ENTRY_PENALTY,
        LOW_EXPECTANCY_CLOSE_PENALTY_SCALE,
        INITIAL_BALANCE,
        EPISODE_LENGTH,
        SPREAD,
        COMMISSION,
        SLIPPAGE,
        PROFIT_TARGET,
        STOP_LOSS,
        REWARD_BALANCE_CHANGE_WEIGHT,
        REWARD_POSITION_HOLDING_PROFIT,
        REWARD_POSITION_HOLDING_LOSS,
        REWARD_TRADE_PROFIT_WEIGHT,
        REWARD_TRADE_LOSS_WEIGHT,
        REWARD_TRADE_FREQUENCY,
        REWARD_MARK_TO_MARKET_WEIGHT,
        PORTFOLIO_EFFICIENT_FRONTIER_POINTS,
        PORTFOLIO_PLOT_HEIGHT,
        PORTFOLIO_PLOT_ASPECT,
        PORTFOLIO_PLOT_STYLE,
        PORTFOLIO_CORRELATION_CMAP,
        PORTFOLIO_CORRELATION_CENTER,
        PORTFOLIO_CORRELATION_FMT,
        PORTFOLIO_PLOT_TITLE_FONTSIZE,
        PORTFOLIO_AXIS_FONTSIZE,
        PPO_LEARNING_RATE,
        PPO_N_STEPS,
        PPO_BATCH_SIZE,
        PPO_N_EPOCHS,
        PPO_GAMMA,
        PPO_GAE_LAMBDA,
        PPO_CLIP_RANGE,
        PPO_ENT_COEF,
        PPO_VF_COEF,
        PPO_MAX_GRAD_NORM,
        PPO_TARGET_KL,
        PPO_USE_SDE,
        PPO_SDE_SAMPLE_FREQ,
        PPO_VERBOSE,
        HP_LEARNING_RATE_MIN,
        HP_LEARNING_RATE_MAX,
        HP_N_STEPS_MIN,
        HP_N_STEPS_MAX,
        HP_BATCH_SIZE_MIN,
        HP_BATCH_SIZE_MAX,
        HP_N_EPOCHS_MIN,
        HP_N_EPOCHS_MAX,
        HP_GAMMA_MIN,
        HP_GAMMA_MAX,
        HP_ENT_COEF_MIN,
        HP_ENT_COEF_MAX
    )

# Configure logging
logger = logging.getLogger(__name__)

class DebugLogger:
    """Simple debug logger for tracking trading metrics"""
    def __init__(self) -> None:
        self.log_file: str = "trading_journal.log"
        logging.basicConfig(
            filename=self.log_file,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    def log_error(self, error: Exception, context: Optional[str] = None) -> None:
        """Log error with optional context"""
        error_msg: str = f"Error: {type(error).__name__} - {str(error)}"
        if context:
            error_msg += f" | Context: {context}"
        logger.error(error_msg)

# Constants
RL_TRAINING_STEPS = RL_TRAINING_STEPS
RL_WINDOW_SIZE = RL_WINDOW_SIZE
MODELS_DIRECTORY = MODELS_DIRECTORY
GENETIC_POPULATION_SIZE = GENETIC_POPULATION_SIZE
GENETIC_GENERATIONS = GENETIC_GENERATIONS

# PPO Parameters
PPO_PARAMS = {
    'learning_rate': PPO_LEARNING_RATE,
    'n_steps': PPO_N_STEPS,
    'batch_size': PPO_BATCH_SIZE,
    'n_epochs': PPO_N_EPOCHS,
    'gamma': PPO_GAMMA,
    'gae_lambda': PPO_GAE_LAMBDA,
    'clip_range': PPO_CLIP_RANGE,
    'ent_coef': PPO_ENT_COEF,
    'vf_coef': PPO_VF_COEF,
    'max_grad_norm': PPO_MAX_GRAD_NORM,
    'target_kl': PPO_TARGET_KL,
    'use_sde': PPO_USE_SDE,
    'sde_sample_freq': PPO_SDE_SAMPLE_FREQ,
    'verbose': PPO_VERBOSE
}

FEATURE_IDX = {
    "ret_1": 0,
    "ret_3": 1,
    "ret_6": 2,
    "ret_12": 3,
    "ret_24": 4,
    "hl_spread": 5,
    "oc_delta": 6,
    "ma_fast_slow": 7,
    "atr_norm": 8,
    "vol_z": 9,
    "weekly_return": 10,
    "monthly_return": 11,
    "rolling_3m_vol": 12,
    "range_compression": 13,
    "dist_52w_high": 14,
    "dist_52w_low": 15,
}

class TradingEnvironment(gym.Env):
    """
    Trading environment for reinforcement learning.

    `mode` controls whether action-level anti-collapse shaping is active:
    - train: shaping enabled by default
    - eval/live: shaping disabled by default
    """

    def __init__(
        self,
        df: pd.DataFrame,
        initial_balance: float = INITIAL_BALANCE,
        mode: str = "train",
        apply_action_shaping: Optional[bool] = None,
    ) -> None:
        super().__init__()
        
        # Store data as numpy arrays for memory efficiency
        self.data: Dict[str, np.ndarray] = {
            'open': df['open'].values,
            'high': df['high'].values,
            'low': df['low'].values,
            'close': df['close'].values,
            'volume': df['volume'].values if 'volume' in df.columns else np.zeros(len(df))
        }
        self.market_features: np.ndarray = self._build_market_features(df)
        self.data_len: int = len(self.data["close"])
        
        self.current_step: int = 0
        self.initial_balance: float = initial_balance
        self.balance: float = initial_balance
        self.position: int = 0
        self.position_size: float = MIN_POSITION_SIZE
        self.trade_history: List[Dict[str, Any]] = []
        mode_value = str(mode).strip().lower()
        if mode_value not in {"train", "eval", "live"}:
            mode_value = "train"
        self.mode: str = mode_value
        if apply_action_shaping is None:
            self.apply_action_shaping = self.mode == "train" or (
                self.mode in {"eval", "live"} and EVAL_ACTION_SHAPING
            )
        else:
            self.apply_action_shaping = bool(apply_action_shaping)
        
        # Trading costs
        self.spread: float = SPREAD      # Cost of the spread (difference between bid and ask)
        self.commission: float = COMMISSION  # Broker's fee per trade (0.01% of trade value)
        self.slippage: float = SLIPPAGE  # Price movement during order execution
        self.profit_target: float = float(PROFIT_TARGET)
        self.stop_loss: float = float(STOP_LOSS)
        
        # Episode tracking
        self.episode: int = 0
        self.episode_profit: float = 0
        self.episode_loss: float = 0
        self.episode_winning_trades: int = 0
        self.episode_losing_trades: int = 0
        self.episode_largest_win: float = 0
        self.episode_largest_loss: float = 0
        self.episode_trades: int = 0
        self.episode_start_balance: float = initial_balance
        self.episode_start_step: int = 0
        
        # Weekly trade tracking
        self.weekly_trades: int = 0
        self.last_week: Optional[Union[int, Tuple[int, int]]] = None
        self.steps_since_trade: int = 0
        self.trade_open_steps = deque(maxlen=max(ACTION_MIN_TRADES_WINDOW_STEPS, 1))
        self.trade_executed_last_step: bool = False
        self.realized_pnl_last_step: float = 0.0
        self.prev_unrealized_pnl: float = 0.0
        
        # Track profit factors for averaging
        self.profit_factors: List[float] = []
        
        # Define action and observation spaces
        # 0: hold, 1: buy, 2: sell, 3: close
        self.action_space: spaces.Discrete = spaces.Discrete(4)
        
        # Calculate observation space size (market features + dynamic position state)
        n_features: int = int(self.market_features.shape[1] + 4)
        self.observation_space: spaces.Box = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(n_features,),
            dtype=np.float32
        )
        
        # Initialize trade tracking
        self.current_trade: Optional[Dict[str, Any]] = None
        self.weekly_trades = 0
        self.last_week = None
        self.df_index = df.index if hasattr(df, 'index') else None
        self.bars_per_day: int = self._infer_bars_per_day()
        self.steps_per_week: int = max(self.bars_per_day * 7, 1)
        self.episode_steps: int = max(int(EPISODE_LENGTH * self.bars_per_day), 1)
        self.force_close_steps: int = max(int(ACTION_FORCE_CLOSE_DAYS * self.bars_per_day), 1)
        self.train_random_start: bool = bool(TRAIN_RANDOM_START)
        if self.mode in {"eval", "live"}:
            self.train_random_start = False
        self.position_balance_scaling: bool = bool(POSITION_BALANCE_SCALING)
        self.position_balance_floor: float = float(POSITION_BALANCE_FLOOR)
        self.regime_filter_enabled: bool = bool(REGIME_FILTER_ENABLED)
        self.regime_block_low_expectancy: bool = bool(REGIME_BLOCK_LOW_EXPECTANCY)
        self.regime_block_in_eval: bool = bool(REGIME_BLOCK_IN_EVAL)
        self.regime_force_exit_on_reversal: bool = bool(REGIME_FORCE_EXIT_ON_REVERSAL)
        self.regime_trend_min_strength: float = float(REGIME_TREND_MIN_STRENGTH)
        self.regime_trend_reversal_mult: float = float(REGIME_TREND_REVERSAL_MULT)
        self.regime_vol_min_atr_norm: float = float(REGIME_VOL_MIN_ATR_NORM)
        self.regime_vol_max_atr_norm: float = float(REGIME_VOL_MAX_ATR_NORM)
        self.entry_expectancy_threshold: float = float(ENTRY_EXPECTANCY_THRESHOLD)
        self.entry_expectancy_reward_weight: float = float(ENTRY_EXPECTANCY_REWARD_WEIGHT)
        self.low_expectancy_entry_penalty: float = float(LOW_EXPECTANCY_ENTRY_PENALTY)
        self.low_expectancy_close_penalty_scale: float = float(LOW_EXPECTANCY_CLOSE_PENALTY_SCALE)
        self.max_episode_start: int = max(self.data_len - self.episode_steps - 2, 0)
        
        # Initialize debug logging
        self.debug_logger: DebugLogger = DebugLogger()
        logger.info(f"Trading Environment initialized with {len(df)} data points")
        if self.mode in {"eval", "live"} and not self.apply_action_shaping:
            logger.info(
                f"Environment mode: {self.mode} | action_shaping={self.apply_action_shaping} "
                f"(expected for unbiased evaluation)"
            )
        else:
            logger.info(f"Environment mode: {self.mode} | action_shaping={self.apply_action_shaping}")
        logger.info(
            f"time_resolution bars_per_day={self.bars_per_day} episode_steps={self.episode_steps} "
            f"force_close_steps={self.force_close_steps}"
        )
        logger.info(
            f"train_random_start={self.train_random_start} max_episode_start={self.max_episode_start} "
            f"position_balance_scaling={self.position_balance_scaling} position_balance_floor={self.position_balance_floor}"
        )
        logger.info(
            f"regime_filter={self.regime_filter_enabled} block_eval={self.regime_block_in_eval} "
            f"trend_min={self.regime_trend_min_strength:.6f} atr_min={self.regime_vol_min_atr_norm:.6f} "
            f"atr_max={self.regime_vol_max_atr_norm:.6f} expectancy_threshold={self.entry_expectancy_threshold:.6f}"
        )
        logger.info(f"Initial balance: ${initial_balance:,.2f}")
        logger.info(f"Position sizing is dynamic. Initial size: {self.position_size} lots")
        logger.info(f"Trading costs - Spread: {self.spread*10000} pips, Commission: {self.commission*100}%")
        logger.info(f"Risk controls - Profit target: {self.profit_target:.4f}, Stop loss: {self.stop_loss:.4f}")

    def _infer_bars_per_day(self) -> int:
        """Infer bars/day from datetime index; fallback to 24."""
        if self.df_index is not None and len(self.df_index) >= 3:
            try:
                idx = pd.to_datetime(self.df_index)
                delta_hours = (
                    idx.to_series().diff().dropna().dt.total_seconds().median() / 3600.0
                )
                if delta_hours and delta_hours > 0:
                    inferred = int(round(24.0 / float(delta_hours)))
                    return max(inferred, 1)
            except Exception:
                pass
        return 24

    def _week_bucket(self, step_idx: int) -> Union[int, Tuple[int, int]]:
        """Return a stable week identifier for the given step."""
        if self.df_index is not None and 0 <= step_idx < len(self.df_index):
            try:
                ts = pd.Timestamp(self.df_index[step_idx])
                iso = ts.isocalendar()
                return int(iso.year), int(iso.week)
            except Exception:
                pass
        return int(step_idx) // self.steps_per_week

    def _build_market_features(self, df: pd.DataFrame) -> np.ndarray:
        """Build market feature matrix used by observations."""
        close = df["close"].astype(float)
        open_ = df["open"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        volume = df["volume"].astype(float) if "volume" in df.columns else pd.Series(0.0, index=df.index)

        bars_per_day = 24
        if isinstance(df.index, pd.DatetimeIndex) and len(df.index) >= 3:
            try:
                diffs = df.index.to_series().diff().dropna().dt.total_seconds().astype(float)
                if not diffs.empty:
                    median_step = float(np.nanmedian(diffs.values))
                    if np.isfinite(median_step) and median_step > 0:
                        inferred = int(round(86400.0 / median_step))
                        bars_per_day = int(np.clip(inferred, 1, 1440))
            except Exception:
                pass

        week_window = max(5 * bars_per_day, 10)
        month_window = max(21 * bars_per_day, 30)
        quarter_window = max(63 * bars_per_day, 90)
        year_52w_window = max(252 * bars_per_day, 250)

        close_safe = close.replace(0.0, np.nan)
        open_safe = open_.replace(0.0, np.nan)
        vol_mean = volume.rolling(24, min_periods=1).mean()
        vol_std = volume.rolling(24, min_periods=1).std().replace(0.0, np.nan)
        ret_1 = close.pct_change(1)

        features = pd.DataFrame(index=df.index)
        features["ret_1"] = ret_1
        features["ret_3"] = close.pct_change(3)
        features["ret_6"] = close.pct_change(6)
        features["ret_12"] = close.pct_change(12)
        features["ret_24"] = close.pct_change(24)
        features["hl_spread"] = (high - low) / close_safe
        features["oc_delta"] = (close - open_) / open_safe
        features["ma_fast_slow"] = (
            close.rolling(12, min_periods=1).mean() - close.rolling(48, min_periods=1).mean()
        ) / close_safe
        features["atr_norm"] = (high - low).rolling(14, min_periods=1).mean() / close_safe
        features["vol_z"] = (volume - vol_mean) / vol_std
        features["weekly_return"] = close.pct_change(week_window)
        features["monthly_return"] = close.pct_change(month_window)
        features["rolling_3m_vol"] = ret_1.rolling(quarter_window, min_periods=10).std()

        range_short = high.rolling(week_window, min_periods=10).max() - low.rolling(week_window, min_periods=10).min()
        range_long = high.rolling(quarter_window, min_periods=20).max() - low.rolling(quarter_window, min_periods=20).min()
        features["range_compression"] = range_short / range_long.replace(0.0, np.nan)

        high_52w = high.rolling(year_52w_window, min_periods=50).max()
        low_52w = low.rolling(year_52w_window, min_periods=50).min()
        features["dist_52w_high"] = (close / high_52w) - 1.0
        features["dist_52w_low"] = (close / low_52w) - 1.0

        features = features.replace([np.inf, -np.inf], 0.0).fillna(0.0)
        return features.astype(np.float32).values
        
    def _calculate_metrics(self) -> Dict[str, Union[float, int]]:
        """Calculate episode trading metrics"""
        if self.episode_trades == 0:
            return {
                'profit_factor': 0.0,
                'avg_profit_factor': 0.0,
                'win_rate': 0.0,
                'avg_profit': 0.0,
                'avg_loss': 0.0,
                'largest_win': 0.0,
                'largest_loss': 0.0,
                'net_profit': 0.0,
                'return_pct': 0.0
            }
            
        win_rate: float = self.episode_winning_trades / self.episode_trades
        avg_profit: float = self.episode_profit / self.episode_winning_trades if self.episode_winning_trades > 0 else 0
        avg_loss: float = self.episode_loss / self.episode_losing_trades if self.episode_losing_trades > 0 else 0
        
        # Calculate current episode profit factor
        profit_factor: float = float('inf') if self.episode_loss == 0 and self.episode_profit > 0 else (
            abs(self.episode_profit / self.episode_loss) if self.episode_loss != 0 else 0
        )
            
        # Add to profit factors list
        self.profit_factors.append(profit_factor)
        
        # Calculate average profit factor
        avg_profit_factor: float = sum(self.profit_factors) / len(self.profit_factors)
        
        net_profit: float = self.episode_profit - self.episode_loss
        return_pct: float = (net_profit / self.episode_start_balance) * 100
        
        return {
            'profit_factor': profit_factor,
            'avg_profit_factor': avg_profit_factor,
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'largest_win': self.episode_largest_win,
            'largest_loss': self.episode_largest_loss,
            'net_profit': net_profit,
            'return_pct': return_pct
        }
        
    def _log_trading_metrics(self):
        """Log current episode trading metrics"""
        metrics = self._calculate_metrics()
        # Use EPISODE_LENGTH for the label and self.episode for the episode number
        logger.info(f"\n=== {EPISODE_LENGTH}-Day Trading Metrics (Episode {self.episode}) ===")
        logger.info(f"Starting Balance: ${self.episode_start_balance:,.2f}")
        logger.info(f"Current Balance: ${self.balance:,.2f}")
        logger.info(f"Net Profit: ${metrics['net_profit']:,.2f} ({metrics['return_pct']:.2f}%)")
        logger.info(f"Total Trades: {self.episode_trades}")
        logger.info(f"Win Rate: {metrics['win_rate']*100:.2f}%")
        logger.info(f"Average Profit Factor: {metrics['avg_profit_factor']:.2f}")
        logger.info(f"Average Profit: ${metrics['avg_profit']:,.2f}")
        logger.info(f"Average Loss: ${metrics['avg_loss']:,.2f}")
        logger.info(f"Largest Win: ${metrics['largest_win']:,.2f}")
        logger.info(f"Largest Loss: ${metrics['largest_loss']:,.2f}")
        logger.info("=============================\n")
        
    def _update_metrics(self, pnl):
        """Update episode trading metrics with new trade result"""
        self.episode_trades += 1
        if pnl > 0:
            self.episode_winning_trades += 1
            self.episode_profit += pnl
            self.episode_largest_win = max(self.episode_largest_win, pnl)
        else:
            self.episode_losing_trades += 1
            self.episode_loss += abs(pnl)
            self.episode_largest_loss = min(self.episode_largest_loss, pnl)
            
    def _get_observation(self):
        """Get current market state observation"""
        market_obs = self.market_features[self.current_step]
        unrealized_pnl = 0.0
        if self.current_trade is not None and self.position != 0:
            entry_price = float(self.current_trade['entry_price'])
            trade_size = self._active_trade_size()
            current_price = float(self.data['close'][self.current_step])
            unrealized_pnl = (current_price - entry_price) * self.position * trade_size * 100000

        dynamic_obs = np.array(
            [
                float(self.position),
                float(self.weekly_trades) / max(float(MAX_TRADES_PER_WEEK), 1.0),
                min(float(self.steps_since_trade) / max(float(self.steps_per_week), 1.0), 1.0),
                float(unrealized_pnl) / max(float(self.initial_balance), 1.0),
            ],
            dtype=np.float32,
        )
        return np.concatenate([market_obs.astype(np.float32), dynamic_obs]).astype(np.float32)
    
    def _calculate_position_size(self) -> float:
        """Calculate position size based on win rate"""
        closed_trades = [
            trade for trade in self.trade_history
            if float(trade.get('reward', 0.0)) != 0.0
        ]
        if not closed_trades:
            return MIN_POSITION_SIZE
        
        # Calculate win rate
        total_trades = len(closed_trades)
        winning_trades = sum(1 for trade in closed_trades if float(trade['reward']) > 0)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Calculate position size based on win rate
        # Linear scaling from MIN_POSITION_SIZE to MAX_POSITION_SIZE based on win rate
        position_size = MIN_POSITION_SIZE + (MAX_POSITION_SIZE - MIN_POSITION_SIZE) * win_rate
        
        # Ensure position size stays within bounds
        return np.clip(position_size, MIN_POSITION_SIZE, MAX_POSITION_SIZE)

    def _sample_episode_start(self) -> int:
        """Sample episode start index for train mode to improve data coverage."""
        if self.mode == "train" and self.train_random_start and self.max_episode_start > 0:
            return int(self.np_random.integers(0, self.max_episode_start + 1))
        return 0

    def _active_trade_size(self) -> float:
        """Return lot size of the currently open trade."""
        if self.current_trade is not None:
            return float(self.current_trade.get("position_size", self.position_size))
        return float(self.position_size)

    def _entry_position_size(self) -> float:
        """Compute lot size for a new entry with optional equity-based scaling."""
        base_size = float(np.clip(self.position_size, MIN_POSITION_SIZE, MAX_POSITION_SIZE))
        if not self.position_balance_scaling:
            return base_size
        balance_ratio = self.balance / max(self.initial_balance, 1e-9)
        scale = float(np.clip(balance_ratio, self.position_balance_floor, 1.0))
        return float(np.clip(base_size * scale, MIN_POSITION_SIZE, MAX_POSITION_SIZE))

    def _direction_sign(self, direction: str) -> float:
        return 1.0 if str(direction).upper() == "LONG" else -1.0

    def _get_feature(self, feature_name: str) -> float:
        idx = FEATURE_IDX[feature_name]
        return float(self.market_features[self.current_step, idx])

    def _entry_expectancy_score(self, direction: str) -> Tuple[float, Dict[str, float]]:
        """Compute directional expectancy score from trend/momentum/volatility context."""
        sign = self._direction_sign(direction)
        trend = self._get_feature("ma_fast_slow")
        ret6 = self._get_feature("ret_6")
        ret24 = self._get_feature("ret_24")
        atr_norm = abs(self._get_feature("atr_norm"))
        hl_spread = abs(self._get_feature("hl_spread"))
        oc_delta = self._get_feature("oc_delta")
        vol_z = self._get_feature("vol_z")

        trend_aligned = sign * trend
        momentum_aligned = sign * (0.65 * ret6 + 0.35 * ret24)

        vol_center = 0.5 * (self.regime_vol_min_atr_norm + self.regime_vol_max_atr_norm)
        vol_half_band = max((self.regime_vol_max_atr_norm - self.regime_vol_min_atr_norm) * 0.5, 1e-9)
        vol_deviation = abs(atr_norm - vol_center) / vol_half_band
        vol_penalty = max(vol_deviation - 1.0, 0.0)
        choppy_penalty = max(abs(vol_z) - 2.5, 0.0)

        score = (
            1.5 * trend_aligned
            + 1.0 * momentum_aligned
            - 0.3 * abs(oc_delta)
            - 0.2 * hl_spread
            - 0.15 * vol_penalty
            - 0.08 * choppy_penalty
        )
        diagnostics = {
            "trend_aligned": trend_aligned,
            "momentum_aligned": momentum_aligned,
            "atr_norm": atr_norm,
            "vol_z": vol_z,
            "score": score,
        }
        return score, diagnostics

    def _entry_gate(self, direction: str) -> Tuple[bool, float, Dict[str, float]]:
        """
        Gate entries by regime/trend/volatility and directional expectancy.
        Returns (allow_entry, score, diagnostics).
        """
        score, diagnostics = self._entry_expectancy_score(direction)
        if not self.regime_filter_enabled:
            return True, score, diagnostics

        trend_ok = diagnostics["trend_aligned"] >= self.regime_trend_min_strength
        atr_norm = diagnostics["atr_norm"]
        vol_ok = self.regime_vol_min_atr_norm <= atr_norm <= self.regime_vol_max_atr_norm
        expectancy_ok = score >= self.entry_expectancy_threshold

        block_low_expectancy = self.regime_block_low_expectancy
        if self.mode in {"eval", "live"} and not self.regime_block_in_eval:
            block_low_expectancy = False

        allow_entry = trend_ok and vol_ok and (expectancy_ok or not block_low_expectancy)
        diagnostics.update(
            {
                "trend_ok": float(trend_ok),
                "vol_ok": float(vol_ok),
                "expectancy_ok": float(expectancy_ok),
            }
        )
        return allow_entry, score, diagnostics

    def _low_expectancy_penalty_ratio(self, score: float) -> float:
        if self.entry_expectancy_threshold <= 0:
            return 0.0
        deficit = max(self.entry_expectancy_threshold - float(score), 0.0)
        ratio = deficit / max(self.entry_expectancy_threshold, 1e-9)
        return min(max(ratio, 0.0), 5.0)

    def _entry_reward_reweight(self, score: float) -> float:
        """Reward high expectancy entries and penalize low expectancy attempts."""
        reward_adj = self.entry_expectancy_reward_weight * float(score)
        if score < self.entry_expectancy_threshold:
            reward_adj += self.low_expectancy_entry_penalty * self._low_expectancy_penalty_ratio(score)
        return reward_adj

    def _close_expectancy_penalty(self) -> float:
        """Penalty applied when closing trades opened under weak expectancy."""
        if not self.apply_action_shaping or self.current_trade is None:
            return 0.0
        score = float(self.current_trade.get("entry_expectancy", 0.0))
        return self.low_expectancy_close_penalty_scale * self._low_expectancy_penalty_ratio(score) * 0.002

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one time step within the environment"""
        max_step = self.data_len - 1
        if self.current_step >= max_step:
            info = {
                "balance": self.balance,
                "position": self.position,
                "position_size": self._active_trade_size(),
                "current_price": float(self.data["close"][self.current_step]),
                "weekly_trades": self.weekly_trades,
                "current_trade": self.current_trade,
                "trade_executed": False,
                "realized_pnl": 0.0,
                "metrics": self._calculate_metrics(),
                "episode_length": EPISODE_LENGTH,
                "episode_steps": self.episode_steps,
                "episode_start_step": self.episode_start_step,
                "current_month": (self.current_step // max(self.bars_per_day * 30, 1)) + 1,
            }
            return self._get_observation(), 0.0, True, False, info

        self.current_step += 1
        elapsed_steps = self.current_step - self.episode_start_step
        episode_done = elapsed_steps > 0 and (elapsed_steps % self.episode_steps == 0)
        done: bool = episode_done or self.current_step >= max_step

        # Get current price
        current_price: float = self.data['close'][self.current_step]
        
        # Reset weekly trades if it's a new week
        current_week = self._week_bucket(self.current_step)
        if self.last_week is not None and current_week != self.last_week:
            self.weekly_trades = 0
            self.last_week = current_week
        elif self.last_week is None:
            self.last_week = current_week
        
        # Update position size based on win rate
        self.position_size = self._calculate_position_size()
        
        # Execute trade
        reward: float = self._execute_trade(action, current_price)
        
        # Get observation
        obs: np.ndarray = self._get_observation()
        
        # Log trading metrics every 10 episodes
        if done and self.episode % 10 == 0:
            self._log_trading_metrics()
        
        # Additional info
        info: Dict[str, Any] = {
            'balance': self.balance,
            'position': self.position,
            'position_size': self._active_trade_size(),
            'current_price': current_price,
            'weekly_trades': self.weekly_trades,
            'current_trade': self.current_trade,
            'trade_executed': self.trade_executed_last_step,
            'realized_pnl': self.realized_pnl_last_step,
            'metrics': self._calculate_metrics(),
            'episode_length': EPISODE_LENGTH,
            'episode_steps': self.episode_steps,
            'episode_start_step': self.episode_start_step,
            'current_month': (self.current_step // max(self.bars_per_day * 30, 1)) + 1
        }
        
        return obs, reward, done, False, info
    
    def reset(self, seed=None, options=None):
        """Reset environment for new episode"""
        super().reset(seed=seed)
        self.episode_start_step = self._sample_episode_start()
        self.current_step = self.episode_start_step
        self.balance = self.initial_balance
        self.position = 0
        self.position_size = MIN_POSITION_SIZE
        self.current_trade = None
        self.weekly_trades = 0
        self.last_week = self._week_bucket(self.current_step)
        self.steps_since_trade = 0
        self.trade_open_steps.clear()
        self.trade_executed_last_step = False
        self.realized_pnl_last_step = 0.0
        self.prev_unrealized_pnl = 0.0
        self.trade_history = []  # Reset trade history for the new episode
        # Reset episode metrics
        self.episode_profit = 0
        self.episode_loss = 0
        self.episode_winning_trades = 0
        self.episode_losing_trades = 0
        self.episode_largest_win = 0
        self.episode_largest_loss = 0
        self.episode_trades = 0
        self.episode_start_balance = self.initial_balance
        self.episode += 1
        return self._get_observation(), {}
    
    def _execute_trade(self, action: int, current_price: float) -> float:
        """Execute trade based on action"""
        reward: float = 0.0
        trade_executed = False
        action_value = int(np.asarray(action).reshape(-1)[0])
        self.trade_executed_last_step = False
        self.realized_pnl_last_step = 0.0

        # Regime reversal exit: flatten when directional trend turns strongly against position.
        if (
            self.current_trade
            and self.position != 0
            and self.regime_filter_enabled
            and self.regime_force_exit_on_reversal
        ):
            position_direction = "LONG" if self.position > 0 else "SHORT"
            _, diagnostics = self._entry_expectancy_score(position_direction)
            reversal_threshold = self.regime_trend_min_strength * self.regime_trend_reversal_mult
            if diagnostics["trend_aligned"] <= -reversal_threshold:
                exit_price = current_price * (1 + self.slippage if self.position == -1 else 1 - self.slippage)
                close_penalty = self._close_expectancy_penalty()
                pnl = self._close_position(exit_price)
                reward += (pnl / self.initial_balance) - close_penalty
                self.realized_pnl_last_step += pnl
                self.current_trade = None
                trade_executed = True

        # Risk-based auto-exit using configured target/stop thresholds.
        if self.current_trade and self.position != 0 and not trade_executed:
            entry_price = float(self.current_trade["entry_price"])
            if entry_price > 0:
                move_pct = ((current_price - entry_price) / entry_price) * float(self.position)
                if move_pct >= self.profit_target or move_pct <= -self.stop_loss:
                    exit_price = current_price * (1 + self.slippage if self.position == -1 else 1 - self.slippage)
                    close_penalty = self._close_expectancy_penalty()
                    pnl = self._close_position(exit_price)
                    reward += (pnl / self.initial_balance) - close_penalty
                    self.realized_pnl_last_step += pnl
                    self.current_trade = None
                    trade_executed = True
        
        # Force close stale positions after a configurable number of days.
        if (
            self.current_trade
            and (self.current_step - self.current_trade['entry_step']) >= self.force_close_steps
            and not trade_executed
        ):
            close_penalty = self._close_expectancy_penalty()
            pnl: float = self._close_position(current_price)
            reward += (pnl / self.initial_balance) - close_penalty  # Normalize reward
            self.realized_pnl_last_step += pnl
            self.current_trade = None
            trade_executed = True
        
        # Execute new trade if we haven't reached weekly limit
        if self.weekly_trades < MAX_TRADES_PER_WEEK and not trade_executed:  # Use imported constant
            if action_value == 1 and self.position == 0:  # Buy
                allow_entry, score, _ = self._entry_gate("LONG")
                if allow_entry:
                    entry_price: float = current_price * (1 + self.slippage)
                    self._open_position(entry_price, "LONG", entry_expectancy=score)
                    self.weekly_trades += 1
                    if self.apply_action_shaping:
                        reward += ACTION_OPEN_REWARD + self._entry_reward_reweight(score)
                    trade_executed = True
                    self.trade_open_steps.append(self.current_step)
                elif self.apply_action_shaping:
                    reward += self.low_expectancy_entry_penalty * max(self._low_expectancy_penalty_ratio(score), 1.0) * 0.5
            elif action_value == 2 and self.position == 0:  # Sell
                allow_entry, score, _ = self._entry_gate("SHORT")
                if allow_entry:
                    entry_price: float = current_price * (1 - self.slippage)
                    self._open_position(entry_price, "SHORT", entry_expectancy=score)
                    self.weekly_trades += 1
                    if self.apply_action_shaping:
                        reward += ACTION_OPEN_REWARD + self._entry_reward_reweight(score)
                    trade_executed = True
                    self.trade_open_steps.append(self.current_step)
                elif self.apply_action_shaping:
                    reward += self.low_expectancy_entry_penalty * max(self._low_expectancy_penalty_ratio(score), 1.0) * 0.5
            elif action_value == 3 and self.position != 0:  # Close position
                # Apply slippage to exit price
                close_penalty = self._close_expectancy_penalty()
                exit_price: float = current_price * (1 + self.slippage if self.position == -1 else 1 - self.slippage)
                pnl: float = self._close_position(exit_price)
                reward += (pnl / self.initial_balance) - close_penalty  # Normalize reward
                self.realized_pnl_last_step += pnl
                self.current_trade = None
                trade_executed = True

        if trade_executed:
            self.steps_since_trade = 0
        else:
            self.steps_since_trade += 1
            if self.apply_action_shaping and action_value == 0:
                if self.position == 0:
                    reward += ACTION_HOLD_FLAT_PENALTY
                else:
                    reward += ACTION_HOLD_POSITION_PENALTY

            # Minimum trade-frequency constraint: force a directional trade when idle too long.
            if (
                self.apply_action_shaping
                and self.mode == "train"
                and ACTION_FORCE_TRADE_ENABLED
                and self.position == 0
                and self.weekly_trades < MAX_TRADES_PER_WEEK
                and ACTION_FORCE_TRADE_IDLE_STEPS > 0
                and self.steps_since_trade >= ACTION_FORCE_TRADE_IDLE_STEPS
            ):
                long_allow, long_score, _ = self._entry_gate("LONG")
                short_allow, short_score, _ = self._entry_gate("SHORT")
                candidate = None
                if long_allow:
                    candidate = ("LONG", long_score)
                if short_allow and (candidate is None or short_score > candidate[1]):
                    candidate = ("SHORT", short_score)
                if candidate is not None:
                    direction, score = candidate
                    entry_price = current_price * (1 + self.slippage if direction == "LONG" else 1 - self.slippage)
                    self._open_position(entry_price, direction, entry_expectancy=score)
                    self.weekly_trades += 1
                    self.steps_since_trade = 0
                    self.trade_open_steps.append(self.current_step)
                    reward += ACTION_OPEN_REWARD + self._entry_reward_reweight(score)
                    trade_executed = True
                else:
                    reward += self.low_expectancy_entry_penalty * 0.5

        if (
            self.apply_action_shaping
            and ACTION_IDLE_THRESHOLD_STEPS > 0
            and self.steps_since_trade > ACTION_IDLE_THRESHOLD_STEPS
        ):
            idle_overage = self.steps_since_trade - ACTION_IDLE_THRESHOLD_STEPS
            reward += ACTION_IDLE_STEP_PENALTY * idle_overage

        if self.apply_action_shaping and ACTION_MIN_TRADES_WINDOW_STEPS > 0 and ACTION_MIN_TRADES_PER_WINDOW > 0:
            window_start = self.current_step - ACTION_MIN_TRADES_WINDOW_STEPS
            while self.trade_open_steps and self.trade_open_steps[0] < window_start:
                self.trade_open_steps.popleft()
            trade_deficit = max(ACTION_MIN_TRADES_PER_WINDOW - len(self.trade_open_steps), 0)
            if trade_deficit > 0:
                reward += ACTION_MIN_TRADE_DEFICIT_PENALTY * trade_deficit

        # Add dense, mark-to-market reward when in position to reduce sparse-reward collapse.
        if self.current_trade is not None and self.position != 0:
            entry_price = float(self.current_trade['entry_price'])
            trade_size = self._active_trade_size()
            unrealized_pnl = (current_price - entry_price) * self.position * trade_size * 100000
            delta_unrealized = unrealized_pnl - self.prev_unrealized_pnl
            reward += (delta_unrealized / self.initial_balance) * REWARD_MARK_TO_MARKET_WEIGHT
            self.prev_unrealized_pnl = unrealized_pnl
        else:
            self.prev_unrealized_pnl = 0.0
        
        # Penalize if no trades were made this week
        if self.apply_action_shaping and self.weekly_trades == 0 and self.current_step > 0:
            reward += ACTION_NO_TRADE_PENALTY

        self.trade_executed_last_step = trade_executed

        return reward

    def _calculate_trading_cost(self, price: float, direction: str, position_size: Optional[float] = None) -> float:
        """Calculate trading costs for a position with dynamic adjustment"""
        lot_size = float(position_size if position_size is not None else self.position_size)
        # Calculate dynamic trading costs based on market volatility
        dynamic_costs = self._calculate_dynamic_trading_costs()
        
        # Spread cost (adjusted by volatility)
        spread_cost = dynamic_costs['spread'] * lot_size * 100000
        
        # Commission cost
        commission_cost = price * lot_size * 100000 * self.commission
        
        # Slippage cost (adjusted by volatility)
        slippage_cost = dynamic_costs['slippage'] * lot_size * 100000
        
        return spread_cost + commission_cost + slippage_cost
    
    def _calculate_dynamic_trading_costs(self) -> Dict[str, float]:
        """Calculate dynamic trading costs based on market volatility"""
        try:
            # Calculate volatility from recent price data
            if self.current_step < 20:
                return {'spread': self.spread, 'slippage': self.slippage}
            
            # Get recent price data for volatility calculation
            recent_prices = self.data['close'][max(0, self.current_step-20):self.current_step]
            if len(recent_prices) < 10:
                return {'spread': self.spread, 'slippage': self.slippage}
            
            # Calculate price volatility
            price_changes = np.diff(recent_prices) / recent_prices[:-1]
            volatility = np.std(price_changes)
            
            # Adjust trading costs based on volatility
            # Higher volatility = higher trading costs
            volatility_multiplier = 1 + (volatility * 100)  # Scale volatility
            
            dynamic_spread = self.spread * volatility_multiplier
            dynamic_slippage = self.slippage * volatility_multiplier
            
            # Ensure costs don't become unreasonable
            dynamic_spread = min(dynamic_spread, self.spread * 3)  # Max 3x base spread
            dynamic_slippage = min(dynamic_slippage, self.slippage * 3)  # Max 3x base slippage
            
            return {
            'spread': dynamic_spread,
            'slippage': dynamic_slippage
}
            
        except Exception as e:
            # Fallback to base costs if calculation fails
            return {'spread': self.spread, 'slippage': self.slippage}

    @property
    def current_datetime(self):
        # Get the datetime for the current step from the data index
        # Assumes the original DataFrame index is datetime and matches the data arrays
        if hasattr(self, 'df_index'):
            return self.df_index[self.current_step]
        return None

    def _open_position(self, price: float, direction: str, entry_expectancy: float = 0.0):
        """Open a new position"""
        self.position = 1 if direction == 'LONG' else -1
        entry_position_size = self._entry_position_size()
        # Calculate trading costs
        trading_cost = self._calculate_trading_cost(price, direction, position_size=entry_position_size)
        self.balance -= trading_cost
        self.current_trade = {
            'direction': direction,
            'entry_price': price,
            'entry_step': self.current_step,
            'trading_cost': trading_cost,
            'position_size': entry_position_size,
            'entry_expectancy': float(entry_expectancy),
            'timestamp': self.current_datetime  # Add timestamp for trade open
        }
        # Append trade to trade_history for per-episode stats
        self.trade_history.append({
            'direction': direction,
            'entry_price': price,
            'entry_step': self.current_step,
            'trading_cost': trading_cost,
            'reward': 0.0,  # Will be updated on close
            'timestamp': self.current_datetime,  # Add timestamp for trade open
            'position_size': entry_position_size,  # Ensure position_size is always present
            'entry_expectancy': float(entry_expectancy),
        })

    def _close_position(self, price: float) -> float:
        """Close current position and calculate P&L"""
        if not self.current_trade:
            return 0
        entry_price: float = self.current_trade['entry_price']
        trade_size = float(self.current_trade.get('position_size', self.position_size))
        # Calculate trading costs
        trading_cost = self._calculate_trading_cost(
            price,
            self.current_trade['direction'],
            position_size=trade_size,
        )
        # Calculate net P&L including close costs.
        pnl: float = (price - entry_price) * self.position * trade_size * 100000
        pnl -= trading_cost
        self.balance += pnl
        # Update metrics
        self._update_metrics(pnl)
        # Update reward and close_timestamp in trade_history for the last trade
        if self.trade_history:
            self.trade_history[-1]['reward'] = pnl
            self.trade_history[-1]['close_step'] = self.current_step
            self.trade_history[-1]['close_timestamp'] = self.current_datetime  # Add close timestamp
            # Ensure position_size is present in the last trade (for legacy/truncated records)
            if 'position_size' not in self.trade_history[-1]:
                self.trade_history[-1]['position_size'] = trade_size
        self.position = 0
        return pnl

class TradingModel(pl.LightningModule):
    """PyTorch Lightning module for trading model"""
    def __init__(self, learning_rate=3e-4, env=None):
        super().__init__()
        self.save_hyperparameters()
        self.env = env
        self.model = None
        self.current_reward = 0
        self.scaler = StandardScaler()
        
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess data for model training"""
        # Create feature set
        feature_cols = ['open', 'high', 'low', 'close', 'volume']
        
        # Filter only available columns
        available_cols = [col for col in feature_cols if col in df.columns]
        
        # Scale features
        df[available_cols] = self.scaler.fit_transform(df[available_cols])
        
        return df
        
    def setup(self, stage=None):
        if stage == 'fit':
            if self.env is None:
                raise ValueError("TradingModel requires an environment before calling setup(stage='fit').")
            self.model = PPO('MlpPolicy', self.env)
            
    def training_step(self, batch, batch_idx):
        # Training logic with progress bar
        with tqdm(total=1000, desc=f"Training step {batch_idx}") as pbar:
            self.model.learn(
                total_timesteps=1000,
                progress_bar=True,
                callback=lambda locals, globals: pbar.update(1)
            )
        self.log('train_reward', self.current_reward)
        return self.current_reward
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
    
    def save_model(self, filepath: str = None):
        """Save the trained model to disk"""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        if filepath is None:
            # Create models directory if it doesn't exist
            if not os.path.exists(MODELS_DIRECTORY):
                os.makedirs(MODELS_DIRECTORY)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(MODELS_DIRECTORY, f"ppo_model_{timestamp}")
        
        # Save model
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")
        
        return filepath
    
    @classmethod
    def load_model(cls, filepath: str):
        """Load a saved model from disk"""
        model = cls()
        model.model = PPO.load(filepath)
        logger.info(f"Model loaded from {filepath}")
        return model

class PortfolioOptimizer:
    """Portfolio optimizer for currency pairs"""
    def __init__(self, returns_data: pd.DataFrame):
        self.returns_data = returns_data
        self.mean_returns = None
        self.cov_matrix = None
        self._preprocess_data()
    
    def _preprocess_data(self):
        """Preprocess returns data"""
        self.mean_returns = self.returns_data.mean()
        self.cov_matrix = self.returns_data.cov()

    def efficient_frontier(self, points=PORTFOLIO_EFFICIENT_FRONTIER_POINTS):
        """Generate random efficient frontier samples."""
        if self.mean_returns is None or self.cov_matrix is None:
            raise ValueError("Returns data is not initialized.")

        n_assets = len(self.mean_returns)
        if n_assets == 0:
            raise ValueError("No assets available for efficient frontier calculation.")

        volatilities = []
        returns = []
        sharpe_ratios = []
        cov_values = np.asarray(self.cov_matrix)

        for _ in range(max(int(points), 1)):
            weights = np.random.random(n_assets)
            weights /= np.sum(weights)

            portfolio_return = float(np.dot(self.mean_returns.values, weights))
            portfolio_volatility = float(np.sqrt(weights.T @ cov_values @ weights))
            sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0.0

            returns.append(portfolio_return)
            volatilities.append(portfolio_volatility)
            sharpe_ratios.append(sharpe_ratio)

        return {
            "volatility": volatilities,
            "return": returns,
            "sharpe_ratio": sharpe_ratios,
        }
        
    def plot_efficient_frontier(self, points=PORTFOLIO_EFFICIENT_FRONTIER_POINTS):
        """Plot the efficient frontier using seaborn"""
        # Calculate efficient frontier points
        frontier_data = self.efficient_frontier(points)
        
        # Create DataFrame for plotting
        frontier_df = pd.DataFrame({
            'Volatility': frontier_data['volatility'],
            'Return': frontier_data['return'],
            'Sharpe Ratio': frontier_data['sharpe_ratio']
        })
        
        # Create the plot
        sns.set_style(PORTFOLIO_PLOT_STYLE)
        plot = sns.relplot(
            data=frontier_df,
            x='Volatility',
            y='Return',
            hue='Sharpe Ratio',
            palette='viridis',
            height=PORTFOLIO_PLOT_HEIGHT,
            aspect=PORTFOLIO_PLOT_ASPECT
        )
        
        # Customize the plot
        plot.fig.suptitle('Efficient Frontier', fontsize=PORTFOLIO_PLOT_TITLE_FONTSIZE, y=1.02)
        plot.ax.set_xlabel('Portfolio Volatility', fontsize=PORTFOLIO_AXIS_FONTSIZE)
        plot.ax.set_ylabel('Expected Return', fontsize=PORTFOLIO_AXIS_FONTSIZE)
        
        return plot

    def plot_correlation_heatmap(self):
        """Plot correlation heatmap using seaborn"""
        # Calculate correlation matrix using imported function
        corr_matrix = cov_to_corr(self.cov_matrix)
        
        # Create the plot
        sns.set_style(PORTFOLIO_PLOT_STYLE)
        plot = sns.heatmap(
            corr_matrix,
            annot=True,
            cmap=PORTFOLIO_CORRELATION_CMAP,
            center=PORTFOLIO_CORRELATION_CENTER,
            square=True,
            fmt=PORTFOLIO_CORRELATION_FMT,
            cbar_kws={'label': 'Correlation'}
        )
        
        # Customize the plot
        plot.set_title('Asset Correlation Matrix', fontsize=PORTFOLIO_PLOT_TITLE_FONTSIZE, pad=20)
        
        return plot

    def plot_returns_distribution(self):
        """Plot returns distribution using seaborn"""
        # Create the plot
        sns.set_style(PORTFOLIO_PLOT_STYLE)
        plot = sns.displot(
            data=self.returns_data,
            kind='kde',
            height=PORTFOLIO_PLOT_HEIGHT,
            aspect=PORTFOLIO_PLOT_ASPECT
        )
        
        # Customize the plot
        plot.fig.suptitle('Returns Distribution', fontsize=PORTFOLIO_PLOT_TITLE_FONTSIZE, y=1.02)
        plot.ax.set_xlabel('Return', fontsize=PORTFOLIO_AXIS_FONTSIZE)
        plot.ax.set_ylabel('Density', fontsize=PORTFOLIO_AXIS_FONTSIZE)
        
        return plot

class CustomRewardWrapper(gym.Wrapper):
    """Custom reward wrapper for PPO model"""
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.last_balance = env.initial_balance
        self.last_position = 0
        self.last_trade_count = 0
        self.last_trade_pnl = 0.0
        self.last_trade_time = None
        self.trade_history = []
        self.episode_rewards = []
        self.current_episode_reward = 0.0
        self.initial_balance = env.initial_balance
        self.consecutive_no_trade_steps = 0
        
    def step(self, action):
        observation, reward, done, truncated, info = self.env.step(action)
        
        # Get current state
        current_balance = info.get('balance', self.last_balance)
        current_position = info.get('position', self.last_position)
        trade_history = getattr(self.env, 'trade_history', [])
        current_trade_count = len(trade_history)
        trade_executed = current_trade_count > self.last_trade_count
        current_trade_pnl = 0.0
        if trade_executed and trade_history:
            current_trade_pnl = float(trade_history[-1].get('reward', reward))
        
        # Calculate reward components
        reward = 0.0
        
        # 1. Balance change reward (primary component)
        balance_change = current_balance - self.last_balance
        if balance_change != 0:
            reward += balance_change / self.last_balance * REWARD_BALANCE_CHANGE_WEIGHT
        
        # 2. Position holding reward/penalty
        if current_position != 0:
            if balance_change > 0:
                reward += REWARD_POSITION_HOLDING_PROFIT * abs(current_position)
            elif balance_change < 0:
                reward -= REWARD_POSITION_HOLDING_LOSS * abs(current_position)
        
        # 3. Trade execution reward/penalty
        if trade_executed:
            self.consecutive_no_trade_steps = 0
            if current_trade_pnl > 0:
                reward += REWARD_TRADE_PROFIT_WEIGHT * (current_trade_pnl / self.last_balance)
            else:
                reward -= REWARD_TRADE_LOSS_WEIGHT * abs(current_trade_pnl / self.last_balance)
        
        # 4. Trade frequency reward
        if current_trade_count > self.last_trade_count:
            reward += REWARD_TRADE_FREQUENCY
        
        # Update last state
        self.last_balance = current_balance
        self.last_position = current_position
        self.last_trade_count = current_trade_count
        self.last_trade_pnl = current_trade_pnl
        
        # Track episode reward
        self.current_episode_reward += reward
        
        # If episode is done, store the episode reward
        if done:
            self.episode_rewards.append(self.current_episode_reward)
            self.current_episode_reward = 0.0
            self.consecutive_no_trade_steps = 0
        
        return observation, reward, done, truncated, info
    
    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.last_balance = self.initial_balance
        self.last_position = 0
        self.last_trade_count = 0
        self.last_trade_pnl = 0.0
        self.last_trade_time = None
        self.current_episode_reward = 0.0
        self.consecutive_no_trade_steps = 0
        return observation, info

def optimize_hyperparameters(df: pd.DataFrame, n_trials=50):
    """Optimize PPO hyperparameters using Optuna"""
    if df is None or df.empty:
        raise ValueError("optimize_hyperparameters requires a non-empty DataFrame.")

    def objective(trial: Trial):
        # Define hyperparameter search space
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
            'n_steps': trial.suggest_int('n_steps', 512, 2048),
            'batch_size': trial.suggest_int('batch_size', 32, 256),
            'n_epochs': trial.suggest_int('n_epochs', 5, 20),
            'gamma': trial.suggest_float('gamma', 0.9, 0.9999),
            'ent_coef': trial.suggest_float('ent_coef', 0.01, 0.1)
        }
        
        # Create and train model with these parameters
        env = TradingEnvironment(df)
        model = PPO('MlpPolicy', env, **params)
        model.learn(total_timesteps=10000)
        
        # Evaluate model
        mean_reward = evaluate_model(model, df)
        return mean_reward

    # Create study and optimize with progress bar
    study = optuna.create_study(direction='maximize')
    with tqdm(total=n_trials, desc="Optimizing hyperparameters") as pbar:
        study.optimize(
            objective, 
            n_trials=n_trials,
            callbacks=[lambda study, trial: pbar.update(1)]
        )
    
    return study.best_params

def create_compatible_env(env):
    """Create a compatible environment using shimmy"""
    return GymV21CompatibilityV0(env)

def evaluate_model(model, df: pd.DataFrame, n_episodes=10):
    """Evaluate model performance over multiple episodes"""
    if df is None or df.empty:
        raise ValueError("evaluate_model requires a non-empty DataFrame.")
    env = TradingEnvironment(df)
    data_len = len(env.data['close'])
    rewards = []
    for _ in range(n_episodes):
        obs = env.reset()[0]
        done = False
        episode_reward = 0
        while not done and (env.current_step + 1) < data_len:
            action, _ = model.predict(obs)
            obs, reward, done, _, _ = env.step(action)
            episode_reward += reward
        rewards.append(episode_reward)
    return np.mean(rewards)

def train_ppo_model(df: pd.DataFrame, n_trials: int = 50, max_epochs: int = 100) -> Dict[str, Any]:
    """
    Train PPO reinforcement learning model with hyperparameter optimization
    
    Parameters
    ----------
    df : pd.DataFrame
        Training data
    n_trials : int
        Number of Optuna trials for hyperparameter optimization
    max_epochs : int
        Maximum number of training epochs
        
    Returns
    -------
    Dict[str, Any]
        Training results and model information
    """
    # Create environment
    env = TradingEnvironment(df)
    compatible_env = create_compatible_env(env)
    
    # Create studies directory if it doesn't exist
    studies_dir = os.path.join(MODELS_DIRECTORY, 'studies')
    if not os.path.exists(studies_dir):
        os.makedirs(studies_dir)
    
    def objective(trial: Trial):
        # Define hyperparameter search space using imported constants
        params = {
            'learning_rate': trial.suggest_float('learning_rate', HP_LEARNING_RATE_MIN, HP_LEARNING_RATE_MAX, log=True),
            'n_steps': trial.suggest_int('n_steps', HP_N_STEPS_MIN, HP_N_STEPS_MAX),
            'batch_size': trial.suggest_int('batch_size', HP_BATCH_SIZE_MIN, HP_BATCH_SIZE_MAX),
            'n_epochs': trial.suggest_int('n_epochs', HP_N_EPOCHS_MIN, HP_N_EPOCHS_MAX),
            'gamma': trial.suggest_float('gamma', HP_GAMMA_MIN, HP_GAMMA_MAX),
            'ent_coef': trial.suggest_float('ent_coef', HP_ENT_COEF_MIN, HP_ENT_COEF_MAX)
        }
        
        # Create model with trial parameters
        model = TradingModel(
            learning_rate=params['learning_rate'],
            env=compatible_env
        )
        
        # Create trainer with trial-specific callbacks
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            callbacks=[
                ModelCheckpoint(
                    monitor='train_reward',
                    mode='max',
                    filename=f'trial_{trial.number}'
                ),
                EarlyStopping(
                    monitor='train_reward',
                    patience=5,
                    mode='max'
                )
            ],
            enable_progress_bar=True
        )
        
        # Train model
        trainer.fit(model)
        
        # Evaluate model
        mean_reward = evaluate_model(model.model, df)
        
        # Report trial result
        trial.report(mean_reward, step=trainer.current_epoch)
        
        # Handle pruning
        if trial.should_prune():
            raise optuna.TrialPruned()
            
        return mean_reward
    
    # Create study and optimize
    study = optuna.create_study(direction='maximize')
    with tqdm(total=n_trials, desc="Optimizing hyperparameters") as pbar:
        study.optimize(
            objective, 
            n_trials=n_trials,
            callbacks=[lambda study, trial: pbar.update(1)]
        )
    
    # Get best trial parameters
    best_params = study.best_params
    logger.info(f"Best hyperparameters: {best_params}")
    
    # Save study results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    study_path = os.path.join(studies_dir, f"study_{timestamp}")
    
    # Save study object
    with open(f"{study_path}.pkl", "wb") as f:
        pickle.dump(study, f)
    
    # Save study visualization
    try:
        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_html(f"{study_path}_history.html")
        
        fig = optuna.visualization.plot_param_importances(study)
        fig.write_html(f"{study_path}_importances.html")
        
        fig = optuna.visualization.plot_parallel_coordinate(study)
        fig.write_html(f"{study_path}_parallel.html")
    except Exception as e:
        logger.warning(f"Failed to create study visualizations: {e}")
    
    # Train final model with best parameters
    final_model = TradingModel(
        learning_rate=best_params['learning_rate'],
        env=compatible_env
    )
    
    final_trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=[
            ModelCheckpoint(monitor='train_reward', mode='max'),
            EarlyStopping(monitor='train_reward', patience=5, mode='max')
        ],
        enable_progress_bar=True
    )
    
    final_trainer.fit(final_model)
    
    # Evaluate final model
    mean_reward = evaluate_model(final_model.model, df)
    logger.info(f"Final mean reward: {mean_reward:.2f}")
    
    # Save model
    model_path = os.path.join(MODELS_DIRECTORY, f"ppo_model_{timestamp}")
    final_model.model.save(model_path)
    
    return {
        'model_path': model_path,
        'mean_reward': mean_reward,
        'hyperparameters': best_params,
        'study_path': study_path
    }
