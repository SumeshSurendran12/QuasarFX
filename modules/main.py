"""
Main training script for the Forex Trading Bot
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
import time
import pandas as pd
import numpy as np
import random
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym
from gymnasium import spaces
import wandb
from model import TradingEnvironment
from debug import DebugLogger
import torch
import torch.nn as nn
from config import (
    MIN_POSITION_SIZE, MAX_POSITION_SIZE, SYMBOL, TIMEFRAME, YEARS,
    MAX_EPISODES, MAX_TIMESTEPS, MIN_EPISODES, TARGET_WEEKLY_PROFIT, MAX_TRADES_PER_WEEK,
    BATCH_SIZE, N_STEPS,
    REWARD_SHAPING,
    INITIAL_LR, FINAL_LR, MIN_LR, WARMUP_STEPS,
    DEVICE_TYPE,
    PPO_PARAMS,
    PPO_VERBOSE,
    EPISODE_LENGTH,
    GRID_SEARCH_PARAMS
)
import json
from data_fetcher import DataFetcher



# Custom Features Extractor
class CustomFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 64):
        super().__init__(observation_space, features_dim)
        
        # Get the actual input shape from the observation space
        n_input = int(np.prod(observation_space.shape))
        
        # Create a more memory-efficient network with proper normalization
        self.network = nn.Sequential(
            nn.Flatten(),  # Flatten the input
            nn.LayerNorm(n_input),  # Add layer normalization
            nn.Linear(n_input, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.LayerNorm(64),  # Add layer normalization
            nn.Linear(64, features_dim),
            nn.ReLU()
        )
        
        # Initialize weights using Kaiming initialization for ReLU
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.zeros_(m.bias)
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Add input validation and normalization
        if torch.isnan(observations).any():
            observations = torch.nan_to_num(observations, nan=0.0)
        return self.network(observations)

# Device selection based on config
if DEVICE_TYPE == "directml":
    try:
        import torch_directml
        DEVICE = torch_directml.device()
        print(f"Using DirectML device: {DEVICE}")
    except Exception as e:
        print(f"Error configuring DirectML: {e}")
        print("Falling back to CPU")
        DEVICE = "cpu"
elif torch.cuda.is_available():
    DEVICE = "cuda"
    print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
else:
    DEVICE = "cpu"
    print("Using CPU device")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define reward shaping configuration
DEFAULT_REWARD_CONFIG = {
    "weekly_trade_bonus": 0.0,
    "win_streak_bonus": 0.0,
    "loss_penalty": 0.0,
    "weekly_profit_bonus": 0.0,
}

# Model selection configuration (tune these for higher-quality search)
MODEL_SELECTION_ENABLED = True
MODEL_SELECTION_TRIALS = 6
MODEL_SELECTION_SEEDS = [0, 1, 2]
MODEL_SELECTION_TIMESTEPS = max(20000, MAX_TIMESTEPS // 5)
FINAL_TRAIN_SEEDS = [0, 1, 2]
VAL_FRACTION = 0.1
TEST_FRACTION = 0.1


def learning_rate_schedule(progress):
    """
    Learning rate schedule with controlled decay and stability monitoring.
    Uses a cosine decay with warmup and minimum learning rate floor.
    
    Args:
        progress: Training progress from 0 to 1
        
    Returns:
        float: Current learning rate
    """
    # Learning rate parameters from config
    initial_lr = INITIAL_LR
    final_lr = FINAL_LR
    min_lr = MIN_LR  # Minimum learning rate floor
    warmup_steps = WARMUP_STEPS  # Fraction of training for warmup
    progress = max(0.0, min(1.0, float(progress)))
    
    if progress < warmup_steps:
        # Linear warmup
        return initial_lr * (progress / warmup_steps)
    else:
        # Cosine decay with minimum floor
        progress = (progress - warmup_steps) / (1 - warmup_steps)  # Adjust progress for decay phase
        cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))  # Cosine decay from 1 to 0
        lr = final_lr + (initial_lr - final_lr) * cosine_decay
        return max(lr, min_lr)  # Ensure learning rate doesn't go below minimum
    
    

def fetch_historical_data(download_if_missing=False, symbol=None, timeframe=None, start_date=None, end_date=None):
    """Download historical data using DataFetcher
    
    Args:
        symbol: Trading symbol (e.g., 'EURUSD')
        timeframe: Timeframe in Alpha Vantage format ('1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w')
        start_date: Optional start date for calculating the number of years to request
        end_date: Optional end date for calculating the number of years to request
    """
    symbol = symbol or SYMBOL
    timeframe = timeframe or TIMEFRAME
    end_date = end_date or datetime.now()
    start_date = start_date or end_date - timedelta(days=YEARS * 365)
    years = max(1, int(round((end_date - start_date).days / 365)))
    
    logger.info(f"Downloading {years} years of historical data for {symbol} at {timeframe} timeframe")
    
    try:
        # Calculate expected data points
        total_days = (end_date - start_date).days
        if timeframe == "15m":
            expected_points = total_days * 24 * 4  # 4 periods per hour, 24 hours per day
        else:
            expected_points = total_days

        logger.info("\n=== Data Download Statistics ===")
        logger.info(f"Start date: {start_date}")
        logger.info(f"End date: {end_date}")
        logger.info(f"Total days: {total_days:,}")
        logger.info(f"Expected data points: {expected_points:,}")
        logger.info(f"Timeframe: {timeframe}")
        logger.info("==============================\n")

        # Use DataFetcher to get historical data
        fetcher = DataFetcher()
        df = fetcher.fetch_historical_data(
            download_if_missing=download_if_missing,
            symbol=symbol,
            timeframe=timeframe,
            years=years,
        )

        if df is None or df.empty:
            logger.error("No data received for the specified date range")
            return None
                
        # Save to CSV using Path
        output_file = Path("data") / f"{symbol}_{timeframe}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
        output_file.parent.mkdir(exist_ok=True)
        df.to_csv(output_file)
        logger.info(f"Saved historical data to {output_file}")
        return df

    except Exception as e:
        logger.error(f"Error downloading historical data: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None

class CustomRewardWrapper(gym.Wrapper):
    def __init__(self, env, reward_config=None):
        super().__init__(env)
        merged_config = DEFAULT_REWARD_CONFIG.copy()
        if reward_config is not None:
            merged_config.update(reward_config)
        self.reward_config = merged_config
        self.win_streak = 0
        self.loss_streak = 0
        self.weekly_trades = 0
        self.weekly_profit = 0
        self.last_week = None
        self.trade_history = []
        self.weekly_trade_cap = MAX_TRADES_PER_WEEK
        self.position_size = MIN_POSITION_SIZE  # Start with minimum position size

    def _calculate_position_size(self):
        """Calculate position size based on win rate"""
        if not self.trade_history:
            return MIN_POSITION_SIZE
        
        # Calculate win rate
        total_trades = len(self.trade_history)
        winning_trades = sum(1 for trade in self.trade_history if trade['reward'] > 0)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Calculate position size based on win rate
        # Linear scaling from MIN_POSITION_SIZE to MAX_POSITION_SIZE based on win rate
        position_size = MIN_POSITION_SIZE + (MAX_POSITION_SIZE - MIN_POSITION_SIZE) * win_rate
        
        # Ensure position size stays within bounds
        return np.clip(position_size, MIN_POSITION_SIZE, MAX_POSITION_SIZE)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        
        # Get current week
        current_week = pd.Timestamp.now().isocalendar()[1]
        
        # Reset weekly counters if it's a new week
        if self.last_week is not None and current_week != self.last_week:
            self.weekly_trades = 0
            self.weekly_profit = 0
        self.last_week = current_week
        
        # Update position size based on win rate
        self.position_size = self._calculate_position_size()
        
        # Apply position size to info
        if action in [1, 2]:  # Buy or Sell actions
            info['position_size'] = self.position_size
        
        # Update trade history
        if action in [1, 2]:  # Buy or Sell actions
            self.trade_history.append({
                'action': action,
                'reward': reward,
                'position_size': self.position_size,
                'timestamp': pd.Timestamp.now()
            })
            
            # Update weekly trade count
            self.weekly_trades += 1
            
            # Apply weekly trade cap penalty
            if self.weekly_trades > self.weekly_trade_cap:
                reward -= self.reward_config['weekly_trade_bonus']
        
        # Update win/loss streaks
        if reward > 0: # type: ignore
            self.win_streak += 1
            self.loss_streak = 0
        elif reward < 0: # type: ignore
            # If reward is negative, it counts as a loss
            self.loss_streak += 1
            self.win_streak = 0

        # Apply win streak bonus
        if self.win_streak > 1:
            reward += self.reward_config['win_streak_bonus'] * min(self.win_streak, 5)
        
        # Apply loss streak penalty
        if self.loss_streak > 1:
            reward -= self.reward_config['loss_penalty'] * min(self.loss_streak, 5)
        
        # Update weekly profit
        self.weekly_profit += reward # type: ignore
        
        # Apply weekly profit bonus
        if self.weekly_profit > 0:
            reward += self.reward_config['weekly_profit_bonus'] * min(self.weekly_profit / TARGET_WEEKLY_PROFIT, 1.0)
        
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        observation, info = self.env.reset(seed=seed, options=options)
        self.win_streak = 0
        self.loss_streak = 0
        self.weekly_trades = 0
        self.weekly_profit = 0
        self.last_week = pd.Timestamp.now().isocalendar()[1]
        self.trade_history = []
        self.position_size = MIN_POSITION_SIZE  # Reset to minimum position size
        return observation, info

def make_env(rank, df, seed=0, reward_config=None):
    """
    Create a new environment instance
    """
    def _init():
        env = TradingEnvironment(df.copy())  # Use a copy for each environment
        env = Monitor(env, f"logs/monitor-{rank}")
        env = CustomRewardWrapper(env, reward_config or REWARD_SHAPING)
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init

def get_base_env(env):
    """Get the base environment without wrappers"""
    while hasattr(env, 'env'):
        env = env.env
    return env

class ProgressCallback(BaseCallback):
    def __init__(self, debug_logger, initial_balance=10000, update_interval=60, verbose=0):
        super().__init__(verbose)
        self.debug_logger = debug_logger
        self.initial_balance = initial_balance
        self.update_interval = update_interval
        self.last_update = time.time()
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0.0
        self.current_episode_length = 0
        self.episode_count = 0
        self.last_episode_reward = None
        self.trade_history = []
        self.best_reward = float('-inf')
        self.best_model_path = None
        self.monthly_performance = {}
        self.current_month = None

    def _calculate_trade_stats(self, trade_history):
        """Calculate trade statistics"""
        if not trade_history:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'avg_profit': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0,
                'max_consecutive_wins': 0,
                'max_consecutive_losses': 0
            }
        
        total_trades = len(trade_history)
        winning_trades = sum(1 for trade in trade_history if trade['reward'] > 0)
        losing_trades = sum(1 for trade in trade_history if trade['reward'] < 0)
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        profits = [trade['reward'] for trade in trade_history if trade['reward'] > 0]
        losses = [abs(trade['reward']) for trade in trade_history if trade['reward'] < 0]
        
        avg_profit = sum(profits) / len(profits) if profits else 0
        avg_loss = sum(losses) / len(losses) if losses else 0
        
        profit_factor = sum(profits) / sum(losses) if sum(losses) > 0 else float('inf')
        
        # Calculate consecutive wins/losses
        current_streak = 0
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        
        for trade in trade_history:
            if trade['reward'] > 0:
                if current_streak > 0:
                    current_streak += 1
                else:
                    current_streak = 1
                max_consecutive_wins = max(max_consecutive_wins, current_streak)
            else:
                if current_streak < 0:
                    current_streak -= 1
                else:
                    current_streak = -1
                max_consecutive_losses = max(max_consecutive_losses, abs(current_streak))
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_consecutive_wins': max_consecutive_wins,
            'max_consecutive_losses': max_consecutive_losses
        }

    def save_model(self, model, episode, reward, is_best=False):
        """Save model checkpoint"""
        # Create models directory if it doesn't exist
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)

        # Save regular checkpoint
        checkpoint_path = models_dir / f"checkpoint_{episode}.zip"
        model.save(str(checkpoint_path))
            
        # Save best model if this is the best so far
        if is_best:
            best_model_path = models_dir / "best_model.zip"
            model.save(str(best_model_path))
            self.best_model_path = str(best_model_path)
                
            # Save model metadata
            metadata = {
                    'episode': episode,
                    'reward': reward,
                'timestamp': datetime.now().isoformat(),
                'trade_stats': self._calculate_trade_stats(self.trade_history)
                }
                
            metadata_path = models_dir / "best_model_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)

    def _on_step(self):
        """Called at each step during training"""
        # Get current time
        current_time = time.time()

        rewards = self.locals.get('rewards')
        dones = self.locals.get('dones')
        if rewards is not None:
            if np.isscalar(rewards):
                self.current_episode_reward += float(rewards)
            else:
                self.current_episode_reward += float(rewards[0])
        self.current_episode_length += 1
        if dones is not None:
            done = bool(dones) if np.isscalar(dones) else bool(dones[0])
            if done:
                self.episode_count += 1
                self.last_episode_reward = self.current_episode_reward
                self.episode_rewards.append(self.current_episode_reward)
                self.episode_lengths.append(self.current_episode_length)
                self.current_episode_reward = 0.0
                self.current_episode_length = 0
        
        # Update only if enough time has passed
        if current_time - self.last_update >= self.update_interval:
            self.last_update = current_time
            
            # Get current episode info - use EPISODE_LENGTH from config instead of max_steps
            episode = self.episode_count
            episode_reward = self.last_episode_reward if self.last_episode_reward is not None else self.current_episode_reward
            episode_length = self.episode_lengths[-1] if self.episode_lengths else self.current_episode_length
                    
            # Calculate running averages
            avg_reward = np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0
            avg_length = np.mean(self.episode_lengths[-100:]) if self.episode_lengths else 0

        # Get trade history from the environment
            env = get_base_env(self.training_env.envs[0]) # type: ignore
            if hasattr(env, 'trade_history'):
                self.trade_history = env.trade_history
            
            # Calculate trade statistics
            trade_stats = self._calculate_trade_stats(self.trade_history)
                
            # Calculate monthly performance
            self._calculate_monthly_performance()
            
            # Log progress
            self.debug_logger.log_training_progress(
                episode=episode,
                timesteps=self.n_calls,
                speed=0,  # Will be calculated if needed
                env_info={
                    'balance': self.initial_balance + episode_reward,
                    'position_size': 0.15,  # Fixed position size
                    'weekly_trades': trade_stats['total_trades'],
                    'metrics': {
                        'net_profit': episode_reward,
                        'win_rate': trade_stats['win_rate'],
                        'avg_profit_factor': trade_stats['profit_factor']
                    }
                }
            )
            
            # Save model if it's the best so far
            if self.last_episode_reward is not None and self.last_episode_reward > self.best_reward:
                self.best_reward = self.last_episode_reward
                self.save_model(self.model, episode, self.last_episode_reward, is_best=True)
            
            # Save regular checkpoint every 1000 episodes
            if self.last_episode_reward is not None and episode > 0 and episode % 1000 == 0:
                self.save_model(self.model, episode, self.last_episode_reward)
            
            # Log to wandb
            wandb.log({
                'episode': episode,
                'reward': episode_reward,
                'avg_reward': avg_reward,
                'avg_length': avg_length,
                'win_rate': trade_stats['win_rate'],
                'profit_factor': trade_stats['profit_factor'],
                'total_trades': trade_stats['total_trades']
            })

        return True
    
    def _calculate_monthly_performance(self):
        """Calculate monthly performance metrics"""
        if not self.trade_history:
            return
        
        # Group trades by month
        trades_df = pd.DataFrame(self.trade_history)
        trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
        trades_df['month'] = trades_df['timestamp'].dt.to_period('M')
        
        # Calculate monthly statistics
        monthly_stats = trades_df.groupby('month').agg({
            'reward': ['sum', 'mean', 'count'],
            'position_size': 'mean'
        }).reset_index()
        
        # Update monthly performance using iterrows and column names
        self.monthly_performance = {}
        for _, row in monthly_stats.iterrows():
            month = row['month']
            self.monthly_performance[str(month)] = {
                'total_profit': row[('reward', 'sum')],
                'avg_profit': row[('reward', 'mean')],
                'total_trades': row[('reward', 'count')],
                'avg_position_size': row[('position_size', 'mean')]
            }

class LearningRateMonitor(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.learning_rates = []
        self.episodes = []
        
    def _on_step(self):
        # Calculate progress
        progress = self.n_calls / MAX_TIMESTEPS
        
        # Get current learning rate
        current_lr = learning_rate_schedule(progress)
        
        # Store learning rate
        self.learning_rates.append(current_lr)
        self.episodes.append(self.n_calls // EPISODE_LENGTH)
        
        # Log to wandb
        wandb.log({
            'learning_rate': current_lr,
            'episode': self.episodes[-1]
        })
        
        return True

def split_train_val_test(df, val_fraction=0.1, test_fraction=0.1):
    """Split data into train/val/test sets preserving time order."""
    if not 0.0 < val_fraction < 1.0 or not 0.0 < test_fraction < 1.0:
        raise ValueError("val_fraction and test_fraction must be between 0 and 1.")
    if val_fraction + test_fraction >= 1.0:
        raise ValueError("val_fraction + test_fraction must be less than 1.")
    train_end = int(len(df) * (1 - val_fraction - test_fraction))
    val_end = int(len(df) * (1 - test_fraction))
    if train_end <= 0 or val_end <= train_end or val_end >= len(df):
        raise ValueError("Not enough data to create a train/val/test split.")
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    return train_df, val_df, test_df

def _calculate_max_drawdown(equity_curve):
    """Calculate max drawdown as a percentage from an equity curve."""
    if not equity_curve:
        return 0.0
    peak = equity_curve[0]
    max_dd = 0.0
    for value in equity_curve:
        if value > peak:
            peak = value
        if peak > 0:
            drawdown = (peak - value) / peak
            if drawdown > max_dd:
                max_dd = drawdown
    return max_dd * 100.0

def evaluate_model(model, eval_df, tag="eval"):
    """Evaluate model on out-of-sample data and return metrics."""
    if eval_df is None or eval_df.empty:
        logger.warning("Evaluation skipped: eval_df is empty.")
        return {}

    env = TradingEnvironment(eval_df.copy())
    obs, _ = env.reset()
    data_len = len(env.data['close'])

    total_reward = 0.0
    episode_reward = 0.0
    episode_rewards = []
    equity_curve = [env.balance]
    episode_count = 0
    steps = 0

    while env.current_step + 1 < data_len:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward
        episode_reward += reward
        steps += 1
        equity_curve.append(env.balance)

        if done:
            episode_count += 1
            episode_rewards.append(episode_reward)
            episode_reward = 0.0

    if episode_reward != 0.0:
        episode_rewards.append(episode_reward)

    trade_history = env.trade_history
    total_trades = len(trade_history)
    profits = [t['reward'] for t in trade_history if t.get('reward', 0) > 0]
    losses = [abs(t['reward']) for t in trade_history if t.get('reward', 0) < 0]
    gross_profit = sum(profits)
    gross_loss = sum(losses)
    net_profit = gross_profit - gross_loss
    win_rate = (len(profits) / total_trades) if total_trades > 0 else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (float('inf') if gross_profit > 0 else 0.0)
    avg_trade_pnl = (net_profit / total_trades) if total_trades > 0 else 0.0
    return_pct = ((equity_curve[-1] - equity_curve[0]) / equity_curve[0] * 100.0) if equity_curve else 0.0
    max_drawdown = _calculate_max_drawdown(equity_curve)

    metrics = {
        "steps": steps,
        "episodes": episode_count,
        "total_reward": total_reward,
        "avg_episode_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
        "total_trades": total_trades,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "net_profit": net_profit,
        "avg_trade_pnl": avg_trade_pnl,
        "return_pct": return_pct,
        "max_drawdown_pct": max_drawdown,
    }

    logger.info("\n=== Out-of-Sample Evaluation ===")
    logger.info(f"Steps: {metrics['steps']}")
    logger.info(f"Episodes: {metrics['episodes']}")
    logger.info(f"Total Reward: {metrics['total_reward']:.4f}")
    logger.info(f"Avg Episode Reward: {metrics['avg_episode_reward']:.4f}")
    logger.info(f"Total Trades: {metrics['total_trades']}")
    logger.info(f"Win Rate: {metrics['win_rate']*100:.2f}%")
    logger.info(f"Profit Factor: {metrics['profit_factor']:.2f}")
    logger.info(f"Net Profit: {metrics['net_profit']:.2f}")
    logger.info(f"Avg Trade PnL: {metrics['avg_trade_pnl']:.2f}")
    logger.info(f"Return %: {metrics['return_pct']:.2f}%")
    logger.info(f"Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
    logger.info("===============================\n")

    try:
        wandb.log({f"{tag}/{k}": v for k, v in metrics.items()})
    except Exception as e:
        logger.warning(f"wandb.log failed during evaluation: {e}")

    return metrics

def _build_model(env, hyperparams=None, verbose_override=None):
    hyperparams = hyperparams or {}
    params = PPO_PARAMS.copy()
    # Update PPO params that are part of PPO_PARAMS
    for key in ['n_epochs', 'gamma', 'gae_lambda', 'clip_range', 'ent_coef', 'vf_coef', 'max_grad_norm', 'use_sde', 'sde_sample_freq', 'target_kl']:
        if key in hyperparams:
            params[key] = hyperparams[key]

    n_steps = hyperparams.get('n_steps', N_STEPS)
    batch_size = hyperparams.get('batch_size', BATCH_SIZE)
    learning_rate = hyperparams.get('learning_rate', learning_rate_schedule(0))
    verbose = PPO_VERBOSE if verbose_override is None else verbose_override

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        device=DEVICE,
        **params,
        policy_kwargs=dict(
            features_extractor_class=CustomFeaturesExtractor,
            features_extractor_kwargs=dict(features_dim=64)
        ),
        verbose=verbose
    )
    return model

def _hyperparams_valid(hyperparams):
    n_steps = hyperparams.get('n_steps', N_STEPS)
    batch_size = hyperparams.get('batch_size', BATCH_SIZE)
    if batch_size > n_steps:
        return False
    return True

def _sample_hyperparam_trials(grid_params, max_trials, rng):
    keys = ['learning_rate', 'n_steps', 'batch_size', 'n_epochs', 'gamma', 'gae_lambda', 'clip_range', 'ent_coef', 'vf_coef']
    trials = []
    seen = set()
    attempts = 0
    while len(trials) < max_trials and attempts < max_trials * 10:
        attempts += 1
        trial = {}
        for key in keys:
            if key in grid_params:
                trial[key] = rng.choice(grid_params[key])
        if not _hyperparams_valid(trial):
            continue
        signature = tuple(sorted(trial.items()))
        if signature in seen:
            continue
        seen.add(signature)
        trials.append(trial)
    return trials

def _selection_score(metrics):
    if not metrics:
        return float('-inf')
    score = metrics.get('return_pct', 0.0) - metrics.get('max_drawdown_pct', 0.0)
    profit_factor = metrics.get('profit_factor', 0.0)
    if profit_factor <= 1.0:
        score -= 50.0
    return score

def _train_single_model(train_df, seed, hyperparams, timesteps, callbacks=None, verbose_override=None):
    set_random_seed(seed)
    env = DummyVecEnv([make_env(0, train_df, seed=seed, reward_config=REWARD_SHAPING)])
    model = _build_model(env, hyperparams=hyperparams, verbose_override=verbose_override)
    model.learn(total_timesteps=timesteps, callback=callbacks or [])
    return model, env

def run_model_selection(train_df, val_df):
    rng = random.Random(42)
    trials = _sample_hyperparam_trials(GRID_SEARCH_PARAMS, MODEL_SELECTION_TRIALS, rng)
    if not trials:
        logger.warning("No hyperparameter trials generated. Falling back to default params.")
        return {}

    best_score = float('-inf')
    best_params = None

    for trial_idx, hyperparams in enumerate(trials, start=1):
        logger.info(f"Starting trial {trial_idx}/{len(trials)} with params: {hyperparams}")
        for seed in MODEL_SELECTION_SEEDS:
            model, env = _train_single_model(
                train_df=train_df,
                seed=seed,
                hyperparams=hyperparams,
                timesteps=MODEL_SELECTION_TIMESTEPS,
                callbacks=[],
                verbose_override=0
            )
            metrics = evaluate_model(model, val_df, tag=f"val_trial{trial_idx}_seed{seed}")
            score = _selection_score(metrics)
            logger.info(f"Trial {trial_idx} seed {seed} score: {score:.4f}")
            if score > best_score:
                best_score = score
                best_params = hyperparams
            env.close()
            del model

    logger.info(f"Best hyperparameters from selection: {best_params}")
    return best_params or {}

def train():
    """Main training function"""
    # Initialize wandb
    wandb.init(project="forex-trading-bot", config={
        'algorithm': 'PPO',
        'batch_size': BATCH_SIZE,
        'n_steps': N_STEPS,
        'learning_rate': learning_rate_schedule(0),
        'max_episodes': MAX_EPISODES,
        'max_timesteps': MAX_TIMESTEPS,
        'min_episodes': MIN_EPISODES,
        'target_weekly_profit': TARGET_WEEKLY_PROFIT,
        'max_trades_per_week': MAX_TRADES_PER_WEEK
    })
    
    # Create debug logger
    debug_logger = DebugLogger()

    # Load historical data before environment creation
    fetcher = DataFetcher()
    df = fetcher.fetch_historical_data(
        symbol=SYMBOL,
        timeframe=TIMEFRAME
    )
    if df is None or df.empty:
        raise ValueError("No historical data loaded for training!")

    # Split into train/val/test sets (time-ordered)
    train_df, val_df, test_df = split_train_val_test(df, val_fraction=VAL_FRACTION, test_fraction=TEST_FRACTION)

    # Run hyperparameter selection (optional)
    selected_params = {}
    if MODEL_SELECTION_ENABLED:
        selected_params = run_model_selection(train_df, val_df)

    # Create vectorized environment
    train_full_df = pd.concat([train_df, val_df]).copy()
    env = DummyVecEnv([make_env(i, train_full_df, reward_config=REWARD_SHAPING) for i in range(1)])

    # Create model
    model = _build_model(env, hyperparams=selected_params)

    # Create callbacks
    progress_callback = ProgressCallback(debug_logger)
    lr_monitor = LearningRateMonitor()
    
    # Train model
    model.learn(
            total_timesteps=MAX_TIMESTEPS,
        callback=[progress_callback, lr_monitor]
        )

    # Evaluate on out-of-sample data (test set)
    evaluate_model(model, test_df, tag="test_final")

    # Train additional seeds for robustness and select best on test set
    best_score = float('-inf')
    best_model_path = Path("models") / "best_model.zip"
    best_metadata_path = Path("models") / "best_model_metadata.json"
    for seed in FINAL_TRAIN_SEEDS:
        candidate_model, candidate_env = _train_single_model(
            train_df=train_full_df,
            seed=seed,
            hyperparams=selected_params,
            timesteps=MAX_TIMESTEPS,
            callbacks=[progress_callback, lr_monitor]
        )
        metrics = evaluate_model(candidate_model, test_df, tag=f"test_seed{seed}")
        score = _selection_score(metrics)
        if score > best_score:
            best_score = score
            candidate_model.save(str(best_model_path))
            with open(best_metadata_path, 'w') as f:
                json.dump({
                    "seed": seed,
                    "score": score,
                    "metrics": metrics,
                    "timestamp": datetime.now().isoformat()
                }, f, indent=4)
        candidate_env.close()
        del candidate_model

        # Save final model
    model.save("models/final_model")
    
    # Close wandb
    wandb.finish()

if __name__ == "__main__":
            train()
    
