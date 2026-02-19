from dotenv import load_dotenv
import os


load_dotenv()

"""
Configuration module for the Forex Trading Bot
- Contains shared constants and configuration values
- Each parameter is commented to explain its purpose and effect
"""

import os
from pathlib import Path
from datetime import datetime, timedelta
from dukascopy_python.instruments import (
    INSTRUMENT_FX_MAJORS_EUR_USD,
    INSTRUMENT_FX_MAJORS_GBP_USD,
    INSTRUMENT_FX_MAJORS_USD_JPY,
    INSTRUMENT_FX_MAJORS_AUD_USD,
    INSTRUMENT_FX_MAJORS_USD_CHF,
    INSTRUMENT_FX_MAJORS_USD_CAD,
    INSTRUMENT_FX_MAJORS_NZD_USD
)
from dukascopy_python import (
    INTERVAL_TICK,
    INTERVAL_HOUR_1,
    INTERVAL_HOUR_4,
    OFFER_SIDE_BID,
)

# === 1. Data Download/Source (Dukascopy) ===
SYMBOL_MAP = {
    INSTRUMENT_FX_MAJORS_EUR_USD,  # EUR/USD symbol for Dukascopy
    INSTRUMENT_FX_MAJORS_GBP_USD,  # GBP/USD symbol for Dukascopy
    INSTRUMENT_FX_MAJORS_USD_JPY,  # USD/JPY symbol for Dukascopy
    INSTRUMENT_FX_MAJORS_AUD_USD,  # AUD/USD symbol for Dukascopy
    INSTRUMENT_FX_MAJORS_USD_CHF,  # USD/CHF symbol for Dukascopy
    INSTRUMENT_FX_MAJORS_USD_CAD,  # USD/CAD symbol for Dukascopy
    INSTRUMENT_FX_MAJORS_NZD_USD   # NZD/USD symbol for Dukascopy
}
TIMEFRAME_MAP = {
    INTERVAL_TICK,     # Tick data interval
    INTERVAL_HOUR_1,  # 1-hour interval
    INTERVAL_HOUR_4,  # 4-hour interval
}
YEARS = 10  # Number of years of historical data to download

# === 2. Directory Paths ===
from pathlib import Path

# === 2. Directory Paths ===
BASE_DIR = Path(__file__).resolve().parent.parent  # Project root directory
DATA_DIRECTORY = BASE_DIR / "data"
MODELS_DIRECTORY = BASE_DIR / "models"
REPORTS_DIRECTORY = BASE_DIR / "reports"
LOGS_DIRECTORY = BASE_DIR / "logs"
LOG_FILE_PATH = LOGS_DIRECTORY / "swing_trading.log"

# Create directories if they don't exist
for directory in [DATA_DIRECTORY, MODELS_DIRECTORY, REPORTS_DIRECTORY, LOGS_DIRECTORY]:
    directory.mkdir(parents=True, exist_ok=True)

    

# === Profile Settings ===
TRAINING_PROFILE = os.getenv("FX_PROFILE", "default").strip().lower()
if TRAINING_PROFILE not in {"default", "gpu", "gpu_quality"}:
    TRAINING_PROFILE = "default"

# === Device/GPU Settings ===
DEVICE_TYPE = os.getenv("FX_DEVICE_TYPE", "cpu")  # Device hint for training/inference
OPENCL_COMPILER_OUTPUT = '0'  # OpenCL compiler output setting
NUM_ENVS = int(os.getenv("FX_NUM_ENVS", "1"))  # Vectorized environment count
VEC_ENV_TYPE = os.getenv("FX_VEC_ENV", "dummy").strip().lower()  # dummy|subproc
FEATURES_DIM = int(os.getenv("FX_FEATURES_DIM", "64"))  # Feature extractor output dim
FEATURE_EXTRACTOR_HIDDEN_DIM = int(os.getenv("FX_FEATURE_HIDDEN_DIM", "64"))  # Feature extractor hidden width
POLICY_NET_ARCH = [64, 64]  # Policy/value MLP sizes


# === 3. Trading Parameters ===
SYMBOL = INSTRUMENT_FX_MAJORS_EUR_USD  # Default trading symbol
TIMEFRAME = INTERVAL_HOUR_1  # Default trading timeframe
INITIAL_BALANCE = 10000  # Starting account balance in USD
EPISODE_LENGTH = 90  # Number of days per training episode
SPREAD = 0.0002  # Spread in price (2 pips)
COMMISSION = 0.0001  # Commission per trade (0.01%)
SLIPPAGE = 0.00005  # Slippage in price (0.5 pips)

# === 4. Position Sizing ===
MIN_POSITION_SIZE = float(os.getenv("FX_MIN_POSITION_SIZE", "0.05"))  # Minimum position size in lots
MAX_POSITION_SIZE = float(os.getenv("FX_MAX_POSITION_SIZE", "0.20"))  # Maximum position size in lots
POSITION_SIZE_INCREMENT = 0.05  # Increment for position sizing

# === 5. Risk Management ===
MAX_TRADES_PER_WEEK = 6  # Max trades allowed per week
MAX_DAILY_TRADES = 10  # Max trades allowed per day
MAX_WEEKLY_TRADES = 50  # Max trades allowed per week (hard cap)
PROFIT_TARGET = 0.0020  # Profit target per trade (20 pips on EURUSD)
STOP_LOSS = 0.0015  # Stop loss per trade (15 pips on EURUSD)
ACTION_OPEN_REWARD = float(os.getenv("FX_ACTION_OPEN_REWARD", "0.01"))  # Reward for opening a trade
ACTION_NO_TRADE_PENALTY = float(os.getenv("FX_ACTION_NO_TRADE_PENALTY", "-0.1"))  # Penalty when no trade is taken
ACTION_FORCE_CLOSE_DAYS = int(os.getenv("FX_ACTION_FORCE_CLOSE_DAYS", "5"))  # Auto-close open position after N days
ACTION_HOLD_FLAT_PENALTY = float(os.getenv("FX_ACTION_HOLD_FLAT_PENALTY", "0.0"))  # Penalty when agent holds while flat
ACTION_HOLD_POSITION_PENALTY = float(os.getenv("FX_ACTION_HOLD_POSITION_PENALTY", "0.0"))  # Penalty when holding an open position
ACTION_IDLE_THRESHOLD_STEPS = int(os.getenv("FX_ACTION_IDLE_THRESHOLD_STEPS", "0"))  # Steps before extra idle penalties kick in
ACTION_IDLE_STEP_PENALTY = float(os.getenv("FX_ACTION_IDLE_STEP_PENALTY", "0.0"))  # Extra penalty per step after idle threshold
ACTION_FORCE_TRADE_ENABLED = os.getenv("FX_ACTION_FORCE_TRADE_ENABLED", "0").strip().lower() in {"1", "true", "yes", "on"}  # Force trade when idle too long
ACTION_FORCE_TRADE_IDLE_STEPS = int(os.getenv("FX_ACTION_FORCE_TRADE_IDLE_STEPS", "0"))  # Idle steps before forced trade
ACTION_MIN_TRADES_WINDOW_STEPS = int(os.getenv("FX_ACTION_MIN_TRADES_WINDOW_STEPS", "0"))  # Rolling window for minimum trade constraint
ACTION_MIN_TRADES_PER_WINDOW = int(os.getenv("FX_ACTION_MIN_TRADES_PER_WINDOW", "0"))  # Minimum trades required within rolling window
ACTION_MIN_TRADE_DEFICIT_PENALTY = float(os.getenv("FX_ACTION_MIN_TRADE_DEFICIT_PENALTY", "0.0"))  # Penalty per missing trade in rolling window
USE_CUSTOM_REWARD_WRAPPER = os.getenv("FX_USE_CUSTOM_REWARD_WRAPPER", "0").strip().lower() in {"1", "true", "yes", "on"}  # Enable extra custom reward wrapper during training
TRAIN_RANDOM_START = os.getenv("FX_TRAIN_RANDOM_START", "1").strip().lower() in {"1", "true", "yes", "on"}  # Start each training episode from a random index
POSITION_BALANCE_SCALING = os.getenv("FX_POSITION_BALANCE_SCALING", "1").strip().lower() in {"1", "true", "yes", "on"}  # Scale lot size down when equity drops
POSITION_BALANCE_FLOOR = float(os.getenv("FX_POSITION_BALANCE_FLOOR", "0.35"))  # Lower bound for lot-size scaling factor
EVAL_ACTION_SHAPING = os.getenv("FX_EVAL_ACTION_SHAPING", "0").strip().lower() in {"1", "true", "yes", "on"}  # Allow train-time shaping logic in eval mode
REGIME_FILTER_ENABLED = os.getenv("FX_REGIME_FILTER_ENABLED", "1").strip().lower() in {"1", "true", "yes", "on"}  # Enable regime-based entry filtering
REGIME_BLOCK_LOW_EXPECTANCY = os.getenv("FX_REGIME_BLOCK_LOW_EXPECTANCY", "1").strip().lower() in {"1", "true", "yes", "on"}  # Block low-expectancy entries
REGIME_BLOCK_IN_EVAL = os.getenv("FX_REGIME_BLOCK_IN_EVAL", "1").strip().lower() in {"1", "true", "yes", "on"}  # Enforce regime gates during eval/live
REGIME_FORCE_EXIT_ON_REVERSAL = os.getenv("FX_REGIME_FORCE_EXIT_ON_REVERSAL", "1").strip().lower() in {"1", "true", "yes", "on"}  # Exit when regime strongly reverses
REGIME_TREND_MIN_STRENGTH = float(os.getenv("FX_REGIME_TREND_MIN_STRENGTH", "0.00012"))  # Minimum directional trend strength for entries
REGIME_TREND_REVERSAL_MULT = float(os.getenv("FX_REGIME_TREND_REVERSAL_MULT", "1.5"))  # Reversal multiple for forced exits
REGIME_VOL_MIN_ATR_NORM = float(os.getenv("FX_REGIME_VOL_MIN_ATR_NORM", "0.00035"))  # Minimum acceptable normalized ATR
REGIME_VOL_MAX_ATR_NORM = float(os.getenv("FX_REGIME_VOL_MAX_ATR_NORM", "0.00190"))  # Maximum acceptable normalized ATR
ENTRY_EXPECTANCY_THRESHOLD = float(os.getenv("FX_ENTRY_EXPECTANCY_THRESHOLD", "0.00005"))  # Minimum expectancy score for entry quality
ENTRY_EXPECTANCY_REWARD_WEIGHT = float(os.getenv("FX_ENTRY_EXPECTANCY_REWARD_WEIGHT", "0.60"))  # Reward scale for high-expectancy entries
LOW_EXPECTANCY_ENTRY_PENALTY = float(os.getenv("FX_LOW_EXPECTANCY_ENTRY_PENALTY", "-0.002"))  # Penalty for attempting low-expectancy entries
LOW_EXPECTANCY_CLOSE_PENALTY_SCALE = float(os.getenv("FX_LOW_EXPECTANCY_CLOSE_PENALTY_SCALE", "0.80"))  # Extra penalty on close for low-expectancy entries

# === 6. Training/Backtesting Parameters ===
MAX_EPISODES = 2200000  # Maximum number of training episodes
MAX_TIMESTEPS = 1000000  # Maximum number of training timesteps (standard real training)
MIN_EPISODES = 1000  # Minimum number of training episodes
TARGET_WEEKLY_PROFIT = 1000.0  # Target profit per week in USD
BATCH_SIZE = int(os.getenv("FX_BATCH_SIZE", "32"))  # Training batch size
N_STEPS = int(os.getenv("FX_N_STEPS", "512"))  # Number of steps per PPO update
QUICK_TRAIN_TIMESTEPS = int(os.getenv("FX_QUICK_TIMESTEPS", "30000"))  # Timesteps for quick profile checks
GRADIENT_ACCUMULATION_STEPS = 4  # Gradient accumulation steps
RESUME_FROM_BEST = False  # Train fresh for current action space compatibility
RESUME_FROM_PATH = "models/best_model.zip"  # Path for auto-resume checkpoint
LOCK_FILE_ENABLED = True  # Prevent duplicate concurrent training runs
LOCK_FILE_PATH = BASE_DIR / "logs" / "train.lock"  # Training lock file path

# GPU profile overrides (tuned for higher CUDA utilization)
if TRAINING_PROFILE == "gpu":
    NUM_ENVS = int(os.getenv("FX_NUM_ENVS", "8"))
    VEC_ENV_TYPE = os.getenv("FX_VEC_ENV", "subproc").strip().lower()
    N_STEPS = int(os.getenv("FX_N_STEPS", "2048"))
    BATCH_SIZE = int(os.getenv("FX_BATCH_SIZE", "1024"))
    FEATURES_DIM = int(os.getenv("FX_FEATURES_DIM", "128"))
    FEATURE_EXTRACTOR_HIDDEN_DIM = int(os.getenv("FX_FEATURE_HIDDEN_DIM", "128"))
    POLICY_NET_ARCH = [256, 256]
elif TRAINING_PROFILE == "gpu_quality":
    NUM_ENVS = int(os.getenv("FX_NUM_ENVS", "10"))
    VEC_ENV_TYPE = os.getenv("FX_VEC_ENV", "subproc").strip().lower()
    N_STEPS = int(os.getenv("FX_N_STEPS", "4096"))
    BATCH_SIZE = int(os.getenv("FX_BATCH_SIZE", "2048"))
    FEATURES_DIM = int(os.getenv("FX_FEATURES_DIM", "192"))
    FEATURE_EXTRACTOR_HIDDEN_DIM = int(os.getenv("FX_FEATURE_HIDDEN_DIM", "192"))
    POLICY_NET_ARCH = [256, 256]
    MAX_TIMESTEPS = int(os.getenv("FX_MAX_TIMESTEPS", "1500000"))
    QUICK_TRAIN_TIMESTEPS = int(os.getenv("FX_QUICK_TIMESTEPS", "120000"))
    MAX_TRADES_PER_WEEK = int(os.getenv("FX_MAX_TRADES_PER_WEEK", "4"))
    ACTION_OPEN_REWARD = float(os.getenv("FX_ACTION_OPEN_REWARD", "0.005"))
    ACTION_NO_TRADE_PENALTY = float(os.getenv("FX_ACTION_NO_TRADE_PENALTY", "-0.005"))
    ACTION_FORCE_CLOSE_DAYS = int(os.getenv("FX_ACTION_FORCE_CLOSE_DAYS", "4"))
    ACTION_HOLD_FLAT_PENALTY = float(os.getenv("FX_ACTION_HOLD_FLAT_PENALTY", "-0.002"))
    ACTION_HOLD_POSITION_PENALTY = float(os.getenv("FX_ACTION_HOLD_POSITION_PENALTY", "-0.0005"))
    ACTION_IDLE_THRESHOLD_STEPS = int(os.getenv("FX_ACTION_IDLE_THRESHOLD_STEPS", "48"))
    ACTION_IDLE_STEP_PENALTY = float(os.getenv("FX_ACTION_IDLE_STEP_PENALTY", "-0.0005"))
    ACTION_FORCE_TRADE_ENABLED = os.getenv("FX_ACTION_FORCE_TRADE_ENABLED", "0").strip().lower() in {"1", "true", "yes", "on"}
    ACTION_FORCE_TRADE_IDLE_STEPS = int(os.getenv("FX_ACTION_FORCE_TRADE_IDLE_STEPS", "120"))
    ACTION_MIN_TRADES_WINDOW_STEPS = int(os.getenv("FX_ACTION_MIN_TRADES_WINDOW_STEPS", "168"))
    ACTION_MIN_TRADES_PER_WINDOW = int(os.getenv("FX_ACTION_MIN_TRADES_PER_WINDOW", "1"))
    ACTION_MIN_TRADE_DEFICIT_PENALTY = float(os.getenv("FX_ACTION_MIN_TRADE_DEFICIT_PENALTY", "-0.002"))
    MIN_POSITION_SIZE = float(os.getenv("FX_MIN_POSITION_SIZE", "0.05"))
    MAX_POSITION_SIZE = float(os.getenv("FX_MAX_POSITION_SIZE", "0.18"))
    POSITION_BALANCE_FLOOR = float(os.getenv("FX_POSITION_BALANCE_FLOOR", "0.30"))
    REGIME_FILTER_ENABLED = os.getenv("FX_REGIME_FILTER_ENABLED", "1").strip().lower() in {"1", "true", "yes", "on"}
    REGIME_BLOCK_LOW_EXPECTANCY = os.getenv("FX_REGIME_BLOCK_LOW_EXPECTANCY", "1").strip().lower() in {"1", "true", "yes", "on"}
    REGIME_BLOCK_IN_EVAL = os.getenv("FX_REGIME_BLOCK_IN_EVAL", "1").strip().lower() in {"1", "true", "yes", "on"}
    REGIME_FORCE_EXIT_ON_REVERSAL = os.getenv("FX_REGIME_FORCE_EXIT_ON_REVERSAL", "1").strip().lower() in {"1", "true", "yes", "on"}
    REGIME_TREND_MIN_STRENGTH = float(os.getenv("FX_REGIME_TREND_MIN_STRENGTH", "0.00012"))
    REGIME_TREND_REVERSAL_MULT = float(os.getenv("FX_REGIME_TREND_REVERSAL_MULT", "1.5"))
    REGIME_VOL_MIN_ATR_NORM = float(os.getenv("FX_REGIME_VOL_MIN_ATR_NORM", "0.00035"))
    REGIME_VOL_MAX_ATR_NORM = float(os.getenv("FX_REGIME_VOL_MAX_ATR_NORM", "0.00190"))
    ENTRY_EXPECTANCY_THRESHOLD = float(os.getenv("FX_ENTRY_EXPECTANCY_THRESHOLD", "0.00005"))
    ENTRY_EXPECTANCY_REWARD_WEIGHT = float(os.getenv("FX_ENTRY_EXPECTANCY_REWARD_WEIGHT", "0.60"))
    LOW_EXPECTANCY_ENTRY_PENALTY = float(os.getenv("FX_LOW_EXPECTANCY_ENTRY_PENALTY", "-0.002"))
    LOW_EXPECTANCY_CLOSE_PENALTY_SCALE = float(os.getenv("FX_LOW_EXPECTANCY_CLOSE_PENALTY_SCALE", "0.80"))

NUM_ENVS = max(NUM_ENVS, 1)
if VEC_ENV_TYPE not in {"dummy", "subproc"}:
    VEC_ENV_TYPE = "dummy"
ACTION_FORCE_CLOSE_DAYS = max(ACTION_FORCE_CLOSE_DAYS, 1)
if ACTION_OPEN_REWARD < 0:
    ACTION_OPEN_REWARD = 0.0
if ACTION_NO_TRADE_PENALTY > 0:
    ACTION_NO_TRADE_PENALTY = -ACTION_NO_TRADE_PENALTY
if ACTION_HOLD_FLAT_PENALTY > 0:
    ACTION_HOLD_FLAT_PENALTY = -ACTION_HOLD_FLAT_PENALTY
if ACTION_HOLD_POSITION_PENALTY > 0:
    ACTION_HOLD_POSITION_PENALTY = -ACTION_HOLD_POSITION_PENALTY
ACTION_IDLE_THRESHOLD_STEPS = max(ACTION_IDLE_THRESHOLD_STEPS, 0)
if ACTION_IDLE_STEP_PENALTY > 0:
    ACTION_IDLE_STEP_PENALTY = -ACTION_IDLE_STEP_PENALTY
ACTION_FORCE_TRADE_IDLE_STEPS = max(ACTION_FORCE_TRADE_IDLE_STEPS, 0)
ACTION_MIN_TRADES_WINDOW_STEPS = max(ACTION_MIN_TRADES_WINDOW_STEPS, 0)
ACTION_MIN_TRADES_PER_WINDOW = max(ACTION_MIN_TRADES_PER_WINDOW, 0)
if ACTION_MIN_TRADE_DEFICIT_PENALTY > 0:
    ACTION_MIN_TRADE_DEFICIT_PENALTY = -ACTION_MIN_TRADE_DEFICIT_PENALTY
QUICK_TRAIN_TIMESTEPS = max(QUICK_TRAIN_TIMESTEPS, 1000)
MIN_POSITION_SIZE = max(MIN_POSITION_SIZE, 0.01)
MAX_POSITION_SIZE = max(MAX_POSITION_SIZE, MIN_POSITION_SIZE)
POSITION_BALANCE_FLOOR = min(max(POSITION_BALANCE_FLOOR, 0.10), 1.0)
REGIME_TREND_MIN_STRENGTH = max(REGIME_TREND_MIN_STRENGTH, 0.0)
REGIME_TREND_REVERSAL_MULT = max(REGIME_TREND_REVERSAL_MULT, 1.0)
REGIME_VOL_MIN_ATR_NORM = max(REGIME_VOL_MIN_ATR_NORM, 0.0)
REGIME_VOL_MAX_ATR_NORM = max(REGIME_VOL_MAX_ATR_NORM, REGIME_VOL_MIN_ATR_NORM + 1e-9)
ENTRY_EXPECTANCY_THRESHOLD = max(ENTRY_EXPECTANCY_THRESHOLD, 0.0)
ENTRY_EXPECTANCY_REWARD_WEIGHT = max(ENTRY_EXPECTANCY_REWARD_WEIGHT, 0.0)
if LOW_EXPECTANCY_ENTRY_PENALTY > 0:
    LOW_EXPECTANCY_ENTRY_PENALTY = -LOW_EXPECTANCY_ENTRY_PENALTY
LOW_EXPECTANCY_CLOSE_PENALTY_SCALE = max(LOW_EXPECTANCY_CLOSE_PENALTY_SCALE, 0.0)

# Keep batch size within rollout capacity to avoid unstable updates
_rollout_capacity = max(N_STEPS * NUM_ENVS, 1)
if BATCH_SIZE > _rollout_capacity:
    BATCH_SIZE = _rollout_capacity

# === 7. Reward Shaping ===
REWARD_SHAPING = {
    'profit_multiplier': 1.0,  # Multiplier for profit rewards
    'loss_penalty': 1.0,  # Penalty for losses
    'win_streak_bonus': 0.2,  # Bonus for consecutive wins
    'pattern_recognition_bonus': 0.1,  # Bonus for recognizing patterns
    'position_size_bonus': 0.1,  # Bonus for optimal position size
    'trade_frequency_bonus': -0.2,  # Penalty for overtrading
    'weekly_profit_bonus': 0.5,  # Bonus for weekly profit
    'drawdown_penalty': 0.5,  # Penalty for drawdown
    'weekly_trade_bonus': -0.3,  # Penalty for not trading weekly
    'profit_target_bonus': 0.5  # Bonus for hitting profit target
}
if TRAINING_PROFILE == "gpu_quality":
    REWARD_SHAPING.update(
        {
            'loss_penalty': float(os.getenv("FX_REWARD_LOSS_PENALTY", "0.35")),
            'win_streak_bonus': float(os.getenv("FX_REWARD_WIN_STREAK_BONUS", "0.12")),
            'weekly_profit_bonus': float(os.getenv("FX_REWARD_WEEKLY_PROFIT_BONUS", "0.30")),
            'drawdown_penalty': float(os.getenv("FX_REWARD_DRAWDOWN_PENALTY", "0.75")),
            # Main wrapper subtracts this value after weekly cap is hit; keep this positive to enforce a penalty.
            'weekly_trade_bonus': float(os.getenv("FX_REWARD_WEEKLY_TRADE_PENALTY", "0.20")),
            'trade_frequency_bonus': float(os.getenv("FX_REWARD_TRADE_FREQUENCY_BONUS", "-0.05")),
        }
    )
REWARD_BALANCE_CHANGE_WEIGHT = 2.0  # Weight for balance change reward
REWARD_POSITION_HOLDING_PROFIT = 0.0005  # Reward for holding profitable position
REWARD_POSITION_HOLDING_LOSS = 0.001  # Penalty for holding losing position
REWARD_TRADE_PROFIT_WEIGHT = 0.3  # Weight for profitable trade reward
REWARD_TRADE_LOSS_WEIGHT = 0.1  # Weight for losing trade penalty
REWARD_TRADE_FREQUENCY = 0.005  # Small reward for executing trades
REWARD_MARK_TO_MARKET_WEIGHT = float(os.getenv("FX_REWARD_MARK_TO_MARKET_WEIGHT", "0.25"))  # Dense reward from unrealized PnL change
if TRAINING_PROFILE == "gpu_quality":
    REWARD_TRADE_PROFIT_WEIGHT = float(os.getenv("FX_REWARD_TRADE_PROFIT_WEIGHT", "0.35"))
    REWARD_TRADE_LOSS_WEIGHT = float(os.getenv("FX_REWARD_TRADE_LOSS_WEIGHT", "0.08"))
    REWARD_TRADE_FREQUENCY = float(os.getenv("FX_REWARD_TRADE_FREQUENCY", "0.008"))
    REWARD_MARK_TO_MARKET_WEIGHT = float(os.getenv("FX_REWARD_MARK_TO_MARKET_WEIGHT", "0.35"))
REWARD_MARK_TO_MARKET_WEIGHT = max(REWARD_MARK_TO_MARKET_WEIGHT, 0.0)

# === 8. RL/PPO/Model Hyperparameters ===
RL_TRAINING_STEPS = 1000000  # Total RL training steps
RL_WINDOW_SIZE = 20  # Window size for RL state
PPO_LEARNING_RATE = 0.0003  # PPO learning rate
PPO_N_STEPS = 2048  # PPO steps per update
PPO_BATCH_SIZE = 128  # PPO batch size
PPO_N_EPOCHS = 10  # PPO epochs per update
PPO_GAMMA = 0.99  # PPO discount factor
PPO_GAE_LAMBDA = 0.95  # PPO GAE lambda
PPO_CLIP_RANGE = 0.2  # PPO clip range
PPO_ENT_COEF = 0.01  # PPO entropy coefficient
PPO_VF_COEF = 0.5  # PPO value function coefficient
PPO_MAX_GRAD_NORM = 0.5  # PPO max gradient norm
PPO_TARGET_KL = 0.015  # PPO target KL divergence
PPO_USE_SDE = False  # PPO state-dependent exploration
PPO_SDE_SAMPLE_FREQ = -1  # PPO SDE sample frequency
PPO_VERBOSE = 1  # PPO verbosity level
PPO_PARAMS = {
    'n_epochs': 10,  # PPO epochs per update
    'gamma': 0.99,  # PPO discount factor
    'gae_lambda': 0.95,  # PPO GAE lambda
    'clip_range': 0.2,  # PPO clip range
    'ent_coef': 0.01,  # PPO entropy coefficient
    'vf_coef': 0.5,  # PPO value function coefficient
    'max_grad_norm': 0.5,  # PPO max gradient norm
    'use_sde': False,  # PPO state-dependent exploration
    'sde_sample_freq': -1,  # PPO SDE sample frequency
    'target_kl': None,  # PPO target KL divergence
}
if TRAINING_PROFILE == "gpu_quality":
    PPO_PARAMS.update(
        {
            'n_epochs': int(os.getenv("FX_PPO_N_EPOCHS", "12")),
            'gamma': float(os.getenv("FX_PPO_GAMMA", "0.995")),
            'gae_lambda': float(os.getenv("FX_PPO_GAE_LAMBDA", "0.97")),
            'clip_range': float(os.getenv("FX_PPO_CLIP_RANGE", "0.15")),
            'ent_coef': float(os.getenv("FX_PPO_ENT_COEF", "0.03")),
            'vf_coef': float(os.getenv("FX_PPO_VF_COEF", "0.6")),
            'max_grad_norm': float(os.getenv("FX_PPO_MAX_GRAD_NORM", "0.5")),
            'target_kl': float(os.getenv("FX_PPO_TARGET_KL", "0.03")),
        }
    )
ENTROPY_SCHEDULE_ENABLED = os.getenv("FX_ENTROPY_SCHEDULE_ENABLED", "0").strip().lower() in {"1", "true", "yes", "on"}
ENTROPY_START_COEF = float(os.getenv("FX_ENTROPY_START_COEF", str(PPO_PARAMS.get("ent_coef", 0.01))))
ENTROPY_END_COEF = float(os.getenv("FX_ENTROPY_END_COEF", str(PPO_PARAMS.get("ent_coef", 0.01))))
ENTROPY_WARMUP_FRACTION = float(os.getenv("FX_ENTROPY_WARMUP_FRACTION", "0.2"))
if TRAINING_PROFILE == "gpu_quality":
    ENTROPY_SCHEDULE_ENABLED = os.getenv("FX_ENTROPY_SCHEDULE_ENABLED", "1").strip().lower() in {"1", "true", "yes", "on"}
    ENTROPY_START_COEF = float(os.getenv("FX_ENTROPY_START_COEF", "0.06"))
    ENTROPY_END_COEF = float(os.getenv("FX_ENTROPY_END_COEF", str(PPO_PARAMS.get("ent_coef", 0.02))))
    ENTROPY_WARMUP_FRACTION = float(os.getenv("FX_ENTROPY_WARMUP_FRACTION", "0.2"))
ENTROPY_START_COEF = max(ENTROPY_START_COEF, 0.0)
ENTROPY_END_COEF = max(ENTROPY_END_COEF, 0.0)
ENTROPY_WARMUP_FRACTION = min(max(ENTROPY_WARMUP_FRACTION, 0.0), 0.95)

# === 9. MetaTrader 5/Live Trading Config ===
MT5_CONFIG = {
    'MT5LOGIN': os.getenv('MT5_LOGIN'),  # MT5 account number
    'MT5PASSWORD': os.getenv('MT5_PASSWORD'),  # MT5 password
    'MT5SERVER': os.getenv('MT5_SERVER'),  # MT5 broker server
    'MT5SYMBOL': "EURUSD",  # MT5 trading symbol
    'MT5TIMEFRAME': "M15",  # MT5 trading timeframe
    'MODEL_PATH': str(Path(__file__).parent.parent / "models" / "best_model.zip"),  # Path to trained model
    'BASE_DEVIATION': 20,  # Base deviation in points
    'MAX_DEVIATION': 50,  # Max allowed deviation in points
    'MIN_DEVIATION': 10,  # Min allowed deviation in points
    'MAGIC_BASE': 234000,  # Base magic number for orders
    'COMMENT': "python script"  # Order comment for MT5
}

# === 10. Optimization/Hyperparameter Search ===
MODEL_PARAMS = {
    'n_estimators': 100,  # Number of trees in ensemble models
    'learning_rate': 0.05,  # Learning rate for boosting
    'max_depth': 4,  # Maximum tree depth
    'min_samples_split': 20,  # Minimum samples to split a node
    'min_samples_leaf': 10,  # Minimum samples at a leaf node
    'subsample': 0.8,  # Fraction of samples for fitting each tree
    'random_state': 42  # Random seed for reproducibility
}
TRAINING_PARAMS = {
    'test_size': 0.2,  # Fraction of data for testing
    'random_state': 42,  # Random seed
    'cv_folds': 5  # Number of cross-validation folds
}
RL_PARAMS = {
    'learning_rate': 0.0003,  # PPO learning rate
    'n_steps': 2048,  # PPO steps per update
    'batch_size': 64,  # PPO batch size
    'n_epochs': 10,  # PPO epochs per update
    'gamma': 0.99,  # PPO discount factor
    'gae_lambda': 0.95,  # PPO GAE lambda
    'clip_range': 0.2,  # PPO clip range
    'ent_coef': 0.01,  # PPO entropy coefficient
    'vf_coef': 0.5,  # PPO value function coefficient
    'max_grad_norm': 0.5  # PPO max gradient norm
}
HP_LEARNING_RATE_MIN = 1e-5  # Min learning rate for search
HP_LEARNING_RATE_MAX = 1e-3  # Max learning rate for search
HP_N_STEPS_MIN = 512  # Min steps for search
HP_N_STEPS_MAX = 2048  # Max steps for search
HP_BATCH_SIZE_MIN = 32  # Min batch size for search
HP_BATCH_SIZE_MAX = 256  # Max batch size for search
HP_N_EPOCHS_MIN = 5  # Min epochs for search
HP_N_EPOCHS_MAX = 20  # Max epochs for search
HP_GAMMA_MIN = 0.9  # Min gamma for search
HP_GAMMA_MAX = 0.9999  # Max gamma for search
HP_ENT_COEF_MIN = 0.01  # Min entropy coef for search
HP_ENT_COEF_MAX = 0.1  # Max entropy coef for search
BAYESIAN_N_CALLS = 30  # Number of Bayesian optimization calls
GRID_SEARCH_PARAMS = {
    'learning_rate': [0.0001, 0.0003, 0.001],  # Learning rates for grid search
    'n_steps': [1024, 2048, 4096],  # Steps for grid search
    'batch_size': [32, 64, 128],  # Batch sizes for grid search
    'n_epochs': [5, 10, 20],  # Epochs for grid search
    'gamma': [0.95, 0.99, 0.995],  # Discount factors for grid search
    'gae_lambda': [0.9, 0.95, 0.98],  # GAE lambdas for grid search
    'clip_range': [0.1, 0.2, 0.3],  # Clip ranges for grid search
    'ent_coef': [0.005, 0.01, 0.02],  # Entropy coefs for grid search
    'vf_coef': [0.3, 0.5, 0.7]  # Value function coefs for grid search
}

# === 11. Analytics/Visualization/Portfolio ===
PORTFOLIO_EFFICIENT_FRONTIER_POINTS = 20  # Points for efficient frontier plot
PORTFOLIO_PLOT_HEIGHT = 8  # Height of portfolio plots
PORTFOLIO_PLOT_ASPECT = 1.5  # Aspect ratio of portfolio plots
PORTFOLIO_PLOT_STYLE = "whitegrid"  # Style for portfolio plots
PORTFOLIO_CORRELATION_CMAP = 'coolwarm'  # Colormap for correlation heatmap
PORTFOLIO_CORRELATION_CENTER = 0  # Center value for correlation heatmap
PORTFOLIO_CORRELATION_FMT = '.2f'  # Format for correlation values
PORTFOLIO_PLOT_TITLE_FONTSIZE = 16  # Font size for plot titles
PORTFOLIO_AXIS_FONTSIZE = 12  # Font size for axis labels


# === 12. Other/Advanced ===
LOG_FILE_PATH = LOGS_DIRECTORY / "swing_trading.log"  # Main log file path
DATA_CSV_PATH = "data/{symbol}_{start}_{end}.csv"  # Path template for historical data CSV
INITIAL_LR = float(os.getenv("FX_INITIAL_LR", "3e-4"))  # Initial learning rate for schedule
FINAL_LR = float(os.getenv("FX_FINAL_LR", "1e-4"))  # Final learning rate for schedule
MIN_LR = float(os.getenv("FX_MIN_LR", "5e-5"))  # Minimum learning rate for schedule
WARMUP_STEPS = float(os.getenv("FX_WARMUP_STEPS", "0.1"))  # Fraction of training for learning rate warmup
if TRAINING_PROFILE == "gpu_quality":
    INITIAL_LR = float(os.getenv("FX_INITIAL_LR", "2.5e-4"))
    FINAL_LR = float(os.getenv("FX_FINAL_LR", "6e-5"))
    MIN_LR = float(os.getenv("FX_MIN_LR", "2e-5"))
    WARMUP_STEPS = float(os.getenv("FX_WARMUP_STEPS", "0.15"))
GENETIC_POPULATION_SIZE = 100  # Population size for genetic algorithm
GENETIC_GENERATIONS = 50  # Number of generations for genetic algorithm

# --- PPO_PARAMS vs GRID_SEARCH_PARAMS ---
# PPO_PARAMS contains the actual hyperparameter values used for a single PPO training run.
# GRID_SEARCH_PARAMS contains lists/ranges of possible values for each hyperparameter, used for hyperparameter search (e.g., grid search, random search, Bayesian optimization).
#
# Best practice: Do NOT reference GRID_SEARCH_PARAMS directly in PPO_PARAMS.
# Instead, your search/experiment code should dynamically create or update PPO_PARAMS for each trial using values from GRID_SEARCH_PARAMS.
# Example:
#   for n_epochs in GRID_SEARCH_PARAMS['n_epochs']:
#       params = PPO_PARAMS.copy()
#       params['n_epochs'] = n_epochs
#       model = PPO(..., **params)
#
# This keeps your config clean, flexible, and easy to maintain.


