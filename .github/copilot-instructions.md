# Forex Trading Bot - AI Coding Agent Instructions

## Architecture Overview
This is a reinforcement learning-based Forex trading bot using Proximal Policy Optimization (PPO) with PyTorch. The system consists of:
- **Training Pipeline**: `modules/main.py` trains PPO models using historical data from Dukascopy
- **Live Trading**: `modules/live_trading.py` executes trades via MetaTrader 5 API
- **Data Layer**: `modules/data_fetcher.py` and `modules/dukascopy_downloader.py` handle forex data acquisition
- **RL Environment**: `modules/model.py` defines the `TradingEnvironment` gym environment
- **Configuration**: `modules/config.py` centralizes all parameters (trading, RL, MT5, etc.)

## Key Workflows
- **Setup**: Create venv, `pip install -r requirements.txt`, `wandb login`, create `.env` for MT5 credentials
- **Data Acquisition**: Run `python modules/dukascopy_downloader.py` or let `DataFetcher` auto-download to `data/` directory
- **Training**: `cd modules && python main.py` (uses CPU by default; DirectML GPU broken in `test_directml.py`)
- **Live Trading**: `cd modules && python live_trading.py` (requires trained model in `models/` and MT5 running)
- **Testing**: `python test_indicators.py` validates technical indicators; `python test_directml.py` checks AMD GPU

## Project Conventions
- **Configuration Centralization**: All parameters in `modules/config.py` - modify here for trading symbols, risk limits, RL hyperparameters
- **Environment Variables**: MT5 credentials via `.env` file (never commit); loaded via `python-dotenv`
- **Directory Structure**: `data/` for CSVs, `models/` for saved PPO models, `logs/` for application logs, `reports/` for analytics
- **Reward Calculation**: Implemented in `TradingEnvironment._execute_trade()` with PnL-based rewards normalized by initial balance; `REWARD_SHAPING` dict in config.py is defined but not currently used
- **Position Sizing**: Dynamic sizing between `MIN_POSITION_SIZE` (0.15 lots) and `MAX_POSITION_SIZE` (0.30 lots)
- **Risk Management**: Weekly trade limits (`MAX_TRADES_PER_WEEK=6`), profit targets (`PROFIT_TARGET=0.02`), stop losses (`STOP_LOSS=0.01`)
- **GPU Acceleration**: AMD-only via DirectML; falls back to CPU if issues (common due to torch-directml bugs)
- **Logging**: Custom logger in `modules/logger.py`; integrates with Weights & Biases for experiment tracking
- **Data Format**: OHLCV CSVs with datetime index; Dukascopy downloads in 3-year chunks to prevent timeouts
- **Missing Components**: `technical_indicators.py` is imported but the file does not exist; TechnicalIndicators class needs implementation

## Integration Points
- **MetaTrader 5**: Live trading requires MT5 desktop app running; uses `mt5` Python package for order execution
- **Weights & Biases**: Automatic experiment logging in training; view at wandb.ai
- **Dukascopy API**: Free historical data; no API key needed; downloads to `data/EUR_USD_20150622_20250619.csv` format
- **Stable Baselines3**: PPO implementation with custom feature extractor in `main.py`
- **Technical Indicators**: Intended to be computed in `DataFetcher.fetch_historical_data()` via `TechnicalIndicators` class (currently missing)

## Common Patterns
- **Environment Creation**: Use `TradingEnvironment` from `model.py` for backtesting; inherits from gymnasium.Env
- **Model Persistence**: Save/load PPO models with `model.save()` and `PPO.load()` to `models/` directory
- **Data Preprocessing**: Standardize features with `StandardScaler`; add indicators before training (when implemented)
- **Hyperparameter Tuning**: Use Optuna integration in `model.py` for automated PPO parameter search
- **Error Handling**: Check MT5 connection in live trading; validate data existence before training

## Critical Files
- `modules/config.py`: All configuration - modify trading params, RL settings, MT5 credentials here
- `modules/model.py`: RL environment definition, reward calculations, hyperparameter optimization
- `modules/main.py`: Training loop with PPO, custom feature extractor, wandb integration
- `modules/live_trading.py`: MT5 integration, real-time position management, order execution
- `modules/data_fetcher.py`: Data loading, indicator calculation (when implemented), Dukascopy integration</content>
<parameter name="filePath">c:\Users\sumes\OneDrive - UTHealth Houston\Desktop\FX\Forex-trading-bot\.github\copilot-instructions.md