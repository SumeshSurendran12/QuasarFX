# Forex Trading Bot with Reinforcement Learning

A sophisticated Forex trading bot that uses Proximal Policy Optimization (PPO) reinforcement learning to make trading decisions. The bot features dynamic position sizing, GPU acceleration via CUDA (for Nvidia GPUs like RTX 3070) or DirectML (for AMD GPUs), and live trading integration with MetaTrader 5.

## ⚡️ Data Provider: Dukascopy (NEW)

**This project now uses [Dukascopy](https://www.dukascopy.com/) as the sole source for historical forex data.**

- Data is automatically downloaded in multi-year intervals and concatenated to create a continuous dataset.
- Saved in the `data/` directory (e.g., `data/EUR_USD_20150622_20250619.csv`).
- No API key or `.env` setup required for data.

### How the Download Works

- Data is fetched in 3-year intervals (by default) to prevent timeouts and memory issues.
- Each chunk is a DataFrame with a datetime index and OHLCV columns.
- All chunks are concatenated and saved as a single CSV with columns: `time, open, high, low, close, volume`.
- The bot handles all data fetching and formatting automatically.

**Example CSV header:**
```csv
time,open,high,low,close,volume
2015-06-23 02:00:00+00:00,1.13214,1.13235,1.12549,1.12602,10090.10
...
```

## 🚀 Features

- **Reinforcement Learning**: PPO for autonomous trading decisions
- **Dynamic Position Sizing**: Adjusts positions based on market/risk
- **GPU Acceleration**: CUDA for Nvidia GPUs (e.g., RTX 3070) or DirectML for AMD GPUs
- **Live Trading**: MetaTrader 5 integration for real-time trading
- **Comprehensive Analytics**: Detailed performance tracking and visualization
- **Risk Management**: Stop-loss, take-profit, trade frequency limits
- **Hyperparameter Optimization**: Automated tuning with Optuna
- **Weights & Biases Integration**: Experiment tracking, logging, and visualization

## ⚠️ Note on GPU Acceleration

- **CUDA is enabled** for Nvidia GPUs like RTX 3070 (automatically detects CUDA if available).
- For AMD GPUs, use DirectML (install torch-directml instead).
- For updates, see [PyTorch CUDA docs](https://pytorch.org/get-started/locally/).

> **Note:**
> - Designed for Nvidia GPUs like RTX 3070 (CUDA) or AMD GPUs like RX 6700 (DirectML).
> - GPU acceleration tested with torch==2.5.1+cu121 (CUDA 12.1).
> - For AMD GPUs, modify requirements to use torch-directml.

## Setup

### Prerequisites

- Python 3.11
- Nvidia GPU with CUDA support (e.g., RTX 3070) or AMD GPU
- MetaTrader 5 account (for live trading)

### Weights & Biases (wandb) Setup

1. Sign up at [wandb.ai](https://wandb.ai/) (free).
2. Log in via terminal:
   ```bash
   wandb login
   ```
3. Paste your API key when prompted.

## Data Provider

- Historical data is automatically downloaded as a CSV in the `data/` directory.
- You can use your own data in the format:
  ```
  time,open,high,low,close,volume
  2024-01-01 00:00:00,1.1000,1.1010,1.0990,1.1005,1000
  ...
  ```

## Technologies and Libraries for GPU Acceleration

- **CUDA**: Hardware-accelerated deep learning on Nvidia GPUs ([Nvidia CUDA](https://developer.nvidia.com/cuda-toolkit))
- **DirectML**: For AMD GPUs ([GitHub](https://github.com/microsoft/DirectML))
- **PyTorch**: Deep learning framework with CUDA backend
- **NumPy, pandas**: Data processing

## Project Structure

```
Forex Trading Bot/
├── .venv/                 # Python virtual environment
├── .vscode/               # VSCode settings
├── data/                  # Stored market data
├── logs/                  # Application logs
├── models/                # Saved ML models
├── modules/   
│   ├── main.py            # Main entry point for backtesting/training
│   ├── config.py          # Configuration settings
│   ├── data_fetcher.py    # Data fetching and preprocessing
│   ├── dukascopy_downloader.py # Downloads forex data
│   ├── live_trading.py    # MetaTrader 5 live trading
│   ├── logger.py          # Logging utilities
│   ├── model.py           # PPO implementation (PyTorch Lightning)
│   ├── debug.py           # Analytics and visualization
│   └── visualization.py   # Performance visualization
├── reports/               # Performance reports and charts
├── wandb/                 # Weights & Biases experiment tracking
├── .env                   # Environment variables (MT5 credentials)
├── .gitignore             # Git ignore rules
├── mypy.ini               # Type checking config
├── py.typed               # Typing marker file
├── requirements.txt       # Python dependencies
├── test_directml.py       # GPU test script
└── README.md              # This file
```
![image2](image2)

## Configuration

All configurable variables are in `modules/config.py`:

- **Trading Parameters**: Symbols, timeframes, position sizes, trade limits
- **Risk Management**: Stop-loss, take-profit, max trades
- **Model Settings**: Learning rates, batch sizes, training parameters
- **MetaTrader 5**: Account credentials, server settings
- **Reward System**: Reward weights and penalties
- **Optimization**: Hyperparameter search spaces

> **Tip:** Lower `BATCH_SIZE` and `N_STEPS` for less memory usage. Tune parameters for your hardware and goals.

### Environment Variables

Create a `.env` file in the project root for MetaTrader 5 credentials:
```
MT5_LOGIN=your_mt5_account_number
MT5_PASSWORD=your_mt5_password
MT5_SERVER=your_broker_server
```
**Important:** Add `.env` to `.gitignore` to keep credentials secure.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Stefodan21/forex-trading-bot.git
   cd forex-trading-bot
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # Linux/Mac
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Test GPU acceleration:
   ```bash
   python test_directml.py
   ```

## Usage

### Backtesting/Training

```bash
cd modules
python main.py
```

### Live Trading

```bash
cd modules
python live_trading.py
```

> **Note:**  
> - `main.py` is for backtesting/training only.  
> - `live_trading.py` is for live trading only.

### Model Saving and Live Trading

- After training, models are saved in the `models/` folder.
- Update `MODEL_PATH` in `modules/config.py` to point to your trained model (e.g., `models/final_model.zip`) for live trading.

## Performance Tracking

- Real-time metrics
- Trade analysis/statistics
- Portfolio visualization
- Risk-adjusted returns
- Drawdown analysis
- Position sizing tracking

## Troubleshooting

- **GPU Issues**: Update AMD drivers, verify DirectML (run `python test_directml.py`), check OpenCL compatibility.
- **MetaTrader 5**: Verify `.env` credentials, make sure MT5 is running/logged in, check server settings.
- **Training Issues**: Confirm data files in `data/`, check system memory, monitor GPU temps.

## Security

- Never commit credentials
- Use `.env` for sensitive info
- Keep dependencies up to date
- Monitor trading activity/balance

## License

This project is licensed under the GNU General Public License v3.0. See the [LICENSE](LICENSE) file for details.

## Disclaimer

This trading bot is for educational and research purposes. Past performance does not guarantee future results. Trading involves substantial risk of loss. Use at your own risk.

---

Feel free to contribute, open issues, or fork for your own research!