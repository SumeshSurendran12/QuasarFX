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
│   └── technical_indicators.py # Indicator feature engineering
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


## Lambda Labs H100 SXM Quick Setup (Sticky Steps 1-3)

Step 1 - Fix permissions + create venv (run on the SXM VM)
```bash
cd ~/forex-trading-bot
sudo chown -R ubuntu:ubuntu .
sudo chmod -R u+rwX .
sudo rm -rf .venv

sudo apt update
sudo apt install -y python3.11 python3.11-venv

python3.11 -m venv .venv
source .venv/bin/activate
python -V
```
You should see Python 3.11.x and the prompt should show (.venv).

Step 2 - Re-copy modules/ from your Windows machine
Run this on Windows PowerShell:
```powershell
scp -i "$env:USERPROFILE\.ssh\lambda_a100" -r "C:\Users\sumes\OneDrive - UTHealth Houston\Desktop\FX\Forex-trading-bot\modules" ubuntu@192.222.55.197:~/forex-trading-bot/
```
Short note: Step 3 will fail if `requirements.txt` is missing. Copy it once:
```powershell
scp -i "$env:USERPROFILE\.ssh\lambda_a100" "C:\Users\sumes\OneDrive - UTHealth Houston\Desktop\FX\Forex-trading-bot\requirements.txt" ubuntu@192.222.55.197:~/forex-trading-bot/
```
Then on the VM:
```bash
ls ~/forex-trading-bot/modules
```
You should see `data_fetcher.py`, `main.py`, etc.

Step 3 - Install deps inside the venv (on VM)
```bash
cd ~/forex-trading-bot
python -m pip install -U pip

pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
grep -vE '^(torch|torchvision|torchaudio)' requirements.txt | pip install -r /dev/stdin
```

Optional Step 4 - Run the benchmark (on VM)
```bash
cd ~/forex-trading-bot/modules
WANDB_MODE=disabled python benchmark.py
```

Benchmark results (20,000 timesteps, Feb 7, 2026):
| GPU | Elapsed (sec) | Steps/sec |
| --- | --- | --- |
| H100 PCIe | 143.85 | 139.0 |
| H100 SXM | 53.23 | 375.7 |
## Usage

## Execution Stages

- Strategy 1: `PAPER_CANDIDATE`
- Strategy 2 deterministic branches: `RESEARCH`
- RLM/RL branches: `EXPERIMENTAL_ONLY`
- Promotion target after paper checks: `LIVE_GATED`

## Strategy 1 Deployment Artifacts

- Frozen profile: `strategy_1_profile.json`
- Frozen deployment manifest: `manifest.json`
- Canonical event schema: `schemas/strategy_1_events.schema.json`
- Canonical runtime logs: `events.jsonl` + `daily_summary.json`
- Canonical run_id format: `YYYY-MM-DD_SESSION_shaXXXXXXXX` (example: `2026-03-07_LONDON_sha9f2c1ab4`)
- Frozen reason_code vocabulary: `signal_pass`, `spread_gate`, `session_cap`, `daily_loss_limit`, `max_open_positions`, `outside_session`, `cooldown_active`, `duplicate_signal`, `no_liquidity`, `manual_disable`, `policy_breach`, `within_limits`, `broker_api_failure`
- Contract versions on every event: `schema_version=1.0.0`, `manifest_version=1.0.0`
- Deployment checklist: `docs/strategy_1_deployment_checklist.md`
- Kill-switch policy: `docs/kill_switch_policy.md`
- Daily summary builder script: `scripts/build_daily_summary.py`
- Paper mode report script: `scripts/paper_trading_mode_report.py`
- Daily health report script: `scripts/daily_health_report.py`
- Run ID generator: `scripts/generate_strategy_1_run_id.py`
- Daily pipeline runner: `scripts/run_daily_paper_pipeline.ps1`
- Task scheduler registration script: `scripts/register_daily_paper_pipeline_task.ps1`

### Daily Paper Pipeline

Run immediately:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_daily_paper_pipeline.ps1
```

Daily artifacts are written under `reports/YYYY-MM-DD/`:
- `reports/YYYY-MM-DD/daily_summary.json`
- `reports/YYYY-MM-DD/paper_report.json`
- `reports/YYYY-MM-DD/daily_health.json`
- `reports/YYYY-MM-DD/daily_pipeline.log`

Generate a canonical run_id for the session:

```powershell
python .\scripts\generate_strategy_1_run_id.py --session LONDON
```

Schedule daily at 18:05 local time:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\register_daily_paper_pipeline_task.ps1 -TaskName FX_Strategy1_DailyPipeline -StartTime 18:05
```

If task registration fails with permissions, run the command in an elevated PowerShell session.

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
> - `live_trading.py` supports canonical paper-event emission via `FX_EXECUTION_MODE=paper` (default).
> - Set `FX_EXECUTION_MODE=live` only when you explicitly want live order routing.

Paper session quick-start (Windows PowerShell):
```powershell
$env:FX_EXECUTION_MODE = "paper"
$env:FX_MAX_DURATION_SECONDS = "600"
cd modules
python live_trading.py
```

### Dashboard (Local Frontend)

Strategy 1 weekly monitoring UI is available in `dashboard/` (Vite + React + TypeScript).

Run locally:

```bash
cd dashboard
npm install
npm run dev
```

Build production bundle:

```bash
cd dashboard
npm run build
```

Open the app in your browser and upload/paste:
- `daily_health.json` (from daily health pipeline output)
- `paper_report.json` (from paper mode report output)

Use **Load demo data** to preview the charts without live files.

### Model Saving and Live Trading

- After training, models are saved in the `models/` folder.
- Update `MODEL_PATH` in `modules/config.py` to point to your trained model (e.g., `models/final_model.zip`) for live trading.

**Model Backup (Local + W&B)**
- Best and final models are saved locally on the VM in `models/`.
- Best and final models are also logged to W&B as versioned artifacts (one version per run).
- To download models to your PC after training, run:
```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\download_models.ps1 -Host 192.222.55.197
```

**Safe Training Launcher (Prevents Duplicate Runs)**
- Use the guard script to start training safely inside `tmux` and avoid accidental duplicates:
```bash
bash ./scripts/run_training.sh
```
- If a run is already active, it will print how to attach instead of starting a second one.

**ETA Helper (Final Stage + Full Completion)**
- Print ETA to final stage and full completion from the VM:
```bash
bash ./scripts/eta.sh
```

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
- **Duplicate Runs**: Training uses a lock file at `logs/train.lock` to prevent multiple concurrent runs. Delete it if you intentionally want to run more than one training job.

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
