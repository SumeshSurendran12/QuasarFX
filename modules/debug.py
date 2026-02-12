"""
Debug module for the Forex Trading Bot
Provides debugging utilities and logging for the current codebase
"""

import logging
import sys
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import seaborn as sns
from pathlib import Path
from config import (
    INITIAL_BALANCE, 
    MIN_POSITION_SIZE, 
    MAX_TRADES_PER_WEEK
)

class DebugLogger:
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Set up logging
        self.logger = logging.getLogger("trading_bot_debug")
        self.logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # File handler
        log_file = self.log_dir / f"debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Initialize metrics storage
        self.metrics_history = []
        self.trade_history = []
        self.error_history = []
        self.training_stats = {
            'episode_number': 0,
            'trades': 0,
            'wins': 0,
            'losses': 0,
            'pnl': 0.0,
            'start_balance': INITIAL_BALANCE,
            'current_balance': INITIAL_BALANCE,
            'win_rate': 0.0,
            'avg_trade_pnl': 0.0,
            'max_profit': 0.0,
            'max_loss': 0.0,
            'weekly_profit': 0.0,
            'weekly_trades': 0,
            'weekly_wins': 0,
            'weekly_losses': 0,
            'weekly_win_rate': 0.0,
            'timesteps': 0,
            'speed': 0.0,
            'profit_factor': 0.0,
            'avg_profit_factor': 0.0,
            'current_position_size': MIN_POSITION_SIZE,
            'dynamic_costs': {'spread': 0.0, 'slippage': 0.0}
        }
        self.week_start_time = datetime.now()
        self.optuna_study = None
        self.portfolio_metrics = {}
    
    def log_training_progress(self, episode: int, timesteps: int, speed: float, env_info: Dict[str, Any] = None): 
        """Log training progress with current environment info"""
        self.training_stats['episode_number'] = episode
        self.training_stats['timesteps'] = timesteps
        self.training_stats['speed'] = speed
        
        # Update from environment info if available
        if env_info:
            metrics = env_info.get('metrics', {})
            self.training_stats.update({
                'current_balance': env_info.get('balance', self.training_stats['current_balance']),
                'pnl': metrics.get('net_profit', self.training_stats['pnl']),
                'win_rate': metrics.get('win_rate', 0.0) * 100,
                'avg_profit_factor': metrics.get('avg_profit_factor', 0.0),
                'weekly_trades': env_info.get('weekly_trades', 0),
                'current_position_size': env_info.get('position_size', self.training_stats['current_position_size'])
            })
        
        self.logger.info("\n=== Training Progress ===")
        self.logger.info(f"Current Episode: {episode}")
        self.logger.info(f"Timesteps: {timesteps}")
        self.logger.info(f"Speed: {speed:.1f} steps/s")
        if env_info and 'eta_seconds' in env_info and env_info['eta_seconds'] is not None:
            try:
                eta_seconds = float(env_info['eta_seconds'])
                if eta_seconds >= 0:
                    hours = int(eta_seconds // 3600)
                    minutes = int((eta_seconds % 3600) // 60)
                    seconds = int(eta_seconds % 60)
                    self.logger.info(f"ETA: {hours:02d}:{minutes:02d}:{seconds:02d}")
            except Exception:
                pass
        self.logger.info(f"Current Episode P&L: ${self.training_stats['pnl']:.2f}")
        self.logger.info(f"Current Balance: ${self.training_stats['current_balance']:.2f}")
        self.logger.info(f"Win Rate: {self.training_stats['win_rate']:.1f}%")
        self.logger.info(f"Avg Profit Factor: {self.training_stats['avg_profit_factor']:.2f}")
        self.logger.info(f"Current Position Size: {self.training_stats['current_position_size']:.3f} lots")
        self.logger.info(f"Weekly Trades: {self.training_stats['weekly_trades']}/{MAX_TRADES_PER_WEEK}")
        
        self.logger.info("\n=== Training Week Statistics ===")
        self.logger.info(f"Training Week: {((datetime.now() - self.week_start_time).days // 7) + 1}")
        self.logger.info(f"Starting Balance: ${self.training_stats['start_balance']:.2f}")
        self.logger.info(f"Current Balance: ${self.training_stats['current_balance']:.2f}")
        self.logger.info(f"Weekly Profit: ${self.training_stats['weekly_profit']:.2f}")
        self.logger.info("="*50 + "\n")
    
    def log_optuna_trial(self, trial_number: int, params: Dict[str, Any], reward: float):
        """Log Optuna hyperparameter optimization trial"""
        self.logger.info(f"Trial {trial_number}: Reward = {reward:.4f}")
        self.logger.info(f"Parameters: {params}")
        
        # Store trial info
        trial_info = {
            'trial_number': trial_number,
            'params': params,
            'reward': reward,
            'timestamp': datetime.now()
        }
        self.metrics_history.append(trial_info)
    
    def log_optuna_study(self, study):
        """Log Optuna study results"""
        self.optuna_study = study
        self.logger.info("\n=== Optuna Study Results ===")
        self.logger.info(f"Best Trial: {study.best_trial.number}")
        self.logger.info(f"Best Reward: {study.best_value:.4f}")
        self.logger.info(f"Best Parameters: {study.best_params}")
        self.logger.info(f"Number of Trials: {len(study.trials)}")
        self.logger.info("="*50 + "\n")
    
    def log_portfolio_metrics(self, portfolio_optimizer, returns_data: pd.DataFrame):
        """Log portfolio optimization metrics"""
        try:
            # Calculate basic portfolio metrics
            mean_returns = returns_data.mean()
            volatility = returns_data.std()
            sharpe_ratio = mean_returns / volatility if volatility > 0 else 0
            
            self.portfolio_metrics = {
                'mean_returns': mean_returns,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'correlation_matrix': portfolio_optimizer.cov_matrix,
                'timestamp': datetime.now()
            }
            
            self.logger.info("\n=== Portfolio Metrics ===")
            self.logger.info(f"Mean Returns: {mean_returns:.6f}")
            self.logger.info(f"Volatility: {volatility:.6f}")
            self.logger.info(f"Sharpe Ratio: {sharpe_ratio:.4f}")
            self.logger.info("="*50 + "\n")
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio metrics: {e}")
    
    def log_live_trading(self, symbol: str, action: int, position_size: float, 
                        current_price: float, balance: float, magic: int):
        """Log live trading actions"""
        action_names = {0: 'HOLD', 1: 'BUY', 2: 'SELL', 3: 'CLOSE'}
        action_name = action_names.get(action, 'UNKNOWN')
        
        self.logger.info(f"Live Trading - {symbol}: {action_name} | "
                        f"Size: {position_size} | Price: {current_price:.5f} | "
                        f"Balance: ${balance:.2f} | Magic: {magic}")
    
    def log_dynamic_costs(self, spread: float, slippage: float, volatility: float):
        """Log dynamic trading costs"""
        self.training_stats['dynamic_costs'] = {
            'spread': spread,
            'slippage': slippage,
            'volatility': volatility
        }
        
        self.logger.debug(f"Dynamic Costs - Volatility: {volatility:.6f}, "
                         f"Spread: {spread:.6f}, Slippage: {slippage:.6f}")
    
    def log_error(self, error: Exception, context: Dict[str, Any] = None):
        """Log an error with essential information"""
        error_info = {
            'timestamp': datetime.now(),
            'error_type': type(error).__name__,
            'error_message': str(error),
            'traceback': traceback.format_exc(),
            'context': context or {}
        }
        self.error_history.append(error_info)
        self.logger.error(f"Error: {error_info['error_type']} - {error_info['error_message']}")
        if context:
            self.logger.error(f"Context: {context}")
    
    def log_episode_summary(self, episode_number: int, episode_metrics: Dict[str, Any] = None):
        """Log episode trading summary"""
        self.training_stats['episode_number'] = episode_number
        
        # Update from episode metrics if provided
        if episode_metrics:
            self.training_stats.update(episode_metrics)
        
        # Print episode summary
        self.logger.info("\n" + "="*50)
        self.logger.info(f"Episode {episode_number} completed!")
        self.logger.info(f"Episode P&L: ${self.training_stats['pnl']:.2f}")
        self.logger.info(f"Current Balance: ${self.training_stats['current_balance']:.2f}")
        self.logger.info(f"Win Rate: {self.training_stats['win_rate']:.1f}%")
        self.logger.info(f"Avg Profit Factor: {self.training_stats['avg_profit_factor']:.2f}")
        self.logger.info("="*50)
        
        # Reset episode stats but keep weekly stats
        self.training_stats.update({
            'episode_number': 0,
            'trades': 0,
            'wins': 0,
            'losses': 0,
            'pnl': 0.0,
            'start_balance': self.training_stats['current_balance'],
            'win_rate': 0.0,
            'avg_trade_pnl': 0.0,
            'max_profit': 0.0,
            'max_loss': 0.0,
            'timesteps': 0,
            'speed': 0.0
        })
    
    def generate_debug_report(self) -> str:
        """Generate a comprehensive debug report"""
        report = []
        report.append("=== Forex Trading Bot Debug Report ===")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Training Summary
        report.append("\n=== Training Summary ===")
        report.append(f"Episode Number: {self.training_stats['episode_number']}")
        report.append(f"Total Trades: {self.training_stats['trades']}")
        report.append(f"Winning Trades: {self.training_stats['wins']}")
        report.append(f"Losing Trades: {self.training_stats['losses']}")
        report.append(f"Win Rate: {self.training_stats['win_rate']:.1f}%")
        report.append(f"Total P&L: ${self.training_stats['pnl']:.2f}")
        report.append(f"Avg Profit Factor: {self.training_stats['avg_profit_factor']:.2f}")
        report.append(f"Current Position Size: {self.training_stats['current_position_size']:.3f} lots")
        report.append(f"Starting Balance: ${self.training_stats['start_balance']:.2f}")
        report.append(f"Current Balance: ${self.training_stats['current_balance']:.2f}")
        
        # Weekly Statistics
        report.append("\n=== Weekly Statistics ===")
        report.append(f"Weekly Trades: {self.training_stats['weekly_trades']}/{MAX_TRADES_PER_WEEK}")
        report.append(f"Weekly Wins: {self.training_stats['weekly_wins']}")
        report.append(f"Weekly Losses: {self.training_stats['weekly_losses']}")
        report.append(f"Weekly Win Rate: {self.training_stats['weekly_win_rate']:.1f}%")
        report.append(f"Weekly Profit: ${self.training_stats['weekly_profit']:.2f}")
        
        # Optuna Study Results
        if self.optuna_study:
            report.append("\n=== Hyperparameter Optimization ===")
            report.append(f"Best Trial: {self.optuna_study.best_trial.number}")
            report.append(f"Best Reward: {self.optuna_study.best_value:.4f}")
            report.append(f"Number of Trials: {len(self.optuna_study.trials)}")
            report.append(f"Best Parameters: {self.optuna_study.best_params}")
        
        # Portfolio Metrics
        if self.portfolio_metrics:
            report.append("\n=== Portfolio Metrics ===")
            report.append(f"Mean Returns: {self.portfolio_metrics['mean_returns']:.6f}")
            report.append(f"Volatility: {self.portfolio_metrics['volatility']:.6f}")
            report.append(f"Sharpe Ratio: {self.portfolio_metrics['sharpe_ratio']:.4f}")
        
        # Error Summary
        if self.error_history:
            error_types = {}
            for error in self.error_history:
                error_type = error['error_type']
                error_types[error_type] = error_types.get(error_type, 0) + 1
            
            report.append("\n=== Errors ===")
            for error_type, count in error_types.items():
                report.append(f"{error_type}: {count}")
        
        return "\n".join(report)
    
    def plot_debug_metrics(self, save_path: Optional[str] = None):
        """Generate focused debug visualization plots using seaborn"""
        if not self.metrics_history:
            return
        
        # Set seaborn style
        sns.set_style("whitegrid")
        
        # Create figure with subplots
        fig, axes = sns.plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        # 1. Optuna Trials (if available)
        if any('trial_number' in m for m in self.metrics_history):
            trial_data = [m for m in self.metrics_history if 'trial_number' in m]
            if trial_data:
                df_trials = pd.DataFrame(trial_data)
                sns.lineplot(data=df_trials, x='trial_number', y='reward', ax=axes[0])
                axes[0].set_title('Optuna Trial Rewards')
                axes[0].set_xlabel('Trial Number')
                axes[0].set_ylabel('Reward')
        
        # 2. Portfolio Metrics (if available)
        if self.portfolio_metrics:
            metrics_data = {
                'Metric': ['Returns', 'Volatility'],
                'Value': [
                    self.portfolio_metrics.get('mean_returns', 0),
                    self.portfolio_metrics.get('volatility', 0)
                ]
            }
            df_metrics = pd.DataFrame(metrics_data)
            sns.barplot(data=df_metrics, x='Metric', y='Value', ax=axes[1])
            axes[1].set_title('Portfolio Metrics')
            axes[1].set_ylabel('Value')
        
        # 3. Training Stats
        stats_data = {
            'Statistic': ['Win Rate', 'Profit Factor', 'Weekly Trades'],
            'Value': [
                self.training_stats['win_rate'],
                self.training_stats['avg_profit_factor'],
                self.training_stats['weekly_trades']
            ]
        }
        df_stats = pd.DataFrame(stats_data)
        sns.barplot(data=df_stats, x='Statistic', y='Value', ax=axes[2])
        axes[2].set_title('Training Statistics')
        axes[2].set_ylabel('Value')
        
        # 4. Balance Over Time (simulated)
        balance_data = {
            'Period': ['Start', 'Current'],
            'Balance': [self.training_stats['start_balance'], self.training_stats['current_balance']]
        }
        df_balance = pd.DataFrame(balance_data)
        sns.lineplot(data=df_balance, x='Period', y='Balance', marker='o', ax=axes[3])
        axes[3].set_title('Account Balance')
        axes[3].set_ylabel('Balance ($)')
        
        # 5. Dynamic Costs (if available)
        if self.training_stats['dynamic_costs']['spread'] > 0:
            costs_data = {
                'Cost Type': ['Spread', 'Slippage'],
                'Value': [
                    self.training_stats['dynamic_costs']['spread'],
                    self.training_stats['dynamic_costs']['slippage']
                ]
            }
            df_costs = pd.DataFrame(costs_data)
            sns.barplot(data=df_costs, x='Cost Type', y='Value', ax=axes[4])
            axes[4].set_title('Dynamic Trading Costs')
            axes[4].set_ylabel('Cost')
        
        # 6. Error Distribution (if available)
        if self.error_history:
            error_types = {}
            for error in self.error_history:
                error_type = error['error_type']
                error_types[error_type] = error_types.get(error_type, 0) + 1
            
            if error_types:
                error_data = {
                    'Error Type': list(error_types.keys()),
                    'Count': list(error_types.values())
                }
                df_errors = pd.DataFrame(error_data)
                sns.barplot(data=df_errors, x='Error Type', y='Count', ax=axes[5])
                axes[5].set_title('Error Distribution')
                axes[5].set_ylabel('Count')
                axes[5].tick_params(axis='x', rotation=45)
        
        # Hide empty subplots
        for i in range(len(axes)):
            if i >= 6 or (i == 1 and not self.portfolio_metrics) or (i == 4 and self.training_stats['dynamic_costs']['spread'] <= 0) or (i == 5 and not self.error_history):
                axes[i].set_visible(False)
        
        sns.plt.tight_layout()
        
        if save_path:
            sns.plt.savefig(save_path, dpi=300, bbox_inches='tight')
            sns.plt.close()
        else:
            sns.plt.show()
    
    def save_debug_report(self, report: str, filepath: str):
        """Save the debug report to a file"""
        with open(filepath, 'w') as f:
            f.write(report)
        self.logger.info(f"Debug report saved to {filepath}")
    
    def save_optuna_visualizations(self, study, save_dir: str = "reports"):
        """Save Optuna study visualizations"""
        try:
            save_path = Path(save_dir)
            save_path.mkdir(exist_ok=True)
            
            # Optimization history
            fig1 = study.optimization_history()
            fig1.write_html(str(save_path / "optimization_history.html"))
            
            # Parameter importance
            fig2 = study.param_importances()
            fig2.write_html(str(save_path / "param_importances.html"))
            
            # Parallel coordinate plot
            fig3 = study.parallel_coordinate()
            fig3.write_html(str(save_path / "parallel_coordinate.html"))
            
            self.logger.info(f"Optuna visualizations saved to {save_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save Optuna visualizations: {e}")
    
    def update_position_size(self, new_position_size: float):
        """Update the current position size"""
        self.training_stats['current_position_size'] = new_position_size
        self.logger.debug(f"Position size updated to: {new_position_size:.3f} lots")
