"""
Data fetcher module for retrieving market data
"""

import os
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Union, Any

import numpy as np
import pandas as pd

try:
    from .logger import logger as _bootstrap_logger
    from .config import SYMBOL, TIMEFRAME, YEARS, DATA_CSV_PATH
    from .dukascopy_downloader import download_dukascopy_csv, download_dukascopy_range_csv
    from .technical_indicators import TechnicalIndicators
except ImportError:  # pragma: no cover - script mode fallback
    from logger import logger as _bootstrap_logger
    from config import SYMBOL, TIMEFRAME, YEARS, DATA_CSV_PATH
    from dukascopy_downloader import download_dukascopy_csv, download_dukascopy_range_csv
    from technical_indicators import TechnicalIndicators

logger = logging.getLogger(__name__)

def cov_to_corr(cov_matrix):
    """
    Convert a covariance matrix to a correlation matrix.
    
    Parameters
    ----------
    cov_matrix : array_like, 2d
        Covariance matrix, assumed to be symmetric
        
    Returns
    -------
    corr_matrix : ndarray, 2d
        Correlation matrix
    """
    cov_array = np.asarray(cov_matrix)
    std = np.sqrt(np.diag(cov_array))
    corr = cov_array / np.outer(std, std)
    return corr

class DataFetcher:
    def __init__(self, symbol: Optional[str] = None, timeframe: Optional[str] = None):
        self.symbol = symbol or SYMBOL
        self.timeframe = timeframe or TIMEFRAME
        self.cache = {}
        self.cache_timeout = timedelta(minutes=5)
        self.data_csv_path = DATA_CSV_PATH
        self.data_dir = Path(__file__).resolve().parent.parent / "data"

    def _normalize_symbol_for_filename(self, symbol: str) -> List[str]:
        safe_symbol = str(symbol).replace("/", "_").replace("\\", "_").replace(" ", "").upper()
        compact = safe_symbol.replace("_", "")
        variants = [safe_symbol]
        if len(compact) == 6:
            variants.append(f"{compact[:3]}_{compact[3:]}")
            variants.append(compact)
        return list(dict.fromkeys(variants))

    def _find_existing_csv(self, symbol: str) -> Optional[Path]:
        candidates: List[Path] = []
        for variant in self._normalize_symbol_for_filename(symbol):
            candidates.extend(self.data_dir.glob(f"{variant}_*.csv"))
        if not candidates:
            return None
        return max(candidates, key=lambda path: path.stat().st_mtime)

    def fetch_historical_data(self, download_if_missing=False, symbol=None, timeframe=None, years: int = YEARS, add_indicators=True) -> pd.DataFrame:
        """
        Load historical forex data from a single CSV file in the data/ directory.
        If the CSV does not exist and download_if_missing is True, download from Dukascopy and reformat.
        The CSV must contain columns: ['open', 'high', 'low', 'close', 'volume', 'time' or 'date'].
        Args:
            download_if_missing (bool): If True, download from Dukascopy if CSV is missing.
            symbol (str): Symbol for Dukascopy download (e.g., 'EURUSD').
            timeframe (str): Timeframe for Dukascopy download (e.g., 'M1').
            years (int): Number of years of historical data to fetch (default: config.YEARS)
            add_indicators (bool): If True, add technical indicators to the data
        Returns:
            pd.DataFrame: DataFrame with columns ['open', 'high', 'low', 'close', 'volume'] and technical indicators with a datetime index.
        Raises:
            FileNotFoundError: If the CSV file does not exist and download_if_missing is False.
            ValueError: If required columns are missing.
        """
        symbol = symbol or self.symbol
        timeframe = timeframe or self.timeframe
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years * 365)
        
        csv_path = self._find_existing_csv(symbol)
        if csv_path is None:
            if not download_if_missing:
                raise FileNotFoundError(
                    f"No historical CSV found for symbol {symbol} in {self.data_dir}. "
                    "Set download_if_missing=True to fetch from Dukascopy."
                )

            logger.info("No local CSV found, downloading fresh data from Dukascopy...")
            downloaded_path = download_dukascopy_range_csv(symbol, timeframe, start_date, end_date)
            if not downloaded_path or not os.path.exists(downloaded_path):
                logger.error(f"CSV file not found or download failed: {downloaded_path}")
                raise FileNotFoundError(f"CSV file not found or download failed: {downloaded_path}")
            csv_path = Path(downloaded_path)

        logger.info(f"Loading historical data from {csv_path}")
        df = pd.read_csv(csv_path)
        
        # Accept either 'time' or 'date' as the datetime column
        datetime_col = 'time' if 'time' in df.columns else 'date' if 'date' in df.columns else None
        if not datetime_col:
            logger.error("CSV must contain a 'time' or 'date' column.")
            raise ValueError("CSV must contain a 'time' or 'date' column.")
        
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                logger.error(f"CSV is missing required column: {col}")
                raise ValueError(f"CSV is missing required column: {col}")
        
        df[datetime_col] = pd.to_datetime(df[datetime_col])
        df.set_index(datetime_col, inplace=True)
        df = df[required_cols]
        
        # Add technical indicators if requested
        if add_indicators:
            logger.info("Adding technical indicators...")
            ti = TechnicalIndicators()
            df = ti.add_all_indicators(df, price_col='close')
            logger.info(f"Added {len(ti.get_feature_names())} technical indicators")

        logger.info(f"Loaded {len(df)} rows from {csv_path}")
        logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
        logger.info(f"Final columns count: {len(df.columns)}")
        
        return df

    def get_latest_price(self, symbol: str) -> float:
        """Get the latest close price for a symbol from the most recent row in the Dukascopy-downloaded CSV file."""
        try:
            # Load the latest data from the CSV file for the given symbol
            df = self.fetch_historical_data(download_if_missing=False, symbol=symbol)
            if not df.empty:
                return float(df['close'].iloc[-1])
            else:
                logger.error(f"No data found for {symbol}")
                return 0.0
        except Exception as e:
            logger.error(f"Error getting latest price for {symbol}: {str(e)}")
            return 0.0

    def clear_cache(self):
        """Clear the data cache"""
        self.cache.clear()
    
    def _analyze_data_gaps(self, df: pd.DataFrame, start_date: datetime, end_date: datetime) -> dict:
        """Analyze gaps in the data and return detailed statistics (for debugging)"""
        # Create a complete date range
        all_dates = pd.date_range(start=start_date, end=end_date, freq='D')

        # Find missing dates
        missing_dates = all_dates.difference(df.index)

        # Calculate gap statistics
        gap_stats = {
            'total_days': len(all_dates),
            'available_days': len(df),
            'missing_days': len(missing_dates),
            'coverage': (len(df) / len(all_dates)) * 100,
            'gaps': []
        }

        # Find continuous gaps
        if len(missing_dates) > 0:
            missing_dates = sorted(missing_dates)
            current_gap_start = missing_dates[0]
            current_gap_end = missing_dates[0]

            for i in range(1, len(missing_dates)):
                if (missing_dates[i] - missing_dates[i - 1]).days == 1:
                    current_gap_end = missing_dates[i]
                else:
                    gap_stats['gaps'].append({
                        'start': current_gap_start,
                        'end': current_gap_end,
                        'duration': (current_gap_end - current_gap_start).days + 1
                    })
                    current_gap_start = missing_dates[i]
                    current_gap_end = missing_dates[i]

            # Add the last gap
            gap_stats['gaps'].append({
                'start': current_gap_start,
                'end': current_gap_end,
                'duration': (current_gap_end - current_gap_start).days + 1
            })

        return gap_stats
    
    def _log_gap_analysis(self, gap_stats: dict):
        """Log the results of gap analysis (for debugging)"""
        logger.info("\n=== Data Gap Analysis ===")
        logger.info(f"Total days: {gap_stats['total_days']:,}")
        logger.info(f"Available days: {gap_stats['available_days']:,}")
        logger.info(f"Missing days: {gap_stats['missing_days']:,}")
        logger.info(f"Data coverage: {gap_stats['coverage']:.2f}%")

        if gap_stats['gaps']:
            logger.info("\n=== Data Gaps Detected ===")
            # Sort gaps by duration
            sorted_gaps = sorted(
                gap_stats['gaps'],
                key=lambda x: x['duration'],
                reverse=True)

            # Log the top 5 largest gaps
            logger.info("Top 5 largest gaps:")
            for i, gap in enumerate(sorted_gaps[:5], 1):
                logger.info(f"Gap {i}:")
                logger.info(f"  Start: {gap['start']}")
                logger.info(f"  End: {gap['end']}")
                logger.info(f"  Duration: {gap['duration']} days") 
