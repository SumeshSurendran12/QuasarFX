import os
import pandas as pd
from datetime import datetime, timedelta
import logging
from dukascopy_python import fetch
from dukascopy_python import TIME_UNIT_MIN, TIME_UNIT_HOUR, TIME_UNIT_DAY
try:
    from . import config
except ImportError:  # pragma: no cover - script mode fallback
    import config

logger = logging.getLogger(__name__)

def download_dukascopy_csv(symbol=None, timeframe=None, start_date=None, end_date=None):
    """
    Download historical data from Dukascopy and save it as a CSV file.
    
    Args:
        symbol: Trading symbol (Dukascopy instrument constant). Defaults to config.SYMBOL.
        timeframe: Timeframe (Dukascopy interval constant). Defaults to config.TIMEFRAME.
        start_date (datetime, optional): Start date for data download.
        end_date (datetime, optional): End date for data download.
    
    Returns:
        str: Path to the saved CSV file, or None if download fails.
    """
    symbol = (config.SYMBOL if symbol is None else symbol).replace('_', '').upper()
    tf = (config.TIMEFRAME if timeframe is None else timeframe).upper()
    
    if start_date is None or end_date is None:
        logger.error("start_date and end_date must be provided.")
        print("[ERROR] start_date and end_date must be provided.")
        return None
    
    logger.info(f"Downloading {symbol} {tf} from {start_date} to {end_date} using dukascopy-python in 3-year chunks...")
    print(f"[INFO] Downloading {symbol} {tf} from {start_date} to {end_date} using dukascopy-python in 3-year chunks...")
    
    max_years = 3
    dfs = []
    current_start = start_date
    
    while current_start < end_date:
        current_end = min(current_start + timedelta(days=365*max_years), end_date)
        logger.info(f"Fetching {symbol} {tf} from {current_start.date()} to {current_end.date()}")
        print(f"[INFO] Fetching {symbol} {tf} from {current_start.date()} to {current_end.date()}")
        
        df = fetch(
            instrument=symbol,
            interval=tf,
            offer_side=config.OFFER_SIDE_BID,
            start=current_start,
            end=current_end
        )
        
        if df is not None and not df.empty:
            logger.info(f"Fetched chunk columns: {df.columns.tolist()}")
            print(f"[DEBUG] Fetched chunk columns: {df.columns.tolist()}")
            print(f"[DEBUG] DataFrame index: {df.index.name}, type: {type(df.index)}")
            print(f"[DEBUG] First few rows of DataFrame:")
            print(df.head())
            dfs.append(df)
        else:
            logger.warning(f"No data for {symbol} {tf} {current_start.date()} to {current_end.date()}")
            print(f"[WARN] No data for {symbol} {tf} {current_start.date()} to {current_end.date()}")
        current_start = current_end + timedelta(days=1)
    
    if not dfs:
        logger.error("No data downloaded from Dukascopy.")
        print("[ERROR] No data downloaded from Dukascopy.")
        return None
    
    # Concatenate while preserving the datetime index
    df_all = pd.concat(dfs, axis=0)
    
    # Debug the concatenated DataFrame
    print(f"[DEBUG] Concatenated DataFrame columns: {df_all.columns.tolist()}")
    print(f"[DEBUG] Concatenated DataFrame index: {df_all.index.name}, type: {type(df_all.index)}")
    print(f"[DEBUG] First few rows of concatenated DataFrame:")
    print(df_all.head())
    
    # Ensure the time column is named 'time'
    if 'timestamp' in df_all.columns:
        df_all = df_all.rename(columns={'timestamp': 'time'})
    elif 'date' in df_all.columns:
        df_all = df_all.rename(columns={'date': 'time'})
    elif 'time' not in df_all.columns:
        # Check if the index is datetime
        if pd.api.types.is_datetime64_any_dtype(df_all.index):
            print(f"[DEBUG] Index is datetime, resetting index to create time column")
            df_all = df_all.reset_index()
            print(f"[DEBUG] After reset_index, columns: {df_all.columns.tolist()}")
            if 'index' in df_all.columns:
                df_all = df_all.rename(columns={'index': 'time'})
                print(f"[DEBUG] After renaming index to time, columns: {df_all.columns.tolist()}")
            elif 'timestamp' in df_all.columns:
                df_all = df_all.rename(columns={'timestamp': 'time'})
                print(f"[DEBUG] After renaming timestamp to time, columns: {df_all.columns.tolist()}")
        else:
            logger.error("No time, timestamp, or date column found in DataFrame or index. Cannot proceed.")
            print("[ERROR] No time, timestamp, or date column found in DataFrame or index. Cannot proceed.")
            return None
    
    print(f"[DEBUG] Final columns before selection: {df_all.columns.tolist()}")
    
    # Always ensure 'time' is the first column if it exists
    if 'time' in df_all.columns:
        cols = ['time', 'open', 'high', 'low', 'close', 'volume']
        available_cols = [col for col in cols if col in df_all.columns]
        df_all = df_all[available_cols]
        df_all['time'] = pd.to_datetime(df_all['time'])
        print(f"[DEBUG] Final DataFrame columns: {df_all.columns.tolist()}")
    else:
        # fallback: just keep whatever columns are available
        df_all = df_all[[col for col in ['open', 'high', 'low', 'close', 'volume'] if col in df_all.columns]]
        print(f"[DEBUG] No time column found, using fallback columns: {df_all.columns.tolist()}")
    
    if df_all.empty:
        logger.error("Downloaded DataFrame is empty. No CSV will be saved.")
        print("[ERROR] Downloaded DataFrame is empty. No CSV will be saved.")
        return None
    
    # Always generate the output path
    os.makedirs('data', exist_ok=True)
    start = start_date.strftime('%Y%m%d')
    end = end_date.strftime('%Y%m%d')
    safe_symbol = symbol.replace('/', '_').replace(' ', '').replace('\\', '_')
    output_path = os.path.join('data', f"{safe_symbol}_{start}_{end}.csv")
    print(f"[INFO] Output path: {output_path}")
    
    df_all.to_csv(output_path, index=False)
    logger.info(f"Saved Dukascopy data to {output_path} ({len(df_all)} rows)")
    print(f"[INFO] Saved Dukascopy data to {output_path} ({len(df_all)} rows)")
    return output_path

def download_dukascopy_range_csv(symbol=None, timeframe=None, start_date=None, end_date=None):
    return download_dukascopy_csv(symbol, timeframe, start_date, end_date) 
