"""
Technical Indicators module for Forex Trading Bot
"""

import pandas as pd
import numpy as np


class TechnicalIndicators:
    def __init__(self):
        self.feature_names = []

    def add_all_indicators(self, df, price_col='close'):
        """Add all technical indicators to the dataframe"""
        df = df.copy()

        # Simple Moving Averages
        df = self.add_sma(df, price_col, periods=[5, 10, 20, 50])

        # Exponential Moving Averages
        df = self.add_ema(df, price_col, periods=[5, 10, 20, 50])

        # RSI
        df = self.add_rsi(df, price_col, period=14)

        # MACD
        df = self.add_macd(df, price_col)

        # Bollinger Bands
        df = self.add_bollinger_bands(df, price_col, period=20, std_dev=2)

        # Volume indicators (if volume exists)
        if 'volume' in df.columns:
            df = self.add_volume_indicators(df)

        return df

    def add_sma(self, df, price_col, periods):
        """Add Simple Moving Averages"""
        for period in periods:
            col_name = f'SMA_{period}'
            df[col_name] = df[price_col].rolling(window=period).mean()
            self.feature_names.append(col_name)
        return df

    def add_ema(self, df, price_col, periods):
        """Add Exponential Moving Averages"""
        for period in periods:
            col_name = f'EMA_{period}'
            df[col_name] = df[price_col].ewm(span=period, adjust=False).mean()
            self.feature_names.append(col_name)
        return df

    def add_rsi(self, df, price_col, period=14):
        """Add Relative Strength Index"""
        delta = df[price_col].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        self.feature_names.append('RSI')
        return df

    def add_macd(self, df, price_col, fast_period=12, slow_period=26, signal_period=9):
        """Add MACD (Moving Average Convergence Divergence)"""
        fast_ema = df[price_col].ewm(span=fast_period, adjust=False).mean()
        slow_ema = df[price_col].ewm(span=slow_period, adjust=False).mean()
        df['MACD'] = fast_ema - slow_ema
        df['MACD_Signal'] = df['MACD'].ewm(span=signal_period, adjust=False).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        self.feature_names.extend(['MACD', 'MACD_Signal', 'MACD_Histogram'])
        return df

    def add_bollinger_bands(self, df, price_col, period=20, std_dev=2):
        """Add Bollinger Bands"""
        sma = df[price_col].rolling(window=period).mean()
        std = df[price_col].rolling(window=period).std()
        df['BB_Upper'] = sma + (std * std_dev)
        df['BB_Lower'] = sma - (std * std_dev)
        df['BB_Middle'] = sma
        self.feature_names.extend(['BB_Upper', 'BB_Lower', 'BB_Middle'])
        return df

    def add_volume_indicators(self, df):
        """Add volume-based indicators"""
        if 'volume' in df.columns and df['volume'].notna().any():
            # Volume Moving Average
            df['Volume_SMA_20'] = df['volume'].rolling(window=20).mean()
            self.feature_names.append('Volume_SMA_20')

            # Volume Rate of Change
            df['Volume_ROC'] = df['volume'].pct_change(periods=1) * 100
            self.feature_names.append('Volume_ROC')
        return df

    def get_feature_names(self):
        """Get list of all added feature names"""
        return self.feature_names