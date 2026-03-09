#!/usr/bin/env python3
"""Regression test for technical indicator feature generation."""

from pathlib import Path

import pytest


def test_indicators():
    """Technical indicator generation should add derived columns to OHLCV input."""
    pd = pytest.importorskip("pandas", reason="pandas is required for indicator tests.")

    csv_path = Path("data/EUR_USD_20150622_20250619.csv")
    if not csv_path.exists():
        pytest.skip(f"Sample indicator CSV is not present: {csv_path}")

    from modules.technical_indicators import TechnicalIndicators

    df = pd.read_csv(csv_path)
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time")[["open", "high", "low", "close", "volume"]]

    ti = TechnicalIndicators()
    df_with_indicators = ti.add_all_indicators(df, price_col="close")

    indicator_cols = [
        col for col in df_with_indicators.columns
        if col not in ["open", "high", "low", "close", "volume"]
    ]

    assert df_with_indicators.shape[1] > df.shape[1]
    assert indicator_cols
    assert df_with_indicators[indicator_cols].iloc[-1].notna().any()
