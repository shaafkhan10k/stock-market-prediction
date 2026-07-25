"""
Notebook 1: S&P 500 Data Fetching & The Data Leakage Trap

In this notebook, we explore:
1. Fetching historical daily S&P 500 price data via yfinance.
2. Understanding why naive train_test_split (random shuffling) creates massive data leakage in time series models.
3. Implementing proper temporal expanding window backtesting.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.data_loader import fetch_sp500_data
from src.features import prepare_dataset
from src.leakage_demo import compare_leakage_impact

print("--- Step 1: Fetching S&P 500 Historical Data ---")
sp500 = fetch_sp500_data(start_date="1990-01-01")
print(f"Data range: {sp500.index[0].date()} to {sp500.index[-1].date()}")
print(sp500.head())

print("\n--- Step 2: Running Data Leakage Audit ---")
compare_leakage_impact()
