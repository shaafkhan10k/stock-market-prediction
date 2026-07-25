"""
Notebook 3: Multi-Horizon Rolling Predictors & Decision Thresholding

In this notebook:
1. We incorporate multi-horizon rolling ratios (2, 5, 60, 250, 1000 days).
2. We demonstrate how shifting Target by 1 prevents rolling trend data leakage.
3. We evaluate custom confidence decision thresholding (T=0.55 and T=0.60).
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import fetch_sp500_data
from src.features import prepare_dataset
from src.backtester import backtest
from src.metrics import compute_ml_metrics, compute_financial_returns, print_performance_report

sp500 = fetch_sp500_data(start_date="1990-01-01")
clean_df, predictors = prepare_dataset(sp500, feature_set="rolling")

print(f"Rolling Predictors ({len(predictors)}): {predictors}")

for t in [0.50, 0.55, 0.60]:
    print(f"\n--- Backtesting Threshold T={t:.2f} ---")
    results = backtest(clean_df, predictors, start=2500, step=125, custom_threshold=t)
    ml_metrics = compute_ml_metrics(results)
    equity_df, fin_metrics = compute_financial_returns(results)
    print_performance_report(ml_metrics, fin_metrics, name=f"Rolling Model (T={t:.2f})")
