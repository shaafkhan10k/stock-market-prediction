"""
Notebook 2: Baseline Random Forest Classifier

In this notebook, we build our initial baseline Random Forest model:
1. Construct binary target: Target = 1 if Tomorrow's Close > Today's Close else 0.
2. Build baseline price/volume ratio predictors.
3. Perform initial walk-forward backtest.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import fetch_sp500_data
from src.features import prepare_dataset
from src.backtester import backtest
from src.metrics import compute_ml_metrics, compute_financial_returns, print_performance_report

sp500 = fetch_sp500_data(start_date="1990-01-01")
clean_df, predictors = prepare_dataset(sp500, feature_set="baseline")

print(f"Predictors ({len(predictors)}): {predictors}")

# Walk-forward backtest
results = backtest(clean_df, predictors, start=2500, step=125, custom_threshold=0.50)

ml_metrics = compute_ml_metrics(results)
equity_df, fin_metrics = compute_financial_returns(results)

print_performance_report(ml_metrics, fin_metrics, name="Baseline Random Forest Model")
