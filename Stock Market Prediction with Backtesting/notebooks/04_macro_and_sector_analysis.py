"""
Notebook 4: Macroeconomic, Sector Correlations & Feature Importance Analysis

In this notebook:
1. We add cross-asset macro indicators: VIX, Nasdaq spread, Treasury Yields, Crude Oil, Gold.
2. We run expanding walk-forward backtests.
3. We extract and rank top feature importances to document which features move the needle and why.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import fetch_sp500_data, fetch_macro_indicators
from src.features import prepare_dataset
from src.backtester import backtest, compute_average_feature_importance
from src.metrics import compute_ml_metrics, compute_financial_returns, print_performance_report

sp500 = fetch_sp500_data(start_date="1990-01-01")
macro = fetch_macro_indicators(start_date="1990-01-01")

clean_df, predictors = prepare_dataset(sp500, macro, feature_set="all")
print(f"Total Predictors ({len(predictors)}): {predictors}")

results = backtest(clean_df, predictors, start=2500, step=125, custom_threshold=0.55)
ml_metrics = compute_ml_metrics(results)
equity_df, fin_metrics = compute_financial_returns(results)

print_performance_report(ml_metrics, fin_metrics, name="Full Macro + Rolling Model (T=0.55)")

# Feature Importance Ranking
imp = compute_average_feature_importance(clean_df, predictors, start=2500, step=125)
print("\n--- FEATURE IMPORTANCE RANKING ---")
print(imp)
