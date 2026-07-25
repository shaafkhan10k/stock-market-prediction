import sys
import pandas as pd
import numpy as np

from src.data_loader import fetch_sp500_data, fetch_macro_indicators
from src.features import prepare_dataset
from src.backtester import backtest, compute_average_feature_importance
from src.metrics import compute_ml_metrics, compute_financial_returns, print_performance_report


def log(msg):
    print(msg, flush=True)


def main():
    log("\n" + "=" * 70)
    log("   S&P 500 PREDICTION — IMPROVED PIPELINE (Ensemble + TA + 5D Target)")
    log("=" * 70)

    # ── 1. Fetch data ──────────────────────────────────────────────────────
    log("\n[Step 1/5] Fetching market data via yfinance...")
    sp500_df = fetch_sp500_data(start_date="1990-01-01")
    macro_df = fetch_macro_indicators(start_date="1990-01-01")
    log(f"  Loaded {len(sp500_df)} S&P 500 trading days "
        f"({sp500_df.index[0].date()} → {sp500_df.index[-1].date()})")

    # ── 2. Experiment matrix ───────────────────────────────────────────────
    log("\n[Step 2/5] Defining experiments...")

    # (label, feature_set, threshold, target_horizon, model_name, regime_aware)
    experiments = [
        # ── Baselines (1-day target, no TA) ──────────────────────────
        ("Original: Baseline RF, 1D target, T=0.50",
         "baseline", 0.50, 1, "RandomForest", False),

        ("Original: All features, RF, 1D target, T=0.55",
         "all", 0.55, 1, "RandomForest", False),

        # ── 5-day target (big upgrade) ────────────────────────────────
        ("Improved: All features, RF, 5D target, T=0.55",
         "all", 0.55, 5, "RandomForest", False),

        # ── Ensemble model ────────────────────────────────────────────
        ("Improved: All features, Ensemble, 5D target, T=0.55",
         "all", 0.55, 5, "Ensemble", False),

        # ── Regime-aware threshold ────────────────────────────────────
        ("Improved: All features, Ensemble, 5D target, T=0.55, Regime-Aware",
         "all", 0.55, 5, "Ensemble", True),

        # ── Tighter threshold for max precision ───────────────────────
        ("Best Precision: All features, Ensemble, 5D target, T=0.60, Regime-Aware",
         "all", 0.60, 5, "Ensemble", True),
    ]

    experiment_results = []

    log("\n[Step 3/5] Running walk-forward backtests...\n")

    for name, f_set, thresh, t_horizon, model_name, regime_aware in experiments:
        log(f"─── {name}")

        clean_df, predictors = prepare_dataset(
            sp500_df, macro_df,
            feature_set=f_set,
            target_horizon=t_horizon,
        )

        bt_df = backtest(
            clean_df, predictors,
            start=2500, step=250,
            custom_threshold=thresh,
            model_name=model_name,
            regime_aware=regime_aware,
            verbose=True,
        )

        ml_m          = compute_ml_metrics(bt_df)
        eq_df, fin_m  = compute_financial_returns(bt_df)

        print_performance_report(ml_m, fin_m, name=name)

        experiment_results.append({
            "Experiment":       name,
            "Feature_Set":      f_set,
            "Target_Horizon":   t_horizon,
            "Threshold":        thresh,
            "Model":            model_name,
            "Regime_Aware":     regime_aware,
            "Num_Predictors":   len(predictors),
            "Precision":        ml_m.get("Precision", 0),
            "Accuracy":         ml_m.get("Accuracy", 0),
            "Trade_Freq":       ml_m.get("Trade_Frequency", 0),
            "Strategy_Return":  fin_m.get("Strategy_Total_Return", 0),
            "Benchmark_Return": fin_m.get("Benchmark_Total_Return", 0),
            "Strategy_Sharpe":  fin_m.get("Strategy_Sharpe", 0),
            "Strategy_Max_DD":  fin_m.get("Strategy_Max_Drawdown", 0),
        })

    # ── 4. Summary table ───────────────────────────────────────────────────
    log("\n[Step 4/5] Experiment Summary:")
    summary_df = pd.DataFrame(experiment_results)
    cols_show = ["Experiment", "Target_Horizon", "Model", "Regime_Aware",
                 "Precision", "Accuracy", "Trade_Freq",
                 "Strategy_Return", "Strategy_Sharpe"]
    log("\n" + summary_df[cols_show].to_string(index=False))

    # ── 5. Feature importance (best config) ───────────────────────────────
    log("\n[Step 5/5] Feature Importance (Best Config — RF for interpretability):")
    clean_best, preds_best = prepare_dataset(
        sp500_df, macro_df, feature_set="all", target_horizon=5
    )
    feat_imp = compute_average_feature_importance(
        clean_best, preds_best,
        start=2500, step=250,
        model_name="RandomForest",
    )

    log("\n─── TOP 15 PREDICTORS ───")
    for feat, imp in feat_imp.head(15).items():
        bar = "█" * int(imp * 200)
        log(f"  {feat:<30}: {imp:.4f}  {bar}")

    # ── Key takeaways ──────────────────────────────────────────────────────
    log("\n" + "=" * 70)
    log("KEY IMPROVEMENTS OVER ORIGINAL:")
    log("  1. 5-day forward target  → smoother signal, less daily noise")
    log("  2. RSI / MACD / BB / ATR → encodes momentum & mean-reversion")
    log("  3. Ensemble (RF+XGB+LGB) → reduces variance vs single model")
    log("  4. Regime-aware threshold → more selective in high-VIX markets")
    log("  5. Calendar features      → Monday/January/month-end effects")
    log("=" * 70)


if __name__ == "__main__":
    main()
