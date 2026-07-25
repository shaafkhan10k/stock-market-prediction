"""
Preloader + Launcher
--------------------
Runs once to pre-train and cache the best model config, then launches
the Streamlit dashboard.

Usage:
    python run.py

To force a full retrain (e.g. after code changes):
    python run.py --retrain
"""

import os
import sys
import subprocess
import numpy as np
import pandas as pd

from src.data_loader import fetch_sp500_data, fetch_macro_indicators, DATA_DIR
from src.features import prepare_dataset
from src.backtester import get_model

# ── Config ────────────────────────────────────────────────────────────────
FEATURE_SET    = "all"
TARGET_HORIZON = 5       # 5-day forward target (improved)
MODEL_NAME     = "Ensemble"
START_WINDOW   = 2500
STEP_DAYS      = 250
BASE_THRESHOLD = 0.55

CACHE_FILE = os.path.join(DATA_DIR, f"precomputed_{FEATURE_SET}_{START_WINDOW}_{STEP_DAYS}_5d.csv")
IMP_FILE   = os.path.join(DATA_DIR, f"precomputed_imp_{FEATURE_SET}_{START_WINDOW}_{STEP_DAYS}_5d.csv")


def preload(force: bool = False):
    if not force and os.path.exists(CACHE_FILE) and os.path.exists(IMP_FILE):
        print("[OK] Cached results found — skipping preload.")
        print("     Pass --retrain to force a full retrain.")
        return

    print("=" * 60)
    print("  PRELOADING: Downloading data & training ensemble...")
    print("  (This only runs once. Future launches are instant!)")
    print("=" * 60)

    print("\n[1/3] Fetching S&P 500 & macro data...")
    sp500_df = fetch_sp500_data()
    macro_df = fetch_macro_indicators()
    print(f"      Loaded {len(sp500_df)} trading days.")

    print("\n[2/3] Engineering features (TA + rolling + macro + calendar)...")
    clean_df, predictors = prepare_dataset(
        sp500_df, macro_df,
        feature_set=FEATURE_SET,
        target_horizon=TARGET_HORIZON,
    )
    print(f"      {len(predictors)} predictors across {len(clean_df)} rows.")

    print(f"\n[3/3] Walk-forward backtest (~{(len(clean_df)-START_WINDOW)//STEP_DAYS} folds)...")
    all_preds  = []
    rf_imps    = []   # collect RF importances for interpretability
    total      = len(range(START_WINDOW, clean_df.shape[0], STEP_DAYS))

    for fold, i in enumerate(range(START_WINDOW, clean_df.shape[0], STEP_DAYS), start=1):
        train = clean_df.iloc[0:i].copy()
        test  = clean_df.iloc[i:(i + STEP_DAYS)].copy()
        if test.empty:
            break

        # Full ensemble for predictions
        model = get_model(MODEL_NAME)
        model.fit(train[predictors], train["Target"])
        probs = model.predict_proba(test[predictors])[:, 1]

        row = pd.DataFrame({
            "Target":        test["Target"],
            "Probabilities": probs,
            "Close":         test["Close"],
        }, index=test.index)

        if "VIX_Level" in test.columns:
            row["VIX_Level"] = test["VIX_Level"].values

        all_preds.append(row)

        # RF-only for importances
        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier(n_estimators=100, min_samples_split=40,
                                    random_state=42, n_jobs=-1)
        rf.fit(train[predictors], train["Target"])
        rf_imps.append(rf.feature_importances_)

        pct    = fold / total * 100
        filled = "#" * int(pct / 5)
        empty  = "-" * (20 - int(pct / 5))
        print(f"      [{filled}{empty}] {pct:5.1f}%  fold {fold}/{total}", end="\r")

    print()

    results  = pd.concat(all_preds)
    feat_imp = pd.Series(np.mean(rf_imps, axis=0),
                         index=predictors).sort_values(ascending=False)

    results.to_csv(CACHE_FILE)
    pd.DataFrame({"Importance": feat_imp}).to_csv(IMP_FILE)

    print(f"\n[DONE] Cached to:")
    print(f"   {CACHE_FILE}")
    print(f"   {IMP_FILE}")


if __name__ == "__main__":
    force_retrain = "--retrain" in sys.argv
    preload(force=force_retrain)

    print("\n" + "=" * 60)
    print("  Launching Streamlit dashboard...")
    print("  Opening: http://localhost:8501")
    print("=" * 60 + "\n")

    subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
