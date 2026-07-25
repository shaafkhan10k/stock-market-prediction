import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, accuracy_score

from src.data_loader import fetch_sp500_data
from src.features import prepare_dataset
from src.backtester import default_model, backtest
from src.metrics import compute_ml_metrics

def run_naive_random_split(df: pd.DataFrame, predictors: list[str]) -> dict:
    """
    Demonstrates the NAIVE approach:
    Randomly shuffling time series data (train_test_split with shuffle=True).
    This leaks future market context into the past, producing fake high metrics.
    """
    X = df[predictors]
    y = df["Target"]
    
    # Flawed: Random shuffle breaks temporal ordering!
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, random_state=42
    )
    
    model = default_model()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    return {
        "Accuracy": float(accuracy_score(y_test, y_pred)),
        "Precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "Split_Type": "Naive Random Shuffle (FLAWED / LEAKAGE)"
    }

def run_proper_temporal_split(df: pd.DataFrame, predictors: list[str]) -> dict:
    """
    Demonstrates PROPER walk-forward temporal backtest:
    No future data is ever seen by the model.
    """
    results = backtest(df, predictors, start=2500, step=125, custom_threshold=0.5)
    ml_metrics = compute_ml_metrics(results)
    
    return {
        "Accuracy": ml_metrics["Accuracy"],
        "Precision": ml_metrics["Precision"],
        "Split_Type": "Walk-Forward Temporal (RIGOROUS / REALISTIC)"
    }

def compare_leakage_impact():
    """
    Runs both naive and proper validation methods and prints the detailed comparison.
    """
    print("=" * 70)
    print("      DATA LEAKAGE DEMONSTRATION: NAIVE vs WALK-FORWARD VALIDATION")
    print("=" * 70)
    
    sp500 = fetch_sp500_data()
    df, predictors = prepare_dataset(sp500, feature_set="rolling")
    
    naive = run_naive_random_split(df, predictors)
    proper = run_proper_temporal_split(df, predictors)
    
    print("\n--- RESULTS COMPARISON ---")
    print(f"1. {naive['Split_Type']}:")
    print(f"   - Accuracy:  {naive['Accuracy']:.4f} ({naive['Accuracy']*100:.1f}%)")
    print(f"   - Precision: {naive['Precision']:.4f} ({naive['Precision']*100:.1f}%)")
    print("\n2. {proper['Split_Type']}:")
    print(f"   - Accuracy:  {proper['Accuracy']:.4f} ({proper['Accuracy']*100:.1f}%)")
    print(f"   - Precision: {proper['Precision']:.4f} ({proper['Precision']*100:.1f}%)")
    print("\n" + "-" * 70)
    print("WHY DO NAIVE SPLITS FAIL IN TIME SERIES?")
    print("Financial data exhibits strong temporal autocorrelation. When you randomly shuffle")
    print("rows, the training set contains day t-1 and day t+1, while the test set contains day t.")
    print("The Random Forest simply learns to interpolate between future and past days that it")
    print("already saw during training! In live trading, tomorrow's price is unknown, causing")
    print("naive models to fail completely when deployed.")
    print("=" * 70)

if __name__ == "__main__":
    compare_leakage_impact()
