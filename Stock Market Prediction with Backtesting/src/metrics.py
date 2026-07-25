import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, accuracy_score, recall_score, confusion_matrix


def compute_ml_metrics(results: pd.DataFrame) -> dict:
    """
    Calculate classification performance metrics from backtest results.
    """
    if results.empty or "Predictions" not in results.columns:
        return {}

    y_true = results["Target"]
    y_pred = results["Predictions"]

    precision = precision_score(y_true, y_pred, zero_division=0)
    accuracy  = accuracy_score(y_true, y_pred)
    recall    = recall_score(y_true, y_pred, zero_division=0)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    total_days     = len(results)
    trades_taken   = int(y_pred.sum())
    trade_frequency = trades_taken / total_days if total_days > 0 else 0

    return {
        "Precision":       float(precision),
        "Accuracy":        float(accuracy),
        "Recall":          float(recall),
        "Total_Days":      total_days,
        "Trades_Taken":    trades_taken,
        "Trade_Frequency": float(trade_frequency),
        "True_Positives":  int(tp),
        "False_Positives": int(fp),
        "True_Negatives":  int(tn),
        "False_Negatives": int(fn),
    }


def compute_financial_returns(
    results: pd.DataFrame,
    transaction_cost_bps: int = 0,
) -> tuple:
    """
    Simulate financial performance of the model strategy vs S&P 500 Buy & Hold.

    Parameters
    ----------
    results : DataFrame with columns Target, Predictions, Probabilities, Close
    transaction_cost_bps : round-trip cost in basis points deducted on every
                           trade entry (when Predictions flips from 0→1 or 1→0).

    Returns
    -------
    (equity_df, summary_metrics_dict)
    """
    if results.empty or "Close" not in results.columns:
        return pd.DataFrame(), {}

    df = results.copy()

    # ── daily returns ────────────────────────────────────────────────────────
    df["Benchmark_Daily_Return"] = df["Close"].pct_change().fillna(0)

    # Hold when yesterday's prediction was 1
    df["Strategy_Daily_Return"] = (
        df["Predictions"].shift(1).fillna(0) * df["Benchmark_Daily_Return"]
    )

    # ── transaction costs ────────────────────────────────────────────────────
    if transaction_cost_bps > 0:
        cost_per_trade = transaction_cost_bps / 10_000
        # A "trade" happens whenever the signal changes (entry or exit)
        signal_change = df["Predictions"].diff().abs().fillna(0).astype(bool)
        df.loc[signal_change, "Strategy_Daily_Return"] -= cost_per_trade

    # ── equity curves ────────────────────────────────────────────────────────
    df["Benchmark_Equity"] = (1 + df["Benchmark_Daily_Return"]).cumprod()
    df["Strategy_Equity"]  = (1 + df["Strategy_Daily_Return"]).cumprod()

    # ── drawdown series (stored for charting) ────────────────────────────────
    bench_peak = df["Benchmark_Equity"].cummax()
    strat_peak = df["Strategy_Equity"].cummax()

    df["Benchmark_Drawdown"] = (df["Benchmark_Equity"] - bench_peak) / bench_peak
    df["Strategy_Drawdown"]  = (df["Strategy_Equity"]  - strat_peak) / strat_peak

    bench_max_dd = float(df["Benchmark_Drawdown"].min())
    strat_max_dd = float(df["Strategy_Drawdown"].min())

    # ── CAGR & Sharpe ────────────────────────────────────────────────────────
    num_years = max(len(df) / 252.0, 0.01)

    bench_cagr = df["Benchmark_Equity"].iloc[-1] ** (1.0 / num_years) - 1
    strat_cagr = df["Strategy_Equity"].iloc[-1]  ** (1.0 / num_years) - 1

    bench_vol = df["Benchmark_Daily_Return"].std() * np.sqrt(252)
    strat_vol = df["Strategy_Daily_Return"].std()  * np.sqrt(252)

    bench_sharpe = bench_cagr / bench_vol if bench_vol > 0 else 0
    strat_sharpe = strat_cagr / strat_vol if strat_vol > 0 else 0

    # ── Calmar Ratio ─────────────────────────────────────────────────────────
    strat_calmar = strat_cagr / abs(strat_max_dd) if strat_max_dd != 0 else 0

    # ── Win rate ─────────────────────────────────────────────────────────────
    traded_days = df[df["Predictions"].shift(1) == 1]
    win_rate = (
        (traded_days["Benchmark_Daily_Return"] > 0).mean()
        if len(traded_days) > 0
        else 0
    )

    summary = {
        "Benchmark_Total_Return":  float(df["Benchmark_Equity"].iloc[-1] - 1),
        "Strategy_Total_Return":   float(df["Strategy_Equity"].iloc[-1]  - 1),
        "Benchmark_CAGR":          float(bench_cagr),
        "Strategy_CAGR":           float(strat_cagr),
        "Benchmark_Volatility":    float(bench_vol),
        "Strategy_Volatility":     float(strat_vol),
        "Benchmark_Sharpe":        float(bench_sharpe),
        "Strategy_Sharpe":         float(strat_sharpe),
        "Benchmark_Max_Drawdown":  bench_max_dd,
        "Strategy_Max_Drawdown":   strat_max_dd,
        "Strategy_Calmar":         float(strat_calmar),
        "Strategy_Win_Rate":       float(win_rate),
    }

    return df, summary


def print_performance_report(
    ml_metrics: dict, financial_metrics: dict, name: str = "Strategy"
) -> None:
    """Format and print quantitative evaluation summary."""
    print("=" * 60)
    print(f"        PERFORMANCE REPORT: {name.upper()}")
    print("=" * 60)
    print("CLASSIFICATION METRICS:")
    print(f"  Precision (Target=1):  {ml_metrics.get('Precision', 0):.4f}")
    print(f"  Accuracy:             {ml_metrics.get('Accuracy', 0):.4f}")
    print(f"  Recall:               {ml_metrics.get('Recall', 0):.4f}")
    print(
        f"  Trade Frequency:      {ml_metrics.get('Trade_Frequency', 0):.2%} "
        f"({ml_metrics.get('Trades_Taken', 0)} / {ml_metrics.get('Total_Days', 0)} days)"
    )
    print("-" * 60)
    print("FINANCIAL PERFORMANCE METRICS:")
    print(f"  Strategy Total Return: {financial_metrics.get('Strategy_Total_Return', 0):.2%}")
    print(f"  S&P 500 Total Return:  {financial_metrics.get('Benchmark_Total_Return', 0):.2%}")
    print(f"  Strategy CAGR:         {financial_metrics.get('Strategy_CAGR', 0):.2%}")
    print(f"  S&P 500 CAGR:          {financial_metrics.get('Benchmark_CAGR', 0):.2%}")
    print(f"  Strategy Sharpe Ratio: {financial_metrics.get('Strategy_Sharpe', 0):.2f}")
    print(f"  S&P 500 Sharpe Ratio:  {financial_metrics.get('Benchmark_Sharpe', 0):.2f}")
    print(f"  Strategy Calmar Ratio: {financial_metrics.get('Strategy_Calmar', 0):.2f}")
    print(f"  Strategy Max Drawdown: {financial_metrics.get('Strategy_Max_Drawdown', 0):.2%}")
    print(f"  S&P 500 Max Drawdown:  {financial_metrics.get('Benchmark_Max_Drawdown', 0):.2%}")
    print(f"  Strategy Win Rate:     {financial_metrics.get('Strategy_Win_Rate', 0):.2%}")
    print("=" * 60)