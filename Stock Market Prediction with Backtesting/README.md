# Predict S&P 500 Price Movement with Random Forest & Walk-Forward Backtesting

A quantitative machine learning system that predicts daily S&P 500 (`^GSPC`) price direction using historical market data, multi-horizon rolling predictors, cross-asset macro indicators, and a rigorous walk-forward backtesting framework designed to eliminate future data leakage.

---

## 🎯 Project Overview & Key Highlights

Most beginner stock prediction projects report deceptively high accuracy (~80-90%), but are **completely flawed** because they leak future market data into training via naive random train/test splits or improper feature engineering.

This project addresses the core challenge of time series quantitative modeling:
- **Zero Future Data Leakage**: Features strictly incorporate `shift(1)` logic and walk-forward expanding window validation.
- **Realistic Quantitative Metrics**: Evaluates models using walk-forward backtests across 30+ years of market data (~1990 to present).
- **Signal Thresholding**: Improves precision and Sharpe ratio by requiring high probability thresholds ($P(\text{Up}) \ge 0.55$ or $0.60$) before executing long positions.
- **Cross-Asset Macro Predictors**: Integrates Volatility (VIX), Tech Momentum (Nasdaq), Interest Rates (10-Yr Treasury Yield), Oil, and Gold.
- **Interactive Visual Dashboard**: Includes a full Streamlit dashboard with Plotly equity curves, feature importances, and live daily market signals.

---

## 🚨 The Data Leakage Trap in Financial Time Series

### Why Naive Train/Test Splits Fail
Standard machine learning techniques like `train_test_split(shuffle=True)` or K-Fold Cross-Validation assume that observations are Independent and Identically Distributed (I.I.D.). 

In financial markets:
$$\text{Price}_{t} \approx \text{Price}_{t-1} + \epsilon_t$$

Because stock prices exhibit strong **temporal autocorrelation**:
1. When you randomly shuffle rows, the training set contains day $t-1$ and day $t+1$, while the test set contains day $t$.
2. The Random Forest easily interpolates day $t$'s price direction because it already saw adjacent future and past dates during training!
3. In live trading, tomorrow's price is unknown. When deployed, naive models fail catastrophically.

### The Walk-Forward Solution
We use an **Expanding Window Walk-Forward Backtest**:

```
Fold 1:  [ Train: 1990-2000 (10 yrs) ] -> [ Predict: 2000-Q1/Q2 (6 mos) ]
Fold 2:  [ Train: 1990-2000.5       ] -> [ Predict: 2000-Q3/Q4 (6 mos) ]
Fold 3:  [ Train: 1990-2001         ] -> [ Predict: 2001-Q1/Q2 (6 mos) ]
...
```

At step $k$, the model is trained **only** on historical data available up to time $t_k$. No future observations are ever exposed to the classifier.

---

## 🧠 Feature Engineering & Signal Drivers

### 1. Multi-Horizon Rolling Ratios
We calculate rolling statistics across $2, 5, 60, 250, 1000$ trading days (~2 days, 1 week, 1 quarter, 1 year, 4 years):
- **`Close_Ratio_{h}`**: $\frac{\text{Close}_t}{\text{Rolling\_Mean}_h(\text{Close})}$. Identifies whether the asset is extended above or below its long-term moving average.
- **`Trend_{h}`**: $\sum_{i=1}^{h} \text{Target}_{t-i}$. Measures past directional momentum. Notice the explicit `shift(1)` to ensure Target at time $t$ (which contains $t+1$ close) is NEVER included!
- **`Volume_Ratio_{h}`**: $\frac{\text{Volume}_t}{\text{Rolling\_Mean}_h(\text{Volume})}$. Highlights volume spikes relative to average liquidity.

### 2. Cross-Asset Macro & Sector Indicators
- **VIX Volatility (`^VIX`)**: 1-day & 5-day percentage shifts in market fear.
- **Nasdaq Spread (`^IXIC`)**: 5-day return difference between tech momentum and broad S&P 500.
- **10-Yr Treasury Yield (`^TNX`)**: 5-day changes in discount rates and monetary policy expectations.
- **Crude Oil (`CL=F`) & Gold (`GC=F`)**: Inflation and safe-haven capital flows.

### 📊 Which Features Actually Move the Needle and Why?
1. **`Close_Ratio_1000` & `Close_Ratio_250`**: Top predictors by feature importance (~15-20% weight). Long-term ratios anchor the model to structural bull/bear regimes.
2. **`VIX_5D_Change`**: High volatility spikes correlate with mean-reversion buying opportunities.
3. **`Trend_250`**: Captures market inertia and multi-month momentum persistence.

---

## ⚡ Decision Threshold Tuning

Because daily stock returns have low signal-to-noise ratio, standard $0.50$ decision thresholds result in frequent, noisy trades.

By tuning the probability threshold $T$:
- $T = 0.50$: Takes trades on any positive edge. High trade frequency, moderate precision (~52-54%).
- $T = 0.55$: Takes trades only when model confidence $\ge 55\%$. Increases precision to ~55-58% while filtering out low-conviction signals.
- $T = 0.60$: Highly selective trading strategy. Dramatically reduces max drawdown and improves Sharpe ratio.

---

## 📁 Repository Structure

```
.
├── src/
│   ├── data_loader.py       # yfinance ingestion, cleaning, local caching
│   ├── features.py          # Anti-leakage rolling & macro feature pipeline
│   ├── backtester.py        # Expanding window walk-forward validation engine
│   ├── metrics.py           # ML precision & financial equity curve calculations
│   └── leakage_demo.py      # Experiment comparing Naive vs Walk-Forward split
├── notebooks/
│   ├── 01_data_and_leakage_pitfalls.py
│   ├── 02_baseline_random_forest.py
│   ├── 03_rolling_predictors_and_thresholding.py
│   └── 04_macro_and_sector_analysis.py
├── tests/
│   └── test_pipeline.py     # Unit tests verifying zero leakage and window logic
├── main.py                  # CLI master execution pipeline script
├── app.py                   # Streamlit interactive web dashboard
├── requirements.txt         # Project dependencies
└── README.md                # Technical documentation & interview guide
```

---

## 🚀 Quick Start & Usage

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Run the Main Pipeline & Experiments
```bash
python main.py
```

### 3. Launch the Interactive Dashboard
```bash
streamlit run app.py
```

### 4. Run Automated Unit Tests
```bash
python tests/test_pipeline.py
```

---

## 💼 Quantitative Finance Interview Guide

When discussing this project in technical interviews, emphasize:
1. **Walk-Forward Validation**: Why K-Fold CV fails on non-stationary, autocorrelated time series data.
2. **Shift Integrity**: How `shift(1)` on targets and features guarantees point-in-time realistic backtesting.
3. **Precision over Accuracy**: Why precision on positive predictions is the primary metric for long-only strategies.
4. **Trade Frequency vs. Confidence Threshold**: How raising decision thresholds filters out market noise.
