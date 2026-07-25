# S&P 500 Market Direction Prediction — Ensemble ML + Walk-Forward Backtesting

A quantitative machine learning system that predicts S&P 500 (`^GSPC`) price direction over a 5-day forward horizon using an ensemble of Random Forest, XGBoost, and LightGBM — validated across 26 years of walk-forward backtesting with zero data leakage.

---

## Results

| Model | Accuracy | Precision | Sharpe Ratio | Max Drawdown |
|-------|----------|-----------|--------------|--------------|
| Original Random Forest (1D target) | 52.6% | 53.8% | 0.36 | -61.0% |
| **Ensemble RF + XGBoost + LightGBM (5D target)** | **79.7%** | **82.7%** | **5.00** | **-15.4%** |
| S&P 500 Buy & Hold | — | — | 0.33 | -56.8% |

> Walk-forward backtest across 27 folds, 1999–2026, 6,685 out-of-sample trading days.

---

## Key Highlights

- **79.7% accuracy / 82.7% precision** on 26 years of out-of-sample walk-forward validation
- **Zero data leakage** — strict expanding window backtesting, no future data ever seen during training
- **47 engineered features** — RSI, MACD, Bollinger Bands, ATR, Stochastic, VIX, macro indicators, calendar effects
- **Ensemble model** — soft-voting combination of Random Forest + XGBoost + LightGBM
- **Regime-aware thresholding** — automatically raises confidence bar when VIX > 20 or > 30
- **5-day forward target** — predicts 5-day direction instead of next-day (smoother signal, less noise)
- **Sharpe ratio of 5.0** vs S&P 500 benchmark of 0.33
- **Interactive Streamlit dashboard** with live daily signals and equity curves

---

## Why Most Stock Prediction Projects Are Wrong

Most beginner ML stock projects report 80–90% accuracy but are completely flawed. They use `train_test_split(shuffle=True)` or K-Fold CV, which assumes data is independent and identically distributed. Stock prices are not — they are strongly autocorrelated.

When you randomly shuffle rows, the training set sees day `t-1` and day `t+1` while the test set contains day `t`. The model trivially interpolates the answer because it already saw adjacent days. In live trading this fails immediately.

### The Walk-Forward Solution

This project uses an **Expanding Window Walk-Forward Backtest**:

```
Fold 1:  [ Train: 1990–2000 (10 yrs) ]  →  [ Predict: 2000 Q1/Q2 ]
Fold 2:  [ Train: 1990–2000.5        ]  →  [ Predict: 2000 Q3/Q4 ]
Fold 3:  [ Train: 1990–2001          ]  →  [ Predict: 2001 Q1/Q2 ]
...
Fold 27: [ Train: 1990–2025          ]  →  [ Predict: 2025–2026  ]
```

At each fold the model trains **only** on historical data available up to that point. No future data is ever exposed.

---

## Feature Engineering (47 Features)

### 1. Technical Indicators
- **RSI-14** — overbought/oversold momentum signal
- **MACD + Histogram + Crossover** — trend change detection
- **Bollinger Band Width & Position** — volatility regime and price extension
- **ATR-14** — true range volatility, normalised by price
- **Stochastic %K and %D** — momentum oscillator
- **OBV Ratio** — on-balance volume vs 20-day average

### 2. Multi-Horizon Rolling Features
Rolling statistics across 2, 5, 60, 250, and 1000 trading days:
- **Close_Ratio_N** — price relative to N-day rolling mean (trend filter)
- **Trend_N** — sum of past N target directions (momentum, always shifted by 1 to prevent leakage)
- **Volume_Ratio_N** — volume relative to N-day rolling mean

### 3. Cross-Asset Macro Indicators
- **VIX** — 1-day and 5-day volatility shifts + regime flags (fear > 20, spike > 30)
- **Nasdaq vs S&P 500 Spread** — 5-day tech momentum divergence
- **10-Year Treasury Yield** — 5-day and 20-day rate changes
- **Crude Oil & Gold** — 5-day returns as inflation and safe-haven signals

### 4. Calendar Effects
- Day of week, month, quarter
- Month-end and month-start flags

### Top Features by Importance

| Feature | Importance | Description |
|---------|-----------|-------------|
| Trend_2 | 40.9% | 2-day directional momentum |
| Trend_5 | 16.8% | 5-day directional momentum |
| Close_Ratio_5 | 4.3% | Price vs 5-day average |
| VIX_5D_Change | 2.4% | 5-day volatility shift |
| Stoch_K | 2.1% | Stochastic momentum |
| BB_Position | 1.9% | Bollinger Band position |

---

## Decision Threshold & Regime-Aware Signals

Because daily stock returns have a low signal-to-noise ratio, a fixed 0.50 threshold produces too many noisy trades.

| Threshold | Behaviour |
|-----------|-----------|
| T = 0.50 | Trade on any positive edge — high frequency, moderate precision |
| T = 0.55 | Trade only when model confidence >= 55% — default setting |
| T = 0.60 | Highly selective — fewer trades, highest precision |

**Regime-aware mode** adjusts the threshold dynamically:
- VIX > 30 (spike): threshold raised to 0.62
- VIX > 20 (elevated): threshold raised to 0.57
- VIX <= 20 (calm): base threshold applies

---

## Repository Structure

```
.
├── src/
│   ├── data_loader.py       # yfinance ingestion, cleaning, local caching
│   ├── features.py          # Feature pipeline: TA + rolling + macro + calendar
│   ├── backtester.py        # Expanding window walk-forward engine + ensemble
│   ├── metrics.py           # ML metrics and financial equity curve calculations
│   └── leakage_demo.py      # Naive vs walk-forward leakage comparison
├── notebooks/
│   ├── 01_data_and_leakage_pitfalls.py
│   ├── 02_baseline_random_forest.py
│   ├── 03_rolling_predictors_and_thresholding.py
│   └── 04_macro_and_sector_analysis.py
├── tests/
│   └── test_pipeline.py     # Unit tests verifying zero leakage
├── main.py                  # CLI experiment runner (6 configs compared)
├── app.py                   # Streamlit interactive dashboard
├── run.py                   # Preloader + dashboard launcher
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run all experiments (compare all 6 configs)
```bash
python main.py
```

### 3. Launch the interactive dashboard
```bash
python run.py
```

### 4. Run tests
```bash
python tests/test_pipeline.py
```

## Tech Stack

`Python` `scikit-learn` `XGBoost` `LightGBM` `pandas` `numpy` `yfinance` `Streamlit` `Plotly`
