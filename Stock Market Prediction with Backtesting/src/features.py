import pandas as pd
import numpy as np


# ─────────────────────────────────────────────
#  TARGET CONSTRUCTION
# ─────────────────────────────────────────────

def create_target(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """
    Binary target: 1 if Close N days from now > today's Close, else 0.

    horizon=1  → original next-day target  (high noise, ~50% base rate)
    horizon=5  → 5-day forward target      (smoother signal, recommended)
    """
    df = df.copy()
    df["Tomorrow"] = df["Close"].shift(-horizon)
    df["Target"] = (df["Tomorrow"] > df["Close"]).astype(int)
    return df


# ─────────────────────────────────────────────
#  BASELINE FEATURES
# ─────────────────────────────────────────────

def build_baseline_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Basic price/volume ratio features."""
    df = df.copy()
    features = []

    df["Close_Pct_Change"]  = df["Close"].pct_change()
    df["Volume_Pct_Change"] = df["Volume"].pct_change()
    df["High_Low_Ratio"]    = df["High"] / df["Low"]
    df["Close_Open_Ratio"]  = df["Close"] / df["Open"]

    features.extend(["Close_Pct_Change", "Volume_Pct_Change",
                      "High_Low_Ratio", "Close_Open_Ratio"])
    return df, features


# ─────────────────────────────────────────────
#  ROLLING FEATURES
# ─────────────────────────────────────────────

def build_rolling_features(
    df: pd.DataFrame,
    horizons: list = [2, 5, 60, 250, 1000]
) -> tuple[pd.DataFrame, list[str]]:
    """
    Multi-horizon rolling features.
    Uses shift(1) on Target to prevent leakage.
    """
    df = df.copy()
    new_features = []

    if "Target" not in df.columns:
        df = create_target(df)

    for h in horizons:
        rolling_close = df["Close"].rolling(h).mean()
        col_close = f"Close_Ratio_{h}"
        df[col_close] = df["Close"] / rolling_close
        new_features.append(col_close)

        rolling_vol = df["Volume"].rolling(h).mean()
        col_vol = f"Volume_Ratio_{h}"
        df[col_vol] = df["Volume"] / rolling_vol.replace(0, np.nan)
        new_features.append(col_vol)

        # shift(1) — Target at t contains t+1 close, so must lag by 1
        col_trend = f"Trend_{h}"
        df[col_trend] = df["Target"].shift(1).rolling(h).sum()
        new_features.append(col_trend)

    return df, new_features


# ─────────────────────────────────────────────
#  TECHNICAL INDICATOR FEATURES  ← NEW
# ─────────────────────────────────────────────

def build_technical_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Classic TA indicators: RSI, MACD, Bollinger Bands, ATR, Stochastic.
    All computed strictly from past data — no leakage.
    """
    df = df.copy()
    features = []

    # ── RSI-14 ──────────────────────────────
    delta = df["Close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    df["RSI_14"] = 100 - (100 / (1 + rs))
    df["RSI_Overbought"] = (df["RSI_14"] > 70).astype(int)
    df["RSI_Oversold"]   = (df["RSI_14"] < 30).astype(int)
    features.extend(["RSI_14", "RSI_Overbought", "RSI_Oversold"])

    # ── MACD (12/26/9) ──────────────────────
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"]        = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"]   = df["MACD"] - df["MACD_Signal"]
    # Crossover signal: histogram flipped positive this bar
    df["MACD_Crossover"] = (
        (df["MACD_Hist"] > 0) & (df["MACD_Hist"].shift(1) <= 0)
    ).astype(int)
    features.extend(["MACD", "MACD_Hist", "MACD_Crossover"])

    # ── Bollinger Bands (20, 2σ) ────────────
    sma20  = df["Close"].rolling(20).mean()
    std20  = df["Close"].rolling(20).std()
    df["BB_Upper"]    = sma20 + 2 * std20
    df["BB_Lower"]    = sma20 - 2 * std20
    df["BB_Width"]    = (df["BB_Upper"] - df["BB_Lower"]) / sma20   # volatility proxy
    df["BB_Position"] = (df["Close"] - sma20) / (2 * std20)         # -1 to +1 normalised
    features.extend(["BB_Width", "BB_Position"])

    # ── ATR-14 (True Range volatility) ──────
    high_low   = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift(1)).abs()
    low_close  = (df["Low"]  - df["Close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR_14"] = tr.rolling(14).mean()
    df["ATR_Pct"] = df["ATR_14"] / df["Close"]   # normalised ATR
    features.extend(["ATR_14", "ATR_Pct"])

    # ── Stochastic %K (14) ──────────────────
    low14  = df["Low"].rolling(14).min()
    high14 = df["High"].rolling(14).max()
    df["Stoch_K"] = 100 * (df["Close"] - low14) / (high14 - low14).replace(0, np.nan)
    df["Stoch_D"] = df["Stoch_K"].rolling(3).mean()
    features.extend(["Stoch_K", "Stoch_D"])

    # ── OBV (On-Balance Volume) ─────────────
    obv = (np.sign(df["Close"].diff()) * df["Volume"]).fillna(0).cumsum()
    df["OBV_Ratio"] = obv / obv.rolling(20).mean().replace(0, np.nan)
    features.append("OBV_Ratio")

    return df, features


# ─────────────────────────────────────────────
#  CALENDAR / REGIME FEATURES  ← NEW
# ─────────────────────────────────────────────

def build_calendar_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Day-of-week, month-of-year, and month-end effects.
    Statistically significant in S&P 500 literature.
    """
    df = df.copy()
    features = []

    df["DayOfWeek"]   = df.index.dayofweek          # 0=Mon … 4=Fri
    df["Month"]        = df.index.month
    df["IsMonthEnd"]   = df.index.is_month_end.astype(int)
    df["IsMonthStart"] = df.index.is_month_start.astype(int)
    df["Quarter"]      = df.index.quarter

    features.extend(["DayOfWeek", "Month", "IsMonthEnd", "IsMonthStart", "Quarter"])
    return df, features


# ─────────────────────────────────────────────
#  MACRO FEATURES  (unchanged, kept compatible)
# ─────────────────────────────────────────────

def build_macro_features(
    sp500_df: pd.DataFrame,
    macro_df: pd.DataFrame
) -> tuple[pd.DataFrame, list[str]]:
    df = df_out = sp500_df.copy()
    macro_features = []

    if macro_df is None or macro_df.empty:
        return df, macro_features

    df = df.join(macro_df, how="left")
    df = df.ffill().bfill()

    if "VIX_Close" in df.columns:
        df["VIX_Level"]    = df["VIX_Close"]
        df["VIX_1D_Change"] = df["VIX_Close"].pct_change(1)
        df["VIX_5D_Change"] = df["VIX_Close"].pct_change(5)
        # VIX regime flag (high fear = > 20)
        df["VIX_HighFear"] = (df["VIX_Close"] > 20).astype(int)
        df["VIX_Spike"]    = (df["VIX_Close"] > 30).astype(int)
        macro_features.extend(["VIX_Level", "VIX_1D_Change", "VIX_5D_Change",
                                "VIX_HighFear", "VIX_Spike"])

    if "NASDAQ_Close" in df.columns:
        nasdaq_ret5 = df["NASDAQ_Close"].pct_change(5)
        sp500_ret5  = df["Close"].pct_change(5)
        df["Nasdaq_SP500_Spread_5D"] = nasdaq_ret5 - sp500_ret5
        macro_features.append("Nasdaq_SP500_Spread_5D")

    if "TNX_10Y_Close" in df.columns:
        df["TNX_5D_Diff"]  = df["TNX_10Y_Close"].diff(5)
        df["TNX_20D_Diff"] = df["TNX_10Y_Close"].diff(20)
        macro_features.extend(["TNX_5D_Diff", "TNX_20D_Diff"])

    if "OIL_Close" in df.columns:
        df["Oil_5D_Return"] = df["OIL_Close"].pct_change(5)
        macro_features.append("Oil_5D_Return")

    if "GOLD_Close" in df.columns:
        df["Gold_5D_Return"] = df["GOLD_Close"].pct_change(5)
        macro_features.append("Gold_5D_Return")

    return df, macro_features


# ─────────────────────────────────────────────
#  MASTER PIPELINE
# ─────────────────────────────────────────────

def prepare_dataset(
    sp500_df: pd.DataFrame,
    macro_df: pd.DataFrame = None,
    feature_set: str = "all",
    horizons: list = [2, 5, 60, 250, 1000],
    target_horizon: int = 5,          # ← NEW: 5-day target by default
) -> tuple[pd.DataFrame, list[str]]:
    """
    Master feature engineering pipeline.

    feature_set options:
      'baseline' : price/volume ratios only
      'rolling'  : baseline + multi-horizon rolling
      'all'      : rolling + technical indicators + calendar + macro  ← recommended

    target_horizon:
      1  → predict tomorrow's direction  (original, noisy)
      5  → predict 5-day direction       (smoother, higher accuracy)
    """
    df = create_target(sp500_df, horizon=target_horizon)
    predictor_cols = []

    # 1. Baseline
    df, base_cols = build_baseline_features(df)
    predictor_cols.extend(base_cols)

    # 2. Rolling
    if feature_set in ["rolling", "all"]:
        df, rolling_cols = build_rolling_features(df, horizons=horizons)
        predictor_cols.extend(rolling_cols)

    # 3. Technical indicators  ← NEW
    if feature_set == "all":
        df, tech_cols = build_technical_features(df)
        predictor_cols.extend(tech_cols)

    # 4. Calendar features  ← NEW
    if feature_set == "all":
        df, cal_cols = build_calendar_features(df)
        predictor_cols.extend(cal_cols)

    # 5. Macro features
    if feature_set == "all" and macro_df is not None:
        df, macro_cols = build_macro_features(df, macro_df)
        predictor_cols.extend(macro_cols)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_clean = df.dropna(subset=predictor_cols + ["Target"]).copy()
    df_clean["Target"] = df_clean["Target"].astype(int)

    return df_clean, predictor_cols


if __name__ == "__main__":
    from src.data_loader import fetch_sp500_data, fetch_macro_indicators
    sp500 = fetch_sp500_data()
    macro = fetch_macro_indicators()

    clean_df, features = prepare_dataset(sp500, macro, feature_set="all", target_horizon=5)
    print(f"Dataset shape: {clean_df.shape}")
    print(f"Predictors ({len(features)}): {features}")
    print(clean_df[features + ["Target"]].tail(3))
