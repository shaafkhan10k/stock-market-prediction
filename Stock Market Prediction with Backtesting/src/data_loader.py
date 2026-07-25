import os
import pandas as pd
import yfinance as yf

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

def ensure_data_dir():
    """Ensure local cache directory exists."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)

def fetch_sp500_data(start_date="1990-01-01", force_download=False) -> pd.DataFrame:
    """
    Fetch S&P 500 (^GSPC) historical daily data using yfinance.
    Caches data locally to data/sp500.csv.
    """
    ensure_data_dir()
    filepath = os.path.join(DATA_DIR, "sp500.csv")
    
    if not force_download and os.path.exists(filepath):
        print(f"[DataLoader] Loading cached S&P 500 data from {filepath}...")
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        return df

    print("[DataLoader] Downloading S&P 500 (^GSPC) data from yfinance...")
    df = yf.download("^GSPC", period="max")
    
    # Handle multi-index column headers returned by recent yfinance versions
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    cols_to_keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[cols_to_keep].copy()
    
    if hasattr(df.index, 'tz') and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    df.sort_index(inplace=True)
    df.dropna(subset=["Close"], inplace=True)
    
    if start_date:
        df = df[df.index >= pd.to_datetime(start_date)].copy()
        
    df.to_csv(filepath)
    print(f"[DataLoader] Saved {len(df)} rows of S&P 500 data to {filepath}.")
    return df

def fetch_macro_indicators(start_date="1990-01-01", force_download=False) -> pd.DataFrame:
    """
    Fetch secondary market & macroeconomic indicators for feature enhancement:
    - ^VIX: CBOE Volatility Index
    - ^IXIC: Nasdaq Composite Index (Tech Momentum)
    - ^TNX: 10-Year Treasury Yield (Interest Rates)
    - CL=F: Crude Oil Futures (Inflation / Commodity)
    - GC=F: Gold Futures (Safe Haven Asset)
    """
    ensure_data_dir()
    filepath = os.path.join(DATA_DIR, "macro_indicators.csv")
    
    if not force_download and os.path.exists(filepath):
        print(f"[DataLoader] Loading cached macro data from {filepath}...")
        return pd.read_csv(filepath, index_col=0, parse_dates=True)
        
    print("[DataLoader] Downloading secondary tickers (^VIX, ^IXIC, ^TNX, CL=F, GC=F)...")
    tickers = {
        "VIX": "^VIX",
        "NASDAQ": "^IXIC",
        "TNX_10Y": "^TNX",
        "OIL": "CL=F",
        "GOLD": "GC=F"
    }
    
    macro_dfs = {}
    for name, sym in tickers.items():
        try:
            data = yf.download(sym, period="max", progress=False)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            if hasattr(data.index, 'tz') and data.index.tz is not None:
                data.index = data.index.tz_localize(None)
            data.index = pd.to_datetime(data.index)
            data.dropna(subset=["Close"], inplace=True)
            macro_dfs[f"{name}_Close"] = data["Close"]
        except Exception as e:
            print(f"[DataLoader] Warning: Failed to download {sym}: {e}")
            
    macro_df = pd.DataFrame(macro_dfs)
    macro_df.index.name = "Date"
    macro_df.sort_index(inplace=True)
    
    macro_df = macro_df.ffill().bfill()
    if start_date:
        macro_df = macro_df[macro_df.index >= pd.to_datetime(start_date)].copy()
        
    macro_df.to_csv(filepath)
    print(f"[DataLoader] Saved macro indicators to {filepath}.")
    return macro_df

if __name__ == "__main__":
    sp500 = fetch_sp500_data()
    print("S&P 500 Head:")
    print(sp500.head())
    
    macro = fetch_macro_indicators()
    print("Macro Head:")
    print(macro.head())
