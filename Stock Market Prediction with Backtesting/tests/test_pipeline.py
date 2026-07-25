import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import pandas as pd
import numpy as np

from src.features import create_target, build_baseline_features, build_rolling_features, prepare_dataset
from src.backtester import backtest

class TestPipeline(unittest.TestCase):

    def setUp(self):
        dates = pd.date_range(start="2020-01-01", periods=3000, freq="D")
        np.random.seed(42)
        
        returns = np.random.normal(loc=0.0005, scale=0.01, size=len(dates))
        price = 100 * np.cumprod(1 + returns)
        volume = np.random.randint(1000, 5000, size=len(dates))
        
        self.df = pd.DataFrame({
            "Open": price * 0.99,
            "High": price * 1.01,
            "Low": price * 0.98,
            "Close": price,
            "Volume": volume
        }, index=dates)

    def test_target_creation(self):
        df_target = create_target(self.df)
        self.assertIn("Tomorrow", df_target.columns)
        self.assertIn("Target", df_target.columns)
        
        first_target = df_target.iloc[0]["Target"]
        expected_target = int(self.df.iloc[1]["Close"] > self.df.iloc[0]["Close"])
        self.assertEqual(first_target, expected_target)

    def test_no_future_leakage_in_rolling_trend(self):
        """
        Verify that Trend_h at index t relies ONLY on Target at t-1 and prior,
        NOT Target at index t (which contains Close at t+1).
        """
        df_target = create_target(self.df)
        df_rolling, features = build_rolling_features(df_target, horizons=[5])
        
        self.assertIn("Trend_5", df_rolling.columns)
        
        t_idx = 10
        actual_trend = df_rolling.iloc[t_idx]["Trend_5"]
        expected_trend = df_target["Target"].iloc[t_idx-5:t_idx].sum()
        
        self.assertEqual(actual_trend, expected_trend)

    def test_prepare_dataset_no_nans(self):
        clean_df, predictors = prepare_dataset(self.df, feature_set="rolling", horizons=[2, 5, 60, 250])
        self.assertFalse(clean_df[predictors].isnull().any().any())
        self.assertFalse(clean_df["Target"].isnull().any())
        self.assertTrue(len(clean_df) > 0)

    def test_backtest_expanding_window_indices(self):
        clean_df, predictors = prepare_dataset(self.df, feature_set="baseline")
        start = 2000
        step = 200
        results = backtest(clean_df, predictors, start=start, step=step)
        
        self.assertFalse(results.empty)
        self.assertEqual(len(results), len(clean_df) - start)

if __name__ == "__main__":
    unittest.main()
