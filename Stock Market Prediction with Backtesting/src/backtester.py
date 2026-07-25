import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False


# ─────────────────────────────────────────────
#  MODEL FACTORY
# ─────────────────────────────────────────────

def get_model(model_name: str = "Ensemble"):
    """
    Return a fresh, untrained classifier by name.

    Options:
      'RandomForest'  – baseline tree ensemble
      'XGBoost'       – gradient boosting (requires xgboost)
      'LightGBM'      – fast gradient boosting (requires lightgbm)
      'Ensemble'      – soft-voting RF + XGB + LGB  ← recommended
    """
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_split=40,
        min_samples_leaf=20,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )

    if model_name == "RandomForest":
        return rf

    if model_name == "XGBoost":
        if not HAS_XGB:
            raise ImportError("xgboost not installed. Run: pip install xgboost")
        return xgb.XGBClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.01,
            subsample=0.8,
            colsample_bytree=0.7,
            min_child_weight=10,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )

    if model_name == "LightGBM":
        if not HAS_LGB:
            raise ImportError("lightgbm not installed. Run: pip install lightgbm")
        return lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.01,
            subsample=0.8,
            colsample_bytree=0.7,
            min_child_samples=20,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )

    # ── Ensemble (default) ────────────────────────────────────────────
    estimators = [("rf", rf)]

    if HAS_XGB:
        estimators.append(("xgb", xgb.XGBClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.01,
            subsample=0.8, colsample_bytree=0.7, min_child_weight=10,
            eval_metric="logloss", random_state=42, n_jobs=-1, verbosity=0,
        )))

    if HAS_LGB:
        estimators.append(("lgb", lgb.LGBMClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.01,
            subsample=0.8, colsample_bytree=0.7, min_child_samples=20,
            random_state=42, n_jobs=-1, verbose=-1,
        )))

    if len(estimators) == 1:
        # neither XGB nor LGB installed — fall back to RF alone
        print("[Backtester] Warning: xgboost/lightgbm not found. Using RandomForest only.")
        return rf

    return VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1)


def default_model(n_estimators=200, min_samples_split=40, random_state=42):
    """Backward-compatible helper (RandomForest only)."""
    return RandomForestClassifier(
        n_estimators=n_estimators,
        min_samples_split=min_samples_split,
        random_state=random_state,
        n_jobs=-1,
    )


# ─────────────────────────────────────────────
#  REGIME-AWARE THRESHOLD  ← NEW
# ─────────────────────────────────────────────

def get_regime_threshold(row: pd.Series, base_threshold: float = 0.55) -> float:
    """
    Adjust the decision threshold based on VIX market regime.
    High-fear markets are noisier → require higher model confidence.

    Falls back to base_threshold if VIX not present in row.
    """
    vix = row.get("VIX_Level", None)
    if vix is None or pd.isna(vix):
        return base_threshold
    if vix > 30:
        return max(base_threshold, 0.62)   # spike — be very selective
    if vix > 20:
        return max(base_threshold, 0.57)   # elevated fear
    return base_threshold                   # calm market


# ─────────────────────────────────────────────
#  PREDICT (single fold)
# ─────────────────────────────────────────────

def predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    predictors: list,
    model=None,
    custom_threshold: float = 0.55,
    regime_aware: bool = True,
) -> pd.DataFrame:
    """
    Fit model on train, predict probabilities on test.
    Applies regime-aware thresholding if enabled and VIX_Level is a feature.
    """
    if model is None:
        model = get_model("Ensemble")

    model.fit(train[predictors], train["Target"])
    probs = model.predict_proba(test[predictors])[:, 1]

    result = pd.DataFrame({
        "Target":        test["Target"],
        "Probabilities": probs,
        "Close":         test["Close"],
    }, index=test.index)

    # Copy VIX if available for regime thresholding
    if "VIX_Level" in test.columns:
        result["VIX_Level"] = test["VIX_Level"].values

    if regime_aware and "VIX_Level" in result.columns:
        result["Predictions"] = result.apply(
            lambda r: 1 if r["Probabilities"] >= get_regime_threshold(r, custom_threshold) else 0,
            axis=1,
        )
    else:
        result["Predictions"] = (probs >= custom_threshold).astype(int)

    return result


# ─────────────────────────────────────────────
#  WALK-FORWARD BACKTESTER
# ─────────────────────────────────────────────

def backtest(
    data: pd.DataFrame,
    predictors: list,
    start: int = 2500,
    step: int = 250,
    model=None,
    custom_threshold: float = 0.55,
    model_name: str = "Ensemble",
    regime_aware: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Expanding-window walk-forward backtest.

    Parameters
    ----------
    data             : full cleaned dataset (from features.prepare_dataset)
    predictors       : list of feature column names
    start            : number of initial training rows before first test fold
    step             : number of rows per test fold (~1 year = 250)
    model            : pass a pre-built model to use it; None = fresh per fold
    custom_threshold : base probability threshold for BUY signal
    model_name       : used when model=None to build fresh instances
    regime_aware     : adjust threshold dynamically on VIX regime
    verbose          : print fold progress
    """
    all_predictions = []
    total_folds = len(range(start, data.shape[0], step))

    for fold_n, i in enumerate(range(start, data.shape[0], step), start=1):
        train = data.iloc[0:i].copy()
        test  = data.iloc[i:(i + step)].copy()

        if test.empty:
            break

        fold_model = get_model(model_name) if model is None else model
        preds = predict(
            train, test, predictors,
            model=fold_model,
            custom_threshold=custom_threshold,
            regime_aware=regime_aware,
        )
        all_predictions.append(preds)

        if verbose:
            pct = fold_n / total_folds * 100
            bar = "#" * int(pct / 5) + "-" * (20 - int(pct / 5))
            print(f"  [{bar}] {pct:5.1f}%  fold {fold_n}/{total_folds}", end="\r")

    if verbose:
        print()

    if not all_predictions:
        return pd.DataFrame()

    return pd.concat(all_predictions)


# ─────────────────────────────────────────────
#  FEATURE IMPORTANCE
# ─────────────────────────────────────────────

def compute_average_feature_importance(
    data: pd.DataFrame,
    predictors: list,
    start: int = 2500,
    step: int = 250,
    model_name: str = "RandomForest",   # VotingClassifier has no .feature_importances_
) -> pd.Series:
    importances = []
    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        if train.empty:
            break
        m = get_model(model_name)
        m.fit(train[predictors], train["Target"])
        if hasattr(m, "feature_importances_"):
            importances.append(m.feature_importances_)

    if not importances:
        return pd.Series(dtype=float)

    return pd.Series(
        np.mean(importances, axis=0), index=predictors
    ).sort_values(ascending=False)
