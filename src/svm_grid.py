"""svm_grid.py

Extended version of the user's original **svm.py**: adds a scikit‑learn
`GridSearchCV` step so you can tune the SVR hyper‑parameters on the fly.

Usage (CLI):
--------------
python svm_grid.py \
       -f path/to/csv            # (default ../data/twitter_trace.csv)
       -x COLUMN_NAME            # column to read & predict
       --grid                    # turn on grid‑search tuning
       --cv 3                    # k‑folds for GridSearch (default 3)
       --n_jobs -1               # parallel workers for GridSearch

If `--grid` is omitted the behaviour is identical to the original script
(default kernel='rbf', C=100, gamma='scale', epsilon=0.1).

The tuned model is cached per run so subsequent predictions are fast.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.svm import SVR

PRED_WINDOW = 4  # how many previous points to feed into each individual model
CACHE_DIR = Path(".svm_cache")
CACHE_DIR.mkdir(exist_ok=True)


def coerce(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    df[col_name] = pd.to_numeric(df[col_name], errors="coerce")
    return df.dropna(subset=[col_name])


def build_feature_series(series: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """Return X, y suitable for SVR training on *this row only*.

    X are time indices (0..N‑1) because we do a per‑row tiny fit,
    y are the values.
    """
    y = series.dropna().values.reshape(-1)
    X = np.arange(len(y)).reshape(-1, 1)
    return X, y


def single_point_predict(series: pd.Series, model: SVR) -> float:
    """Assumes *model* is already fitted to the tiny per‑row sequence."""
    next_x = [[len(series)]]
    return model.predict(next_x)[0]


def tune_svr_params(X: np.ndarray, y: np.ndarray, cv: int = 3,
                    n_jobs: int = -1) -> Dict[str, float]:
    """Run a quick GridSearchCV for SVR hyper‑parameters.

    TimeSeriesSplit is used so we don't leak future information.
    """
    param_grid = {
        "kernel": ["rbf", "poly", "sigmoid"],
        "C": [1, 10, 100],
        "gamma": ["scale", "auto"],
        "epsilon": [0.01, 0.1, 1.0],
    }
    tscv = TimeSeriesSplit(n_splits=cv)
    gs = GridSearchCV(SVR(), param_grid, cv=tscv, n_jobs=n_jobs,
                      scoring="neg_root_mean_squared_error", verbose=0)
    gs.fit(X, y)
    return gs.best_params_


def cached_best_params(cache_key: str) -> Dict[str, float] | None:
    cache_file = CACHE_DIR / f"{cache_key}.joblib"
    if cache_file.exists():
        return load(cache_file)
    return None


def save_best_params(cache_key: str, params: Dict[str, float]):
    cache_file = CACHE_DIR / f"{cache_key}.joblib"
    dump(params, cache_file)


def predict_next_svm(series: pd.Series,
                     default_params: Dict[str, float],
                     use_grid: bool = False,
                     cv: int = 3,
                     n_jobs: int = -1) -> float:
    """Fit per‑row SVR model, optionally tuning the hyper‑params.

    *series* contains the window of past points for this row.
    """
    try:
        X, y = build_feature_series(series)
        if len(y) < 3:
            return np.nan
        # decide params
        params = default_params.copy()
        if use_grid:
            cache_key = f"len{len(y)}"
            best = cached_best_params(cache_key)
            if best is None:
                best = tune_svr_params(X, y, cv=cv, n_jobs=n_jobs)
                save_best_params(cache_key, best)
            params.update(best)
        model = SVR(**params)
        model.fit(X, y)
        return single_point_predict(series, model)
    except Exception as exc:
        print(f"Predict error: {exc}")
        return np.nan


def run(args: argparse.Namespace):
    # Load data
    df = pd.read_csv(args.file)
    df = coerce(df, args.column)

    # Prepare shifted cols
    col = args.column
    df["Actual_Next"] = df[col].shift(-1)
    for i in range(PRED_WINDOW):
        df[f"prev{i}"] = df[col].shift(i + 1, fill_value=0)

    default_params = {
        "kernel": "rbf",
        "C": 100,
        "gamma": "scale",
        "epsilon": 0.1,
    }

    # apply row‑wise
    def _row_pred(row):
        series = pd.Series([row[col]] + [row[f"prev{i}"] for i in range(PRED_WINDOW)])
        return predict_next_svm(series, default_params,
                                use_grid=args.grid, cv=args.cv, n_jobs=args.n_jobs)

    df["Predicted_Next"] = df.apply(_row_pred, axis=1)
    df["Prediction_Error"] = df["Actual_Next"] - df["Predicted_Next"]

    valid = df.dropna(subset=["Predicted_Next", "Actual_Next"])
    if not valid.empty:
        rmse = mean_squared_error(valid["Actual_Next"], valid["Predicted_Next"], squared=False)
    else:
        rmse = np.nan

    out_file = "predicted_output_svm.csv"
    df.to_csv(out_file, index=False)

    print("Saved predictions ->", out_file)
    print(f"RMSE: {rmse:.4f}")
    if args.grid:
        print("Grid‑search used: best params are cached in ./.svm_cache/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="SVR predictor with optional GridSearchCV tuning")
    ap.add_argument("-f", "--file", default=os.path.join(Path(__file__).parent, "..", "data", "twitter_trace.csv"))
    ap.add_argument("-x", "--column", default="Tweets 09-May-2023", help="column to predict")
    ap.add_argument("--grid", action="store_true", help="enable GridSearchCV")
    ap.add_argument("--cv", type=int, default=3, help="folds for GridSearchCV (default 3)")
    ap.add_argument("--n_jobs", type=int, default=-1, help="parallel workers for GridSearchCV")
    args = ap.parse_args()

    run(args)

