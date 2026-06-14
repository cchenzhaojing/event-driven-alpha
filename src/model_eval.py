"""
Prediction evaluation utilities.

This script reads the rolling prediction output file and produces:

1) prediction_daily_ic_rankic.csv
   - Per trading day:
       - rank_ic: cross-sectional RankIC between predictions and realized ret_fwd_H
       - ic     : cross-sectional Pearson correlation between predictions and realized ret_fwd_H

2) prediction_summary_metrics.csv
   - One-row summary metrics:
       - mean_rank_ic: mean daily RankIC (ignoring NaNs)
       - mean_ic     : mean daily IC (ignoring NaNs)
       - mse         : full-sample MSE = mean((ret_fwd_H - pred_H)^2)
"""

# ===== Standard library imports =====
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ===== Third-party imports =====
import numpy as np
import pandas as pd

# ===== Local imports =====
from config import (
    CLOSE_COL,
    DATE_COL,
    FF5_REG_WINDOW_DAYS,
    FUTURE_WINDOWS,
    HORIZON,
    ID_COL,
    LGB_PARAMS,
    MIN_OBS_FOR_MEAN,
    MIN_STOCKS_PER_DAY,
    OHLCV_WINDOWS,
    OUTPUT_ROOT,
    TEST_WINDOW,
    TRAIN_WINDOW,
)

warnings.filterwarnings("ignore")

PRED_COL = f"pred_{HORIZON}"
ACTUAL_COL = f"ret_fwd_{HORIZON}"

# ======================================================================


def build_experiment_name() -> str:
    """
    Build a deterministic experiment name based on key configuration parameters.

    Returns:
        str: Experiment name used to locate prediction files and write analysis outputs.
    """
    win_str = "-".join(str(w) for w in OHLCV_WINDOWS)
    exp_name = (
        f"h{HORIZON}"
        f"_tr{TRAIN_WINDOW}"
        f"_te{TEST_WINDOW}"
        f"_win{win_str}"
        f"_lr{LGB_PARAMS['learning_rate']}"
        f"_ne{LGB_PARAMS['n_estimators']}"
        f"_md{LGB_PARAMS['max_depth']}"
        f"_nleaves{LGB_PARAMS['num_leaves']}"
        f"_ss{LGB_PARAMS['subsample']}"
        f"_cs{LGB_PARAMS['colsample_bytree']}"
        f"_ff5w{FF5_REG_WINDOW_DAYS}"  # Tag the FF5 regression lookback window (days)
    )
    return exp_name


EXP_NAME = build_experiment_name()

# Prediction file path (must match the training script exactly)
PREDICTION_FILE = OUTPUT_ROOT / EXP_NAME / "predictions.csv"

# Analysis output directory (organized under each experiment folder)
ANALYSIS_OUTPUT_DIR = OUTPUT_ROOT / EXP_NAME / "analysis"


# ===================== Utility functions =====================


def load_and_prepare_base_df() -> pd.DataFrame:
    """
    Load the prediction file, filter valid samples, and compute deviation.

    Steps:
        - Load predictions.csv and parse DATE_COL
        - If split_id exists, keep only split_id > 0 (i.e., truly out-of-sample predictions)
        - Drop rows with missing values in required columns
        - Compute deviation = actual - prediction

    Returns:
        pd.DataFrame: Prepared DataFrame with a 'deviation' column.
    """
    df = pd.read_csv(PREDICTION_FILE)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])

    print(f"✅ Prediction file loaded: {PREDICTION_FILE}")
    print(f"   Total rows: {len(df):,}")

    # Use only rows with actual predictions (if split_id exists, keep split_id > 0)
    if "split_id" in df.columns:
        df = df[df["split_id"] > 0].copy()

    # Required columns must be present and non-missing
    needed_cols = [PRED_COL, ACTUAL_COL, CLOSE_COL]
    df = df.dropna(subset=[c for c in needed_cols if c in df.columns])

    # deviation = actual - prediction
    df["deviation"] = df[ACTUAL_COL] - df[PRED_COL]

    print(f"   Valid rows after filtering: {len(df):,}")
    return df


def compute_prediction_accuracy_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute prediction accuracy metrics for t + horizon:
      - Daily RankIC (prediction vs actual) and IC (prediction vs actual)
      - Full-sample summary: mean_rank_ic, mean_ic, MSE

    Parameters:
        df (pd.DataFrame): Prepared DataFrame that includes PRED_COL and ACTUAL_COL.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            - daily_metrics: index=DATE_COL, columns=[rank_ic, ic]
            - summary_df: one-row DataFrame with mean_rank_ic, mean_ic, mse
    """

    def _daily_metrics(group: pd.DataFrame) -> pd.Series:
        g = group[[PRED_COL, ACTUAL_COL]].dropna()
        if len(g) < MIN_STOCKS_PER_DAY:
            return pd.Series({"rank_ic": np.nan, "ic": np.nan})
        if g[PRED_COL].nunique() < 2 or g[ACTUAL_COL].nunique() < 2:
            return pd.Series({"rank_ic": np.nan, "ic": np.nan})

        rank_ic = g[PRED_COL].rank().corr(g[ACTUAL_COL].rank())
        ic = g[PRED_COL].corr(g[ACTUAL_COL])
        return pd.Series({"rank_ic": rank_ic, "ic": ic})

    daily_metrics = df.groupby(DATE_COL).apply(_daily_metrics).sort_index()

    # Summary metrics (mean over non-NaN days)
    summary = {}
    summary["mean_rank_ic"] = daily_metrics["rank_ic"].mean(skipna=True)
    summary["mean_ic"] = daily_metrics["ic"].mean(skipna=True)

    # Full-sample MSE
    diff = df[ACTUAL_COL] - df[PRED_COL]
    summary["mse"] = np.mean(diff**2)

    summary_df = pd.DataFrame([summary])
    return daily_metrics, summary_df


# ===================== Main entrypoint =====================


def main():
    # 1) Load prediction file and prepare the base DataFrame (including deviation)
    df = load_and_prepare_base_df()

    # 2) Compute prediction accuracy metrics: RankIC, IC, MSE
    print("📈 Computing prediction accuracy metrics (t + horizon)...")
    daily_pred_metrics, pred_summary = compute_prediction_accuracy_metrics(df)
    print("✅ Prediction accuracy summary:")
    print(pred_summary)

    ANALYSIS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    daily_pred_metrics.to_csv(ANALYSIS_OUTPUT_DIR / "prediction_daily_ic_rankic.csv")
    pred_summary.to_csv(
        ANALYSIS_OUTPUT_DIR / "prediction_summary_metrics.csv", index=False
    )
    print("✅ All analysis outputs saved to:", ANALYSIS_OUTPUT_DIR)


if __name__ == "__main__":
    main()
