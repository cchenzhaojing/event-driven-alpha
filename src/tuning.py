"""
pipeline.py

Experiment orchestration utilities for running training + analysis and logging results.

This module provides:
- collect_pred_metrics(): reads per-experiment analysis outputs and computes prediction stability.
- append_result_to_csv(): appends a single result row into a CSV (creates file if missing).
- run_single_experiment_with_current_config(): runs train + analysis using current config values.
- run_model_param_sweep(): sweeps LightGBM hyperparameters only (keeps training setup fixed).
- run_training_param_sweep(): sweeps training setup parameters (horizon/train/test windows).

Notes:
- This script assumes train.main() produces predictions under:
    config.OUTPUT_ROOT / EXP_NAME / predictions.csv
  and analysis.main() produces analysis outputs under:
    config.OUTPUT_ROOT / EXP_NAME / analysis/
- Any change in experiment naming rules between training and analysis will break file alignment.
"""

# ===== Standard library imports =====
import importlib
import itertools
from pathlib import Path

# ===== Third-party imports =====
import pandas as pd

# ===== Local imports =====
import model_eval  # Must expose main()
import config  # Centralized hyperparameters and paths
import train  # Must expose main() and EXP_NAME


def collect_pred_metrics(exp_name: str) -> dict:
    """
    Given an EXP_NAME, read the corresponding analysis outputs and compute Pred_Stability.

    Pred_Stability is computed by:
        1) grouping daily RankIC by year-month,
        2) taking the mean RankIC per month,
        3) returning mean(monthly_mean_rank_ic) / std(monthly_mean_rank_ic).

    Returns a dictionary containing:
        - exp_name
        - pred_stability
        - mean_rank_ic
        - mean_ic
        - mse

    Parameters:
        exp_name (str): Experiment name used to locate outputs.

    Returns:
        dict: Metrics dictionary.
    """
    analysis_dir = config.OUTPUT_ROOT / exp_name / "analysis"

    # 1) Read daily RankIC series
    daily_path = analysis_dir / "prediction_daily_ic_rankic.csv"
    daily = pd.read_csv(daily_path)

    # Parse date for monthly bucketing
    daily["dt"] = pd.to_datetime(daily["dt"])
    daily["ym"] = daily["dt"].dt.to_period("M")

    # Monthly block means of RankIC
    block_mean = daily.groupby("ym")["rank_ic"].mean()

    mean_block = block_mean.mean()
    std_block = block_mean.std(ddof=0)  # Population std for simplicity

    pred_stability = mean_block / (std_block + 1e-8)  # Avoid division by zero

    # 2) Read summary metrics (mean_rank_ic / mean_ic / mse)
    summary_path = analysis_dir / "prediction_summary_metrics.csv"
    summary = pd.read_csv(summary_path)

    mean_rank_ic = float(summary.loc[0, "mean_rank_ic"])
    mean_ic = float(summary.loc[0, "mean_ic"])
    mse = float(summary.loc[0, "mse"])

    return dict(
        exp_name=exp_name,
        pred_stability=pred_stability,
        mean_rank_ic=mean_rank_ic,
        mean_ic=mean_ic,
        mse=mse,
    )


def append_result_to_csv(row: dict, csv_path: Path):
    """
    Append a single result row into a CSV file.

    If the file does not exist, create it and write the header.
    If the file exists, append without writing a header.

    Parameters:
        row (dict): Row data to write (one experiment).
        csv_path (Path): Output CSV path.
    """
    df = pd.DataFrame([row])
    if csv_path.exists():
        df.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df.to_csv(csv_path, index=False)


def run_single_experiment_with_current_config():
    """
    Run a single training + analysis pass using the current config values.

    This helper does not modify any config parameters. It reloads the modules
    before calling main() to ensure the latest config state is used.
    """
    importlib.reload(train)
    train.main()

    importlib.reload(model_eval)
    model_eval.main()


def run_model_param_sweep():
    """
    Pipeline for sweeping tree-model parameters only.

    This function keeps:
        HORIZON / TRAIN_WINDOW / TEST_WINDOW fixed
    and sweeps:
        LGB_PARAMS (n_estimators, learning_rate, num_leaves, max_depth, etc.)

    Results are appended to:
        config.OUTPUT_ROOT / "model_param_scores.csv"
    """
    print(
        "Fixed training setup: "
        f"HORIZON={config.HORIZON}, "
        f"TRAIN_WINDOW={config.TRAIN_WINDOW}, "
        f"TEST_WINDOW={config.TEST_WINDOW}"
    )

    base_lgb_params = config.LGB_PARAMS.copy()
    score_csv_path = config.OUTPUT_ROOT / "model_param_scores.csv"

    # Coarse sweep grid
    n_estimators_list = [400, 600, 800]
    learning_rates = [0.02, 0.03, 0.05]
    num_leaves_list = [16, 31, 63]
    max_depth_list = [-1, 6, 10]
    subsamples = [0.8]
    colsample_bys = [0.8]

    for (n_estimator, lr, nl, md, ss, cs) in itertools.product(
        n_estimators_list,
        learning_rates,
        num_leaves_list,
        max_depth_list,
        subsamples,
        colsample_bys,
    ):
        # Update model parameters
        config.LGB_PARAMS.update(
            dict(
                n_estimators=n_estimator,
                learning_rate=lr,
                num_leaves=nl,
                max_depth=md,
                subsample=ss,
                colsample_bytree=cs,
            )
        )

        print("\n================= MODEL PARAM EXPERIMENT =================")
        print(
            f"n_estimators={n_estimator}, learning_rate={lr}, num_leaves={nl}, "
            f"max_depth={md}, subsample={ss}, colsample_bytree={cs}"
        )
        print("==========================================================")

        # Run training + analysis
        run_single_experiment_with_current_config()
        exp_name = train.EXP_NAME

        # Collect prediction-level metrics (including pred_stability)
        pred_metrics = collect_pred_metrics(exp_name)

        # Assemble result row: params + metrics
        row = {
            "exp_name": exp_name,
            "horizon": config.HORIZON,
            "train_window": config.TRAIN_WINDOW,
            "test_window": config.TEST_WINDOW,
            "learning_rate": lr,
            "num_leaves": nl,
            "max_depth": md,
            "n_estimators": n_estimator,
            "subsample": ss,
            "colsample_bytree": cs,
        }
        row.update(pred_metrics)

        append_result_to_csv(row, score_csv_path)

    # Restore original params (optional)
    config.LGB_PARAMS = base_lgb_params


def run_training_param_sweep():
    """
    Pipeline for sweeping training setup parameters.

    This function keeps model parameters fixed and sweeps:
        HORIZON / TRAIN_WINDOW / TEST_WINDOW

    Results are appended to:
        config.OUTPUT_ROOT / "train_param_scores.csv"
    """
    print(
        "Fixed model setup: "
        f"n_estimators={config.LGB_PARAMS['n_estimators']}, "
        f"lr={config.LGB_PARAMS['learning_rate']}, "
        f"num_leaves={config.LGB_PARAMS['num_leaves']}, "
        f"max_depth={config.LGB_PARAMS['max_depth']}"
    )

    base_train_horizon = config.HORIZON
    base_train_window = config.TRAIN_WINDOW
    base_test_window = config.TEST_WINDOW

    score_csv_path = config.OUTPUT_ROOT / "train_param_scores.csv"

    # Training parameter sweep grid
    horizon_list = [3, 5, 10, 20]
    train_window_list = [80, 150, 252, 300]
    test_window_list = [3, 5, 10]

    for (hozizon, train_window, test_window) in itertools.product(
        horizon_list, train_window_list, test_window_list
    ):
        if not ((hozizon == 5) and (train_window == 252) and (test_window == 5)):
            continue

        # Update training parameters
        config.HORIZON = hozizon
        config.TRAIN_WINDOW = train_window
        config.TEST_WINDOW = test_window

        print("\n================= TRAIN PARAM EXPERIMENT =================")
        print(f"HORIZON={hozizon}, TRAIN_WINDOW={train_window}, TEST_WINDOW={test_window}")
        print("==========================================================")

        # Run training + analysis
        run_single_experiment_with_current_config()
        exp_name = train.EXP_NAME

        # Collect prediction-level metrics
        pred_metrics = collect_pred_metrics(exp_name)

        # Assemble result row
        row = {
            "exp_name": exp_name,
            "horizon": config.HORIZON,
            "train_window": config.TRAIN_WINDOW,
            "test_window": config.TEST_WINDOW,
        }
        row.update(pred_metrics)

        append_result_to_csv(row, score_csv_path)

    # Restore original training setup (optional)
    config.HORIZON = base_train_horizon
    config.TRAIN_WINDOW = base_train_window
    config.TEST_WINDOW = base_test_window


if __name__ == "__main__":
    # Run training parameter sweep by default
    run_training_param_sweep()
