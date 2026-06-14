"""
config.py

Global configuration constants for the project.

This module defines:
- Data paths and output directories
- Shared column name conventions (train / analysis)
- Feature engineering settings (OHLCV windows, filtering thresholds)
- Rolling training settings (horizon, train/test window lengths)
- SHAP toggle for interpretability analysis
- LightGBM hyperparameters (must remain consistent across scripts)
- Analysis-related parameters (FF5 beta inputs, regression window, future horizons)

Notes:
- This file intentionally contains only constants and simple conditional switches.
- Downstream scripts should import from here to avoid hard-coded values.
"""

from pathlib import Path
from typing import Dict, List, Tuple

# ====== Project root (repo root) ======
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ====== Data paths ======
TRAIN_DATA_PATH = PROJECT_ROOT / "data" / "issuer_data_textual.csv"
OUTPUT_ROOT = PROJECT_ROOT / "outputs_final_no_industry"

# ====== Shared column name configuration (train / analysis) ======
DATE_COL = "dt"
ID_COL = "corp_tkr"
CLOSE_COL = "close"
OHLCV_COLS = ["open", "high", "low", "close", "volume"]

# ====== OHLCV feature engineering ======
OHLCV_WINDOWS: Tuple[int, ...] = (14, 20, 63)  # Rolling windows for technical indicators
VOL_STD_MIN: float = 1e-4  # Threshold for dropping samples with near-zero volatility
CONST_VAR_THRESHOLD: float = 1e-8  # Variance threshold for removing near-constant features

ONLY_INDUSTRY = False
DROP_INDUSTRY = False
DEBUG_SHUFFLE_LABEL = False
DELETE_FEATURES = True

# ====== Rolling training settings ======
HORIZON: int = 5  # Predict K-day forward return (train.py default is 5)
TRAIN_WINDOW: int = 252  # Training window length per rolling iteration
TEST_WINDOW: int = 5  # Test window length per rolling iteration

# ====== SHAP settings ======
RUN_SHAP_ANALYSIS = True  # If True, run SHAP; if False, skip SHAP

# ====== LightGBM parameters (must match train.py / analysis.py) ======
LGB_PARAMS: Dict = dict(
    n_estimators=800,
    learning_rate=0.03,
    max_depth=-1,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="regression_l2",
    n_jobs=-1,
    force_row_wise=True,
)

# ====== Analysis script parameters ======
FF5_BETA_PATH = PROJECT_ROOT / "data" / "ff5_rolling_betas.parquet"
FF5_REG_WINDOW_DAYS: int = 120  # FF5 regression window length (used to build EXP_NAME in analysis.py)

FUTURE_WINDOWS: List[int] = [1, 5, 10, 20]  # Future horizons (k-day returns) for deviation analysis
MIN_STOCKS_PER_DAY: int = 30  # Minimum cross-sectional stock count per day
MIN_OBS_FOR_MEAN: int = 30  # Minimum sample size when estimating conditional means

# ====== Output directory override logic (do not change behavior) ======
if ONLY_INDUSTRY:
    OUTPUT_ROOT = PROJECT_ROOT / "outputs_final_only_industry"
elif DROP_INDUSTRY:
    OUTPUT_ROOT = PROJECT_ROOT / "outputs_final_no_industry"
