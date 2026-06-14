"""
Feature engineering utilities for OHLCV-based technical indicators.

This module defines:
- OHLCVFeatureEngineer: Generates rolling technical indicators from multi-asset OHLCV data.

Assumptions:
- Input DataFrame contains at least:
  - DATE_COL (e.g., "dt"): datetime-like
  - ID_COL (e.g., "corp_tkr"): asset identifier
  - CLOSE_COL (e.g., "close")
  - "volume"
  - "high" and "low" (used for tradability filtering)

Outputs:
- add_features() returns a copy of the input DataFrame augmented with engineered features.
"""

# ===== Standard library imports =====
from typing import Tuple

# ===== Third-party imports =====
import numpy as np
import pandas as pd

# ===== Local imports =====
from config import (
    CLOSE_COL,
    CONST_VAR_THRESHOLD,
    DATE_COL,
    FF5_BETA_PATH,
    FF5_REG_WINDOW_DAYS,
    HORIZON,
    ID_COL,
    LGB_PARAMS,
    MIN_STOCKS_PER_DAY,
    OHLCV_COLS,
    OHLCV_WINDOWS,
    OUTPUT_ROOT,
    TEST_WINDOW,
    TRAIN_DATA_PATH,
    TRAIN_WINDOW,
    VOL_STD_MIN,
)

# ========= 1. Technical indicator engineering =========


class OHLCVFeatureEngineer:
    """
    Generate technical indicators from multi-asset OHLCV data.

    The input DataFrame must contain at least:
        - DATE_COL   : datetime-like date column
        - ID_COL     : asset identifier (string)
        - CLOSE_COL  : close price column
        - "volume"   : trading volume

    add_features() returns:
        - The input DataFrame augmented with engineered technical feature columns.
    """

    def __init__(self, windows: Tuple[int, ...] = OHLCV_WINDOWS):
        """
        Initialize the feature engineer.

        Parameters:
            windows (Tuple[int, ...]): Rolling windows used to compute indicators.
        """
        self.windows = windows

    @staticmethod
    def wilder_rsi(series: pd.Series, n: int) -> pd.Series:
        """
        Compute Wilder's RSI (exponential smoothing with alpha=1/n).

        Parameters:
            series (pd.Series): Price series.
            n (int): RSI window length.

        Returns:
            pd.Series: RSI values aligned with the input index.
        """
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
        avg_loss = loss.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - 100 / (1 + rs)
        return rsi

    def rsi_weighted(self, prices: pd.Series, window: int, lam: float = 0.99) -> pd.Series:
        """
        Compute an exponentially-weighted RSI variant with recursive updates.

        Notes:
            - If the series length is too short (<= window), this returns an all-NaN series
              to avoid raising errors and to preserve alignment.

        Parameters:
            prices (pd.Series): Price series.
            window (int): RSI lookback window.
            lam (float): Exponential decay factor in (0, 1).

        Returns:
            pd.Series: RSI values aligned with the input index.
        """
        if not 0 < lam < 1:
            raise ValueError("lam must be in (0, 1).")

        prices = prices.astype(float)
        n = len(prices)

        # Key behavior: return an all-NaN Series instead of raising for short inputs.
        if n <= window:
            return pd.Series(np.nan, index=prices.index)

        delta = prices.diff()

        # Gains and losses
        gains = delta.clip(lower=0.0).fillna(0.0)
        losses = -delta.clip(upper=0.0).fillna(0.0)

        avg_gain = np.zeros(n)
        avg_loss = np.zeros(n)

        # Initialize at index = window
        init_slice = slice(1, window + 1)
        avg_gain[window] = gains.iloc[init_slice].mean()
        avg_loss[window] = losses.iloc[init_slice].mean()

        # Recursive update
        for t in range(window + 1, n):
            g_t = gains.iloc[t]
            l_t = losses.iloc[t]
            avg_gain[t] = lam * avg_gain[t - 1] + (1.0 - lam) * g_t
            avg_loss[t] = lam * avg_loss[t - 1] + (1.0 - lam) * l_t

        # Convert to Series aligned with prices
        avg_gain_series = pd.Series(avg_gain, index=prices.index)
        avg_loss_series = pd.Series(avg_loss, index=prices.index)

        # Relative strength and RSI
        rs = avg_gain_series / avg_loss_series.replace(0.0, np.nan)
        rsi = 100.0 - 100.0 / (1.0 + rs)

        # Set pre-window values to NaN
        rsi.iloc[: window + 1] = np.nan

        return rsi

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add OHLCV-based technical features to the input DataFrame.

        Steps:
            1) Sort by (ID_COL, DATE_COL) and compute group-wise features.
            2) Generate rolling-window features for each window in self.windows.
            3) Clean infinities, filter non-tradable rows, drop near-constant features, and drop NaNs.
            4) Sanitize column names for LightGBM compatibility.

        Parameters:
            df (pd.DataFrame): Raw OHLCV DataFrame.

        Returns:
            pd.DataFrame: Feature-augmented and cleaned DataFrame.
        """
        df = df.copy()
        df = df.sort_values([ID_COL, DATE_COL])
        g = df.groupby(ID_COL, group_keys=False)

        # Simple daily return
        df["ret_simple"] = g[CLOSE_COL].pct_change(1)

        # Generate a feature block per rolling window
        for n in self.windows:
            # Cumulative return over window n
            df[f"cumret_simple_{n}"] = g[CLOSE_COL].apply(lambda x: x / x.shift(n) - 1)

            # Price relative to SMA
            sma_n = g[CLOSE_COL].transform(lambda x: x.rolling(n, min_periods=n).mean())
            df[f"price_sma_ratio_{n}"] = df[CLOSE_COL] / sma_n - 1

            # Momentum (normalized)
            df[f"mom_norm_{n}"] = g[CLOSE_COL].apply(lambda x: (x - x.shift(n)) / x.shift(n))

            # RSI
            # df[f"rsi_{n}"] = g[CLOSE_COL].apply(lambda x: self.wilder_rsi(x, n))
            df[f"rsi_{n}"] = g[CLOSE_COL].apply(lambda x: self.rsi_weighted(x, window=n, lam=0.99))

            # Realized volatility proxy (sum of squared log returns)
            ret_log = g[CLOSE_COL].apply(lambda x: np.log(x / x.shift(1)))
            df[f"rv_{n}"] = ret_log.pow(2).groupby(df[ID_COL]).transform(
                lambda x: x.rolling(n, min_periods=n).sum()
            )

            # Volume z-score
            vol_mean = g["volume"].transform(lambda x: x.rolling(n, min_periods=n).mean())
            vol_std = g["volume"].transform(lambda x: x.rolling(n, min_periods=n).std(ddof=0))
            df[f"vol_z_{n}"] = (df["volume"] - vol_mean) / vol_std.replace(0, np.nan)

            # Dollar volume relative to its rolling mean
            dv = df[CLOSE_COL] * df["volume"]
            dv_sma = (
                g.apply(lambda x: (x[CLOSE_COL] * x["volume"]).rolling(n, min_periods=n).mean())
                .reset_index(level=0, drop=True)
            )
            df[f"dollar_vol_sma_diff_pct_{n}"] = dv / dv_sma - 1

        # Daily volume change
        df["vol_pct"] = g["volume"].pct_change(1)

        # Safety cleanup: replace inf / -inf with NaN
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Filter non-tradable rows (no volume or no intraday range)
        mask_tradable = (df["volume"] > 0) & (df["high"] > df["low"])
        df = df[mask_tradable]

        # Drop near-zero volatility samples (use vol_std_20 as a representative feature if present)
        if "vol_std_20" in df.columns:
            df = df[df["vol_std_20"] > VOL_STD_MIN]

        # Drop near-constant numeric feature columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        var_series = df[numeric_cols].var()
        constant_cols = var_series[var_series < CONST_VAR_THRESHOLD].index.tolist()
        if constant_cols:
            print(f"⚠️ Dropping near-constant feature columns: {constant_cols}")
            df.drop(columns=constant_cols, inplace=True)

        # Drop rows that are all-NaN across numeric columns
        df.dropna(axis=0, how="all", subset=numeric_cols, inplace=True)

        # Conservative NaN cleanup: ensure core columns exist
        df = df.dropna(subset=[CLOSE_COL, "volume"])

        # Sanitize column names to avoid unsupported symbols in LightGBM
        df.columns = (
            df.columns.str.replace(r"[^0-9a-zA-Z_]+", "_", regex=True)
            .str.replace(r"_+", "_", regex=True)
            .str.strip("_")
            .str.lower()
        )

        return df
