"""
Rolling LightGBM training script for event-driven forward return prediction.

This module defines:
- Experiment naming and output directory construction.
- EventReturnModel: a rolling-window LightGBM regressor that predicts K-day forward returns.
- main(): end-to-end workflow (load data, merge FF5 betas, engineer OHLCV features, train/predict, save outputs).

Outputs:
- predictions.csv (includes ret_fwd_{HORIZON} and pred_{HORIZON})
- feature_importance_mean.csv (mean gain/split importance across rolling splits)
- Optional SHAP artifacts (feature_importance_shap.csv, shap_summary_plot.png)

Notes:
- This script focuses on generating labels and predictions only; it does not run backtests or analysis.
- Configuration is expected to be provided via config.py to avoid hard-coded values.
"""

# ===== Standard library imports =====
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ===== Third-party imports =====
import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from sklearn.preprocessing import RobustScaler

# ===== Local imports =====
from feature_engineering import OHLCVFeatureEngineer
from config import (
    CLOSE_COL,
    CONST_VAR_THRESHOLD,
    DATE_COL,
    DEBUG_SHUFFLE_LABEL,
    DELETE_FEATURES,
    DROP_INDUSTRY,
    FF5_BETA_PATH,
    FF5_REG_WINDOW_DAYS,
    HORIZON,
    ID_COL,
    LGB_PARAMS,
    MIN_STOCKS_PER_DAY,
    OHLCV_COLS,
    OHLCV_WINDOWS,
    ONLY_INDUSTRY,
    OUTPUT_ROOT,
    RUN_SHAP_ANALYSIS,
    TEST_WINDOW,
    TRAIN_DATA_PATH,
    TRAIN_WINDOW,
    VOL_STD_MIN,
)

warnings.filterwarnings("ignore")


CATEGORICAL_COLS_DEFAULT = [
    "corp_tkr",
    "ml_industry_lvl_2",
    "ml_industry_lvl_3",
    "ml_industry_lvl_4",
]

CS_NORM_COLS = [
    "10y_oas",
    "mv",
    "open",
    "low",
    "high",
    "volume",
    "ml_industry_lvl_2_roll0d",
    "ml_industry_lvl_2_roll5d",
    "ml_industry_lvl_2_roll10d",
    "ml_industry_lvl_3_roll0d",
    "ml_industry_lvl_3_roll5d",
    "ml_industry_lvl_3_roll10d",
    "ml_industry_lvl_4_roll0d",
    "ml_industry_lvl_4_roll5d",
    "ml_industry_lvl_4_roll10d",
    "pca1",
    "pca2",
    "pca3",
    "alpha",
    "beta_mkt_rf",
    "beta_smb",
    "beta_hml",
    "beta_rmw",
    "beta_cma",
    "ret_simple",
    "cumret_simple_14",
    "price_sma_ratio_14",
    "mom_norm_14",
    "rsi_14",
    "rv_14",
    "vol_z_14",
    "dollar_vol_sma_diff_pct_14",
    "cumret_simple_20",
    "price_sma_ratio_20",
    "mom_norm_20",
    "rsi_20",
    "rv_20",
    "vol_z_20",
    "dollar_vol_sma_diff_pct_20",
    "cumret_simple_63",
    "price_sma_ratio_63",
    "mom_norm_63",
    "rsi_63",
    "rv_63",
    "vol_z_63",
    "dollar_vol_sma_diff_pct_63",
    "vol_pct",
]

DROP_FEATURES = [
    "volume",
    "low",
    "vol_z_14",
    "cumret_simple_14",
    "vol_z_20",
    "high",
    "rsi_20",
    "rsi_14",
    "ret_simple",
    "dollar_vol_sma_diff_pct_20",
    "mom_norm_63",
    "dollar_vol_sma_diff_pct_14",
    "mom_norm_14",
    "mom_norm_20",
    "vol_pct",
    "roll10d_events_count",
    "roll5d_events_count",
    "corporate_actions",
    "operations_business_changes",
    "event_count",
    "analyst_research_events",
    "strategic_financing_intent",
    "capital_markets_deals",
    "dividends",
    "index_membership",
    "m_a",
    "product_events",
    "guidance",
    "esg",
    "regulatory",
    "leadership",
    "legal",
    "bankruptcy_restructuring",
]

INDUSTRY = [
    "ml_industry_lvl_2_roll0d",
    "ml_industry_lvl_2_roll5d",
    "ml_industry_lvl_2_roll10d",
    "ml_industry_lvl_3_roll0d",
    "ml_industry_lvl_3_roll5d",
    "ml_industry_lvl_3_roll10d",
    "ml_industry_lvl_4_roll0d",
    "ml_industry_lvl_4_roll5d",
    "ml_industry_lvl_4_roll10d",
    "pca1",
    "pca2",
    "pca3",
]

if DROP_INDUSTRY:
    # 1) Drop categorical industry columns (ml_industry_lvl_x)
    industry_cat_cols = [c for c in CATEGORICAL_COLS_DEFAULT if "industry" in c]

    # 2) Drop cross-sectional normalized industry rolling features (ml_industry_lvl_x_rollXd)
    #    as well as PCA columns (pca1, pca2, pca3)
    industry_cs_cols = [
        c for c in CS_NORM_COLS if ("industry" in c) or (c.lower().startswith("pca"))
    ]

    # 3) Merge these columns into DROP_FEATURES (use set to avoid duplicates)
    cols_to_drop = set(industry_cat_cols + industry_cs_cols)
    DROP_FEATURES = list(set(DROP_FEATURES).union(cols_to_drop))

    print(
        f"🚫 DROP_INDUSTRY is enabled. Additionally dropping {len(cols_to_drop)} industry/PCA-related features."
    )


# ======================================================================


# ========= Build experiment name & output directory from key parameters =========
def build_experiment_name() -> str:
    """
    Build a deterministic experiment name based on key configuration parameters.

    Returns:
        str: Experiment name used to create the output directory.
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
OUTPUT_DIR = OUTPUT_ROOT / EXP_NAME  # Final output directory: OUTPUT_ROOT/EXP_NAME
print(f"Current experiment output directory: {OUTPUT_DIR}")


# ========= 2. LGBM forward return prediction model =========


@dataclass
class EventReturnModel:
    """
    Train a unified LightGBM model to predict K-day forward returns with rolling splits.

    This class generates:
    - ret_fwd_{K}: forward return label
    - pred_{K}: model predictions

    It does NOT perform backtesting or performance analytics.
    """

    horizon: int = HORIZON  # Predict K-day forward return
    date_col: str = DATE_COL
    id_col: str = ID_COL
    close_col: str = CLOSE_COL
    feature_cols: Optional[List[str]] = None  # If None, infer automatically
    categorical_cols: Optional[List[str]] = None

    lgb_params: Dict = field(default_factory=lambda: LGB_PARAMS)

    def _make_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create the forward return label column: ret_fwd_{horizon}.

        Parameters:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: DataFrame with an added forward return label column.
        """
        df = df.sort_values([self.id_col, self.date_col]).copy()
        g = df.groupby(self.id_col)

        # Forward return: close_{t+h} / close_t - 1
        future_price = g[self.close_col].shift(-self.horizon)
        df[f"ret_fwd_{self.horizon}"] = future_price / df[self.close_col] - 1

        return df

    def _infer_feature_cols(self, df: pd.DataFrame) -> List[str]:
        """
        Infer usable feature columns by excluding identifiers, targets, metadata, and object dtypes.

        Exclusions:
        - id/date/close/target
        - common metadata columns (equity_name, country, fs_entity_id, etc.)
        - "Unnamed: 0"
        - optionally DROP_FEATURES when DELETE_FEATURES is enabled

        Parameters:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            List[str]: Inferred feature column names.
        """
        exclude = {
            self.id_col,
            self.date_col,
            self.close_col,
            f"ret_fwd_{self.horizon}",
            "Unnamed: 0",
            "equity_name",
            "country",
            "fs_entity_id",
            "renamed_ticker",
        }
        if DELETE_FEATURES:
            exclude = exclude.union(set(DROP_FEATURES))

        numeric_or_category_cols = [
            c
            for c in df.columns
            if (df[c].dtype.name not in ["object"]) and (c not in exclude)
        ]
        return numeric_or_category_cols

    def fit_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rolling time-split training for LightGBM and writing predictions back to the DataFrame.

        This method:
        - Creates the label ret_fwd_{horizon}
        - Runs rolling splits in date space:
          train on last TRAIN_WINDOW trading days -> predict next TEST_WINDOW trading days
        - Writes pred_{horizon} and split_id to df
        - Optionally computes SHAP artifacts

        Parameters:
            df (pd.DataFrame): Feature-augmented DataFrame.

        Returns:
            pd.DataFrame: DataFrame with prediction columns added.
        """
        df = df.copy()
        df[self.date_col] = pd.to_datetime(df[self.date_col])

        # 1) Create forward return label
        df = self._make_target(df)
        df = df.dropna(subset=[f"ret_fwd_{self.horizon}"]).reset_index(drop=True)

        if DEBUG_SHUFFLE_LABEL:
            label_col = f"ret_fwd_{self.horizon}"
            df[label_col] = np.random.permutation(df[label_col].values)
            print(
                "⚠️ DEBUG_SHUFFLE_LABEL = True. ret_fwd has been randomly permuted for leakage diagnostics."
            )

        # 2) Infer features if not provided
        if self.feature_cols is None:
            self.feature_cols = self._infer_feature_cols(df)

        # 3) Categorical columns
        if self.categorical_cols is None:
            self.categorical_cols = [c for c in CATEGORICAL_COLS_DEFAULT if c in df.columns]
        for col in self.categorical_cols:
            df[col] = df[col].astype("category")

        # 4) Missing value handling / numeric sanitation
        df = df.dropna(subset=OHLCV_COLS)
        num_cols_all = df[self.feature_cols].select_dtypes(include=[np.number]).columns
        df[num_cols_all] = df[num_cols_all].replace([np.inf, -np.inf], np.nan)

        print("\n================= FEATURE SUMMARY =================")
        print(f"📌 Number of features: {len(self.feature_cols):,}")
        print(f"📌 Feature columns: {self.feature_cols}")
        print("\n📌 Feature preview (first 3 rows):")
        print(df[self.feature_cols].head(3))
        print("====================================================\n")

        # 5) Time sort
        df = df.sort_values(self.date_col).reset_index(drop=True)
        X = df[self.feature_cols]
        y = df[f"ret_fwd_{self.horizon}"]

        # 6) Rolling splits by trading days
        all_dates = df[self.date_col].sort_values().unique()

        df[f"pred_{self.horizon}"] = np.nan
        df["split_id"] = -1

        # Accumulate feature importance across rolling splits
        feat_names_for_imp = None
        feat_imp_gain_sum = None
        feat_imp_split_sum = None
        n_splits_for_imp = 0

        all_shap_values = []
        all_X_test_scaled = []

        # Roll forward in date space (advance by TEST_WINDOW trading days)
        for split_id, start in enumerate(
            range(TRAIN_WINDOW, len(all_dates) - TEST_WINDOW + 1, TEST_WINDOW),
            start=1,
        ):
            # Train dates: last TRAIN_WINDOW trading days
            train_dates = all_dates[start - TRAIN_WINDOW : start]
            # Test dates: next TEST_WINDOW trading days
            test_dates = all_dates[start : start + TEST_WINDOW]

            train_mask = df[self.date_col].isin(train_dates)
            test_mask = df[self.date_col].isin(test_dates)

            train_idx = np.where(train_mask)[0]
            test_idx = np.where(test_mask)[0]

            X_train, y_train = X.iloc[train_idx].copy(), y.iloc[train_idx]
            X_test = X.iloc[test_idx].copy()

            # Cache dates for temporary per-day cross-sectional normalization
            train_dates_series = df[self.date_col].iloc[train_idx]
            test_dates_series = df[self.date_col].iloc[test_idx]

            # Robust scaling on numeric features (excluding categorical columns)
            num_cols = X_train.select_dtypes(include=[np.number]).columns.difference(self.categorical_cols)
            scaler = RobustScaler()
            scaler.fit(X_train[num_cols])
            X_train[num_cols] = scaler.transform(X_train[num_cols])
            X_test[num_cols] = scaler.transform(X_test[num_cols])

            # Cross-sectional z-score normalization by date for selected columns
            cols_to_norm = [col for col in CS_NORM_COLS if col in X_train.columns]

            # Temporary date column for groupby
            X_train["_dt_tmp_"] = train_dates_series.values
            X_test["_dt_tmp_"] = test_dates_series.values

            for col in cols_to_norm:
                # Train: per-day cross-sectional z-score
                X_train[col] = X_train.groupby("_dt_tmp_")[col].transform(
                    lambda x: (x - x.mean()) / (x.std(ddof=0) + 1e-6)
                )
                # Test: per-day cross-sectional z-score
                X_test[col] = X_test.groupby("_dt_tmp_")[col].transform(
                    lambda x: (x - x.mean()) / (x.std(ddof=0) + 1e-6)
                )

            # Remove temporary date column before modeling
            X_train.drop(columns=["_dt_tmp_"], inplace=True)
            X_test.drop(columns=["_dt_tmp_"], inplace=True)

            X_train = X_train.fillna(0)
            X_test = X_test.fillna(0)

            if ONLY_INDUSTRY:
                X_train = X_train[INDUSTRY]
                X_test = X_test[INDUSTRY]

            # Train LightGBM
            model = lgb.LGBMRegressor(**self.lgb_params)
            model.fit(
                X_train,
                y_train,
                categorical_feature=[c for c in self.categorical_cols if c in X_train.columns],
            )

            if RUN_SHAP_ANALYSIS:
                try:
                    # TreeExplainer is typically the fastest for tree-based models
                    explainer = shap.TreeExplainer(model)
                    cur_shap_values = explainer.shap_values(X_test)

                    # SHAP can return a list for multiclass; regression returns an array
                    all_shap_values.append(cur_shap_values)
                    all_X_test_scaled.append(X_test.copy())
                except Exception as e:
                    print(f"⚠️ SHAP computation failed (Split {split_id}): {e}")

            # Accumulate current split feature importance
            cur_feat_names = X_train.columns.tolist()
            imp_gain = model.booster_.feature_importance(importance_type="gain")
            imp_split = model.booster_.feature_importance(importance_type="split")

            if feat_imp_gain_sum is None:
                feat_names_for_imp = cur_feat_names
                feat_imp_gain_sum = imp_gain.astype(float)
                feat_imp_split_sum = imp_split.astype(float)
            else:
                feat_imp_gain_sum += imp_gain
                feat_imp_split_sum += imp_split

            n_splits_for_imp += 1

            preds = model.predict(X_test)
            df.loc[test_idx, f"pred_{self.horizon}"] = preds
            df.loc[test_idx, "split_id"] = split_id

        # Save mean feature importance across splits (gain + split)
        if (feat_imp_gain_sum is not None) and (n_splits_for_imp > 0):
            importance_df = pd.DataFrame(
                {
                    "feature": feat_names_for_imp,
                    "importance_mean_gain": feat_imp_gain_sum / n_splits_for_imp,
                    "importance_mean_split": feat_imp_split_sum / n_splits_for_imp,
                }
            )

            # Sort by mean gain (often more informative in practice)
            importance_df = importance_df.sort_values("importance_mean_gain", ascending=False)

            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            fi_path = OUTPUT_DIR / "feature_importance_mean.csv"
            importance_df.to_csv(fi_path, index=False)

            print(f"📊 Mean feature importance saved to: {fi_path}")
            print("Top 10 features by gain:")
            print(importance_df.head(10))
        else:
            print("⚠️ No feature importance was computed.")

        # Aggregate SHAP outputs across splits (optional)
        if RUN_SHAP_ANALYSIS and all_shap_values:
            print("⏳ Aggregating SHAP results...")
            try:
                merged_shap = np.vstack(all_shap_values)
                merged_X = pd.concat(all_X_test_scaled, axis=0)

                # A) Save SHAP importance ranking
                shap_imp = (
                    pd.DataFrame(
                        {
                            "feature": self.feature_cols,
                            "shap_importance": np.abs(merged_shap).mean(axis=0),
                        }
                    )
                    .sort_values("shap_importance", ascending=False)
                )
                shap_imp.to_csv(OUTPUT_DIR / "feature_importance_shap.csv", index=False)

                # B) Save SHAP summary plot
                import matplotlib.pyplot as plt

                plt.figure(figsize=(12, 10))
                shap.summary_plot(merged_shap, merged_X, show=False)
                plt.title(f"SHAP Summary Plot (Horizon: {self.horizon})")
                plt.tight_layout()
                plt.savefig(OUTPUT_DIR / "shap_summary_plot.png", dpi=150)
                plt.close()

                print(f"✅ SHAP analysis completed. See: {OUTPUT_DIR / 'shap_summary_plot.png'}")
            except Exception as e:
                print(f"⚠️ SHAP aggregation failed: {e}")

        return df


# ===================== 3. Main entrypoint =====================


def main():
    """
    End-to-end workflow:
    1) Load training data
    2) Merge FF5 betas by (date, ticker)
    3) Engineer OHLCV features
    4) Rolling train/predict with LightGBM
    5) Save predictions to OUTPUT_DIR/predictions.csv
    """
    # 1) Load input data
    data_path = TRAIN_DATA_PATH
    df = pd.read_csv(data_path)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    print(f"✅ Raw data loaded: {len(df):,} rows, {df[ID_COL].nunique()} unique tickers")

    # 1.5) Load FF5 betas and merge by date + ticker
    print("⏳ Loading FF5 beta exposures and merging by (date, ticker)...")
    ff5 = pd.read_parquet(FF5_BETA_PATH)

    # Align FF5 column names to the main table
    ff5 = ff5.rename(
        columns={
            "date": DATE_COL,
            "corp_tkr": ID_COL,
        }
    )
    ff5[DATE_COL] = pd.to_datetime(ff5[DATE_COL])

    # Left join to preserve all rows in the main dataset
    df = df.merge(ff5, on=[DATE_COL, ID_COL], how="left")

    # Drop noisy columns (e.g., unnamed) and a few known unused identifiers if present
    drop_cols = [c for c in df.columns if "unnamed" in c.lower()]
    drop_cols += [c for c in ["equity_price", "capiq_id"] if c in df.columns]

    if drop_cols:
        print(f"🧹 Dropping unused columns: {drop_cols}")
        df = df.drop(columns=drop_cols)

    # Fill missing FF5 exposures with 0 (treat as zero exposure when unavailable)
    beta_cols = [c for c in df.columns if c == "alpha" or c.startswith("beta_")]
    if beta_cols:
        df[beta_cols] = df[beta_cols].fillna(0.0)
        print(f"✅ FF5 exposures merged. Columns: {beta_cols}")

    # 2) Generate OHLCV technical features
    print("⏳ Generating OHLCV technical indicator features...")
    fe = OHLCVFeatureEngineer(windows=OHLCV_WINDOWS)
    df_feat = fe.add_features(df)
    print(f"✅ Feature generation completed. Total columns: {len(df_feat.columns)}")

    # 3) Initialize model
    print("⚙️ Initializing the LightGBM forward return prediction model...")
    model = EventReturnModel(
        horizon=HORIZON,
        date_col=DATE_COL,
        id_col=ID_COL,
        close_col=CLOSE_COL,
    )

    # 4) Train & predict
    print("🚀 Starting rolling training and prediction (generating ret_fwd_H and pred_H only)...")
    result_df = model.fit_predict(df_feat)
    print("✅ Model prediction completed.")

    # Preview output
    print(result_df[[DATE_COL, ID_COL, f"ret_fwd_{HORIZON}", f"pred_{HORIZON}", "split_id"]].head())

    # 5) Save results for downstream analysis
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result_path = OUTPUT_DIR / "predictions.csv"
    result_df.to_csv(result_path, index=False)

    print(f"✅ Predictions saved to: {result_path}")


if __name__ == "__main__":
    main()
