"""
Deviation factor analysis for horizon return prediction outputs.

This script analyzes the output of the rolling prediction pipeline:
    outputs/<EXP_NAME>/predictions.csv

It produces analysis artifacts under:
    outputs/<EXP_NAME>/analysis/

Key reminders (path & parameters must align):
- This script assumes the prediction file is located at:
      outputs/<EXP_NAME>/predictions.csv
- EXP_NAME generation (or the explicit BASE_ROOT you set) must match the training script output.
  If any key parameters differ, the script may fail to locate the prediction file.

Outputs (examples):
- *_ic_detailed_summary.csv: detailed IC statistics by future window
- *_conditional_means.csv: conditional mean returns given factor sign
- plots/*.png: RankIC time-series diagnostics and quintile cumulative return plots
- *_performance.csv: long-short performance metrics for quintile strategies
"""

# ===== Standard library imports =====
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ===== Third-party imports =====
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ===== Local imports =====
from config import (
    CLOSE_COL,
    DATE_COL,
    FF5_REG_WINDOW_DAYS,
    FUTURE_WINDOWS,
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

BP = 10
slip = BP / 10_000
HORIZON = 5
PRED_COL = f"pred_{HORIZON}"
ACTUAL_COL = "ret_fwd_1"

# ======================================================================

# Prediction file path (must align with the training output)
BASE_ROOT = (
    "outputs_final_only_industry/"
    "h5_tr252_te5_win14-20-63_lr0.03_ne800_md-1_nleaves31_ss0.8_cs0.8_ff5w120"
)
PREDICTION_FILE = os.path.join(BASE_ROOT, "predictions.csv")
DATA_PATH = Path(PREDICTION_FILE)

# Analysis output directory (stored under the experiment folder)
OUTPUT_DIR = Path(os.path.join(BASE_ROOT, "analysis"))


class FactorBuilder:
    """
    Build post-event return columns and factor variants based on deviation and filters.
    """

    def __init__(self, horizon: int):
        """
        Parameters:
            horizon (int): Prediction horizon (kept for naming/consistency).
        """
        self.horizon = horizon
        self.ret_col = ACTUAL_COL

    def add_post_window_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute post-event forward returns: P_{t+k} / P_t - 1 for each k in FUTURE_WINDOWS.

        Parameters:
            df (pd.DataFrame): Input DataFrame with at least [ID_COL, DATE_COL, CLOSE_COL].

        Returns:
            pd.DataFrame: DataFrame with added columns ret_post_{k}.
        """
        df = df.copy().sort_values([ID_COL, DATE_COL])
        g = df.groupby(ID_COL, group_keys=False)
        price = df[CLOSE_COL]
        for k in FUTURE_WINDOWS:
            price_k = g[CLOSE_COL].shift(-k)
            df[f"ret_post_{k}"] = price_k / price - 1
        return df

    def add_volume_ts_cs_deviation_factors(
        self, df: pd.DataFrame, ts_threshold=0.7, cs_threshold=0.7
    ) -> pd.DataFrame:
        """
        Build deviation factors filtered by volume percentiles.

        - Time-series filter: volume >= rolling quantile (per stock)
        - Cross-sectional filter: volume >= daily market percentile

        Parameters:
            df (pd.DataFrame): Input DataFrame with 'volume' and 'deviation'.
            ts_threshold (float): Rolling quantile threshold in time-series dimension.
            cs_threshold (float): Percentile threshold in cross-sectional dimension.

        Returns:
            pd.DataFrame: DataFrame with added columns dev_vol_ts and dev_vol_cs.
        """
        df = df.copy().sort_values([ID_COL, DATE_COL])

        # Time-series filter: volume >= 20-day rolling quantile (per stock)
        vol_ts_q = (
            df.groupby(ID_COL)["volume"]
            .rolling(window=20, min_periods=10)
            .quantile(ts_threshold)
            .reset_index(level=0, drop=True)
        )
        df["dev_vol_ts"] = np.where(df["volume"] >= vol_ts_q, df["deviation"], np.nan)

        # Cross-sectional filter: volume >= daily market percentile
        vol_cs_pct = df.groupby(DATE_COL)["volume"].rank(pct=True)
        df["dev_vol_cs"] = np.where(vol_cs_pct >= cs_threshold, df["deviation"], np.nan)

        return df

    def add_return_ts_cs_percentile_factors(
        self, df: pd.DataFrame, ts_threshold=0.7, cs_threshold=0.7
    ) -> pd.DataFrame:
        """
        Build return-based filtered factors using return percentiles.

        Parameters:
            df (pd.DataFrame): Input DataFrame with self.ret_col present.
            ts_threshold (float): Rolling quantile threshold in time-series dimension.
            cs_threshold (float): Percentile threshold in cross-sectional dimension.

        Returns:
            pd.DataFrame: DataFrame with added filtered return columns.
        """
        df = df.copy().sort_values([ID_COL, DATE_COL])

        # Time-series filter (based on t-day return)
        ret_ts_q = (
            df.groupby(ID_COL)[self.ret_col]
            .rolling(window=20, min_periods=10)
            .quantile(ts_threshold)
            .reset_index(level=0, drop=True)
        )
        df[f"{self.ret_col}_ts_pct_filter"] = np.where(
            df[self.ret_col] >= ret_ts_q, df[self.ret_col], np.nan
        )

        # Cross-sectional filter (based on t-day return)
        ret_cs_pct = df.groupby(DATE_COL)[self.ret_col].rank(pct=True)
        df[f"{self.ret_col}_cs_pct_filter"] = np.where(
            ret_cs_pct >= cs_threshold, df[self.ret_col], np.nan
        )

        return df


class FactorAnalyzer:
    """
    Analyze factor efficacy via conditional means, IC metrics, time-series diagnostics,
    and overlapping-hold quintile portfolio returns.
    """

    def __init__(self, output_dir: Path):
        """
        Parameters:
            output_dir (Path): Directory to write analysis artifacts.
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plot_dir = self.output_dir / "plots"
        self.plot_dir.mkdir(parents=True, exist_ok=True)

    def deviation_conditional_means(self, df: pd.DataFrame, dev_col: str) -> pd.DataFrame:
        """
        Compute conditional mean post-window returns given factor sign.

        Parameters:
            df (pd.DataFrame): Input DataFrame with dev_col and ret_post_{k}.
            dev_col (str): Factor column name.

        Returns:
            pd.DataFrame: Rows indexed by future_window with positive/negative sample counts and means.
        """
        df_eval = df[df["event_count"] != 0].copy()
        rows = []
        for k in FUTURE_WINDOWS:
            ret_col = f"ret_post_{k}"
            sub = df_eval[[dev_col, ret_col]].dropna()
            pos = sub[sub[dev_col] > 0][ret_col]
            neg = sub[sub[dev_col] < 0][ret_col]
            rows.append(
                {
                    "future_window": k,
                    "n_pos": len(pos),
                    "mean_ret_pos": pos.mean() if len(pos) >= MIN_OBS_FOR_MEAN else np.nan,
                    "n_neg": len(neg),
                    "mean_ret_neg": neg.mean() if len(neg) >= MIN_OBS_FOR_MEAN else np.nan,
                }
            )
        return pd.DataFrame(rows).set_index("future_window")

    def compute_detailed_ic_metrics(self, df: pd.DataFrame, dev_col: str) -> pd.DataFrame:
        """
        Compute detailed IC statistics for daily RankIC series:
        - Mean, Std, Annualized IR, Positive ratio, Significant ratio, Number of days

        Parameters:
            df (pd.DataFrame): Input DataFrame with dev_col and ret_post_{k}.
            dev_col (str): Factor column name.

        Returns:
            pd.DataFrame: Rows indexed by future_window with detailed IC metrics.
        """
        df_eval = df[df["event_count"] != 0].copy()
        results = []

        for k in FUTURE_WINDOWS:
            ret_col = f"ret_post_{k}"

            def daily_corr(group):
                valid = group[[dev_col, ret_col]].dropna()
                if len(valid) < MIN_STOCKS_PER_DAY:
                    return np.nan
                return valid[dev_col].corr(valid[ret_col], method="spearman")

            ic_series = df_eval.groupby(DATE_COL).apply(daily_corr).dropna()

            if len(ic_series) > 0:
                mean_ic = ic_series.mean()
                std_ic = ic_series.std()

                ic_ir = (mean_ic / std_ic * np.sqrt(252)) if std_ic != 0 else np.nan
                pos_ratio = (ic_series > 0).mean()
                sig_ratio = (ic_series.abs() > 0.02).mean()  # Threshold: |IC| > 0.02

                results.append(
                    {
                        "future_window": k,
                        "IC_Mean": mean_ic,
                        "IC_Std": std_ic,
                        "IC_IR_Annual": ic_ir,
                        "IC_Positive_Ratio": pos_ratio,
                        "IC_Significant_Ratio": sig_ratio,
                        "Num_Days": len(ic_series),
                    }
                )

        return pd.DataFrame(results).set_index("future_window")

    def compute_and_plot_ic_ts(self, df: pd.DataFrame, dev_col: str):
        """
        Compute daily IC / RankIC time series, save the series, and generate plots.

        Parameters:
            df (pd.DataFrame): Input DataFrame with dev_col and ret_post_{k}.
            dev_col (str): Factor column name.
        """
        df_eval = df[df["event_count"] != 0].copy()

        for k in FUTURE_WINDOWS:
            ret_col = f"ret_post_{k}"

            def get_daily_metrics(group):
                valid = group[[dev_col, ret_col]].dropna()
                if len(valid) < MIN_STOCKS_PER_DAY:
                    return pd.Series({"ic": np.nan, "rank_ic": np.nan})

                ic = valid[dev_col].corr(valid[ret_col], method="pearson")
                rank_ic = valid[dev_col].corr(valid[ret_col], method="spearman")
                return pd.Series({"ic": ic, "rank_ic": rank_ic})

            ic_ts = df_eval.groupby(DATE_COL).apply(get_daily_metrics).dropna()

            if len(ic_ts) == 0:
                continue

            ic_ts.to_csv(self.output_dir / f"{dev_col}_window_{k}_ic_ts.csv")

            # Plot RankIC over time (the helper uses daily_res with sample_count/rank_ic)
            # This function is kept as-is; plotting is handled in compute_and_plot_all().
            # self._generate_ic_plot(ic_ts["rank_ic"], dev_col, k, "RankIC")

    def _generate_ic_plot(self, daily_metrics: pd.DataFrame, dev_col: str, window: int):
        """
        Generate a time-series diagnostic plot.

        Parameters:
            daily_metrics (pd.DataFrame): Must include columns ['rank_ic', 'sample_count'] indexed by DATE_COL.
            dev_col (str): Factor column name.
            window (int): Future window used for title/filename.
        """
        daily_metrics = daily_metrics.sort_index()

        fig, ax1 = plt.subplots(figsize=(15, 7))

        # 1) Sample count bar chart (right axis)
        ax2 = ax1.twinx()
        ax2.bar(
            daily_metrics.index,
            daily_metrics["sample_count"],
            color="gray",
            alpha=0.2,
            width=1.0,
            label="Sample Count",
        )
        ax2.set_ylabel("Effective Sample Count", color="gray", fontsize=12)
        ax2.axhline(
            y=30,
            color="orange",
            linestyle=":",
            alpha=0.8,
            label="Min Sample Threshold (30)",
        )

        # 2) RankIC line (left axis)
        ax1.plot(
            daily_metrics.index,
            daily_metrics["rank_ic"],
            color="#1f77b4",
            linewidth=1.5,
            marker="o",
            markersize=3,
            label="Daily RankIC (Spearman)",
        )

        ax1.axhline(y=0, color="black", linewidth=1)
        mean_ic = daily_metrics["rank_ic"].mean()
        ax1.axhline(y=mean_ic, color="red", linestyle="--", label=f"Mean IC: {mean_ic:.4f}")
        ax1.set_ylabel("RankIC Value", fontsize=12)
        ax1.set_ylim(-1.05, 1.05)

        # 3) Format x-axis (dates)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45)

        # 4) Title & legend
        plt.title(
            f"Factor Consistency Check: {dev_col} (Window: {window}d)\n"
            f"[Gaps indicate < 30 samples]",
            fontsize=14,
        )

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

        ax1.grid(True, alpha=0.3)
        plt.tight_layout()

        save_path = self.plot_dir / f"{dev_col}_w{window}_ts_analysis.png"
        plt.savefig(save_path, dpi=150)
        plt.close()

    def compute_and_plot_all(self, df: pd.DataFrame, dev_col: str):
        """
        Unified routine to compute daily RankIC with sample counts and generate plots.

        Parameters:
            df (pd.DataFrame): Input DataFrame with dev_col and ret_post_{k}.
            dev_col (str): Factor column name.
        """
        df_eval = df[df["event_count"] != 0].copy()

        for k in FUTURE_WINDOWS:
            ret_col = f"ret_post_{k}"

            def calc_daily(group):
                valid = group[[dev_col, ret_col]].dropna()
                n = len(valid)
                if n < 30:
                    return pd.Series({"rank_ic": np.nan, "sample_count": n})

                ric = valid[dev_col].corr(valid[ret_col], method="spearman")
                return pd.Series({"rank_ic": ric, "sample_count": n})

            daily_res = df_eval.groupby(DATE_COL).apply(calc_daily)

            if not daily_res.empty:
                self._generate_ic_plot(daily_res, dev_col, k)

    def compute_quintile_overlapping_returns(
        self, df: pd.DataFrame, signal_col: str, k_holding: int = 5
    ):
        """
        Compute quintile overlapping returns.

        Entry logic:
        - Signal at day T, enter at T+1 open and measure intraday return to close:
              Close_{T+1} / Open_{T+1} - 1 (with slippage)

        Holding logic:
        - For days 2..k_holding after entry, measure daily close-to-close returns.

        Parameters:
            df (pd.DataFrame): Input DataFrame with required OHLC and signal columns.
            signal_col (str): Factor signal column used to form quintiles.
            k_holding (int): Holding period for overlapping portfolios.

        Returns:
            pd.DataFrame: Date-indexed DataFrame with columns Q1..Q5 daily overlapping returns.
        """
        df = df.sort_values([ID_COL, DATE_COL]).copy()

        # Precompute returns
        df["ret_intraday"] = df[CLOSE_COL] / (df["open"] * (1 + slip)) - 1
        df["ret_daily"] = df.groupby(ID_COL)[CLOSE_COL].pct_change()

        # Daily cross-sectional quintile assignment (signal day T)
        def assign_quintiles(group):
            valid_signal = group[signal_col].dropna()
            if len(valid_signal) >= 5:
                group.loc[valid_signal.index, "quintile"] = pd.qcut(
                    valid_signal.rank(method="first"), 5, labels=[1, 2, 3, 4, 5]
                )
            else:
                group["quintile"] = np.nan
            return group

        df_eval = df[df[signal_col].notna()].copy()
        df_eval = df_eval.groupby(DATE_COL, group_keys=False).apply(assign_quintiles)

        df = df.merge(df_eval[[ID_COL, DATE_COL, "quintile"]], on=[ID_COL, DATE_COL], how="left")

        all_dates = sorted(df[DATE_COL].unique())

        quintile_map = (
            df.dropna(subset=["quintile"])
            .groupby([DATE_COL, "quintile"])[ID_COL]
            .apply(list)
            .to_dict()
        )

        intraday_map = df.set_index([DATE_COL, ID_COL])["ret_intraday"].to_dict()
        daily_map = df.set_index([DATE_COL, ID_COL])["ret_daily"].to_dict()

        results = []

        for i, today in enumerate(all_dates):
            daily_q_rets = {f"Q{q}": 0.0 for q in range(1, 6)}
            found_any_signal = False

            # Overlapping logic: check signals from the past k_holding days
            for lookback in range(1, k_holding + 1):
                sig_date_idx = i - lookback
                if sig_date_idx < 0:
                    continue

                sig_date = all_dates[sig_date_idx]

                for q in range(1, 6):
                    stocks = quintile_map.get((sig_date, q), [])
                    if not stocks:
                        continue

                    stock_rets = []
                    for s in stocks:
                        if lookback == 1:
                            res = intraday_map.get((today, s))
                        else:
                            res = daily_map.get((today, s))

                        if pd.notna(res):
                            stock_rets.append(res)

                    if stock_rets:
                        daily_q_rets[f"Q{q}"] += np.mean(stock_rets) / k_holding
                        found_any_signal = True

            if found_any_signal:
                daily_q_rets["date"] = today
                results.append(daily_q_rets)

        return pd.DataFrame(results).set_index("date")

    def plot_quintile_cumulative_returns(self, return_df: pd.DataFrame, factor_name: str, k_holding: int):
        """
        Plot cumulative returns by quintiles.

        Parameters:
            return_df (pd.DataFrame): Output of compute_quintile_overlapping_returns (Q1..Q5).
            factor_name (str): Factor name used in titles/filenames.
            k_holding (int): Holding period used for titles/filenames.
        """
        if return_df.empty:
            return

        cum_rets = (1 + return_df.fillna(0)).cumprod()

        plt.figure(figsize=(12, 6))

        colors = ["#d62728", "#ff7f0e", "#7f7f7f", "#1f77b4", "#003f5c"]

        for i, col in enumerate(["Q1", "Q2", "Q3", "Q4", "Q5"]):
            if col not in cum_rets.columns:
                continue

            linewidth = 2.5 if col == "Q1" else 1.5
            alpha = 1.0 if col in ["Q1", "Q5"] else 0.7

            plt.plot(
                cum_rets.index,
                cum_rets[col],
                label=f"{col} (Aggressive)" if col == "Q1" else col,
                color=colors[i],
                linewidth=linewidth,
                alpha=alpha,
            )

        plt.title(
            f"Cumulative Returns by {factor_name} Quintiles\n"
            f"(Hold={k_holding}d, Q1=Most Negative Deviation)",
            fontsize=14,
        )
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Net Value (Base 1.0)", fontsize=12)
        plt.axhline(y=1.0, color="black", linestyle="--", alpha=0.5)

        plt.legend(loc="upper left", fontsize=10)
        plt.grid(True, alpha=0.2)

        plt.gcf().autofmt_xdate()
        plt.tight_layout()

        save_path = self.plot_dir / f"quintile_returns_{factor_name}_h{k_holding}.png"
        plt.savefig(save_path, dpi=150)
        print(f"✅ Quintile return plot saved to: {save_path}")
        plt.close()

    def compute_strategy_performance(self, return_df: pd.DataFrame, factor_name: str):
        """
        Compute long-short (Q1 - Q5) performance metrics.

        Parameters:
            return_df (pd.DataFrame): Quintile return DataFrame (Q1..Q5).
            factor_name (str): Factor name used in output filenames.
        """
        if return_df.empty:
            print("⚠️ Return series is empty; cannot compute metrics.")
            return

        ls_ret = return_df["Q1"] - return_df["Q5"]
        cum_ls_ret = (1 + ls_ret.fillna(0)).cumprod()

        mean_ret = ls_ret.mean()
        std_ret = ls_ret.std()
        n_days = len(ls_ret)

        t_stat = (mean_ret / (std_ret / np.sqrt(n_days))) if std_ret != 0 and n_days > 0 else 0

        ann_ret = mean_ret * 252
        ann_vol = std_ret * np.sqrt(252)
        sharpe = (ann_ret / ann_vol) if ann_vol != 0 else 0
        win_rate = (ls_ret > 0).mean()

        rolling_max = cum_ls_ret.expanding().max()
        drawdown = (cum_ls_ret - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        print("\n" + "🚀" + "=" * 48)
        print(f"Factor Strategy Performance Report: {factor_name}")
        print("-" * 50)
        print(f"Backtest days:        {n_days}")
        print(f"LS daily mean:        {mean_ret:.4%}")
        print(f"LS annual return:     {ann_ret:.2%}")
        print(f"LS annual Sharpe:     {sharpe:.2f}")
        print(f"LS max drawdown:      {max_drawdown:.2%}")
        print(f"LS win rate (daily):  {win_rate:.2%}")
        print(
            f"LS t-stat:            {t_stat:.4f} "
            + ("⭐ Significant" if abs(t_stat) > 2 else "✖️ Not significant")
        )
        print("=" * 50 + "\n")

        metrics = {
            "Factor": factor_name,
            "Daily_Mean": mean_ret,
            "Ann_Return": ann_ret,
            "Ann_Sharpe": sharpe,
            "Max_Drawdown": max_drawdown,
            "T_Stat": t_stat,
            "Win_Rate": win_rate,
            "Days": n_days,
        }
        save_file = self.output_dir / f"{factor_name}_performance.csv"
        pd.DataFrame([metrics]).to_csv(save_file, index=False)


def main():
    # 1) Load data
    df = pd.read_csv(PREDICTION_FILE, parse_dates=[DATE_COL])
    if "split_id" in df.columns:
        df = df[df["split_id"] > 0].copy()

    # Compute ACTUAL_COL on the fly (kept as-is)
    df[ACTUAL_COL] = df.groupby(ID_COL)["close"].pct_change()
    df["deviation"] = df[ACTUAL_COL] - df[PRED_COL]

    # 2) Factor construction
    builder = FactorBuilder(HORIZON)
    df = builder.add_post_window_returns(df)
    df = builder.add_volume_ts_cs_deviation_factors(df)
    df = builder.add_return_ts_cs_percentile_factors(df)

    # 3) Analysis & export
    analyzer = FactorAnalyzer(OUTPUT_DIR)
    target_cols = [
        "deviation",
        "dev_vol_ts",
        "dev_vol_cs",
        f"{ACTUAL_COL}_ts_pct_filter",
        f"{ACTUAL_COL}_cs_pct_filter",
    ]

    for col in target_cols:
        if col not in df.columns:
            continue

        print(f"📊 Computing detailed factor metrics: {col}")

        ic_metrics = analyzer.compute_detailed_ic_metrics(df, col)
        ic_metrics.to_csv(OUTPUT_DIR / f"{col}_ic_detailed_summary.csv")

        cond_means = analyzer.deviation_conditional_means(df, col)
        cond_means.to_csv(OUTPUT_DIR / f"{col}_conditional_means.csv")

        analyzer.compute_and_plot_all(df, col)

        return_test = analyzer.compute_quintile_overlapping_returns(df, col)
        analyzer.plot_quintile_cumulative_returns(return_test, col, FUTURE_WINDOWS[1])

        analyzer.compute_strategy_performance(return_test, col)

    print(f"✅ Analysis completed. Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
