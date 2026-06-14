# Event-Driven Alpha

**A Machine Learning Approach to Detecting Event-Driven Price Deviations**

> **Disclaimer:** This repository contains research code and methodology only. **Proprietary market and textual data are not included** due to licensing and privacy constraints.

---

## Table of Contents

1. [Background](#1-background)
2. [Introduction & Objectives](#2-introduction--objectives)
3. [Key Definitions](#3-key-definitions)
4. [Methodology](#4-methodology)
5. [Repository Structure](#5-repository-structure)
6. [Getting Started](#6-getting-started)
7. [Evaluation Framework](#7-evaluation-framework)
8. [Limitations & Future Work](#8-limitations--future-work)
9. [References](#9-references)

---

## 1. Background

Corporate events—M&A, earnings guidance, regulatory actions, litigation, leadership changes—can rapidly shift investor perceptions of firm value and risk. Markets do not always price these shocks efficiently at disclosure: reactions may **overreact** or **underreact**, creating short-horizon mispricing that corrects over subsequent days.

This project asks a practical question:

> On an event day, does the market's price reaction deviate from a model-implied short-term reasonable return—suggesting overvaluation or undervaluation?

We build a supervised return-prediction benchmark (LightGBM) using only information available at the close of day *t*, then compare the **realized event-day return** against the **predicted forward return** to construct a **Deviation** signal. Quintile backtests on event-day samples test whether the signal predicts subsequent price corrections.

---

## 2. Introduction & Objectives

### Core Research Question

Does the market's event-day reaction diverge from what a multi-factor ML model would expect over the next *h* trading days?

### Project Objectives

| # | Objective | Description |
|---|-----------|-------------|
| (a) | **Predictive benchmark** | Train LightGBM (and linear regression baseline) to forecast *h*-day forward returns using event, price, and risk features observable at time *t*. |
| (b) | **Deviation metric** | Define `Deviation = R(t) − pred(t+h)` on event days (`event_count > 0`). |
| (c) | **Economic evaluation** | Sort event-day stocks into quintiles by Deviation; measure subsequent return paths and long-short (Q1−Q5) performance. |
| (d) | **Robustness** | Compare models and run ablation studies (full / no-event / event-only / no-industry). |
| (e) | **Interpretability** | Use SHAP and gain-based feature importance to explain non-linear pricing logic. |

### End-to-End Pipeline

```mermaid
flowchart LR
    A[Raw Data] --> B[Data Processing]
    B --> C[Feature Engineering]
    C --> D[Rolling ML Training]
    D --> E[Deviation Signal]
    E --> F[Quintile Backtest]
    F --> G[Ablation & SHAP]
```

---

## 3. Key Definitions

| Term | Definition |
|------|------------|
| **Event Day** | A trading day with at least one recorded corporate event (`event_count > 0`). |
| **Prediction Horizon (*h*)** | Number of forward trading days for the return label. Default: **h = 5**. |
| **Predicted Return** | Model forecast of the *h*-day ahead return, using features available at or before day *t*. |
| **Realized Return R(t)** | Actual return on the event day—the market's immediate reaction. |
| **Deviation** | `R(t) − pred(t+h)`. Large positive values suggest potential overreaction; large negative values suggest underreaction. |
| **Quintile Portfolio** | Event-day stocks sorted into Q1–Q5 by Deviation; Q1 = most negative (underreaction candidates). |
| **Overlapping Holdings** | Portfolio holds *K* equally weighted sub-portfolios initiated over the past *K* days to smooth signal volatility. |

---

## 4. Methodology

### 4.1 Assumptions

- **No look-ahead bias:** All features use information available at or before the close of day *t*.
- **Predictive benchmark:** LightGBM predictions represent a cross-sectionally comparable short-term return benchmark.
- **Event-day conditioning:** Conclusions apply to event days, not general trading days.
- **Execution simplification:** Entry at *t+1* open with 10 bps slippage in selected analyses.

### 4.2 Data Sources *(not shipped in this repo)*

The full study combines:

| Layer | Content | Notes |
|-------|---------|-------|
| **Equity pricing** | Daily OHLCV, market cap | ~400 large-cap U.S. issuers |
| **Credit metrics** | 10Y OAS | Debt-market sentiment |
| **Risk factors** | Fama-French 5-factor betas | Rolling regression alignment |
| **Textual events** | Earnings calls, major developments, news flows | 15+ event categories |
| **Industry context** | Level-2 to Level-4 classifications | Sector aggregation features |

**Expected local inputs** (place under `data/`):

```
data/
├── issuer_data_textual.csv    # Panel: issuer-date features + event dummies
└── ff5_rolling_betas.parquet  # Rolling FF5 betas per issuer-date
```

See [`data/README.md`](data/README.md) for the expected schema.

### 4.3 Feature Engineering

#### 4.3.1 Price–Volume Features

Computed over rolling windows **n ∈ {14, 20, 63}** (~3 weeks, 1 month, 1 quarter):

- **Momentum:** cumulative returns, SMA, SMA deviation
- **Volatility:** realized variance from log returns
- **Liquidity:** volume z-score, dollar-volume deviation
- **RSI:** EWMA-smoothed gains/losses (overbought ≥ 70, oversold ≤ 30)

Implementation: [`src/feature_engineering.py`](src/feature_engineering.py)

#### 4.3.2 Anomaly Features

Anomaly z-score: `z = (r_t − μ) / σ` over a rolling window. Two approaches for systematic vs. idiosyncratic shocks:

1. **Rule-based:** classify as systematic if market proxy is also anomalous
2. **β-adjusted residuals:** anomalies on `ε_t = r^s_t − β_t · r^m_t`

#### 4.3.3 Alternative (Event) Features

| Type | Description | Source notebooks |
|------|-------------|------------------|
| **Statistical textual** | `event_count`, category dummies, rolling counts (5d/10d), cross-sectional transforms | `notebooks/data_processing/02_price_volume_textual_features.ipynb`, `01_news_cleaning_merged.ipynb` |
| **Sentiment** | Topic-specific scores in [-1, 1] via instruction-tuned LLM (Qwen2.5-3B) | `notebooks/data_processing/03_sentiments_factor_pipeline.ipynb`, `src/sentiment_pipeline.py` |

Event categories include M&A, Leadership, Regulatory, Legal, ESG, Guidance, Product Events, and others.

### 4.4 Modeling

Three-stage tuning under a **rolling training framework**:

```mermaid
flowchart TB
    subgraph Stage I
        A1[Fix train window / test window / horizon] --> A2[Sweep LGBM hyperparameters]
        A2 --> A3[Select by mean RankIC / std RankIC]
    end
    subgraph Stage II
        B1[Fix LGBM architecture] --> B2[Sweep horizon / train / test windows]
        B2 --> B3[Random label shuffle test]
        B3 --> B4[Check label overlap / autocorrelation]
    end
    subgraph Stage III
        C1[Fix model + training config] --> C2[Feature importance pruning]
        C2 --> C3[Retain 32 features]
    end
    Stage I --> Stage II --> Stage III
```

#### Rolling Training

At each prediction time *T*, train on `[T − train_window, T]`, predict the test block `[T+1, T+test_window]`, then roll forward by `test_window`.

#### Final Configuration

| Parameter | Value |
|-----------|-------|
| Horizon (*h*) | 5 |
| Train window | 252 |
| Test window | 5 |
| `n_estimators` | 800 |
| `learning_rate` | 0.03 |
| `num_leaves` | 31 |
| `max_depth` | -1 |
| Selected features | 32 (after gain-based pruning) |

#### Baseline

Rolling **linear regression** with multicollinearity filtering, cross-sectional normalization, and training-only scaling—serves as an interpretable "logical anchor."

Scripts: [`src/train.py`](src/train.py), [`src/tuning.py`](src/tuning.py), [`src/config.py`](src/config.py)

### 4.5 Deviation Signal Construction

On event days only:

```
Deviation_i(t) = R_i(t) − pred_i(t+h)
```

- **R_i(t):** realized event-day return (market reaction)
- **pred_i(t+h):** LightGBM benchmark (multi-dimensional features)

**Portfolio rules:**

- Rank event-day stocks into quintiles by Deviation
- Signal after close on *t*; enter at *t+1* open (10 bps slippage)
- Overlapping *K*-day holding for smoother NAV

Evaluation: [`src/factor_eval.py`](src/factor_eval.py)

### 4.6 Model Comparison & Ablation

| Model / Variant | Role |
|-----------------|------|
| **Linear Regression** | Interpretable baseline; coefficient stability checks |
| **LightGBM (Full)** | Primary benchmark; balances reversal logic with event context |
| **No-Event Features** | Isolates price-volume reversal alpha |
| **Event-Only Features** | Tests standalone event predictive power |
| **No-Industry / Only-Industry** | Industry ablation via `config.py` flags |

Interpretability: SHAP analysis on LightGBM (`RUN_SHAP_ANALYSIS = True` in config).

### 4.7 Factor Enhancement Trials

Optional filters layered on Q1:

| Variant | Logic |
|---------|-------|
| `dev_vol_ts` | Time-series z-score of return vs. 20-day volatility |
| `dev_vol_cs` | Cross-sectional top-30% risk-adjusted return |
| `dev_ret_ts` / `dev_ret_cs` | Absolute return thresholds |

Enhancement trials generally reduced sample breadth and did not improve risk-adjusted performance vs. the base Deviation signal.

---

## 5. Repository Structure

```
event-driven-alpha/
├── README.md
├── requirements.txt
├── .gitignore
├── data/                              # User-provided inputs (gitignored)
│   └── README.md                      # Expected schema (no raw data)
├── src/                               # Core Python pipeline
│   ├── config.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── model_eval.py
│   ├── factor_eval.py
│   ├── tuning.py
│   └── sentiment_pipeline.py
└── notebooks/
    ├── data_processing/
    │   ├── 01_news_cleaning_merged.ipynb
    │   ├── 02_price_volume_textual_features.ipynb
    │   └── 03_sentiments_factor_pipeline.ipynb
    └── modeling/
        ├── 01_linear_regression_horizon1.ipynb
        ├── 02_lr_factor_analysis.ipynb
        └── 03_lr_model_eval.ipynb
```

---

## 6. Getting Started

### Prerequisites

- Python 3.9+
- CUDA GPU (optional; required for sentiment pipeline)

### Installation

```bash
git clone <your-repo-url>
cd "Event Driven Alpha"
pip install -r requirements.txt
```

### Prepare Data

1. Obtain licensed equity, credit, and news/event data from your provider.
2. Build `data/issuer_data_textual.csv` and `data/ff5_rolling_betas.parquet` following [`data/README.md`](data/README.md).
3. Or run the data-processing notebooks in `notebooks/data_processing/` (update paths in the first cells if needed).

### Run the Pipeline

Run all commands from the **repository root**:

```bash
# 1. Train rolling LightGBM and write predictions
python src/train.py

# 2. Evaluate prediction IC / RankIC
python src/model_eval.py

# 3. Run Deviation factor quintile analysis
python src/factor_eval.py
```

### Configuration

Edit [`src/config.py`](src/config.py):

```python
TRAIN_DATA_PATH = PROJECT_ROOT / "data" / "issuer_data_textual.csv"
HORIZON = 5
TRAIN_WINDOW = 252
TEST_WINDOW = 5
DROP_INDUSTRY = False   # Ablation: exclude industry features
ONLY_INDUSTRY = False   # Ablation: industry-only features
RUN_SHAP_ANALYSIS = True
```

Outputs are written to `outputs_final_no_industry/<experiment_name>/` at the repo root.

---

## 7. Evaluation Framework

### Prediction Quality (model selection)

- Daily **IC** and **RankIC** between `pred_h` and `ret_fwd_h`
- **Pred_Stability** = mean(monthly RankIC) / std(monthly RankIC)
- Random label shuffle test to detect leakage

### Factor Quality (investment logic)

- Quintile cumulative return curves (event days only)
- Long-short Q1−Q5: annualized return, Sharpe, max drawdown, t-stat, win rate
- Minimum 30 stocks per day threshold for cross-sectional statistics

---

## 8. Limitations & Future Work

- **Turnover & capacity:** Transaction costs and strategy capacity not fully modeled.
- **Return decomposition:** Sentiment residuals via regression-based decomposition remain unexplored.
- **Cross-event interactions:** Event clusters (e.g., insider buying + earnings surprise) not jointly modeled.
- **Macro conditioning:** Deviation signal not yet adapted to rate/liquidity regimes.

---

## 9. References

- Gu, S., Kelly, B., & Xiu, D. (2020). *Empirical Asset Pricing via Machine Learning.* Review of Financial Studies.
- Ke, R., Kelly, B., & Xiu, D. (2019). *Predicting Returns with Text Data.*
- Da, Z., Engelberg, J., & Gao, P. (2011). *In Search of Attention.* Journal of Finance.
- Al-Sulaiman, T. (2022). *Predicting Reactions to Anomalies in Stock Movements Using Deep Learning.*

---

## License

Code is provided for academic and research purposes. **Data is not included** and must be sourced separately under appropriate licenses. Contact the authors for questions about reproduction with alternative datasets.
