# Data Directory

**No proprietary data is included in this repository.**

Place your locally sourced datasets here before running the modeling pipeline.

## Required Files

### `issuer_data_textual.csv`

Issuer–date panel with at minimum:

| Column group | Examples |
|--------------|----------|
| Keys | `dt`, `corp_tkr` |
| Prices | `open`, `high`, `low`, `close`, `volume` |
| Fundamentals | `mv`, `10y_oas` |
| Industry | `ml_industry_lvl_2`, `ml_industry_lvl_3`, `ml_industry_lvl_4` |
| Event dummies | `m_a`, `leadership`, `regulatory`, `legal`, `esg`, `guidance`, ... |
| Event aggregates | `event_count`, `roll5d_events_count`, `roll10d_events_count` |
| Industry rolls | `ml_industry_lvl_*_roll0d`, `_roll5d`, `_roll10d` |
| PCA | `pca1`, `pca2`, `pca3` |
| Sentiment (optional) | `*_sent` columns per event category |

### `ff5_rolling_betas.parquet`

Rolling Fama-French 5-factor loadings per issuer-date:

- `alpha`, `beta_mkt_rf`, `beta_smb`, `beta_hml`, `beta_rmw`, `beta_cma`

## Building the Panel

Use the notebooks under:

```
Final_src/Code/Data Processing&Alternative feature creation/
```

Update all hard-coded Colab / Google Drive paths to point to your local `data/` directory.

## Privacy Notice

Do not commit files containing real ticker symbols, entity IDs, or licensed vendor content to a public repository.
