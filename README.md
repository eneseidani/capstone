# Machine Learning-Enhanced Portfolio Allocation: A Systematic ETF-Based Strategy

MSc Financial Engineering capstone project that combines machine-learning classification with traditional mean-variance optimization. The research pipeline collects market and macroeconomic data, constructs point-in-time features, trains walk-forward return-direction classifiers, and backtests diversified portfolios net of estimated trading costs.


## Workflow

| Stage | Notebook                               | Purpose                                                                                   | Output folder                                                   |
|-------|----------------------------------------|-------------------------------------------------------------------------------------------|-----------------------------------------------------------------|
| S01   | `S01_get_raw_data.ipynb`               | Download price, macroeconomic, and factor data                                            | `output/AS_OF_DATE/raw_data/`                                   |
| S02   | `S02_prepare_input_data.ipynb`         | Clean raw data, estimate bid–ask spreads and average dollar volume                        | `output/AS_OF_DATE/input_data/`                                 |
| S03   | `S03_construct_features.ipynb`         | Generate asset and macro features, forward-return labels, and market regimes              | `output/AS_OF_DATE/features_data/`                              |
| S04A  | `S04A_return_classification_lr.ipynb`  | Walk-forward Logistic Regression classification and probability screening                 | `output/AS_OF_DATE/classification_results/logistic_regression/` |
| S04B  | `S04B_return_classification_xgb.ipynb` | Walk-forward XGBoost classification and probability screening                             | `output/AS_OF_DATE/classification_results/xgboost/`             |
| S05   | `S05_portfolio_optimization.ipynb`     | Run validation and test portfolio backtests, comparisons and sensitivity analyses         | `output/AS_OF_DATE/portfolio_results/`                          |
| S06   | `S06_return_attribution.ipynb`         | Analyze validation and test portfolio returns using Fama–French five factors and momentum | `output/AS_OF_DATE/return_attribution/`                         |

Execution order: **S01 → S02 → S03 → S04A and S04B → S05 → S06**.

## Investable universe

`Investable_Universe.csv`: 48 risky instruments and one cash proxy. The risky universe includes U.S. and international equities, fixed income, foreign exchange, commodities, and volatility-linked ETFs, as ell as Bitcoin. The cash proxy is the FRED three-month Treasury yield series.

Asset eligibility is point-in-time: portfolios require adequate historical prices and usable transaction-cost inputs. The regime-aware covariance estimator may use up to five years of regime-matched history, but it does **not** require each asset to have five years of data; it falls back to unconditional Ledoit–Wolf when the conditional sample is insufficient.
