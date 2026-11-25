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

Execution order: **S01 → S02 → S03 → S04A, S04B → S05 → S06**.

## Investable universe

`Investable_Universe.csv`: 48 risky instruments and one cash proxy. The risky universe includes U.S. and international equities, fixed income, foreign exchange, commodities, and volatility-linked ETFs, as well as Bitcoin. The cash proxy is the FRED three-month Treasury yield series.

Asset eligibility is point-in-time: portfolios require adequate historical prices and usable transaction-cost inputs. The regime-aware covariance estimator may use up to 5 years of regime-matched history, but it does **not** require each asset to have five years of data; it falls back to unconditional Ledoit–Wolf when the conditional sample is insufficient.


## Setup

Use **Python 3.11**. Keep the notebooks, `config.py`, `transaction_cost_helper.py`, `requirements.txt`, and `Investable_Universe.csv` in the project root. Set `fred_api_key` in `config.py` to a valid [FRED API key](https://fred.stlouisfed.org/docs/api/api_key.html) and review `AS_OF_DATE` before running S01. Do not commit API credentials to version control.

Install dependencies from the project root:

```bash
python -m pip install -r requirements.txt
```

## Running the pipeline

Run the notebooks in the execution order shown above. S04A and S04B may run in either order, but both must finish before S05.

For a first run without saved classifier configurations, set `HYPERPARAMETER_TUNING = True` in `config.py` before running S04A and S04B. This performs validation-based tuning and saves `selected_config.json` in each classifier's output folder. Set it to `False` only when those files are already present and you want to reuse the saved model hyperparameters.

All generated files are stored under `output/<AS_OF_DATE>/`, where `<AS_OF_DATE>` is the date configured in `config.py`. S05 exports validation and test portfolio NAVs, weights, trades, rebalance summaries, and performance summaries. S06 exports `factor_regression_results.csv` and `factor_return_attribution.csv`.

Historical data may be updated or revised by external providers. Trading costs are estimates, and backtest results do not represent live execution or investment advice.
