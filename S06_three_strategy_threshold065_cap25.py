"""
S06 THREE-STRATEGY SYSTEM
========================

One-file implementation of the Capstone portfolio allocation experiment.

Strategies:
    1. XGBoost-selected universe: every ticker with P(positive 60d return) >= 0.65.
       No Top-K restriction.
    2. Full Universe: all risky tickers in Investable_Universe.csv.
    3. SPY buy-and-hold: common benchmark, bought at the first execution open.

The XGBoost probabilities are produced by the S04B notebook from MAIN:
    classification_results/xgboost/all_test_predictions.csv

The allocation code in this file consolidates the direct-utility S06 engine and
the regime-aware Ledoit-Wolf estimator. It does NOT import another S06 module.

External project modules intentionally retained:
    - config.py
    - transaction_fee_helper.py

Run from the repository root:
    python S06_three_strategy_threshold065_cap25.py

The script reruns both portfolio strategies through the same allocation engine,
same dates, same market data, same risk-free series, and same transaction fee
helper. It does not reuse a previously calculated Full Universe NAV.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

import config
from transaction_fee_helper import calculate_transaction_fees, load_fee_context


# =============================================================================
# GLOBAL SETTINGS
# =============================================================================

LOG = logging.getLogger("S06_THREE_STRATEGY")

CONFIGURATION = "04_RegimeAware_LedoitWolf"
XGB_STRATEGY = "XGBoost_Probability_0.65"
FULL_STRATEGY = "Full_Universe"
SPY_STRATEGY = "SPY_Buy_And_Hold"

CASH = "DGS3MO"
MAX_WEIGHT = 0.25
MAX_CASH_WEIGHT = 0.80
MIN_RISKY_WEIGHT = 1.0 - MAX_CASH_WEIGHT

HORIZON = 60
EXPECTED_RETURN_LOOKBACK = 120
COVARIANCE_LOOKBACK = 252
TRADING_DAYS = 252
DEFAULT_RISK_AVERSION = 3.0

# Exact 20-date calendar in the currently inspected MAIN S04B output.
MAIN_REBALANCE_DATES = [
    "2022-01-03",
    "2022-03-30",
    "2022-06-27",
    "2022-09-21",
    "2022-12-15",
    "2023-03-15",
    "2023-06-09",
    "2023-09-06",
    "2023-11-30",
    "2024-02-28",
    "2024-05-23",
    "2024-08-20",
    "2024-11-13",
    "2025-02-12",
    "2025-05-09",
    "2025-08-06",
    "2025-10-30",
    "2026-01-28",
    "2026-04-24",
    "2026-07-22",
]

SOURCE_PROVENANCE = {
    "S04B_notebook": {
        "branch": "main",
        "path": "S04B_return_classification_xgb.ipynb",
        "git_blob_sha": "779fd577abd2be08ad10513e24f8515a5d05ec89",
    },
    "S04B_probability_export": {
        "branch": "main",
        "path": "classification_results/xgboost/all_test_predictions.csv",
        "git_blob_sha": "2529e19356af42b1d618aedaf07460baea1f8f3b",
    },
    "S04_liquidity_data_step": {
        "branch": "feature/S05_predict_forward_return",
        "path": "S04_add_liqudity_data.ipynb",
        "git_blob_sha": "eb81c84ce8d5c0f0d4ed6a0df5605ff82cef216f",
        "note": "The existing Yahoo OHLCV files must contain the Spread and ADV_Dollars columns created by this step.",
    },
    "transaction_fee_helper": {
        "branch": "feature/S05_predict_forward_return",
        "path": "transaction_fee_helper.py",
        "git_blob_sha": "dbf83654ac244c22efcd3381e2c80514e31a3b34",
    },
    "config": {
        "branch": "main",
        "path": "config.py",
    },
}


# =============================================================================
# COMMAND-LINE CONFIGURATION
# =============================================================================

def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regime-aware Ledoit-Wolf: XGBoost vs Full Universe vs SPY",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("classification_results/xgboost/all_test_predictions.csv"),
        help="Complete Date/Ticker/Probability/Rank export from the MAIN S04B notebook.",
    )
    parser.add_argument(
        "--universe",
        type=Path,
        default=Path(config.UNIVERSE_FILE),
        help="Investable universe CSV.",
    )
    parser.add_argument(
        "--macro",
        type=Path,
        default=Path(config.INPUT_DATA_DIR) / "fred" / "DGS3MO.csv",
        help="DGS3MO annualized yield CSV with Date and DGS3MO columns.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("portfolio_results/S06_threshold065_cap25"),
        help="Root output directory.",
    )
    parser.add_argument("--threshold", type=float, default=0.65)
    parser.add_argument("--max-weight", type=float, default=0.25,
                        help="Maximum allocation weight for each risky asset.")
    parser.add_argument("--gamma", type=float, default=DEFAULT_RISK_AVERSION)
    parser.add_argument("--direct-starts", type=int, default=12)
    parser.add_argument("--initial-capital", type=float, default=100000.0)

    parser.add_argument("--regime-lookback", type=int, default=756)
    parser.add_argument("--regime-vol-window", type=int, default=21)
    parser.add_argument("--regime-threshold-window", type=int, default=252)
    parser.add_argument("--regime-min-days", type=int, default=90)
    parser.add_argument("--regime-condition-weight", type=float, default=0.50)

    parser.add_argument(
        "--allow-different-dates",
        action="store_true",
        help="Allow a regenerated S04B export to use a calendar other than the inspected 20 main dates.",
    )
    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> None:
    if not 0.50 <= args.threshold <= 1.00:
        raise ValueError("--threshold must be between 0.50 and 1.00.")
    if not np.isfinite(args.max_weight) or not MIN_RISKY_WEIGHT <= args.max_weight <= 1.0:
        raise ValueError("--max-weight must be finite and in [0.20, 1.00] so a single eligible asset remains feasible.")
    if not np.isfinite(args.gamma) or args.gamma <= 0:
        raise ValueError("--gamma must be finite and positive.")
    if args.direct_starts < 1:
        raise ValueError("--direct-starts must be at least 1.")
    if not np.isfinite(args.initial_capital) or args.initial_capital <= 0:
        raise ValueError("--initial-capital must be finite and positive.")
    if args.regime_lookback < 252:
        raise ValueError("--regime-lookback must be at least 252.")
    if args.regime_vol_window < 5:
        raise ValueError("--regime-vol-window must be at least 5.")
    if args.regime_threshold_window < 30:
        raise ValueError("--regime-threshold-window must be at least 30.")
    if args.regime_min_days < 30:
        raise ValueError("--regime-min-days must be at least 30.")
    if not 0.0 <= args.regime_condition_weight <= 1.0:
        raise ValueError("--regime-condition-weight must lie between 0 and 1.")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# =============================================================================
# UNIVERSE AND XGBOOST INPUT
# =============================================================================

def read_universe(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Investable universe not found: {path}")

    universe = pd.read_csv(path)
    required = {"Ticker", "Asset Class"}
    missing = required.difference(universe.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    tickers = universe["Ticker"].astype(str).str.strip().tolist()
    if any(not ticker or ticker.lower() == "nan" for ticker in tickers):
        raise ValueError("Investable universe contains blank ticker values.")
    if len(tickers) != len(set(tickers)):
        raise ValueError("Investable universe contains duplicate ticker values.")
    if CASH not in tickers:
        raise ValueError(f"Investable universe must contain the cash proxy {CASH}.")
    if "SPY" not in tickers:
        raise ValueError("Investable universe must contain SPY.")

    risky_tickers = [ticker for ticker in tickers if ticker != CASH]
    if len(risky_tickers) != 48:
        raise ValueError(
            f"Expected 48 risky tickers in the investable universe; found {len(risky_tickers)}."
        )
    return risky_tickers


def read_xgboost_probabilities(
    path: Path,
    risky_tickers: list[str],
    threshold: float,
    allow_different_dates: bool,
) -> tuple[pd.DataFrame, dict[pd.Timestamp, list[str]]]:
    """
    Read the MAIN S04B all_test_predictions.csv file.

    The selection flag is recalculated from Probability >= threshold.
    No Top-K ranking restriction is applied.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"XGBoost prediction export not found: {path}\n"
            "Run S04B_return_classification_xgb.ipynb from main first."
        )

    panel = pd.read_csv(path, parse_dates=["Date"])
    required = {"Date", "Ticker", "Probability", "Rank"}
    missing = required.difference(panel.columns)
    if missing:
        raise ValueError(
            f"{path} is missing columns: {sorted(missing)}. "
            "Expected Date, Ticker, Probability, Rank."
        )

    panel = panel.copy()
    panel["Date"] = pd.to_datetime(panel["Date"], errors="raise").dt.normalize()
    panel["Ticker"] = panel["Ticker"].astype(str).str.strip()
    panel["Probability"] = pd.to_numeric(panel["Probability"], errors="raise")
    panel["Rank"] = pd.to_numeric(panel["Rank"], errors="raise")

    if panel[["Date", "Ticker", "Probability", "Rank"]].isna().any().any():
        raise ValueError("XGBoost export contains missing required values.")
    if panel["Ticker"].eq("").any() or panel["Ticker"].str.lower().eq("nan").any():
        raise ValueError("XGBoost export contains blank ticker values.")
    if not np.isfinite(panel["Probability"].to_numpy(float)).all():
        raise ValueError("XGBoost probabilities contain non-finite values.")
    if not panel["Probability"].between(0.0, 1.0).all():
        raise ValueError("XGBoost probabilities must be between zero and one.")
    if not np.equal(
        panel["Rank"].to_numpy(float),
        np.floor(panel["Rank"].to_numpy(float)),
    ).all() or not panel["Rank"].ge(1).all():
        raise ValueError("XGBoost Rank must contain positive integers.")

    panel["Rank"] = panel["Rank"].astype(int)

    if panel.duplicated(["Date", "Ticker"]).any():
        raise ValueError("Duplicate Date/Ticker observations found in XGBoost export.")

    outside = sorted(set(panel["Ticker"]) - set(risky_tickers))
    missing_tickers = sorted(set(risky_tickers) - set(panel["Ticker"]))
    if outside or missing_tickers:
        raise ValueError(
            "S04B prediction universe does not match Investable_Universe.csv. "
            f"Outside universe: {outside}; missing tickers: {missing_tickers}."
        )

    actual_dates = sorted(panel["Date"].dt.strftime("%Y-%m-%d").unique().tolist())
    if not allow_different_dates and actual_dates != MAIN_REBALANCE_DATES:
        raise ValueError(
            "The prediction export does not contain the inspected MAIN S04B rebalance calendar.\n"
            f"Expected: {MAIN_REBALANCE_DATES}\n"
            f"Found:    {actual_dates}\n"
            "If you intentionally regenerated S04B with a different calendar, "
            "rerun with --allow-different-dates."
        )

    selected_by_date: dict[pd.Timestamp, list[str]] = {}
    expected_count = len(risky_tickers)

    for date, day in panel.groupby("Date", sort=True):
        if len(day) != expected_count or set(day["Ticker"]) != set(risky_tickers):
            raise ValueError(
                f"{date.date()}: expected exactly {expected_count} unique ticker "
                f"predictions, found {len(day)}."
            )

        ranked = day.sort_values("Rank", kind="stable")
        if ranked["Rank"].tolist() != list(range(1, expected_count + 1)):
            raise ValueError(
                f"{date.date()}: Rank must run consecutively from 1 to {expected_count}."
            )

        expected_order = day.sort_values(
            ["Probability", "Ticker"],
            ascending=[False, True],
            kind="stable",
        )["Ticker"].tolist()
        if ranked["Ticker"].tolist() != expected_order:
            raise ValueError(
                f"{date.date()}: Rank does not agree with descending probability order."
            )

        selected_by_date[pd.Timestamp(date)] = (
            ranked.loc[ranked["Probability"].ge(threshold), "Ticker"]
            .astype(str)
            .tolist()
        )

    panel["Selected"] = panel["Probability"].ge(threshold).astype(int)
    panel["Threshold"] = float(threshold)
    panel["Selection_Rule"] = f"Probability >= {threshold:.2f}"
    panel = panel.sort_values(["Date", "Rank"]).reset_index(drop=True)

    if len(panel) != len(selected_by_date) * expected_count:
        raise AssertionError("Prediction panel row-count reconciliation failed.")

    return panel, selected_by_date


# =============================================================================
# MARKET DATA AND RISK-FREE SERIES
# =============================================================================

def read_market(ticker: str) -> pd.DataFrame:
    path = Path(config.INPUT_DATA_DIR) / "yahoo" / f"{ticker}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing Yahoo market file: {path}")

    data = pd.read_csv(path, parse_dates=["Date"])
    required = {"Date", "Open", "Close", "Adj Close"}
    if not required.issubset(data.columns):
        raise ValueError(f"{path} missing columns: {sorted(required - set(data.columns))}")
    if data["Date"].duplicated().any():
        raise ValueError(f"{ticker}: duplicate market dates.")

    data = data.sort_values("Date").set_index("Date")
    for column in ["Open", "Close", "Adj Close"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    positive = data[["Open", "Close", "Adj Close"]]
    if (positive <= 0).any().any():
        raise ValueError(f"{ticker}: non-positive price observed.")

    # Adjusted Open/Close are in a consistent total-return unit.
    # Raw Open is retained for commission-per-share calculations.
    data["Adj Open"] = data["Open"] * data["Adj Close"] / data["Close"]
    return data


def load_daily_cash_rates(path: Path, spy_dates: pd.DatetimeIndex) -> pd.Series:
    """
    DGS3MO annualized percent yield, lagged one SPY session for publication.

    A quote reported on T is used from the next SPY session onward.
    No future-value interpolation is performed.
    """
    if not path.is_file():
        raise FileNotFoundError(f"DGS3MO series not found: {path}")

    data = pd.read_csv(path, usecols=["Date", CASH], parse_dates=["Date"])
    data[CASH] = pd.to_numeric(data[CASH].replace(".", np.nan), errors="coerce")
    data = (
        data.dropna(subset=[CASH])
        .sort_values("Date")
        .drop_duplicates("Date", keep="last")
    )
    if data.empty:
        raise ValueError(f"No valid {CASH} observations in {path}.")

    annual_yield = data.set_index("Date")[CASH] / 100.0
    if (annual_yield <= -1.0).any():
        raise ValueError("DGS3MO yields below -100%.")

    asof = annual_yield.reindex(spy_dates, method="ffill").shift(1)
    if asof.isna().any():
        first_valid = asof.first_valid_index()
        if first_valid is None:
            raise ValueError("DGS3MO has no observations available on the SPY calendar.")
        asof = asof.loc[first_valid:]

    return (1.0 + asof).pow(1.0 / TRADING_DAYS).sub(1.0)


def market_panels(
    tickers: set[str],
    calendar: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build adjusted close/open and raw open panels on SPY sessions."""
    adjusted_close: dict[str, pd.Series] = {}
    adjusted_open: dict[str, pd.Series] = {}
    raw_open: dict[str, pd.Series] = {}

    for ticker in sorted(tickers):
        market = read_market(ticker)

        # Carry only short historical close gaps, never backfill from future.
        adjusted_close[ticker] = market["Adj Close"].reindex(calendar).ffill(limit=2)
        adjusted_open[ticker] = market["Adj Open"].reindex(calendar)
        raw_open[ticker] = market["Open"].reindex(calendar)

    return (
        pd.DataFrame(adjusted_close, index=calendar),
        pd.DataFrame(adjusted_open, index=calendar),
        pd.DataFrame(raw_open, index=calendar),
    )


# =============================================================================
# REGIME-AWARE LEDOIT-WOLF ESTIMATION
# =============================================================================

def covariance_lw(
    daily_returns: np.ndarray,
    horizon: int = HORIZON,
) -> tuple[np.ndarray, float]:
    """
    Ledoit-Wolf covariance, finite-sample corrected and scaled to the horizon.

    sklearn's LedoitWolf covariance uses the maximum-likelihood denominator.
    Multiplying by n/(n-1) applies the finite-sample degrees-of-freedom
    correction before scaling daily covariance to the 60-session horizon.
    """
    values = np.asarray(daily_returns, dtype=float)
    if (
        values.ndim != 2
        or values.shape[0] < 3
        or values.shape[1] < 1
        or not np.isfinite(values).all()
    ):
        raise ValueError("Ledoit-Wolf covariance requires a complete finite return matrix.")

    fitted = LedoitWolf(assume_centered=False).fit(values)
    covariance = (
        np.atleast_2d(fitted.covariance_)
        * (len(values) / (len(values) - 1.0))
        * horizon
    )
    covariance = (covariance + covariance.T) / 2.0
    return covariance, float(fitted.shrinkage_)


def regime_covariance(
    prices: pd.DataFrame,
    date: pd.Timestamp,
    tickers: list[str],
    spy_close: pd.Series,
    unconditional_covariance: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Estimate regime-conditioned Ledoit-Wolf covariance.

    Regime state:
        High volatility if 21-session SPY realized volatility is above the
        trailing 252-session median of realized volatility. The median is
        shifted one session to prevent the threshold from using future data.

    Conditional covariance:
        Ledoit-Wolf covariance estimated from up to 756 prior daily return
        observations belonging to the same volatility state.

    Blend:
        50% conditional covariance + 50% unconditional covariance by default.
        If fewer than 90 same-state observations are available, the function
        falls back to unconditional Ledoit-Wolf covariance.
    """
    spy_history = spy_close.loc[:date]
    spy_returns = spy_history.pct_change(fill_method=None)

    realized_volatility = spy_returns.rolling(
        args.regime_vol_window,
        min_periods=args.regime_vol_window,
    ).std()

    past_only_threshold = realized_volatility.shift(1).rolling(
        args.regime_threshold_window,
        min_periods=args.regime_threshold_window,
    ).median()

    if date not in realized_volatility.index or date not in past_only_threshold.index:
        raise ValueError(f"{date.date()}: SPY regime state is unavailable.")

    current_volatility = realized_volatility.loc[date]
    current_threshold = past_only_threshold.loc[date]
    if not np.isfinite(current_volatility) or not np.isfinite(current_threshold):
        raise ValueError(f"{date.date()}: insufficient history for SPY regime classification.")

    high_volatility = bool(current_volatility > current_threshold)
    regime_name = "High_Vol" if high_volatility else "Low_Vol"

    historical_returns = (
        prices.loc[:date, tickers]
        .pct_change(fill_method=None)
        .tail(args.regime_lookback)
    )

    regime_state = (realized_volatility > past_only_threshold).where(
        realized_volatility.notna() & past_only_threshold.notna()
    )
    state_mask = (
        regime_state.reindex(historical_returns.index)
        .eq(high_volatility)
        .fillna(False)
    )

    conditional_returns = historical_returns.loc[state_mask]
    conditional_returns = conditional_returns.loc[
        np.isfinite(conditional_returns.to_numpy(float)).all(axis=1)
    ]

    common = {
        "Regime": regime_name,
        "State_Observations": int(len(conditional_returns)),
        "SPY_Realized_Vol": float(current_volatility),
        "SPY_Regime_Threshold": float(current_threshold),
    }

    if len(conditional_returns) < args.regime_min_days:
        LOG.warning(
            "%s | regime=%s | observations=%d | using unconditional Ledoit-Wolf",
            date.date(),
            regime_name,
            len(conditional_returns),
        )
        return unconditional_covariance, {
            **common,
            "Regime_Fallback": True,
            "Regime_Covariance_LW_Shrinkage": np.nan,
        }

    conditional_covariance, conditional_shrinkage = covariance_lw(
        conditional_returns.to_numpy(float),
        HORIZON,
    )

    blended = (
        args.regime_condition_weight * conditional_covariance
        + (1.0 - args.regime_condition_weight) * unconditional_covariance
    )
    blended = (blended + blended.T) / 2.0

    return blended, {
        **common,
        "Regime_Fallback": False,
        "Regime_Covariance_LW_Shrinkage": conditional_shrinkage,
    }


def make_regime_estimator(
    spy_close: pd.Series,
    args: argparse.Namespace,
    strategy_name: str,
    diagnostics: list[dict[str, Any]],
):
    """
    Return the estimation function used by the integrated direct optimizer.

    Expected returns remain the original 120-session raw-momentum estimate.
    The covariance matrix is replaced by regime-aware Ledoit-Wolf covariance.
    """
    def estimator(
        prices: pd.DataFrame,
        date: pd.Timestamp,
        tickers: list[str],
        daily_rf: pd.Series,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        price_window = prices.loc[:date, tickers].tail(EXPECTED_RETURN_LOOKBACK + 1)
        price_values = price_window.to_numpy(dtype=float)

        if (
            len(price_window) != EXPECTED_RETURN_LOOKBACK + 1
            or not np.isfinite(price_values).all()
            or (price_values <= 0).any()
        ):
            raise ValueError(
                f"{date.date()}: incomplete {EXPECTED_RETURN_LOOKBACK}-session "
                f"expected-return lookback for {tickers}."
            )

        cumulative_return = (
            price_window.iloc[-1].to_numpy(float)
            / price_window.iloc[0].to_numpy(float)
            - 1.0
        )
        if not np.isfinite(cumulative_return).all() or (cumulative_return <= -1.0).any():
            raise ValueError(f"{date.date()}: invalid 120-session cumulative return.")

        expected_return_60d = np.expm1(
            np.log1p(cumulative_return)
            * (HORIZON / EXPECTED_RETURN_LOOKBACK)
        )

        daily_returns = (
            prices.loc[:date, tickers]
            .pct_change(fill_method=None)
            .tail(COVARIANCE_LOOKBACK)
        )
        return_values = daily_returns.to_numpy(dtype=float)
        if (
            len(daily_returns) != COVARIANCE_LOOKBACK
            or not np.isfinite(return_values).all()
        ):
            raise ValueError(
                f"{date.date()}: incomplete {COVARIANCE_LOOKBACK}-session "
                f"covariance lookback for {tickers}."
            )

        unconditional_covariance, baseline_shrinkage = covariance_lw(
            return_values,
            HORIZON,
        )

        regime_cov, regime_details = regime_covariance(
            prices,
            date,
            tickers,
            spy_close,
            unconditional_covariance,
            args,
        )

        daily_cash_rate = float(daily_rf.loc[date])
        if not np.isfinite(daily_cash_rate) or daily_cash_rate <= -1.0:
            raise ValueError(f"{date.date()}: invalid one-session DGS3MO rate.")

        cash_return_60d = float(
            np.expm1(HORIZON * np.log1p(daily_cash_rate))
        )

        diagnostics.append({
            "Configuration": CONFIGURATION,
            "Strategy": strategy_name,
            "Signal_Date": str(pd.Timestamp(date).date()),
            "Asset_Count": len(tickers),
            "Baseline_LW_Shrinkage": baseline_shrinkage,
            "Covariance_Condition_Number": float(np.linalg.cond(regime_cov)),
            "Raw_Momentum_Mean_60D": float(np.mean(expected_return_60d)),
            "Cash_60D": cash_return_60d,
            **regime_details,
        })

        return expected_return_60d, regime_cov, cash_return_60d

    return estimator


# =============================================================================
# TRANSACTION COST ACCOUNTING
# =============================================================================

def fee_context_at_decision(
    signal_date: pd.Timestamp,
    execution_date: pd.Timestamp,
    tickers: list[str],
    raw_opens: pd.DataFrame,
) -> pd.DataFrame:
    """
    Use point-in-time Spread and ADV at signal close.

    Commission share price is updated to the raw execution-day opening price.
    """
    context = load_fee_context(signal_date, tickers=tickers)

    for ticker in tickers:
        if ticker == CASH:
            continue
        opening = raw_opens.at[execution_date, ticker]
        if not np.isfinite(opening) or opening <= 0:
            raise ValueError(
                f"{ticker}: missing raw OPEN on execution date {execution_date.date()}."
            )
        context.at[ticker, "Price"] = float(opening)

    return context


def fee_and_trades(
    pre_trade_dollars: pd.Series,
    desired_post_fee_weights: pd.Series,
    context: pd.DataFrame,
) -> tuple[float, pd.Series, pd.Series, pd.DataFrame]:
    """
    Solve fees with post-fee target weights, preserving exact self-financing.

    The desired weights refer to post-fee NAV. The fee is found by fixed point
    because order sizes depend on final NAV. The helper computes spread,
    market impact and commission from the supplied point-in-time context.
    """
    tickers = list(context.index)
    current = pre_trade_dollars.reindex(tickers, fill_value=0.0).astype(float)
    wealth = float(current.sum())

    if wealth <= 0 or not np.isfinite(wealth):
        raise ValueError("Invalid portfolio NAV.")

    weights = desired_post_fee_weights.reindex(tickers, fill_value=0.0).astype(float)
    if abs(float(weights.sum()) - 1.0) > 1e-8 or (weights < -1e-9).any():
        raise ValueError("Invalid candidate target weights.")

    current_weights = current / wealth

    def charge(cost: float, breakdown: bool = False):
        after_cost_nav = wealth - cost
        if after_cost_nav <= 0:
            raise ValueError("Transaction cost exceeds portfolio NAV.")

        wanted_dollars = weights * after_cost_nav

        # Cash is the balancing leg before commission settlement.
        helper_target = wanted_dollars / wealth
        helper_target.loc[CASH] += cost / wealth

        if not np.isclose(helper_target.sum(), 1.0, atol=1e-9):
            raise ArithmeticError("Transaction-fee accounting target is not self-financing.")

        result = calculate_transaction_fees(
            current_weights,
            helper_target,
            wealth,
            context,
            return_breakdown=breakdown,
        )
        return result, wanted_dollars

    fee = 0.0
    for _ in range(100):
        next_fee, _ = charge(fee)
        if abs(next_fee - fee) <= max(1e-7, wealth * 1e-12):
            fee = float(next_fee)
            break
        fee = float(next_fee)
    else:
        raise ArithmeticError("Transaction-fee fixed point did not converge.")

    (calculated_fee, breakdown), wanted = charge(fee, breakdown=True)

    if abs(calculated_fee - fee) > max(1e-5, wealth * 1e-9):
        raise ArithmeticError("Transaction-fee reconciliation failed.")
    if abs(float(wanted.sum()) + fee - wealth) > max(1e-6, wealth * 1e-10):
        raise ArithmeticError("Post-fee NAV accounting failed.")

    trades = wanted - current
    return float(fee), wanted, trades, breakdown


# =============================================================================
# DIRECT TRANSACTION-COST-AWARE MEAN-VARIANCE OPTIMIZER
# =============================================================================

def optimize_direct_utility(
    date: pd.Timestamp,
    chosen: list[str],
    dollars: pd.Series,
    price_history: pd.DataFrame,
    risk_free: pd.Series,
    estimator,
    risk_aversion: float = DEFAULT_RISK_AVERSION,
    starts: int = 12,
) -> tuple[pd.Series, dict[str, Any]]:
    """
    Maximize cost-adjusted mean-variance utility directly over asset weights.

    The optimizer considers:
        - every currently XGBoost-selected ticker;
        - every risky ticker already held in the portfolio.

    This preserves holdings continuity. A ticker does not become ineligible
    for liquidation/reallocation merely because it failed the latest filter.

    There is NO minimum-three-assets restriction:
        - 0 eligible risky assets: cash;
        - 1 or 2 eligible assets: valid optimization problem;
        - 3+ eligible assets: valid optimization problem.

    Each risky ticker is capped at 20%. When invested, total risky exposure is
    at least 20%. The portfolio can hold 100% cash if the best incremental
    expected utility after transaction costs is non-positive.

    Objective:
        U(w) = w'(mu_60 - r_cash_60)
               - incremental_transaction_costs / NAV
               - gamma/2 * w' Sigma_60 w
    """
    if not np.isfinite(risk_aversion) or risk_aversion <= 0:
        raise ValueError("Risk aversion gamma must be finite and positive.")
    if starts < 1:
        raise ValueError("Optimizer starts must be at least one.")

    held = [
        ticker for ticker, value in dollars.items()
        if ticker != CASH and value > 1e-7
    ]
    eligible = list(dict.fromkeys(list(chosen) + held))
    tickers = eligible + [CASH]

    prior = dollars.reindex(tickers, fill_value=0.0).astype(float)
    wealth = float(prior.sum())
    if wealth <= 0 or not np.isfinite(wealth):
        raise ValueError(f"{date.date()}: invalid prior portfolio NAV.")

    context = load_fee_context(date, tickers=tickers)

    def portfolio(risky_weights: np.ndarray) -> pd.Series:
        # Project only tiny numerical excursions onto the feasible box/simplex.
        weights = np.clip(np.asarray(risky_weights, dtype=float), 0.0, MAX_WEIGHT)
        risky_total = float(weights.sum())

        if risky_total > 1.0:
            weights = weights / risky_total
            risky_total = float(weights.sum())

        target = pd.Series(0.0, index=tickers, dtype=float)
        if eligible:
            target.loc[eligible] = weights
        target.loc[CASH] = max(0.0, 1.0 - risky_total)
        target.loc[CASH] += 1.0 - float(target.sum())
        return target

    cash_target = portfolio(np.zeros(len(eligible), dtype=float))
    cash_fee, _, _, _ = fee_and_trades(prior, cash_target, context)

    base = {
        "direct_eligible_count": len(eligible),
        "direct_risk_aversion_gamma": float(risk_aversion),
        "direct_start_count": int(starts),
        "direct_solver_converged_count": 0,
        "direct_selected_source": "cash_rule",
        "direct_mean_variance_utility": 0.0,
        "direct_expected_net_excess_return": 0.0,
        "direct_estimated_cash_liquidation_cost": float(cash_fee),
    }

    # Only zero eligible assets is a count-based reason to remain in cash.
    if len(eligible) == 0:
        LOG.info(
            "%s | direct utility | eligible=0 | risky=0.00%% | cash=100.00%%",
            date.date(),
        )
        return cash_target, {
            **base,
            "direct_cash_reason": "no_eligible_risky_assets",
        }

    expected_returns, covariance, cash_return_60d = estimator(
        price_history,
        date,
        eligible,
        risk_free,
    )
    covariance = (covariance + covariance.T) / 2.0
    excess_expected_returns = expected_returns - cash_return_60d
    tolerance = 1e-7
    n = len(eligible)

    def feasible(weights: np.ndarray) -> bool:
        weights = np.asarray(weights, dtype=float)
        total = float(weights.sum())
        return (
            weights.shape == (n,)
            and np.isfinite(weights).all()
            and weights.min() >= -tolerance
            and weights.max() <= MAX_WEIGHT + tolerance
            and total >= MIN_RISKY_WEIGHT - tolerance
            and total <= 1.0 + tolerance
        )

    # Cache objective evaluations because SLSQP finite differences often
    # revisit the same points and every evaluation includes a fee calculation.
    cache: dict[tuple[float, ...], tuple[float, float, float]] = {}

    def measure(weights: np.ndarray) -> tuple[float, float, float]:
        weights = np.asarray(weights, dtype=float)
        key = tuple(np.round(weights, 12))

        if key not in cache:
            if not feasible(weights):
                return -1e6, -1e6, np.nan

            target = portfolio(weights)
            fee, _, _, _ = fee_and_trades(prior, target, context)

            gross_excess_return = float(weights @ excess_expected_returns)
            incremental_net_return = (
                gross_excess_return - (fee - cash_fee) / wealth
            )
            variance = max(0.0, float(weights @ covariance @ weights))
            utility = incremental_net_return - risk_aversion * variance / 2.0

            cache[key] = (
                float(utility),
                float(incremental_net_return),
                float(fee),
            )

        return cache[key]

    def objective(weights: np.ndarray) -> float:
        return -measure(weights)[0]

    # Deterministic feasible starting portfolios.
    rng = np.random.default_rng(int(pd.Timestamp(date).strftime("%Y%m%d")) + 149)
    seeds: list[np.ndarray] = [
        np.full(n, MIN_RISKY_WEIGHT / n, dtype=float)
    ]

    if n >= 5:
        seeds.append(np.full(n, 1.0 / n, dtype=float))

    prior_risky = prior.reindex(eligible).to_numpy(dtype=float)
    if prior_risky.sum() > 0:
        seed = np.clip(prior_risky / wealth, 0.0, MAX_WEIGHT)
        if seed.sum() < MIN_RISKY_WEIGHT:
            seed += (MIN_RISKY_WEIGHT - seed.sum()) / n
        if feasible(seed):
            seeds.append(seed)

    max_exposure = min(1.0, n * MAX_WEIGHT)

    while len(seeds) < starts:
        exposure = float(rng.uniform(MIN_RISKY_WEIGHT, max_exposure))
        raw = rng.dirichlet(np.ones(n) * 2.0)

        # Project the random composition onto the capped simplex with the
        # chosen total risky exposure.
        low = float(raw.min() - MAX_WEIGHT)
        high = float(raw.max())

        for _ in range(75):
            midpoint = 0.5 * (low + high)
            if np.clip(raw - midpoint, 0.0, MAX_WEIGHT).sum() > exposure:
                low = midpoint
            else:
                high = midpoint

        weights = np.clip(raw - 0.5 * (low + high), 0.0, MAX_WEIGHT)
        difference = exposure - float(weights.sum())

        for index in np.argsort(-raw if difference > 0 else raw):
            if abs(difference) < 1e-10:
                break
            if difference > 0:
                adjustment = min(difference, MAX_WEIGHT - weights[index])
            else:
                adjustment = min(-difference, weights[index])
            weights[index] += np.sign(difference) * adjustment
            difference -= np.sign(difference) * adjustment

        if feasible(weights):
            seeds.append(weights)

    seeds = seeds[:starts]

    best: tuple[float, np.ndarray, str] | None = None
    converged = 0

    for seed in seeds:
        if feasible(seed):
            candidate = (measure(seed)[0], seed.copy(), "feasible_seed")
            if best is None or candidate[0] > best[0]:
                best = candidate

        result = minimize(
            objective,
            seed,
            method="SLSQP",
            bounds=[(0.0, MAX_WEIGHT)] * n,
            constraints=[
                {
                    "type": "ineq",
                    "fun": lambda weights: float(np.sum(weights) - MIN_RISKY_WEIGHT),
                },
                {
                    "type": "ineq",
                    "fun": lambda weights: float(1.0 - np.sum(weights)),
                },
            ],
            options={"maxiter": 250, "ftol": 1e-8},
        )

        if result.success and feasible(result.x):
            converged += 1

        if feasible(result.x):
            candidate = (
                measure(result.x)[0],
                np.asarray(result.x, dtype=float).copy(),
                "solver_converged" if result.success else "solver_feasible_unconverged",
            )
            if best is None or candidate[0] > best[0]:
                best = candidate

    if best is None:
        raise RuntimeError(
            f"{date.date()}: direct utility optimizer found no feasible candidate."
        )

    utility, selected_weights, selected_source = best
    _, net_excess_return, selected_fee = measure(selected_weights)

    meta = {
        **base,
        "direct_solver_converged_count": int(converged),
        "direct_selected_source": selected_source,
        "direct_best_feasible_risky_utility": float(utility),
        "direct_expected_net_excess_return": float(net_excess_return),
        "direct_estimated_transaction_cost": float(selected_fee),
        "direct_evaluations": int(len(cache)),
    }

    if utility <= 0 or net_excess_return <= 0:
        LOG.info(
            "%s | direct utility | eligible=%d | risky=0.00%% | cash=100.00%% "
            "| reason=non_positive_incremental_utility | utility=%.6f | net=%.6f",
            date.date(),
            n,
            utility,
            net_excess_return,
        )
        return cash_target, {
            **meta,
            "direct_mean_variance_utility": 0.0,
            "direct_cash_reason": "non_positive_incremental_utility",
        }

    target = portfolio(selected_weights)
    if not feasible(selected_weights) or target[CASH] > MAX_CASH_WEIGHT + tolerance:
        raise AssertionError("Direct utility optimizer produced an invalid allocation.")

    LOG.info(
        "%s | direct utility gamma=%.3f | eligible=%d | risky=%.2f%% "
        "| utility=%.6f | starts=%d converged=%d",
        date.date(),
        risk_aversion,
        n,
        100.0 * float(selected_weights.sum()),
        utility,
        starts,
        converged,
    )

    return target, {
        **meta,
        "direct_mean_variance_utility": float(utility),
        "direct_cash_reason": "direct_joint_utility",
    }


# =============================================================================
# DAILY BACKTEST AND PORTFOLIO EXECUTION
# =============================================================================

def run_strategy(
    strategy_name: str,
    selections: dict[pd.Timestamp, list[str]],
    args: argparse.Namespace,
    spy_close: pd.Series,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """
    Run one portfolio strategy through the same execution and allocation code.

    Decisions are made at rebalance-date close and executed at the next SPY
    session open. The SPY buy-and-hold benchmark is initialized at that same
    first execution open.
    """
    if not selections:
        raise ValueError(f"{strategy_name}: no rebalance selections supplied.")

    strategy_output = args.out / strategy_name
    strategy_output.mkdir(parents=True, exist_ok=True)

    decisions = sorted(pd.Timestamp(date) for date in selections)
    spy_market = read_market("SPY")
    sessions = spy_market.index

    if any(date not in sessions for date in decisions):
        missing = [date for date in decisions if date not in sessions]
        raise ValueError(
            f"{strategy_name}: rebalance dates are not SPY sessions: "
            f"{[str(date.date()) for date in missing]}"
        )

    first_signal = decisions[0]
    first_execution_index = sessions.get_loc(first_signal) + 1
    if first_execution_index >= len(sessions):
        raise ValueError("No SPY session exists after the first rebalance date.")
    first_execution = sessions[first_execution_index]

    # Load every ticker that can enter the portfolio at any rebalance, plus SPY.
    # Previously held assets remain in the eligible set in the optimizer.
    needed_tickers = set().union(*(set(selections[date]) for date in decisions))
    needed_tickers.update({"SPY"})
    close, adjusted_open, raw_open = market_panels(needed_tickers, sessions)

    risk_free = load_daily_cash_rates(args.macro, sessions)
    if risk_free.loc[first_signal:sessions[-1]].isna().any():
        raise ValueError("DGS3MO is missing during the backtest; zero rates will not be substituted.")

    signals_by_execution: dict[pd.Timestamp, pd.Timestamp] = {}
    for signal_date in decisions:
        next_index = sessions.get_loc(signal_date) + 1
        if next_index >= len(sessions):
            LOG.warning(
                "%s: skipping final signal %s because no next-session execution exists.",
                strategy_name,
                signal_date.date(),
            )
            continue

        execution_date = sessions[next_index]
        if execution_date in signals_by_execution:
            raise ValueError("Multiple signals map to the same execution date.")
        signals_by_execution[execution_date] = signal_date

    if not signals_by_execution:
        raise ValueError(f"{strategy_name}: no executable rebalance dates.")

    diagnostics: list[dict[str, Any]] = []
    estimator = make_regime_estimator(
        spy_close,
        args,
        strategy_name,
        diagnostics,
    )

    # Dollar holdings in adjusted-price total-return units; cash in dollars.
    shares = pd.Series(dtype=float)
    cash = float(args.initial_capital)
    cash_rate_at_close = float(risk_free.loc[first_signal])

    benchmark_shares: float | None = None
    benchmark_cash = float(args.initial_capital)

    daily_rows: list[dict[str, Any]] = [{
        "Date": first_signal,
        "Portfolio_NAV": cash,
        "SPY_NAV": benchmark_cash,
        "Risky_Weight": 0.0,
        "Cash_Weight": 1.0,
        "Net_Daily_Return": np.nan,
        "SPY_Daily_Return": np.nan,
    }]

    weights_rows: list[dict[str, Any]] = []
    rebalance_rows: list[dict[str, Any]] = []
    trade_frames: list[pd.DataFrame] = []
    spread_fallback_frames: list[pd.DataFrame] = []

    LOG.info(
        "START | %s | rebalances=%d | gamma=%.3f | threshold=%.2f",
        strategy_name,
        len(decisions),
        args.gamma,
        args.threshold,
    )

    for current_date in sessions[sessions > first_signal]:
        # The optimizer only sees information available at the signal-date close.
        if current_date in signals_by_execution:
            signal_date = signals_by_execution[current_date]

            prior_close_dollars = pd.Series({
                ticker: float(quantity * close.at[signal_date, ticker])
                for ticker, quantity in shares.items()
            })
            prior_close_dollars.loc[CASH] = cash

            target, optimizer_meta = optimize_direct_utility(
                date=signal_date,
                chosen=selections[signal_date],
                dollars=prior_close_dollars,
                price_history=close,
                risk_free=risk_free,
                estimator=estimator,
                risk_aversion=args.gamma,
                starts=args.direct_starts,
            )
        else:
            signal_date = None
            target = None
            optimizer_meta = {}

        # Cash earns the last known one-session DGS3MO rate.
        cash *= 1.0 + float(cash_rate_at_close)
        cash_rate_at_close = float(risk_free.loc[current_date])

        # Buy the common SPY benchmark at the first strategy execution open.
        if benchmark_shares is None:
            benchmark_open = adjusted_open.at[first_execution, "SPY"]
            if not np.isfinite(benchmark_open) or benchmark_open <= 0:
                raise ValueError("SPY adjusted opening price missing on first execution date.")
            benchmark_shares = benchmark_cash / float(benchmark_open)
            benchmark_cash = 0.0

        if current_date in signals_by_execution:
            assert signal_date is not None and target is not None

            pre_trade_values: dict[str, float] = {}
            for ticker, quantity in shares.items():
                if abs(quantity) <= 1e-10:
                    continue
                opening = adjusted_open.at[current_date, ticker]
                if not np.isfinite(opening) or opening <= 0:
                    raise ValueError(
                        f"{ticker}: missing adjusted OPEN on {current_date.date()}."
                    )
                pre_trade_values[ticker] = float(quantity * opening)

            pre_trade_values[CASH] = cash
            pre_trade_dollars = pd.Series(pre_trade_values)
            eligible_for_execution = target.index.tolist()

            fee_context = fee_context_at_decision(
                signal_date,
                current_date,
                eligible_for_execution,
                raw_open,
            )

            # Record all spread fallbacks used by the fee context, including
            # assets that were eligible but did not ultimately receive a trade.
            if "Spread_Fallback" in fee_context.columns:
                fallback_rows = fee_context.loc[
                    fee_context["Spread_Fallback"].fillna(False).astype(bool)
                ].reset_index().rename(columns={"Spread": "Spread_Used"})

                if not fallback_rows.empty:
                    fallback_rows.insert(0, "Signal_Date", signal_date)
                    fallback_rows.insert(1, "Execution_Date", current_date)
                    spread_fallback_frames.append(fallback_rows)

                    for _, fallback in fallback_rows.iterrows():
                        LOG.warning(
                            "%s: spread fallback=%s ticker=%s age=%s spread=%.8f",
                            signal_date.date(),
                            fallback.get("Spread_Fallback_Method"),
                            fallback.get("Ticker"),
                            fallback.get("Spread_Fallback_Age_Trading_Days"),
                            float(fallback.get("Spread_Used")),
                        )

            fee, post_trade_dollars, trade_dollars, breakdown = fee_and_trades(
                pre_trade_dollars,
                target,
                fee_context,
            )

            pre_trade_nav = float(pre_trade_dollars.sum())
            post_trade_nav = float(post_trade_dollars.sum())

            shares = pd.Series({
                ticker: float(post_trade_dollars[ticker] / adjusted_open.at[current_date, ticker])
                for ticker in eligible_for_execution
                if ticker != CASH and post_trade_dollars[ticker] > 1e-10
            }, dtype=float)

            cash = float(post_trade_dollars.loc[CASH])
            if cash < -1e-6:
                raise AssertionError("Negative cash after transaction costs.")
            if abs(post_trade_nav + fee - pre_trade_nav) > max(1e-6, pre_trade_nav * 1e-10):
                raise AssertionError("Portfolio NAV does not reconcile after transaction costs.")

            selected_count = len(selections[signal_date])
            rebalance_rows.append({
                "Signal_Date": signal_date,
                "Execution_Date": current_date,
                "Pre_Trade_NAV": pre_trade_nav,
                "Post_Trade_NAV": post_trade_nav,
                "Transaction_Cost_USD": fee,
                "Selected_Asset_Count": selected_count,
                "Eligible_Including_Holdings": len(eligible_for_execution) - 1,
                "Actual_Risky_Positions": int(len(shares)),
                "Target_Cash_Weight": float(target.loc[CASH]),
                **optimizer_meta,
            })

            for ticker, weight in target.items():
                weights_rows.append({
                    "Signal_Date": signal_date,
                    "Execution_Date": current_date,
                    "Ticker": ticker,
                    "Target_Weight": float(weight),
                    "Post_Cost_Target_Dollars": float(post_trade_dollars.loc[ticker]),
                    "Selected_At_Signal": ticker in selections[signal_date],
                })

            if not breakdown.empty:
                breakdown.insert(0, "Signal_Date", signal_date)
                breakdown.insert(1, "Execution_Date", current_date)
                trade_frames.append(breakdown)

        # Mark every actual holding to adjusted close.
        for ticker in shares.index:
            closing = close.at[current_date, ticker]
            if not np.isfinite(closing) or closing <= 0:
                raise ValueError(f"{ticker}: missing valuation close on {current_date.date()}.")

        portfolio_nav = float(
            cash
            + sum(quantity * float(close.at[current_date, ticker])
                  for ticker, quantity in shares.items())
        )

        if benchmark_shares is None:
            raise AssertionError("SPY benchmark shares were not initialized.")

        spy_nav = float(benchmark_shares * close.at[current_date, "SPY"])
        previous_portfolio_nav = float(daily_rows[-1]["Portfolio_NAV"])
        previous_spy_nav = float(daily_rows[-1]["SPY_NAV"])

        daily_rows.append({
            "Date": current_date,
            "Portfolio_NAV": portfolio_nav,
            "SPY_NAV": spy_nav,
            "Cash_Weight": cash / portfolio_nav,
            "Risky_Weight": 1.0 - cash / portfolio_nav,
            "Net_Daily_Return": portfolio_nav / previous_portfolio_nav - 1.0,
            "SPY_Daily_Return": spy_nav / previous_spy_nav - 1.0,
        })

    nav = pd.DataFrame(daily_rows)
    nav["Portfolio_Drawdown"] = nav["Portfolio_NAV"] / nav["Portfolio_NAV"].cummax() - 1.0
    nav["SPY_Drawdown"] = nav["SPY_NAV"] / nav["SPY_NAV"].cummax() - 1.0

    transaction_columns = [
        "Signal_Date", "Execution_Date", "Ticker", "Trade_USD",
        "Spread_Cost", "Market_Impact_Cost", "IBKR_Commission",
        "Spread_Used", "Spread_Original", "Spread_Fallback",
        "Spread_Fallback_Method", "Spread_Fallback_Age_Trading_Days",
        "Spread_Reference_Date", "Spread_Fallback_Quantile", "Total_Cost",
    ]
    trades = (
        pd.concat(trade_frames, ignore_index=True)
        if trade_frames
        else pd.DataFrame(columns=transaction_columns)
    )

    spread_audit_columns = [
        "Signal_Date", "Execution_Date", "Ticker", "Asset Class", "Price",
        "ADV_Dollars", "Spread_Original", "Spread_Used", "Spread_Fallback",
        "Spread_Fallback_Method", "Spread_Fallback_Age_Trading_Days",
        "Spread_Reference_Date", "Spread_Fallback_Quantile",
        "Spread_Fallback_History_Count", "Spread_Fallback_History_Start",
        "Spread_Fallback_History_End",
    ]
    spread_audit = (
        pd.concat(spread_fallback_frames, ignore_index=True)
        if spread_fallback_frames
        else pd.DataFrame(columns=spread_audit_columns)
    )

    rebalance_summary = pd.DataFrame(rebalance_rows)
    portfolio_weights = pd.DataFrame(weights_rows)

    strategy_output.mkdir(parents=True, exist_ok=True)
    nav.to_csv(strategy_output / "daily_nav.csv", index=False)
    rebalance_summary.to_csv(strategy_output / "rebalance_summary.csv", index=False)
    portfolio_weights.to_csv(strategy_output / "portfolio_weights.csv", index=False)
    trades.to_csv(strategy_output / "transaction_cost_breakdown.csv", index=False)
    spread_audit.to_csv(strategy_output / "spread_fallback_audit.csv", index=False)

    metrics = calculate_performance_statistics(nav, trades, risk_free)
    # Count rebalance events, not daily NAV observations or non-zero-cost trades.
    metrics["rebalance_count"] = int(len(rebalance_summary))
    metrics["spread_fallback_count"] = int(len(spread_audit))
    metrics["rebalance_count"] = int(len(rebalance_summary))
    metrics["mean_selected_assets_per_rebalance"] = float(
        np.mean([len(selections[date]) for date in decisions])
    )

    manifest = {
        "configuration": CONFIGURATION,
        "strategy": strategy_name,
        "allocation_method": "Direct joint transaction-cost-aware mean-variance utility over individual risky-asset weights",
        "expected_return_estimation": (
            "120-session adjusted-price momentum converted to a 60-session equivalent "
            "using geometric compounding"
        ),
        "covariance_estimation": (
            "Regime-conditioned Ledoit-Wolf covariance blended with unconditional "
            "252-session Ledoit-Wolf covariance"
        ),
        "risk_aversion_gamma": args.gamma,
        "direct_optimizer_starts": args.direct_starts,
        "minimum_eligible_risky_assets": 1,
        "minimum_three_asset_restriction": False,
        "maximum_weight_per_risky_asset": MAX_WEIGHT,
        "maximum_cash_weight_when_invested": MAX_CASH_WEIGHT,
        "minimum_risky_exposure_when_invested": MIN_RISKY_WEIGHT,
        "cash_only_allowed": True,
        "cash_only_conditions": [
            "No eligible risky assets, including previously held positions",
            "Best incremental net mean-variance utility versus cash is non-positive",
        ],
        "regime_definition": (
            "High volatility when 21-session SPY realized volatility exceeds "
            "the trailing 252-session median of realized volatility; threshold "
            "shifted by one session"
        ),
        "regime_lookback": args.regime_lookback,
        "regime_vol_window": args.regime_vol_window,
        "regime_threshold_window": args.regime_threshold_window,
        "regime_min_days": args.regime_min_days,
        "regime_condition_weight": args.regime_condition_weight,
        "execution": (
            "Signal at rebalance close; execution at next SPY session open; "
            "Spread and ADV at signal date; commission share price at execution open"
        ),
        "transaction_cost_helper": "transaction_fee_helper.py",
        "transaction_cost_components": ["Spread", "Market Impact", "IBKR Commission"],
        "spread_fallback_policy": {
            "short_gap_max_trading_observations": 5,
            "short_gap_method": "Most recent valid historical spread",
            "long_gap_method": "Ticker-specific trailing 252-observation 90th percentile",
            "long_gap_minimum_valid_observations": 20,
            "lookahead": "None; only observations strictly before signal date are used",
            "missing_history": "Fail closed; never substitute zero",
        },
        "benchmark": "SPY buy-and-hold from first execution open, no fees",
        "first_signal": str(first_signal.date()),
        "first_execution": str(first_execution.date()),
        "last_observation": str(nav["Date"].iloc[-1].date()),
        "rebalance_dates": [str(date.date()) for date in decisions],
        "results": metrics,
    }
    (strategy_output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    (strategy_output / "performance_summary.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    LOG.info(
        "COMPLETED | %s | final NAV=%.2f | transaction costs=%.2f",
        strategy_name,
        float(nav["Portfolio_NAV"].iloc[-1]),
        float(metrics["total_transaction_cost_USD"]),
    )

    return {
        "results": metrics,
        "nav": nav,
        "rebalance_summary": rebalance_summary,
        "portfolio_weights": portfolio_weights,
        "transaction_costs": trades,
        "spread_fallback_audit": spread_audit,
    }, diagnostics


def calculate_performance_statistics(
    daily_nav: pd.DataFrame,
    trades: pd.DataFrame,
    risk_free: pd.Series,
) -> dict[str, Any]:
    """Calculate net portfolio and SPY performance on the same daily calendar."""
    frame = daily_nav.copy()
    portfolio_returns = frame["Net_Daily_Return"].dropna()
    spy_returns = frame["SPY_Daily_Return"].dropna()

    if portfolio_returns.empty or spy_returns.empty:
        raise ValueError("No daily returns available for performance statistics.")

    observation_count = len(portfolio_returns)
    elapsed_years = observation_count / TRADING_DAYS
    if elapsed_years <= 0:
        raise ValueError("Invalid annualization period.")

    portfolio_growth = float(
        frame["Portfolio_NAV"].iloc[-1] / frame["Portfolio_NAV"].iloc[0]
    )
    spy_growth = float(frame["SPY_NAV"].iloc[-1] / frame["SPY_NAV"].iloc[0])

    risk_free_aligned = risk_free.reindex(frame["Date"]).to_numpy(dtype=float)
    portfolio_excess = frame["Net_Daily_Return"].to_numpy(dtype=float) - risk_free_aligned
    spy_excess = frame["SPY_Daily_Return"].to_numpy(dtype=float) - risk_free_aligned

    def annualized_sharpe(excess: np.ndarray) -> float:
        valid = excess[np.isfinite(excess)]
        if len(valid) <= 1:
            return np.nan
        volatility = float(np.std(valid, ddof=1))
        if volatility <= 0:
            return np.nan
        return float(np.mean(valid) / volatility * np.sqrt(TRADING_DAYS))

    return {
        "portfolio_total_return": portfolio_growth - 1.0,
        "spy_total_return": spy_growth - 1.0,
        "portfolio_CAGR": portfolio_growth ** (1.0 / elapsed_years) - 1.0,
        "spy_CAGR": spy_growth ** (1.0 / elapsed_years) - 1.0,
        "portfolio_annualized_sharpe": annualized_sharpe(portfolio_excess),
        "spy_annualized_sharpe": annualized_sharpe(spy_excess),
        "portfolio_max_drawdown": float(frame["Portfolio_Drawdown"].min()),
        "spy_max_drawdown": float(frame["SPY_Drawdown"].min()),
        "total_transaction_cost_USD": (
            float(trades["Total_Cost"].sum())
            if len(trades) and "Total_Cost" in trades.columns
            else 0.0
        ),
        "rebalance_count": 0,  # Set to the actual rebalance-row count by run_strategy().
        "daily_return_count": int(observation_count),
    }


# =============================================================================
# COMPARISON, AUDIT OUTPUTS AND REPORTING
# =============================================================================

def create_comparison_chart(comparison: pd.DataFrame, initial_capital: float, output: Path, threshold: float) -> None:
    fig, ax = plt.subplots(figsize=(13, 7))

    ax.plot(
        comparison["Date"],
        comparison["XGBoost_NAV"] / initial_capital * 100.0,
        label=f"XGBoost (probability >= {threshold:.2f}), net of costs",
        linewidth=1.9,
    )
    ax.plot(
        comparison["Date"],
        comparison["Full_Universe_NAV"] / initial_capital * 100.0,
        label="Full Universe, net of costs",
        linewidth=1.9,
    )
    ax.plot(
        comparison["Date"],
        comparison["SPY_NAV"] / initial_capital * 100.0,
        label="SPY buy-and-hold",
        linestyle="--",
        linewidth=1.9,
    )

    ax.set(
        title=f"Regime-Aware Ledoit-Wolf: XGBoost P >= {threshold:.2f} vs Full Universe vs SPY",
        xlabel="Date",
        ylabel="Portfolio NAV (initial value = 100)",
    )
    ax.grid(alpha=0.25)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output / "comparison_nav.png", dpi=180)
    plt.close(fig)


def aggregate_outputs(
    output: Path,
    xgb_result: dict[str, Any],
    full_result: dict[str, Any],
) -> None:
    frames = {
        "all_portfolio_weights.csv": [],
        "all_rebalance_summaries.csv": [],
        "all_transaction_costs.csv": [],
        "all_spread_fallback_audits.csv": [],
    }

    mapping = {
        "all_portfolio_weights.csv": "portfolio_weights",
        "all_rebalance_summaries.csv": "rebalance_summary",
        "all_transaction_costs.csv": "transaction_costs",
        "all_spread_fallback_audits.csv": "spread_fallback_audit",
    }

    for strategy_name, result in [
        (XGB_STRATEGY, xgb_result),
        (FULL_STRATEGY, full_result),
    ]:
        for output_name, key in mapping.items():
            frame = result[key].copy()
            if not frame.empty:
                frame.insert(0, "Strategy", strategy_name)
                frame.insert(0, "Configuration", CONFIGURATION)
            frames[output_name].append(frame)

    for output_name, pieces in frames.items():
        combined = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
        combined.to_csv(output / output_name, index=False)

    transaction_costs = pd.read_csv(output / "all_transaction_costs.csv")
    component_columns = [
        column for column in
        ["Spread_Cost", "Market_Impact_Cost", "IBKR_Commission", "Total_Cost"]
        if column in transaction_costs.columns
    ]

    if not transaction_costs.empty and component_columns:
        cost_summary = transaction_costs.groupby("Strategy", as_index=False)[
            component_columns
        ].sum()
    else:
        cost_summary = pd.DataFrame(columns=["Strategy"] + component_columns)

    cost_summary.to_csv(output / "transaction_cost_summary.csv", index=False)


def build_performance_summary(
    xgb_result: dict[str, Any],
    full_result: dict[str, Any],
    xgb_nav: pd.DataFrame,
    full_nav: pd.DataFrame,
    probability_panel: pd.DataFrame,
    args: argparse.Namespace,
    risky_ticker_count: int,
) -> pd.DataFrame:
    xgb_metrics = xgb_result["results"]
    full_metrics = full_result["results"]

    selected_counts = probability_panel.groupby("Date")["Selected"].sum()
    spy_metrics = {
        "total_return": xgb_metrics["spy_total_return"],
        "CAGR": xgb_metrics["spy_CAGR"],
        "sharpe": xgb_metrics["spy_annualized_sharpe"],
        "max_drawdown": xgb_metrics["spy_max_drawdown"],
    }

    rows = [
        {
            "Configuration": CONFIGURATION,
            "Strategy": XGB_STRATEGY,
            "Selection_Rule": f"Probability >= {args.threshold:.2f}; no Top-K restriction",
            "Total_Return": xgb_metrics["portfolio_total_return"],
            "CAGR": xgb_metrics["portfolio_CAGR"],
            "Annualized_Sharpe": xgb_metrics["portfolio_annualized_sharpe"],
            "Max_Drawdown": xgb_metrics["portfolio_max_drawdown"],
            "Transaction_Cost_USD": xgb_metrics["total_transaction_cost_USD"],
            "Average_Daily_Cash_Weight": float(xgb_nav["Cash_Weight"].mean()),
            "Rebalance_Count": int(len(selected_counts)),
            "Mean_Eligible_Assets_Per_Rebalance": float(selected_counts.mean()),
        },
        {
            "Configuration": CONFIGURATION,
            "Strategy": FULL_STRATEGY,
            "Selection_Rule": f"All {risky_ticker_count} risky assets",
            "Total_Return": full_metrics["portfolio_total_return"],
            "CAGR": full_metrics["portfolio_CAGR"],
            "Annualized_Sharpe": full_metrics["portfolio_annualized_sharpe"],
            "Max_Drawdown": full_metrics["portfolio_max_drawdown"],
            "Transaction_Cost_USD": full_metrics["total_transaction_cost_USD"],
            "Average_Daily_Cash_Weight": float(full_nav["Cash_Weight"].mean()),
            "Rebalance_Count": int(len(selected_counts)),
            "Mean_Eligible_Assets_Per_Rebalance": float(risky_ticker_count),
        },
        {
            "Configuration": CONFIGURATION,
            "Strategy": SPY_STRATEGY,
            "Selection_Rule": "Buy-and-hold from first common execution open; no transaction fees",
            "Total_Return": spy_metrics["total_return"],
            "CAGR": spy_metrics["CAGR"],
            "Annualized_Sharpe": spy_metrics["sharpe"],
            "Max_Drawdown": spy_metrics["max_drawdown"],
            "Transaction_Cost_USD": 0.0,
            "Average_Daily_Cash_Weight": 0.0,
            "Rebalance_Count": 0,
            "Mean_Eligible_Assets_Per_Rebalance": np.nan,
        },
    ]
    return pd.DataFrame(rows)


# =============================================================================
# MAIN SYSTEM
# =============================================================================

def main() -> None:
    args = arguments()
    validate_arguments(args)

    # Apply requested scenario consistently to all optimizer calculations.
    global MAX_WEIGHT, XGB_STRATEGY
    MAX_WEIGHT = float(args.max_weight)
    XGB_STRATEGY = f"XGBoost_Probability_{args.threshold:.2f}"

    for path, label in [
        (args.predictions, "MAIN S04B probability export"),
        (args.universe, "Investable universe"),
        (args.macro, "DGS3MO risk-free series"),
    ]:
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")

    yahoo_directory = Path(config.INPUT_DATA_DIR) / "yahoo"
    if not yahoo_directory.is_dir():
        raise FileNotFoundError(f"Yahoo input directory not found: {yahoo_directory}")

    args.out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(args.out / "run.log", encoding="utf-8"),
        ],
        force=True,
    )

    risky_tickers = read_universe(args.universe)
    probability_panel, xgb_selections = read_xgboost_probabilities(
        args.predictions,
        risky_tickers,
        args.threshold,
        args.allow_different_dates,
    )

    dates = sorted(xgb_selections)
    full_universe_selections = {
        date: list(risky_tickers)
        for date in dates
    }

    if set(xgb_selections) != set(full_universe_selections):
        raise AssertionError("XGBoost and Full Universe rebalance dates differ.")

    # Write the complete audit before running the optimizer.
    selection_directory = args.out / "selection_audit"
    selection_directory.mkdir(parents=True, exist_ok=True)

    probability_panel.to_csv(
        args.out / "portfolio_signals_all_candidates.csv",
        index=False,
    )
    probability_panel.loc[probability_panel["Selected"].eq(1)].to_csv(
        args.out / "portfolio_signals_selected.csv",
        index=False,
    )
    probability_panel.to_csv(
        selection_directory / "xgboost_probability_audit.csv",
        index=False,
    )

    selected_by_date = (
        probability_panel.loc[probability_panel["Selected"].eq(1)]
        .groupby("Date")["Ticker"]
        .agg(lambda values: ";".join(values))
    )

    selection_summary = (
        probability_panel.groupby("Date", as_index=False)
        .agg(
            Candidate_Assets=("Ticker", "nunique"),
            Selected_Assets=("Selected", "sum"),
            Minimum_Probability=("Probability", "min"),
            Maximum_Probability=("Probability", "max"),
        )
    )
    selection_summary["Threshold"] = args.threshold
    selection_summary["Selected_Tickers"] = (
        selection_summary["Date"].map(selected_by_date).fillna("")
    )
    selection_summary.to_csv(
        selection_directory / "selection_summary_by_rebalance.csv",
        index=False,
    )

    LOG.info(
        "S04B input validated | rows=%d | dates=%d | tickers/date=%d",
        len(probability_panel),
        len(dates),
        len(risky_tickers),
    )
    LOG.info(
        "Threshold-only selection | P >= %.2f | selected asset-date rows=%d | mean selected/date=%.2f",
        args.threshold,
        int(probability_panel["Selected"].sum()),
        float(probability_panel.groupby("Date")["Selected"].sum().mean()),
    )
    LOG.info(
        "Rebalance dates: %s",
        ", ".join(date.strftime("%Y-%m-%d") for date in dates),
    )

    # Regime state is calculated from point-in-time SPY adjusted closes.
    spy_market = read_market("SPY")
    spy_close = spy_market["Adj Close"]

    # Both strategies use the same integrated allocator and are rerun from
    # scratch. The Full Universe is not imported from a previous experiment.
    xgb_result, xgb_diagnostics = run_strategy(
        XGB_STRATEGY,
        xgb_selections,
        args,
        spy_close,
    )
    full_result, full_diagnostics = run_strategy(
        FULL_STRATEGY,
        full_universe_selections,
        args,
        spy_close,
    )

    xgb_nav = xgb_result["nav"].copy()
    full_nav = full_result["nav"].copy()

    xgb_nav["Date"] = pd.to_datetime(xgb_nav["Date"]).dt.normalize()
    full_nav["Date"] = pd.to_datetime(full_nav["Date"]).dt.normalize()

    if not xgb_nav["Date"].equals(full_nav["Date"]):
        raise ValueError(
            "XGBoost and Full Universe daily NAV calendars differ. "
            "The system refuses to compare different evaluation periods."
        )

    if not np.allclose(
        xgb_nav["SPY_NAV"].to_numpy(float),
        full_nav["SPY_NAV"].to_numpy(float),
        rtol=0.0,
        atol=1e-6,
    ):
        raise ValueError(
            "SPY benchmark paths differ between strategy runs. "
            "The system refuses to combine different benchmark paths."
        )

    comparison = pd.DataFrame({
        "Date": xgb_nav["Date"],
        "XGBoost_NAV": xgb_nav["Portfolio_NAV"],
        "XGBoost_Cash_Weight": xgb_nav["Cash_Weight"],
        "XGBoost_Net_Daily_Return": xgb_nav["Net_Daily_Return"],
        "XGBoost_Drawdown": xgb_nav["Portfolio_Drawdown"],
        "Full_Universe_NAV": full_nav["Portfolio_NAV"],
        "Full_Universe_Cash_Weight": full_nav["Cash_Weight"],
        "Full_Universe_Net_Daily_Return": full_nav["Net_Daily_Return"],
        "Full_Universe_Drawdown": full_nav["Portfolio_Drawdown"],
        "SPY_NAV": xgb_nav["SPY_NAV"],
        "SPY_Daily_Return": xgb_nav["SPY_Daily_Return"],
        "SPY_Drawdown": xgb_nav["SPY_Drawdown"],
    })
    comparison.to_csv(args.out / "comparison_daily_nav.csv", index=False)

    performance_summary = build_performance_summary(
        xgb_result,
        full_result,
        xgb_nav,
        full_nav,
        probability_panel,
        args,
        len(risky_tickers),
    )
    performance_summary.to_csv(
        args.out / "performance_summary.csv",
        index=False,
    )

    aggregate_outputs(args.out, xgb_result, full_result)

    regime_diagnostics = pd.DataFrame(xgb_diagnostics + full_diagnostics)
    regime_diagnostics.to_csv(
        args.out / "regime_diagnostics.csv",
        index=False,
    )

    create_comparison_chart(
        comparison,
        args.initial_capital,
        args.out,
        args.threshold,
    )

    manifest = {
        "system": "S06_three_strategy_system.py",
        "configuration": CONFIGURATION,
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_parameters": {"selection_threshold": args.threshold, "maximum_risky_asset_weight": args.max_weight},
        "python_version": platform.python_version(),
        "source_provenance": SOURCE_PROVENANCE,
        "source_files": {
            "predictions": {
                "path": str(args.predictions),
                "sha256": sha256_file(args.predictions),
            },
            "universe": {
                "path": str(args.universe),
                "sha256": sha256_file(args.universe),
            },
            "risk_free": {
                "path": str(args.macro),
                "sha256": sha256_file(args.macro),
            },
        },
        "selection": {
            "model": "XGBoost",
            "source_notebook": "main/S04B_return_classification_xgb.ipynb",
            "rule": f"Probability >= {args.threshold:.2f}",
            "top_k_restriction": None,
            "candidate_assets_per_date": len(risky_tickers),
            "rebalance_count": len(dates),
            "rebalance_dates": [str(date.date()) for date in dates],
            "candidate_probability_rows": int(len(probability_panel)),
            "selected_asset_date_rows": int(probability_panel["Selected"].sum()),
            "mean_selected_assets_per_rebalance": float(
                probability_panel.groupby("Date")["Selected"].sum().mean()
            ),
        },
        "strategies": [
            "XGBoost probability-threshold portfolio",
            "Full Universe portfolio",
            "SPY buy-and-hold benchmark",
        ],
        "comparison_controls": {
            "same_rebalance_dates": True,
            "same_daily_evaluation_calendar": True,
            "same_allocation_code": True,
            "same_regime_estimator": True,
            "same_transaction_fee_helper": True,
            "same_risk_free_series": True,
            "same_initial_capital": True,
            "full_universe_rerun": True,
            "previous_full_universe_nav_reused": False,
            "spy_paths_verified_identical": True,
        },
        "allocation": {
            "method": "Direct joint transaction-cost-aware mean-variance utility",
            "expected_return": (
                "120-session adjusted-price momentum converted to a 60-session "
                "equivalent using geometric compounding"
            ),
            "unconditional_covariance": (
                "252-session Ledoit-Wolf covariance, finite-sample corrected "
                "and scaled to 60 sessions"
            ),
            "conditional_covariance": (
                "Same-regime Ledoit-Wolf covariance estimated over up to "
                "756 historical observations"
            ),
            "covariance_blend_weight": args.regime_condition_weight,
            "risk_aversion_gamma": args.gamma,
            "direct_optimizer_starts": args.direct_starts,
            "maximum_weight_per_risky_asset": MAX_WEIGHT,
            "maximum_cash_weight_when_invested": MAX_CASH_WEIGHT,
            "minimum_risky_exposure_when_invested": MIN_RISKY_WEIGHT,
            "minimum_eligible_risky_assets": 1,
            "minimum_three_asset_restriction": False,
            "previously_held_assets_remain_eligible": True,
            "cash_only_allowed": True,
            "cash_only_rule": (
                "No eligible risky assets, or best incremental net "
                "mean-variance utility versus cash is non-positive"
            ),
        },
        "regime": {
            "state_variable": "SPY realized volatility",
            "realized_volatility_window": args.regime_vol_window,
            "threshold_window": args.regime_threshold_window,
            "threshold_uses_one_session_lag": True,
            "conditional_covariance_lookback": args.regime_lookback,
            "minimum_same_regime_observations": args.regime_min_days,
        },
        "execution": {
            "decision": "Rebalance-date close",
            "execution": "Next SPY trading-session open",
            "spread_and_adv": "Point-in-time signal-date observations",
            "commission_price": "Raw execution-day opening price",
            "cash_rate": "DGS3MO lagged one SPY session",
        },
        "transaction_cost_model": {
            "helper": "transaction_fee_helper.py",
            "components": ["Spread", "Market Impact", "IBKR Commission"],
            "spread_fallback": (
                "Most recent valid spread for gaps up to five ticker observations; "
                "otherwise trailing 252-observation ticker-specific 90th percentile "
                "with at least 20 valid observations; otherwise fail closed"
            ),
            "cash_proxy_transaction_cost": 0.0,
            "spy_benchmark_transaction_cost": 0.0,
        },
        "outputs": [
            "portfolio_signals_all_candidates.csv",
            "portfolio_signals_selected.csv",
            "selection_audit/xgboost_probability_audit.csv",
            "selection_audit/selection_summary_by_rebalance.csv",
            "XGBoost_Probability_0.60/daily_nav.csv",
            "Full_Universe/daily_nav.csv",
            "comparison_daily_nav.csv",
            "performance_summary.csv",
            "all_portfolio_weights.csv",
            "all_rebalance_summaries.csv",
            "all_transaction_costs.csv",
            "transaction_cost_summary.csv",
            "all_spread_fallback_audits.csv",
            "regime_diagnostics.csv",
            "comparison_nav.png",
            "run.log",
        ],
    }

    (args.out / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    LOG.info("S06 THREE-STRATEGY SYSTEM COMPLETED.")
    LOG.info("Output directory: %s", args.out)
    LOG.info("\n%s", performance_summary.to_string(index=False))


if __name__ == "__main__":
    main()
