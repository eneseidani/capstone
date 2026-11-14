"""S06 expected-return and covariance-estimation experiment.

Run from the Capstone project root:
    python S06_expected_return_and_covariance_shrinkage.py

The script preserves the S06 FOUR_FILTERS direct-utility architecture. It does
not introduce a Sharpe-ratio sleeve. It runs the same direct S06 optimizer,
execution engine, constraints and transaction-fee helper for:
  1) original 120-session momentum + original sample covariance (replication)
  2) original 120-session momentum + Ledoit-Wolf covariance
  3) walk-forward calibrated momentum + Ledoit-Wolf covariance
  4) Jorion-style Bayes-Stein historical mean + Ledoit-Wolf covariance

Each configuration is run for the same three candidate universes:
  - Our Random Forest selections
  - Peer XGBoost exported selections (no retraining)
  - Full point-in-time eligible investable universe
SPY buy-and-hold is the benchmark in every chart.

Expected-return calibration is pooled across the full investable universe,
using non-overlapping 60-session historical outcomes that have matured by the
current rebalance date. The calibration is estimated independently at every
rebalance date and never uses future outcomes.

The Bayes-Stein mean estimator is a Jorion-style empirical-Bayes shrinkage of
120-session daily sample means toward the GLS global-minimum-variance mean.
A Ledoit-Wolf covariance estimate is used in the GLS target/shrinkage formula
for numerical stability. The portfolio risk covariance is estimated separately
from 252 daily returns, using either the original sample estimator or
Ledoit-Wolf shrinkage.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

LOG = logging.getLogger("S06_mean_cov_shrinkage")

STRATEGIES = ["Our_RandomForest", "Peer_XGBoost", "Full_Universe"]
STRATEGY_LABELS = {
    "Our_RandomForest": "Our Random Forest",
    "Peer_XGBoost": "Peer XGBoost",
    "Full_Universe": "Full investable universe",
}
COLORS = {
    "Our_RandomForest": "#1f77b4",
    "Peer_XGBoost": "#d62728",
    "Full_Universe": "#2ca02c",
    "SPY": "#ff7f0e",
}

CONFIGS = [
    ("01_RawMomentum_SampleCov", "Raw 120-day momentum", "sample"),
    ("02_RawMomentum_LedoitWolf", "Raw 120-day momentum", "ledoit_wolf"),
    ("03_CalibratedMomentum_LedoitWolf", "Calibrated momentum", "ledoit_wolf"),
    ("04_BayesSteinMean_LedoitWolf", "Jorion-style Bayes-Stein mean", "ledoit_wolf"),
]


def arguments():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--predictions", type=Path,
                   default=Path("model_output/S05_peer_calendar_classification/predictions_test.csv"))
    p.add_argument("--peer-signals", type=Path, default=Path("portfolio_signals.csv"))
    p.add_argument("--peer-schedule", type=Path, default=Path("portfolio_rebalance_schedule.csv"))
    p.add_argument("--peer-hyperparameters", type=Path, default=Path("best_hyperparameters (1).csv"))
    p.add_argument("--universe", type=Path, default=Path("Investable_Universe.csv"))
    p.add_argument("--macro", type=Path, default=Path("features_data/macro_features.csv"))
    p.add_argument("--out", type=Path, default=Path("model_output/S06_expected_return_and_covariance_shrinkage"))
    p.add_argument("--four-filters-dir", type=Path, default=Path("model_output/S06_four_universe_comparison"),
                   help="Optional prior four_filters output folder for baseline replication comparison")
    p.add_argument("--gamma", type=float, default=3.0)
    p.add_argument("--direct-starts", type=int, default=12)
    p.add_argument("--initial-capital", type=float, default=100000.0)
    p.add_argument("--rf-threshold", type=float, default=0.50)
    p.add_argument("--xgb-threshold", type=float, default=0.60)
    p.add_argument("--top-k", type=int, default=15)
    p.add_argument("--calibration-lookback-sessions", type=int, default=756,
                   help="Trailing signal-date window used to estimate momentum calibration beta")
    p.add_argument("--calibration-step-sessions", type=int, default=60,
                   help="Spacing between historical calibration signal dates; 60 avoids overlapping forward labels")
    p.add_argument("--calibration-min-observations", type=int, default=100)
    return p.parse_args()


def read_universe(path: Path, cash: str) -> list[str]:
    d = pd.read_csv(path)
    if "Ticker" not in d:
        raise ValueError(f"Missing Ticker column in {path}")
    raw = d.Ticker.astype(str).str.strip().tolist()
    if any((not t) or t.lower() == "nan" for t in raw) or len(raw) != len(set(raw)):
        raise ValueError("Invalid or duplicated tickers in Investable_Universe.csv")
    if "SPY" not in raw or cash not in raw:
        raise ValueError("Investable universe must include SPY and DGS3MO")
    return [t for t in raw if t != cash]


def read_peer_xgboost(signals_path: Path, schedule_path: Path, calendar: list[pd.Timestamp],
                      allowed: set[str], top_k: int, threshold: float) -> tuple[dict, pd.DataFrame]:
    signals = pd.read_csv(signals_path, parse_dates=["Date"])
    schedule = pd.read_csv(schedule_path, parse_dates=["Date"])
    signal_cols = {"Date", "Model", "Ticker", "Probability", "Rank_Within_Date", "Threshold", "Specification"}
    schedule_cols = {"Date", "Model", "Selected_Assets", "Threshold", "Top_K", "Specification"}
    if not signal_cols.issubset(signals.columns):
        raise ValueError(f"Peer signals missing columns: {signal_cols - set(signals.columns)}")
    if not schedule_cols.issubset(schedule.columns):
        raise ValueError(f"Peer schedule missing columns: {schedule_cols - set(schedule.columns)}")
    if signals.duplicated(["Date", "Model", "Ticker"]).any() or schedule.duplicated(["Date", "Model"]).any():
        raise ValueError("Duplicate peer signal or schedule rows")
    calendar = [pd.Timestamp(x) for x in calendar]
    if set(calendar) != set(schedule.loc[schedule.Model.eq("XGBoost"), "Date"]):
        raise ValueError("Peer XGBoost rebalance dates do not exactly match S05 Random Forest dates")
    sig = signals.loc[signals.Model.eq("XGBoost")].copy()
    sch = schedule.loc[schedule.Model.eq("XGBoost")].copy()
    if set(sig.Date) - set(calendar):
        raise ValueError("Peer XGBoost signals contain dates outside the common calendar")
    if (sch.Threshold.astype(float).sub(threshold).abs() > 1e-9).any():
        raise ValueError("Peer XGBoost threshold differs from configured 0.60")
    if (sch.Top_K.astype(int) != top_k).any():
        raise ValueError("Peer XGBoost Top_K differs from configured Top_K")
    if not sch.Specification.eq("With Regimes").all() or not sig.Specification.eq("With Regimes").all():
        raise ValueError("Expected peer XGBoost specification 'With Regimes'")
    if not sig.Ticker.astype(str).isin(allowed).all():
        raise ValueError("Peer XGBoost contains ticker outside Investable_Universe.csv")

    result, audit = {}, []
    for dt in calendar:
        day = sig.loc[sig.Date.eq(dt)].copy()
        day["Probability"] = pd.to_numeric(day.Probability, errors="raise")
        day["Rank_Within_Date"] = pd.to_numeric(day.Rank_Within_Date, errors="raise")
        if len(day):
            if not np.isfinite(day.Probability.to_numpy(float)).all():
                raise ValueError(f"{dt.date()}: non-finite XGBoost probability")
            if ((day.Probability < threshold - 1e-9) | (day.Probability > 1)).any():
                raise ValueError(f"{dt.date()}: XGBoost probability outside threshold/[0,1]")
            day = day.sort_values("Rank_Within_Date")
            if day.Rank_Within_Date.tolist() != list(range(1, len(day) + 1)):
                raise ValueError(f"{dt.date()}: XGBoost ranks are not consecutive")
            if len(day) > 1 and (day.Probability.to_numpy()[:-1] < day.Probability.to_numpy()[1:] - 1e-10).any():
                raise ValueError(f"{dt.date()}: XGBoost ranks are not descending by probability")
            if (day.Threshold.astype(float).sub(threshold).abs() > 1e-9).any():
                raise ValueError(f"{dt.date()}: signal threshold inconsistent with schedule")
        planned = sch.loc[sch.Date.eq(dt)].iloc[0]
        if len(day) != int(planned.Selected_Assets):
            raise ValueError(f"{dt.date()}: exported XGBoost count differs from schedule")
        if len(day) > top_k:
            raise ValueError(f"{dt.date()}: XGBoost selected more than Top_K assets")
        selected = day.Ticker.astype(str).tolist()
        result[dt] = selected
        audit.append({"Date": dt, "Selected_Count": len(selected), "Selected_Tickers": "|".join(selected),
                      "Peer_Primary_Backtest": bool(planned.get("Primary_Backtest", True)),
                      "Threshold": threshold, "Specification": "With Regimes"})
    return result, pd.DataFrame(audit)


def full_universe_pit(s06, tickers: list[str], dates: list[pd.Timestamp]) -> tuple[dict, pd.DataFrame]:
    """Same point-in-time eligibility logic used by S06_four_universe_comparison."""
    market = {}
    for ticker in tickers:
        try:
            market[ticker] = s06.read_market(ticker)
        except (ValueError, FileNotFoundError) as exc:
            LOG.warning("Market file unusable for %s: %s", ticker, exc)
            market[ticker] = None
    spy = s06.read_market("SPY")
    sessions = spy.index
    eligible_by_date, audit = {}, []
    for dt in dates:
        dt = pd.Timestamp(dt)
        if dt not in sessions:
            raise ValueError(f"Signal date not on SPY calendar: {dt.date()}")
        idx = sessions.get_loc(dt)
        if idx + 1 >= len(sessions):
            raise ValueError(f"No execution session after {dt.date()}")
        execution = sessions[idx + 1]
        eligible = []
        for ticker in tickers:
            d = market[ticker]
            reason = "eligible"
            if d is None:
                reason = "missing_ohlcv"
            else:
                hist = d["Adj Close"].reindex(sessions[:idx + 1]).ffill(limit=2).tail(253)
                if len(hist) != 253 or not np.isfinite(hist.to_numpy(float)).all() or (hist <= 0).any():
                    reason = "insufficient_historical_253_closes"
                elif not np.isfinite(float(d["Open"].get(execution, np.nan))) or float(d["Open"].get(execution, np.nan)) <= 0:
                    reason = "missing_execution_open"
                else:
                    try:
                        s06.load_fee_context(dt, tickers=[ticker, s06.CASH])
                    except (ValueError, KeyError, ArithmeticError, FileNotFoundError) as exc:
                        reason = "missing_point_in_time_fee:" + type(exc).__name__
            if reason == "eligible":
                eligible.append(ticker)
            audit.append({"Signal_Date": dt, "Execution_Date": execution, "Ticker": ticker,
                          "Eligible": reason == "eligible", "Reason": reason})
        eligible_by_date[dt] = eligible
        LOG.info("%s | full universe eligible %d/%d", dt.date(), len(eligible), len(tickers))
    return eligible_by_date, pd.DataFrame(audit)


def load_global_prices(s06, universe: list[str], sessions: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the complete investable universe on SPY sessions for pooled calibration."""
    columns = {}
    failures = []
    for ticker in universe:
        try:
            d = s06.read_market(ticker)
            columns[ticker] = d["Adj Close"].reindex(sessions).ffill(limit=2)
        except (FileNotFoundError, ValueError, KeyError) as exc:
            failures.append({"Ticker": ticker, "Error": str(exc)})
            LOG.warning("Calibration price history unavailable for %s: %s", ticker, exc)
    if "SPY" not in columns:
        raise ValueError("Could not load SPY adjusted-close history for the calibration calendar")
    return pd.DataFrame(columns, index=sessions), pd.DataFrame(failures)


def _safe_condition_number(matrix: np.ndarray) -> float:
    try:
        value = float(np.linalg.cond(np.asarray(matrix, float)))
        return value if np.isfinite(value) else float("inf")
    except np.linalg.LinAlgError:
        return float("inf")


def calibration_beta(date: pd.Timestamp, global_prices: pd.DataFrame, daily_rf: pd.Series,
                     lookback_sessions: int, step_sessions: int,
                     minimum_observations: int) -> dict:
    """Pooled OLS-through-origin calibration of momentum excess-return forecasts.

    Historical signal dates are spaced by one 60-session horizon. For each
    date s, the 120-session momentum forecast is known at s and its subsequent
    60-session return is fully realized by the current date. This avoids
    overlapping forward labels in the calibration sample.
    """
    sessions = global_prices.index
    current_idx = sessions.get_loc(pd.Timestamp(date))
    x_all, y_all, tickers_used, dates_used = [], [], set(), set()
    first_idx = max(120, current_idx - int(lookback_sessions))
    # Start at t-60: its forward 60-session outcome ends at t and is mature.
    for signal_idx in range(current_idx - 60, first_idx - 1, -int(step_sessions)):
        end_idx = signal_idx + 60
        if signal_idx < 120 or end_idx > current_idx:
            continue
        signal_date = sessions[signal_idx]
        if signal_date not in daily_rf.index or not np.isfinite(float(daily_rf.loc[signal_date])):
            continue
        forward_rf = daily_rf.iloc[signal_idx + 1:end_idx + 1].to_numpy(float)
        if len(forward_rf) != 60 or not np.isfinite(forward_rf).all() or (forward_rf <= -1).any():
            continue
        cash_forecast = float(np.expm1(60.0 * np.log1p(float(daily_rf.loc[signal_date]))))
        cash_realized = float(np.expm1(np.log1p(forward_rf).sum()))
        p_start = global_prices.iloc[signal_idx - 120].to_numpy(float)
        p_signal = global_prices.iloc[signal_idx].to_numpy(float)
        p_end = global_prices.iloc[end_idx].to_numpy(float)
        valid = (np.isfinite(p_start) & np.isfinite(p_signal) & np.isfinite(p_end)
                 & (p_start > 0) & (p_signal > 0) & (p_end > 0))
        if not valid.any():
            continue
        momentum = np.expm1(np.log(p_signal[valid] / p_start[valid]) * 0.5)
        realized = p_end[valid] / p_signal[valid] - 1.0
        x = momentum - cash_forecast
        y = realized - cash_realized
        finite = np.isfinite(x) & np.isfinite(y)
        if finite.any():
            x_all.extend(x[finite].tolist())
            y_all.extend(y[finite].tolist())
            valid_columns = np.flatnonzero(valid)[finite]
            tickers_used.update(global_prices.columns[valid_columns].tolist())
            dates_used.add(signal_date)

    x_arr = np.asarray(x_all, float)
    y_arr = np.asarray(y_all, float)
    denominator = float(x_arr @ x_arr) if len(x_arr) else 0.0
    fallback = (len(x_arr) < minimum_observations or len(dates_used) < 6
                or len(tickers_used) < 5 or denominator <= 1e-12)
    if fallback:
        beta = 1.0  # fail-safe: retain the existing raw momentum forecast
        reason = "insufficient_matured_calibration_sample_raw_momentum_used"
    else:
        beta_unconstrained = float((x_arr @ y_arr) / denominator)
        beta = float(np.clip(beta_unconstrained, 0.0, 1.0))
        reason = "estimated_constrained_pooled_ols_through_origin"
    return {
        "Date": pd.Timestamp(date), "Beta": beta,
        "Beta_Unconstrained": (float((x_arr @ y_arr) / denominator) if denominator > 1e-12 else np.nan),
        "Calibration_Observation_Count": int(len(x_arr)),
        "Calibration_Unique_Signal_Dates": int(len(dates_used)),
        "Calibration_Unique_Tickers": int(len(tickers_used)),
        "Calibration_Lookback_Sessions": int(lookback_sessions),
        "Calibration_Step_Sessions": int(step_sessions),
        "Calibration_Fallback": bool(fallback), "Calibration_Method": reason,
    }


def estimate_jorion_bayes_stein(daily_returns: np.ndarray) -> tuple[np.ndarray, dict]:
    """Jorion-style Bayes-Stein shrinkage of daily sample means.

    The GLS shrinkage target is the mean of the global-minimum-variance
    portfolio. Ledoit-Wolf covariance is used in this mean-estimator formula
    to stabilize inversion. This is the mean-vector component of Jorion's
    estimator; S06's portfolio-risk covariance is handled separately.
    """
    x = np.asarray(daily_returns, dtype=float)
    t, n = x.shape
    if t <= n + 2:
        raise ValueError(f"Bayes-Stein requires T > N+2; got T={t}, N={n}")
    if not np.isfinite(x).all():
        raise ValueError("Bayes-Stein daily returns contain missing/non-finite values")
    sample_mean = x.mean(axis=0)
    lw = LedoitWolf(assume_centered=False).fit(x)
    sigma = np.asarray(lw.covariance_, float) * (t / (t - 1.0))
    sigma = 0.5 * (sigma + sigma.T)
    ones = np.ones(n, dtype=float)
    try:
        inv_sigma_ones = np.linalg.solve(sigma, ones)
        inv_sigma_mean = np.linalg.solve(sigma, sample_mean)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Bayes-Stein Ledoit-Wolf covariance is singular") from exc
    denominator = float(ones @ inv_sigma_ones)
    if not np.isfinite(denominator) or denominator <= 0:
        raise ValueError("Invalid GLS denominator in Bayes-Stein estimator")
    mu0 = float((ones @ inv_sigma_mean) / denominator)
    deviation = sample_mean - mu0 * ones
    inv_sigma_deviation = np.linalg.solve(sigma, deviation)
    mahalanobis = max(0.0, float(deviation @ inv_sigma_deviation))
    phi = float((n + 2.0) / (n + 2.0 + t * mahalanobis))
    phi = float(np.clip(phi, 0.0, 1.0))
    shrunk = (1.0 - phi) * sample_mean + phi * mu0 * ones
    diagnostics = {
        "BayesStein_Target_Daily_Return": mu0,
        "BayesStein_Shrinkage_Intensity": phi,
        "BayesStein_Mahalanobis_Distance": mahalanobis,
        "BayesStein_Sample_Size": int(t),
        "BayesStein_Asset_Count": int(n),
        "BayesStein_Mean_Covariance_LW_Shrinkage": float(lw.shrinkage_),
        "BayesStein_Mean_Covariance_Condition_Number": _safe_condition_number(sigma),
    }
    return shrunk, diagnostics


def make_estimator(s06, method: str, covariance_method: str,
                   calibration_by_date: dict[pd.Timestamp, dict],
                   forecast_audit: list[dict], covariance_audit: list[dict]):
    """Create an S06-compatible estimator(date, prices, tickers, daily_rf)."""
    def estimator(prices, date, tickers, daily_rf):
        date = pd.Timestamp(date)
        tickers = list(tickers)
        # Current S06 raw momentum estimator: 121 closes / 120 daily intervals.
        price_window = prices.loc[:date, tickers].tail(s06.EXPECTED_RETURN_LOOKBACK + 1)
        p = price_window.to_numpy(float)
        if (len(price_window) != s06.EXPECTED_RETURN_LOOKBACK + 1
                or not np.isfinite(p).all() or (p <= 0).any()):
            raise ValueError(f"{date.date()}: insufficient price history for 120-session momentum: {tickers}")
        cumulative_120 = p[-1] / p[0] - 1.0
        if (cumulative_120 <= -1.0).any() or not np.isfinite(cumulative_120).all():
            raise ValueError(f"{date.date()}: invalid 120-session cumulative return")
        raw_mu60 = np.expm1(np.log1p(cumulative_120) * (s06.HORIZON / s06.EXPECTED_RETURN_LOOKBACK))

        daily_returns_252 = prices.loc[:date, tickers].pct_change(fill_method=None).tail(s06.COVARIANCE_LOOKBACK)
        x252 = daily_returns_252.to_numpy(float)
        if len(daily_returns_252) != s06.COVARIANCE_LOOKBACK or not np.isfinite(x252).all():
            raise ValueError(f"{date.date()}: incomplete 252-session covariance history for {tickers}")
        sample_cov = np.atleast_2d(np.cov(x252, rowvar=False, ddof=1))
        if covariance_method == "sample":
            daily_cov = sample_cov
            lw_shrinkage = 0.0
        elif covariance_method == "ledoit_wolf":
            lw = LedoitWolf(assume_centered=False).fit(x252)
            daily_cov = np.asarray(lw.covariance_, float) * (len(x252) / (len(x252) - 1.0))
            daily_cov = 0.5 * (daily_cov + daily_cov.T)
            lw_shrinkage = float(lw.shrinkage_)
        else:
            raise ValueError(f"Unknown covariance method: {covariance_method}")
        cov60 = s06.HORIZON * daily_cov

        daily_rate = float(daily_rf.loc[date])
        if not np.isfinite(daily_rate) or daily_rate <= -1.0:
            raise ValueError(f"{date.date()}: invalid DGS3MO one-session rate")
        cash60 = float(np.expm1(s06.HORIZON * np.log1p(daily_rate)))

        method_diagnostics = {}
        if method == "Raw_120D_Momentum":
            mu60 = raw_mu60.copy()
        elif method == "Calibrated_Momentum":
            beta_info = calibration_by_date[date]
            beta = float(beta_info["Beta"])
            mu60 = cash60 + beta * (raw_mu60 - cash60)
            method_diagnostics.update({k: v for k, v in beta_info.items() if k != "Date"})
        elif method == "Jorion_Bayes_Stein":
            daily_returns_120 = prices.loc[:date, tickers].pct_change(fill_method=None).tail(s06.EXPECTED_RETURN_LOOKBACK)
            x120 = daily_returns_120.to_numpy(float)
            if len(daily_returns_120) != s06.EXPECTED_RETURN_LOOKBACK or not np.isfinite(x120).all():
                raise ValueError(f"{date.date()}: incomplete 120-session returns for Bayes-Stein: {tickers}")
            bs_daily, bs_diag = estimate_jorion_bayes_stein(x120)
            # Arithmetic expected return over HORIZON sessions: H * daily mean.
            mu60 = s06.HORIZON * bs_daily
            method_diagnostics.update(bs_diag)
        else:
            raise ValueError(f"Unknown expected-return method: {method}")

        if not np.isfinite(mu60).all() or not np.isfinite(cov60).all():
            raise ValueError(f"{date.date()}: non-finite expected return or covariance")
        cond_sample = _safe_condition_number(sample_cov)
        cond_used = _safe_condition_number(daily_cov)
        covariance_audit.append({
            "Date": date, "Expected_Return_Method": method,
            "Covariance_Method": covariance_method, "Asset_Count": len(tickers),
            "Covariance_Observations": len(x252),
            "Ledoit_Wolf_Shrinkage_Intensity": lw_shrinkage,
            "Sample_Covariance_Condition_Number": cond_sample,
            "Used_Covariance_Condition_Number": cond_used,
            "Sample_Covariance_Min_Eigenvalue": float(np.linalg.eigvalsh(sample_cov).min()),
            "Used_Covariance_Min_Eigenvalue": float(np.linalg.eigvalsh(daily_cov).min()),
        })
        for i, ticker in enumerate(tickers):
            forecast_audit.append({
                "Date": date, "Ticker": ticker, "Expected_Return_Method": method,
                "Covariance_Method": covariance_method,
                "Raw_Momentum_60D": float(raw_mu60[i]),
                "Selected_Expected_Return_60D": float(mu60[i]),
                "Cash_Expected_Return_60D": cash60,
                "Expected_Excess_Return_60D": float(mu60[i] - cash60),
                "Beta": method_diagnostics.get("Beta", np.nan),
                "BayesStein_Target_Daily_Return": method_diagnostics.get("BayesStein_Target_Daily_Return", np.nan),
                "BayesStein_Shrinkage_Intensity": method_diagnostics.get("BayesStein_Shrinkage_Intensity", np.nan),
            })
        return np.asarray(mu60, float), np.asarray(cov60, float), cash60
    return estimator


def s06_namespace(args, out):
    return argparse.Namespace(
        predictions=args.predictions, out=out, threshold=args.rf_threshold,
        model="RandomForest", top_k=args.top_k, initial_capital=args.initial_capital,
        risk_aversion=args.gamma, risk_aversion_grid="none", skip_direct_utility=True,
        direct_starts=args.direct_starts, macro=args.macro,
        allow_feasible_fallback=False, initial_date=None,
    )


def validate_choices(s06, choices: dict, allowed: list[str], dates: list[pd.Timestamp]):
    for strategy, by_date in choices.items():
        if list(by_date) != list(dates):
            raise ValueError(f"{strategy}: rebalance dates differ from RF calendar")
        for dt, selected in by_date.items():
            if s06.CASH in selected or len(selected) != len(set(selected)):
                raise ValueError(f"{strategy} {dt.date()}: cash or duplicate ticker in selection")
            extra = set(selected) - set(allowed)
            if extra:
                raise ValueError(f"{strategy} {dt.date()}: unknown tickers {sorted(extra)}")


def main(args):
    import S06_portfolio_allocation as s06

    if not np.isfinite(args.gamma) or args.gamma <= 0:
        raise ValueError("--gamma must be finite and positive")
    if args.direct_starts < 1 or args.top_k < 1 or args.initial_capital <= 0:
        raise ValueError("Invalid direct starts, Top_K or initial capital")
    if args.calibration_lookback_sessions < 360 or args.calibration_step_sessions < 60:
        raise ValueError("Calibration lookback must be >=360 and step must be >=60 sessions")

    args.out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(args.out / "run.log", encoding="utf-8")],
        force=True)

    universe = read_universe(args.universe, s06.CASH)
    rf_selections = s06.read_selections(args.predictions, "RandomForest", args.rf_threshold, args.top_k)
    dates = list(rf_selections)
    peer_selections, peer_audit = read_peer_xgboost(
        args.peer_signals, args.peer_schedule, dates, set(universe), args.top_k, args.xgb_threshold)
    full_selections, universe_audit = full_universe_pit(s06, universe, dates)
    choices = {
        "Our_RandomForest": rf_selections,
        "Peer_XGBoost": peer_selections,
        "Full_Universe": full_selections,
    }
    validate_choices(s06, choices, universe, dates)

    spy = s06.read_market("SPY")
    sessions = spy.index
    daily_rf = s06.load_daily_cash_rates(args.macro, sessions).reindex(sessions)
    global_prices, load_failures = load_global_prices(s06, universe, sessions)

    # Estimate each calibration coefficient once per rebalance date. The same
    # beta is shared by RF, peer XGBoost and full-universe portfolios.
    calibration_by_date = {}
    calibration_rows = []
    for dt in dates:
        info = calibration_beta(dt, global_prices, daily_rf,
            args.calibration_lookback_sessions, args.calibration_step_sessions,
            args.calibration_min_observations)
        calibration_by_date[pd.Timestamp(dt)] = info
        calibration_rows.append(info)
        LOG.info("%s | momentum calibration beta=%.4f | obs=%d | dates=%d | fallback=%s",
                 pd.Timestamp(dt).date(), info["Beta"], info["Calibration_Observation_Count"],
                 info["Calibration_Unique_Signal_Dates"], info["Calibration_Fallback"])

    peer_audit.to_csv(args.out / "peer_xgboost_selection_audit.csv", index=False)
    universe_audit.to_csv(args.out / "full_universe_eligibility_audit.csv", index=False)
    pd.DataFrame(calibration_rows).to_csv(args.out / "momentum_calibration_audit.csv", index=False)
    load_failures.to_csv(args.out / "calibration_price_load_failures.csv", index=False)
    if args.peer_hyperparameters.exists():
        hp = pd.read_csv(args.peer_hyperparameters)
        (args.out / "peer_xgboost_hyperparameters_metadata.json").write_text(
            hp.to_json(orient="records", indent=2), encoding="utf-8")

    all_nav = {}
    summary_rows, weight_frames, rebalance_frames, trade_frames = [], [], [], []
    forecast_rows, covariance_rows = [], []
    baseline_spy_metrics = None

    for config_name, expected_label, covariance_method in CONFIGS:
        expected_method = {
            "01_RawMomentum_SampleCov": "Raw_120D_Momentum",
            "02_RawMomentum_LedoitWolf": "Raw_120D_Momentum",
            "03_CalibratedMomentum_LedoitWolf": "Calibrated_Momentum",
            "04_BayesSteinMean_LedoitWolf": "Jorion_Bayes_Stein",
        }[config_name]
        all_nav[config_name] = {}
        for strategy in STRATEGIES:
            directory = args.out / config_name / strategy
            settings = s06_namespace(args, directory)
            estimator = make_estimator(s06, expected_method, covariance_method,
                                       calibration_by_date, forecast_rows, covariance_rows)
            LOG.info("START | %s | %s | covariance=%s | expected=%s",
                     config_name, strategy, covariance_method, expected_method)
            with patch.object(s06, "read_selections", return_value=choices[strategy]), \
                 patch.object(s06, "estimation", estimator):
                result = s06.run_single(settings, mode="direct")
            nav = pd.read_csv(directory / "daily_nav.csv", parse_dates=["Date"])
            all_nav[config_name][strategy] = nav
            metric = result["results"]
            if baseline_spy_metrics is None:
                baseline_spy_metrics = metric
            rebalance = pd.read_csv(directory / "rebalance_summary.csv")
            weights = pd.read_csv(directory / "portfolio_weights.csv")
            trades = pd.read_csv(directory / "transaction_cost_breakdown.csv")
            for frame, sink, label in [
                (rebalance, rebalance_frames, "Rebalance"),
                (weights, weight_frames, "Weights"),
                (trades, trade_frames, "Trades"),
            ]:
                if not frame.empty:
                    frame.insert(0, "Configuration", config_name)
                    frame.insert(1, "Expected_Return_Method", expected_method)
                    frame.insert(2, "Covariance_Method", covariance_method)
                    frame.insert(3, "Strategy", strategy)
                    sink.append(frame)
            summary_rows.append({
                "Configuration": config_name, "Expected_Return_Method": expected_method,
                "Covariance_Method": covariance_method, "Strategy": strategy,
                "Total_Return": metric["portfolio_total_return"],
                "CAGR": metric["portfolio_CAGR"],
                "Annualized_Sharpe": metric["portfolio_annualized_sharpe"],
                "Max_Drawdown": metric["portfolio_max_drawdown"],
                "Transaction_Cost_USD": metric["total_transaction_cost_USD"],
                "Average_Daily_Cash_Weight": float(nav.Cash_Weight.mean()),
                "Average_Risky_Weight": float(nav.Risky_Weight.mean()),
                "Average_Eligible_Assets": float(rebalance.Eligible_Including_Holdings.mean()) if len(rebalance) else 0.0,
                "Average_Risky_Positions": float(rebalance.Actual_Risky_Positions.mean()) if len(rebalance) else 0.0,
                "Rebalances": int(len(rebalance)),
            })
            LOG.info("END | %s | %s | return=%+.2f%% | Sharpe=%.3f | costs=$%.2f",
                     config_name, strategy, 100 * metric["portfolio_total_return"],
                     metric["portfolio_annualized_sharpe"], metric["total_transaction_cost_USD"])

    # Four charts: original four_filters replication, covariance-only change,
    # calibrated momentum, and Jorion-style Bayes-Stein expected means.
    anchor = all_nav[CONFIGS[0][0]]["Our_RandomForest"]
    daily_comparison = pd.DataFrame({"Date": anchor.Date})
    for config_name, expected_label, covariance_method in CONFIGS:
        for strategy in STRATEGIES:
            nav = all_nav[config_name][strategy]
            if not anchor.Date.equals(nav.Date):
                raise AssertionError(f"NAV dates misaligned: {config_name}/{strategy}")
            if not np.allclose(anchor.SPY_NAV.to_numpy(float), nav.SPY_NAV.to_numpy(float), rtol=0, atol=1e-5):
                raise AssertionError(f"SPY NAV differs: {config_name}/{strategy}")
            daily_comparison[f"{config_name}_{strategy}_NAV"] = nav.Portfolio_NAV.to_numpy(float)
            daily_comparison[f"{config_name}_{strategy}_Cash_Weight"] = nav.Cash_Weight.to_numpy(float)
        daily_comparison[f"{config_name}_SPY_NAV"] = anchor.SPY_NAV.to_numpy(float)

        fig, ax = plt.subplots(figsize=(13, 7))
        for strategy in STRATEGIES:
            nav = all_nav[config_name][strategy]
            ax.plot(nav.Date, nav.Portfolio_NAV / args.initial_capital * 100,
                    label=STRATEGY_LABELS[strategy], color=COLORS[strategy], lw=1.9)
        ax.plot(anchor.Date, anchor.SPY_NAV / args.initial_capital * 100,
                label="SPY buy-and-hold", color=COLORS["SPY"], lw=2.0, ls="--")
        ax.set(title=f"{config_name} | {expected_label} + {covariance_method} covariance | net of costs",
               xlabel="Date", ylabel="Portfolio NAV (initial = 100)")
        ax.grid(alpha=0.25); ax.legend(ncol=2); fig.autofmt_xdate(); fig.tight_layout()
        fig.savefig(args.out / f"{config_name.lower()}_vs_SPY.png", dpi=185)
        plt.close(fig)

    # Add the common SPY benchmark as a separate summary row per configuration.
    for config_name, _, _ in CONFIGS:
        spy = anchor.SPY_NAV.to_numpy(float)
        years = (pd.Timestamp(anchor.Date.iloc[-1]) - pd.Timestamp(anchor.Date.iloc[0])).days / 365.25
        spy_total = float(spy[-1] / spy[0] - 1.0)
        spy_cagr = float((spy[-1] / spy[0]) ** (1.0 / years) - 1.0) if years > 0 else np.nan
        summary_rows.append({
            "Configuration": config_name, "Expected_Return_Method": "Benchmark",
            "Covariance_Method": "Not applicable", "Strategy": "SPY_Buy_And_Hold",
            "Total_Return": spy_total, "CAGR": spy_cagr,
            "Annualized_Sharpe": baseline_spy_metrics["spy_annualized_sharpe"],
            "Max_Drawdown": baseline_spy_metrics["spy_max_drawdown"],
            "Transaction_Cost_USD": 0.0, "Average_Daily_Cash_Weight": 0.0,
            "Average_Risky_Weight": 1.0, "Average_Eligible_Assets": np.nan,
            "Average_Risky_Positions": 1.0, "Rebalances": 0,
        })

    pd.DataFrame(summary_rows).to_csv(args.out / "expected_return_covariance_performance_summary.csv", index=False)
    daily_comparison.to_csv(args.out / "comparison_daily_nav.csv", index=False)
    if forecast_rows:
        pd.DataFrame(forecast_rows).to_csv(args.out / "expected_return_forecasts.csv", index=False)
    if covariance_rows:
        pd.DataFrame(covariance_rows).to_csv(args.out / "covariance_diagnostics.csv", index=False)
    if weight_frames:
        pd.concat(weight_frames, ignore_index=True).to_csv(args.out / "all_portfolio_weights.csv", index=False)
    if rebalance_frames:
        pd.concat(rebalance_frames, ignore_index=True).to_csv(args.out / "all_rebalance_summaries.csv", index=False)
    if trade_frames:
        pd.concat(trade_frames, ignore_index=True).to_csv(args.out / "all_transaction_costs.csv", index=False)

    # If the previous four_filters output exists, compare the raw-momentum /
    # sample-covariance replication with the prior saved performance summary.
    replication = []
    prior_summary_path = args.four_filters_dir / "comparison_summary.csv"
    if prior_summary_path.exists():
        prior = pd.read_csv(prior_summary_path)
        current = pd.DataFrame(summary_rows)
        baseline_rows = current.loc[current.Configuration.eq(CONFIGS[0][0])]
        for strategy in STRATEGIES:
            old = prior.loc[prior.Strategy.eq(strategy)]
            new = baseline_rows.loc[baseline_rows.Strategy.eq(strategy)]
            if len(old) == 1 and len(new) == 1:
                old_ret = float(old.Total_Return.iloc[0])
                new_ret = float(new.Total_Return.iloc[0])
                replication.append({"Strategy": strategy, "Prior_Four_Filters_Total_Return": old_ret,
                                    "Current_Baseline_Total_Return": new_ret,
                                    "Difference": new_ret - old_ret,
                                    "Matches_Within_1bp": abs(new_ret - old_ret) <= 0.0001})
        if replication:
            pd.DataFrame(replication).to_csv(args.out / "four_filters_replication_check.csv", index=False)

    manifest = {
        "script": Path(__file__).name,
        "purpose": "Isolate covariance shrinkage and expected-return estimation effects under the four_filters direct S06 allocator",
        "allocator": "Native S06 run_single(mode='direct') and native optimize_direct_utility",
        "stage1_sharpe_optimizer_used": False,
        "gamma": args.gamma,
        "direct_multistart_count": args.direct_starts,
        "maximum_individual_weight": s06.MAX_WEIGHT,
        "maximum_normal_cash_weight": s06.MAX_CASH_WEIGHT,
        "minimum_risky_weight_when_invested": s06.MIN_RISKY_WEIGHT,
        "universe_strategies": STRATEGIES,
        "peer_xgboost_retrained": False,
        "peer_xgboost_source": str(args.peer_signals),
        "peer_xgboost_schedule_source": str(args.peer_schedule),
        "rebalance_dates": [str(pd.Timestamp(x).date()) for x in dates],
        "expected_return_covariance_configurations": [
            {"name": name, "expected_return": method, "covariance": cov} for name, method, cov in CONFIGS
        ],
        "raw_momentum_formula": "(P_t/P_t-120)^(60/120)-1",
        "calibrated_momentum_formula": "cash_60 + beta_t*(raw_momentum_60-cash_60); beta estimated by pooled constrained OLS through origin on full-universe historical forecasts whose non-overlapping 60-session outcomes have matured by t",
        "calibration_lookback_sessions": args.calibration_lookback_sessions,
        "calibration_step_sessions": args.calibration_step_sessions,
        "calibration_minimum_observations": args.calibration_min_observations,
        "bayes_stein_formula": "mu_BS=(1-phi)*sample_mean+phi*mu_GLS; phi=(N+2)/(N+2+T*(mu-mu_GLS*1)'S_LW^-1*(mu-mu_GLS*1)); daily sample mean over 120 sessions scaled linearly by 60",
        "bayes_stein_covariance_for_mean_shrinkage": "120-session Ledoit-Wolf covariance, used only to stabilize GLS target and Mahalanobis distance",
        "portfolio_risk_covariance": "252 daily returns; original sample covariance in exact four_filters replication and Ledoit-Wolf covariance in the other three configurations; scaled by 60",
        "ledoit_wolf_scaling": "sklearn LedoitWolf covariance multiplied by T/(T-1) to align with the ddof=1 sample covariance scale before multiplying by 60",
        "execution": "Signal close decision; next SPY session adjusted open execution; native S06 fee helper",
        "transaction_costs": "Native S06 spread + market impact + IBKR commissions, charged in NAV",
        "cash_proxy": s06.CASH,
        "benchmark": "SPY buy-and-hold from first next-session adjusted open, no benchmark fees",
        "known_caveats": [
            "The calibration beta is pooled across the investable universe and is common to RF, peer XGBoost and full-universe strategies at each date.",
            "Calibration observations use non-overlapping 60-session forward horizons, but cross-sectional asset returns remain dependent.",
            "The Jorion-style mean estimator uses Ledoit-Wolf covariance in the GLS shrinkage calculation; it is an explicitly documented regularized implementation of the Jorion mean shrinkage formula.",
            "Bayes-Stein expected returns use 120-session arithmetic daily sample means scaled by 60; raw and calibrated momentum use geometric 120-to-60 conversion.",
            "The prior test period has already been inspected in previous experiments; results are exploratory, not a pristine independent holdout.",
            "S06 retains previously held tickers as eligible candidates even if they are no longer selected at the current date, matching four_filters behavior."
        ],
    }
    (args.out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    LOG.info("COMPLETED expected-return/covariance experiment. Results: %s", args.out)
    LOG.info("\n%s", pd.DataFrame(summary_rows).to_string(index=False))
    return pd.DataFrame(summary_rows)


if __name__ == "__main__":
    main(arguments())
