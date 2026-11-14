"""
S06: XGBoost-only rerun using Regime-Aware Ledoit-Wolf, with historical
Full Universe and SPY NAV series reused for the three-line comparison.

Run from the Capstone project root:
    python S06_regime_aware_xgboost_only_no_min3.py

What this script runs:
    - One allocation run: Peer_XGBoost using the 0.60 probability selection.
    - One optimizer configuration: Regime-Aware Ledoit-Wolf.
    - No Full Universe allocation rerun.
    - No Random Forest or other allocation methods.

What this script reuses:
    - Full Universe daily NAV and cash weights from the existing
      comparison_daily_nav.csv, specifically the 04_RegimeAware_LedoitWolf
      columns.
    - SPY buy-and-hold NAV from the SPY_NAV column in the same daily_nav.csv.
    - Full Universe and SPY performance metrics from the existing
      allocation_methods_performance_summary.csv.

The existing comparison files must correspond to the previous accepted
Regime-Aware Ledoit-Wolf comparison. The script validates date alignment
and checks the SPY benchmark produced internally by S06 against the stored
SPY series, but plots and reports the stored benchmark series.

The allocator imported is S06_portfolio_allocation_no_min3.py, which removes
the arbitrary minimum-three-eligible-assets restriction. One- and two-asset
candidate sets can therefore reach the optimizer. Zero selected assets remain
a valid all-cash rebalance, subject to the allocation engine's existing
holdings-continuity rules.

XGBoost input:
    portfolio_signals_all_candidates.csv
It must contain all 48 tickers for each of the 19 rebalance dates, including
dates where no ticker passes the threshold.
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


LOG = logging.getLogger("S06_regime_aware_xgboost_only_no_min3")

CONFIG = "04_RegimeAware_LedoitWolf"
STRATEGY = "Peer_XGBoost"
FULL_UNIVERSE_NAV_COLUMN = f"{CONFIG}_Full_Universe_NAV"
FULL_UNIVERSE_CASH_COLUMN = f"{CONFIG}_Full_Universe_Cash_Weight"
SPY_NAV_COLUMN = f"{CONFIG}_SPY_NAV"

COLORS = {
    "Peer_XGBoost": "#d62728",
    "Full_Universe": "#2ca02c",
    "SPY": "#ff7f0e",
}


def arguments():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--peer-signals",
        type=Path,
        default=Path("portfolio_signals_all_candidates.csv"),
        help="Complete 48-ticker x 19-date XGBoost prediction panel.",
    )
    p.add_argument(
        "--universe",
        type=Path,
        default=Path("Investable_Universe.csv"),
    )
    p.add_argument(
        "--macro",
        type=Path,
        default=Path("features_data/macro_features.csv"),
    )
    p.add_argument(
        "--full-universe-nav",
        type=Path,
        default=Path(
            "model_output/S06_blacklitterman_robust_regime_comparison/"
            "04_RegimeAware_LedoitWolf/Full_Universe/daily_nav.csv"
        ),
        help=(
            "Existing Full Universe daily_nav.csv. Its SPY_NAV column is reused "
            "as the SPY benchmark; neither strategy is rerun."
        ),
    )
    p.add_argument(
        "--comparison-summary",
        type=Path,
        default=Path(
            "model_output/S06_blacklitterman_robust_regime_comparison/"
            "allocation_methods_performance_summary.csv"
        ),
        help=(
            "Optional previous summary. If absent, Full Universe and SPY "
            "performance metrics are calculated from daily_nav.csv."
        ),
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("model_output/S06_regime_aware_xgboost_only_no_min3"),
    )
    p.add_argument("--gamma", type=float, default=3.0)
    p.add_argument("--direct-starts", type=int, default=12)
    p.add_argument("--initial-capital", type=float, default=100000.0)
    p.add_argument("--xgb-threshold", type=float, default=0.60)
    p.add_argument("--expected-tickers", type=int, default=48)
    p.add_argument("--regime-lookback", type=int, default=756)
    p.add_argument("--regime-vol-window", type=int, default=21)
    p.add_argument("--regime-threshold-window", type=int, default=252)
    p.add_argument("--regime-min-days", type=int, default=90)
    p.add_argument(
        "--regime-condition-weight",
        type=float,
        default=0.5,
        help="Blend weight on regime-conditional Ledoit-Wolf covariance.",
    )
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

def read_peer_xgboost(
    signals_path: Path,
    allowed: set[str],
    threshold: float,
    expected_tickers: int,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """
    Read the full ticker/date prediction panel and use its Selected flag.

    Required columns:
        Date, Ticker, Probability, Rank, Selected

    The file must contain every candidate ticker on every rebalance date.
    This is important because a date with zero selected assets must still
    remain in the rebalance calendar.
    """
    if not signals_path.exists():
        raise FileNotFoundError(
            f"XGBoost selection input not found: {signals_path}\n"
            "Use portfolio_signals_all_candidates.csv (912 rows), not the "
            "selected-only 520-row portfolio_signals.csv."
        )

    signals = pd.read_csv(signals_path)

    required = {"Date", "Ticker", "Probability", "Rank", "Selected"}
    missing = required.difference(signals.columns)
    if missing:
        raise ValueError(
            f"XGBoost input is missing columns: {sorted(missing)}. "
            "Expected Date, Ticker, Probability, Rank, Selected."
        )

    signals = signals.copy()
    signals["Date"] = pd.to_datetime(signals["Date"], errors="raise").dt.normalize()
    signals["Ticker"] = signals["Ticker"].astype(str).str.strip()
    signals["Probability"] = pd.to_numeric(signals["Probability"], errors="raise")
    signals["Rank"] = pd.to_numeric(signals["Rank"], errors="raise")
    signals["Selected"] = pd.to_numeric(signals["Selected"], errors="raise")

    if signals["Ticker"].eq("").any() or signals["Ticker"].str.lower().eq("nan").any():
        raise ValueError("XGBoost input contains blank ticker values.")
    if signals[["Probability", "Rank", "Selected"]].isna().any().any():
        raise ValueError("XGBoost input contains missing Probability, Rank or Selected values.")
    if not np.isfinite(signals["Probability"].to_numpy(float)).all():
        raise ValueError("XGBoost probabilities contain non-finite values.")
    if not signals["Probability"].between(0.0, 1.0).all():
        raise ValueError("XGBoost probabilities must be between zero and one.")
    if not signals["Rank"].ge(1).all() or not np.equal(
        signals["Rank"].to_numpy(float),
        np.floor(signals["Rank"].to_numpy(float)),
    ).all():
        raise ValueError("XGBoost Rank must contain positive integers.")
    if not signals["Selected"].isin([0, 1]).all():
        raise ValueError("XGBoost Selected column must contain only 0 or 1.")

    signals["Rank"] = signals["Rank"].astype(int)
    signals["Selected"] = signals["Selected"].astype(int)

    if signals.duplicated(["Date", "Ticker"]).any():
        duplicate_rows = signals.loc[
            signals.duplicated(["Date", "Ticker"], keep=False),
            ["Date", "Ticker"],
        ]
        raise ValueError(
            "Duplicate XGBoost Date/Ticker rows found:\n"
            + duplicate_rows.head(20).to_string(index=False)
        )

    outside = sorted(set(signals["Ticker"]) - allowed)
    if outside:
        raise ValueError(
            "XGBoost input contains tickers outside Investable_Universe.csv: "
            + ", ".join(outside)
        )

    # Recalculate the threshold flag rather than trusting a possibly stale
    # selection flag. The file's flag must agree with the requested rule.
    expected_selected = (signals["Probability"] >= threshold).astype(int)
    if not np.array_equal(signals["Selected"].to_numpy(), expected_selected.to_numpy()):
        mismatches = signals.loc[
            signals["Selected"].ne(expected_selected),
            ["Date", "Ticker", "Probability", "Selected"],
        ].head(20)
        raise ValueError(
            f"Selected flags do not match Probability >= {threshold:.2f}. "
            "Regenerate or correct the input file. Examples:\n"
            + mismatches.to_string(index=False)
        )

    dates = sorted(pd.Timestamp(x) for x in signals["Date"].unique())
    if not dates:
        raise ValueError("No rebalance dates found in XGBoost input.")

    selected_by_date = {}
    for dt in dates:
        day = signals.loc[signals["Date"].eq(dt)].copy()

        if len(day) != expected_tickers:
            raise ValueError(
                f"{dt.date()}: expected {expected_tickers} ticker rows but found "
                f"{len(day)}. The selected-only file is incomplete for dates "
                "with zero or fewer selected tickers; use the 912-row "
                "portfolio_signals_all_candidates.csv."
            )

        if set(day["Ticker"]) != set(signals.loc[signals["Date"].eq(dt), "Ticker"]):
            raise AssertionError(f"{dt.date()}: ticker panel validation failed.")

        day = day.sort_values("Rank")
        if day["Rank"].tolist() != list(range(1, expected_tickers + 1)):
            raise ValueError(f"{dt.date()}: Rank must run from 1 to {expected_tickers}.")
        probs = day["Probability"].to_numpy(float)
        if len(probs) > 1 and (probs[:-1] < probs[1:] - 1e-10).any():
            raise ValueError(f"{dt.date()}: ranks are not in descending probability order.")

        selected = day.loc[day["Selected"].eq(1)].sort_values("Rank")
        selected_by_date[dt] = selected["Ticker"].astype(str).tolist()

        LOG.info(
            "%s | XGBoost selected %d/%d assets at probability >= %.2f",
            dt.date(), len(selected), expected_tickers, threshold,
        )

    # Keep the audit ticker-level: one row per ticker per rebalance date.
    audit = signals.sort_values(["Date", "Rank", "Ticker"]).reset_index(drop=True)
    return selected_by_date, audit, signals

def covariance_lw(x: np.ndarray, horizon: int=60) -> tuple[np.ndarray,float]:
    x=np.asarray(x, dtype=float)
    if x.ndim!=2 or x.shape[0]<3 or x.shape[1]<1 or not np.isfinite(x).all():
        raise ValueError('Covariance requires a complete finite return matrix')
    lw=LedoitWolf(assume_centered=False).fit(x)
    cov=np.atleast_2d(lw.covariance_) * (len(x)/(len(x)-1.)) * horizon
    cov=(cov+cov.T)/2
    return cov,float(lw.shrinkage_)

def raw_estimation(s06, prices, date, tickers, rf):
    p=prices.loc[:date,tickers].tail(s06.EXPECTED_RETURN_LOOKBACK+1)
    if len(p)!=s06.EXPECTED_RETURN_LOOKBACK+1 or not np.isfinite(p.to_numpy(float)).all() or (p.to_numpy(float)<=0).any():
        raise ValueError(f'{date}: incomplete raw-momentum lookback')
    mu=np.expm1(np.log(p.iloc[-1].to_numpy(float)/p.iloc[0].to_numpy(float))*
                (s06.HORIZON/s06.EXPECTED_RETURN_LOOKBACK))
    dr=prices.loc[:date,tickers].pct_change(fill_method=None).tail(s06.COVARIANCE_LOOKBACK)
    if len(dr)!=s06.COVARIANCE_LOOKBACK or not np.isfinite(dr.to_numpy(float)).all():
        raise ValueError(f'{date}: incomplete Ledoit-Wolf lookback')
    cov, intensity=covariance_lw(dr.to_numpy(float),s06.HORIZON)
    day_rf=float(rf.loc[date]); cash60=float(np.expm1(s06.HORIZON*np.log1p(day_rf)))
    return mu,cov,cash60,intensity

def regime_covariance(prices,date,tickers,spy_close,base_cov,args):
    """Regime-conditioned LW with a past-only SPY volatility state variable.

    State is 21-day SPY realized volatility relative to median 21-day
    volatility over the preceding 252 sessions (shifted by one day). Daily
    return dates classified using contemporaneously observed closes; regime
    conditioning is estimated only from historical dates <= decision date.
    """
    spy=spy_close.loc[:date]
    spy_ret=spy.pct_change(fill_method=None)
    v=spy_ret.rolling(args.regime_vol_window,min_periods=args.regime_vol_window).std()
    threshold=v.shift(1).rolling(args.regime_threshold_window,
                                  min_periods=args.regime_threshold_window).median()
    regime=(v>threshold).where(v.notna() & threshold.notna())
    current=regime.loc[date]
    if pd.isna(current): raise ValueError(f'{date}: insufficient SPY regime history')
    state='High_Vol' if bool(current) else 'Low_Vol'
    hist=prices.loc[:date,tickers].pct_change(fill_method=None).tail(args.regime_lookback)
    mask=regime.reindex(hist.index).eq(bool(current)).fillna(False)
    sub=hist.loc[mask]
    sub=sub.loc[np.isfinite(sub.to_numpy(float)).all(axis=1)]
    if len(sub)<args.regime_min_days:
        LOG.warning('%s regime %s has only %d historical days: using unconditional LW',
                    date.date(),state,len(sub))
        return base_cov,{'Regime':state,'State_Observations':int(len(sub)),
                         'Regime_Fallback':True,'Regime_Covariance_LW_Shrinkage':np.nan,
                         'SPY_Realized_Vol':float(v.loc[date]),
                         'SPY_Regime_Threshold':float(threshold.loc[date])}
    cond,shrink=covariance_lw(sub.to_numpy(float),60)
    blended=args.regime_condition_weight*cond+(1.-args.regime_condition_weight)*base_cov
    return blended,{'Regime':state,'State_Observations':int(len(sub)),
                    'Regime_Fallback':False,'Regime_Covariance_LW_Shrinkage':shrink,
                    'SPY_Realized_Vol':float(v.loc[date]),
                    'SPY_Regime_Threshold':float(threshold.loc[date])}

def make_estimator(s06, args, spy_close, diagnostics):
    """Use raw momentum expected returns and regime-aware Ledoit-Wolf covariance."""
    def estimator(prices, date, tickers, rf):
        date = pd.Timestamp(date)
        mu, base_cov, cash60, shrinkage = raw_estimation(
            s06, prices, date, tickers, rf
        )
        regime_cov, regime_diag = regime_covariance(
            prices, date, tickers, spy_close, base_cov, args
        )
        diagnostics.append({
            "Configuration": CONFIG,
            "Signal_Date": str(date.date()),
            "Asset_Count": len(tickers),
            "Baseline_LW_Shrinkage": shrinkage,
            "Covariance_Condition_Number": float(np.linalg.cond(regime_cov)),
            "Raw_Momentum_Mean_60D": float(mu.mean()),
            "Cash_60D": cash60,
            **regime_diag,
        })
        return mu, regime_cov, cash60

    return estimator

def s06_settings(args, out):
    """
    Build the settings object expected by S06_portfolio_allocation.run_single.

    read_selections is patched during each run, so no separate RF or peer
    prediction file is loaded by the S06 allocation engine.
    """
    return argparse.Namespace(
        predictions=args.peer_signals,
        out=out,
        threshold=args.xgb_threshold,
        model="XGBoost",
        top_k=args.expected_tickers,
        initial_capital=args.initial_capital,
        risk_aversion=args.gamma,
        risk_aversion_grid="none",
        skip_direct_utility=True,
        direct_starts=args.direct_starts,
        macro=args.macro,
        allow_feasible_fallback=False,
        initial_date=None,
    )


def _nav_metrics(nav: pd.Series, daily_returns: pd.Series, drawdown: pd.Series) -> dict:
    """Calculate NAV-based metrics when the original summary file is unavailable.

    Sharpe is calculated with a zero daily risk-free proxy because the supplied
    daily_nav.csv does not contain a risk-free return series. Transaction costs
    cannot be recovered from NAV alone and are therefore left missing.
    """
    nav = pd.to_numeric(nav, errors="raise").astype(float)
    returns = pd.to_numeric(daily_returns, errors="coerce").astype(float).dropna()
    drawdown = pd.to_numeric(drawdown, errors="coerce").astype(float)

    total_return = float(nav.iloc[-1] / nav.iloc[0] - 1.0)
    elapsed_years = (nav.index[-1] - nav.index[0]).days / 365.25
    cagr = float((nav.iloc[-1] / nav.iloc[0]) ** (1.0 / elapsed_years) - 1.0) if elapsed_years > 0 else np.nan
    volatility = float(returns.std(ddof=1)) if len(returns) > 1 else np.nan
    sharpe = (
        float(returns.mean() / volatility * np.sqrt(252.0))
        if np.isfinite(volatility) and volatility > 0
        else np.nan
    )
    return {
        "Total_Return": total_return,
        "CAGR": cagr,
        "Annualized_Sharpe": sharpe,
        "Max_Drawdown": float(drawdown.min()),
    }


def load_previous_comparison(args):
    """Load existing Full Universe NAV and its embedded SPY benchmark.

    Expected source format is the daily_nav.csv from:
      .../04_RegimeAware_LedoitWolf/Full_Universe/daily_nav.csv

    It contains Portfolio_NAV, Cash_Weight and SPY_NAV. No Full Universe or
    SPY allocation/backtest is run by this script.
    """
    nav_path = args.full_universe_nav
    if not nav_path.exists():
        raise FileNotFoundError(
            "Previous Full Universe daily_nav.csv not found. Expected the file "
            "from the Full_Universe folder. Supply its location with "
            "--full-universe-nav. Looked for: " + str(nav_path)
        )

    source = pd.read_csv(nav_path)
    required = {"Date", "Portfolio_NAV", "Cash_Weight", "SPY_NAV"}
    missing = required.difference(source.columns)
    if missing:
        raise ValueError(
            f"{nav_path} is missing required columns: " + ", ".join(sorted(missing))
        )

    source["Date"] = pd.to_datetime(source["Date"], errors="raise").dt.normalize()
    source = source.sort_values("Date").reset_index(drop=True)
    if source["Date"].duplicated().any():
        raise ValueError(f"{nav_path} contains duplicate dates.")

    numeric_columns = ["Portfolio_NAV", "Cash_Weight", "SPY_NAV"]
    for col in numeric_columns:
        source[col] = pd.to_numeric(source[col], errors="raise")
    if source[numeric_columns].isna().any().any():
        raise ValueError(f"{nav_path} contains missing NAV or cash-weight values.")
    if not np.isfinite(source[numeric_columns].to_numpy(float)).all():
        raise ValueError(f"{nav_path} contains non-finite NAV or cash-weight values.")
    if (source["Portfolio_NAV"] <= 0).any() or (source["SPY_NAV"] <= 0).any():
        raise ValueError("Full Universe and SPY NAV values must be strictly positive.")
    if not source["Cash_Weight"].between(-1e-8, 1.0 + 1e-8).all():
        raise ValueError("Full Universe Cash_Weight must be between zero and one.")

    previous_nav = pd.DataFrame({
        "Date": source["Date"],
        FULL_UNIVERSE_NAV_COLUMN: source["Portfolio_NAV"],
        FULL_UNIVERSE_CASH_COLUMN: source["Cash_Weight"],
        SPY_NAV_COLUMN: source["SPY_NAV"],
    })

    # Use the exact historical summary if it exists; otherwise calculate
    # comparable NAV-based metrics and leave unavailable transaction costs blank.
    needed = None
    summary_path = args.comparison_summary
    if summary_path and summary_path.exists():
        previous_summary = pd.read_csv(summary_path)
        if {"Configuration", "Strategy"}.issubset(previous_summary.columns):
            candidate = previous_summary[
                previous_summary["Configuration"].eq(CONFIG)
                & previous_summary["Strategy"].isin(["Full_Universe", "SPY_Buy_And_Hold"])
            ].copy()
            if set(candidate["Strategy"]) == {"Full_Universe", "SPY_Buy_And_Hold"} and not candidate["Strategy"].duplicated().any():
                needed = candidate

    if needed is None:
        indexed = source.set_index("Date")
        if "Net_Daily_Return" in source.columns:
            full_returns = pd.to_numeric(source["Net_Daily_Return"], errors="coerce")
        else:
            full_returns = source["Portfolio_NAV"].pct_change()
        if "SPY_Daily_Return" in source.columns:
            spy_returns = pd.to_numeric(source["SPY_Daily_Return"], errors="coerce")
        else:
            spy_returns = source["SPY_NAV"].pct_change()

        if "Portfolio_Drawdown" in source.columns:
            full_dd = pd.to_numeric(source["Portfolio_Drawdown"], errors="coerce")
        else:
            full_dd = source["Portfolio_NAV"] / source["Portfolio_NAV"].cummax() - 1.0
        if "SPY_Drawdown" in source.columns:
            spy_dd = pd.to_numeric(source["SPY_Drawdown"], errors="coerce")
        else:
            spy_dd = source["SPY_NAV"] / source["SPY_NAV"].cummax() - 1.0

        full_metrics = _nav_metrics(
            indexed["Portfolio_NAV"],
            full_returns,
            full_dd,
        )
        spy_metrics = _nav_metrics(
            indexed["SPY_NAV"],
            spy_returns,
            spy_dd,
        )
        needed = pd.DataFrame([
            {
                "Configuration": CONFIG,
                "Strategy": "Full_Universe",
                **full_metrics,
                "Transaction_Cost_USD": np.nan,
                "Average_Daily_Cash_Weight": float(source["Cash_Weight"].mean()),
                "Metrics_Source": "Calculated from supplied Full Universe daily_nav.csv",
            },
            {
                "Configuration": CONFIG,
                "Strategy": "SPY_Buy_And_Hold",
                **spy_metrics,
                "Transaction_Cost_USD": 0.0,
                "Average_Daily_Cash_Weight": 0.0,
                "Metrics_Source": "Calculated from supplied daily_nav.csv; zero risk-free proxy for Sharpe",
            },
        ])
        LOG.warning(
            "Previous summary file unavailable or incompatible. Performance "
            "metrics for Full Universe and SPY are calculated from daily_nav.csv. "
            "Transaction costs for Full Universe are not derivable from NAV and remain blank."
        )

    args.full_universe_nav = nav_path
    if needed is not None and summary_path and summary_path.exists():
        args.comparison_summary = summary_path
    else:
        args.comparison_summary = None

    LOG.info(
        "Loaded previous Full Universe and SPY NAV: %s | rows=%d | %s to %s",
        nav_path, len(previous_nav), previous_nav["Date"].min().date(),
        previous_nav["Date"].max().date(),
    )
    return previous_nav, needed


def main(args):
    import S06_portfolio_allocation_no_min3 as s06

    if not (
        args.gamma > 0
        and args.direct_starts >= 1
        and args.initial_capital > 0
        and 0.0 <= args.xgb_threshold <= 1.0
        and args.expected_tickers >= 1
        and args.regime_lookback >= 252
        and args.regime_vol_window >= 5
        and args.regime_threshold_window >= 30
        and args.regime_min_days >= 30
        and 0.0 <= args.regime_condition_weight <= 1.0
    ):
        raise ValueError("Invalid experiment parameters.")

    for path, label in [
        (args.universe, "Investable universe"),
        (args.macro, "Macro feature file"),
        (args.peer_signals, "XGBoost selection input"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    args.out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(args.out / "run.log", encoding="utf-8"),
        ],
        force=True,
    )

    # Load old Full Universe and SPY results before starting the XGBoost run.
    previous_nav, previous_summary = load_previous_comparison(args)

    universe = read_universe(args.universe, s06.CASH)
    peer_selections, peer_audit, peer_panel = read_peer_xgboost(
        args.peer_signals,
        set(universe),
        args.xgb_threshold,
        args.expected_tickers,
    )
    dates = list(peer_selections)

    # Validate that the previous benchmark results cover the complete XGBoost
    # backtest calendar. Do not silently shorten either strategy's history.
    previous_dates = set(previous_nav["Date"])
    signal_dates = set(pd.to_datetime(dates).normalize())
    if not signal_dates.issubset(previous_dates):
        missing_dates = sorted(signal_dates - previous_dates)
        raise ValueError(
            "Previous comparison NAV does not cover all XGBoost rebalance dates: "
            + ", ".join(str(d.date()) for d in missing_dates)
        )

    # SPY close is used to identify the market regime, not to rerun a SPY strategy.
    spy = s06.read_market("SPY")
    spy_close = spy["Adj Close"]

    peer_audit.to_csv(args.out / "peer_xgboost_selection_audit.csv", index=False)
    peer_audit.loc[peer_audit["Selected"].eq(1)].to_csv(
        args.out / "peer_xgboost_selected_tickers.csv", index=False
    )

    dest = args.out / CONFIG / STRATEGY
    settings = s06_settings(args, dest)
    diagnostics = []
    estimator = make_estimator(s06, args, spy_close, diagnostics)

    LOG.info(
        "START | %s | %s | ONLY ALLOCATION RUN | threshold >= %.2f",
        CONFIG, STRATEGY, args.xgb_threshold,
    )

    with (
        patch.object(s06, "read_selections", return_value=peer_selections),
        patch.object(s06, "estimation", estimator),
    ):
        result = s06.run_single(settings, mode="direct")

    nav_path = dest / "daily_nav.csv"
    if not nav_path.exists():
        raise FileNotFoundError(f"XGBoost daily NAV was not created: {nav_path}")

    xgb_nav = pd.read_csv(nav_path, parse_dates=["Date"])
    required_xgb = {"Date", "Portfolio_NAV", "Cash_Weight", "SPY_NAV"}
    missing_xgb = required_xgb.difference(xgb_nav.columns)
    if missing_xgb:
        raise ValueError(
            f"{nav_path} is missing columns: " + ", ".join(sorted(missing_xgb))
        )
    xgb_nav["Date"] = pd.to_datetime(xgb_nav["Date"], errors="raise").dt.normalize()
    xgb_nav = xgb_nav.sort_values("Date").reset_index(drop=True)

    if xgb_nav["Date"].duplicated().any():
        raise ValueError("New XGBoost NAV contains duplicate dates.")
    if not xgb_nav["Date"].equals(previous_nav["Date"]):
        xgb_dates = set(xgb_nav["Date"])
        old_dates = set(previous_nav["Date"])
        missing_from_xgb = sorted(old_dates - xgb_dates)
        missing_from_old = sorted(xgb_dates - old_dates)
        raise ValueError(
            "New XGBoost NAV calendar does not exactly match the stored "
            "Full Universe/SPY calendar. "
            f"Missing from XGBoost (first 10): {[str(x.date()) for x in missing_from_xgb[:10]]}; "
            f"missing from previous comparison (first 10): "
            f"{[str(x.date()) for x in missing_from_old[:10]]}."
        )

    # Ensure the internally generated benchmark matches the stored benchmark,
    # then use the stored SPY series for the chart and combined comparison.
    stored_spy = previous_nav[SPY_NAV_COLUMN].to_numpy(float)
    current_spy = xgb_nav["SPY_NAV"].to_numpy(float)
    if not np.allclose(current_spy, stored_spy, rtol=0, atol=1e-5):
        max_diff = float(np.max(np.abs(current_spy - stored_spy)))
        raise ValueError(
            "The SPY benchmark generated during the XGBoost run does not match "
            "the previously stored SPY series. Refusing to combine different "
            f"benchmark paths. Maximum absolute NAV difference: {max_diff:.8f}"
        )

    xgb_metric = result["results"]
    xgb_summary_row = {
        "Configuration": CONFIG,
        "Strategy": "Peer_XGBoost",
        "Total_Return": xgb_metric["portfolio_total_return"],
        "CAGR": xgb_metric["portfolio_CAGR"],
        "Annualized_Sharpe": xgb_metric["portfolio_annualized_sharpe"],
        "Max_Drawdown": xgb_metric["portfolio_max_drawdown"],
        "Transaction_Cost_USD": xgb_metric["total_transaction_cost_USD"],
        "Average_Daily_Cash_Weight": float(xgb_nav["Cash_Weight"].mean()),
    }

    # Preserve the previous Full Universe and SPY metrics without recalculating.
    full_universe_summary = previous_summary.loc[
        previous_summary["Strategy"].eq("Full_Universe")
    ].iloc[0].to_dict()
    spy_summary = previous_summary.loc[
        previous_summary["Strategy"].eq("SPY_Buy_And_Hold")
    ].iloc[0].to_dict()

    summary = pd.DataFrame([xgb_summary_row, full_universe_summary, spy_summary])

    # Join the newly computed XGBoost path to the stored Full Universe and SPY paths.
    comparison = pd.DataFrame({
        "Date": xgb_nav["Date"],
        f"{CONFIG}_Peer_XGBoost_NAV": xgb_nav["Portfolio_NAV"].to_numpy(float),
        f"{CONFIG}_Peer_XGBoost_Cash_Weight": xgb_nav["Cash_Weight"].to_numpy(float),
        FULL_UNIVERSE_NAV_COLUMN: previous_nav[FULL_UNIVERSE_NAV_COLUMN].to_numpy(float),
        FULL_UNIVERSE_CASH_COLUMN: previous_nav[FULL_UNIVERSE_CASH_COLUMN].to_numpy(float),
        SPY_NAV_COLUMN: stored_spy,
    })

    # Chart has exactly three lines: new XGBoost, stored Full Universe, stored SPY.
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.plot(
        comparison["Date"],
        comparison[f"{CONFIG}_Peer_XGBoost_NAV"] / args.initial_capital * 100.0,
        label="XGBoost (probability >= 0.60)",
        color=COLORS["Peer_XGBoost"],
        linewidth=1.9,
    )
    ax.plot(
        comparison["Date"],
        comparison[FULL_UNIVERSE_NAV_COLUMN] / args.initial_capital * 100.0,
        label="Full Universe (previous results)",
        color=COLORS["Full_Universe"],
        linewidth=1.9,
    )
    ax.plot(
        comparison["Date"],
        comparison[SPY_NAV_COLUMN] / args.initial_capital * 100.0,
        label="SPY buy-and-hold (previous benchmark)",
        color=COLORS["SPY"],
        linestyle="--",
        linewidth=2.0,
    )
    ax.set(
        title="Regime-Aware Ledoit-Wolf | XGBoost vs Full Universe vs SPY",
        xlabel="Date",
        ylabel="Portfolio NAV (initial = 100)",
    )
    ax.legend()
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    chart_path = args.out / "regimeaware_ledoitwolf_xgboost_full_universe_spy.png"
    fig.savefig(chart_path, dpi=180)
    plt.close(fig)

    summary.to_csv(args.out / "allocation_methods_performance_summary.csv", index=False)
    comparison.to_csv(args.out / "comparison_daily_nav.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(
        args.out / "allocation_method_diagnostics.csv", index=False
    )

    # Aggregate outputs from the one XGBoost run only.
    output_map = [
        ("portfolio_weights.csv", "all_portfolio_weights.csv"),
        ("rebalance_summary.csv", "all_rebalance_summaries.csv"),
        ("transaction_cost_breakdown.csv", "all_transaction_costs.csv"),
    ]
    for source_name, output_name in output_map:
        source_file = dest / source_name
        if not source_file.exists():
            raise FileNotFoundError(
                f"Expected XGBoost output was not created: {source_file}"
            )
        frame = pd.read_csv(source_file)
        if not frame.empty:
            frame.insert(0, "Configuration", CONFIG)
            frame.insert(1, "Strategy", STRATEGY)
        frame.to_csv(args.out / output_name, index=False)

    manifest = {
        "script": Path(__file__).name,
        "allocation_module": "S06_portfolio_allocation_no_min3.py",
        "methods": [CONFIG],
        "strategies_run": ["Peer_XGBoost"],
        "strategies_reused": ["Full_Universe", "SPY_Buy_And_Hold"],
        "full_universe_source": str(args.full_universe_nav),
        "performance_summary_source": str(args.comparison_summary) if args.comparison_summary else "Metrics calculated from supplied daily_nav.csv",
        "architecture": "Direct asset-weight optimization (no Sharpe stage)",
        "benchmark": "Previously stored SPY buy-and-hold NAV; not rerun as a strategy",
        "gamma": args.gamma,
        "lookback_momentum": s06.EXPECTED_RETURN_LOOKBACK,
        "covariance_lookback": s06.COVARIANCE_LOOKBACK,
        "horizon": s06.HORIZON,
        "mean_estimation": "Raw 120-session momentum converted to 60-session expected return",
        "covariance_estimation": (
            "Regime-conditioned Ledoit-Wolf covariance blended with unconditional "
            "252-session Ledoit-Wolf covariance"
        ),
        "regime_definition": (
            "21-session SPY realized volatility > trailing 252-session median "
            "of 21-session SPY realized volatility, using past-only threshold"
        ),
        "regime_lookback": args.regime_lookback,
        "regime_vol_window": args.regime_vol_window,
        "regime_threshold_window": args.regime_threshold_window,
        "regime_min_days": args.regime_min_days,
        "regime_condition_weight": args.regime_condition_weight,
        "xgboost_input": str(args.peer_signals),
        "xgboost_selection_rule": f"Probability >= {args.xgb_threshold:.2f}",
        "xgboost_ticker_date_rows": int(len(peer_panel)),
        "xgboost_rebalance_dates": len(dates),
        "xgboost_selected_observations": int(peer_panel["Selected"].sum()),
        "minimum_eligible_asset_restriction": "Removed; 1- and 2-asset candidate sets may be optimized",
        "rebalance_dates": [str(pd.Timestamp(x).date()) for x in dates],
        "lookahead": "Point-in-time historical returns and signal-date market state only",
        "runs_performed": ["Regime-Aware Ledoit-Wolf / Peer XGBoost"],
        "reused_without_rerun": [
            "Regime-Aware Ledoit-Wolf / Full Universe NAV and cash weights",
            "SPY buy-and-hold NAV",
            "Full Universe and SPY performance metrics",
        ],
        "not_run": [
            "Full Universe allocation",
            "Random Forest",
            "Mean-Variance Ledoit-Wolf",
            "Black-Litterman Ledoit-Wolf",
            "Robust Ledoit-Wolf",
        ],
    }
    (args.out / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    LOG.info("COMPLETED: XGBoost-only rerun. Full Universe and SPY were reused.")
    LOG.info("Output directory: %s", args.out)
    LOG.info("Three-line comparison chart: %s", chart_path)
    LOG.info("\n%s", summary.to_string(index=False))


if __name__ == "__main__":
    main(arguments())
