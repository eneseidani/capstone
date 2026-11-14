"""
S06: Regime-Aware Ledoit-Wolf comparison — XGBoost vs Full Universe vs SPY.

Run from the Capstone project root:
    python S06_regime_aware_xgboost_full_universe.py

This script runs exactly ONE portfolio allocation methodology:
    04_RegimeAware_LedoitWolf

It runs exactly TWO actively managed strategies:
    1. Peer_XGBoost — uses the Selected flag in the 912-row XGBoost
       ticker/date input file (probability >= 0.60).
    2. Full_Universe — point-in-time eligible investable universe.

SPY buy-and-hold is shown as a benchmark using the existing S06 benchmark
calculation. No Random Forest, Black-Litterman, Robust, or standalone
Mean-Variance allocation runs are performed.

The XGBoost input must contain all candidate tickers for all 19 rebalance
dates, including dates with zero selected assets. Use
portfolio_signals_all_candidates.csv, not the selected-only 520-row file.

The existing S06_portfolio_allocation.py remains responsible for portfolio
construction, holdings continuity, execution timing, fees, and SPY benchmark
calculation. This script only supplies the XGBoost selection map and the
regime-aware Ledoit-Wolf covariance estimator.
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


LOG = logging.getLogger("S06_regime_aware_xgboost_full_universe")

CONFIG = "04_RegimeAware_LedoitWolf"
STRATEGIES = ("Peer_XGBoost", "Full_Universe")
LABELS = {
    "Peer_XGBoost": "XGBoost (probability >= 0.60)",
    "Full_Universe": "Full investable universe",
}
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
        help=(
            "912-row ticker/date CSV containing Date, Ticker, Probability, "
            "Rank and Selected. Use the all-candidates file so zero-selection "
            "rebalance dates remain in the calendar."
        ),
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
        "--out",
        type=Path,
        default=Path("model_output/S06_regime_aware_xgboost_full_universe"),
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
        help=(
            "Blend weight on regime-conditional Ledoit-Wolf covariance; "
            "remaining weight is unconditional 252-session Ledoit-Wolf covariance."
        ),
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

def main(args):
    import S06_portfolio_allocation as s06

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

    if not args.universe.exists():
        raise FileNotFoundError(f"Investable universe file not found: {args.universe}")
    if not args.macro.exists():
        raise FileNotFoundError(f"Macro feature file not found: {args.macro}")

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

    universe = read_universe(args.universe, s06.CASH)

    # Calendar and XGBoost selections both come from the complete 912-row
    # ticker/date input. This retains rebalance dates with zero selections.
    peer_selections, peer_audit, peer_panel = read_peer_xgboost(
        args.peer_signals,
        set(universe),
        args.xgb_threshold,
        args.expected_tickers,
    )
    dates = list(peer_selections)

    full_selections, universe_audit = full_universe_pit(s06, universe, dates)

    choices = {
        "Peer_XGBoost": peer_selections,
        "Full_Universe": full_selections,
    }

    for strategy, by_date in choices.items():
        if list(by_date) != dates:
            raise ValueError(f"{strategy}: rebalance calendars differ.")
        for dt, selected in by_date.items():
            if len(selected) != len(set(selected)):
                raise ValueError(f"{strategy}/{dt}: duplicate selected tickers.")
            if set(selected) - set(universe):
                raise ValueError(f"{strategy}/{dt}: selected ticker outside universe.")

    spy = s06.read_market("SPY")
    spy_close = spy["Adj Close"]

    # Export the exact ticker-level XGBoost input used by the allocator.
    peer_audit.to_csv(args.out / "peer_xgboost_selection_audit.csv", index=False)
    peer_audit.loc[peer_audit["Selected"].eq(1)].to_csv(
        args.out / "peer_xgboost_selected_tickers.csv", index=False
    )
    universe_audit.to_csv(
        args.out / "full_universe_eligibility_audit.csv", index=False
    )

    navs = {}
    summary_rows = []
    weight_frames = []
    rebalance_frames = []
    tx_frames = []
    diagnostics = []
    spy_metrics = None

    # Exactly two allocation runs: XGBoost and Full Universe.
    for strategy in STRATEGIES:
        dest = args.out / CONFIG / strategy
        settings = s06_settings(args, dest)
        estimator = make_estimator(s06, args, spy_close, diagnostics)

        LOG.info("START | %s | %s", CONFIG, strategy)
        with (
            patch.object(s06, "read_selections", return_value=choices[strategy]),
            patch.object(s06, "estimation", estimator),
        ):
            result = s06.run_single(settings, mode="direct")

        nav_path = dest / "daily_nav.csv"
        if not nav_path.exists():
            raise FileNotFoundError(
                f"Expected S06 NAV output was not created: {nav_path}"
            )
        nav = pd.read_csv(nav_path, parse_dates=["Date"])
        required_nav_columns = {"Date", "Portfolio_NAV", "Cash_Weight", "SPY_NAV"}
        if not required_nav_columns.issubset(nav.columns):
            raise ValueError(
                f"{nav_path} is missing columns: "
                f"{sorted(required_nav_columns - set(nav.columns))}"
            )
        navs[strategy] = nav

        metric = result["results"]
        if spy_metrics is None:
            spy_metrics = metric

        summary_rows.append({
            "Configuration": CONFIG,
            "Strategy": strategy,
            "Total_Return": metric["portfolio_total_return"],
            "CAGR": metric["portfolio_CAGR"],
            "Annualized_Sharpe": metric["portfolio_annualized_sharpe"],
            "Max_Drawdown": metric["portfolio_max_drawdown"],
            "Transaction_Cost_USD": metric["total_transaction_cost_USD"],
            "Average_Daily_Cash_Weight": float(nav["Cash_Weight"].mean()),
        })

        for sink, filename in [
            (weight_frames, "portfolio_weights.csv"),
            (rebalance_frames, "rebalance_summary.csv"),
            (tx_frames, "transaction_cost_breakdown.csv"),
        ]:
            file_path = dest / filename
            if not file_path.exists():
                raise FileNotFoundError(f"Expected S06 output was not created: {file_path}")
            frame = pd.read_csv(file_path)
            if not frame.empty:
                frame.insert(0, "Configuration", CONFIG)
                frame.insert(1, "Strategy", strategy)
                sink.append(frame)

        LOG.info(
            "END | %s | %s | return=%+.2f%% | Sharpe=%.3f",
            CONFIG,
            strategy,
            100.0 * metric["portfolio_total_return"],
            metric["portfolio_annualized_sharpe"],
        )

    # Check that both portfolio runs use the same daily calendar and identical
    # SPY benchmark series. SPY is plotted once, not rerun as a third optimizer.
    anchor = navs["Peer_XGBoost"]
    full_nav = navs["Full_Universe"]
    if not anchor["Date"].equals(full_nav["Date"]):
        raise AssertionError("NAV date mismatch between XGBoost and Full Universe.")
    if not np.allclose(
        anchor["SPY_NAV"].to_numpy(float),
        full_nav["SPY_NAV"].to_numpy(float),
        rtol=0,
        atol=1e-5,
    ):
        raise AssertionError("SPY benchmark differs between the two portfolio runs.")

    joined = pd.DataFrame({"Date": anchor["Date"]})
    joined[f"{CONFIG}_Peer_XGBoost_NAV"] = anchor["Portfolio_NAV"].to_numpy(float)
    joined[f"{CONFIG}_Peer_XGBoost_Cash_Weight"] = anchor["Cash_Weight"].to_numpy(float)
    joined[f"{CONFIG}_Full_Universe_NAV"] = full_nav["Portfolio_NAV"].to_numpy(float)
    joined[f"{CONFIG}_Full_Universe_Cash_Weight"] = full_nav["Cash_Weight"].to_numpy(float)
    joined[f"{CONFIG}_SPY_NAV"] = anchor["SPY_NAV"].to_numpy(float)

    # Single chart: XGBoost vs Full Universe vs SPY.
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.plot(
        anchor["Date"],
        anchor["Portfolio_NAV"] / args.initial_capital * 100.0,
        label=LABELS["Peer_XGBoost"],
        color=COLORS["Peer_XGBoost"],
        linewidth=1.9,
    )
    ax.plot(
        full_nav["Date"],
        full_nav["Portfolio_NAV"] / args.initial_capital * 100.0,
        label=LABELS["Full_Universe"],
        color=COLORS["Full_Universe"],
        linewidth=1.9,
    )
    ax.plot(
        anchor["Date"],
        anchor["SPY_NAV"] / args.initial_capital * 100.0,
        label="SPY buy-and-hold",
        color=COLORS["SPY"],
        linestyle="--",
        linewidth=2.0,
    )
    ax.set(
        title="Regime-Aware Ledoit-Wolf | XGBoost vs Full Universe vs SPY",
        xlabel="Date",
        ylabel="Portfolio NAV (initial = 100)",
    )
    ax.legend(ncol=1)
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    chart_path = args.out / "regimeaware_ledoitwolf_xgboost_full_universe_spy.png"
    fig.savefig(chart_path, dpi=180)
    plt.close(fig)

    years = (anchor["Date"].iloc[-1] - anchor["Date"].iloc[0]).days / 365.25
    spy_return = float(
        anchor["SPY_NAV"].iloc[-1] / anchor["SPY_NAV"].iloc[0] - 1.0
    )
    summary_rows.append({
        "Configuration": CONFIG,
        "Strategy": "SPY_Buy_And_Hold",
        "Total_Return": spy_return,
        "CAGR": float((1.0 + spy_return) ** (1.0 / years) - 1.0),
        "Annualized_Sharpe": spy_metrics["spy_annualized_sharpe"],
        "Max_Drawdown": spy_metrics["spy_max_drawdown"],
        "Transaction_Cost_USD": 0.0,
        "Average_Daily_Cash_Weight": 0.0,
    })

    pd.DataFrame(summary_rows).to_csv(
        args.out / "allocation_methods_performance_summary.csv", index=False
    )
    joined.to_csv(args.out / "comparison_daily_nav.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(
        args.out / "allocation_method_diagnostics.csv", index=False
    )

    for filename, frames in [
        ("all_portfolio_weights.csv", weight_frames),
        ("all_rebalance_summaries.csv", rebalance_frames),
        ("all_transaction_costs.csv", tx_frames),
    ]:
        if frames:
            pd.concat(frames, ignore_index=True).to_csv(
                args.out / filename, index=False
            )

    manifest = {
        "script": Path(__file__).name,
        "methods": [CONFIG],
        "architecture": "Direct asset-weight optimization (no Sharpe stage)",
        "strategies": list(STRATEGIES),
        "benchmark": "SPY buy-and-hold, using S06 benchmark calculation; no SPY fees",
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
        "xgboost_selected_flag": "Validated against Probability threshold in input CSV",
        "xgboost_ticker_date_rows": int(len(peer_panel)),
        "xgboost_rebalance_dates": len(dates),
        "xgboost_selected_observations": int(peer_panel["Selected"].sum()),
        "full_universe_selection": "Point-in-time eligibility using S06 OHLCV, execution-open and fee-context checks",
        "transaction_costs": "S06 transaction_fee_helper native spread, impact and commissions",
        "rebalance_dates": [str(pd.Timestamp(x).date()) for x in dates],
        "lookahead": "Point-in-time historical returns and signal-date market state only",
        "runs_performed": [
            "Regime-Aware Ledoit-Wolf / Peer XGBoost",
            "Regime-Aware Ledoit-Wolf / Full Universe",
        ],
        "not_run": [
            "Random Forest",
            "Mean-Variance Ledoit-Wolf",
            "Black-Litterman Ledoit-Wolf",
            "Robust Ledoit-Wolf",
        ],
    }
    (args.out / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    LOG.info("COMPLETED: %s", args.out)
    LOG.info("Chart: %s", chart_path)
    LOG.info("\n%s", pd.DataFrame(summary_rows).to_string(index=False))


if __name__ == "__main__":
    main(arguments())
