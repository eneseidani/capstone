"""Controlled S05 selection-value experiment: RF vs peer LR vs peer XGBoost vs no filter.

Run from the Capstone project root:
    python S06_four_universe_comparison.py

Requires: existing S06_portfolio_allocation.py, config.py, transaction_fee_helper.py,
Investable_Universe.csv, portfolio_signals.csv, portfolio_rebalance_schedule.csv,
the original S05 test predictions and S04 price/fee data. The XGBoost
hyperparameter CSV is optional metadata (peer models are NOT retrained).

Uses EXACT SAME S06 direct joint mean-variance utility allocator for all four cases.
No Sharpe-ratio screening or re-training is performed. SPY is an external fifth
benchmark; it is also allowed as a risky candidate (20% cap) in the full universe.
The full universe is point-in-time screened for historical data and S04 fees.
Peer signals are taken as originally exported, with dates that selected ZERO assets
preserved from the peer schedule. The peer's same-day execution dates are NOT used;
S06's next-session-open execution convention is applied equally to all.

Caution: reusing existing held assets that are no longer selected is an unchanged
S06 rule; results measure the incremental value of the candidate filter under
S06's holdings-aware policy, not a pure hard-exclusion treatment.
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

LOG = logging.getLogger("S06_four_universe")
PEER_MODELS = {"Peer_Logistic": ("Logistic Regression", 0.50),
               "Peer_XGBoost": ("XGBoost", 0.60)}


def arguments():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--predictions", type=Path, default=Path("model_output/S05_peer_calendar_classification/predictions_test.csv"))
    p.add_argument("--peer-signals", type=Path, default=Path("portfolio_signals.csv"))
    p.add_argument("--peer-schedule", type=Path, default=Path("portfolio_rebalance_schedule.csv"))
    p.add_argument("--peer-hyperparameters", type=Path, default=Path("best_hyperparameters (1).csv"))
    p.add_argument("--universe", type=Path, default=Path("Investable_Universe.csv"))
    p.add_argument("--macro", type=Path, default=Path("features_data/macro_features.csv"))
    p.add_argument("--out", type=Path, default=Path("model_output/S06_four_universe_comparison"))
    p.add_argument("--gamma", type=float, default=3.0)
    p.add_argument("--direct-starts", type=int, default=12)
    p.add_argument("--initial-capital", type=float, default=100000.0)
    p.add_argument("--rf-threshold", type=float, default=0.50)
    p.add_argument("--top-k", type=int, default=15)
    return p.parse_args()


def read_universe(path: Path, cash: str):
    d = pd.read_csv(path)
    if "Ticker" not in d:
        raise ValueError(f"Missing Ticker column in {path}")
    raw = d.Ticker.astype(str).str.strip().tolist()
    if any(not t or t.lower() == "nan" for t in raw) or len(raw) != len(set(raw)):
        raise ValueError("Invalid or duplicated tickers in investable universe")
    if "SPY" not in raw or cash not in raw:
        raise ValueError("Investable universe must include SPY and DGS3MO")
    return [t for t in raw if t != cash]


def read_peer_exports(signals_path, schedule_path, calendar, allowed, top_k):
    signals = pd.read_csv(signals_path, parse_dates=["Date"])
    schedule = pd.read_csv(schedule_path, parse_dates=["Date"])
    signals_cols = {"Date", "Model", "Ticker", "Probability", "Rank_Within_Date", "Threshold", "Specification"}
    sched_cols = {"Date", "Model", "Selected_Assets", "Threshold", "Top_K", "Specification"}
    if not signals_cols.issubset(signals):
        raise ValueError(f"Peer signals missing columns: {signals_cols - set(signals)}")
    if not sched_cols.issubset(schedule):
        raise ValueError(f"Peer schedule missing columns: {sched_cols - set(schedule)}")
    if signals.duplicated(["Date", "Model", "Ticker"]).any() or schedule.duplicated(["Date", "Model"]).any():
        raise ValueError("Duplicate peer selection or schedule rows")
    calendar = [pd.Timestamp(d) for d in calendar]
    if set(calendar) != set(schedule.Date.unique()):
        raise ValueError("Peer rebalance calendar is not identical to S05")
    result, audit = {}, []
    for name, (peer_model, threshold) in PEER_MODELS.items():
        sub_s = signals.loc[signals.Model.eq(peer_model)].copy()
        sub_c = schedule.loc[schedule.Model.eq(peer_model)].copy()
        if set(sub_c.Date) != set(calendar):
            raise ValueError(f"{name}: missing/extra dates in peer schedule")
        if set(sub_s.Date) - set(calendar):
            raise ValueError(f"{name}: out-of-calendar signals")
        if (sub_c.Threshold.astype(float).sub(threshold).abs() > 1e-9).any():
            raise ValueError(f"{name}: unexpected selection threshold in schedule")
        if (sub_c.Top_K.astype(int) != top_k).any():
            raise ValueError(f"{name}: peer Top_K differs from the chosen comparison")
        if not sub_c.Specification.eq("With Regimes").all() or not sub_s.Specification.eq("With Regimes").all():
            raise ValueError(f"{name}: peer specification differs from With Regimes")
        if not sub_s.Ticker.astype(str).isin(allowed).all():
            raise ValueError(f"{name}: signals contain tickers outside the declared universe")
        if not sub_s.empty:
            sub_s["Probability"] = pd.to_numeric(sub_s.Probability, errors="raise")
            sub_s["Rank_Within_Date"] = pd.to_numeric(sub_s.Rank_Within_Date, errors="raise")
            if not np.isfinite(sub_s.Probability.to_numpy(float)).all():
                raise ValueError(f"{name}: nonfinite probabilities")
            if (sub_s.Probability < threshold - 1e-9).any() or (sub_s.Probability > 1).any():
                raise ValueError(f"{name}: invalid probability/threshold")
            if (sub_s.Threshold.astype(float).sub(threshold).abs() > 1e-9).any():
                raise ValueError(f"{name}: signals threshold inconsistent")
        chosen = {}
        for date in calendar:
            data = sub_s.loc[sub_s.Date.eq(date)].sort_values("Rank_Within_Date")
            planned = sub_c.loc[sub_c.Date.eq(date)].iloc[0]
            count = int(planned.Selected_Assets)
            if len(data) != count:
                raise ValueError(f"{name} {date.date()}: {len(data)} exported signals but schedule says {count}")
            if len(data) > top_k:
                raise ValueError(f"{name} {date.date()}: over Top_K cap")
            if data.Rank_Within_Date.tolist() != list(range(1, len(data) + 1)):
                raise ValueError(f"{name} {date.date()}: incorrect within-date ranks")
            if any(data.Probability.to_numpy()[:-1] < data.Probability.to_numpy()[1:] - 1e-10):
                raise ValueError(f"{name} {date.date()}: probability rank not descending")
            selected = data.Ticker.astype(str).tolist()
            chosen[date] = selected
            audit.append({"Strategy": name, "Signal_Date": date, "Selected_Count":len(selected),
                          "Tickers": "|".join(selected), "Cash_Only_At_Screening":len(selected) == 0,
                          "Peer_Primary_Backtest": bool(planned.get("Primary_Backtest", True))})
        result[name] = chosen
    return result, pd.DataFrame(audit)


def full_universe_pit(s06, tickers, dates):
    """Exclude only assets lacking historical estimation/fees; execution must be available.

    The execution-open check is a data-completeness check at execution, not a
    return-based selection rule. It must not be used to filter on future gains.
    """
    available_market = {}
    for tk in tickers:
        try:
            available_market[tk] = s06.read_market(tk)
        except (ValueError, FileNotFoundError) as exc:
            LOG.warning("Market file unusable for %s: %s", tk, exc)
            available_market[tk] = None
    spy = s06.read_market("SPY")
    sessions = spy.index
    eligible_by_date, checks = {}, []
    for dt in dates:
        dt = pd.Timestamp(dt)
        if dt not in sessions:
            raise ValueError(f"Signal date not on SPY market calendar: {dt.date()}")
        index = sessions.get_loc(dt)
        if index + 1 >= len(sessions):
            raise ValueError(f"No execution session after {dt.date()}")
        exec_dt = sessions[index + 1]
        eligible = []
        for tk in tickers:
            d = available_market[tk]
            reason = "eligible"
            if d is None:
                reason = "missing_ohlcv"
            else:
                historical = d["Adj Close"].reindex(sessions[:index + 1]).ffill(limit=2).tail(253)
                if len(historical) != 253 or not np.isfinite(historical.to_numpy(float)).all() or (historical <= 0).any():
                    reason = "insufficient_historical_253_closes"
                elif not np.isfinite(float(d["Open"].get(exec_dt, np.nan))) or float(d["Open"].get(exec_dt, np.nan)) <= 0:
                    reason = "missing_execution_open"
                else:
                    try:
                        # Exactly S06's point-in-time fee helper, same strictness.
                        s06.load_fee_context(dt, tickers=[tk, s06.CASH])
                    except (ValueError, KeyError, ArithmeticError, FileNotFoundError) as exc:
                        reason = "missing_point_in_time_fee:" + type(exc).__name__
            if reason == "eligible":
                eligible.append(tk)
            checks.append({"Signal_Date":dt,"Ticker":tk,"Eligible":reason == "eligible","Reason":reason})
        eligible_by_date[dt] = eligible
        LOG.info("%s | full-universe eligible %d / %d", dt.date(), len(eligible), len(tickers))
    return eligible_by_date, pd.DataFrame(checks)


def validate_lists(s06, choices, allowed, reference_dates):
    for name, by_date in choices.items():
        if list(by_date) != list(reference_dates):
            raise ValueError(f"{name}: selection calendar/order differs")
        for dt, tickers in by_date.items():
            extra = set(tickers) - set(allowed)
            if extra:
                raise ValueError(f"{name} {dt.date()}: unknown tickers {sorted(extra)}")
            if s06.CASH in tickers or len(tickers) != len(set(tickers)):
                raise ValueError(f"{name} {dt.date()}: invalid or duplicate tickers")


def s06_namespace(args, out):
    return argparse.Namespace(predictions=args.predictions, out=out, threshold=args.rf_threshold,
                              model="RandomForest", top_k=args.top_k,
                              initial_capital=args.initial_capital, risk_aversion=args.gamma,
                              risk_aversion_grid="none", skip_direct_utility=True,
                              direct_starts=args.direct_starts, macro=args.macro,
                              allow_feasible_fallback=False, initial_date=None)


def run_with_candidates(s06, args, candidates, directory):
    """Use the actual S06 direct utility and actual S06 NAV/fee execution engine."""
    settings = s06_namespace(args, directory)
    with patch.object(s06, "read_selections", return_value=candidates):
        result = s06.run_single(settings, mode="direct")
    return result


def main(args):
    import S06_portfolio_allocation as s06
    if not np.isfinite(args.gamma) or args.gamma <= 0:
        raise ValueError("--gamma must be finite and positive")
    if args.direct_starts < 1 or args.top_k < 1 or args.initial_capital <= 0:
        raise ValueError("Invalid optimizer starts, Top_K, or initial capital")
    args.out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(args.out / "run.log", encoding="utf-8")],
                        force=True)
    universe = read_universe(args.universe, s06.CASH)
    rf = s06.read_selections(args.predictions, "RandomForest", args.rf_threshold, args.top_k)
    dates = list(rf)
    peer, peer_audit = read_peer_exports(args.peer_signals, args.peer_schedule, dates, set(universe), args.top_k)
    full, universe_audit = full_universe_pit(s06, universe, dates)
    choices = {"Our_RandomForest":rf, "Peer_Logistic":peer["Peer_Logistic"],
               "Peer_XGBoost":peer["Peer_XGBoost"], "Full_Universe":full}
    validate_lists(s06, choices, universe, dates)
    peer_audit.to_csv(args.out / "peer_selection_audit.csv", index=False)
    universe_audit.to_csv(args.out / "full_universe_eligibility_audit.csv", index=False)
    if args.peer_hyperparameters.exists():
        hp = pd.read_csv(args.peer_hyperparameters)
        if len(hp) != 1 or str(hp.iloc[0].get("Algorithm")) != "XGBoost":
            raise ValueError("Peer XGBoost hyperparameters metadata invalid")
        hyperparameters = hp.iloc[0].to_dict()
    else:
        LOG.warning("XGBoost hyperparameter metadata file not found; peer selections will still be evaluated without retraining")
        hyperparameters = None

    out_nav, stats, results_by_name = {}, [], {}
    for name, candidates in choices.items():
        LOG.info("START %s | direct utility gamma=%s starts=%s", name, args.gamma, args.direct_starts)
        folder = args.out / name
        result = run_with_candidates(s06, args, candidates, folder)
        daily = pd.read_csv(folder / "daily_nav.csv", parse_dates=["Date"])
        out_nav[name] = daily
        rebal = pd.read_csv(folder / "rebalance_summary.csv")
        results = result["results"]
        results_by_name[name] = results
        stats.append({"Strategy":name,
                      "Total_Return":results["portfolio_total_return"],
                      "CAGR":results["portfolio_CAGR"],
                      "Annualized_Sharpe":results["portfolio_annualized_sharpe"],
                      "Max_Drawdown":results["portfolio_max_drawdown"],
                      "Transaction_Cost_USD":results["total_transaction_cost_USD"],
                      "Average_Daily_Cash_Weight":float(daily.Cash_Weight.mean()),
                      "Average_Eligible_Assets":float(rebal.Eligible_Including_Holdings.mean()),
                      "Average_Risky_Positions":float(rebal.Actual_Risky_Positions.mean()),
                      "Rebalances":int(len(rebal))})
        LOG.info("DONE %s | net return %+.2f%% | costs $%.2f", name,
                 100 * results["portfolio_total_return"], results["total_transaction_cost_USD"])

    baseline = out_nav["Our_RandomForest"]
    merged = pd.DataFrame({"Date":baseline.Date})
    for name, daily in out_nav.items():
        if not baseline.Date.equals(daily.Date):
            raise ValueError(f"NAV dates differ for {name}")
        if not np.allclose(baseline.SPY_NAV, daily.SPY_NAV, rtol=0, atol=1e-5):
            raise ValueError(f"SPY benchmark differs for {name}")
        merged[name + "_NAV"] = daily.Portfolio_NAV.to_numpy(float)
        merged[name + "_Cash_Weight"] = daily.Cash_Weight.to_numpy(float)
    merged["SPY_NAV"] = baseline.SPY_NAV.to_numpy(float)
    merged.to_csv(args.out / "comparison_daily_nav.csv", index=False)
    benchmark_results = results_by_name["Our_RandomForest"]
    stats.append({"Strategy":"SPY_Buy_And_Hold",
                  "Total_Return":benchmark_results["spy_total_return"],
                  "CAGR":benchmark_results["spy_CAGR"],
                  "Annualized_Sharpe":benchmark_results["spy_annualized_sharpe"],
                  "Max_Drawdown":benchmark_results["spy_max_drawdown"],
                  "Transaction_Cost_USD":0.0,
                  "Average_Daily_Cash_Weight":0.0,
                  "Average_Eligible_Assets":np.nan,"Average_Risky_Positions":1.0,
                  "Rebalances":0})
    summary = pd.DataFrame(stats)
    summary.to_csv(args.out / "comparison_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(13, 7))
    colors = {"Our_RandomForest":"#1F77B4", "Peer_Logistic":"#9467BD",
              "Peer_XGBoost":"#D62728", "Full_Universe":"#2CA02C", "SPY":"#FF7F0E"}
    for name in choices:
        ax.plot(merged.Date, merged[name + "_NAV"] / args.initial_capital * 100,
                label=name.replace("_", " "), lw=1.8, color=colors[name])
    ax.plot(merged.Date, merged.SPY_NAV / args.initial_capital * 100,
            label="SPY buy-and-hold", lw=1.9, ls="--", color=colors["SPY"])
    ax.set(title=f"Classification filter ablation | S06 direct utility γ={args.gamma:g} | net of costs",
           xlabel="Date", ylabel="Portfolio value (initial = 100)")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(args.out / "four_filters_vs_SPY.png", dpi=180)
    plt.close(fig)

    manifesto = {
        "purpose":"Measure contribution of classifier candidate sets with constant S06 direct utility allocation",
        "strategies":list(choices) + ["SPY_Buy_And_Hold"],
        "allocator":"S06 optimize_direct_utility; Stage 1 Sharpe sleeve NOT used",
        "risk_aversion_gamma":args.gamma,"direct_starts":args.direct_starts,
        "maximum_per_asset_weight":s06.MAX_WEIGHT,"maximum_normal_cash_weight":s06.MAX_CASH_WEIGHT,
        "expected_return":"120-day compounded momentum converted to 60 days",
        "covariance":"252 daily returns scaled to 60-day covariance",
        "execution":"signal close, following SPY session adjusted opening",
        "transaction_costs":"S06 existing spread, market impact, commission; net NAV",
        "peer_signals_file":str(args.peer_signals),
        "peer_schedule_file":str(args.peer_schedule),
        "peer_xgboost_hyperparameters_file":str(args.peer_hyperparameters),
        "peer_xgboost_hyperparameters_metadata_only":hyperparameters,
        "peer_retrained":False,
        "peer_zero_selected_dates_preserved":True,
        "uses_all_19_rebalance_dates_including_nonprimary_peer_last_date":True,
        "nonselected_existing_holdings_remain_eligible":"S06 original behavior",
        "rf_predictions_file":str(args.predictions),
        "full_universe_file":str(args.universe),
        "full_universe_candidate_count":len(universe),
        "notes":["SPY remains a risky candidate with maximum 20% in full universe and an independent buy-hold benchmark",
                 "Peer original same-day execution assumption is replaced with S06 next-day opening execution",
                 "Test sample has been examined repeatedly; report comparisons as exploratory, not fresh independent out-of-sample discovery"],
    }
    (args.out / "comparison_manifest.json").write_text(json.dumps(manifesto, indent=2, default=str), encoding="utf-8")
    LOG.info("COMPLETED four-way comparison, saved to %s", args.out)
    LOG.info("\n%s", summary.to_string(index=False))
    return summary


if __name__ == "__main__":
    main(arguments())
