"""S05 classification ablation: SAME S06 direct-utility allocator, two universes.

Run from the Capstone project root (beside S06_portfolio_allocation.py):

    python S06_classification_ablation.py

Compares (A) S05 RandomForest Top-15 p>=0.50 and (B) the entire point-in-time
tradable investable universe, using the identical 60-session mean-variance
utility function, gamma, covariance/expected-return estimators, fee helper,
next-session execution, 20% position cap, 80% normal cash cap, and defensive
cash exception. SPY buy-and-hold is a third curve and NEVER a training label.

The full universe comes from Investable_Universe.csv, not S05 probabilities.
It includes SPY as a *candidate ticker*, capped at 20%, as well as an external
100% SPY benchmark. The DGS3MO cash proxy is never a risky candidate.

Only past prices and point-in-time S04 fee inputs determine universe eligibility.
Ticker exclusions are written to an audit CSV. The optimizer may also retain
existing holdings that the S05 classifier did not reselect, exactly as in S06.
No modifications are made to the S05 or S06 files.
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LOG = logging.getLogger("S06_ablation")


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--universe", type=Path, default=Path("Investable_Universe.csv"))
    p.add_argument("--predictions", type=Path,
                   default=Path("model_output/S05_peer_calendar_classification/predictions_test.csv"))
    p.add_argument("--macro", type=Path, default=Path("features_data/macro_features.csv"))
    p.add_argument("--out", type=Path, default=Path("model_output/S06_classification_ablation"))
    p.add_argument("--gamma", type=float, default=3.0)
    p.add_argument("--direct-starts", type=int, default=12)
    p.add_argument("--threshold", type=float, default=0.50)
    p.add_argument("--top-k", type=int, default=15)
    p.add_argument("--initial-capital", type=float, default=100000.0)
    return p.parse_args()


def universe_tickers(path):
    if not path.exists():
        raise FileNotFoundError(f"Investable universe missing: {path}")
    universe = pd.read_csv(path)
    if "Ticker" not in universe:
        raise ValueError("Investable_Universe.csv must have a Ticker column")
    candidates = universe["Ticker"].astype(str).str.strip().tolist()
    if len(set(candidates)) != len(candidates):
        raise ValueError("Duplicate universe tickers")
    if "SPY" not in candidates:
        raise ValueError("SPY must be in investable universe for benchmark parity")
    if "DGS3MO" not in candidates:
        raise ValueError("Cash DGS3MO must be in investable universe")
    return [t for t in candidates if t and t != "DGS3MO"]


def point_in_time_universe(s06, raw_universe, dates):
    """Check each candidate before each decision, never its forward return.

    A candidate needs 253 historical adjusted closes for S06's covariance,
    valid historical 120-day return, and usable *as-of-date* S04 fees.
    The next-session opening-price requirement is handled by S06 execution;
    we additionally check its presence to avoid a phantom tradeable asset.
    This is a tradeability check, not a future return screen.
    """
    series = {}
    audit = []
    for ticker in raw_universe:
        try:
            series[ticker] = s06.read_market(ticker)
        except (FileNotFoundError, ValueError) as exc:
            series[ticker] = None
            LOG.warning("Unable to load %s: %s", ticker, exc)

    spy = s06.read_market("SPY")
    sessions = spy.index
    choices = {}
    for signal in dates:
        signal = pd.Timestamp(signal)
        if signal not in sessions:
            raise ValueError(f"Decision date is not SPY session: {signal.date()}")
        loc = sessions.get_loc(signal)
        if loc + 1 >= len(sessions):
            raise ValueError(f"No next session after {signal.date()}")
        exec_date = sessions[loc+1]
        available = []
        for ticker in raw_universe:
            d = series[ticker]
            reason = None
            if d is None:
                reason = "missing_or_invalid_OHLCV_file"
            else:
                # Match S06's previous-observation-only close fill convention.
                last_253 = d["Adj Close"].reindex(sessions[:loc+1]).ffill(limit=2).tail(253)
                if len(last_253) != 253 or not np.isfinite(last_253.to_numpy(float)).all():
                    reason = "insufficient_253_historical_closes"
                elif (last_253 <= 0).any():
                    reason = "invalid_historical_price"
                elif (not np.isfinite(float(d["Open"].get(exec_date, np.nan)))
                      or float(d["Open"].get(exec_date, np.nan)) <= 0):
                    reason = "unavailable_next_open_for_execution"
                else:
                    try:
                        s06.load_fee_context(signal, tickers=[ticker, s06.CASH])
                    except (ValueError, FileNotFoundError, KeyError, ArithmeticError) as exc:
                        reason = "no_point_in_time_fee_data:" + type(exc).__name__
            if reason is None:
                available.append(ticker)
            audit.append({"Signal_Date":signal, "Ticker":ticker,
                          "Eligible":reason is None, "Reason":reason or "eligible"})
        if len(available) < 3:
            LOG.warning("%s: only %d universe assets eligible; S06 will choose cash",
                        signal.date(), len(available))
        LOG.info("%s: %d/%d universe assets eligible", signal.date(),
                 len(available), len(raw_universe))
        choices[signal] = available
    return choices, pd.DataFrame(audit)


def main(args):
    import S06_portfolio_allocation as s06
    if not np.isfinite(args.gamma) or args.gamma <= 0:
        raise ValueError("Gamma must be positive")
    if args.direct_starts < 1:
        raise ValueError("--direct-starts must be >= 1")
    args.out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s",
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(args.out/"run.log", encoding="utf-8")],
                        force=True)

    selections = s06.read_selections(args.predictions, "RandomForest", args.threshold, args.top_k)
    universe = universe_tickers(args.universe)
    # Insist on same declared investable universe, not an alternative proxy.
    unknown = sorted(set().union(*map(set, selections.values())) - set(universe))
    if unknown:
        raise ValueError(f"S05 selected names missing from declared universe: {unknown}")
    dates = list(selections)
    all_choices, audit = point_in_time_universe(s06, universe, dates)
    audit.to_csv(args.out/"universe_eligibility_audit.csv", index=False)

    s06_args = argparse.Namespace(
        predictions=args.predictions, out=args.out/"S05_filtered", threshold=args.threshold,
        model="RandomForest", top_k=args.top_k, initial_capital=args.initial_capital,
        risk_aversion=args.gamma, risk_aversion_grid="none", skip_direct_utility=True,
        direct_starts=args.direct_starts, macro=args.macro,
        allow_feasible_fallback=False, initial_date=None,
    )
    LOG.info("RUN 1/2: S05-selected candidates, S06 DIRECT mean-variance utility")
    filtered = s06.run_single(s06_args, mode="direct")

    LOG.info("RUN 2/2: All investable tickers, SAME S06 DIRECT mean-variance utility")
    original = s06.read_selections
    try:
        # Override ONLY the candidate list. Everything else is original S06.
        s06.read_selections = lambda path, model, threshold, top_k: all_choices
        full_args = copy.copy(s06_args)
        full_args.out = args.out/"Full_Universe"
        full = s06.run_single(full_args, mode="direct")
    finally:
        s06.read_selections = original

    df_a = pd.read_csv(s06_args.out/"daily_nav.csv", parse_dates=["Date"])
    df_b = pd.read_csv(full_args.out/"daily_nav.csv", parse_dates=["Date"])
    if not df_a.Date.equals(df_b.Date):
        raise ValueError("The strategy NAV calendars differ: comparison not valid")
    if not np.allclose(df_a.SPY_NAV, df_b.SPY_NAV, rtol=0, atol=1e-5):
        raise ValueError("The SPY benchmarks differ: comparison not valid")
    nav = pd.DataFrame({"Date":df_a.Date, "S05_Filtered_NAV":df_a.Portfolio_NAV,
                        "Full_Universe_NAV":df_b.Portfolio_NAV, "SPY_NAV":df_a.SPY_NAV,
                        "S05_Filtered_Cash_Weight":df_a.Cash_Weight,
                        "Full_Universe_Cash_Weight":df_b.Cash_Weight})
    nav.to_csv(args.out/"comparison_daily_nav.csv", index=False)
    labels = ["S05 filtered", "Full universe (no S05 filter)", "SPY"]
    columns = ["S05_Filtered_NAV", "Full_Universe_NAV", "SPY_NAV"]
    fig, ax = plt.subplots(figsize=(12,6))
    for col, label in zip(columns, labels):
        ax.plot(nav.Date, nav[col]/args.initial_capital*100, label=label)
    ax.set(title=f"S05 classification ablation | same direct utility | gamma={args.gamma:g} | net strategies",
           xlabel="Date", ylabel="Normalized NAV (initial = 100)")
    ax.legend(); ax.grid(alpha=.25); fig.autofmt_xdate(); fig.tight_layout()
    fig.savefig(args.out/"S05_filter_vs_full_universe_vs_SPY.png", dpi=180)
    plt.close(fig)

    filtered_results = filtered["results"]
    full_results = full["results"]
    summary_rows = []
    for label, results, pth in [(labels[0], filtered_results, s06_args.out),
                                (labels[1], full_results, full_args.out)]:
        rebal = pd.read_csv(pth/"rebalance_summary.csv")
        weights = pd.read_csv(pth/"portfolio_weights.csv")
        summary_rows.append({"Strategy":label,
                             "Total_Return":results["portfolio_total_return"],
                             "CAGR":results["portfolio_CAGR"],
                             "Sharpe":results["portfolio_annualized_sharpe"],
                             "Max_Drawdown":results["portfolio_max_drawdown"],
                             "Transaction_Cost_USD":results["total_transaction_cost_USD"],
                             "Average_Cash_Weight":float(pd.read_csv(pth/"daily_nav.csv").Cash_Weight.mean()),
                             "Mean_Eligible_Assets":float(rebal.Eligible_Including_Holdings.mean()),
                             "Mean_Risky_Positions":float(rebal.Actual_Risky_Positions.mean()),
                             "Mean_Target_Cash":float(rebal.Target_Cash_Weight.mean()),
                             "Total_Selected_Weight_Rows":len(weights)})
    summary_rows.append({"Strategy":"SPY", "Total_Return":filtered_results["spy_total_return"],
                         "CAGR":filtered_results["spy_CAGR"],
                         "Sharpe":filtered_results["spy_annualized_sharpe"],
                         "Max_Drawdown":filtered_results["spy_max_drawdown"],
                         "Transaction_Cost_USD":0.0})
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.out/"comparison_summary.csv",index=False)
    manifest = {"experiment":"S05 classification filter ablation",
                "investable_universe_file":str(args.universe),
                "predictions_file":str(args.predictions),
                "candidate_rules":{"filtered":"S05 RandomForest Top-K probability>=threshold + existing holdings",
                                   "full":"all point-in-time eligible tickers in universe CSV + existing holdings"},
                "universe_includes_SPY_as_candidate":True,
                "SPY_candidate_max_weight":s06.MAX_WEIGHT,
                "SPY_benchmark":"100% SPY buy and hold; zero benchmark fees per S06 convention",
                "allocator_for_both":"S06 optimize_direct_utility; no Stage1 Sharpe sleeve",
                "gamma":args.gamma, "direct_starts":args.direct_starts,
                "same_decision_dates":True,
                "same_covariance_expected_returns_fees_and_execution":True,
                "s06_filtered_manifest":filtered,
                "s06_full_universe_manifest":full,
                "note":"Full universe has a larger feasible set; that does not guarantee higher realized returns. Test-period analysis is exploratory."}
    (args.out/"comparison_manifest.json").write_text(json.dumps(manifest,indent=2,default=str),encoding="utf-8")
    LOG.info("Completed. Outputs in %s", args.out)
    LOG.info("\n%s", summary.to_string(index=False))
    return summary


if __name__ == "__main__":
    main(arguments())
