"""S06 portfolio allocation: two-stage risky Sharpe + capital utility.

Expected returns use the trailing 120-session adjusted-price return, converted
into a 60-session expected return by geometric compounding. Risk uses the
sample covariance of 252 daily returns, scaled to a 60-session horizon.

Decision at rebalance day's CLOSE; execute at NEXT SPY session's OPEN.
Each risky asset is capped at 20%; normal DGS3MO cash is capped at 80%.
100% cash is the explicit exception when fewer than three risky assets are
eligible or allocating to risky assets has non-positive incremental mean-variance
utility over DGS3MO. One hundred feasible Sharpe optimizer starts are used.

Do not feed realized forward-return columns into the optimizer.
"""
from __future__ import annotations

import argparse
import json
import logging
import copy
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar

import config
from transaction_fee_helper import calculate_transaction_fees, load_fee_context

LOG = logging.getLogger("S06")
CASH = "DGS3MO"
MAX_WEIGHT = 0.20
MAX_CASH_WEIGHT = 0.80
MIN_RISKY_WEIGHT = 1.0 - MAX_CASH_WEIGHT
HORIZON = 60
EXPECTED_RETURN_LOOKBACK = 120
COVARIANCE_LOOKBACK = 252
TRADING_DAYS = 252
N_STARTS = 100  # Number of feasible multistart portfolios per rebalance.
DEFAULT_RISK_AVERSION = 3.0  # Illustrative, NOT fitted on test-period results.


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", type=Path,
                   default=Path("model_output/S05_peer_calendar_classification/predictions_test.csv"))
    p.add_argument("--out", type=Path, default=Path("model_output/S06_portfolio_allocation"))
    p.add_argument("--threshold", type=float, default=0.50)
    p.add_argument("--model", default="RandomForest")
    p.add_argument("--top-k", type=int, default=15)
    p.add_argument("--initial-capital", type=float, default=100000.0)
    p.add_argument("--risk-aversion", type=float, default=DEFAULT_RISK_AVERSION,
                   help="Mean-variance risk aversion gamma on 60-session returns (default 3.0, illustrative; conduct sensitivity analysis).")
    p.add_argument("--risk-aversion-grid", type=str, default="1,2,3,5,7,10",
                   help="Comma-separated gamma values for independent full-history sensitivity runs; default 1,2,3,5,7,10. Use 'none' to disable.")
    p.add_argument("--skip-direct-utility", action="store_true",
                   help="Skip the unconstrained-by-Stage-1 direct mean-variance comparator at baseline gamma.")
    p.add_argument("--direct-starts", type=int, default=12,
                   help="Number of deterministic feasible SLSQP starts for the direct utility comparator (default 12).")
    p.add_argument("--macro", type=Path, default=Path("features_data/macro_features.csv"))
    p.add_argument("--allow-feasible-fallback", action="store_true",
                   help="Explicitly allow a feasible seed/unsuccessful solver result when no converged solver solution exists; logged and marked as UNVERIFIED.")
    p.add_argument("--initial-date", type=str, default=None,
                   help="Optional first decision date (must exist in predictions)")
    return p.parse_args()


def read_market(ticker):
    path = Path(config.INPUT_DATA_DIR) / "yahoo" / f"{ticker}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing Yahoo file: {path}")
    d = pd.read_csv(path, parse_dates=["Date"])
    required = {"Date", "Open", "Close", "Adj Close"}
    if not required.issubset(d):
        raise ValueError(f"{path} missing columns {required - set(d)}")
    if d["Date"].duplicated().any():
        raise ValueError(f"{ticker}: duplicate OHLCV dates")
    d = d.sort_values("Date").set_index("Date")
    for col in ["Open", "Close", "Adj Close"]:
        d[col] = pd.to_numeric(d[col], errors="coerce")
    if ((d["Open"] <= 0) | (d["Close"] <= 0) | (d["Adj Close"] <= 0)).any():
        raise ValueError(f"{ticker}: nonpositive price observed")
    # Adjusted Open/Close are expressed in a consistent total-return unit.
    # Fee helper still receives *raw Open* for commission-per-share pricing.
    d["Adj Open"] = d["Open"] * d["Adj Close"] / d["Close"]
    return d


def read_selections(path, model, threshold, top_k):
    d = pd.read_csv(path, parse_dates=["Date"])
    required = {"Date", "Model", "Ticker", "P_Positive_60d"}
    if not required.issubset(d):
        raise ValueError(f"Predictions missing {required-set(d)}")
    d = d.loc[d.Model.eq(model)].copy()
    if d.empty:
        raise ValueError(f"No {model} predictions in {path}")
    d["P_Positive_60d"] = pd.to_numeric(d["P_Positive_60d"], errors="coerce")
    if not np.isfinite(d["P_Positive_60d"]).all():
        raise ValueError("Non-finite model probabilities")
    if d.duplicated(["Date", "Ticker"]).any():
        raise ValueError("Duplicate model predictions per date/ticker")
    if (d["P_Positive_60d"] < 0).any() or (d["P_Positive_60d"] > 1).any():
        raise ValueError("Probabilities must lie between 0 and 1")
    choices = {}
    for dt, grp in d.groupby("Date"):
        ranked = grp.loc[grp.Ticker.ne(CASH)].sort_values(
            ["P_Positive_60d", "Ticker"], ascending=[False, True]
        )
        choices[pd.Timestamp(dt)] = ranked.head(top_k).loc[
            lambda x: x.P_Positive_60d >= threshold, "Ticker"
        ].tolist()
    return dict(sorted(choices.items()))


def load_daily_cash_rates(path, spy_dates):
    """DGS3MO annualized percent yield; lag one SPY session for publication.

    Quote reported on T is used from the next SPY session onward. This is a
    deliberately conservative rule; it does not interpolate future values.
    """
    if not path.exists():
        raise FileNotFoundError(f"Cash yield series not found: {path}")
    d = pd.read_csv(path, usecols=["Date", CASH], parse_dates=["Date"])
    d[CASH] = pd.to_numeric(d[CASH].replace(".", np.nan), errors="coerce")
    d = d.dropna(subset=[CASH]).sort_values("Date").drop_duplicates("Date", keep="last")
    if d.empty:
        raise ValueError(f"No valid {CASH} data")
    annual = d.set_index("Date")[CASH] / 100.0
    if (annual <= -1).any():
        raise ValueError("DGS3MO yields below -100%")
    # Carry last quoted yield forward, then delay the quote one equity session.
    asof = annual.reindex(spy_dates, method="ffill").shift(1)
    if asof.isna().any():
        # Rates before first available observation only; no forecast/fill from future.
        asof = asof.loc[asof.first_valid_index():]
    return (1.0 + asof).pow(1.0 / TRADING_DAYS).sub(1.0)


def market_panels(tickers, calendar, start_date, end_date):
    """Build daily OHLC panels on SPY sessions without backfilling missing data."""
    full_close = {}; full_open = {}; raw_open = {}
    for ticker in sorted(tickers):
        d = read_market(ticker)
        # Holiday/small stale-gap carry uses only *past* observations.
        # Any missing actual OPEN on an execution day triggers a hard error.
        close = d["Adj Close"].reindex(calendar).ffill(limit=2)
        full_close[ticker] = close
        full_open[ticker] = d["Adj Open"].reindex(calendar)
        raw_open[ticker] = d["Open"].reindex(calendar)
        # An asset need not have existed at the start of the backtest.
        # Missing valuations are checked only when that asset is actually held.
    return (pd.DataFrame(full_close, index=calendar),
            pd.DataFrame(full_open, index=calendar),
            pd.DataFrame(raw_open, index=calendar))


def estimation(prices, date, tickers, daily_rf):
    """Estimate 60-session returns and risk using point-in-time information.

    Expected return:
        1) Measure adjusted-price cumulative return over the previous 120
           trading intervals (121 closing-price observations).
        2) Convert that 120-session return to a 60-session equivalent using
           geometric compounding.

    Risk:
        Sample covariance of the previous 252 daily simple returns, scaled
        linearly by 60 as the standard horizon approximation.

    Cash:
        Compound the latest known one-session DGS3MO effective rate over
        60 sessions. The DGS3MO series has already been lagged one SPY session.
    """
    price_window = prices.loc[:date, tickers].tail(EXPECTED_RETURN_LOOKBACK + 1)
    price_values = price_window.to_numpy(dtype=float)
    if (len(price_window) != EXPECTED_RETURN_LOOKBACK + 1
            or not np.isfinite(price_values).all()
            or (price_values <= 0).any()):
        raise ValueError(
            f"{date.date()}: need {EXPECTED_RETURN_LOOKBACK + 1} valid adjusted closes "
            f"for the expected-return estimate: {tickers}"
        )

    cumulative_120 = price_window.iloc[-1].to_numpy(float) / price_window.iloc[0].to_numpy(float) - 1.0
    if not np.isfinite(cumulative_120).all() or (cumulative_120 <= -1.0).any():
        raise ValueError(f"{date.date()}: invalid 120-session cumulative return")
    # Geometric conversion: (1 + R_120) ** (60 / 120) - 1.
    mu_60 = np.expm1(
        np.log1p(cumulative_120) * (HORIZON / EXPECTED_RETURN_LOOKBACK)
    )

    daily_returns = prices.loc[:date, tickers].pct_change(fill_method=None).tail(COVARIANCE_LOOKBACK)
    x = daily_returns.to_numpy(dtype=float)
    if len(daily_returns) != COVARIANCE_LOOKBACK or not np.isfinite(x).all():
        raise ValueError(
            f"{date.date()}: fewer than {COVARIANCE_LOOKBACK} complete daily returns "
            f"for covariance: {tickers}"
        )
    daily_cov = np.atleast_2d(np.cov(x, rowvar=False, ddof=1))
    cov_60 = HORIZON * daily_cov

    daily_rf = float(daily_rf.loc[date])
    if not np.isfinite(daily_rf) or daily_rf <= -1.0:
        raise ValueError(f"{date.date()}: invalid one-session DGS3MO rate")
    cash_60 = float(np.expm1(HORIZON * np.log1p(daily_rf)))
    return mu_60, cov_60, cash_60


def fee_context_at_decision(signal_date, execution_date, tickers, raw_opens):
    """S04 Spread/ADV from signal close, commission share-price from next OPEN."""
    ctx = load_fee_context(signal_date, tickers=tickers)
    for ticker in tickers:
        if ticker == CASH:
            continue
        opening = raw_opens.at[execution_date, ticker]
        if not np.isfinite(opening) or opening <= 0:
            raise ValueError(f"{ticker}: missing OPEN on execution {execution_date.date()}")
        ctx.at[ticker, "Price"] = float(opening)
    return ctx


def fee_and_trades(pre_dollars, desired_weights, context):
    """Solve fees with post-fee target weights, preserving exact self-financing.

    Transaction helper insists both input weight vectors sum to one. Its
    'target' here is pre-fee *trade-dollar* weights (cash absorbs costs),
    whereas desired_weights are the exact POST-fee NAV weights.
    The fee is found by fixed point because order sizes depend on final NAV.
    """
    tickers = list(context.index)
    current = pre_dollars.reindex(tickers, fill_value=0.0).astype(float)
    wealth = float(current.sum())
    if wealth <= 0 or not np.isfinite(wealth):
        raise ValueError("Invalid portfolio NAV")
    w = desired_weights.reindex(tickers, fill_value=0.0).astype(float)
    if abs(w.sum() - 1.0) > 1e-8 or (w < -1e-9).any():
        raise ValueError("Invalid candidate target weights")
    current_w = current / wealth
    def charge(cost, breakdown=False):
        after = wealth - cost
        if after <= 0:
            raise ValueError("Transaction cost exceeds NAV")
        wanted = w * after
        # Cash trade is the balancing leg before commission settlement.
        proxy = wanted / wealth
        proxy.loc[CASH] += cost / wealth
        if not np.isclose(proxy.sum(), 1.0, atol=1e-9):
            raise ArithmeticError("Fee accounting target not self financing")
        result = calculate_transaction_fees(current_w, proxy, wealth, context,
                                            return_breakdown=breakdown)
        return result, wanted
    fee = 0.0
    for _ in range(100):
        next_fee, _ = charge(fee)
        if abs(next_fee - fee) <= max(1e-7, wealth * 1e-12):
            fee = next_fee
            break
        fee = next_fee
    else:
        raise ArithmeticError("Transaction fees did not converge")
    (calculated, breakdown), wanted = charge(fee, breakdown=True)
    if abs(calculated - fee) > max(1e-5, wealth * 1e-9):
        raise ArithmeticError("Transaction fee reconciliation failed")
    if abs(float(wanted.sum()) + fee - wealth) > max(1e-6, wealth * 1e-10):
        raise ArithmeticError("Post-fee NAV accounting failed")
    trades = wanted - current
    return fee, wanted, trades, breakdown


def optimize(date, chosen, dollars, price_hist, rf, allow_feasible_fallback=False,
             risk_aversion=DEFAULT_RISK_AVERSION):
    """Two-stage allocation with final risky weights w_i = k * q_i.

    Stage 1: maximize GROSS 60-session risky-sleeve Sharpe over q_i >= 0,
    sum(q)=1. Costs are deferred to Stage 2 because the dollar trades depend
    on k. For n>=5, q_i<=20%, allowing full risky deployment without breaching
    the final per-asset cap. For n=3 or 4 the minimally feasible sleeve cap
    is 1/n (and equal-weight composition is unavoidable under that cap).

    Stage 2: maximize incremental 60-session mean-variance utility:
       U(k) = k*q'(mu-rf) - (fee(k)-fee(all cash))/wealth
              - (gamma/2)*k^2*q'Cov*q.
    k lies in [20%, min(100%, 20%/max(q))]. A zero-weight all-cash portfolio
    is additionally permissible when fewer than three names are eligible or
    investing has non-positive best incremental utility. Transaction costs
    use the exact existing fee helper, including spread, impact and commission.
    """
    if risk_aversion <= 0 or not np.isfinite(risk_aversion):
        raise ValueError("Risk aversion gamma must be finite and > 0")
    held = [tk for tk, value in dollars.items() if tk != CASH and value > 1e-7]
    eligible = list(dict.fromkeys(chosen + held))
    all_tickers = eligible + [CASH]
    prior = dollars.reindex(all_tickers, fill_value=0.0).astype(float)
    wealth = float(prior.sum())
    if wealth <= 0 or not np.isfinite(wealth):
        raise ValueError(f"{date.date()}: invalid prior portfolio NAV")
    ctx = load_fee_context(date, tickers=all_tickers)
    n = len(eligible)

    def portfolio(w):
        w = np.asarray(w, dtype=float)
        target = pd.Series(0.0, index=all_tickers, dtype=float)
        if n:
            target.loc[eligible] = w
        target.loc[CASH] = max(0.0, 1.0 - float(w.sum()))
        return target

    cash_target = portfolio(np.zeros(n))
    cash_fee, _, _, _ = fee_and_trades(prior, cash_target, ctx)
    base = {
        "eligible_count": n,
        "estimated_cash_liquidation_cost": float(cash_fee),
        "optimizer_success": None,
        "optimizer_status": "NOT_APPLICABLE",
        "optimizer_source": "cash_rule",
        "optimizer_solver_success_count": 0,
        "optimizer_solver_attempt_count": 0,
        "optimizer_candidate_count": 0,
        "optimizer_best_start_sharpe": None,
        "optimizer_best_converged_sharpe": None,
        "optimizer_max_weight_violation": 0.0,
        "optimizer_sum_weight_violation": 0.0,
        "optimizer_selected_minus_best_seed": None,
        "risk_aversion_gamma": float(risk_aversion),
        "sleeve_weight_cap": None,
        "optimal_risky_exposure": 0.0,
        "stage2_expected_excess_return_gross": None,
        "stage2_incremental_fees_as_fraction_nav": None,
        "stage2_expected_net_excess_return": None,
        "stage2_variance_60d": None,
        "stage2_mean_variance_utility": None,
        "stage2_max_feasible_exposure": None,
    }
    if n < 3:
        return cash_target, {
            **base, "cash_reason": "fewer_than_three_eligible_assets",
            "expected_net_excess_over_cash": None, "estimated_sharpe": None,
        }

    mu_60, cov_60, cash_60 = estimation(price_hist, date, eligible, rf)
    cov_60 = 0.5 * (cov_60 + cov_60.T)
    excess_mu = mu_60 - cash_60

    # A fully invested risky sleeve has sum(q)=1. 20% sleeve cap is feasible
    # only with five names. For n=3/4 the smallest feasible cap is 1/n.
    sleeve_cap = max(MAX_WEIGHT, 1.0 / n)
    lower_tol = 1e-7
    bounds = [(0.0, sleeve_cap)] * n
    constraints = [{"type": "eq", "fun": lambda q: float(np.sum(q) - 1.0)}]

    def sleeve_valid(q):
        if q is None:
            return False
        q = np.asarray(q, dtype=float)
        return (q.shape == (n,) and np.isfinite(q).all()
                and q.min() >= -lower_tol and q.max() <= sleeve_cap + lower_tol
                and abs(float(q.sum()) - 1.0) <= lower_tol)

    def project_capped_simplex(scores):
        """Exact projection onto the 100%-invested capped risky sleeve."""
        scores = np.asarray(scores, dtype=float)
        lo, hi = float(scores.min() - sleeve_cap), float(scores.max())
        for _ in range(100):
            mid = (lo + hi) / 2.0
            if np.clip(scores - mid, 0.0, sleeve_cap).sum() > 1.0:
                lo = mid
            else:
                hi = mid
        result = np.clip(scores - (lo + hi) / 2.0, 0.0, sleeve_cap)
        residual = 1.0 - result.sum()
        # Nudge floating-point residual without changing the shape materially.
        for idx in np.argsort(-scores if residual > 0 else scores):
            if abs(residual) < 1e-12:
                break
            if residual > 0:
                delta = min(residual, sleeve_cap - result[idx])
                result[idx] += delta
                residual -= delta
            else:
                delta = min(-residual, result[idx])
                result[idx] -= delta
                residual += delta
        if not sleeve_valid(result):
            raise AssertionError("Stage 1 seed projection is infeasible")
        return result

    def sleeve_score(q):
        q = np.asarray(q, dtype=float)
        variance = float(q @ cov_60 @ q)
        if variance <= 1e-16:
            return -1e6
        return float((q @ excess_mu) / np.sqrt(variance))

    # Exactly 100 reproducible, feasible starting compositions.
    rng = np.random.default_rng(int(date.strftime("%Y%m%d")))
    seeds = [project_capped_simplex(np.zeros(n)),
             project_capped_simplex(excess_mu),
             project_capped_simplex(-excess_mu)]
    prior_q = prior.reindex(eligible).to_numpy(dtype=float)
    if prior_q.sum() > 1e-10:
        seeds.append(project_capped_simplex(prior_q / prior_q.sum()))
    scale = max(float(np.std(excess_mu)), 1e-10)
    while len(seeds) < N_STARTS:
        random_scores = rng.normal(size=n)
        if len(seeds) % 2 == 0:
            random_scores += 0.15 * (excess_mu - np.mean(excess_mu)) / scale
        seeds.append(project_capped_simplex(random_scores))
    seeds = seeds[:N_STARTS]
    assert all(sleeve_valid(q) for q in seeds)

    candidates = []
    messages = []
    converged_count = 0
    for i, seed in enumerate(seeds):
        candidates.append({"weights":seed.copy(), "score":sleeve_score(seed),
                           "source":"seed", "converged":False, "seed":i})
        fit = minimize(lambda q: -sleeve_score(q), seed, method="SLSQP",
                       bounds=bounds, constraints=constraints,
                       options={"maxiter":500, "ftol":1e-9})
        messages.append(str(fit.message))
        if fit.success and sleeve_valid(fit.x):
            converged_count += 1
        if sleeve_valid(fit.x):
            candidates.append({"weights":np.asarray(fit.x).copy(),
                               "score":sleeve_score(fit.x),
                               "source":"solver_converged" if fit.success else "solver_unconverged",
                               "converged":bool(fit.success), "seed":i})
    converged = [c for c in candidates if c["converged"]]
    best_seed = max(c["score"] for c in candidates if c["source"] == "seed")
    best_converged = max((c["score"] for c in converged), default=None)
    if not converged and not allow_feasible_fallback:
        raise RuntimeError(f"{date.date()}: zero successful Stage 1 Sharpe SLSQP solutions")
    considered = candidates if allow_feasible_fallback else [
        c for c in candidates if c["source"] == "seed" or c["converged"]]
    for c in considered:
        c["score"] = sleeve_score(c["weights"])
    selected = max(considered, key=lambda c: (c["score"], int(c["converged"])))
    if selected["source"] == "solver_converged":
        status = "CONVERGED"
    elif selected["source"] == "seed":
        status = "FEASIBLE_SEED_BEST" if converged else "UNVERIFIED_FALLBACK"
    else:
        status = "UNVERIFIED_FALLBACK"
    if selected["score"] < best_seed - 1e-8:
        raise AssertionError("Stage 1 selected Sharpe inferior to feasible seed")
    q = selected["weights"]
    max_exposure = min(1.0, MAX_WEIGHT / float(q.max()))
    if max_exposure < MIN_RISKY_WEIGHT - 1e-8:
        raise AssertionError("Stage 1 sleeve incompatible with minimum risky exposure")

    # Stage 2: solve ONE dimension, and evaluate the real transaction costs
    # at each proposed k. Commission minimums can make this non-smooth, so
    # a global grid is compared against local bounded refinements.
    sleeve_excess = float(q @ excess_mu)
    sleeve_variance = max(0.0, float(q @ cov_60 @ q))
    evaluations = {}

    def exposure_result(k):
        k = float(k)
        key = round(k, 12)
        if key not in evaluations:
            w = q * k
            target = portfolio(w)
            fee, _, _, _ = fee_and_trades(prior, target, ctx)
            gross = k * sleeve_excess
            fee_delta = (fee - cash_fee) / wealth
            net_excess = gross - fee_delta
            variance = k * k * sleeve_variance
            utility = net_excess - 0.5 * risk_aversion * variance
            evaluations[key] = (float(utility), float(net_excess), float(fee),
                                float(gross), float(fee_delta), float(variance))
        return evaluations[key]

    grid = np.linspace(MIN_RISKY_WEIGHT, max_exposure, 101)
    for k in grid:
        exposure_result(k)
    # Refine around each of the five best coarse-grid candidates.
    ranked = sorted(grid, key=lambda k: exposure_result(k)[0], reverse=True)
    step = float(grid[1] - grid[0]) if len(grid) > 1 else 0.0
    for k0 in ranked[:5]:
        a = max(MIN_RISKY_WEIGHT, float(k0) - step)
        b = min(max_exposure, float(k0) + step)
        if b > a + 1e-10:
            res = minimize_scalar(lambda k: -exposure_result(k)[0],
                                  method="bounded", bounds=(a, b),
                                  options={"xatol":1e-9})
            if res.success:
                exposure_result(res.x)
    k_best = max(evaluations, key=lambda k: evaluations[k][0])
    utility, net_excess, fee, gross, fee_delta, variance = evaluations[k_best]
    diagnostics = {
        **base,
        "optimizer_success": bool(selected["converged"]),
        "optimizer_status": status,
        "optimizer_source": selected["source"],
        "optimizer_solver_success_count": converged_count,
        "optimizer_solver_attempt_count": N_STARTS,
        "optimizer_candidate_count": len(candidates),
        "optimizer_best_start_sharpe": float(best_seed),
        "optimizer_best_converged_sharpe": best_converged,
        "optimizer_max_weight_violation": max(0.0, float(q.max() - sleeve_cap)),
        "optimizer_sum_weight_violation": abs(float(q.sum())-1.0),
        "optimizer_selected_minus_best_seed": float(selected["score"] - best_seed),
        "optimizer_solver_message": messages[selected["seed"]] if selected["source"] != "seed" else "Feasible stage 1 seed",
        "optimizer_best_eligible_candidate_sharpe": float(selected["score"]),
        "optimizer_selected_is_converged": bool(selected["converged"]),
        "optimizer_selected_is_seed": selected["source"] == "seed",
        "sleeve_weight_cap": float(sleeve_cap),
        "optimal_risky_exposure": float(k_best),
        "stage2_expected_excess_return_gross": gross,
        "stage2_incremental_fees_as_fraction_nav": fee_delta,
        "stage2_expected_net_excess_return": net_excess,
        "stage2_variance_60d": variance,
        "stage2_mean_variance_utility": utility,
        "stage2_max_feasible_exposure": float(max_exposure),
        "stage1_gross_sleeve_sharpe": float(selected["score"]),
        "stage2_exposure_grid_evaluations": len(evaluations),
    }
    if utility <= 0 or net_excess <= 0:
        reason = "non_positive_utility_vs_cash" if utility <= 0 else "cash_beats_risky_net"
        LOG.info("%s | Stage1 Sharpe=%.6f | Stage2 utility=%.6f | 100%% cash: %s",
                 date.date(), selected["score"], utility, reason)
        return cash_target, {
            **diagnostics, "cash_reason":reason,
            "expected_net_excess_over_cash":net_excess,
            "estimated_sharpe":None,
            "optimal_risky_exposure":0.0,
        }
    target = portfolio(q * k_best)
    if abs(float(target.sum()) - 1.0) > 1e-7:
        raise AssertionError("Stage 2 target weights do not sum to one")
    if target.loc[CASH] > MAX_CASH_WEIGHT + 1e-7:
        raise AssertionError("Normal cash exceeds 80%")
    if float(target.loc[eligible].max()) > MAX_WEIGHT + 1e-7:
        raise AssertionError("Stage 2 asset concentration exceeds 20%")
    LOG.info("%s | Stage1=%s (%d/%d) | sleeve Sharpe=%.6f | Stage2 gamma=%.3f | risky=%.2f%% | utility=%.6f",
             date.date(), status, converged_count, N_STARTS, selected["score"],
             risk_aversion, 100*k_best, utility)
    return target, {
        **diagnostics,
        "cash_reason":"two_stage_sharpe_utility",
        "expected_net_excess_over_cash":net_excess,
        "estimated_sharpe":net_excess / np.sqrt(variance) if variance > 1e-16 else None,
    }



def optimize_direct_utility(date, chosen, dollars, price_hist, rf,
                            risk_aversion=DEFAULT_RISK_AVERSION, starts=12):
    """Maximize the SAME cost-adjusted utility directly over all ticker weights.

    Unlike the two-stage version, individual w_i are decision variables; no
    Sharpe-optimal sleeve q is imposed. This is a numerical ex-ante comparator,
    NOT a hindsight/oracle result or a proven global optimum.
    """
    if not np.isfinite(risk_aversion) or risk_aversion <= 0:
        raise ValueError("Risk aversion gamma must be finite and > 0")
    if starts < 1:
        raise ValueError("Direct utility starts must be positive")
    held = [tk for tk, value in dollars.items() if tk != CASH and value > 1e-7]
    eligible = list(dict.fromkeys(chosen + held))
    n = len(eligible)
    tickers = eligible + [CASH]
    prior = dollars.reindex(tickers, fill_value=0.0).astype(float)
    wealth = float(prior.sum())
    if wealth <= 0 or not np.isfinite(wealth):
        raise ValueError("Invalid NAV in direct utility comparator")
    context = load_fee_context(date, tickers=tickers)

    def portfolio(w):
        """Make SLSQP finite-difference boundary trials self-financing.

        SLSQP evaluates points ~1e-8 outside the simplex to estimate a
        numerical gradient.  Such tiny excursions are accepted by `feasible`
        (1e-7 tolerance) but used to produce negative cash, which the fee
        helper correctly rejects.  Project only numerical boundary drift
        onto the closed box/simplex; materially infeasible points remain
        rejected by `measure` before reaching this helper.
        """
        w = np.clip(np.asarray(w, dtype=float), 0.0, MAX_WEIGHT)
        risky_total = float(w.sum())
        if risky_total > 1.0:
            w = w / risky_total
            risky_total = float(w.sum())
        target = pd.Series(0.0, index=tickers, dtype=float)
        if n:
            target.loc[eligible] = w
        target.loc[CASH] = max(0.0, 1.0 - risky_total)
        # Absorb machine-precision summation error in cash, not an asset.
        target.loc[CASH] += 1.0 - float(target.sum())
        return target

    cash_target = portfolio(np.zeros(n))
    cash_fee, _, _, _ = fee_and_trades(prior, cash_target, context)
    base = {"direct_eligible_count": n, "direct_risk_aversion_gamma": risk_aversion,
            "direct_start_count": starts, "direct_solver_converged_count": 0,
            "direct_selected_source": "cash_rule", "direct_mean_variance_utility": 0.0,
            "direct_expected_net_excess_return": 0.0,
            "direct_estimated_cash_liquidation_cost": float(cash_fee)}
    if n < 3:
        return cash_target, {**base, "direct_cash_reason": "fewer_than_three_eligible_assets"}

    mu, cov, rf60 = estimation(price_hist, date, eligible, rf)
    cov = 0.5 * (cov + cov.T)
    excess = mu - rf60
    tolerance = 1e-7

    def feasible(w):
        w = np.asarray(w, float)
        total = float(w.sum())
        return (w.shape == (n,) and np.isfinite(w).all()
                and w.min() >= -tolerance and w.max() <= MAX_WEIGHT+tolerance
                and total >= MIN_RISKY_WEIGHT-tolerance and total <= 1+tolerance)

    # Caching avoids repeating the expensive fee fixed-point calculation for
    # identical points (as often happens during SLSQP finite differences).
    cache = {}
    def measure(w):
        w = np.asarray(w, float)
        key = tuple(np.round(w, 12))
        if key not in cache:
            if not feasible(w):
                return (-1e6, -1e6, np.nan)
            target = portfolio(w)
            fee, _, _, _ = fee_and_trades(prior, target, context)
            gross = float(w @ excess)
            net = gross - (fee-cash_fee)/wealth
            variance = max(0.0, float(w @ cov @ w))
            util = net - risk_aversion*variance/2
            cache[key] = (float(util), float(net), float(fee))
        return cache[key]

    def objective(w):
        return -measure(w)[0]

    # Deterministic starting portfolios ranging from the minimum 20% exposure
    # to full investment; direct optimizer can change asset proportions.
    rng = np.random.default_rng(int(date.strftime("%Y%m%d"))+149)
    seeds = [np.full(n, MIN_RISKY_WEIGHT/n)]
    if n >= 5:
        seeds.append(np.full(n, 1.0/n))
    prior_risky = prior.reindex(eligible).to_numpy(float)
    if prior_risky.sum() > 0:
        seed = np.clip(prior_risky / wealth, 0, MAX_WEIGHT)
        if seed.sum() < MIN_RISKY_WEIGHT:
            seed += (MIN_RISKY_WEIGHT-seed.sum())/n
        if feasible(seed):
            seeds.append(seed)
    while len(seeds) < starts:
        exposure = float(rng.uniform(MIN_RISKY_WEIGHT, min(1.0, n*MAX_WEIGHT)))
        raw = rng.dirichlet(np.ones(n)*2)
        # Project onto capped simplex with specified sum via bisection.
        low, high = float(raw.min()-MAX_WEIGHT), float(raw.max())
        for _ in range(75):
            mid = 0.5*(low+high)
            if np.clip(raw-mid, 0, MAX_WEIGHT).sum() > exposure:
                low = mid
            else:
                high = mid
        w = np.clip(raw-0.5*(low+high), 0, MAX_WEIGHT)
        # Correct tiny floating-point residual.
        diff = exposure - w.sum()
        for j in np.argsort(-raw if diff>0 else raw):
            delta = min(abs(diff), MAX_WEIGHT-w[j] if diff>0 else w[j])
            w[j] += np.sign(diff)*delta
            diff -= np.sign(diff)*delta
            if abs(diff)<1e-10:
                break
        if feasible(w):
            seeds.append(w)
    seeds = seeds[:starts]

    best = None
    converged = 0
    for seed in seeds:
        if feasible(seed):
            candidate = (measure(seed)[0], seed.copy(), "feasible_seed")
            if best is None or candidate[0] > best[0]:
                best = candidate
        result = minimize(objective, seed, method="SLSQP",
                          bounds=[(0.0, MAX_WEIGHT)]*n,
                          constraints=[{"type":"ineq", "fun":lambda w:float(np.sum(w)-MIN_RISKY_WEIGHT)},
                                       {"type":"ineq", "fun":lambda w:float(1.0-np.sum(w))}],
                          options={"maxiter":250,"ftol":1e-8})
        if result.success and feasible(result.x):
            converged += 1
        if feasible(result.x):
            candidate = (measure(result.x)[0], np.asarray(result.x).copy(),
                         "solver_converged" if result.success else "solver_feasible_unconverged")
            if candidate[0] > best[0]:
                best = candidate
    if best is None:
        raise RuntimeError(f"{date.date()}: direct utility comparator found no feasible candidate")
    utility, w, source = best
    _, net, selected_fee = measure(w)
    meta = {**base, "direct_solver_converged_count": converged,
            "direct_selected_source": source,
            "direct_best_feasible_risky_utility": float(utility),
            "direct_expected_net_excess_return": float(net),
            "direct_estimated_transaction_cost": float(selected_fee),
            "direct_evaluations": len(cache)}
    if utility <= 0 or net <= 0:
        return cash_target, {**meta,"direct_mean_variance_utility":0.0,
                             "direct_cash_reason":"non_positive_incremental_utility"}
    target = portfolio(w)
    if not feasible(w) or target[CASH] > MAX_CASH_WEIGHT+tolerance:
        raise AssertionError("Invalid direct utility allocation")
    LOG.info("%s | direct utility gamma=%.3f | risky=%.2f%% | utility=%.6f | starts=%d converged=%d",
             date.date(),risk_aversion,100*float(w.sum()),utility,starts,converged)
    return target, {**meta,"direct_mean_variance_utility":float(utility),
                     "direct_cash_reason":"direct_joint_utility"}


def statistics(daily, trade, risk_free):
    frame = daily.copy()
    returns = frame["Net_Daily_Return"].dropna()
    spy_ret = frame["SPY_Daily_Return"].dropna()
    if returns.empty:
        raise ValueError("No portfolio returns")
    n = len(returns)
    years = n/TRADING_DAYS
    pnl = float(frame["Portfolio_NAV"].iloc[-1] / frame["Portfolio_NAV"].iloc[0])
    spy_growth = float(frame["SPY_NAV"].iloc[-1] / frame["SPY_NAV"].iloc[0])
    rf_aligned = risk_free.reindex(frame.Date).to_numpy(float)
    p_excess = frame["Net_Daily_Return"].to_numpy(float) - rf_aligned
    s_excess = frame["SPY_Daily_Return"].to_numpy(float) - rf_aligned
    def ratio(excess):
        x = excess[np.isfinite(excess)]
        return float(np.mean(x) / np.std(x, ddof=1) * np.sqrt(TRADING_DAYS)) if len(x)>1 and np.std(x, ddof=1)>0 else None
    out = {
        "portfolio_total_return": pnl-1, "spy_total_return": spy_growth-1,
        "portfolio_CAGR": pnl**(1/years)-1, "spy_CAGR": spy_growth**(1/years)-1,
        "portfolio_annualized_sharpe": ratio(p_excess), "spy_annualized_sharpe": ratio(s_excess),
        "portfolio_max_drawdown": float(frame.Portfolio_Drawdown.min()),
        "spy_max_drawdown": float(frame.SPY_Drawdown.min()),
        "total_transaction_cost_USD": float(trade.Total_Cost.sum()) if len(trade) else 0.0,
        "rebalance_count": int(trade.Execution_Date.nunique()) if len(trade) else 0,
        "daily_return_count": n,
    }
    return out


def run_single(a, mode="two_stage"):
    if not 0.5 <= a.threshold <= 1.0: raise ValueError("Threshold must lie in [0.5, 1]")
    if a.initial_capital <= 0 or a.top_k < 1: raise ValueError("Initial capital/top-k invalid")
    if not np.isfinite(a.risk_aversion) or a.risk_aversion <= 0: raise ValueError("--risk-aversion must be finite and > 0")
    if mode not in {"two_stage", "direct"}: raise ValueError("Unknown mode")
    choices = read_selections(a.predictions, a.model, a.threshold, a.top_k)
    if a.initial_date:
        cutoff = pd.Timestamp(a.initial_date)
        if cutoff not in choices: raise ValueError(f"Initial decision {cutoff.date()} not in predictions")
        choices = {k:v for k,v in choices.items() if k>=cutoff}
    decisions = list(choices)
    spy = read_market("SPY")
    sessions = spy.index
    if not decisions: raise ValueError("No decisions")
    if any(dt not in sessions for dt in decisions):
        raise ValueError("All peer calendar decision dates must be SPY trading sessions")
    first_signal = decisions[0]
    first_execution_idx = sessions.get_loc(first_signal)+1
    if first_execution_idx >= len(sessions): raise ValueError("No session after first decision")
    first_trade = sessions[first_execution_idx]
    needed = set().union(*(set(x) for x in choices.values())) | {"SPY"}
    close, opening, raw_open = market_panels(needed, sessions, first_trade, sessions[-1])
    rf = load_daily_cash_rates(a.macro, sessions)
    if rf.loc[first_signal:sessions[-1]].isna().any():
        raise ValueError("DGS3MO missing during backtest; cannot substitute zero")
    signals_by_execution = {}
    for signal in decisions:
        next_idx = sessions.get_loc(signal) + 1
        if next_idx >= len(sessions):
            LOG.warning("Skipping last signal %s: next SPY opening unavailable", signal.date())
            continue
        trade_day = sessions[next_idx]
        if trade_day in signals_by_execution:
            raise ValueError("Overlapping execution days")
        signals_by_execution[trade_day] = signal
    if not signals_by_execution: raise ValueError("No executable decisions")

    # Dollar holdings in constant total-return units; cash balance in dollars.
    shares = pd.Series(dtype=float)
    cash = float(a.initial_capital)
    last_date = first_signal
    last_prices = close.loc[first_signal]
    bench_shares = None
    bench_cash = float(a.initial_capital)
    daily = [{"Date": first_signal, "Portfolio_NAV": cash, "SPY_NAV": bench_cash,
              "Risky_Weight": 0.0, "Cash_Weight":1.0, "Net_Daily_Return":np.nan,
              "SPY_Daily_Return":np.nan}]
    weights_rows=[]; rebal_rows=[]; trade_frames=[]; spread_fallback_frames=[]
    cash_rate_at_close = rf.loc[first_signal]

    # Mark to opening using today's adjusted open, then mark to today's close.
    for dt in sessions[sessions>first_signal]:
        # Overnight cash earns the previous session's known short-term yield.
        # Optimize exactly ONCE using prior-close holdings, before seeing next OPEN.
        if dt in signals_by_execution:
            signal = signals_by_execution[dt]
            prior_close = pd.Series({tk:float(q*close.at[signal,tk]) for tk,q in shares.items()})
            prior_close.loc[CASH] = cash
            if mode == "direct":
                target, meta = optimize_direct_utility(
                    signal, choices[signal], prior_close, close, rf,
                    risk_aversion=a.risk_aversion, starts=a.direct_starts)
            else:
                target, meta = optimize(signal, choices[signal], prior_close, close, rf,
                                        allow_feasible_fallback=a.allow_feasible_fallback,
                                        risk_aversion=a.risk_aversion)
        cash *= 1.0 + float(cash_rate_at_close)
        cash_rate_at_close = float(rf.loc[dt])
        if bench_shares is None:
            # First SPY purchase is at the first strategy execution opening.
            bench_shares = bench_cash / float(opening.at[first_trade, "SPY"])
            bench_cash = 0.0
        if dt in signals_by_execution:
            pre_dollars = {}
            for tk,q in shares.items():
                if abs(q)<=1e-10: continue
                op = opening.at[dt, tk]
                if not np.isfinite(op) or op<=0:
                    raise ValueError(f"{tk}: missing execution OPEN on {dt.date()}")
                pre_dollars[tk] = float(q*op)
            pre_dollars[CASH] = cash
            prior = pd.Series(pre_dollars)
            eligible = target.index.tolist()
            context = fee_context_at_decision(signal, dt, eligible, raw_open)

            # Record every spread fallback used by the optimizer/execution
            # context, including eligible assets that ultimately receive no trade.
            if "Spread_Fallback" in context.columns:
                fallback_rows = context.loc[
                    context["Spread_Fallback"].fillna(False).astype(bool)
                ].reset_index().rename(columns={"Spread": "Spread_Used"})
                if not fallback_rows.empty:
                    fallback_rows.insert(0, "Signal_Date", signal)
                    fallback_rows.insert(1, "Execution_Date", dt)
                    spread_fallback_frames.append(fallback_rows)
                    for _, fallback in fallback_rows.iterrows():
                        LOG.warning(
                            "%s: %s spread fallback for %s; age=%s ticker observations; "
                            "spread used=%.8f; reference date=%s",
                            signal.date(),
                            fallback["Spread_Fallback_Method"],
                            fallback["Ticker"],
                            fallback["Spread_Fallback_Age_Trading_Days"],
                            float(fallback["Spread_Used"]),
                            pd.Timestamp(fallback["Spread_Reference_Date"]).date()
                            if pd.notna(fallback["Spread_Reference_Date"]) else "n/a",
                        )

            fee, post_dollars, trade_usd, breakdown = fee_and_trades(prior, target, context)
            pre_nav = prior.sum()
            post_nav = post_dollars.sum()
            shares = pd.Series({tk:float(post_dollars[tk] / opening.at[dt, tk]) for tk in eligible
                                if tk != CASH and post_dollars[tk] > 1e-10}, dtype=float)
            cash = float(post_dollars.loc[CASH])
            if cash < -1e-6: raise AssertionError("Negative cash after costs")
            if abs(post_nav + fee - pre_nav) > max(1e-6,pre_nav*1e-10):
                raise AssertionError("NAV reconciliation")
            rebal_rows.append({"Signal_Date":signal,"Execution_Date":dt,
                               "Pre_Trade_NAV":pre_nav,"Post_Trade_NAV":post_nav,
                               "Transaction_Cost_USD":fee,"Selected_S05_Count":len(choices[signal]),
                               "Eligible_Including_Holdings":len(eligible)-1,
                               "Actual_Risky_Positions":len(shares),
                               "Target_Cash_Weight":float(target.loc[CASH]),**meta})
            for tk, w in target.items():
                weights_rows.append({"Signal_Date":signal,"Execution_Date":dt,
                                     "Ticker":tk,"Target_Weight":float(w),
                                     "Post_Cost_Target_Dollars":float(post_dollars.loc[tk]),
                                     "S05_Selected":tk in choices[signal]})
            if not breakdown.empty:
                breakdown.insert(0,"Signal_Date",signal)
                breakdown.insert(1,"Execution_Date",dt)
                trade_frames.append(breakdown)
        for tk in shares.index:
            if not np.isfinite(close.at[dt,tk]):
                raise ValueError(f"{tk}: missing valuation price on {dt.date()}")
        portfolio_nav = float(cash + sum(q * float(close.at[dt, tk]) for tk,q in shares.items()))
        spy_nav = float(bench_shares*close.at[dt, "SPY"])
        prev_p = float(daily[-1]["Portfolio_NAV"])
        prev_b = float(daily[-1]["SPY_NAV"])
        daily.append({"Date":dt,"Portfolio_NAV":portfolio_nav,"SPY_NAV":spy_nav,
                      "Cash_Weight":cash / portfolio_nav,
                      "Risky_Weight":1.0-cash/portfolio_nav,
                      "Net_Daily_Return":portfolio_nav/prev_p-1.0,
                      "SPY_Daily_Return":spy_nav/prev_b-1.0})
        last_date=dt
    nav = pd.DataFrame(daily)
    nav["Portfolio_Drawdown"] = nav.Portfolio_NAV/nav.Portfolio_NAV.cummax()-1
    nav["SPY_Drawdown"] = nav.SPY_NAV/nav.SPY_NAV.cummax()-1
    tx_cols=["Signal_Date","Execution_Date","Ticker","Trade_USD","Spread_Cost",
             "Market_Impact_Cost","IBKR_Commission","Spread_Used","Spread_Original",
             "Spread_Fallback","Spread_Fallback_Method",
             "Spread_Fallback_Age_Trading_Days","Spread_Reference_Date",
             "Spread_Fallback_Quantile","Total_Cost"]
    trades = (pd.concat(trade_frames,ignore_index=True) if trade_frames else pd.DataFrame(columns=tx_cols))
    spread_audit_cols = ["Signal_Date", "Execution_Date", "Ticker", "Asset Class",
                         "Price", "ADV_Dollars", "Spread_Original", "Spread_Used", "Spread_Fallback",
                         "Spread_Fallback_Method", "Spread_Fallback_Age_Trading_Days",
                         "Spread_Reference_Date", "Spread_Fallback_Quantile",
                         "Spread_Fallback_History_Count", "Spread_Fallback_History_Start",
                         "Spread_Fallback_History_End"]
    spread_audit = (pd.concat(spread_fallback_frames, ignore_index=True)
                    if spread_fallback_frames else pd.DataFrame(columns=spread_audit_cols))
    rebal = pd.DataFrame(rebal_rows)
    weights = pd.DataFrame(weights_rows)
    a.out.mkdir(parents=True,exist_ok=True)
    nav.to_csv(a.out/"daily_nav.csv",index=False)
    rebal.to_csv(a.out/"rebalance_summary.csv",index=False)
    weights.to_csv(a.out/"portfolio_weights.csv",index=False)
    trades.to_csv(a.out/"transaction_cost_breakdown.csv",index=False)
    spread_audit.to_csv(a.out/"spread_fallback_audit.csv",index=False)
    metrics=statistics(nav,trades,rf)
    metrics["spread_fallback_count"] = int(len(spread_audit))
    if not spread_audit.empty:
        metrics["spread_fallback_method_counts"] = (
            spread_audit["Spread_Fallback_Method"].value_counts().to_dict()
        )
    else:
        metrics["spread_fallback_method_counts"] = {}
    metrics["rebalance_count"] = len(rebal)
    if not rebal.empty and "optimizer_status" in rebal:
        status_counts = rebal["optimizer_status"].fillna("NOT_APPLICABLE").value_counts().to_dict()
    else:
        status_counts = {}
    metrics["optimization_status_counts"] = status_counts
    if "optimizer_status" in rebal:
        metrics["unverified_fallback_count"] = int((rebal["optimizer_status"] == "UNVERIFIED_FALLBACK").sum())
        metrics["feasible_seed_best_count"] = int((rebal["optimizer_status"] == "FEASIBLE_SEED_BEST").sum())
    else:
        metrics["unverified_fallback_count"] = 0
        metrics["feasible_seed_best_count"] = 0
    if "optimizer_selected_minus_best_seed" in rebal:
        deficits = pd.to_numeric(rebal["optimizer_selected_minus_best_seed"], errors="coerce")
        if (deficits < -1e-8).any():
            raise AssertionError("Selected optimizer result inferior to a feasible seed")
    summary={"script":"S06_portfolio_allocation.py","prediction_file":str(a.predictions),
             "model":a.model,"threshold":a.threshold,"top_k":a.top_k,
             "initial_capital":a.initial_capital,"first_decision":str(first_signal.date()),
             "first_execution":str(first_trade.date()),"last_observation":str(last_date.date()),
             "expected_return_lookback_sessions":EXPECTED_RETURN_LOOKBACK,
             "expected_return_horizon_sessions":HORIZON,
             "covariance_lookback_sessions":COVARIANCE_LOOKBACK,
             "horizon_sessions":HORIZON,
             "maximum_risky_asset_weight":MAX_WEIGHT,
             "maximum_normal_cash_weight":MAX_CASH_WEIGHT,
             "minimum_risky_weight_when_invested":MIN_RISKY_WEIGHT,
             "minimum_eligible_assets_for_risky_investment":3,
             "optimizer_starting_points":N_STARTS,
             "allocation_method":("direct joint mean-variance utility over ticker weights" if mode=="direct" else "two-stage: gross risky sleeve maximum Sharpe, then transaction-cost-aware mean-variance utility over risky exposure"),
             "risk_aversion_gamma":a.risk_aversion,
             "risk_aversion_note":"3.0 is an illustrative default, not calibrated or optimized using test-set performance",
             "stage1_sleeve_cap_rule":(None if mode=="direct" else "20% when n>=5; 1/n for n=3 or 4 to ensure a feasible 100%-invested sleeve"),
             "covariance":"sample covariance of daily simple returns (ddof=1), scaled by 60 sessions",
             "expected_returns":"120-session cumulative adjusted-price return converted to a 60-session equivalent using geometric compounding",
             "risk_free":"DGS3MO lagged one SPY session; one-session effective rate compounded over 60 sessions for optimization and accrued daily in NAV",
             "transaction_cost_source":"transaction_fee_helper.py (point-in-time spread fallback; S04 source values unchanged)",
             "spread_fallback_policy":{
                 "short_gap_max_trading_observations":5,
                 "short_gap_method":"most recent valid historical spread",
                 "long_gap_method":"ticker-specific 90th percentile of valid S04 spreads in prior 252 ticker observations",
                 "long_gap_minimum_valid_observations":20,
                 "lookahead":"none; only observations strictly before the signal date are used",
                 "missing_history":"fail closed; never substitute zero"
             },
             "allow_feasible_fallback": bool(a.allow_feasible_fallback),
             "execution":"weights fixed at signal close; next SPY open execution; prior-close liquidity; opening commission price",
             "constraints":"if fewer than 3 eligible risky assets: 100% cash; otherwise final risky weights <=20% each and cash <=80%; 100% cash when best candidate has non-positive incremental mean-variance utility relative to cash",
             "benchmark":"SPY buy-and-hold from first execution open, no fees",
             "run_mode":mode, "direct_utility_starts":a.direct_starts if mode=="direct" else None,
             "results":metrics}
    (a.out/"run_manifest.json").write_text(json.dumps(summary,indent=2,allow_nan=False),encoding="utf-8")
    (a.out/"performance_summary.json").write_text(json.dumps(metrics,indent=2,allow_nan=False),encoding="utf-8")
    fig,ax=plt.subplots(figsize=(11,6))
    ax.plot(nav.Date,nav.Portfolio_NAV/a.initial_capital*100,label="S06 portfolio (net costs)",linewidth=1.9)
    ax.plot(nav.Date,nav.SPY_NAV/a.initial_capital*100,label="SPY buy-and-hold",linewidth=1.7)
    ax.set(title="Portfolio vs. SPY | $100 initial normalized value",
           ylabel="Value (initial = 100)",xlabel="Date")
    ax.grid(alpha=0.25);ax.legend();fig.autofmt_xdate();fig.tight_layout()
    fig.savefig(a.out/"portfolio_vs_SPY.png",dpi=180)
    plt.close(fig)
    LOG.info("Saved %s results to %s; final NAV %.2f",mode,a.out,nav.Portfolio_NAV.iloc[-1])
    return summary



def parse_gamma_grid(raw):
    if raw.strip().lower() in {"", "none", "off"}:
        return []
    values = []
    for item in raw.split(","):
        try:
            gamma = float(item.strip())
        except ValueError as ex:
            raise ValueError(f"Invalid gamma in --risk-aversion-grid: {item!r}") from ex
        if not np.isfinite(gamma) or gamma <= 0:
            raise ValueError("Every sensitivity gamma must be finite and positive")
        if gamma not in values:
            values.append(gamma)
    return values


def gamma_path(gamma):
    return "gamma_" + format(gamma, ".8g").replace(".", "p")


def run(a):
    """Independent full backtests for each gamma; baseline direct comparator.

    Each gamma maintains its OWN holdings, transaction costs, and S05-held
    eligibility (no shared state). The direct comparator likewise maintains a
    separate account. The baseline output filenames remain backward-compatible.
    """
    if a.direct_starts < 1:
        raise ValueError("--direct-starts must be >=1")
    gamma_grid = parse_gamma_grid(a.risk_aversion_grid)
    base_out = a.out
    baseline = run_single(a, mode="two_stage")
    baseline_nav = pd.read_csv(base_out/"daily_nav.csv", parse_dates=["Date"])
    baseline_metrics = json.loads((base_out/"performance_summary.json").read_text())

    def exposure_summary(folder):
        nav_path = folder/"daily_nav.csv"
        rebal_path = folder/"rebalance_summary.csv"
        nav_frame = pd.read_csv(nav_path, usecols=["Cash_Weight"])
        result = {"average_daily_cash_weight":float(nav_frame.Cash_Weight.mean())}
        if rebal_path.exists():
            rb = pd.read_csv(rebal_path, usecols=["Target_Cash_Weight"])
            result["average_target_cash_weight"] = float(rb.Target_Cash_Weight.mean())
            result["full_cash_rebalances"] = int((rb.Target_Cash_Weight >= 1-1e-8).sum())
            result["fully_invested_rebalances"] = int((rb.Target_Cash_Weight <= 1e-8).sum())
        return result

    sweep = [{"gamma":float(a.risk_aversion), "is_baseline":True,
              "output_directory":str(base_out), **baseline_metrics,
              **exposure_summary(base_out)}]
    for gamma in gamma_grid:
        if np.isclose(gamma, a.risk_aversion, atol=1e-12, rtol=0):
            continue
        variant = copy.copy(a)
        variant.risk_aversion = gamma
        variant.out = base_out / "gamma_sensitivity" / gamma_path(gamma)
        LOG.info("GAMMA SENSITIVITY | gamma=%s | out=%s",gamma,variant.out)
        result = run_single(variant,mode="two_stage")
        sweep.append({"gamma":float(gamma),"is_baseline":False,
                      "output_directory":str(variant.out),**result["results"],
                      **exposure_summary(variant.out)})

    if not a.skip_direct_utility:
        variant = copy.copy(a)
        variant.out = base_out/"direct_utility_comparator"
        LOG.info("DIRECT UTILITY COMPARATOR | gamma=%.4f | out=%s",a.risk_aversion,variant.out)
        direct = run_single(variant, mode="direct")
        direct_nav = pd.read_csv(variant.out/"daily_nav.csv",parse_dates=["Date"])
        if not baseline_nav.Date.equals(direct_nav.Date):
            raise ValueError("Direct comparator and baseline backtest calendars differ")
        baseline_nav["Direct_Utility_NAV"] = direct_nav["Portfolio_NAV"].to_numpy(float)
        baseline_nav["Direct_Utility_Net_Daily_Return"] = direct_nav["Net_Daily_Return"].to_numpy(float)
        baseline_nav["Direct_Utility_Cash_Weight"] = direct_nav["Cash_Weight"].to_numpy(float)
        baseline_nav["Direct_Utility_Drawdown"] = (baseline_nav.Direct_Utility_NAV/
                                                     baseline_nav.Direct_Utility_NAV.cummax()-1)
        baseline_nav.to_csv(base_out/"daily_nav.csv",index=False)
        baseline_metrics.update({"direct_utility_"+key:val for key,val in direct["results"].items()
                                 if key.startswith("portfolio_") or key=="total_transaction_cost_USD"})
        (base_out/"performance_summary.json").write_text(json.dumps(baseline_metrics, indent=2,allow_nan=False))
        baseline["results"] = baseline_metrics
        baseline["direct_utility_comparator"] = {
            "mode":"direct_joint_mean_variance_utility",
            "risk_aversion_gamma":float(a.risk_aversion),
            "output_directory":str(variant.out),
            "interpretation":"best numerically found ex-ante expected utility without a fixed Stage 1 Sharpe sleeve; not hindsight optimal realized return; global optimum not guaranteed",
            "results":direct["results"]}

    sweep.sort(key=lambda row:row["gamma"])
    sweep_df = pd.DataFrame(sweep)
    sweep_df.to_csv(base_out/"risk_aversion_sensitivity.csv",index=False)
    baseline["risk_aversion_sensitivity"] = {
        "grid_gamma":sorted(set(gamma_grid+[float(a.risk_aversion)])),
        "selection_warning":"Test-set sensitivity is descriptive only. Do not select gamma using final out-of-sample results.",
        "file":"risk_aversion_sensitivity.csv"}
    (base_out/"run_manifest.json").write_text(json.dumps(baseline, indent=2,allow_nan=False))

    fig, ax = plt.subplots(figsize=(11,6))
    ax.plot(baseline_nav.Date,baseline_nav.Portfolio_NAV/a.initial_capital*100,
            label=f"S06 two-stage (gamma={a.risk_aversion:g}, net)", linewidth=1.9)
    if "Direct_Utility_NAV" in baseline_nav:
        ax.plot(baseline_nav.Date,baseline_nav.Direct_Utility_NAV/a.initial_capital*100,
                label=f"Direct utility, no Sharpe-sleeve constraint (gamma={a.risk_aversion:g}, net)",linewidth=1.7)
    ax.plot(baseline_nav.Date,baseline_nav.SPY_NAV/a.initial_capital*100,
            label="SPY buy-and-hold",linewidth=1.6)
    ax.set(title="Two-stage vs direct mean-variance utility vs SPY | initial = 100",
           ylabel="Value (initial = 100)",xlabel="Date")
    ax.grid(alpha=0.25);ax.legend();fig.autofmt_xdate();fig.tight_layout()
    fig.savefig(base_out/"portfolio_vs_SPY.png",dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(sweep_df.gamma, 100*sweep_df.portfolio_total_return,marker="o",label="Net total return (%)")
    ax.set(xlabel="Risk aversion gamma",ylabel="Net cumulative return (%)",
           title="Risk-aversion sensitivity (out-of-sample descriptive, not tuning)")
    ax.grid(alpha=0.25);ax.legend();fig.tight_layout()
    fig.savefig(base_out/"risk_aversion_sensitivity.png",dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(sweep_df.gamma,100*sweep_df.average_daily_cash_weight,
            marker="o",label="Mean daily DGS3MO weight (%)")
    ax.set(xlabel="Risk aversion gamma",ylabel="Mean cash allocation (%)",
           title="Exposure sensitivity by risk aversion")
    ax.set_ylim(0,100)
    ax.grid(alpha=0.25);ax.legend();fig.tight_layout()
    fig.savefig(base_out/"risk_aversion_cash_exposure.png",dpi=180)
    plt.close(fig)
    return baseline


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,format="%(asctime)s | %(levelname)s | %(message)s")
    print(json.dumps(run(arguments()),indent=2))
