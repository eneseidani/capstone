"""DIRECT portfolio construction comparison: BL, robust, regime-aware.

From Capstone project root:
  python S06_blacklitterman_robust_regime_comparison.py

Requires existing S06_portfolio_allocation.py and transaction_fee_helper.py,
existing project data (OHLCV, DGS3MO, exported RF/XGB predictions).
No model retraining, no maximum-Sharpe first stage. Four individual charts:
  01: raw momentum + Ledoit-Wolf mean-variance reference
  02: Black-Litterman posterior returns + Ledoit-Wolf mean-variance
  03: Ellipsoidal robust mean-variance + Ledoit-Wolf
  04: Regime-conditioned Ledoit-Wolf covariance + raw momentum

Black-Litterman: the raw momentum forecasts are the views, so posterior
expected returns are NOT identical to raw momentum. Equal-weight reference
is an explicit modeling proxy, not a genuine market-cap equilibrium portfolio.
All prior and regime estimations use data through the signal close only.
"""
from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

LOG = logging.getLogger('S06_alternative_allocations')
STRATEGIES = ('Our_RandomForest', 'Peer_XGBoost', 'Full_Universe')
LABELS = {'Our_RandomForest':'Our Random Forest', 'Peer_XGBoost':'Peer XGBoost',
          'Full_Universe':'Full investable universe'}
COLORS = {'Our_RandomForest':'#1f77b4', 'Peer_XGBoost':'#d62728',
          'Full_Universe':'#2ca02c', 'SPY':'#ff7f0e'}
CONFIGS = ('01_MeanVariance_LedoitWolf', '02_BlackLitterman_LedoitWolf',
           '03_Robust_LedoitWolf', '04_RegimeAware_LedoitWolf')


def arguments():
    p=argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--predictions',type=Path,default=Path('model_output/S05_peer_calendar_classification/predictions_test.csv'))
    p.add_argument('--peer-signals',type=Path,default=Path('portfolio_signals.csv'))
    p.add_argument('--peer-schedule',type=Path,default=Path('portfolio_rebalance_schedule.csv'))
    p.add_argument('--universe',type=Path,default=Path('Investable_Universe.csv'))
    p.add_argument('--macro',type=Path,default=Path('features_data/macro_features.csv'))
    p.add_argument('--out',type=Path,default=Path('model_output/S06_blacklitterman_robust_regime_comparison'))
    p.add_argument('--gamma',type=float,default=3.)
    p.add_argument('--direct-starts',type=int,default=12)
    p.add_argument('--initial-capital',type=float,default=100000.)
    p.add_argument('--rf-threshold',type=float,default=.5)
    p.add_argument('--xgb-threshold',type=float,default=.6)
    p.add_argument('--top-k',type=int,default=15)
    p.add_argument('--bl-tau',type=float,default=.05,
                   help='BL prior covariance scaling; fixed a priori')
    p.add_argument('--bl-view-noise-multiple',type=float,default=1.,
                   help='Omega = multiple * tau * diag(Sigma), strictly positive')
    p.add_argument('--robust-radius',type=float,default=.5,
                   help='Ellipsoidal return-estimation uncertainty radius, fixed a priori')
    p.add_argument('--uncertainty-lookback',type=int,default=756,
                   help='Past trading sessions to estimate historical 60-day momentum forecast errors')
    p.add_argument('--regime-lookback',type=int,default=756)
    p.add_argument('--regime-vol-window',type=int,default=21)
    p.add_argument('--regime-threshold-window',type=int,default=252)
    p.add_argument('--regime-min-days',type=int,default=90)
    p.add_argument('--regime-condition-weight',type=float,default=.5,
                   help='Blend conditional LW covariance and standard 252-day LW covariance')
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


def black_litterman(mu, cov, cash60, gamma, tau, omega_multiple):
    """Posterior of returns in excess of cash, from all-asset absolute momentum views.

    Prior Pi=gamma Sigma @ equal_weight_reference, Q=mu_raw-rf and P=I.
    Inverse-free Kalman form: Pi + (tau*Sigma) @ solve(tau*Sigma+Omega,Q-Pi).
    Omega proportional to individual return variances (diagonal).
    """
    n=len(mu)
    if n==0: raise ValueError('Empty BL asset universe')
    eq=np.full(n,1./n)
    pi=gamma*cov@eq
    views=mu-cash60
    prior_unc=tau*cov
    omega=np.diag(np.maximum(np.diag(prior_unc)*omega_multiple,1e-12))
    posterior=pi+prior_unc@np.linalg.solve(prior_unc+omega,views-pi)
    return cash60+posterior, pi, views


def historical_forecast_uncertainty(prices, date, tickers, lookback, horizon=60, momentum_window=120):
    """Diagonal ellipsoidal uncertainty from prior out-of-sample forecast errors.

    Sample nonoverlapping labels to avoid falsely treating overlapping 60-day
    outcomes as independent; use realized forecast errors only for fully
    matured windows. The resulting uncertainty scale is standard error of
    mean forecast error across historical windows. This is a proxy, NOT a
    formally calibrated confidence region for the momentum model.
    """
    frame=prices.loc[:date,tickers].tail(lookback+momentum_window+horizon+2)
    x=frame.to_numpy(float); t=len(frame)
    end=t-1
    errs=[]
    for signal in range(end-horizon, momentum_window-1,-horizon):
        future=signal+horizon
        if future>end: continue
        a=x[signal-momentum_window]; b=x[signal]; c=x[future]
        ok=(np.isfinite(a)&np.isfinite(b)&np.isfinite(c)&(a>0)&(b>0)&(c>0))
        forecast=np.full(len(tickers),np.nan)
        realized=np.full(len(tickers),np.nan)
        forecast[ok]=np.expm1(np.log(b[ok]/a[ok])*(horizon/momentum_window))
        realized[ok]=c[ok]/b[ok]-1.
        errs.append(realized-forecast)
        if len(errs)>=max(1,lookback//horizon):break
    if len(errs)<6: raise ValueError(f'{date}: fewer than six matured forecast-error blocks')
    e=np.asarray(errs,float)
    # ETFs can have 252 valid recent sessions yet shorter historical records.
    # Do not exclude an eligible ETF solely for this uncertainty estimate.
    # For >=6 matured errors, use its own standard error. Otherwise use a
    # conservative cross-sectional fallback (75th percentile). This fallback
    # is disclosed in the diagnostic logger and is never set to zero.
    counts=np.isfinite(e).sum(axis=0)
    valid=counts>=6
    stderr=np.full(e.shape[1],np.nan,dtype=float)
    for j in np.flatnonzero(valid):
        obs=e[np.isfinite(e[:,j]),j]
        stderr[j]=float(np.std(obs,ddof=1)/np.sqrt(len(obs)))
    if not valid.any():
        raise ValueError(f'{date}: no assets have six matured momentum forecast errors')
    fallback=max(float(np.percentile(stderr[valid],75)),1e-6)
    if (~valid).any():
        LOG.warning('%s | robust uncertainty | %d/%d tickers use conservative '
                    'historical-error fallback',date.date(),int((~valid).sum()),len(tickers))
    stderr[~valid]=fallback
    return np.maximum(stderr,1e-6),len(e)


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


def make_estimator(s06,config,args,spy_close,diagnostics,robust_scales):
    def estimator(prices,date,tickers,rf):
        date=pd.Timestamp(date)
        mu,cov,cash60,lw=raw_estimation(s06,prices,date,tickers,rf)
        diag={'Configuration':config,'Signal_Date':str(date.date()),
              'Asset_Count':len(tickers),'LW_Shrinkage':lw,
              'Covariance_Condition_Number':float(np.linalg.cond(cov)),
              'Raw_Momentum_Mean_60D':float(mu.mean()),
              'Cash_60D':cash60}
        if config==CONFIGS[1]:
            mu,pi,views=black_litterman(mu,cov,cash60,args.gamma,args.bl_tau,args.bl_view_noise_multiple)
            diag.update(BL_Reference='Equal_Weight_Eligible',
                        BL_Prior_Excess_Mean=float(pi.mean()),
                        BL_View_Excess_Mean=float(views.mean()),
                        BL_Posterior_Mean_60D=float(mu.mean()))
        elif config==CONFIGS[2]:
            sigma,nblocks=historical_forecast_uncertainty(
                prices,date,tickers,args.uncertainty_lookback,s06.HORIZON,s06.EXPECTED_RETURN_LOOKBACK)
            robust_scales[(date,tuple(tickers))]=sigma
            diag.update(Robust_StdErr_Median_60D=float(np.median(sigma)),
                        Robust_Matured_Blocks=int(nblocks))
        elif config==CONFIGS[3]:
            cov,reg=regime_covariance(prices,date,tickers,spy_close,cov,args)
            diag.update(reg)
        diagnostics.append(diag)
        return mu,cov,cash60
    return estimator


def robust_direct_optimizer(s06, date, chosen, dollars, price_hist, rf,
                            risk_aversion=3., starts=12, scales=None, radius=.5):
    """S06 direct optimizer, with added ellipsoidal return-uncertainty penalty.

    U= w'(mu-rf) - gamma/2*w'Sigma*w - rho*sqrt(sum((stderr_i*w_i)^2))
       -(fee(w)-fee(cash))/wealth.
    Uses identical S06 fee helper, constraints, next-open execution and cash rule.
    """
    if not np.isfinite(radius) or radius<0:raise ValueError('Invalid robustness radius')
    held=[tk for tk,value in dollars.items() if tk!=s06.CASH and value>1e-7]
    eligible=list(dict.fromkeys(chosen+held))
    n=len(eligible); tickers=eligible+[s06.CASH]
    prior=dollars.reindex(tickers,fill_value=0.).astype(float)
    wealth=float(prior.sum())
    if wealth<=0 or not np.isfinite(wealth):raise ValueError('Invalid pretrade wealth')
    ctx=s06.load_fee_context(date,tickers=tickers)

    def portfolio(w):
        w=np.clip(np.asarray(w,float),0.,s06.MAX_WEIGHT)
        total=float(w.sum())
        if total>1.: w=w/total; total=float(w.sum())
        target=pd.Series(0.,index=tickers,dtype=float)
        if n:target.loc[eligible]=w
        target.loc[s06.CASH]=max(0.,1.-total)
        target.loc[s06.CASH]+=1.-float(target.sum())
        return target

    cash_target=portfolio(np.zeros(n))
    cash_fee,_,_,_=s06.fee_and_trades(prior,cash_target,ctx)
    meta={'direct_eligible_count':n,'direct_risk_aversion_gamma':risk_aversion,
          'direct_start_count':starts,'direct_solver_converged_count':0,
          'direct_mean_variance_utility':0.,'direct_expected_net_excess_return':0.,
          'direct_estimated_cash_liquidation_cost':float(cash_fee),
          'direct_robust_radius':radius}
    if n<3:return cash_target,{**meta,'direct_cash_reason':'fewer_than_three_eligible_assets'}
    mu,cov,cash60=s06.estimation(price_hist,date,eligible,rf)
    sig=scales.get((pd.Timestamp(date),tuple(eligible)))
    if sig is None or len(sig)!=n:raise RuntimeError('Robust forecast uncertainty unavailable')
    excess=np.asarray(mu,float)-cash60
    cov=(cov+cov.T)/2
    tol=1e-7
    def feasible(w):
        w=np.asarray(w,float)
        return (w.shape==(n,) and np.isfinite(w).all() and w.min()>=-tol
                and w.max()<=s06.MAX_WEIGHT+tol
                and s06.MIN_RISKY_WEIGHT-tol<=w.sum()<=1.+tol)
    cache={}
    def measure(w):
        w=np.asarray(w,float)
        key=tuple(np.round(w,12))
        if key not in cache:
            if not feasible(w):return (-1e6,-1e6,np.nan)
            target=portfolio(w)
            fee,_,_,_=s06.fee_and_trades(prior,target,ctx)
            net=float(w@excess)-(fee-cash_fee)/wealth
            uncertainty=radius*float(np.linalg.norm(sig*w))
            util=net-.5*risk_aversion*max(0.,float(w@cov@w))-uncertainty
            cache[key]=(float(util),float(net),float(fee))
        return cache[key]
    seeds=[np.full(n,s06.MIN_RISKY_WEIGHT/n)]
    if n>=5:seeds.append(np.full(n,1./n))
    prior_w=prior.reindex(eligible).to_numpy(float)/wealth
    if prior_w.sum()>1e-12:
        prior_w=np.clip(prior_w,0,s06.MAX_WEIGHT)
        if prior_w.sum()<s06.MIN_RISKY_WEIGHT:prior_w+= (s06.MIN_RISKY_WEIGHT-prior_w.sum())/n
        if feasible(prior_w):seeds.append(prior_w)
    rng=np.random.default_rng(int(pd.Timestamp(date).strftime('%Y%m%d'))+149)
    max_exp=min(1.,n*s06.MAX_WEIGHT)
    for _ in range(max(0,starts-len(seeds))):
        exposure=float(rng.uniform(s06.MIN_RISKY_WEIGHT,max_exp))
        raw=rng.dirichlet(np.ones(n)*2.)
        lo=float(raw.min()-s06.MAX_WEIGHT);hi=float(raw.max())
        for _ in range(75):
            mid=(lo+hi)/2
            if np.clip(raw-mid,0,s06.MAX_WEIGHT).sum()>exposure:lo=mid
            else:hi=mid
        w=np.clip(raw-(lo+hi)/2,0,s06.MAX_WEIGHT)
        residual=exposure-float(w.sum())
        for j in np.argsort(-raw if residual>0 else raw):
            change=min(abs(residual),s06.MAX_WEIGHT-w[j] if residual>0 else w[j])
            if change<=0:continue
            w[j]+=np.sign(residual)*change
            residual-=np.sign(residual)*change
            if abs(residual)<1e-10:break
        if feasible(w):seeds.append(w)
    seeds=seeds[:starts]
    best=None;converged=0
    for seed in seeds:
        if feasible(seed):
            val=measure(seed)[0]
            if best is None or val>best[0]:best=(val,seed.copy(),'seed')
        res=minimize(lambda w:-measure(w)[0],seed,method='SLSQP',
                     bounds=[(0.,s06.MAX_WEIGHT)]*n,
                     constraints=[{'type':'ineq','fun':lambda w:float(w.sum()-s06.MIN_RISKY_WEIGHT)},
                                  {'type':'ineq','fun':lambda w:float(1.-w.sum())}],
                     options={'maxiter':250,'ftol':1e-8})
        if res.success and feasible(res.x):converged+=1
        if feasible(res.x):
            val=measure(res.x)[0]
            if best is None or val>best[0]:best=(val,np.asarray(res.x).copy(),'solver')
    if best is None:raise RuntimeError(f'{date}: No feasible robust candidate')
    utility,w,source=best
    _,net,fee=measure(w)
    meta.update(direct_solver_converged_count=converged,
                direct_selected_source=source,direct_best_feasible_risky_utility=utility,
                direct_expected_net_excess_return=net,direct_estimated_transaction_cost=fee,
                direct_evaluations=len(cache),direct_robust_uncertainty_penalty=radius*float(np.linalg.norm(sig*w)))
    if utility<=0 or net<=0:
        return cash_target,{**meta,'direct_cash_reason':'non_positive_incremental_utility'}
    target=portfolio(w)
    if not feasible(w) or target[s06.CASH]>s06.MAX_CASH_WEIGHT+tol:
        raise AssertionError('Robust portfolio violated constraints')
    LOG.info('%s | direct ROBUST | risky=%.2f%% | utility=%.6f | converged=%d/%d',
             date.date(),100*w.sum(),utility,converged,starts)
    return target,{**meta,'direct_mean_variance_utility':float(utility),
                   'direct_cash_reason':'direct_robust_utility'}


def s06_settings(args,out):
    return argparse.Namespace(predictions=args.predictions,out=out,
       threshold=args.rf_threshold,model='RandomForest',top_k=args.top_k,
       initial_capital=args.initial_capital,risk_aversion=args.gamma,
       risk_aversion_grid='none',skip_direct_utility=True,
       direct_starts=args.direct_starts,macro=args.macro,
       allow_feasible_fallback=False,initial_date=None)


def main(args):
    import S06_portfolio_allocation as s06
    if not (args.gamma>0 and args.direct_starts>=1 and args.initial_capital>0
            and args.bl_tau>0 and args.bl_view_noise_multiple>0
            and args.robust_radius>=0 and args.uncertainty_lookback>=360
            and args.regime_lookback>=252 and args.regime_vol_window>=5
            and args.regime_min_days>=30 and 0<=args.regime_condition_weight<=1):
        raise ValueError('Invalid experiment parameters')
    args.out.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[logging.StreamHandler(),logging.FileHandler(args.out/'run.log',encoding='utf-8')],force=True)

    universe=read_universe(args.universe,s06.CASH)
    rf_selections=s06.read_selections(args.predictions,'RandomForest',args.rf_threshold,args.top_k)
    dates=list(rf_selections)
    peer_selections,peer_audit=read_peer_xgboost(args.peer_signals,args.peer_schedule,dates,
                                                 set(universe),args.top_k,args.xgb_threshold)
    full_selections,universe_audit=full_universe_pit(s06,universe,dates)
    choices={'Our_RandomForest':rf_selections,'Peer_XGBoost':peer_selections,'Full_Universe':full_selections}
    for strategy,by_date in choices.items():
        if list(by_date)!=dates:raise ValueError(f'{strategy}: rebalance calendars differ')
        for dt,selected in by_date.items():
            if len(selected)!=len(set(selected)) or set(selected)-set(universe):
                raise ValueError(f'{strategy}/{dt}: invalid candidates')
    spy=s06.read_market('SPY');spy_close=spy['Adj Close']
    peer_audit.to_csv(args.out/'peer_xgboost_selection_audit.csv',index=False)
    universe_audit.to_csv(args.out/'full_universe_eligibility_audit.csv',index=False)

    navs={};summary_rows=[];weight_frames=[];rebalance_frames=[];tx_frames=[];diag=[]
    spy_metrics=None
    original_direct=s06.optimize_direct_utility
    for config in CONFIGS:
        navs[config]={}
        for strategy in STRATEGIES:
            dest=args.out/config/strategy
            settings=s06_settings(args,dest)
            robust_scales={}
            estimator=make_estimator(s06,config,args,spy_close,diag,robust_scales)
            LOG.info('START | %s | %s',config,strategy)
            def optimizer(date,chosen,dollars,price_hist,rf,risk_aversion=3.,starts=12):
                return robust_direct_optimizer(s06,date,chosen,dollars,price_hist,rf,
                                               risk_aversion,starts,robust_scales,args.robust_radius)
            with patch.object(s06,'read_selections',return_value=choices[strategy]),\
                 patch.object(s06,'estimation',estimator),\
                 patch.object(s06,'optimize_direct_utility',optimizer if config==CONFIGS[2] else original_direct):
                result=s06.run_single(settings,mode='direct')
            nav=pd.read_csv(dest/'daily_nav.csv',parse_dates=['Date'])
            navs[config][strategy]=nav
            metric=result['results']
            if spy_metrics is None:spy_metrics=metric
            summary_rows.append({'Configuration':config,'Strategy':strategy,
                'Total_Return':metric['portfolio_total_return'],
                'CAGR':metric['portfolio_CAGR'],
                'Annualized_Sharpe':metric['portfolio_annualized_sharpe'],
                'Max_Drawdown':metric['portfolio_max_drawdown'],
                'Transaction_Cost_USD':metric['total_transaction_cost_USD'],
                'Average_Daily_Cash_Weight':float(nav.Cash_Weight.mean())})
            for name,sink,filename in [
                ('weights',weight_frames,'portfolio_weights.csv'),
                ('rebalance',rebalance_frames,'rebalance_summary.csv'),
                ('trades',tx_frames,'transaction_cost_breakdown.csv')]:
                df=pd.read_csv(dest/filename)
                if not df.empty:
                    df.insert(0,'Configuration',config);df.insert(1,'Strategy',strategy)
                    sink.append(df)
            LOG.info('END | %s | %s | return=%+.2f%% | sharpe=%.3f',
                     config,strategy,100*metric['portfolio_total_return'],metric['portfolio_annualized_sharpe'])

    anchor=navs[CONFIGS[0]]['Our_RandomForest']
    joined=pd.DataFrame({'Date':anchor.Date})
    for config in CONFIGS:
        for strategy in STRATEGIES:
            nav=navs[config][strategy]
            if not nav.Date.equals(anchor.Date):raise AssertionError('NAV date mismatch across strategies')
            if not np.allclose(anchor.SPY_NAV,nav.SPY_NAV,rtol=0,atol=1e-5):
                raise AssertionError('SPY benchmark differs across runs')
            joined[f'{config}_{strategy}_NAV']=nav.Portfolio_NAV.to_numpy(float)
            joined[f'{config}_{strategy}_Cash_Weight']=nav.Cash_Weight.to_numpy(float)
        joined[f'{config}_SPY_NAV']=anchor.SPY_NAV.to_numpy(float)
        fig,ax=plt.subplots(figsize=(13,7))
        for strategy in STRATEGIES:
            nav=navs[config][strategy]
            ax.plot(nav.Date,nav.Portfolio_NAV/args.initial_capital*100,
                    label=LABELS[strategy],color=COLORS[strategy],lw=1.9)
        ax.plot(anchor.Date,anchor.SPY_NAV/args.initial_capital*100,
                label='SPY buy-and-hold',color=COLORS['SPY'],ls='--',lw=2)
        ax.set(title=f'{config} | Direct portfolio optimization | net of trading costs',
               xlabel='Date',ylabel='Portfolio NAV (initial = 100)')
        ax.legend(ncol=2);ax.grid(alpha=.25);fig.autofmt_xdate();fig.tight_layout()
        fig.savefig(args.out/f'{config.lower()}_vs_SPY.png',dpi=180)
        plt.close(fig)

    years=(anchor.Date.iloc[-1]-anchor.Date.iloc[0]).days/365.25
    spy_return=float(anchor.SPY_NAV.iloc[-1]/anchor.SPY_NAV.iloc[0]-1.)
    for config in CONFIGS:
        summary_rows.append({'Configuration':config,'Strategy':'SPY_Buy_And_Hold',
                'Total_Return':spy_return,
                'CAGR':float((1+spy_return)**(1/years)-1),
                'Annualized_Sharpe':spy_metrics['spy_annualized_sharpe'],
                'Max_Drawdown':spy_metrics['spy_max_drawdown'],
                'Transaction_Cost_USD':0.,'Average_Daily_Cash_Weight':0.})
    pd.DataFrame(summary_rows).to_csv(args.out/'allocation_methods_performance_summary.csv',index=False)
    joined.to_csv(args.out/'comparison_daily_nav.csv',index=False)
    pd.DataFrame(diag).to_csv(args.out/'allocation_method_diagnostics.csv',index=False)
    for filename,frames in [('all_portfolio_weights.csv',weight_frames),
                            ('all_rebalance_summaries.csv',rebalance_frames),
                            ('all_transaction_costs.csv',tx_frames)]:
        if frames:pd.concat(frames,ignore_index=True).to_csv(args.out/filename,index=False)
    manifest={'script':Path(__file__).name,'methods':list(CONFIGS),
              'architecture':'Direct asset-weight optimization (no Sharpe stage)',
              'benchmark':'SPY buy-and-hold, no SPY fees',
              'gamma':args.gamma,'lookback_momentum':s06.EXPECTED_RETURN_LOOKBACK,
              'covariance_lookback':s06.COVARIANCE_LOOKBACK,'horizon':s06.HORIZON,
              'mean_variance_covariance':'Ledoit-Wolf scaled T/(T-1), 252 daily sessions',
              'robust_objective':'raw_excess - gamma/2 variance - rho*||diag(forecast_std_error)w||_2 - relative transaction costs',
              'robust_radius':args.robust_radius,
              'uncertainty_estimation':'diagonal stderr from matured nonoverlapping 60-session momentum forecast errors within trailing window',
              'uncertainty_lookback':args.uncertainty_lookback,
              'bl_tau':args.bl_tau,'bl_view_noise_multiple':args.bl_view_noise_multiple,
              'bl_prior':'gamma*60d_LW_covariance @ equal_weight_eligible_reference (proxy, not market equilibrium)',
              'bl_views':'raw 120d momentum converted to 60d expected excess returns; P=identity',
              'regime_definition':'21-session SPY volatility > past 252-session median 21-session SPY volatility',
              'regime_lookback':args.regime_lookback,
              'regime_min_days':args.regime_min_days,
              'regime_condition_weight':args.regime_condition_weight,
              'regime_risk':'weighted blend of regime-conditional LW and baseline 252-session LW; raw momentum unchanged',
              'strategies':list(STRATEGIES),
              'rebalance_dates':[str(pd.Timestamp(x).date()) for x in dates],
              'transaction_costs':'S06 transaction_fee_helper native spread, impact and commissions',
              'implementation_note':'Black-Litterman is posterior return estimation followed by native mean-variance optimization; robust changes objective; regime-aware changes covariance input.',
              'lookahead':'Point-in-time historical returns and signal-date market state only',
              'warning':'Fixed plausible hyperparameters not cross-validated; past test period repeatedly inspected and not a pristine untouched holdout.'}
    (args.out/'run_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    LOG.info('COMPLETED: %s',args.out)
    LOG.info('\n%s',pd.DataFrame(summary_rows).to_string(index=False))

if __name__=='__main__':
    main(arguments())
