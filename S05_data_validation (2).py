"""S05 independent data and target audit. Run this before

Raw Yahoo per-ticker CSVs are audited using the EXACT S03 return convention:
remove Saturday/Sunday BEFORE applying the 60-observation forward shift.
No old label lookup is trusted. An audit must pass before model training.

Usage (from repo root): python S05_data_validation.py

S03 drops weekend observations for all assets, including cryptocurrencies,
then computes Adj Close.shift(-60) / Adj Close - 1. This audit matches it.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

H = 60

def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets',type=Path,default=Path('features_data/asset_features.csv'))
    p.add_argument('--macro',type=Path,default=Path('features_data/macro_features.csv'))
    p.add_argument('--regimes',type=Path,default=Path('features_data/market_regimes.csv'))
    p.add_argument('--universe',type=Path,default=Path('Investable_Universe.csv'))
    p.add_argument('--price-dir',type=Path,default=Path('input_data/yahoo'))
    p.add_argument('--price-column',default='Adj Close',help='Price convention used in S03; default Adj Close')
    p.add_argument('--out',type=Path,default=Path('model_output/S05_data_validation'))
    p.add_argument('--atol',type=float,default=1e-6)
    p.add_argument('--max-example-errors',type=int,default=1000)
    return p.parse_args()

def load_csv(path, required):
    if not path.is_file(): raise FileNotFoundError(f'Input not found: {path}')
    x=pd.read_csv(path,low_memory=False)
    if x.columns.duplicated().any(): raise ValueError(f'Duplicated column names: {path}')
    missing=set(required)-set(x.columns)
    if missing: raise ValueError(f'{path}: missing columns {sorted(missing)}')
    if 'Date' in x:
        x['Date']=pd.to_datetime(x['Date'],errors='raise').dt.normalize()
        if x.Date.isna().any(): raise ValueError(f'{path}: invalid Date')
    return x

def yahoo_path(folder,ticker):
    candidates=[folder/(ticker+suffix) for suffix in ('.csv','.CSV')]
    for c in candidates:
        if c.exists(): return c
    # Windows files may hide extensions but actual file still has .csv.
    raise FileNotFoundError(f'Yahoo price file for {ticker} not found in {folder}')

def verify_ticker(ticker, block, folder, price_column, atol):
    src=load_csv(yahoo_path(folder,ticker),['Date',price_column])
    if src.Date.duplicated().any(): raise ValueError(f'{ticker}: duplicate raw price dates')
    # Use the FULL raw chronology: a feature observed at t needs a source price
    # at t+60 for its forward label. Truncating source data at the last feature
    # date would incorrectly mark the last 60 labelled observations immature.
    # S03 load_yahoo_data() drops weekend observations BEFORE constructing
    # returns and targets. In particular, Bitcoin uses 60 *weekday* rows,
    # not 60 consecutive crypto calendar-day observations.
    src=src.loc[src.Date.dt.dayofweek < 5].sort_values('Date').copy()
    px=pd.to_numeric(src[price_column],errors='coerce')
    if (px.dropna()<=0).any(): raise ValueError(f'{ticker}: nonpositive raw {price_column}')
    src['Audit_Price']=px
    src['Audit_End_Date']=src.Date.shift(-H)
    src['Audit_Return_60d']=src.Audit_Price.shift(-H).div(src.Audit_Price).sub(1)
    d=block.merge(src[['Date','Audit_End_Date','Audit_Return_60d','Audit_Price']],on='Date',how='left',validate='one_to_one')
    observed = d.Audit_Price.notna()
    valid = observed & d.Audit_End_Date.notna() & d.Audit_Return_60d.notna()
    supplied_return=pd.to_numeric(d.Forward_Return_60d,errors='coerce')
    supplied_target=pd.to_numeric(d.Target_60d,errors='coerce')
    mismatch=np.zeros(len(d),dtype=bool)
    reason=np.full(len(d),'',dtype=object)
    def add(mask,msg):
        nonlocal mismatch,reason
        mask=np.asarray(mask,dtype=bool)
        reason[mask]=np.where(reason[mask]=='',msg,reason[mask]+';'+msg)
        mismatch |= mask
    add(~observed,'no_matching_raw_price')
    add(valid & (~np.isfinite(supplied_return.to_numpy())),'missing_60d_return')
    add(valid & ~np.isclose(supplied_return.to_numpy(dtype=float),d.Audit_Return_60d.to_numpy(dtype=float),rtol=0,atol=atol,equal_nan=False),'return_difference')
    add(valid & (~supplied_target.isin([0,1]).to_numpy()),'missing_or_nonbinary_target')
    add(valid & (supplied_target.to_numpy(dtype=float)!=(d.Audit_Return_60d.to_numpy(dtype=float)>0).astype(float)),'target_difference')
    # Allow last H rows to be unlabelled; never train on them.
    add(~valid & observed & (supplied_return.notna().to_numpy() | supplied_target.notna().to_numpy()),'immature_label_present')
    d['Audit_Issue']=reason
    # Exact maturity comes from RAW source chronology, not from feature data.
    d['Label_End_Date']=d.Audit_End_Date
    d['Audit_Forward_Return_60d']=d.Audit_Return_60d
    d['Audit_Target_60d']=np.where(valid,(d.Audit_Return_60d>0).astype(float),np.nan)
    d['Audit_Mismatch']=mismatch
    return d

def main(a=None):
    a=a or arguments()
    a.out.mkdir(parents=True,exist_ok=True)
    report={'status':'FAIL','horizon_asset_observations':H,'s03_weekdays_only':True,'price_column':a.price_column,'source_price_audit_passed':False,
            'macro_publication_vintage_audited':False,'regime_formation_point_in_time_audited':False,
            'limitations':['This checks raw-price labels, not release-time availability of macro series or historical revisions.',
                           'Regime construction itself must be verified as causal in S04.',
                           'Features observed at the close of t require execution no earlier than the next tradable opportunity.',
                           'S03 excludes Saturdays and Sundays for all assets; asset-specific market holidays can still produce different elapsed calendar horizons.']}
    try:
        uni=load_csv(a.universe,['Ticker','Asset Class','Source'])
        uni=uni.loc[uni.Source.astype(str).str.strip().str.casefold().eq('yahoo finance'),['Ticker','Asset Class']].copy()
        uni['Ticker']=uni.Ticker.astype(str).str.strip()
        if uni.Ticker.duplicated().any() or uni['Asset Class'].isna().any(): raise ValueError('Invalid universe ticker/class keys')
        asset=load_csv(a.assets,['Date','Ticker','Forward_Return_60d','Target_60d'])
        asset['Ticker']=asset.Ticker.astype(str).str.strip()
        asset=asset.loc[asset.Date.ge('2008-01-01')].merge(uni,on='Ticker',how='inner',validate='many_to_one')
        if asset.empty or asset.duplicated(['Ticker','Date']).any(): raise ValueError('Empty/duplicated eligible asset panel')
        audited=[]
        for ticker,sub in asset.groupby('Ticker',sort=True):
            audited.append(verify_ticker(ticker,sub,a.price_dir,a.price_column,a.atol))
        checked=pd.concat(audited,ignore_index=True)
        errors=checked.loc[checked.Audit_Mismatch,['Ticker','Date','Audit_Issue','Forward_Return_60d','Audit_Forward_Return_60d','Target_60d','Audit_Target_60d','Label_End_Date']]
        # Preserve the complete discrepancy inventory for diagnosis. The
        # --max-example-errors option only bounds the quick-inspection sample.
        errors.to_csv(a.out/'label_discrepancies_full.csv',index=False)
        errors.head(a.max_example_errors).to_csv(a.out/'label_discrepancies.csv',index=False)
        if not errors.empty:
            by_reason = (errors.assign(Reason=errors.Audit_Issue.str.split(';'))
                         .explode('Reason').groupby('Reason',dropna=False).size()
                         .rename('Count').reset_index().sort_values('Count',ascending=False))
            by_reason.to_csv(a.out/'discrepancy_reason_counts.csv',index=False)
            print('DISCREPANCIES BY REASON:')
            print(by_reason.to_string(index=False))
            print('SAMPLE DISCREPANCIES:')
            print(errors.head(10).to_string(index=False))
        report.update({'price_rows_checked':int(len(checked)),'tickers_checked':int(checked.Ticker.nunique()),
                       'label_mismatches':int(len(errors)), 'error_types':checked.loc[checked.Audit_Mismatch,'Audit_Issue'].value_counts().to_dict(),
                       'first_date':str(checked.Date.min().date()),'last_date':str(checked.Date.max().date())})
        if len(errors): raise ValueError(f'{len(errors)} source-price/label discrepancies: see label_discrepancies.csv')
        macro=load_csv(a.macro,['Date']);reg=load_csv(a.regimes,['Date','Market_Regime'])
        if macro.Date.duplicated().any() or reg.Date.duplicated().any(): raise ValueError('Duplicate macro/regime date')
        if not reg.Market_Regime.isin(['Risk-On','Risk-Off','Transition']).all(): raise ValueError('Invalid regime categories')
        # Macro as-of matching matches S03-style daily panel alignment, but
        # cannot prove publication-time availability or historical vintage.
        # Existing earlier macro columns are not silently allowed to overwrite.
        panel=checked.drop(columns=['Audit_Price','Audit_End_Date','Audit_Return_60d','Audit_Mismatch','Audit_Issue'])
        overlapping=(set(panel.columns)&set(macro.columns))-{'Date'}
        if overlapping: raise ValueError(f'Unexpected macro column collisions: {sorted(overlapping)}')
        panel=pd.merge_asof(panel.sort_values('Date'),macro.sort_values('Date'),on='Date',direction='backward',tolerance=pd.Timedelta(days=4))
        if 'Market_Regime' not in panel:
            panel=pd.merge_asof(panel.sort_values('Date'),reg[['Date','Market_Regime']].sort_values('Date'),on='Date',direction='backward',tolerance=pd.Timedelta(days=4))
        if panel.Market_Regime.isna().any(): raise ValueError('Missing regime labels after causal date join')
        if panel.duplicated(['Date','Ticker']).any(): raise ValueError('Joined panel has duplicate date/ticker')
        lookup=panel[['Date','Ticker','Label_End_Date','Audit_Forward_Return_60d','Audit_Target_60d']].copy()
        lookup.to_csv(a.out/'label_maturity_60d.csv',index=False)
        panel.to_csv(a.out/'combined_features_validated.csv',index=False)
        report.update({'status':'PASS_SOURCE_PRICE_CHECK','source_price_audit_passed':True,
                       'combined_rows':int(len(panel)),'numeric_features':int(panel.select_dtypes('number').shape[1])})
    except Exception as exc:
        report['failure']=str(exc)
        (a.out/'audit_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        raise
    (a.out/'audit_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
    return report

if __name__=='__main__': main()
