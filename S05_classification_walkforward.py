"""S05 Classification Model v2: point-in-time 60-session walk-forward experiment.

The script predicts whether each asset's *forward* 60-observation return is
positive. Forward returns/targets are outcomes only, never model inputs.

Data assumptions
----------------
combined_features.csv: Date, Ticker, Target_60d, Forward_Return_60d, and
numeric predictors. Investable_Universe.csv: Ticker, Asset Class, Source.
market_regimes.csv: Date, Market_Regime (Risk-On/Risk-Off/Transition), unless
Market_Regime is already in combined_features.csv.
The label lookup must contain Date, Ticker, Label_End_Date. Label_End_Date is
the date on which the forward outcome became observable.

Validation (2016-2021) selects hyperparameters and screening rules. Test
(2022 onward) is report-only. During test, models may learn from newly matured
historical labels, but parameters and screening rules remain frozen.
"""
from __future__ import annotations
import argparse, json, logging, warnings, tempfile, shutil, hashlib, time
from pathlib import Path
import numpy as np
import pandas as pd
import optuna
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (roc_auc_score, matthews_corrcoef,
    balanced_accuracy_score, average_precision_score, log_loss, f1_score,
    precision_score, recall_score, confusion_matrix, accuracy_score)
from sklearn.inspection import permutation_importance
from dataclasses import dataclass

LOG = logging.getLogger("S05_v2")
TRAIN_START = pd.Timestamp("2008-01-01")
VALID_START = pd.Timestamp("2016-01-01")
TEST_START = pd.Timestamp("2022-01-01")
HORIZON = 60
STEP = 60
THRESHOLD = 0.50
HIST_COL = "Ticker_Historical_Positive_Rate_60d"

# Fixed provisional feature exclusions based on pre-2022 sampled |Pearson r| > 0.90.
# Upstream CSVs remain unchanged. The list is frozen for this experiment.
CORRELATION_THRESHOLD = 0.90
CORRELATION_EXCLUSION_CANDIDATES = frozenset([
    'Mean_Reversion_20d',
    'THREEFYTP5_RollingMean_20d',
    'THREEFYTP10_RollingMean_20d',
    'Mean_Reversion_60d',
    'DGS10_RollingMean_20d',
    'THREEFYTP5_RollingMean_60d',
    'T10Y3M_RollingMean_60d',
    'DGS30_RollingMean_60d',
    'DGS2_RollingMean_60d',
    'MA_50d',
    'MA_20d',
    'THREEFYTP10_RollingMean_60d',
    'T10Y2Y_RollingMean_60d',
    'DGS3MO_RollingMean_20d',
    'VXDCLS_RollingMean_20d',
    'DGS30_RollingMean_20d',
    'DGS2_RollingMean_20d',
    'THREEFYTP5_RollingMean_252d',
    'DGS30_RollingMean_252d',
    'THREEFYTP2_RollingMean_60d',
    'DX-Y.NYB_RollingMean_60d',
    'VIXCLS_RollingMean_20d',
    'T5YIE_RollingMean_20d',
    'T10Y3M_RollingMean_20d',
    'THREEFYTP5_RollingZScore_252d',
    'T10YIE_RollingStd_252d',
    'THREEFYTP5_RollingZScore_60d',
    'THREEFYTP5_RollingStd_60d',
    'THREEFYTP5_Change_60d',
    'T10YIE_RollingMean_60d',
    'DGS3MO_RollingMean_60d',
    'CL=F_RollingMean_60d',
    'BAA10Y_RollingMean_60d',
    'AAA10Y_RollingMean_60d',
    'THREEFYTP5_RollingZScore_20d',
    'THREEFYTP5_RollingStd_20d',
    'THREEFYTP5_Change_20d',
    'THREEFYTP2_RollingMean_20d',
    'T5YIFR_RollingMean_20d',
    'T10Y2Y_RollingMean_20d',
    'DX-Y.NYB_RollingMean_20d',
    'VXDCLS_RollingZScore_252d',
    'VXDCLS_RollingStd_252d',
    'VXDCLS_RollingMean_252d',
    'THREEFYTP5_RollingStd_252d',
    'THREEFYTP10_RollingMean_252d',
    'T5YIFR_RollingMean_252d',
    'T5YIE_RollingZScore_252d',
    'T10Y3M_RollingZScore_252d',
    'T10Y3M_RollingMean_252d',
    'DGS3MO_RollingStd_252d',
    'DGS3MO_RollingMean_252d',
    'DGS30_RollingZScore_252d',
    'VXDCLS_RollingZScore_60d',
    'VXDCLS_RollingStd_60d',
    'VXDCLS_RollingMean_60d',
    'VXDCLS_Change_60d',
    'T5YIE_RollingStd_60d',
    'T5YIE_Change_60d',
    'DGS30_RollingZScore_60d',
    'DGS30_RollingStd_60d',
    'DGS30_Change_60d',
    'DGS10_RollingMean_60d',
    'VXDCLS_RollingZScore_20d',
    'VXDCLS_RollingStd_20d',
    'VXDCLS_Change_20d',
    'THREEFYTP2_RollingZScore_20d',
    'T10YIE_RollingMean_20d',
    'DGS30_RollingZScore_20d',
    'DGS30_Change_20d',
    'CL=F_RollingMean_20d',
    'BAA10Y_RollingMean_20d',
    'AAA10Y_RollingMean_20d',
    'THREEFYTP5',
    'Volatility_120d',
    'THREEFYTP10',
    'DGS3MO',
    'VXDCLS',
    'T5YIE',
    'T10Y3M',
    'Price_to_MA_200d',
    'MVGFD027MNFRBDAL',
    'DGS30',
    'Volume_Ratio_60d',
    'Volatility_60d',
])

MODELS = ("LogisticRegression", "RandomForest", "XGBoost", "MLP")
METRICS = ("ROC_AUC", "MCC", "Balanced_Accuracy", "PR_AUC", "Log_Loss", "F1")


def args_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", type=Path, default=Path("model_output/S05_data_validation/combined_features_validated.csv"))
    p.add_argument("--universe", type=Path, default=Path("Investable_Universe.csv"))
    p.add_argument("--regimes", type=Path, default=Path("features_data/market_regimes.csv"))
    p.add_argument("--label-lookup", type=Path, default=Path("model_output/S05_data_validation/label_maturity_60d.csv"))
    p.add_argument("--audit-report", type=Path, default=Path("model_output/S05_data_validation/audit_report.json"))
    p.add_argument("--out", type=Path, default=Path("model_output/S05_classification_walkforward"))
    p.add_argument("--trials", type=int, default=32, help="Optuna trials PER ML algorithm")
    p.add_argument("--tuning-folds", type=int, default=6, help="Chronologically distributed validation dates used inside each Optuna trial; winning configuration is evaluated on every validation date")
    p.add_argument("--startup-trials", type=int, default=8)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--top-k", type=int, nargs="+", default=[10, 15])
    p.add_argument("--screen-thresholds", type=float, nargs="+", default=[.50, .55, .60])
    p.add_argument("--screen-min-average", type=float, default=10.0)
    p.add_argument("--permutation-repeats", type=int, default=5)
    p.add_argument("--skip-importance", action="store_true")
    p.add_argument("--skip-ablation", action="store_true")
    p.add_argument("--keep-fold-cache", action="store_true", help="Retain disk-based preprocessing cache after run (can occupy GB)")
    return p.parse_args()


def check_cols(df, required, label):
    missing = set(required) - set(df.columns)
    if missing: raise ValueError(f"{label}: missing columns {sorted(missing)}")


def load_data(a):
    if not a.audit_report.is_file():
        raise FileNotFoundError(f"Run S05_data_validation.py first; missing {a.audit_report}")
    audit = json.loads(a.audit_report.read_text(encoding="utf-8"))
    if audit.get("status") != "PASS_SOURCE_PRICE_CHECK" or not audit.get("source_price_audit_passed"):
        raise ValueError("Source-price audit did not PASS; S05 training is blocked.")
    u = pd.read_csv(a.universe)
    check_cols(u, {"Ticker", "Asset Class", "Source"}, "Universe")
    u = u.loc[u.Source.astype(str).str.strip().str.casefold().eq("yahoo finance"), ["Ticker", "Asset Class"]].copy()
    u.Ticker = u.Ticker.astype(str).str.strip()
    if u.Ticker.duplicated().any() or u["Asset Class"].isna().any():
        raise ValueError("Universe has duplicate tickers or missing asset classes")
    d = pd.read_csv(a.features, low_memory=False)
    check_cols(d, {"Date", "Ticker", "Target_60d", "Forward_Return_60d"}, "Features")
    if d.columns.duplicated().any(): raise ValueError("Duplicate feature column names")
    d.Date = pd.to_datetime(d.Date, errors="raise").dt.normalize()
    d.Ticker = d.Ticker.astype(str).str.strip()
    if "Asset Class" in d.columns:
        d=d.rename(columns={"Asset Class":"Panel_Asset_Class"})
    d = d.loc[d.Date.ge(TRAIN_START)].merge(u, on="Ticker", how="inner", validate="many_to_one")
    if "Panel_Asset_Class" in d.columns:
        if not d.Panel_Asset_Class.eq(d["Asset Class"]).all():
            raise ValueError("Asset class differs between validated panel and investable universe")
        d=d.drop(columns=["Panel_Asset_Class"])
    if d.empty or d.duplicated(["Date", "Ticker"]).any(): raise ValueError("Empty data or duplicate ticker-date rows")
    if "Market_Regime" not in d:
        r = pd.read_csv(a.regimes, usecols=["Date", "Market_Regime"])
        r.Date = pd.to_datetime(r.Date, errors="raise").dt.normalize()
        r.Market_Regime = r.Market_Regime.astype(str).str.strip()
        if r.Date.duplicated().any(): raise ValueError("Regime dates are duplicated")
        d = pd.merge_asof(d.sort_values("Date"), r.sort_values("Date"), on="Date", direction="backward", tolerance=pd.Timedelta(days=4))
    if d.Market_Regime.isna().any(): raise ValueError("Market regime missing on one or more dates")
    allowed = {"Risk-On", "Risk-Off", "Transition"}
    if not set(d.Market_Regime.unique()).issubset(allowed): raise ValueError("Unexpected regime labels")
    if not a.label_lookup.exists(): raise FileNotFoundError(f"Audited label lookup not found: {a.label_lookup}")
    lab = pd.read_csv(a.label_lookup, low_memory=False)
    check_cols(lab, {"Date", "Ticker", "Label_End_Date"}, "Label lookup")
    lab.Date = pd.to_datetime(lab.Date, errors="raise").dt.normalize()
    lab.Label_End_Date = pd.to_datetime(lab.Label_End_Date, errors="coerce").dt.normalize()
    if lab.duplicated(["Date", "Ticker"]).any(): raise ValueError("Duplicate label lookup keys")
    if "Label_End_Date" in d.columns:
        d = d.rename(columns={"Label_End_Date": "Panel_Label_End_Date"})
    d = d.merge(lab[["Date", "Ticker", "Label_End_Date"]], on=["Date", "Ticker"], how="left", validate="one_to_one")
    if "Panel_Label_End_Date" in d.columns:
        panel_end = pd.to_datetime(d.Panel_Label_End_Date, errors="coerce").dt.normalize()
        if not (panel_end.fillna(pd.Timestamp("1900-01-01")) == d.Label_End_Date.fillna(pd.Timestamp("1900-01-01"))).all():
            raise ValueError("Maturity dates disagree between validated panel and audit lookup")
        d = d.drop(columns=["Panel_Label_End_Date"])
    # Exact maturity comes exclusively from the independently reconstructed
    # price audit. No inference/fallback to the old pre-2023 lookup.
    if d.Label_End_Date.isna().all():
        raise ValueError("The audit has no mature labels")
    # A valid audited label must have an observed target and return.
    mature=d.Label_End_Date.notna()
    if d.loc[mature,['Target_60d','Forward_Return_60d']].isna().any().any():
        raise ValueError("Audited mature labels are missing targets or forward returns")
    audited_fields={"Audit_Target_60d", "Audit_Forward_Return_60d"}
    if audited_fields.issubset(d.columns):
        # audit columns are duplicated outcome information, never features
        pass
    d.Target_60d = pd.to_numeric(d.Target_60d, errors="coerce")
    d.Forward_Return_60d = pd.to_numeric(d.Forward_Return_60d, errors="coerce")
    mature = d.Label_End_Date.notna()
    valid = mature & d.Target_60d.notna() & d.Forward_Return_60d.notna()
    if not d.loc[valid, "Target_60d"].isin([0,1]).all(): raise ValueError("Non-binary mature target")
    expected = (d.loc[valid, "Forward_Return_60d"] > 0).astype(int)
    if not np.array_equal(d.loc[valid, "Target_60d"].astype(int).to_numpy(), expected.to_numpy()):
        raise ValueError("Target_60d inconsistent with forward return sign")
    # A feature column containing obvious target/future information is rejected.
    excluded = {"Date","Ticker","Asset Class","Source","Market_Regime","Label_End_Date",
                "Target_60d","Forward_Return_60d","Audit_Target_60d","Audit_Forward_Return_60d",
                "Category","Instrument","Currency","MA_20d","MA_50d","MA_200d"}
    suspicious = [c for c in d if c not in excluded and any(k in c.lower() for k in ("future","lookahead","next_","label_","exit_","forward_return","target_","audit_"))]
    if suspicious: raise ValueError(f"Potential leakage columns: {suspicious}")
    numeric = [c for c in d.select_dtypes(include=[np.number]).columns if c not in excluded]
    # Upstream raw price and volume columns are not part of the selected
    # technical/macro feature set; avoid unit-scale and universe artefacts.
    numeric = [c for c in numeric if c not in {"Open", "High", "Low", "Close", "Adj Close", "Volume"}]
    # Exclude the fixed correlated-feature candidates from model inputs only.
    matched = sorted(set(numeric) & CORRELATION_EXCLUSION_CANDIDATES)
    unmatched = sorted(CORRELATION_EXCLUSION_CANDIDATES - set(numeric))
    # The MA_20d and MA_50d columns were already excluded in the original S05.
    # Other missing candidates indicate a schema mismatch and should not pass silently.
    unexpected = sorted(set(unmatched) - {"MA_20d", "MA_50d"})
    if unexpected:
        raise ValueError(f"Missing expected correlation exclusion candidates: {unexpected}")
    numeric = [c for c in numeric if c not in CORRELATION_EXCLUSION_CANDIDATES]
    if not numeric: raise ValueError("No numeric predictors found after correlation exclusions")
    a.correlation_excluded_features = matched
    a.correlation_already_excluded_features = sorted(set(unmatched) & {"MA_20d", "MA_50d"})
    LOG.info("Correlation filter: %d candidates, %d excluded now, %d already excluded; %d numerical inputs remain",
             len(CORRELATION_EXCLUSION_CANDIDATES), len(matched), len(a.correlation_already_excluded_features), len(numeric))
    for c in numeric: d[c] = pd.to_numeric(d[c], errors="coerce").replace([np.inf,-np.inf],np.nan).astype("float32")
    return d.sort_values(["Date","Ticker"]).reset_index(drop=True), numeric


def add_point_in_time_frequency(frame):
    """For row (i,t), use only same-ticker labels with Label_End_Date < t.

    The calculation is deliberately row-wise by ticker and chronological date.
    A label whose end date equals t is not considered known at the start of t.
    """
    out = frame.copy()
    out[HIST_COL] = np.nan
    for ticker, ix in out.groupby("Ticker", sort=False).groups.items():
        g = out.loc[ix].sort_values("Date")
        # Sort matured outcomes by their availability date, not their origin date.
        known = g.loc[g.Label_End_Date.notna() & g.Target_60d.notna(), ["Label_End_Date","Target_60d"]].sort_values("Label_End_Date")
        end_dates = known.Label_End_Date.to_numpy(dtype="datetime64[ns]")
        y = known.Target_60d.to_numpy(dtype=float)
        cumulative_pos = np.concatenate(([0.0], np.cumsum(y)))
        counts = np.searchsorted(end_dates, g.Date.to_numpy(dtype="datetime64[ns]"), side="left")
        pos = cumulative_pos[counts]
        out.loc[g.index, HIST_COL] = (pos + 1.0) / (counts + 2.0)
    out[HIST_COL] = out[HIST_COL].fillna(0.5).astype("float32")
    return out


def make_transform(numeric, scale, include_categories=True):
    num_steps = [("imputer", SimpleImputer(strategy="median", keep_empty_features=True))]
    if scale: num_steps.append(("scaler", StandardScaler()))
    transformers = [("num", Pipeline(num_steps), numeric)]
    if include_categories:
        transformers.extend([
            ("ticker", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["Ticker"]),
            ("class", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["Asset Class"]),
            ("regime", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["Market_Regime"]),
        ])
    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=0)


def score(y, p, threshold=THRESHOLD):
    y = np.asarray(y, dtype=int); p = np.asarray(p, dtype=float)
    if len(y)==0 or not np.isin(y,[0,1]).all() or not np.isfinite(p).all(): raise ValueError("Invalid metric inputs")
    yh = (p >= threshold).astype(int); both = len(np.unique(y)) == 2
    tn,fp,fn,tp = confusion_matrix(y,yh,labels=[0,1]).ravel()
    return {"N":len(y),"Positive_Rate":float(y.mean()),
        "ROC_AUC":float(roc_auc_score(y,p)) if both else np.nan,
        "MCC":float(matthews_corrcoef(y,yh)),
        "Balanced_Accuracy":float(balanced_accuracy_score(y,yh)) if both else np.nan,
        "PR_AUC":float(average_precision_score(y,p)) if both else np.nan,
        "Log_Loss":float(log_loss(y,np.clip(p,1e-15,1-1e-15),labels=[0,1])),
        "F1":float(f1_score(y,yh,zero_division=0)),
        "Precision":float(precision_score(y,yh,zero_division=0)),
        "Recall":float(recall_score(y,yh,zero_division=0)),
        "Accuracy":float(accuracy_score(y,yh)),"TP":int(tp),"FP":int(fp),"TN":int(tn),"FN":int(fn)}


def metric_tuple(m):
    return tuple((-m[k] if k=="Log_Loss" else m[k]) if pd.notna(m[k]) else -np.inf for k in METRICS)


def schedule(dates, start):
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(pd.Series(dates).dropna().unique())))
    return list(dates[dates >= pd.Timestamp(start)][::STEP])


def eligible_history(history, date):
    return history.loc[(history.Date < date) & (history.Label_End_Date < date) & history.Target_60d.notna()].copy()


def normalize_params(name, params):
    params=dict(params)
    if name == "MLP" and "architecture" in params:
        arch=params.pop("architecture")
        params["hidden_layer_sizes"]={"32":(32,),"64":(64,),"128":(128,),
                                      "64x32":(64,32),"128x64":(128,64),"128x64x32":(128,64,32)}[arch]
        params["solver"]="adam"
    return params

def make_estimator(name, params, seed, workers):
    params=normalize_params(name,params)
    if name == "LogisticRegression": return LogisticRegression(max_iter=1000, solver="lbfgs", random_state=seed, **params)
    if name == "RandomForest": return RandomForestClassifier(random_state=seed, n_jobs=workers, **params)
    if name == "XGBoost":
        from xgboost import XGBClassifier
        return XGBClassifier(objective="binary:logistic", eval_metric="logloss", tree_method="hist", n_jobs=workers, random_state=seed, **params)
    if name == "MLP": return MLPClassifier(random_state=seed, early_stopping=True, n_iter_no_change=15, max_iter=300, **params)
    raise ValueError(name)


def suggest_params(trial, name):
    if name == "LogisticRegression":
        return {"C":trial.suggest_float("C",1e-4,100,log=True),"class_weight":trial.suggest_categorical("class_weight",[None,"balanced"])}
    if name == "RandomForest":
        # Bounded search space: still meaningful, but avoids hundreds of deep,
        # oversized forests during each walk-forward tuning trial.
        return {"n_estimators":trial.suggest_int("n_estimators",100,300,step=50),"max_depth":trial.suggest_int("max_depth",5,15),"min_samples_split":trial.suggest_int("min_samples_split",10,100,step=10),"min_samples_leaf":trial.suggest_int("min_samples_leaf",5,80,step=5),"max_features":trial.suggest_categorical("max_features",["sqrt",0.3,0.5,0.7]),"max_samples":trial.suggest_float("max_samples",0.5,0.9),"class_weight":trial.suggest_categorical("class_weight",[None,"balanced","balanced_subsample"])}
    if name == "XGBoost":
        return {"n_estimators":trial.suggest_int("n_estimators",150,900,step=50),"max_depth":trial.suggest_int("max_depth",2,8),"learning_rate":trial.suggest_float("learning_rate",0.01,0.15,log=True),"min_child_weight":trial.suggest_float("min_child_weight",1,30,log=True),"subsample":trial.suggest_float("subsample",0.5,1.0),"colsample_bytree":trial.suggest_float("colsample_bytree",0.5,1.0),"gamma":trial.suggest_float("gamma",0,5),"reg_lambda":trial.suggest_float("reg_lambda",0.1,50,log=True),"reg_alpha":trial.suggest_float("reg_alpha",1e-5,10,log=True)}
    if name == "MLP":
        layer_id = trial.suggest_categorical("architecture",["32", "64", "128", "64x32", "128x64", "128x64x32"])
        layers = {"32":(32,),"64":(64,),"128":(128,),"64x32":(64,32),"128x64":(128,64),"128x64x32":(128,64,32)}[layer_id]
        return {"hidden_layer_sizes":layers,"activation":trial.suggest_categorical("activation",["relu","tanh"]),"alpha":trial.suggest_float("alpha",1e-5,1.0,log=True),"learning_rate_init":trial.suggest_float("learning_rate_init",1e-4,1e-2,log=True),"batch_size":trial.suggest_categorical("batch_size",[64,128,256]),"solver":"adam"}
    raise ValueError(name)


def build_fold_data(history, date, numeric, include_hist=True, include_categories=True):
    fit = eligible_history(history, date)
    pred = history.loc[history.Date.eq(date)].copy()
    if fit.empty or fit.Target_60d.nunique()<2: return None
    pred = pred.loc[pred.Ticker.isin(fit.Ticker.unique())]
    if pred.empty: return None
    cols = list(numeric)
    if not include_hist and HIST_COL in cols: cols.remove(HIST_COL)
    categorical = ["Ticker","Asset Class","Market_Regime"] if include_categories else []
    tr_cols = cols + categorical
    transformer = make_transform(cols, scale=True, include_categories=include_categories)
    x = transformer.fit_transform(fit[tr_cols]).astype("float32",copy=False)
    z = transformer.transform(pred[tr_cols]).astype("float32",copy=False)
    return fit,pred,x,fit.Target_60d.astype(int).to_numpy(),z


def predict_date(name, params, history, date, numeric, a, include_hist=True, include_categories=True):
    if name == "HistoricalFrequency":
        fit = eligible_history(history, date)
        pred = history.loc[history.Date.eq(date)].copy()
        if fit.empty or pred.empty: return None
        pred = pred.loc[pred.Ticker.isin(fit.Ticker.unique())]
        if pred.empty: return None
        hist = fit.groupby("Ticker").Target_60d.agg(["sum","count"])
        probs = ((hist["sum"]+1)/(hist["count"]+2)).to_dict()
        prior = (fit.Target_60d.sum()+1)/(len(fit)+2)
        p = pred.Ticker.map(probs).fillna(prior).to_numpy(float)
    else:
        fold = build_fold_data(history,date,numeric,include_hist,include_categories)
        if fold is None: return None
        fit,pred,x,y,z = fold
        model = make_estimator(name,params,a.seed,a.workers)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(x,y)
        p = model.predict_proba(z)[:,1]
    out = pred[["Date","Ticker","Asset Class","Target_60d","Forward_Return_60d","Label_End_Date"]].copy()
    out["P_Positive_60d"] = np.clip(p,0,1)
    out["Model"] = name
    out["Fit_Rows"] = len(fit)
    out["Last_Known_Label_End"] = fit.Label_End_Date.max()
    return out


def walk_predict(name, params, history, dates, numeric, a, phase, include_hist=True, include_categories=True):
    blocks=[]
    for n,date in enumerate(dates,1):
        b=predict_date(name,params,history,date,numeric,a,include_hist,include_categories)
        if b is not None:
            b["Phase"]=phase; b["Rebalance_Date"]=date
            blocks.append(b)
        if n%10==0: LOG.info("%s %s: %d/%d rebalance dates",name,phase,n,len(dates))
    return pd.concat(blocks,ignore_index=True) if blocks else pd.DataFrame()


def mature_metrics(pred):
    known=pred.loc[pred.Label_End_Date.notna() & pred.Target_60d.notna()]
    return score(known.Target_60d,known.P_Positive_60d) if len(known) else {k:np.nan for k in [*METRICS,"Precision","Recall"]}|{"N":0}


@dataclass
class FoldArrays:
    """Paths + small audit metadata. Large arrays live on disk (memory-mapped)."""
    date: pd.Timestamp
    x_path: Path
    y_path: Path
    z_path: Path
    prediction_rows: pd.DataFrame
    n_fit: int
    last_label_end: pd.Timestamp


class FoldCache:
    """Prepare each chronological fold once; reuse arrays across Optuna trials.

    Disk-backed float32 arrays avoid retaining 27 increasingly large training
    matrices in RAM. Storage is temporary and is removed after a successful
    experiment by default. Cache keys incorporate phase + feature definition.
    """
    def __init__(self, folder):
        self.folder=Path(folder)
        self.folder.mkdir(parents=True,exist_ok=True)
        self.index={}
        self.ready={}

    @staticmethod
    def key(phase, date, cols, categories):
        payload=json.dumps([phase,str(pd.Timestamp(date).date()),cols,bool(categories)],sort_keys=True)
        return hashlib.sha256(payload.encode('utf-8')).hexdigest()[:20]

    def build(self, history, dates, numeric, phase, include_categories=True):
        keytuple=(phase,tuple(numeric),bool(include_categories))
        if keytuple in self.ready:
            return self.ready[keytuple]
        LOG.info('Preparing %s preprocessing cache: %d dates; feature count=%d',phase,len(dates),len(numeric))
        folds=[]
        for j,date in enumerate(dates,1):
            fit=eligible_history(history,date)
            pred=history.loc[history.Date.eq(date)]
            if fit.empty or fit.Target_60d.nunique()<2 or pred.empty:
                continue
            pred=pred.loc[pred.Ticker.isin(fit.Ticker.unique())].copy()
            if pred.empty:
                continue
            cat_cols=['Ticker','Asset Class','Market_Regime'] if include_categories else []
            cols=list(numeric)+cat_cols
            transform=make_transform(list(numeric),scale=True,include_categories=include_categories)
            x=transform.fit_transform(fit[cols]).astype(np.float32,copy=False)
            z=transform.transform(pred[cols]).astype(np.float32,copy=False)
            y=fit.Target_60d.to_numpy(dtype=np.int8)
            key=self.key(phase,date,list(numeric),include_categories)
            xp,yp,zp=[self.folder/(key+'_'+suffix+'.npy') for suffix in ('x','y','z')]
            np.save(xp,x,allow_pickle=False)
            np.save(yp,y,allow_pickle=False)
            np.save(zp,z,allow_pickle=False)
            small=pred[['Date','Ticker','Asset Class','Target_60d','Forward_Return_60d','Label_End_Date']].copy()
            folds.append(FoldArrays(pd.Timestamp(date),xp,yp,zp,small,len(fit),fit.Label_End_Date.max()))
            del x,z,y,fit,transform
            if j%5==0: LOG.info('Prepared %s folds: %d/%d',phase,j,len(dates))
        if not folds: raise ValueError('No eligible folds for '+phase)
        self.ready[keytuple]=folds
        LOG.info('Prepared %s: %d folds; matrices cached on disk (%s)',phase,len(folds),self.folder)
        return folds


def fit_cached_fold(name,params,fold,a):
    """Train only the estimator, not its preprocessing."""
    x=np.load(fold.x_path,mmap_mode='r')
    y=np.load(fold.y_path,mmap_mode='r')
    z=np.load(fold.z_path,mmap_mode='r')
    model=make_estimator(name,params,a.seed,a.workers)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model.fit(x,y)
    p=np.asarray(model.predict_proba(z)[:,1],dtype=float)
    block=fold.prediction_rows.copy()
    block['P_Positive_60d']=np.clip(p,0,1)
    block['Model']=name
    block['Fit_Rows']=fold.n_fit
    block['Last_Known_Label_End']=fold.last_label_end
    block['Phase']='Validation' if 'Validation' in str(fold.x_path) else 'Test'
    block['Rebalance_Date']=fold.date
    del x,y,z,model
    return block


def predict_cached_folds(name,params,folds,a,phase,report=None):
    pieces=[]
    for step,fold in enumerate(folds,1):
        part=fit_cached_fold(name,params,fold,a)
        part['Phase']=phase
        pieces.append(part)
        if report is not None:
            report(step,pieces)
        if step%10==0: LOG.info('%s %s: %d/%d rebalances',name,phase,step,len(folds))
    return pd.concat(pieces,ignore_index=True)


def select_tuning_dates(full_dates, n_dates):
    """Choose evenly spaced dates across the complete validation period."""
    dates = list(full_dates)
    if n_dates >= len(dates):
        return dates
    if n_dates < 3:
        raise ValueError("--tuning-folds must be at least 3 to cover validation regimes")
    positions = np.linspace(0, len(dates) - 1, num=n_dates, dtype=int)
    selected = [dates[i] for i in positions]
    # Guard against duplicate positions for unusual inputs.
    return list(dict.fromkeys(selected))


def optuna_tune(name, history, dates, numeric, a, cache):
    """Tune on a small chronological subset; evaluate the winner on all validation dates.

    The Optuna objective uses `tuning_dates` only. After selecting the best
    parameters, the winning configuration is fitted once per each of the full
    validation dates. These full-validation predictions—not the tuning subset—
    drive screening and model ranking.
    """
    tuning_dates = select_tuning_dates(dates, min(a.tuning_folds, len(dates)))
    LOG.info("%s tuning uses %d/%d chronological validation dates: %s", name,
             len(tuning_dates), len(dates), [str(pd.Timestamp(x).date()) for x in tuning_dates])
    tuning_folds = cache.build(history, tuning_dates, numeric, 'Tuning', include_categories=True)
    sampler=optuna.samplers.TPESampler(n_startup_trials=min(a.startup_trials,a.trials-1),n_ei_candidates=24,multivariate=True,seed=a.seed)
    pruner=optuna.pruners.MedianPruner(n_startup_trials=min(a.startup_trials,a.trials-1),n_warmup_steps=2,interval_steps=1)
    study_name=f'S05_subsampled_tuning_v2_{name}'
    study=optuna.create_study(study_name=study_name,storage=f"sqlite:///{(a.out/'optuna_studies.db').resolve()}",load_if_exists=True,direction='maximize',sampler=sampler,pruner=pruner)
    source_signature = {
        'implementation':'subsampled_tuning_v2', 'model':name,
        'features':str(a.features.resolve()), 'features_size':a.features.stat().st_size,
        'features_mtime_ns':a.features.stat().st_mtime_ns,
        'label_lookup':str(a.label_lookup.resolve()), 'label_lookup_size':a.label_lookup.stat().st_size,
        'label_lookup_mtime_ns':a.label_lookup.stat().st_mtime_ns,
        'dates':[str(pd.Timestamp(x).date()) for x in dates],
        'tuning_dates':[str(pd.Timestamp(x).date()) for x in tuning_dates],
        'numeric':numeric, 'seed':a.seed, 'startup_trials':a.startup_trials,
        'tuning_folds':a.tuning_folds, 'trials':a.trials,
    }
    fingerprint=hashlib.sha256(json.dumps(source_signature,sort_keys=True).encode('utf-8')).hexdigest()
    previous=study.user_attrs.get('experiment_fingerprint')
    if previous is not None and previous != fingerprint:
        raise ValueError(f'Existing Optuna study {study_name} has incompatible inputs/settings. Use a fresh --out directory.')
    if previous is None and len(study.trials):
        raise ValueError(f'Existing Optuna study {study_name} has no experiment fingerprint. Use a fresh --out directory.')
    study.set_user_attr('experiment_fingerprint',fingerprint)
    study.set_user_attr('tuning_dates',[str(pd.Timestamp(x).date()) for x in tuning_dates])
    study.set_user_attr('full_validation_dates',[str(pd.Timestamp(x).date()) for x in dates])
    pred_dir=a.out/'optuna_best_trial_predictions'; pred_dir.mkdir(exist_ok=True)

    def objective(trial):
        params=suggest_params(trial,name)
        pieces=[]
        for step,fold in enumerate(tuning_folds,1):
            pieces.append(fit_cached_fold(name,params,fold,a))
            if step >= 2:
                partial=pd.concat(pieces,ignore_index=True)
                m=mature_metrics(partial)
                auc=m.get('ROC_AUC',np.nan)
                if np.isfinite(auc):
                    trial.report(float(auc),step=step)
                    if trial.should_prune():
                        raise optuna.TrialPruned(f'Partial tuning ROC-AUC={auc:.4f} after {step} dates')
        preds=pd.concat(pieces,ignore_index=True)
        m=mature_metrics(preds)
        if not np.isfinite(m.get('ROC_AUC',np.nan)):
            raise optuna.TrialPruned('No valid pooled tuning ROC-AUC')
        for k,v in m.items():
            if isinstance(v,(int,float,np.integer,np.floating)) and np.isfinite(v): trial.set_user_attr(k,float(v))
        return float(m['ROC_AUC'])

    remaining=max(0,a.trials-len(study.trials))
    if remaining:
        study.optimize(objective,n_trials=remaining,n_jobs=1,show_progress_bar=False,gc_after_trial=False)
    complete=[t for t in study.trials if t.state==optuna.trial.TrialState.COMPLETE]
    if not complete: raise RuntimeError(f'No completed Optuna trials for {name}')
    best=max(complete,key=lambda t:t.value)
    LOG.info("%s best tuning ROC-AUC=%.6f; now evaluating winning parameters on ALL %d validation dates",name,best.value,len(dates))
    full_validation_folds=cache.build(history,dates,numeric,'Validation',include_categories=True)
    best_preds=predict_cached_folds(name,best.params,full_validation_folds,a,'Validation')
    save(best_preds,pred_dir/f'{name}_full_validation_best.csv')
    trials=[]
    for t in study.trials:
        trials.append({'Model':name,'Trial':t.number,'State':t.state.name,'Tuning_ROC_AUC':t.value,
                       'Parameters':json.dumps(t.params,default=str),**t.user_attrs})
    return best.params,best_preds,pd.DataFrame(trials)


def select_assets(pred, rule, value):
    z=pred.copy(); z["Selected"]=False
    for _,ix in z.groupby("Rebalance_Date").groups.items():
        g=z.loc[ix]
        chosen=(g.sort_values(["P_Positive_60d","Ticker"],ascending=[False,True]).head(int(value)).index if rule=="TopK" else g.index[g.P_Positive_60d.ge(float(value))])
        z.loc[chosen,"Selected"]=True
    return z


def screen_stat(pred,rule,value):
    z=select_assets(pred,rule,value)
    k=z.loc[z.Label_End_Date.notna() & z.Target_60d.notna()]
    s=k.loc[k.Selected]
    return {"Rule":rule,"Value":value,"N_Selected":len(s),"Mean_Selected_Per_Rebalance":len(s)/k.Rebalance_Date.nunique() if k.Rebalance_Date.nunique() else np.nan,"Selection_Precision":float(s.Target_60d.mean()) if len(s) else np.nan,"Unconditional_Positive_Rate":float(k.Target_60d.mean()) if len(k) else np.nan,"Precision_Lift":float(s.Target_60d.mean()-k.Target_60d.mean()) if len(s) and len(k) else np.nan,"Selected_Forward_Return_Mean":float(s.Forward_Return_60d.mean()) if len(s) else np.nan}


def choose_screen(pred,a):
    candidates=pd.DataFrame([screen_stat(pred,r,v) for r,vals in (("TopK",a.top_k),("Threshold",a.screen_thresholds)) for v in vals])
    eligible=candidates.loc[(candidates.N_Selected>0)&(candidates.Mean_Selected_Per_Rebalance>=a.screen_min_average)]
    if eligible.empty: raise ValueError("No validation screening rule meets minimum average selected assets")
    chosen=eligible.sort_values(["Precision_Lift","N_Selected"],ascending=False).iloc[0]
    return candidates, {"rule":chosen.Rule,"value":float(chosen.Value)}


def save(frame,path):
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+".tmp"); frame.to_csv(tmp,index=False); tmp.replace(path)


def run(a):
    if a.trials<2 or a.startup_trials<1 or a.startup_trials>=a.trials: raise ValueError("Require trials >= 2 and startup-trials between 1 and trials-1")
    if a.tuning_folds < 3: raise ValueError("--tuning-folds must be at least 3")
    a.out.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format="%(asctime)s | %(levelname)s | %(message)s",handlers=[logging.StreamHandler(),logging.FileHandler(a.out/"run.log",encoding="utf-8")],force=True)
    d,numeric=load_data(a)
    d=add_point_in_time_frequency(d)
    numeric=numeric+[HIST_COL]
    if HIST_COL in CORRELATION_EXCLUSION_CANDIDATES:
        raise AssertionError("Historical frequency must never be excluded")
    # Prediction dates use the common dates in the panel, not ticker-specific clocks.
    all_dates=sorted(d.Date.unique())
    val_dates=[x for x in schedule(all_dates,VALID_START) if x<TEST_START]
    test_dates=[x for x in schedule(all_dates,TEST_START)]
    if not val_dates or not test_dates: raise ValueError("No validation or test rebalance dates; inspect data coverage")
    # Validation history excludes test-origin observations. The expanding fit
    # naturally includes only labels whose end date is before each prediction.
    pre=d.loc[d.Date<TEST_START].copy()
    test_history=d.copy()
    params_by_model={}; val_frames=[]; test_frames=[]; trial_frames=[]; screen_rows=[]; ranking_rows=[]
    cache_dir=a.out/'_temporary_fold_cache'
    if cache_dir.exists():
        shutil.rmtree(cache_dir)  # never reuse stale preprocessing from another data run
    fold_cache=FoldCache(cache_dir)
    for name in MODELS:
        LOG.info("Tuning %s: up to %d trials",name,a.trials)
        best_params,val_pred,trials=optuna_tune(name,pre,val_dates,numeric,a,fold_cache)
        params_by_model[name]=best_params; trial_frames.append(trials); val_frames.append(val_pred)
        candidates,screen=choose_screen(val_pred,a); candidates.insert(0,"Model",name); screen_rows.append(candidates)
        LOG.info("%s selected validation screening: %s",name,screen)
        # Final test: fixed parameters/screen, but use only matured labels available at each date.
        test_folds=fold_cache.build(test_history,test_dates,numeric,'Test',include_categories=True)
        test_pred=predict_cached_folds(name,best_params,test_folds,a,'Test')
        if not test_pred.empty:
            test_pred=select_assets(test_pred,screen["rule"],screen["value"])
            test_pred["Screen_Rule"]=screen["rule"]; test_pred["Screen_Value"]=screen["value"]
            test_frames.append(test_pred)
        val_pred=select_assets(val_pred,screen["rule"],screen["value"])
        val_pred["Screen_Rule"]=screen["rule"]; val_pred["Screen_Value"]=screen["value"]
        val_frames[-1]=val_pred
        ranking_rows.append({"Model":name,"Phase":"Validation",**mature_metrics(val_pred)})
        if not test_pred.empty: ranking_rows.append({"Model":name,"Phase":"Test",**mature_metrics(test_pred)})
        save(pd.concat(trial_frames,ignore_index=True),a.out/"hyperparameter_trials.csv")
        save(pd.concat(val_frames,ignore_index=True),a.out/"predictions_validation.csv")
        if test_frames: save(pd.concat(test_frames,ignore_index=True),a.out/"predictions_test.csv")
    # Historical-frequency benchmark: direct ticker-specific smoothed frequency,
    # calculated only from labels matured at each rebalance date.
    hf_val = walk_predict("HistoricalFrequency", {}, pre, val_dates, numeric, a, "Validation")
    hf_candidates, hf_screen = choose_screen(hf_val, a)
    hf_candidates.insert(0, "Model", "HistoricalFrequency")
    screen_rows.append(hf_candidates)
    hf_val = select_assets(hf_val, hf_screen["rule"], hf_screen["value"])
    hf_val["Screen_Rule"] = hf_screen["rule"]; hf_val["Screen_Value"] = hf_screen["value"]
    val_frames.append(hf_val)
    hf_test = walk_predict("HistoricalFrequency", {}, test_history, test_dates, numeric, a, "Test")
    if not hf_test.empty:
        hf_test = select_assets(hf_test, hf_screen["rule"], hf_screen["value"])
        hf_test["Screen_Rule"] = hf_screen["rule"]; hf_test["Screen_Value"] = hf_screen["value"]
        test_frames.append(hf_test)
    ranking_rows.append({"Model":"HistoricalFrequency","Phase":"Validation",**mature_metrics(hf_val)})
    if not hf_test.empty:
        ranking_rows.append({"Model":"HistoricalFrequency","Phase":"Test",**mature_metrics(hf_test)})
    val=pd.concat(val_frames,ignore_index=True); test=pd.concat(test_frames,ignore_index=True) if test_frames else pd.DataFrame()
    metric_frame=pd.DataFrame(ranking_rows)
    save(metric_frame,a.out/"model_metrics.csv")
    for phase in ("Validation", "Test"):
        part=metric_frame.loc[metric_frame.Phase.eq(phase)].copy()
        if not part.empty:
            # Agreed lexicographic ordering; precision is reported but does not
            # displace the prespecified ranking hierarchy.
            part=part.sort_values(list(METRICS),ascending=[False,False,False,False,True,False],na_position="last",kind="stable").reset_index(drop=True)
            part.insert(0,"Rank",np.arange(1,len(part)+1))
            save(part,a.out/f"model_ranking_{phase.lower()}.csv")
    save(pd.concat(screen_rows,ignore_index=True),a.out/"screening_validation_candidates.csv")
    screening_output=[]
    for frame in (val,test):
        if frame.empty: continue
        for model,g in frame.groupby('Model'):
            known=g.loc[g.Label_End_Date.notna() & g.Target_60d.notna()]
            chosen=known.loc[known.Selected]
            screening_output.append({"Phase":str(g.Phase.iloc[0]),"Model":model,
                "N_Selected":len(chosen),"Selection_Precision":float(chosen.Target_60d.mean()) if len(chosen) else np.nan,
                "Selection_Recall":float(chosen.Target_60d.sum()/known.Target_60d.sum()) if known.Target_60d.sum()>0 else np.nan,
                "Selection_Lift":float(chosen.Target_60d.mean()-known.Target_60d.mean()) if len(chosen) else np.nan,
                "Screen_Rule":str(g.Screen_Rule.iloc[0]),"Screen_Value":float(g.Screen_Value.iloc[0])})
    save(pd.DataFrame(screening_output),a.out/"screening_results.csv")
    (a.out/"selected_hyperparameters.json").write_text(json.dumps(params_by_model,indent=2,default=str),encoding="utf-8")
    # Ablation uses the selected model hyperparameters, same dates, and the
    # same fold transformations across all four models. The two additional
    # feature sets are prepared once each and evicted after use.
    if not a.skip_ablation:
        rows=[]
        for label,cols,categories in (
            ("A_original_features",[c for c in numeric if c!=HIST_COL],True),
            ("B_original_plus_history",numeric,True),
            ("C_history_only",[HIST_COL],False),
        ):
            phase_key='Ablation_'+label if label != 'B_original_plus_history' else 'Validation'
            folds=fold_cache.build(pre,val_dates,cols,phase_key,include_categories=categories)
            for name in MODELS:
                pp=predict_cached_folds(name,params_by_model[name],folds,a,"Validation_Ablation")
                rows.append({"Model":name,"Experiment":label,**mature_metrics(pp)})
                save(pd.DataFrame(rows),a.out/"feature_ablation.csv")
            if label != 'B_original_plus_history':
                # Keep storage bounded: A and C are not needed again.
                for fold in folds:
                    for path in (fold.x_path,fold.y_path,fold.z_path):
                        path.unlink(missing_ok=True)
                fold_cache.ready.pop((phase_key,tuple(cols),bool(categories)),None)
    # Grouped permutation importance on a validation-only rebalance date.
    # One-hot groups are shuffled together at their ORIGINAL column level.
    # This is descriptive, not a full-period or independent inference estimate.
    importance_rows=[]; coefficient_rows=[]; summary_rows=[]
    if not a.skip_importance:
        for name in MODELS:
            result=None
            for date in reversed(val_dates):
                fit=eligible_history(pre,date)
                pred=pre.loc[pre.Date.eq(date)]
                pred=pred.loc[pred.Ticker.isin(fit.Ticker.unique())]
                if len(pred)>=8 and pred.Target_60d.notna().all() and pred.Target_60d.nunique()==2 and fit.Target_60d.nunique()==2:
                    result=(fit,pred,date)
                    break
            if result is None:
                LOG.warning("No two-class validation date available for importance: %s",name)
                continue
            fit,pred,date=result
            inputs=numeric+["Ticker","Asset Class","Market_Regime"]
            trans=make_transform(numeric,scale=True)
            x=trans.fit_transform(fit[inputs]).astype("float32")
            model=make_estimator(name,params_by_model[name],a.seed,a.workers)
            model.fit(x,fit.Target_60d.astype(int).to_numpy())
            baseline=roc_auc_score(pred.Target_60d,model.predict_proba(trans.transform(pred[inputs]).astype('float32'))[:,1])
            rng=np.random.default_rng(a.seed)
            group_to_col={c:c for c in inputs}
            for group,col in group_to_col.items():
                impacts=[]
                for _ in range(a.permutation_repeats):
                    shuffled=pred[inputs].copy()
                    shuffled[col]=rng.permutation(shuffled[col].to_numpy())
                    psh=model.predict_proba(trans.transform(shuffled).astype('float32'))[:,1]
                    impacts.append(baseline-roc_auc_score(pred.Target_60d,psh))
                importance_rows.append({"Model":name,"Validation_Date":str(date.date()),"Feature":group,
                                        "Importance_Mean":float(np.mean(impacts)),"Importance_SD":float(np.std(impacts,ddof=0)),
                                        "Baseline_ROC_AUC":float(baseline)})
            encoded=list(trans.get_feature_names_out())
            if name=="LogisticRegression":
                for feature,coef in zip(encoded,model.coef_[0]):
                    coefficient_rows.append({"Model":name,"Transformed_Feature":feature,"Coefficient":float(coef),
                                             "Absolute_Coefficient":abs(float(coef))})
            if name in ("RandomForest","XGBoost"):
                native=getattr(model,"feature_importances_",None)
                if native is not None:
                    for feature,value in zip(encoded,native):
                        coefficient_rows.append({"Model":name,"Transformed_Feature":feature,
                                                 "Native_Importance":float(value),"Importance_Type":"impurity" if name=="RandomForest" else "gain"})
        if importance_rows:
            imp=pd.DataFrame(importance_rows)
            save(imp,a.out/"feature_importance_by_model.csv")
            for model,sub in imp.groupby('Model'):
                positive=sub.Importance_Mean.clip(lower=0)
                total=float(positive.sum())
                h=float(positive[sub.Feature.eq(HIST_COL)].sum())
                summary_rows.append({"Model":model,"Historical_Frequency_Importance_Share":h/total if total>0 else np.nan,
                                     "Top_Feature":str(sub.sort_values('Importance_Mean',ascending=False).iloc[0].Feature),
                                     "Positive_Importance_Total":total,"Validation_Date":str(sub.Validation_Date.iloc[0])})
            save(pd.DataFrame(summary_rows),a.out/"feature_importance_summary.csv")
        if coefficient_rows: save(pd.DataFrame(coefficient_rows),a.out/"feature_coefficients.csv")
    manifest={"experiment":"S05 Classification Model v2","train_start":str(TRAIN_START.date()),"validation_start":str(VALID_START.date()),"test_start":str(TEST_START.date()),"horizon_observations":HORIZON,"rebalance_sessions":STEP,"models":list(MODELS)+["HistoricalFrequency"],"optuna_trials_per_ml_model":a.trials,"maximum_ml_trials_total":a.trials*len(MODELS),"optuna_startup_trials":a.startup_trials,"optuna_sampler":"TPESampler","implementation":"subsampled_tuning_v2","preprocessing_cached_by_fold":True,"best_trial_predictions_reused":False,"optuna_tuning_validation_dates":a.tuning_folds,"full_validation_dates_used_for_final_model_evaluation":len(val_dates),"tuning_date_selection":"evenly spaced chronological dates across 2016-2021","random_forest_search_space":{"n_estimators":[100,300,50],"max_depth":[5,15],"max_samples":[0.5,0.9]},"optuna_objective":"pooled validation ROC-AUC","metric_reporting":[*METRICS,"Precision","Recall"],"historical_frequency_formula":"(positive matured ticker outcomes + 1)/(all matured ticker outcomes + 2)","screening_selected_on_validation_only":True,"test_hyperparameters_and_screening_frozen":True,"important_limitations":["Verify upstream macroeconomic feature publication lags and point-in-time availability.","Overlapping 60-observation forward labels induce dependence; ordinary row-level metrics are descriptive, not independent-sample inference.","Permutation importance is calculated on transformed columns for a representative final validation fold; one-hot feature interpretation requires care.","Source-price label audit is required and checked before model fitting; historical price-file convention must match S03."]}
    manifest["audit_report"]=str(a.audit_report)
    manifest["audit_status"]="PASS_SOURCE_PRICE_CHECK"
    manifest["feature_count"]=len(numeric)
    manifest["correlation_filter"]={
        "method":"fixed exclusion list from pre-2022 sampled absolute Pearson correlations",
        "threshold":CORRELATION_THRESHOLD,
        "number_of_candidates":len(CORRELATION_EXCLUSION_CANDIDATES),
        "excluded_from_model_inputs":a.correlation_excluded_features,
        "already_excluded_by_original_S05":a.correlation_already_excluded_features,
        "remaining_numeric_inputs_including_historical_frequency":len(numeric),
        "note":"Heuristic reduction; not proof of redundant predictive information."
    }
    save(pd.DataFrame({"Excluded_Feature":sorted(CORRELATION_EXCLUSION_CANDIDATES),
                       "Exclusion_Status":["already_excluded_by_S05" if c in a.correlation_already_excluded_features else "excluded_by_correlation_filter" for c in sorted(CORRELATION_EXCLUSION_CANDIDATES)]}),
         a.out/"correlation_excluded_features.csv")
    (a.out/"run_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    readme="""S05 Classification Model v2 outputs\n\ncorrelation_excluded_features.csv: fixed pre-2022 high-correlation exclusion candidates; all kept out of S05 model input.\n\nmodel_metrics.csv: validation and test classification metrics. Precision is TP/(TP+FP) at probability threshold 0.50.\npredictions_validation.csv / predictions_test.csv: ticker-date probabilities and validation-selected screening flags.\nhyperparameter_trials.csv: Optuna trial parameters, tuning-subset objective values and recorded metrics. Optuna tunes on six evenly spaced validation dates by default; winning parameters are then evaluated on all validation dates.\nselected_hyperparameters.json: selected parameters per ML model.\nscreening_validation_candidates.csv: validation-only screening candidates and precision lift.\nfeature_ablation.csv: original features, original plus historical frequency, and historical frequency only.\nfeature_importance_by_model.csv: permutation importance for transformed input columns on a representative validation fold.\nrun_manifest.json: experiment dates, settings and limitations.\n\nROC-AUC: ranking/discrimination. MCC: balanced correlation between labels and hard predictions. Balanced Accuracy: mean sensitivity and specificity. PR-AUC: average precision. Log Loss: probability calibration/quality (lower is better). F1: harmonic mean of precision and recall. Precision: fraction of predicted positives that are truly positive. Recall: fraction of actual positives identified.\n"""
    (a.out/"README.txt").write_text(readme,encoding="utf-8")
    if not a.keep_fold_cache:
        shutil.rmtree(cache_dir,ignore_errors=True)
        LOG.info('Removed temporary preprocessing cache')
    LOG.info("Completed S05 cached v2. Outputs: %s",a.out)
    return ranking_rows

if __name__=="__main__":
    run(args_parser())
