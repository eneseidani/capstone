"""Final S05 classification: Historical Frequency vs tuned Random Forest/XGBoost.

A standalone script (no import of previous S05/S06 modules, no Optuna).

- S05 independently audited features, labels, and existing correlation exclusions.
- Rebalance every 60 actual SPY trading sessions, anchored to the first SPY
  session on/after 2016-01-01, continuously across validation and test.
- 60 forward *asset observations* define the target, as in S03; this is not
  necessarily the same clock as the SPY rebalancing schedule.
- Expanding window with ONLY outcomes whose label end is strictly before the
  rebalance date and is verifiably within the available data endpoint.
- Tuned RF/XGBoost parameters are loaded without further search.
- Select the best method using validation mean Precision@15 ONLY; test results
  are descriptive because S05/S06 test results have already been inspected.

From D:/Capstone:
    python S05_final_classification.py

Requires existing successful S05_data_validation.py audit results and the
selected_hyperparameters.json from S05_optimized_v2.
"""
from __future__ import annotations
import argparse
import json
import logging
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, matthews_corrcoef,
    balanced_accuracy_score, average_precision_score, log_loss, f1_score,
    precision_score, recall_score, confusion_matrix, accuracy_score)

LOG = logging.getLogger("S05_final")

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

def eligible_history(history, date):
    return history.loc[(history.Date < date) & (history.Label_End_Date < date) & history.Target_60d.notna()].copy()

def make_estimator(name, params, seed, workers):
    if name == "RandomForest":
        return RandomForestClassifier(random_state=seed, n_jobs=workers, **params)
    if name == "XGBoost":
        from xgboost import XGBClassifier
        return XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                             tree_method="hist", n_jobs=workers, random_state=seed, **params)
    raise ValueError(f"Unsupported fitted model: {name}")


def save(frame,path):
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+".tmp"); frame.to_csv(tmp,index=False); tmp.replace(path)


MODELS = ("HistoricalFrequency", "RandomForest", "XGBoost")


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", type=Path, default=Path("model_output/S05_data_validation/combined_features_validated.csv"))
    p.add_argument("--universe", type=Path, default=Path("Investable_Universe.csv"))
    p.add_argument("--regimes", type=Path, default=Path("features_data/market_regimes.csv"))
    p.add_argument("--label-lookup", type=Path, default=Path("model_output/S05_data_validation/label_maturity_60d.csv"))
    p.add_argument("--audit-report", type=Path, default=Path("model_output/S05_data_validation/audit_report.json"))
    p.add_argument("--spy", type=Path, default=Path("input_data/yahoo/SPY.csv"))
    p.add_argument("--tuned-params", type=Path, default=Path("model_output/S05_optimized_v2/selected_hyperparameters.json"))
    p.add_argument("--out", type=Path, default=Path("model_output/S05_final_classification"))
    p.add_argument("--top-k", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=4)
    return p.parse_args()


def get_spy_rebalance_dates(spy_path, panel):
    """Actual SPY observations, never union-of-asset or synthetic weekdays."""
    if not spy_path.is_file():
        raise FileNotFoundError(f"SPY OHLCV data required to define market sessions: {spy_path}")
    spy = pd.read_csv(spy_path, usecols=["Date"])
    spy_dates = pd.DatetimeIndex(pd.to_datetime(spy["Date"], errors="raise").dt.normalize().dropna().unique()).sort_values()
    if spy_dates.has_duplicates:
        raise ValueError("Duplicate SPY price dates")
    # A SPY trading session without a feature-panel observation must not silently
    # vanish and shift all subsequent 60-session rebalances.
    anchor_index = spy_dates.searchsorted(VALID_START)
    if anchor_index >= len(spy_dates):
        raise ValueError("No SPY sessions on/after validation start")
    # Do not schedule dates beyond the current validated panel endpoint.
    spy_dates = spy_dates[spy_dates <= panel.Date.max()]
    all_rebalances = list(spy_dates[anchor_index::STEP])
    panel_spy = set(panel.loc[panel.Ticker.eq("SPY"), "Date"])
    missing = [d for d in all_rebalances if d not in panel_spy]
    if missing:
        raise ValueError(
            f"{len(missing)} scheduled SPY sessions are missing from the validated "
            f"feature panel. First missing: {missing[:4]}. Fix panel coverage; "
            "do not silently shift the rebalance schedule."
        )
    validation = [d for d in all_rebalances if VALID_START <= d < TEST_START]
    test = [d for d in all_rebalances if d >= TEST_START]
    if not validation or not test:
        raise ValueError("No SPY-based validation or test rebalances")
    return validation, test


def enforce_observed_maturity(panel):
    """Treat a label as observed only if its end is within price-panel coverage.

    The audit lookup may carry an end date after the current available sample.
    Also check each ticker's available last feature-panel date, conservatively.
    Target and forward return remain in output for auditing but never inputs.
    """
    d = panel.copy()
    global_end = d.Date.max()
    ticker_end = d.groupby("Ticker").Date.transform("max")
    valid = (d.Label_End_Date.notna() & (d.Label_End_Date <= global_end)
             & (d.Label_End_Date <= ticker_end)
             & d.Target_60d.notna() & d.Forward_Return_60d.notna())
    rejected = int((d.Label_End_Date.notna() & ~valid).sum())
    if rejected:
        LOG.warning("Invalidated %d non-observed/non-finite forward labels (coverage ends %s)", rejected, global_end.date())
    d.loc[~valid, ["Target_60d", "Forward_Return_60d", "Label_End_Date"]] = [np.nan, np.nan, pd.NaT]
    return d, global_end, rejected


def dates_and_predictions(name, params, history, rebalances, numeric, a, phase):
    blocks = []
    for n, date in enumerate(rebalances, 1):
        fit = eligible_history(history, date)
        pred = history.loc[history.Date.eq(date)].copy()
        pred = pred.loc[pred.Ticker.isin(fit.Ticker.unique())].copy()
        if fit.empty or pred.empty:
            LOG.warning("%s %s: no eligible prediction at %s", name, phase, date.date())
            continue
        if name == "HistoricalFrequency":
            prob = pd.to_numeric(
                pred["Ticker_Historical_Positive_Rate_60d"],
                errors="coerce"
            ).to_numpy(dtype=float)

            if not np.isfinite(prob).all():
                raise ValueError("Invalid Historical Frequency values.")
        else:
            if fit.Target_60d.nunique() < 2:
                LOG.warning("%s %s: single-class training data at %s", name, phase, date.date())
                continue
            features = numeric + ["Ticker", "Asset Class", "Market_Regime"]
            transform = make_transform(numeric, scale=True, include_categories=True)
            X_train = transform.fit_transform(fit[features]).astype("float32", copy=False)
            X_pred = transform.transform(pred[features]).astype("float32", copy=False)
            estimator = make_estimator(name, params, a.seed, a.workers)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                estimator.fit(X_train, fit.Target_60d.astype(int).to_numpy())
            prob = estimator.predict_proba(X_pred)[:, 1]
        result = pred[["Date", "Ticker", "Asset Class", "Target_60d",
                       "Forward_Return_60d", "Label_End_Date"]].copy()
        result["Rebalance_Date"] = date
        result["Model"] = name
        result["Phase"] = phase
        result["P_Positive_60d"] = np.asarray(prob, dtype=float)
        result["Fit_Rows"] = len(fit)
        result["Last_Known_Label_End"] = fit.Label_End_Date.max()
        result = result.sort_values(["P_Positive_60d", "Ticker"], ascending=[False, True])
        result["Rank"] = np.arange(1, len(result) + 1)
        # Never pretend a full Top-15 portfolio exists on partial asset dates.
        result["Selected_TopK"] = (len(result) >= a.top_k) & (result.Rank <= a.top_k)
        blocks.append(result)
        if n % 5 == 0 or n == len(rebalances):
            LOG.info("%s %s: %d/%d SPY rebalances", name, phase, n, len(rebalances))
    if not blocks:
        raise RuntimeError(f"No predictions for {name} {phase}")
    return pd.concat(blocks, ignore_index=True)


def per_rebalance(predictions, k):
    rows = []
    for (model, phase, dt), g in predictions.groupby(["Model", "Phase", "Rebalance_Date"]):
        matured = g.loc[g.Label_End_Date.notna() & g.Target_60d.notna()]
        row = {"Model": model, "Phase": phase, "Rebalance_Date": dt,
               "N_Predicted": len(g), "N_Matured": len(matured),
               "Full_Universe_Available": len(g) >= k}
        # Top-K diagnostics require every predicted asset's label to have matured:
        # otherwise rank/outcome comparisons are not a complete cross-section.
        if len(g) >= k and len(matured) == len(g):
            top = g.loc[g.Rank <= k]
            row[f"Precision_at_{k}"] = float(top.Target_60d.mean())
            n_pos = int(g.Target_60d.sum())
            row[f"Recall_at_{k}"] = float(top.Target_60d.sum() / n_pos) if n_pos else np.nan
            row["Selected_Forward_Return_Mean"] = float(top.Forward_Return_60d.mean())
            row["Unconditional_Positive_Rate"] = float(g.Target_60d.mean())
        else:
            row[f"Precision_at_{k}"] = np.nan
            row[f"Recall_at_{k}"] = np.nan
            row["Selected_Forward_Return_Mean"] = np.nan
            row["Unconditional_Positive_Rate"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def compare_models(preds, date_metrics, k):
    result = []
    for (phase, model), g in preds.groupby(["Phase", "Model"], sort=False):
        valid = g.loc[g.Target_60d.notna() & g.Label_End_Date.notna()]
        metrics = score(valid.Target_60d, valid.P_Positive_60d) if not valid.empty else {}
        daily = date_metrics.loc[(date_metrics.Phase == phase) & (date_metrics.Model == model)]
        result.append({"Phase": phase, "Model": model, **metrics,
                       f"Mean_Precision_at_{k}": daily[f"Precision_at_{k}"].mean(),
                       f"Mean_Recall_at_{k}": daily[f"Recall_at_{k}"].mean(),
                       "Mean_Selected_Forward_Return": daily.Selected_Forward_Return_Mean.mean(),
                       "N_Full_Matured_Rebalances": int(daily[f"Precision_at_{k}"].notna().sum()),
                       "N_Rebalances": len(daily)})
    return pd.DataFrame(result)


def main():
    a = arguments()
    if a.top_k < 1:
        raise ValueError("--top-k must be >= 1")
    a.out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(a.out / "run.log", encoding="utf-8")], force=True)
    tuned = json.loads(a.tuned_params.read_text(encoding="utf-8"))
    for name in ("RandomForest", "XGBoost"):
        if not isinstance(tuned.get(name), dict) or not tuned[name]:
            raise ValueError(f"S05 tuned hyperparameters missing for {name}: {a.tuned_params}")
    LOG.info("Reading independently audited S05 panel")
    d, numeric = load_data(a)
    d, observed_end, invalidated = enforce_observed_maturity(d)

    if HIST_COL not in d.columns:
        raise ValueError(
            f"{HIST_COL} missing from input features. "
            "Check that S02 and S03 have been rerun."
        )

    d[HIST_COL] = pd.to_numeric(d[HIST_COL], errors="coerce")

    numeric = list(dict.fromkeys(numeric + [HIST_COL]))
    val_dates, test_dates = get_spy_rebalance_dates(a.spy, d)
    LOG.info("SPY dates: validation=%d, test=%d; data cutoff=%s", len(val_dates), len(test_dates), observed_end.date())
    predictions = []
    for name in MODELS:
        params = {} if name == "HistoricalFrequency" else tuned[name]
        for phase, dates in (("Validation", val_dates), ("Test", test_dates)):
            LOG.info("Starting %s on %s using frozen S05 settings", name, phase)
            part = dates_and_predictions(name, params, d, dates, numeric, a, phase)
            predictions.append(part)
            save(pd.concat(predictions, ignore_index=True), a.out / "predictions_all.csv")
    predictions = pd.concat(predictions, ignore_index=True)
    daily = per_rebalance(predictions, a.top_k)
    metrics = compare_models(predictions, daily, a.top_k)
    save(daily, a.out / "metrics_by_rebalance.csv")
    save(metrics, a.out / "model_comparison.csv")

    rank_field = f"Mean_Precision_at_{a.top_k}"
    ranking = metrics.loc[(metrics.Phase == "Validation") & (metrics.N_Full_Matured_Rebalances > 0)].copy()
    if ranking.empty:
        raise RuntimeError("No fully observed validation Top-K cross-sections; cannot select a model")
    # Predeclared decision rule: validation Top-K first, ROC-AUC tie-break,
    # then simpler model (historical frequency) wins exact ties.
    complexity = {"HistoricalFrequency": 0, "RandomForest": 1, "XGBoost": 2}
    ranking["Complexity_Tiebreak"] = ranking.Model.map(complexity)
    ranking = ranking.sort_values([rank_field, "ROC_AUC", "Complexity_Tiebreak"],
                                  ascending=[False, False, True], na_position="last")
    chosen = ranking.iloc[0].Model
    save(ranking, a.out / "validation_model_ranking.csv")
    for phase in ("Validation", "Test"):
        p = predictions.loc[predictions.Phase.eq(phase)]
        save(p, a.out / f"predictions_{phase.lower()}.csv")
        selected = p.loc[p.Selected_TopK].copy()
        save(selected, a.out / f"top_{a.top_k}_{phase.lower()}_all_models.csv")
        save(selected.loc[selected.Model.eq(chosen)], a.out / f"top_{a.top_k}_{phase.lower()}_chosen_model.csv")

    manifest = {
        "script": "S05_final_classification.py",
        "models": list(MODELS),
        "hyperparameter_source": str(a.tuned_params),
        "models_retuned": False,
        "calendar_source": str(a.spy),
        "calendar_anchor": str(val_dates[0].date()),
        "calendar_step_spy_sessions": STEP,
        "validation_dates": [str(x.date()) for x in val_dates],
        "test_dates": [str(x.date()) for x in test_dates],
        "forward_target": "60 asset observations, per S03 and S05 audit",
        "training_rule": "Date < rebalance and Label_End_Date < rebalance",
        "last_observed_panel_date": str(observed_end.date()),
        "invalidated_non_observed_labels": invalidated,
        "numeric_features": numeric,
        "selection_rule": f"max validation mean Precision@{a.top_k}; tie break validation ROC-AUC then model simplicity",
        "chosen_model": chosen,
        "test_warning": "Test period has already been reviewed in prior S05/S06 experiments; not an untouched holdout",
        "macro_warning": "Macro publication-vintage and market regime formation audit remains outstanding",
    }
    (a.out / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    LOG.info("Finished. Validation-selected model = %s; results at %s", chosen, a.out)


if __name__ == "__main__":
    main()
